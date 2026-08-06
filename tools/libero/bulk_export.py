"""Bulk export LIBERO .npz clips using the OSMesa / native mujoco renderer.

For every ``*_demo.hdf5`` under ``--libero_root`` and every ``demo_<i>`` group
inside it we emit overlapping 1-second PointWorld clips. LIBERO is recorded at
20 Hz, so the default samples raw indices ``s, s+2, ..., s+20`` and advances
window starts by 10 raw indices (5 PointWorld steps, or 0.5 seconds). The
per-task / per-demo split is recorded in
``manifest.json`` so the training dataloader can later reconstruct the
train/val assignment.

This is a thin driver around :mod:`tools.libero.export_clip_native` so we
do not duplicate the BDDL / camera-layout / state-setting logic. We just
import the entry point and shell out one Python process per clip -- the
env + mujoco.Renderer construction is dominated by OSMesa + LIBERO BDDL
imports and does not survive pickling into multiprocessing pools cleanly,
so per-clip subprocesses are the simplest reliable strategy.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import h5py

from tools.libero.sample_schema import T_FRAMES


def _list_tasks(libero_root: Path, task_glob: str) -> list[Path]:
    return sorted(path for path in libero_root.rglob(task_glob) if path.is_file())


def _list_demos(hdf5_path: Path) -> list[str]:
    def demo_sort_key(name: str) -> tuple[int, int | str]:
        prefix, separator, suffix = name.rpartition("_")
        if prefix == "demo" and separator and suffix.isdigit():
            return (0, int(suffix))
        return (1, name)

    with h5py.File(hdf5_path, "r") as f:
        return sorted(f["data"].keys(), key=demo_sort_key)


def _demo_num_states(hdf5_path: Path, demo_id: str) -> int:
    with h5py.File(hdf5_path, "r") as f:
        return int(len(f[f"data/{demo_id}/states"]))


def _short_task_name(task_stem: str, suffix: str = "_demo") -> str:
    # task_stem looks like "pick_up_the_black_bowl_..._plate_demo"
    if task_stem.endswith(suffix):
        task_stem = task_stem[: -len(suffix)]
    return task_stem


def _run_one_clip(
    *,
    hdf5: Path,
    demo_id: str,
    start_idx: int,
    output: Path,
    camera_layout: str,
    cameras: list[str],
    bddl_search: list[str],
    workspace_bounds: list[float] | None,
    frame_step: int,
    window_stride_raw: int,
    allow_bddl_reconstruction: bool,
) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "tools.libero.export_clip_native",
        "--demo_hdf5",
        str(hdf5),
        "--demo_id",
        demo_id,
        "--start_idx",
        str(start_idx),
        "--frame_step",
        str(frame_step),
        "--window_stride_raw",
        str(window_stride_raw),
        "--camera_layout",
        camera_layout,
        "--camera_names",
        *cameras,
        "--output",
        str(output),
    ]
    for sd in bddl_search:
        cmd.extend(["--bddl_search_dir", sd])
    if workspace_bounds is not None:
        cmd.extend(["--workspace_bounds", *[str(value) for value in workspace_bounds]])
    if allow_bddl_reconstruction:
        cmd.append("--allow_bddl_reconstruction")
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "osmesa")
    env["PYOPENGL_PLATFORM"] = "osmesa"
    # ``tools.libero.export_clip._resolve_bddl_for_demo`` looks up BDDL
    # files under ``$LIBERO_SRC/libero/bddl_files`` (and a hard-coded
    # fallback that does not exist on this machine). The pointworld-env
    # ships LIBERO at site-packages/libero, so set LIBERO_SRC to the
    # site-packages root -- the resolver appends ``/libero/bddl_files``.
    env.setdefault(
        "LIBERO_SRC",
        str(Path(sys.executable).parent.parent / "lib" / "python3.10" / "site-packages" / "libero"),
    )
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    elapsed = time.time() - t0
    return {
        "hdf5": str(hdf5),
        "demo_id": demo_id,
        "start_idx": int(start_idx),
        "frame_step": int(frame_step),
        "window_stride_raw": int(window_stride_raw),
        "output": str(output),
        "returncode": int(proc.returncode),
        "elapsed_s": round(elapsed, 2),
        "stderr_tail": proc.stderr.splitlines()[-6:],
    }


def _run_demo_windows(
    *,
    hdf5: Path,
    demo_id: str,
    items: list[dict],
    camera_layout: str,
    cameras: list[str],
    bddl_search: list[str],
    workspace_bounds: list[float] | None,
    frame_step: int,
    window_stride_raw: int,
    allow_bddl_reconstruction: bool,
) -> tuple[int, list[dict], list[str]]:
    for item in items:
        item["output"].parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="libero_demo_results_", suffix=".json", delete=False
    ) as handle:
        result_path = Path(handle.name)
    cmd = [
        sys.executable,
        "-m",
        "tools.libero.export_demo_windows_native",
        "--demo_hdf5", str(hdf5),
        "--demo_id", demo_id,
        "--start_indices", *[str(item["start_idx"]) for item in items],
        "--output_paths", *[str(item["output"]) for item in items],
        "--result_json", str(result_path),
        "--frame_step", str(frame_step),
        "--window_stride_raw", str(window_stride_raw),
        "--camera_layout", camera_layout,
        "--camera_names", *cameras,
    ]
    for search_dir in bddl_search:
        cmd.extend(["--bddl_search_dir", search_dir])
    if workspace_bounds is not None:
        cmd.extend(["--workspace_bounds", *map(str, workspace_bounds)])
    if allow_bddl_reconstruction:
        cmd.append("--allow_bddl_reconstruction")
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "osmesa")
    env["PYOPENGL_PLATFORM"] = "osmesa"
    env.setdefault(
        "LIBERO_SRC",
        str(Path(sys.executable).parent.parent / "lib" / "python3.10" / "site-packages" / "libero"),
    )
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    try:
        results = json.loads(result_path.read_text()) if result_path.stat().st_size else []
    finally:
        result_path.unlink(missing_ok=True)
    return int(proc.returncode), results, proc.stderr.splitlines()[-8:]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--libero_root", required=True,
                    help="Root directory containing <task>_demo.hdf5 files.")
    ap.add_argument("--output_root", required=True,
                    help="Output root; clips are written to <output_root>/<split>/<task_short>__<demo_id>.npz.")
    ap.add_argument("--n_demos_per_task", type=int, default=-1,
                    help="If > 0, only use the first N demos per task (for smoke tests).")
    ap.add_argument("--n_val_per_task", type=int, default=5,
                    help="Last N demos per task go to val (default 5).")
    ap.add_argument(
        "--task_glob", default="*_demo.hdf5",
        help="HDF5 filename glob below --libero_root (default: *_demo.hdf5).",
    )
    ap.add_argument(
        "--max_demos_per_split", type=int, default=0,
        help="If > 0, keep only this many demos from each train/val split.",
    )
    ap.add_argument("--start_idx", type=int, default=0,
                    help="First raw LIBERO index considered in each demo (default 0).")
    ap.add_argument(
        "--frame_step", type=int, default=2,
        help="Raw 20 Hz LIBERO states per PointWorld step (default 2).",
    )
    ap.add_argument(
        "--window_stride", type=int, default=5,
        help="Window-start stride in PointWorld steps (default 5 = 0.5 s).",
    )
    ap.add_argument(
        "--max_windows_per_demo", type=int, default=0,
        help="If > 0, export at most this many windows per demo (smoke tests).",
    )
    ap.add_argument("--camera_layout", default="oblique_triplet",
                    choices=("native", "oblique_pair", "oblique_triplet"))
    ap.add_argument("--camera_names", nargs="+",
                    default=["frontview", "sideview", "birdview"])
    ap.add_argument("--bddl_search_dir", action="append", default=None)
    ap.add_argument(
        "--workspace_bounds", nargs=6, type=float, default=None,
        metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
        help="Optional world-frame crop forwarded to every clip exporter.",
    )
    ap.add_argument(
        "--allow_bddl_reconstruction",
        action="store_true",
        help=(
            "Allow input demos without embedded model_file. This rebuilds "
            "fixtures from BDDL and is not faithful to the recorded scene."
        ),
    )
    ap.add_argument("--skip_existing", action="store_true",
                    help="If output file already exists, skip rather than re-export.")
    ap.add_argument(
        "--plan_only", action="store_true",
        help="Write manifest.json with the resolved plan without exporting clips.",
    )
    args = ap.parse_args()
    if args.frame_step < 1:
        raise ValueError("--frame_step must be >= 1")
    if args.window_stride < 1:
        raise ValueError("--window_stride must be >= 1")
    if args.max_windows_per_demo < 0:
        raise ValueError("--max_windows_per_demo must be >= 0")
    if args.max_demos_per_split < 0:
        raise ValueError("--max_demos_per_split must be >= 0")
    raw_window_stride = int(args.frame_step) * int(args.window_stride)
    raw_window_span = (T_FRAMES - 1) * int(args.frame_step)

    libero_root = Path(args.libero_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"

    tasks = _list_tasks(libero_root, args.task_glob)
    print(f"[bulk_export] {len(tasks)} task files under {libero_root}", flush=True)

    plan: list[dict] = []
    for hdf5 in tasks:
        demos = _list_demos(hdf5)
        if args.n_demos_per_task > 0:
            demos = demos[: args.n_demos_per_task]
        n_val = max(0, min(args.n_val_per_task, len(demos)))
        train_demos = demos[:-n_val] if n_val else demos
        val_demos = demos[-n_val:] if n_val else []
        if args.max_demos_per_split > 0:
            train_demos = train_demos[: args.max_demos_per_split]
            val_demos = val_demos[: args.max_demos_per_split]
        task_short = _short_task_name(hdf5.stem)
        for split, split_demos in (("train", train_demos), ("val", val_demos)):
            for demo_id in split_demos:
                num_states = _demo_num_states(hdf5, demo_id)
                last_start = num_states - 1 - raw_window_span
                starts = list(range(args.start_idx, last_start + 1, raw_window_stride))
                if args.max_windows_per_demo > 0:
                    starts = starts[: args.max_windows_per_demo]
                for start_idx in starts:
                    out = output_root / split / (
                        f"{task_short}__{demo_id}__start{start_idx:06d}.npz"
                    )
                    plan.append({
                        "hdf5": hdf5,
                        "demo_id": demo_id,
                        "start_idx": start_idx,
                        "output": out,
                        "split": split,
                    })

    print(f"[bulk_export] {len(plan)} clips planned "
          f"(train={sum(1 for p in plan if p['split']=='train')}, "
          f"val={sum(1 for p in plan if p['split']=='val')})", flush=True)

    manifest: dict = {
        "libero_root": str(libero_root),
        "output_root": str(output_root),
        "camera_layout": args.camera_layout,
        "camera_names": list(args.camera_names),
        "start_idx": int(args.start_idx),
        "frame_step": int(args.frame_step),
        "model_step_seconds": float(args.frame_step / 20.0),
        "window_stride": int(args.window_stride),
        "window_stride_raw": int(raw_window_stride),
        "max_windows_per_demo": int(args.max_windows_per_demo),
        "workspace_bounds": args.workspace_bounds,
        "model_policy": (
            "allow_bddl_reconstruction"
            if args.allow_bddl_reconstruction else "require_embedded_model_file"
        ),
        "n_val_per_task": int(args.n_val_per_task),
        "task_glob": args.task_glob,
        "max_demos_per_split": int(args.max_demos_per_split),
        "plan_size": len(plan),
        "results": [],
    }
    if args.plan_only:
        manifest["plan"] = [
            {
                "hdf5": str(item["hdf5"]),
                "demo_id": item["demo_id"],
                "start_idx": int(item["start_idx"]),
                "output": str(item["output"]),
                "split": item["split"],
            }
            for item in plan
        ]
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"[bulk_export] plan only: wrote {manifest_path}", flush=True)
        return

    groups: dict[tuple[Path, str, str], list[dict]] = defaultdict(list)
    for item in plan:
        groups[(item["hdf5"], item["demo_id"], item["split"])].append(item)

    t_run = time.time()
    n_done = n_skip = n_fail = n_processed = 0
    for group_index, ((hdf5, demo_id, split), items) in enumerate(groups.items(), start=1):
        pending = []
        for item in items:
            if args.skip_existing and item["output"].exists():
                manifest["results"].append({
                    "output": str(item["output"]),
                    "start_idx": int(item["start_idx"]),
                    "skipped_existing": True,
                })
                n_skip += 1
                n_processed += 1
            else:
                pending.append(item)
        if pending:
            returncode, results, stderr_tail = _run_demo_windows(
                hdf5=hdf5,
                demo_id=demo_id,
                items=pending,
                camera_layout=args.camera_layout,
                cameras=args.camera_names,
                bddl_search=args.bddl_search_dir or [],
                workspace_bounds=args.workspace_bounds,
                frame_step=int(args.frame_step),
                window_stride_raw=raw_window_stride,
                allow_bddl_reconstruction=args.allow_bddl_reconstruction,
            )
            if returncode == 0 and len(results) == len(pending):
                manifest["results"].extend(results)
                n_done += len(results)
            else:
                n_fail += len(pending)
                manifest["results"].extend({
                    "output": str(item["output"]),
                    "start_idx": int(item["start_idx"]),
                    "returncode": returncode,
                    "stderr_tail": stderr_tail,
                } for item in pending)
                print(
                    f"[bulk_export] FAILED {demo_id} @ {hdf5.name}: "
                    f"rc={returncode} results={len(results)}/{len(pending)} "
                    f"stderr_tail={stderr_tail}",
                    flush=True,
                )
            n_processed += len(pending)

        elapsed = time.time() - t_run
        eta = (len(plan) - n_processed) * (elapsed / max(1, n_processed))
        print(
            f"[bulk_export] demos={group_index}/{len(groups)} clips={n_processed}/{len(plan)} "
            f"(done={n_done} skip={n_skip} fail={n_fail}) "
            f"elapsed={elapsed/60:.1f}min eta={eta/60:.1f}min",
            flush=True,
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[bulk_export] DONE: done={n_done} skip={n_skip} fail={n_fail} "
          f"manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
