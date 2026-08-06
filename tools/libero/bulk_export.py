"""Bulk export LIBERO .npz clips using the OSMesa / native mujoco renderer.

For every ``*_demo.hdf5`` under ``--libero_root`` and every ``demo_<i>`` group
inside it we emit exactly one 11-frame clip starting at the very first
recorded action. The per-task / per-demo split is recorded in
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
import time
from pathlib import Path

import h5py


def _list_tasks(libero_root: Path) -> list[Path]:
    return sorted(libero_root.rglob("*_demo.hdf5"))


def _list_demos(hdf5_path: Path) -> list[str]:
    with h5py.File(hdf5_path, "r") as f:
        return sorted(f["data"].keys())


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
        "--camera_layout",
        camera_layout,
        "--camera_names",
        *cameras,
        "--output",
        str(output),
    ]
    for sd in bddl_search:
        cmd.extend(["--bddl_search_dir", sd])
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
        "output": str(output),
        "returncode": int(proc.returncode),
        "elapsed_s": round(elapsed, 2),
        "stderr_tail": proc.stderr.splitlines()[-6:],
    }


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
    ap.add_argument("--start_idx", type=int, default=0,
                    help="Start frame inside each demo (default 0; needs demo len >= T_FRAMES).")
    ap.add_argument("--camera_layout", default="oblique_triplet",
                    choices=("native", "oblique_pair", "oblique_triplet"))
    ap.add_argument("--camera_names", nargs="+",
                    default=["frontview", "sideview", "birdview"])
    ap.add_argument("--bddl_search_dir", action="append", default=None)
    ap.add_argument("--skip_existing", action="store_true",
                    help="If output file already exists, skip rather than re-export.")
    args = ap.parse_args()

    libero_root = Path(args.libero_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"

    tasks = _list_tasks(libero_root)
    print(f"[bulk_export] {len(tasks)} task files under {libero_root}", flush=True)

    plan: list[dict] = []
    for hdf5 in tasks:
        demos = _list_demos(hdf5)
        if args.n_demos_per_task > 0:
            demos = demos[: args.n_demos_per_task]
        n_val = max(0, min(args.n_val_per_task, len(demos)))
        train_demos = demos[:-n_val] if n_val else demos
        val_demos = demos[-n_val:] if n_val else []
        task_short = _short_task_name(hdf5.stem)
        for demo_id in train_demos:
            out = output_root / "train" / f"{task_short}__{demo_id}.npz"
            plan.append({"hdf5": hdf5, "demo_id": demo_id, "start_idx": args.start_idx,
                         "output": out, "split": "train"})
        for demo_id in val_demos:
            out = output_root / "val" / f"{task_short}__{demo_id}.npz"
            plan.append({"hdf5": hdf5, "demo_id": demo_id, "start_idx": args.start_idx,
                         "output": out, "split": "val"})

    print(f"[bulk_export] {len(plan)} clips planned "
          f"(train={sum(1 for p in plan if p['split']=='train')}, "
          f"val={sum(1 for p in plan if p['split']=='val')})", flush=True)

    manifest: dict = {
        "libero_root": str(libero_root),
        "output_root": str(output_root),
        "camera_layout": args.camera_layout,
        "camera_names": list(args.camera_names),
        "start_idx": int(args.start_idx),
        "n_val_per_task": int(args.n_val_per_task),
        "plan_size": len(plan),
        "results": [],
    }

    t_run = time.time()
    n_done = 0
    n_skip = 0
    n_fail = 0
    for i, item in enumerate(plan):
        out_path: Path = item["output"]
        if args.skip_existing and out_path.exists():
            result = {"output": str(out_path), "skipped_existing": True}
            n_skip += 1
        else:
            result = _run_one_clip(
                hdf5=item["hdf5"],
                demo_id=item["demo_id"],
                start_idx=item["start_idx"],
                output=out_path,
                camera_layout=args.camera_layout,
                cameras=args.camera_names,
                bddl_search=args.bddl_search_dir or [],
            )
            if result["returncode"] == 0:
                n_done += 1
            else:
                n_fail += 1
                print(f"[bulk_export] FAILED {item['demo_id']} @ {item['hdf5'].name}: "
                      f"rc={result['returncode']} stderr_tail={result['stderr_tail']}", flush=True)
        manifest["results"].append(result)
        if (i + 1) % 10 == 0 or i + 1 == len(plan):
            elapsed = time.time() - t_run
            eta = (len(plan) - i - 1) * (elapsed / max(1, i + 1))
            print(f"[bulk_export] {i + 1}/{len(plan)} "
                  f"(done={n_done} skip={n_skip} fail={n_fail}) "
                  f"elapsed={elapsed/60:.1f}min eta={eta/60:.1f}min", flush=True)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[bulk_export] DONE: done={n_done} skip={n_skip} fail={n_fail} "
          f"manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
