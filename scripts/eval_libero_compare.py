"""Compare two PointWorld checkpoints on stratified LIBERO validation clips."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("WANDB_MODE", "disabled")

import numpy as np
import torch

from dataset_components.cameras import select_cameras_in_order
from dataset_components.collate import custom_collate_fn
from dataset_components.pipeline import apply_release_pipeline_to_sample
from dataset_components.robot import canonicalize_gripper_keys_and_flags
from scripts.eval_libero_clip import _build_args, _checkpoint_domains
from training.trainer import Trainer
from tools.libero.sample_schema import flatten_for_pointworld, load_npz


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split_manifest", required=True)
    parser.add_argument("--baseline_checkpoint", required=True)
    parser.add_argument("--finetuned_checkpoint", required=True)
    parser.add_argument("--baseline_norm_stats", default="stats/droid")
    parser.add_argument("--finetuned_norm_stats", default="stats/libero_spatial_paper")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_candidates", type=int, default=30)
    parser.add_argument("--num_selected", type=int, default=12)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def _task_from_clip(path: Path) -> str:
    return path.name.split("__demo_", 1)[0]


def _motion_stats(path: Path) -> dict[str, float]:
    with np.load(path, allow_pickle=True) as clip:
        points = np.asarray(clip["camera_0_scene_flows"], dtype=np.float32)
        movement = np.linalg.norm(points[1:] - points[:1], axis=-1).max(axis=0)
    moved = movement > 0.005
    return {
        "motion_mean_m": float(movement.mean()),
        "motion_p95_m": float(np.quantile(movement, 0.95)),
        "moved_fraction": float(moved.mean()),
    }


def _candidate_paths(data_root: Path, manifest_path: Path, count: int) -> list[dict]:
    payload = json.loads(manifest_path.read_text())
    entries = payload["splits"]["val"]
    records: list[dict] = []
    for index, relative in enumerate(entries, start=1):
        path = data_root / relative
        stats = _motion_stats(path)
        records.append({"clip": str(path), "task": _task_from_clip(path), **stats})
        if index % 50 == 0:
            print(f"[selection] scanned {index}/{len(entries)} clips", flush=True)

    by_task: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_task[record["task"]].append(record)
    per_task = max(1, count // max(len(by_task), 1))
    selected: list[dict] = []
    for task in sorted(by_task):
        ranked = sorted(by_task[task], key=lambda item: item["motion_mean_m"])
        quantiles = np.linspace(0.5, 1.0, per_task)
        for quantile in quantiles:
            selected.append(ranked[round((len(ranked) - 1) * float(quantile))])
    if len(selected) < count:
        used = {record["clip"] for record in selected}
        remaining = sorted(records, key=lambda item: item["motion_mean_m"], reverse=True)
        selected.extend(record for record in remaining if record["clip"] not in used)
    return selected[:count]


def _trainer_cli(
    checkpoint: str,
    domain: str,
    norm_stats: str,
    device: str,
    exp_name: str,
) -> tuple[argparse.Namespace, Trainer]:
    cli = argparse.Namespace(
        model_path=checkpoint,
        device=device,
        num_cameras=3,
        model_domain=domain,
        norm_stats_path=norm_stats,
        log_dir="/tmp/pointworld_eval_compare",
        exp_name=exp_name,
    )
    cli._checkpoint_domains = _checkpoint_domains(checkpoint)
    trainer_args = _build_args(cli)
    trainer = Trainer(trainer_args, inference_only=True, data_info_dict=None)
    trainer.model.eval()
    return trainer_args, trainer


def _run_checkpoint(
    *,
    tag: str,
    checkpoint: str,
    domain: str,
    norm_stats: str,
    device: str,
    candidates: list[dict],
    out_dir: Path,
) -> dict[str, dict]:
    result: dict[str, dict] = {}
    prediction_dir = out_dir / tag
    prediction_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for record in candidates:
        clip_path = Path(record["clip"])
        output = prediction_dir / f"{clip_path.stem}.npz"
        if output.is_file():
            with np.load(output) as cached:
                epe = cached["per_point_epe"]
                moved = cached["moved_mask"].astype(bool)
                static = ~moved
                metrics = {
                    "epe_all_m": float(epe[1:].mean()),
                    "epe_moved_m": float(epe[1:, moved].mean()) if moved.any() else None,
                    "epe_static_m": float(epe[1:, static].mean()) if static.any() else None,
                    "final_epe_all_m": float(epe[-1].mean()),
                    "final_epe_moved_m": float(epe[-1, moved].mean()) if moved.any() else None,
                    "n_points": int(moved.size),
                    "n_moved_points": int(moved.sum()),
                    "moved_fraction": float(moved.mean()),
                }
            result[str(clip_path)] = {"prediction": str(output), "metrics": metrics}
        else:
            pending.append(record)
    if not pending:
        print(f"[{tag}] reused {len(result)} cached predictions", flush=True)
        return result

    args, trainer = _trainer_cli(checkpoint, domain, norm_stats, device, f"libero_{tag}_eval")
    for index, record in enumerate(pending, start=1):
        clip_path = Path(record["clip"])
        raw = load_npz(clip_path)
        sample = flatten_for_pointworld(raw)
        sample["__domain__"] = domain
        sample = select_cameras_in_order(sample, num_cameras=3)
        sample = canonicalize_gripper_keys_and_flags(sample)
        sample = apply_release_pipeline_to_sample(
            sample=sample,
            domain=domain,
            mode="test",
            args=args,
            has_bimanual_robot=False,
            include_scene_data=True,
        )
        batch = custom_collate_fn([sample], args=args)
        batch = {
            key: value.to(trainer.device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        with torch.inference_mode():
            outputs = trainer.model(batch, training=False)
        pred = outputs["scene_flows"][0].float().cpu().numpy().astype(np.float32)
        gt = batch["gt_scene_flows"][0].float().cpu().numpy().astype(np.float32)
        colors = np.clip(
            np.rint(np.asarray(sample["scene_colors"], dtype=np.float32) * 255.0),
            0,
            255,
        ).astype(np.uint8)
        shift = np.asarray(sample["__shift_amount__"], dtype=np.float32).reshape(-1)[:3]
        per_frame_epe = np.linalg.norm(pred - gt, axis=-1).astype(np.float32)
        movement = np.linalg.norm(gt[1:] - gt[:1], axis=-1).max(axis=0)
        moved = movement > 0.005
        static = ~moved
        metrics = {
            "epe_all_m": float(per_frame_epe[1:].mean()),
            "epe_moved_m": float(per_frame_epe[1:, moved].mean()) if moved.any() else None,
            "epe_static_m": float(per_frame_epe[1:, static].mean()) if static.any() else None,
            "final_epe_all_m": float(per_frame_epe[-1].mean()),
            "final_epe_moved_m": float(per_frame_epe[-1, moved].mean()) if moved.any() else None,
            "n_points": int(gt.shape[1]),
            "n_moved_points": int(moved.sum()),
            "moved_fraction": float(moved.mean()),
        }
        output = prediction_dir / f"{clip_path.stem}.npz"
        np.savez_compressed(
            output,
            pred_scene_flows=pred,
            gt_scene_flows=gt,
            scene_colors=colors,
            shift_amount=shift,
            per_point_epe=per_frame_epe,
            moved_mask=moved,
        )
        result[str(clip_path)] = {"prediction": str(output), "metrics": metrics}
        print(f"[{tag}] {index}/{len(pending)} {clip_path.name}", flush=True)

    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _select_for_html(records: list[dict], count: int) -> list[dict]:
    per_category = max(1, count // 3)
    selected: list[dict] = []
    used: set[str] = set()

    def take(ranked: list[dict], category: str) -> None:
        for record in ranked:
            if record["clip"] in used:
                continue
            item = dict(record)
            item["category"] = category
            selected.append(item)
            used.add(record["clip"])
            if sum(entry["category"] == category for entry in selected) >= per_category:
                break

    take(sorted(records, key=lambda item: item["motion_mean_m"], reverse=True), "high_motion")
    take(
        sorted(
            records,
            key=lambda item: item["finetuned"]["metrics"]["epe_moved_m"] or -1,
            reverse=True,
        ),
        "difficult",
    )
    errors = np.array([
        item["finetuned"]["metrics"]["epe_moved_m"] or 0.0 for item in records
    ])
    median = float(np.median(errors))
    take(
        sorted(
            records,
            key=lambda item: abs((item["finetuned"]["metrics"]["epe_moved_m"] or 0.0) - median),
        ),
        "typical",
    )
    return selected[:count]


def main() -> None:
    args = _parse_args()
    if args.num_candidates < args.num_selected:
        raise ValueError("--num_candidates must be >= --num_selected")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_cache = out_dir / "candidate_clips.json"
    if candidate_cache.is_file():
        candidates = json.loads(candidate_cache.read_text())
        print(f"[selection] loaded {len(candidates)} cached candidates", flush=True)
    else:
        candidates = _candidate_paths(
            Path(args.data_root), Path(args.split_manifest), args.num_candidates
        )
        candidate_cache.write_text(json.dumps(candidates, indent=2) + "\n")
    baseline = _run_checkpoint(
        tag="baseline",
        checkpoint=args.baseline_checkpoint,
        domain="droid",
        norm_stats=args.baseline_norm_stats,
        device=args.device,
        candidates=candidates,
        out_dir=out_dir,
    )
    finetuned = _run_checkpoint(
        tag="finetuned",
        checkpoint=args.finetuned_checkpoint,
        domain="libero",
        norm_stats=args.finetuned_norm_stats,
        device=args.device,
        candidates=candidates,
        out_dir=out_dir,
    )
    records = []
    for candidate in candidates:
        item = dict(candidate)
        item["baseline"] = baseline[item["clip"]]
        item["finetuned"] = finetuned[item["clip"]]
        records.append(item)
    selected = _select_for_html(records, args.num_selected)
    manifest = {
        "baseline_checkpoint": args.baseline_checkpoint,
        "finetuned_checkpoint": args.finetuned_checkpoint,
        "candidate_count": len(records),
        "selected_count": len(selected),
        "selected": selected,
        "candidates": records,
    }
    output = out_dir / "comparison_manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[eval_libero_compare] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
