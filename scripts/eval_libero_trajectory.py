# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate a complete LIBERO interval with fixed-horizon PointWorld windows.

The checkpoint is evaluated independently on every 11-frame exported window.
The resulting timeline keeps only frames that were not already emitted by an
earlier window, so the artifact covers the complete requested trajectory
without pretending that PointWorld supports a longer autoregressive rollout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("WANDB_MODE", "disabled")

from dataset_components.cameras import select_cameras_in_order
from dataset_components.collate import custom_collate_fn
from dataset_components.pipeline import apply_release_pipeline_to_sample
from dataset_components.robot import canonicalize_gripper_keys_and_flags
from scripts.eval_libero_clip import _build_args, _checkpoint_domains
from tools.libero.sample_schema import T_FRAMES, flatten_for_pointworld, load_npz
from tools.libero.trajectory import (
    TrajectoryWindow,
    newly_covered_local_indices,
)
from training.trainer import Trainer


@dataclass(slots=True)
class WindowPrediction:
    window: TrajectoryWindow
    pred_centered: np.ndarray
    gt_centered: np.ndarray
    colors: np.ndarray
    supervised: np.ndarray
    shift_amount: np.ndarray
    robot_world: np.ndarray
    right_gripper_pose_world: np.ndarray
    right_gripper_open: np.ndarray


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", required=True)
    p.add_argument(
        "--manifest",
        required=True,
        help="manifest.json produced by tools.libero.export_trajectory_windows.",
    )
    p.add_argument("--out_trajectory", required=True)
    p.add_argument("--out_metrics", default=None)
    p.add_argument(
        "--out_window_dir",
        default=None,
        help=(
            "Directory for independent per-window viewer payloads. Defaults to "
            "<out_trajectory stem>_windows."
        ),
    )
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument(
        "--num_cameras",
        type=int,
        choices=[1, 2, 3],
        default=3,
        help="Number of exported cameras to evaluate in on-disk order.",
    )
    p.add_argument("--model_domain", default="behavior")
    p.add_argument("--norm_stats_path", default="stats/droid_behavior")
    p.add_argument("--has_bimanual_robot", action="store_true", default=True)
    p.add_argument("--no_bimanual", dest="has_bimanual_robot", action="store_false")
    p.add_argument("--exp_name", default="libero_trajectory_eval")
    p.add_argument("--log_dir", default="/tmp/pointworld_log")
    return p.parse_args()


def _as_numpy(value: object) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _parse_window_start(raw: dict, source_path: Path) -> int:
    key = str(raw.get("__key__", ""))
    try:
        _, indices = key.rsplit("-", 1)
        start_text, _ = indices.split(":", 1)
        return int(start_text)
    except (TypeError, ValueError):
        stem = source_path.stem
        if stem.startswith("clip_") and stem[5:].isdigit():
            return int(stem[5:])
        raise ValueError(
            f"Could not recover window start index from {source_path} (key={key!r})"
        )


def _manifest_paths(manifest_path: Path) -> list[Path]:
    manifest = json.loads(manifest_path.read_text())
    entries = manifest.get("windows")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{manifest_path} does not contain any exported windows")

    paths: list[Path] = []
    for entry in entries:
        if not isinstance(entry, dict) or "output" not in entry:
            raise ValueError(f"Malformed window entry in {manifest_path}: {entry!r}")
        candidate = Path(str(entry["output"]))
        if not candidate.exists() and not candidate.is_absolute():
            candidate = manifest_path.parent / candidate
        if not candidate.is_file():
            raise FileNotFoundError(f"Window clip from manifest not found: {candidate}")
        paths.append(candidate)
    return paths


def _prepare_window_sample(cli_args: argparse.Namespace, raw: dict) -> dict:
    sample = flatten_for_pointworld(raw)
    sample["__domain__"] = cli_args.model_domain
    sample = select_cameras_in_order(sample, num_cameras=cli_args.num_cameras)
    sample = canonicalize_gripper_keys_and_flags(sample)
    return apply_release_pipeline_to_sample(
        sample=sample,
        domain=cli_args.model_domain,
        mode="test",
        args=cli_args._trainer_args,
        has_bimanual_robot=cli_args.has_bimanual_robot,
        include_scene_data=True,
    )


def _run_window(
    *,
    cli_args: argparse.Namespace,
    trainer: Trainer,
    raw: dict,
    source_path: Path,
) -> tuple[WindowPrediction, dict]:
    sample = _prepare_window_sample(cli_args, raw)
    batch = custom_collate_fn([sample], args=cli_args._trainer_args)
    batch = {
        key: (value.to(trainer.device) if torch.is_tensor(value) else value)
        for key, value in batch.items()
    }
    with torch.inference_mode():
        outputs = trainer.model(batch, training=False)

    pred = _as_numpy(outputs["scene_flows"])[0].astype(np.float32, copy=False)
    gt = _as_numpy(batch["gt_scene_flows"])[0].astype(np.float32, copy=False)
    colors = np.clip(
        np.rint(_as_numpy(sample["scene_colors"]) * 255.0), 0.0, 255.0
    ).astype(np.uint8)
    supervised = _as_numpy(sample["scene_supervised_mask"]).astype(bool)
    shift = _as_numpy(sample["__shift_amount__"]).astype(np.float32).reshape(-1)
    if shift.size < 3:
        raise ValueError(f"Invalid __shift_amount__ shape: {shift.shape}")

    start_idx = _parse_window_start(raw, source_path)
    raw_robot = np.asarray(raw["robot_flows"], dtype=np.float32)
    raw_pose = np.asarray(raw["right_gripper_pose"], dtype=np.float32)
    raw_open = np.asarray(raw["right_gripper_open"], dtype=np.float32)
    if raw_robot.shape[0] != T_FRAMES or raw_pose.shape != (T_FRAMES, 7):
        raise ValueError(f"Unexpected raw robot trajectory in {source_path}")
    if raw_open.shape != (T_FRAMES, 1):
        raise ValueError(f"Unexpected right_gripper_open in {source_path}")

    result = WindowPrediction(
        window=TrajectoryWindow(start_idx, start_idx + T_FRAMES - 1),
        pred_centered=pred,
        gt_centered=gt,
        colors=colors,
        supervised=supervised,
        shift_amount=shift[:3],
        robot_world=raw_robot,
        right_gripper_pose_world=raw_pose,
        right_gripper_open=raw_open,
    )
    return result, sample


def _pad_scene_frame(
    frame: np.ndarray, target_points: int, *, dtype: np.dtype
) -> np.ndarray:
    out = np.zeros((target_points, 3), dtype=dtype)
    out[: frame.shape[0]] = frame
    return out


def _build_trajectory_payload(
    *,
    records: list[WindowPrediction],
    anchor_sample: dict,
    model_domain: str,
    clip_key: str,
) -> tuple[dict[str, np.ndarray], dict[str, float | int]]:
    if not records:
        raise ValueError("No window predictions were provided")
    records = sorted(records, key=lambda record: record.window.start_idx)
    anchor_shift = records[0].shift_amount
    max_scene_points = max(record.pred_centered.shape[1] for record in records)
    max_robot_points = max(record.robot_world.shape[1] for record in records)

    pred_frames: list[np.ndarray] = []
    gt_frames: list[np.ndarray] = []
    color_frames: list[np.ndarray] = []
    exists_frames: list[np.ndarray] = []
    supervised_frames: list[np.ndarray] = []
    robot_frames: list[np.ndarray] = []
    robot_exists_frames: list[np.ndarray] = []
    pose_frames: list[np.ndarray] = []
    open_frames: list[np.ndarray] = []
    frame_indices: list[int] = []
    source_starts: list[int] = []
    source_local_indices: list[int] = []

    covered_through = records[0].window.start_idx - 1
    for record in records:
        for local_idx in newly_covered_local_indices(
            record.window, covered_through=covered_through
        ):
            # Each window has a different center shift. Convert it into the
            # first window's coordinate frame, which is also the camera frame
            # retained for native PredictionVisualizer rendering.
            pred_anchor = (
                record.pred_centered[local_idx]
                - record.shift_amount[None, :]
                + anchor_shift[None, :]
            )
            gt_anchor = (
                record.gt_centered[local_idx]
                - record.shift_amount[None, :]
                + anchor_shift[None, :]
            )
            n_scene = pred_anchor.shape[0]
            pred_frames.append(
                _pad_scene_frame(pred_anchor, max_scene_points, dtype=np.float32)
            )
            gt_frames.append(
                _pad_scene_frame(gt_anchor, max_scene_points, dtype=np.float32)
            )
            color_frames.append(
                _pad_scene_frame(
                    record.colors[local_idx], max_scene_points, dtype=np.uint8
                )
            )
            exists = np.zeros((max_scene_points,), dtype=bool)
            exists[:n_scene] = True
            exists_frames.append(exists)
            supervised = np.zeros((max_scene_points,), dtype=bool)
            supervised[:n_scene] = record.supervised[local_idx]
            supervised_frames.append(supervised)

            robot_anchor = record.robot_world[local_idx] + anchor_shift[None, :]
            robot_frame = np.zeros((max_robot_points, 3), dtype=np.float32)
            robot_frame[: robot_anchor.shape[0]] = robot_anchor
            robot_frames.append(robot_frame)
            robot_exists = np.zeros((max_robot_points,), dtype=bool)
            robot_exists[: robot_anchor.shape[0]] = True
            robot_exists_frames.append(robot_exists)

            pose = record.right_gripper_pose_world[local_idx].copy()
            pose[:3] += anchor_shift
            pose_frames.append(pose)
            open_frames.append(record.right_gripper_open[local_idx])
            frame_indices.append(record.window.start_idx + local_idx)
            source_starts.append(record.window.start_idx)
            source_local_indices.append(local_idx)

        covered_through = max(covered_through, record.window.end_idx)

    if not pred_frames:
        raise ValueError("Window plan did not emit any trajectory frames")

    payload: dict[str, np.ndarray] = {
        "__key__": np.asarray(clip_key),
        "__domain__": np.asarray(model_domain),
        "scene_flows": np.stack(gt_frames, axis=0),
        "gt_scene_flows": np.stack(gt_frames, axis=0),
        "scene_prediction": np.stack(pred_frames, axis=0),
        "scene_colors": np.stack(color_frames, axis=0),
        "scene_exists": np.stack(exists_frames, axis=0),
        "scene_supervised_mask": np.stack(supervised_frames, axis=0),
        "robot_flows": np.stack(robot_frames, axis=0),
        "robot_exists": np.stack(robot_exists_frames, axis=0),
        "right_gripper_pose": np.stack(pose_frames, axis=0),
        "right_gripper_open": np.stack(open_frames, axis=0),
        "frame_indices": np.asarray(frame_indices, dtype=np.int32),
        "source_window_start": np.asarray(source_starts, dtype=np.int32),
        "source_local_index": np.asarray(source_local_indices, dtype=np.int32),
    }
    source_starts_np = payload["source_window_start"]
    source_local_np = payload["source_local_index"]
    same_window_step = (
        (source_starts_np[1:] == source_starts_np[:-1])
        & (source_local_np[1:] == source_local_np[:-1] + 1)
    )
    payload["scene_flow_transition_mask"] = (
        same_window_step[:, None]
        & payload["scene_exists"][:-1]
        & payload["scene_exists"][1:]
    )
    payload["robot_flow_transition_mask"] = (
        same_window_step[:, None]
        & payload["robot_exists"][:-1]
        & payload["robot_exists"][1:]
    )
    for key, value in anchor_sample.items():
        if key.startswith("cam") and key.endswith(
            ("_initial_rgb", "_initial_depth", "_intrinsic", "_extrinsic")
        ):
            payload[key] = _as_numpy(value)

    pred_all = payload["scene_prediction"]
    gt_all = payload["gt_scene_flows"]
    exists_all = payload["scene_exists"]
    errors = np.linalg.norm(pred_all - gt_all, axis=-1)
    prediction_mask = exists_all.copy()
    prediction_mask[0] = False  # The first frame is the context frame.
    metrics: dict[str, float | int] = {
        "timeline_frames": int(pred_all.shape[0]),
        "window_count": int(len(records)),
        "max_scene_points": int(max_scene_points),
        "max_robot_points": int(max_robot_points),
        "stitched_epe_all_m": (
            float(errors[prediction_mask].mean())
            if np.any(prediction_mask)
            else float("nan")
        ),
    }
    return payload, metrics


def _window_viewer_payload(
    *,
    record: WindowPrediction,
    sample: dict,
    model_domain: str,
) -> dict[str, np.ndarray]:
    """Serialize exactly one model window for coordinate-safe visualization.

    The release pipeline applies the same centering transform to scene points,
    robot points and camera extrinsics. Keeping that complete post-pipeline
    payload together prevents a later trajectory viewer from mixing camera
    data from one window with scene points from another.
    """
    scene_exists = np.ones(record.gt_centered.shape[:2], dtype=bool)
    robot = _as_numpy(sample["robot_flows"]).astype(np.float32, copy=False)
    robot_exists = np.ones(robot.shape[:2], dtype=bool)
    payload: dict[str, np.ndarray] = {
        "__key__": np.asarray(
            f"libero_window:{record.window.start_idx}:{record.window.end_idx}"
        ),
        "__domain__": np.asarray(model_domain),
        "start_idx": np.asarray(record.window.start_idx, dtype=np.int32),
        "end_idx": np.asarray(record.window.end_idx, dtype=np.int32),
        "pred_scene_flows": record.pred_centered.astype(np.float32, copy=False),
        "gt_scene_flows": record.gt_centered.astype(np.float32, copy=False),
        "scene_colors": record.colors.astype(np.uint8, copy=False),
        "scene_exists": scene_exists,
        "scene_supervised_mask": record.supervised.astype(bool, copy=False),
        "robot_flows": robot,
        "robot_exists": robot_exists,
    }
    for key, value in sample.items():
        if key.startswith("cam") and key.endswith(
            ("_initial_rgb", "_initial_depth", "_intrinsic", "_extrinsic")
        ):
            payload[key] = _as_numpy(value)
    if not any(key.endswith("_initial_rgb") for key in payload):
        raise RuntimeError("Window payload is missing post-pipeline camera images")
    return payload


def main() -> None:
    cli_args = _parse_args()
    cli_args._checkpoint_domains = _checkpoint_domains(str(cli_args.model_path))
    cli_args._trainer_args = _build_args(cli_args)
    trainer = Trainer(cli_args._trainer_args, inference_only=True, data_info_dict=None)
    trainer.model.eval()

    manifest_path = Path(cli_args.manifest).resolve()
    output_path = Path(cli_args.out_trajectory)
    window_dir = (
        Path(cli_args.out_window_dir)
        if cli_args.out_window_dir is not None
        else output_path.with_name(f"{output_path.stem}_windows")
    )
    window_dir.mkdir(parents=True, exist_ok=True)
    records: list[WindowPrediction] = []
    anchor_sample: dict | None = None
    window_entries: list[dict[str, int | str]] = []
    for path in _manifest_paths(manifest_path):
        raw = load_npz(str(path))
        record, sample = _run_window(
            cli_args=cli_args,
            trainer=trainer,
            raw=raw,
            source_path=path,
        )
        if anchor_sample is None:
            anchor_sample = sample
        records.append(record)
        window_path = window_dir / f"window_{record.window.start_idx:06d}.npz"
        np.savez_compressed(
            window_path,
            **_window_viewer_payload(
                record=record,
                sample=sample,
                model_domain=cli_args.model_domain,
            ),
        )
        window_entries.append(
            {
                "start_idx": record.window.start_idx,
                "end_idx": record.window.end_idx,
                "output": str(window_path.resolve()),
            }
        )
        print(
            f"[eval_trajectory] evaluated window "
            f"{record.window.start_idx}:{record.window.end_idx}",
            flush=True,
        )

    if anchor_sample is None:
        raise RuntimeError("No valid windows were evaluated")
    payload, metrics = _build_trajectory_payload(
        records=records,
        anchor_sample=anchor_sample,
        model_domain=cli_args.model_domain,
        clip_key=f"libero_trajectory:{records[0].window.start_idx}:{records[-1].window.end_idx}",
    )
    metrics.update(
        {
            "checkpoint": str(cli_args.model_path),
            "manifest": str(manifest_path),
            "model_domain": cli_args.model_domain,
            "window_size": T_FRAMES,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    window_manifest_path = window_dir / "window_manifest.json"
    window_manifest_path.write_text(
        json.dumps(
            {
                "source_manifest": str(manifest_path),
                "model_domain": cli_args.model_domain,
                "window_size": T_FRAMES,
                "windows": window_entries,
            },
            indent=2,
        )
        + "\n"
    )
    metrics_path = (
        Path(cli_args.out_metrics)
        if cli_args.out_metrics is not None
        else output_path.with_suffix(".metrics.json")
    )
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"[eval_trajectory] wrote {output_path}")
    print(f"[eval_trajectory] wrote {metrics_path}")
    print(f"[eval_trajectory] wrote {window_manifest_path}")


if __name__ == "__main__":
    main()
