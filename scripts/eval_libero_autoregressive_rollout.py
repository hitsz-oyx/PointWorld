# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create an experimental long-horizon PointWorld scene rollout on LIBERO.

The released PointWorld evaluator predicts one 1-context + 10-frame window
from a recorded observation.  This script deliberately changes that contract:
after ``--gt_prefix_end`` it feeds the previous window's predicted terminal
scene into the next window.  Robot trajectories remain recorded exogenous
conditions because the model has no robot-motion prediction head.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("WANDB_MODE", "disabled")

from dataset_components.collate import custom_collate_fn
from scripts.eval_libero_clip import _build_args, _checkpoint_domains
from scripts.eval_libero_trajectory import _prepare_window_sample
from tools.libero.sample_schema import T_FRAMES, load_npz
from training.trainer import Trainer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument(
        "--manifest",
        required=True,
        help="Full-window manifest produced by export_trajectory_windows.",
    )
    parser.add_argument(
        "--gt_trajectory",
        required=True,
        help="GT-reset trajectory .npz used only to display the GT prefix.",
    )
    parser.add_argument("--out", required=True, help="Output rollout .npz.")
    parser.add_argument("--out_metrics", default=None)
    parser.add_argument("--gt_prefix_end", type=int, default=30)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--num_cameras", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--model_domain", default="behavior")
    parser.add_argument("--norm_stats_path", default="stats/droid_behavior")
    parser.add_argument("--has_bimanual_robot", action="store_true", default=True)
    parser.add_argument("--no_bimanual", dest="has_bimanual_robot", action="store_false")
    parser.add_argument("--exp_name", default="libero_autoregressive_rollout")
    parser.add_argument("--log_dir", default="/tmp/pointworld_log")
    return parser.parse_args()


def _numpy(value: object) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _manifest_paths(path: Path) -> dict[int, Path]:
    manifest = json.loads(path.read_text())
    entries = manifest.get("windows")
    if not isinstance(entries, list):
        raise ValueError(f"{path} has no window list")
    paths: dict[int, Path] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"Malformed window entry: {entry!r}")
        start = int(entry["start_idx"])
        clip = Path(str(entry["output"]))
        if not clip.is_absolute():
            clip = path.parent / clip
        if not clip.is_file():
            raise FileNotFoundError(f"Missing window clip: {clip}")
        paths[start] = clip
    return paths


def _shift_amount(sample: dict) -> np.ndarray:
    shift = _numpy(sample["__shift_amount__"]).astype(np.float32).reshape(-1)
    if shift.size != 3:
        raise ValueError(f"Expected a 3D shift, got {shift.shape}")
    return shift


def _scene_features(
    *,
    scene_context: np.ndarray,
    scene_colors: np.ndarray,
    scene_normals: np.ndarray,
    right_open: np.ndarray,
    left_open: np.ndarray,
    robot_flows: np.ndarray,
) -> np.ndarray:
    """Rebuild the scene features after replacing the scene context.

    This exactly follows ``dataset_components.robot.gather_features`` for the
    released feature order: position, color, normal, per-timestep gripper
    openness, and per-timestep distance to the known robot trajectory.
    """
    context = np.asarray(scene_context, dtype=np.float32)
    colors = np.asarray(scene_colors, dtype=np.float32)
    normals = np.asarray(scene_normals, dtype=np.float32)
    robot = np.asarray(robot_flows, dtype=np.float32)
    if context.ndim != 2 or context.shape[1] != 3:
        raise ValueError(f"scene_context must be (N,3), got {context.shape}")
    if colors.shape != context.shape or normals.shape != context.shape:
        raise ValueError("Static scene colors/normals must align with the context")
    if robot.ndim != 3 or robot.shape[2] != 3:
        raise ValueError(f"robot_flows must be (T,N,3), got {robot.shape}")

    # (T, N): minimum distance from every predicted scene point to each
    # known robot configuration in the upcoming 11-frame action segment.
    deltas = context[None, :, None, :] - robot[:, None, :, :]
    distances = np.linalg.norm(deltas, axis=-1).min(axis=2).astype(np.float32)
    gripper = np.concatenate(
        [
            np.asarray(right_open, dtype=np.float32).reshape(robot.shape[0], -1),
            np.asarray(left_open, dtype=np.float32).reshape(robot.shape[0], -1),
        ],
        axis=1,
    )
    gripper_feature = np.broadcast_to(
        gripper.reshape(1, 1, -1), (1, context.shape[0], gripper.size)
    ).copy()
    return np.concatenate(
        [
            context[None],
            colors[None],
            normals[None],
            gripper_feature,
            distances.transpose(1, 0)[None],
        ],
        axis=-1,
    ).astype(np.float32, copy=False)


def _rollout_sample(
    *,
    prepared_window: dict,
    scene_context: np.ndarray,
    scene_colors: np.ndarray,
    scene_normals: np.ndarray,
    rollout_shift: np.ndarray,
    camera_payload: dict[str, torch.Tensor],
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Build one model batch with a predicted scene and recorded robot input.

    ``prepared_window`` contributes only robot conditioning. Its scene tensors
    are intentionally not copied into the returned sample.
    """
    local_shift = _shift_amount(prepared_window)
    robot_local = _numpy(prepared_window["robot_flows"]).astype(np.float32)
    robot = robot_local - local_shift[None, None, :] + rollout_shift[None, None, :]
    robot_features = _numpy(prepared_window["robot_features"]).astype(np.float32).copy()
    if robot_features.shape[:2] != robot.shape[:2] or robot_features.shape[-1] < 3:
        raise ValueError("Robot feature tensor is incompatible with robot_flows")
    # ``robot_flows`` is the first released robot feature; velocities and
    # normals are translation-invariant and remain valid after the reframe.
    robot_features[..., :3] = robot

    right_open = _numpy(prepared_window["right_gripper_open"]).astype(np.float32)
    left_open = _numpy(prepared_window["left_gripper_open"]).astype(np.float32)
    scene_input = np.broadcast_to(
        np.asarray(scene_context, dtype=np.float32)[None],
        (T_FRAMES, scene_context.shape[0], 3),
    ).copy()
    features = _scene_features(
        scene_context=scene_context,
        scene_colors=scene_colors,
        scene_normals=scene_normals,
        right_open=right_open,
        left_open=left_open,
        robot_flows=robot,
    )

    sample = {
        "__key__": "libero_autoregressive_rollout",
        "__domain__": str(prepared_window["__domain__"]),
        "scene_flows": torch.from_numpy(scene_input),
        "scene_features": torch.from_numpy(features),
        # A finite placeholder is sufficient in inference-only mode; it is
        # never supplied to the model as a feature.
        "gt_scene_flows": torch.from_numpy(scene_input.copy()),
        "robot_flows": torch.from_numpy(robot.astype(np.float32, copy=False)),
        "robot_features": torch.from_numpy(robot_features),
    }
    # ``custom_collate_fn`` builds point weights unconditionally, even for
    # inference. These placeholders are labels only and never enter the model.
    n_scene = scene_context.shape[0]
    context_mask = np.zeros((T_FRAMES, n_scene, 1), dtype=np.float32)
    context_mask[0] = 1.0
    sample.update(
        {
            "scene_context_mask": torch.from_numpy(context_mask),
            "scene_selector_gt": torch.zeros((T_FRAMES, n_scene), dtype=torch.float32),
            "scene_moved_mask": torch.zeros((T_FRAMES, n_scene), dtype=torch.bool),
            "scene_static_mask": torch.ones((T_FRAMES, n_scene), dtype=torch.bool),
            "scene_supervised_mask": torch.ones((T_FRAMES, n_scene), dtype=torch.bool),
        }
    )
    # The scene encoder consumes RGB-D alongside the point tensors. Keep the
    # frame-30 observations fixed for every later model call instead of
    # leaking a new GT camera observation at each rollout boundary.
    sample.update(camera_payload)
    return sample, robot, local_shift


def _run_model(trainer: Trainer, sample: dict, trainer_args) -> np.ndarray:
    batch = custom_collate_fn([sample], args=trainer_args)
    batch = {
        key: (value.to(trainer.device) if torch.is_tensor(value) else value)
        for key, value in batch.items()
    }
    with torch.inference_mode():
        result = trainer.model(batch, training=False)
    pred = result["scene_flows"][0].detach().cpu().numpy().astype(np.float32)
    if pred.shape[0] != T_FRAMES or pred.shape[-1] != 3:
        raise ValueError(f"Unexpected model prediction shape: {pred.shape}")
    return pred


def _write_prediction_window(
    *,
    start: int,
    prediction_rollout_frame: np.ndarray,
    robot_rollout_frame: np.ndarray,
    rollout_shift: np.ndarray,
    display_shift: np.ndarray,
    colors: np.ndarray,
    positions_out: np.ndarray,
    colors_out: np.ndarray,
    exists_out: np.ndarray,
    robot_out: np.ndarray,
    robot_exists_out: np.ndarray,
    predicted_contexts: dict[int, np.ndarray],
    model_mask: np.ndarray,
    transition_mask: np.ndarray,
    local_start: int = 1,
) -> None:
    """Write new rollout frames, retaining one fixed scene-point identity."""
    display_prediction = (
        prediction_rollout_frame - rollout_shift[None, None, :] + display_shift[None, None, :]
    )
    display_robot = (
        robot_rollout_frame - rollout_shift[None, None, :] + display_shift[None, None, :]
    )
    for local_idx in range(local_start, T_FRAMES):
        frame = start + local_idx
        if frame >= positions_out.shape[0]:
            break
        positions_out[frame] = display_prediction[local_idx]
        colors_out[frame] = colors
        exists_out[frame] = True
        robot_out[frame].fill(0.0)
        robot_out[frame, : display_robot.shape[1]] = display_robot[local_idx]
        robot_exists_out[frame] = False
        robot_exists_out[frame, : display_robot.shape[1]] = True
        predicted_contexts[frame] = prediction_rollout_frame[local_idx].copy()
        model_mask[frame] = True
        if frame > 0 and model_mask[frame - 1]:
            transition_mask[frame - 1] = True


def main() -> None:
    args = _parse_args()
    args._checkpoint_domains = _checkpoint_domains(str(args.model_path))
    args._trainer_args = _build_args(args)
    trainer = Trainer(args._trainer_args, inference_only=True, data_info_dict=None)
    trainer.model.eval()

    paths = _manifest_paths(Path(args.manifest).resolve())
    required_starts = list(range(args.gt_prefix_end, 141, 10)) + [144]
    missing_starts = [start for start in required_starts if start not in paths]
    if missing_starts:
        raise ValueError(f"Manifest lacks required rollout windows: {missing_starts}")

    with np.load(args.gt_trajectory, allow_pickle=False) as reference:
        gt_positions = np.asarray(reference["gt_scene_flows"], dtype=np.float32)
        gt_colors = np.asarray(reference["scene_colors"], dtype=np.uint8)
        gt_exists = np.asarray(reference["scene_exists"], dtype=bool)
        gt_robot = np.asarray(reference["robot_flows"], dtype=np.float32)
        gt_robot_exists = np.asarray(reference["robot_exists"], dtype=bool)
    if not (gt_positions.shape[:2] == gt_colors.shape[:2] == gt_exists.shape):
        raise ValueError("GT trajectory scene tensors are inconsistent")
    if args.gt_prefix_end < 0 or args.gt_prefix_end >= gt_positions.shape[0] - 1:
        raise ValueError("--gt_prefix_end must leave at least one prediction frame")

    raw_prefix = load_npz(str(paths[args.gt_prefix_end]))
    initial = _prepare_window_sample(args, raw_prefix)
    rollout_shift = _shift_amount(initial)
    raw_display_anchor = load_npz(str(paths[0]))
    display_anchor = _prepare_window_sample(args, raw_display_anchor)
    display_shift = _shift_amount(display_anchor)

    initial_pred = _run_model(trainer, initial, args._trainer_args)
    initial_colors = _numpy(initial["scene_colors"])[0].astype(np.float32)
    initial_normals = _numpy(initial["scene_normals"])[0].astype(np.float32)
    camera_payload = {
        key: value
        for key, value in initial.items()
        if key.startswith("cam")
        and key.endswith(("_initial_rgb", "_initial_depth", "_intrinsic", "_extrinsic"))
    }
    if len(camera_payload) != args.num_cameras * 4:
        raise RuntimeError(
            "Initial rollout context does not contain all requested camera payloads"
        )
    static_colors = np.clip(np.rint(initial_colors * 255.0), 0.0, 255.0).astype(np.uint8)
    if initial_pred.shape[1] != gt_positions.shape[1]:
        raise ValueError(
            "Rollout and GT display point counts differ: "
            f"{initial_pred.shape[1]} vs {gt_positions.shape[1]}"
        )

    positions = gt_positions.copy()
    colors = gt_colors.copy()
    exists = gt_exists.copy()
    robot = gt_robot.copy()
    robot_exists = gt_robot_exists.copy()
    model_mask = np.zeros((positions.shape[0],), dtype=bool)
    transition_mask = np.zeros((positions.shape[0] - 1,), dtype=bool)
    predicted_contexts: dict[int, np.ndarray] = {
        args.gt_prefix_end: initial_pred[0].copy(),
    }

    # The context frame itself remains visible as GT. The first model output
    # written to the timeline is therefore frame 31.
    initial_robot = _numpy(initial["robot_flows"]).astype(np.float32)
    _write_prediction_window(
        start=args.gt_prefix_end,
        prediction_rollout_frame=initial_pred,
        robot_rollout_frame=initial_robot,
        rollout_shift=rollout_shift,
        display_shift=display_shift,
        colors=static_colors,
        positions_out=positions,
        colors_out=colors,
        exists_out=exists,
        robot_out=robot,
        robot_exists_out=robot_exists,
        predicted_contexts=predicted_contexts,
        model_mask=model_mask,
        transition_mask=transition_mask,
    )

    for start in required_starts[1:]:
        if start not in predicted_contexts:
            raise RuntimeError(
                f"No predicted context available for rollout window {start}"
            )
        raw = load_npz(str(paths[start]))
        prepared = _prepare_window_sample(args, raw)
        sample, robot_rollout, _ = _rollout_sample(
            prepared_window=prepared,
            scene_context=predicted_contexts[start],
            scene_colors=initial_colors,
            scene_normals=initial_normals,
            rollout_shift=rollout_shift,
            camera_payload=camera_payload,
        )
        prediction = _run_model(trainer, sample, args._trainer_args)
        _write_prediction_window(
            start=start,
            prediction_rollout_frame=prediction,
            robot_rollout_frame=robot_rollout,
            rollout_shift=rollout_shift,
            display_shift=display_shift,
            colors=static_colors,
            positions_out=positions,
            colors_out=colors,
            exists_out=exists,
            robot_out=robot,
            robot_exists_out=robot_exists,
            predicted_contexts=predicted_contexts,
            model_mask=model_mask,
            transition_mask=transition_mask,
        )
        print(f"[autoregressive_rollout] predicted window {start}:{start + 10}", flush=True)

    if not np.all(model_mask[args.gt_prefix_end + 1 :]):
        missing = np.flatnonzero(~model_mask[args.gt_prefix_end + 1 :]) + args.gt_prefix_end + 1
        raise RuntimeError(f"Rollout did not fill frames: {missing.tolist()}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        scene_positions=positions,
        scene_colors=colors,
        scene_exists=exists,
        robot_positions=robot,
        robot_exists=robot_exists,
        model_prediction_mask=model_mask,
        flow_transition_mask=transition_mask,
        gt_prefix_end=np.asarray(args.gt_prefix_end, dtype=np.int32),
        frame_indices=np.arange(positions.shape[0], dtype=np.int32),
    )
    metrics = {
        "mode": "experimental_autoregressive_scene_rollout",
        "gt_prefix_frames": f"0..{args.gt_prefix_end}",
        "predicted_frames": f"{args.gt_prefix_end + 1}..{positions.shape[0] - 1}",
        "window_count": len(required_starts),
        "scene_points": int(positions.shape[1]),
        "robot_condition": "recorded_gt_exogenous",
        "manifest": str(Path(args.manifest).resolve()),
        "checkpoint": str(args.model_path),
    }
    metrics_path = Path(args.out_metrics) if args.out_metrics else out_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"[autoregressive_rollout] wrote {out_path}")
    print(f"[autoregressive_rollout] wrote {metrics_path}")


if __name__ == "__main__":
    main()
