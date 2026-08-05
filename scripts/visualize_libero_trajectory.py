# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Viser viewer for a long LIBERO trajectory evaluated in fixed windows.

PointWorld predicts a fixed 11-frame point-flow window. A longer LIBERO demo
therefore has no valid global point identity or RGB-D background. This viewer
keeps every window independent: a global frame selects one source window, and
that window's own centered camera data is used for RGB-D upsampling.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import viser

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.libero.trajectory import TrajectoryWindow, build_trajectory_frame_map
from visualization.viser_flow.upsampling import VoxelAssignment, build_voxel_assignment
from visualization.viser_tools.visualization_utils import (
    CameraObservation,
    merge_camera_point_cloud,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window_manifest",
        required=True,
        help="window_manifest.json produced by scripts/eval_libero_trajectory.py",
    )
    parser.add_argument("--viewer_host", default="0.0.0.0")
    parser.add_argument("--viewer_port", type=int, default=8093)
    parser.add_argument(
        "--scene_point_size",
        type=float,
        default=0.0015,
        help="Rendered point size in meters (default: 0.0015).",
    )
    parser.add_argument(
        "--dense_grid_size",
        type=float,
        default=0.015,
        help="Voxel size used by PointWorld's RGB-D upsampling path.",
    )
    return parser.parse_args()


def _array(archive: np.lib.npyio.NpzFile, key: str, *, dtype: np.dtype) -> np.ndarray:
    if key not in archive:
        raise KeyError(f"Window payload is missing required field {key!r}")
    return np.asarray(archive[key], dtype=dtype)


@dataclass
class WindowPayload:
    path: Path
    start_idx: int
    end_idx: int
    pred: np.ndarray
    gt: np.ndarray
    colors: np.ndarray
    scene_exists: np.ndarray
    supervised: np.ndarray
    robot: np.ndarray
    robot_exists: np.ndarray
    cameras: list[CameraObservation]

    @classmethod
    def load(cls, path: Path, *, start_idx: int, end_idx: int) -> "WindowPayload":
        with np.load(path, allow_pickle=False) as archive:
            pred = _array(archive, "pred_scene_flows", dtype=np.float32)
            gt = _array(archive, "gt_scene_flows", dtype=np.float32)
            colors = _array(archive, "scene_colors", dtype=np.uint8)
            scene_exists = _array(archive, "scene_exists", dtype=bool)
            supervised = _array(archive, "scene_supervised_mask", dtype=bool)
            robot = _array(archive, "robot_flows", dtype=np.float32)
            robot_exists = _array(archive, "robot_exists", dtype=bool)
            cameras = _load_cameras(archive)

        expected_frames = end_idx - start_idx + 1
        if pred.shape != gt.shape or pred.ndim != 3 or pred.shape[-1] != 3:
            raise ValueError(f"Invalid prediction/GT shape in {path}: {pred.shape}, {gt.shape}")
        if pred.shape[0] != expected_frames:
            raise ValueError(
                f"{path} covers {pred.shape[0]} frames, expected {expected_frames}"
            )
        if colors.shape != pred.shape:
            raise ValueError(f"scene_colors shape mismatch in {path}: {colors.shape}")
        if scene_exists.shape != pred.shape[:2]:
            raise ValueError(f"scene_exists shape mismatch in {path}: {scene_exists.shape}")
        if supervised.shape != pred.shape[:2]:
            raise ValueError(f"scene_supervised_mask shape mismatch in {path}")
        if robot.ndim != 3 or robot.shape[-1] != 3:
            raise ValueError(f"robot_flows must have shape (T,N,3) in {path}")
        if robot_exists.shape != robot.shape[:2]:
            raise ValueError(f"robot_exists shape mismatch in {path}")
        return cls(
            path=path,
            start_idx=start_idx,
            end_idx=end_idx,
            pred=pred,
            gt=gt,
            colors=colors,
            scene_exists=scene_exists,
            supervised=supervised,
            robot=robot,
            robot_exists=robot_exists,
            cameras=cameras,
        )


def _load_cameras(archive: np.lib.npyio.NpzFile) -> list[CameraObservation]:
    prefixes = sorted(
        key[: -len("_initial_rgb")]
        for key in archive.files
        if key.endswith("_initial_rgb")
    )
    cameras: list[CameraObservation] = []
    for prefix in prefixes:
        fields = {
            "rgb": f"{prefix}_initial_rgb",
            "depth": f"{prefix}_initial_depth",
            "intrinsic": f"{prefix}_intrinsic",
            "extrinsic": f"{prefix}_extrinsic",
        }
        missing = [key for key in fields.values() if key not in archive]
        if missing:
            raise KeyError(f"Camera {prefix!r} missing fields {missing}")
        cameras.append(
            CameraObservation(
                name=prefix,
                rgb=np.asarray(archive[fields["rgb"]]),
                depth=np.asarray(archive[fields["depth"]], dtype=np.float32),
                intrinsic=np.asarray(archive[fields["intrinsic"]], dtype=np.float32),
                extrinsic_world_to_cam=np.asarray(
                    archive[fields["extrinsic"]], dtype=np.float32
                ),
            )
        )
    if not cameras:
        raise ValueError("Window payload contains no camera RGB-D data")
    return cameras


def _dense_assignment(payload: WindowPayload, *, grid_size: float) -> VoxelAssignment:
    valid_gt = payload.gt[payload.scene_exists]
    if valid_gt.size == 0:
        raise ValueError(f"No valid scene points in {payload.path}")
    margin = np.float32(0.04)
    bounds_min = valid_gt.min(axis=0) - margin
    bounds_max = valid_gt.max(axis=0) + margin
    background_points, background_colors = merge_camera_point_cloud(
        payload.cameras,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        include_out_of_bounds=True,
    )
    if background_points.size == 0:
        raise ValueError(f"RGB-D projection produced no points for {payload.path}")
    assignment: VoxelAssignment = build_voxel_assignment(
        background_points,
        background_colors,
        payload.gt,
        payload.scene_exists,
        grid_size=grid_size,
    )
    return assignment


def _dense_frame(
    assignment: VoxelAssignment,
    payload: WindowPayload,
    *,
    local_idx: int,
    use_gt: bool,
) -> tuple[np.ndarray, np.ndarray]:
    positions = payload.gt if use_gt else payload.pred
    return assignment.build_frame(
        positions[local_idx],
        payload.scene_exists[local_idx],
        supervised=payload.supervised[local_idx],
    )


def _coarse_frame(
    payload: WindowPayload, *, local_idx: int, use_gt: bool
) -> tuple[np.ndarray, np.ndarray]:
    mask = payload.scene_exists[local_idx]
    positions = payload.gt if use_gt else payload.pred
    return positions[local_idx, mask], payload.colors[local_idx, mask]


def _one_step_segments(
    payload: WindowPayload,
    *,
    local_idx: int,
    use_gt: bool,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Show only the current valid one-step flow, never an accumulated trail."""
    if local_idx >= payload.gt.shape[0] - 1:
        return _empty_segments()
    gt_step = payload.gt[local_idx + 1] - payload.gt[local_idx]
    motion = np.linalg.norm(gt_step, axis=1)
    valid = (
        payload.scene_exists[local_idx]
        & payload.scene_exists[local_idx + 1]
        & np.isfinite(motion)
        & (motion >= 0.002)
    )
    indices = np.flatnonzero(valid)
    if indices.size == 0:
        return _empty_segments()
    if indices.size > max_points:
        # Stable ranking makes a given window look identical after revisiting it.
        order = np.argsort(motion[indices], kind="stable")[-max_points:]
        indices = indices[order]
    positions = payload.gt if use_gt else payload.pred
    segments = np.stack(
        [positions[local_idx, indices], positions[local_idx + 1, indices]], axis=1
    ).astype(np.float32, copy=False)
    color = np.array([60, 180, 255] if use_gt else [255, 170, 50], dtype=np.uint8)
    colors = np.broadcast_to(color, segments.shape).copy()
    return segments, colors


def _prediction_error_segments(
    payload: WindowPayload,
    *,
    local_idx: int,
    scale: float,
    max_points: int = 300,
    max_length: float = 0.08,
) -> tuple[np.ndarray, np.ndarray]:
    """Build capped, magnified GT-to-prediction arrows for visual comparison."""
    if scale <= 0.0:
        return _empty_segments()
    errors = payload.pred[local_idx] - payload.gt[local_idx]
    magnitudes = np.linalg.norm(errors, axis=1)
    valid = payload.scene_exists[local_idx] & np.isfinite(magnitudes)
    indices = np.flatnonzero(valid & (magnitudes > 1e-7))
    if indices.size == 0:
        return _empty_segments()
    if indices.size > max_points:
        order = np.argsort(magnitudes[indices], kind="stable")[-max_points:]
        indices = indices[order]
    offsets = errors[indices] * float(scale)
    lengths = np.linalg.norm(offsets, axis=1)
    clip_scale = np.minimum(1.0, float(max_length) / np.maximum(lengths, 1e-8))
    endpoints = payload.gt[local_idx, indices] + offsets * clip_scale[:, None]
    segments = np.stack(
        [payload.gt[local_idx, indices], endpoints], axis=1
    ).astype(np.float32, copy=False)
    colors = np.broadcast_to(
        np.array([255, 40, 200], dtype=np.uint8), segments.shape
    ).copy()
    return segments, colors


def _empty_segments() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.empty((0, 2, 3), dtype=np.float32),
        np.empty((0, 2, 3), dtype=np.uint8),
    )


def _load_manifest(path: Path) -> list[WindowPayload]:
    manifest = json.loads(path.read_text())
    raw_windows = manifest.get("windows")
    if not isinstance(raw_windows, list) or not raw_windows:
        raise ValueError(f"{path} does not contain any window payloads")
    payloads: list[WindowPayload] = []
    for entry in raw_windows:
        if not isinstance(entry, dict):
            raise ValueError(f"Malformed window entry: {entry!r}")
        try:
            start_idx = int(entry["start_idx"])
            end_idx = int(entry["end_idx"])
            output = Path(str(entry["output"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed window entry: {entry!r}") from exc
        if not output.is_absolute():
            output = path.parent / output
        if not output.is_file():
            raise FileNotFoundError(f"Window payload not found: {output}")
        payloads.append(WindowPayload.load(output, start_idx=start_idx, end_idx=end_idx))
    return sorted(payloads, key=lambda payload: payload.start_idx)


def main() -> None:
    args = _parse_args()
    if args.scene_point_size <= 0.0:
        raise ValueError("--scene_point_size must be positive")
    if args.dense_grid_size <= 0.0:
        raise ValueError("--dense_grid_size must be positive")
    manifest_path = Path(args.window_manifest).resolve()
    payloads = _load_manifest(manifest_path)
    windows = [
        TrajectoryWindow(payload.start_idx, payload.end_idx) for payload in payloads
    ]
    frame_map = build_trajectory_frame_map(windows)
    frame_indices = sorted(frame_map)
    if frame_indices != list(range(frame_indices[0], frame_indices[-1] + 1)):
        raise ValueError("Window manifest does not form a contiguous trajectory")

    server = viser.ViserServer(
        host=str(args.viewer_host), port=int(args.viewer_port), label="libero-trajectory"
    )
    server.scene.world_axes.visible = False
    server.scene.enable_default_lights()
    server.gui.add_markdown(
        "## PointWorld LIBERO trajectory\n"
        "Each frame uses its own fixed 11-frame source window and RGB-D background. "
        "Flow lines are one step only, so window boundaries cannot create long trails. "
        "Use the frame buttons for single-frame steps or the slider to jump."
    )
    with server.gui.add_folder("Trajectory controls", expand_by_default=True):
        previous_frame_button = server.gui.add_button("Previous frame")
        next_frame_button = server.gui.add_button("Next frame")
        frame_slider = server.gui.add_slider(
            "Frame",
            min=frame_indices[0],
            max=frame_indices[-1],
            step=1,
            initial_value=frame_indices[0],
        )
        frame_text = server.gui.add_text(
            "Recorded frame",
            initial_value=f"{frame_indices[0]} / {frame_indices[-1]}",
            disabled=True,
        )
        gt_toggle = server.gui.add_checkbox("Ground-truth", initial_value=False)
        dense_toggle = server.gui.add_checkbox("Dense RGB-D", initial_value=True)
        flow_toggle = server.gui.add_checkbox("One-step flow", initial_value=True)
        robot_toggle = server.gui.add_checkbox("Robot points", initial_value=True)
        point_size_slider = server.gui.add_slider(
            "Point size", min=0.0005, max=0.01, step=0.0005,
            initial_value=float(args.scene_point_size),
        )
        flow_count_slider = server.gui.add_slider(
            "Flow point limit", min=50, max=1000, step=50, initial_value=300
        )
        error_scale_slider = server.gui.add_slider(
            "Pred-GT error x (0=off)",
            min=0.0,
            max=100.0,
            step=5.0,
            # Keep the diagnostic overlay opt-in: its magnified errors can be
            # mistaken for scene flow, especially on static background points.
            initial_value=0.0,
        )
        source_text = server.gui.add_text("Source window", initial_value="", disabled=True)

    empty_points = np.empty((0, 3), dtype=np.float32)
    empty_colors = np.empty((0, 3), dtype=np.uint8)
    coarse_handle = server.scene.add_point_cloud(
        "scene/coarse", points=empty_points, colors=empty_colors,
        point_size=float(args.scene_point_size), point_shape="rounded", precision="float32",
    )
    dense_handle = server.scene.add_point_cloud(
        "scene/dense", points=empty_points, colors=empty_colors,
        point_size=float(args.scene_point_size), point_shape="rounded", precision="float32",
    )
    robot_handle = server.scene.add_point_cloud(
        "robot/points", points=empty_points, colors=empty_colors,
        point_size=float(args.scene_point_size) * 1.2, point_shape="rounded", precision="float32",
    )
    flow_points, flow_colors = _empty_segments()
    flow_handle = server.scene.add_line_segments(
        "scene/one_step_flow", points=flow_points, colors=flow_colors, line_width=2.5
    )
    error_points, error_colors = _empty_segments()
    error_handle = server.scene.add_line_segments(
        "scene/prediction_error",
        points=error_points,
        colors=error_colors,
        line_width=2.5,
    )

    dense_cache: dict[str, object] = {
        "window_idx": None,
        "assignment": None,
    }
    render_lock = threading.RLock()
    frame_state = {"position": 0}

    def _update() -> None:
        frame = frame_indices[int(frame_state["position"])]
        window_idx, local_idx = frame_map[frame]
        payload = payloads[window_idx]
        use_gt = bool(gt_toggle.value)
        use_dense = bool(dense_toggle.value)

        branch_label = "GT" if use_gt else "prediction"
        frame_text.value = f"{frame} / {frame_indices[-1]}"
        source_text.value = (
            f"{branch_label}, frame {frame}: window {payload.start_idx}:{payload.end_idx}, "
            f"local {local_idx}"
        )
        if use_dense:
            if dense_cache["window_idx"] != window_idx:
                dense_cache["assignment"] = _dense_assignment(
                    payload, grid_size=float(args.dense_grid_size)
                )
                dense_cache["window_idx"] = window_idx
            assignment = dense_cache["assignment"]
            if not isinstance(assignment, VoxelAssignment):
                raise RuntimeError("Dense voxel assignment cache was not initialized")
            points, colors = _dense_frame(
                assignment, payload, local_idx=local_idx, use_gt=use_gt
            )
            dense_handle.points = points.astype(np.float32, copy=False)
            dense_handle.colors = colors.astype(np.uint8, copy=False)
        else:
            points, colors = _coarse_frame(
                payload, local_idx=local_idx, use_gt=use_gt
            )
            coarse_handle.points = points.astype(np.float32, copy=False)
            coarse_handle.colors = colors.astype(np.uint8, copy=False)

        coarse_handle.visible = not use_dense
        dense_handle.visible = use_dense
        point_size = float(point_size_slider.value)
        coarse_handle.point_size = point_size
        dense_handle.point_size = point_size
        robot_handle.point_size = point_size * 1.2

        robot_mask = payload.robot_exists[local_idx]
        robot_handle.points = payload.robot[local_idx, robot_mask]
        robot_handle.colors = np.full(
            (int(robot_mask.sum()), 3), [230, 90, 80], dtype=np.uint8
        )
        robot_handle.visible = bool(robot_toggle.value)

        segments, segment_colors = _one_step_segments(
            payload,
            local_idx=local_idx,
            use_gt=use_gt,
            max_points=int(flow_count_slider.value),
        )
        flow_handle.points = segments
        flow_handle.colors = segment_colors
        flow_handle.visible = bool(flow_toggle.value) and segments.shape[0] > 0

        error_segments, error_segment_colors = _prediction_error_segments(
            payload,
            local_idx=local_idx,
            scale=float(error_scale_slider.value),
        )
        error_handle.points = error_segments
        error_handle.colors = error_segment_colors
        error_handle.visible = error_segments.shape[0] > 0

    def _step_frame(delta: int) -> None:
        with render_lock:
            old_position = int(frame_state["position"])
            new_position = min(
                max(old_position + int(delta), 0), len(frame_indices) - 1
            )
            if new_position == old_position:
                return
            frame_state["position"] = new_position
            frame_slider.value = frame_indices[new_position]
            _update()

    def _frame_changed(event: object) -> None:
        requested_frame = int(event.target.value)
        with render_lock:
            frame_state["position"] = requested_frame - frame_indices[0]
            _update()

    def _control_changed(_event: object) -> None:
        with render_lock:
            _update()

    previous_frame_button.on_click(lambda _event: _step_frame(-1))
    next_frame_button.on_click(lambda _event: _step_frame(1))
    frame_slider.on_update(_frame_changed)
    for control in (
        gt_toggle,
        dense_toggle,
        flow_toggle,
        robot_toggle,
        point_size_slider,
        flow_count_slider,
        error_scale_slider,
    ):
        control.on_update(_control_changed)

    _update()
    print(
        f"[trajectory_viz] {len(payloads)} independent windows, "
        f"frames {frame_indices[0]}..{frame_indices[-1]}"
    )
    print(f"[trajectory_viz] Live viewer running at http://127.0.0.1:{args.viewer_port}")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
