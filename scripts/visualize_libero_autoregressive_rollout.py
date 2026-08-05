# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Button-only Viser viewer for an autoregressive LIBERO scene rollout."""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np
import viser

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", required=True, help="Rollout .npz output.")
    parser.add_argument("--viewer_host", default="0.0.0.0")
    parser.add_argument("--viewer_port", type=int, default=8096)
    parser.add_argument("--scene_point_size", type=float, default=0.0015)
    parser.add_argument("--flow_point_limit", type=int, default=300)
    return parser.parse_args()


def _empty_segments() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.empty((0, 2, 3), dtype=np.float32),
        np.empty((0, 2, 3), dtype=np.uint8),
    )


def _one_step_flow(
    *,
    frame: int,
    positions: np.ndarray,
    exists: np.ndarray,
    transition_mask: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    if frame >= positions.shape[0] - 1 or not bool(transition_mask[frame]):
        return _empty_segments()
    displacement = positions[frame + 1] - positions[frame]
    magnitude = np.linalg.norm(displacement, axis=1)
    valid = exists[frame] & exists[frame + 1] & np.isfinite(magnitude)
    indices = np.flatnonzero(valid & (magnitude >= 0.002))
    if indices.size == 0:
        return _empty_segments()
    if indices.size > max_points:
        order = np.argsort(magnitude[indices], kind="stable")[-max_points:]
        indices = indices[order]
    segments = np.stack(
        [positions[frame, indices], positions[frame + 1, indices]], axis=1
    ).astype(np.float32, copy=False)
    colors = np.broadcast_to(
        np.array([255, 170, 50], dtype=np.uint8), segments.shape
    ).copy()
    return segments, colors


def main() -> None:
    args = _parse_args()
    if args.scene_point_size <= 0.0:
        raise ValueError("--scene_point_size must be positive")
    if args.flow_point_limit < 1:
        raise ValueError("--flow_point_limit must be >= 1")

    with np.load(args.rollout, allow_pickle=False) as archive:
        positions = np.asarray(archive["scene_positions"], dtype=np.float32)
        colors = np.asarray(archive["scene_colors"], dtype=np.uint8)
        exists = np.asarray(archive["scene_exists"], dtype=bool)
        robot_positions = np.asarray(archive["robot_positions"], dtype=np.float32)
        robot_exists = np.asarray(archive["robot_exists"], dtype=bool)
        model_mask = np.asarray(archive["model_prediction_mask"], dtype=bool)
        transition_mask = np.asarray(archive["flow_transition_mask"], dtype=bool)
        gt_prefix_end = int(np.asarray(archive["gt_prefix_end"]).item())

    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(f"scene_positions must be (T,N,3), got {positions.shape}")
    if colors.shape != positions.shape or exists.shape != positions.shape[:2]:
        raise ValueError("Scene colors/existence masks do not match positions")
    if robot_positions.shape[:2] != robot_exists.shape or robot_positions.shape[-1] != 3:
        raise ValueError("Robot positions/existence masks are inconsistent")
    if model_mask.shape != (positions.shape[0],):
        raise ValueError("model_prediction_mask has an invalid shape")
    if transition_mask.shape != (positions.shape[0] - 1,):
        raise ValueError("flow_transition_mask has an invalid shape")

    server = viser.ViserServer(
        host=str(args.viewer_host),
        port=int(args.viewer_port),
        label="libero-autoregressive-rollout",
    )
    server.scene.world_axes.visible = False
    server.scene.enable_default_lights()
    server.gui.add_markdown(
        "## PointWorld autoregressive scene rollout\n"
        f"Frames 0..{gt_prefix_end} are recorded GT. Later scene frames are chained "
        "model predictions; only the robot trajectory remains a recorded action condition."
    )
    with server.gui.add_folder("Trajectory controls", expand_by_default=True):
        previous_button = server.gui.add_button("Previous frame")
        next_button = server.gui.add_button("Next frame")
        frame_text = server.gui.add_text(
            "Recorded frame", initial_value=f"0 / {positions.shape[0] - 1}", disabled=True
        )
        source_text = server.gui.add_text("Frame source", initial_value="", disabled=True)
        robot_toggle = server.gui.add_checkbox("Robot points", initial_value=True)
        flow_toggle = server.gui.add_checkbox("One-step flow", initial_value=True)
        point_size_slider = server.gui.add_slider(
            "Point size", min=0.0005, max=0.01, step=0.0005,
            initial_value=float(args.scene_point_size),
        )
        flow_limit_slider = server.gui.add_slider(
            "Flow point limit", min=50, max=1000, step=50,
            initial_value=min(max(int(args.flow_point_limit), 50), 1000),
        )

    empty_points = np.empty((0, 3), dtype=np.float32)
    empty_colors = np.empty((0, 3), dtype=np.uint8)
    scene_handle = server.scene.add_point_cloud(
        "scene/rollout",
        points=empty_points,
        colors=empty_colors,
        point_size=float(args.scene_point_size),
        point_shape="rounded",
        precision="float32",
    )
    robot_handle = server.scene.add_point_cloud(
        "robot/recorded_condition",
        points=empty_points,
        colors=empty_colors,
        point_size=float(args.scene_point_size) * 1.2,
        point_shape="rounded",
        precision="float32",
    )
    flow_points, flow_colors = _empty_segments()
    flow_handle = server.scene.add_line_segments(
        "scene/one_step_flow", points=flow_points, colors=flow_colors, line_width=2.5
    )

    state = {"position": 0}
    lock = threading.RLock()

    def _update() -> None:
        frame = int(state["position"])
        scene_mask = exists[frame]
        scene_handle.points = positions[frame, scene_mask]
        scene_handle.colors = colors[frame, scene_mask]
        point_size = float(point_size_slider.value)
        scene_handle.point_size = point_size
        robot_handle.point_size = point_size * 1.2

        robot_mask = robot_exists[frame]
        robot_handle.points = robot_positions[frame, robot_mask]
        robot_handle.colors = np.full(
            (int(robot_mask.sum()), 3), [230, 90, 80], dtype=np.uint8
        )
        robot_handle.visible = bool(robot_toggle.value)

        segments, segment_colors = _one_step_flow(
            frame=frame,
            positions=positions,
            exists=exists,
            transition_mask=transition_mask,
            max_points=int(flow_limit_slider.value),
        )
        flow_handle.points = segments
        flow_handle.colors = segment_colors
        flow_handle.visible = bool(flow_toggle.value) and segments.shape[0] > 0

        frame_text.value = f"{frame} / {positions.shape[0] - 1}"
        source_text.value = (
            "Model autoregressive prediction"
            if bool(model_mask[frame])
            else "Recorded GT prefix"
        )

    def _step(delta: int) -> None:
        with lock:
            old = int(state["position"])
            new = min(max(old + int(delta), 0), positions.shape[0] - 1)
            if new != old:
                state["position"] = new
                _update()

    def _control_changed(_event: object) -> None:
        with lock:
            _update()

    previous_button.on_click(lambda _event: _step(-1))
    next_button.on_click(lambda _event: _step(1))
    for control in (robot_toggle, flow_toggle, point_size_slider, flow_limit_slider):
        control.on_update(_control_changed)

    _update()
    print(
        f"[autoregressive_viz] frames 0..{positions.shape[0] - 1}, "
        f"GT prefix 0..{gt_prefix_end}"
    )
    print(f"[autoregressive_viz] Live viewer running at http://127.0.0.1:{args.viewer_port}")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
