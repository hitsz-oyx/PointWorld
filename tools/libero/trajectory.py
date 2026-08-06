# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Window planning helpers for fixed-horizon LIBERO evaluation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrajectoryWindow:
    """An inclusive fixed-size model window over raw demo indices."""

    start_idx: int
    end_idx: int
    frame_step: int = 1


def plan_trajectory_windows(
    *,
    start_idx: int,
    end_idx: int,
    window_size: int,
    stride: int,
    frame_step: int = 1,
) -> list[TrajectoryWindow]:
    """Cover ``[start_idx, end_idx]`` with fixed windows.

    The final window is always anchored at the last valid start so the tail is
    covered even when ``stride`` does not divide the trajectory length.
    """
    if window_size < 2:
        raise ValueError(f"window_size must be >= 2, got {window_size}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if frame_step < 1:
        raise ValueError(f"frame_step must be >= 1, got {frame_step}")
    if start_idx < 0:
        raise ValueError(f"start_idx must be >= 0, got {start_idx}")
    if end_idx < start_idx:
        raise ValueError(
            f"end_idx must be >= start_idx, got {end_idx} < {start_idx}"
        )

    window_span = (window_size - 1) * frame_step
    last_start = end_idx - window_span
    if last_start < start_idx:
        raise ValueError(
            f"Trajectory [{start_idx}, {end_idx}] is shorter than "
            f"window_size={window_size}, frame_step={frame_step}."
        )

    raw_stride = stride * frame_step
    starts = list(range(start_idx, last_start + 1, raw_stride))
    if starts[-1] != last_start:
        starts.append(last_start)
    return [
        TrajectoryWindow(start_idx=s, end_idx=s + window_span, frame_step=frame_step)
        for s in starts
    ]


def newly_covered_local_indices(
    window: TrajectoryWindow,
    *,
    covered_through: int,
) -> range:
    """Return local frame indices not already emitted by earlier windows."""
    if window.frame_step < 1:
        raise ValueError(f"window frame_step must be >= 1, got {window.frame_step}")
    model_steps = (window.end_idx - window.start_idx) // window.frame_step
    raw_offset = max(0, int(covered_through) + 1 - window.start_idx)
    local_start = (raw_offset + window.frame_step - 1) // window.frame_step
    if local_start > model_steps:
        return range(0, 0)
    return range(local_start, model_steps + 1)


def build_trajectory_frame_map(
    windows: list[TrajectoryWindow],
) -> dict[int, tuple[int, int]]:
    """Map each emitted global frame to its source window and local frame.

    This is intentionally the same first-window-wins policy used for the
    stitched metrics artifact. The viewer uses the mapping only to select a
    local 11-frame payload; it never treats points from different windows as
    one continuous point track.
    """
    if not windows:
        raise ValueError("At least one trajectory window is required")

    frame_map: dict[int, tuple[int, int]] = {}
    covered_through = min(window.start_idx for window in windows) - 1
    for window_idx, window in enumerate(sorted(windows, key=lambda item: item.start_idx)):
        for local_idx in newly_covered_local_indices(
            window, covered_through=covered_through
        ):
            raw_idx = window.start_idx + local_idx * window.frame_step
            frame_map[raw_idx] = (window_idx, local_idx)
        covered_through = max(covered_through, window.end_idx)

    if not frame_map:
        raise ValueError("Trajectory windows did not cover any frames")
    return frame_map


__all__ = [
    "TrajectoryWindow",
    "build_trajectory_frame_map",
    "newly_covered_local_indices",
    "plan_trajectory_windows",
]
