# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tools.libero.trajectory import (
    build_trajectory_frame_map,
    newly_covered_local_indices,
    plan_trajectory_windows,
)


def test_plan_trajectory_windows_covers_tail() -> None:
    windows = plan_trajectory_windows(
        start_idx=60, end_idx=154, window_size=11, stride=10
    )

    assert [(w.start_idx, w.end_idx) for w in windows] == [
        (60, 70),
        (70, 80),
        (80, 90),
        (90, 100),
        (100, 110),
        (110, 120),
        (120, 130),
        (130, 140),
        (140, 150),
        (144, 154),
    ]


def test_newly_covered_indices_emit_each_global_frame_once() -> None:
    windows = plan_trajectory_windows(
        start_idx=0, end_idx=24, window_size=11, stride=10
    )
    covered = -1
    emitted = []
    for window in windows:
        local_indices = newly_covered_local_indices(
            window, covered_through=covered
        )
        emitted.extend(window.start_idx + local for local in local_indices)
        if local_indices:
            covered = window.start_idx + max(local_indices)

    assert emitted == list(range(25))


def test_trajectory_frame_map_keeps_each_frame_in_its_source_window() -> None:
    windows = plan_trajectory_windows(
        start_idx=60, end_idx=84, window_size=11, stride=10
    )

    frame_map = build_trajectory_frame_map(windows)

    assert list(frame_map) == list(range(60, 85))
    assert frame_map[60] == (0, 0)
    assert frame_map[70] == (0, 10)
    assert frame_map[71] == (1, 1)
    assert frame_map[80] == (1, 10)
    assert frame_map[81] == (2, 7)
    assert frame_map[84] == (2, 10)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"start_idx": 0, "end_idx": 9, "window_size": 11, "stride": 1}, "shorter"),
        ({"start_idx": 0, "end_idx": 10, "window_size": 11, "stride": 0}, "stride"),
    ],
)
def test_plan_trajectory_windows_rejects_invalid_ranges(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        plan_trajectory_windows(**kwargs)
