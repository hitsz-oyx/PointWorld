# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import numpy as np

from visualization.viser_flow.upsampling import build_voxel_assignment


def test_build_frame_matches_the_corresponding_timeline_frame() -> None:
    background_points = np.array(
        [[0.001, 0.0, 0.0], [0.009, 0.0, 0.0], [0.100, 0.0, 0.0]],
        dtype=np.float32,
    )
    background_colors = np.array(
        [[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.uint8
    )
    positions = np.array(
        [
            [[0.0, 0.0, 0.0]],
            [[0.020, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    exists = np.ones((2, 1), dtype=bool)
    assignment = build_voxel_assignment(
        background_points, background_colors, positions, exists, grid_size=0.02
    )

    timeline = assignment.build_timeline(positions, exists)
    frame_points, frame_colors = assignment.build_frame(positions[1], exists[1])
    expected_points, expected_colors = timeline.frame(1)

    np.testing.assert_allclose(frame_points, expected_points)
    np.testing.assert_array_equal(frame_colors, expected_colors)
