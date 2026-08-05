# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from visualization.viser_flow.timeline import build_rainbow_flow_timeline


def test_transition_mask_breaks_flow_at_window_boundary() -> None:
    positions = np.array(
        [[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]],
         [[100.0, 0.0, 0.0]], [[101.0, 0.0, 0.0]]],
        dtype=np.float32,
    )
    exists = np.ones((4, 1), dtype=bool)
    transitions = np.array([[True], [False], [True]], dtype=bool)

    timeline = build_rainbow_flow_timeline(
        positions,
        exists,
        colormap=lambda values: np.stack([values, values, values, values], axis=-1),
        min_brightness=1.0,
        transition_mask=transitions,
    )

    assert timeline.segments.shape == (2, 2, 3)
    np.testing.assert_allclose(timeline.segments[:, :, 0], [[0.0, 1.0], [100.0, 101.0]])
