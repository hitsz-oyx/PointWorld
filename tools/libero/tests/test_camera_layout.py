# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import numpy as np

from tools.libero.camera_layout import (
    look_at_rotation,
    oblique_pair_poses,
    oblique_triplet_poses,
)


def test_oblique_pair_has_120_degree_horizontal_separation() -> None:
    positions, quaternions = oblique_pair_poses()
    target = np.array([0.0, 0.0, 0.82], dtype=np.float32)
    offsets = positions[:, :2] - target[:2]
    directions = offsets / np.linalg.norm(offsets, axis=1, keepdims=True)

    assert positions.shape == (2, 3)
    assert quaternions.shape == (2, 4)
    assert np.allclose(np.dot(directions[0], directions[1]), -0.5, atol=1e-6)
    assert np.allclose(np.linalg.norm(quaternions, axis=1), 1.0, atol=1e-6)


def test_look_at_rotation_points_mujoco_negative_z_at_target() -> None:
    position = np.array([0.65, 1.1258, 1.67], dtype=np.float32)
    target = np.array([0.0, 0.0, 0.82], dtype=np.float32)
    rotation = look_at_rotation(position, target)
    rendered_forward = -rotation[:, 2]
    expected_forward = (target - position) / np.linalg.norm(target - position)

    assert np.allclose(rendered_forward, expected_forward, atol=1e-6)


def test_oblique_triplet_adds_nadir_birdview() -> None:
    positions, quaternions = oblique_triplet_poses()
    target = np.array([0.0, 0.0, 0.82], dtype=np.float32)
    rotation = look_at_rotation(positions[2], target)

    assert positions.shape == (3, 3)
    assert quaternions.shape == (3, 4)
    assert np.allclose(positions[2, :2], 0.0)
    assert positions[2, 2] > target[2]
    assert np.allclose(-rotation[:, 2], [0.0, 0.0, -1.0], atol=1e-6)
