# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Pure geometry for fixed LIBERO export camera layouts."""

from __future__ import annotations

import numpy as np


CAMERA_LAYOUT_NATIVE = "native"
CAMERA_LAYOUT_OBLIQUE_PAIR = "oblique_pair"
CAMERA_LAYOUT_OBLIQUE_TRIPLET = "oblique_triplet"
OBLIQUE_CAMERA_NAMES = ("frontview", "sideview")
OBLIQUE_TRIPLET_CAMERA_NAMES = ("frontview", "sideview", "birdview")


def _normalize(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length <= 1e-8:
        raise ValueError("Camera look-at vectors must be non-zero")
    return vector / length


def look_at_rotation(camera_position: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return MuJoCo camera axes as a world-frame rotation matrix.

    MuJoCo renders along local ``-Z``. Therefore the local ``+Z`` axis points
    from the target back toward the camera. Local ``+Y`` is chosen as close to
    world up as possible, preventing an arbitrary image roll.
    """
    position = np.asarray(camera_position, dtype=np.float64)
    look_target = np.asarray(target, dtype=np.float64)
    back_axis = _normalize(position - look_target)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right_axis = np.cross(world_up, back_axis)
    if float(np.linalg.norm(right_axis)) <= 1e-8:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right_axis = np.cross(world_up, back_axis)
    right_axis = _normalize(right_axis)
    up_axis = _normalize(np.cross(back_axis, right_axis))
    rotation = np.column_stack((right_axis, up_axis, back_axis))
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise RuntimeError("Camera look-at rotation is not orthonormal")
    if float(np.linalg.det(rotation)) <= 0.0:
        raise RuntimeError("Camera look-at rotation must be right-handed")
    return rotation.astype(np.float32)


def rotation_matrix_to_quat_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Convert a right-handed rotation matrix to MuJoCo's ``wxyz`` quaternion."""
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"rotation must have shape (3, 3), got {matrix.shape}")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quat = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ],
            dtype=np.float64,
        )
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
        quat = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
            ],
            dtype=np.float64,
        )
    elif matrix[1, 1] > matrix[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
        quat = np.array(
            [
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
            ],
            dtype=np.float64,
        )
    else:
        scale = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
        quat = np.array(
            [
                (matrix[1, 0] - matrix[0, 1]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
            ],
            dtype=np.float64,
        )
    return (quat / np.linalg.norm(quat)).astype(np.float32)


def oblique_pair_poses() -> tuple[np.ndarray, np.ndarray]:
    """Return fixed left/right diagonal camera positions and orientations.

    Both cameras look at the tabletop center from elevation. Their horizontal
    directions are +60 and -60 degrees around the target, so the two views
    have a 120-degree azimuth separation.
    """
    target = np.array([0.0, 0.0, 0.82], dtype=np.float32)
    radius = np.float32(1.30)
    elevation = np.float32(0.85)
    positions = np.array(
        [
            [0.5 * radius, np.sqrt(3.0) * 0.5 * radius, target[2] + elevation],
            [0.5 * radius, -np.sqrt(3.0) * 0.5 * radius, target[2] + elevation],
        ],
        dtype=np.float32,
    )
    quaternions = np.stack(
        [rotation_matrix_to_quat_wxyz(look_at_rotation(position, target)) for position in positions],
        axis=0,
    )
    return positions, quaternions


def oblique_triplet_poses() -> tuple[np.ndarray, np.ndarray]:
    """Return two oblique cameras plus a nadir bird's-eye camera.

    The first two positions match :func:`oblique_pair_poses`; the third
    camera looks directly down at the workspace to fill their occlusions.
    """
    oblique_positions, oblique_quaternions = oblique_pair_poses()
    target = np.array([0.0, 0.0, 0.82], dtype=np.float32)
    birdview_position = np.array([0.0, 0.0, 2.62], dtype=np.float32)
    birdview_quaternion = rotation_matrix_to_quat_wxyz(
        look_at_rotation(birdview_position, target)
    )
    return (
        np.concatenate((oblique_positions, birdview_position[None]), axis=0),
        np.concatenate((oblique_quaternions, birdview_quaternion[None]), axis=0),
    )


__all__ = [
    "CAMERA_LAYOUT_NATIVE",
    "CAMERA_LAYOUT_OBLIQUE_PAIR",
    "CAMERA_LAYOUT_OBLIQUE_TRIPLET",
    "OBLIQUE_CAMERA_NAMES",
    "OBLIQUE_TRIPLET_CAMERA_NAMES",
    "look_at_rotation",
    "oblique_pair_poses",
    "oblique_triplet_poses",
    "rotation_matrix_to_quat_wxyz",
]
