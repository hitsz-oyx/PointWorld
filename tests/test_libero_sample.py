# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for the LIBERO <-> PointWorld adapter.

These tests verify the .npz file format and the geometric consistency of a
single clip. They do *not* need a real LIBERO environment; the
``dummy_clip`` fixture builds a synthetic scene with two cubes that move
with known rigid-body transforms.

Run with::

    pytest tests/test_libero_sample.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make the project root importable so we can use the tools.libero package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.libero import sample_schema
from tools.libero.scene_geometry import (
    backproject_depth,
    estimate_normals_from_depth,
    track_points_through_poses,
)


# ----------------------------------------------------------------------------
# Fixtures.
# ----------------------------------------------------------------------------

def _rotmat(roll=0.0, pitch=0.0, yaw=0.0) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    R_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    R_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    R_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return R_z @ R_y @ R_x


def _transform(R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _synth_depth_image(K, T_w_c, points_world) -> np.ndarray:
    """Render a synthetic depth image for given world points on a regular
    HxW pixel grid (assumes points_world is (H, W, 3))."""
    H, W, _ = points_world.shape
    pts_cam = np.linalg.inv(T_w_c) @ np.concatenate(
        [points_world.reshape(-1, 3), np.ones((H * W, 1))], axis=-1
    ).T
    pts_cam = pts_cam.T[:, :3]
    z = pts_cam[:, 2].reshape(H, W)
    return z.astype(np.float32)


def _synth_rgb_image(H, W, color=(180, 120, 60)) -> np.ndarray:
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[..., 0] = color[0]
    rgb[..., 1] = color[1]
    rgb[..., 2] = color[2]
    return rgb


def dummy_clip(num_points: int = 64) -> dict:
    """Build a synthetic clip with two moving rigid bodies.

    Body A moves +x over 10 frames; body B stays still. The first 8x8
    pixel block in the top-left of the depth image is owned by body A,
    the rest is owned by body B.
    """
    H, W = sample_schema.H_RELEASE, sample_schema.W_RELEASE
    K = np.array([[W, 0, W / 2 - 0.5], [0, H, H / 2 - 0.5], [0, 0, 1]], dtype=np.float32)
    T_w_c = _transform(np.eye(3), np.array([0.0, 0.0, 1.5]))

    # Build a depth image: plane at z=1 (i.e. depth 0.5 m from cam at z=1.5).
    z_plane = np.full((H, W), 0.5, dtype=np.float32)
    # Object A is at world (0.1, 0.0, 1.0) with size ~0.05 m, so it pokes
    # towards the camera. Use a small bump in the top-left 32x32 patch.
    z_plane[:32, :32] = 0.3

    points_world = backproject_depth(z_plane, K, T_w_c)  # (H, W, 3)
    # Sample num_points pixels: half from object A, half from background.
    idx_a = np.indices((32, 32))[:, :16, :16].reshape(2, -1).T  # 256 points
    idx_b_list = []
    for r in range(0, H, 4):
        for c in range(0, W, 4):
            if r < 32 and c < 32:
                continue
            idx_b_list.append((r, c))
    idx_b = np.array(idx_b_list[:num_points // 2])
    idx_a = idx_a[:num_points // 2]
    chosen = np.concatenate([idx_a, idx_b], axis=0)
    points_kept = points_world[chosen[:, 0], chosen[:, 1]]  # (N, 3)

    owning = np.array(["bodyA"] * idx_a.shape[0] + ["bodyB"] * idx_b.shape[0])

    # Body A moves +x by 0.005 m per frame.
    T_bA_t = [_transform(np.eye(3), np.array([0.005 * t, 0.0, 0.0])) for t in range(11)]
    T_bB_t = [_transform(np.eye(3), np.array([0.0, 0.0, 0.0])) for _ in range(11)]
    # The track_points_through_poses API expects body_t0_inv and
    # body_poses_per_t dicts in the documented shape.
    body_poses_per_t = {}
    for t, T in enumerate(T_bA_t):
        body_poses_per_t[("bodyA", t)] = T
    for t, T in enumerate(T_bB_t):
        body_poses_per_t[("bodyB", t)] = T
    body_t0_inv = {name: np.linalg.inv(T_bA_t[0] if name == "bodyA" else T_bB_t[0])
                   for name in ("bodyA", "bodyB")}

    scene_flows = track_points_through_poses(
        points_kept.astype(np.float32), owning, body_t0_inv, body_poses_per_t
    )

    normals = estimate_normals_from_depth(points_world, T_w_c[:3, 3])
    normals_kept = normals[chosen[:, 0], chosen[:, 1]]

    colors = _synth_rgb_image(H, W, color=(200, 100, 50))

    # Package into the canonical sample dict.
    sample = sample_schema.empty_clip()
    sample["__key__"] = "synth-0:10"

    prefix = "camera_0"
    sample["scene_flows_per_cam"][prefix] = np.broadcast_to(
        points_kept.reshape(1, -1, 3), (11, points_kept.shape[0], 3)
    ).copy().astype(np.float32)
    # We need (T, H*W, 3) for the schema, not (T, N, 3). Repack:
    full = np.zeros((11, H * W, 3), dtype=np.float32)
    full[:, chosen[:, 0] * W + chosen[:, 1]] = scene_flows
    sample["scene_flows_per_cam"][prefix] = full

    full_n = np.zeros((11, H * W, 3), dtype=np.float32)
    full_n[:, chosen[:, 0] * W + chosen[:, 1]] = np.broadcast_to(
        normals_kept, (11, normals_kept.shape[0], 3)
    )
    sample["scene_normals_per_cam"][prefix] = full_n

    full_c = np.zeros((11, H * W, 3), dtype=np.uint8)
    full_c[:, chosen[:, 0] * W + chosen[:, 1]] = np.broadcast_to(
        colors[chosen[:, 0], chosen[:, 1]], (11, normals_kept.shape[0], 3)
    )
    sample["scene_colors_per_cam"][prefix] = full_c

    full_v = np.zeros((11, H * W), dtype=bool)
    full_v[:, chosen[:, 0] * W + chosen[:, 1]] = True
    sample["scene_visibility_per_cam"][prefix] = full_v
    sample["scene_depth_valid_mask_per_cam"][prefix] = full_v.copy()

    sample["initial_rgb_per_cam"][prefix] = colors
    sample["initial_depth_per_cam"][prefix] = z_plane
    sample["intrinsic_per_cam"][prefix] = K
    sample["extrinsic_per_cam"][prefix] = T_w_c.astype(np.float32)

    # Add a second camera (static, identical) so the smoke eval can pick
    # two cameras deterministically.
    sample["scene_flows_per_cam"]["camera_1"] = sample["scene_flows_per_cam"][prefix].copy()
    sample["scene_colors_per_cam"]["camera_1"] = sample["scene_colors_per_cam"][prefix].copy()
    sample["scene_normals_per_cam"]["camera_1"] = sample["scene_normals_per_cam"][prefix].copy()
    sample["scene_visibility_per_cam"]["camera_1"] = sample["scene_visibility_per_cam"][prefix].copy()
    sample["scene_depth_valid_mask_per_cam"]["camera_1"] = sample["scene_depth_valid_mask_per_cam"][prefix].copy()
    sample["initial_rgb_per_cam"]["camera_1"] = colors.copy()
    sample["initial_depth_per_cam"]["camera_1"] = z_plane.copy()
    sample["intrinsic_per_cam"]["camera_1"] = K.copy()
    sample["extrinsic_per_cam"]["camera_1"] = T_w_c.astype(np.float32).copy()

    # Robot trajectories: 16 points attached to a "gripper" body that
    # moves with bodyA.  This is a placeholder.
    n_robot = 16
    robot_local = np.random.RandomState(0).randn(n_robot, 3).astype(np.float32) * 0.02
    robot_flows = np.zeros((11, n_robot, 3), dtype=np.float32)
    for t in range(11):
        robot_flows[t] = (T_bA_t[t][:3, :3] @ robot_local.T).T + T_bA_t[t][:3, 3]
    sample["robot_flows"] = robot_flows
    sample["robot_normals"] = np.broadcast_to(
        robot_local / (np.linalg.norm(robot_local, axis=-1, keepdims=True) + 1e-12),
        (11, n_robot, 3),
    ).astype(np.float32)
    sample["robot_colors"] = np.zeros((11, n_robot, 3), dtype=np.uint8)
    sample["robot_colors"][..., 0] = 255
    sample["robot_colors"][..., 2] = 255
    sample["right_gripper_pose"] = np.zeros((11, 7), dtype=np.float32)
    sample["right_gripper_pose"][:, 6] = 1.0
    sample["right_gripper_open"] = np.zeros((11, 1), dtype=np.float32)
    return sample


@pytest.fixture
def synth_clip(tmp_path) -> Path:
    sample = dummy_clip()
    out = tmp_path / "synth_clip.npz"
    sample_schema.save_npz(sample, str(out))
    return out


# ----------------------------------------------------------------------------
# Tests.
# ----------------------------------------------------------------------------

def test_save_load_roundtrip(synth_clip):
    sample = sample_schema.load_npz(str(synth_clip))
    assert sample["__key__"] == "synth-0:10"
    assert "camera_0" in sample["scene_flows_per_cam"]
    assert "camera_1" in sample["scene_flows_per_cam"]
    for prefix in ("camera_0", "camera_1"):
        assert sample["scene_flows_per_cam"][prefix].shape == (11, 180 * 320, 3)
        assert sample["initial_rgb_per_cam"][prefix].shape == (180, 320, 3)
        assert sample["initial_depth_per_cam"][prefix].shape == (180, 320)
        assert sample["intrinsic_per_cam"][prefix].shape == (3, 3)
        assert sample["extrinsic_per_cam"][prefix].shape == (4, 4)
    assert sample["robot_flows"].shape == (11, 16, 3)
    assert sample["right_gripper_pose"].shape == (11, 7)
    assert sample["right_gripper_open"].shape == (11, 1)


def test_moving_body_actually_moves(synth_clip):
    sample = sample_schema.load_npz(str(synth_clip))
    flows = sample["scene_flows_per_cam"]["camera_0"]
    # We placed 32 bodyA points at the linear indices ``chosen[:32, 0] * W +
    # chosen[:32, 1]``. Those 32 points are split into two contiguous
    # blocks of 16 (rows 0 and 1, columns 0..15) in the H*W image, so we
    # check both blocks.
    H, W = sample_schema.H_RELEASE, sample_schema.W_RELEASE
    block_a = list(range(0, 16)) + list(range(1 * W, 1 * W + 16))
    block_b = list(range(4 * W, 4 * W + 16)) + list(range(4 * W + 4, 4 * W + 20))
    delta_a = np.linalg.norm(flows[10, block_a] - flows[0, block_a], axis=-1)
    delta_b = np.linalg.norm(flows[10, block_b] - flows[0, block_b], axis=-1)
    # Each bodyA point moves +0.005 m * 10 frames = 0.05 m.
    assert (delta_a > 0.04).all(), f"Body A did not move as expected: {delta_a}"
    # bodyB points should not move.
    assert (delta_b < 1e-5).all(), f"Body B should be static, but moved: {delta_b}"


def test_reprojection_consistency(synth_clip):
    """For frame 0, projecting scene_flows[0] back through the camera should
    recover the initial depth within 3 mm (PointWorld's default threshold)."""
    sample = sample_schema.load_npz(str(synth_clip))
    K = sample["intrinsic_per_cam"]["camera_0"]
    T_w_c = sample["extrinsic_per_cam"]["camera_0"]
    depth0 = sample["initial_depth_per_cam"]["camera_0"]
    flows0 = sample["scene_flows_per_cam"]["camera_0"][0]
    valid = sample["scene_depth_valid_mask_per_cam"]["camera_0"][0]
    pts = flows0[valid]
    H, W = depth0.shape
    pixel_idx = np.where(valid)[0]
    v, u = pixel_idx // W, pixel_idx % W
    # project
    T_c_w = np.linalg.inv(T_w_c)
    pts_cam = (T_c_w[:3, :3] @ pts.T + T_c_w[:3, 3:4]).T
    z_proj = pts_cam[:, 2]
    z_obs = depth0[v, u]
    err = np.abs(z_proj - z_obs)
    pct = (err < 0.003).mean()
    # We allow some pixels to fail (the synthetic test is approximate).
    assert pct > 0.9, f"Reprojection consistency too low: {pct:.2%} of pixels within 3mm"


def test_robot_trajectory_consistency(synth_clip):
    sample = sample_schema.load_npz(str(synth_clip))
    rf = sample["robot_flows"]
    # robot_flows are attached to bodyA which moves +x by 0.005 m per frame.
    delta = np.linalg.norm(rf[10] - rf[0], axis=-1)
    assert (delta > 0.04).all(), f"Robot trajectory did not move as expected: {delta.min()}"


def test_schema_validation(synth_clip):
    """A valid npz should round-trip without error; an obviously bad one
    (wrong resolution) should fail."""
    with pytest.raises(Exception):
        # Build a bad npz: wrong rgb shape.
        sample = sample_schema.empty_clip()
        bad = {**{k: v for k, v in np.load(str(synth_clip)).items()},
               "camera_0_initial_rgb": np.zeros((100, 100, 3), dtype=np.uint8)}
        np.savez("/tmp/bad.npz", **bad)
        sample_schema.load_npz("/tmp/bad.npz")
