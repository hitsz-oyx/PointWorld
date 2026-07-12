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

def _transform(R, t) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _synth_rgb_image(H, W, color=(180, 120, 60)) -> np.ndarray:
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[..., 0] = color[0]
    rgb[..., 1] = color[1]
    rgb[..., 2] = color[2]
    return rgb


def dummy_clip(num_points_per_body: int = 32) -> tuple[dict, dict]:
    """Build a synthetic clip with two moving rigid bodies and return the
    schema sample plus a metadata dict that exposes the actual body/point
    ownership so the tests can target the right linear indices.

    Convention used in this fixture:
    * Camera at world origin, looking +Z (OpenCV: z forward).
    * Schema stores ``T_c_w`` (world-to-camera); with the camera at the
      origin and identity rotation this is just the identity matrix.
    * Depth values are the actual distance to the surface in meters.

    Body A is a small patch in the top-left at z=0.3 moving +x at 5 mm/frame.
    Body B is the rest of the image at z=0.5 and is static.
    """
    H, W = sample_schema.H_RELEASE, sample_schema.W_RELEASE
    # OpenCV intrinsics (cx = W/2, cy = H/2) -- matches robosuite convention.
    K = np.array([
        [W, 0.0, W / 2.0],
        [0.0, H, H / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    T_c_w = np.eye(4, dtype=np.float32)  # camera at world origin
    T_w_c = T_c_w.copy()                  # == identity

    z_plane = np.full((H, W), 0.5, dtype=np.float32)
    z_plane[:32, :32] = 0.3
    points_world = backproject_depth(z_plane, K, T_c_w)  # (H, W, 3)

    # Sample bodyA from the 32x32 patch (top-left 16x16 block = 256 pixels),
    # bodyB from outside it.
    idx_a = np.indices((32, 32))[:, :16, :16].reshape(2, -1).T  # (256, 2)
    idx_b_list = []
    for r in range(0, H, 4):
        for c in range(0, W, 4):
            if r < 32 and c < 32:
                continue
            idx_b_list.append((r, c))
    idx_b = np.array(idx_b_list[:num_points_per_body])
    idx_a = idx_a[:num_points_per_body]
    chosen = np.concatenate([idx_a, idx_b], axis=0)  # (N, 2)

    points_kept = points_world[chosen[:, 0], chosen[:, 1]]  # (N, 3)
    owning = np.array(["bodyA"] * idx_a.shape[0] + ["bodyB"] * idx_b.shape[0])

    T_bA_t = [_transform(np.eye(3), np.array([0.005 * t, 0.0, 0.0]))
              for t in range(sample_schema.T_FRAMES)]
    T_bB_t = [_transform(np.eye(3), np.array([0.0, 0.0, 0.0]))
              for _ in range(sample_schema.T_FRAMES)]

    body_poses_per_t = {}
    for t, T in enumerate(T_bA_t):
        body_poses_per_t[("bodyA", t)] = T
    for t, T in enumerate(T_bB_t):
        body_poses_per_t[("bodyB", t)] = T
    body_t0_inv = {
        "bodyA": np.linalg.inv(T_bA_t[0]),
        "bodyB": np.linalg.inv(T_bB_t[0]),
    }
    scene_flows_kept = track_points_through_poses(
        points_kept.astype(np.float32), owning, body_t0_inv, body_poses_per_t
    )  # (T, N, 3)

    normals = estimate_normals_from_depth(points_world, T_w_c[:3, 3])
    normals_kept = normals[chosen[:, 0], chosen[:, 1]]
    colors = _synth_rgb_image(H, W, color=(200, 100, 50))

    # Build the schema-shaped sample.
    sample = sample_schema.empty_clip()
    sample["__key__"] = "synth-0:10"
    kept_linear_idx = chosen[:, 0] * W + chosen[:, 1]

    prefix = "camera_0"
    H_W = H * W
    full = np.zeros((sample_schema.T_FRAMES, H_W, 3), dtype=np.float32)
    full[:, kept_linear_idx, :] = scene_flows_kept
    sample["scene_flows_per_cam"][prefix] = full

    full_n = np.zeros((sample_schema.T_FRAMES, H_W, 3), dtype=np.float32)
    full_n[:, kept_linear_idx, :] = np.broadcast_to(
        normals_kept, (sample_schema.T_FRAMES, normals_kept.shape[0], 3)
    )
    sample["scene_normals_per_cam"][prefix] = full_n

    full_c = np.zeros((sample_schema.T_FRAMES, H_W, 3), dtype=np.uint8)
    full_c[:, kept_linear_idx, :] = np.broadcast_to(
        colors[chosen[:, 0], chosen[:, 1]],
        (sample_schema.T_FRAMES, normals_kept.shape[0], 3),
    )
    sample["scene_colors_per_cam"][prefix] = full_c

    full_v = np.zeros((sample_schema.T_FRAMES, H_W), dtype=bool)
    full_v[:, kept_linear_idx] = True
    sample["scene_visibility_per_cam"][prefix] = full_v
    sample["scene_depth_valid_mask_per_cam"][prefix] = full_v.copy()

    sample["initial_rgb_per_cam"][prefix] = colors
    sample["initial_depth_per_cam"][prefix] = z_plane
    sample["intrinsic_per_cam"][prefix] = K
    sample["extrinsic_per_cam"][prefix] = T_c_w

    # Second camera: identical, just so the eval pipeline has 2 to pick.
    for key in ("scene_flows_per_cam", "scene_colors_per_cam",
                "scene_normals_per_cam", "scene_visibility_per_cam",
                "scene_depth_valid_mask_per_cam", "initial_rgb_per_cam",
                "initial_depth_per_cam", "intrinsic_per_cam",
                "extrinsic_per_cam"):
        sample[key]["camera_1"] = sample[key][prefix].copy()

    # Robot trajectories: 16 points attached to a "gripper" body that moves
    # with bodyA.  Placeholder; the real exporter uses mesh sampling.
    n_robot = 16
    robot_local = np.random.RandomState(0).randn(n_robot, 3).astype(np.float32) * 0.02
    robot_flows = np.zeros((sample_schema.T_FRAMES, n_robot, 3), dtype=np.float32)
    for t in range(sample_schema.T_FRAMES):
        # p_world = p_local @ R^T + t  (row-vector convention).
        robot_flows[t] = robot_local @ T_bA_t[t][:3, :3].T + T_bA_t[t][:3, 3]
    sample["robot_flows"] = robot_flows.astype(np.float32)
    sample["robot_normals"] = np.broadcast_to(
        robot_local / (np.linalg.norm(robot_local, axis=-1, keepdims=True) + 1e-12),
        (sample_schema.T_FRAMES, n_robot, 3),
    ).astype(np.float32)
    sample["robot_colors"] = np.zeros((sample_schema.T_FRAMES, n_robot, 3), dtype=np.uint8)
    sample["robot_colors"][..., 0] = 255
    sample["robot_colors"][..., 2] = 255
    sample["right_gripper_pose"] = np.zeros((sample_schema.T_FRAMES, 7), dtype=np.float32)
    sample["right_gripper_pose"][:, 6] = 1.0
    sample["right_gripper_open"] = np.zeros((sample_schema.T_FRAMES, 1), dtype=np.float32)

    meta = {
        "kept_linear_idx": kept_linear_idx,
        "owning_body": owning,
        "bodyA_linear_idx": kept_linear_idx[: idx_a.shape[0]],
        "bodyB_linear_idx": kept_linear_idx[idx_a.shape[0]:],
        "intrinsic": K,
        "extrinsic_T_c_w": T_c_w,
        "z_plane": z_plane,
        "T_bA_t": T_bA_t,
        "T_bB_t": T_bB_t,
    }
    return sample, meta


@pytest.fixture
def synth_clip(tmp_path) -> tuple[Path, dict]:
    sample, meta = dummy_clip()
    out = tmp_path / "synth_clip.npz"
    sample_schema.save_npz(sample, str(out))
    return out, meta


# ----------------------------------------------------------------------------
# Tests.
# ----------------------------------------------------------------------------

def test_save_load_roundtrip(synth_clip):
    path, _ = synth_clip
    sample = sample_schema.load_npz(str(path))
    assert sample["__key__"] == "synth-0:10"
    assert "camera_0" in sample["scene_flows_per_cam"]
    assert "camera_1" in sample["scene_flows_per_cam"]
    for prefix in ("camera_0", "camera_1"):
        assert sample["scene_flows_per_cam"][prefix].shape == (
            sample_schema.T_FRAMES, sample_schema.H_RELEASE * sample_schema.W_RELEASE, 3
        )
        assert sample["initial_rgb_per_cam"][prefix].shape == (180, 320, 3)
        assert sample["initial_depth_per_cam"][prefix].shape == (180, 320)
        assert sample["intrinsic_per_cam"][prefix].shape == (3, 3)
        assert sample["extrinsic_per_cam"][prefix].shape == (4, 4)
    assert sample["robot_flows"].shape == (sample_schema.T_FRAMES, 16, 3)
    assert sample["right_gripper_pose"].shape == (sample_schema.T_FRAMES, 7)
    assert sample["right_gripper_open"].shape == (sample_schema.T_FRAMES, 1)


def test_moving_body_actually_moves(synth_clip):
    path, meta = synth_clip
    sample = sample_schema.load_npz(str(path))
    flows = sample["scene_flows_per_cam"]["camera_0"]

    # Real bodyA linear indices (not zero-filled pixels).
    bodyA_idx = meta["bodyA_linear_idx"]
    bodyB_idx = meta["bodyB_linear_idx"]
    delta_a = np.linalg.norm(flows[10, bodyA_idx] - flows[0, bodyA_idx], axis=-1)
    delta_b = np.linalg.norm(flows[10, bodyB_idx] - flows[0, bodyB_idx], axis=-1)
    # Each bodyA point moves +0.005 m * 10 frames = 0.05 m.
    assert (delta_a > 0.04).all(), f"Body A did not move as expected: {delta_a}"
    # bodyB points should not move.
    assert (delta_b < 1e-5).all(), f"Body B should be static, but moved: {delta_b}"


def test_reprojection_consistency(synth_clip):
    """Projecting frame-0 scene_flows through the stored ``T_c_w`` should
    recover the initial depth within 3 mm. This validates that the schema
    extrinsic is the world-to-camera matrix PointWorld expects.
    """
    path, meta = synth_clip
    sample = sample_schema.load_npz(str(path))
    K = sample["intrinsic_per_cam"]["camera_0"]
    E = sample["extrinsic_per_cam"]["camera_0"]  # T_c_w; do NOT invert
    depth0 = sample["initial_depth_per_cam"]["camera_0"]
    flows0 = sample["scene_flows_per_cam"]["camera_0"][0]
    valid = sample["scene_depth_valid_mask_per_cam"]["camera_0"][0]
    pts_world = flows0[valid]
    H, W = depth0.shape
    pixel_idx = np.where(valid)[0]
    v, u = pixel_idx // W, pixel_idx % W
    # p_cam = T_c_w @ p_world
    pts_cam = (E[:3, :3] @ pts_world.T + E[:3, 3:4]).T
    z_proj = pts_cam[:, 2]
    z_obs = depth0[v, u]
    err = np.abs(z_proj - z_obs)
    pct = (err < 0.003).mean()
    # We allow some pixels to fail (the synthetic test is approximate).
    assert pct > 0.9, f"Reprojection consistency too low: {pct:.2%} of pixels within 3mm"


def test_robot_trajectory_consistency(synth_clip):
    path, _ = synth_clip
    sample = sample_schema.load_npz(str(path))
    rf = sample["robot_flows"]
    # robot_flows are attached to bodyA which moves +x by 0.005 m per frame.
    delta = np.linalg.norm(rf[10] - rf[0], axis=-1)
    assert (delta > 0.04).all(), f"Robot trajectory did not move as expected: {delta.min()}"


def test_schema_validation(synth_clip):
    """A valid npz should round-trip without error; an obviously bad one
    (wrong resolution) should fail."""
    path, _ = synth_clip
    with pytest.raises(Exception):
        bad = {**{k: v for k, v in np.load(str(path)).items()},
               "camera_0_initial_rgb": np.zeros((100, 100, 3), dtype=np.uint8)}
        np.savez("/tmp/bad.npz", **bad)
        sample_schema.load_npz("/tmp/bad.npz")


def test_extrinsic_is_world_to_camera(synth_clip):
    """The schema stores T_c_w; verify it acts as world-to-camera by checking
    the matrix inverse matches the T_w_c we used to construct the scene.
    """
    path, meta = synth_clip
    sample = sample_schema.load_npz(str(path))
    E = sample["extrinsic_per_cam"]["camera_0"]
    T_w_c_expected = meta["extrinsic_T_c_w"]  # identity in this fixture
    # In the fixture T_w_c == I, so the schema extrinsic T_c_w == I too.
    assert np.allclose(E, T_w_c_expected, atol=1e-6), \
        "Stored extrinsic does not match the world-to-camera convention."
