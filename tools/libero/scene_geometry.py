# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""LIBERO-side scene geometry helpers.

This module is the LIBERO-bound half of the LIBERO <-> PointWorld adapter. It
exposes a small set of pure functions that:

* backproject RGB-D pixels into world coordinates (per-camera),
* estimate per-point normals from depth,
* look up body / camera poses from the MuJoCo sim at any timestep, and
* track scene points across frames by binding them to their owning rigid body
  at t=0 and re-applying that body's per-frame pose.

The body ownership step is intentionally conservative for the smoke test: we
assign each valid pixel to the nearest non-robot body whose bounding sphere
contains the point. This is approximate but it does give the PointWorld model
the right rigid-body correspondence structure, which is the property the
adapter is trying to verify first.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


# ----------------------------------------------------------------------------
# Camera intrinsics / extrinsics.
# ----------------------------------------------------------------------------

def get_camera_intrinsic(env, cam_name: str, height: int, width: int) -> np.ndarray:
    """Build a 3x3 pinhole intrinsic from a MuJoCo camera definition.

    MuJoCo stores the vertical FOV in degrees and assumes square pixels; we
    derive focal length and principal point from that and the requested image
    size.
    """
    cam_id = env.sim.model.camera_name2id(cam_name)
    fovy_deg = float(env.sim.model.cam_fovy[cam_id])
    fovy_rad = np.deg2rad(fovy_deg)
    fy = 0.5 * height / np.tan(0.5 * fovy_rad)
    fx = fy
    cx = 0.5 * (width - 1)
    cy = 0.5 * (height - 1)
    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    return K


def get_camera_extrinsic(env, cam_name: str) -> np.ndarray:
    """Return the 4x4 world-from-camera transform for ``cam_name``.

    ``cam_xmat`` is the rotation matrix that takes camera-frame vectors to
    world-frame vectors; ``cam_xpos`` is the camera origin in world frame.
    """
    cam_id = env.sim.model.camera_name2id(cam_name)
    xpos = np.asarray(env.sim.data.cam_xpos[cam_id], dtype=np.float64)
    xmat = np.asarray(env.sim.data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = xmat
    T[:3, 3] = xpos
    return T


# ----------------------------------------------------------------------------
# Depth back-projection and normal estimation.
# ----------------------------------------------------------------------------

def backproject_depth(depth: np.ndarray, K: np.ndarray, T_w_c: np.ndarray) -> np.ndarray:
    """Backproject a (H, W) depth image to (H, W, 3) world points.

    ``T_w_c`` is a 4x4 world-from-camera transform. Returns float32.
    """
    depth = depth.astype(np.float32)
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing="xy")
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (u - cx) * depth / fx
    y_cam = (v - cy) * depth / fy
    z_cam = depth
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1).reshape(-1, 3)

    R = T_w_c[:3, :3].astype(np.float32)
    t = T_w_c[:3, 3].astype(np.float32)
    # pts_cam is (N, 3); we want (R @ p + t) per point.
    pts_world = pts_cam @ R.T + t  # (N, 3)
    return pts_world.reshape(H, W, 3)


def estimate_normals_from_depth(points_world: np.ndarray,
                                camera_position: np.ndarray) -> np.ndarray:
    """Estimate per-point normals from a (H, W, 3) world point cloud.

    Uses forward differences and flips each normal so it points toward
    ``camera_position``. Returns unit-norm float32 (H, W, 3) normals.
    """
    H, W, _ = points_world.shape
    dx = np.roll(points_world, -1, axis=1) - points_world
    dy = np.roll(points_world, -1, axis=0) - points_world
    normals = np.cross(dx, dy)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-12
    normals = normals / norm

    cam = np.asarray(camera_position, dtype=np.float32).reshape(1, 1, 3)
    view = cam - points_world
    view_dot_n = np.sum(view * normals, axis=-1, keepdims=True)
    flip = view_dot_n < 0
    normals = np.where(flip, -normals, normals)
    return normals.astype(np.float32)


# ----------------------------------------------------------------------------
# Body pose lookup.
# ----------------------------------------------------------------------------

def get_body_pose(env, body_name: str) -> np.ndarray:
    """Return the 4x4 world-from-body transform for ``body_name`` at the
    current sim timestep."""
    body_id = env.sim.model.body_name2id(body_name)
    xpos = np.asarray(env.sim.data.body(body_id).xpos, dtype=np.float64)
    xmat = np.asarray(env.sim.data.body(body_id).xmat, dtype=np.float64).reshape(3, 3)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = xmat
    T[:3, 3] = xpos
    return T


def list_non_robot_body_names(env, exclude_prefixes: Sequence[str] = ("robot0:",)) -> list[str]:
    """Return all body names whose name does not start with any excluded prefix.

    MuJoCo stores a world body (typically named ``"world"``) and parent
    sub-bodies. We skip the world body and any robot-prefixed bodies by default.
    """
    out = []
    for i in range(env.sim.model.nbody):
        name = env.sim.model.body(i).name
        if not name:
            continue
        if name == "world":
            continue
        if any(name.startswith(p) for p in exclude_prefixes):
            continue
        out.append(name)
    return out


def find_owning_body(env, world_point: np.ndarray,
                     candidate_bodies: Iterable[str]) -> str | None:
    """Return the body whose bounding sphere contains ``world_point``.

    Falls back to the closest body within 0.5 m if no body's bounding sphere
    actually contains the point (which can happen for points near edges).
    Returns ``None`` if no candidate is within 0.5 m.
    """
    best_name = None
    best_signed = np.inf
    for name in candidate_bodies:
        body_id = env.sim.model.body_name2id(name)
        xpos = np.asarray(env.sim.data.body(body_id).xpos, dtype=np.float64)
        # rbound is the radius of the body's bounding sphere; 0 if body has
        # no geoms (we treat that as 0.05 m so we still consider it).
        rbound = float(env.sim.model.body(body_id).rbound)
        r = rbound if rbound > 1e-6 else 0.05
        dist = float(np.linalg.norm(world_point - xpos))
        # Prefer a body whose sphere *contains* the point; among those pick
        # the smallest residual (i.e. the tightest fit). If none contains,
        # fall back to the closest.
        if dist <= r and (r - dist) < best_signed:
            best_signed = r - dist
            best_name = name
    if best_name is not None:
        return best_name

    # Fallback: nearest candidate within 0.5 m.
    best_dist = 0.5
    for name in candidate_bodies:
        body_id = env.sim.model.body_name2id(name)
        xpos = np.asarray(env.sim.data.body(body_id).xpos, dtype=np.float64)
        dist = float(np.linalg.norm(world_point - xpos))
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name


# ----------------------------------------------------------------------------
# Scene trajectory tracking.
# ----------------------------------------------------------------------------

def track_points_through_poses(world_points_0: np.ndarray,
                               owning_body: np.ndarray,
                               body_poses_t0_inv: dict,
                               body_poses_per_t: dict) -> np.ndarray:
    """Build a (T, N, 3) world trajectory given a precomputed pose cache.

    Args:
        world_points_0: (N, 3) world points at t=0.
        owning_body: (N,) array of body name strings (or None).
        body_poses_t0_inv: ``{body_name: 4x4}`` of T_b0^-1 for each named body.
        body_poses_per_t: ``{(body_name, t): 4x4}`` of T_bt per timestep.

    Returns:
        (T, N, 3) world trajectory.
    """
    N = world_points_0.shape[0]
    # Discover T from the cache: any key of body_poses_per_t has the time axis.
    if not body_poses_per_t:
        T = 1
    else:
        T = 1 + max(t for _, t in body_poses_per_t.keys())
    out = np.empty((T, N, 3), dtype=np.float32)

    for i in range(N):
        body = owning_body[i]
        if not body:
            out[:, i, :] = world_points_0[i]
            continue
        T_inv = body_poses_t0_inv.get(body)
        if T_inv is None:
            out[:, i, :] = world_points_0[i]
            continue
        p_local = T_inv @ np.array([*world_points_0[i], 1.0], dtype=np.float64)
        for t in range(T):
            T_t = body_poses_per_t.get((body, t))
            if T_t is None:
                out[t, i, :] = world_points_0[i]
                continue
            p_w = T_t @ np.array([*p_local[:3], 1.0], dtype=np.float64)
            out[t, i, :] = p_w[:3].astype(np.float32)
    return out


# ----------------------------------------------------------------------------
# Robot gripper pose snapshot.
# ----------------------------------------------------------------------------

def get_gripper_pose(env, gripper_body_name: str = "gripper0_eef") -> np.ndarray:
    """Return the 7D (xyz + qx,qy,qz,qw) world-frame gripper pose."""
    T = get_body_pose(env, gripper_body_name)
    t = T[:3, 3]
    R = T[:3, :3]
    q = _rotmat_to_quat_xyzw(R)
    return np.array([t[0], t[1], t[2], q[0], q[1], q[2], q[3]], dtype=np.float32)


def get_gripper_open(env, gripper_body_name: str = "gripper0_eef") -> float:
    """Return a scalar gripper openness in [0, 1].

    Reads the ``gripper0_finger_joint1`` (or similar) joint position. The
    mapping from joint position to openness is approximate; we only use this
    as a *feature* for the model, so any smooth [0, 1] signal works.
    """
    # The Panda gripper in LIBERO has a finger joint called
    # ``robot0_gripper_joint1`` (or ``gripper0_finger_joint1``) with a positive
    # range. We pick the first joint whose name contains ``finger`` and
    # belongs to the gripper body subtree.
    try:
        for jid in range(env.sim.model.njnt):
            name = env.sim.model.joint(jid).name
            if "finger" not in name:
                continue
            lo, hi = env.sim.model.jnt_range[jid]
            if hi <= lo:
                continue
            qpos_addr = env.sim.model.jnt_qposadr[jid]
            qpos = float(env.sim.data.qpos[qpos_addr])
            return float((qpos - lo) / (hi - lo))
    except Exception:
        return 0.0
    return 0.0


def _rotmat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> (qx, qy, qz, qw)."""
    R = np.asarray(R, dtype=np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    return np.array([qx, qy, qz, qw], dtype=np.float64)


__all__ = [
    "get_camera_intrinsic",
    "get_camera_extrinsic",
    "backproject_depth",
    "estimate_normals_from_depth",
    "get_body_pose",
    "list_non_robot_body_names",
    "find_owning_body",
    "track_points_through_poses",
    "get_gripper_pose",
    "get_gripper_open",
]
