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

* look up camera intrinsics / extrinsics from the MuJoCo sim
  (delegated to :mod:`robosuite.utils.camera_utils` to avoid reinventing the
   OpenCV / MuJoCo axis correction),
* convert the normalized depth buffer that robosuite exposes to a metric
  distance map (:func:`get_real_depth`),
* backproject RGB-D pixels into world coordinates (per-camera),
* estimate per-point normals from depth,
* look up body / camera poses from the MuJoCo sim at any timestep, and
* track scene points across frames by binding them to their owning rigid body
  (which the LIBERO element-segmentation gives us exactly per pixel) and
  re-applying that body's per-frame pose.

For the robot side this module also provides :func:`get_gripper_body_names`
and :func:`sample_gripper_mesh_points` so the exporter can build a real
Panda gripper surface trajectory (hand + two fingers) instead of the random
sphere placeholder the first draft used.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# robosuite utilities are imported lazily so this module can be loaded
# inside the PointWorld env (no robosuite installed) for unit-testing the
# pure-Python geometry helpers. The functions that need robosuite will
# raise a clear error if it is not available.
def _camera_utils():
    from robosuite.utils import camera_utils  # type: ignore
    return camera_utils


# ----------------------------------------------------------------------------
# Camera intrinsics / extrinsics.
# ----------------------------------------------------------------------------

def get_camera_intrinsic(env, cam_name: str, height: int, width: int) -> np.ndarray:
    """3x3 pinhole intrinsic from a MuJoCo camera definition (OpenCV: cx=W/2,
    cy=H/2). Delegates to :func:`robosuite.utils.camera_utils` to match the
    convention PointWorld's renderer expects."""
    return _camera_utils().get_camera_intrinsic_matrix(
        env.sim, cam_name, height, width
    ).astype(np.float32)


def get_camera_extrinsic_c_w(env, cam_name: str) -> np.ndarray:
    """4x4 world-to-camera transform (``T_c_w``) for ``cam_name``.

    The official :func:`robosuite.utils.camera_utils.get_camera_extrinsic_matrix`
    returns the OpenCV-corrected camera-to-world pose; we invert it so the
    stored extrinsic matches the ``p_cam = T_c_w @ p_world`` convention that
    PointWorld's renderer uses.
    """
    T_w_c = _camera_utils().get_camera_extrinsic_matrix(env.sim, cam_name)
    return np.linalg.inv(T_w_c).astype(np.float32)


def get_camera_extrinsic_w_c(env, cam_name: str) -> np.ndarray:
    """4x4 camera-to-world transform (``T_w_c``) for ``cam_name``, OpenCV
    axis convention. Used internally for things like camera position lookup
    (e.g. for normal flipping)."""
    return _camera_utils().get_camera_extrinsic_matrix(env.sim, cam_name).astype(np.float32)


# ----------------------------------------------------------------------------
# Depth: convert robosuite normalized buffer to metric meters.
# ----------------------------------------------------------------------------

def get_real_depth(raw_depth: np.ndarray, sim) -> np.ndarray:
    """Convert a (H, W) or (H, W, 1) normalized depth buffer to meters.

    robosuite 1.4 returns the depth observable as ``(H, W, 1)`` from a
    normalized ([0, 1]) MuJoCo depth buffer. Calling
    :func:`robosuite.utils.camera_utils.get_real_depth_map` gives back the
    actual distance in meters, which is what PointWorld's scene encoder
    expects via the ``gt_depth`` field.
    """
    if raw_depth.ndim == 3:
        raw_depth = raw_depth.squeeze(-1)
    return _camera_utils().get_real_depth_map(sim, raw_depth.astype(np.float32)).astype(np.float32)


# ----------------------------------------------------------------------------
# Depth back-projection and normal estimation.
# ----------------------------------------------------------------------------

def backproject_depth(depth: np.ndarray, K: np.ndarray, T_c_w: np.ndarray) -> np.ndarray:
    """Backproject a (H, W) metric depth image to (H, W, 3) world points.

    ``T_c_w`` is the 4x4 world-to-camera transform (the schema convention).
    Returns float32 world points.
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

    # p_world = inv(T_c_w) @ p_cam.
    R = T_c_w[:3, :3].astype(np.float32)
    t = T_c_w[:3, 3].astype(np.float32)
    pts_world = pts_cam @ R.T - (R.T @ t)  # (N, 3)
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


def list_robot_body_names(env) -> list[str]:
    """Return the names of all bodies that belong to any robot.

    robosuite prefixes robot bodies with ``robot{idx}_`` (e.g.
    ``robot0_link0``). Some gripper sub-bodies are also under the
    ``robot0_`` prefix -- ``robot0_right_hand``, ``robot0_leftfinger``,
    ``robot0_rightfinger`` for the Panda -- and must therefore also be
    excluded from scene ownership.
    """
    out = []
    for i in range(env.sim.model.nbody):
        name = env.sim.model.body(i).name
        if name and name.startswith("robot0_"):
            out.append(name)
    return out


def get_gripper_body_names(env) -> list[str]:
    """Return the names of the Panda gripper sub-bodies (hand + two fingers).

    LIBERO's Panda defines a palm / hand body and two finger bodies; they
    carry the actual visible mesh, while the EEF body (``gripper0_eef``) is
    only a reference frame. We want the visible ones for robot_flows.
    """
    out = []
    for name in list_robot_body_names(env):
        lname = name.lower()
        if "hand" in lname or "finger" in lname or "gripper" in lname:
            # Skip pure reference frames that carry no geom / mesh.
            out.append(name)
    return out


# ----------------------------------------------------------------------------
# Quat <-> rotmat helper.
# ----------------------------------------------------------------------------

def quat_wxyz_to_rotmat(quat: np.ndarray) -> np.ndarray:
    """Convert a (4,) MuJoCo wxyz quaternion to a 3x3 rotation matrix."""
    qw, qx, qy, qz = quat
    # Normalize (MuJoCo quaternions are already unit-norm but be safe).
    n = (qw * qw + qx * qx + qy * qy + qz * qz) ** 0.5 + 1e-12
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    R = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)
    return R


# ----------------------------------------------------------------------------
# Gripper mesh sampling.
# ----------------------------------------------------------------------------

def sample_gripper_mesh_points(env, body_names: Sequence[str], n_per_body: int = 64,
                               seed: int = 0) -> tuple[np.ndarray, list[str]]:
    """Sample points from the visible mesh of each gripper sub-body.

    Returns:
        (N, 3) float32 points in body-local coordinates, plus a list of
        length N of the body name each point belongs to.
    """
    import mujoco

    rng = np.random.RandomState(seed)
    local_points: list[np.ndarray] = []
    body_for_point: list[str] = []

    for body_name in body_names:
        body_id = env.sim.model.body_name2id(body_name)
        n_geoms = env.sim.model.body_geomnum[body_id]
        if n_geoms <= 0:
            continue
        geom_adr = env.sim.model.body_geomadr[body_id]

        body_verts: list[np.ndarray] = []
        for g in range(geom_adr, geom_adr + n_geoms):
            if env.sim.model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            data_id = env.sim.model.geom_dataid[g]
            if data_id < 0:
                continue
            vert_adr = int(env.sim.model.mesh_vertadr[data_id])
            vert_num = int(env.sim.model.mesh_vertnum[data_id])
            if vert_num == 0:
                continue
            verts = np.asarray(
                env.sim.model.mesh_vert[vert_adr:vert_adr + vert_num],
                dtype=np.float64,
            ).copy()
            # Vertices are stored in the geom's local frame; transform them
            # into the body frame using geom_pos and geom_quat (wxyz).
            geom_pos = np.asarray(env.sim.model.geom_pos[g], dtype=np.float64)
            geom_quat = np.asarray(env.sim.model.geom_quat[g], dtype=np.float64)
            R_g = quat_wxyz_to_rotmat(geom_quat)
            verts = verts @ R_g.T + geom_pos
            body_verts.append(verts)
        if not body_verts:
            continue
        body_verts = np.concatenate(body_verts, axis=0)
        if body_verts.shape[0] > n_per_body:
            idx = rng.choice(body_verts.shape[0], n_per_body, replace=False)
            body_verts = body_verts[idx]
        local_points.append(body_verts.astype(np.float32))
        body_for_point.extend([body_name] * body_verts.shape[0])

    if not local_points:
        return np.zeros((0, 3), dtype=np.float32), []
    return np.concatenate(local_points, axis=0).astype(np.float32), body_for_point


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
    "get_camera_extrinsic_c_w",
    "get_camera_extrinsic_w_c",
    "get_real_depth",
    "backproject_depth",
    "estimate_normals_from_depth",
    "get_body_pose",
    "list_robot_body_names",
    "get_gripper_body_names",
    "sample_gripper_mesh_points",
    "track_points_through_poses",
    "get_gripper_pose",
    "get_gripper_open",
    "quat_wxyz_to_rotmat",
]
