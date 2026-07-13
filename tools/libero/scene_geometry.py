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

    ``T_c_w`` is the 4x4 world-to-camera transform (the schema convention
    used by the rest of the pipeline). Returns float32 world points.

    The mapping is ``p_cam = T_c_w @ p_world`` (homogeneous), so
    ``p_world = inv(T_c_w) @ p_cam``. We invert the extrinsic explicitly
    rather than re-deriving ``R^T @ (p - t)`` because the earlier
    hand-derived formula had the wrong sign of the translation when the
    camera was off-origin (the synthetic unit test that only used
    ``T_c_w = I`` masked the bug).
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
    pts_cam = pts_cam.astype(np.float32)

    # p_world = inv(T_c_w) @ p_cam.
    T_w_c = np.linalg.inv(T_c_w.astype(np.float64)).astype(np.float32)
    ones = np.ones((pts_cam.shape[0], 1), dtype=np.float32)
    pts_cam_h = np.concatenate([pts_cam, ones], axis=-1)
    pts_world = (T_w_c @ pts_cam_h.T).T[:, :3]
    return pts_world.reshape(H, W, 3)


# ----------------------------------------------------------------------------
# Body ownership from robosuite's element segmentation.
# ----------------------------------------------------------------------------

def get_per_pixel_bodies(env, seg_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a (H, W, 1) element-segmentation image to per-pixel body IDs
    and body names.

    The element segmentation channel from robosuite is the **raw geom
    id** (no ``+1`` offset). The ``+1`` shift only applies to
    instance / class seg maps, which go through the ``mapping`` table
    in ``CameraObservable``. Our previous ``seg - 1`` mapping silently
    shifted every geom index down by one, which meant the body lookup
    for every scene point was off by one -- the root cause of the
    "table / world only" body ownership symptom on real LIBERO scenes.

    Returns
    -------
    body_id : (H, W) int32
        1-based body id per pixel. -1 = background / no hit.
    body_name : (H, W) object array of strings
    """
    seg = seg_map.squeeze(-1).astype(np.int32)
    H, W = seg.shape
    nbody = env.sim.model.nbody
    ngeom = env.sim.model.ngeom
    valid_geom = (seg >= 0) & (seg < ngeom)
    # ``model.geom_bodyid`` is 0-based; we add 1 so the no-hit value can
    # be a clean 0 (sentinel for "no body"). Pixels with no hit stay at
    # the default 0 in the empty ``body_id`` array.
    body_id = np.zeros(seg.shape, dtype=np.int32)
    body_id[valid_geom] = env.sim.model.geom_bodyid[seg[valid_geom]] + 1
    body_name = np.empty((H, W), dtype=object)
    for b in range(nbody):
        mask = body_id == (b + 1)
        if not mask.any():
            continue
        # ``model.body(b).name`` is a ``bytes`` object in mujoco; decode
        # it so it matches the ``str`` keys in ``body_poses_per_t`` and
        # the ``robot_body_set`` membership checks downstream.
        raw_name = env.sim.model.body(b).name
        if isinstance(raw_name, bytes):
            raw_name = raw_name.decode("utf-8", errors="ignore")
        body_name[mask] = raw_name
    body_name[body_id == 0] = ""
    return body_id, body_name


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
    current sim timestep. ``body_name`` is a ``str`` (not bytes) and
    matches the keys in ``body_poses_per_t``."""
    # ``body_name2id`` accepts str directly in mujoco 2.x.
    if isinstance(body_name, str):
        try:
            body_id = env.sim.model.body_name2id(body_name)
        except Exception:
            # Some mujoco versions still want bytes; fall back.
            body_id = env.sim.model.body_name2id(body_name.encode("utf-8"))
    else:
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

    All names are returned as ``str`` (not ``bytes``) to match the
    decoded body names produced by ``get_per_pixel_bodies``.
    """
    out = []
    for i in range(env.sim.model.nbody):
        name = env.sim.model.body(i).name
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="ignore")
        if name and name.startswith("robot0_"):
            out.append(name)
    return out


def _get_mesh_name(env, mesh_id: int) -> str | None:
    """Return the name of the mesh with id ``mesh_id``, or ``None``.

    Works on both modern mujoco (``model.id2name`` available) and the
    mujoco 2.3.x that ships with LIBERO 0.1.0 / robosuite 1.4
    (no ``id2name`` but has ``model.names`` + ``name_meshadr``).
    The fallback reads the mesh name out of ``model.names`` via the
    mesh's own name offset.
    """
    model = env.sim.model
    # Modern path.
    try:
        import mujoco
        name = model.id2name(mujoco.mjtObj.mjOBJ_MESH, mesh_id)
        if name is not None:
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="ignore")
            return name
    except Exception:
        pass
    # Fallback: walk the ``model.names`` blob. mujoco packs all names
    # as a single null-separated byte string; ``name_meshadr[mesh_id]``
    # gives the offset for each mesh.
    try:
        if not hasattr(model, "names") or not hasattr(model, "name_meshadr"):
            return None
        names_blob = model.names
        if isinstance(names_blob, bytes):
            names_blob = names_blob.decode("utf-8", errors="ignore")
        adr = int(model.name_meshadr[mesh_id])
        end = names_blob.find("\x00", adr)
        if end < 0:
            return None
        return names_blob[adr:end]
    except Exception:
        return None


def get_gripper_body_names(env, gripper_eef_body: str = "gripper0_eef") -> list[str]:
    """Return the names of the Panda gripper / hand sub-bodies with visible
    mesh.

    Two-pass selection:

    1. **Body-level** — any robot body whose name contains a
       gripper keyword (``"hand"`` / ``"finger"`` / ``"gripper"`` /
       ``"palm"``) and which owns at least one visible geom. This
       matches ``robot0_*finger*`` / ``robot0_hand`` on the
       real Franka / Panda.

    2. **Mesh-level** — for bodies that don't have a gripper keyword
       in their name, we still pick them up if any of their
       attached *mesh* geoms has a gripper keyword in the mesh name
       (the "hand mesh is on link7" case).

    3. **EEF-proximity fallback** — for LIBERO scenes where the
       gripper meshes don't carry gripper keywords (the meshes are
       named e.g. ``robot0_link7_vis_0``), we additionally include
       the body that owns the **gripper EEF** and all of its
       ancestors in the kinematic tree that have visible geoms.
       This is the only way to identify the gripper bodies on the
       actual LIBERO ``libero_spatial`` scenes without hard-coding
       ``("robot0_link6", "robot0_link7")`` (the bug the previous
       implementation had — see docs/指导.md §P1-3).

    The function never hard-codes a body list. It does however rely
    on the LIBERO / robosuite convention that the gripper EEF is
    named ``gripper0_eef`` and lives at the end of the kinematic
    chain. Override via the ``gripper_eef_body`` argument if a
    different convention is in use.

    Bodies with no geoms at all are excluded so
    :func:`sample_gripper_mesh_points` always produces a non-empty
    ``robot_flows`` cloud.
    """
    import mujoco  # local: only needed for the geom-type enum

    robot_bodies = list_robot_body_names(env)
    gripper_keywords = ("hand", "finger", "gripper", "palm")
    keep: set[str] = set()

    # Build a (bid -> body_name) map once.
    bid_to_name: dict[int, str] = {i: env.sim.model.body(i).name for i in range(env.sim.model.nbody)}
    # Decode bytes → str.
    for k, v in list(bid_to_name.items()):
        if isinstance(v, bytes):
            bid_to_name[k] = v.decode("utf-8", errors="ignore")
    name_to_bid: dict[str, int] = {v: k for k, v in bid_to_name.items()}

    # Pass 1 + 2: name + mesh keyword matching.
    for g in range(env.sim.model.ngeom):
        owning_body = int(env.sim.model.geom_bodyid[g])
        body_name = bid_to_name[owning_body]
        if not body_name.startswith("robot0_"):
            continue
        lname = body_name.lower()
        if any(k in lname for k in gripper_keywords):
            if env.sim.model.body_geomnum[owning_body] > 0:
                keep.add(body_name)
            continue
        if env.sim.model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        data_id = int(env.sim.model.geom_dataid[g])
        if data_id < 0:
            continue
        mesh_name = _get_mesh_name(env, data_id)
        if mesh_name is None:
            continue
        if any(k in mesh_name.lower() for k in gripper_keywords):
            if env.sim.model.body_geomnum[owning_body] > 0:
                keep.add(body_name)

    # Pass 3: EEF-proximity fallback. If we found no gripper body
    # by name/mesh matching, the LIBERO scene's gripper meshes
    # don't carry gripper keywords (they're typically named
    # ``robot0_link7_vis_*`` etc.). Take the last
    # ``max_distal`` robot bodies with visible geoms; on the
    # LIBERO / Panda layout that's ``link6`` and ``link7`` (the
    # wrist + hand cluster). We never want to pull in the whole
    # arm, so the cap is critical.
    if not keep and gripper_eef_body in name_to_bid:
        robot_with_geoms = [
            n for n in robot_bodies
            if env.sim.model.body_geomnum[name_to_bid[n]] > 0
        ]
        # robot_bodies is ordered by body id (proximal -> distal).
        # Take the trailing ``max_distal`` entries.
        max_distal = 2
        for name in robot_with_geoms[-max_distal:]:
            keep.add(name)

    # Preserve the original (deterministic) order from
    # ``list_robot_body_names`` for output stability.
    return [name for name in robot_bodies if name in keep]


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
# Gripper mesh sampling (surface, with face normals).
# ----------------------------------------------------------------------------

def _sample_box_surface(
    rng: np.random.RandomState,
    half_sizes: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``n`` points on the surface of an axis-aligned box.

    Faces are weighted by area. Returns:
        pts:      (n, 3) points in box-local coords (centered at origin).
        normals:  (n, 3) outward unit face normals (one of ±x, ±y, ±z).
    """
    hs = np.asarray(half_sizes, dtype=np.float64)
    # Face areas (in pairs of opposing faces): [2*hs[1]*hs[2], 2*hs[0]*hs[2], 2*hs[0]*hs[1]]
    # for ±x, ±y, ±z.
    face_areas = np.array([
        2.0 * hs[1] * hs[2],
        2.0 * hs[0] * hs[2],
        2.0 * hs[0] * hs[1],
    ])
    if face_areas.sum() <= 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)
    # Pick which axis the face is on (x, y, or z), then which side (+/-).
    axis_pick = rng.choice(3, size=n, p=face_areas / face_areas.sum())
    sign_pick = rng.choice([-1, 1], size=n)
    # 2D coords in the face's tangent plane.
    uv = rng.uniform(0.0, 1.0, size=(n, 2))
    pts = np.zeros((n, 3), dtype=np.float64)
    normals = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        ax = int(axis_pick[i])
        sgn = float(sign_pick[i])
        u, v = uv[i]
        # Map u, v in [0, 1] to the face's two tangent axes, both
        # centered at 0 with extents hs[t1], hs[t2].
        t1 = (ax + 1) % 3
        t2 = (ax + 2) % 3
        face = np.zeros(3, dtype=np.float64)
        face[ax] = sgn * hs[ax]
        face[t1] = (2.0 * u - 1.0) * hs[t1]
        face[t2] = (2.0 * v - 1.0) * hs[t2]
        pts[i] = face
        n_vec = np.zeros(3, dtype=np.float64)
        n_vec[ax] = sgn
        normals[i] = n_vec
    return pts, normals


def _sample_cylinder_surface(
    rng: np.random.RandomState,
    radius: float,
    half_height: float,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``n`` points on the surface of a cylinder of given radius and
    half-height, oriented along the z-axis (geom-local). The cylinder has
    3 surface components (lateral + 2 caps); we weight by area.

    Returns:
        pts:      (n, 3) points in cylinder-local coords.
        normals:  (n, 3) outward unit surface normals.
    """
    r, h = float(radius), float(half_height)
    if r <= 0 or h <= 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)
    lateral_area = 2.0 * np.pi * r * (2.0 * h)
    cap_area = np.pi * r * r
    weights = np.array([lateral_area, cap_area, cap_area])
    if weights.sum() <= 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)
    comp_pick = rng.choice(3, size=n, p=weights / weights.sum())
    pts = np.zeros((n, 3), dtype=np.float64)
    normals = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        comp = int(comp_pick[i])
        if comp == 0:
            # Lateral surface.
            theta = rng.uniform(0.0, 2.0 * np.pi)
            z = rng.uniform(-h, h)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            pts[i] = (x, y, z)
            n_vec = np.array([np.cos(theta), np.sin(theta), 0.0])
            normals[i] = n_vec
        else:
            # Cap (top if comp == 1 else bottom).
            r_pick = r * np.sqrt(rng.uniform(0.0, 1.0))
            theta = rng.uniform(0.0, 2.0 * np.pi)
            x = r_pick * np.cos(theta)
            y = r_pick * np.sin(theta)
            z = h if comp == 1 else -h
            pts[i] = (x, y, z)
            normals[i] = (0.0, 0.0, 1.0 if comp == 1 else -1.0)
    return pts, normals


def _sample_sphere_surface(
    rng: np.random.RandomState,
    radius: float,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``n`` points on the surface of a sphere, uniformly by area.

    Returns:
        pts:      (n, 3) points in sphere-local coords.
        normals:  (n, 3) outward unit normals (== pts / radius).
    """
    if radius <= 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12
    pts = v * radius
    return pts, v


def _sample_mesh_surface(
    rng: np.random.RandomState,
    verts: np.ndarray,
    faces: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``n`` points on the surface of a triangle mesh by
    area-weighted face sampling.

    For each sample we pick a face weighted by its area, then a
    uniformly-weighted point in the triangle's interior via two
    random barycentric coordinates (the sqrt is the standard
    area-preserving mapping). The returned normal is the face normal
    (constant over the triangle).

    Returns:
        pts:      (n, 3) points in mesh-local coords.
        normals:  (n, 3) unit face normals (constant per face).
    """
    if faces.shape[0] == 0 or verts.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)
    tri = verts[faces]  # (F, 3, 3)
    # Per-face area = 0.5 * ||(b - a) x (c - a)||
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    cross = np.cross(e1, e2)
    area = 0.5 * np.linalg.norm(cross, axis=-1)
    face_normals = cross / (np.linalg.norm(cross, axis=-1, keepdims=True) + 1e-12)
    if area.sum() <= 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)
    probs = area / area.sum()
    face_pick = rng.choice(faces.shape[0], size=n, p=probs)
    # Barycentric (u, v) ~ U(triangle). Standard formulation:
    #   r1 = sqrt(u), r2 = v, point = (1 - sqrt(u)) * a
    #       + sqrt(u) * (1 - v) * b + sqrt(u) * v * c
    u = np.sqrt(rng.uniform(0.0, 1.0, size=n))
    v = rng.uniform(0.0, 1.0, size=n)
    chosen = tri[face_pick]  # (n, 3, 3) -- a, b, c
    pts = (
        (1.0 - u)[:, None] * chosen[:, 0]
        + (u * (1.0 - v))[:, None] * chosen[:, 1]
        + (u * v)[:, None] * chosen[:, 2]
    )
    normals = face_normals[face_pick]
    return pts, normals


def sample_gripper_mesh_points(env, body_names: Sequence[str], n_per_body: int = 64,
                               seed: int = 0) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Sample points and outward unit normals from the visible surface of
    each gripper sub-body.

    Sampling is **on the surface** (not in the volume) and the
    returned normal is the **surface normal**, not a radial-from-body-
    origin vector. This matches PointWorld's training-time
    ``deterministic_sample_surface`` (see docs/指导.md §P1-2) and
    gives the model a meaningful ``robot_normals`` feature on the
    Panda hand / fingers.

    Returns:
        pts:        (N, 3) float32 points in body-local coords.
        normals:    (N, 3) float32 unit outward normals (body-local).
        body_for_point: list of length N of body names.
    """
    import mujoco

    rng = np.random.RandomState(seed)
    local_points: list[np.ndarray] = []
    local_normals: list[np.ndarray] = []
    body_for_point: list[str] = []

    for body_name in body_names:
        body_id = env.sim.model.body_name2id(body_name)
        n_geoms = env.sim.model.body_geomnum[body_id]
        if n_geoms <= 0:
            continue
        geom_adr = env.sim.model.body_geomadr[body_id]

        body_verts: list[np.ndarray] = []
        body_norms: list[np.ndarray] = []
        for g in range(geom_adr, geom_adr + n_geoms):
            geom_pos = np.asarray(env.sim.model.geom_pos[g], dtype=np.float64)
            geom_quat = np.asarray(env.sim.model.geom_quat[g], dtype=np.float64)
            R_g = quat_wxyz_to_rotmat(geom_quat)
            gtype = env.sim.model.geom_type[g]
            gsize = np.asarray(env.sim.model.geom_size[g], dtype=np.float64)
            n_geom_samp = max(1, n_per_body // max(1, n_geoms))

            if gtype == mujoco.mjtGeom.mjGEOM_MESH:
                data_id = env.sim.model.geom_dataid[g]
                if data_id < 0:
                    continue
                vert_adr = int(env.sim.model.mesh_vertadr[data_id])
                vert_num = int(env.sim.model.mesh_vertnum[data_id])
                face_adr = int(env.sim.model.mesh_faceadr[data_id])
                face_num = int(env.sim.model.mesh_facenum[data_id])
                if vert_num == 0 or face_num == 0:
                    continue
                verts = np.asarray(
                    env.sim.model.mesh_vert[vert_adr:vert_adr + vert_num],
                    dtype=np.float64,
                ).copy()
                faces = np.asarray(
                    env.sim.model.mesh_face[face_adr:face_adr + face_num],
                    dtype=np.int64,
                ).copy()
                # Translate into geom-local then to body frame.
                pts_local, normals_local = _sample_mesh_surface(
                    rng, verts, faces, n_geom_samp
                )
            elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
                pts_local, normals_local = _sample_box_surface(
                    rng, gsize[:3], n_geom_samp
                )
            elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
                r = gsize[0]
                h = gsize[1]
                pts_local, normals_local = _sample_cylinder_surface(
                    rng, r, h, n_geom_samp
                )
            elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
                pts_local, normals_local = _sample_sphere_surface(
                    rng, gsize[0], n_geom_samp
                )
            else:
                # Unknown primitive: skip.
                continue

            # Transform from geom-local to body-local.
            pts_body = pts_local @ R_g.T + geom_pos
            normals_body = normals_local @ R_g.T
            body_verts.append(pts_body)
            body_norms.append(normals_body)

        if not body_verts:
            continue
        body_verts = np.concatenate(body_verts, axis=0)
        body_norms = np.concatenate(body_norms, axis=0)
        if body_verts.shape[0] > n_per_body:
            idx = rng.choice(body_verts.shape[0], n_per_body, replace=False)
            body_verts = body_verts[idx]
            body_norms = body_norms[idx]
        # Renormalize after the geom-rotation (R_g was a proper rotation
        # so the magnitude is preserved, but be safe against tiny float
        # drift).
        body_norms = body_norms / (
            np.linalg.norm(body_norms, axis=-1, keepdims=True) + 1e-12
        )
        local_points.append(body_verts.astype(np.float32))
        local_normals.append(body_norms.astype(np.float32))
        body_for_point.extend([body_name] * body_verts.shape[0])

    if not local_points:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            [],
        )
    return (
        np.concatenate(local_points, axis=0).astype(np.float32),
        np.concatenate(local_normals, axis=0).astype(np.float32),
        body_for_point,
    )


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
# Body pose snapshot helper (used by ``export_clip`` to cache poses for
# the trajectory tracker).
# ----------------------------------------------------------------------------

def snapshot_body_poses(env, body_names: Sequence[str], t_idx: int,
                        cache: dict) -> None:
    """Fill ``cache[(body, t_idx)] = T_w_b`` for every body that still
    exists in the current sim state.

    Bodies whose name lookup raises (e.g. an object that was removed
    mid-clip) are silently skipped. ``t_idx`` is the per-frame index
    the caller is using to key the cache.
    """
    for body in body_names:
        try:
            cache[(body, t_idx)] = get_body_pose(env, body)
        except Exception:
            continue


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
    "get_per_pixel_bodies",
    "sample_gripper_mesh_points",
    "track_points_through_poses",
    "snapshot_body_poses",
    "get_gripper_pose",
    "get_gripper_open",
    "quat_wxyz_to_rotmat",
]
