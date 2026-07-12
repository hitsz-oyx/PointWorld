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
"""Export a single 11-frame LIBERO clip as a PointWorld-ready .npz.

The demo HDF5 file is the source of truth for the env: we read its
``model_file`` and ``env_args`` attributes, postprocess the model XML, build
the env exactly like the original LIBERO replay does, then run 11 timesteps
of action playback and dump the per-camera RGB-D / intrinsics / extrinsics,
per-element body ownership, and the gripper mesh-vertex point flow.

Usage (run from the project root inside the LIBERO conda env)::

    python -m tools.libero.export_clip \
        --demo_hdf5 /path/to/libero_spatial/<task>/demo.hdf5 \
        --demo_id demo_0 \
        --start_idx 100 \
        --output /tmp/libero_clip.npz

This script does not import any PointWorld code; the only contract is the
.npz file format defined in :mod:`tools.libero.sample_schema`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

# LIBERO / robosuite imports. These are only available in the LIBERO env.
try:
    from libero.libero.envs import OffScreenRenderEnv  # type: ignore
    # The XML post-processing helper lives in libero.libero.utils.utils
    # in the official LIBERO source; the previous ``env_utils`` name was
    # a misnomer carried over from an older draft.
    from libero.libero.utils import utils as libero_env_utils  # type: ignore
    # Postprocess that also rewrites ``chiliocosm/assets`` paths (used by
    # the official LIBERO recordings) to the local assets dir.
    from libero.libero.envs import utils as libero_xml_postprocess  # type: ignore
except ImportError as e:  # pragma: no cover - import error path
    print(
        "FATAL: this script must be run inside the LIBERO environment. "
        f"Failed to import libero: {e}",
        file=sys.stderr,
    )
    raise


def _rewrite_libero_asset_paths(xml_str: str) -> str:
    """Rewrite the absolute ``/Users/yifengz/.../chiliocosm/assets/...``
    paths baked into the recorded XMLs to point at the assets dir of the
    LIBERO source checkout on this machine. ``libero_env_utils.postprocess_model_xml``
    only fixes the ``robosuite/...``-style paths and leaves
    ``stable_scanned_objects``, ``turbosquid_objects``, etc. as absolute
    paths that no longer resolve."""
    import xml.etree.ElementTree as ET  # local import: not needed elsewhere
    from libero.libero import get_libero_path

    assets_dir = Path(get_libero_path("assets")).resolve()
    if not assets_dir.is_dir():
        return xml_str

    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        return xml_str

    asset = root.find("asset")
    if asset is None:
        return xml_str

    # Asset subfolders we know about. They are sibling subdirs of the
    # assets dir, e.g. ``stable_scanned_objects/akita_black_bowl/...``.
    subfolders = {
        p.name for p in assets_dir.iterdir() if p.is_dir()
    }

    def _rewrite_path(old: str) -> str:
        if old is None or os.path.isabs(old) and Path(old).is_file():
            return old
        # Find the first asset subfolder in the path. Everything from
        # that subfolder onwards is mapped into our local assets dir.
        parts = Path(old).parts
        for i, part in enumerate(parts):
            if part in subfolders:
                rel = Path(*parts[i:])
                candidate = assets_dir / rel
                if candidate.is_file():
                    return str(candidate)
        return old

    for elem in asset.findall("mesh") + asset.findall("texture"):
        old = elem.get("file")
        if old is None:
            continue
        new = _rewrite_path(old)
        if new != old:
            elem.set("file", new)

    return ET.tostring(root, encoding="utf8").decode("utf8")

from .sample_schema import (
    T_FRAMES,
    H_RELEASE,
    W_RELEASE,
    empty_clip,
    save_npz,
)
from .scene_geometry import (
    backproject_depth,
    estimate_normals_from_depth,
    get_body_pose,
    get_camera_extrinsic_c_w,
    get_camera_extrinsic_w_c,
    get_camera_intrinsic,
    get_gripper_body_names,
    get_gripper_open,
    get_gripper_pose,
    get_real_depth,
    list_robot_body_names,
    sample_gripper_mesh_points,
    track_points_through_poses,
)


# ----------------------------------------------------------------------------
# Environment construction from demo metadata.
# ----------------------------------------------------------------------------

def _resolve_bddl_for_demo(
    demo_hdf5: str,
    demo_group,
    cli_bddl: str | None,
    extra_search_dirs: list[str] | None,
) -> str:
    """Find the BDDL file for a demo.

    Priority:
    1. ``--bddl`` CLI flag (if provided and exists on disk).
    2. ``env_args/bddl_file_name`` stored in the demo group (legacy
       recording format produced by ``record_one_demo``).
    3. The BDDL with the same stem as the HDF5, in any of the search
       dirs (covers the real LIBERO dataset layout).
    """
    if cli_bddl is not None:
        if not Path(cli_bddl).is_file():
            raise FileNotFoundError(
                f"--bddl {cli_bddl!r} was passed but does not exist on disk."
            )
        return str(Path(cli_bddl).resolve())

    # (2) legacy in-HDF5 env_args
    raw = demo_group.attrs.get("env_args", None)
    if raw is not None:
        if isinstance(raw, (str, bytes)):
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = {}
        elif isinstance(raw, dict):
            parsed = dict(raw)
        else:
            parsed = {}
        candidate = parsed.get("bddl_file_name")
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())

    # (3) stem-based search across (a) the LIBERO bddl_files tree and
    #     (b) any directories the user passed via --bddl_search_dir.
    stem = Path(demo_hdf5).stem  # e.g. "..._demo"
    if stem.endswith("_demo"):
        stem = stem[: -len("_demo")]
    candidate_names = [f"{stem}.bddl", f"{stem}.libero.bddl"]

    search_dirs: list[str] = []
    if extra_search_dirs:
        search_dirs.extend(extra_search_dirs)
    # The default LIBERO bddl_files layout, mirrored per suite.
    bddl_root = "/home/u2023312616/test_ws/LIBERO/libero/libero/bddl_files"
    if Path(bddl_root).is_dir():
        for suite in Path(bddl_root).iterdir():
            if suite.is_dir():
                search_dirs.append(str(suite))

    for d in search_dirs:
        for name in candidate_names:
            p = Path(d) / name
            if p.is_file():
                return str(p.resolve())

    raise FileNotFoundError(
        f"Could not locate a BDDL file for demo {Path(demo_hdf5).name!r}. "
        f"Tried stems {candidate_names} in {search_dirs}. "
        "Pass it explicitly via --bddl."
    )


def make_libero_env_from_demo(
    demo_hdf5: str,
    demo_id: str,
    camera_names: Sequence[str],
    height: int,
    width: int,
    bddl: str | None = None,
    bddl_search_dirs: list[str] | None = None,
):
    """Build a LIBERO env by reading ``model_file`` / ``env_args`` out of the
    demo HDF5 and replaying the canonical LIBERO reset_from_xml_string path.

    Returns ``(env, demo_group)``. The HDF5 is kept open by the caller.
    """
    f = h5py.File(demo_hdf5, "r")
    if f"data/{demo_id}" not in f:
        f.close()
        raise ValueError(f"demo_id '{demo_id}' not found in {demo_hdf5}")
    demo_group = f[f"data/{demo_id}"]

    if "model_file" not in demo_group.attrs:
        # Fall back to a "model_file" dataset if the XML is too large to
        # fit as an attribute (HDF5 caps attributes at 64 KB). The recorded
        # demo produced by ``tools.libero.record_one_demo`` uses this
        # fallback for any non-trivial LIBERO scene.
        if "model_file" in demo_group:
            model_file = demo_group["model_file"][()]
            if isinstance(model_file, bytes):
                model_file = model_file.decode("utf-8")
        else:
            f.close()
            raise ValueError(
                f"Demo group {demo_id} is missing the 'model_file' "
                "attribute and 'model_file' dataset. This exporter "
                "requires a HDF5 file produced by a recent LIBERO."
            )
    else:
        model_file = demo_group.attrs["model_file"]
        if isinstance(model_file, bytes):
            model_file = model_file.decode("utf-8")
    model_xml = libero_env_utils.postprocess_model_xml(model_file, {})
    # The recorded XML also references ``chiliocosm/assets/...`` paths
    # for the LIBERO-specific meshes; rewrite them to the local assets dir.
    model_xml = _rewrite_libero_asset_paths(model_xml)

    raw_env_args = demo_group.attrs.get("env_args", None)
    # ``env_args`` may be a JSON string (as written by
    # ``tools.libero.record_one_demo``) rather than a real dict, depending
    # on which tool produced the HDF5. Normalize to a dict.
    if raw_env_args is None:
        env_args = {}
    elif isinstance(raw_env_args, (str, bytes)):
        if isinstance(raw_env_args, bytes):
            raw_env_args = raw_env_args.decode("utf-8")
        try:
            env_args = json.loads(raw_env_args)
        except json.JSONDecodeError:
            env_args = {}
    elif isinstance(raw_env_args, dict):
        env_args = dict(raw_env_args)
    else:
        env_args = {}

    # Resolve the BDDL: prefer the explicit --bddl flag, then the in-HDF5
    # env_args (legacy recording format), then the BDDL with the same stem
    # as the HDF5 (real LIBERO dataset layout).
    bddl_file = _resolve_bddl_for_demo(
        demo_hdf5=demo_hdf5,
        demo_group=demo_group,
        cli_bddl=bddl,
        extra_search_dirs=bddl_search_dirs,
    )
    if bddl_file is None:
        f.close()
        raise ValueError(
            f"Demo group {demo_id} is missing 'env_args/bddl_file_name' "
            "and no --bddl flag was provided. Pass --bddl explicitly."
        )

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_names=list(camera_names),
        camera_widths=width,
        camera_heights=height,
        camera_depths=True,
        camera_segmentations="element",  # per-geom; see `get_per_pixel_bodies`.
    )
    env.reset()
    # Canonical LIBERO replay pattern (see docs/指导.md §5 P0#5):
    env.reset_from_xml_string(model_xml)
    env.sim.reset()
    return env, f, demo_group


# ----------------------------------------------------------------------------
# Per-pixel body ownership via element segmentation.
# ----------------------------------------------------------------------------

def get_per_pixel_bodies(env, seg_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a (H, W, 1) element-segmentation image to per-pixel body IDs
    and body names.

    Returns
    -------
    body_id : (H, W) int32
        1-based body id per pixel. 0 = background / no hit.
    body_name : (H, W) object array of strings
    """
    seg = seg_map.squeeze(-1).astype(np.int32)  # 1-based; 0 = no hit
    H, W = seg.shape
    nbody = env.sim.model.nbody
    ngeom = env.sim.model.ngeom
    # The segmentation channel from liberosuite's
    # ``<cam>_segmentation_element`` is the **geom id + 1** (0 = no hit).
    # We must clip into ``[0, ngeom - 1]`` to index ``model.geom_bodyid``;
    # clipping into ``nbody`` (smaller!) used to silently collapse the
    # body of every geom with id >= nbody onto the last body, which is
    # why a LIBERO stove scene with 200+ geoms but fewer bodies used to
    # come out as just "table" + "world".
    safe = np.clip(seg - 1, -1, ngeom - 1)
    body_id = np.where(seg > 0, env.sim.model.geom_bodyid[safe] + 1, 0).astype(np.int32)
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


# ----------------------------------------------------------------------------
# Frame capture.
# ----------------------------------------------------------------------------

def _get_obs(env):
    for obj in (env, getattr(env, "env", None), getattr(env, "_env", None)):
        if obj is None:
            continue
        if hasattr(obj, "_get_observations"):
            return obj._get_observations()
    raise RuntimeError("Could not find _get_observations on env")


def capture_frame(env, camera_names: Sequence[str]):
    """Return per-camera (rgb, depth_metric, seg_element) and gripper pose/open."""
    obs = _get_obs(env)
    per_cam = {}
    for cam in camera_names:
        rgb = np.asarray(obs[f"{cam}_image"], dtype=np.uint8)
        depth = get_real_depth(obs[f"{cam}_depth"], env.sim)
        seg = np.asarray(obs[f"{cam}_segmentation_element"], dtype=np.int32)
        per_cam[cam] = (rgb, depth, seg)
    gripper_pose = get_gripper_pose(env)
    gripper_open = get_gripper_open(env)
    return per_cam, gripper_pose, gripper_open


# ----------------------------------------------------------------------------
# Scene trajectory construction (per camera).
# ----------------------------------------------------------------------------

def build_scene_trajectory(
    env,
    camera_name: str,
    depth_per_t: list,
    body_name_t0: np.ndarray,
    robot_body_set: set,
    body_poses_per_t: dict,
) -> dict:
    """Build the per-camera scene payload from t=0 body ownership + per-frame
    body poses. Skips robot/gripper pixels entirely.

    Returns
    -------
    dict with keys scene_flows, scene_colors, scene_normals,
    scene_visibility, scene_depth_valid_mask, initial_rgb, initial_depth,
    intrinsic, extrinsic, valid_mask, kept_linear_idx.
    """
    T = len(depth_per_t)
    H, W = depth_per_t[0].shape
    K = get_camera_intrinsic(env, camera_name, H, W)
    T_c_w = get_camera_extrinsic_c_w(env, camera_name)
    T_w_c = get_camera_extrinsic_w_c(env, camera_name)
    cam_pos = T_w_c[:3, 3]

    depth0 = depth_per_t[0]
    points_t0 = backproject_depth(depth0, K, T_c_w)  # (H, W, 3)
    normals_t0 = estimate_normals_from_depth(points_t0, cam_pos)
    rgb0 = _get_obs(env)[f"{camera_name}_image"]
    rgb0 = np.asarray(rgb0, dtype=np.uint8)

    points_flat = points_t0.reshape(-1, 3)
    normals_flat = normals_t0.reshape(-1, 3)
    colors_flat = rgb0.reshape(-1, 3)

    # Scene pixels = valid depth AND non-robot/gripper body.
    valid_depth = (depth0 > 0.01) & (depth0 < 5.0) & np.isfinite(depth0)
    valid_depth_flat = valid_depth.reshape(-1)
    is_robot = np.array(
        [body_name_t0.flat[i] in robot_body_set for i in range(body_name_t0.size)],
        dtype=bool,
    ).reshape(-1)
    valid = valid_depth_flat & ~is_robot
    keep_idx = np.where(valid)[0]
    points_kept = points_flat[keep_idx]
    owning_body_kept = body_name_t0.reshape(-1)[keep_idx]

    # Build pose cache restricted to the bodies that actually own points.
    used_bodies = set(str(b) for b in owning_body_kept if b)
    body_t0_inv = {}
    for body in used_bodies:
        T0 = body_poses_per_t.get((body, 0))
        if T0 is not None:
            body_t0_inv[body] = np.linalg.inv(T0)

    scene_flows_kept = track_points_through_poses(
        points_kept.astype(np.float32),
        owning_body_kept,
        body_t0_inv,
        body_poses_per_t,
    )  # (T, N_kept, 3)

    # Re-pack into (T, H*W, 3) with zeros for invalid pixels.
    N_total = H * W
    full_flows = np.zeros((T, N_total, 3), dtype=np.float32)
    full_normals = np.zeros((T, N_total, 3), dtype=np.float32)
    full_colors = np.zeros((T, N_total, 3), dtype=np.uint8)
    full_visibility = np.zeros((T, N_total), dtype=bool)
    full_depth_valid = np.zeros((T, N_total), dtype=bool)

    for t in range(T):
        full_flows[t, keep_idx, :] = scene_flows_kept[t]
        full_normals[t, keep_idx, :] = normals_flat[keep_idx]
        full_colors[t, keep_idx, :] = colors_flat[keep_idx]
        depth_t_flat = depth_per_t[t].reshape(-1)
        full_depth_valid[t] = (depth_t_flat > 0.01) & (depth_t_flat < 5.0)
        # Visibility: scene is "visible" if the pixel still has a valid depth
        # at frame t; proper occlusion is left to the model.
        full_visibility[t] = full_depth_valid[t] & valid

    return {
        "scene_flows": full_flows,
        "scene_colors": full_colors,
        "scene_normals": full_normals,
        "scene_visibility": full_visibility,
        "scene_depth_valid_mask": full_depth_valid,
        "initial_rgb": rgb0,
        "initial_depth": depth0.astype(np.float32),
        "intrinsic": K,
        "extrinsic": T_c_w.astype(np.float32),
        "valid_mask": valid.reshape(H, W),
        "kept_linear_idx": keep_idx,
    }


# ----------------------------------------------------------------------------
# Robot gripper mesh point flow.
# ----------------------------------------------------------------------------

def build_robot_trajectory(
    env,
    gripper_body_names: list,
    body_poses_per_t: dict,
    n_per_body: int = 64,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample gripper mesh points in body-local frame at t=0, then track them
    through the per-frame body poses. Returns (T, N, 3) world flows, normals,
    and a magenta (T, N, 3) color array."""
    local_points, body_for_point = sample_gripper_mesh_points(
        env, gripper_body_names, n_per_body=n_per_body, seed=seed
    )
    if local_points.shape[0] == 0:
        N = 0
    else:
        N = local_points.shape[0]

    # Discover T from the cache.
    if body_poses_per_t:
        T_total = 1 + max(t for _, t in body_poses_per_t.keys())
    else:
        T_total = T_FRAMES

    robot_flows = np.zeros((T_total, N, 3), dtype=np.float32)
    robot_normals = np.zeros((T_total, N, 3), dtype=np.float32)
    robot_colors = np.zeros((T_total, N, 3), dtype=np.uint8)
    robot_colors[..., 0] = 255
    robot_colors[..., 2] = 255

    if N == 0:
        return robot_flows, robot_normals, robot_colors

    # Compute body-local normals (outward from the body's local origin; this
    # is only a visual prior since the model uses normals as a feature).
    body_origin = np.zeros(3, dtype=np.float32)
    normals_local = local_points - body_origin
    n = np.linalg.norm(normals_local, axis=-1, keepdims=True) + 1e-6
    normals_local = normals_local / n

    for i, body in enumerate(body_for_point):
        T_b0 = body_poses_per_t.get((body, 0))
        if T_b0 is None:
            continue
        R_0 = T_b0[:3, :3].astype(np.float32)
        t_0 = T_b0[:3, 3].astype(np.float32)
        # World position at t=0.
        p_world_0 = local_points[i] @ R_0.T + t_0
        robot_flows[0, i] = p_world_0
        # World normal at t=0: rotate body-local normal by R_0.
        n_world_0 = normals_local[i] @ R_0.T
        robot_normals[0, i] = n_world_0
        for t in range(1, T_total):
            T_t = body_poses_per_t.get((body, t), T_b0)
            R_t = T_t[:3, :3].astype(np.float32)
            t_t = T_t[:3, 3].astype(np.float32)
            robot_flows[t, i] = local_points[i] @ R_t.T + t_t
            robot_normals[t, i] = normals_local[i] @ R_t.T

    return robot_flows, robot_normals, robot_colors


# ----------------------------------------------------------------------------
# Body pose snapshot helper.
# ----------------------------------------------------------------------------

def snapshot_body_poses(env, body_names: list, t_idx: int, cache: dict) -> None:
    """Fill ``cache[(body, t_idx)] = T_w_b`` for every body that still exists
    in the current sim state. Bodies that disappeared (e.g. a removed object)
    are simply skipped."""
    for body in body_names:
        try:
            cache[(body, t_idx)] = get_body_pose(env, body)
        except Exception:
            continue


# ----------------------------------------------------------------------------
# CLI.
# ----------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Export a single 11-frame LIBERO clip as a PointWorld-ready npz. "
            "Run this in the LIBERO conda env, e.g.:\n"
            "  python -m tools.libero.export_clip \\\n"
            "      --demo_hdf5 /path/to/demo.hdf5 \\\n"
            "      --demo_id demo_0 \\\n"
            "      --start_idx 100 \\\n"
            "      --output /tmp/libero_clip.npz"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--demo_hdf5", required=True,
                   help="Path to a LIBERO demo.hdf5 file.")
    p.add_argument("--demo_id", default="demo_0",
                   help="Demo group name inside the HDF5 (default: demo_0).")
    p.add_argument("--start_idx", type=int, required=True,
                   help="Index in the demo to use as the context frame.")
    p.add_argument(
        "--bddl",
        default=None,
        help=(
            "Path to the BDDL file for this task. If omitted, the exporter "
            "falls back to (a) the BDDL referenced inside the demo group's "
            "env_args (legacy recording format) or (b) the BDDL with the "
            "same stem as the HDF5 (real LIBERO dataset layout: "
            "``<task>_demo.hdf5`` -> ``<task>.bddl``) found in the standard "
            "BDDL search paths."
        ),
    )
    p.add_argument(
        "--bddl_search_dir",
        action="append",
        default=None,
        help=(
            "Additional directory to search for a BDDL file when the HDF5 "
            "doesn't reference one. May be passed multiple times. Defaults "
            "to the standard LIBERO ``bddl_files/{suite}/`` layout."
        ),
    )
    p.add_argument("--camera_names", nargs="+",
                   default=["agentview", "robot0_eye_in_hand"],
                   help="Camera names (default: agentview robot0_eye_in_hand).")
    p.add_argument("--camera_height", type=int, default=H_RELEASE)
    p.add_argument("--camera_width", type=int, default=W_RELEASE)
    p.add_argument("--gripper_eef_body", default="gripper0_eef",
                   help="MuJoCo body name for the gripper EEF reference frame.")
    p.add_argument("--robot_points_per_body", type=int, default=64,
                   help="Mesh vertices to sample per gripper sub-body.")
    p.add_argument("--output", "-o", required=True, help="Output .npz path.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    t_start = time.time()

    # ----------------------------------------------------------------
    # Build the env *exactly* from the demo's own metadata.
    # ----------------------------------------------------------------
    env, h5_file, demo_group = make_libero_env_from_demo(
        args.demo_hdf5, args.demo_id, args.camera_names,
        args.camera_height, args.camera_width,
        bddl=args.bddl,
        bddl_search_dirs=args.bddl_search_dir,
    )
    try:
        actions = np.asarray(demo_group["actions"], dtype=np.float32)
        states = np.asarray(demo_group["states"], dtype=np.float32)

        if args.start_idx < 0 or args.start_idx + (T_FRAMES - 1) >= len(actions):
            raise ValueError(
                f"start_idx={args.start_idx} out of range for demo with "
                f"{len(actions)} actions (need at least {T_FRAMES - 1} steps left)."
            )

        # Reset sim to the chosen demo state, following the canonical LIBERO
        # replay contract.
        env.sim.set_state_from_flattened(states[args.start_idx])
        env.sim.forward()

        # ----------------------------------------------------------------
        # Identify bodies: scene vs robot.
        # ----------------------------------------------------------------
        robot_body_set = set(list_robot_body_names(env))
        gripper_body_names = get_gripper_body_names(env)
        # Bodies whose poses we want to snapshot for tracking.
        all_body_names = []
        for i in range(env.sim.model.nbody):
            name = env.sim.model.body(i).name
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="ignore")
            if name:
                all_body_names.append(name)

        body_poses_per_t: dict[tuple[str, int], np.ndarray] = {}

        # ----------------------------------------------------------------
        # Capture frame 0 (context).
        # ----------------------------------------------------------------
        per_cam0, gpose0, gopen0 = capture_frame(env, args.camera_names)
        depth_per_t: dict[str, list] = {c: [d for _, d, _ in [per_cam0[c]]]
                                         for c in args.camera_names}
        rgb0_per_cam: dict[str, np.ndarray] = {c: per_cam0[c][0] for c in args.camera_names}
        seg0_per_cam: dict[str, np.ndarray] = {c: per_cam0[c][2] for c in args.camera_names}
        body_name0_per_cam: dict[str, np.ndarray] = {}
        for c in args.camera_names:
            _, body_name = get_per_pixel_bodies(env, seg0_per_cam[c])
            body_name0_per_cam[c] = body_name

        gripper_poses = [gpose0]
        gripper_opens = [gopen0]
        snapshot_body_poses(env, all_body_names, 0, body_poses_per_t)

        # ----------------------------------------------------------------
        # Replay T_FRAMES - 1 more steps and snapshot.
        # ----------------------------------------------------------------
        for k in range(T_FRAMES - 1):
            action = actions[args.start_idx + k]
            env.step(action)
            per_cam, gpose, gopen = capture_frame(env, args.camera_names)
            for c in args.camera_names:
                _, depth, _ = per_cam[c]
                depth_per_t[c].append(depth)
            gripper_poses.append(gpose)
            gripper_opens.append(gopen)
            snapshot_body_poses(env, all_body_names, k + 1, body_poses_per_t)

        # ----------------------------------------------------------------
        # Build the in-memory sample.
        # ----------------------------------------------------------------
        sample = empty_clip()
        sample["__key__"] = f"{args.demo_id}-{args.start_idx}:{args.start_idx + T_FRAMES - 1}"

        for i, cam in enumerate(args.camera_names):
            prefix = f"camera_{i}"
            payload = build_scene_trajectory(
                env, cam, depth_per_t[cam], body_name0_per_cam[cam],
                robot_body_set, body_poses_per_t,
            )
            sample["scene_flows_per_cam"][prefix] = payload["scene_flows"]
            sample["scene_colors_per_cam"][prefix] = payload["scene_colors"]
            sample["scene_normals_per_cam"][prefix] = payload["scene_normals"]
            sample["scene_visibility_per_cam"][prefix] = payload["scene_visibility"]
            sample["scene_depth_valid_mask_per_cam"][prefix] = payload["scene_depth_valid_mask"]
            sample["initial_rgb_per_cam"][prefix] = payload["initial_rgb"]
            sample["initial_depth_per_cam"][prefix] = payload["initial_depth"]
            sample["intrinsic_per_cam"][prefix] = payload["intrinsic"]
            sample["extrinsic_per_cam"][prefix] = payload["extrinsic"]

        # ----------------------------------------------------------------
        # Gripper mesh point flow (Panda hand + two fingers).
        # ----------------------------------------------------------------
        if gripper_body_names:
            robot_flows, robot_normals, robot_colors = build_robot_trajectory(
                env, gripper_body_names, body_poses_per_t,
                n_per_body=args.robot_points_per_body,
            )
        else:
            print("WARNING: no gripper bodies found; emitting empty robot_flows",
                  file=sys.stderr)
            robot_flows = np.zeros((T_FRAMES, 0, 3), dtype=np.float32)
            robot_normals = np.zeros((T_FRAMES, 0, 3), dtype=np.float32)
            robot_colors = np.zeros((T_FRAMES, 0, 3), dtype=np.uint8)

        sample["robot_flows"] = robot_flows
        sample["robot_normals"] = robot_normals
        sample["robot_colors"] = robot_colors
        sample["right_gripper_pose"] = np.stack(gripper_poses, axis=0)
        sample["right_gripper_open"] = np.array(
            gripper_opens, dtype=np.float32
        ).reshape(T_FRAMES, 1)
    finally:
        h5_file.close()

    # ----------------------------------------------------------------
    # Save.
    # ----------------------------------------------------------------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_npz(sample, str(out_path))

    print(
        f"Saved {out_path}  "
        f"(T={T_FRAMES}, robot_points={robot_flows.shape[1]}, "
        f"cameras={list(args.camera_names)}, "
        f"gripper_bodies={gripper_body_names})  "
        f"elapsed={time.time() - t_start:.1f}s",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
