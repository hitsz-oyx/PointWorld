# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Export a single 11-frame LIBERO clip as a PointWorld-ready .npz.

Usage::

    python tools/libero/export_clip.py \
        --benchmark libero_spatial \
        --task_id 0 \
        --demo_hdf5 /path/to/libero_spatial/pick_up_the_black_bowl/demo.hdf5 \
        --demo_id demo_0 \
        --start_idx 100 \
        --output /tmp/libero_clip.npz

This script must be run inside a Python environment that has LIBERO and its
dependencies (MuJoCo, robosuite) installed. It does *not* import any
PointWorld code; the only contract is the .npz file format defined in
:mod:`tools.libero.sample_schema`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

# LIBERO imports. These are only available in the LIBERO env.
try:
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv  # type: ignore
except ImportError as e:  # pragma: no cover - import error path
    print(
        "FATAL: this script must be run inside the LIBERO environment. "
        f"Failed to import libero: {e}",
        file=sys.stderr,
    )
    raise

from .sample_schema import (
    T_FRAMES,
    H_RELEASE,
    W_RELEASE,
    DEFAULT_CAMERA_NAMES,
    empty_clip,
    save_npz,
)
from .scene_geometry import (
    backproject_depth,
    estimate_normals_from_depth,
    find_owning_body,
    get_body_pose,
    get_camera_extrinsic,
    get_camera_intrinsic,
    get_gripper_open,
    get_gripper_pose,
    list_non_robot_body_names,
    track_points_through_poses,
)


# ----------------------------------------------------------------------------
# Environment construction.
# ----------------------------------------------------------------------------

def make_libero_env(
    benchmark_name: str,
    task_id: int | None,
    task_name: str | None,
    camera_names: Sequence[str],
    height: int,
    width: int,
):
    """Create a LIBERO OffScreenRenderEnv with the requested camera config."""
    benchmark_dict = benchmark.get_benchmark_dict()
    if benchmark_name not in benchmark_dict:
        raise ValueError(
            f"Unknown benchmark '{benchmark_name}'. Known: {list(benchmark_dict)}"
        )
    task_suite = benchmark_dict[benchmark_name]()

    task = None
    if task_id is not None:
        task = task_suite.get_task(task_id=task_id)
    elif task_name is not None:
        for t in task_suite.tasks:
            if t.name == task_name:
                task = t
                break
        if task is None:
            raise ValueError(
                f"Task '{task_name}' not found in benchmark '{benchmark_name}'"
            )
    else:
        raise ValueError("Either --task_id or --task_name must be provided")

    env = OffScreenRenderEnv(
        bddl_file_name=task.bddl_file,
        camera_names=list(camera_names),
        camera_widths=width,
        camera_heights=height,
        camera_depths=True,
    )
    return env, task


# ----------------------------------------------------------------------------
# Frame capture helpers.
# ----------------------------------------------------------------------------

def _get_obs(env):
    """Wrapper that handles nested env wrappers (e.g. Task wrapper)."""
    # OffScreenRenderEnv sometimes wraps the inner env; try a few paths.
    for obj in (env, getattr(env, "env", None), getattr(env, "_env", None)):
        if obj is None:
            continue
        if hasattr(obj, "_get_observations"):
            return obj._get_observations()
    raise RuntimeError("Could not find _get_observations on env")


def capture_frame(env, camera_names: Sequence[str]):
    """Return per-camera (rgb, depth) and the current gripper pose/open."""
    obs = _get_obs(env)
    per_cam = {}
    for cam in camera_names:
        rgb = np.asarray(obs[f"{cam}_image"], dtype=np.uint8)
        depth = np.asarray(obs[f"{cam}_depth"], dtype=np.float32)
        per_cam[cam] = (rgb, depth)
    gripper_pose = get_gripper_pose(env)
    gripper_open = get_gripper_open(env)
    return per_cam, gripper_pose, gripper_open


# ----------------------------------------------------------------------------
# Robot surface points (smoke-test placeholder).
# ----------------------------------------------------------------------------

def sample_robot_points_around_body(env, body_name: str, n_points: int,
                                    seed: int = 0) -> np.ndarray:
    """Sample points uniformly inside a sphere around the gripper EEF.

    This is a placeholder for proper mesh-surface sampling. The points are
    anchored to the body's frame at t=0 and tracked via the body pose in
    later frames, so even this rough proxy gives the model a recognizable
    robot trajectory.
    """
    body_id = env.sim.model.body_name2id(body_name)
    xpos = np.asarray(env.sim.data.body(body_id).xpos, dtype=np.float64)
    rbound = float(env.sim.model.body(body_id).rbound)
    r = max(rbound, 0.05)  # at least 5 cm so the model can see them

    rng = np.random.RandomState(seed)
    v = rng.randn(n_points, 3)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    r_rand = r * rng.uniform(0.3, 1.0, size=(n_points, 1))
    return (xpos[None, :] + v * r_rand).astype(np.float32)


# ----------------------------------------------------------------------------
# Per-camera scene trajectory construction.
# ----------------------------------------------------------------------------

def build_scene_trajectory_for_camera(
    env_at_t0,
    depth_per_t: list,
    camera_name: str,
    candidate_bodies: list,
    body_poses_per_t: dict,
):
    """Given a list of (T,) depth frames and a pose cache, build the per-camera
    scene payload expected by PointWorld.

    Returns a dict with the keys:
        scene_flows, scene_colors, scene_normals, scene_visibility,
        scene_depth_valid_mask, initial_rgb, initial_depth, intrinsic, extrinsic.
    """
    T = len(depth_per_t)
    H, W = depth_per_t[0].shape
    K = get_camera_intrinsic(env_at_t0, camera_name, H, W)
    T_w_c = get_camera_extrinsic(env_at_t0, camera_name)
    cam_pos = T_w_c[:3, 3]

    depth0 = depth_per_t[0]
    points_t0 = backproject_depth(depth0, K, T_w_c)  # (H, W, 3)
    normals_t0 = estimate_normals_from_depth(points_t0, cam_pos)  # (H, W, 3)
    rgb0 = _get_obs(env_at_t0)[f"{camera_name}_image"]
    rgb0 = np.asarray(rgb0, dtype=np.uint8)

    points_flat = points_t0.reshape(-1, 3)
    depth_flat = depth0.reshape(-1)
    normals_flat = normals_t0.reshape(-1, 3)
    colors_flat = rgb0.reshape(-1, 3)

    valid = (depth_flat > 0.01) & (depth_flat < 5.0) & np.isfinite(depth_flat)
    N_total = points_flat.shape[0]
    keep_idx = np.where(valid)[0]
    points_kept = points_flat[keep_idx]
    normals_kept = normals_flat[keep_idx]
    colors_kept = colors_flat[keep_idx]

    owning_body = []
    body_t0_inv = {}
    for p in points_kept:
        body = find_owning_body(env_at_t0, p, candidate_bodies)
        owning_body.append(body)
        if body is not None and body not in body_t0_inv:
            T_b0 = body_poses_per_t.get((body, 0))
            if T_b0 is not None:
                body_t0_inv[body] = np.linalg.inv(T_b0)

    owning_arr = np.array(owning_body, dtype=object)
    scene_flows_kept = track_points_through_poses(
        points_kept, owning_arr, body_t0_inv, body_poses_per_t
    )  # (T, N_kept, 3)

    # Re-pack into (T, H*W, 3) with zeros for invalid pixels.
    full_flows = np.zeros((T, N_total, 3), dtype=np.float32)
    full_normals = np.zeros((T, N_total, 3), dtype=np.float32)
    full_colors = np.zeros((T, N_total, 3), dtype=np.uint8)
    full_visibility = np.zeros((T, N_total), dtype=bool)
    full_depth_valid = np.zeros((T, N_total), dtype=bool)

    for t in range(T):
        full_flows[t, keep_idx, :] = scene_flows_kept[t]
        # For non-t=0 frames, the surface point may have moved slightly
        # (rigid body motion); we re-project to get visibility and color
        # by reading the corresponding frame's RGB/depth. We do not change
        # color/normal per-frame in this smoke version; they stay at t=0.
        full_normals[t, keep_idx, :] = normals_kept
        full_colors[t, keep_idx, :] = colors_kept
        depth_t_flat = depth_per_t[t].reshape(-1)
        full_depth_valid[t] = (depth_t_flat > 0.01) & (depth_t_flat < 5.0)
        # Visibility: scene is "visible" if the projected depth at the
        # owning body is close to the rendered depth. For the smoke test
        # we mark visible if depth_valid; proper occlusion is out of scope.
        full_visibility[t] = full_depth_valid[t]

    return {
        "scene_flows": full_flows.reshape(T, H, W, 3).reshape(T, H * W, 3),
        "scene_colors": full_colors.reshape(T, H, W, 3).reshape(T, H * W, 3),
        "scene_normals": full_normals.reshape(T, H, W, 3).reshape(T, H * W, 3),
        "scene_visibility": full_visibility.reshape(T, H * W),
        "scene_depth_valid_mask": full_depth_valid.reshape(T, H * W),
        "initial_rgb": rgb0,
        "initial_depth": depth0,
        "intrinsic": K,
        "extrinsic": T_w_c.astype(np.float32),
        "owning_body": owning_arr,
        "valid_mask": valid,
    }


# ----------------------------------------------------------------------------
# Main entry point.
# ----------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Export a single 11-frame LIBERO clip as a PointWorld-ready npz. "
            "This script runs in the LIBERO environment and does not import "
            "any PointWorld code."
        )
    )
    p.add_argument("--benchmark", required=True,
                   help="LIBERO benchmark name (e.g. libero_spatial, libero_10).")
    p.add_argument("--task_id", type=int, default=None,
                   help="Task index inside the benchmark (0-indexed).")
    p.add_argument("--task_name", type=str, default=None,
                   help="Task name inside the benchmark (alternative to --task_id).")
    p.add_argument("--demo_hdf5", required=True,
                   help="Path to a LIBERO demo.hdf5 file.")
    p.add_argument("--demo_id", default="demo_0",
                   help="Demo group name inside the HDF5 (default: demo_0).")
    p.add_argument("--start_idx", type=int, required=True,
                   help="Index in the demo to use as the context frame.")
    p.add_argument("--camera_names", nargs="+", default=list(DEFAULT_CAMERA_NAMES),
                   help=f"Camera names (default: {' '.join(DEFAULT_CAMERA_NAMES)}).")
    p.add_argument("--camera_height", type=int, default=H_RELEASE)
    p.add_argument("--camera_width", type=int, default=W_RELEASE)
    p.add_argument("--robot_body", default="gripper0_eef",
                   help="MuJoCo body name for the robot gripper EEF.")
    p.add_argument("--robot_points", type=int, default=128,
                   help="Number of robot surface points to sample (smoke placeholder).")
    p.add_argument("--output", "-o", required=True, help="Output .npz path.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    t_start = time.time()
    env, task = make_libero_env(
        args.benchmark,
        args.task_id,
        args.task_name,
        args.camera_names,
        args.camera_height,
        args.camera_width,
    )

    # ----------------------------------------------------------------
    # Load the demo and pick the start state.
    # ----------------------------------------------------------------
    with h5py.File(args.demo_hdf5, "r") as f:
        demo_group = f[f"data/{args.demo_id}"]
        actions = np.asarray(demo_group["actions"], dtype=np.float32)
        states = np.asarray(demo_group["states"], dtype=np.float32)

    if args.start_idx < 0 or args.start_idx + (T_FRAMES - 1) >= len(actions):
        raise ValueError(
            f"start_idx={args.start_idx} out of range for demo with "
            f"{len(actions)} actions (need at least {T_FRAMES - 1} steps left)."
        )

    # ----------------------------------------------------------------
    # Replay the demo and snapshot per-frame state.
    # ----------------------------------------------------------------
    env.reset()
    env.regenerate_obs_from_state(states[args.start_idx])

    candidate_bodies = list_non_robot_body_names(env)
    print(f"Found {len(candidate_bodies)} candidate non-robot bodies: "
          f"{candidate_bodies[:5]}{'...' if len(candidate_bodies) > 5 else ''}",
          file=sys.stderr)

    depth_per_t: dict[str, list] = {c: [] for c in args.camera_names}
    rgb0_per_cam: dict[str, np.ndarray] = {}
    gripper_poses = []
    gripper_opens = []
    body_poses_per_t: dict[tuple[str, int], np.ndarray] = {}

    def snapshot_body_poses(t_idx: int):
        for body in candidate_bodies:
            try:
                body_poses_per_t[(body, t_idx)] = get_body_pose(env, body)
            except Exception:
                # Body name disappeared (e.g. a removed object); skip.
                continue

    # Frame 0 (context).
    per_cam, gpose, gopen = capture_frame(env, args.camera_names)
    for cam in args.camera_names:
        rgb, depth = per_cam[cam]
        rgb0_per_cam[cam] = rgb
        depth_per_t[cam].append(depth)
    gripper_poses.append(gpose)
    gripper_opens.append(gopen)
    snapshot_body_poses(0)

    # Frames 1..10 (predicted).
    for k in range(T_FRAMES - 1):
        action = actions[args.start_idx + k]
        env.step(action)
        per_cam, gpose, gopen = capture_frame(env, args.camera_names)
        for cam in args.camera_names:
            _, depth = per_cam[cam]
            depth_per_t[cam].append(depth)
        gripper_poses.append(gpose)
        gripper_opens.append(gopen)
        snapshot_body_poses(k + 1)

    # ----------------------------------------------------------------
    # Build the per-camera scene payload at t=0 (intrinsic / extrinsic
    # only need the t=0 sim state; the per-frame depths and body poses
    # are used to construct the trajectories).
    # ----------------------------------------------------------------
    # Reset the sim back to t=0 to read K, T_w_c, RGB at t=0 cleanly.
    env.reset()
    env.regenerate_obs_from_state(states[args.start_idx])
    t0_obs = _get_obs(env)

    sample = empty_clip()
    sample["__key__"] = (
        f"{args.benchmark}-{task.name}-{args.demo_id}-"
        f"{args.start_idx}:{args.start_idx + T_FRAMES - 1}"
    )

    for i, cam in enumerate(args.camera_names):
        prefix = f"camera_{i}"
        payload = build_scene_trajectory_for_camera(
            env, depth_per_t[cam], cam, candidate_bodies, body_poses_per_t
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
    # Robot trajectory: sample points in gripper local frame at t=0,
    # then track them via the gripper body pose over time.
    # ----------------------------------------------------------------
    env.reset()
    env.regenerate_obs_from_state(states[args.start_idx])
    # Build gripper pose cache for the robot trajectory.
    gripper_pose_cache = {}
    try:
        gripper_pose_cache[("gripper", 0)] = get_body_pose(env, args.robot_body)
    except Exception as e:
        raise RuntimeError(
            f"Could not look up body '{args.robot_body}'. "
            f"Available bodies: "
            f"{[env.sim.model.body(i).name for i in range(env.sim.model.nbody)]}"
        ) from e
    env.reset()
    env.regenerate_obs_from_state(states[args.start_idx])
    for k in range(T_FRAMES - 1):
        env.step(actions[args.start_idx + k])
        try:
            gripper_pose_cache[("gripper", k + 1)] = get_body_pose(env, args.robot_body)
        except Exception:
            continue

    env.reset()
    env.regenerate_obs_from_state(states[args.start_idx])
    robot_local = sample_robot_points_around_body(env, args.robot_body, args.robot_points)
    # Convert world->local at t=0.
    T_g0 = gripper_pose_cache[("gripper", 0)]
    T_g0_inv = np.linalg.inv(T_g0)
    ones = np.ones((robot_local.shape[0], 1), dtype=np.float32)
    robot_local_h = np.concatenate([robot_local, ones], axis=-1)
    p_local = (T_g0_inv @ robot_local_h.T).T[:, :3]
    # Forward to world at each t.
    robot_flows = np.zeros((T_FRAMES, args.robot_points, 3), dtype=np.float32)
    for t in range(T_FRAMES):
        T_gt = gripper_pose_cache.get(("gripper", t), T_g0)
        p_world_h = T_gt @ np.concatenate([p_local, ones], axis=-1).T
        robot_flows[t] = p_world_h.T[:, :3].astype(np.float32)
    # Normals: outward from gripper center at t=0.
    robot_normals = (robot_local - T_g0[:3, 3].astype(np.float32))
    robot_normals = robot_normals / (np.linalg.norm(robot_normals, axis=-1, keepdims=True) + 1e-12)
    robot_normals = np.broadcast_to(robot_normals, (T_FRAMES, args.robot_points, 3)).astype(np.float32)
    # Colors: magenta (1, 0, 1) as per PointWorld convention.
    robot_colors = np.zeros((T_FRAMES, args.robot_points, 3), dtype=np.uint8)
    robot_colors[..., 0] = 255
    robot_colors[..., 2] = 255

    sample["robot_flows"] = robot_flows
    sample["robot_normals"] = robot_normals
    sample["robot_colors"] = robot_colors
    sample["right_gripper_pose"] = np.stack(gripper_poses, axis=0)
    sample["right_gripper_open"] = np.array(gripper_opens, dtype=np.float32).reshape(T_FRAMES, 1)

    # ----------------------------------------------------------------
    # Save.
    # ----------------------------------------------------------------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_npz(sample, str(out_path))

    print(
        f"Saved {out_path}  "
        f"(T={T_FRAMES}, robot_points={args.robot_points}, "
        f"cameras={list(args.camera_names)})  "
        f"elapsed={time.time() - t_start:.1f}s",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
