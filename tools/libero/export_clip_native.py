# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Native-mujoco-renderer variant of :mod:`tools.libero.export_clip`.

This script produces the exact same ``libero_clip.npz`` schema as
:mod:`tools.libero.export_clip` but uses ``mujoco.Renderer`` directly
(backed by Mesa's OSMesa CPU renderer) instead of the
``OffScreenRenderEnv`` / robosuite offscreen rendering path. The latter
needs an EGL device that is unavailable on many headless GPU servers
(``/dev/dri/renderD*`` is usually cgroup-restricted to root, and the
``EGL_EXT_platform_device`` query on a remote EGL display frequently
returns no usable devices).

Why a separate file
-------------------
``export_clip.py`` is the source of truth when the LIBERO conda env
ships a working EGL stack. This file is the **fallback** for the cases
where EGL is unavailable.  Keeping the two renderers in separate files
means we can ship the native path upstream without breaking the
official EGL path and without having to maintain a complex
``# EGL vs OSMesa`` switch inside a single 1000-line script.

What is different vs ``export_clip.py``
---------------------------------------
* Env creation: ``ControlEnv(use_camera_obs=False,
  has_offscreen_renderer=False, has_renderer=False)`` -- the EGL
  context is never instantiated, the env is only used as a builder
  for the underlying mujoco model/data.
* State setting: directly poke ``data.qpos`` / ``data.qvel`` + call
  ``mujoco.mj_forward``. We don't use
  ``env.regenerate_obs_from_state`` (which depends on the env's
  observable cache, which we explicitly avoided by setting
  ``use_camera_obs=False``).
* Frame capture: we call ``mujoco.Renderer`` directly.  This gives us
  RGB + **metric-meter** depth (no need for the
  ``get_real_depth_map`` normalized->metric conversion that the
  robosuite path does) + element segmentation (which we feed straight
  into :func:`tools.libero.scene_geometry.get_per_pixel_bodies`).

What is reused from ``export_clip.py``
--------------------------------------
* BDDL resolution logic, asset-path rewriting, env-arg parsing.
* :func:`tools.libero.scene_geometry.backproject_depth`
* :func:`tools.libero.scene_geometry.estimate_normals_from_depth`
* :func:`tools.libero.scene_geometry.get_per_pixel_bodies`
* :func:`tools.libero.scene_geometry.get_gripper_body_names`
* :func:`tools.libero.scene_geometry.list_robot_body_names`
* :func:`tools.libero.scene_geometry.sample_gripper_mesh_points`
* :func:`tools.libero.scene_geometry.track_points_through_poses`
* :func:`tools.libero.scene_geometry.snapshot_body_poses`
* :func:`tools.libero.scene_geometry.get_gripper_pose`
* :func:`tools.libero.scene_geometry.get_gripper_open`
* :func:`tools.libero.scene_geometry.get_camera_intrinsic`
* :func:`tools.libero.scene_geometry.get_camera_extrinsic_c_w`
* :func:`tools.libero.scene_geometry.get_camera_extrinsic_w_c`
* :func:`build_scene_trajectory` (private but well-defined)
* :func:`build_robot_trajectory` (private but well-defined)
* :func:`_apply_camera_layout`
* :func:`_make_libero_env_from_demo` / :func:`_resolve_bddl_for_demo`
* :func:`tools.libero.sample_schema.save_npz` / :func:`empty_clip`

Usage
-----
Run with the OSMesa environment variables exported *before* the
interpreter starts (so that the OpenGL backend auto-detector doesn't
pick EGL at import time)::

    unset PYOPENGL_PLATFORM
    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \\
        python -m tools.libero.export_clip_native \\
        --demo_hdf5 /path/to/<task>_demo.hdf5 \\
        --demo_id demo_0 \\
        --start_idx 0 \\
        --output /tmp/clip_demo0_0.npz
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Force the OSMesa / native-mujoco path BEFORE anything imports
# ``OpenGL`` / ``mujoco`` / ``robosuite``.  In particular, ``robosuite``'s
# ``binding_utils`` module switches between ``EGLGLContext`` and
# ``OSMesaGLContext`` based on ``MUJOCO_GL`` at import time.
# ---------------------------------------------------------------------------
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "osmesa"
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

# Now safe to import project-local shim + libero + mujoco.
from tools.libero import _egl_compat  # noqa: E402

_egl_compat.apply()

import h5py  # noqa: E402
import numpy as np  # noqa: E402

# Reuse the original BDDL / env-arg / camera-layout / traj-builder code
# from ``export_clip`` -- we deliberately don't reimplement them here.
from tools.libero import export_clip as _exp  # noqa: E402
from tools.libero import sample_schema  # noqa: E402
from tools.libero.scene_geometry import (  # noqa: E402
    backproject_depth,
    estimate_normals_from_depth,
    get_camera_extrinsic_c_w,
    get_camera_extrinsic_w_c,
    get_camera_intrinsic,
    get_gripper_body_names,
    get_gripper_open,
    get_gripper_pose,
    get_per_pixel_bodies,
    list_robot_body_names,
    sample_gripper_mesh_points,
    snapshot_body_poses,
    track_points_through_poses,
)
from tools.libero.camera_layout import (  # noqa: E402
    CAMERA_LAYOUT_OBLIQUE_TRIPLET,
)

import mujoco  # noqa: E402

# Re-export the constants the rest of the project imports from
# ``sample_schema`` so callers that ``import from export_clip_native``
# don't have to think about the indirection.
T_FRAMES = sample_schema.T_FRAMES
H_RELEASE = sample_schema.H_RELEASE
W_RELEASE = sample_schema.W_RELEASE
empty_clip = sample_schema.empty_clip
save_npz = sample_schema.save_npz
crop_scene_to_workspace = sample_schema.crop_scene_to_workspace


# ---------------------------------------------------------------------------
# State setting.
# ---------------------------------------------------------------------------

def _set_state_from_flattened(
    env,
    model: "mujoco.MjModel",
    data: "mujoco.MjData",
    flat: np.ndarray,
) -> None:
    """Set the raw mujoco state from a (nq + nv + ...) flattened buffer.

    LIBERO's HDF5 ``states`` arrays are length 92 (= nq + nv for the
    standard Panda + 1 table object set); the first element is a
    ``time``/timestamp field, then ``qpos`` (length ``model.nq``), then
    ``qvel`` (length ``model.nv``).  The robosuite wrapper has its own
    ``set_state_from_flattened`` that also pokes the ``mocap_*``
    addresses, but on the no-renderer env we instantiate here the
    mocap bodies are usually empty so the raw mujoco call is enough.
    """
    nq, nv = model.nq, model.nv
    # ``flat[0]`` is the timestamp; skip it.
    if flat.shape[0] < 1 + nq + nv:
        raise ValueError(
            f"flat state too short: got {flat.shape[0]} need at least "
            f"1 + nq={nq} + nv={nv} = {1 + nq + nv}"
        )
    data.qpos[:] = flat[1:1 + nq]
    data.qvel[:] = flat[1 + nq:1 + nq + nv]
    if hasattr(data, "mocap_pos") and data.mocap_pos is not None and data.mocap_pos.shape[0] > 0:
        # Best-effort copy: only the first few elements; if the demo
        # has no mocap bodies these are zeros.
        nmocap = data.mocap_pos.shape[0]
        if 1 + nq + nv + 7 * nmocap <= flat.shape[0]:
            data.mocap_pos[:] = flat[
                1 + nq + nv:1 + nq + nv + 3 * nmocap
            ].reshape(nmocap, 3)
            data.mocap_quat[:] = flat[
                1 + nq + nv + 3 * nmocap:1 + nq + nv + 7 * nmocap
            ].reshape(nmocap, 4)
    mujoco.mj_forward(model, data)


# ---------------------------------------------------------------------------
# Frame capture (native mujoco.Renderer path).
# ---------------------------------------------------------------------------

def capture_frame_native(
    renderer: "mujoco.Renderer",
    model: "mujoco.MjModel",
    data: "mujoco.MjData",
    env,
    camera_names,
    height: int,
    width: int,
):
    """Render RGB + depth (metric) + seg-element for each camera.

    Returns ``(per_cam, gripper_pose, gripper_open)`` in the same
    shape as :func:`tools.libero.export_clip.capture_frame`:

    * ``per_cam[camera]`` is a ``(rgb_uint8, depth_float32, seg_int32)``
      triple where ``seg`` has shape ``(H, W, 1)`` and is the **geom
      id** (the format :func:`get_per_pixel_bodies` expects).
    """
    per_cam = {}
    for cam in camera_names:
        # --- RGB ---
        renderer.update_scene(data, camera=cam)
        rgb = np.asarray(renderer.render(), dtype=np.uint8)
        # --- Depth (already in metric meters) ---
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=cam)
        depth = np.asarray(renderer.render(), dtype=np.float32)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        renderer.disable_depth_rendering()
        # --- Segmentation: (H, W, 2) where seg[..., 0] is geom_id ---
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=cam)
        seg_full = renderer.render()
        renderer.disable_segmentation_rendering()
        # ``seg_full`` is ``(H, W, 2)`` int32: [geom_id, body_id+1].
        # Reshape geom_id channel to ``(H, W, 1)`` so that
        # ``get_per_pixel_bodies`` (which calls ``.squeeze(-1)``) sees
        # the same on-the-wire format that the robosuite
        # ``element_segmentation`` observable produces.
        seg = np.asarray(seg_full[..., 0:1], dtype=np.int32)
        per_cam[cam] = (rgb, depth, seg)
    gripper_pose = get_gripper_pose(env)
    gripper_open = get_gripper_open(env)
    return per_cam, gripper_pose, gripper_open


# ---------------------------------------------------------------------------
# Environment construction (BDDL + camera layout, no EGL context).
# ---------------------------------------------------------------------------

def _build_env_native(
    bddl: str,
    camera_names,
    height: int,
    width: int,
    camera_layout: str,
    model_xml: str | None = None,
):
    """Build a ``ControlEnv`` with the offscreen renderer disabled and
    apply the requested camera layout. Returns ``(env, sim, model, data)``.

    We never call ``env.regenerate_obs_from_state`` -- the env's
    observable cache is invalid because we set ``use_camera_obs=False``,
    and we drive the simulator state directly with
    :func:`_set_state_from_flattened` + ``mujoco.mj_forward``.
    """
    from libero.libero.envs.env_wrapper import ControlEnv  # type: ignore

    env = ControlEnv(
        bddl_file_name=bddl,
        has_offscreen_renderer=False,
        has_renderer=False,
        use_camera_obs=False,
    )
    if model_xml is not None:
        # LIBERO stores the exact MuJoCo XML used to record each demo. It
        # includes model-level fixture poses that are not represented in the
        # flattened qpos/qvel state, so loading it is required for faithful
        # and deterministic reconstruction.
        env.reset()
        env.reset_from_xml_string(model_xml)
        env.sim.reset()
    # The native renderer doesn't use robosuite's camera-resolution
    # knobs (which are baked into the env's offscreen-renderer width
    # and height attributes), but the LIBERO BDDL env's *intrinsic
    # computation* via ``get_camera_intrinsic_matrix`` is derived from
    # the camera's ``fovy`` attribute, which is independent of the
    # offscreen-renderer resolution.  We do however set the per-camera
    # width/height on the renderer itself, and we also adjust the
    # ``cam_pos``/``cam_quat``/``fovy`` for the oblique layouts.
    _exp._apply_camera_layout(
        env, camera_names=camera_names, camera_layout=camera_layout,
    )

    sim = env.env.sim
    model = sim.model._model
    data = sim.data._data
    return env, sim, model, data


# ---------------------------------------------------------------------------
# Main export function.
# ---------------------------------------------------------------------------

class NativeDemoExporter:
    """Reusable native renderer for all windows from one LIBERO demo."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.camera_names = list(args.camera_names)
        self.camera_layout = str(args.camera_layout)
        self.height = int(args.camera_height)
        self.width = int(args.camera_width)

        with h5py.File(args.demo_hdf5, "r") as f:
            if args.demo_id not in f["data"]:
                raise KeyError(
                    f"Demo group {args.demo_id!r} not found in {args.demo_hdf5}. "
                    f"Available: {list(f['data'].keys())[:5]}..."
                )
            demo_group = f[f"data/{args.demo_id}"]
            self.states = np.asarray(demo_group["states"], dtype=np.float32)
            if "model_file" in demo_group.attrs:
                model_file = demo_group.attrs["model_file"]
            elif "model_file" in demo_group:
                model_file = demo_group["model_file"][()]
            else:
                model_file = None
            if isinstance(model_file, bytes):
                model_file = model_file.decode("utf-8")
            if model_file is None:
                if not getattr(args, "allow_bddl_reconstruction", False):
                    raise ValueError(
                        f"Demo group {args.demo_id!r} in {args.demo_hdf5} has no "
                        "model_file. A flattened LIBERO state cannot restore "
                        "model-level fixture poses. Use the official HDF5 with "
                        "embedded model_file, or explicitly pass "
                        "--allow_bddl_reconstruction for a non-faithful preview."
                    )
                model_xml = None
                self.model_source = "bddl_reconstruction"
            else:
                model_xml = _exp.libero_env_utils.postprocess_model_xml(
                    str(model_file), {}
                )
                model_xml = _exp._rewrite_libero_asset_paths(model_xml)
                self.model_source = "embedded_model_file"
            bddl = _exp._resolve_bddl_for_demo(
                demo_hdf5=args.demo_hdf5,
                f=f,
                demo_group=demo_group,
                cli_bddl=args.bddl,
                extra_search_dirs=args.bddl_search_dir,
            )

        self.env, self.sim, self.model, self.data = _build_env_native(
            bddl=bddl,
            camera_names=self.camera_names,
            height=self.height,
            width=self.width,
            camera_layout=self.camera_layout,
            model_xml=model_xml,
        )
        self.renderer = mujoco.Renderer(
            self.model, height=self.height, width=self.width
        )
        self.robot_body_set = set(list_robot_body_names(self.env))
        self.gripper_body_names = get_gripper_body_names(self.env)
        self.all_body_names = []
        for i in range(self.model.nbody):
            name = self.model.body(i).name
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="ignore")
            if name:
                self.all_body_names.append(name)

    def close(self) -> None:
        try:
            self.renderer.close()
        finally:
            self.env.close()

    def export(self, args: argparse.Namespace) -> dict:
        t_start = time.time()
        frame_step = int(
            getattr(args, "frame_step", sample_schema.DEFAULT_FRAME_STEP)
        )
        if frame_step < 1:
            raise ValueError(f"frame_step must be >= 1, got {frame_step}")
        last_raw_idx = int(args.start_idx) + (T_FRAMES - 1) * frame_step
        if args.start_idx < 0 or last_raw_idx >= len(self.states):
            raise ValueError(
                f"start_idx={args.start_idx} out of range for demo with "
                f"{len(self.states)} states (frame_step={frame_step} requires "
                f"last raw index {last_raw_idx})."
            )

        body_poses_per_t: dict = {}
        _set_state_from_flattened(
            self.env, self.model, self.data, self.states[args.start_idx]
        )
        per_cam0, gpose0, gopen0 = capture_frame_native(
            self.renderer, self.model, self.data, self.env,
            self.camera_names, self.height, self.width,
        )
        depth_per_t = {c: [per_cam0[c][1]] for c in self.camera_names}
        rgb0_per_cam = {c: per_cam0[c][0] for c in self.camera_names}
        seg0_per_cam = {c: per_cam0[c][2] for c in self.camera_names}
        body_name0_per_cam = {}
        K_t0_per_cam = {}
        T_c_w_t0_per_cam = {}
        T_w_c_t0_per_cam = {}
        for camera in self.camera_names:
            _, body_name = get_per_pixel_bodies(self.env, seg0_per_cam[camera])
            body_name0_per_cam[camera] = body_name
            K_t0_per_cam[camera] = get_camera_intrinsic(
                self.env, camera, self.height, self.width
            )
            T_w_c_t0_per_cam[camera] = get_camera_extrinsic_w_c(self.env, camera)
            T_c_w_t0_per_cam[camera] = get_camera_extrinsic_c_w(self.env, camera)
        gripper_poses = [gpose0]
        gripper_opens = [gopen0]
        snapshot_body_poses(self.env, self.all_body_names, 0, body_poses_per_t)

        for k in range(T_FRAMES - 1):
            raw_idx = int(args.start_idx) + (k + 1) * frame_step
            _set_state_from_flattened(
                self.env, self.model, self.data, self.states[raw_idx]
            )
            per_cam, gpose, gopen = capture_frame_native(
                self.renderer, self.model, self.data, self.env,
                self.camera_names, self.height, self.width,
            )
            for camera in self.camera_names:
                depth_per_t[camera].append(per_cam[camera][1])
            gripper_poses.append(gpose)
            gripper_opens.append(gopen)
            snapshot_body_poses(
                self.env, self.all_body_names, k + 1, body_poses_per_t
            )

        sample = empty_clip()
        sample["__key__"] = (
            f"{args.demo_id}-{args.start_idx}:{last_raw_idx}:step{frame_step}"
        )
        sample["source_control_freq_hz"] = float(
            getattr(args, "source_control_freq_hz", sample_schema.DEFAULT_CONTROL_FREQ_HZ)
        )
        sample["frame_step"] = frame_step
        sample["model_step_seconds"] = frame_step / sample["source_control_freq_hz"]
        sample["window_stride_raw"] = getattr(args, "window_stride_raw", None)
        sample["camera_names"] = np.asarray(self.camera_names, dtype=object)

        for i, camera in enumerate(self.camera_names):
            prefix = f"camera_{i}"
            payload = _exp.build_scene_trajectory(
                self.env, camera, depth_per_t[camera], body_name0_per_cam[camera],
                self.robot_body_set, body_poses_per_t,
                K_t0=K_t0_per_cam[camera],
                T_c_w_t0=T_c_w_t0_per_cam[camera],
                T_w_c_t0=T_w_c_t0_per_cam[camera],
                rgb_t0=rgb0_per_cam[camera],
            )
            for field in (
                "scene_flows", "scene_colors", "scene_normals",
                "scene_visibility", "scene_depth_valid_mask",
            ):
                sample[f"{field}_per_cam"][prefix] = payload[field]
            sample["initial_rgb_per_cam"][prefix] = payload["initial_rgb"]
            sample["initial_depth_per_cam"][prefix] = payload["initial_depth"]
            sample["intrinsic_per_cam"][prefix] = payload["intrinsic"]
            sample["extrinsic_per_cam"][prefix] = payload["extrinsic"]

        if getattr(args, "workspace_bounds", None) is not None:
            crop_scene_to_workspace(sample, args.workspace_bounds)

        if self.gripper_body_names:
            robot_flows, robot_normals, robot_colors = _exp.build_robot_trajectory(
                self.env, self.gripper_body_names, body_poses_per_t,
                n_per_body=args.robot_points_per_body,
            )
        else:
            robot_flows = np.zeros((T_FRAMES, 0, 3), dtype=np.float32)
            robot_normals = np.zeros((T_FRAMES, 0, 3), dtype=np.float32)
            robot_colors = np.zeros((T_FRAMES, 0, 3), dtype=np.uint8)
        sample["robot_flows"] = robot_flows
        sample["robot_normals"] = robot_normals
        sample["robot_colors"] = robot_colors
        sample["right_gripper_pose"] = np.stack(gripper_poses, axis=0)
        sample["right_gripper_open"] = np.asarray(
            gripper_opens, dtype=np.float32
        ).reshape(T_FRAMES, 1)

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_npz(sample, str(out_path))
        elapsed = time.time() - t_start
        return {
            "out_path": str(out_path),
            "robot_points": int(robot_flows.shape[1]),
            "scene_points_per_cam": {
                key: int(value.shape[1])
                for key, value in sample["scene_flows_per_cam"].items()
            },
            "elapsed_s": elapsed,
            "workspace_bounds": (
                sample["workspace_bounds"].tolist()
                if sample.get("workspace_bounds") is not None else None
            ),
            "start_idx": int(args.start_idx),
            "end_idx": int(last_raw_idx),
            "frame_step": int(frame_step),
            "model_step_seconds": float(sample["model_step_seconds"]),
            "model_source": self.model_source,
        }


def export_clip_native(args: argparse.Namespace) -> dict:
    """Export one clip, using the same reusable runtime as bulk export."""
    exporter = NativeDemoExporter(args)
    try:
        result = exporter.export(args)
    finally:
        exporter.close()
    print(
        f"Saved {result['out_path']} (start={result['start_idx']}, "
        f"end={result['end_idx']}, elapsed={result['elapsed_s']:.1f}s)",
        file=sys.stderr,
    )
    return result


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Native-mujoco-renderer variant of tools.libero.export_clip. "
            "Produces the exact same libero_clip.npz schema, but uses "
            "mujoco.Renderer + OSMesa instead of the robosuite "
            "OffScreenRenderEnv / EGL path.  Use this on headless GPU "
            "servers where /dev/dri is cgroup-locked.\n\n"
            "Run with:\n"
            "  MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python -m "
            "tools.libero.export_clip_native <args>"
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
        "--frame_step", type=int, default=sample_schema.DEFAULT_FRAME_STEP,
        help="Raw 20 Hz LIBERO states per 0.1 s PointWorld step (default: 2).",
    )
    p.add_argument(
        "--window_stride_raw", type=int, default=None,
        help="Optional raw-index window stride recorded as NPZ metadata.",
    )
    p.add_argument("--bddl", default=None,
                   help=(
                       "Path to the BDDL file for this task. If omitted, the "
                       "exporter searches the LIBERO standard BDDL dirs."
                   ))
    p.add_argument("--bddl_search_dir", action="append", default=None,
                   help="Additional BDDL search dir (may repeat).")
    p.add_argument("--camera_names", nargs="+",
                   default=list(sample_schema.DEFAULT_CAMERA_NAMES),
                   help=(
                       "Camera names to render. Defaults to "
                       "``frontview sideview birdview`` (oblique_triplet)."
                   ))
    p.add_argument("--camera_layout", default=CAMERA_LAYOUT_OBLIQUE_TRIPLET,
                   choices=("native", "oblique_pair", "oblique_triplet"),
                   help="Fixed external camera layout (default: oblique_triplet).")
    p.add_argument("--camera_height", type=int, default=H_RELEASE)
    p.add_argument("--camera_width", type=int, default=W_RELEASE)
    p.add_argument("--gripper_eef_body", default="gripper0_eef",
                   help="MuJoCo body name for the gripper EEF.")
    p.add_argument("--robot_points_per_body", type=int, default=64,
                   help="Mesh vertices to sample per gripper sub-body.")
    p.add_argument(
        "--allow_bddl_reconstruction",
        action="store_true",
        help=(
            "Allow demos without embedded model_file by rebuilding from BDDL. "
            "This cannot restore recorded fixture poses and is only suitable "
            "for non-faithful previews."
        ),
    )
    p.add_argument(
        "--workspace_bounds", nargs=6, type=float, default=None,
        metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
        help=(
            "Optional frame-0 LIBERO world-frame crop. Recommended tabletop "
            "bounds: "
            f"{' '.join(str(v) for v in sample_schema.DEFAULT_WORKSPACE_BOUNDS)}."
        ),
    )
    p.add_argument("--output", "-o", required=True, help="Output .npz path.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    export_clip_native(args)


if __name__ == "__main__":
    main()
