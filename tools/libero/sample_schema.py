# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Schema for the in-memory LIBERO clip npz consumed by PointWorld.

A single ``libero_clip.npz`` is the boundary between the LIBERO environment
and the PointWorld model. The two pieces of code are not required to share a
Python interpreter, only this file format.

Conventions
-----------
* All arrays are float32 unless noted.
* Camera prefixes follow the PointWorld convention ``camera_<i>_*`` so the
  existing :func:`dataset_components.cameras.sample_cameras` can rename them
  to ``cam0_*`` / ``cam1_*`` without modification.
* Camera intrinsics ``camera_<i>_intrinsic`` are a 3x3 pinhole matrix in
  OpenCV convention (cx = W/2, cy = H/2).
* Camera extrinsics ``camera_<i>_extrinsic`` are a 4x4 **world-to-camera**
  transform ``T_c_w`` -- i.e. applying the matrix to a homogeneous world
  point gives the corresponding camera-frame point. This matches what
  PointWorld's renderer expects in ``extr`` (see the ``@property`` of
  :class:`dataset_components.scene_featurizer.PTRenderer`).

    p_cam = T_c_w @ p_world

  The exporter computes this by inverting the camera-to-world pose returned
  by :func:`robosuite.utils.camera_utils.get_camera_extrinsic_matrix`.
"""

from __future__ import annotations

import numpy as np


# Release contract constants.
T_FRAMES = 11                 # 1 context + 10 predicted frames
CONTEXT_HORIZON = 1
H_RELEASE = 180
W_RELEASE = 320

# Two fixed oblique views plus birdview provide complementary coverage of
# tabletop objects. The inference bridge keeps their on-disk order fixed.
DEFAULT_CAMERAS = ("camera_0", "camera_1")
DEFAULT_CAMERA_NAMES = ("frontview", "sideview", "birdview")


def _as_float32(x):
    return np.asarray(x, dtype=np.float32)


def _as_uint8(x):
    return np.asarray(x, dtype=np.uint8)


def _as_bool(x):
    return np.asarray(x, dtype=bool)


def empty_clip() -> dict:
    """Return a fresh dict with the canonical keys and the right dtypes/shapes.

    All per-camera scene tensors are pre-allocated with N=0; the exporter
    fills them once it has run the LIBERO replay.
    """
    sample = {
        "__key__": "",
        "scene_flows_per_cam": {},
        "scene_colors_per_cam": {},
        "scene_normals_per_cam": {},
        "scene_visibility_per_cam": {},
        "scene_depth_valid_mask_per_cam": {},
        "initial_rgb_per_cam": {},
        "initial_depth_per_cam": {},
        "intrinsic_per_cam": {},
        "extrinsic_per_cam": {},
        "robot_flows": np.zeros((T_FRAMES, 0, 3), dtype=np.float32),
        "robot_normals": np.zeros((T_FRAMES, 0, 3), dtype=np.float32),
        "robot_colors": np.zeros((T_FRAMES, 0, 3), dtype=np.uint8),
        "right_gripper_pose": np.tile(
            np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32), (T_FRAMES, 1)
        ),
        "right_gripper_open": np.zeros((T_FRAMES, 1), dtype=np.float32),
        "camera_names": np.array([], dtype=object),
        "point_object_names": np.array([], dtype=object),
    }
    return sample


def save_npz(sample: dict, path: str) -> None:
    """Flatten the nested per-camera dicts into PointWorld-style flat keys
    and write a single .npz file.

    The on-disk schema exactly matches the input expected by
    :func:`dataset_components.cameras.sample_cameras` plus the
    ``robot_*`` / ``right_gripper_*`` fields.
    """
    flat: dict[str, np.ndarray] = {}
    for prefix, arr in sample["scene_flows_per_cam"].items():
        flat[f"{prefix}_scene_flows"] = _as_float32(arr)
    for prefix, arr in sample["scene_colors_per_cam"].items():
        flat[f"{prefix}_scene_colors"] = _as_uint8(arr)
    for prefix, arr in sample["scene_normals_per_cam"].items():
        flat[f"{prefix}_scene_normals"] = _as_float32(arr)
    for prefix, arr in sample["scene_visibility_per_cam"].items():
        flat[f"{prefix}_scene_visibility"] = _as_bool(arr)
    for prefix, arr in sample["scene_depth_valid_mask_per_cam"].items():
        flat[f"{prefix}_scene_depth_valid_mask"] = _as_bool(arr)
    for prefix, arr in sample["initial_rgb_per_cam"].items():
        flat[f"{prefix}_initial_rgb"] = _as_uint8(arr)
    for prefix, arr in sample["initial_depth_per_cam"].items():
        flat[f"{prefix}_initial_depth"] = _as_float32(arr)
    for prefix, arr in sample["intrinsic_per_cam"].items():
        flat[f"{prefix}_intrinsic"] = _as_float32(arr)
    for prefix, arr in sample["extrinsic_per_cam"].items():
        flat[f"{prefix}_extrinsic"] = _as_float32(arr)

    flat["robot_flows"] = _as_float32(sample["robot_flows"])
    flat["robot_normals"] = _as_float32(sample["robot_normals"])
    flat["robot_colors"] = _as_uint8(sample["robot_colors"])
    flat["right_gripper_pose"] = _as_float32(sample["right_gripper_pose"])
    flat["right_gripper_open"] = _as_float32(sample["right_gripper_open"])

    # Metadata.
    flat["__key__"] = np.array(sample["__key__"], dtype=object)
    flat["camera_names"] = np.array(
        sample.get("camera_names", []), dtype=object
    )
    flat["point_object_names"] = np.array(
        sample.get("point_object_names", []), dtype=object
    )

    np.savez(path, **flat)


def load_npz(path: str) -> dict:
    """Load a clip npz and validate basic shape/dtype contracts.

    Raises ``ValueError`` if anything is off; the caller is expected to treat
    that as a hard failure (no silent truncation).
    """
    raw = np.load(path, allow_pickle=True)

    # Identify per-camera scene keys.
    scene_prefixes = sorted({
        k.split("_scene_")[0]
        for k in raw.files
        if "_scene_" in k and not k.startswith("scene_")
    })
    if not scene_prefixes:
        raise ValueError(f"No camera-prefixed scene_* keys found in {path}")

    cam_prefixes = sorted({
        k.split("_initial_")[0]
        for k in raw.files
        if "_initial_rgb" in k or "_initial_depth" in k
    })
    if set(scene_prefixes) != set(cam_prefixes):
        raise ValueError(
            f"Scene prefixes {scene_prefixes} do not match camera prefixes {cam_prefixes}"
        )

    sample = {
        "__key__": str(raw["__key__"]) if "__key__" in raw.files else "",
        "scene_flows_per_cam": {},
        "scene_colors_per_cam": {},
        "scene_normals_per_cam": {},
        "scene_visibility_per_cam": {},
        "scene_depth_valid_mask_per_cam": {},
        "initial_rgb_per_cam": {},
        "initial_depth_per_cam": {},
        "intrinsic_per_cam": {},
        "extrinsic_per_cam": {},
        "robot_flows": np.asarray(raw["robot_flows"], dtype=np.float32),
        "robot_normals": np.asarray(raw["robot_normals"], dtype=np.float32),
        "robot_colors": np.asarray(raw["robot_colors"], dtype=np.uint8),
        "right_gripper_pose": np.asarray(raw["right_gripper_pose"], dtype=np.float32),
        "right_gripper_open": np.asarray(raw["right_gripper_open"], dtype=np.float32),
        "camera_names": (
            list(raw["camera_names"])
            if "camera_names" in raw.files
            else []
        ),
        "point_object_names": (
            list(raw["point_object_names"])
            if "point_object_names" in raw.files
            else []
        ),
    }

    for prefix in cam_prefixes:
        sf = raw[f"{prefix}_scene_flows"]
        if sf.ndim != 3 or sf.shape[0] != T_FRAMES or sf.shape[2] != 3:
            raise ValueError(
                f"{prefix}_scene_flows must be (T={T_FRAMES}, N, 3), got {sf.shape}"
            )
        sample["scene_flows_per_cam"][prefix] = sf.astype(np.float32)
        sample["scene_colors_per_cam"][prefix] = raw[f"{prefix}_scene_colors"].astype(np.uint8)
        sample["scene_normals_per_cam"][prefix] = raw[f"{prefix}_scene_normals"].astype(np.float32)

        if f"{prefix}_scene_visibility" in raw.files:
            sample["scene_visibility_per_cam"][prefix] = raw[f"{prefix}_scene_visibility"].astype(bool)
        if f"{prefix}_scene_depth_valid_mask" in raw.files:
            sample["scene_depth_valid_mask_per_cam"][prefix] = raw[f"{prefix}_scene_depth_valid_mask"].astype(bool)

        rgb = raw[f"{prefix}_initial_rgb"]
        if rgb.shape != (H_RELEASE, W_RELEASE, 3):
            raise ValueError(
                f"{prefix}_initial_rgb must be ({H_RELEASE},{W_RELEASE},3), got {rgb.shape}"
            )
        sample["initial_rgb_per_cam"][prefix] = rgb.astype(np.uint8)

        depth = raw[f"{prefix}_initial_depth"]
        if depth.shape != (H_RELEASE, W_RELEASE):
            raise ValueError(
                f"{prefix}_initial_depth must be ({H_RELEASE},{W_RELEASE}), got {depth.shape}"
            )
        sample["initial_depth_per_cam"][prefix] = depth.astype(np.float32)

        K = raw[f"{prefix}_intrinsic"]
        if K.shape != (3, 3):
            raise ValueError(f"{prefix}_intrinsic must be (3,3), got {K.shape}")
        sample["intrinsic_per_cam"][prefix] = K.astype(np.float32)

        E = raw[f"{prefix}_extrinsic"]
        if E.shape != (4, 4):
            raise ValueError(f"{prefix}_extrinsic must be (4,4), got {E.shape}")
        sample["extrinsic_per_cam"][prefix] = E.astype(np.float32)

    # Robot tensors.
    if sample["robot_flows"].shape[0] != T_FRAMES or sample["robot_flows"].shape[2] != 3:
        raise ValueError(
            f"robot_flows must be (T={T_FRAMES}, Nr, 3), got {sample['robot_flows'].shape}"
        )
    if sample["right_gripper_pose"].shape != (T_FRAMES, 7):
        raise ValueError(
            f"right_gripper_pose must be ({T_FRAMES}, 7), got {sample['right_gripper_pose'].shape}"
        )
    if sample["right_gripper_open"].shape != (T_FRAMES, 1):
        raise ValueError(
            f"right_gripper_open must be ({T_FRAMES}, 1), got {sample['right_gripper_open'].shape}"
        )

    return sample


def flatten_for_pointworld(sample: dict) -> dict:
    """Convert the in-memory sample into the flat dict expected by
    :func:`dataset_components.cameras.sample_cameras` /
    :func:`dataset_components.pipeline.apply_release_pipeline_to_sample`.

    Returns a fresh dict (does not mutate the input).
    """
    out: dict = {"__key__": sample["__key__"]}
    for prefix in sample["scene_flows_per_cam"]:
        out[f"{prefix}_scene_flows"] = sample["scene_flows_per_cam"][prefix]
        out[f"{prefix}_scene_colors"] = sample["scene_colors_per_cam"][prefix]
        out[f"{prefix}_scene_normals"] = sample["scene_normals_per_cam"][prefix]
        if prefix in sample["scene_visibility_per_cam"]:
            out[f"{prefix}_scene_visibility"] = sample["scene_visibility_per_cam"][prefix]
        if prefix in sample["scene_depth_valid_mask_per_cam"]:
            out[f"{prefix}_scene_depth_valid_mask"] = sample["scene_depth_valid_mask_per_cam"][prefix]
        out[f"{prefix}_initial_rgb"] = sample["initial_rgb_per_cam"][prefix]
        out[f"{prefix}_initial_depth"] = sample["initial_depth_per_cam"][prefix]
        out[f"{prefix}_intrinsic"] = sample["intrinsic_per_cam"][prefix]
        out[f"{prefix}_extrinsic"] = sample["extrinsic_per_cam"][prefix]

    out["robot_flows"] = sample["robot_flows"]
    out["robot_normals"] = sample["robot_normals"]
    out["robot_colors"] = sample["robot_colors"]
    out["right_gripper_pose"] = sample["right_gripper_pose"]
    out["right_gripper_open"] = sample["right_gripper_open"]
    if "camera_names" in sample:
        out["camera_names"] = np.asarray(sample["camera_names"], dtype=object)
    if "point_object_names" in sample:
        out["point_object_names"] = np.asarray(
            sample["point_object_names"], dtype=object
        )
    return out


__all__ = [
    "T_FRAMES",
    "CONTEXT_HORIZON",
    "H_RELEASE",
    "W_RELEASE",
    "DEFAULT_CAMERAS",
    "DEFAULT_CAMERA_NAMES",
    "empty_clip",
    "save_npz",
    "load_npz",
    "flatten_for_pointworld",
]
