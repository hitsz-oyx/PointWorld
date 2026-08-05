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

import re

import numpy as np
import torch

from dataset_components.utils import _stable_int_hash


def _camera_prefix_sort_key(prefix: str) -> tuple[int, str]:
    match = re.fullmatch(r"camera_(\d+)", prefix)
    if match is None:
        return (10**9, prefix)
    return (int(match.group(1)), prefix)


def _discover_camera_prefixes(sample):
    camera_prefixes = set()
    for key in sample.keys():
        if '__' not in key and "_scene_flows" in key and key != "scene_flows":
            camera_prefixes.add(key.split("_scene_")[0])
    return sorted(camera_prefixes, key=_camera_prefix_sort_key)


def _apply_camera_selection(sample, selected_prefixes):
    # Create mapping from old prefixes to standardized prefixes
    prefix_mapping = {}
    for i, old_prefix in enumerate(selected_prefixes):
        prefix_mapping[old_prefix] = f"cam{i}"

    # Dynamically discover scene attributes from the keys
    scene_attributes = set()
    for key in sample.keys():
        for prefix in selected_prefixes:
            if key.startswith(f"{prefix}_scene_"):
                attribute = key[len(prefix) + 1:]
                scene_attributes.add(attribute)

    # Initialize outputs without prefixes
    for attr in scene_attributes:
        sample[attr] = None

    # Process each selected camera prefix for the flow data
    for old_prefix in selected_prefixes:
        for attr in scene_attributes:
            prefixed_key = f"{old_prefix}_{attr}"
            assert prefixed_key in sample, f"Attribute {prefixed_key} not found in sample"
            if sample[attr] is None:
                sample[attr] = sample[prefixed_key]
            else:
                if sample[attr].shape[0] != sample[prefixed_key].shape[0]:
                    raise ValueError(
                        f"Time dimension mismatch for {attr}: "
                        f"{sample[attr].shape[0]} vs {sample[prefixed_key].shape[0]}"
                    )
                sample[attr] = np.concatenate(
                    [sample[attr], sample[prefixed_key]], axis=1
                )

    # Handle image data with standardized prefixes.
    image_suffixes = ['_initial_rgb', '_initial_depth', '_intrinsic', '_extrinsic']
    for old_prefix in selected_prefixes:
        new_prefix = prefix_mapping[old_prefix]
        for suffix in image_suffixes:
            old_key = f"{old_prefix}{suffix}"
            new_key = f"{new_prefix}{suffix}"
            if old_key in sample:
                sample[new_key] = sample[old_key]

    # Keep a small audit trail for downstream debug / metadata export.
    sample["__selected_camera_prefixes__"] = np.asarray(
        selected_prefixes, dtype=object
    )
    if "camera_names" in sample:
        camera_names = list(sample["camera_names"])
        selected_names = []
        for prefix in selected_prefixes:
            match = re.fullmatch(r"camera_(\d+)", prefix)
            if match is None:
                selected_names.append(prefix)
                continue
            idx = int(match.group(1))
            if 0 <= idx < len(camera_names):
                selected_names.append(camera_names[idx])
            else:
                selected_names.append(prefix)
        sample["camera_names"] = np.asarray(selected_names, dtype=object)

    # Remove scene-related prefixed keys and non-selected camera image keys to save memory
    camera_prefixes = _discover_camera_prefixes(sample)
    keys_to_remove = []
    for key in sample.keys():
        for prefix in camera_prefixes:
            if key.startswith(f"{prefix}_scene_"):
                keys_to_remove.append(key)

    for key in sample.keys():
        for prefix in camera_prefixes:
            for suffix in image_suffixes:
                if key.startswith(f"{prefix}{suffix}"):
                    keys_to_remove.append(key)

    for key in keys_to_remove:
        sample.pop(key, None)

    assert 'scene_flows' in sample, "No scene flows found after sampling cameras"
    assert sample['scene_flows'].shape[1] > 0, "No points left after sampling cameras"
    assert 'robot_flows' in sample, "No robot flows found after sampling cameras"
    assert sample['robot_flows'].shape[1] > 0, "No points left after sampling cameras"

    return sample


def select_cameras_in_order(sample, num_cameras: int):
    """Select the first ``num_cameras`` exported cameras in their on-disk order.

    This is the deterministic counterpart to :func:`sample_cameras` and is
    useful for release-style evaluation where ``camera_0`` / ``camera_1``
    already encode the intended fixed view pair.
    """
    camera_prefixes = _discover_camera_prefixes(sample)
    assert len(camera_prefixes) > 0, "No camera prefixes found in sample"
    num_cameras = int(num_cameras)
    if num_cameras < 1:
        raise ValueError(f"num_cameras must be >= 1, got {num_cameras}")
    if len(camera_prefixes) < num_cameras:
        raise ValueError(
            f"Requested {num_cameras} cameras but sample only has "
            f"{len(camera_prefixes)}: {camera_prefixes}"
        )
    selected_prefixes = camera_prefixes[:num_cameras]
    return _apply_camera_selection(sample, selected_prefixes)


def sample_cameras(sample, min_num_cameras=1, max_num_cameras=None,
                   deterministic: bool = False, seed: int | None = None):
    """
    Randomly samples scene data from a subset of available cameras, concatenates them,
    and removes the camera prefixes from the keys. Selected cameras are renamed to
    standardized prefixes (cam0, cam1, etc.).

    Args:
        sample (dict): Sample dictionary that may contain camera-prefixed entries
        min_num_cameras (int | None): Minimum number of cameras to sample
            (None = auto lower bound, clipped by sample availability).
        max_num_cameras (int | None): Maximum number of cameras to sample
            (None = use all available in the sample).
    Returns:
        sample (dict): Updated sample with concatenated scene data and standardized camera prefixes
    """
    # Find all unique camera prefixes in the sample
    camera_prefixes = _discover_camera_prefixes(sample)
    assert len(camera_prefixes) > 0, "No camera prefixes found in sample"

    # Determine how many cameras to sample
    num_cameras = len(camera_prefixes)
    if min_num_cameras is None:
        min_num_cameras = 1
    if max_num_cameras is None:
        max_num_cameras = num_cameras

    # Ensure min_num_cameras doesn't exceed available cameras
    min_num_cameras = min(max(int(min_num_cameras), 1), num_cameras)

    # Ensure max_num_cameras doesn't exceed available cameras
    max_num_cameras = min(max(int(max_num_cameras), 1), num_cameras)
    max_num_cameras = max(max_num_cameras, min_num_cameras)

    # RNG: deterministic per-sample if requested
    if deterministic:
        key = str(sample.get('__key__', ''))
        base = 0 if seed is None else int(seed)
        rs = np.random.RandomState(_stable_int_hash(base, key, 'cam'))
    else:
        rs = np.random
    # Sample a number of cameras between min and max
    num_cameras_to_sample = rs.randint(min_num_cameras, max_num_cameras + 1)
    # Randomly select camera prefixes
    selected_prefixes = rs.choice(
        camera_prefixes, size=num_cameras_to_sample, replace=False
    )

    return _apply_camera_selection(sample, selected_prefixes)


def convert_to_tensors(sample):
    """
    Converts each npy array in the sample to a torch.Tensor.
    Modify this as needed for your downstream model (e.g. float32).
    """
    tensor_sample = {}
    for k, v in sample.items():
        if isinstance(v, np.ndarray):
            try:
                tensor_sample[k] = torch.from_numpy(v)
            except ValueError as e:
                if 'with array.copy()' in str(e):
                    tensor_sample[k] = torch.from_numpy(v.copy())
                else:
                    raise e
            if v.dtype in [np.float64, np.float32, np.float16]:
                tensor_sample[k] = tensor_sample[k].float()
        else:
            tensor_sample[k] = v
    return tensor_sample
