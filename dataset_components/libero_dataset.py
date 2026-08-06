# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""LIBERO ``.npz`` torch Dataset.

This is the LIBERO fine-tuning counterpart to the DROID/BEHAVIOR
WebDataset pipeline in :mod:`dataset_components.dataloader`.  It
loads ``.npz`` files produced by
:mod:`tools.libero.export_clip_native` (or the upstream
:mod:`tools.libero.export_clip` if EGL is available), runs them
through the LIBERO-specific single-sample processing chain
(:func:`tools.libero.sample_schema.load_npz` +
:func:`tools.libero.sample_schema.flatten_for_pointworld` +
:func:`dataset_components.cameras.select_cameras_in_order` +
:func:`dataset_components.robot.canonicalize_gripper_keys_and_flags` +
:func:`dataset_components.pipeline.apply_release_pipeline_to_sample`),
and yields the same dict shape the WDS path returns after its
``.map(...)`` stack.

Differences vs the WDS path
---------------------------
* The LIBERO path needs a **fixed camera order** across train and
  test (``frontview``, ``sideview`` then ``birdview`` by default)
  because the oblique_triplet layout captures
  complementary coverage that we don't want to randomly permute at
  training time.  We use
  :func:`dataset_components.cameras.select_cameras_in_order` instead
  of :func:`dataset_components.cameras.sample_cameras`.
* The WDS path's :func:`dataset_components.decoders.build_flow_sample`
  constructs the sample from raw tar members; we already have the
  full PointWorld-shape sample in the ``.npz`` so we skip that step.
* We use a regular ``torch.utils.data.Dataset`` + ``DataLoader`` so
  multiprocess workers can be used without coordinating WDS
  resampling / shard shuffling (which is unnecessary for our
  pre-windowed clips).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np
from torch.utils.data import Dataset

from dataset_components.cameras import select_cameras_in_order
from dataset_components.pipeline import apply_release_pipeline_to_sample
from dataset_components.robot import canonicalize_gripper_keys_and_flags
from tools.libero.sample_schema import flatten_for_pointworld, load_npz


POINTWORLD_STEP_SECONDS = 0.1


def validate_temporal_metadata(sample: dict, path: Path) -> None:
    required = (
        "source_control_freq_hz",
        "frame_step",
        "model_step_seconds",
    )
    missing = [key for key in required if sample.get(key) is None]
    if missing:
        raise ValueError(
            f"LIBERO clip {path} lacks temporal metadata {missing}. This is "
            "usually a legacy 20 Hz consecutive-frame export covering only "
            "0.5 seconds; re-export it with --frame_step 2."
        )
    control_hz = float(sample["source_control_freq_hz"])
    frame_step = int(sample["frame_step"])
    model_dt = float(sample["model_step_seconds"])
    if control_hz <= 0 or frame_step < 1:
        raise ValueError(
            f"Invalid temporal metadata in {path}: control_hz={control_hz}, "
            f"frame_step={frame_step}."
        )
    derived_dt = frame_step / control_hz
    if not np.isclose(model_dt, derived_dt, atol=1e-6):
        raise ValueError(
            f"Inconsistent temporal metadata in {path}: model_step_seconds={model_dt} "
            f"but frame_step/control_hz={derived_dt}."
        )
    if not np.isclose(model_dt, POINTWORLD_STEP_SECONDS, atol=1e-6):
        raise ValueError(
            f"LIBERO clip {path} uses {model_dt}s per model step; PointWorld "
            f"requires {POINTWORLD_STEP_SECONDS}s."
        )


class LiberoNPZDataset(Dataset):
    """Map-style ``Dataset`` over a directory of LIBERO ``.npz`` clips.

    Each ``__getitem__`` returns the dict produced by the standard
    LIBERO single-sample processing chain, with the same final
    fields the WDS path yields (after ``convert_to_tensors``,
    ``gather``, ``compute_helper_variables``).
    """

    def __init__(
        self,
        root: str,
        mode: str,
        args,
        num_cameras: int = 2,
        domain: str = "libero",
        file_glob: str = "*.npz",
        file_list: Optional[Sequence[str]] = None,
    ) -> None:
        if mode not in ("train", "test"):
            raise ValueError(f"LiberoNPZDataset mode must be 'train' or 'test', got {mode!r}")
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(
                f"LiberoNPZDataset root not found: {self.root.resolve()}"
            )
        if file_list is not None:
            self.files: List[Path] = [Path(f) for f in file_list]
        else:
            self.files = sorted(self.root.rglob(file_glob))
        if not self.files:
            raise RuntimeError(
                f"No LIBERO .npz files found under {self.root} "
                f"(glob={file_glob!r})"
            )
        self.mode = mode
        self.args = args
        self.num_cameras = int(num_cameras)
        self.domain = domain

    def __len__(self) -> int:
        return len(self.files)

    # ------------------------------------------------------------------
    # Per-sample processing chain.
    # ------------------------------------------------------------------
    def _process_sample(self, sample: dict) -> dict:
        sample = select_cameras_in_order(
            sample, num_cameras=self.num_cameras,
        )
        sample = canonicalize_gripper_keys_and_flags(sample)
        sample = apply_release_pipeline_to_sample(
            sample=sample,
            domain=self.domain,
            mode=self.mode,
            args=self.args,
            has_bimanual_robot=False,
        )
        return sample

    def __getitem__(self, index: int) -> dict:
        path = self.files[index]
        # ``load_npz`` re-opens the file (and re-does allow_pickle /
        # object-array handling).  We pass through it for consistency
        # with the eval scripts.  When debugging large datasets it is
        # much faster to inline ``np.load(..., allow_pickle=True)``
        # here and skip the round-trip.
        raw = load_npz(str(path))
        if getattr(self.args, "libero_require_temporal_metadata", True):
            validate_temporal_metadata(raw, path)
        sample = flatten_for_pointworld(raw)
        sample["__domain__"] = self.domain
        sample["__key__"] = path.stem
        return self._process_sample(sample)


__all__ = ["LiberoNPZDataset", "validate_temporal_metadata"]
