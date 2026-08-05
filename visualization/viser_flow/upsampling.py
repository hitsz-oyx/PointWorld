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

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional

import numpy as np

from .timeline import PointTimeline


_GREEN = np.array([0, 255, 0], dtype=np.float32)


@dataclasses.dataclass(slots=True)
class VoxelAssignment:
    background_points: np.ndarray
    background_colors: np.ndarray
    initial_points: np.ndarray
    flow_to_points: List[np.ndarray]
    static_indices: np.ndarray
    background_flow_indices: np.ndarray

    def build_frame(
        self,
        flow_positions: np.ndarray,
        flow_exists: np.ndarray,
        *,
        supervised: Optional[np.ndarray] = None,
        tint_alpha: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build one dense frame without materializing the whole timeline.

        This is the interactive path used while a slider moves. The cached
        background-to-flow lookup keeps the update vectorized rather than
        iterating through every sparse flow point in Python.
        """
        positions = np.asarray(flow_positions, dtype=np.float32)
        exists = np.asarray(flow_exists, dtype=bool)
        if positions.shape != self.initial_points.shape:
            raise ValueError(
                "flow_positions must have shape "
                f"{self.initial_points.shape}, got {positions.shape}"
            )
        if exists.shape != self.initial_points.shape[:1]:
            raise ValueError(
                "flow_exists must have shape "
                f"{self.initial_points.shape[:1]}, got {exists.shape}"
            )
        if self.background_flow_indices.shape != self.background_points.shape[:1]:
            raise ValueError("background_flow_indices shape mismatch")

        supervised_mask = None
        if supervised is not None:
            supervised_mask = np.asarray(supervised, dtype=bool)
            if supervised_mask.shape != exists.shape:
                raise ValueError("supervised mask must match flow_exists shape")

        flow_indices = self.background_flow_indices
        assigned = flow_indices >= 0
        active = assigned.copy()
        active[assigned] = exists[flow_indices[assigned]]
        dynamic_indices = np.flatnonzero(active)

        static_points = self.background_points[self.static_indices]
        static_colors = self.background_colors[self.static_indices]
        if dynamic_indices.size == 0:
            return (
                static_points.astype(np.float32, copy=False),
                static_colors.astype(np.uint8, copy=False),
            )

        dynamic_flows = flow_indices[dynamic_indices]
        deltas = positions[dynamic_flows] - self.initial_points[dynamic_flows]
        dynamic_points = self.background_points[dynamic_indices] + deltas
        dynamic_colors = self.background_colors[dynamic_indices].astype(
            np.float32, copy=True
        )
        if supervised_mask is not None:
            unsupervised = ~supervised_mask[dynamic_flows]
            if np.any(unsupervised):
                dynamic_colors[unsupervised] = (
                    (1.0 - float(tint_alpha)) * dynamic_colors[unsupervised]
                    + float(tint_alpha) * _GREEN
                )
        return (
            np.concatenate([static_points, dynamic_points], axis=0).astype(
                np.float32, copy=False
            ),
            np.concatenate(
                [static_colors, dynamic_colors.clip(0.0, 255.0).astype(np.uint8)],
                axis=0,
            ).astype(np.uint8, copy=False),
        )

    def build_timeline(
        self,
        flow_positions: np.ndarray,
        flow_exists: np.ndarray,
        *,
        supervised: Optional[np.ndarray] = None,
        tint_alpha: float = 0.5,
    ) -> PointTimeline:
        positions = np.asarray(flow_positions, dtype=np.float32)
        exists = np.asarray(flow_exists, dtype=bool)
        if positions.ndim != 3 or positions.shape[-1] != 3:
            raise ValueError("flow_positions must have shape (T, N, 3)")
        if exists.shape != positions.shape[:2]:
            raise ValueError("flow_exists must have shape (T, N)")

        if supervised is not None:
            supervised_mask = np.asarray(supervised, dtype=bool)
            if supervised_mask.shape != exists.shape:
                raise ValueError("supervised mask must match flow_exists shape")
        else:
            supervised_mask = None

        T = positions.shape[0]
        frames: List[np.ndarray] = []
        frame_colors: List[np.ndarray] = []

        for t in range(T):
            frame_points, frame_colors_t = self.build_frame(
                positions[t],
                exists[t],
                supervised=(None if supervised_mask is None else supervised_mask[t]),
                tint_alpha=tint_alpha,
            )
            frames.append(frame_points)
            frame_colors.append(frame_colors_t)

        return PointTimeline(frames, frame_colors)


def build_voxel_assignment(
    background_points: np.ndarray,
    background_colors: np.ndarray,
    flow_positions: np.ndarray,
    flow_exists: np.ndarray,
    *,
    grid_size: float,
) -> VoxelAssignment:
    if grid_size <= 0.0:
        raise ValueError("grid_size must be positive for voxel assignment")

    bg_pts = np.asarray(background_points, dtype=np.float32)
    bg_cols = np.asarray(background_colors, dtype=np.uint8)
    flows = np.asarray(flow_positions, dtype=np.float32)
    exists = np.asarray(flow_exists, dtype=bool)

    if flows.ndim != 3 or flows.shape[-1] != 3:
        raise ValueError("flow_positions must have shape (T, N, 3)")
    if exists.shape != flows.shape[:2]:
        raise ValueError("flow_exists must match flow_positions shape")

    initial_positions = flows[0]
    initial_exists = exists[0]

    voxel_map: Dict[tuple[int, int, int], List[int]] = {}
    for idx in range(initial_positions.shape[0]):
        if not initial_exists[idx]:
            continue
        key = tuple(np.floor(initial_positions[idx] / grid_size).astype(np.int64))
        voxel_map.setdefault(key, []).append(idx)

    assigned = np.zeros((bg_pts.shape[0],), dtype=bool)
    background_flow_indices = np.full((bg_pts.shape[0],), -1, dtype=np.int64)
    flow_to_points: List[List[int]] = [[] for _ in range(initial_positions.shape[0])]

    for i, point in enumerate(bg_pts):
        key = tuple(np.floor(point / grid_size).astype(np.int64))
        candidates = voxel_map.get(key)
        if not candidates:
            continue
        if len(candidates) == 1:
            chosen = candidates[0]
        else:
            initial_positions_subset = initial_positions[candidates]
            distances = np.linalg.norm(initial_positions_subset - point, axis=1)
            chosen = candidates[int(np.argmin(distances))]
        assigned[i] = True
        background_flow_indices[i] = chosen
        flow_to_points[chosen].append(i)

    flow_point_arrays: List[np.ndarray] = []
    for lst in flow_to_points:
        if lst:
            flow_point_arrays.append(np.asarray(lst, dtype=np.int64))
        else:
            flow_point_arrays.append(np.empty((0,), dtype=np.int64))

    static_indices = np.nonzero(~assigned)[0]

    return VoxelAssignment(
        background_points=bg_pts,
        background_colors=bg_cols,
        initial_points=initial_positions.astype(np.float32, copy=False),
        flow_to_points=flow_point_arrays,
        static_indices=static_indices.astype(np.int64, copy=False),
        background_flow_indices=background_flow_indices,
    )
