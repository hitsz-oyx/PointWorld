# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export one complete LIBERO demo interval as fixed 11-frame windows.

Run this from the LIBERO environment. PointWorld checkpoints have a fixed
1-context + 10-prediction-frame contract, so a longer demo is exported as
overlapping windows and evaluated by ``scripts/eval_libero_trajectory.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py

from tools.libero.export_clip import export_clip_from_args
from tools.libero.camera_layout import (
    CAMERA_LAYOUT_NATIVE,
    CAMERA_LAYOUT_OBLIQUE_PAIR,
    CAMERA_LAYOUT_OBLIQUE_TRIPLET,
)
from tools.libero.sample_schema import (
    DEFAULT_CAMERA_NAMES,
    H_RELEASE,
    T_FRAMES,
    W_RELEASE,
)
from tools.libero.trajectory import plan_trajectory_windows


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--demo_hdf5", required=True)
    p.add_argument("--demo_id", default="demo_0")
    p.add_argument("--start_idx", type=int, required=True)
    p.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="Inclusive final action index (default: final recorded action).",
    )
    p.add_argument(
        "--window_stride",
        type=int,
        default=T_FRAMES - 1,
        help="Window-start stride; default 10 retains each 10-step prediction.",
    )
    p.add_argument("--output_dir", required=True)
    p.add_argument("--bddl", default=None)
    p.add_argument("--bddl_search_dir", action="append", default=None)
    p.add_argument("--camera_names", nargs="+", default=list(DEFAULT_CAMERA_NAMES))
    p.add_argument(
        "--camera_layout",
        choices=[
            CAMERA_LAYOUT_NATIVE,
            CAMERA_LAYOUT_OBLIQUE_PAIR,
            CAMERA_LAYOUT_OBLIQUE_TRIPLET,
        ],
        default=CAMERA_LAYOUT_OBLIQUE_TRIPLET,
    )
    p.add_argument("--camera_height", type=int, default=H_RELEASE)
    p.add_argument("--camera_width", type=int, default=W_RELEASE)
    p.add_argument("--gripper_eef_body", default="gripper0_eef")
    p.add_argument("--robot_points_per_body", type=int, default=64)
    p.add_argument(
        "--trajectory_source",
        choices=["recorded", "replay"],
        default="recorded",
        help="Future-frame source; recorded HDF5 states are the faithful default.",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _last_action_index(demo_hdf5: str, demo_id: str) -> int:
    with h5py.File(demo_hdf5, "r") as f:
        actions = f[f"data/{demo_id}/actions"]
        if len(actions) == 0:
            raise ValueError(f"{demo_id} contains no actions")
        return len(actions) - 1


def main() -> None:
    args = _parse_args()
    end_idx = (
        _last_action_index(args.demo_hdf5, args.demo_id)
        if args.end_idx is None
        else int(args.end_idx)
    )
    windows = plan_trajectory_windows(
        start_idx=int(args.start_idx),
        end_idx=end_idx,
        window_size=T_FRAMES,
        stride=int(args.window_stride),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "demo_hdf5": str(Path(args.demo_hdf5).resolve()),
        "demo_id": str(args.demo_id),
        "start_idx": int(args.start_idx),
        "end_idx": end_idx,
        "window_size": T_FRAMES,
        "window_stride": int(args.window_stride),
        "trajectory_source": str(args.trajectory_source),
        "camera_layout": str(args.camera_layout),
        "windows": [],
    }

    for window in windows:
        output = output_dir / f"clip_{window.start_idx:06d}.npz"
        if output.exists() and not args.overwrite:
            result = {
                "output": str(output),
                "start_idx": window.start_idx,
                "end_idx": window.end_idx,
                "skipped_existing": True,
            }
        else:
            export_args = argparse.Namespace(
                demo_hdf5=args.demo_hdf5,
                demo_id=args.demo_id,
                start_idx=window.start_idx,
                bddl=args.bddl,
                bddl_search_dir=args.bddl_search_dir,
                camera_names=args.camera_names,
                camera_layout=args.camera_layout,
                camera_height=args.camera_height,
                camera_width=args.camera_width,
                gripper_eef_body=args.gripper_eef_body,
                robot_points_per_body=args.robot_points_per_body,
                trajectory_source=args.trajectory_source,
                output=str(output),
            )
            result = export_clip_from_args(export_args)
        manifest["windows"].append(result)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(
            f"[export_trajectory] window {window.start_idx}:{window.end_idx} -> {output}",
            flush=True,
        )

    print(f"[export_trajectory] wrote {manifest_path}")


if __name__ == "__main__":
    main()
