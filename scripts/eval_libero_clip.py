# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Run a single 11-frame LIBERO clip through a released PointWorld checkpoint
and report full-scene / moved-point EPE.

Usage::

    python scripts/eval_libero_clip.py \
        --model_path /path/to/large-droid+behavior/model-best.pt \
        --libero_clip /tmp/libero_clip.npz \
        --device cuda \
        --out /tmp/libero_eval.json

The script must be run inside the PointWorld conda environment
(pointworld-env). It only reads the .npz that ``tools.libero.export_clip``
produced; it never imports LIBERO itself.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Make sure the project root is importable when this script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# wandb is optional; we disable it eagerly so imports don't try to phone home.
os.environ.setdefault("WANDB_MODE", "disabled")

import numpy as np
import torch

from arguments import parse_args
from dataset_components.cameras import sample_cameras
from dataset_components.collate import custom_collate_fn
from dataset_components.pipeline import apply_release_pipeline_to_sample
from dataset_components.robot import canonicalize_gripper_keys_and_flags
from training.trainer import Trainer
from tools.libero.sample_schema import (
    T_FRAMES,
    flatten_for_pointworld,
    load_npz,
)


# ----------------------------------------------------------------------------
# CLI plumbing.
# ----------------------------------------------------------------------------

def _default_argv(args: argparse.Namespace) -> list[str]:
    """Build a sys.argv shape that parse_args() is happy with.

    The Trainer in inference-only mode skips the dataloader, so most fields
    are ignored — we just need to satisfy the parser and the model-arg
    contract that the checkpoint will override.
    """
    return [
        "eval_libero_clip.py",
        "--model_path", str(args.model_path),
        "--device", args.device,
        "--seed", "42",
        "--robot_features",
        "robot_flows,robot_colors,robot_normals,gripper_open,robot_velocity,robot_acceleration",
        "--scene_features",
        "scene_flows,scene_colors,scene_normals,gripper_open,dist2robot",
        # Even in inference-only mode the Trainer must know which domains
        # to load per-timestep normalization stats for. The released
        # ``large-droid+behavior`` checkpoint was trained on two domains
        # (DROID and BEHAVIOR), so the per-domain stat buffers in the
        # model have shape (2, ...). Pass both so checkpoint loading
        # matches the saved shape; the per-sample domain selector
        # (``sample["__domain__"]``) still picks a single row.
        "--domains", "droid,behavior",
        "--norm_stats_path", args.norm_stats_path,
        "--log_dir", str(args.log_dir),
    ]


def _build_args(cli_args: argparse.Namespace):
    """Build the Trainer-compatible args namespace."""
    sys.argv = _default_argv(cli_args)
    args = parse_args()
    # Force inference-only knobs that parse_args does not set.
    args.exp_name = cli_args.exp_name
    args.batch_size = 1
    args.eval_min_num_cameras = 2
    args.eval_max_num_cameras = 2
    args.train_min_num_cameras = 2
    args.train_max_num_cameras = 2
    return args


# ----------------------------------------------------------------------------
# Metrics.
# ----------------------------------------------------------------------------

def compute_epe_metrics(pred_scene_flows: torch.Tensor,
                        gt_scene_flows: torch.Tensor) -> dict:
    """Full-scene EPE in meters plus a "ever-moved" EPE over points whose
    max over t of ||p^t - p^0||_2 exceeds 5 mm.

    Note: this is *not* the same metric as PointWorld's official soft-movement
    selector (which uses ||p^t - p^{t-1}|| with a 0.5 threshold). The two
    should be clearly separated when comparing numbers.
    """
    # pred_scene_flows, gt_scene_flows: (B, T, N, 3)
    err = torch.linalg.norm(pred_scene_flows[:, 1:] - gt_scene_flows[:, 1:], dim=-1)

    # GT movement: max over t of ||p^t - p^0||_2.
    gt_movement = torch.linalg.norm(gt_scene_flows[:, 1:] - gt_scene_flows[:, :1], dim=-1)
    moved_mask = (gt_movement.max(dim=1).values > 0.005)  # (B, N)

    epe_all_m = err.mean().item()

    expanded_moved_mask = moved_mask[:, None, :].expand_as(err)
    moved_err = err[expanded_moved_mask]
    epe_ever_moved_m = (
        moved_err.mean().item() if moved_err.numel() > 0 else float("nan")
    )

    return {
        "epe_all_m": float(epe_all_m),
        "epe_ever_moved_m": float(epe_ever_moved_m),
        "n_moved_points": int(moved_mask.sum().item()),
        "n_total_points": int(moved_mask.numel()),
    }


# ----------------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", required=True,
                   help="Path to a PointWorld checkpoint (.pt).")
    p.add_argument("--libero_clip", required=True,
                   help="Path to a clip .npz produced by tools.libero.export_clip.")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--model_domain", default="behavior",
                   help=(
                       "Domain name to use for normalization stats lookup. "
                       "For the smoke test we alias LIBERO onto the released "
                       "'behavior' checkpoint (see docs/指导.md §十二)."
                   ))
    p.add_argument(
        "--norm_stats_path",
        default="stats/droid_behavior",
        help=(
            "Folder containing precomputed per-domain normalization stats. "
            "Defaults to ``stats/droid_behavior`` which matches the released "
            "``large-droid+behavior`` checkpoint."
        ),
    )
    p.add_argument("--has_bimanual_robot", action="store_true", default=True,
                   help="Use bimanual feature layout (left slot = identity).")
    p.add_argument("--no_bimanual", dest="has_bimanual_robot", action="store_false",
                   help="Use single-arm feature layout.")
    p.add_argument("--out", default=None, help="Optional path to dump the metric JSON.")
    p.add_argument("--exp_name", default="libero_clip_eval")
    p.add_argument("--log_dir", default="/tmp/pointworld_log")
    return p.parse_args()


def main() -> None:
    cli_args = _parse_args()
    args = _build_args(cli_args)

    # ----------------------------------------------------------------
    # Load the .npz and project it into the PointWorld sample shape.
    # ----------------------------------------------------------------
    raw = load_npz(cli_args.libero_clip)
    sample = flatten_for_pointworld(raw)
    sample["__domain__"] = cli_args.model_domain

    # Deterministic camera selection: always pick both cameras in order.
    sample = sample_cameras(
        sample,
        min_num_cameras=2,
        max_num_cameras=2,
        deterministic=True,
        seed=42,
    )

    # Fill in left/right gripper slots (left = identity by canonicalize).
    sample = canonicalize_gripper_keys_and_flags(sample)

    sample = apply_release_pipeline_to_sample(
        sample=sample,
        domain=cli_args.model_domain,
        mode="test",
        args=args,
        has_bimanual_robot=cli_args.has_bimanual_robot,
    )

    # ----------------------------------------------------------------
    # Build the Trainer in inference-only mode and run forward.
    # ----------------------------------------------------------------
    trainer = Trainer(args, inference_only=True, data_info_dict=None)
    batch = custom_collate_fn([sample], args=args)
    batch = {
        k: (v.to(trainer.device) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }
    with torch.no_grad():
        outputs = trainer.model(batch, training=False)

    pred = outputs["scene_flows"]  # (B, T, N, 3)
    gt = batch["gt_scene_flows"]  # (B, T, N, 3)
    metrics = compute_epe_metrics(pred, gt)

    # Add a few diagnostics.
    metrics["model_domain"] = cli_args.model_domain
    metrics["has_bimanual_robot"] = bool(cli_args.has_bimanual_robot)
    metrics["checkpoint"] = str(cli_args.model_path)
    metrics["clip"] = str(cli_args.libero_clip)
    metrics["T"] = int(gt.shape[1])
    metrics["n_scene_points"] = int(gt.shape[2])
    # NOTE: PointWorld overwrites log variance to a SIM_VAR_CONST for any
    # domain whose name contains "behavior" (see BaseModel.forward), so
    # outputs["confidence"] here is *not* a learned uncertainty. We do not
    # report it; callers that want real confidence should use a DROID
    # domain and the corresponding normalization.

    print(json.dumps(metrics, indent=2))
    if cli_args.out is not None:
        Path(cli_args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(cli_args.out, "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
