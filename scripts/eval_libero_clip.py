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
from pointworld.checkpoint_contract import (
    read_checkpoint_contract,
    train_domains_from_data_contract,
)
from training.trainer import Trainer
from tools.libero.sample_schema import (
    T_FRAMES,
    flatten_for_pointworld,
    load_npz,
)


# ----------------------------------------------------------------------------
# CLI plumbing.
# ----------------------------------------------------------------------------

def _checkpoint_domains(model_path: str) -> list[str]:
    """Read the training-time domain list out of a PointWorld checkpoint.

    Falls back to ``["droid", "behavior"]`` if the checkpoint predates the
    contract metadata (legacy v0 / v1 checkpoints used those two domains
    and had the per-domain stat buffers baked in at that shape). The
    hardcoded fallback matches the released ``large-droid+behavior``
    checkpoint and is also the only combination the BEHAVIOR
    ``--model_domain behavior`` selector supports.
    """
    try:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(
            f"WARNING: failed to load checkpoint {model_path} to read "
            f"data_contract; falling back to ['droid', 'behavior']. Error: {e}",
            file=sys.stderr,
        )
        return ["droid", "behavior"]
    try:
        _, data_contract = read_checkpoint_contract(
            ckpt, context="LIBERO clip eval checkpoint"
        )
        return train_domains_from_data_contract(
            data_contract, context="LIBERO clip eval checkpoint"
        )
    except Exception as e:
        print(
            f"WARNING: checkpoint {model_path} has no data_contract; "
            f"falling back to ['droid', 'behavior']. Error: {e}",
            file=sys.stderr,
        )
        return ["droid", "behavior"]


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
        # The per-domain stat buffers in the model have one row per
        # training domain. Read that list out of the checkpoint's
        # ``data_contract`` instead of hardcoding it: any future
        # checkpoint with a different domain set would otherwise load
        # with the wrong stat buffer shape.
        "--domains", ",".join(args._checkpoint_domains),
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
    p.add_argument(
        "--out_pred_npz",
        default=None,
        help=(
            "Optional path to dump the per-point predicted scene flow as a "
            "small .npz (keys: pred_scene_flows (T, N, 3) and "
            "gt_scene_flows (T, N, 3), plus scene_colors (T, N, 3) uint8 "
            "copied from the input clip). Used for offline visualization "
            "(see tools/libero/visualize_clip.py --pred_npz)."
        ),
    )
    p.add_argument("--exp_name", default="libero_clip_eval")
    p.add_argument("--log_dir", default="/tmp/pointworld_log")
    return p.parse_args()


def main() -> None:
    cli_args = _parse_args()
    # Resolve the per-domain stat buffer count from the checkpoint's
    # data contract before handing the CLI namespace to ``_build_args``
    # (which uses it to build the fake argv that the Trainer parser
    # consumes).
    cli_args._checkpoint_domains = _checkpoint_domains(str(cli_args.model_path))
    print(
        f"[eval_libero_clip] using {len(cli_args._checkpoint_domains)} "
        f"domain(s) from checkpoint data_contract: "
        f"{cli_args._checkpoint_domains}",
        file=sys.stderr,
    )
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
    # Switch to eval mode. ``training=False`` only suppresses some
    # auxiliary losses in ``BaseModel.forward``; it does **not** call
    # ``model.eval()``. PTV3 with ``drop_path=0.3`` is still stochastic
    # in train mode, which makes EPE non-reproducible.
    trainer.model.eval()
    with torch.inference_mode():
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

    # ----------------------------------------------------------------
    # Optional: dump the predicted flow + GT flow + colors as a small
    # .npz for offline visualization (see
    # tools/libero/visualize_clip.py --pred_npz).
    # ----------------------------------------------------------------
    if cli_args.out_pred_npz is not None:
        # Both pred and gt come out of the model in world units (meters);
        # we keep them as (T, N, 3) float32 to match the clip's flow
        # convention. The colormap we want for the viser overlay is the
        # per-point *displacement magnitude*, so the caller doesn't need
        # the colors split per-cam; we just take the agentview colors
        # (camera_0) as a representative palette.
        pred_np = pred[0].detach().cpu().numpy().astype(np.float32)  # (T, N, 3)
        gt_np = gt[0].detach().cpu().numpy().astype(np.float32)  # (T, N, 3)
        # Reconstruct the per-point color lookup from the raw clip.
        # We want (T, N, 3) uint8 colors that line up with the model's
        # per-point ordering. After ``sample_cameras`` + the
        # release pipeline, the model gets a per-point ``scene_colors``
        # tensor indexed identically to ``scene_flows``; we recover it
        # by walking back through the raw clip's per-camera arrays in
        # the same order the sampler used.
        try:
            colors_t = _recover_scene_colors(raw)
        except Exception as e:
            print(
                f"WARNING: could not recover scene_colors for "
                f"out_pred_npz ({e}); saving zeros.",
                file=sys.stderr,
            )
            colors_t = np.zeros(
                (pred_np.shape[0], pred_np.shape[1], 3), dtype=np.uint8
            )
        out_path = Path(cli_args.out_pred_npz)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_path,
            pred_scene_flows=pred_np,
            gt_scene_flows=gt_np,
            scene_colors=colors_t,
            # Carry the per-point error so viser can colormap by it
            # without re-doing the math.
            per_point_epe=np.linalg.norm(pred_np - gt_np, axis=-1).astype(
                np.float32
            ),
        )
        print(f"[eval_libero_clip] wrote predicted-flow npz to {out_path}",
              file=sys.stderr)


def _recover_scene_colors(raw: dict) -> np.ndarray:
    """Reconstruct the (T, N, 3) uint8 color tensor that the model sees.

    After :func:`sample_cameras` + the release pipeline, the model's
    per-point color ordering is: camera_0's points (in the order
    they appear in the clip), then camera_1's points. Both cameras
    contribute the same number of valid points per frame, indexed by
    the depth-validity AND visibility mask. We replicate that here so
    the viser overlay can color predicted vs GT points the same way
    the model saw them.
    """
    prefixes = [f"camera_{i}" for i in range(2) if f"camera_{i}_scene_colors" in raw]
    parts = []
    for prefix in prefixes:
        cols = raw[f"{prefix}_scene_colors"]  # (T, Np, 3) uint8
        # Per-point validity mask (depth + visibility).
        if f"{prefix}_scene_depth_valid_mask" in raw:
            mask = raw[f"{prefix}_scene_depth_valid_mask"].astype(bool)
        else:
            mask = np.ones(cols.shape[:2], dtype=bool)
        if f"{prefix}_scene_visibility" in raw:
            mask &= raw[f"{prefix}_scene_visibility"].astype(bool)
        # In the pipeline the invalid points are dropped (kept as -1 in
        # the model's tensor but ``scene_colors`` keeps the original
        # point order). To get a clean visualization, set invalid
        # point colors to black.
        cols = cols.copy()
        cols[~mask] = 0
        parts.append(cols)
    return np.concatenate(parts, axis=1)


if __name__ == "__main__":
    main()
