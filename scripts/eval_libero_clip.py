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
# Robot-trajectory extrapolation for autoregressive rollout.
# ----------------------------------------------------------------------------

def _extrapolate_robot_traj(
    robot_flows_orig: torch.Tensor,
    robot_exists_orig: torch.Tensor,
    mode: str = "linear",
) -> torch.Tensor:
    """Build an extrapolated/repeated robot trajectory for the autoregressive
    rollout step. The model's condition is the *future* robot pose trajectory;
    in the autoregressive loop we don't have the GT future, so we have to
    synthesize one from the last few frames of the input clip.

    Parameters
    ----------
    robot_flows_orig : (B, T, Nr, 3) tensor of robot point positions in the
        centered frame (what the model sees).
    robot_exists_orig : (B, T, Nr) bool tensor indicating which robot points
        are valid at each frame.
    mode : "linear" | "hold" | "repeat"
        - "linear": extrapolate the last 3 frames of robot pose with constant
          per-point velocity (position at t+k = position at t-1 + k * vel).
          This is the default; it gives the model a plausible "the robot is
          moving in a straight line" continuation.
        - "hold": keep the last frame's pose constant for the entire future
          trajectory. Good fallback if the last few input frames already
          capture the desired motion and we want the model to "rest".
        - "repeat": cycle the original T-frame trajectory. Useful when the
          input clip already contains a full demonstration cycle and we want
          the model to keep producing the same motion.

    Returns
    -------
    (B, T, Nr, 3) tensor of the new "future" robot trajectory, in the same
    shape and frame as ``robot_flows_orig``. ``robot_exists_orig`` is
    unchanged.
    """
    B, T, Nr, _ = robot_flows_orig.shape
    if mode == "hold":
        # All future frames == the last frame.
        return robot_flows_orig[:, -1:].expand(-1, T, -1, -1).contiguous()
    if mode == "repeat":
        return robot_flows_orig.clone()
    # "linear": extrapolate the last 3 frames with constant velocity per point.
    if T < 2:
        return robot_flows_orig.clone()
    # Use the last up-to-3 frames to estimate per-point velocity.
    # ``vel[t, n] = (p[t] - p[t-1])`` averaged over the trailing window.
    n_vel = min(3, T - 1)
    diffs = robot_flows_orig[:, -n_vel:] - robot_flows_orig[:, -n_vel - 1:-1]  # (B, n_vel, Nr, 3)
    vel = diffs.mean(dim=1)  # (B, Nr, 3) per-point constant velocity
    # Build the future trajectory: at step k in [0, T), the per-point position
    # is ``p[last] + (k+1) * vel`` (we use k+1 because at k=0 we are one
    # step past ``last``).
    last = robot_flows_orig[:, -1]  # (B, Nr, 3)
    ks = torch.arange(1, T + 1, device=robot_flows_orig.device, dtype=robot_flows_orig.dtype)  # (T,)
    new_traj = last.unsqueeze(1) + ks.view(1, T, 1, 1) * vel.unsqueeze(1)
    return new_traj


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
    p.add_argument(
        "--rollout",
        type=int,
        default=1,
        help=(
            "Number of autoregressive model calls. ``1`` (default) is the "
            "single forward pass that predicts T future frames from the "
            "11-frame clip. ``N > 1`` keeps feeding the predicted scene "
            "back into the model to cover a longer horizon (~(N * (T-1) + "
            "1) frames total). The robot's future trajectory is "
            "linearly extrapolated from the last 3 frames of the clip "
            "so the model has a plausible 'continuing action' to "
            "condition on."
        ),
    )
    p.add_argument(
        "--robot_extrap_mode",
        choices=["linear", "hold", "repeat"],
        default="linear",
        help=(
            "How to construct the future robot trajectory passed to the "
            "model in the autoregressive steps. ``linear`` (default) "
            "extrapolates the last 3 frames of robot pose with constant "
            "velocity; ``hold`` keeps the last frame's pose constant; "
            "``repeat`` cycles the original 11-frame trajectory."
        ),
    )
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

    # ----------------------------------------------------------------
    # Autoregressive rollout (--rollout N).
    #
    # The single forward pass above predicts T=11 future frames. The
    # LIBERO demo trajectory is typically much longer (50-200 frames),
    # so for visualization we want the model to keep predicting until
    # the object is fully grasped / placed. We do this by feeding the
    # model's last predicted scene back into the model as the new
    # initial state, with an extrapolated / repeated robot trajectory
    # to give the model a plausible "continuing action" to condition on.
    #
    # The output ``pred`` becomes (B, N * (T - 1) + 1, N_pts, 3); the
    # intermediate "frame" at the end of each chunk is dropped to avoid
    # duplicating it. We also extend ``gt`` to match (filled with the
    # last available GT frame -- there is no ground truth for the
    # extrapolated frames, but the user only looks at pred).
    # ----------------------------------------------------------------
    if cli_args.rollout > 1:
        from dataset_components.collate import custom_collate_fn as _ccf
        # Cache the inputs we need to rebuild the batch on each step.
        # We re-use frame 0 of the original scene/robot features (the
        # features at the actual t=0), which is a reasonable
        # approximation; the model's per-point features are designed
        # to be roughly stationary over the horizon we predict (the
        # scene+robot geometry is the dominant signal).
        scene_flows0 = batch["scene_flows"][:, 0].clone()  # (B, N, 3)
        scene_features0 = batch["scene_features"][:, 0].clone()  # (B, N, F)
        scene_exists0 = batch["scene_exists"][:, 0].clone()  # (B, N)
        if "scene_visibility" in batch:
            scene_visibility0 = batch["scene_visibility"][:, 0].clone()
        else:
            scene_visibility0 = None
        # Robot: keep the full T-frame trajectory, but we'll replace
        # it chunk-by-chunk in the loop.
        robot_flows_orig = batch["robot_flows"].clone()  # (B, T, Nr, 3)
        robot_features_orig = batch["robot_features"].clone()  # (B, T, Nr, Fr)
        robot_exists_orig = batch["robot_exists"].clone()  # (B, T, Nr)
        T_clip = int(gt.shape[1])

        # Concatenate predictions across chunks. We keep the per-chunk
        # pred so the file consumers can see the model "thinking"
        # each time it sees its own output.
        pred_chunks = [pred]  # (B, T, N, 3) each
        gt_chunks = [gt]  # (B, T, N, 3) each
        current_pred_last = pred[:, -1].clone()  # (B, N, 3) for next seed

        for step in range(1, cli_args.rollout):
            # ------------------------------------------------------------------
            # Build the new batch: scene[:, 0] = last predicted,
            # scene[:, 1:] = the previously-predicted frames (so the
            # scene features have some temporal context). The robot
            # trajectory is extrapolated from the last few frames.
            # ------------------------------------------------------------------
            new_batch = dict(batch)
            # New scene_flows seed: pred[:, -1] (last predicted of previous chunk)
            # Then for the remaining T-1 frames, use the trailing T-1
            # predictions from the previous chunk so the model has
            # context. (We hold the seed constant at frame 0 and
            # propagate the trailing frames into frames 1..T-1.)
            new_scene_flows = torch.cat(
                [current_pred_last.unsqueeze(1), pred[:, 1:]], dim=1
            )  # (B, T, N, 3)
            new_batch["scene_flows"] = new_scene_flows
            # Use the same scene features (frame 0) for all frames --
            # an approximation, but the model's scene feature encoder
            # is robust to small per-frame feature drift because the
            # geometry (the scene_flows positions) is the dominant
            # signal. Saves us from having to recompute normals /
            # dist2robot on the predicted positions.
            new_batch["scene_features"] = scene_features0.unsqueeze(1).expand(
                -1, T_clip, -1, -1
            ).contiguous()
            new_batch["scene_exists"] = scene_exists0.unsqueeze(1).expand(
                -1, T_clip, -1
            ).contiguous()
            if scene_visibility0 is not None:
                new_batch["scene_visibility"] = scene_visibility0.unsqueeze(1).expand(
                    -1, T_clip, -1
                ).contiguous()
            # Extrapolate the robot trajectory.
            new_batch["robot_flows"] = _extrapolate_robot_traj(
                robot_flows_orig, robot_exists_orig, mode=cli_args.robot_extrap_mode,
            )
            new_batch["robot_features"] = robot_features_orig  # reuse
            new_batch["robot_exists"] = robot_exists_orig

            with torch.inference_mode():
                outputs_step = trainer.model(new_batch, training=False)
            pred_step = outputs_step["scene_flows"]  # (B, T, N, 3)
            # The first frame of pred_step is the identity (model
            # convention); drop it so we can concatenate without
            # duplicating. ``pred[:, 0] == scene_coord0``, which after
            # the first rollout equals current_pred_last.
            pred_chunks.append(pred_step[:, 1:])  # (B, T-1, N, 3)
            # Update the seed: the last predicted position of this chunk
            # is the *second-to-last* of pred_step (since pred_step[:, -1]
            # is the model's prediction for its own frame 0, which is
            # the same as current_pred_last).
            current_pred_last = pred_step[:, -1].clone()
            # No real GT for the extrapolated frames; pad with the last
            # available GT frame so any consumer that expects
            # ``len(gt) == len(pred)`` doesn't crash.
            gt_chunks.append(
                gt[:, -1:].expand(-1, T_clip - 1, -1, -1).contiguous()
            )
            print(
                f"[eval_libero_clip] rollout step {step+1}/{cli_args.rollout} "
                f"complete (pred shape now "
                f"({sum(c.shape[1] for c in pred_chunks)}, {pred_chunks[0].shape[2]}, 3))",
                file=sys.stderr,
            )

        # Concatenate predictions. Keep the first frame of the very
        # first chunk (it's the model's t=0 identity, which equals the
        # actual scene at t=0).
        pred = torch.cat(pred_chunks, dim=1)  # (B, T + N*(T-1), N, 3)
        gt = torch.cat(gt_chunks, dim=1)  # (B, T + N*(T-1), N, 3)
        # EPE is only meaningful on the first chunk (where we have GT);
        # we re-compute and report it on that slice for diagnostics.
        first_chunk_pred = pred_chunks[0]
        first_chunk_gt = gt_chunks[0]
        chunk_metrics = compute_epe_metrics(first_chunk_pred, first_chunk_gt)
        metrics["first_chunk_epe_all_m"] = chunk_metrics["epe_all_m"]
        metrics["first_chunk_epe_ever_moved_m"] = chunk_metrics["epe_ever_moved_m"]
        metrics["rollout_steps"] = int(cli_args.rollout)
        metrics["T_total"] = int(pred.shape[1])
        print(
            f"[eval_libero_clip] autoregressive rollout complete: "
            f"T_total = {pred.shape[1]} (T_clip={T_clip}, steps={cli_args.rollout})",
            file=sys.stderr,
        )

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
        # CRITICAL: ``apply_release_pipeline_to_sample`` runs
        # ``center_shift`` in test mode, which subtracts the t=0 mean
        # of (downsampled scene + robot) points from the input. The
        # model produces outputs in that centered frame, so ``pred`` and
        # ``gt`` are also in the centered frame. We save them **as-is**
        # (still centered) along with the cumulative shift amount
        # (``__shift_amount__``) so that downstream consumers (e.g.
        # ``tools/libero/visualize_clip.py``) can convert to world frame
        # if needed. We deliberately do **not** pre-correct here --
        # pre-correcting would cause the visualization to double-correct
        # when it also applies the shift, which manifests as a
        # uniform Z offset ("z-axis off by a section").
        shift_amount = sample.get("__shift_amount__", None)
        if shift_amount is None:
            print(
                "WARNING: sample has no __shift_amount__; saving pred in "
                "centered frame. Visualization will fall back to a "
                "recomputed shift from the raw clip, which may be off by "
                "a few cm.",
                file=sys.stderr,
            )
        else:
            # ``__shift_amount__`` semantics (see center_shift in
            # dataset_components/transforms.py): the function does
            # ``scene -= shift`` and ``__shift_amount__ -= shift``, so
            # the cumulative ``__shift_amount__`` at the end equals
            # ``-sum(shift_i)``. The model's first-frame scene_coord0
            # therefore sits at ``world = centered - __shift_amount__``,
            # or equivalently ``world = centered + sum(shift_i)``.
            print(
                f"[eval_libero_clip] saving pred in centered frame; "
                f"__shift_amount__ = {np.asarray(shift_amount)} "
                f"(use ``pred_world = pred - shift_amount`` in the "
                f"visualizer to convert to world frame).",
                file=sys.stderr,
            )
        # Save the raw model output (centered frame) so the npz is a
        # faithful snapshot of what the model produced. The viz is
        # responsible for converting to world frame using
        # ``shift_amount``.
        pred_np = pred[0].detach().cpu().numpy().astype(np.float32)  # (T, N, 3) centered
        gt_np = gt[0].detach().cpu().numpy().astype(np.float32)  # (T, N, 3) centered
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
        save_kwargs = dict(
            pred_scene_flows=pred_np,
            gt_scene_flows=gt_np,
            scene_colors=colors_t,
            # Carry the per-point error so viser can colormap by it
            # without re-doing the math.
            per_point_epe=np.linalg.norm(pred_np - gt_np, axis=-1).astype(
                np.float32
            ),
        )
        if shift_amount is not None:
            # Carry the shift so viser (or other downstream tools) can
            # audit/regenerate the world transform if needed. Note that
            # ``__shift_amount__`` is the cumulative inverse-shift
            # vector stored on the sample; pred_world = pred - shift.
            save_kwargs["shift_amount"] = np.asarray(
                shift_amount, dtype=np.float32
            )
        np.savez(out_path, **save_kwargs)
        print(f"[eval_libero_clip] wrote predicted-flow npz to {out_path}",
              file=sys.stderr)


def _recover_scene_colors(raw: dict) -> np.ndarray:
    """Reconstruct the (T, N, 3) uint8 color tensor that the model sees.

    After :func:`sample_cameras` + the release pipeline, the model's
    per-point color ordering is: camera_0's points (in the order
    they appear in the clip), then camera_1's points, then camera_2's,
    etc. (the pipeline concatenates every camera's points and then
    subsamples; the point order is preserved per camera). We replicate
    that here so the viser overlay can color predicted vs GT points
    the same way the model saw them.

    We scan for camera_0..camera_K rather than hardcoding a 2-camera
    layout, so 5-camera exports (e.g. agentview + birdview + sideview
    + frontview + robot0_eye_in_hand) also work.
    """
    # Discover the available camera prefix ids. We use the
    # ``camera_i_scene_flows`` key as the authoritative "this camera
    # exists" probe; ``scene_flows`` is always emitted by the
    # exporter when the camera was actually rendered.
    cam_ids = []
    for i in range(8):  # up to 8 cameras is more than enough
        if f"camera_{i}_scene_flows" in raw:
            cam_ids.append(i)
    if not cam_ids:
        raise KeyError(
            "no camera_N_scene_flows keys in raw clip; cannot recover "
            "scene colors"
        )
    parts = []
    for cid in cam_ids:
        prefix = f"camera_{cid}"
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
