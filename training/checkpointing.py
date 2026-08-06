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

import os
from pathlib import Path

import torch
import torch.distributed as dist
from utils import _print
from pointworld.checkpoint_contract import attach_checkpoint_contract


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


# Names of buffers/parameters that hold the *precomputed* dataset
# normalization statistics.  These are populated at training time
# from the JSON file under ``--norm_stats_path``; they are *not* a
# learnable part of the network.  When we fine-tune a pre-trained
# model on a new dataset, we want the network weights to come from
# the source checkpoint but the normalization statistics to come
# from the new dataset's stats JSON.  The downstream training code
# re-creates the normalization buffers after the model is built,
# so we strip these keys from the loaded state dict.
NORM_BUFFER_NAMES = (
    "norm_stats_per_step_mean",
    "norm_stats_per_step_var",
    "robot_norm_mean",
    "robot_norm_var",
    "scene_norm_mean",
    "scene_norm_var",
)


def _is_norm_buffer(key: str) -> bool:
    return any(key.endswith(name) for name in NORM_BUFFER_NAMES)


def load_finetune_weights(model, checkpoint_path: str) -> None:
    """Load a pre-trained ``model_state_dict`` into ``model`` for fine-tuning.

    This is the fine-tune counterpart to
    :func:`load_checkpoint`: it loads only the learnable model
    weights, skips the dataset-derived normalization buffers, and
    **does not** touch the optimizer or training counters.  Missing
    keys are allowed only for the normalization buffers; the rest
    of the model state dict must match the checkpoint exactly.

    Use this when transferring a network trained on DROID/BEHAVIOR
    to a new domain (e.g. LIBERO).  The downstream training code
    then re-creates the ``NORM_BUFFER_NAMES`` from
    ``args.norm_stats_path`` so the new run uses its own dataset
    statistics.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(
        str(checkpoint_path),
        map_location="cpu",
        weights_only=False,
    )
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        raise KeyError(
            f"checkpoint {checkpoint_path} has neither 'model' nor "
            f"'model_state_dict' keys; cannot fine-tune from it."
        )

    cleaned = {}
    for k, v in state_dict.items():
        # Strip DDP 'module.' prefix if present.
        if k.startswith("module."):
            k = k[len("module."):]
        if _is_norm_buffer(k):
            continue
        cleaned[k] = v

    model_to_load = _unwrap_model(model)
    result = model_to_load.load_state_dict(cleaned, strict=False)
    unexpected = set(result.unexpected_keys)
    missing = set(result.missing_keys)
    # Norm buffers are allowed to be missing: they will be set from
    # the LIBERO (or other new-domain) stats JSON.
    expected_missing = {
        k for k in model_to_load.state_dict() if _is_norm_buffer(k)
    }
    invalid_missing = missing - expected_missing
    if unexpected:
        raise RuntimeError(
            f"Unexpected keys in fine-tune checkpoint: {sorted(unexpected)[:20]}"
        )
    if invalid_missing:
        raise RuntimeError(
            f"Unexpected missing model keys after fine-tune load: "
            f"{sorted(invalid_missing)[:20]}"
        )
    _print(
        f"Loaded fine-tune weights from {checkpoint_path} "
        f"(stripped NORM_BUFFER_NAMES; "
        f"{len(cleaned)} params loaded, "
        f"{len(expected_missing)} norm buffers reinitialised from "
        f"--norm_stats_path)"
    )


def save_checkpoint_now(trainer, adjusted_batch_count, log_dict=None):
    if trainer.args.distributed:
        dist.barrier()
    if trainer.rank == 0:
        _print("Saving checkpoint")
        save_checkpoint(trainer, log_dict or {})
    if trainer.args.distributed:
        dist.barrier()


def save_checkpoint(trainer, log_dict=None):
    if trainer.rank != 0:
        return

    save_dict = {
        "model": _unwrap_model(trainer.model).state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "args": trainer.args,
        "exp_name": trainer.exp_name,
        "wandb_id": trainer.wandb_id,
        "batch_count": trainer.batch_count,
        "epoch_count": trainer.epoch_count,
        "sample_count": trainer.sample_count,
    }
    attach_checkpoint_contract(save_dict, args=trainer.args, context="training save checkpoint")

    final_local_path = os.path.join(trainer.save_dir, "model-last.pt")
    torch.save(save_dict, final_local_path)

    return trainer.save_dir


def load_checkpoint_from_path(trainer, model_path=None):
    if model_path is None:
        if trainer.rank == 0:
            _print("No checkpoint path provided, starting from scratch")
        return None
    assert os.path.exists(model_path), f"Checkpoint not found: {model_path}"
    if trainer.rank == 0:
        _print(f"Loading checkpoint from {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if "wandb_id" in checkpoint:
        checkpoint.pop("wandb_id")
    if "exp_name" in checkpoint:
        checkpoint.pop("exp_name")
    return checkpoint


def load_checkpoint(trainer, checkpoint):
    model_state_dict = checkpoint["model"]
    expected_keys = set(_unwrap_model(trainer.model).state_dict().keys())
    ckpt_keys = set(model_state_dict.keys())
    if ckpt_keys != expected_keys:
        missing = sorted(expected_keys - ckpt_keys)
        extra = sorted(ckpt_keys - expected_keys)
        missing_suffix = "..." if len(missing) > 10 else ""
        extra_suffix = "..." if len(extra) > 10 else ""
        raise RuntimeError(
            "Checkpoint model keys do not match the current model state_dict. "
            "Ensure the distributed/Non-DDP setting and model config match. "
            f"Missing keys: {missing[:10]}{missing_suffix}. "
            f"Extra keys: {extra[:10]}{extra_suffix}."
        )
    _unwrap_model(trainer.model).load_state_dict(model_state_dict)

    if not trainer.inference_only:
        if "optimizer" not in checkpoint:
            if getattr(trainer.args, "allow_optimizer_reset", False):
                _print("Checkpoint missing optimizer state; resetting optimizer as requested.")
            else:
                raise RuntimeError(
                    "Checkpoint missing optimizer state. "
                    "Rerun with --allow_optimizer_reset=true to skip optimizer restore."
                )
        else:
            try:
                trainer.optimizer.load_state_dict(checkpoint["optimizer"])
            except ValueError as exc:
                if getattr(trainer.args, "allow_optimizer_reset", False):
                    _print(
                        "Optimizer state incompatible with current optimizer; "
                        "resetting optimizer state as requested."
                    )
                else:
                    raise RuntimeError(
                        "Optimizer state incompatible with current optimizer. "
                        "Rerun with --allow_optimizer_reset=true to skip optimizer restore."
                    ) from exc
    assert "batch_count" in checkpoint, "Checkpoint missing batch_count."
    assert "epoch_count" in checkpoint, "Checkpoint missing epoch_count."
    assert "sample_count" in checkpoint, "Checkpoint missing sample_count."
    trainer.batch_count = checkpoint["batch_count"]
    trainer.epoch_count = checkpoint["epoch_count"]
    trainer.sample_count = checkpoint["sample_count"]
