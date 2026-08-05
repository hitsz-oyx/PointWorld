# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launch PointWorld's native prediction visualizer on a LIBERO eval sample.

This bridges the LIBERO clip export / eval path onto the same
``PredictionVisualizer`` used by ``evaluation.Tester``. It rebuilds the
post-pipeline sample dict (camera payloads, masks, centered scene points,
robot state) from a clip ``.npz`` and combines it with a predicted-flow
``.npz`` produced by ``scripts/eval_libero_clip.py --out_pred_npz``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_components.cameras import select_cameras_in_order
from dataset_components.collate import custom_collate_fn
from dataset_components.pipeline import apply_release_pipeline_to_sample
from dataset_components.robot import canonicalize_gripper_keys_and_flags
from scripts.eval_libero_clip import _build_args
from tools.libero.sample_schema import flatten_for_pointworld, load_npz
from utils import resolve_default_robot_urdf
from visualization.prediction_viz import (
    PredictionVisualizer,
    PredictionVisualizerConfig,
    build_sample_from_dictionary,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clip", required=True, help="Path to LIBERO clip .npz")
    p.add_argument(
        "--pred_npz",
        required=True,
        help="Path to pred .npz from scripts/eval_libero_clip.py",
    )
    p.add_argument(
        "--model_path",
        default="",
        help="Optional PointWorld checkpoint path. Not needed for visualization.",
    )
    p.add_argument(
        "--domains",
        default="droid,behavior",
        help="Comma-separated training domains for parse_args compatibility.",
    )
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument(
        "--num_cameras",
        type=int,
        choices=[1, 2, 3],
        default=3,
        help="Number of exported cameras to visualize in on-disk order.",
    )
    p.add_argument(
        "--model_domain",
        default="behavior",
        help="Domain name for the release pipeline (default: behavior).",
    )
    p.add_argument(
        "--norm_stats_path",
        default="stats/droid_behavior",
        help="Norm stats folder matching the checkpoint.",
    )
    p.add_argument("--viewer_port", type=int, default=8091)
    p.add_argument("--viewer_host", default="0.0.0.0")
    p.add_argument(
        "--scene_point_size",
        type=float,
        default=0.003,
        help="Rendered behavior point size in meters (default: 0.003).",
    )
    p.add_argument(
        "--no_upsample",
        action="store_true",
        help="Start on the model's 12k coarse cloud instead of RGB-D upsampling.",
    )
    p.add_argument("--exp_name", default="libero_eval_visualizer")
    p.add_argument("--log_dir", default="/tmp/pointworld_log")
    p.add_argument("--has_bimanual_robot", action="store_true", default=True)
    p.add_argument("--no_bimanual", dest="has_bimanual_robot", action="store_false")
    return p.parse_args()


def _prepare_sample(cli_args: argparse.Namespace) -> dict:
    cli_args._checkpoint_domains = [
        d.strip() for d in str(cli_args.domains).split(",") if d.strip()
    ]
    if not cli_args._checkpoint_domains:
        raise ValueError("At least one domain is required via --domains")
    args = _build_args(cli_args)

    raw = load_npz(str(cli_args.clip))
    sample = flatten_for_pointworld(raw)
    sample["__domain__"] = cli_args.model_domain
    sample = select_cameras_in_order(sample, num_cameras=cli_args.num_cameras)
    sample = canonicalize_gripper_keys_and_flags(sample)
    sample = apply_release_pipeline_to_sample(
        sample=sample,
        domain=cli_args.model_domain,
        mode="test",
        args=args,
        has_bimanual_robot=cli_args.has_bimanual_robot,
        include_scene_data=True,
    )
    batch = custom_collate_fn([sample], args=args)

    batch_np: dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                batch_np[key] = value.item()
            else:
                batch_np[key] = value[0] if value.shape[0] == 1 else value
        elif hasattr(value, "detach"):
            arr = value.detach().cpu().numpy()
            if arr.ndim == 0:
                batch_np[key] = arr.item()
            else:
                batch_np[key] = arr[0] if arr.shape[0] == 1 else arr
        elif isinstance(value, (list, tuple)):
            batch_np[key] = value[0] if len(value) == 1 else value
        else:
            batch_np[key] = value
    return batch_np


def main() -> None:
    cli_args = _parse_args()

    print("[prediction_viz] preparing sample...", flush=True)
    sample_dict = _prepare_sample(cli_args)
    print("[prediction_viz] loading pred npz...", flush=True)
    pred_npz = np.load(str(cli_args.pred_npz), allow_pickle=True)

    print("[prediction_viz] building visualizer sample...", flush=True)
    viz_sample = build_sample_from_dictionary(
        sample_dict=sample_dict,
        predictions={"scene_flows": np.asarray(pred_npz["pred_scene_flows"])},
    )

    viz_config = PredictionVisualizerConfig()
    viz_config.viewer_host = str(cli_args.viewer_host)
    viz_config.viewer_port = int(cli_args.viewer_port)
    if cli_args.scene_point_size <= 0.0:
        raise ValueError("--scene_point_size must be positive")
    viz_config.behavior_scene_point_size = float(cli_args.scene_point_size)
    viz_config.initial_upsample = not bool(cli_args.no_upsample)

    visualizer = PredictionVisualizer(
        viz_config,
        urdf_path=Path(resolve_default_robot_urdf([cli_args.model_domain])),
    )
    print("[prediction_viz] launching native PointWorld viewer...", flush=True)
    result = visualizer.visualize(viz_sample, launch_viewer=True)
    live_session = result.get("live_session")
    if live_session is None:
        raise RuntimeError("PredictionVisualizer failed to launch a live viewer.")

    host, port = visualizer.viewer_endpoint()
    display_host = "localhost" if host in {"0.0.0.0", "127.0.0.1"} else host
    print(f"[prediction_viz] Live viewer running at http://{display_host}:{port}")
    print(f"[prediction_viz] clip={cli_args.clip}")
    print(f"[prediction_viz] pred={cli_args.pred_npz}")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        live_session.close()


if __name__ == "__main__":
    main()
