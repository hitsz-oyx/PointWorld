"""Compute LIBERO normalization statistics from a directory of exported ``.npz`` clips.

We reuse the *exact* single-sample processing chain from
:class:`dataset_components.libero_dataset.LiberoNPZDataset` so the
statistics describe the same tensors the trainer will eventually see.

Output JSON layout (matches :func:`pointworld.norm_stats.load_stats_from_json_folder`):

    {
        "statistics": {
            "libero": {
                "robot_features":  {"mean": [...], "variance": [...]},
                "scene_features":  {"mean": [...], "variance": [...]}
            }
        },
        "per_timestep_statistics": {
            "libero": {
                "gt_scene_flows_relative": {
                    "timestep_0": {"mean": [...], "variance": [...]},
                    ...
                    "timestep_10": {...}
                }
            }
        }
    }

The computed ``gt_scene_flows_relative`` is the per-timestep difference
``gt_scene_flows - gt_scene_flows[0:1]`` clipped to
``RELEASE_MAX_RELATIVE_MOVEMENT``, the same tensor the loss normalizes.

Usage::

    python -m tools.libero.compute_norm_stats \\
        --data_dir /path/to/libero_pointworld/train \\
        --output stats/libero/norm_stats.json

For a shared clip pool, add ``--split_manifest splits/paper.json`` and select
the manifest split with ``--split`` (default: ``train``).
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np

# Make project root importable when run as a module.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ``dataset_components.utils`` is numba-accelerated; we want the real
# implementation (numba is a hard dep of PointWorld).  No stub needed.


# Imports that touch libero / robosuite need the headless EGL shim.
from tools.libero import _egl_compat  # noqa: E402
_egl_compat.apply()


from dataset_components.cameras import select_cameras_in_order  # noqa: E402
from dataset_components.constants import RELEASE_MAX_RELATIVE_MOVEMENT  # noqa: E402
from dataset_components.pipeline import apply_release_pipeline_to_sample  # noqa: E402
from dataset_components.robot import (  # noqa: E402
    canonicalize_gripper_keys_and_flags,
    gather_features,
)
from dataset_components.transforms import (  # noqa: E402
    compute_helper_variables,
    make_gt_copy,
)
from tools.libero.sample_schema import (  # noqa: E402
    T_FRAMES,
    flatten_for_pointworld,
    load_npz,
)


# ---------------------------------------------------------------------------
# Tiny args bundle to satisfy ``apply_release_pipeline_to_sample``.
# ---------------------------------------------------------------------------
@dataclass
class _StatsArgs:
    deterministic_train: bool = True
    grid_size: float = 0.015
    max_scene_points: int = 12000
    seed: int = 0
    robot_features: List[str] = field(default_factory=lambda: [
        "robot_flows", "robot_colors", "robot_normals",
        "gripper_open", "robot_velocity", "robot_acceleration",
    ])
    scene_features: List[str] = field(default_factory=lambda: [
        "scene_flows", "scene_colors", "scene_normals",
        "gripper_open", "dist2robot",
    ])


def _process_one(npz_path: Path, args: _StatsArgs, num_cameras: int) -> dict:
    """Run the same single-sample chain the trainer uses and return the
    post-pipeline sample (which is already gathered, gt-copied,
    helper-computed, and tensor-converted by
    :func:`apply_release_pipeline_to_sample`)."""
    raw = load_npz(str(npz_path))
    sample = flatten_for_pointworld(raw)
    sample["__domain__"] = "libero"
    sample["__key__"] = npz_path.stem

    sample = select_cameras_in_order(sample, num_cameras=num_cameras)
    sample = canonicalize_gripper_keys_and_flags(sample)
    sample = apply_release_pipeline_to_sample(
        sample=sample,
        domain="libero",
        mode="test",
        args=args,
        has_bimanual_robot=False,
    )
    return sample


def _to_numpy(x):
    """Detach torch tensors to numpy float arrays (the trainer
    pipeline emits torch tensors via ``convert_to_tensors``)."""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def _sample_moments(
    path: Path,
    pipeline_args: _StatsArgs,
    num_cameras: int,
):
    """Return mergeable sufficient statistics for one exported clip."""
    try:
        sample = _process_one(path, pipeline_args, num_cameras=num_cameras)
        rf = _to_numpy(sample["robot_features"]).reshape(
            -1, sample["robot_features"].shape[-1],
        ).astype(np.float64)
        sf = _to_numpy(sample["scene_features"]).reshape(
            -1, sample["scene_features"].shape[-1],
        ).astype(np.float64)
        gt = _to_numpy(sample["gt_scene_flows_relative"]).astype(np.float64)
        if "scene_supervised_mask" in sample:
            mask = _to_numpy(sample["scene_supervised_mask"]).astype(bool)
        else:
            mask = np.ones(gt.shape[:-1], dtype=bool)
        gt = gt.reshape(gt.shape[-3], gt.shape[-2], gt.shape[-1]) if gt.ndim == 4 else gt
        mask = mask.reshape(mask.shape[-3], mask.shape[-2]) if mask.ndim == 4 else mask

        per_t_sum = np.zeros((T_FRAMES, 3), dtype=np.float64)
        per_t_sq = np.zeros((T_FRAMES, 3), dtype=np.float64)
        per_t_n = np.zeros(T_FRAMES, dtype=np.int64)
        for timestep in range(T_FRAMES):
            valid = mask[timestep] & np.isfinite(gt[timestep]).all(axis=-1)
            points = gt[timestep][valid]
            if points.size:
                per_t_sum[timestep] = points.sum(axis=0)
                per_t_sq[timestep] = (points ** 2).sum(axis=0)
                per_t_n[timestep] = points.shape[0]
        return {
            "path": str(path),
            "error": None,
            "robot_sum": rf.sum(axis=0),
            "robot_sq": (rf ** 2).sum(axis=0),
            "robot_n": rf.shape[0],
            "scene_sum": sf.sum(axis=0),
            "scene_sq": (sf ** 2).sum(axis=0),
            "scene_n": sf.shape[0],
            "per_t_sum": per_t_sum,
            "per_t_sq": per_t_sq,
            "per_t_n": per_t_n,
        }
    except Exception as error:
        return {
            "path": str(path),
            "error": f"{type(error).__name__}: {error}",
        }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", required=True,
                   help="LIBERO NPZ directory or shared pool root.")
    p.add_argument(
        "--split_manifest", default=None,
        help="Optional clip-level split manifest; paths are relative to --data_dir.",
    )
    p.add_argument(
        "--split", default="train",
        help="Manifest split to process when --split_manifest is set (default: train).",
    )
    p.add_argument("--output", required=True,
                   help="Output JSON path.")
    p.add_argument("--domain", default="libero",
                   help="Domain name to record in the JSON (default: libero).")
    p.add_argument("--num_cameras", type=int, default=3,
                   help="Number of cameras to use (default: 3).")
    p.add_argument("--grid_size", type=float, default=0.015,
                   help="Voxel size used by training (default: 0.015).")
    p.add_argument("--max_scene_points", type=int, default=12000,
                   help="Training scene-point cap (default: 12000).")
    p.add_argument("--max_files", type=int, default=0,
                   help="If > 0, only process the first N .npz files (smoke test).")
    p.add_argument(
        "--num_workers", type=int, default=1,
        help="Processes used to compute mergeable per-clip moments (default: 1).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(data_dir)

    if args.split_manifest:
        from dataset_components.libero_manifest import load_split_files

        files = load_split_files(data_dir, args.split_manifest, args.split)
    else:
        files = sorted(data_dir.rglob("*.npz"))
    if args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        raise RuntimeError(f"No .npz files found under {data_dir}")

    if args.num_cameras < 1:
        raise ValueError("--num_cameras must be >= 1")
    if args.grid_size <= 0:
        raise ValueError("--grid_size must be > 0")
    if args.max_scene_points < 1:
        raise ValueError("--max_scene_points must be >= 1")
    if args.num_workers < 1:
        raise ValueError("--num_workers must be >= 1")
    pipeline_args = _StatsArgs(
        grid_size=float(args.grid_size),
        max_scene_points=int(args.max_scene_points),
    )

    # Streaming accumulators.
    robot_sum = None
    robot_sq = None
    robot_n = 0
    scene_sum = None
    scene_sq = None
    scene_n = 0

    per_t_sum = [np.zeros(3, dtype=np.float64) for _ in range(T_FRAMES)]
    per_t_sq = [np.zeros(3, dtype=np.float64) for _ in range(T_FRAMES)]
    per_t_n = [0] * T_FRAMES

    print(
        f"[compute_norm_stats] {len(files)} clips from {data_dir} "
        f"(cameras={args.num_cameras}, grid_size={pipeline_args.grid_size}, "
        f"max_scene_points={pipeline_args.max_scene_points})"
    )
    worker_args = ((path, pipeline_args, args.num_cameras) for path in files)
    if args.num_workers == 1:
        moments = (_sample_moments(*item) for item in worker_args)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=args.num_workers)
        moments = executor.map(_sample_moments, files,
                               [pipeline_args] * len(files),
                               [args.num_cameras] * len(files))

    for i, moment in enumerate(moments):
        if moment["error"] is not None:
            print(f"  [{i}] SKIP {Path(moment['path']).name}: {moment['error']}")
            continue
        if robot_sum is None:
            robot_sum = np.zeros_like(moment["robot_sum"])
            robot_sq = np.zeros_like(moment["robot_sq"])
            scene_sum = np.zeros_like(moment["scene_sum"])
            scene_sq = np.zeros_like(moment["scene_sq"])

        robot_sum += moment["robot_sum"]
        robot_sq += moment["robot_sq"]
        robot_n += moment["robot_n"]
        scene_sum += moment["scene_sum"]
        scene_sq += moment["scene_sq"]
        scene_n += moment["scene_n"]
        for timestep in range(T_FRAMES):
            per_t_sum[timestep] += moment["per_t_sum"][timestep]
            per_t_sq[timestep] += moment["per_t_sq"][timestep]
            per_t_n[timestep] += int(moment["per_t_n"][timestep])

        if (i + 1) % 5 == 0 or i + 1 == len(files):
            print(f"  [{i + 1}/{len(files)}] processed "
                  f"(robot_n={robot_n}, scene_n={scene_n})")
    if executor is not None:
        executor.shutdown()

    if robot_n == 0 or scene_n == 0:
        raise RuntimeError("No usable features extracted; cannot compute stats.")

    robot_mean = robot_sum / robot_n
    robot_var = np.maximum(robot_sq / robot_n - robot_mean ** 2, 0.0)
    scene_mean = scene_sum / scene_n
    scene_var = np.maximum(scene_sq / scene_n - scene_mean ** 2, 0.0)

    per_t_mean = []
    per_t_var = []
    for t in range(T_FRAMES):
        if per_t_n[t] == 0:
            mean_t = np.zeros(3, dtype=np.float64)
            var_t = np.zeros(3, dtype=np.float64)
        else:
            mean_t = per_t_sum[t] / per_t_n[t]
            var_t = np.maximum(per_t_sq[t] / per_t_n[t] - mean_t ** 2, 0.0)
        per_t_mean.append(mean_t)
        per_t_var.append(var_t)

    out = {
        "statistics": {
            args.domain: {
                "robot_features": {
                    "mean": robot_mean.tolist(),
                    "variance": robot_var.tolist(),
                },
                "scene_features": {
                    "mean": scene_mean.tolist(),
                    "variance": scene_var.tolist(),
                },
            }
        },
        "per_timestep_statistics": {
            args.domain: {
                "gt_scene_flows_relative": {
                    f"timestep_{t}": {
                        "mean": per_t_mean[t].tolist(),
                        "variance": per_t_var[t].tolist(),
                    } for t in range(T_FRAMES)
                }
            }
        },
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[compute_norm_stats] wrote {out_path} "
          f"(robot_dim={robot_mean.size}, scene_dim={scene_mean.size})")


if __name__ == "__main__":
    main()
