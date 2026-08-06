from argparse import Namespace
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pointworld.checkpoint_contract import apply_model_contract_to_args


def test_finetune_keeps_target_camera_policy_and_stats() -> None:
    args = Namespace(
        ptv3_size="base",
        ptv3_patch_size=128,
        predictor_dim=256,
        max_scene_points=10000,
        max_robot_points=1000,
        grid_size=0.1,
        depth_threshold=0.01,
        norm_stats_path="stats/libero",
        train_min_num_cameras=3,
        train_max_num_cameras=3,
        eval_min_num_cameras=3,
        eval_max_num_cameras=3,
    )
    source = {
        "ptv3_size": "small",
        "ptv3_patch_size": 256,
        "predictor_dim": 128,
        "max_scene_points": 12000,
        "max_robot_points": 500,
        "grid_size": 0.015,
        "depth_threshold": 0.003,
        "norm_stats_path": "stats/droid",
        "train_min_num_cameras": 1,
        "train_max_num_cameras": 2,
        "eval_min_num_cameras": 2,
        "eval_max_num_cameras": 2,
    }

    apply_model_contract_to_args(
        args,
        source,
        context="test checkpoint",
        explicit_cli_dests={
            "norm_stats_path",
            "train_min_num_cameras",
            "train_max_num_cameras",
            "eval_min_num_cameras",
            "eval_max_num_cameras",
        },
        skip_data_contract=True,
    )

    assert args.ptv3_size == "small"
    assert args.max_scene_points == 12000
    assert args.norm_stats_path == "stats/libero"
    assert (
        args.train_min_num_cameras,
        args.train_max_num_cameras,
        args.eval_min_num_cameras,
        args.eval_max_num_cameras,
    ) == (3, 3, 3, 3)
