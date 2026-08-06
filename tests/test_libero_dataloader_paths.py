from argparse import Namespace
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_components.dataloader import _resolve_libero_data_dir


def _args(root: Path | None = None, **overrides) -> Namespace:
    values = {
        "data_dirs": [str(root)] if root is not None else [],
        "libero_data_dir_train": None,
        "libero_data_dir_val": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_libero_root_resolves_disjoint_train_and_val(tmp_path: Path) -> None:
    train = tmp_path / "train"
    val = tmp_path / "val"
    train.mkdir()
    val.mkdir()

    args = _args(tmp_path)
    assert _resolve_libero_data_dir(args, "train") == str(train.resolve())
    assert _resolve_libero_data_dir(args, "test") == str(val.resolve())


def test_libero_root_is_never_recursively_used_for_both_splits(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Do not point both loaders"):
        _resolve_libero_data_dir(_args(tmp_path), "train")


def test_libero_explicit_split_overlap_is_rejected(tmp_path: Path) -> None:
    train = tmp_path / "train"
    nested_val = train / "val"
    nested_val.mkdir(parents=True)
    args = _args(
        libero_data_dir_train=str(train),
        libero_data_dir_val=str(nested_val),
    )
    with pytest.raises(ValueError, match="must be disjoint"):
        _resolve_libero_data_dir(args, "train")
