from argparse import Namespace
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_components.dataloader import (
    _resolve_libero_data_dir,
    _resolve_libero_data_source,
)


def _args(root: Path | None = None, **overrides) -> Namespace:
    values = {
        "data_dirs": [str(root)] if root is not None else [],
        "libero_data_dir_train": None,
        "libero_data_dir_val": None,
        "libero_data_root": None,
        "libero_split_manifest": None,
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


def test_libero_manifest_source_selects_train_and_val_files(tmp_path: Path) -> None:
    train = tmp_path / "task__demo_0__start000000.npz"
    val = tmp_path / "task__demo_1__start000000.npz"
    train.touch()
    val.touch()
    manifest = tmp_path / "paper.json"
    manifest.write_text(
        '{"splits": {"train": ["task__demo_0__start000000.npz"], '
        '"val": ["task__demo_1__start000000.npz"]}}'
    )
    args = _args(
        libero_data_root=str(tmp_path),
        libero_split_manifest=str(manifest),
    )

    train_root, train_files, _ = _resolve_libero_data_source(args, "train")
    val_root, val_files, _ = _resolve_libero_data_source(args, "test")

    assert train_root == val_root == str(tmp_path.resolve())
    assert train_files == [train.resolve()]
    assert val_files == [val.resolve()]
