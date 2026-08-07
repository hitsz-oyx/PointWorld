from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import pytest

from dataset_components.libero_manifest import load_split_files, validate_split_manifest
from tools.libero.create_split_manifest import build_manifest
from tools.libero.bulk_export import main as bulk_export_main


def _touch_clip(root: Path, relative: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def _write_manifest(path: Path, splits: dict[str, list[str]]) -> None:
    path.write_text(json.dumps({"version": 1, "splits": splits}))


def test_load_split_files_resolves_relative_pool_paths(tmp_path: Path) -> None:
    train = _touch_clip(tmp_path, "clips/task_a__demo_0__start000000.npz")
    val = _touch_clip(tmp_path, "legacy/val/task_a__demo_1__start000000.npz")
    manifest = tmp_path / "paper.json"
    _write_manifest(
        manifest,
        {
            "train": ["clips/task_a__demo_0__start000000.npz"],
            "val": ["legacy/val/task_a__demo_1__start000000.npz"],
        },
    )

    assert load_split_files(tmp_path, manifest, "train") == [train.resolve()]
    assert load_split_files(tmp_path, manifest, "val") == [val.resolve()]


@pytest.mark.parametrize(
    "splits, error",
    [
        ({"train": ["../outside.npz"], "val": []}, "relative to data root"),
        (
            {"train": ["x.npz", "x.npz"], "val": []},
            "Duplicate path",
        ),
        (
            {"train": ["x.npz"], "val": ["x.npz"]},
            "appears in both",
        ),
    ],
)
def test_manifest_rejects_bad_paths(
    tmp_path: Path, splits: dict[str, list[str]], error: str
) -> None:
    _touch_clip(tmp_path, "x.npz")
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, splits)
    with pytest.raises(ValueError, match=error):
        validate_split_manifest(tmp_path, manifest)


def test_manifest_rejects_windows_from_same_demo_across_splits(tmp_path: Path) -> None:
    first = "task_a__demo_2__start000000.npz"
    second = "task_a__demo_2__start000010.npz"
    _touch_clip(tmp_path, first)
    _touch_clip(tmp_path, second)
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, {"train": [first], "val": [second]})

    with pytest.raises(ValueError, match="one demo cross splits"):
        validate_split_manifest(tmp_path, manifest)


def test_manifest_rejects_symlink_aliases_across_splits(tmp_path: Path) -> None:
    clip = _touch_clip(tmp_path, "clips/task_a__demo_0__start000000.npz")
    alias = tmp_path / "aliases" / "different_name.npz"
    alias.parent.mkdir()
    alias.symlink_to(clip)
    manifest = tmp_path / "manifest.json"
    _write_manifest(
        manifest,
        {"train": ["clips/task_a__demo_0__start000000.npz"],
         "val": ["aliases/different_name.npz"]},
    )

    with pytest.raises(ValueError, match="resolves to the same file"):
        validate_split_manifest(tmp_path, manifest)


def test_build_manifest_maps_paper_demo_records_without_copying(tmp_path: Path) -> None:
    pool = tmp_path / "pool"
    train_clip = _touch_clip(pool, "old/train/task_a__demo_0__start000000.npz")
    val_clip = _touch_clip(pool, "old/val/task_a__demo_1__start000000.npz")
    train_json = tmp_path / "train.json"
    val_json = tmp_path / "val.json"
    train_json.write_text(json.dumps({
        "split": "train",
        "records": [{"path": "/source/task_a_demo_pcd.hdf5", "demo_keys": ["demo_0"]}],
    }))
    val_json.write_text(json.dumps({
        "split": "val",
        "records": [{"path": "/source/task_a_demo_pcd.hdf5", "demo_keys": ["demo_1"]}],
    }))
    output = pool / "splits" / "paper.json"

    splits = build_manifest(pool, train_json, val_json, output)

    assert splits == {
        "train": [train_clip.relative_to(pool).as_posix()],
        "val": [val_clip.relative_to(pool).as_posix()],
    }
    assert train_clip.exists() and val_clip.exists()


def test_build_manifest_rejects_assigned_demo_without_clips(tmp_path: Path) -> None:
    pool = tmp_path / "pool"
    _touch_clip(pool, "task_a__demo_0__start000000.npz")
    train_json = tmp_path / "train.json"
    val_json = tmp_path / "val.json"
    train_json.write_text(json.dumps({
        "split": "train",
        "records": [{
            "path": "/source/task_a_demo_pcd.hdf5",
            "demo_keys": ["demo_0", "demo_2"],
        }],
    }))
    val_json.write_text(json.dumps({
        "split": "val",
        "records": [{
            "path": "/source/task_a_demo_pcd.hdf5",
            "demo_keys": ["demo_1"],
        }],
    }))

    with pytest.raises(ValueError, match="assigned demos have no exported clips"):
        build_manifest(pool, train_json, val_json, pool / "paper.json")


def test_bulk_export_pool_layout_plans_unassigned_clip_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = source_root / "task_a_demo.hdf5"
    with h5py.File(source, "w") as handle:
        handle.create_dataset("data/demo_0/states", shape=(21, 1), dtype="f4")
    output_root = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", [
        "bulk_export",
        "--libero_root", str(source_root),
        "--output_root", str(output_root),
        "--output_layout", "pool",
        "--plan_only",
    ])

    bulk_export_main()

    manifest = json.loads((output_root / "manifest.json").read_text())
    assert manifest["output_layout"] == "pool"
    assert manifest["plan_size"] == 1
    assert manifest["plan"][0]["split"] == "pool"
    assert Path(manifest["plan"][0]["output"]).parent == output_root / "clips"
