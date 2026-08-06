from pathlib import Path

import h5py

from tools.libero.bulk_export import _list_demos


def test_list_demos_uses_numeric_order(tmp_path: Path) -> None:
    path = tmp_path / "task_demo.hdf5"
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        for name in ("demo_10", "demo_2", "demo_1", "custom"):
            data.create_group(name)

    assert _list_demos(path) == ["demo_1", "demo_2", "demo_10", "custom"]
