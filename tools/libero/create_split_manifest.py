"""Create a clip-level LIBERO manifest from demo-level train/val JSON files.

The input JSON format is the paper split format with a ``records`` list; each
record contains an HDF5 ``path`` and its assigned ``demo_keys``. Exported clips
are discovered recursively under ``--data_root`` and are not moved or copied.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_components.libero_manifest import (  # noqa: E402
    clip_demo_identity,
    validate_split_manifest,
    write_split_manifest,
)


def _task_name(source_path: str) -> str:
    stem = Path(source_path).stem
    for suffix in ("_demo_pcd", "_demo"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _read_demo_assignments(
    path: Path,
    expected_split: str,
) -> Dict[Tuple[str, str], str]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError(f"Demo split JSON needs a 'records' list: {path}")
    declared = payload.get("split")
    if declared is not None and declared != expected_split:
        raise ValueError(
            f"Expected {expected_split!r} JSON but {path} declares split={declared!r}"
        )
    assignments: Dict[Tuple[str, str], str] = {}
    for record_index, record in enumerate(payload["records"]):
        if not isinstance(record, dict):
            raise ValueError(f"{path}: records[{record_index}] must be an object")
        source = record.get("path")
        demo_keys = record.get("demo_keys")
        if not isinstance(source, str) or not isinstance(demo_keys, list):
            raise ValueError(
                f"{path}: records[{record_index}] needs string 'path' and list 'demo_keys'"
            )
        task = _task_name(source)
        for demo in demo_keys:
            if not isinstance(demo, str) or not demo.startswith("demo_"):
                raise ValueError(f"Invalid demo key in {path}: {demo!r}")
            key = (task, demo)
            if key in assignments:
                raise ValueError(f"Duplicate demo assignment in {path}: {task} {demo}")
            assignments[key] = expected_split
    return assignments


def build_manifest(
    data_root: Path,
    train_json: Path,
    val_json: Path,
    output: Path,
    clip_glob: str = "*.npz",
) -> dict[str, list[str]]:
    root = data_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    assignments = _read_demo_assignments(train_json, "train")
    val_assignments = _read_demo_assignments(val_json, "val")
    overlap = sorted(set(assignments) & set(val_assignments))
    if overlap:
        task, demo = overlap[0]
        raise ValueError(f"Demo is assigned to both train and val: {task} {demo}")
    assignments.update(val_assignments)

    clips = sorted(path for path in root.rglob(clip_glob) if path.is_file())
    if not clips:
        raise RuntimeError(f"No clips matching {clip_glob!r} under {root}")
    splits: dict[str, list[str]] = {"train": [], "val": []}
    unmatched: list[Path] = []
    matched_demos: set[Tuple[str, str]] = set()
    represented_tasks: set[str] = set()
    for clip in clips:
        identity = clip_demo_identity(clip)
        if identity is None or identity not in assignments:
            unmatched.append(clip)
            continue
        represented_tasks.add(identity[0])
        matched_demos.add(identity)
        splits[assignments[identity]].append(clip.relative_to(root).as_posix())
    if unmatched:
        preview = ", ".join(str(path.relative_to(root)) for path in unmatched[:5])
        raise ValueError(
            f"{len(unmatched)} exported clips could not be matched to the demo splits; "
            f"first entries: {preview}"
        )
    expected_demos = {
        identity for identity in assignments if identity[0] in represented_tasks
    }
    missing_demos = sorted(expected_demos - matched_demos)
    if missing_demos:
        preview = ", ".join(f"{task} {demo}" for task, demo in missing_demos[:5])
        raise ValueError(
            f"{len(missing_demos)} assigned demos have no exported clips; "
            f"first entries: {preview}"
        )
    if not splits["train"] or not splits["val"]:
        raise RuntimeError(
            f"Both train and val must contain clips; got "
            f"train={len(splits['train'])}, val={len(splits['val'])}"
        )

    write_split_manifest(
        output,
        splits,
        metadata={
            "name": "libero_paper_split",
            "source_demo_splits": {
                "train": train_json.name,
                "val": val_json.name,
            },
        },
    )
    validate_split_manifest(root, output)
    return splits


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True, help="NPZ pool root (searched recursively).")
    parser.add_argument("--train_demo_split", required=True, help="Paper train demo-level JSON.")
    parser.add_argument("--val_demo_split", required=True, help="Paper val demo-level JSON.")
    parser.add_argument("--output", required=True, help="Output clip-level manifest JSON.")
    parser.add_argument(
        "--clip_glob", default="*.npz",
        help="Filename glob used recursively below data root (default: *.npz).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    splits = build_manifest(
        Path(args.data_root),
        Path(args.train_demo_split),
        Path(args.val_demo_split),
        Path(args.output),
        args.clip_glob,
    )
    print(
        f"[create_split_manifest] wrote {args.output} "
        f"(train={len(splits['train'])}, val={len(splits['val'])})"
    )


if __name__ == "__main__":
    main()
