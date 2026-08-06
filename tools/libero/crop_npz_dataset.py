"""Apply a world-frame workspace box to an existing LIBERO NPZ dataset.

The source tree is never modified. Relative paths (including ``train/`` and
``val/``) are preserved under a different output root, so the result can be
passed directly to ``train.py --data_dirs`` without split leakage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.libero.sample_schema import crop_scene_to_workspace, load_npz, save_npz


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument(
        "--workspace_bounds", nargs=6, type=float, required=True,
        metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
    )
    parser.add_argument(
        "--max_files_per_split", type=int, default=0,
        help="If > 0, retain only the first N files in each train/val split.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)
    if input_root == output_root or input_root in output_root.parents:
        raise ValueError("--output_root must be outside --input_root")
    if args.max_files_per_split < 0:
        raise ValueError("--max_files_per_split must be >= 0")

    plan: list[Path] = []
    split_counts: dict[str, int] = {}
    for split in ("train", "val"):
        split_root = input_root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Required split directory not found: {split_root}")
        files = sorted(split_root.rglob("*.npz"))
        if args.max_files_per_split > 0:
            files = files[: args.max_files_per_split]
        if not files:
            raise RuntimeError(f"No NPZ clips found under {split_root}")
        split_counts[split] = len(files)
        plan.extend(files)

    results = []
    for index, source in enumerate(plan, start=1):
        relative = source.relative_to(input_root)
        destination = output_root / relative
        if destination.exists() and not args.overwrite:
            results.append({"source": str(source), "output": str(destination), "skipped": True})
            continue
        sample = load_npz(str(source))
        crop_scene_to_workspace(sample, args.workspace_bounds)
        destination.parent.mkdir(parents=True, exist_ok=True)
        save_npz(sample, str(destination))
        results.append({
            "source": str(source),
            "output": str(destination),
            "points_before": sample["workspace_points_before_per_cam"].tolist(),
            "points_after": sample["workspace_points_after_per_cam"].tolist(),
        })
        print(f"[crop_npz_dataset] {index}/{len(plan)} {relative}", flush=True)

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "workspace_bounds": list(args.workspace_bounds),
        "split_counts": split_counts,
        "results": results,
    }
    (output_root / "crop_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[crop_npz_dataset] wrote {output_root} ({split_counts})", flush=True)


if __name__ == "__main__":
    main()
