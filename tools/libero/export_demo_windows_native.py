"""Export multiple PointWorld windows from one LIBERO demo with one renderer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.libero import sample_schema
from tools.libero.export_clip_native import NativeDemoExporter


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo_hdf5", required=True)
    parser.add_argument("--demo_id", required=True)
    parser.add_argument("--start_indices", nargs="+", type=int, required=True)
    parser.add_argument("--output_paths", nargs="+", required=True)
    parser.add_argument("--result_json", required=True)
    parser.add_argument("--frame_step", type=int, default=sample_schema.DEFAULT_FRAME_STEP)
    parser.add_argument("--window_stride_raw", type=int, default=10)
    parser.add_argument("--bddl", default=None)
    parser.add_argument("--bddl_search_dir", action="append", default=None)
    parser.add_argument(
        "--camera_names", nargs="+", default=list(sample_schema.DEFAULT_CAMERA_NAMES)
    )
    parser.add_argument(
        "--camera_layout", default="oblique_triplet",
        choices=("native", "oblique_pair", "oblique_triplet"),
    )
    parser.add_argument("--camera_height", type=int, default=sample_schema.H_RELEASE)
    parser.add_argument("--camera_width", type=int, default=sample_schema.W_RELEASE)
    parser.add_argument("--robot_points_per_body", type=int, default=64)
    parser.add_argument(
        "--allow_bddl_reconstruction",
        action="store_true",
        help="Permit non-faithful BDDL reconstruction when model_file is absent.",
    )
    parser.add_argument(
        "--workspace_bounds", nargs=6, type=float, default=None,
        metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if len(args.start_indices) != len(args.output_paths):
        raise ValueError("--start_indices and --output_paths must have equal length")
    if args.frame_step < 1 or args.window_stride_raw < 1:
        raise ValueError("frame_step and window_stride_raw must be >= 1")

    exporter = NativeDemoExporter(args)
    results = []
    try:
        for index, (start_idx, output) in enumerate(
            zip(args.start_indices, args.output_paths), start=1
        ):
            job_args = argparse.Namespace(
                **vars(args), start_idx=int(start_idx), output=str(output)
            )
            result = exporter.export(job_args)
            results.append(result)
            print(
                f"[export_demo_windows_native] {index}/{len(args.start_indices)} "
                f"start={start_idx} elapsed={result['elapsed_s']:.1f}s",
                flush=True,
            )
    finally:
        exporter.close()

    result_path = Path(args.result_json)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
