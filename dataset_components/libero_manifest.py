"""Validation and loading for LIBERO clip split manifests."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple


_CLIP_NAME_RE = re.compile(
    r"^(?P<task>.+)__demo_(?P<demo>\d+)__start(?P<start>\d+)\.npz$"
)


def clip_demo_identity(path: Path | str) -> Tuple[str, str] | None:
    """Return ``(task, demo_N)`` for the standard bulk-export filename."""
    match = _CLIP_NAME_RE.match(Path(path).name)
    if match is None:
        return None
    return match.group("task"), f"demo_{int(match.group('demo'))}"


def _read_manifest(path: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid LIBERO split manifest JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"LIBERO split manifest must contain a JSON object: {path}")
    return payload


def validate_split_manifest(
    data_root: Path | str,
    manifest_path: Path | str,
) -> Dict[str, List[Path]]:
    """Validate a manifest and resolve every clip against ``data_root``.

    Paths in the JSON must be relative to the data root. Symlink resolution is
    included in the containment check so a manifest cannot escape the pool.
    """
    root = Path(data_root).expanduser().resolve()
    manifest = Path(manifest_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"LIBERO data root not found: {root}")
    if not manifest.is_file():
        raise FileNotFoundError(f"LIBERO split manifest not found: {manifest}")

    payload = _read_manifest(manifest)
    splits = payload.get("splits")
    if not isinstance(splits, dict) or not splits:
        raise ValueError(f"LIBERO split manifest {manifest} needs a non-empty 'splits' object")

    resolved: Dict[str, List[Path]] = {}
    relative_owner: Dict[str, str] = {}
    resolved_owner: Dict[Path, str] = {}
    demo_owner: Dict[Tuple[str, str], str] = {}
    for split, entries in splits.items():
        if not isinstance(split, str) or not split:
            raise ValueError("LIBERO manifest split names must be non-empty strings")
        if not isinstance(entries, list):
            raise ValueError(f"LIBERO manifest split {split!r} must be a list")
        split_files: List[Path] = []
        seen: set[str] = set()
        for index, entry in enumerate(entries):
            if not isinstance(entry, str) or not entry:
                raise ValueError(
                    f"LIBERO manifest {split!r}[{index}] must be a non-empty relative path"
                )
            relative = Path(entry)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(
                    f"LIBERO manifest path must be relative to data root: {entry!r}"
                )
            normalized = relative.as_posix()
            if normalized in seen:
                raise ValueError(f"Duplicate path in LIBERO split {split!r}: {entry!r}")
            seen.add(normalized)
            owner = relative_owner.get(normalized)
            if owner is not None:
                raise ValueError(
                    f"LIBERO clip appears in both {owner!r} and {split!r}: {entry!r}"
                )
            relative_owner[normalized] = split

            clip = (root / relative).resolve()
            try:
                clip.relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    f"LIBERO manifest path escapes data root: {entry!r}"
                ) from exc
            if clip.suffix.lower() != ".npz":
                raise ValueError(f"LIBERO manifest path must end in .npz: {entry!r}")
            if not clip.is_file():
                raise FileNotFoundError(f"LIBERO manifest clip not found: {clip}")
            physical_owner = resolved_owner.get(clip)
            if physical_owner is not None:
                raise ValueError(
                    "LIBERO clip resolves to the same file in both "
                    f"{physical_owner!r} and {split!r}: {entry!r}"
                )
            resolved_owner[clip] = split

            identity = clip_demo_identity(relative)
            if identity is not None:
                previous = demo_owner.get(identity)
                if previous is not None and previous != split:
                    raise ValueError(
                        "LIBERO windows from one demo cross splits: "
                        f"{identity[0]} {identity[1]} is in {previous!r} and {split!r}"
                    )
                demo_owner[identity] = split
            split_files.append(clip)
        resolved[split] = split_files
    return resolved


def load_split_files(
    data_root: Path | str,
    manifest_path: Path | str,
    split: str,
) -> List[Path]:
    """Return validated absolute clip paths for one manifest split."""
    splits = validate_split_manifest(data_root, manifest_path)
    if split not in splits:
        raise KeyError(
            f"LIBERO split {split!r} is absent from {Path(manifest_path)}; "
            f"available splits: {sorted(splits)}"
        )
    if not splits[split]:
        raise RuntimeError(f"LIBERO manifest split {split!r} contains no clips")
    return splits[split]


def write_split_manifest(
    output: Path | str,
    splits: Mapping[str, Sequence[str]],
    *,
    metadata: Mapping[str, object] | None = None,
) -> None:
    """Write the stable clip-level manifest schema used by the loader."""
    payload: dict[str, object] = {"version": 1}
    if metadata:
        payload.update(metadata)
    payload["splits"] = {name: list(paths) for name, paths in splits.items()}
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")


__all__ = [
    "clip_demo_identity",
    "load_split_files",
    "validate_split_manifest",
    "write_split_manifest",
]
