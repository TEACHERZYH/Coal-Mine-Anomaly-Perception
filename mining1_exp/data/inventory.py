from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd

from ..provenance import sha256_file
from .manifests import validate_archive_manifest


PathLike = Union[str, Path]


def read_archive_manifest(path: PathLike) -> pd.DataFrame:
    frame = pd.read_csv(Path(path), dtype="string", keep_default_na=False)
    if "byte_size" in frame.columns:
        frame["byte_size"] = pd.to_numeric(frame["byte_size"], errors="raise")
    return validate_archive_manifest(frame)


def bounded_file_inventory(root: PathLike, max_files: int) -> Dict[str, Any]:
    """Inventory at most max_files without extracting archives or scanning past the bound."""
    if max_files <= 0:
        raise ValueError("max_files must be positive")
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"Inventory root not found: {root_path}")

    pending = [root_path]
    records: List[Dict[str, Any]] = []
    truncated = False
    while pending and not truncated:
        directory = pending.pop()
        entries = sorted(os.scandir(directory), key=lambda item: item.name)
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                pending.append(Path(entry.path))
                continue
            if not entry.is_file(follow_symlinks=False):
                continue
            if len(records) >= max_files:
                truncated = True
                break
            file_path = Path(entry.path).resolve()
            records.append(
                {
                    "relative_path": file_path.relative_to(root_path).as_posix(),
                    "byte_size": file_path.stat().st_size,
                    "sha256": sha256_file(file_path),
                }
            )
    return {
        "root": str(root_path),
        "max_files": max_files,
        "files": records,
        "file_count": len(records),
        "truncated": truncated,
        "archives_extracted": False,
    }
