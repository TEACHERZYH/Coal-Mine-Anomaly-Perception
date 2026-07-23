from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Union

import yaml

from .provenance import canonical_json_sha256, sha256_file


PathLike = Union[str, Path]


class ConfigError(ValueError):
    """Raised when a structured configuration cannot be validated."""


def load_mapping(path: PathLike) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"Configuration file not found: {config_path}")
    suffix = config_path.suffix.lower()
    with config_path.open("r", encoding="utf-8-sig") as handle:
        if suffix == ".json":
            payload = json.load(handle)
        elif suffix in {".yaml", ".yml"}:
            payload = yaml.safe_load(handle)
        else:
            raise ConfigError(f"Unsupported configuration format: {suffix or '<none>'}")
    if not isinstance(payload, dict):
        raise ConfigError(f"Configuration root must be a mapping: {config_path}")
    return payload


def resolved_config_snapshot(path: PathLike) -> Dict[str, Any]:
    config_path = Path(path).resolve()
    payload = load_mapping(config_path)
    return {
        "source_path": str(config_path),
        "source_sha256": sha256_file(config_path),
        "resolved_sha256": canonical_json_sha256(payload),
        "resolved": payload,
    }
