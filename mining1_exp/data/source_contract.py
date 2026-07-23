from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Mapping


SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class DatasetSourceContractError(ValueError):
    """Raised when a reviewed source archive set is incomplete or ambiguous."""


def _archive_part(
    payload: Mapping[str, Any],
    *,
    part_id: str,
    default_transfer_route: str,
) -> Dict[str, str]:
    if not SAFE_ID.fullmatch(part_id):
        raise DatasetSourceContractError(f"Unsafe archive_part_id: {part_id}")
    source = str(payload.get("source_path_or_url", "")).strip()
    digest = str(payload.get("archive_sha256", "")).strip()
    route = str(payload.get("transfer_route", default_transfer_route)).strip()
    if not source:
        raise DatasetSourceContractError(f"Archive part {part_id} lacks source_path_or_url")
    if not SHA256.fullmatch(digest):
        raise DatasetSourceContractError(f"Archive part {part_id} has an invalid SHA-256")
    if not route:
        raise DatasetSourceContractError(f"Archive part {part_id} lacks transfer_route")
    result = {
        "archive_part_id": part_id,
        "source_path_or_url": source,
        "archive_sha256": digest,
        "transfer_route": route,
    }
    minimal_path = str(payload.get("minimal_sample_local_path", "")).strip()
    if minimal_path:
        result["minimal_sample_local_path"] = minimal_path
    return result


def archive_parts_from_entry(entry: Mapping[str, Any]) -> list[Dict[str, str]]:
    """Normalize one primary archive and optional reviewed companion archives."""

    if not isinstance(entry, Mapping):
        raise DatasetSourceContractError("Dataset source entry must be an object")
    default_route = str(entry.get("transfer_route", "")).strip()
    primary = _archive_part(entry, part_id="primary", default_transfer_route=default_route)
    companions = entry.get("companion_archives", [])
    if not isinstance(companions, list):
        raise DatasetSourceContractError("companion_archives must be a list")
    parts = [primary]
    seen = {"primary"}
    for companion in companions:
        if not isinstance(companion, Mapping):
            raise DatasetSourceContractError("Each companion archive must be an object")
        part_id = str(companion.get("archive_part_id", "")).strip()
        if not part_id or part_id in seen:
            raise DatasetSourceContractError(
                f"Duplicate or empty companion archive_part_id: {part_id}"
            )
        seen.add(part_id)
        parts.append(
            _archive_part(
                companion,
                part_id=part_id,
                default_transfer_route=default_route,
            )
        )
    return parts


def archive_set_sha256(parts: list[Mapping[str, Any]]) -> str:
    normalized = [
        {
            "archive_part_id": str(part["archive_part_id"]),
            "sha256": str(part.get("sha256", part.get("archive_sha256", ""))),
        }
        for part in sorted(parts, key=lambda value: str(value["archive_part_id"]))
    ]
    if not normalized or any(not SHA256.fullmatch(item["sha256"]) for item in normalized):
        raise DatasetSourceContractError("Archive set contains an invalid SHA-256")
    data = json.dumps(normalized, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("ascii")).hexdigest()
