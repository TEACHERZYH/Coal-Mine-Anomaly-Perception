from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Union

from ..provenance import ArtifactDigest, canonical_json_bytes, describe_artifact


PathLike = Union[str, Path]
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class GovernanceContractError(ValueError):
    """Raised when an evidence-governance invariant is violated."""


@dataclass(frozen=True)
class WriteOnceResult:
    artifact: ArtifactDigest
    created: bool


def require_sha256(value: Any, field: str) -> str:
    normalized = str(value)
    if SHA256_PATTERN.fullmatch(normalized) is None:
        raise GovernanceContractError(f"{field} must be a lowercase SHA-256 string")
    return normalized


def require_timestamp(value: Any, field: str) -> str:
    normalized = str(value)
    parse_value = re.sub(r"(\.\d{6})\d+(Z|[+-]\d\d:\d\d)$", r"\1\2", normalized)
    try:
        parsed = datetime.fromisoformat(parse_value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise GovernanceContractError(f"{field} must be an ISO timestamp") from exc
    if parsed.tzinfo is None:
        raise GovernanceContractError(f"{field} must include a timezone")
    return normalized


def write_once_bytes(path: PathLike, data: bytes) -> WriteOnceResult:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file() or target.read_bytes() != data:
            raise GovernanceContractError(f"immutable artifact differs: {target}")
        return WriteOnceResult(describe_artifact(target), created=False)

    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(str(temporary), str(target))
        except FileExistsError:
            if target.read_bytes() != data:
                raise GovernanceContractError(f"immutable artifact race differs: {target}")
            return WriteOnceResult(describe_artifact(target), created=False)
        return WriteOnceResult(describe_artifact(target), created=True)
    finally:
        temporary.unlink(missing_ok=True)


def promote_once_file(path: PathLike, staged_path: PathLike) -> WriteOnceResult:
    """Atomically retain a staged file without loading large artifacts into memory."""
    target = Path(path).resolve()
    staged = Path(staged_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if not staged.is_file() or staged.parent != target.parent:
        raise GovernanceContractError("staged artifact must be a file beside its target")
    staged_artifact = describe_artifact(staged)
    try:
        if target.exists():
            target_artifact = describe_artifact(target)
            if (
                target_artifact.bytes != staged_artifact.bytes
                or target_artifact.sha256 != staged_artifact.sha256
            ):
                raise GovernanceContractError(f"immutable artifact differs: {target}")
            return WriteOnceResult(target_artifact, created=False)
        try:
            os.link(str(staged), str(target))
        except FileExistsError:
            target_artifact = describe_artifact(target)
            if (
                target_artifact.bytes != staged_artifact.bytes
                or target_artifact.sha256 != staged_artifact.sha256
            ):
                raise GovernanceContractError(f"immutable artifact race differs: {target}")
            return WriteOnceResult(target_artifact, created=False)
        return WriteOnceResult(describe_artifact(target), created=True)
    finally:
        staged.unlink(missing_ok=True)


def write_once_json(path: PathLike, payload: Mapping[str, Any]) -> WriteOnceResult:
    return write_once_bytes(path, canonical_json_bytes(dict(payload)) + b"\n")
