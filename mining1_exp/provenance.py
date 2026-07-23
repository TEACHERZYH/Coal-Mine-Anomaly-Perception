from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Dict, Iterable, Mapping, Optional, Union


PathLike = Union[str, Path]
HASH_ALGORITHM = "sha256"
ATTESTATION_SCHEME = "canonical_json_sha256_v1"
DEFAULT_BRANCH_ARTIFACTS = {
    "feature_store": "data/locked/branch_test_features.parquet",
    "feature_manifest": "data/locked/branch_test_feature_manifest.json",
    "detection_truth": "data/sealed/branch_detection_truth.parquet",
    "concept_truth": "data/sealed/branch_concept_truth.parquet",
    "truth_manifest": "data/sealed/branch_test_truth_manifest.json",
    "candidate_seal": "data/seals/branch_test_seal_candidate.json",
    "final_seal": "data/seals/branch_test_seal.json",
}


def _source_lock_amendment_path(root: Path) -> Path:
    candidates = []
    for path in root.glob("configs/source_lock_amendment.v*.json"):
        match = re.fullmatch(r"source_lock_amendment[.]v(\d+)[.]json", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]
    return root / "configs/source_lock_amendment.v2.json"


@dataclass(frozen=True)
class ArtifactDigest:
    path: str
    sha256: str
    bytes: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: PathLike, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_json_sha256(payload: Any) -> str:
    return sha256_bytes(canonical_json_bytes(payload))


def validate_source_lock_amendment(project_root: PathLike) -> Optional[Dict[str, str]]:
    root = Path(project_root).resolve()
    amendment_path = _source_lock_amendment_path(root)
    if not amendment_path.is_file():
        return None
    payload = json.loads(amendment_path.read_text(encoding="utf-8-sig"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("status") != "locked"
    ):
        raise ValueError("source-lock amendment is not locked")
    protocol_path = root / "configs/protocol_lock.pretest.yaml"
    if sha256_file(protocol_path) != payload.get("base_protocol_sha256"):
        raise ValueError("source-lock amendment base protocol hash drifted")
    manifest_relative = Path(str(payload.get("source_manifest_path", "")))
    if manifest_relative.is_absolute() or ".." in manifest_relative.parts:
        raise ValueError("source-lock amendment manifest path is unsafe")
    manifest_path = root / manifest_relative
    if (
        not manifest_path.is_file()
        or sha256_file(manifest_path) != payload.get("source_manifest_sha256")
    ):
        raise ValueError("source-lock amendment manifest hash drifted")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(files, list) or not files:
        raise ValueError("source-lock amendment manifest is empty")
    normalized = []
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ValueError("source-lock amendment contains an invalid file entry")
        relative = Path(str(item["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("source-lock amendment contains an unsafe source path")
        source = root / relative
        if not source.is_file() or sha256_file(source) != item["sha256"]:
            raise ValueError(f"source-lock amendment source drifted: {relative.as_posix()}")
        normalized.append({"path": relative.as_posix(), "sha256": str(item["sha256"])})
    if normalized != sorted(normalized, key=lambda value: value["path"]):
        raise ValueError("source-lock amendment manifest is not canonically sorted")
    source_hash = canonical_json_sha256(normalized)
    if (
        manifest.get("source_code_sha256") != source_hash
        or payload.get("amended_source_code_sha256") != source_hash
    ):
        raise ValueError("source-lock amendment source hash is invalid")
    amendment_id = str(payload.get("amendment_id", "")).strip()
    if not amendment_id:
        raise ValueError("source-lock amendment ID is missing")
    return {
        "amendment_id": amendment_id,
        "amendment_sha256": sha256_file(amendment_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_code_sha256": source_hash,
    }


def branch_artifact_path(project_root: PathLike, artifact_key: str) -> Path:
    if artifact_key not in DEFAULT_BRANCH_ARTIFACTS:
        raise ValueError(f"unknown branch artifact key: {artifact_key}")
    root = Path(project_root).resolve()
    amendment_path = _source_lock_amendment_path(root)
    relative_value = DEFAULT_BRANCH_ARTIFACTS[artifact_key]
    if amendment_path.is_file():
        payload = json.loads(amendment_path.read_text(encoding="utf-8-sig"))
        artifacts = payload.get("branch_test_artifacts")
        if not isinstance(artifacts, dict) or artifact_key not in artifacts:
            raise ValueError(f"source-lock amendment lacks branch artifact: {artifact_key}")
        relative_value = str(artifacts[artifact_key])
    relative = Path(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("branch artifact path is unsafe")
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"branch artifact is missing: {relative.as_posix()}")
    return path


def add_canonical_json_attestation(
    payload: Mapping[str, Any], signer_identity: str
) -> Dict[str, Any]:
    if not signer_identity.strip():
        raise ValueError("attestation signer identity is required")
    attested = dict(payload)
    attested["signer_identity"] = signer_identity
    attested["attestation_scheme"] = ATTESTATION_SCHEME
    attested["attestation_sha256"] = canonical_json_sha256(attested)
    return attested


def validate_canonical_json_attestation(payload: Mapping[str, Any]) -> None:
    if payload.get("attestation_scheme") != ATTESTATION_SCHEME:
        raise ValueError("attestation scheme is invalid")
    if not str(payload.get("signer_identity", "")).strip():
        raise ValueError("attestation signer identity is missing")
    observed = payload.get("attestation_sha256")
    if not isinstance(observed, str) or re.fullmatch(r"[0-9a-f]{64}", observed) is None:
        raise ValueError("attestation_sha256 must be a SHA-256 string")
    unsigned = dict(payload)
    del unsigned["attestation_sha256"]
    if canonical_json_sha256(unsigned) != observed:
        raise ValueError("attestation does not bind the artifact payload")


def describe_artifact(path: PathLike) -> ArtifactDigest:
    artifact_path = Path(path).resolve()
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")
    return ArtifactDigest(
        path=str(artifact_path),
        sha256=sha256_file(artifact_path),
        bytes=artifact_path.stat().st_size,
    )


def atomic_write_bytes(path: PathLike, data: bytes) -> ArtifactDigest:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary_path), str(target))
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return describe_artifact(target)


def atomic_write_json(path: PathLike, payload: Any) -> ArtifactDigest:
    return atomic_write_bytes(path, canonical_json_bytes(payload) + b"\n")


def _describe_many(paths: Iterable[PathLike]) -> list[Dict[str, Any]]:
    return [describe_artifact(path).to_dict() for path in paths]


def build_receipt(
    *,
    step_id: str,
    status: str,
    command: str,
    inputs: Iterable[PathLike] = (),
    outputs: Iterable[PathLike] = (),
    config_snapshot: Optional[Mapping[str, Any]] = None,
    details: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    if status not in {"pass", "fail", "blocked", "not_applicable"}:
        raise ValueError(f"Unsupported receipt status: {status}")
    if not step_id.strip() or not command.strip():
        raise ValueError("Receipt step_id and command must be non-empty")
    return {
        "schema_version": 1,
        "step_id": step_id,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "hash_algorithm": HASH_ALGORITHM,
        "inputs": _describe_many(inputs),
        "outputs": _describe_many(outputs),
        "config_snapshot": dict(config_snapshot or {}),
        "details": dict(details or {}),
    }


def write_receipt(path: PathLike, receipt: Mapping[str, Any]) -> ArtifactDigest:
    return atomic_write_json(path, dict(receipt))
