from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

from ..data.seals import (
    SealValidationError,
    build_test_release,
    validate_test_release,
    validate_test_seal,
)
from ..provenance import (
    add_canonical_json_attestation,
    sha256_file,
    validate_canonical_json_attestation,
)
from .immutable import (
    GovernanceContractError,
    WriteOnceResult,
    require_sha256,
    require_timestamp,
    write_once_json,
)
from .prediction_lock import validate_prediction_lock


PathLike = Union[str, Path]
PROHIBITED_LABEL_ROLES = {
    "trainer",
    "selector",
    "calibrator",
    "policy_fitter",
    "test_predictor",
    "human_reviewer",
}


def _attest(payload: dict[str, Any], signer_identity: str) -> None:
    try:
        attested = add_canonical_json_attestation(payload, signer_identity)
    except ValueError as exc:
        raise GovernanceContractError(str(exc)) from exc
    payload.clear()
    payload.update(attested)


def _validate_attestation(payload: Mapping[str, Any]) -> None:
    try:
        validate_canonical_json_attestation(payload)
    except ValueError as exc:
        raise GovernanceContractError(str(exc)) from exc


def write_test_release(
    *,
    output_path: PathLike,
    scope: str,
    seal_path: PathLike,
    prediction_lock_path: PathLike,
    protocol_hash: str,
    source_hash: str,
    matrix_hash: str,
    evaluator_identity: str,
    evaluator_label_acl: Sequence[str],
    signer_identity: str,
) -> WriteOnceResult:
    seal = Path(seal_path).resolve()
    lock = Path(prediction_lock_path).resolve()
    if not seal.is_file() or not lock.is_file():
        raise GovernanceContractError("release requires existing seal and prediction lock")
    seal_before = sha256_file(seal)
    lock_before = sha256_file(lock)
    seal_payload = json.loads(seal.read_text(encoding="utf-8"))
    validate_test_seal(seal_payload)
    if seal_payload["scope"] != scope or seal_payload["stage"] != "final":
        raise GovernanceContractError("release requires the matching final test seal")
    lock_payload = json.loads(lock.read_text(encoding="utf-8"))
    validate_prediction_lock(lock_payload)
    if lock_payload["scope"] != scope or lock_payload["seal_hash"] != seal_before:
        raise GovernanceContractError("release seal or scope does not match prediction lock")
    expected_hashes = {
        "protocol_hash": protocol_hash,
        "source_hash": source_hash,
        "matrix_hash": matrix_hash,
    }
    for field, expected in expected_hashes.items():
        require_sha256(expected, field)
        if lock_payload[field] != expected or seal_payload[field] != expected:
            raise GovernanceContractError(
                f"release {field} does not match both seal and prediction lock"
            )
    expected_acl = [f"allow:evaluator:{evaluator_identity}"]
    if list(evaluator_label_acl) != expected_acl:
        raise GovernanceContractError("release ACL must name only the independent evaluator")
    if signer_identity == evaluator_identity:
        raise GovernanceContractError("release authority and evaluator must be distinct")
    payload = build_test_release(
        scope=scope,
        seal_hash=seal_before,
        prediction_lock_hash=lock_before,
        protocol_hash=protocol_hash,
        source_hash=source_hash,
        matrix_hash=matrix_hash,
        evaluator_identity=evaluator_identity,
        evaluator_label_acl=evaluator_label_acl,
        signer_identity=signer_identity,
    )
    result = write_once_json(output_path, payload)
    if sha256_file(seal) != seal_before or sha256_file(lock) != lock_before:
        raise GovernanceContractError("release mutated the seal or prediction lock")
    return result


def validate_label_access(
    *,
    role: str,
    principal_identity: str,
    seal_hash: str,
    prediction_lock_hash: str,
    release_payload: Optional[Mapping[str, Any]],
) -> None:
    if role in PROHIBITED_LABEL_ROLES:
        raise GovernanceContractError(f"role is never authorized for test labels: {role}")
    if role != "evaluator" or release_payload is None:
        raise GovernanceContractError("test labels remain sealed before evaluator release")
    try:
        validate_test_release(release_payload)
    except SealValidationError as exc:
        raise GovernanceContractError(str(exc)) from exc
    if release_payload["seal_hash"] != seal_hash:
        raise GovernanceContractError("release does not match the requested seal")
    if release_payload["prediction_lock_hash"] != prediction_lock_hash:
        raise GovernanceContractError("release does not match the prediction lock")
    expected_acl = [f"allow:evaluator:{principal_identity}"]
    if release_payload["evaluator_identity"] != principal_identity:
        raise GovernanceContractError("release evaluator identity does not match the caller")
    if release_payload["evaluator_label_acl"] != expected_acl:
        raise GovernanceContractError("evaluator ACL is absent or contains extra principals")


def build_invalidation(
    *,
    scope: str,
    seal_hash: str,
    prediction_lock_hash: str,
    reason: str,
    observed_prediction_hashes: Mapping[str, str],
    signer_identity: str,
) -> dict[str, Any]:
    if scope not in {"branch", "episode"} or not reason.strip():
        raise GovernanceContractError("invalidation scope and reason are required")
    require_sha256(seal_hash, "seal_hash")
    require_sha256(prediction_lock_hash, "prediction_lock_hash")
    if not observed_prediction_hashes:
        raise GovernanceContractError("invalidation requires observed drift hashes")
    for family, value in observed_prediction_hashes.items():
        if not str(family).strip():
            raise GovernanceContractError("invalidation family is empty")
        require_sha256(value, f"observed_prediction_hashes.{family}")
    payload = {
        "scope": scope,
        "seal_hash": seal_hash,
        "prediction_lock_hash": prediction_lock_hash,
        "reason": reason,
        "observed_prediction_hashes": dict(observed_prediction_hashes),
        "invalidated_at": datetime.now(timezone.utc).isoformat(),
        "status": "invalidated",
    }
    _attest(payload, signer_identity)
    return payload


def validate_invalidation(payload: Mapping[str, Any]) -> None:
    required = {
        "scope",
        "seal_hash",
        "prediction_lock_hash",
        "reason",
        "observed_prediction_hashes",
        "signer_identity",
        "attestation_scheme",
        "attestation_sha256",
        "invalidated_at",
        "status",
    }
    if not required.issubset(payload) or payload["status"] != "invalidated":
        raise GovernanceContractError("invalidation record is incomplete")
    if payload["scope"] not in {"branch", "episode"} or not str(payload["reason"]).strip():
        raise GovernanceContractError("invalidation scope or reason is invalid")
    require_sha256(payload["seal_hash"], "seal_hash")
    require_sha256(payload["prediction_lock_hash"], "prediction_lock_hash")
    require_timestamp(payload["invalidated_at"], "invalidated_at")
    _validate_attestation(payload)
    hashes = payload["observed_prediction_hashes"]
    if not isinstance(hashes, dict) or not hashes:
        raise GovernanceContractError("invalidation drift hashes are missing")
    for family, value in hashes.items():
        require_sha256(value, f"observed_prediction_hashes.{family}")


def write_invalidation(
    path: PathLike, payload: Mapping[str, Any]
) -> WriteOnceResult:
    validate_invalidation(payload)
    return write_once_json(path, payload)
