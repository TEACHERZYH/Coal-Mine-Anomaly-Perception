from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import re
from typing import Any, Dict, Iterable, Mapping, Optional

from ..provenance import (
    add_canonical_json_attestation,
    validate_canonical_json_attestation,
)


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
DENIED_LABEL_ROLES = {
    "deny:trainer",
    "deny:selector",
    "deny:calibrator",
    "deny:policy_fitter",
    "deny:test_predictor",
    "deny:human_reviewer",
}
PROHIBITED_LABEL_READERS = {
    "trainer",
    "selector",
    "calibrator",
    "policy_fitter",
    "test_predictor",
    "human_reviewer",
}
TEST_SEAL_REQUIRED_FIELDS = {
    "scope",
    "stage",
    "split_hash",
    "feature_manifest_hash",
    "label_manifest_hash",
    "feature_acl",
    "label_acl",
    "artifact_contract_hash",
    "sealed_at",
    "status",
}
TEST_RELEASE_REQUIRED_FIELDS = {
    "scope",
    "seal_hash",
    "prediction_lock_hash",
    "protocol_hash",
    "source_hash",
    "matrix_hash",
    "evaluator_identity",
    "evaluator_label_acl",
    "signer_identity",
    "attestation_scheme",
    "attestation_sha256",
    "authorized_at",
    "authorization_status",
}


class SealValidationError(ValueError):
    """Raised when test seals or release records violate the append-only contract."""


def _require_sha(value: Any, field: str) -> None:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise SealValidationError(f"{field} must be a SHA-256 string")


def build_candidate_test_seal(
    *,
    scope: str,
    split_hash: str,
    feature_manifest_hash: str,
    label_manifest_hash: str,
    artifact_contract_hash: str,
    feature_acl: Iterable[str],
    label_acl: Iterable[str],
) -> Dict[str, Any]:
    payload = {
        "scope": scope,
        "stage": "candidate",
        "split_hash": split_hash,
        "feature_manifest_hash": feature_manifest_hash,
        "label_manifest_hash": label_manifest_hash,
        "feature_acl": list(feature_acl),
        "label_acl": list(label_acl),
        "artifact_contract_hash": artifact_contract_hash,
        "sealed_at": datetime.now(timezone.utc).isoformat(),
        "status": "sealed",
    }
    validate_test_seal(payload)
    return payload


def validate_test_seal(payload: Mapping[str, Any]) -> None:
    if not TEST_SEAL_REQUIRED_FIELDS.issubset(payload):
        raise SealValidationError("test seal is missing required fields")
    if payload["scope"] not in {"branch", "episode"}:
        raise SealValidationError("invalid test seal scope")
    if payload["stage"] not in {"candidate", "final"}:
        raise SealValidationError("invalid test seal stage")
    if payload["status"] != "sealed":
        raise SealValidationError("test seal status must be sealed")
    if not isinstance(payload["feature_acl"], list) or not isinstance(
        payload["label_acl"], list
    ):
        raise SealValidationError("test seal ACL fields must be lists")
    for field in (
        "split_hash",
        "feature_manifest_hash",
        "label_manifest_hash",
        "artifact_contract_hash",
    ):
        _require_sha(payload[field], field)
    if not DENIED_LABEL_ROLES.issubset(set(payload["label_acl"])):
        raise SealValidationError("test labels are not denied to all prohibited roles")
    conflicting = {
        entry
        for entry in payload["label_acl"]
        if entry in {f"allow:{role}" for role in PROHIBITED_LABEL_READERS}
    }
    if conflicting:
        raise SealValidationError("test label ACL contains conflicting read access")
    try:
        datetime.fromisoformat(str(payload["sealed_at"]).replace("Z", "+00:00"))
    except ValueError as exc:
        raise SealValidationError("sealed_at must be an ISO timestamp") from exc
    if payload["stage"] == "final":
        for field in ("candidate_seal_hash", "protocol_hash", "source_hash", "matrix_hash"):
            if field not in payload:
                raise SealValidationError(f"final test seal requires {field}")
            _require_sha(payload[field], field)
        if payload["scope"] == "episode":
            field = "episode_skeleton_candidate_seal_hash"
            if field not in payload:
                raise SealValidationError(f"final episode seal requires {field}")
            _require_sha(payload[field], field)


def build_test_release(
    *,
    scope: str,
    seal_hash: str,
    prediction_lock_hash: str,
    protocol_hash: str,
    source_hash: str,
    matrix_hash: str,
    evaluator_identity: str,
    evaluator_label_acl: Iterable[str],
    signer_identity: str,
    sealed_payload: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    original = deepcopy(dict(sealed_payload)) if sealed_payload is not None else None
    for field, value in {
        "seal_hash": seal_hash,
        "prediction_lock_hash": prediction_lock_hash,
        "protocol_hash": protocol_hash,
        "source_hash": source_hash,
        "matrix_hash": matrix_hash,
    }.items():
        _require_sha(value, field)
    if scope not in {"branch", "episode"} or not evaluator_identity.strip():
        raise SealValidationError("release scope and evaluator identity are required")
    payload = add_canonical_json_attestation(
        {
            "scope": scope,
            "seal_hash": seal_hash,
            "prediction_lock_hash": prediction_lock_hash,
            "protocol_hash": protocol_hash,
            "source_hash": source_hash,
            "matrix_hash": matrix_hash,
            "evaluator_identity": evaluator_identity,
            "evaluator_label_acl": list(evaluator_label_acl),
            "authorized_at": datetime.now(timezone.utc).isoformat(),
            "authorization_status": "released",
        },
        signer_identity,
    )
    if sealed_payload is not None and dict(sealed_payload) != original:
        raise SealValidationError("release construction mutated the test seal")
    validate_test_release(payload)
    return payload


def validate_test_release(payload: Mapping[str, Any]) -> None:
    if not TEST_RELEASE_REQUIRED_FIELDS.issubset(payload):
        raise SealValidationError("test release is missing required fields")
    try:
        validate_canonical_json_attestation(payload)
    except ValueError as exc:
        raise SealValidationError(str(exc)) from exc
    if payload["scope"] not in {"branch", "episode"}:
        raise SealValidationError("invalid test release scope")
    if payload["authorization_status"] != "released":
        raise SealValidationError("test release status must be released")
    for field in (
        "seal_hash",
        "prediction_lock_hash",
        "protocol_hash",
        "source_hash",
        "matrix_hash",
    ):
        _require_sha(payload[field], field)
    if not str(payload["evaluator_identity"]).strip():
        raise SealValidationError("evaluator identity is required")
    if payload["signer_identity"] == payload["evaluator_identity"]:
        raise SealValidationError("release authority and evaluator must be distinct")
    acl = payload["evaluator_label_acl"]
    expected_acl = [f"allow:evaluator:{payload['evaluator_identity']}"]
    if not isinstance(acl, list) or acl != expected_acl:
        raise SealValidationError("only the named evaluator ACL may receive released labels")
    try:
        authorized_at = datetime.fromisoformat(
            str(payload["authorized_at"]).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise SealValidationError("authorized_at must be an ISO timestamp") from exc
    if authorized_at.tzinfo is None or authorized_at.utcoffset() is None:
        raise SealValidationError("authorized_at must include a timezone")
