from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Optional, Sequence, Union

from .immutable import (
    GovernanceContractError,
    WriteOnceResult,
    require_sha256,
    require_timestamp,
    write_once_json,
)


PathLike = Union[str, Path]
NEW_REMOTE_HOST = "xinxi-zhyh@211.87.115.228"


def validate_gate(payload: Mapping[str, Any]) -> None:
    required = {
        "gate_id",
        "status",
        "protocol_hash",
        "code_hash",
        "experiment_matrix_hash",
        "artifact_contract_hash",
        "required_steps",
        "input_artifacts",
        "output_artifacts",
        "checks",
        "failures",
        "waivers",
        "allowed_next_steps",
        "slurm_job_ids",
        "remote_closeout_artifact",
        "host",
        "created_at",
    }
    if not required.issubset(payload):
        raise GovernanceContractError("gate is missing required fields")
    if payload["status"] not in {"pass", "fail", "blocked"}:
        raise GovernanceContractError("gate status is invalid")
    if not str(payload["gate_id"]).strip():
        raise GovernanceContractError("gate ID is required")
    for field in (
        "protocol_hash",
        "code_hash",
        "experiment_matrix_hash",
        "artifact_contract_hash",
    ):
        require_sha256(payload[field], field)
    require_timestamp(payload["created_at"], "created_at")
    for field in (
        "required_steps",
        "input_artifacts",
        "output_artifacts",
        "checks",
        "failures",
        "waivers",
        "allowed_next_steps",
        "slurm_job_ids",
    ):
        if not isinstance(payload[field], list):
            raise GovernanceContractError(f"gate {field} must be a list")
    if (
        not payload["required_steps"]
        or len(set(payload["required_steps"])) != len(payload["required_steps"])
        or any(not str(value).strip() for value in payload["required_steps"])
    ):
        raise GovernanceContractError("gate required steps must be unique and nonempty")
    if len(set(payload["slurm_job_ids"])) != len(payload["slurm_job_ids"]):
        raise GovernanceContractError("gate Slurm job IDs must be unique")
    if any(
        re.fullmatch(r"\d+(?:_\d+)?", str(value)) is None
        for value in payload["slurm_job_ids"]
    ):
        raise GovernanceContractError("gate contains an invalid Slurm job ID")
    for collection_name in ("input_artifacts", "output_artifacts"):
        artifact_paths = []
        for artifact in payload[collection_name]:
            if not isinstance(artifact, dict) or not {"path", "sha256"}.issubset(artifact):
                raise GovernanceContractError("gate artifact digest is incomplete")
            if not str(artifact["path"]).strip():
                raise GovernanceContractError("gate artifact path is empty")
            artifact_path = str(artifact["path"])
            parsed_path = PurePosixPath(artifact_path)
            if parsed_path.is_absolute() or ".." in parsed_path.parts:
                raise GovernanceContractError("gate artifact path must stay project-relative")
            artifact_paths.append(artifact_path)
            require_sha256(artifact["sha256"], f"{collection_name}.sha256")
        if len(set(artifact_paths)) != len(artifact_paths):
            raise GovernanceContractError("gate artifact paths must be unique")
    check_ids = []
    for check in payload["checks"]:
        if (
            not isinstance(check, dict)
            or not str(check.get("id", "")).strip()
            or check.get("status") not in {"pass", "fail", "blocked"}
        ):
            raise GovernanceContractError("gate check is malformed")
        check_ids.append(str(check["id"]))
    if len(set(check_ids)) != len(check_ids):
        raise GovernanceContractError("gate check IDs must be unique")
    failed_checks = [
        check
        for check in payload["checks"]
        if not isinstance(check, dict) or check.get("status") != "pass"
    ]
    passed_check_ids = {
        str(check.get("id"))
        for check in payload["checks"]
        if isinstance(check, dict) and check.get("status") == "pass"
    }
    missing_step_checks = sorted(set(payload["required_steps"]).difference(check_ids))
    if missing_step_checks:
        raise GovernanceContractError(
            f"gate lacks checks for required steps: {missing_step_checks}"
        )
    if payload["status"] == "pass":
        unpassed_steps = sorted(set(payload["required_steps"]).difference(passed_check_ids))
        if unpassed_steps:
            raise GovernanceContractError(
                f"gate lacks passing checks for required steps: {unpassed_steps}"
            )
        if failed_checks or payload["failures"] or payload["waivers"]:
            raise GovernanceContractError("passing gate cannot contain failures or waivers")
        if not payload["allowed_next_steps"]:
            raise GovernanceContractError("passing gate must authorize a next step")
    if payload["host"] is not None:
        if payload["host"] != NEW_REMOTE_HOST:
            raise GovernanceContractError("gate references a forbidden remote host")
        if payload["status"] == "pass" and not str(
            payload["remote_closeout_artifact"] or ""
        ).strip():
            raise GovernanceContractError("remote passing gate requires closeout evidence")
        if payload["status"] == "pass" and "remote_closeout" not in passed_check_ids:
            raise GovernanceContractError(
                "remote passing gate requires a validated closeout check"
            )
        closeout_path = str(payload["remote_closeout_artifact"] or "")
        bound_artifact_paths = {
            str(artifact["path"])
            for field in ("input_artifacts", "output_artifacts")
            for artifact in payload[field]
        }
        if payload["status"] == "pass" and closeout_path not in bound_artifact_paths:
            raise GovernanceContractError(
                "remote passing gate does not hash-bind its closeout artifact"
            )
    elif payload["remote_closeout_artifact"] is not None:
        raise GovernanceContractError("local gate cannot claim a remote closeout")
    elif payload["slurm_job_ids"]:
        raise GovernanceContractError("local gate cannot list Slurm job IDs")


def build_gate(
    *,
    gate_id: str,
    status: str,
    protocol_hash: str,
    code_hash: str,
    experiment_matrix_hash: str,
    artifact_contract_hash: str,
    required_steps: Sequence[str],
    input_artifacts: Sequence[Mapping[str, Any]],
    output_artifacts: Sequence[Mapping[str, Any]],
    checks: Sequence[Mapping[str, Any]],
    failures: Sequence[str],
    waivers: Sequence[str],
    allowed_next_steps: Sequence[str],
    slurm_job_ids: Sequence[str] = (),
    remote_closeout_artifact: Optional[str] = None,
    host: Optional[str] = None,
) -> dict[str, Any]:
    payload = {
        "gate_id": gate_id,
        "status": status,
        "protocol_hash": protocol_hash,
        "code_hash": code_hash,
        "experiment_matrix_hash": experiment_matrix_hash,
        "artifact_contract_hash": artifact_contract_hash,
        "required_steps": list(required_steps),
        "input_artifacts": [dict(value) for value in input_artifacts],
        "output_artifacts": [dict(value) for value in output_artifacts],
        "checks": [dict(value) for value in checks],
        "failures": list(failures),
        "waivers": list(waivers),
        "allowed_next_steps": list(allowed_next_steps),
        "slurm_job_ids": list(slurm_job_ids),
        "remote_closeout_artifact": remote_closeout_artifact,
        "host": host,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    validate_gate(payload)
    return payload


def write_gate(path: PathLike, payload: Mapping[str, Any]) -> WriteOnceResult:
    validate_gate(payload)
    return write_once_json(path, payload)
