from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence, Union

import pandas as pd
import yaml

from ..provenance import canonical_json_sha256, sha256_file
from .immutable import (
    GovernanceContractError,
    WriteOnceResult,
    require_sha256,
    require_timestamp,
    write_once_bytes,
    write_once_json,
)
from .prediction_lock import assert_truth_free_prediction


PathLike = Union[str, Path]
REQUIRED_RUN_FILES = {
    "run_manifest.json",
    "config_resolved.yaml",
    "environment.json",
    "data_hashes.json",
    "metrics_per_epoch.csv",
    "execution.log",
    "last_checkpoint",
    "artifact_sha256.txt",
}
RUN_ID_LOCK_FIELDS = (
    "family_id",
    "protocol_hash",
    "artifact_contract_hash",
    "code_hash",
    "environment_ready_sha256",
    "package_inventory_sha256",
    "command_receipt_sha256",
    "parent_checkpoint_hashes",
    "data_and_split_hashes",
    "seeds",
    "host_and_slurm_job_id",
)


def canonical_run_id(manifest: Mapping[str, Any]) -> str:
    missing = [field for field in RUN_ID_LOCK_FIELDS if field not in manifest]
    if missing:
        raise GovernanceContractError(f"run ID inputs are missing: {missing}")
    return canonical_json_sha256({field: manifest[field] for field in RUN_ID_LOCK_FIELDS})


def validate_run_manifest(manifest: Mapping[str, Any]) -> None:
    required = set(RUN_ID_LOCK_FIELDS).union(
        {
            "run_id",
            "started_at_and_finished_at",
            "exit_code",
            "status",
            "remote_closeout_reference",
        }
    )
    if not required.issubset(manifest):
        raise GovernanceContractError("run manifest is missing required fields")
    if not str(manifest["family_id"]).strip():
        raise GovernanceContractError("run family ID is required")
    for field in (
        "protocol_hash",
        "artifact_contract_hash",
        "code_hash",
        "environment_ready_sha256",
        "package_inventory_sha256",
        "command_receipt_sha256",
    ):
        require_sha256(manifest[field], field)
    data_hashes = manifest["data_and_split_hashes"]
    if not isinstance(data_hashes, dict) or not data_hashes:
        raise GovernanceContractError("run data and split hashes are required")
    for name, value in data_hashes.items():
        if not str(name).strip():
            raise GovernanceContractError("run data hash name is empty")
        require_sha256(value, f"data_and_split_hashes.{name}")
    parent_hashes = manifest["parent_checkpoint_hashes"]
    if not isinstance(parent_hashes, dict):
        raise GovernanceContractError("parent checkpoint hashes must be a mapping")
    for name, value in parent_hashes.items():
        if not str(name).strip():
            raise GovernanceContractError("parent checkpoint role is empty")
        require_sha256(value, f"parent_checkpoint_hashes.{name}")
    if not isinstance(manifest["seeds"], dict) or not manifest["seeds"]:
        raise GovernanceContractError("run seeds are required")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in manifest["seeds"].values()
    ):
        raise GovernanceContractError("run seeds must be positive integers")
    host_job = manifest["host_and_slurm_job_id"]
    host_fields = {"host", "slurm_job_id", "allocation_id", "compute_node"}
    if not isinstance(host_job, dict) or not host_fields.issubset(host_job):
        raise GovernanceContractError("run host and Slurm job ID are required")
    if host_job["host"] != "xinxi-zhyh@211.87.115.228":
        raise GovernanceContractError("run host or Slurm job ID is invalid")
    for field in ("slurm_job_id", "allocation_id"):
        if re.fullmatch(r"\d+(?:_\d+)?", str(host_job[field])) is None:
            raise GovernanceContractError(f"run {field} is invalid")
    compute_node = str(host_job["compute_node"]).strip()
    if not compute_node or compute_node.lower() in {"mu01", "login", "login-node"}:
        raise GovernanceContractError("run compute node is missing or is a login node")
    times = manifest["started_at_and_finished_at"]
    if not isinstance(times, dict) or not {"started_at", "finished_at"}.issubset(times):
        raise GovernanceContractError("run timestamps are required")
    require_timestamp(times["started_at"], "started_at")
    require_timestamp(times["finished_at"], "finished_at")
    started = datetime.fromisoformat(str(times["started_at"]).replace("Z", "+00:00"))
    finished = datetime.fromisoformat(str(times["finished_at"]).replace("Z", "+00:00"))
    if finished < started:
        raise GovernanceContractError("run finish time predates its start")
    if manifest["status"] not in {"pass", "fail"}:
        raise GovernanceContractError("final run status must be pass or fail")
    if not isinstance(manifest["exit_code"], int) or isinstance(
        manifest["exit_code"], bool
    ):
        raise GovernanceContractError("run exit code must be an integer")
    if manifest["status"] == "pass" and manifest["exit_code"] != 0:
        raise GovernanceContractError("passing run must have exit code zero")
    if manifest["status"] == "fail" and manifest["exit_code"] == 0:
        raise GovernanceContractError("failed run must have a nonzero exit code")
    closeout = manifest["remote_closeout_reference"]
    closeout_fields = {"path", "sha256", "remote_action_id"}
    if not isinstance(closeout, dict) or not closeout_fields.issubset(closeout):
        raise GovernanceContractError("remote run requires a closeout reference mapping")
    require_sha256(closeout["sha256"], "remote_closeout_reference.sha256")
    closeout_path = PurePosixPath(str(closeout["path"]))
    if (
        closeout_path.is_absolute()
        or ".." in closeout_path.parts
        or str(closeout_path) in {"", "."}
        or not str(closeout["remote_action_id"]).strip()
    ):
        raise GovernanceContractError("remote closeout reference is invalid")
    expected_run_id = canonical_run_id(manifest)
    if manifest["run_id"] != expected_run_id:
        raise GovernanceContractError("run ID is not the canonical locked-input hash")


def write_run_bundle_file(
    bundle_root: PathLike, relative_path: str, data: bytes
) -> WriteOnceResult:
    root = Path(bundle_root).resolve()
    target = (root / relative_path).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise GovernanceContractError("run bundle path escapes its root") from exc
    return write_once_bytes(target, data)


def write_run_manifest(
    bundle_root: PathLike, manifest: Mapping[str, Any]
) -> WriteOnceResult:
    validate_run_manifest(manifest)
    return write_once_json(Path(bundle_root) / "run_manifest.json", manifest)


def _parse_artifact_hashes(path: Path, root: Path) -> dict[str, str]:
    hashes = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        parts = raw_line.split(None, 1)
        if len(parts) != 2:
            raise GovernanceContractError(f"invalid artifact hash line {line_number}")
        digest, relative = parts[0], parts[1].strip().lstrip("*")
        require_sha256(digest, f"artifact_sha256 line {line_number}")
        relative_path = PurePosixPath(relative)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or "\\" in relative
            or relative in {"", "."}
        ):
            raise GovernanceContractError("artifact hash manifest path is invalid")
        artifact = (root / relative).resolve()
        try:
            artifact.relative_to(root)
        except ValueError as exc:
            raise GovernanceContractError("artifact hash path escapes run root") from exc
        if relative == "artifact_sha256.txt" or not artifact.is_file():
            raise GovernanceContractError("artifact hash manifest references an invalid file")
        if relative in hashes:
            raise GovernanceContractError("artifact hash manifest contains duplicate paths")
        hashes[relative] = digest
    return hashes


def validate_run_bundle(
    bundle_root: PathLike,
    *,
    selected_trainable_run: bool,
    probability_calibration_applies: bool,
    scheduled_prediction_family: bool,
    label_release_authorized: bool,
    expected_selected_remote_python: str,
    expected_environment_ready_sha256: str,
    expected_environment_fingerprint_sha256: str,
    expected_package_inventory_sha256: str,
) -> dict[str, Any]:
    root = Path(bundle_root).resolve()
    if not root.is_dir():
        raise GovernanceContractError("run bundle directory is missing")
    required = set(REQUIRED_RUN_FILES)
    if selected_trainable_run:
        required.add("best_checkpoint")
    if probability_calibration_applies:
        required.add("calibrator.json")
    if scheduled_prediction_family:
        required.add("predictions.parquet")
    missing = sorted(name for name in required if not (root / name).is_file())
    if missing:
        raise GovernanceContractError(f"run bundle files are missing: {missing}")
    if (root / "metrics.json").exists() and not label_release_authorized:
        raise GovernanceContractError("metrics cannot exist before matching label release")

    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    validate_run_manifest(manifest)
    require_sha256(expected_environment_ready_sha256, "expected_environment_ready_sha256")
    require_sha256(expected_package_inventory_sha256, "expected_package_inventory_sha256")
    if manifest["environment_ready_sha256"] != expected_environment_ready_sha256:
        raise GovernanceContractError("run environment hash does not match remote-ready lock")
    if manifest["package_inventory_sha256"] != expected_package_inventory_sha256:
        raise GovernanceContractError("run package inventory hash does not match remote-ready lock")
    failure_path = root / "failure.json"
    if manifest["status"] == "pass" and failure_path.exists():
        raise GovernanceContractError("passing run cannot contain failure.json")
    if manifest["status"] == "fail" and not failure_path.is_file():
        raise GovernanceContractError("failed run requires failure.json")
    if scheduled_prediction_family:
        assert_truth_free_prediction(root / "predictions.parquet")
    resolved_config = yaml.safe_load(
        (root / "config_resolved.yaml").read_text(encoding="utf-8")
    )
    if not isinstance(resolved_config, dict):
        raise GovernanceContractError("resolved run config must be a YAML mapping")
    if resolved_config.get("family_id") != manifest["family_id"]:
        raise GovernanceContractError("resolved run config family drifted from manifest")
    environment = json.loads((root / "environment.json").read_text(encoding="utf-8"))
    environment_required = {
        "selected_remote_python",
        "python_version",
        "framework_version",
        "cuda_runtime",
        "environment_ready_sha256",
        "package_inventory_sha256",
        "environment_fingerprint_sha256",
        "slurm_job_id",
        "allocation_id",
        "compute_node",
    }
    if not isinstance(environment, dict) or not environment_required.issubset(environment):
        raise GovernanceContractError("environment.json is missing locked runtime fields")
    for field in (
        "environment_ready_sha256",
        "package_inventory_sha256",
        "environment_fingerprint_sha256",
    ):
        require_sha256(environment[field], f"environment.{field}")
    require_sha256(
        expected_environment_fingerprint_sha256,
        "expected_environment_fingerprint_sha256",
    )
    if environment["environment_ready_sha256"] != expected_environment_ready_sha256:
        raise GovernanceContractError("environment snapshot ready hash drifted")
    if environment["package_inventory_sha256"] != expected_package_inventory_sha256:
        raise GovernanceContractError("environment snapshot package hash drifted")
    if (
        environment["environment_fingerprint_sha256"]
        != expected_environment_fingerprint_sha256
    ):
        raise GovernanceContractError("environment snapshot fingerprint drifted")
    expected_python = str(PurePosixPath(expected_selected_remote_python))
    if (
        not PurePosixPath(expected_python).is_absolute()
        or environment["selected_remote_python"] != expected_python
    ):
        raise GovernanceContractError("environment snapshot selected Python drifted")
    if environment["slurm_job_id"] != str(
        manifest["host_and_slurm_job_id"]["slurm_job_id"]
    ):
        raise GovernanceContractError("environment snapshot Slurm job ID drifted")
    for field in ("allocation_id", "compute_node"):
        if str(environment[field]) != str(manifest["host_and_slurm_job_id"][field]):
            raise GovernanceContractError(f"environment snapshot {field} drifted")
    if not PurePosixPath(str(environment["selected_remote_python"])).is_absolute():
        raise GovernanceContractError("environment snapshot Python path is not absolute")
    if any(
        not str(environment[field]).strip()
        for field in ("python_version", "framework_version", "cuda_runtime", "compute_node")
    ):
        raise GovernanceContractError("environment snapshot contains empty runtime fields")
    data_hashes = json.loads((root / "data_hashes.json").read_text(encoding="utf-8"))
    if data_hashes != manifest["data_and_split_hashes"]:
        raise GovernanceContractError("data_hashes.json does not match the run manifest")
    metrics = pd.read_csv(root / "metrics_per_epoch.csv")
    if metrics.empty:
        if manifest["status"] == "pass":
            raise GovernanceContractError("passing run requires progress rows")
    if not (root / "execution.log").read_bytes():
        raise GovernanceContractError("execution.log must not be empty")
    if not (root / "last_checkpoint").read_bytes():
        raise GovernanceContractError("last_checkpoint pointer must not be empty")
    if manifest["status"] == "fail":
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
        failure_fields = {
            "error_type",
            "error",
            "last_safe_checkpoint",
            "retry_eligible",
            "frozen_retry_rule",
            "failed_at",
            "exit_code",
        }
        if not isinstance(failure, dict) or not failure_fields.issubset(failure):
            raise GovernanceContractError("failure.json is incomplete")
        if not str(failure["error_type"]).strip() or not str(failure["error"]).strip():
            raise GovernanceContractError("failure.json lacks an exception description")
        if not isinstance(failure["retry_eligible"], bool):
            raise GovernanceContractError("failure retry eligibility must be boolean")
        if failure["retry_eligible"] and not str(failure["frozen_retry_rule"]).strip():
            raise GovernanceContractError("retry-eligible failure lacks a frozen retry rule")
        require_timestamp(failure["failed_at"], "failure.failed_at")
        if failure["exit_code"] != manifest["exit_code"]:
            raise GovernanceContractError("failure exit code drifted from run manifest")

    recorded = _parse_artifact_hashes(root / "artifact_sha256.txt", root)
    expected_hashed_files = {
        name for name in required if name != "artifact_sha256.txt"
    }
    if (root / "metrics.json").is_file():
        expected_hashed_files.add("metrics.json")
    actual_hashed_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != root / "artifact_sha256.txt"
    }
    if not expected_hashed_files.issubset(recorded) or set(recorded) != actual_hashed_files:
        raise GovernanceContractError("artifact hash manifest does not close the run bundle")
    mismatches = [
        name
        for name, digest in recorded.items()
        if sha256_file(root / name) != digest
    ]
    if mismatches:
        raise GovernanceContractError(f"run bundle artifact hash drift: {mismatches}")
    return {
        "run_id": manifest["run_id"],
        "status": manifest["status"],
        "verified_file_count": len(recorded),
    }
