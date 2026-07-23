from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence, Union

import pandas as pd
import pyarrow.parquet as pq

from ..provenance import sha256_file
from .immutable import (
    GovernanceContractError,
    WriteOnceResult,
    require_sha256,
    require_timestamp,
    write_once_json,
)


PathLike = Union[str, Path]
FORBIDDEN_PREDICTION_COLUMNS = {
    "accuracy",
    "auroc",
    "event_truth",
    "f1",
    "state_truth",
    "label",
    "loss",
    "map50",
    "map50_95",
    "metric",
    "precision",
    "recall",
    "target",
    "test_metric",
    "test_label",
    "truth",
    "ground_truth",
    "future_value",
    "forecast_label",
}


def _read_prediction_columns(path: Path) -> tuple[str, ...]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return tuple(str(value) for value in pq.read_schema(path).names)
    if suffix == ".csv":
        return tuple(str(value) for value in pd.read_csv(path, nrows=0).columns)
    if suffix in {".json", ".jsonl"}:
        frame = pd.read_json(path, lines=suffix == ".jsonl")
        return tuple(str(value) for value in frame.columns)
    raise GovernanceContractError(f"unsupported prediction format: {path.suffix}")


def assert_truth_free_prediction(path: PathLike) -> tuple[str, ...]:
    prediction_path = Path(path).resolve()
    if not prediction_path.is_file():
        raise GovernanceContractError(f"prediction artifact is missing: {prediction_path}")
    columns = _read_prediction_columns(prediction_path)
    forbidden = sorted(
        column
        for column in columns
        if column.lower() in FORBIDDEN_PREDICTION_COLUMNS
        or column.lower().endswith(("_truth", "_label", "_target", "_metric"))
        or column.lower().startswith(("truth_", "label_", "target_", "test_metric_"))
    )
    if forbidden:
        raise GovernanceContractError(
            f"prediction artifact contains truth fields: {forbidden}"
        )
    return columns


def build_prediction_lock(
    *,
    scope: str,
    seal_hash: str,
    protocol_hash: str,
    source_hash: str,
    matrix_hash: str,
    required_prediction_families: Sequence[str],
    prediction_paths: Mapping[str, PathLike],
    model_hashes: Mapping[str, str],
    calibrator_and_policy_hashes: Mapping[str, str],
) -> dict[str, Any]:
    if scope not in {"branch", "episode"}:
        raise GovernanceContractError("prediction lock scope must be branch or episode")
    families = tuple(str(value) for value in required_prediction_families)
    if not families or len(set(families)) != len(families) or any(
        not value.strip() for value in families
    ):
        raise GovernanceContractError("required prediction families must be unique")
    expected = set(families)
    for name, mapping in {
        "prediction_paths": prediction_paths,
        "model_hashes": model_hashes,
        "calibrator_and_policy_hashes": calibrator_and_policy_hashes,
    }.items():
        if set(mapping) != expected:
            raise GovernanceContractError(f"{name} does not close every scheduled family")
    prediction_store_hashes = {}
    for family in families:
        assert_truth_free_prediction(prediction_paths[family])
        prediction_store_hashes[family] = sha256_file(prediction_paths[family])
        require_sha256(model_hashes[family], f"model_hashes.{family}")
        require_sha256(
            calibrator_and_policy_hashes[family],
            f"calibrator_and_policy_hashes.{family}",
        )
    for field, value in {
        "seal_hash": seal_hash,
        "protocol_hash": protocol_hash,
        "source_hash": source_hash,
        "matrix_hash": matrix_hash,
    }.items():
        require_sha256(value, field)
    return {
        "scope": scope,
        "seal_hash": seal_hash,
        "protocol_hash": protocol_hash,
        "source_hash": source_hash,
        "matrix_hash": matrix_hash,
        "required_prediction_families": list(families),
        "prediction_store_hashes": prediction_store_hashes,
        "model_hashes": dict(model_hashes),
        "calibrator_and_policy_hashes": dict(calibrator_and_policy_hashes),
        "closed_at": datetime.now(timezone.utc).isoformat(),
        "truth_fields_scan_pass": True,
    }


def validate_prediction_lock(payload: Mapping[str, Any]) -> None:
    required = {
        "scope",
        "seal_hash",
        "protocol_hash",
        "source_hash",
        "matrix_hash",
        "required_prediction_families",
        "prediction_store_hashes",
        "model_hashes",
        "calibrator_and_policy_hashes",
        "closed_at",
        "truth_fields_scan_pass",
    }
    if not required.issubset(payload):
        raise GovernanceContractError("prediction lock is missing required fields")
    if payload["scope"] not in {"branch", "episode"}:
        raise GovernanceContractError("prediction lock scope is invalid")
    families = payload["required_prediction_families"]
    if (
        not isinstance(families, list)
        or not families
        or len(set(families)) != len(families)
        or any(not isinstance(value, str) or not value.strip() for value in families)
    ):
        raise GovernanceContractError("prediction lock family list is invalid")
    expected = set(families)
    for field in ("prediction_store_hashes", "model_hashes", "calibrator_and_policy_hashes"):
        mapping = payload[field]
        if not isinstance(mapping, dict) or set(mapping) != expected:
            raise GovernanceContractError(f"prediction lock {field} is incomplete")
        for family, value in mapping.items():
            require_sha256(value, f"{field}.{family}")
    for field in ("seal_hash", "protocol_hash", "source_hash", "matrix_hash"):
        require_sha256(payload[field], field)
    require_timestamp(payload["closed_at"], "closed_at")
    if payload["truth_fields_scan_pass"] is not True:
        raise GovernanceContractError("prediction lock truth-field scan did not pass")


def close_prediction_lock(path: PathLike, payload: Mapping[str, Any]) -> WriteOnceResult:
    validate_prediction_lock(payload)
    return write_once_json(path, payload)


def verify_prediction_lock(
    lock_path: PathLike, prediction_paths: Mapping[str, PathLike]
) -> dict[str, str]:
    payload = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    validate_prediction_lock(payload)
    expected = set(payload["required_prediction_families"])
    if set(prediction_paths) != expected:
        raise GovernanceContractError("prediction verification family set drifted")
    drift = {}
    for family in sorted(expected):
        assert_truth_free_prediction(prediction_paths[family])
        observed = sha256_file(prediction_paths[family])
        if observed != payload["prediction_store_hashes"][family]:
            drift[family] = observed
    if drift:
        raise GovernanceContractError(f"prediction hash drift: {sorted(drift)}")
    return {
        family: payload["prediction_store_hashes"][family]
        for family in sorted(expected)
    }
