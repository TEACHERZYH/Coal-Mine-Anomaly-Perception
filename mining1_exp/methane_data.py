from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd
import yaml

from .data.adapters import adapter_contract_from_entry
from .data.manifests import read_file_manifest, read_split_manifest
from .provenance import canonical_json_sha256, sha256_file
from .workflow_common import (
    WorkflowExecutionError,
    load_json,
    write_json_artifact,
    write_parquet_artifact,
)


METHANE_CONCEPT_ID = "methane_future_risk"
SAFE_FEATURE = re.compile(r"[^A-Za-z0-9_]+")


def _window_protocol(project_root: Path) -> Dict[str, Any]:
    frozen = project_root / "configs/protocol_lock.pretest.yaml"
    path = frozen if frozen.is_file() else project_root / "configs/protocol_lock.template.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise WorkflowExecutionError("Methane window protocol is unavailable")
    methane = payload.get("data", {}).get("methane", {})
    required = {
        "target_sensor_ids",
        "history_seconds",
        "horizon_seconds",
        "stride_seconds",
        "risk_concentration_threshold",
    }
    if not required.issubset(methane) or "TBD" in json.dumps(
        {key: methane[key] for key in required}, sort_keys=True
    ):
        raise WorkflowExecutionError("Methane window-defining fields are not frozen")
    return payload


def _methane_source_contract(project_root: Path) -> tuple[str, Dict[str, Any]]:
    decision = load_json(project_root / "evidence/data/dataset_source_decision.json")
    entry = decision.get("roles", {}).get("methane_dataset")
    if not isinstance(entry, Mapping):
        raise WorkflowExecutionError("Methane dataset role is unavailable")
    dataset_id = str(entry.get("dataset_id", "")).strip()
    contract = adapter_contract_from_entry(entry)
    if not dataset_id or contract["kind"] != "methane_csv":
        raise WorkflowExecutionError("Methane role does not use the reviewed CSV adapter")
    return dataset_id, contract


def _feature_name(value: str) -> str:
    normalized = SAFE_FEATURE.sub("_", value.strip()).strip("_").lower()
    if not normalized:
        raise WorkflowExecutionError(f"Methane feature name is invalid: {value!r}")
    return normalized


def _resample_causal(
    frame: pd.DataFrame,
    *,
    timestamp_column: str,
    source_columns: list[str],
    stride_seconds: int,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    ordered = frame.sort_values(timestamp_column).copy()
    timestamps = pd.to_datetime(ordered[timestamp_column], utc=True, errors="raise")
    epoch = timestamps.astype("int64").to_numpy(dtype=np.int64) // 1_000_000_000
    if len(epoch) < 2 or np.any(np.diff(epoch) <= 0):
        raise WorkflowExecutionError("Methane block timestamps must be strictly increasing")
    start = int(math.ceil(epoch[0] / stride_seconds) * stride_seconds)
    stop = int(math.floor(epoch[-1] / stride_seconds) * stride_seconds)
    grid = np.arange(start, stop + 1, stride_seconds, dtype=np.int64)
    if grid.size == 0:
        raise WorkflowExecutionError("Methane block has no complete stride endpoint")
    values = ordered[source_columns].apply(pd.to_numeric, errors="raise").to_numpy(
        dtype=np.float64
    )
    sampled = np.full((len(grid), len(source_columns)), np.nan, dtype=np.float64)
    source_indices = np.searchsorted(epoch, grid, side="right") - 1
    valid = source_indices >= 0
    valid_indices = source_indices[valid]
    valid[valid] = epoch[valid_indices] > grid[valid] - stride_seconds
    sampled[valid] = values[source_indices[valid]]
    masks = np.isfinite(sampled).astype(np.float64)
    feature_names = [_feature_name(column) for column in source_columns]
    model_values = np.concatenate((sampled, masks), axis=1)
    model_names = [*feature_names, *(f"{name}_observed_mask" for name in feature_names)]
    return grid, model_values, model_names, epoch


def _window_rows_for_block(
    frame: pd.DataFrame,
    *,
    dataset_id: str,
    raw_group_id: str,
    pool: str,
    sensor_group_id: str,
    timestamp_column: str,
    value_column: str,
    feature_columns: list[str],
    history_seconds: int,
    horizon_seconds: int,
    stride_seconds: int,
    risk_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    source_columns = [value_column, *feature_columns]
    grid, model_values, model_names, raw_epoch = _resample_causal(
        frame,
        timestamp_column=timestamp_column,
        source_columns=source_columns,
        stride_seconds=stride_seconds,
    )
    raw_values = pd.to_numeric(frame[value_column], errors="raise").to_numpy(dtype=np.float64)
    history_steps = history_seconds // stride_seconds
    if history_steps <= 0 or history_steps * stride_seconds != history_seconds:
        raise WorkflowExecutionError("Methane history must be divisible by the stride")
    feature_rows: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for endpoint_index in range(history_steps - 1, len(grid)):
        history_end = int(grid[endpoint_index])
        forecast_end = history_end + horizon_seconds
        future_start = int(np.searchsorted(raw_epoch, history_end, side="right"))
        future_stop = int(np.searchsorted(raw_epoch, forecast_end, side="right"))
        future = raw_values[future_start:future_stop]
        if future.size == 0:
            continue
        history = model_values[endpoint_index - history_steps + 1 : endpoint_index + 1]
        if not np.isfinite(history[:, len(source_columns) :]).all():
            raise WorkflowExecutionError("Methane missingness masks are non-finite")
        if not history[:, : len(source_columns)].shape[0] == history_steps:
            raise WorkflowExecutionError("Methane history window is incomplete")
        window_id = hashlib.sha256(
            f"{dataset_id}|{raw_group_id}|{sensor_group_id}|{history_end}".encode("utf-8")
        ).hexdigest()
        history_payload = {
            "feature_names": model_names,
            "values": [
                [None if not np.isfinite(value) else float(value) for value in row]
                for row in history
            ],
        }
        feature_hash = canonical_json_sha256(history_payload)
        event_truth = int(bool(np.any(future >= risk_threshold)))
        label_payload = {
            "concept_id": METHANE_CONCEPT_ID,
            "event_truth": event_truth,
            "future_max": float(np.max(future)),
            "forecast_end": forecast_end,
        }
        label_hash = canonical_json_sha256(label_payload)
        common = {
            "dataset_id": dataset_id,
            "window_id": window_id,
            "raw_group_id": raw_group_id,
            "sensor_group_id": sensor_group_id,
            "pool": pool,
        }
        feature_rows.append(
            {
                **common,
                "history_start": pd.Timestamp(
                    history_end - history_seconds + stride_seconds, unit="s", tz="UTC"
                ),
                "history_end": pd.Timestamp(history_end, unit="s", tz="UTC"),
                "forecast_end": pd.Timestamp(forecast_end, unit="s", tz="UTC"),
                "target_sensor_variant": "all_sensor",
                "history_json": json.dumps(history_payload, sort_keys=True),
                "feature_manifest_hash": feature_hash,
            }
        )
        truth_rows.append(
            {
                **common,
                "concept_id": METHANE_CONCEPT_ID,
                "timestamp_seconds": history_end,
                "raw_value": float(
                    raw_values[int(np.searchsorted(raw_epoch, history_end, side="right")) - 1]
                ),
                "history_end_epoch_seconds": history_end,
                "forecast_end_epoch_seconds": forecast_end,
                "event_truth": event_truth,
                "future_max": float(np.max(future)),
                "label_manifest_hash": label_hash,
            }
        )
        manifest_rows.append(
            {
                **common,
                "history_start": pd.Timestamp(
                    history_end - history_seconds + stride_seconds, unit="s", tz="UTC"
                ),
                "history_end": pd.Timestamp(history_end, unit="s", tz="UTC"),
                "forecast_end": pd.Timestamp(forecast_end, unit="s", tz="UTC"),
                "target_sensor_variant": "all_sensor",
                "feature_manifest_hash": feature_hash,
                "label_manifest_hash": label_hash,
            }
        )
    return feature_rows, truth_rows, manifest_rows


def build_methane_window_store(
    project_root: Path,
    *,
    methane_role_lock: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    protocol = _window_protocol(project_root)
    dataset_id, contract = _methane_source_contract(project_root)
    section = contract["methane"]
    methane_lock_path = project_root / "data/locked/methane_role_lock.json"
    existing_lock = load_json(methane_lock_path) if methane_lock_path.is_file() else None
    if methane_role_lock is not None:
        methane_lock = dict(methane_role_lock)
        if existing_lock is not None and existing_lock != methane_lock:
            raise WorkflowExecutionError("Existing methane role lock differs from the candidate")
    elif existing_lock is not None:
        methane_lock = existing_lock
    else:
        raise WorkflowExecutionError("Methane role lock candidate is unavailable")
    if methane_lock.get("dataset_id") != dataset_id or methane_lock.get("status") != "pass":
        raise WorkflowExecutionError("Methane role lock does not match the selected dataset")
    file_path = project_root / "data/locked/file_manifest.parquet"
    split_path = project_root / "data/locked/split_manifest.parquet"
    file_manifest = read_file_manifest(file_path)
    split = read_split_manifest(split_path)
    rows = file_manifest.loc[
        (file_manifest["dataset_id"].astype(str) == dataset_id)
        & (file_manifest["modality"].astype(str) == "methane")
    ].merge(
        split[["dataset_id", "record_id", "pool"]],
        on=["dataset_id", "record_id"],
        validate="one_to_one",
    )
    if rows.empty:
        raise WorkflowExecutionError("Selected methane dataset has no canonical records")
    data_contract = protocol["data"]["methane"]
    target_sensors = {str(value) for value in data_contract["target_sensor_ids"]}
    all_features: list[dict[str, Any]] = []
    all_truth: list[dict[str, Any]] = []
    all_manifest: list[dict[str, Any]] = []
    observed_sensors: set[str] = set()
    for item in rows.sort_values(["raw_group_id", "record_id"]).itertuples(index=False):
        source = project_root / str(item.relative_path)
        frame = pd.read_parquet(source)
        sensor_values = frame[str(section["sensor_group_column"])].astype(str).unique()
        if len(sensor_values) != 1:
            raise WorkflowExecutionError(f"Methane block mixes sensor groups: {source}")
        sensor = str(sensor_values[0])
        if sensor not in target_sensors:
            continue
        observed_sensors.add(sensor)
        feature_rows, truth_rows, manifest_rows = _window_rows_for_block(
            frame,
            dataset_id=dataset_id,
            raw_group_id=str(item.raw_group_id),
            pool=str(item.pool),
            sensor_group_id=sensor,
            timestamp_column="__timestamp_utc",
            value_column=str(section["value_column"]),
            feature_columns=[str(value) for value in section["feature_columns"]],
            history_seconds=int(data_contract["history_seconds"]),
            horizon_seconds=int(data_contract["horizon_seconds"]),
            stride_seconds=int(data_contract["stride_seconds"]),
            risk_threshold=float(data_contract["risk_concentration_threshold"]),
        )
        all_features.extend(feature_rows)
        all_truth.extend(truth_rows)
        all_manifest.extend(manifest_rows)
    if observed_sensors != target_sensors:
        missing = sorted(target_sensors - observed_sensors)
        raise WorkflowExecutionError(f"Locked target methane sensors are absent: {missing}")
    if not all_manifest:
        raise WorkflowExecutionError("Methane window construction produced no complete windows")
    features = pd.DataFrame.from_records(all_features).sort_values("window_id").reset_index(drop=True)
    truth = pd.DataFrame.from_records(all_truth).sort_values("window_id").reset_index(drop=True)
    manifest = pd.DataFrame.from_records(all_manifest).sort_values("window_id").reset_index(drop=True)
    if manifest["window_id"].duplicated().any():
        raise WorkflowExecutionError("Methane window IDs are not unique")
    allowed_pools = {
        "D_b_tr", "D_b_sel", "D_b_prob", "D_b_te",
        "D_e_tr", "D_e_sel", "D_e_pol", "D_e_te",
    }
    if not set(manifest["pool"]).issubset(allowed_pools):
        raise WorkflowExecutionError("Methane windows escaped the frozen branch and episode pools")
    branch_test_features = features.loc[features["pool"].astype(str) == "D_b_te"].copy()
    non_test_truth = truth.loc[
        truth["pool"].astype(str).isin({"D_b_tr", "D_b_sel", "D_b_prob"})
    ].copy()
    branch_test_truth = truth.loc[truth["pool"].astype(str) == "D_b_te"].copy()
    if branch_test_features.empty or non_test_truth.empty or branch_test_truth.empty:
        raise WorkflowExecutionError("Methane window store lacks non-test or sealed test partitions")
    feature_path = project_root / "data/locked/methane_window_features.parquet"
    branch_feature_path = project_root / "data/locked/branch_methane_test_features.parquet"
    non_test_truth_path = project_root / "data/locked/methane_non_test_truth.parquet"
    branch_truth_path = project_root / "data/sealed/branch_methane_truth.parquet"
    episode_feature_path = project_root / "data/locked/episode_methane_features.parquet"
    episode_truth_path = project_root / "data/locked/episode_methane_truth.parquet"
    manifest_path = project_root / "data/locked/methane_window_manifest.parquet"
    write_parquet_artifact(feature_path, features)
    write_parquet_artifact(branch_feature_path, branch_test_features)
    write_parquet_artifact(non_test_truth_path, non_test_truth)
    write_parquet_artifact(branch_truth_path, branch_test_truth)
    episode_features = features.loc[features["pool"].astype(str).str.startswith("D_e_")].copy()
    episode_truth = truth.loc[truth["pool"].astype(str).str.startswith("D_e_")].copy()
    if not episode_features.empty or not episode_truth.empty:
        if episode_features.empty or episode_truth.empty:
            raise WorkflowExecutionError("Methane episode features and truth are not aligned")
        write_parquet_artifact(episode_feature_path, episode_features)
        write_parquet_artifact(episode_truth_path, episode_truth)
    write_parquet_artifact(manifest_path, manifest)
    paths = {
        "features": feature_path.relative_to(project_root).as_posix(),
        "branch_test_features": branch_feature_path.relative_to(project_root).as_posix(),
        "non_test_truth": non_test_truth_path.relative_to(project_root).as_posix(),
        "branch_test_truth": branch_truth_path.relative_to(project_root).as_posix(),
        "manifest": manifest_path.relative_to(project_root).as_posix(),
    }
    hashes = {
        "features": sha256_file(feature_path),
        "branch_test_features": sha256_file(branch_feature_path),
        "non_test_truth": sha256_file(non_test_truth_path),
        "branch_test_truth": sha256_file(branch_truth_path),
        "manifest": sha256_file(manifest_path),
        "file_manifest": sha256_file(file_path),
        "split_manifest": sha256_file(split_path),
    }
    if episode_feature_path.is_file():
        paths["episode_features"] = episode_feature_path.relative_to(project_root).as_posix()
        paths["episode_truth"] = episode_truth_path.relative_to(project_root).as_posix()
        hashes["episode_features"] = sha256_file(episode_feature_path)
        hashes["episode_truth"] = sha256_file(episode_truth_path)
    if existing_lock is None:
        write_json_artifact(methane_lock_path, methane_lock)
    hashes["role_lock"] = sha256_file(methane_lock_path)
    return {
        "status": "pass",
        "dataset_id": dataset_id,
        "window_count": len(manifest),
        "sensor_count": len(observed_sensors),
        "feature_names": json.loads(features.iloc[0]["history_json"])["feature_names"],
        "paths": paths,
        "hashes": hashes,
    }
