from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd
import pyarrow.parquet as pq

from .governance.immutable import write_once_json
from .methane_data import _window_protocol
from .methane_materialization import (
    MATERIALIZATION_RECEIPT,
    METHANE_POOLS,
    OUTPUT_PATHS,
    load_verified_methane_window_store,
)
from .provenance import sha256_file
from .workflow_common import WorkflowExecutionError, load_json


REVIEW_PATH = "evidence/data/E059_independent_reconciliation.json"


def _read(path: Path, columns: Iterable[str]) -> pd.DataFrame:
    return pd.read_parquet(path, columns=list(columns))


def _schema_columns(path: Path) -> set[str]:
    return set(pq.ParquetFile(path).schema_arrow.names)


def _assert_partition(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    hash_column: str,
    label: str,
) -> None:
    columns = ["window_id", "pool", hash_column]
    left = actual.loc[:, columns].astype(str).sort_values("window_id").reset_index(drop=True)
    right = expected.loc[:, columns].astype(str).sort_values("window_id").reset_index(drop=True)
    if left["window_id"].duplicated().any() or right["window_id"].duplicated().any():
        raise WorkflowExecutionError(f"E059 {label} contains duplicate window IDs")
    if not left.equals(right):
        raise WorkflowExecutionError(f"E059 {label} does not match the frozen manifest")


def review_methane_materialization(project_root: Path) -> Dict[str, Any]:
    root = project_root.resolve()
    verified = load_verified_methane_window_store(root)
    receipt = load_json(root / MATERIALIZATION_RECEIPT)
    paths = {key: root / relative for key, relative in OUTPUT_PATHS.items()}
    for key, record in receipt["outputs"].items():
        rows = int(pq.ParquetFile(paths[key]).metadata.num_rows)
        if rows != int(record["row_count"]) or rows <= 0:
            raise WorkflowExecutionError(f"E059 {key} row count drifted")

    manifest = _read(
        paths["manifest"],
        (
            "window_id",
            "pool",
            "sensor_group_id",
            "history_start",
            "history_end",
            "forecast_end",
            "feature_manifest_hash",
            "label_manifest_hash",
        ),
    )
    if manifest["window_id"].duplicated().any():
        raise WorkflowExecutionError("E059 manifest window IDs are not unique")
    if set(manifest["pool"].astype(str)) != set(METHANE_POOLS):
        raise WorkflowExecutionError("E059 manifest does not contain all eight frozen pools")
    expected_sensors = {
        str(value)
        for value in _window_protocol(root)["data"]["methane"]["target_sensor_ids"]
    }
    if set(manifest["sensor_group_id"].astype(str)) != expected_sensors:
        raise WorkflowExecutionError("E059 manifest target sensor set drifted")
    for column in ("history_start", "history_end", "forecast_end"):
        manifest[column] = pd.to_datetime(manifest[column], utc=True, errors="raise")
    if not (
        (manifest["history_start"] <= manifest["history_end"])
        & (manifest["history_end"] < manifest["forecast_end"])
    ).all():
        raise WorkflowExecutionError("E059 manifest contains a noncausal window")

    feature_columns = _schema_columns(paths["features"])
    branch_feature_columns = _schema_columns(paths["branch_test_features"])
    truth_columns = (
        _schema_columns(paths["non_test_truth"])
        | _schema_columns(paths["branch_test_truth"])
        | _schema_columns(paths["episode_truth"])
    )
    forbidden_feature_columns = {
        "event_truth",
        "future_max",
        "label_manifest_hash",
        "raw_value",
    }
    forbidden_truth_columns = {"history_json", "feature_manifest_hash"}
    if feature_columns.intersection(forbidden_feature_columns) or branch_feature_columns.intersection(
        forbidden_feature_columns
    ):
        raise WorkflowExecutionError("E059 feature files expose truth columns")
    if truth_columns.intersection(forbidden_truth_columns):
        raise WorkflowExecutionError("E059 truth files expose feature payloads")

    all_features = _read(
        paths["features"], ("window_id", "pool", "feature_manifest_hash")
    )
    branch_features = _read(
        paths["branch_test_features"],
        ("window_id", "pool", "feature_manifest_hash"),
    )
    non_test_truth = _read(
        paths["non_test_truth"], ("window_id", "pool", "label_manifest_hash")
    )
    branch_truth = _read(
        paths["branch_test_truth"], ("window_id", "pool", "label_manifest_hash")
    )
    episode_features = _read(
        paths["episode_features"], ("window_id", "pool", "feature_manifest_hash")
    )
    episode_truth = _read(
        paths["episode_truth"], ("window_id", "pool", "label_manifest_hash")
    )
    _assert_partition(
        all_features,
        manifest,
        hash_column="feature_manifest_hash",
        label="complete feature store",
    )
    branch_expected = manifest.loc[manifest["pool"].astype(str) == "D_b_te"]
    _assert_partition(
        branch_features,
        branch_expected,
        hash_column="feature_manifest_hash",
        label="branch test features",
    )
    _assert_partition(
        branch_truth,
        branch_expected,
        hash_column="label_manifest_hash",
        label="sealed branch truth",
    )
    non_test_expected = manifest.loc[
        manifest["pool"].astype(str).isin({"D_b_tr", "D_b_sel", "D_b_prob"})
    ]
    _assert_partition(
        non_test_truth,
        non_test_expected,
        hash_column="label_manifest_hash",
        label="non-test truth",
    )
    episode_expected = manifest.loc[manifest["pool"].astype(str).str.startswith("D_e_")]
    _assert_partition(
        episode_features,
        episode_expected,
        hash_column="feature_manifest_hash",
        label="episode features",
    )
    _assert_partition(
        episode_truth,
        episode_expected,
        hash_column="label_manifest_hash",
        label="episode truth",
    )
    boundary = receipt.get("execution_boundary", {})
    if boundary != {
        "model_fit_count": 0,
        "threshold_selection_count": 0,
        "test_metric_count": 0,
        "raw_dataset_synced_locally": False,
        "performance_claims_authorized": False,
    }:
        raise WorkflowExecutionError("E059 execution boundary is not strictly no-fit")

    review = {
        "schema_version": 1,
        "step_id": "E059",
        "status": "pass",
        "materialization_receipt_sha256": verified["receipt_sha256"],
        "window_count": len(manifest),
        "pool_window_counts": {
            pool: int((manifest["pool"].astype(str) == pool).sum()) for pool in METHANE_POOLS
        },
        "target_sensors": sorted(expected_sensors),
        "checks": {
            "source_and_output_hashes_match": True,
            "parquet_row_counts_match_receipt": True,
            "all_eight_pools_present": True,
            "all_target_sensors_present": True,
            "window_ids_unique": True,
            "windows_causal": True,
            "feature_truth_partitions_match_manifest": True,
            "feature_truth_payloads_physically_separate": True,
            "model_fit_count_zero": True,
            "threshold_selection_count_zero": True,
            "test_metric_count_zero": True,
        },
        "output_sha256": verified["hashes"],
        "performance_claims_authorized": False,
    }
    output_path = root / REVIEW_PATH
    write_once_json(output_path, review)
    return {
        "status": "pass",
        "output_path": REVIEW_PATH,
        "output_sha256": sha256_file(output_path),
        "window_count": len(manifest),
    }
