from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .data.manifests import read_file_manifest, read_split_manifest
from .governance.immutable import promote_once_file, write_once_json
from .methane_data import (
    _methane_source_contract,
    _window_protocol,
    _window_rows_for_block,
)
from .provenance import sha256_file
from .workflow_common import WorkflowExecutionError, load_json


METHANE_POOLS = (
    "D_b_tr",
    "D_b_sel",
    "D_b_prob",
    "D_b_te",
    "D_e_tr",
    "D_e_sel",
    "D_e_pol",
    "D_e_te",
)
OUTPUT_PATHS = {
    "features": "data/locked/methane_window_features.parquet",
    "branch_test_features": "data/locked/branch_methane_test_features.parquet",
    "non_test_truth": "data/locked/methane_non_test_truth.parquet",
    "branch_test_truth": "data/sealed/branch_methane_truth.parquet",
    "episode_features": "data/locked/episode_methane_features.parquet",
    "episode_truth": "data/locked/episode_methane_truth.parquet",
    "manifest": "data/locked/methane_window_manifest.parquet",
}
MATERIALIZATION_RECEIPT = "evidence/data/E059_methane_window_materialization.json"


class _ParquetSinks:
    def __init__(self, project_root: Path) -> None:
        self.root = project_root.resolve()
        self.targets = {
            key: (self.root / relative).resolve() for key, relative in OUTPUT_PATHS.items()
        }
        self.staged: Dict[str, Path] = {}
        self.writers: Dict[str, pq.ParquetWriter] = {}
        self.schemas: Dict[str, pa.Schema] = {}
        self.row_counts: Counter[str] = Counter()

    def write(self, key: str, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        target = self.targets[key]
        target.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(frame, preserve_index=False)
        schema = table.schema.remove_metadata()
        if key not in self.writers:
            staged = target.parent / f".{target.name}.{os.getpid()}.E059.stage"
            if staged.exists():
                raise WorkflowExecutionError(f"Stale E059 staging file exists: {staged}")
            self.staged[key] = staged
            self.schemas[key] = schema
            self.writers[key] = pq.ParquetWriter(
                staged,
                table.schema,
                compression="zstd",
                use_dictionary=True,
                write_statistics=True,
            )
        elif self.schemas[key] != schema:
            raise WorkflowExecutionError(f"E059 Parquet schema drifted for {key}")
        self.writers[key].write_table(table)
        self.row_counts[key] += len(frame)

    def close(self) -> None:
        for writer in self.writers.values():
            writer.close()
        self.writers.clear()

    def abort(self) -> None:
        for writer in self.writers.values():
            try:
                writer.close()
            except Exception:
                pass
        self.writers.clear()
        for staged in self.staged.values():
            staged.unlink(missing_ok=True)

    def promote(self) -> Dict[str, Dict[str, Any]]:
        missing = sorted(set(OUTPUT_PATHS).difference(self.staged))
        if missing:
            raise WorkflowExecutionError(f"E059 did not produce required outputs: {missing}")
        outputs: Dict[str, Dict[str, Any]] = {}
        for key in OUTPUT_PATHS:
            result = promote_once_file(self.targets[key], self.staged[key])
            outputs[key] = {
                "path": OUTPUT_PATHS[key],
                "sha256": result.artifact.sha256,
                "bytes": result.artifact.bytes,
                "row_count": int(self.row_counts[key]),
            }
        return outputs


def _verified_methane_rows(project_root: Path) -> tuple[str, Mapping[str, Any], pd.DataFrame]:
    dataset_id, contract = _methane_source_contract(project_root)
    file_manifest = read_file_manifest(project_root / "data/locked/file_manifest.parquet")
    split = read_split_manifest(project_root / "data/locked/split_manifest.parquet")
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
    return dataset_id, contract, rows.sort_values(["raw_group_id", "record_id"])


def materialize_methane_windows(
    project_root: Path,
    run_root: Path,
    config_path: Optional[Path],
) -> Dict[str, Any]:
    del run_root
    if config_path is not None:
        raise WorkflowExecutionError("E059 does not accept a mutable runtime config")
    root = project_root.resolve()
    protocol = _window_protocol(root)
    data_contract = protocol["data"]["methane"]
    target_sensors = {str(value) for value in data_contract["target_sensor_ids"]}
    dataset_id, contract, rows = _verified_methane_rows(root)
    source_contract = contract["methane"]
    role_lock_path = root / "data/locked/methane_role_lock.json"
    role_lock = load_json(role_lock_path)
    if role_lock.get("status") != "pass" or role_lock.get("dataset_id") != dataset_id:
        raise WorkflowExecutionError("E059 methane role lock does not match the selected dataset")

    sinks = _ParquetSinks(root)
    pool_counts: Counter[str] = Counter()
    sensor_counts: Counter[str] = Counter()
    seen_window_ids: set[str] = set()
    feature_names: Optional[list[str]] = None
    canonical_bytes = 0
    try:
        for item in rows.itertuples(index=False):
            source = root / str(item.relative_path)
            if (
                not source.is_file()
                or source.stat().st_size != int(item.byte_size)
                or sha256_file(source) != str(item.sha256)
            ):
                raise WorkflowExecutionError(f"Canonical methane source hash drifted: {source}")
            canonical_bytes += int(item.byte_size)
            frame = pd.read_parquet(source)
            sensor_values = frame[str(source_contract["sensor_group_column"])].astype(str).unique()
            if len(sensor_values) != 1:
                raise WorkflowExecutionError(f"Methane block mixes sensor groups: {source}")
            sensor = str(sensor_values[0])
            if sensor not in target_sensors:
                continue
            pool = str(item.pool)
            if pool not in METHANE_POOLS:
                raise WorkflowExecutionError(f"Methane row escaped the frozen pools: {pool}")
            feature_rows, truth_rows, manifest_rows = _window_rows_for_block(
                frame,
                dataset_id=dataset_id,
                raw_group_id=str(item.raw_group_id),
                pool=pool,
                sensor_group_id=sensor,
                timestamp_column="__timestamp_utc",
                value_column=str(source_contract["value_column"]),
                feature_columns=[str(value) for value in source_contract["feature_columns"]],
                history_seconds=int(data_contract["history_seconds"]),
                horizon_seconds=int(data_contract["horizon_seconds"]),
                stride_seconds=int(data_contract["stride_seconds"]),
                risk_threshold=float(data_contract["risk_concentration_threshold"]),
            )
            if not manifest_rows:
                raise WorkflowExecutionError(f"Methane block produced no complete windows: {source}")
            block_ids = [str(value["window_id"]) for value in manifest_rows]
            if len(block_ids) != len(set(block_ids)) or seen_window_ids.intersection(block_ids):
                raise WorkflowExecutionError("E059 methane window IDs are not globally unique")
            seen_window_ids.update(block_ids)
            features = pd.DataFrame.from_records(feature_rows)
            truth = pd.DataFrame.from_records(truth_rows)
            manifest = pd.DataFrame.from_records(manifest_rows)
            current_feature_names = json.loads(str(features.iloc[0]["history_json"]))[
                "feature_names"
            ]
            if feature_names is None:
                feature_names = [str(value) for value in current_feature_names]
            elif feature_names != [str(value) for value in current_feature_names]:
                raise WorkflowExecutionError("E059 methane feature schema drifted between blocks")
            sinks.write("features", features)
            sinks.write("manifest", manifest)
            if pool == "D_b_te":
                sinks.write("branch_test_features", features)
                sinks.write("branch_test_truth", truth)
            elif pool in {"D_b_tr", "D_b_sel", "D_b_prob"}:
                sinks.write("non_test_truth", truth)
            else:
                sinks.write("episode_features", features)
                sinks.write("episode_truth", truth)
            pool_counts[pool] += len(manifest)
            sensor_counts[sensor] += len(manifest)
        if set(pool_counts) != set(METHANE_POOLS):
            raise WorkflowExecutionError(
                f"E059 does not cover all eight frozen pools: {sorted(pool_counts)}"
            )
        if set(sensor_counts) != target_sensors:
            raise WorkflowExecutionError(
                f"E059 target sensor coverage drifted: {sorted(sensor_counts)}"
            )
        sinks.close()
        outputs = sinks.promote()
    except Exception:
        sinks.abort()
        raise

    source_paths = {
        "dataset_source_decision": root / "evidence/data/dataset_source_decision.json",
        "role_lock": role_lock_path,
        "file_manifest": root / "data/locked/file_manifest.parquet",
        "split_manifest": root / "data/locked/split_manifest.parquet",
    }
    receipt = {
        "schema_version": 1,
        "step_id": "E059",
        "status": "pass",
        "dataset_id": dataset_id,
        "source_artifacts": {
            key: {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
            }
            for key, path in source_paths.items()
        },
        "canonical_record_files": len(rows),
        "canonical_bytes": canonical_bytes,
        "target_sensors": sorted(target_sensors),
        "sensor_window_counts": dict(sorted(sensor_counts.items())),
        "pool_window_counts": {pool: int(pool_counts[pool]) for pool in METHANE_POOLS},
        "window_count": len(seen_window_ids),
        "feature_names": feature_names or [],
        "outputs": outputs,
        "execution_boundary": {
            "model_fit_count": 0,
            "threshold_selection_count": 0,
            "test_metric_count": 0,
            "raw_dataset_synced_locally": False,
            "performance_claims_authorized": False,
        },
    }
    receipt_path = root / MATERIALIZATION_RECEIPT
    write_once_json(receipt_path, receipt)
    return {
        "status": "pass",
        "output_paths": [*OUTPUT_PATHS.values(), MATERIALIZATION_RECEIPT],
        "details": {
            "dataset_id": dataset_id,
            "window_count": len(seen_window_ids),
            "sensor_count": len(sensor_counts),
            "pool_count": len(pool_counts),
            "materialization_receipt_sha256": sha256_file(receipt_path),
        },
    }


def load_verified_methane_window_store(project_root: Path) -> Dict[str, Any]:
    root = project_root.resolve()
    receipt_path = root / MATERIALIZATION_RECEIPT
    receipt = load_json(receipt_path)
    if receipt.get("status") != "pass" or receipt.get("step_id") != "E059":
        raise WorkflowExecutionError("E059 materialization receipt is unavailable")
    for source in receipt.get("source_artifacts", {}).values():
        path = root / str(source.get("path", ""))
        if not path.is_file() or sha256_file(path) != source.get("sha256"):
            raise WorkflowExecutionError(f"E059 source artifact drifted: {path}")
    outputs = receipt.get("outputs", {})
    if set(outputs) != set(OUTPUT_PATHS):
        raise WorkflowExecutionError("E059 materialization output set is incomplete")
    for key, expected_path in OUTPUT_PATHS.items():
        record = outputs[key]
        path = root / expected_path
        if (
            record.get("path") != expected_path
            or not path.is_file()
            or path.stat().st_size != int(record.get("bytes", -1))
            or sha256_file(path) != record.get("sha256")
        ):
            raise WorkflowExecutionError(f"E059 materialized artifact drifted: {key}")
    return {
        "status": "pass",
        "dataset_id": receipt["dataset_id"],
        "window_count": int(receipt["window_count"]),
        "sensor_count": len(receipt["target_sensors"]),
        "feature_names": list(receipt["feature_names"]),
        "paths": {key: value["path"] for key, value in outputs.items()},
        "hashes": {key: value["sha256"] for key, value in outputs.items()},
        "receipt_sha256": sha256_file(receipt_path),
    }
