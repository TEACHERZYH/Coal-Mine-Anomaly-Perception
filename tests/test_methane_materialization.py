from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from mining1_exp.methane_materialization import (
    MATERIALIZATION_RECEIPT,
    METHANE_POOLS,
    OUTPUT_PATHS,
    load_verified_methane_window_store,
    materialize_methane_windows,
)
from mining1_exp.review_methane_materialization import review_methane_materialization
from mining1_exp.workflow_common import WorkflowExecutionError


def _resolve_tbd(value):
    if isinstance(value, dict):
        return {key: _resolve_tbd(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_tbd(item) for item in value]
    if isinstance(value, str) and "TBD" in value:
        return 1
    return value


def _build_fixture(root: Path) -> None:
    for relative in ("configs", "evidence/data", "data/locked", "data/sealed", "data/canonical"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    template = yaml.safe_load(
        (Path(__file__).parents[1] / "configs/protocol_lock.template.yaml").read_text(
            encoding="utf-8-sig"
        )
    )
    protocol = _resolve_tbd(template)
    (root / "configs/protocol_lock.pretest.yaml").write_text(
        yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8"
    )
    adapter = {
        "schema_version": 1,
        "kind": "methane_csv",
        "archive_subdir": ".",
        "methane": {
            "csv_globs": ["*.csv"],
            "timestamp_column": "time",
            "value_column": "ch4",
            "sensor_group_column": "sensor",
            "feature_columns": ["temperature"],
            "group_duration_seconds": 3600,
            "timezone": "UTC",
        },
    }
    (root / "evidence/data/dataset_source_decision.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "roles": {
                    "methane_dataset": {
                        "dataset_id": "methane-public",
                        "adapter_contract": adapter,
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    sensors = ("MM263", "MM264", "MM256")
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    file_rows = []
    split_rows = []
    for pool_index, pool in enumerate(METHANE_POOLS):
        for sensor in sensors:
            group = f"{sensor}-{pool}"
            path = root / "data/canonical" / f"{group}.parquet"
            timestamps = pd.date_range(
                start + pd.Timedelta(hours=2 * pool_index), periods=41, freq="30s"
            )
            frame = pd.DataFrame(
                {
                    "__timestamp_utc": timestamps,
                    "sensor": sensor,
                    "ch4": [0.2] * 31 + [1.2] * 10,
                    "temperature": [20.0] * len(timestamps),
                }
            )
            frame.to_parquet(path, index=False)
            file_rows.append(
                {
                    "dataset_id": "methane-public",
                    "record_id": group,
                    "archive_id": "archive-a",
                    "relative_path": path.relative_to(root).as_posix(),
                    "modality": "methane",
                    "raw_group_id": group,
                    "pair_id": None,
                    "sequence_id": sensor,
                    "timestamp_or_order": timestamps[0].isoformat(),
                    "label_summary_json": "{}",
                    "byte_size": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
            split_rows.append(
                {
                    "dataset_id": "methane-public",
                    "record_id": group,
                    "raw_group_id": group,
                    "pool": pool,
                    "split_seed": 13007,
                    "split_version": "test",
                    "ontology_hash": "a" * 64,
                    "dedup_report_hash": "b" * 64,
                }
            )
    pd.DataFrame(file_rows).to_parquet(root / "data/locked/file_manifest.parquet", index=False)
    pd.DataFrame(split_rows).to_parquet(root / "data/locked/split_manifest.parquet", index=False)
    (root / "data/locked/methane_role_lock.json").write_text(
        json.dumps({"status": "pass", "dataset_id": "methane-public"}, sort_keys=True),
        encoding="utf-8",
    )


def test_e059_streams_reviews_reuses_and_rejects_drift(tmp_path: Path) -> None:
    _build_fixture(tmp_path)
    run_root = tmp_path / "runs/slurm/E059/test"
    run_root.mkdir(parents=True)
    first = materialize_methane_windows(tmp_path, run_root, None)
    review = review_methane_materialization(tmp_path)
    verified = load_verified_methane_window_store(tmp_path)
    assert first["status"] == review["status"] == verified["status"] == "pass"
    receipt_path = tmp_path / MATERIALIZATION_RECEIPT
    receipt_hash = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    output_hashes = {
        key: hashlib.sha256((tmp_path / path).read_bytes()).hexdigest()
        for key, path in OUTPUT_PATHS.items()
    }
    second = materialize_methane_windows(tmp_path, run_root, None)
    assert second["status"] == "pass"
    assert hashlib.sha256(receipt_path.read_bytes()).hexdigest() == receipt_hash
    assert output_hashes == {
        key: hashlib.sha256((tmp_path / path).read_bytes()).hexdigest()
        for key, path in OUTPUT_PATHS.items()
    }
    assert verified["window_count"] > 0
    assert verified["sensor_count"] == 3
    branch_features = pd.read_parquet(tmp_path / OUTPUT_PATHS["branch_test_features"])
    branch_truth = pd.read_parquet(tmp_path / OUTPUT_PATHS["branch_test_truth"])
    assert set(branch_features["pool"]) == set(branch_truth["pool"]) == {"D_b_te"}
    assert "event_truth" not in branch_features.columns
    assert "history_json" not in branch_truth.columns

    with (tmp_path / OUTPUT_PATHS["manifest"]).open("ab") as handle:
        handle.write(b"drift")
    with pytest.raises(WorkflowExecutionError, match="materialized artifact drifted"):
        load_verified_methane_window_store(tmp_path)
