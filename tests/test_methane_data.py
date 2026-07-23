from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from mining1_exp.methane_data import build_methane_window_store


def _resolve_tbd(value):
    if isinstance(value, dict):
        return {key: _resolve_tbd(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_tbd(item) for item in value]
    if isinstance(value, str) and "TBD" in value:
        return 1
    return value


def _adapter() -> dict:
    return {
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


def test_methane_windows_keep_history_truth_physically_separate(tmp_path: Path) -> None:
    root = tmp_path / "project"
    for relative in ("configs", "evidence/data", "data/locked", "data/canonical"):
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
    (root / "evidence/data/dataset_source_decision.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "roles": {
                    "methane_dataset": {
                        "dataset_id": "methane-public",
                        "adapter_contract": _adapter(),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    file_rows = []
    split_rows = []
    sensors = ("MM263", "MM264", "MM256")
    pools = ("D_b_tr", "D_b_sel", "D_b_prob", "D_b_te")
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    for pool_index, pool in enumerate(pools):
        for sensor in sensors:
            group = f"{sensor}-{pool}"
            path = root / "data/canonical" / f"{group}.parquet"
            timestamps = pd.date_range(start + pd.Timedelta(hours=pool_index), periods=41, freq="30s")
            values = [0.2] * 31 + [1.2] * 10
            frame = pd.DataFrame(
                {
                    "__timestamp_utc": timestamps,
                    "sensor": sensor,
                    "ch4": values,
                    "temperature": [20.0] * len(timestamps),
                }
            )
            frame.to_parquet(path, index=False)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
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
                    "sha256": digest,
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
        json.dumps({"status": "pass", "dataset_id": "methane-public"}), encoding="utf-8"
    )

    result = build_methane_window_store(root)
    features = pd.read_parquet(root / result["paths"]["features"])
    branch_features = pd.read_parquet(root / result["paths"]["branch_test_features"])
    non_test_truth = pd.read_parquet(root / result["paths"]["non_test_truth"])
    branch_truth = pd.read_parquet(root / result["paths"]["branch_test_truth"])
    truth = pd.concat([non_test_truth, branch_truth], ignore_index=True)
    manifest = pd.read_parquet(root / result["paths"]["manifest"])
    assert set(manifest["pool"]) == set(pools)
    assert set(manifest["sensor_group_id"]) == set(sensors)
    assert "event_truth" not in features.columns
    assert set(branch_features["pool"]) == {"D_b_te"}
    assert "D_b_te" not in set(non_test_truth["pool"])
    assert set(branch_truth["pool"]) == {"D_b_te"}
    assert "history_json" not in truth.columns
    positive_ids = set(truth.loc[truth["event_truth"] == 1, "window_id"])
    earliest = features.loc[features["window_id"].isin(positive_ids)].sort_values("history_end").iloc[0]
    payload = json.loads(earliest["history_json"])
    assert max(row[0] for row in payload["values"] if row[0] is not None) < 1.0
