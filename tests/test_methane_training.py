from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from mining1_exp.train.methane import (
    _classification_event_metrics,
    _fit_persistence_rule,
    load_methane_arrays,
    s1_gru_seed,
    train_s1_gru,
)
from mining1_exp.workflow_common import WorkflowExecutionError


def _history(index: int, *, target_names: bool = False) -> str:
    label_signal = 1.2 if index % 4 >= 2 else 0.2
    values = [[label_signal, 20.0, 1.0, 1.0] for _ in range(10)]
    feature_names = (
        ["target_value", "temperature", "target_value_observed_mask", "temperature_observed_mask"]
        if target_names
        else ["ch4", "temperature", "ch4_observed_mask", "temperature_observed_mask"]
    )
    return json.dumps(
        {
            "feature_names": feature_names,
            "values": values,
        },
        sort_keys=True,
    )


def _write_store(root: Path) -> None:
    (root / "data/locked").mkdir(parents=True)
    (root / "data/sealed").mkdir(parents=True)
    features = []
    truth = []
    for pool in ("D_b_tr", "D_b_sel", "D_b_te"):
        for index in range(8):
            window_id = f"{pool}-{index}"
            features.append(
                {
                    "dataset_id": "methane",
                    "window_id": window_id,
                    "raw_group_id": f"{pool}-group-{index // 4}",
                    "sensor_group_id": "MM263",
                    "pool": pool,
                    "history_end": pd.Timestamp("2026-01-01T00:00:00Z")
                    + pd.Timedelta(seconds=30 * index),
                    "history_json": _history(index),
                }
            )
            truth.append(
                {
                    "window_id": window_id,
                    "pool": pool,
                    "event_truth": int(index % 4 >= 2),
                    "timestamp_seconds": 30 * index,
                    "raw_value": 1.2 if index % 4 >= 2 else 0.2,
                }
            )
    feature_frame = pd.DataFrame(features)
    truth_frame = pd.DataFrame(truth)
    feature_frame.to_parquet(
        root / "data/locked/methane_window_features.parquet", index=False
    )
    feature_frame.loc[feature_frame["pool"] == "D_b_te"].to_parquet(
        root / "data/locked/branch_methane_test_features.parquet", index=False
    )
    truth_frame.loc[truth_frame["pool"] != "D_b_te"].to_parquet(
        root / "data/locked/methane_non_test_truth.parquet", index=False
    )
    truth_frame.loc[truth_frame["pool"] == "D_b_te"].to_parquet(
        root / "data/sealed/branch_methane_truth.parquet", index=False
    )


def _write_runtime_contracts(root: Path) -> Path:
    (root / "configs").mkdir(parents=True)
    (root / "evidence/pilot").mkdir(parents=True)
    protocol = {
        "data": {
            "methane": {
                "stride_seconds": 30,
                "risk_concentration_threshold": 1.0,
                "event_merge_gap_seconds": 30,
                "horizon_seconds": 30,
            }
        },
        "models": {"methane": {"hidden_size": 64, "layers": 1, "dropout": 0.20}},
        "training": {
            "methane": {
                "effective_batch_size": 2,
                "max_updates": 1,
                "learning_rate": 0.001,
                "max_false_alarms_per_hour": 100.0,
            }
        },
    }
    (root / "configs/protocol_lock.pretest.yaml").write_text(
        yaml.safe_dump(protocol), encoding="utf-8"
    )
    (root / "evidence/pilot/resource_pilot.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "decisions": {
                    "precision": "fp32",
                    "micro_batch_size": {"methane": 2},
                    "gradient_accumulation": {"methane": 1},
                },
            }
        ),
        encoding="utf-8",
    )
    run_root = root / "job"
    run_root.mkdir()
    (run_root / "slurm_environment_probe.json").write_text("{}", encoding="utf-8")
    return run_root


def test_training_loader_refuses_branch_test_labels(tmp_path: Path) -> None:
    _write_store(tmp_path)
    with pytest.raises(WorkflowExecutionError, match="cannot open D_b_te labels"):
        load_methane_arrays(tmp_path, pools={"D_b_te"}, include_labels=True)
    inference = load_methane_arrays(tmp_path, pools={"D_b_te"}, include_labels=False)
    assert inference.labels is None
    assert len(inference.history) == 8


def test_training_loader_renames_reviewed_target_sensor_history(tmp_path: Path) -> None:
    _write_store(tmp_path)
    features = pd.read_parquet(tmp_path / "data/locked/methane_window_features.parquet")
    features["history_json"] = [_history(index, target_names=True) for index in range(len(features))]
    features.to_parquet(tmp_path / "data/locked/methane_window_features.parquet", index=False)

    arrays = load_methane_arrays(tmp_path, pools={"D_b_tr"}, include_labels=True)

    assert arrays.feature_names == (
        "ch4_value",
        "temperature",
        "ch4_value_observed_mask",
        "temperature_observed_mask",
    )
    assert arrays.history.shape[2] == 4


def test_selection_metric_uses_raw_events_and_frozen_horizon() -> None:
    metadata = pd.DataFrame(
        {
            "sensor_group_id": ["MM263"] * 4,
            "timestamp_seconds": [0, 30, 60, 90],
            "raw_value": [0.2, 1.2, 1.2, 0.2],
        }
    )
    metrics = _classification_event_metrics(
        metadata,
        np.zeros(4, dtype=np.int64),
        np.asarray([0.9, 0.1, 0.1, 0.1]),
        threshold=0.5,
        stride_seconds=30,
        risk_threshold=1.0,
        event_merge_gap_seconds=30,
        horizon_seconds=30,
    )
    assert metrics == {
        "false_alarms_per_hour": 0.0,
        "event_macro_f1": 1.0,
        "event_miss_rate": 0.0,
    }


def test_persistence_rule_uses_train_derived_fallback_for_all_missing_history() -> None:
    history = np.asarray(
        [
            [[np.nan], [np.nan], [np.nan]],
            [[0.1], [0.2], [0.4]],
            [[0.8], [1.0], [np.nan]],
        ],
        dtype=np.float64,
    )
    rule = _fit_persistence_rule(history, concentration_threshold=1.0)
    assert rule.missing_value == pytest.approx(0.7)
    scores = rule.score(history)
    assert np.isfinite(scores).all()
    assert scores[0] == pytest.approx(
        1.0 / (1.0 + np.exp(-((0.7 - 1.0) / rule.transition_scale)))
    )


def test_gru_array_mapping_and_one_update_smoke(tmp_path: Path) -> None:
    _write_store(tmp_path)
    run_root = _write_runtime_contracts(tmp_path)
    assert [s1_gru_seed(index) for index in range(3)] == [1701, 2903, 4219]
    result = train_s1_gru(tmp_path, run_root, 0, max_updates_override=1)
    assert result["status"] == "pass"
    assert result["train_seed"] == 1701
    assert result["max_optimizer_updates"] == 1
    assert (tmp_path / result["checkpoint_path"]).is_file()
