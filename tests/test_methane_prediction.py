from __future__ import annotations

import io
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from mining1_exp.models.methane import (
    CausalStandardizer,
    MethaneGRU,
    MethaneHGB,
    PersistenceRiskRule,
    build_causal_stat_features,
)
from mining1_exp.predict.methane import predict_s1_sealed
from mining1_exp.provenance import sha256_file
from mining1_exp.train.methane import fit_sequence_preprocessor
from mining1_exp.workflow_common import WorkflowExecutionError


FEATURE_NAMES = (
    "ch4",
    "temperature",
    "ch4_observed_mask",
    "temperature_observed_mask",
)


def _history(value: float) -> str:
    return json.dumps(
        {
            "feature_names": list(FEATURE_NAMES),
            "values": [[value, 20.0, 1.0, 1.0] for _ in range(10)],
        },
        sort_keys=True,
    )


def _prepare_project(root: Path) -> Path:
    for relative in (
        "configs",
        "evidence/data",
        "data/locked",
        "data/sealed",
        "data/seals",
        "evidence/pilot",
        "runs/S1/baselines",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    protocol = {
        "models": {"methane": {"hidden_size": 64, "layers": 1, "dropout": 0.20}},
        "evaluation": {
            "calibration_group_cv_folds": 2,
            "calibration_candidates": ["temperature"],
        },
    }
    (root / "configs/protocol_lock.pretest.yaml").write_text(
        yaml.safe_dump(protocol), encoding="utf-8"
    )
    (root / "configs/experiment_matrix.template.csv").write_text("family_id\nS1\n", encoding="utf-8")
    (root / "evidence/data/dataset_source_decision.json").write_text(
        json.dumps({"status": "pass"}), encoding="utf-8"
    )
    (root / "data/seals/branch_test_seal.json").write_text(
        json.dumps({"status": "sealed"}), encoding="utf-8"
    )
    features = []
    truth = []
    for pool, group_count in (("D_b_prob", 4), ("D_b_te", 2)):
        for group in range(group_count):
            for label in (0, 1):
                window = f"{pool}-{group}-{label}"
                features.append(
                    {
                        "dataset_id": "methane",
                        "window_id": window,
                        "raw_group_id": f"{pool}-group-{group}",
                        "sensor_group_id": "MM263",
                        "pool": pool,
                        "history_end": pd.Timestamp("2026-01-01T00:00:00Z")
                        + pd.Timedelta(seconds=len(features) * 30),
                        "history_json": _history(0.2 if label == 0 else 1.2),
                    }
                )
                truth.append(
                    {
                        "window_id": window,
                        "raw_group_id": f"{pool}-group-{group}",
                        "sensor_group_id": "MM263",
                        "pool": pool,
                        "event_truth": label,
                        "timestamp_seconds": len(features) * 30,
                        "raw_value": 0.2 if label == 0 else 1.2,
                    }
                )
    feature_frame = pd.DataFrame(features)
    truth_frame = pd.DataFrame(truth)
    feature_frame.to_parquet(root / "data/locked/methane_window_features.parquet", index=False)
    feature_frame.loc[feature_frame["pool"] == "D_b_te"].to_parquet(
        root / "data/locked/branch_methane_test_features.parquet", index=False
    )
    truth_frame.loc[truth_frame["pool"] != "D_b_te"].to_parquet(
        root / "data/locked/methane_non_test_truth.parquet", index=False
    )
    truth_frame.loc[truth_frame["pool"] == "D_b_te"].to_parquet(
        root / "data/sealed/branch_methane_truth.parquet", index=False
    )
    probability = feature_frame.loc[feature_frame["pool"] == "D_b_prob"]
    histories = np.stack(
        [np.asarray(json.loads(value)["values"], dtype=np.float64) for value in probability["history_json"]]
    )
    labels = np.asarray([0, 1] * 4, dtype=np.int64)
    statistical, names = build_causal_stat_features(histories[:, :, :2], sample_period_seconds=30)
    standardizer = CausalStandardizer(names).fit(statistical, pool="D_b_tr")
    hgb = MethaneHGB(names).fit(standardizer.transform(statistical), labels, pool="D_b_tr")
    baseline_path = root / "runs/S1/baselines/baseline_models.pkl"
    baseline_path.write_bytes(
        pickle.dumps(
            {
                "rule": PersistenceRiskRule(1.0, 0.1),
                "hgb": hgb,
                "standardizer": standardizer,
                "feature_names": names,
                "stride_seconds": 30,
            },
            protocol=4,
        )
    )
    (root / "runs/S1/baselines/run_manifest.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "model_path": baseline_path.relative_to(root).as_posix(),
                "model_sha256": sha256_file(baseline_path),
            }
        ),
        encoding="utf-8",
    )
    (root / "runs/S1/baselines/selection.json").write_text(
        json.dumps(
            {
                "family_selections": {
                    "S1-RULE": {"threshold": 0.5},
                    "S1-HGB": {"threshold": 0.5},
                }
            }
        ),
        encoding="utf-8",
    )
    preprocessor = fit_sequence_preprocessor(histories, FEATURE_NAMES)
    for seed in (1701, 2903, 4219):
        run = root / f"runs/S1/gru/S1-GRU/seed-{seed}"
        run.mkdir(parents=True)
        torch.manual_seed(seed)
        model = MethaneGRU(FEATURE_NAMES, hidden_size=64, layers=1, dropout=0.20)
        checkpoint = run / "best.pt"
        buffer = io.BytesIO()
        torch.save(
            {
                "family_id": "S1-GRU",
                "feature_names": list(FEATURE_NAMES),
                "preprocessor": preprocessor,
                "model_state_dict": model.state_dict(),
            },
            buffer,
        )
        checkpoint.write_bytes(buffer.getvalue())
        (run / "run_manifest.json").write_text(
            json.dumps(
                {
                    "status": "pass",
                    "train_seed": seed,
                    "selection_threshold": 0.5,
                    "checkpoint_path": checkpoint.relative_to(root).as_posix(),
                    "checkpoint_sha256": sha256_file(checkpoint),
                }
            ),
            encoding="utf-8",
        )
    run_root = root / "job"
    run_root.mkdir()
    (run_root / "slurm_environment_probe.json").write_text("{}", encoding="utf-8")
    return run_root


def test_s1_prediction_closes_five_truth_free_runs(tmp_path: Path) -> None:
    run_root = _prepare_project(tmp_path)
    result = predict_s1_sealed(tmp_path, run_root, None)
    predictions = pd.read_parquet(tmp_path / "predictions/locked/S1.parquet")
    lock = json.loads(
        (tmp_path / "predictions/locked/S1/prediction_lock.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "pass"
    assert predictions["run_id"].nunique() == 5
    assert set(lock["required_prediction_families"]) == {
        "S1-RULE",
        "S1-HGB",
        "S1-GRU-seed-1701",
        "S1-GRU-seed-2903",
        "S1-GRU-seed-4219",
    }
    assert not any("truth" in column or "label" in column for column in predictions.columns)


def test_s1_prediction_rejects_missing_probe_before_writing(tmp_path: Path) -> None:
    run_root = _prepare_project(tmp_path)
    (run_root / "slurm_environment_probe.json").unlink()
    with pytest.raises(WorkflowExecutionError, match="Slurm environment probe"):
        predict_s1_sealed(tmp_path, run_root, None)
    assert not (tmp_path / "predictions/locked/S1.parquet").exists()
    assert not (tmp_path / "predictions/locked/S1/prediction_lock.json").exists()
