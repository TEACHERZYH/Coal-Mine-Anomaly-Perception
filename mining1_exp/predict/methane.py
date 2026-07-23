from __future__ import annotations

import hashlib
from pathlib import Path
import pickle
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd
import torch

from ..governance.immutable import write_once_bytes
from ..governance.prediction_lock import build_prediction_lock, close_prediction_lock
from ..methane_data import METHANE_CONCEPT_ID
from ..models.methane import MethaneGRU, build_causal_stat_features
from ..models.yolo_adapter import select_calibrator_group_cv
from ..provenance import branch_artifact_path, canonical_json_sha256, sha256_file
from ..train.methane import (
    TRAINING_SEEDS,
    MethaneArrays,
    load_methane_arrays,
    transform_sequences,
)
from ..training_data import load_frozen_protocol
from ..workflow_common import WorkflowExecutionError, load_json, write_parquet_artifact


def _baseline_scores(
    arrays: MethaneArrays,
    payload: Mapping[str, Any],
    family_id: str,
) -> np.ndarray:
    count = next(
        index for index, name in enumerate(arrays.feature_names) if name.endswith("_observed_mask")
    )
    if family_id == "S1-RULE":
        return np.asarray(
            payload["rule"].score(arrays.history[:, :, :count][:, :, 0:1]),
            dtype=np.float64,
        )
    if family_id == "S1-HGB":
        features, names = build_causal_stat_features(
            arrays.history[:, :, :count],
            sample_period_seconds=int(payload["stride_seconds"]),
        )
        if tuple(names) != tuple(payload["feature_names"]):
            raise WorkflowExecutionError("S1 baseline feature contract drifted")
        return np.asarray(
            payload["hgb"].predict_proba(payload["standardizer"].transform(features)),
            dtype=np.float64,
        )
    raise WorkflowExecutionError(f"Unknown S1 baseline family: {family_id}")


def _gru_scores(
    arrays: MethaneArrays,
    checkpoint_path: Path,
    protocol: Mapping[str, Any],
) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("family_id") != "S1-GRU":
        raise WorkflowExecutionError(f"Invalid S1 GRU checkpoint: {checkpoint_path}")
    feature_names = tuple(str(value) for value in checkpoint["feature_names"])
    if feature_names != arrays.feature_names:
        raise WorkflowExecutionError("S1 GRU checkpoint feature contract drifted")
    model = MethaneGRU(
        feature_names,
        hidden_size=int(protocol["models"]["methane"]["hidden_size"]),
        layers=int(protocol["models"]["methane"]["layers"]),
        dropout=float(protocol["models"]["methane"]["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    values = transform_sequences(arrays.history, checkpoint["preprocessor"])
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(values), 1024):
            batch = torch.from_numpy(values[start : start + 1024]).to(device)
            outputs.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(outputs).astype(np.float64)


def _calibrate(
    probability_arrays: MethaneArrays,
    raw_scores: np.ndarray,
    protocol: Mapping[str, Any],
) -> tuple[Any, bytes]:
    if probability_arrays.labels is None:
        raise WorkflowExecutionError("S1 calibration labels are unavailable")
    selection = select_calibrator_group_cv(
        raw_scores,
        probability_arrays.labels,
        probability_arrays.metadata["raw_group_id"].astype(str),
        pool="D_b_prob",
        negatives_verified=True,
        folds=int(protocol["evaluation"]["calibration_group_cv_folds"]),
        methods=tuple(protocol["evaluation"]["calibration_candidates"]),
    )
    payload = {
        "schema_version": 1,
        "method": selection.method,
        "fold_count": selection.fold_count,
        "cv_brier_by_method": selection.cv_brier_by_method,
        "calibrator": selection.calibrator,
    }
    return selection.calibrator, pickle.dumps(payload, protocol=4)


def _boundary_hashes(project_root: Path) -> Dict[str, str]:
    paths = {
        "seal_hash": branch_artifact_path(project_root, "final_seal"),
        "protocol_hash": project_root / "configs/protocol_lock.pretest.yaml",
        "source_hash": project_root / "evidence/data/dataset_source_decision.json",
        "matrix_hash": project_root / "configs/experiment_matrix.template.csv",
    }
    for path in paths.values():
        if not path.is_file():
            raise WorkflowExecutionError(f"S1 prediction boundary artifact is missing: {path}")
    return {name: sha256_file(path) for name, path in paths.items()}


def _run_specs(project_root: Path, protocol: Mapping[str, Any]) -> list[Dict[str, Any]]:
    baseline_manifest = load_json(project_root / "runs/S1/baselines/run_manifest.json")
    baseline_selection = load_json(project_root / "runs/S1/baselines/selection.json")
    baseline_model_path = project_root / str(baseline_manifest["model_path"])
    if sha256_file(baseline_model_path) != baseline_manifest.get("model_sha256"):
        raise WorkflowExecutionError("S1 baseline model hash drifted")
    baseline_payload = pickle.loads(baseline_model_path.read_bytes())
    specs: list[Dict[str, Any]] = []
    for family_id in ("S1-RULE", "S1-HGB"):
        selection = baseline_selection.get("family_selections", {}).get(family_id)
        if not isinstance(selection, Mapping):
            raise WorkflowExecutionError(f"S1 baseline threshold is missing: {family_id}")
        specs.append(
            {
                "lock_id": family_id,
                "family_id": family_id,
                "train_seed": None,
                "threshold": float(selection["threshold"]),
                "model_path": baseline_model_path,
                "score": lambda arrays, family_id=family_id: _baseline_scores(
                    arrays, baseline_payload, family_id
                ),
            }
        )
    for seed in TRAINING_SEEDS:
        manifest = load_json(
            project_root / f"runs/S1/gru/S1-GRU/seed-{seed}/run_manifest.json"
        )
        checkpoint_path = project_root / str(manifest["checkpoint_path"])
        if (
            manifest.get("status") != "pass"
            or int(manifest.get("train_seed", -1)) != seed
            or sha256_file(checkpoint_path) != manifest.get("checkpoint_sha256")
        ):
            raise WorkflowExecutionError(f"S1 GRU manifest drifted: seed {seed}")
        specs.append(
            {
                "lock_id": f"S1-GRU-seed-{seed}",
                "family_id": "S1-GRU",
                "train_seed": seed,
                "threshold": float(manifest["selection_threshold"]),
                "model_path": checkpoint_path,
                "score": lambda arrays, path=checkpoint_path: _gru_scores(
                    arrays, path, protocol
                ),
            }
        )
    return specs


def predict_s1_sealed(
    project_root: Path,
    run_root: Path,
    config_path: Optional[Path],
) -> Dict[str, Any]:
    del config_path
    probe = run_root / "slurm_environment_probe.json"
    if not probe.is_file():
        raise WorkflowExecutionError("S1 prediction step lacks its Slurm environment probe")
    protocol = load_frozen_protocol(project_root)
    probability = load_methane_arrays(
        project_root, pools={"D_b_prob"}, include_labels=True
    )
    test = load_methane_arrays(project_root, pools={"D_b_te"}, include_labels=False)
    specs = _run_specs(project_root, protocol)
    rows = []
    calibrator_payloads: Dict[str, bytes] = {}
    model_hashes: Dict[str, str] = {}
    policy_hashes: Dict[str, str] = {}
    for spec in specs:
        probability_scores = np.asarray(spec["score"](probability), dtype=np.float64)
        test_scores = np.asarray(spec["score"](test), dtype=np.float64)
        calibrator, calibrator_bytes = _calibrate(probability, probability_scores, protocol)
        calibrated = calibrator.predict(test_scores)
        lock_id = str(spec["lock_id"])
        calibrator_payloads[lock_id] = calibrator_bytes
        model_hash = sha256_file(spec["model_path"])
        policy_hash = canonical_json_sha256(
            {
                "calibrator_sha256": hashlib.sha256(calibrator_bytes).hexdigest(),
                "threshold": float(spec["threshold"]),
                "selection_pool": "D_b_sel",
            }
        )
        model_hashes[lock_id] = model_hash
        policy_hashes[lock_id] = policy_hash
        run_id = canonical_json_sha256(
            {
                "family_id": spec["family_id"],
                "train_seed": spec["train_seed"],
                "model_sha256": model_hash,
                "policy_sha256": policy_hash,
            }
        )
        for item, raw, probability_value in zip(
            test.metadata.itertuples(index=False), test_scores, calibrated
        ):
            rows.append(
                {
                    "run_id": run_id,
                    "family_id": str(spec["family_id"]),
                    "train_seed": spec["train_seed"],
                    "window_id": str(item.window_id),
                    "raw_group_id": str(item.raw_group_id),
                    "prediction_role": "branch_test",
                    "pool_role": "D_b_te",
                    "concept_id": METHANE_CONCEPT_ID,
                    "score_raw": np.float32(raw),
                    "score_calibrated": np.float32(probability_value),
                    "threshold": np.float32(spec["threshold"]),
                    "abstained": False,
                    "model_hash": model_hash,
                    "policy_hash": policy_hash,
                }
            )
    predictions = pd.DataFrame.from_records(rows).sort_values(
        ["run_id", "window_id", "concept_id"]
    )
    expected = len(test.metadata) * len(specs)
    if len(predictions) != expected or predictions.duplicated(
        ["run_id", "window_id", "concept_id"]
    ).any():
        raise WorkflowExecutionError("S1 sealed prediction coverage is incomplete")
    output = project_root / "predictions/locked/S1.parquet"
    write_parquet_artifact(output, predictions)
    calibrator_store = project_root / "predictions/locked/S1/calibrators.pkl"
    write_once_bytes(calibrator_store, pickle.dumps(calibrator_payloads, protocol=4))
    boundaries = _boundary_hashes(project_root)
    prediction_paths = {lock_id: output for lock_id in sorted(model_hashes)}
    lock = build_prediction_lock(
        scope="branch",
        required_prediction_families=sorted(model_hashes),
        prediction_paths=prediction_paths,
        model_hashes=model_hashes,
        calibrator_and_policy_hashes=policy_hashes,
        **boundaries,
    )
    lock["prediction_paths"] = {
        lock_id: output.relative_to(project_root).as_posix()
        for lock_id in sorted(model_hashes)
    }
    lock["calibrator_store_path"] = calibrator_store.relative_to(project_root).as_posix()
    lock["calibrator_store_sha256"] = sha256_file(calibrator_store)
    lock_path = project_root / "predictions/locked/S1/prediction_lock.json"
    close_prediction_lock(lock_path, lock)
    return {
        "status": "pass",
        "output_paths": [
            output.relative_to(project_root).as_posix(),
            lock_path.relative_to(project_root).as_posix(),
        ],
        "details": {
            "prediction_count": len(predictions),
            "run_count": len(specs),
            "test_truth_opened": False,
        },
    }
