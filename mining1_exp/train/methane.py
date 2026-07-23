from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import pickle
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..evaluate.event_metrics import evaluate_methane_events, select_s1_threshold
from ..governance.immutable import write_once_bytes, write_once_json
from ..methane_materialization import load_verified_methane_window_store
from ..models.methane import (
    CausalStandardizer,
    MethaneGRU,
    MethaneHGB,
    PersistenceRiskRule,
    build_causal_stat_features,
)
from ..provenance import sha256_file, validate_source_lock_amendment
from ..training_data import load_frozen_protocol
from ..workflow_common import WorkflowExecutionError, load_json


TRAINING_SEEDS = (1701, 2903, 4219)


@dataclass(frozen=True)
class MethaneArrays:
    history: np.ndarray
    labels: Optional[np.ndarray]
    metadata: pd.DataFrame
    feature_names: tuple[str, ...]


def _read_history_payloads(frame: pd.DataFrame) -> tuple[np.ndarray, tuple[str, ...]]:
    histories = []
    expected_names: Optional[tuple[str, ...]] = None
    expected_shape: Optional[tuple[int, int]] = None
    for value in frame["history_json"]:
        payload = json.loads(str(value))
        names = tuple(str(name) for name in payload.get("feature_names", []))
        rows = payload.get("values")
        if not names or not isinstance(rows, list) or not rows:
            raise WorkflowExecutionError("Methane history payload is incomplete")
        values = np.asarray(
            [[np.nan if item is None else float(item) for item in row] for row in rows],
            dtype=np.float64,
        )
        if values.ndim != 2 or values.shape[1] != len(names) or np.isinf(values).any():
            raise WorkflowExecutionError("Methane history payload has an invalid shape")
        if expected_names is None:
            expected_names = names
            expected_shape = values.shape
        if names != expected_names or values.shape != expected_shape:
            raise WorkflowExecutionError("Methane history feature contract drifted")
        histories.append(values)
    if expected_names is None:
        raise WorkflowExecutionError("Methane feature store is empty")
    return np.stack(histories), expected_names


def _safe_sequence_feature_names(feature_names: Sequence[str]) -> tuple[str, ...]:
    replacements = {
        "target_value": "ch4_value",
        "target_value_observed_mask": "ch4_value_observed_mask",
    }
    names = tuple(replacements.get(str(name), str(name)) for name in feature_names)
    if len(set(names)) != len(names):
        raise WorkflowExecutionError("Methane sequence feature repair produced duplicate names")
    return names


def load_methane_arrays(
    project_root: Path,
    *,
    pools: Iterable[str],
    include_labels: bool,
) -> MethaneArrays:
    requested = {str(pool) for pool in pools}
    allowed = {"D_b_tr", "D_b_sel", "D_b_prob", "D_b_te"}
    if not requested or not requested.issubset(allowed):
        raise WorkflowExecutionError("Methane pool request is invalid")
    if include_labels and "D_b_te" in requested:
        raise WorkflowExecutionError("Training and selection code cannot open D_b_te labels")
    if "D_b_te" in requested and requested != {"D_b_te"}:
        raise WorkflowExecutionError("Methane test features cannot be mixed with non-test pools")
    feature_path = (
        project_root / "data/locked/branch_methane_test_features.parquet"
        if requested == {"D_b_te"}
        else project_root / "data/locked/methane_window_features.parquet"
    )
    features = pd.read_parquet(feature_path)
    selected = features.loc[features["pool"].astype(str).isin(requested)].copy()
    if selected.empty:
        raise WorkflowExecutionError(f"Methane pools contain no windows: {sorted(requested)}")
    selected = selected.sort_values("window_id").reset_index(drop=True)
    history, names = _read_history_payloads(selected)
    names = _safe_sequence_feature_names(names)
    labels = None
    if include_labels:
        truth = pd.read_parquet(project_root / "data/locked/methane_non_test_truth.parquet")
        allowed_truth = truth.loc[
            truth["pool"].astype(str).isin(requested),
            ["window_id", "event_truth", "timestamp_seconds", "raw_value"],
        ]
        selected = selected.merge(allowed_truth, on="window_id", validate="one_to_one")
        selected = selected.sort_values("window_id").reset_index(drop=True)
        history, names = _read_history_payloads(selected)
        names = _safe_sequence_feature_names(names)
        labels = pd.to_numeric(selected.pop("event_truth"), errors="raise").to_numpy(
            dtype=np.int64
        )
        if not set(labels).issubset({0, 1}):
            raise WorkflowExecutionError("Methane labels are not binary")
    return MethaneArrays(history=history, labels=labels, metadata=selected, feature_names=names)


def fit_sequence_preprocessor(history: np.ndarray, feature_names: Sequence[str]) -> Dict[str, Any]:
    values = np.asarray(history, dtype=np.float64)
    names = tuple(str(value) for value in feature_names)
    if values.ndim != 3 or values.shape[2] != len(names):
        raise WorkflowExecutionError("Methane sequence preprocessing received the wrong shape")
    mask_start = next(
        (index for index, name in enumerate(names) if name.endswith("_observed_mask")),
        len(names),
    )
    if mask_start <= 0 or any(not name.endswith("_observed_mask") for name in names[mask_start:]):
        raise WorkflowExecutionError("Methane missingness-mask columns are not trailing")
    flattened = values[:, :, :mask_start].reshape(-1, mask_start)
    medians = np.asarray(
        [
            float(np.median(column[np.isfinite(column)]))
            if np.isfinite(column).any()
            else 0.0
            for column in flattened.T
        ],
        dtype=np.float64,
    )
    imputed = np.where(np.isfinite(flattened), flattened, medians)
    means = np.mean(imputed, axis=0)
    scales = np.where(np.std(imputed, axis=0) > 0, np.std(imputed, axis=0), 1.0)
    return {
        "schema_version": 1,
        "fitted_pool": "D_b_tr",
        "feature_names": list(names),
        "continuous_feature_count": mask_start,
        "medians": medians.tolist(),
        "means": means.tolist(),
        "scales": scales.tolist(),
    }


def transform_sequences(history: np.ndarray, preprocessor: Mapping[str, Any]) -> np.ndarray:
    values = np.asarray(history, dtype=np.float64).copy()
    count = int(preprocessor["continuous_feature_count"])
    if values.ndim != 3 or values.shape[2] != len(preprocessor["feature_names"]):
        raise WorkflowExecutionError("Methane sequence transform shape drifted")
    medians = np.asarray(preprocessor["medians"], dtype=np.float64)
    means = np.asarray(preprocessor["means"], dtype=np.float64)
    scales = np.asarray(preprocessor["scales"], dtype=np.float64)
    flattened = values[:, :, :count].reshape(-1, count)
    flattened = np.where(np.isfinite(flattened), flattened, medians)
    values[:, :, :count] = ((flattened - means) / scales).reshape(
        values.shape[0], values.shape[1], count
    )
    masks = values[:, :, count:]
    if not np.isfinite(values).all() or not np.isin(masks, (0.0, 1.0)).all():
        raise WorkflowExecutionError("Methane preprocessing produced invalid values or masks")
    return values.astype(np.float32)


def _classification_event_metrics(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
    stride_seconds: int,
    risk_threshold: float,
    event_merge_gap_seconds: float,
    horizon_seconds: float,
) -> Dict[str, float]:
    required = {"sensor_group_id", "timestamp_seconds", "raw_value"}
    if not required.issubset(metadata.columns):
        raise WorkflowExecutionError("Methane selection metadata lacks raw event fields")
    frame = metadata[["sensor_group_id", "timestamp_seconds", "raw_value"]].copy()
    frame["score"] = np.asarray(scores, dtype=np.float64)
    if len(frame) != len(labels) or len(frame) != len(scores):
        raise WorkflowExecutionError("Methane event metric arrays are not aligned")
    unit_results = []
    for _, sensor in frame.groupby("sensor_group_id", sort=True):
        ordered = sensor.sort_values("timestamp_seconds")
        positions = ordered["timestamp_seconds"].to_numpy(dtype=np.float64)
        gaps = np.diff(positions)
        continuity = float(np.median(gaps)) if len(gaps) else float(stride_seconds)
        unit_results.append(
            evaluate_methane_events(
                raw_positions_seconds=positions,
                raw_values=ordered["raw_value"].to_numpy(dtype=np.float64),
                risk_threshold=float(risk_threshold),
                prediction_positions_seconds=positions,
                prediction_scores=ordered["score"].to_numpy(dtype=np.float64),
                score_threshold=float(threshold),
                raw_continuity_seconds=continuity,
                prediction_continuity_seconds=continuity,
                event_merge_gap_seconds=float(event_merge_gap_seconds),
                horizon_seconds=float(horizon_seconds),
                valid_observed_sensor_hours=max(
                    continuity * len(ordered) / 3600.0,
                    continuity / 3600.0,
                ),
            )
        )
    f1_values = [
        float(result["event_f1"])
        for result in unit_results
        if result["eligible_for_macro_f1"]
    ]
    total_false = sum(float(result["false_alarm_count"]) for result in unit_results)
    total_hours = sum(float(result["valid_observed_sensor_hours"]) for result in unit_results)
    total_truth = sum(float(result["truth_event_count"]) for result in unit_results)
    total_missed = sum(float(result["missed_event_count"]) for result in unit_results)
    if not f1_values or total_hours <= 0 or total_truth <= 0:
        raise WorkflowExecutionError("Methane selection pool lacks event-metric support")
    return {
        "false_alarms_per_hour": float(total_false / total_hours),
        "event_macro_f1": float(np.mean(f1_values)),
        "event_miss_rate": float(total_missed / total_truth),
    }


def _candidate_table(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    scores_by_family: Mapping[str, np.ndarray],
    *,
    stride_seconds: int,
    methane_contract: Mapping[str, Any],
) -> pd.DataFrame:
    rows = []
    for family_id, scores in sorted(scores_by_family.items()):
        for threshold in np.round(np.arange(0.10, 0.9001, 0.05), 2):
            metrics = _classification_event_metrics(
                metadata,
                labels,
                scores,
                threshold=float(threshold),
                stride_seconds=stride_seconds,
                risk_threshold=float(methane_contract["risk_concentration_threshold"]),
                event_merge_gap_seconds=float(methane_contract["event_merge_gap_seconds"]),
                horizon_seconds=float(methane_contract["horizon_seconds"]),
            )
            rows.append(
                {
                    "candidate_id": f"{family_id}@{threshold:.2f}",
                    "family_id": family_id,
                    "threshold": float(threshold),
                    **metrics,
                }
            )
    return pd.DataFrame.from_records(rows)


def _statistical_features(arrays: MethaneArrays, stride_seconds: int) -> tuple[np.ndarray, tuple[str, ...]]:
    continuous_count = next(
        index for index, name in enumerate(arrays.feature_names) if name.endswith("_observed_mask")
    )
    return build_causal_stat_features(
        arrays.history[:, :, :continuous_count], sample_period_seconds=stride_seconds
    )


def _fit_persistence_rule(
    ch4_history: np.ndarray,
    *,
    concentration_threshold: float,
) -> PersistenceRiskRule:
    values = np.asarray(ch4_history, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] == 0 or values.shape[2] != 1:
        raise WorkflowExecutionError("Persistence-rule CH4 history has an invalid shape")
    observed_last = []
    for row in values[:, :, 0]:
        finite = row[np.isfinite(row)]
        if finite.size:
            observed_last.append(float(finite[-1]))
    if not observed_last:
        raise WorkflowExecutionError("D_b_tr has no observed CH4 value for the persistence rule")
    observed = np.asarray(observed_last, dtype=np.float64)
    return PersistenceRiskRule(
        concentration_threshold,
        max(float(np.std(observed)), 1.0e-3),
        missing_value=float(np.median(observed)),
    )


def fit_s1_baselines(project_root: Path, run_root: Path, config_path: Optional[Path]) -> Dict[str, Any]:
    del config_path
    source_lock_amendment = validate_source_lock_amendment(project_root)
    store = load_verified_methane_window_store(project_root)
    protocol = load_frozen_protocol(project_root)
    methane = protocol["data"]["methane"]
    train = load_methane_arrays(project_root, pools={"D_b_tr"}, include_labels=True)
    select = load_methane_arrays(project_root, pools={"D_b_sel"}, include_labels=True)
    if set(train.labels.tolist()) != {0, 1} or set(select.labels.tolist()) != {0, 1}:
        raise WorkflowExecutionError("Methane train and selection pools require both classes")
    stride = int(methane["stride_seconds"])
    train_features, feature_names = _statistical_features(train, stride)
    select_features, selected_names = _statistical_features(select, stride)
    if feature_names != selected_names:
        raise WorkflowExecutionError("Methane classical feature names drifted")
    standardizer = CausalStandardizer(feature_names).fit(train_features, pool="D_b_tr")
    hgb = MethaneHGB(feature_names, random_state=1701, max_iter=100).fit(
        standardizer.transform(train_features), train.labels, pool="D_b_tr"
    )
    continuous_count = next(
        index for index, name in enumerate(train.feature_names) if name.endswith("_observed_mask")
    )
    ch4_history = train.history[:, :, :continuous_count][:, :, 0:1]
    rule = _fit_persistence_rule(
        ch4_history,
        concentration_threshold=float(methane["risk_concentration_threshold"]),
    )
    select_scores = {
        "S1-RULE": rule.score(select.history[:, :, :continuous_count][:, :, 0:1]),
        "S1-HGB": hgb.predict_proba(standardizer.transform(select_features)),
    }
    candidates = _candidate_table(
        select.metadata,
        select.labels,
        select_scores,
        stride_seconds=stride,
        methane_contract=methane,
    )
    result = select_s1_threshold(
        candidates,
        pool="D_b_sel",
        max_false_alarms_per_hour=float(
            protocol["training"]["methane"]["max_false_alarms_per_hour"]
        ),
    )
    selected_id = result.selected_id or result.diagnostic_id
    selected_row = candidates.loc[candidates["candidate_id"] == selected_id].iloc[0]
    family_selections = {}
    for family_id, family_candidates in candidates.groupby("family_id", sort=True):
        family_result = select_s1_threshold(
            family_candidates,
            pool="D_b_sel",
            max_false_alarms_per_hour=float(
                protocol["training"]["methane"]["max_false_alarms_per_hour"]
            ),
        )
        family_id_selected = family_result.selected_id or family_result.diagnostic_id
        family_row = family_candidates.loc[
            family_candidates["candidate_id"] == family_id_selected
        ].iloc[0]
        family_selections[str(family_id)] = {
            "status": family_result.status,
            "candidate_id": family_id_selected,
            "threshold": float(family_row["threshold"]),
            "event_macro_f1": float(family_row["event_macro_f1"]),
            "false_alarms_per_hour": float(family_row["false_alarms_per_hour"]),
        }
    output_root = project_root / "runs/S1/baselines"
    output_root.mkdir(parents=True, exist_ok=True)
    model_path = output_root / "baseline_models.pkl"
    write_once_bytes(
        model_path,
        pickle.dumps(
            {
                "schema_version": 1,
                "rule": rule,
                "hgb": hgb,
                "standardizer": standardizer,
                "feature_names": feature_names,
                "stride_seconds": stride,
            },
            protocol=4,
        ),
    )
    metrics_path = output_root / "selection_candidates.csv"
    write_once_bytes(metrics_path, candidates.to_csv(index=False).encode("utf-8"))
    selection_path = output_root / "selection.json"
    write_once_json(
        selection_path,
        {
            "schema_version": 1,
            "status": result.status,
            "selected_candidate_id": result.selected_id,
            "diagnostic_candidate_id": result.diagnostic_id,
            "selected_or_diagnostic_family_id": str(selected_row["family_id"]),
            "selected_or_diagnostic_threshold": float(selected_row["threshold"]),
            "family_selections": family_selections,
            "selection_pool": "D_b_sel",
            "reason": result.reason,
            "rule_missing_value": rule.missing_value,
            "rule_missing_value_fit_pool": "D_b_tr",
            "rule_transition_scale": rule.transition_scale,
            "source_lock_amendment": source_lock_amendment,
            "test_labels_opened": False,
        },
    )
    probe = run_root / "slurm_environment_probe.json"
    if not probe.is_file():
        raise WorkflowExecutionError("S1 baseline step lacks its Slurm environment probe")
    manifest_path = output_root / "run_manifest.json"
    write_once_json(
        manifest_path,
        {
            "schema_version": 1,
            "status": "pass",
            "step_id": "E120",
            "family_ids": ["S1-RULE", "S1-HGB"],
            "selected_reference_status": result.status,
            "selected_reference_family_id": str(selected_row["family_id"]),
            "selected_reference_threshold": float(selected_row["threshold"]),
            "model_path": model_path.relative_to(project_root).as_posix(),
            "model_sha256": sha256_file(model_path),
            "selection_path": selection_path.relative_to(project_root).as_posix(),
            "selection_sha256": sha256_file(selection_path),
            "rule_missing_value": rule.missing_value,
            "rule_missing_value_fit_pool": "D_b_tr",
            "rule_transition_scale": rule.transition_scale,
            "source_lock_amendment": source_lock_amendment,
            "window_store": store,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "environment_probe_sha256": sha256_file(probe),
            "test_labels_opened": False,
            "remote_closeout_status": "pending_local_post_job_closeout",
            "finished_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return {
        "status": "pass",
        "output_paths": [output_root.relative_to(project_root).as_posix()],
        "details": {"selected_reference_status": result.status},
    }


def s1_gru_seed(array_index: int) -> int:
    if array_index < 0 or array_index >= len(TRAINING_SEEDS):
        raise WorkflowExecutionError("E122 array index is outside the frozen range")
    return TRAINING_SEEDS[array_index]


def _pilot_training_decisions(project_root: Path) -> Dict[str, Any]:
    payload = load_json(project_root / "evidence/pilot/resource_pilot.json")
    decisions = payload.get("decisions", {})
    if payload.get("status") != "pass" or not isinstance(decisions, dict):
        raise WorkflowExecutionError("Resource pilot decisions are unavailable")
    return decisions


def train_s1_gru(
    project_root: Path,
    run_root: Path,
    array_index: int,
    *,
    max_updates_override: Optional[int] = None,
) -> Dict[str, Any]:
    seed = s1_gru_seed(array_index)
    protocol = load_frozen_protocol(project_root)
    train = load_methane_arrays(project_root, pools={"D_b_tr"}, include_labels=True)
    select = load_methane_arrays(project_root, pools={"D_b_sel"}, include_labels=True)
    if set(train.labels.tolist()) != {0, 1} or set(select.labels.tolist()) != {0, 1}:
        raise WorkflowExecutionError("GRU train and selection pools require both classes")
    preprocessor = fit_sequence_preprocessor(train.history, train.feature_names)
    train_values = transform_sequences(train.history, preprocessor)
    select_values = transform_sequences(select.history, preprocessor)
    decisions = _pilot_training_decisions(project_root)
    batch = int(decisions["micro_batch_size"]["methane"])
    accumulation = int(decisions["gradient_accumulation"]["methane"])
    if batch * accumulation != int(protocol["training"]["methane"]["effective_batch_size"]):
        raise WorkflowExecutionError("GRU pilot decision changed the locked effective batch")
    max_updates = int(protocol["training"]["methane"]["max_updates"])
    if max_updates_override is not None:
        max_updates = int(max_updates_override)
    if max_updates <= 0:
        raise WorkflowExecutionError("GRU max updates must be positive")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = MethaneGRU(
        train.feature_names,
        hidden_size=int(protocol["models"]["methane"]["hidden_size"]),
        layers=int(protocol["models"]["methane"]["layers"]),
        dropout=float(protocol["models"]["methane"]["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(protocol["training"]["methane"]["learning_rate"])
    )
    loss_function = nn.BCEWithLogitsLoss()
    use_amp = str(decisions["precision"]).lower() in {
        "amp",
        "amp_fp16",
        "fp16",
        "mixed_fp16",
    } and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    generator = np.random.default_rng(seed)
    metrics = []
    model.train()
    for update in range(max_updates):
        optimizer.zero_grad(set_to_none=True)
        cumulative = 0.0
        for _ in range(accumulation):
            indices = generator.integers(0, len(train_values), size=batch)
            inputs = torch.from_numpy(train_values[indices]).to(device)
            targets = torch.from_numpy(train.labels[indices].astype(np.float32)).to(device)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                loss = loss_function(model(inputs), targets) / accumulation
            scaler.scale(loss).backward()
            cumulative += float(loss.detach().cpu())
        scaler.step(optimizer)
        scaler.update()
        if update == 0 or (update + 1) % 100 == 0 or update + 1 == max_updates:
            metrics.append({"optimizer_update": update + 1, "train_loss": cumulative})
    model.eval()
    with torch.inference_mode():
        selection_scores = torch.sigmoid(
            model(torch.from_numpy(select_values).to(device))
        ).cpu().numpy()
    stride = int(protocol["data"]["methane"]["stride_seconds"])
    candidates = _candidate_table(
        select.metadata,
        select.labels,
        {"S1-GRU": selection_scores},
        stride_seconds=stride,
        methane_contract=protocol["data"]["methane"],
    )
    selection = select_s1_threshold(
        candidates,
        pool="D_b_sel",
        max_false_alarms_per_hour=float(
            protocol["training"]["methane"]["max_false_alarms_per_hour"]
        ),
    )
    selected_id = selection.selected_id or selection.diagnostic_id
    selected_row = candidates.loc[candidates["candidate_id"] == selected_id].iloc[0]
    output_root = project_root / f"runs/S1/gru/S1-GRU/seed-{seed}"
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint = output_root / "best.pt"
    buffer = io.BytesIO()
    torch.save(
        {
            "schema_version": 1,
            "family_id": "S1-GRU",
            "train_seed": seed,
            "feature_names": list(train.feature_names),
            "preprocessor": preprocessor,
            "model_state_dict": model.cpu().state_dict(),
        },
        buffer,
    )
    write_once_bytes(checkpoint, buffer.getvalue())
    write_once_bytes(
        output_root / "metrics.csv",
        pd.DataFrame(metrics).to_csv(index=False).encode("utf-8"),
    )
    write_once_bytes(
        output_root / "selection_candidates.csv",
        candidates.to_csv(index=False).encode("utf-8"),
    )
    probe = run_root / "slurm_environment_probe.json"
    if not probe.is_file():
        raise WorkflowExecutionError("S1 GRU step lacks its Slurm environment probe")
    manifest = {
        "schema_version": 1,
        "status": "pass",
        "step_id": "E122",
        "family_id": "S1-GRU",
        "train_seed": seed,
        "selection_status": selection.status,
        "selection_threshold": float(selected_row["threshold"]),
        "selection_metric_value": float(selected_row["event_macro_f1"]),
        "selection_pool": "D_b_sel",
        "checkpoint_path": checkpoint.relative_to(project_root).as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint),
        "max_optimizer_updates": max_updates,
        "effective_batch_size": batch * accumulation,
        "protocol_sha256": sha256_file(
            project_root / "configs/protocol_lock.pretest.yaml"
        ),
        "environment_probe_sha256": sha256_file(probe),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "test_labels_opened": False,
        "remote_closeout_status": "pending_local_post_job_closeout",
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    write_once_json(output_root / "run_manifest.json", manifest)
    return manifest
