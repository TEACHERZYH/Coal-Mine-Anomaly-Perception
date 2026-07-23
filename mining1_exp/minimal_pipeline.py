from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Union
import uuid

import numpy as np
import pandas as pd
import torch
from torch import nn

from .data.groups import assign_raw_group_ids, audit_duplicates
from .data.episodes import make_skeleton_item_id
from .data.manifests import (
    validate_file_manifest,
    validate_split_manifest,
    write_parquet_atomic,
)
from .data.splits import FEWSHOT_COLUMNS, assign_group_pools, build_fewshot_manifest
from .evaluate.event_metrics import evaluate_episode_events, evaluate_methane_events
from .governance.immutable import write_once_bytes, write_once_json
from .models.episode_fusion import (
    CalibratedLogitFusion,
    EventMemoryPolicy,
    MemoryPolicyConfig,
    ReliabilityGraphFusion,
    RobustQualityScaler,
    mean_fusion,
    reliability_brier_loss,
    reliability_supervision,
    validate_observed_pair_edges,
)
from .models.methane import (
    CausalStandardizer,
    MethaneGRU,
    MethaneHGB,
    PersistenceRiskRule,
    build_causal_stat_features,
)
from .models.rgbt_branches import IndependentT1Branches, build_branch_batch
from .models.yolo_adapter import (
    BinaryProbabilityCalibrator,
    YoloV8nAdapter,
    aggregate_concept_scores,
)
from .provenance import canonical_json_sha256, sha256_bytes, sha256_file


PathLike = Union[str, Path]
PIPELINE_STEP = "I080"
POOL_WEIGHTS = {"D_b_tr": 1.0, "D_b_prob": 1.0, "D_e_tr": 1.0, "D_e_te": 1.0}
PARQUET_OUTPUTS = (
    "duplicate_report.parquet",
    "file_manifest.parquet",
    "split_manifest.parquet",
    "fewshot_manifest.parquet",
    "branch_predictions.parquet",
    "methane_predictions.parquet",
    "episode_predictions.parquet",
)
JSON_OUTPUTS = ("metrics.json", "summary.json")
TEXT_OUTPUTS = ("execution.log",)
RECEIPT_NAME = "integration_receipt.json"
LOCK_NAME = ".I080.lock"
FORBIDDEN_PREDICTION_TOKENS = ("truth", "label", "target", "test_metric")
EXPECTED_MODELS = (
    "visible_yolov8n_adapter_synthetic_backend",
    "thermal_yolov8n_adapter_synthetic_backend",
    "concept_max_aggregation_and_platt_calibration",
    "methane_persistence_rule",
    "methane_hgb",
    "methane_gru_one_step",
    "episode_mean",
    "episode_calibrated_logit_one_step",
    "episode_reliability_graph_one_step",
    "episode_memory_and_event_evaluation",
)


class MinimalPipelineError(ValueError):
    """Raised when the local synthetic integration run violates its contract."""


class _SyntheticDetector(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, 2, kernel_size=1)
        with torch.no_grad():
            self.projection.weight.fill_(scale)
            self.projection.bias.copy_(torch.tensor([-0.2, 0.2]))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(images).mean(dim=(2, 3))


def _artifact_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _write_parquet_once(frame: pd.DataFrame, path: Path) -> None:
    target = path.resolve()
    staged = target.with_name(f".{target.name}.{uuid.uuid4().hex}.stage")
    try:
        write_parquet_atomic(frame, staged)
        try:
            os.link(staged, target)
        except FileExistsError as exc:
            raise MinimalPipelineError(
                f"refusing to overwrite integration artifact: {target.name}"
            ) from exc
    finally:
        staged.unlink(missing_ok=True)


def _build_minimal_manifests() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for index in range(256):
        record_id = f"synthetic_{index:03d}"
        rows.append(
            {
                "dataset_id": "synthetic_minimal",
                "record_id": record_id,
                "archive_id": "synthetic_in_memory",
                "relative_path": f"memory/{record_id}.bin",
                "modality": "synthetic",
                "source_group": f"source_{index:03d}",
                "pair_id": None,
                "sequence_id": f"sequence_{index:03d}",
                "timestamp_or_order": index,
                "label_summary_json": json.dumps({}, sort_keys=True),
                "byte_size": 0,
                "sha256": sha256_bytes(record_id.encode("ascii")),
            }
        )
    candidates = assign_raw_group_ids(pd.DataFrame(rows), ["dataset_id", "source_group"])
    duplicate_report = audit_duplicates(candidates)
    ontology_hash = sha256_bytes(b"synthetic-minimal-ontology-v1")
    dedup_hash = canonical_json_sha256(duplicate_report.to_dict("records"))
    all_splits = assign_group_pools(
        candidates,
        pool_weights=POOL_WEIGHTS,
        seed=1701,
        split_version="synthetic_i080_v1",
        ontology_hash=ontology_hash,
        dedup_report_hash=dedup_hash,
    )
    selected = (
        all_splits.sort_values(["pool", "raw_group_id"])
        .groupby("pool", sort=True, group_keys=False)
        .head(2)
        .reset_index(drop=True)
    )
    if set(selected["pool"]) != set(POOL_WEIGHTS):
        raise MinimalPipelineError("synthetic split did not cover every required smoke pool")
    split_manifest = validate_split_manifest(selected)
    selected_keys = set(zip(selected["dataset_id"], selected["record_id"]))
    file_manifest = candidates.loc[
        [
            (dataset_id, record_id) in selected_keys
            for dataset_id, record_id in zip(
                candidates["dataset_id"], candidates["record_id"]
            )
        ],
        [
            "dataset_id",
            "record_id",
            "archive_id",
            "relative_path",
            "modality",
            "raw_group_id",
            "pair_id",
            "sequence_id",
            "timestamp_or_order",
            "label_summary_json",
            "byte_size",
            "sha256",
        ],
    ].reset_index(drop=True)
    file_manifest = validate_file_manifest(file_manifest)
    group_classes = {
        (str(row.dataset_id), str(row.raw_group_id)): ("helmet", "person")
        for row in split_manifest.itertuples(index=False)
    }
    fewshot_manifest = build_fewshot_manifest(
        split_manifest,
        direction_id="synthetic_direction",
        subset_seeds=(1701, 1702, 1703),
        group_classes=group_classes,
        parent_manifest_hash=canonical_json_sha256(
            split_manifest.sort_values(["dataset_id", "record_id"]).to_dict("records")
        ),
    )
    return duplicate_report, file_manifest, split_manifest, fewshot_manifest


def _run_branch_smoke(split_manifest: pd.DataFrame) -> pd.DataFrame:
    torch.manual_seed(1701)
    identifiers = split_manifest.loc[
        split_manifest["pool"] == "D_b_prob", "record_id"
    ].astype(str).tolist()
    visible_images = [np.full((32, 32, 3), 64 + 64 * index, dtype=np.uint8) for index in range(2)]
    thermal_images = [np.full((32, 32), 48 + 96 * index, dtype=np.uint8) for index in range(2)]
    visible_batch = build_branch_batch(
        visible_images, record_ids=identifiers, modality="visible_only"
    )
    thermal_batch = build_branch_batch(
        thermal_images, record_ids=identifiers, modality="thermal_only"
    )
    branches = IndependentT1Branches(
        YoloV8nAdapter(_SyntheticDetector(0.05)),
        YoloV8nAdapter(_SyntheticDetector(-0.03)),
        training_seed=1701,
    ).eval()
    with torch.no_grad():
        visible_model_mean = float(
            torch.sigmoid(branches.forward_visible(visible_batch)).mean().item()
        )
        thermal_model_mean = float(
            torch.sigmoid(branches.forward_thermal(thermal_batch)).mean().item()
        )
    detections = pd.DataFrame(
        [
            {"record_id": identifiers[0], "concept_id": "helmet", "score_raw": 0.12},
            {"record_id": identifiers[0], "concept_id": "person", "score_raw": 0.25},
            {"record_id": identifiers[1], "concept_id": "helmet", "score_raw": 0.88},
            {"record_id": identifiers[1], "concept_id": "person", "score_raw": 0.93},
        ]
    )
    frames = []
    calibration_scores = np.asarray([0.02, 0.08, 0.15, 0.30, 0.65, 0.80, 0.92, 0.98])
    calibration_labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    for modality, offset, model_mean in (
        ("visible", 0.0, visible_model_mean),
        ("thermal", -0.03, thermal_model_mean),
    ):
        raw = aggregate_concept_scores(
            detections.assign(score_raw=lambda frame: np.clip(frame["score_raw"] + offset, 0, 1)),
            record_ids=identifiers,
            concept_ids=("helmet", "person"),
            modality=modality,
            post_nms=True,
        )
        calibrator = BinaryProbabilityCalibrator("platt").fit(
            np.clip(calibration_scores + offset, 0, 1),
            calibration_labels,
            pool="D_b_prob",
            negatives_verified=True,
        )
        raw["step_probability"] = calibrator.predict(raw["step_score_raw"])
        raw["model_probability_mean"] = model_mean
        frames.append(raw)
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(["record_id", "concept_id", "modality"]).any():
        raise MinimalPipelineError("branch prediction primary key is not unique")
    return result


def _run_methane_smoke() -> tuple[pd.DataFrame, dict[str, Any]]:
    random = np.random.default_rng(1701)
    labels = np.asarray([0, 1] * 12, dtype=np.int64)
    history = random.normal(0.2, 0.1, size=(24, 10, 2))
    history += labels[:, None, None] * 0.9
    features, feature_names = build_causal_stat_features(
        history, sample_period_seconds=30
    )
    train_slice = slice(0, 20)
    inference_slice = slice(20, 24)
    standardizer = CausalStandardizer(feature_names).fit(
        features[train_slice], pool="D_b_tr"
    )
    standardized_train = standardizer.transform(features[train_slice])
    standardized_inference = standardizer.transform(features[inference_slice])
    hgb = MethaneHGB(feature_names, random_state=1701, max_iter=5).fit(
        standardized_train, labels[train_slice], pool="D_b_tr"
    )
    rule = PersistenceRiskRule(concentration_threshold=0.8, transition_scale=0.2)
    torch.manual_seed(1701)
    gru = MethaneGRU(("sensor_0", "sensor_1"), hidden_size=64, layers=1, dropout=0.20)
    optimizer = torch.optim.SGD(gru.parameters(), lr=0.01)
    sequence_train = torch.tensor(history[train_slice], dtype=torch.float32)
    sequence_inference = torch.tensor(history[inference_slice], dtype=torch.float32)
    target = torch.tensor(labels[train_slice], dtype=torch.float32)
    optimizer.zero_grad()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        gru(sequence_train), target
    )
    loss.backward()
    optimizer.step()
    gru.eval()
    with torch.no_grad():
        gru_probabilities = gru.probabilities(sequence_inference).cpu().numpy()
    frame = pd.DataFrame(
        {
            "window_id": [f"methane_{index}" for index in range(4)],
            "rule_probability": rule.score(history[inference_slice]),
            "hgb_probability": hgb.predict_proba(standardized_inference),
            "gru_probability": gru_probabilities,
        }
    )
    fit_hash = sha256_bytes(np.ascontiguousarray(history[train_slice]).tobytes())
    inference_hash = sha256_bytes(
        np.ascontiguousarray(history[inference_slice]).tobytes()
    )
    return frame, {
        "fit_input_sha256": fit_hash,
        "inference_input_sha256": inference_hash,
        "fit_and_inference_disjoint": fit_hash != inference_hash,
        "fit_rows": 20,
        "inference_rows": 4,
    }


def _run_episode_smoke() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    torch.manual_seed(1701)
    training_probabilities = torch.tensor(
        [
            [0.12, 0.18],
            [0.78, 0.82],
            [0.20, 0.24],
            [0.86, 0.80],
            [0.16, 0.22],
            [0.90, 0.85],
            [0.25, 0.30],
            [0.88, 0.92],
        ],
        dtype=torch.float32,
    )
    inference_probabilities = torch.tensor(
        [
            [0.10, 0.20],
            [0.88, 0.84],
            [0.92, 0.90],
            [0.94, 0.91],
            [0.96, 0.93],
            [0.90, 0.88],
            [0.20, 0.25],
            [0.15, float("nan")],
        ],
        dtype=torch.float32,
    )
    training_availability = torch.isfinite(training_probabilities)
    inference_availability = torch.isfinite(inference_probabilities)
    mean_probability, mean_abstained = mean_fusion(
        inference_probabilities, inference_availability
    )
    logit = CalibratedLogitFusion(node_count=2)
    logit_optimizer = torch.optim.SGD(logit.parameters(), lr=0.01)
    training_target = torch.tensor([0, 1, 1, 1, 1, 1, 0, 0], dtype=torch.float32)
    logit_optimizer.zero_grad()
    training_logit_probability, _ = logit(
        training_probabilities, training_availability
    )
    torch.nn.functional.binary_cross_entropy(
        training_logit_probability, training_target
    ).backward()
    logit_optimizer.step()
    logit_probability, _ = logit(inference_probabilities, inference_availability)

    training_quality = np.asarray(
        [[[0.75, 0.15, 1.0], [0.65, 0.25, 1.0]]] * len(training_probabilities),
        dtype=np.float32,
    )
    inference_quality = np.asarray(
        [[[0.8, 0.1, 1.0], [0.7, 0.2, 1.0]]] * len(inference_probabilities),
        dtype=np.float32,
    )
    inference_quality[-1, 1] = np.nan
    scaler = RobustQualityScaler(
        ("quality_mean", "quality_spread", "quality_valid")
    ).fit(training_quality, pool="D_e_tr")
    scaled_training_quality, _ = scaler.transform(training_quality)
    scaled_inference_quality, _ = scaler.transform(inference_quality)
    edge_index = validate_observed_pair_edges(
        pd.DataFrame(
            [
                {"source_node": 0, "target_node": 1, "edge_provenance": "observed_pair"},
                {"source_node": 1, "target_node": 0, "edge_provenance": "observed_pair"},
            ]
        ),
        node_count=2,
    )
    graph = ReliabilityGraphFusion(
        node_count=2,
        quality_feature_names=("quality_mean", "quality_spread", "quality_valid"),
        concept_dim=2,
        pair_feature_names=("pair_time_delta", "pair_quality"),
        graph_layers=2,
    )
    training_pair_features = torch.tensor([[[0.0, 0.9], [0.0, 0.9]]] * 8)
    inference_pair_features = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]] * 8)
    training_concept_embedding = torch.tensor([[0.3, 0.7]] * 8)
    inference_concept_embedding = torch.tensor([[0.2, 0.8]] * 8)
    graph_optimizer = torch.optim.SGD(graph.parameters(), lr=0.01)
    graph_optimizer.zero_grad()
    training_output = graph(
        training_probabilities,
        torch.tensor(scaled_training_quality, dtype=torch.float32),
        training_availability,
        training_concept_embedding,
        edge_index=edge_index,
        pair_features=training_pair_features,
        abstention_threshold=0.10,
    )
    reliability_target = reliability_supervision(
        training_target,
        training_probabilities,
        training_availability,
        pool="D_e_tr",
    )
    graph_loss = reliability_brier_loss(
        training_output.reliability, reliability_target, training_availability
    ) + torch.nn.functional.binary_cross_entropy(
        training_output.probability, training_target
    )
    graph_loss.backward()
    graph_optimizer.step()
    output = graph(
        inference_probabilities,
        torch.tensor(scaled_inference_quality, dtype=torch.float32),
        inference_availability,
        inference_concept_embedding,
        edge_index=edge_index,
        pair_features=inference_pair_features,
        abstention_threshold=0.10,
    )

    memory = EventMemoryPolicy(
        MemoryPolicyConfig(
            beta=0.70,
            memory_k=2,
            low_threshold=0.30,
            high_threshold=0.60,
            alarm_threshold=0.80,
        )
    )
    trace = memory.run(
        mean_probability.detach().tolist(),
        mean_abstained.tolist(),
        alarm_allowed=True,
    )
    skeleton_nonce = b"i080-synthetic-sealed-nonce-v1"
    frame = pd.DataFrame(
        {
            "skeleton_item_id": [
                make_skeleton_item_id(
                    skeleton_nonce,
                    record_id=f"synthetic_episode_record_{index}",
                    concept_id="ppe",
                    node_id="visible_thermal_pair",
                )
                for index in range(8)
            ],
            "mean_probability": mean_probability.detach().numpy(),
            "logit_probability": logit_probability.detach().numpy(),
            "graph_probability": output.probability.detach().numpy(),
            "abstained": output.abstained.detach().numpy(),
            "effective_state": [row.state for row in trace],
            "memory_probability": [row.memory for row in trace],
        }
    )
    metrics = evaluate_episode_events(
        truth_event=[0, 0, 0, 1, 1, 1, 0, 0],
        effective_states=frame["effective_state"].tolist(),
        episode_ids=["synthetic_episode"] * 8,
        step_indices=list(range(8)),
    )
    fit_hash = sha256_bytes(
        np.ascontiguousarray(training_probabilities.detach().numpy()).tobytes()
    )
    inference_hash = sha256_bytes(
        np.ascontiguousarray(inference_probabilities.detach().numpy()).tobytes()
    )
    return frame, metrics, {
        "fit_input_sha256": fit_hash,
        "inference_input_sha256": inference_hash,
        "fit_and_inference_disjoint": fit_hash != inference_hash,
        "fit_rows": 8,
        "inference_rows": 8,
    }


def _validate_prediction_columns(frame: pd.DataFrame, name: str) -> None:
    forbidden = sorted(
        column
        for column in frame.columns
        if any(token in str(column).lower() for token in FORBIDDEN_PREDICTION_TOKENS)
    )
    if forbidden:
        raise MinimalPipelineError(f"{name} contains forbidden prediction columns: {forbidden}")


def _validate_probability_table(
    frame: pd.DataFrame,
    *,
    name: str,
    required_columns: tuple[str, ...],
    probability_columns: tuple[str, ...],
    primary_key: tuple[str, ...],
) -> None:
    missing = sorted(set(required_columns).difference(frame.columns))
    if missing or frame.empty:
        raise MinimalPipelineError(f"{name} is empty or missing columns: {missing}")
    if frame.duplicated(list(primary_key)).any():
        raise MinimalPipelineError(f"{name} primary key is not unique")
    for column in probability_columns:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise MinimalPipelineError(f"{name}.{column} is not a finite probability")


def _validate_boundary_hashes(summary: Mapping[str, Any], field: str) -> None:
    boundary = summary.get(field)
    if not isinstance(boundary, dict) or boundary.get("fit_and_inference_disjoint") is not True:
        raise MinimalPipelineError(f"{field} does not prove disjoint smoke inputs")
    fit_hash = str(boundary.get("fit_input_sha256", ""))
    inference_hash = str(boundary.get("inference_input_sha256", ""))
    if (
        re.fullmatch(r"[0-9a-f]{64}", fit_hash) is None
        or re.fullmatch(r"[0-9a-f]{64}", inference_hash) is None
        or fit_hash == inference_hash
        or int(boundary.get("fit_rows", 0)) <= 0
        or int(boundary.get("inference_rows", 0)) <= 0
    ):
        raise MinimalPipelineError(f"{field} fit/inference evidence is invalid")


def validate_minimal_pipeline_run(run_root: PathLike) -> dict[str, Any]:
    unresolved_root = Path(run_root)
    if unresolved_root.is_symlink():
        raise MinimalPipelineError("integration run root may not be a symbolic link")
    root = unresolved_root.resolve()
    receipt_path = root / RECEIPT_NAME
    if not receipt_path.is_file():
        raise MinimalPipelineError("integration receipt is missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema_version") != 1
        or receipt.get("step_id") != PIPELINE_STEP
        or receipt.get("status") != "pass"
    ):
        raise MinimalPipelineError("integration receipt has an invalid state")
    try:
        created_at = datetime.fromisoformat(
            str(receipt.get("created_at", "")).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise MinimalPipelineError("integration receipt timestamp is invalid") from exc
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise MinimalPipelineError("integration receipt timestamp lacks a timezone")
    records = receipt.get("artifacts")
    if not isinstance(records, list) or not records:
        raise MinimalPipelineError("integration receipt has no artifact hashes")
    required_artifacts = set((*PARQUET_OUTPUTS, *JSON_OUTPUTS, *TEXT_OUTPUTS))
    recorded_paths = [str(record.get("path", "")) for record in records]
    if (
        len(recorded_paths) != len(set(recorded_paths))
        or set(recorded_paths) != required_artifacts
    ):
        raise MinimalPipelineError("integration receipt does not close the exact output set")
    expected = {RECEIPT_NAME}
    for record in records:
        relative = Path(str(record.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
            raise MinimalPipelineError("integration artifact path is unsafe")
        artifact = root / relative
        if not artifact.is_file() or artifact.is_symlink():
            raise MinimalPipelineError(f"integration artifact is missing: {relative}")
        if artifact.stat().st_size != record.get("bytes") or sha256_file(artifact) != record.get(
            "sha256"
        ):
            raise MinimalPipelineError(f"integration artifact hash drift: {relative}")
        expected.add(relative.as_posix())
    unexpected_entries = [
        path.name
        for path in root.iterdir()
        if path.name != LOCK_NAME and (not path.is_file() or path.is_symlink())
    ]
    if unexpected_entries:
        raise MinimalPipelineError("integration run contains an untracked directory or link")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.iterdir()
        if path.is_file() and path.name != LOCK_NAME
    }
    if actual != expected:
        raise MinimalPipelineError("integration run contains missing or untracked files")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if receipt.get("summary_sha256") != sha256_file(root / "summary.json"):
        raise MinimalPipelineError("integration summary hash does not match the receipt")
    if (
        summary.get("status") != "pass"
        or summary.get("schema_version") != 1
        or summary.get("step_id") != PIPELINE_STEP
        or summary.get("synthetic_only") is not True
        or summary.get("fit_and_inference_samples_disjoint") is not True
        or summary.get("full_local_dataset_extractions") != 0
        or summary.get("remote_connections") != 0
        or summary.get("slurm_jobs_created") != 0
        or tuple(summary.get("models_exercised", ())) != EXPECTED_MODELS
    ):
        raise MinimalPipelineError("integration summary violates the local synthetic boundary")
    try:
        summary_created_at = datetime.fromisoformat(
            str(summary.get("created_at", "")).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise MinimalPipelineError("integration summary timestamp is invalid") from exc
    if summary_created_at.tzinfo is None or summary_created_at.utcoffset() is None:
        raise MinimalPipelineError("integration summary timestamp lacks a timezone")
    _validate_boundary_hashes(summary, "methane_boundary")
    _validate_boundary_hashes(summary, "episode_boundary")
    file_manifest = validate_file_manifest(pd.read_parquet(root / "file_manifest.parquet"))
    split_manifest = validate_split_manifest(pd.read_parquet(root / "split_manifest.parquet"))
    group_counts = split_manifest.groupby("pool")["raw_group_id"].nunique()
    if (
        len(file_manifest) != len(split_manifest)
        or set(group_counts.index) != set(POOL_WEIGHTS)
        or (group_counts > 2).any()
        or summary.get("max_groups_per_pool") != int(group_counts.max())
        or summary.get("pool_group_counts")
        != {str(pool): int(count) for pool, count in group_counts.items()}
    ):
        raise MinimalPipelineError("integration sample exceeds the two-group-per-pool limit")
    duplicate_report = pd.read_parquet(root / "duplicate_report.parquet")
    if set(duplicate_report.columns) != {
        "left_dataset_id",
        "left_record_id",
        "right_dataset_id",
        "right_record_id",
        "duplicate_type",
        "phash_hamming_distance",
    }:
        raise MinimalPipelineError("duplicate report schema is invalid")
    fewshot = pd.read_parquet(root / "fewshot_manifest.parquet")
    if (
        not FEWSHOT_COLUMNS.issubset(fewshot.columns)
        or fewshot.empty
        or set(fewshot["subset_seed"]) != {1701, 1702, 1703}
        or set(fewshot["ratio_percent"]) != {10}
    ):
        raise MinimalPipelineError("few-shot smoke manifest is invalid")
    branch = pd.read_parquet(root / "branch_predictions.parquet")
    _validate_prediction_columns(branch, "branch_predictions")
    _validate_probability_table(
        branch,
        name="branch_predictions",
        required_columns=(
            "record_id",
            "concept_id",
            "modality",
            "step_score_raw",
            "step_probability",
            "model_probability_mean",
        ),
        probability_columns=(
            "step_score_raw",
            "step_probability",
            "model_probability_mean",
        ),
        primary_key=("record_id", "concept_id", "modality"),
    )
    methane = pd.read_parquet(root / "methane_predictions.parquet")
    _validate_probability_table(
        methane,
        name="methane_predictions",
        required_columns=(
            "window_id",
            "rule_probability",
            "hgb_probability",
            "gru_probability",
        ),
        probability_columns=(
            "rule_probability",
            "hgb_probability",
            "gru_probability",
        ),
        primary_key=("window_id",),
    )
    episode = pd.read_parquet(root / "episode_predictions.parquet")
    _validate_prediction_columns(episode, "episode_predictions")
    _validate_probability_table(
        episode,
        name="episode_predictions",
        required_columns=(
            "skeleton_item_id",
            "mean_probability",
            "logit_probability",
            "graph_probability",
            "abstained",
            "effective_state",
            "memory_probability",
        ),
        probability_columns=(
            "mean_probability",
            "logit_probability",
            "graph_probability",
            "memory_probability",
        ),
        primary_key=("skeleton_item_id",),
    )
    if (
        not episode["skeleton_item_id"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all()
        or not set(episode["effective_state"]).issubset(
            {"normal", "attention", "prewarning", "alarm", "abstain"}
        )
        or not episode["abstained"].isin([True, False]).all()
    ):
        raise MinimalPipelineError("episode prediction state or join key is invalid")
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    if set(metrics) != {"methane_event", "episode_event"}:
        raise MinimalPipelineError("integration metrics bundle is incomplete")
    for name, required_counts in {
        "methane_event": ("truth_event_count", "prediction_event_count", "matched_event_count"),
        "episode_event": ("episode_count", "eligible_step_count", "matched_event_count"),
    }.items():
        if any(int(metrics[name].get(field, -1)) < 0 for field in required_counts):
            raise MinimalPipelineError(f"{name} contains an invalid diagnostic count")
    execution_log = (root / "execution.log").read_text(encoding="ascii")
    if not execution_log.strip() or "no dataset extraction" not in execution_log:
        raise MinimalPipelineError("integration execution log is incomplete")
    return {
        "status": "pass",
        "step_id": PIPELINE_STEP,
        "receipt_sha256": sha256_file(receipt_path),
        "artifact_count": len(records),
        "mode": "validated_existing",
    }


def run_minimal_pipeline(run_root: PathLike) -> dict[str, Any]:
    unresolved_root = Path(run_root)
    if unresolved_root.is_symlink():
        raise MinimalPipelineError("integration run root may not be a symbolic link")
    root = unresolved_root.resolve()
    if root == Path.cwd().resolve() or root.parent == root:
        raise MinimalPipelineError("integration run root must be a dedicated subdirectory")
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / LOCK_NAME
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise MinimalPipelineError("integration run is already locked") from exc
    os.close(descriptor)
    try:
        if (root / RECEIPT_NAME).exists():
            return validate_minimal_pipeline_run(root)
        existing = [path for path in root.iterdir() if path.name != LOCK_NAME]
        if existing:
            raise MinimalPipelineError("new integration run root must be empty")

        duplicate_report, file_manifest, split_manifest, fewshot_manifest = (
            _build_minimal_manifests()
        )
        branch_predictions = _run_branch_smoke(split_manifest)
        methane_predictions, methane_boundary = _run_methane_smoke()
        episode_predictions, episode_metrics, episode_boundary = _run_episode_smoke()
        _validate_prediction_columns(branch_predictions, "branch_predictions")
        _validate_prediction_columns(episode_predictions, "episode_predictions")
        methane_metrics = evaluate_methane_events(
            raw_positions_seconds=[100, 110, 120, 130],
            raw_values=[2.0, 2.0, 0.0, 0.0],
            risk_threshold=1.0,
            prediction_positions_seconds=[80, 90, 100, 110],
            prediction_scores=[0.9, 0.0, 0.9, 0.0],
            score_threshold=0.5,
            raw_continuity_seconds=10,
            prediction_continuity_seconds=5,
            event_merge_gap_seconds=0,
            horizon_seconds=30,
            valid_observed_sensor_hours=2.0,
        )
        frames = {
            "duplicate_report.parquet": duplicate_report,
            "file_manifest.parquet": file_manifest,
            "split_manifest.parquet": split_manifest,
            "fewshot_manifest.parquet": fewshot_manifest,
            "branch_predictions.parquet": branch_predictions,
            "methane_predictions.parquet": methane_predictions,
            "episode_predictions.parquet": episode_predictions,
        }
        for name, frame in frames.items():
            _write_parquet_once(frame, root / name)
        metrics = {"methane_event": methane_metrics, "episode_event": episode_metrics}
        write_once_json(root / "metrics.json", metrics)
        group_counts = {
            str(pool): int(count)
            for pool, count in split_manifest.groupby("pool")["raw_group_id"].nunique().items()
        }
        summary = {
            "schema_version": 1,
            "step_id": PIPELINE_STEP,
            "status": "pass",
            "synthetic_only": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "file_manifest_rows": len(file_manifest),
            "pool_group_counts": group_counts,
            "max_groups_per_pool": max(group_counts.values()),
            "fit_and_inference_samples_disjoint": (
                methane_boundary["fit_and_inference_disjoint"]
                and episode_boundary["fit_and_inference_disjoint"]
            ),
            "methane_boundary": methane_boundary,
            "episode_boundary": episode_boundary,
            "models_exercised": list(EXPECTED_MODELS),
            "full_local_dataset_extractions": 0,
            "remote_connections": 0,
            "slurm_jobs_created": 0,
        }
        write_once_json(root / "summary.json", summary)
        log_lines = [
            "I080 synthetic manifests validated",
            "I080 branch, methane, fusion, memory, and event paths completed",
            "I080 no dataset extraction, remote connection, or Slurm submission",
        ]
        write_once_bytes(root / "execution.log", ("\n".join(log_lines) + "\n").encode("ascii"))
        artifact_names = sorted((*PARQUET_OUTPUTS, *JSON_OUTPUTS, *TEXT_OUTPUTS))
        artifacts = [_artifact_record(root / name, root) for name in artifact_names]
        receipt = {
            "schema_version": 1,
            "step_id": PIPELINE_STEP,
            "status": "pass",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "summary_sha256": sha256_file(root / "summary.json"),
            "artifacts": artifacts,
        }
        write_once_json(root / RECEIPT_NAME, receipt)
        result = validate_minimal_pipeline_run(root)
        result["mode"] = "created"
        return result
    except Exception as exc:
        if not (root / RECEIPT_NAME).exists() and not (root / "failure.json").exists():
            failure = {
                "schema_version": 1,
                "step_id": PIPELINE_STEP,
                "status": "failed",
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "last_safe_artifacts": sorted(
                    path.name
                    for path in root.iterdir()
                    if path.is_file() and path.name != LOCK_NAME
                ),
                "retry_rule": "repair_then_use_fresh_run_root",
            }
            write_once_json(root / "failure.json", failure)
        raise
    finally:
        lock_path.unlink(missing_ok=True)
