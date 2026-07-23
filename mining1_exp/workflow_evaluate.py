from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss
import yaml

from .data.episodes import validate_fusion_eligibility_lock
from .evaluate.event_metrics import (
    evaluate_episode_events,
    evaluate_methane_events,
    macro_event_f1,
)
from .governance.prediction_lock import validate_prediction_lock
from .governance.release import validate_label_access
from .provenance import branch_artifact_path, canonical_json_sha256, sha256_file
from .workflow_common import (
    hash_existing_inputs,
    load_json,
    utc_now,
    write_json_artifact,
    write_parquet_artifact,
)


PACKAGE_IDS = {"T1", "S1", "V2", "R1", "E2_E3"}


def _protocol(root: Path) -> Dict[str, Any]:
    path = root / "configs/protocol_lock.pretest.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise RuntimeError("Frozen protocol must be a mapping")
    return payload


def _ensure_no_test_selection(payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, sort_keys=True).lower()
    forbidden = ("d_b_te", "test_metric", "test_score", "test_loss")
    if any(token in encoded for token in forbidden):
        raise RuntimeError("Checkpoint selection evidence contains test information")


def _v2_not_applicable_repeats(root: Path) -> list[Dict[str, Any]]:
    receipts = sorted(
        (root / "evidence/remote_runs/E202").glob(
            "**/V2-A-10-MULTI_not_applicable.json"
        )
    )
    accepted: list[Dict[str, Any]] = []
    seen = set()
    for receipt_path in receipts:
        payload = load_json(receipt_path)
        _ensure_no_test_selection(payload)
        if payload.get("status") != "not_applicable":
            continue
        if payload.get("step_id") != "E202":
            continue
        if payload.get("family_id") != "V2-A-10-MULTI":
            continue
        if payload.get("condition") != "multi_coal":
            continue
        key = (
            "V2-A-10-MULTI",
            int(payload["train_seed"]),
            int(payload["subset_seed"]),
        )
        if key in seen:
            continue
        seen.add(key)
        accepted.append(
            {
                "family_id": "V2-A-10-MULTI",
                "train_seed": key[1],
                "subset_seed": key[2],
                "status": "not_applicable",
                "reason_code": str(payload.get("fallback", {}).get("reason_code", "")),
                "receipt_path": receipt_path.relative_to(root).as_posix(),
                "receipt_sha256": sha256_file(receipt_path),
            }
        )
    return sorted(
        accepted,
        key=lambda value: (value["train_seed"], value["subset_seed"]),
    )


def select_checkpoints(
    root: Path, row: Mapping[str, str], arguments: Mapping[str, Any]
) -> Dict[str, Any]:
    family = str(arguments["family"])
    if family != "V2":
        raise RuntimeError("The frozen checkpoint-selection step is V2 only")
    manifests = sorted((root / "runs/V2/finetune").rglob("run_manifest.json"))
    if not manifests:
        raise RuntimeError("No V2 fine-tuning run manifests exist")
    selected = []
    seen = set()
    for manifest_path in manifests:
        manifest = load_json(manifest_path)
        _ensure_no_test_selection(manifest)
        required = {
            "run_id",
            "family_id",
            "train_seed",
            "subset_seed",
            "selection_pool",
            "selection_metric",
            "selection_metric_value",
            "checkpoint_path",
            "checkpoint_sha256",
        }
        if not required.issubset(manifest):
            raise RuntimeError(f"Incomplete V2 run manifest: {manifest_path}")
        if manifest["selection_pool"] != "D_b_sel":
            raise RuntimeError("V2 checkpoint selection may use only D_b_sel")
        if manifest["selection_metric"] != "map50_95":
            raise RuntimeError("V2 checkpoint selection metric drifted")
        key = (
            str(manifest["family_id"]),
            int(manifest["train_seed"]),
            int(manifest["subset_seed"]),
        )
        if key in seen:
            raise RuntimeError(f"Duplicate V2 repeat cell: {key}")
        seen.add(key)
        checkpoint = root / str(manifest["checkpoint_path"])
        checkpoint_availability = "remote_manifest_bound"
        if checkpoint.is_file():
            if sha256_file(checkpoint) != manifest["checkpoint_sha256"]:
                raise RuntimeError(f"V2 checkpoint hash mismatch: {checkpoint}")
            checkpoint_availability = "local_hash_verified"
        selected.append(
            {
                "run_id": str(manifest["run_id"]),
                "family_id": str(manifest["family_id"]),
                "train_seed": int(manifest["train_seed"]),
                "subset_seed": int(manifest["subset_seed"]),
                "selection_pool": "D_b_sel",
                "selection_metric": "map50_95",
                "selection_metric_value": float(manifest["selection_metric_value"]),
                "checkpoint_path": str(manifest["checkpoint_path"]),
                "checkpoint_sha256": str(manifest["checkpoint_sha256"]),
                "checkpoint_availability": checkpoint_availability,
                "run_manifest_path": manifest_path.relative_to(root).as_posix(),
                "run_manifest_sha256": sha256_file(manifest_path),
            }
        )
    expected_families = {
        "V2-A-10-SCR",
        "V2-A-10-GEN",
        "V2-A-10-SINGLE",
        "V2-A-10-MULTI",
    }
    selected_families = {item["family_id"] for item in selected}
    not_applicable = _v2_not_applicable_repeats(root)
    not_applicable_families = {item["family_id"] for item in not_applicable}
    if selected_families | not_applicable_families != expected_families:
        raise RuntimeError("V2 checkpoint selection does not cover all four conditions")
    if len(selected) + len(not_applicable) != 12:
        raise RuntimeError("V2 checkpoint selection requires twelve resolved repeat cells")
    if not_applicable and (
        not_applicable_families != {"V2-A-10-MULTI"} or len(not_applicable) != 3
    ):
        raise RuntimeError("V2 not-applicable repeats must be exactly the locked MULTI cells")
    payload = {
        "schema_version": 1,
        "step_id": row["step_id"],
        "status": "pass",
        "family": family,
        "selection_rule": "validation_only_map50_95_per_locked_repeat_cell",
        "test_information_used": False,
        "not_applicable": not_applicable,
        "selected": sorted(
            selected,
            key=lambda value: (
                value["family_id"],
                value["train_seed"],
                value["subset_seed"],
            ),
        ),
        "created_at": utc_now(),
    }
    output = "runs/V2/checkpoint_selection.json"
    write_json_artifact(root / output, payload)
    return {
        "status": "pass",
        "output_paths": [output],
        "inputs": hash_existing_inputs(
            root,
            [path.relative_to(root).as_posix() for path in manifests]
            + [item["receipt_path"] for item in not_applicable],
        ),
        "details": {
            "selected_repeat_cells": len(selected),
            "not_applicable_repeat_cells": len(not_applicable),
            "resolved_repeat_cells": len(selected) + len(not_applicable),
        },
    }


def _validate_release(root: Path, package: str) -> tuple[Path, Path, Path]:
    if package == "E2_E3":
        release = root / "data/releases/episode_test_release.json"
        seal = root / "data/seals/episode_test_seal.json"
        lock = root / "predictions/locked/episodes_prediction_lock.json"
    else:
        release = root / "data/releases/branch_test_release.json"
        seal = branch_artifact_path(root, "final_seal")
        lock = root / "predictions/locked/branch_prediction_lock.json"
    release_payload = load_json(release)
    lock_payload = load_json(lock)
    validate_prediction_lock(lock_payload)
    validate_label_access(
        role="evaluator",
        principal_identity="mining1_independent_evaluator",
        seal_hash=sha256_file(seal),
        prediction_lock_hash=sha256_file(lock),
        release_payload=release_payload,
    )
    return release, seal, lock


def _ece(scores: Sequence[float], labels: Sequence[int], bins: int = 10) -> float:
    probabilities = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    if len(probabilities) != len(truth) or len(truth) == 0:
        raise RuntimeError("ECE inputs must be aligned and non-empty")
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (probabilities >= edges[index]) & (probabilities <= edges[index + 1])
        else:
            mask = (probabilities >= edges[index]) & (probabilities < edges[index + 1])
        if mask.any():
            result += float(mask.mean()) * abs(
                float(probabilities[mask].mean()) - float(truth[mask].mean())
            )
    return result


def _iou(left: Sequence[float], right: Sequence[float]) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(
        0.0, float(left[3]) - float(left[1])
    )
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(
        0.0, float(right[3]) - float(right[1])
    )
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _average_precision(recalls: np.ndarray, precisions: np.ndarray) -> float:
    recall = np.concatenate(([0.0], recalls, [1.0]))
    precision = np.concatenate(([1.0], precisions, [0.0]))
    for index in range(len(precision) - 2, -1, -1):
        precision[index] = max(precision[index], precision[index + 1])
    changes = np.flatnonzero(recall[1:] != recall[:-1])
    return float(np.sum((recall[changes + 1] - recall[changes]) * precision[changes + 1]))


def _class_ap(
    predictions: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    concept_id: str,
    iou_threshold: float,
) -> tuple[float, int, int, int]:
    class_truth = truth.loc[truth["concept_id"].astype(str) == concept_id]
    class_predictions = predictions.loc[
        predictions["concept_id"].astype(str) == concept_id
    ].sort_values("score_calibrated", ascending=False)
    matched: Dict[str, set[int]] = {}
    true_positive = []
    false_positive = []
    for prediction in class_predictions.itertuples(index=False):
        record_truth = class_truth.loc[
            class_truth["record_id"].astype(str) == str(prediction.record_id)
        ].reset_index(drop=True)
        used = matched.setdefault(str(prediction.record_id), set())
        best_index = None
        best_iou = -1.0
        for index, target in enumerate(record_truth.itertuples(index=False)):
            if index in used:
                continue
            overlap = _iou(
                (prediction.x1, prediction.y1, prediction.x2, prediction.y2),
                (target.x1, target.y1, target.x2, target.y2),
            )
            if overlap > best_iou:
                best_iou = overlap
                best_index = index
        is_match = best_index is not None and best_iou >= iou_threshold
        if is_match:
            used.add(int(best_index))
        true_positive.append(1 if is_match else 0)
        false_positive.append(0 if is_match else 1)
    truth_count = len(class_truth)
    if truth_count == 0:
        return math.nan, 0, int(sum(false_positive)), 0
    tp = np.cumsum(np.asarray(true_positive, dtype=float))
    fp = np.cumsum(np.asarray(false_positive, dtype=float))
    recalls = tp / truth_count
    precisions = tp / np.maximum(tp + fp, 1.0)
    ap = _average_precision(recalls, precisions) if len(tp) else 0.0
    matched_count = int(tp[-1]) if len(tp) else 0
    return ap, matched_count, int(fp[-1]) if len(fp) else 0, truth_count - matched_count


def _metric_row(
    *,
    package: str,
    family: str,
    run_id: str,
    repeat_key: str,
    metric_name: str,
    value: float,
    result_level: str,
    group_id: str,
    population_hash: str,
    prediction_lock_hash: str,
    label_manifest_hash: str,
    diagnostics: Mapping[str, Any],
) -> Dict[str, Any]:
    definition = {
        "package": package,
        "metric": metric_name,
        "version": "minimal_v2_1",
        "aggregation_unit": "smallest_independent_raw_group",
    }
    return {
        "package_id": package,
        "family_id": family,
        "run_id": run_id,
        "repeat_key": repeat_key,
        "group_id": group_id,
        "result_level": result_level,
        "metric_name": metric_name,
        "metric_value": float(value),
        "metric_version": "minimal_v2_1",
        "metric_definition_hash": canonical_json_sha256(definition),
        "aggregation_unit": definition["aggregation_unit"],
        "population_hash": population_hash,
        "prediction_lock_hash": prediction_lock_hash,
        "label_manifest_hash": label_manifest_hash,
        "diagnostic_counts_json": json.dumps(dict(diagnostics), sort_keys=True),
        "decision_status": "exploratory_only",
    }


def _repeat_key(frame: pd.DataFrame, fallback: str) -> str:
    values = []
    for column in ("train_seed", "subset_seed", "episode_seed"):
        if column in frame.columns:
            unique = frame[column].dropna().unique()
            if len(unique) == 0:
                continue
            if len(unique) != 1:
                raise RuntimeError(f"Repeat cell has multiple {column} values")
            values.append(f"{column}={unique[0]}")
    return "|".join(values) if values else fallback


def _detection_metrics(
    root: Path,
    *,
    package: str,
    lock_hash: str,
) -> pd.DataFrame:
    predictions = pd.read_parquet(root / f"predictions/locked/{package}/detections.parquet")
    truth_path = branch_artifact_path(root, "detection_truth")
    truth = pd.read_parquet(truth_path)
    required_prediction = {
        "run_id",
        "family_id",
        "record_id",
        "raw_group_id",
        "concept_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "score_calibrated",
    }
    required_truth = {
        "record_id",
        "raw_group_id",
        "concept_id",
        "x1",
        "y1",
        "x2",
        "y2",
    }
    if not required_prediction.issubset(predictions.columns):
        raise RuntimeError(f"{package} detection prediction schema is incomplete")
    if not required_truth.issubset(truth.columns):
        raise RuntimeError("Branch detection truth schema is incomplete")
    population_hash = canonical_json_sha256(
        sorted(set(truth["raw_group_id"].astype(str)))
    )
    rows = []
    thresholds = np.round(np.arange(0.50, 0.951, 0.05), 2)
    for (run_id, family), run_predictions in predictions.groupby(
        ["run_id", "family_id"], sort=True
    ):
        concepts = sorted(set(truth["concept_id"].astype(str)))
        ap_by_threshold = []
        ap50_values = []
        for threshold in thresholds:
            class_values = []
            for concept in concepts:
                ap, _, _, _ = _class_ap(
                    run_predictions,
                    truth,
                    concept_id=concept,
                    iou_threshold=float(threshold),
                )
                if np.isfinite(ap):
                    class_values.append(ap)
            if not class_values:
                raise RuntimeError("Detection evaluation has no truth-bearing class")
            mean_value = float(np.mean(class_values))
            ap_by_threshold.append(mean_value)
            if math.isclose(float(threshold), 0.50):
                ap50_values = class_values
        metrics = {
            "map50": float(np.mean(ap50_values)),
            "map50_95": float(np.mean(ap_by_threshold)),
        }
        repeat_key = _repeat_key(run_predictions, str(run_id))
        for metric_name, value in metrics.items():
            rows.append(
                _metric_row(
                    package=package,
                    family=str(family),
                    run_id=str(run_id),
                    repeat_key=repeat_key,
                    metric_name=metric_name,
                    value=value,
                    result_level="repeat",
                    group_id="__all__",
                    population_hash=population_hash,
                    prediction_lock_hash=lock_hash,
                    label_manifest_hash=sha256_file(truth_path),
                    diagnostics={"truth_box_count": len(truth), "class_count": len(concepts)},
                )
            )
        for group_id, group_truth in truth.groupby("raw_group_id", sort=True):
            group_predictions = run_predictions.loc[
                run_predictions["raw_group_id"].astype(str) == str(group_id)
            ]
            tp = fp = fn = 0
            for concept in concepts:
                _, concept_tp, concept_fp, concept_fn = _class_ap(
                    group_predictions,
                    group_truth,
                    concept_id=concept,
                    iou_threshold=0.50,
                )
                tp += concept_tp
                fp += concept_fp
                fn += concept_fn
            f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
            rows.append(
                _metric_row(
                    package=package,
                    family=str(family),
                    run_id=str(run_id),
                    repeat_key=repeat_key,
                    metric_name="group_detection_f1_iou50",
                    value=f1,
                    result_level="group",
                    group_id=str(group_id),
                    population_hash=population_hash,
                    prediction_lock_hash=lock_hash,
                    label_manifest_hash=sha256_file(truth_path),
                    diagnostics={"tp": tp, "fp": fp, "fn": fn},
                )
            )
            _, group_map50_95 = _map_values(group_predictions, group_truth)
            rows.append(
                _metric_row(
                    package=package,
                    family=str(family),
                    run_id=str(run_id),
                    repeat_key=repeat_key,
                    metric_name="group_map50_95",
                    value=group_map50_95,
                    result_level="group",
                    group_id=str(group_id),
                    population_hash=population_hash,
                    prediction_lock_hash=lock_hash,
                    label_manifest_hash=sha256_file(truth_path),
                    diagnostics={"truth_box_count": len(group_truth)},
                )
            )
    concept_prediction_path = root / f"predictions/locked/{package}/concepts.parquet"
    concept_truth_path = branch_artifact_path(root, "concept_truth")
    if concept_prediction_path.is_file() and concept_truth_path.is_file():
        concept_predictions = pd.read_parquet(concept_prediction_path)
        concept_truth = pd.read_parquet(concept_truth_path)
        required_prediction = {
            "run_id",
            "family_id",
            "record_id",
            "concept_id",
            "concept_probability_calibrated",
        }
        required_truth = {"record_id", "concept_id", "concept_truth", "raw_group_id"}
        if not required_prediction.issubset(concept_predictions.columns):
            raise RuntimeError(f"{package} concept prediction schema is incomplete")
        if not required_truth.issubset(concept_truth.columns):
            raise RuntimeError("Branch concept truth schema is incomplete")
        for (run_id, family), run in concept_predictions.groupby(
            ["run_id", "family_id"], sort=True
        ):
            joined = run.merge(
                concept_truth,
                on=["record_id", "concept_id"],
                validate="many_to_one",
            )
            scores = joined["concept_probability_calibrated"].to_numpy(dtype=float)
            labels = joined["concept_truth"].to_numpy(dtype=int)
            repeat_key = _repeat_key(run, str(run_id))
            for metric_name, value in {
                "concept_brier_score": float(brier_score_loss(labels, scores)),
                "concept_expected_calibration_error": _ece(scores, labels),
            }.items():
                rows.append(
                    _metric_row(
                        package=package,
                        family=str(family),
                        run_id=str(run_id),
                        repeat_key=repeat_key,
                        metric_name=metric_name,
                        value=value,
                        result_level="repeat",
                        group_id="__all__",
                        population_hash=population_hash,
                        prediction_lock_hash=lock_hash,
                        label_manifest_hash=sha256_file(concept_truth_path),
                        diagnostics={"concept_row_count": len(joined)},
                    )
                )
    return pd.DataFrame(rows)


def _map_values(predictions: pd.DataFrame, truth: pd.DataFrame) -> tuple[float, float]:
    concepts = sorted(set(truth["concept_id"].astype(str)))
    thresholds = np.round(np.arange(0.50, 0.951, 0.05), 2)
    values = []
    map50 = None
    for threshold in thresholds:
        class_values = []
        for concept in concepts:
            ap, _, _, _ = _class_ap(
                predictions,
                truth,
                concept_id=concept,
                iou_threshold=float(threshold),
            )
            if np.isfinite(ap):
                class_values.append(ap)
        if not class_values:
            raise RuntimeError("Detection evaluation has no truth-bearing class")
        mean_value = float(np.mean(class_values))
        values.append(mean_value)
        if math.isclose(float(threshold), 0.50):
            map50 = mean_value
    return float(map50), float(np.mean(values))


def _r1_metrics(root: Path, lock_hash: str) -> pd.DataFrame:
    corrupt_path = root / "predictions/locked/R1.parquet"
    clean_path = root / "predictions/locked/V2/detections.parquet"
    truth_path = branch_artifact_path(root, "detection_truth")
    corrupt = pd.read_parquet(corrupt_path)
    clean = pd.read_parquet(clean_path)
    truth = pd.read_parquet(truth_path)
    required = {"run_id", "family_id", "corruption_type", "severity"}
    if not required.issubset(corrupt.columns):
        raise RuntimeError("R1 predictions lack corruption cell identifiers")
    family_map = {
        "R1-GEN": "V2-A-10-GEN",
        "R1-MULTI": "V2-A-10-MULTI",
    }
    population_hash = canonical_json_sha256(
        sorted(set(truth["raw_group_id"].astype(str)))
    )
    rows = []
    for (run_id, family), run in corrupt.groupby(["run_id", "family_id"], sort=True):
        if str(family) not in family_map:
            raise RuntimeError(f"Unexpected R1 family: {family}")
        clean_family = family_map[str(family)]
        clean_run = clean.loc[
            (clean["family_id"].astype(str) == clean_family)
            & (clean["run_id"].astype(str) == str(run_id).replace("R1", "V2"))
        ]
        if clean_run.empty:
            repeat_columns = [
                column
                for column in ("train_seed", "subset_seed")
                if column in run.columns and column in clean.columns
            ]
            if repeat_columns:
                mask = clean["family_id"].astype(str) == clean_family
                for column in repeat_columns:
                    mask &= clean[column] == run[column].iloc[0]
                clean_run = clean.loc[mask]
        if clean_run.empty:
            raise RuntimeError(f"R1 clean checkpoint pair is missing: {run_id}")
        _, clean_map = _map_values(clean_run, truth)
        for (corruption_type, severity), cell in run.groupby(
            ["corruption_type", "severity"], sort=True
        ):
            _, corrupted_map = _map_values(cell, truth)
            relative_drop = (
                100.0 * (clean_map - corrupted_map) / clean_map
                if clean_map > 0
                else math.nan
            )
            diagnostics = {
                "corruption_type": str(corruption_type),
                "severity": int(severity),
                "clean_map50_95": clean_map,
                "corrupted_map50_95": corrupted_map,
            }
            for metric_name, value in {
                "corrupted_map50_95": corrupted_map,
                "relative_drop_percent": relative_drop,
            }.items():
                rows.append(
                    _metric_row(
                        package="R1",
                        family=str(family),
                        run_id=str(run_id),
                        repeat_key=_repeat_key(run, str(run_id)),
                        metric_name=metric_name,
                        value=value,
                        result_level="corruption_cell",
                        group_id=f"{corruption_type}:{int(severity)}",
                        population_hash=population_hash,
                        prediction_lock_hash=lock_hash,
                        label_manifest_hash=sha256_file(truth_path),
                        diagnostics=diagnostics,
                    )
                )
    frame = pd.DataFrame(rows)
    if frame.empty or not np.isfinite(frame["metric_value"].astype(float)).all():
        raise RuntimeError("R1 metrics are empty or clean validity is undefined")
    expected_cells = {
        (corruption_type, severity)
        for corruption_type in ("low_light", "dust_fog_proxy")
        for severity in (1, 2, 3)
    }
    for _, group in frame.loc[
        frame["metric_name"] == "relative_drop_percent"
    ].groupby(["run_id", "family_id"]):
        cells = {
            tuple(str(value).split(":"))
            for value in group["group_id"]
        }
        normalized = {(kind, int(level)) for kind, level in cells}
        if normalized != expected_cells:
            raise RuntimeError("R1 does not contain the locked 2 by 3 corruption cells")
    return frame


def _s1_metrics(root: Path, lock_hash: str) -> pd.DataFrame:
    prediction_path = root / "predictions/locked/S1.parquet"
    truth_path = root / "data/sealed/branch_methane_truth.parquet"
    predictions = pd.read_parquet(prediction_path)
    truth = pd.read_parquet(truth_path)
    merged = predictions.merge(truth, on=["window_id", "raw_group_id"], validate="many_to_one")
    required = {
        "run_id",
        "family_id",
        "sensor_group_id",
        "timestamp_seconds",
        "raw_value",
        "score_calibrated",
        "threshold",
    }
    if not required.issubset(merged.columns):
        raise RuntimeError("S1 evaluation schema is incomplete")
    protocol = _protocol(root)
    methane = protocol["data"]["methane"]
    population_hash = canonical_json_sha256(
        sorted(set(merged["raw_group_id"].astype(str)))
    )
    rows = []
    for (run_id, family), run in merged.groupby(["run_id", "family_id"], sort=True):
        unit_results = []
        for sensor_id, sensor in run.groupby("sensor_group_id", sort=True):
            sensor = sensor.sort_values("timestamp_seconds")
            positions = sensor["timestamp_seconds"].to_numpy(dtype=float)
            gaps = np.diff(positions)
            continuity = float(np.median(gaps)) if len(gaps) else float(methane["stride_seconds"])
            result = evaluate_methane_events(
                raw_positions_seconds=positions,
                raw_values=sensor["raw_value"].to_numpy(dtype=float),
                risk_threshold=float(methane["risk_concentration_threshold"]),
                prediction_positions_seconds=positions,
                prediction_scores=sensor["score_calibrated"].to_numpy(dtype=float),
                score_threshold=float(sensor["threshold"].iloc[0]),
                raw_continuity_seconds=continuity,
                prediction_continuity_seconds=continuity,
                event_merge_gap_seconds=float(methane["event_merge_gap_seconds"]),
                horizon_seconds=float(methane["horizon_seconds"]),
                valid_observed_sensor_hours=max(
                    continuity * len(sensor) / 3600.0,
                    continuity / 3600.0,
                ),
            )
            unit_results.append(result)
            rows.append(
                _metric_row(
                    package="S1",
                    family=str(family),
                    run_id=str(run_id),
                    repeat_key=_repeat_key(run, str(run_id)),
                    metric_name="sensor_event_f1",
                    value=float(result["event_f1"] or 0.0),
                    result_level="group",
                    group_id=str(sensor_id),
                    population_hash=population_hash,
                    prediction_lock_hash=lock_hash,
                    label_manifest_hash=sha256_file(truth_path),
                    diagnostics=result,
                )
            )
        macro_f1 = macro_event_f1(unit_results)
        labels = (run["raw_value"].to_numpy(dtype=float) >= float(
            methane["risk_concentration_threshold"]
        )).astype(int)
        scores = run["score_calibrated"].to_numpy(dtype=float)
        total_hours = sum(result["valid_observed_sensor_hours"] for result in unit_results)
        total_false_alarms = sum(result["false_alarm_count"] for result in unit_results)
        miss_rates = [
            result["event_miss_rate"]
            for result in unit_results
            if result["event_miss_rate"] is not None
        ]
        lead_times = [
            value
            for result in unit_results
            for value in result["lead_times_seconds"]
        ]
        aggregate = {
            "event_macro_f1": macro_f1,
            "pr_auc": float(average_precision_score(labels, scores)),
            "brier_score": float(brier_score_loss(labels, scores)),
            "expected_calibration_error": _ece(scores, labels),
            "false_alarms_per_hour": float(total_false_alarms / total_hours),
            "event_miss_rate": float(np.mean(miss_rates)) if miss_rates else 0.0,
            "median_lead_time_seconds": (
                float(np.median(lead_times)) if lead_times else 0.0
            ),
        }
        for metric_name, value in aggregate.items():
            rows.append(
                _metric_row(
                    package="S1",
                    family=str(family),
                    run_id=str(run_id),
                    repeat_key=_repeat_key(run, str(run_id)),
                    metric_name=metric_name,
                    value=value,
                    result_level="repeat",
                    group_id="__all__",
                    population_hash=population_hash,
                    prediction_lock_hash=lock_hash,
                    label_manifest_hash=sha256_file(truth_path),
                    diagnostics={"sensor_count": len(unit_results)},
                )
            )
    return pd.DataFrame(rows)


def _episode_metrics(root: Path, lock_hash: str) -> tuple[pd.DataFrame, Dict[str, Any]]:
    prediction_path = root / "predictions/locked/episodes.parquet"
    controls_path = root / "predictions/locked/episode_shortcut_controls.parquet"
    label_path = root / "data/locked/episode_labels/D_e_te.parquet"
    predictions = pd.read_parquet(prediction_path)
    controls = pd.read_parquet(controls_path)
    labels = pd.read_parquet(label_path)
    join_columns = ["episode_id", "step_index", "concept_id"]
    label_columns = [*join_columns, "event_truth", "source_component_id"]
    required_prediction = {
        "run_id",
        "family_id",
        "train_seed",
        "episode_id",
        "step_index",
        "concept_id",
        "score_calibrated",
        "state_prediction",
        "abstained",
        "model_hash",
        "policy_hash",
    }
    required_labels = set(label_columns)
    if not required_prediction.issubset(predictions.columns) or not required_prediction.issubset(
        controls.columns
    ):
        raise RuntimeError("Episode prediction schema is incomplete")
    if not required_labels.issubset(labels.columns):
        raise RuntimeError("Episode label schema is incomplete")
    if labels.duplicated(join_columns).any():
        raise RuntimeError("Episode labels are not unique on the sealed join key")
    population_hash = canonical_json_sha256(
        sorted(set(labels["source_component_id"].astype(str)))
    )
    label_hash = sha256_file(label_path)

    def evaluate_run(run_id: str, family: str, run: pd.DataFrame) -> tuple[list[Dict[str, Any]], Dict[str, float], Dict[str, Any]]:
        if run.duplicated(join_columns).any():
            raise RuntimeError(f"Episode predictions duplicate a sealed row: {run_id}")
        concept_results = []
        metric_rows: list[Dict[str, Any]] = []
        repeat_key = _repeat_key(run, run_id)
        for concept_id, concept in run.groupby("concept_id", sort=True):
            concept = concept.sort_values(["episode_id", "step_index"])
            result = evaluate_episode_events(
                truth_event=concept["event_truth"].astype(int).tolist(),
                effective_states=concept["state_prediction"].astype(str).tolist(),
                episode_ids=concept["episode_id"].astype(str).tolist(),
                step_indices=concept["step_index"].astype(int).tolist(),
            )
            concept_results.append(result)
            metric_rows.append(
                _metric_row(
                    package="E2_E3",
                    family=family,
                    run_id=run_id,
                    repeat_key=repeat_key,
                    metric_name="concept_event_f1",
                    value=float(result["event_f1"] or 0.0),
                    result_level="group",
                    group_id=str(concept_id),
                    population_hash=population_hash,
                    prediction_lock_hash=lock_hash,
                    label_manifest_hash=label_hash,
                    diagnostics=result,
                )
            )
        truth_count = sum(int(result["truth_event_count"]) for result in concept_results)
        prediction_count = sum(
            int(result["prediction_event_count"]) for result in concept_results
        )
        matched_count = sum(int(result["matched_event_count"]) for result in concept_results)
        missed_count = sum(int(result["missed_event_count"]) for result in concept_results)
        false_alarm_count = sum(
            int(result["false_alarm_count"]) for result in concept_results
        )
        delays = [
            float(value)
            for result in concept_results
            for value in result["detection_delays_steps"]
        ]
        flips = [
            int(value)
            for result in concept_results
            for value in result["state_flips_per_episode"]
        ]
        answered = ~run["abstained"].astype(bool).to_numpy()
        truth = run["event_truth"].astype(int).to_numpy()
        positive = run["state_prediction"].astype(str).isin({"prewarning", "alarm"}).to_numpy()
        scores = run["score_calibrated"].astype(float).to_numpy()
        episode_count = int(run["episode_id"].astype(str).nunique())
        answer_coverage = float(np.mean(answered))
        mean_delay = float(np.mean(delays)) if delays else 0.0
        mean_flips = float(np.mean(flips)) if flips else 0.0
        selective_risk = (
            float(np.mean(positive[answered].astype(int) != truth[answered]))
            if answered.any()
            else 1.0
        )
        aggregate_values = {
            "event_macro_f1": macro_event_f1(concept_results),
            "event_precision": float(matched_count / prediction_count)
            if prediction_count
            else 0.0,
            "event_recall": float(matched_count / truth_count) if truth_count else 0.0,
            "false_alarms_per_100_episodes": float(
                false_alarm_count * 100.0 / episode_count
            ),
            "event_miss_rate": float(missed_count / truth_count) if truth_count else 0.0,
            "detection_delay_steps": mean_delay,
            "mean_detection_delay_steps": mean_delay,
            "state_flips": mean_flips,
            "mean_state_flips_per_episode": mean_flips,
            "expected_calibration_error": _ece(scores, truth),
            "answer_coverage": answer_coverage,
            "abstention_rate": 1.0 - answer_coverage,
            "selective_risk": selective_risk,
        }
        diagnostics = {
            "concept_count": len(concept_results),
            "evaluated_episode_count": episode_count,
            "truth_event_count": truth_count,
            "prediction_event_count": prediction_count,
            "matched_event_count": matched_count,
            "missed_event_count": missed_count,
            "false_alarm_count": false_alarm_count,
            "eligible_step_rows": len(run),
            "abstained_step_rows": int((~answered).sum()),
            "answer_coverage": answer_coverage,
        }
        for metric_name, value in aggregate_values.items():
            metric_rows.append(
                _metric_row(
                    package="E2_E3",
                    family=family,
                    run_id=run_id,
                    repeat_key=repeat_key,
                    metric_name=metric_name,
                    value=value,
                    result_level="repeat",
                    group_id="__all__",
                    population_hash=population_hash,
                    prediction_lock_hash=lock_hash,
                    label_manifest_hash=label_hash,
                    diagnostics=diagnostics,
                )
            )
        return metric_rows, aggregate_values, diagnostics

    main = predictions.merge(labels[label_columns], on=join_columns, validate="many_to_one")
    control_frame = controls.merge(
        labels[label_columns], on=join_columns, validate="many_to_one"
    )
    rows: list[Dict[str, Any]] = []
    full_hashes: Dict[int, str] = {}
    for (run_id, family), run in main.groupby(["run_id", "family_id"], sort=True):
        run_rows, _, _ = evaluate_run(str(run_id), str(family), run)
        rows.extend(run_rows)
        if str(family) == "E3-FULL":
            seeds = run["train_seed"].dropna().astype(int).unique()
            hashes = run["model_hash"].astype(str).unique()
            if len(seeds) != 1 or len(hashes) != 1:
                raise RuntimeError("E3-FULL repeat does not bind one seed and checkpoint")
            full_hashes[int(seeds[0])] = str(hashes[0])
    shortcut: Dict[str, Any] = {
        "status": "pass",
        "same_checkpoint_required": True,
        "test_information_used_for_selection": False,
        "controls": {},
    }
    for (run_id, family), run in control_frame.groupby(
        ["run_id", "family_id"], sort=True
    ):
        control_types = run["control_type"].dropna().astype(str).unique()
        seeds = run["train_seed"].dropna().astype(int).unique()
        hashes = run["model_hash"].astype(str).unique()
        if len(control_types) != 1 or len(seeds) != 1 or len(hashes) != 1:
            raise RuntimeError("Episode shortcut control identity is ambiguous")
        control_type = str(control_types[0])
        seed = int(seeds[0])
        model_hash = str(hashes[0])
        if full_hashes.get(seed) != model_hash:
            raise RuntimeError("Episode shortcut control does not reuse E3-FULL checkpoint")
        control_rows, aggregate_values, diagnostics = evaluate_run(
            str(run_id), str(family), run
        )
        if str(family) == "E3-SHUFFLE":
            rows.extend(control_rows)
        audit_id = f"{control_type}-seed-{seed}"
        shortcut["controls"][audit_id] = {
            "family_id": str(family),
            "train_seed": seed,
            "control_type": control_type,
            "model_hash": model_hash,
            "same_full_checkpoint": True,
            "metrics": aggregate_values,
            "diagnostic_counts": diagnostics,
        }
    if not shortcut["controls"]:
        raise RuntimeError("Episode shortcut control predictions are empty")
    return pd.DataFrame(rows), shortcut


def evaluate_package(
    root: Path, row: Mapping[str, str], arguments: Mapping[str, Any]
) -> Dict[str, Any]:
    package = str(arguments["package"])
    if package not in PACKAGE_IDS:
        raise RuntimeError(f"Unknown evaluation package: {package}")
    if package == "E2_E3":
        eligibility = validate_fusion_eligibility_lock(
            pd.read_parquet(
                root / "data/locked/fusion_eligibility_lock.parquet"
            )
        )
        if not eligibility["fusion_primary_eligible"].astype(bool).any():
            output = "evidence/episode/E310_not_applicable.json"
            payload = {
                "schema_version": 1,
                "step_id": row["step_id"],
                "status": "accepted_not_applicable",
                "reason": "fusion_eligible_population_empty_at_E058",
                "fusion_eligibility_sha256": sha256_file(
                    root / "data/locked/fusion_eligibility_lock.parquet"
                ),
                "created_at": utc_now(),
            }
            write_json_artifact(root / output, payload)
            return {
                "status": "pass",
                "output_paths": [output],
                "inputs": hash_existing_inputs(
                    root, ["data/locked/fusion_eligibility_lock.parquet"]
                ),
                "details": {"workflow_status": "accepted_not_applicable"},
            }
    release, _, lock = _validate_release(root, package)
    lock_hash = sha256_file(lock)
    shortcut: Dict[str, Any] = {}
    if package in {"T1", "V2"}:
        metrics = _detection_metrics(root, package=package, lock_hash=lock_hash)
    elif package == "R1":
        metrics = _r1_metrics(root, lock_hash)
    elif package == "S1":
        metrics = _s1_metrics(root, lock_hash)
    else:
        metrics, shortcut = _episode_metrics(root, lock_hash)
    if metrics.empty or not np.isfinite(metrics["metric_value"].astype(float)).all():
        raise RuntimeError(f"{package} evaluation produced no finite metric rows")
    output = f"results/{package}/metrics.parquet"
    write_parquet_artifact(root / output, metrics)
    output_paths = [output]
    if package == "E2_E3":
        shortcut_output = "results/E2_E3/shortcut_audits.json"
        write_json_artifact(
            root / shortcut_output,
            {
                "schema_version": 1,
                "status": "pass",
                "same_checkpoint_controls": shortcut,
                "created_at": utc_now(),
            },
        )
        output_paths.append(shortcut_output)
    return {
        "status": "pass",
        "output_paths": output_paths,
        "inputs": hash_existing_inputs(
            root,
            [
                release.relative_to(root).as_posix(),
                lock.relative_to(root).as_posix(),
            ],
        ),
        "details": {
            "package": package,
            "metric_row_count": len(metrics),
            "test_threshold_searches": 0,
        },
    }
