from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ..data.manifests import read_file_feature_manifest, read_file_manifest, read_split_manifest
from ..governance.immutable import write_once_bytes
from ..governance.prediction_lock import build_prediction_lock, close_prediction_lock
from ..models.yolo_adapter import aggregate_concept_scores, select_calibrator_group_cv
from ..provenance import branch_artifact_path, canonical_json_sha256, sha256_file
from ..training_data import (
    _ontology_mapping,
    dataset_ids_for_role,
    load_frozen_protocol,
    target_branch_modalities,
)
from ..workflow_common import WorkflowExecutionError, load_json, write_parquet_artifact


def _boundary_hashes(project_root: Path) -> Dict[str, str]:
    paths = {
        "seal_hash": branch_artifact_path(project_root, "final_seal"),
        "protocol_hash": project_root / "configs/protocol_lock.pretest.yaml",
        "source_hash": project_root / "evidence/data/dataset_source_decision.json",
        "matrix_hash": project_root / "configs/experiment_matrix.template.csv",
    }
    for path in paths.values():
        if not path.is_file():
            raise WorkflowExecutionError(f"Detection prediction boundary is missing: {path}")
    return {name: sha256_file(path) for name, path in paths.items()}


def _records_for_pool(
    project_root: Path,
    *,
    dataset_id: str,
    modality: str,
    pool: str,
    include_labels: bool = False,
) -> pd.DataFrame:
    manifest_path = (
        branch_artifact_path(project_root, "feature_store")
        if pool == "D_b_te" and not include_labels
        else project_root / "data/locked/file_manifest.parquet"
    )
    files = (
        read_file_manifest(manifest_path)
        if include_labels
        else read_file_feature_manifest(manifest_path)
    )
    split = read_split_manifest(project_root / "data/locked/split_manifest.parquet")
    rows = files.merge(
        split[["dataset_id", "record_id", "raw_group_id", "pool"]],
        on=["dataset_id", "record_id", "raw_group_id"],
        validate="one_to_one",
    )
    branch_modalities = target_branch_modalities(
        project_root,
        dataset_id=dataset_id,
        source_modalities=rows["modality"],
    )
    selected = rows.loc[
        (rows["dataset_id"].astype(str) == dataset_id)
        & (branch_modalities == modality.lower())
        & (rows["pool"].astype(str) == pool)
    ].copy()
    if selected.empty:
        raise WorkflowExecutionError(
            f"Detection dataset {dataset_id}/{modality} has no records in {pool}"
        )
    return selected.sort_values("record_id").reset_index(drop=True)


def _concept_labels(
    rows: pd.DataFrame,
    mapping: Mapping[tuple[str, str], str],
    concepts: Sequence[str],
) -> pd.DataFrame:
    output = []
    for item in rows.itertuples(index=False):
        dataset_concepts = {
            concept
            for (dataset_id, _), concept in mapping.items()
            if dataset_id == str(item.dataset_id)
        }
        if dataset_concepts != set(concepts):
            raise WorkflowExecutionError(
                f"Calibration dataset lacks complete frozen ontology coverage: {item.dataset_id}"
            )
        summary = json.loads(str(item.label_summary_json))
        if summary.get("negative_annotation_verified") is not True:
            raise WorkflowExecutionError(
                f"Detection calibration lacks verified negatives: {item.record_id}"
            )
        present = {
            mapping[(str(item.dataset_id), str(box["source_label"]))]
            for box in summary.get("boxes", [])
            if (str(item.dataset_id), str(box.get("source_label"))) in mapping
        }
        for concept in concepts:
            output.append(
                {
                    "record_id": str(item.record_id),
                    "raw_group_id": str(item.raw_group_id),
                    "concept_id": str(concept),
                    "label_value": int(concept in present),
                }
            )
    return pd.DataFrame.from_records(output)


def _extract_result_rows(
    result: Any,
    *,
    record: Any,
    concepts: Sequence[str],
) -> pd.DataFrame:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return pd.DataFrame(
            columns=["record_id", "raw_group_id", "concept_id", "x1", "y1", "x2", "y2", "score_raw"]
        )
    xyxy = boxes.xyxy.detach().cpu().numpy()
    confidence = boxes.conf.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy().astype(int)
    if xyxy.shape != (len(confidence), 4) or len(classes) != len(confidence):
        raise WorkflowExecutionError("Ultralytics prediction boxes have an invalid shape")
    rows = []
    for coordinates, score, class_index in zip(xyxy, confidence, classes):
        if class_index < 0 or class_index >= len(concepts):
            raise WorkflowExecutionError("Detector emitted a class outside the frozen ontology")
        if not np.isfinite(coordinates).all() or not np.isfinite(score) or not 0 <= score <= 1:
            raise WorkflowExecutionError("Detector emitted non-finite coordinates or confidence")
        rows.append(
            {
                "record_id": str(record.record_id),
                "raw_group_id": str(record.raw_group_id),
                "concept_id": str(concepts[class_index]),
                "x1": np.float32(coordinates[0]),
                "y1": np.float32(coordinates[1]),
                "x2": np.float32(coordinates[2]),
                "y2": np.float32(coordinates[3]),
                "score_raw": np.float32(score),
            }
        )
    return pd.DataFrame.from_records(rows)


def _predict_records(
    model: Any,
    rows: pd.DataFrame,
    *,
    project_root: Path,
    concepts: Sequence[str],
    imgsz: int,
) -> pd.DataFrame:
    parts = []
    for item in rows.itertuples(index=False):
        source = str((project_root / str(item.relative_path)).resolve())
        result = model.predict(
            source=source,
            imgsz=imgsz,
            conf=0.001,
            iou=0.70,
            device=0,
            verbose=False,
        )
        if not isinstance(result, (list, tuple)) or len(result) != 1:
            raise WorkflowExecutionError("Detector must return exactly one result per record")
        parts.append(_extract_result_rows(result[0], record=item, concepts=concepts))
    nonempty = [part for part in parts if not part.empty]
    if not nonempty:
        return pd.DataFrame(
            columns=["record_id", "raw_group_id", "concept_id", "x1", "y1", "x2", "y2", "score_raw"]
        )
    return pd.concat(nonempty, ignore_index=True)


def _calibrators_for_run(
    probability_rows: pd.DataFrame,
    probability_detections: pd.DataFrame,
    *,
    mapping: Mapping[tuple[str, str], str],
    concepts: Sequence[str],
    modality: str,
    protocol: Mapping[str, Any],
) -> tuple[Dict[str, Any], bytes]:
    labels = _concept_labels(probability_rows, mapping, concepts)
    raw = aggregate_concept_scores(
        probability_detections,
        record_ids=probability_rows["record_id"].astype(str),
        concept_ids=concepts,
        modality=modality,
        post_nms=True,
    ).merge(
        probability_rows[["record_id", "raw_group_id"]],
        on="record_id",
        validate="many_to_one",
    ).merge(
        labels,
        on=["record_id", "raw_group_id", "concept_id"],
        validate="one_to_one",
    )
    calibrators = {}
    metadata = {}
    for concept, concept_rows in raw.groupby("concept_id", sort=True):
        selection = select_calibrator_group_cv(
            concept_rows["step_score_raw"],
            concept_rows["label_value"],
            concept_rows["raw_group_id"],
            pool="D_b_prob",
            negatives_verified=True,
            folds=int(protocol["evaluation"]["calibration_group_cv_folds"]),
            methods=tuple(protocol["evaluation"]["calibration_candidates"]),
        )
        calibrators[str(concept)] = selection.calibrator
        metadata[str(concept)] = {
            "method": selection.method,
            "fold_count": selection.fold_count,
            "cv_brier_by_method": selection.cv_brier_by_method,
        }
    payload = {"schema_version": 1, "metadata": metadata, "calibrators": calibrators}
    return calibrators, pickle.dumps(payload, protocol=4)


def _run_specs(project_root: Path, package: str) -> list[Dict[str, Any]]:
    if package == "T1":
        manifests = sorted((project_root / "runs/T1").rglob("run_manifest.json"))
        if len(manifests) != 2:
            raise WorkflowExecutionError("T1 prediction requires two completed branch runs")
        selected = [load_json(path) for path in manifests]
    elif package == "V2":
        payload = load_json(project_root / "runs/V2/checkpoint_selection.json")
        selected = payload.get("selected", [])
        if payload.get("status") != "pass" or not selected:
            raise WorkflowExecutionError("V2 prediction requires selected repeat cells")
    else:
        raise WorkflowExecutionError(f"Unsupported detection prediction package: {package}")
    specs = []
    for item in selected:
        checkpoint = project_root / str(item["checkpoint_path"])
        if not checkpoint.is_file() or sha256_file(checkpoint) != item["checkpoint_sha256"]:
            raise WorkflowExecutionError(f"Detection checkpoint hash drifted: {checkpoint}")
        family = str(item["family_id"])
        modality = (
            str(item.get("modality") or ("thermal" if family == "T1-THERM" else "visible"))
        )
        lock_id = family
        if item.get("train_seed") is not None:
            lock_id += f"-seed-{int(item['train_seed'])}"
        if item.get("subset_seed") is not None:
            lock_id += f"-subset-{int(item['subset_seed'])}"
        specs.append(
            {
                **dict(item),
                "checkpoint": checkpoint,
                "modality": modality,
                "lock_id": lock_id,
            }
        )
    if len({item["lock_id"] for item in specs}) != len(specs):
        raise WorkflowExecutionError("Detection prediction run lock IDs are not unique")
    return specs


def predict_detection_package(
    project_root: Path,
    run_root: Path,
    package: str,
    *,
    yolo_factory: Optional[Callable[[str], Any]] = None,
) -> Dict[str, Any]:
    probe = run_root / "slurm_environment_probe.json"
    if not probe.is_file():
        raise WorkflowExecutionError(f"{package} prediction lacks its Slurm probe")
    if yolo_factory is None:
        from ultralytics import YOLO

        yolo_factory = YOLO
    protocol = load_frozen_protocol(project_root)
    mapping, concepts = _ontology_mapping(project_root)
    role = "primary_rgbt_dataset" if package == "T1" else "primary_visual_target"
    dataset_id = dataset_ids_for_role(project_root, role)[0]
    specs = _run_specs(project_root, package)
    detection_parts = []
    concept_parts = []
    calibrator_store = {}
    model_hashes = {}
    policy_hashes = {}
    imgsz = int(protocol["models"]["visible"]["input_size"])
    ontology_hash = sha256_file(project_root / "data/locked/ontology_lock.yaml")
    aggregation_hash = canonical_json_sha256(
        {"rule": "max_post_nms_compatible_box_score_else_zero", "version": 1}
    )
    for spec in specs:
        probability_rows = _records_for_pool(
            project_root,
            dataset_id=dataset_id,
            modality=spec["modality"],
            pool="D_b_prob",
            include_labels=True,
        )
        test_rows = _records_for_pool(
            project_root,
            dataset_id=dataset_id,
            modality=spec["modality"],
            pool="D_b_te",
        )
        model = yolo_factory(str(spec["checkpoint"]))
        probability_detections = _predict_records(
            model,
            probability_rows,
            project_root=project_root,
            concepts=concepts,
            imgsz=imgsz,
        )
        test_detections = _predict_records(
            model,
            test_rows,
            project_root=project_root,
            concepts=concepts,
            imgsz=imgsz,
        )
        calibrators, calibrator_bytes = _calibrators_for_run(
            probability_rows,
            probability_detections,
            mapping=mapping,
            concepts=concepts,
            modality=spec["modality"],
            protocol=protocol,
        )
        lock_id = str(spec["lock_id"])
        calibrator_store[lock_id] = calibrator_bytes
        calibrator_hash = hashlib.sha256(calibrator_bytes).hexdigest()
        model_hash = sha256_file(spec["checkpoint"])
        policy_hash = canonical_json_sha256(
            {
                "calibrator_sha256": calibrator_hash,
                "aggregation_sha256": aggregation_hash,
                "nms_iou": 0.70,
                "inference_confidence_floor": 0.001,
            }
        )
        model_hashes[lock_id] = model_hash
        policy_hashes[lock_id] = policy_hash
        run_id = str(spec.get("run_id") or canonical_json_sha256({"lock_id": lock_id, "model": model_hash}))
        if not test_detections.empty:
            calibrated_detection = []
            for concept, part in test_detections.groupby("concept_id", sort=True):
                copied = part.copy()
                copied["score_calibrated"] = calibrators[str(concept)].predict(
                    copied["score_raw"]
                ).astype(np.float32)
                calibrated_detection.append(copied)
            detections = pd.concat(calibrated_detection, ignore_index=True)
        else:
            detections = test_detections.assign(score_calibrated=pd.Series(dtype="float32"))
        detections = detections.sort_values(
            ["record_id", "concept_id", "score_raw"], ascending=[True, True, False]
        ).reset_index(drop=True)
        detections["detection_id"] = detections.groupby("record_id").cumcount().astype("int64")
        detections = detections.assign(
            run_id=run_id,
            family_id=str(spec["family_id"]),
            train_seed=spec.get("train_seed"),
            subset_seed=spec.get("subset_seed"),
            prediction_role="branch_test",
            pool_role="D_b_te",
            modality=spec["modality"],
            available=True,
            model_hash=model_hash,
            calibrator_hash=calibrator_hash,
        )
        detection_parts.append(detections)
        concepts_frame = aggregate_concept_scores(
            test_detections,
            record_ids=test_rows["record_id"].astype(str),
            concept_ids=concepts,
            modality=spec["modality"],
            post_nms=True,
        ).merge(
            test_rows[["record_id", "raw_group_id"]],
            on="record_id",
            validate="many_to_one",
        )
        concepts_frame["concept_probability_calibrated"] = [
            np.float32(calibrators[str(concept)].predict([score])[0])
            for concept, score in concepts_frame[["concept_id", "step_score_raw"]].itertuples(
                index=False, name=None
            )
        ]
        concepts_frame = concepts_frame.assign(
            run_id=run_id,
            family_id=str(spec["family_id"]),
            train_seed=spec.get("train_seed"),
            subset_seed=spec.get("subset_seed"),
            prediction_role="branch_test",
            pool_role="D_b_te",
            available=True,
            detector_hash=model_hash,
            ontology_hash=ontology_hash,
            aggregation_rule_hash=aggregation_hash,
            calibrator_hash=calibrator_hash,
        )
        concept_parts.append(concepts_frame)
    detections = pd.concat(detection_parts, ignore_index=True)
    concepts_frame = pd.concat(concept_parts, ignore_index=True)
    expected_concepts = sum(
        len(
            _records_for_pool(
                project_root,
                dataset_id=dataset_id,
                modality=spec["modality"],
                pool="D_b_te",
            )
        )
        * len(concepts)
        for spec in specs
    )
    if len(concepts_frame) != expected_concepts or concepts_frame.duplicated(
        ["run_id", "record_id", "concept_id", "modality"]
    ).any():
        raise WorkflowExecutionError("Detection concept prediction coverage is incomplete")
    output_root = project_root / f"predictions/locked/{package}"
    detection_path = output_root / "detections.parquet"
    concept_path = output_root / "concepts.parquet"
    write_parquet_artifact(detection_path, detections)
    write_parquet_artifact(concept_path, concepts_frame)
    calibrator_path = output_root / "calibrators.pkl"
    write_once_bytes(calibrator_path, pickle.dumps(calibrator_store, protocol=4))
    paths = {lock_id: detection_path for lock_id in sorted(model_hashes)}
    lock = build_prediction_lock(
        scope="branch",
        required_prediction_families=sorted(model_hashes),
        prediction_paths=paths,
        model_hashes=model_hashes,
        calibrator_and_policy_hashes=policy_hashes,
        **_boundary_hashes(project_root),
    )
    lock["prediction_paths"] = {
        lock_id: detection_path.relative_to(project_root).as_posix()
        for lock_id in sorted(model_hashes)
    }
    lock["concept_prediction_path"] = concept_path.relative_to(project_root).as_posix()
    lock["concept_prediction_sha256"] = sha256_file(concept_path)
    lock["calibrator_store_path"] = calibrator_path.relative_to(project_root).as_posix()
    lock["calibrator_store_sha256"] = sha256_file(calibrator_path)
    lock_path = output_root / "prediction_lock.json"
    close_prediction_lock(lock_path, lock)
    return {
        "status": "pass",
        "output_paths": [
            detection_path.relative_to(project_root).as_posix(),
            concept_path.relative_to(project_root).as_posix(),
            lock_path.relative_to(project_root).as_posix(),
        ],
        "details": {
            "run_count": len(specs),
            "detection_count": len(detections),
            "concept_row_count": len(concepts_frame),
            "test_truth_opened": False,
        },
    }
