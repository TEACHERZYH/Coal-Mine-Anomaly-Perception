from __future__ import annotations

import json
from pathlib import Path
import pickle
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image

from .data.corruptions import (
    CORRUPTION_SEVERITIES,
    CORRUPTION_TYPES,
    apply_corruption,
    array_sha256,
    build_corruption_record,
    validate_corruption_manifest,
)
from .governance.prediction_lock import build_prediction_lock, close_prediction_lock
from .predict.detection import _boundary_hashes, _predict_records, _run_specs
from .provenance import branch_artifact_path, canonical_json_sha256, sha256_file
from .training_data import _ontology_mapping, dataset_ids_for_role, load_frozen_protocol
from .workflow_common import (
    WorkflowExecutionError,
    write_json_artifact,
    write_parquet_artifact,
)


OPERATOR_VERSION = "locked_photometric_v1"


def _probe(run_root: Path, step_id: str) -> Path:
    path = run_root / "slurm_environment_probe.json"
    if not path.is_file():
        raise WorkflowExecutionError(f"{step_id} lacks its Slurm environment probe")
    return path


def _corrupted_path(project_root: Path, record_id: str, kind: str, severity: int) -> Path:
    record_key = canonical_json_sha256({"record_id": str(record_id)})[:24]
    return project_root / f"data/locked/corruptions/{record_key}/{kind}-{severity}.png"


def build_locked_corruptions(
    project_root: Path,
    run_root: Path,
    config_path: Optional[Path],
) -> Dict[str, Any]:
    del config_path
    probe = _probe(run_root, "E220")
    protocol = load_frozen_protocol(project_root)
    robustness = protocol["evaluation"]["robustness"]
    kinds = tuple(str(value) for value in robustness["corruption_types"])
    severities = tuple(int(value) for value in robustness["severity_levels"])
    if set(kinds) != CORRUPTION_TYPES or set(severities) != CORRUPTION_SEVERITIES:
        raise WorkflowExecutionError("E220 corruption grid drifted from the implementation contract")
    dataset_id = dataset_ids_for_role(project_root, "primary_visual_target")[0]
    features = pd.read_parquet(branch_artifact_path(project_root, "feature_store"))
    records = features.loc[
        (features["dataset_id"].astype(str) == dataset_id)
        & (features["modality"].astype(str).str.lower() == "visible")
    ].copy()
    if records.empty or records["record_id"].duplicated().any():
        raise WorkflowExecutionError("E220 has no unique primary visual test records")
    operator_hash = sha256_file(Path(__file__).parent / "data/corruptions.py")
    rows = []
    for item in records.sort_values("record_id").itertuples(index=False):
        source = project_root / str(item.relative_path)
        if not source.is_file() or sha256_file(source) != str(item.sha256):
            raise WorkflowExecutionError(f"E220 source image hash drifted: {item.record_id}")
        with Image.open(source) as decoded:
            image = np.asarray(decoded.convert("RGB"), dtype=np.uint8)
        for kind in kinds:
            for severity in severities:
                corrupted = apply_corruption(
                    image,
                    record_id=str(item.record_id),
                    corruption_type=kind,
                    severity=severity,
                    operator_version=OPERATOR_VERSION,
                )
                target = _corrupted_path(
                    project_root, str(item.record_id), kind, severity
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    with Image.open(target) as existing:
                        existing_values = np.asarray(existing.convert("RGB"), dtype=np.uint8)
                    if array_sha256(existing_values) != array_sha256(corrupted):
                        raise WorkflowExecutionError(f"E220 immutable corruption drifted: {target}")
                else:
                    Image.fromarray(corrupted, mode="RGB").save(target, format="PNG")
                row = build_corruption_record(
                    image=image,
                    record_id=str(item.record_id),
                    raw_group_id=str(item.raw_group_id),
                    corruption_type=kind,
                    severity=severity,
                    operator_version=OPERATOR_VERSION,
                    generator_hash=operator_hash,
                    source_feature_hash=str(item.sha256),
                )
                if row["corrupted_feature_hash"] != array_sha256(corrupted):
                    raise WorkflowExecutionError("E220 corruption reconstruction is not deterministic")
                rows.append(row)
    manifest = validate_corruption_manifest(pd.DataFrame.from_records(rows))
    expected = len(records) * len(kinds) * len(severities)
    if len(manifest) != expected:
        raise WorkflowExecutionError("E220 corruption grid coverage is incomplete")
    output = project_root / "data/locked/corruption_manifest.parquet"
    write_parquet_artifact(output, manifest)
    summary = project_root / "evidence/robustness/corruption_build.json"
    write_json_artifact(
        summary,
        {
            "schema_version": 1,
            "step_id": "E220",
            "status": "pass",
            "record_count": len(records),
            "cell_count": len(kinds) * len(severities),
            "manifest_row_count": len(manifest),
            "manifest_sha256": sha256_file(output),
            "operator_version": OPERATOR_VERSION,
            "operator_sha256": operator_hash,
            "label_inputs_used": False,
            "slurm_probe_sha256": sha256_file(probe),
        },
    )
    return {
        "status": "pass",
        "output_paths": [
            output.relative_to(project_root).as_posix(),
            summary.relative_to(project_root).as_posix(),
        ],
        "details": {"record_count": len(records), "manifest_row_count": len(manifest)},
    }


def _load_v2_calibrators(project_root: Path, lock_id: str) -> Dict[str, Any]:
    store = pickle.loads((project_root / "predictions/locked/V2/calibrators.pkl").read_bytes())
    if lock_id not in store:
        raise WorkflowExecutionError(f"R1 cannot locate V2 calibrators: {lock_id}")
    payload = pickle.loads(store[lock_id])
    calibrators = payload.get("calibrators")
    if not isinstance(calibrators, dict) or not calibrators:
        raise WorkflowExecutionError(f"R1 V2 calibrator payload is invalid: {lock_id}")
    return calibrators


def predict_r1_sealed(
    project_root: Path,
    run_root: Path,
    config_path: Optional[Path],
    *,
    yolo_factory: Optional[Callable[[str], Any]] = None,
) -> Dict[str, Any]:
    del config_path
    _probe(run_root, "E221")
    if yolo_factory is None:
        from ultralytics import YOLO

        yolo_factory = YOLO
    protocol = load_frozen_protocol(project_root)
    _, concepts = _ontology_mapping(project_root)
    manifest_path = project_root / "data/locked/corruption_manifest.parquet"
    manifest = validate_corruption_manifest(pd.read_parquet(manifest_path))
    base_features = pd.read_parquet(branch_artifact_path(project_root, "feature_store"))
    dataset_id = dataset_ids_for_role(project_root, "primary_visual_target")[0]
    base = base_features.loc[
        (base_features["dataset_id"].astype(str) == dataset_id)
        & (base_features["modality"].astype(str).str.lower() == "visible")
    ][["record_id", "raw_group_id"]]
    records = manifest.merge(base, on=["record_id", "raw_group_id"], validate="many_to_one")
    records["relative_path"] = [
        _corrupted_path(project_root, record_id, kind, int(severity))
        .relative_to(project_root)
        .as_posix()
        for record_id, kind, severity in records[
            ["record_id", "corruption_type", "severity"]
        ].itertuples(index=False, name=None)
    ]
    for item in records.itertuples(index=False):
        path = project_root / str(item.relative_path)
        with Image.open(path) as decoded:
            values = np.asarray(decoded.convert("RGB"), dtype=np.uint8)
        if array_sha256(values) != str(item.corrupted_feature_hash):
            raise WorkflowExecutionError(f"R1 corrupted feature hash drifted: {item.record_id}")
    specs = [
        item
        for item in _run_specs(project_root, "V2")
        if item["family_id"] in {"V2-A-10-GEN", "V2-A-10-MULTI"}
    ]
    if len(specs) != 6:
        raise WorkflowExecutionError("R1 requires three generic and three multi checkpoints")
    detection_parts = []
    model_hashes = {}
    policy_hashes = {}
    imgsz = int(protocol["models"]["visible"]["input_size"])
    for spec in specs:
        calibrators = _load_v2_calibrators(project_root, str(spec["lock_id"]))
        model = yolo_factory(str(spec["checkpoint"]))
        run_parts = []
        for (kind, severity), cell in records.groupby(
            ["corruption_type", "severity"], sort=True
        ):
            detections = _predict_records(
                model,
                cell,
                project_root=project_root,
                concepts=concepts,
                imgsz=imgsz,
            )
            if not detections.empty:
                calibrated = []
                for concept, part in detections.groupby("concept_id", sort=True):
                    copied = part.copy()
                    copied["score_calibrated"] = calibrators[str(concept)].predict(
                        copied["score_raw"]
                    ).astype(np.float32)
                    calibrated.append(copied)
                detections = pd.concat(calibrated, ignore_index=True)
            else:
                detections["score_calibrated"] = pd.Series(dtype="float32")
            detections["corruption_type"] = str(kind)
            detections["severity"] = int(severity)
            run_parts.append(detections)
        run = pd.concat(run_parts, ignore_index=True)
        r1_family = "R1-GEN" if spec["family_id"] == "V2-A-10-GEN" else "R1-MULTI"
        model_hash = sha256_file(spec["checkpoint"])
        run_id = canonical_json_sha256(
            {
                "family_id": r1_family,
                "train_seed": spec.get("train_seed"),
                "subset_seed": spec.get("subset_seed"),
                "model_sha256": model_hash,
                "corruption_manifest_sha256": sha256_file(manifest_path),
            }
        )
        run["detection_id"] = run.groupby(
            ["record_id", "corruption_type", "severity"]
        ).cumcount().astype("int64")
        run = run.assign(
            run_id=run_id,
            family_id=r1_family,
            train_seed=spec.get("train_seed"),
            subset_seed=spec.get("subset_seed"),
            prediction_role="branch_test_corruption",
            pool_role="D_b_te",
            modality="visible",
            available=True,
            model_hash=model_hash,
        )
        detection_parts.append(run)
        lock_id = f"{r1_family}-seed-{spec.get('train_seed')}-subset-{spec.get('subset_seed')}"
        model_hashes[lock_id] = model_hash
        policy_hashes[lock_id] = canonical_json_sha256(
            {
                "source_v2_lock_id": spec["lock_id"],
                "corruption_manifest_sha256": sha256_file(manifest_path),
                "nms_iou": 0.70,
                "inference_confidence_floor": 0.001,
            }
        )
    predictions = pd.concat(detection_parts, ignore_index=True)
    output = project_root / "predictions/locked/R1.parquet"
    write_parquet_artifact(output, predictions)
    lock = build_prediction_lock(
        scope="branch",
        required_prediction_families=sorted(model_hashes),
        prediction_paths={key: output for key in model_hashes},
        model_hashes=model_hashes,
        calibrator_and_policy_hashes=policy_hashes,
        **_boundary_hashes(project_root),
    )
    lock["prediction_paths"] = {
        key: output.relative_to(project_root).as_posix() for key in model_hashes
    }
    lock["corruption_manifest_path"] = manifest_path.relative_to(project_root).as_posix()
    lock["corruption_manifest_sha256"] = sha256_file(manifest_path)
    lock_path = project_root / "predictions/locked/R1/prediction_lock.json"
    close_prediction_lock(lock_path, lock)
    return {
        "status": "pass",
        "output_paths": [
            output.relative_to(project_root).as_posix(),
            lock_path.relative_to(project_root).as_posix(),
        ],
        "details": {
            "run_count": len(specs),
            "prediction_count": len(predictions),
            "test_truth_opened": False,
        },
    }
