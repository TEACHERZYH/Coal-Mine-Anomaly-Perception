from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ..data.episodes import validate_fusion_eligibility_lock
from ..governance.immutable import write_once_json
from ..governance.prediction_lock import (
    assert_truth_free_prediction,
    validate_prediction_lock,
)
from ..models.yolo_adapter import aggregate_concept_scores
from ..provenance import canonical_json_sha256, sha256_file
from ..train.methane import MethaneArrays, TRAINING_SEEDS, _read_history_payloads
from ..training_data import _ontology_mapping, load_frozen_protocol
from ..workflow_common import WorkflowExecutionError, load_json, write_parquet_artifact
from .detection import _predict_records, _run_specs as detection_run_specs
from .methane import _gru_scores


GENERATOR_MODALITIES = {
    "T1-VIS": "visible",
    "T1-THERM": "thermal",
    "V2-A-10-MULTI": "visible",
    "S1-GRU": "methane",
}
PROJECTION_COLUMNS = {
    "skeleton_item_id",
    "record_id",
    "concept_id",
    "node_id",
    "generator_family_id",
}
FEATURE_COLUMNS = {
    "dataset_id",
    "record_id",
    "raw_group_id",
    "pair_id",
    "modality",
    "feature_kind",
    "relative_path",
    "history_json",
    "source_feature_sha256",
}
FORBIDDEN_INPUT_TOKENS = {
    "episode_id",
    "step_index",
    "pool",
    "episode_seed",
    "template_instance_id",
    "template_family",
    "event_position_role",
    "event_truth",
    "state_truth",
    "label",
    "truth",
    "future",
}


def _median_validation_rank(
    records: Sequence[Mapping[str, Any]], *, family_id: str
) -> Dict[str, Any]:
    candidates = [
        dict(item) for item in records if str(item.get("family_id")) == family_id
    ]
    if len(candidates) != 3:
        raise WorkflowExecutionError(
            f"{family_id} representative selection requires three validation repeats"
        )
    for item in candidates:
        if str(item.get("selection_pool")) != "D_b_sel":
            raise WorkflowExecutionError(
                f"{family_id} representative selection escaped D_b_sel"
            )
        value = float(item.get("selection_metric_value", np.nan))
        if not np.isfinite(value):
            raise WorkflowExecutionError(
                f"{family_id} representative selection metric is non-finite"
            )
    ranked = sorted(
        candidates,
        key=lambda item: (
            -float(item["selection_metric_value"]),
            int(item.get("train_seed", 0)),
            int(item.get("subset_seed") or 0),
        ),
    )
    return ranked[len(ranked) // 2]


def _selected_specs(
    project_root: Path, required_families: set[str]
) -> Dict[str, Dict[str, Any]]:
    specs: Dict[str, Dict[str, Any]] = {}
    if required_families.intersection({"T1-VIS", "T1-THERM"}):
        for item in detection_run_specs(project_root, "T1"):
            family = str(item["family_id"])
            if family in required_families:
                specs[family] = dict(item)
    if "V2-A-10-MULTI" in required_families:
        payload = load_json(project_root / "runs/V2/checkpoint_selection.json")
        item = _median_validation_rank(
            payload.get("selected", []), family_id="V2-A-10-MULTI"
        )
        checkpoint = project_root / str(item["checkpoint_path"])
        if not checkpoint.is_file() or sha256_file(checkpoint) != item.get(
            "checkpoint_sha256"
        ):
            raise WorkflowExecutionError("Selected V2 episode checkpoint drifted")
        specs["V2-A-10-MULTI"] = {
            **item,
            "checkpoint": checkpoint,
            "lock_id": (
                f"V2-A-10-MULTI-seed-{int(item['train_seed'])}"
                f"-subset-{int(item['subset_seed'])}"
            ),
            "modality": "visible",
        }
    if "S1-GRU" in required_families:
        manifests = []
        for seed in TRAINING_SEEDS:
            path = project_root / f"runs/S1/gru/S1-GRU/seed-{seed}/run_manifest.json"
            item = load_json(path)
            if int(item.get("train_seed", -1)) != seed:
                raise WorkflowExecutionError("S1 GRU seed manifest drifted")
            manifests.append(item)
        item = _median_validation_rank(manifests, family_id="S1-GRU")
        checkpoint = project_root / str(item["checkpoint_path"])
        if not checkpoint.is_file() or sha256_file(checkpoint) != item.get(
            "checkpoint_sha256"
        ):
            raise WorkflowExecutionError("Selected S1 episode checkpoint drifted")
        seed = int(item["train_seed"])
        specs["S1-GRU"] = {
            **item,
            "checkpoint": checkpoint,
            "lock_id": f"S1-GRU-seed-{seed}",
            "modality": "methane",
        }
    if set(specs) != required_families:
        raise WorkflowExecutionError(
            f"Episode generator selection is incomplete: {sorted(required_families - set(specs))}"
        )
    return specs


def _package_for_family(family_id: str) -> str:
    if family_id.startswith("T1-"):
        return "T1"
    if family_id.startswith("V2-"):
        return "V2"
    if family_id == "S1-GRU":
        return "S1"
    raise WorkflowExecutionError(f"Unknown episode generator family: {family_id}")


def _load_locked_calibrator(
    project_root: Path, *, family_id: str, lock_id: str, checkpoint_hash: str
) -> tuple[Any, str, bytes]:
    package = _package_for_family(family_id)
    lock_path = project_root / f"predictions/locked/{package}/prediction_lock.json"
    lock = load_json(lock_path)
    validate_prediction_lock(lock)
    if lock_id not in set(lock["required_prediction_families"]):
        raise WorkflowExecutionError(
            f"Frozen {package} prediction lock lacks representative {lock_id}"
        )
    if lock["model_hashes"].get(lock_id) != checkpoint_hash:
        raise WorkflowExecutionError(
            f"Frozen {package} model hash differs from the selected checkpoint"
        )
    store_path = project_root / str(lock.get("calibrator_store_path", ""))
    if (
        not store_path.is_file()
        or sha256_file(store_path) != lock.get("calibrator_store_sha256")
    ):
        raise WorkflowExecutionError(f"Frozen {package} calibrator store drifted")
    store = pickle.loads(store_path.read_bytes())
    payload_bytes = store.get(lock_id)
    if not isinstance(payload_bytes, bytes):
        raise WorkflowExecutionError(f"Frozen calibrator payload is missing: {lock_id}")
    payload = pickle.loads(payload_bytes)
    if family_id == "S1-GRU":
        calibrator = payload.get("calibrator")
    else:
        calibrator = payload.get("calibrators")
    if calibrator is None:
        raise WorkflowExecutionError(f"Frozen calibrator payload is malformed: {lock_id}")
    policy_hash = str(lock["calibrator_and_policy_hashes"][lock_id])
    return calibrator, policy_hash, payload_bytes


def _read_truth_free_inputs(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    projection_path = (
        project_root / "data/seals/episode_skeleton_inference_projection.parquet"
    )
    feature_path = project_root / "data/locked/episode_branch_feature_manifest.parquet"
    projection = pd.read_parquet(projection_path)
    features = pd.read_parquet(feature_path)
    if set(projection.columns) != PROJECTION_COLUMNS:
        raise WorkflowExecutionError("E300 inference projection schema drifted")
    if not FEATURE_COLUMNS.issubset(features.columns):
        raise WorkflowExecutionError("E300 branch feature manifest schema is incomplete")
    forbidden = sorted(
        column
        for column in [*projection.columns, *features.columns]
        if str(column).lower() in FORBIDDEN_INPUT_TOKENS
        or str(column).lower().endswith(("_truth", "_label", "_target"))
    )
    if forbidden:
        raise WorkflowExecutionError(f"E300 input exposes forbidden fields: {forbidden}")
    if projection.empty or projection["skeleton_item_id"].duplicated().any():
        raise WorkflowExecutionError("E300 inference projection is empty or non-unique")
    if features["record_id"].astype(str).duplicated().any():
        raise WorkflowExecutionError("E300 feature records are not globally unique")
    joined = projection.merge(features, on="record_id", validate="many_to_one")
    if len(joined) != len(projection):
        raise WorkflowExecutionError("E300 projection records do not resolve to features")
    for item in joined.itertuples(index=False):
        family = str(item.generator_family_id)
        if family not in GENERATOR_MODALITIES:
            raise WorkflowExecutionError(f"E300 generator is not frozen: {family}")
        expected_modality = GENERATOR_MODALITIES[family]
        if str(item.modality) != expected_modality:
            raise WorkflowExecutionError(
                f"E300 feature modality differs from {family}: {item.record_id}"
            )
        expected_kind = "causal_history_json" if family == "S1-GRU" else "image_path"
        if str(item.feature_kind) != expected_kind:
            raise WorkflowExecutionError(
                f"E300 feature kind differs from {family}: {item.record_id}"
            )
    return projection, joined


def _image_rows(
    project_root: Path,
    rows: pd.DataFrame,
    *,
    spec: Mapping[str, Any],
    calibrators: Mapping[str, Any],
    model_hash: str,
    policy_hash: str,
    projection_hash: str,
    skeleton_hash: str,
    concepts: Sequence[str],
    imgsz: int,
    yolo_factory: Callable[[str], Any],
) -> list[Dict[str, Any]]:
    records = rows[
        ["record_id", "raw_group_id", "relative_path"]
    ].drop_duplicates("record_id")
    model = yolo_factory(str(spec["checkpoint"]))
    detections = _predict_records(
        model,
        records,
        project_root=project_root,
        concepts=concepts,
        imgsz=imgsz,
    )
    raw = aggregate_concept_scores(
        detections,
        record_ids=records["record_id"].astype(str),
        concept_ids=concepts,
        modality=str(spec["modality"]),
        post_nms=True,
    ).set_index(["record_id", "concept_id"])["step_score_raw"]
    output = []
    for item in rows.itertuples(index=False):
        key = (str(item.record_id), str(item.concept_id))
        if key not in raw.index or str(item.concept_id) not in calibrators:
            raise WorkflowExecutionError(f"E300 image concept is not calibratable: {key}")
        score = float(raw.loc[key])
        calibrated = float(calibrators[str(item.concept_id)].predict([score])[0])
        output.append(
            _prediction_row(
                item,
                score=score,
                calibrated=calibrated,
                model_hash=model_hash,
                policy_hash=policy_hash,
                projection_hash=projection_hash,
                skeleton_hash=skeleton_hash,
            )
        )
    return output


def _methane_rows(
    rows: pd.DataFrame,
    *,
    spec: Mapping[str, Any],
    calibrator: Any,
    model_hash: str,
    policy_hash: str,
    projection_hash: str,
    skeleton_hash: str,
    protocol: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    records = rows[["record_id", "raw_group_id", "history_json"]].drop_duplicates(
        "record_id"
    )
    history, names = _read_history_payloads(records)
    arrays = MethaneArrays(
        history=history,
        labels=None,
        metadata=records.rename(columns={"record_id": "window_id"}),
        feature_names=names,
    )
    scores = _gru_scores(arrays, Path(spec["checkpoint"]), protocol)
    score_by_record = dict(zip(records["record_id"].astype(str), scores))
    output = []
    for item in rows.itertuples(index=False):
        score = float(score_by_record[str(item.record_id)])
        calibrated = float(calibrator.predict([score])[0])
        output.append(
            _prediction_row(
                item,
                score=score,
                calibrated=calibrated,
                model_hash=model_hash,
                policy_hash=policy_hash,
                projection_hash=projection_hash,
                skeleton_hash=skeleton_hash,
            )
        )
    return output


def _prediction_row(
    item: Any,
    *,
    score: float,
    calibrated: float,
    model_hash: str,
    policy_hash: str,
    projection_hash: str,
    skeleton_hash: str,
) -> Dict[str, Any]:
    if not np.isfinite(score) or not np.isfinite(calibrated):
        raise WorkflowExecutionError("E300 emitted a non-finite branch probability")
    if not 0 <= score <= 1 or not 0 <= calibrated <= 1:
        raise WorkflowExecutionError("E300 branch probability lies outside [0, 1]")
    return {
        "skeleton_item_id": str(item.skeleton_item_id),
        "record_id": str(item.record_id),
        "concept_id": str(item.concept_id),
        "node_id": str(item.node_id),
        "modality": str(item.modality),
        "step_score_raw": np.float32(score),
        "calibrated_probability": np.float32(calibrated),
        "available": True,
        "generator_family_id": str(item.generator_family_id),
        "model_hash": model_hash,
        "calibrator_or_policy_hash": policy_hash,
        "skeleton_manifest_hash": skeleton_hash,
        "skeleton_inference_projection_hash": projection_hash,
    }


def predict_episode_branches(
    project_root: Path,
    run_root: Path,
    config_path: Optional[Path],
    *,
    yolo_factory: Optional[Callable[[str], Any]] = None,
) -> Dict[str, Any]:
    del config_path
    probe = run_root / "slurm_environment_probe.json"
    if not probe.is_file():
        raise WorkflowExecutionError("E300 lacks its Slurm environment probe")
    eligibility_path = project_root / "data/locked/fusion_eligibility_lock.parquet"
    eligibility = validate_fusion_eligibility_lock(
        pd.read_parquet(eligibility_path)
    )
    if not eligibility["fusion_primary_eligible"].astype(bool).any():
        raise WorkflowExecutionError(
            "E300 must not be submitted when the fusion population is empty"
        )
    protocol = load_frozen_protocol(project_root)
    projection, joined = _read_truth_free_inputs(project_root)
    required_families = set(joined["generator_family_id"].astype(str))
    specs = _selected_specs(project_root, required_families)
    if yolo_factory is None and required_families - {"S1-GRU"}:
        from ultralytics import YOLO

        yolo_factory = YOLO
    projection_path = (
        project_root / "data/seals/episode_skeleton_inference_projection.parquet"
    )
    projection_hash = sha256_file(projection_path)
    skeleton_hashes = set(eligibility["skeleton_manifest_sha256"].astype(str))
    if len(skeleton_hashes) != 1:
        raise WorkflowExecutionError("E300 eligibility lock has inconsistent skeleton hashes")
    skeleton_hash = next(iter(skeleton_hashes))
    _, concepts = _ontology_mapping(project_root)
    output_rows: list[Dict[str, Any]] = []
    selection_rows = []
    for family_id in sorted(required_families):
        spec = specs[family_id]
        model_hash = sha256_file(Path(spec["checkpoint"]))
        calibrator, policy_hash, calibrator_bytes = _load_locked_calibrator(
            project_root,
            family_id=family_id,
            lock_id=str(spec["lock_id"]),
            checkpoint_hash=model_hash,
        )
        family_rows = joined.loc[
            joined["generator_family_id"].astype(str) == family_id
        ].copy()
        if family_id == "S1-GRU":
            output_rows.extend(
                _methane_rows(
                    family_rows,
                    spec=spec,
                    calibrator=calibrator,
                    model_hash=model_hash,
                    policy_hash=policy_hash,
                    projection_hash=projection_hash,
                    skeleton_hash=skeleton_hash,
                    protocol=protocol,
                )
            )
        else:
            assert yolo_factory is not None
            output_rows.extend(
                _image_rows(
                    project_root,
                    family_rows,
                    spec=spec,
                    calibrators=calibrator,
                    model_hash=model_hash,
                    policy_hash=policy_hash,
                    projection_hash=projection_hash,
                    skeleton_hash=skeleton_hash,
                    concepts=concepts,
                    imgsz=int(protocol["models"]["visible"]["input_size"]),
                    yolo_factory=yolo_factory,
                )
            )
        selection_rows.append(
            {
                "family_id": family_id,
                "lock_id": str(spec["lock_id"]),
                "train_seed": spec.get("train_seed"),
                "subset_seed": spec.get("subset_seed"),
                "selection_pool": spec.get("selection_pool"),
                "selection_metric_value": spec.get("selection_metric_value"),
                "checkpoint_sha256": model_hash,
                "calibrator_payload_sha256": hashlib.sha256(
                    calibrator_bytes
                ).hexdigest(),
                "calibrator_or_policy_hash": policy_hash,
            }
        )
    predictions = pd.DataFrame.from_records(output_rows).sort_values(
        "skeleton_item_id"
    )
    if len(predictions) != len(projection) or predictions[
        "skeleton_item_id"
    ].duplicated().any():
        raise WorkflowExecutionError("E300 prediction coverage is incomplete")
    if set(predictions["skeleton_item_id"].astype(str)) != set(
        projection["skeleton_item_id"].astype(str)
    ):
        raise WorkflowExecutionError("E300 predictions do not close the projection")
    output = project_root / "predictions/episode_branches/all.parquet"
    write_parquet_artifact(output, predictions)
    assert_truth_free_prediction(output)
    selection_payload = {
        "schema_version": 1,
        "step_id": "E300",
        "status": "pass",
        "selection_rule": "median_validation_rank_then_smallest_seed_or_locked_single_seed",
        "test_information_used": False,
        "full_skeleton_opened": False,
        "inference_projection_sha256": projection_hash,
        "skeleton_manifest_sha256": skeleton_hash,
        "generator_selections": selection_rows,
        "selection_sha256": canonical_json_sha256(selection_rows),
    }
    selection_path = project_root / "evidence/episode/E300_generator_selection.json"
    write_once_json(selection_path, selection_payload)
    return {
        "status": "pass",
        "output_paths": [
            output.relative_to(project_root).as_posix(),
            selection_path.relative_to(project_root).as_posix(),
        ],
        "details": {
            "prediction_count": len(predictions),
            "generator_count": len(required_families),
            "test_truth_opened": False,
            "full_skeleton_opened": False,
            "environment_probe_sha256": sha256_file(probe),
        },
    }
