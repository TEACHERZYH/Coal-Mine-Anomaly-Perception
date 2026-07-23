from __future__ import annotations

import csv
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

from .data.seals import (
    DENIED_LABEL_ROLES,
    build_candidate_test_seal,
    validate_test_seal,
)
from .data.manifests import read_file_manifest, read_split_manifest
from .data.ontology import validate_ontology_lock
from .governance.gates import build_gate, write_gate
from .governance.immutable import write_once_bytes, write_once_json
from .governance.prediction_lock import (
    close_prediction_lock,
    validate_prediction_lock,
    verify_prediction_lock,
)
from .governance.release import write_test_release
from .provenance import branch_artifact_path, canonical_json_sha256, sha256_file
from .training_data import target_branch_modalities
from .workflow_common import hash_existing_inputs, load_json, utc_now, write_parquet_artifact


_GATE_NEXT = {
    "G0": ["E010"],
    "G1": ["E100", "E120", "E200"],
    "G2": ["E300"],
    "G3": ["E300"],
    "G4": ["E402"],
    "G5": ["E400", "E402"],
    "G6": ["E500"],
    "G7": ["E900"],
}


def _path(root: Path, relative: str) -> Path:
    return (root / relative).resolve()


def _load_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"YAML root must be a mapping: {path}")
    return payload


def _experiment_steps_contract_sha256(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(not row.get("step_id") for row in rows):
        raise RuntimeError("Experiment-step contract is empty or malformed")
    step_ids = [str(row["step_id"]) for row in rows]
    if len(step_ids) != len(set(step_ids)):
        raise RuntimeError("Experiment-step contract contains duplicate step IDs")
    contract = [
        {key: value for key, value in row.items() if key != "status"}
        for row in rows
    ]
    return canonical_json_sha256(contract)


def _protocol_path(root: Path) -> Path:
    frozen = root / "configs/protocol_lock.pretest.yaml"
    return frozen if frozen.is_file() else root / "configs/protocol_lock.template.yaml"


def _source_hash(root: Path) -> str:
    for relative in (
        "evidence/data/dataset_source_decision.json",
        "evidence/data/staging_receipts.json",
    ):
        candidate = root / relative
        if candidate.is_file():
            return sha256_file(candidate)
    raise RuntimeError("Dataset source evidence is missing")


def _contract_hashes(root: Path) -> Dict[str, str]:
    review = root / "evidence/reviews/I090.json"
    if not review.is_file():
        raise RuntimeError("Passing I090 implementation review is missing")
    return {
        "protocol_hash": sha256_file(_protocol_path(root)),
        "code_hash": sha256_file(review),
        "experiment_matrix_hash": sha256_file(
            root / "configs/experiment_matrix.template.csv"
        ),
        "artifact_contract_hash": sha256_file(
            root / "configs/artifact_contract.template.yaml"
        ),
    }


def _review_digest(root: Path, step_id: str) -> Dict[str, Any]:
    path = root / "evidence/step_reviews" / f"{step_id}.json"
    payload = load_json(path)
    if payload.get("status") not in {"pass", "not_applicable", "accepted_not_applicable"}:
        raise RuntimeError(f"Gate dependency did not pass: {step_id}")
    if payload.get("advance_allowed") is not True:
        raise RuntimeError(f"Gate dependency blocks advancement: {step_id}")
    return {
        "id": step_id,
        "status": "pass",
        "review_status": payload["status"],
        "review_sha256": sha256_file(path),
    }


def _branch_truth_rows(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    files = read_file_manifest(root / "data/locked/file_manifest.parquet")
    split = read_split_manifest(root / "data/locked/split_manifest.parquet")
    joined = files.merge(
        split[["dataset_id", "record_id", "raw_group_id", "pool"]],
        on=["dataset_id", "record_id", "raw_group_id"],
        validate="one_to_one",
    )
    branch_modalities = pd.Series(index=joined.index, dtype="object")
    for dataset_id, indexes in joined.groupby("dataset_id", sort=False).groups.items():
        branch_modalities.loc[indexes] = target_branch_modalities(
            root,
            dataset_id=str(dataset_id),
            source_modalities=joined.loc[indexes, "modality"],
        )
    test = joined.loc[
        (joined["pool"].astype(str) == "D_b_te")
        & branch_modalities.isin({"visible", "thermal"})
    ].copy()
    if test.empty or test["record_id"].astype(str).duplicated().any():
        raise RuntimeError("Branch detection test records are empty or not globally unique")
    ontology = validate_ontology_lock(
        _load_yaml(root / "data/locked/ontology_lock.yaml")
    )
    mapping = {
        (str(item["dataset_id"]), str(item["source_label"])): str(
            item["canonical_concept_id"]
        )
        for item in ontology["entries"]
        if item["mapping_status"] == "compatible"
    }
    concepts = sorted(set(mapping.values()))
    if not concepts:
        raise RuntimeError("Branch truth has no compatible ontology concepts")
    feature_columns = [
        "dataset_id",
        "record_id",
        "archive_id",
        "relative_path",
        "modality",
        "raw_group_id",
        "pair_id",
        "sequence_id",
        "timestamp_or_order",
        "byte_size",
        "sha256",
    ]
    features = test[feature_columns].sort_values(["dataset_id", "record_id"]).reset_index(
        drop=True
    )
    detection_rows = []
    concept_rows = []
    for item in test.sort_values(["dataset_id", "record_id"]).itertuples(index=False):
        summary = json.loads(str(item.label_summary_json))
        if summary.get("negative_annotation_verified") is not True:
            raise RuntimeError(f"Branch test negatives are not verified: {item.record_id}")
        dataset_concepts = {
            concept for (dataset_id, _), concept in mapping.items() if dataset_id == str(item.dataset_id)
        }
        if dataset_concepts != set(concepts):
            raise RuntimeError(
                f"Branch dataset does not support the complete frozen ontology: {item.dataset_id}"
            )
        present = set()
        for box_index, box in enumerate(summary.get("boxes", [])):
            concept = mapping.get((str(item.dataset_id), str(box.get("source_label"))))
            if concept is None:
                continue
            present.add(concept)
            if {"x1", "y1", "x2", "y2"}.issubset(box):
                x1, y1, x2, y2 = (float(box[name]) for name in ("x1", "y1", "x2", "y2"))
            else:
                width = float(summary.get("image_width", 0))
                height = float(summary.get("image_height", 0))
                xc = float(box["x_center_normalized"])
                yc = float(box["y_center_normalized"])
                bw = float(box["width_normalized"])
                bh = float(box["height_normalized"])
                x1, x2 = (xc - bw / 2.0) * width, (xc + bw / 2.0) * width
                y1, y2 = (yc - bh / 2.0) * height, (yc + bh / 2.0) * height
            coordinates = np.asarray([x1, y1, x2, y2], dtype=np.float64)
            if not np.isfinite(coordinates).all() or x2 <= x1 or y2 <= y1:
                raise RuntimeError(f"Branch truth box is invalid: {item.record_id}/{box_index}")
            detection_rows.append(
                {
                    "dataset_id": str(item.dataset_id),
                    "record_id": str(item.record_id),
                    "raw_group_id": str(item.raw_group_id),
                    "concept_id": concept,
                    "box_id": int(box_index),
                    "x1": np.float32(x1),
                    "y1": np.float32(y1),
                    "x2": np.float32(x2),
                    "y2": np.float32(y2),
                }
            )
        for concept in concepts:
            concept_rows.append(
                {
                    "dataset_id": str(item.dataset_id),
                    "record_id": str(item.record_id),
                    "raw_group_id": str(item.raw_group_id),
                    "concept_id": concept,
                    "concept_truth": int(concept in present),
                }
            )
    detection = pd.DataFrame.from_records(
        detection_rows,
        columns=[
            "dataset_id", "record_id", "raw_group_id", "concept_id", "box_id",
            "x1", "y1", "x2", "y2",
        ],
    ).sort_values(["dataset_id", "record_id", "box_id"]).reset_index(drop=True)
    concepts_frame = pd.DataFrame.from_records(concept_rows).sort_values(
        ["dataset_id", "record_id", "concept_id"]
    ).reset_index(drop=True)
    if detection.empty or set(detection["concept_id"].astype(str)) != set(concepts):
        raise RuntimeError("Branch truth does not contain a positive box for every concept")
    return features, detection, concepts_frame


def _verify_branch_truth_bundle(root: Path, manifest_relative: Optional[str] = None) -> Dict[str, Any]:
    manifest_path = root / manifest_relative if manifest_relative else branch_artifact_path(root, "truth_manifest")
    manifest = load_json(manifest_path)
    for field in ("detection_truth", "concept_truth", "methane_truth"):
        item = manifest[field]
        path = root / str(item["path"])
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Branch sealed truth drifted: {field}")
        if len(pd.read_parquet(path)) != int(item["row_count"]):
            raise RuntimeError(f"Branch sealed truth row count drifted: {field}")
    return manifest


def _verify_branch_feature_bundle(root: Path, manifest_relative: Optional[str] = None) -> Dict[str, Any]:
    manifest_path = root / manifest_relative if manifest_relative else branch_artifact_path(root, "feature_manifest")
    manifest = load_json(manifest_path)
    for field in ("detection_features", "methane_features"):
        item = manifest[field]
        path = root / str(item["path"])
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Branch test features drifted: {field}")
        frame = pd.read_parquet(path)
        if len(frame) != int(item["row_count"]):
            raise RuntimeError(f"Branch test feature row count drifted: {field}")
        if any("truth" in str(column).lower() or "label" in str(column).lower() for column in frame):
            raise RuntimeError(f"Branch test feature view contains truth fields: {field}")
    return manifest


def close_gate(
    root: Path, row: Mapping[str, str], arguments: Mapping[str, Any]
) -> Dict[str, Any]:
    gate_id = str(arguments["gate"])
    if gate_id not in _GATE_NEXT:
        raise RuntimeError(f"Unknown evidence gate: {gate_id}")
    expected = f"G{int(row['step_id'][1:]) // 100}" if row["step_id"] == "E549" else None
    gate_by_step = {
        "E006": "G0",
        "E069": "G1",
        "E129": "G2",
        "E209": "G3",
        "E229": "G4",
        "E319": "G5",
        "E409": "G6",
        "E549": "G7",
    }
    if gate_by_step.get(row["step_id"], expected) != gate_id:
        raise RuntimeError("Gate argument does not match the frozen plan step")
    required = [value for value in row["depends_on"].split("|") if value != "none"]
    checks = [_review_digest(root, step_id) for step_id in required]
    inputs = []
    for step_id in required:
        for relative in (
            f"evidence/step_reviews/{step_id}.json",
            f"evidence/command_receipts/{step_id}.json",
        ):
            path = root / relative
            if path.is_file():
                inputs.append({"path": relative, "sha256": sha256_file(path)})
    output = f"evidence/gates/{gate_id}.json"
    payload = build_gate(
        gate_id=gate_id,
        status="pass",
        required_steps=required,
        input_artifacts=inputs,
        output_artifacts=[],
        checks=checks,
        failures=[],
        waivers=[],
        allowed_next_steps=_GATE_NEXT[gate_id],
        **_contract_hashes(root),
    )
    write_gate(root / output, payload)
    return {
        "status": "pass",
        "output_paths": [output],
        "inputs": inputs,
        "details": {"gate_id": gate_id, "reviewed_steps": required},
    }


def create_branch_test_seal(
    root: Path, row: Mapping[str, str], arguments: Mapping[str, Any]
) -> Dict[str, Any]:
    if arguments.get("candidate") is not True:
        raise RuntimeError("The branch seal must first be created as a candidate")
    split = root / "data/locked/split_manifest.parquet"
    labels = root / "data/locked/file_manifest.parquet"
    if not split.is_file() or not labels.is_file():
        raise RuntimeError("Branch seal requires the locked split and file manifests")
    features, detection_truth, concept_truth = _branch_truth_rows(root)
    feature_path = root / "data/locked/branch_test_features.parquet"
    methane_feature_path = root / "data/locked/branch_methane_test_features.parquet"
    methane_truth_path = root / "data/sealed/branch_methane_truth.parquet"
    if not methane_feature_path.is_file() or not methane_truth_path.is_file():
        raise RuntimeError("Branch seal requires E050 methane feature and truth partitions")
    detection_path = root / "data/sealed/branch_detection_truth.parquet"
    concept_path = root / "data/sealed/branch_concept_truth.parquet"
    write_parquet_artifact(feature_path, features)
    write_parquet_artifact(detection_path, detection_truth)
    write_parquet_artifact(concept_path, concept_truth)
    methane_features = pd.read_parquet(methane_feature_path)
    methane_truth = pd.read_parquet(methane_truth_path)
    if (
        methane_features.empty
        or methane_truth.empty
        or set(methane_features["pool"].astype(str)) != {"D_b_te"}
        or set(methane_truth["pool"].astype(str)) != {"D_b_te"}
    ):
        raise RuntimeError("Branch methane test partitions are incomplete")
    feature_manifest_path = root / "data/locked/branch_test_feature_manifest.json"
    feature_manifest = {
        "schema_version": 1,
        "scope": "branch",
        "pool": "D_b_te",
        "detection_features": {
            "path": feature_path.relative_to(root).as_posix(),
            "sha256": sha256_file(feature_path),
            "row_count": len(features),
        },
        "methane_features": {
            "path": methane_feature_path.relative_to(root).as_posix(),
            "sha256": sha256_file(methane_feature_path),
            "row_count": len(methane_features),
        },
    }
    write_once_json(feature_manifest_path, feature_manifest)
    truth_manifest_path = root / "data/sealed/branch_test_truth_manifest.json"
    truth_manifest = {
        "schema_version": 1,
        "scope": "branch",
        "pool": "D_b_te",
        "detection_truth": {
            "path": detection_path.relative_to(root).as_posix(),
            "sha256": sha256_file(detection_path),
            "row_count": len(detection_truth),
        },
        "concept_truth": {
            "path": concept_path.relative_to(root).as_posix(),
            "sha256": sha256_file(concept_path),
            "row_count": len(concept_truth),
        },
        "methane_truth": {
            "path": methane_truth_path.relative_to(root).as_posix(),
            "sha256": sha256_file(methane_truth_path),
            "row_count": len(methane_truth),
        },
        "record_count": len(features),
        "raw_group_count": int(features["raw_group_id"].nunique()),
    }
    write_once_json(truth_manifest_path, truth_manifest)
    payload = build_candidate_test_seal(
        scope="branch",
        split_hash=sha256_file(split),
        feature_manifest_hash=sha256_file(feature_manifest_path),
        label_manifest_hash=sha256_file(truth_manifest_path),
        artifact_contract_hash=sha256_file(
            root / "configs/artifact_contract.template.yaml"
        ),
        feature_acl=["allow:test_predictor", "allow:evaluator"],
        label_acl=sorted(DENIED_LABEL_ROLES),
    )
    payload.update(
        {
            "feature_manifest_path": feature_manifest_path.relative_to(root).as_posix(),
            "label_manifest_path": truth_manifest_path.relative_to(root).as_posix(),
        }
    )
    validate_test_seal(payload)
    output = "data/seals/branch_test_seal_candidate.json"
    write_once_json(root / output, payload)
    return {
        "status": "pass",
        "output_paths": [
            feature_path.relative_to(root).as_posix(),
            feature_manifest_path.relative_to(root).as_posix(),
            detection_path.relative_to(root).as_posix(),
            concept_path.relative_to(root).as_posix(),
            truth_manifest_path.relative_to(root).as_posix(),
            output,
        ],
        "inputs": hash_existing_inputs(
            root,
            [
                "data/locked/split_manifest.parquet",
                "data/locked/file_manifest.parquet",
                "configs/artifact_contract.template.yaml",
            ],
        ),
        "details": {
            "scope": "branch",
            "stage": "candidate",
            "test_record_count": len(features),
            "detection_truth_count": len(detection_truth),
            "concept_truth_count": len(concept_truth),
            "methane_test_window_count": len(methane_truth),
        },
    }


def finalize_branch_test_seal(
    root: Path, row: Mapping[str, str], arguments: Mapping[str, Any]
) -> Dict[str, Any]:
    protocol = _path(root, str(arguments["protocol"]))
    if protocol != (root / "configs/protocol_lock.pretest.yaml").resolve():
        raise RuntimeError("Branch seal must bind the frozen PRETEST protocol")
    candidate_path = root / "data/seals/branch_test_seal_candidate.json"
    candidate = load_json(candidate_path)
    validate_test_seal(candidate)
    if candidate["scope"] != "branch" or candidate["stage"] != "candidate":
        raise RuntimeError("Invalid branch candidate seal")
    feature_path = root / str(candidate.get("feature_manifest_path", ""))
    truth_manifest_path = root / str(candidate.get("label_manifest_path", ""))
    if (
        not feature_path.is_file()
        or sha256_file(feature_path) != candidate["feature_manifest_hash"]
        or not truth_manifest_path.is_file()
        or sha256_file(truth_manifest_path) != candidate["label_manifest_hash"]
    ):
        raise RuntimeError("Branch candidate seal artifact hashes drifted")
    _verify_branch_feature_bundle(root)
    _verify_branch_truth_bundle(root)
    payload = deepcopy(candidate)
    payload.update(
        {
            "stage": "final",
            "sealed_at": utc_now(),
            "candidate_seal_hash": sha256_file(candidate_path),
            "protocol_hash": sha256_file(protocol),
            "source_hash": _source_hash(root),
            "matrix_hash": sha256_file(root / "configs/experiment_matrix.template.csv"),
        }
    )
    validate_test_seal(payload)
    output = "data/seals/branch_test_seal.json"
    write_once_json(root / output, payload)
    return {
        "status": "pass",
        "output_paths": [output],
        "inputs": hash_existing_inputs(
            root,
            [
                "configs/protocol_lock.pretest.yaml",
                "data/seals/branch_test_seal_candidate.json",
                "evidence/data/dataset_source_decision.json",
                "configs/experiment_matrix.template.csv",
            ],
        ),
        "details": {"scope": "branch", "stage": "final"},
    }


def _merge_prediction_locks(
    root: Path,
    *,
    scope: str,
    lock_paths: Sequence[str],
    output: str,
) -> Path:
    merged: Dict[str, Any] = {
        "scope": scope,
        "required_prediction_families": [],
        "prediction_store_hashes": {},
        "model_hashes": {},
        "calibrator_and_policy_hashes": {},
        "truth_fields_scan_pass": True,
        "closed_at": utc_now(),
    }
    boundary_fields = ("seal_hash", "protocol_hash", "source_hash", "matrix_hash")
    for relative in lock_paths:
        path = root / relative
        payload = load_json(path)
        validate_prediction_lock(payload)
        if payload["scope"] != scope:
            raise RuntimeError(f"Prediction lock scope mismatch: {relative}")
        prediction_paths = {
            family: root / prediction_path
            for family, prediction_path in payload.get("prediction_paths", {}).items()
        }
        if set(prediction_paths) != set(payload["required_prediction_families"]):
            raise RuntimeError(f"Prediction lock does not expose verifiable paths: {relative}")
        verify_prediction_lock(path, prediction_paths)
        for field in boundary_fields:
            observed = payload[field]
            if field in merged and merged[field] != observed:
                raise RuntimeError(f"Prediction lock boundary drift: {field}")
            merged[field] = observed
        for family in payload["required_prediction_families"]:
            if family in merged["prediction_store_hashes"]:
                raise RuntimeError(f"Prediction family appears in two locks: {family}")
            merged["required_prediction_families"].append(family)
            for field in (
                "prediction_store_hashes",
                "model_hashes",
                "calibrator_and_policy_hashes",
            ):
                merged[field][family] = payload[field][family]
    merged["required_prediction_families"] = sorted(
        merged["required_prediction_families"]
    )
    validate_prediction_lock(merged)
    target = root / output
    close_prediction_lock(target, merged)
    return target


def _seal_path_for_prediction_locks(root: Path, lock_paths: Sequence[str]) -> Path:
    seal_hashes = set()
    for relative in lock_paths:
        payload = load_json(root / relative)
        validate_prediction_lock(payload)
        seal_hashes.add(str(payload["seal_hash"]))
    if len(seal_hashes) != 1:
        raise RuntimeError("Selected branch prediction locks bind different seals")
    expected_hash = next(iter(seal_hashes))
    candidates = [
        root / "data/seals/branch_test_seal.json",
        *sorted((root / "data/seals").glob("branch_test_seal.*.json")),
    ]
    for candidate in candidates:
        if candidate.is_file() and sha256_file(candidate) == expected_hash:
            return candidate
    raise RuntimeError(f"No local branch seal matches prediction lock hash: {expected_hash}")


def release_branch_test(
    root: Path, row: Mapping[str, str], arguments: Mapping[str, Any]
) -> Dict[str, Any]:
    lock_by_package = {
        "T1": "predictions/locked/T1/prediction_lock.json",
        "S1": "predictions/locked/S1/prediction_lock.json",
        "V2": "predictions/locked/V2/prediction_lock.json",
        "R1": "predictions/locked/R1/prediction_lock.json",
    }
    requested_packages = tuple(str(value).upper() for value in arguments.get("packages", ()))
    if not requested_packages:
        requested_packages = tuple(lock_by_package)
    unknown_packages = sorted(set(requested_packages) - set(lock_by_package))
    if unknown_packages:
        raise RuntimeError(f"Unknown branch release package(s): {unknown_packages}")
    if len(set(requested_packages)) != len(requested_packages):
        raise RuntimeError("Branch release packages must be unique")
    locks = [lock_by_package[package] for package in requested_packages]
    seal = _seal_path_for_prediction_locks(root, locks)
    seal_payload = load_json(seal)
    feature_manifest_path = seal_payload.get("feature_manifest_path")
    label_manifest_path = seal_payload.get("label_manifest_path")
    _verify_branch_feature_bundle(
        root, str(feature_manifest_path) if feature_manifest_path else None
    )
    truth_manifest = _verify_branch_truth_bundle(
        root, str(label_manifest_path) if label_manifest_path else None
    )
    aggregate_relative = "predictions/locked/branch_prediction_lock.json"
    aggregate = _merge_prediction_locks(
        root,
        scope="branch",
        lock_paths=locks,
        output=aggregate_relative,
    )
    protocol = root / "configs/protocol_lock.pretest.yaml"
    output = "data/releases/branch_test_release.json"
    write_test_release(
        output_path=root / output,
        scope="branch",
        seal_path=seal,
        prediction_lock_path=aggregate,
        protocol_hash=sha256_file(protocol),
        source_hash=_source_hash(root),
        matrix_hash=sha256_file(root / "configs/experiment_matrix.template.csv"),
        evaluator_identity="mining1_independent_evaluator",
        evaluator_label_acl=["allow:evaluator:mining1_independent_evaluator"],
        signer_identity="mining1_release_authority",
    )
    return {
        "status": "pass",
        "output_paths": [aggregate_relative, output],
        "inputs": hash_existing_inputs(root, [*locks, str(seal.relative_to(root)), str(protocol.relative_to(root))]),
        "details": {
            "scope": "branch",
            "packages": list(requested_packages),
            "prediction_lock_count": len(locks),
            "sealed_detection_truth_sha256": truth_manifest["detection_truth"]["sha256"],
            "sealed_concept_truth_sha256": truth_manifest["concept_truth"]["sha256"],
            "sealed_methane_truth_sha256": truth_manifest["methane_truth"]["sha256"],
        },
    }


def _precision_feasibility(root: Path, protocol: Mapping[str, Any]) -> Dict[str, Any]:
    endpoints = {
        "V2": (5.0, 0.10),
        "S1": (5.0, 0.10),
        "E2_E3": (5.0, 0.10),
        "E3_RELIABILITY": (3.0, 0.06),
        "E3_GRAPH": (3.0, 0.06),
        "E3_MEMORY": (3.0, 0.06),
        "R1": (5.0, 0.10),
    }
    rng = np.random.default_rng(int(protocol["seeds"]["statistics"]))
    simulations = int(protocol["data"]["precision_feasibility"]["simulation_repetitions"])
    rows = []
    for endpoint, (effect_pp, paired_sd) in endpoints.items():
        effect = effect_pp / 100.0
        required = max(2, math.ceil((1.96 * paired_sd / effect) ** 2))
        draws = rng.normal(effect, paired_sd / math.sqrt(required), simulations)
        success_probability = float(np.mean(np.abs(draws - effect) <= effect))
        rows.append(
            {
                "endpoint": endpoint,
                "minimum_meaningful_effect_pp": effect_pp,
                "conservative_paired_sd": paired_sd,
                "simulated_independent_group_requirement": required,
                "success_probability": success_probability,
                "uses_model_predictions": False,
                "uses_test_outcomes": False,
            }
        )
    hard_min = int(protocol["data"]["independent_group_floor_hard_min"])
    return {
        "schema_version": 1,
        "status": "pass",
        "simulation_seed": int(protocol["seeds"]["statistics"]),
        "simulation_repetitions": simulations,
        "variance_envelope": {
            row["endpoint"]: row["conservative_paired_sd"] for row in rows
        },
        "variance_envelope_hash": canonical_json_sha256(
            {row["endpoint"]: row["conservative_paired_sd"] for row in rows}
        ),
        "endpoint_results": rows,
        "independent_group_floor": max(
            hard_min,
            max(row["simulated_independent_group_requirement"] for row in rows),
        ),
        "positive_group_floor_per_claimed_class": int(
            protocol["data"]["positive_group_floor_per_claimed_class_hard_min"]
        ),
        "created_at": utc_now(),
    }


def _load_or_create_precision_feasibility(
    root: Path, protocol: Mapping[str, Any]
) -> Dict[str, Any]:
    output = root / "evidence/g1/precision_feasibility.json"
    expected = _precision_feasibility(root, protocol)
    if output.is_file():
        existing = load_json(output)
        expected["created_at"] = existing.get("created_at")
        if existing != expected:
            raise RuntimeError(
                "Existing precision feasibility evidence differs from the frozen simulation"
            )
        return existing
    write_once_json(output, expected)
    return expected


def _artifact_value(root: Path, relative: str) -> str:
    path = root / relative
    if not path.exists():
        raise RuntimeError(f"PRETEST producer artifact is missing: {relative}")
    return relative


def _source_roles(root: Path) -> Dict[str, Any]:
    payload = load_json(root / "evidence/data/dataset_source_decision.json")
    roles = payload.get("selected_roles", payload.get("roles", {}))
    if not isinstance(roles, dict):
        raise RuntimeError("Dataset source decision lacks selected roles")
    return roles


def _role_id(roles: Mapping[str, Any], name: str) -> Any:
    value = roles.get(name)
    if isinstance(value, dict):
        return value.get("dataset_id", value.get("dataset_ids"))
    return value


def _verification_payload(receipt: Mapping[str, Any], scope: str) -> Dict[str, Any]:
    excerpts = receipt.get("verification_output_excerpt")
    if not isinstance(excerpts, list) or not excerpts:
        raise RuntimeError(f"{scope} environment receipt lacks verification output")
    payload: Optional[Dict[str, Any]] = None
    for candidate in reversed(excerpts):
        if isinstance(candidate, Mapping):
            parsed = dict(candidate)
        elif isinstance(candidate, str):
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
        else:
            continue
        if isinstance(parsed, dict) and parsed.get("scope") == scope:
            payload = parsed
            break
    if payload is None:
        raise RuntimeError(f"{scope} environment receipt lacks a structured verification payload")
    required = {
        "packages",
        "python_version",
        "scope",
        "selected_python",
        "status",
        "torch_cuda_runtime",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise RuntimeError(
            f"{scope} environment verification lacks fields: {', '.join(missing)}"
        )
    if payload["status"] != "pass" or not isinstance(payload["packages"], dict):
        raise RuntimeError(f"{scope} environment verification is not a passing package record")
    return payload


def _remote_driver_version(
    root: Path,
    decision: Mapping[str, Any],
    remote_ready: Mapping[str, Any],
    remote_verification: Mapping[str, Any],
) -> str:
    direct = str(remote_ready.get("cuda_driver", "")).strip()
    if direct:
        return direct
    relative = "evidence/preimplementation/E067_remote_driver_repair.json"
    path = root / relative
    if not path.is_file():
        raise RuntimeError(
            "Remote environment receipt lacks cuda_driver and the reviewed E067 "
            "driver repair receipt is missing"
        )
    receipt = load_json(path)
    expected = {
        "status": "pass",
        "step_id": "E067",
        "probe_kind": "nvidia_driver_metadata",
        "job_state": "COMPLETED",
        "exit_code": "0:0",
        "remote_host": "xinxi-zhyh@211.87.115.228",
        "selected_remote_python": decision["selected_remote_python"],
        "python_version": remote_verification["python_version"],
        "torch_cuda_runtime": remote_verification["torch_cuda_runtime"],
        "decision_sha256": sha256_file(
            root / "configs/environment_decision.lock.json"
        ),
        "remote_ready_receipt_sha256": sha256_file(
            root / "evidence/preimplementation/remote_environment_ready.json"
        ),
        "cuda_available": True,
        "training_performed": False,
        "dataset_accessed": False,
        "stderr_bytes": 0,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise RuntimeError(
                f"E067 driver repair receipt {key} does not match its producer"
            )
    version = str(receipt.get("driver_version", "")).strip()
    if not version or not all(part.isdigit() for part in version.split(".")):
        raise RuntimeError(
            "E067 driver repair receipt lacks a numeric NVIDIA driver version"
        )
    return version


def _environment_values(root: Path) -> Dict[str, Any]:
    decision = load_json(root / "configs/environment_decision.lock.json")
    local = load_json(root / "evidence/preimplementation/local_environment_ready.json")
    remote = load_json(root / "evidence/preimplementation/remote_environment_ready.json")
    local_verification = _verification_payload(local, "local")
    remote_verification = _verification_payload(remote, "remote")
    for scope, receipt, verification, selected_key in (
        ("local", local, local_verification, "selected_local_python"),
        ("remote", remote, remote_verification, "selected_remote_python"),
    ):
        selected = decision[selected_key]
        if receipt.get("selected_python") != selected:
            raise RuntimeError(
                f"{scope} ready receipt Python differs from the environment decision"
            )
        if verification["selected_python"] != selected:
            raise RuntimeError(
                f"{scope} verification Python differs from the environment decision"
            )
    local_packages = local_verification["packages"]
    remote_packages = remote_verification["packages"]
    required_local_packages = {"torch"}
    required_remote_packages = {"torch", "ultralytics", "numpy", "scipy", "pandas", "pyarrow"}
    if not required_local_packages.issubset(local_packages):
        raise RuntimeError("Local verification package inventory is incomplete")
    if not required_remote_packages.issubset(remote_packages):
        raise RuntimeError("Remote verification package inventory is incomplete")
    compiler = str(decision.get("selected_remote_compiler", "")).strip()
    if not compiler:
        raise RuntimeError("Environment decision lacks the selected remote compiler")
    remote_driver = _remote_driver_version(
        root, decision, remote, remote_verification
    )
    return {
        "preimplementation_decision_sha256": sha256_file(
            root / "configs/environment_decision.lock.json"
        ),
        "local_ready_receipt_sha256": sha256_file(
            root / "evidence/preimplementation/local_environment_ready.json"
        ),
        "remote_ready_receipt_sha256": sha256_file(
            root / "evidence/preimplementation/remote_environment_ready.json"
        ),
        "local_strategy": decision["local_strategy"],
        "remote_strategy": decision["remote_strategy"],
        "selected_local_python": decision["selected_local_python"],
        "selected_remote_python": decision["selected_remote_python"],
        "local_python_version": local_verification["python_version"],
        "remote_python_version": remote_verification["python_version"],
        "local_pytorch_version": local_packages["torch"],
        "remote_pytorch_version": remote_packages["torch"],
        "remote_cuda_runtime": remote_verification["torch_cuda_runtime"],
        "remote_cuda_driver": remote_driver,
        "remote_compiler": compiler,
        "ultralytics_version": remote_packages["ultralytics"],
        "transformers_or_detector_version": remote_packages["ultralytics"],
        "numpy_version": remote_packages["numpy"],
        "scipy_version": remote_packages["scipy"],
        "pandas_version": remote_packages["pandas"],
        "pyarrow_version": remote_packages["pyarrow"],
        "local_environment_lock_file": decision["local_environment_specification"],
        "remote_environment_lock_file": decision["remote_environment_specification"],
    }


def _pilot_values(root: Path) -> Dict[str, Any]:
    pilot = load_json(root / "evidence/pilot/resource_pilot.json")
    decisions = pilot.get("decisions")
    if not isinstance(decisions, dict):
        raise RuntimeError("Pilot receipt lacks frozen technical decisions")
    required = {
        "precision",
        "workers",
        "checkpoint_interval_updates",
        "fixed_cpu_affinity",
        "max_array_concurrency",
        "gpu_hour_caps",
    }
    if not required.issubset(decisions):
        raise RuntimeError("Pilot technical decisions are incomplete")
    return decisions


_RESOURCE_PACKAGE = {
    "V2": "V2",
    "T1": "T1",
    "S1": "S1",
    "R1": "R1",
    "E1": "E2_E3",
    "E2": "E2_E3",
    "E3": "E2_E3",
    "C1": "C1",
}
_RESOURCE_PRIORITY = {
    "V2": "V2_primary",
    "S1": "S1_primary",
    "E2_E3": "E2_E3_primary",
    "T1": "T1_input_validity",
    "R1": "R1_secondary",
    "C1": "efficiency",
}
_PILOT_FAMILY = {
    "V2": "visible",
    "T1": "rgbt",
    "S1": "methane",
    "R1": "visible",
    "E2_E3": "episode",
}
_GPU_TRAIN_FAMILIES = frozenset(
    {
        "V2-PRE-GEN",
        "V2-PRE-SINGLE",
        "V2-PRE-MULTI",
        "V2-A-10-SCR",
        "V2-A-10-GEN",
        "V2-A-10-SINGLE",
        "V2-A-10-MULTI",
        "T1-VIS",
        "T1-THERM",
        "S1-GRU",
        "E3-NOREL",
        "E3-NOGRAPH",
        "E3-FULL",
    }
)


def _matrix_run_count(value: Any, family_id: str) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid planned run count for {family_id}") from exc
    if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
        raise RuntimeError(f"Invalid planned run count for {family_id}")
    return int(numeric)


def _gpu_max_updates(protocol: Mapping[str, Any], family_id: str) -> int:
    training = protocol["training"]
    if family_id.startswith("V2-PRE-"):
        value = training["matched_pretraining"]["optimizer_updates"]
    elif family_id.startswith("V2-A-"):
        value = training["visible"]["max_updates"]
    elif family_id.startswith("T1-"):
        value = training["rgbt"]["max_updates"]
    elif family_id == "S1-GRU":
        value = training["methane"]["max_updates"]
    elif family_id in {"E3-NOREL", "E3-NOGRAPH", "E3-FULL"}:
        value = training["episode"]["max_updates"]
    else:
        raise RuntimeError(f"GPU resource mapping is missing for {family_id}")
    updates = _matrix_run_count(value, family_id)
    if updates <= 0:
        raise RuntimeError(f"GPU max updates must be positive for {family_id}")
    return updates


def _build_family_resource_budget(
    root: Path,
    protocol: Mapping[str, Any],
    pilot_receipt: Mapping[str, Any],
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    matrix_path = root / "configs/experiment_matrix.template.csv"
    matrix = pd.read_csv(matrix_path)
    required_columns = {"family_id", "package_id", "planned_trained_runs"}
    missing_columns = sorted(required_columns.difference(matrix.columns))
    if missing_columns:
        raise RuntimeError(f"Resource budget matrix lacks columns: {missing_columns}")
    if matrix["family_id"].astype(str).duplicated().any():
        raise RuntimeError("Resource budget matrix has duplicate family IDs")

    decisions = pilot_receipt.get("decisions")
    if pilot_receipt.get("status") != "pass" or not isinstance(decisions, dict):
        raise RuntimeError("Passing resource-pilot decisions are required")
    selected_precision = str(decisions.get("precision", ""))
    benchmarks = pilot_receipt.get("benchmarks")
    if not isinstance(benchmarks, list):
        raise RuntimeError("Resource-pilot benchmarks are missing")
    selected_benchmarks: Dict[str, Mapping[str, Any]] = {}
    for record in benchmarks:
        if str(record.get("precision")) != selected_precision:
            continue
        family = str(record.get("family", ""))
        if family in selected_benchmarks:
            raise RuntimeError(f"Duplicate selected-precision pilot benchmark: {family}")
        updates_per_second = float(record.get("updates_per_second", 0.0))
        peak_memory_bytes = int(record.get("peak_memory_bytes", 0))
        if (
            record.get("status") != "pass"
            or record.get("numerically_finite") is not True
            or not math.isfinite(updates_per_second)
            or updates_per_second <= 0.0
            or peak_memory_bytes <= 0
        ):
            raise RuntimeError(f"Invalid selected-precision pilot benchmark: {family}")
        selected_benchmarks[family] = record

    cap_basis = pilot_receipt.get("resource_cap_basis")
    package_caps = cap_basis.get("package_caps_gpu_hours") if isinstance(cap_basis, dict) else None
    if not isinstance(package_caps, dict):
        raise RuntimeError("Resource-pilot package caps are missing")
    for package in _RESOURCE_PRIORITY:
        cap = float(package_caps.get(package, 0.0))
        if not math.isfinite(cap) or cap <= 0.0:
            raise RuntimeError(f"Invalid GPU-hour cap for {package}")

    priority_order = list(protocol["resources"]["priority_order"])
    if priority_order != list(_RESOURCE_PRIORITY.values()):
        raise RuntimeError("Resource priority order differs from the frozen protocol")

    rows: list[Dict[str, Any]] = []
    for record in matrix.to_dict(orient="records"):
        matrix_package = str(record["package_id"])
        package = _RESOURCE_PACKAGE.get(matrix_package)
        if package is None:
            continue
        family_id = str(record["family_id"])
        run_count = _matrix_run_count(record["planned_trained_runs"], family_id)
        pilot_family = _PILOT_FAMILY.get(package)
        benchmark = selected_benchmarks.get(pilot_family) if pilot_family else None
        if pilot_family is not None and benchmark is None:
            raise RuntimeError(
                f"Selected-precision pilot benchmark is missing for {pilot_family}"
            )
        gpu_training = family_id in _GPU_TRAIN_FAMILIES
        max_updates = _gpu_max_updates(protocol, family_id) if gpu_training else 0
        seconds_per_update = (
            1.0 / float(benchmark["updates_per_second"])
            if gpu_training and benchmark is not None
            else 0.0
        )
        projected_gpu_hours = (
            run_count * max_updates * seconds_per_update / 3600.0
        )
        storage_envelope = int(benchmark["peak_memory_bytes"]) if benchmark else 0
        projected_storage_gib = run_count * storage_envelope / float(1024**3)
        execution_resource = (
            "gpu_training"
            if gpu_training
            else "cpu_fit"
            if run_count > 0
            else "reuse_or_nontraining"
        )
        rows.append(
            {
                "family_id": family_id,
                "package_id": package,
                "run_count": run_count,
                "pilot_family": pilot_family,
                "selected_precision": selected_precision if pilot_family else None,
                "execution_resource": execution_resource,
                "max_updates_per_run": max_updates,
                "pilot_seconds_per_update": round(seconds_per_update, 12),
                "projected_gpu_hours": round(projected_gpu_hours, 12),
                "projected_storage_gib": round(projected_storage_gib, 12),
                "storage_envelope_bytes_per_run": storage_envelope if run_count else 0,
                "priority": _RESOURCE_PRIORITY[package],
                "cap_decision": "pending_package_reconciliation",
            }
        )
    if not rows:
        raise RuntimeError("Resource budget cannot be empty")

    expected_families = {
        str(record["family_id"])
        for record in matrix.to_dict(orient="records")
        if str(record["package_id"]) in _RESOURCE_PACKAGE
    }
    if {row["family_id"] for row in rows} != expected_families:
        raise RuntimeError("Resource budget does not cover every governed matrix family")

    summaries = []
    for package, priority in _RESOURCE_PRIORITY.items():
        package_rows = [row for row in rows if row["package_id"] == package]
        projected_gpu_hours = sum(
            float(row["projected_gpu_hours"]) for row in package_rows
        )
        projected_storage_gib = sum(
            float(row["projected_storage_gib"]) for row in package_rows
        )
        cap = float(package_caps[package])
        if projected_gpu_hours > cap + 1.0e-12:
            raise RuntimeError(
                f"Pilot projection exceeds the frozen GPU-hour cap for {package}"
            )
        cap_decision = "within_frozen_package_cap"
        for row in package_rows:
            row["cap_decision"] = cap_decision
        summaries.append(
            {
                "package_id": package,
                "priority": priority,
                "family_count": len(package_rows),
                "run_count": sum(int(row["run_count"]) for row in package_rows),
                "projected_gpu_hours": round(projected_gpu_hours, 12),
                "gpu_hour_cap": cap,
                "remaining_gpu_hour_margin": round(cap - projected_gpu_hours, 12),
                "projected_storage_gib": round(projected_storage_gib, 12),
                "cap_decision": cap_decision,
            }
        )

    required_fields = set(protocol["resources"]["resource_budget_required_fields"])
    for row in rows:
        if not required_fields.issubset(row):
            raise RuntimeError(f"Resource budget row is incomplete: {row['family_id']}")
        for field in (
            "pilot_seconds_per_update",
            "projected_gpu_hours",
            "projected_storage_gib",
        ):
            value = float(row[field])
            if not math.isfinite(value) or value < 0.0:
                raise RuntimeError(
                    f"Resource budget {field} is invalid for {row['family_id']}"
                )
    return rows, summaries


def _replace_placeholders(
    node: Any,
    *,
    path: tuple[str, ...],
    exact: Mapping[tuple[str, ...], Any],
    provenance: Dict[str, str],
) -> Any:
    if isinstance(node, dict):
        return {
            key: _replace_placeholders(
                value,
                path=(*path, str(key)),
                exact=exact,
                provenance=provenance,
            )
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [
            _replace_placeholders(
                value,
                path=(*path, str(index)),
                exact=exact,
                provenance=provenance,
            )
            for index, value in enumerate(node)
        ]
    if isinstance(node, str) and node.startswith("TBD_"):
        if path not in exact:
            raise RuntimeError(f"No frozen producer is defined for placeholder: {'.'.join(path)}")
        provenance[".".join(path)] = str(exact[(path)])[:0] or "frozen_producer"
        return deepcopy(exact[path])
    return node


def _source_code_manifest(root: Path, scopes: Iterable[str]) -> Dict[str, Any]:
    files = []
    for scope in scopes:
        path = root / scope
        candidates = path.rglob("*") if path.is_dir() else [path]
        for candidate in candidates:
            if candidate.is_file() and "__pycache__" not in candidate.parts:
                files.append(
                    {
                        "path": candidate.relative_to(root).as_posix(),
                        "sha256": sha256_file(candidate),
                    }
                )
    files = sorted(files, key=lambda value: value["path"])
    if not files:
        raise RuntimeError("Source-code hash scope is empty")
    return {
        "schema_version": 1,
        "files": files,
        "source_code_sha256": canonical_json_sha256(files),
    }


def fill_pretest_lock(
    root: Path, row: Mapping[str, str], arguments: Mapping[str, Any]
) -> Dict[str, Any]:
    template_path = root / "configs/protocol_lock.template.yaml"
    protocol = _load_yaml(template_path)
    roles = _source_roles(root)
    precision = _load_or_create_precision_feasibility(root, protocol)
    precision_output = "evidence/g1/precision_feasibility.json"
    environment = _environment_values(root)
    pilot = _pilot_values(root)
    scopes = protocol["locks"]["source_code_hash_scope"]
    source_manifest = _source_code_manifest(root, scopes)
    source_manifest_output = "evidence/pretest/source_code_hashes.json"
    exact: Dict[tuple[str, ...], Any] = {
        ("data", "archive_manifest"): _artifact_value(
            root, "evidence/data/remote_archive_verification.json"
        ),
        ("data", "file_manifest"): _artifact_value(
            root, "data/locked/file_manifest.parquet"
        ),
        ("data", "ontology_lock"): _artifact_value(
            root, "data/locked/ontology_lock.yaml"
        ),
        ("data", "event_semantics_lock"): _artifact_value(
            root, "data/locked/methane_role_lock.json"
        ),
        ("data", "split_manifest"): _artifact_value(
            root, "data/locked/split_manifest.parquet"
        ),
        ("data", "fewshot_manifest"): _artifact_value(
            root, "data/locked/fewshot_manifest.parquet"
        ),
        ("data", "graph_eligibility_lock"): _artifact_value(
            root, "data/locked/graph_eligibility_lock.json"
        ),
        ("data", "fusion_eligibility_lock"): _artifact_value(
            root, "data/locked/fusion_eligibility_lock.parquet"
        ),
        ("data", "episode_template_lock"): _artifact_value(
            root, "data/locked/episode_template_lock.json"
        ),
        ("data", "episode_skeleton_manifest"): _artifact_value(
            root, "data/locked/episode_skeleton_manifest.parquet"
        ),
        ("data", "episode_test_skeleton_candidate"): _artifact_value(
            root, "data/seals/episode_test_skeleton_candidate.json"
        ),
        ("data", "branch_test_seal_candidate"): _artifact_value(
            root, "data/seals/branch_test_seal_candidate.json"
        ),
        ("data", "primary_visual_target"): _role_id(roles, "primary_visual_target"),
        ("data", "primary_visual_sources"): _role_id(roles, "primary_visual_sources"),
        ("data", "primary_rgbt_dataset"): _role_id(roles, "primary_rgbt_dataset"),
        ("data", "methane_dataset"): _role_id(roles, "methane_dataset"),
        ("data", "independent_group_floor"): precision["independent_group_floor"],
        ("data", "positive_group_floor_per_claimed_class"): precision[
            "positive_group_floor_per_claimed_class"
        ],
        ("data", "precision_feasibility", "variance_envelope_hash"): precision[
            "variance_envelope_hash"
        ],
        ("data", "near_duplicate_rule", "hamming_threshold"): 8,
        ("data", "near_duplicate_rule", "audit_sample_size"): 200,
        ("data", "methane", "risk_unit"): "percent_CH4_volume",
        ("data", "methane", "event_merge_gap_seconds"): 60,
        ("training", "visible", "max_updates"): 10000,
        ("training", "rgbt", "max_updates"): 5000,
        ("training", "methane", "max_updates"): 5000,
        ("training", "methane", "max_false_alarms_per_hour"): 1.0,
        ("training", "episode", "max_updates"): 5000,
        ("training", "matched_pretraining", "source_validation_fraction"): 0.10,
        ("training", "matched_pretraining", "unique_source_images"): 20000,
        ("training", "matched_pretraining", "optimizer_updates"): 10000,
        ("training", "matched_pretraining", "augmentation_family_hash"): (
            canonical_json_sha256({"family": "ultralytics_default_locked", "version": 1})
        ),
        ("evaluation", "calibration_group_cv_folds"): 3,
        ("episodes", "shortcut_max_metric"): 0.60,
        ("episodes", "graph_primary", "eligibility_lock"): (
            "data/locked/graph_eligibility_lock.json"
        ),
        (
            "episodes",
            "graph_primary",
            "edge_trace_minimum_nonzero_scored_step_fraction",
        ): 0.05,
        ("episodes", "policy_constraints", "max_false_alarms_per_100_episodes"): 10.0,
        ("episodes", "policy_constraints", "max_event_miss_rate"): 0.20,
        ("confirmatory_endpoints", "V2", "minimum_meaningful_effect_pp"): 5.0,
        ("confirmatory_endpoints", "S1", "minimum_meaningful_effect_pp"): 5.0,
        ("confirmatory_endpoints", "E2_E3", "minimum_meaningful_effect_pp"): 5.0,
        (
            "confirmatory_endpoints",
            "E3_RELIABILITY",
            "minimum_meaningful_effect_pp",
        ): 3.0,
        ("confirmatory_endpoints", "E3_GRAPH", "minimum_meaningful_effect_pp"): 3.0,
        ("confirmatory_endpoints", "E3_MEMORY", "minimum_meaningful_effect_pp"): 3.0,
        ("confirmatory_endpoints", "R1", "clean_map50_95_validity_floor"): 0.05,
        ("confirmatory_endpoints", "R1", "minimum_meaningful_effect_pp"): 5.0,
        ("efficiency", "module_input_contracts", "sensor_inference"): [1, 32, 30],
        ("efficiency", "runtime_backend"): "pytorch_eager",
        ("efficiency", "model_format"): "state_dict",
        ("efficiency", "precision"): pilot["precision"],
        ("locks", "source_code_sha256"): source_manifest["source_code_sha256"],
        ("locks", "artifact_contract_sha256"): sha256_file(
            root / "configs/artifact_contract.template.yaml"
        ),
        ("locks", "experiment_matrix_sha256"): sha256_file(
            root / "configs/experiment_matrix.template.csv"
        ),
        ("locks", "experiment_steps_sha256"): _experiment_steps_contract_sha256(
            root / "plans/experiment_steps.csv"
        ),
        ("locks", "implementation_pass_sha256"): sha256_file(
            root / "evidence/reviews/I090.json"
        ),
    }
    for key, value in environment.items():
        exact[("environment", key)] = value
    for key in ("precision", "workers", "checkpoint_interval_updates"):
        exact[("training", "common", key)] = pilot[key]
    exact[("efficiency", "fixed_cpu_affinity")] = pilot["fixed_cpu_affinity"]
    exact[("resources", "max_array_concurrency")] = pilot["max_array_concurrency"]
    for key, value in pilot["gpu_hour_caps"].items():
        exact[("resources", "gpu_hour_caps", key)] = value
    provenance: Dict[str, str] = {}
    frozen = _replace_placeholders(
        protocol,
        path=(),
        exact=exact,
        provenance=provenance,
    )
    frozen["status"] = "pretest_locked"
    frozen["created_at"] = frozen.get("created_at") or utc_now()
    frozen["locked_at"] = utc_now()
    frozen["pretest_lock_provenance"] = {
        key: {"producer": "frozen_plan_artifact", "status": "bound"}
        for key in sorted(provenance)
    }
    encoded = yaml.safe_dump(frozen, sort_keys=False, allow_unicode=False).encode("utf-8")
    if b"TBD_" in encoded:
        raise RuntimeError("PRETEST protocol still contains unresolved placeholders")
    pilot_receipt = load_json(root / "evidence/pilot/resource_pilot.json")
    budget_rows, package_summaries = _build_family_resource_budget(
        root, frozen, pilot_receipt
    )
    protocol_sha256 = hashlib.sha256(encoded).hexdigest()
    output = "configs/protocol_lock.pretest.yaml"
    write_once_json(root / source_manifest_output, source_manifest)
    write_once_bytes(root / output, encoded)
    sidecar = "evidence/pretest/protocol_lock.sha256"
    write_once_bytes(root / sidecar, (protocol_sha256 + "\n").encode("ascii"))
    resource_budget = {
        "schema_version": 1,
        "status": "locked",
        "result_role": "resource_governance_only",
        "pilot_sha256": sha256_file(root / "evidence/pilot/resource_pilot.json"),
        "experiment_matrix_sha256": sha256_file(
            root / "configs/experiment_matrix.template.csv"
        ),
        "protocol_sha256": protocol_sha256,
        "max_array_concurrency": pilot["max_array_concurrency"],
        "gpu_hour_caps": pilot["gpu_hour_caps"],
        "projection_methods": {
            "gpu_hours": (
                "selected_precision_pilot_seconds_per_update_x_frozen_max_updates_"
                "x_planned_gpu_run_count"
            ),
            "storage": (
                "selected_precision_pilot_peak_memory_working_set_envelope_"
                "x_planned_fitted_run_count"
            ),
            "scope": (
                "diagnostic_projection_only; hard_package_caps_govern_execution; "
                "test_metrics_and_model_rankings_not_used"
            ),
        },
        "families": budget_rows,
        "package_summaries": package_summaries,
        "total_planned_fitted_runs": sum(row["run_count"] for row in budget_rows),
        "created_at": utc_now(),
    }
    resource_output = "evidence/pretest/resource_budget.lock.json"
    write_once_json(root / resource_output, resource_budget)
    return {
        "status": "pass",
        "output_paths": [
            output,
            precision_output,
            source_manifest_output,
            sidecar,
            resource_output,
        ],
        "inputs": hash_existing_inputs(
            root,
            [
                "configs/protocol_lock.template.yaml",
                "configs/environment_decision.lock.json",
                "evidence/preimplementation/local_environment_ready.json",
                "evidence/preimplementation/remote_environment_ready.json",
                "evidence/preimplementation/E067_remote_driver_repair.json",
                "evidence/pilot/resource_pilot.json",
                "evidence/data/dataset_source_decision.json",
            ],
        ),
        "details": {
            "placeholder_count": len(provenance),
            "protocol_sha256": sha256_file(root / output),
        },
    }
