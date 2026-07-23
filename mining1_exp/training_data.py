from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import pandas as pd
import yaml

from .data.manifests import read_file_manifest, read_split_manifest
from .data.ontology import validate_ontology_lock
from .governance.immutable import write_once_bytes, write_once_json
from .provenance import canonical_json_sha256, sha256_file
from .workflow_common import WorkflowExecutionError, load_json


DETECTION_MODALITIES = {"visible", "thermal"}
SOURCE_CONDITIONS = {"generic_matched", "single_coal", "multi_coal"}
TRANSFER_FALLBACK_REASON = "fewer_than_two_eligible_non_target_coal_sources"


def load_frozen_protocol(project_root: Path) -> Dict[str, Any]:
    path = project_root / "configs/protocol_lock.pretest.yaml"
    if not path.is_file():
        raise WorkflowExecutionError("Frozen PRETEST protocol is missing")
    payload = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or "TBD" in json.dumps(payload, sort_keys=True):
        raise WorkflowExecutionError("Frozen PRETEST protocol is invalid or unresolved")
    return payload


def _decision_entries(project_root: Path) -> list[Dict[str, Any]]:
    decision = load_json(project_root / "evidence/data/dataset_source_decision.json")
    roles = decision.get("roles")
    if decision.get("status") != "pass" or not isinstance(roles, Mapping):
        raise WorkflowExecutionError("Dataset source decision is not available")
    entries = []
    for role, value in roles.items():
        values = value if isinstance(value, list) else [value]
        for item in values:
            entry = dict(item)
            entry["planned_role"] = str(role)
            entries.append(entry)
    return entries


def dataset_ids_for_role(project_root: Path, role: str) -> tuple[str, ...]:
    values = [
        str(item["dataset_id"])
        for item in _decision_entries(project_root)
        if item["planned_role"] == role
    ]
    if not values:
        raise WorkflowExecutionError(f"Dataset role is empty: {role}")
    return tuple(values)


def source_dataset_ids(project_root: Path, condition: str) -> tuple[str, ...]:
    if condition not in SOURCE_CONDITIONS:
        raise WorkflowExecutionError(f"Unknown source condition: {condition}")
    values = []
    for item in _decision_entries(project_root):
        if item["planned_role"] != "primary_visual_sources":
            continue
        memberships = item.get("pretraining_conditions", [])
        if condition in memberships:
            values.append(str(item["dataset_id"]))
    if condition in {"generic_matched", "single_coal"} and len(values) != 1:
        raise WorkflowExecutionError(f"Condition {condition} must resolve to one dataset")
    if condition == "multi_coal" and len(values) < 2:
        raise WorkflowExecutionError("Multi-coal condition requires at least two datasets")
    return tuple(sorted(values))


def transfer_claim_not_applicable(project_root: Path) -> Optional[Dict[str, Any]]:
    decision = load_json(project_root / "evidence/data/dataset_source_decision.json")
    eligibility = decision.get("claim_eligibility")
    if not isinstance(eligibility, Mapping):
        return None
    fallback = eligibility.get("C-TRANSFER")
    if not isinstance(fallback, Mapping):
        return None
    if (
        fallback.get("status") != "not_applicable"
        or fallback.get("reason_code") != TRANSFER_FALLBACK_REASON
    ):
        return None
    eligible_ids = fallback.get("eligible_non_target_coal_dataset_ids")
    review_hashes = fallback.get("candidate_review_hashes")
    if not isinstance(eligible_ids, list) or len(eligible_ids) >= 2:
        raise WorkflowExecutionError("C-TRANSFER fallback eligible-source evidence drifted")
    if not isinstance(review_hashes, list) or not review_hashes:
        raise WorkflowExecutionError("C-TRANSFER fallback candidate review hashes are missing")
    return {
        "claim_id": "C-TRANSFER",
        "status": "not_applicable",
        "reason_code": TRANSFER_FALLBACK_REASON,
        "eligible_non_target_coal_dataset_ids": [str(value) for value in eligible_ids],
        "authority_path": str(fallback.get("authority_path", "")),
        "authority_sha256": str(fallback.get("authority_sha256", "")),
        "candidate_review_hashes": [dict(item) for item in review_hashes],
    }


def _ontology_mapping(project_root: Path) -> tuple[Dict[tuple[str, str], str], tuple[str, ...]]:
    path = project_root / "data/locked/ontology_lock.yaml"
    payload = validate_ontology_lock(yaml.safe_load(path.read_text(encoding="utf-8-sig")))
    mapping = {}
    for item in payload["entries"]:
        if item["mapping_status"] != "compatible":
            continue
        key = (str(item["dataset_id"]), str(item["source_label"]))
        if key in mapping and mapping[key] != str(item["canonical_concept_id"]):
            raise WorkflowExecutionError(f"Ontology maps one source label twice: {key}")
        mapping[key] = str(item["canonical_concept_id"])
    concepts = tuple(sorted(set(mapping.values())))
    if not concepts:
        raise WorkflowExecutionError("Ontology has no compatible detection concept")
    return mapping, concepts


def target_branch_modalities(
    project_root: Path,
    *,
    dataset_id: str,
    source_modalities: pd.Series,
) -> pd.Series:
    mapping = {modality: modality for modality in DETECTION_MODALITIES}
    lock_path = project_root / "data/locked/rgbt_input_lock.json"
    if lock_path.is_file():
        lock = load_json(lock_path)
        if str(lock.get("dataset_id")) == str(dataset_id):
            configured = lock.get("source_modality_to_branch_role")
            if lock.get("status") != "pass" or not isinstance(configured, Mapping):
                raise WorkflowExecutionError("RGBT source-to-branch modality lock is invalid")
            normalized = {
                str(source).lower(): str(branch).lower()
                for source, branch in configured.items()
            }
            if not normalized or not set(normalized.values()).issubset(DETECTION_MODALITIES):
                raise WorkflowExecutionError("RGBT modality lock contains an invalid branch role")
            mapping.update(normalized)
    return source_modalities.astype(str).str.lower().map(mapping)


def target_detection_records(
    project_root: Path,
    *,
    dataset_id: str,
    train_pool: str = "D_b_tr",
    validation_pool: str = "D_b_sel",
    subset_seed: Optional[int] = None,
    modality: Optional[str] = None,
) -> pd.DataFrame:
    files = read_file_manifest(project_root / "data/locked/file_manifest.parquet")
    split = read_split_manifest(project_root / "data/locked/split_manifest.parquet")
    joined = files.merge(
        split[["dataset_id", "record_id", "raw_group_id", "pool"]],
        on=["dataset_id", "record_id", "raw_group_id"],
        validate="one_to_one",
    )
    rows = joined.loc[
        (joined["dataset_id"].astype(str) == str(dataset_id))
        & joined["pool"].isin({train_pool, validation_pool})
    ].copy()
    branch_modalities = target_branch_modalities(
        project_root,
        dataset_id=dataset_id,
        source_modalities=rows["modality"],
    )
    rows = rows.loc[branch_modalities.isin(DETECTION_MODALITIES)].copy()
    rows["branch_modality"] = branch_modalities.loc[rows.index]
    if modality is not None:
        requested = modality.lower()
        if requested not in DETECTION_MODALITIES:
            raise WorkflowExecutionError(f"Unknown target branch modality: {modality}")
        rows = rows.loc[rows["branch_modality"] == requested]
    if subset_seed is not None:
        fewshot = pd.read_parquet(project_root / "data/locked/fewshot_manifest.parquet")
        selected_groups = set(
            fewshot.loc[
                (fewshot["subset_seed"].astype(int) == int(subset_seed))
                & fewshot["included"].astype(bool),
                "raw_group_id",
            ].astype(str)
        )
        train_mask = rows["pool"] == train_pool
        rows = rows.loc[~train_mask | rows["raw_group_id"].astype(str).isin(selected_groups)]
    if rows.empty or not {train_pool, validation_pool}.issubset(set(rows["pool"])):
        raise WorkflowExecutionError("Target detection view lacks train or validation records")
    rows["view_split"] = rows["pool"].map(
        {train_pool: "train", validation_pool: "val"}
    )
    return rows.reset_index(drop=True)


def _hash_score(*parts: Any) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def source_detection_records(
    project_root: Path,
    *,
    condition: str,
    train_seed: int,
) -> pd.DataFrame:
    protocol = load_frozen_protocol(project_root)
    contract = protocol["training"]["matched_pretraining"]
    validation_fraction = float(contract["source_validation_fraction"])
    budget = int(contract["unique_source_images"])
    if not 0.0 < validation_fraction < 0.5 or budget <= 0:
        raise WorkflowExecutionError("Matched source budget or validation fraction is invalid")
    dataset_ids = source_dataset_ids(project_root, condition)
    files = read_file_manifest(project_root / "data/locked/file_manifest.parquet")
    rows = files.loc[
        files["dataset_id"].astype(str).isin(dataset_ids)
        & files["modality"].astype(str).str.lower().isin(DETECTION_MODALITIES)
    ].copy()
    if len(rows) < budget:
        raise WorkflowExecutionError(f"Source condition {condition} lacks unique image capacity")
    if condition == "multi_coal":
        per_source = max(1, int(math.floor(budget / len(dataset_ids))))
        selected_parts = []
        remaining = budget
        for position, dataset_id in enumerate(dataset_ids):
            candidates = rows.loc[rows["dataset_id"].astype(str) == dataset_id].copy()
            count = remaining if position == len(dataset_ids) - 1 else per_source
            if len(candidates) < count:
                raise WorkflowExecutionError(
                    f"Multi-coal source {dataset_id} cannot satisfy stratified budget"
                )
            candidates["__rank"] = [
                _hash_score(train_seed, condition, dataset_id, record_id)
                for record_id in candidates["record_id"]
            ]
            selected_parts.append(candidates.sort_values("__rank").head(count))
            remaining -= count
        selected = pd.concat(selected_parts, ignore_index=True)
    else:
        rows["__rank"] = [
            _hash_score(train_seed, condition, dataset_id, record_id)
            for dataset_id, record_id in rows[["dataset_id", "record_id"]].itertuples(
                index=False, name=None
            )
        ]
        selected = rows.sort_values("__rank").head(budget).copy()
    group_rows = selected[["dataset_id", "raw_group_id"]].drop_duplicates()
    validation_groups = set()
    for dataset_id, group_id in group_rows.itertuples(index=False, name=None):
        score = int(_hash_score(train_seed, "source_val", dataset_id, group_id), 16) / float(
            2**256
        )
        if score < validation_fraction:
            validation_groups.add((str(dataset_id), str(group_id)))
    selected["view_split"] = [
        "val" if (str(dataset_id), str(group_id)) in validation_groups else "train"
        for dataset_id, group_id in selected[["dataset_id", "raw_group_id"]].itertuples(
            index=False, name=None
        )
    ]
    if set(selected["view_split"]) != {"train", "val"}:
        raise WorkflowExecutionError("Source group split did not produce train and validation")
    return selected.drop(columns=["__rank"], errors="ignore").reset_index(drop=True)


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(str(source), str(target))
    except OSError:
        with source.open("rb") as input_handle, target.open("xb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)


def _remapped_label_lines(
    row: Mapping[str, Any],
    mapping: Mapping[tuple[str, str], str],
    concept_index: Mapping[str, int],
) -> list[str]:
    summary = json.loads(str(row["label_summary_json"]))
    boxes = summary.get("boxes", [])
    if not isinstance(boxes, list):
        raise WorkflowExecutionError("Detection label summary lacks a box list")
    width = float(summary.get("image_width", 0))
    height = float(summary.get("image_height", 0))
    lines = []
    for box in boxes:
        source_label = str(box["source_label"])
        concept = mapping.get((str(row["dataset_id"]), source_label))
        if concept is None:
            continue
        if "x_center_normalized" in box:
            x_center = float(box["x_center_normalized"])
            y_center = float(box["y_center_normalized"])
            box_width = float(box["width_normalized"])
            box_height = float(box["height_normalized"])
        else:
            if width <= 0 or height <= 0:
                raise WorkflowExecutionError("Absolute detection box lacks image dimensions")
            x1, y1, x2, y2 = (float(box[name]) for name in ("x1", "y1", "x2", "y2"))
            x_center = (x1 + x2) / (2.0 * width)
            y_center = (y1 + y2) / (2.0 * height)
            box_width = (x2 - x1) / width
            box_height = (y2 - y1) / height
        if not all(0.0 <= value <= 1.0 for value in (x_center, y_center, box_width, box_height)):
            raise WorkflowExecutionError("Remapped YOLO box is outside [0,1]")
        lines.append(
            f"{concept_index[concept]} {x_center:.10f} {y_center:.10f} "
            f"{box_width:.10f} {box_height:.10f}"
        )
    return lines


def build_yolo_view(
    project_root: Path,
    *,
    view_id: str,
    records: pd.DataFrame,
) -> Dict[str, Any]:
    if not view_id or any(value in view_id for value in ("/", "\\", "..")):
        raise WorkflowExecutionError("YOLO view ID is unsafe")
    required = {
        "dataset_id",
        "record_id",
        "raw_group_id",
        "relative_path",
        "label_summary_json",
        "view_split",
    }
    if not required.issubset(records.columns) or records.empty:
        raise WorkflowExecutionError("YOLO view records are empty or incomplete")
    if set(records["view_split"]) != {"train", "val"}:
        raise WorkflowExecutionError("YOLO view must contain train and validation rows")
    mapping, concepts = _ontology_mapping(project_root)
    concept_index = {concept: index for index, concept in enumerate(concepts)}
    view_root = project_root / "state/yolo_views" / view_id
    receipt_path = view_root / "view_receipt.json"
    source_signature = canonical_json_sha256(
        sorted(
            [
                str(item.dataset_id),
                str(item.record_id),
                str(item.raw_group_id),
                str(item.view_split),
                str(item.sha256),
                canonical_json_sha256(json.loads(str(item.label_summary_json))),
            ]
            for item in records.itertuples(index=False)
        )
    )
    if receipt_path.is_file():
        receipt = load_json(receipt_path)
        yaml_path = view_root / "dataset.yaml"
        if (
            receipt.get("status") != "pass"
            or receipt.get("source_signature_sha256") != source_signature
            or receipt.get("dataset_yaml_sha256") != sha256_file(yaml_path)
        ):
            raise WorkflowExecutionError(f"YOLO view receipt drift: {view_id}")
        return receipt
    if view_root.exists():
        raise WorkflowExecutionError(f"Incomplete YOLO view requires review: {view_id}")
    rows = []
    for item in records.sort_values(["view_split", "dataset_id", "record_id"]).to_dict(
        orient="records"
    ):
        source = project_root / str(item["relative_path"])
        if not source.is_file() or sha256_file(source) != str(item["sha256"]):
            raise WorkflowExecutionError(f"YOLO view source drift: {source}")
        stem = _hash_score(item["dataset_id"], item["record_id"])[:24]
        split = str(item["view_split"])
        image_target = view_root / "images" / split / f"{stem}{source.suffix.lower()}"
        label_target = view_root / "labels" / split / f"{stem}.txt"
        _link_or_copy(source, image_target)
        lines = _remapped_label_lines(item, mapping, concept_index)
        label_target.parent.mkdir(parents=True, exist_ok=True)
        write_once_bytes(
            label_target, ("\n".join(lines) + ("\n" if lines else "")).encode("ascii")
        )
        rows.append(
            {
                "dataset_id": str(item["dataset_id"]),
                "record_id": str(item["record_id"]),
                "raw_group_id": str(item["raw_group_id"]),
                "view_split": split,
                "image_path": image_target.relative_to(project_root).as_posix(),
                "label_path": label_target.relative_to(project_root).as_posix(),
                "source_sha256": str(item["sha256"]),
            }
        )
    dataset_yaml = {
        "path": view_root.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "names": {index: concept for index, concept in enumerate(concepts)},
        "nc": len(concepts),
    }
    yaml_path = view_root / "dataset.yaml"
    write_once_bytes(
        yaml_path,
        yaml.safe_dump(dataset_yaml, sort_keys=True, allow_unicode=False).encode("ascii"),
    )
    manifest_path = view_root / "view_manifest.parquet"
    pd.DataFrame.from_records(rows).to_parquet(manifest_path, index=False)
    receipt = {
        "schema_version": 1,
        "status": "pass",
        "view_id": view_id,
        "record_count": len(rows),
        "train_count": sum(item["view_split"] == "train" for item in rows),
        "validation_count": sum(item["view_split"] == "val" for item in rows),
        "concepts": list(concepts),
        "source_signature_sha256": source_signature,
        "dataset_yaml_path": yaml_path.relative_to(project_root).as_posix(),
        "dataset_yaml_sha256": sha256_file(yaml_path),
        "view_manifest_path": manifest_path.relative_to(project_root).as_posix(),
        "view_manifest_sha256": sha256_file(manifest_path),
    }
    write_once_json(receipt_path, receipt)
    return receipt
