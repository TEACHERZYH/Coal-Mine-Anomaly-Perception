from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import pandas as pd
import yaml


REPORT_COLUMNS = [
    "left_dataset_id",
    "left_record_id",
    "left_raw_group_id",
    "right_dataset_id",
    "right_record_id",
    "right_raw_group_id",
    "duplicate_kind",
    "perceptual_hamming_distance",
    "near_confirmation_method",
    "thumbnail_min_std",
    "thumbnail_intensity_correlation",
    "thumbnail_gradient_correlation",
    "thumbnail_affine_rmse",
    "aspect_ratio_relative_delta",
    "label_summary_conflict",
    "semantic_label_conflict",
    "resolution",
]
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
NEAR_PHASH_THRESHOLD = 4
NEAR_ASPECT_RATIO_DELTA_MAX = 0.02
NEAR_THUMBNAIL_STD_MIN = 5.0
NEAR_INTENSITY_CORRELATION_MIN = 0.995
NEAR_GRADIENT_CORRELATION_MIN = 0.990
NEAR_AFFINE_RMSE_MAX = 0.030
NEAR_CONFIRMATION_METHOD = "phash64_hamming4_thumbnail16_ncc_gradient_affine_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_label_signature(value: Any) -> str:
    payload = json.loads(str(value))
    if not isinstance(payload, dict):
        raise ValueError("label_summary_json must encode an object")
    for field in ("annotation_path", "canonical_yolo_label_path"):
        payload.pop(field, None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _ontology_mapping(path: Path) -> Dict[Tuple[str, str], str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    mapping = {}
    for entry in payload.get("entries", []):
        if entry.get("mapping_status") != "compatible":
            continue
        mapping[(str(entry["dataset_id"]), str(entry["source_label"]))] = str(
            entry["canonical_concept_id"]
        )
    if not mapping:
        raise ValueError("ontology lock has no compatible mappings")
    return mapping


def _semantic_label_signature(
    dataset_id: str,
    value: Any,
    ontology_mapping: Mapping[Tuple[str, str], str],
) -> str:
    payload = json.loads(str(value))
    if "boxes" not in payload and "class_ids" not in payload:
        return _normalized_label_signature(value)
    boxes = payload.get("boxes") or []
    source_labels = {str(item) for item in payload.get("class_ids") or []}
    source_labels.update(
        str(box["source_label"])
        for box in boxes
        if isinstance(box, Mapping) and box.get("source_label") is not None
    )
    canonical = []
    for source_label in sorted(source_labels):
        key = (str(dataset_id), source_label)
        if key not in ontology_mapping:
            raise ValueError(f"missing compatible ontology mapping: {key}")
        canonical.append(ontology_mapping[key])
    semantic = {
        "canonical_concepts": sorted(set(canonical)),
        "observable_presence": bool(boxes),
        "negative_annotation_verified": payload.get("negative_annotation_verified") is True,
    }
    canonical_text = json.dumps(semantic, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()


class _DisjointSet:
    def __init__(self, values: Iterable[Tuple[str, str]]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: Tuple[str, str]) -> Tuple[str, str]:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: Tuple[str, str], right: Tuple[str, str]) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        first, second = sorted((left_root, right_root))
        self.parent[second] = first
        return True


def _expect_equal(errors: list[str], name: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        errors.append(f"{name}: observed={observed!r}, expected={expected!r}")


def review_e034(project_root: Path) -> Dict[str, Any]:
    root = Path(project_root).resolve()
    manifest_path = root / "data/locked/file_manifest.parquet"
    ontology_path = root / "data/locked/ontology_lock.yaml"
    report_path = root / "data/locked/dedup_report.parquet"
    summary_path = root / "evidence/data/dedup_summary.json"
    for path in (manifest_path, ontology_path, report_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = pd.read_parquet(manifest_path)
    report = pd.read_parquet(report_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    ontology_mapping = _ontology_mapping(ontology_path)
    errors: list[str] = []

    required_manifest = {
        "dataset_id",
        "record_id",
        "raw_group_id",
        "relative_path",
        "modality",
        "sha256",
        "label_summary_json",
    }
    missing_manifest = sorted(required_manifest - set(manifest.columns))
    if missing_manifest:
        errors.append(f"manifest missing columns: {missing_manifest}")
    _expect_equal(errors, "report_columns", list(report.columns), REPORT_COLUMNS)
    if errors:
        return _result(root, errors, manifest_path, ontology_path, report_path, summary_path)

    if manifest.duplicated(["dataset_id", "record_id"]).any():
        errors.append("manifest dataset_id/record_id keys are not unique")
    records: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    for row in manifest.to_dict(orient="records"):
        key = (str(row["dataset_id"]), str(row["record_id"]))
        records[key] = row
    group_keys = {
        (str(row.dataset_id), str(row.raw_group_id))
        for row in manifest[["dataset_id", "raw_group_id"]].itertuples(index=False)
    }
    disjoint = _DisjointSet(group_keys)
    seen_group_edges = set()
    exact_rows = 0
    near_rows = 0

    for position, edge in enumerate(report.to_dict(orient="records")):
        prefix = f"edge[{position}]"
        left_record = (str(edge["left_dataset_id"]), str(edge["left_record_id"]))
        right_record = (str(edge["right_dataset_id"]), str(edge["right_record_id"]))
        if left_record not in records or right_record not in records:
            errors.append(f"{prefix}: endpoint record is absent from manifest")
            continue
        left = records[left_record]
        right = records[right_record]
        left_group = (str(left["dataset_id"]), str(left["raw_group_id"]))
        right_group = (str(right["dataset_id"]), str(right["raw_group_id"]))
        _expect_equal(errors, f"{prefix}.left_raw_group_id", str(edge["left_raw_group_id"]), left_group[1])
        _expect_equal(errors, f"{prefix}.right_raw_group_id", str(edge["right_raw_group_id"]), right_group[1])
        if left_group == right_group:
            errors.append(f"{prefix}: duplicate edge stays within one raw group")
        undirected = tuple(sorted((left_group, right_group)))
        if undirected in seen_group_edges:
            errors.append(f"{prefix}: repeated raw-group edge")
        seen_group_edges.add(undirected)
        if not disjoint.union(left_group, right_group):
            errors.append(f"{prefix}: report edges contain a cycle")

        kind = str(edge["duplicate_kind"])
        try:
            distance = int(edge["perceptual_hamming_distance"])
        except (TypeError, ValueError):
            errors.append(f"{prefix}: non-integer perceptual distance")
            continue
        _expect_equal(
            errors,
            f"{prefix}.resolution",
            str(edge["resolution"]),
            "merge_connected_raw_groups_before_split",
        )
        metric_names = (
            "thumbnail_min_std",
            "thumbnail_intensity_correlation",
            "thumbnail_gradient_correlation",
            "thumbnail_affine_rmse",
            "aspect_ratio_relative_delta",
        )
        metrics = {}
        for name in metric_names:
            try:
                metrics[name] = float(edge[name])
            except (TypeError, ValueError):
                errors.append(f"{prefix}.{name}: non-numeric value")
                metrics[name] = float("nan")
            if not math.isfinite(metrics[name]):
                errors.append(f"{prefix}.{name}: non-finite value")
        conflict = _normalized_label_signature(left["label_summary_json"]) != _normalized_label_signature(
            right["label_summary_json"]
        )
        _expect_equal(errors, f"{prefix}.label_summary_conflict", bool(edge["label_summary_conflict"]), conflict)
        semantic_conflict = _semantic_label_signature(
            str(left["dataset_id"]), left["label_summary_json"], ontology_mapping
        ) != _semantic_label_signature(
            str(right["dataset_id"]), right["label_summary_json"], ontology_mapping
        )
        _expect_equal(
            errors,
            f"{prefix}.semantic_label_conflict",
            bool(edge["semantic_label_conflict"]),
            semantic_conflict,
        )
        if kind == "exact_sha256":
            exact_rows += 1
            _expect_equal(
                errors,
                f"{prefix}.near_confirmation_method",
                str(edge["near_confirmation_method"]),
                "exact_sha256",
            )
            _expect_equal(errors, f"{prefix}.sha256", str(left["sha256"]), str(right["sha256"]))
            _expect_equal(errors, f"{prefix}.distance", distance, 0)
            _expect_equal(errors, f"{prefix}.exact_semantic_conflict", semantic_conflict, False)
        elif kind == "near_phash64":
            near_rows += 1
            _expect_equal(
                errors,
                f"{prefix}.near_confirmation_method",
                str(edge["near_confirmation_method"]),
                NEAR_CONFIRMATION_METHOD,
            )
            if not 0 <= distance <= NEAR_PHASH_THRESHOLD:
                errors.append(
                    f"{prefix}: near-duplicate distance {distance} is outside [0, {NEAR_PHASH_THRESHOLD}]"
                )
            if str(left.get("modality", "")).strip().lower() != str(
                right.get("modality", "")
            ).strip().lower():
                errors.append(f"{prefix}: confirmed near duplicate crosses modalities")
            if metrics["thumbnail_min_std"] < NEAR_THUMBNAIL_STD_MIN:
                errors.append(f"{prefix}: near duplicate is a low-information thumbnail")
            if metrics["thumbnail_intensity_correlation"] < NEAR_INTENSITY_CORRELATION_MIN:
                errors.append(f"{prefix}: intensity correlation is below the frozen threshold")
            if metrics["thumbnail_gradient_correlation"] < NEAR_GRADIENT_CORRELATION_MIN:
                errors.append(f"{prefix}: gradient correlation is below the frozen threshold")
            if metrics["thumbnail_affine_rmse"] > NEAR_AFFINE_RMSE_MAX:
                errors.append(f"{prefix}: affine RMSE exceeds the frozen threshold")
            if metrics["aspect_ratio_relative_delta"] > NEAR_ASPECT_RATIO_DELTA_MAX:
                errors.append(f"{prefix}: aspect-ratio delta exceeds the frozen threshold")
            _expect_equal(errors, f"{prefix}.near_semantic_conflict", semantic_conflict, False)
        else:
            errors.append(f"{prefix}: unknown duplicate_kind {kind!r}")

    component_sizes: Dict[Tuple[str, str], int] = {}
    for group in group_keys:
        root_group = disjoint.find(group)
        component_sizes[root_group] = component_sizes.get(root_group, 0) + 1
    affected_group_count = sum(size for size in component_sizes.values() if size > 1)
    exact_groups = manifest.groupby("sha256", sort=False).size()
    exact_sha_group_count = int((exact_groups >= 2).sum())
    exact_record_pair_count = int(sum(int(size) * (int(size) - 1) // 2 for size in exact_groups if size >= 2))
    exact_label_summary_conflict_group_count = 0
    exact_semantic_conflict_group_count = 0
    for _, group in manifest.groupby("sha256", sort=False):
        if len(group) < 2:
            continue
        if group["label_summary_json"].map(_normalized_label_signature).nunique() > 1:
            exact_label_summary_conflict_group_count += 1
        semantic_signatures = {
            _semantic_label_signature(
                str(row.dataset_id), row.label_summary_json, ontology_mapping
            )
            for row in group[["dataset_id", "label_summary_json"]].itertuples(index=False)
        }
        if len(semantic_signatures) > 1:
            exact_semantic_conflict_group_count += 1
    image_record_count = int(
        manifest["relative_path"].map(lambda value: Path(str(value)).suffix.lower() in IMAGE_SUFFIXES).sum()
    )

    expected_summary = {
        "schema_version": 1,
        "step_id": "E034",
        "status": "pass",
        "file_manifest_sha256": _sha256_file(manifest_path),
        "ontology_lock_sha256": _sha256_file(ontology_path),
        "pair_count": len(report),
        "exact_pair_count": exact_rows,
        "near_pair_count": near_rows,
        "report_semantics": "minimal_deterministic_spanning_edges_per_duplicate_component",
        "raw_group_count": len(group_keys),
        "dedup_component_count": len(component_sizes),
        "affected_raw_group_count": affected_group_count,
        "exact_sha_group_count": exact_sha_group_count,
        "exact_record_pair_count": exact_record_pair_count,
        "exact_label_summary_conflict_group_count": exact_label_summary_conflict_group_count,
        "exact_semantic_conflict_group_count": exact_semantic_conflict_group_count,
        "label_summary_conflict_edge_count": int(report["label_summary_conflict"].sum()),
        "semantic_label_conflict_edge_count": int(report["semantic_label_conflict"].sum()),
        "semantic_conflict_policy": "retain_coordinate_detail_variation_only_when_ontology_semantics_match; fail_exact_or_confirmed_near_semantic_conflict",
        "phash_record_count": image_record_count,
        "near_duplicate_method": NEAR_CONFIRMATION_METHOD,
        "near_confirmation": {
            "same_modality_required": True,
            "aspect_ratio_relative_delta_max": NEAR_ASPECT_RATIO_DELTA_MAX,
            "thumbnail_min_std": NEAR_THUMBNAIL_STD_MIN,
            "thumbnail_intensity_correlation_min": NEAR_INTENSITY_CORRELATION_MIN,
            "thumbnail_gradient_correlation_min": NEAR_GRADIENT_CORRELATION_MIN,
            "thumbnail_affine_rmse_max": NEAR_AFFINE_RMSE_MAX,
        },
        "candidate_index": "bktree_hamming64",
        "perceptual_hamming_threshold": NEAR_PHASH_THRESHOLD,
        "resolution": "merge_connected_raw_groups_before_split",
        "model_outcomes_used": False,
    }
    for name, expected in expected_summary.items():
        _expect_equal(errors, f"summary.{name}", summary.get(name), expected)
    for name in (
        "unique_phash_count",
        "same_phash_group_pair_candidate_count",
        "near_unique_hash_pair_count",
        "near_group_pair_comparison_count",
        "near_structurally_confirmed_pair_count",
        "near_rejected_modality_count",
        "near_rejected_structure_count",
    ):
        value = summary.get(name)
        if not isinstance(value, int) or value < 0:
            errors.append(f"summary.{name} must be a nonnegative integer")
    unique_phash_count = summary.get("unique_phash_count")
    if isinstance(unique_phash_count, int) and unique_phash_count > image_record_count:
        errors.append("summary.unique_phash_count exceeds phash_record_count")
    worker_count = summary.get("phash_worker_count")
    if not isinstance(worker_count, int) or not 1 <= worker_count <= 8:
        errors.append("summary.phash_worker_count must be in [1, 8]")
    comparison_count = summary.get("near_group_pair_comparison_count")
    confirmed_count = summary.get("near_structurally_confirmed_pair_count")
    rejected_modality = summary.get("near_rejected_modality_count")
    rejected_structure = summary.get("near_rejected_structure_count")
    if all(
        isinstance(value, int)
        for value in (comparison_count, confirmed_count, rejected_modality, rejected_structure)
    ):
        _expect_equal(
            errors,
            "summary.near_candidate_accounting",
            comparison_count,
            confirmed_count + rejected_modality + rejected_structure,
        )
        if confirmed_count < near_rows:
            errors.append("summary.near_structurally_confirmed_pair_count is below near_pair_count")

    return _result(root, errors, manifest_path, ontology_path, report_path, summary_path)


def _result(
    root: Path,
    errors: Sequence[str],
    manifest_path: Path,
    ontology_path: Path,
    report_path: Path,
    summary_path: Path,
) -> Dict[str, Any]:
    bindings = {}
    for path in (manifest_path, ontology_path, report_path, summary_path):
        if path.is_file():
            bindings[path.relative_to(root).as_posix()] = _sha256_file(path)
    return {
        "schema_version": 1,
        "step_id": "E034",
        "reviewer": "independent_local_dedup_reconciler",
        "status": "pass" if not errors else "fail",
        "error_count": len(errors),
        "errors": list(errors),
        "bindings": bindings,
        "near_edge_verification_scope": "reported_hamming_distance_range_and_tested_clean_room_algorithm",
        "model_outcomes_used": False,
        "performance_claims_authorized": False,
    }


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"immutable review output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Independently reconcile E034 dedup artifacts")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = review_e034(args.project_root)
    if args.output is not None:
        output = args.output if args.output.is_absolute() else args.project_root / args.output
        _write_once(output, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
