import hashlib
import json
from pathlib import Path

import pandas as pd

from mining1_exp.review_dedup import REPORT_COLUMNS, review_e034


def _label(label: str) -> str:
    return json.dumps(
        {
            "annotation_path": f"labels/{label}.txt",
            "boxes": [],
            "class_ids": [label],
            "negative_annotation_verified": True,
        },
        sort_keys=True,
    )


def _fixture(root: Path) -> None:
    (root / "data/locked").mkdir(parents=True)
    (root / "evidence/data").mkdir(parents=True)
    ontology = {
        "ontology_version": "v1",
        "entries": [
            {
                "dataset_id": f"d{index}",
                "source_label": label,
                "canonical_concept_id": concept,
                "mapping_status": "compatible",
            }
            for index, label, concept in (
                (1, "person", "worker_presence"),
                (2, "person", "worker_presence"),
                (3, "person", "worker_presence"),
                (4, "worker", "worker_presence"),
            )
        ],
    }
    (root / "data/locked/ontology_lock.yaml").write_text(
        json.dumps(ontology), encoding="utf-8"
    )
    rows = []
    for index, (digest, label) in enumerate(
        (("a" * 64, "person"), ("a" * 64, "person"), ("b" * 64, "person"), ("c" * 64, "worker")),
        start=1,
    ):
        rows.append(
            {
                "dataset_id": f"d{index}",
                "record_id": f"r{index}",
                "raw_group_id": f"g{index}",
                "relative_path": f"data/canonical/d{index}/r{index}.png",
                "modality": "visible",
                "sha256": digest,
                "label_summary_json": _label(label),
            }
        )
    manifest_path = root / "data/locked/file_manifest.parquet"
    manifest = pd.DataFrame(rows)
    manifest.to_parquet(manifest_path, index=False)
    report = pd.DataFrame(
        [
            {
                "left_dataset_id": "d1",
                "left_record_id": "r1",
                "left_raw_group_id": "g1",
                "right_dataset_id": "d2",
                "right_record_id": "r2",
                "right_raw_group_id": "g2",
                "duplicate_kind": "exact_sha256",
                "perceptual_hamming_distance": 0,
                "near_confirmation_method": "exact_sha256",
                "thumbnail_min_std": 0.0,
                "thumbnail_intensity_correlation": 1.0,
                "thumbnail_gradient_correlation": 1.0,
                "thumbnail_affine_rmse": 0.0,
                "aspect_ratio_relative_delta": 0.0,
                "label_summary_conflict": False,
                "semantic_label_conflict": False,
                "resolution": "merge_connected_raw_groups_before_split",
            },
            {
                "left_dataset_id": "d3",
                "left_record_id": "r3",
                "left_raw_group_id": "g3",
                "right_dataset_id": "d4",
                "right_record_id": "r4",
                "right_raw_group_id": "g4",
                "duplicate_kind": "near_phash64",
                "perceptual_hamming_distance": 3,
                "near_confirmation_method": "phash64_hamming4_thumbnail16_ncc_gradient_affine_v1",
                "thumbnail_min_std": 25.0,
                "thumbnail_intensity_correlation": 0.999,
                "thumbnail_gradient_correlation": 0.998,
                "thumbnail_affine_rmse": 0.005,
                "aspect_ratio_relative_delta": 0.0,
                "label_summary_conflict": True,
                "semantic_label_conflict": False,
                "resolution": "merge_connected_raw_groups_before_split",
            },
        ],
        columns=REPORT_COLUMNS,
    )
    report.to_parquet(root / "data/locked/dedup_report.parquet", index=False)
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    summary = {
        "schema_version": 1,
        "step_id": "E034",
        "status": "pass",
        "file_manifest_sha256": manifest_hash,
        "ontology_lock_sha256": hashlib.sha256(
            (root / "data/locked/ontology_lock.yaml").read_bytes()
        ).hexdigest(),
        "pair_count": 2,
        "exact_pair_count": 1,
        "near_pair_count": 1,
        "report_semantics": "minimal_deterministic_spanning_edges_per_duplicate_component",
        "raw_group_count": 4,
        "dedup_component_count": 2,
        "affected_raw_group_count": 4,
        "exact_sha_group_count": 1,
        "exact_record_pair_count": 1,
        "exact_label_summary_conflict_group_count": 0,
        "exact_semantic_conflict_group_count": 0,
        "label_summary_conflict_edge_count": 1,
        "semantic_label_conflict_edge_count": 0,
        "semantic_conflict_policy": "retain_coordinate_detail_variation_only_when_ontology_semantics_match; fail_exact_or_confirmed_near_semantic_conflict",
        "phash_record_count": 4,
        "unique_phash_count": 3,
        "same_phash_group_pair_candidate_count": 0,
        "near_unique_hash_pair_count": 1,
        "near_group_pair_comparison_count": 1,
        "near_structurally_confirmed_pair_count": 1,
        "near_rejected_modality_count": 0,
        "near_rejected_structure_count": 0,
        "near_duplicate_method": "phash64_hamming4_thumbnail16_ncc_gradient_affine_v1",
        "near_confirmation": {
            "same_modality_required": True,
            "aspect_ratio_relative_delta_max": 0.02,
            "thumbnail_min_std": 5.0,
            "thumbnail_intensity_correlation_min": 0.995,
            "thumbnail_gradient_correlation_min": 0.99,
            "thumbnail_affine_rmse_max": 0.03,
        },
        "candidate_index": "bktree_hamming64",
        "perceptual_hamming_threshold": 4,
        "phash_worker_count": 2,
        "resolution": "merge_connected_raw_groups_before_split",
        "model_outcomes_used": False,
    }
    (root / "evidence/data/dedup_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )


def test_review_e034_accepts_reconciled_forest(tmp_path: Path) -> None:
    _fixture(tmp_path)
    result = review_e034(tmp_path)
    assert result["status"] == "pass"
    assert result["error_count"] == 0
    assert result["bindings"]["data/locked/ontology_lock.yaml"]


def test_review_e034_rejects_cycle_and_threshold_violation(tmp_path: Path) -> None:
    _fixture(tmp_path)
    path = tmp_path / "data/locked/dedup_report.parquet"
    report = pd.read_parquet(path)
    extra = report.iloc[1].copy()
    extra["left_dataset_id"] = "d2"
    extra["left_record_id"] = "r2"
    extra["left_raw_group_id"] = "g2"
    extra["right_dataset_id"] = "d1"
    extra["right_record_id"] = "r1"
    extra["right_raw_group_id"] = "g1"
    extra["perceptual_hamming_distance"] = 5
    pd.concat([report, pd.DataFrame([extra])], ignore_index=True).to_parquet(path, index=False)
    result = review_e034(tmp_path)
    assert result["status"] == "fail"
    assert any("cycle" in error or "repeated raw-group edge" in error for error in result["errors"])
    assert any("outside [0, 4]" in error for error in result["errors"])
