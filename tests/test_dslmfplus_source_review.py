from __future__ import annotations

import pytest

from tools.data.review_dslmfplus_source import (
    EXPECTED_EXCLUSIONS,
    DsLMFSourceReviewError,
    build_adapter_contract,
    validate_archive_audit,
)


def _mapping() -> dict:
    ranges = [
        {"value": f"scenario-{index:02d}", "start": str(index), "end": str(index)}
        for index in range(1, 59)
    ]
    ranges.extend(
        {
            "value": "scenario-58",
            "start": f"p{index}-1",
            "end": f"p{index}-1",
        }
        for index in range(1, 56)
    )
    return {
        "schema_version": "mining1.dslmfplus_scenario_map.v1",
        "scenario_count": 58,
        "adapter_ranges": ranges,
        "excluded_image_stems": EXPECTED_EXCLUSIONS,
    }


def test_adapter_contract_binds_official_ranges_and_reviewed_drops() -> None:
    contract = build_adapter_contract(_mapping())
    assert len(contract["raw_group_rule"]["filename_range_lookup"]["ranges"]) == 113
    assert contract["yolo"]["excluded_image_stems"] == sorted(EXPECTED_EXCLUSIONS)
    assert contract["yolo"]["missing_label_policy"] == "error"


def test_adapter_contract_rejects_changed_exclusion_set() -> None:
    mapping = _mapping()
    mapping["excluded_image_stems"] = EXPECTED_EXCLUSIONS[:-1]
    with pytest.raises(DsLMFSourceReviewError, match="exclusion set changed"):
        build_adapter_contract(mapping)


def test_source_evidence_rejects_nonunique_scene_mapping() -> None:
    audit = {
        "yolo": {
            "total_image_count": 30704,
            "total_label_count": 30704,
            "counts_match_official_report": True,
            "cross_split_image_overlap_count": 0,
            "zero_byte_label_count": 13,
            "pairing": {
                "train": {"image_label_ids_equal": True},
                "val": {"image_label_ids_equal": True},
            },
        },
        "scenario_mapping": {
            "declared_scenario_count": 58,
            "observed_scenario_count": 58,
            "eligible_image_count": 30699,
            "all_eligible_images_mapped_exactly_once": False,
            "unmapped_image_count": 1,
            "multiply_mapped_image_count": 0,
            "observed_excluded_filename_stems": EXPECTED_EXCLUSIONS,
        },
        "coco": {
            "train": {"invalid_bbox_count": 0},
            "val": {"invalid_bbox_count": 0},
            "summary": {
                "total_image_count_equal": True,
                "execution_truth": "yolo_original_names_and_splits",
            },
        },
    }
    with pytest.raises(DsLMFSourceReviewError, match="scenario mapping failed"):
        validate_archive_audit(audit)
