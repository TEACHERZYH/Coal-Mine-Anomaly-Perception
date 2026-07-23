from __future__ import annotations

import json

import pytest

from tools.data.audit_dslmfplus_archive import audit_coco_files, audit_entries
from tools.data.extract_dslmf_scenario_map import (
    ScenarioMapError,
    _adapter_ranges,
    _overlap_audit,
    parse_scenario_text,
)


def _mapping() -> dict:
    scenarios = parse_scenario_text(
        """
Scenario 1
0000001.jpg-0000010.jpg
p1_1.jpg-p1_9.jpg
Scenario 2
0000011.jpg-0000020.jpg
p2_1.jpg-p2_9.jpg
""",
        expected_scenarios=2,
    )
    return {
        "scenario_count": 2,
        "scenarios": scenarios,
        "adapter_ranges": [item for scenario in scenarios for item in scenario["ranges"]],
        "ambiguous_filename_stems": [],
        "excluded_image_stems": [],
    }


def test_scenario_parser_normalizes_dataset_hyphens_and_rejects_missing_headings() -> None:
    scenarios = _mapping()["scenarios"]
    assert scenarios[0]["ranges"][1] == {
        "value": "scenario-01",
        "start": "p1-1",
        "end": "p1-9",
    }
    with pytest.raises(ScenarioMapError, match="headings are incomplete"):
        parse_scenario_text("Scenario 1\n1.jpg-2.jpg", expected_scenarios=2)


def test_archive_audit_checks_pairs_and_exact_scenario_coverage() -> None:
    root = "DsLMF/data2023_yolo/coal_miner_data2023_yolo"
    entries = []
    for split, stems in (("train", ("0000001", "p1-2")), ("val", ("0000011", "p2-2"))):
        for stem in stems:
            entries.extend(
                [
                    {"path": f"{root}/images/{split}/{stem}.jpg", "size": 10, "is_dir": False},
                    {"path": f"{root}/labels/{split}/{stem}.txt", "size": 8, "is_dir": False},
                ]
            )
    result = audit_entries(entries, _mapping())
    assert result["yolo"]["pairing"]["train"]["image_label_ids_equal"] is True
    assert result["yolo"]["pairing"]["val"]["image_label_ids_equal"] is True
    assert result["scenario_mapping"]["all_eligible_images_mapped_exactly_once"] is True
    assert result["scenario_mapping"]["observed_scenario_count"] == 2


def test_coco_audit_matches_yolo_ids_and_detects_invalid_box(tmp_path) -> None:
    payloads = {
        "train": {
            "images": [{"id": 1, "file_name": "0000001.jpg", "width": 10, "height": 8}],
            "categories": [{"id": 0, "name": "coal miner"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [1, 1, 4, 5]}],
        },
        "val": {
            "images": [{"id": 2, "file_name": "0000011.jpg", "width": 10, "height": 8}],
            "categories": [{"id": 0, "name": "coal miner"}],
            "annotations": [{"id": 2, "image_id": 2, "category_id": 0, "bbox": [8, 1, 4, 5]}],
        },
    }
    paths = {}
    for split, payload in payloads.items():
        path = tmp_path / f"{split}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[split] = path
    result = audit_coco_files(
        paths["train"], paths["val"], {"train": ["0000001"], "val": ["0000011"]}
    )
    assert result["train"]["yolo_image_ids_equal"] is True
    assert result["train"]["invalid_bbox_count"] == 0
    assert result["val"]["invalid_bbox_count"] == 1
    assert result["summary"]["total_image_count_equal"] is True


def test_official_overlap_is_preserved_and_ambiguous_ids_are_dropped() -> None:
    scenarios = parse_scenario_text(
        """
Scenario 1
0000001.jpg-0000012.jpg
Scenario 2
0000011.jpg-0000020.jpg
""",
        expected_scenarios=2,
    )
    overlaps = _overlap_audit(scenarios)
    ambiguous = set(overlaps[0]["ambiguous_filename_stems"])
    assert ambiguous == {"0000011", "0000012"}
    assert _adapter_ranges(scenarios, ambiguous) == [
        {"value": "scenario-01", "start": "0000001", "end": "0000010"},
        {"value": "scenario-02", "start": "0000013", "end": "0000020"},
    ]
