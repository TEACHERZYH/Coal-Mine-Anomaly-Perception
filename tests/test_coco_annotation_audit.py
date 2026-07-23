from __future__ import annotations

import json

from tools.data.audit_coco_person_annotations import audit


def test_coco_boundary_audit_classifies_invalid_boxes(tmp_path):
    annotation_path = tmp_path / "instances.json"
    annotation_path.write_text(
        json.dumps(
            {
                "images": [{"id": 7, "width": 100, "height": 80, "license": 4}],
                "annotations": [
                    {"id": 1, "image_id": 7, "category_id": 1, "bbox": [1, 2, 3, 4]},
                    {"id": 2, "image_id": 7, "category_id": 1, "bbox": [99, 2, 3, 4]},
                    {"id": 3, "image_id": 7, "category_id": 1, "bbox": [-1, 2, 3, 0]},
                    {"id": 4, "image_id": 8, "category_id": 1, "bbox": [1, 2, 3, 4]},
                    {"id": 5, "image_id": 7, "category_id": 2, "bbox": [0, 0, 1, 1]},
                ],
            }
        ),
        encoding="utf-8",
    )

    result = audit(
        annotation_path,
        category_id=1,
        example_limit=10,
        allowed_license_ids=[4],
    )

    assert result["counts"] == {
        "selected_annotations": 4,
        "missing_image": 1,
        "nonfinite_or_invalid_shape": 0,
        "nonpositive": 1,
        "negative_origin": 1,
        "exceeds_image_bounds": 1,
    }
    assert [item["annotation_id"] for item in result["examples"]] == [2, 3, 4]
    assert result["license_boundary"] == {
        "allowed_license_ids": [4],
        "person_image_license_counts": {"-1": 1, "4": 1},
        "eligible_person_image_count": 1,
        "excluded_person_image_count": 1,
        "usable_eligible_person_image_count": 1,
        "eligible_without_usable_noncrowd_box_count": 0,
    }
