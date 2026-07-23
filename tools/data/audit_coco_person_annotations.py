from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation_json", type=Path)
    parser.add_argument("--category-id", type=int, default=1)
    parser.add_argument("--allowed-license-id", type=int, action="append")
    parser.add_argument("--example-limit", type=int, default=20)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def audit(
    annotation_json: Path,
    category_id: int,
    example_limit: int,
    allowed_license_ids: list[int] | None = None,
) -> dict:
    payload = json.loads(annotation_json.read_text(encoding="utf-8"))
    image_dimensions = {
        image["id"]: (image["width"], image["height"])
        for image in payload["images"]
    }
    image_licenses = {
        image["id"]: int(image.get("license", -1)) for image in payload["images"]
    }
    counts = {
        "selected_annotations": 0,
        "missing_image": 0,
        "nonfinite_or_invalid_shape": 0,
        "nonpositive": 0,
        "negative_origin": 0,
        "exceeds_image_bounds": 0,
    }
    examples: list[dict] = []
    selected_image_ids: set[int] = set()
    usable_noncrowd_image_ids: set[int] = set()

    for annotation in payload["annotations"]:
        if annotation.get("category_id") != category_id:
            continue
        counts["selected_annotations"] += 1
        image_id = annotation.get("image_id")
        selected_image_ids.add(image_id)
        dimensions = image_dimensions.get(image_id)
        bbox = annotation.get("bbox", [])
        reasons: list[str] = []

        if dimensions is None:
            counts["missing_image"] += 1
            reasons.append("missing_image")
        if len(bbox) != 4 or not all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in bbox
        ):
            counts["nonfinite_or_invalid_shape"] += 1
            reasons.append("nonfinite_or_invalid_shape")
        else:
            x, y, width, height = map(float, bbox)
            if width <= 0 or height <= 0:
                counts["nonpositive"] += 1
                reasons.append("nonpositive")
            if x < 0 or y < 0:
                counts["negative_origin"] += 1
                reasons.append("negative_origin")
            if dimensions is not None and (
                x + width > dimensions[0] + 1e-9
                or y + height > dimensions[1] + 1e-9
            ):
                counts["exceeds_image_bounds"] += 1
                reasons.append("exceeds_image_bounds")

        if reasons and len(examples) < example_limit:
            examples.append(
                {
                    "annotation_id": annotation.get("id"),
                    "image_id": image_id,
                    "bbox": bbox,
                    "image_dimensions": dimensions,
                    "iscrowd": annotation.get("iscrowd"),
                    "reasons": reasons,
                }
            )
        if not reasons and int(annotation.get("iscrowd", 0)) == 0:
            usable_noncrowd_image_ids.add(image_id)

    person_image_license_counts: dict[str, int] = {}
    for image_id in selected_image_ids:
        license_id = str(image_licenses.get(image_id, -1))
        person_image_license_counts[license_id] = (
            person_image_license_counts.get(license_id, 0) + 1
        )
    allowed = sorted(set(allowed_license_ids or []))
    eligible_image_ids = {
        image_id
        for image_id in selected_image_ids
        if not allowed or image_licenses.get(image_id, -1) in allowed
    }
    usable_eligible_image_ids = usable_noncrowd_image_ids & eligible_image_ids
    return {
        "schema_version": "mining1.coco_annotation_boundary_audit.v1",
        "annotation_json": annotation_json.as_posix(),
        "category_id": category_id,
        "counts": counts,
        "license_boundary": {
            "allowed_license_ids": allowed,
            "person_image_license_counts": dict(
                sorted(person_image_license_counts.items(), key=lambda item: int(item[0]))
            ),
            "eligible_person_image_count": len(eligible_image_ids),
            "excluded_person_image_count": len(selected_image_ids - eligible_image_ids),
            "usable_eligible_person_image_count": len(usable_eligible_image_ids),
            "eligible_without_usable_noncrowd_box_count": len(
                eligible_image_ids - usable_eligible_image_ids
            ),
        },
        "examples": examples,
    }


def main() -> int:
    args = parse_args()
    result = audit(
        args.annotation_json,
        args.category_id,
        args.example_limit,
        args.allowed_license_id,
    )
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
