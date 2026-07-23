from __future__ import annotations

import argparse
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree as ET


IMAGE_PATTERN = re.compile(
    r"^LLVIP/(?P<modality>visible|infrared)/(?P<split>train|test)/"
    r"(?P<record>[0-9]{6})[.]jpg$"
)
ANNOTATION_PATTERN = re.compile(r"^LLVIP/Annotations/(?P<record>[0-9]{6})[.]xml$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--class-name", default="person")
    parser.add_argument("--example-limit", type=int, default=20)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _normalized_member(name: str) -> str:
    return name.replace("\\", "/")


def _is_safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts


def audit(archive: Path, class_name: str, example_limit: int) -> dict:
    records: dict[tuple[str, str], set[str]] = defaultdict(set)
    annotations: dict[str, zipfile.ZipInfo] = {}
    casefold_names: Counter[str] = Counter()
    unsafe_members: list[str] = []

    with zipfile.ZipFile(archive) as handle:
        for info in handle.infolist():
            if info.is_dir():
                continue
            member = _normalized_member(info.filename)
            casefold_names[member.casefold()] += 1
            if not _is_safe_member(member):
                unsafe_members.append(member)
                continue
            image_match = IMAGE_PATTERN.fullmatch(member)
            if image_match:
                records[
                    (image_match.group("modality"), image_match.group("split"))
                ].add(image_match.group("record"))
                continue
            annotation_match = ANNOTATION_PATTERN.fullmatch(member)
            if annotation_match:
                annotations[annotation_match.group("record")] = info

        visible_all = records[("visible", "train")] | records[("visible", "test")]
        infrared_all = records[("infrared", "train")] | records[("infrared", "test")]
        class_counts: Counter[str] = Counter()
        class_positive_images: Counter[str] = Counter()
        class_positive_groups: dict[str, set[str]] = defaultdict(set)
        difficult_counts: Counter[str] = Counter()
        invalid_bbox_count = 0
        invalid_examples: list[dict] = []
        filename_mismatch_count = 0
        malformed_xml_count = 0

        for record_id, info in sorted(annotations.items()):
            try:
                root = ET.fromstring(handle.read(info))
            except (ET.ParseError, UnicodeDecodeError):
                malformed_xml_count += 1
                continue
            declared = PurePosixPath(
                (root.findtext("filename") or "").strip().replace("\\", "/")
            ).stem
            if declared != record_id:
                filename_mismatch_count += 1
            size = root.find("size")
            try:
                width = float(size.findtext("width", "nan")) if size is not None else math.nan
                height = (
                    float(size.findtext("height", "nan"))
                    if size is not None
                    else math.nan
                )
            except ValueError:
                width = math.nan
                height = math.nan

            present_classes: set[str] = set()
            for object_node in root.findall("object"):
                label = (object_node.findtext("name") or "").strip()
                class_counts[label] += 1
                present_classes.add(label)
                try:
                    difficult = int(object_node.findtext("difficult", "0"))
                except ValueError:
                    difficult = -1
                if difficult:
                    difficult_counts[label] += 1
                bbox = object_node.find("bndbox")
                reasons: list[str] = []
                try:
                    values = [
                        float(bbox.findtext(key, "nan"))
                        for key in ("xmin", "ymin", "xmax", "ymax")
                    ] if bbox is not None else [math.nan] * 4
                except ValueError:
                    values = [math.nan] * 4
                if not all(math.isfinite(value) for value in values):
                    reasons.append("nonfinite_or_missing")
                else:
                    xmin, ymin, xmax, ymax = values
                    if xmax <= xmin or ymax <= ymin:
                        reasons.append("nonpositive")
                    if xmin < 0 or ymin < 0:
                        reasons.append("negative_origin")
                    if math.isfinite(width) and math.isfinite(height) and (
                        xmax > width or ymax > height
                    ):
                        reasons.append("exceeds_declared_dimensions")
                if reasons:
                    invalid_bbox_count += 1
                    if len(invalid_examples) < example_limit:
                        invalid_examples.append(
                            {
                                "record_id": record_id,
                                "class_name": label,
                                "bbox": values,
                                "declared_dimensions": [width, height],
                                "reasons": reasons,
                            }
                        )
            for label in present_classes:
                class_positive_images[label] += 1
                class_positive_groups[label].add(record_id[:2])

    train_ids = records[("visible", "train")]
    test_ids = records[("visible", "test")]
    all_ids = train_ids | test_ids
    group_counts = {
        split: dict(sorted(Counter(record[:2] for record in ids).items()))
        for split, ids in (("train", train_ids), ("test", test_ids))
    }
    return {
        "schema_version": "mining1.llvip_archive_audit.v1",
        "archive": archive.as_posix(),
        "class_name": class_name,
        "archive_structure": {
            "unsafe_member_count": len(unsafe_members),
            "casefold_duplicate_count": sum(
                count - 1 for count in casefold_names.values() if count > 1
            ),
            "annotation_count": len(annotations),
            "visible_train_count": len(records[("visible", "train")]),
            "visible_test_count": len(records[("visible", "test")]),
            "infrared_train_count": len(records[("infrared", "train")]),
            "infrared_test_count": len(records[("infrared", "test")]),
        },
        "pairing": {
            "visible_infrared_train_ids_equal": records[("visible", "train")]
            == records[("infrared", "train")],
            "visible_infrared_test_ids_equal": records[("visible", "test")]
            == records[("infrared", "test")],
            "annotation_ids_equal_image_pair_ids": set(annotations) == all_ids,
            "cross_split_record_overlap_count": len(train_ids & test_ids),
        },
        "group_prefix_audit": {
            "rule": "first_two_digits_of_six_digit_record_id",
            "train": group_counts["train"],
            "test": group_counts["test"],
            "total_group_count": len({record[:2] for record in all_ids}),
            "cross_split_group_overlap": sorted(
                {record[:2] for record in train_ids}
                & {record[:2] for record in test_ids}
            ),
        },
        "annotations": {
            "class_object_counts": dict(sorted(class_counts.items())),
            "class_positive_image_counts": dict(sorted(class_positive_images.items())),
            "class_positive_group_counts": {
                label: len(groups)
                for label, groups in sorted(class_positive_groups.items())
            },
            "difficult_object_counts": dict(sorted(difficult_counts.items())),
            "malformed_xml_count": malformed_xml_count,
            "filename_mismatch_count": filename_mismatch_count,
            "invalid_bbox_count": invalid_bbox_count,
            "invalid_bbox_examples": invalid_examples,
        },
    }


def main() -> int:
    args = parse_args()
    result = audit(args.archive, args.class_name, args.example_limit)
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
