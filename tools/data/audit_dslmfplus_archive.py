from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Any, Iterable, Iterator, Mapping

from mining1_exp.data.adapters import _seven_zip_executable


YOLO_ROOT = "DsLMF/data2023_yolo/coal_miner_data2023_yolo"
YOLO_PATTERN = re.compile(
    rf"^{re.escape(YOLO_ROOT)}/(?P<kind>images|labels)/(?P<split>train|val)/"
    r"(?P<stem>(?:[0-9]+|[pP][0-9]+-[0-9]+))[.](?P<suffix>jpg|txt)$"
)
EXPECTED_COUNTS = {"train": 24563, "val": 6141}


class DsLMFAuditError(ValueError):
    """Raised when the bounded DsLMF+ audit cannot establish source integrity."""


def _safe_member(value: str) -> str:
    text = value.replace("\\", "/").strip()
    path = PurePosixPath(text)
    if not text or path.is_absolute() or ".." in path.parts:
        raise DsLMFAuditError(f"Unsafe 7z member: {value}")
    return text


def _finish_slt_record(current: Mapping[str, str]) -> dict[str, Any] | None:
    if not current:
        return None
    path = _safe_member(current.get("Path", ""))
    attributes = current.get("Attributes", "")
    if "Symbolic Link" in current or "Hard Link" in current or attributes.lower().startswith("l"):
        raise DsLMFAuditError(f"Link member is forbidden: {path}")
    if current.get("Encrypted", "-") == "+":
        raise DsLMFAuditError(f"Encrypted member is forbidden: {path}")
    size = int(current.get("Size", "0"))
    if size < 0:
        raise DsLMFAuditError(f"Negative member size: {path}")
    return {"path": path, "size": size, "is_dir": attributes.upper().startswith("D")}


def iter_7z_entries(archive: Path) -> Iterator[dict[str, Any]]:
    include = r"-ir!DsLMF\data2023_yolo\coal_miner_data2023_yolo\*"
    process = subprocess.Popen(
        [_seven_zip_executable(), "l", "-slt", "-ba", "-sccUTF-8", str(archive), include],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if process.stdout is None or process.stderr is None:
        raise DsLMFAuditError("Could not capture 7z listing streams")
    current: dict[str, str] = {}
    for raw_line in process.stdout:
        line = raw_line.rstrip("\r\n")
        if not line.strip():
            entry = _finish_slt_record(current)
            if entry is not None:
                yield entry
            current = {}
        elif " = " in line:
            key, value = line.split(" = ", 1)
            current[key] = value
    entry = _finish_slt_record(current)
    if entry is not None:
        yield entry
    stderr = process.stderr.read()
    return_code = process.wait()
    if return_code != 0:
        raise DsLMFAuditError(f"7z listing failed: {stderr.strip()[-500:]}")


def _filename_key(value: str) -> tuple[int, int, int]:
    if re.fullmatch(r"[0-9]+", value):
        return (0, int(value), 0)
    match = re.fullmatch(r"[pP]([0-9]+)-([0-9]+)", value)
    if match is None:
        raise DsLMFAuditError(f"Unsupported coal-miner filename stem: {value}")
    return (1, int(match.group(1)), int(match.group(2)))


def _scenario_intervals(mapping: Mapping[str, Any]) -> list[tuple[Any, Any, str]]:
    intervals = [
        (
            _filename_key(str(item["start"])),
            _filename_key(str(item["end"])),
            str(item["value"]),
        )
        for item in mapping.get("adapter_ranges", [])
    ]
    if not intervals:
        for scenario in mapping.get("scenarios", []):
            scenario_id = str(scenario.get("scenario_id", ""))
            for item in scenario.get("ranges", []):
                intervals.append(
                    (
                        _filename_key(str(item["start"])),
                        _filename_key(str(item["end"])),
                        scenario_id,
                    )
                )
    if not intervals:
        raise DsLMFAuditError("Scenario mapping contains no intervals")
    return intervals


def audit_entries(
    entries: Iterable[Mapping[str, Any]], mapping: Mapping[str, Any]
) -> dict[str, Any]:
    records: dict[tuple[str, str], set[str]] = defaultdict(set)
    casefold_paths: Counter[str] = Counter()
    unexpected_files: list[str] = []
    zero_byte_labels = 0
    zero_byte_label_examples: list[str] = []
    member_count = 0
    for raw_entry in entries:
        path = _safe_member(str(raw_entry["path"]))
        casefold_paths[path.casefold()] += 1
        if bool(raw_entry.get("is_dir")):
            continue
        member_count += 1
        match = YOLO_PATTERN.fullmatch(path)
        if match is None:
            if len(unexpected_files) < 20:
                unexpected_files.append(path)
            continue
        kind = match.group("kind")
        suffix = match.group("suffix")
        if (kind == "images" and suffix != "jpg") or (kind == "labels" and suffix != "txt"):
            raise DsLMFAuditError(f"YOLO member has a kind/suffix mismatch: {path}")
        stem = match.group("stem").casefold()
        split = match.group("split")
        records[(kind, split)].add(stem)
        if kind == "labels" and int(raw_entry.get("size", 0)) == 0:
            zero_byte_labels += 1
            if len(zero_byte_label_examples) < 20:
                zero_byte_label_examples.append(stem)

    intervals = _scenario_intervals(mapping)
    all_images = records[("images", "train")] | records[("images", "val")]
    excluded_stems = {
        str(value).casefold()
        for value in mapping.get(
            "excluded_image_stems", mapping.get("ambiguous_filename_stems", [])
        )
    }
    observed_excluded = all_images & excluded_stems
    eligible_images = all_images - excluded_stems
    assignments: dict[str, str] = {}
    unmapped = []
    multiply_mapped = []
    scenario_counts: Counter[str] = Counter()
    split_scenarios: dict[str, set[str]] = defaultdict(set)
    for stem in sorted(eligible_images):
        key = _filename_key(stem)
        matched = [scenario_id for start, end, scenario_id in intervals if start <= key <= end]
        if not matched:
            if len(unmapped) < 20:
                unmapped.append(stem)
            continue
        if len(matched) != 1:
            if len(multiply_mapped) < 20:
                multiply_mapped.append({"stem": stem, "scenarios": matched})
            continue
        assignments[stem] = matched[0]
        scenario_counts[matched[0]] += 1
    for split in ("train", "val"):
        split_scenarios[split] = {
            assignments[stem]
            for stem in records[("images", split)]
            if stem in assignments
        }

    image_counts = {split: len(records[("images", split)]) for split in ("train", "val")}
    label_counts = {split: len(records[("labels", split)]) for split in ("train", "val")}
    pairing = {
        split: {
            "image_label_ids_equal": records[("images", split)] == records[("labels", split)],
            "missing_label_count": len(records[("images", split)] - records[("labels", split)]),
            "orphan_label_count": len(records[("labels", split)] - records[("images", split)]),
        }
        for split in ("train", "val")
    }
    return {
        "archive_structure": {
            "coal_miner_yolo_file_count": member_count,
            "unexpected_file_count": max(0, member_count - sum(image_counts.values()) - sum(label_counts.values())),
            "unexpected_file_examples": unexpected_files,
            "casefold_duplicate_count": sum(
                count - 1 for count in casefold_paths.values() if count > 1
            ),
        },
        "yolo": {
            "image_counts": image_counts,
            "label_counts": label_counts,
            "total_image_count": sum(image_counts.values()),
            "total_label_count": sum(label_counts.values()),
            "expected_image_counts": EXPECTED_COUNTS,
            "counts_match_official_report": image_counts == EXPECTED_COUNTS,
            "pairing": pairing,
            "cross_split_image_overlap_count": len(
                records[("images", "train")] & records[("images", "val")]
            ),
            "zero_byte_label_count": zero_byte_labels,
            "zero_byte_label_examples": zero_byte_label_examples,
        },
        "scenario_mapping": {
            "declared_scenario_count": int(mapping.get("scenario_count", 0)),
            "observed_scenario_count": len(scenario_counts),
            "all_eligible_images_mapped_exactly_once": len(assignments) == len(eligible_images),
            "eligible_image_count": len(eligible_images),
            "declared_excluded_image_count": len(excluded_stems),
            "observed_excluded_image_count": len(observed_excluded),
            "observed_excluded_filename_stems": sorted(observed_excluded),
            "unmapped_image_count": len(eligible_images) - len(assignments) - len(multiply_mapped),
            "unmapped_examples": unmapped,
            "multiply_mapped_image_count": len(multiply_mapped),
            "multiply_mapped_examples": multiply_mapped,
            "image_counts_by_scenario": dict(sorted(scenario_counts.items())),
            "train_scenario_count": len(split_scenarios["train"]),
            "val_scenario_count": len(split_scenarios["val"]),
            "official_split_scenario_overlap_count": len(
                split_scenarios["train"] & split_scenarios["val"]
            ),
            "canonical_group_rule": "official_inclusive_filename_ranges_from_coal_miner_DsLMF.pdf",
        },
        "_observed_ids": {
            split: sorted(records[("images", split)]) for split in ("train", "val")
        },
    }


def audit_coco_file(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    categories = {
        int(item["id"]): str(item["name"]).strip()
        for item in payload.get("categories", [])
    }
    images = {int(item["id"]): item for item in payload.get("images", [])}
    stems = {
        Path(str(item.get("file_name", ""))).stem.casefold() for item in images.values()
    }
    invalid_boxes = 0
    unknown_image_annotations = 0
    unknown_category_annotations = 0
    category_counts: Counter[str] = Counter()
    max_boundary_overflow = 0.0
    bbox_tolerance_pixels = 0.01
    for annotation in payload.get("annotations", []):
        image_id = int(annotation.get("image_id", -1))
        category_id = int(annotation.get("category_id", -1))
        image = images.get(image_id)
        if image is None:
            unknown_image_annotations += 1
        if category_id not in categories:
            unknown_category_annotations += 1
        else:
            category_counts[categories[category_id]] += 1
        bbox = annotation.get("bbox")
        valid = isinstance(bbox, list) and len(bbox) == 4
        if valid:
            try:
                x, y, width, height = map(float, bbox)
                valid = width > 0 and height > 0 and x >= 0 and y >= 0
                if image is not None:
                    overflow = max(
                        0.0,
                        x + width - float(image["width"]),
                        y + height - float(image["height"]),
                    )
                    max_boundary_overflow = max(max_boundary_overflow, overflow)
                    valid = valid and overflow <= bbox_tolerance_pixels
            except (KeyError, TypeError, ValueError):
                valid = False
        if not valid:
            invalid_boxes += 1
    return {
        "image_count": len(images),
        "annotation_count": len(payload.get("annotations", [])),
        "categories": dict(sorted(categories.items())),
        "category_annotation_counts": dict(sorted(category_counts.items())),
        "invalid_bbox_count": invalid_boxes,
        "bbox_boundary_tolerance_pixels": bbox_tolerance_pixels,
        "max_bbox_boundary_overflow_pixels": max_boundary_overflow,
        "unknown_image_annotation_count": unknown_image_annotations,
        "unknown_category_annotation_count": unknown_category_annotations,
        "stems": stems,
    }


def audit_coco_files(
    train_path: Path,
    val_path: Path,
    observed_ids: Mapping[str, list[str]],
) -> dict[str, Any]:
    result = {}
    for split, path in (("train", train_path), ("val", val_path)):
        audit = audit_coco_file(path)
        stems = set(audit.pop("stems"))
        yolo_stems = set(observed_ids[split])
        sequential_names = bool(stems) and all(
            re.fullmatch(r"[0-9]{12}", stem) for stem in stems
        )
        audit["file_name_scheme"] = (
            "sequential_12_digit_conversion_ids" if sequential_names else "source_stems"
        )
        audit["direct_yolo_id_comparison_applicable"] = not sequential_names
        audit["yolo_image_ids_equal"] = None if sequential_names else stems == yolo_stems
        audit["missing_from_coco_count"] = (
            None if sequential_names else len(yolo_stems - stems)
        )
        audit["missing_from_yolo_count"] = (
            None if sequential_names else len(stems - yolo_stems)
        )
        result[split] = audit
    result["summary"] = {
        "total_image_count": result["train"]["image_count"]
        + result["val"]["image_count"],
        "yolo_total_image_count": len(observed_ids["train"]) + len(observed_ids["val"]),
        "total_image_count_equal": (
            result["train"]["image_count"] + result["val"]["image_count"]
            == len(observed_ids["train"]) + len(observed_ids["val"])
        ),
        "train_count_delta_from_yolo": result["train"]["image_count"]
        - len(observed_ids["train"]),
        "val_count_delta_from_yolo": result["val"]["image_count"]
        - len(observed_ids["val"]),
        "execution_truth": "yolo_original_names_and_splits",
    }
    return result


def audit(
    archive: Path,
    mapping_path: Path,
    coco_train: Path,
    coco_val: Path,
) -> dict[str, Any]:
    mapping = json.loads(mapping_path.read_text(encoding="utf-8-sig"))
    result = audit_entries(iter_7z_entries(archive), mapping)
    observed = result.pop("_observed_ids")
    result["coco"] = audit_coco_files(coco_train, coco_val, observed)
    return {
        "schema_version": "mining1.dslmfplus_archive_audit.v1",
        "archive": archive.as_posix(),
        "mapping_path": mapping_path.as_posix(),
        **result,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--coco-train", type=Path, required=True)
    parser.add_argument("--coco-val", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = audit(args.archive, args.mapping, args.coco_train, args.coco_val)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
