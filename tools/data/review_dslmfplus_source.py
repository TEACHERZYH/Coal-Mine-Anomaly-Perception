from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping
import zipfile

import pandas as pd

from mining1_exp.data.adapters import materialize_dataset, validate_adapter_contract
from mining1_exp.provenance import sha256_file


DATASET_ID = "dslmfplus_coal_miner_v1"
ARCHIVE_ID = "figshare-22654945-v1-file-40215118"
ARCHIVE_SHA256 = "685453ff0409c9dd241bcce716901948e441fd122b38262b5dc60c19ab199fa0"
ARCHIVE_MD5 = "71b41987bbc0a95a877fdff7c3fe9840"
ARCHIVE_BYTES = 24_421_851_538
EXPECTED_EXCLUSIONS = ["0010171", "0010172", "p12-1", "p12-2", "p63-1"]
EXPECTED_SAMPLE_RECORDS = {
    "train-0000001": "scenario-01",
    "train-0000853": "scenario-02",
    "train-0001507": "scenario-03",
}
EXPECTED_NEGATIVE_RECORD = "train-0001507"
ADAPTER_SUBDIR = "DsLMF/data2023_yolo/coal_miner_data2023_yolo"
IMAGE_PATTERN = (
    r"images/(?P<split>train|val)/"
    r"(?P<record>(?:[0-9]+|[pP][0-9]+-[0-9]+))[.]jpg"
)


class DsLMFSourceReviewError(ValueError):
    """Raised when the source cannot satisfy the frozen G1 review gates."""


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise DsLMFSourceReviewError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _project_relative(project_root: Path, path: Path) -> str:
    resolved_root = project_root.resolve()
    resolved = path.resolve()
    if resolved_root not in resolved.parents:
        raise DsLMFSourceReviewError(f"Evidence path escapes project root: {path}")
    return resolved.relative_to(resolved_root).as_posix()


def build_adapter_contract(mapping: Mapping[str, Any]) -> dict[str, Any]:
    if mapping.get("schema_version") != "mining1.dslmfplus_scenario_map.v1":
        raise DsLMFSourceReviewError("Unexpected DsLMF+ scenario-map schema")
    if mapping.get("scenario_count") != 58:
        raise DsLMFSourceReviewError("DsLMF+ scenario count must equal 58")
    ranges = mapping.get("adapter_ranges")
    if not isinstance(ranges, list) or len(ranges) != 113:
        raise DsLMFSourceReviewError("DsLMF+ adapter ranges must contain 113 entries")
    exclusions = sorted(str(value).casefold() for value in mapping.get("excluded_image_stems", []))
    if exclusions != sorted(EXPECTED_EXCLUSIONS):
        raise DsLMFSourceReviewError("DsLMF+ reviewed exclusion set changed")
    scenario_ids = {str(item.get("value", "")) for item in ranges if isinstance(item, Mapping)}
    expected_scenarios = {f"scenario-{index:02d}" for index in range(1, 59)}
    if scenario_ids != expected_scenarios:
        raise DsLMFSourceReviewError("DsLMF+ adapter ranges do not cover all 58 scenarios")
    contract = {
        "schema_version": 1,
        "kind": "yolo_detection",
        "archive_subdir": ADAPTER_SUBDIR,
        "record_id_rule": {
            "regex": {
                "source": "relative_path",
                "pattern": IMAGE_PATTERN,
                "template": "{split}-{record}",
            }
        },
        "raw_group_rule": {
            "filename_range_lookup": {
                "source": "relative_path",
                "pattern": IMAGE_PATTERN,
                "record_group": "record",
                "ranges": ranges,
            }
        },
        "modality_rule": {"constant": "visible"},
        "pair_id_rule": {"null": True},
        "sequence_id_rule": {"null": True},
        "timestamp_rule": {"null": True},
        "yolo": {
            "image_globs": ["images/**/*.jpg"],
            "image_root": "images",
            "label_root": "labels",
            "class_names": ["coal_miner"],
            "missing_label_policy": "error",
            "excluded_image_stems": EXPECTED_EXCLUSIONS,
        },
    }
    return validate_adapter_contract(contract)


def validate_source_evidence(
    *,
    integrity: Mapping[str, Any],
    audit: Mapping[str, Any],
    mapping: Mapping[str, Any],
    article: Mapping[str, Any],
    archive: Path,
) -> None:
    if integrity.get("status") != "pass":
        raise DsLMFSourceReviewError("Full archive integrity evidence is not pass")
    if integrity.get("observed_sha256") != ARCHIVE_SHA256:
        raise DsLMFSourceReviewError("Full archive SHA-256 does not match the reviewed value")
    if integrity.get("observed_md5") != ARCHIVE_MD5:
        raise DsLMFSourceReviewError("Full archive MD5 does not match the official value")
    seven_zip = integrity.get("seven_zip_integrity_test", {})
    if seven_zip.get("exit_code") != 0 or seven_zip.get("result") != "Everything is Ok":
        raise DsLMFSourceReviewError("Full 7z CRC test did not pass")
    if seven_zip.get("files") != 414_067:
        raise DsLMFSourceReviewError("Full archive file count changed")
    if not archive.is_file() or archive.stat().st_size != ARCHIVE_BYTES:
        raise DsLMFSourceReviewError("Full archive path or byte size is invalid")

    files = article.get("files")
    license_data = article.get("license")
    if (
        article.get("id") != 22_654_945
        or not isinstance(files, list)
        or len(files) != 1
        or files[0].get("id") != 40_215_118
        or files[0].get("supplied_md5") != ARCHIVE_MD5
        or files[0].get("size") != ARCHIVE_BYTES
        or not isinstance(license_data, Mapping)
        or license_data.get("name") != "CC0"
    ):
        raise DsLMFSourceReviewError("Official article identity, file, or license changed")

    validate_archive_audit(audit)
    build_adapter_contract(mapping)


def validate_archive_audit(audit: Mapping[str, Any]) -> None:
    """Validate annotation, grouping, and secondary metadata evidence only."""

    yolo = audit.get("yolo", {})
    scenario = audit.get("scenario_mapping", {})
    coco = audit.get("coco", {})
    if (
        yolo.get("total_image_count") != 30_704
        or yolo.get("total_label_count") != 30_704
        or yolo.get("counts_match_official_report") is not True
        or yolo.get("cross_split_image_overlap_count") != 0
        or yolo.get("zero_byte_label_count") != 13
    ):
        raise DsLMFSourceReviewError("YOLO counts, pairing, or negative-label evidence failed")
    pairing = yolo.get("pairing", {})
    if any(
        pairing.get(split, {}).get("image_label_ids_equal") is not True
        for split in ("train", "val")
    ):
        raise DsLMFSourceReviewError("YOLO image-label pairing is incomplete")
    if (
        scenario.get("declared_scenario_count") != 58
        or scenario.get("observed_scenario_count") != 58
        or scenario.get("eligible_image_count") != 30_699
        or scenario.get("all_eligible_images_mapped_exactly_once") is not True
        or scenario.get("unmapped_image_count") != 0
        or scenario.get("multiply_mapped_image_count") != 0
        or sorted(scenario.get("observed_excluded_filename_stems", []))
        != sorted(EXPECTED_EXCLUSIONS)
    ):
        raise DsLMFSourceReviewError("Official scenario mapping failed its coverage gate")
    if (
        coco.get("summary", {}).get("total_image_count_equal") is not True
        or coco.get("summary", {}).get("execution_truth")
        != "yolo_original_names_and_splits"
        or any(coco.get(split, {}).get("invalid_bbox_count") != 0 for split in ("train", "val"))
    ):
        raise DsLMFSourceReviewError("Secondary COCO metadata audit failed")


def _sample_member_stems(sample_archive: Path) -> tuple[set[str], set[str]]:
    with zipfile.ZipFile(sample_archive) as handle:
        if handle.testzip() is not None:
            raise DsLMFSourceReviewError("Minimal sample ZIP CRC failed")
        files = sorted(name for name in handle.namelist() if not name.endswith("/"))
    if len(files) != 16:
        raise DsLMFSourceReviewError("Minimal sample ZIP must contain exactly 16 files")
    images = {
        Path(name).stem.casefold()
        for name in files
        if "/images/" in name and name.lower().endswith(".jpg")
    }
    labels = {
        Path(name).stem.casefold()
        for name in files
        if "/labels/" in name and name.lower().endswith(".txt")
    }
    if images != labels:
        raise DsLMFSourceReviewError("Minimal sample image-label stems differ")
    expected = {record.split("-", 1)[1] for record in EXPECTED_SAMPLE_RECORDS}
    expected.update(EXPECTED_EXCLUSIONS)
    if images != expected:
        raise DsLMFSourceReviewError("Minimal sample membership changed")
    return images, labels


def run_minimal_adapter(
    *,
    sample_archive: Path,
    contract: Mapping[str, Any],
    output_manifest: Path,
) -> dict[str, Any]:
    _sample_member_stems(sample_archive)
    with tempfile.TemporaryDirectory(prefix="dslmf-review-") as work:
        frame = materialize_dataset(
            project_root=Path(work),
            dataset_id="dslmfplus_coal_miner_review_sample_v1",
            archive_id=ARCHIVE_ID,
            archive_path=sample_archive,
            contract=contract,
        )
        frame = frame.sort_values("record_id").reset_index(drop=True)
        records = frame.where(pd.notna(frame), None).to_dict(orient="records")
    observed = dict(zip(frame["record_id"], frame["raw_group_id"]))
    if observed != EXPECTED_SAMPLE_RECORDS:
        raise DsLMFSourceReviewError("Minimal adapter record-to-scenario mapping changed")
    label_summaries = {
        row["record_id"]: json.loads(row["label_summary_json"]) for row in records
    }
    negative = label_summaries[EXPECTED_NEGATIVE_RECORD]
    if negative.get("boxes") != [] or negative.get("negative_annotation_verified") is not True:
        raise DsLMFSourceReviewError("Zero-byte negative-label semantics are not preserved")
    for record_id in set(EXPECTED_SAMPLE_RECORDS) - {EXPECTED_NEGATIVE_RECORD}:
        boxes = label_summaries[record_id].get("boxes", [])
        if not boxes or {box.get("source_class_id") for box in boxes} != {0}:
            raise DsLMFSourceReviewError("Positive sample class mapping changed")
    manifest_payload = {
        "schema_version": "mining1.dslmfplus_minimal_manifest.v1",
        "records": records,
    }
    _write_json(output_manifest, manifest_payload)
    return {
        "status": "pass",
        "sample_archive_path": sample_archive.as_posix(),
        "sample_archive_sha256": sha256_file(sample_archive),
        "sample_archive_bytes": sample_archive.stat().st_size,
        "sample_member_count": 16,
        "record_count": len(records),
        "record_ids": sorted(observed),
        "raw_group_ids": sorted(set(observed.values())),
        "negative_record_ids": [EXPECTED_NEGATIVE_RECORD],
        "reviewed_excluded_image_stems": EXPECTED_EXCLUSIONS,
        "manifest_path": output_manifest.as_posix(),
        "manifest_sha256": sha256_file(output_manifest),
    }


def review_source(
    *,
    project_root: Path,
    archive: Path,
    sample_archive: Path,
    integrity_path: Path,
    mapping_path: Path,
    audit_path: Path,
    article_path: Path,
    output_review: Path,
    output_test: Path,
    output_manifest: Path,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    archive = archive.resolve()
    sample_archive = sample_archive.resolve()
    integrity = _load_json(integrity_path)
    mapping = _load_json(mapping_path)
    audit = _load_json(audit_path)
    article = _load_json(article_path)
    validate_source_evidence(
        integrity=integrity,
        audit=audit,
        mapping=mapping,
        article=article,
        archive=archive,
    )
    contract = build_adapter_contract(mapping)
    minimal = run_minimal_adapter(
        sample_archive=sample_archive,
        contract=contract,
        output_manifest=output_manifest,
    )
    for key in ("sample_archive_path", "manifest_path"):
        minimal[key] = _project_relative(project_root, Path(minimal[key]))
    _write_json(output_test, minimal)

    review = {
        "schema_version": 1,
        "reviewed_at": datetime.now().astimezone().isoformat(),
        "status": "pass",
        "dataset_entry": {
            "dataset_id": DATASET_ID,
            "dataset_version": ARCHIVE_ID,
            "source_location": "local_existing_verified_official_download",
            "source_path_or_url": archive.as_posix(),
            "archive_sha256": ARCHIVE_SHA256,
            "archive_md5": ARCHIVE_MD5,
            "license_id": "CC0-1.0",
            "license_evidence_path": _project_relative(project_root, article_path),
            "transfer_route": "local_existing_upload",
            "selection_evidence": (
                "Selected only from official CC0 identity, full archive integrity, "
                "verified coal-miner ontology, 58 official scene groups, and a real "
                "minimal adapter run; no model predictions or performance were read."
            ),
            "pretraining_conditions": ["single_coal", "multi_coal"],
            "adapter_contract": contract,
            "minimal_sample_records": [
                {
                    "record_id": record_id,
                    "raw_group_id": group_id,
                    "member_path": (
                        f"{ADAPTER_SUBDIR}/images/train/{record_id.split('-', 1)[1]}.jpg"
                    ),
                    "modality": "visible",
                }
                for record_id, group_id in EXPECTED_SAMPLE_RECORDS.items()
            ],
        },
        "official_source": {
            "article_id": 22_654_945,
            "doi": "10.6084/m9.figshare.22654945.v1",
            "file_id": 40_215_118,
            "file_name": "DsLMF.7z",
            "file_bytes": ARCHIVE_BYTES,
            "official_md5": ARCHIVE_MD5,
            "license_name": "CC0",
            "article_snapshot_sha256": sha256_file(article_path),
        },
        "archive_integrity": {
            "status": "pass",
            "evidence_path": _project_relative(project_root, integrity_path),
            "evidence_sha256": sha256_file(integrity_path),
            "seven_zip_crc": "pass",
            "archive_file_count": 414_067,
            "full_local_extractions": 0,
        },
        "annotation_and_group_audit": {
            "status": "pass",
            "evidence_path": _project_relative(project_root, audit_path),
            "evidence_sha256": sha256_file(audit_path),
            "yolo_image_label_pairs": 30_704,
            "verified_negative_label_count": 13,
            "official_scene_group_count": 58,
            "eligible_image_count": 30_699,
            "reviewed_excluded_image_stems": EXPECTED_EXCLUSIONS,
            "official_train_val_scene_overlap_count": 56,
            "canonical_split_requirement": "ignore_official_random_split_and_split_by_official_scene_group",
            "mapping_path": _project_relative(project_root, mapping_path),
            "mapping_sha256": sha256_file(mapping_path),
            "mapping_pdf_path": str(mapping.get("source_pdf", "")),
            "mapping_pdf_sha256": str(mapping.get("source_pdf_sha256", "")),
            "secondary_coco_execution_truth": "yolo_original_names_and_splits",
        },
        "minimal_adapter_test": {
            **minimal,
            "evidence_path": _project_relative(project_root, output_test),
            "evidence_sha256": sha256_file(output_test),
        },
        "eligibility_decision": {
            "license_access": "pass_cc0",
            "archive_integrity": "pass",
            "annotation_and_ontology": "pass_with_five_reviewed_filename_drops",
            "independent_group_floor": "pass",
            "single_coal_source": "eligible",
            "confirmatory_multi_coal_source": "eligible",
        },
        "scope_boundary": {
            "full_local_extractions": 0,
            "minimal_local_extracted_file_count": 16,
            "minimal_local_extracted_bytes": 475_844,
            "model_predictions_read": 0,
            "performance_metrics_read": 0,
            "performance_claims_authorized": False,
        },
    }
    _write_json(output_review, review)
    return review


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--sample-archive", type=Path, required=True)
    parser.add_argument("--integrity", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--article", type=Path, required=True)
    parser.add_argument("--output-review", type=Path, required=True)
    parser.add_argument("--output-test", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    review_source(
        project_root=args.project_root,
        archive=args.archive,
        sample_archive=args.sample_archive,
        integrity_path=args.integrity,
        mapping_path=args.mapping,
        audit_path=args.audit,
        article_path=args.article,
        output_review=args.output_review,
        output_test=args.output_test,
        output_manifest=args.output_manifest,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
