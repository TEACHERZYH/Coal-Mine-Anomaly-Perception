from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

from mining1_exp.provenance import sha256_file


SELECTED_REVIEWS = {
    "primary_visual_target": "evidence/data/source_reviews/dsdpm66_coal_miner_v1.json",
    "generic_visual_source": "evidence/data/source_reviews/coco2017_person.json",
    "coal_visual_source": "evidence/data/source_reviews/dslmfplus_coal_miner_v1.json",
    "primary_rgbt_dataset": "evidence/data/source_reviews/llvip_v1.json",
    "methane_dataset": "evidence/data/source_reviews/mendeley_methane_v1.json",
}
TRANSFER_CANDIDATE_REVIEWS = [
    "evidence/data/source_reviews/dslmfplus_coal_miner_v1.json",
    "evidence/data/source_reviews/sciencedb_drilling_v1.json",
    "evidence/data/source_reviews/cumt_helmet_v1.json",
    "evidence/data/source_reviews/cumt_belt_v1.json",
]
EXPECTED_DATASET_IDS = {
    "primary_visual_target": "dsdpm66_coal_miner_v1",
    "generic_visual_source": "coco2017_person",
    "coal_visual_source": "dslmfplus_coal_miner_v1",
    "primary_rgbt_dataset": "llvip_v1",
    "methane_dataset": "mendeley_methane_v1",
}
AUTHORITY_RELATIVE = "notes/minimal_experiment_plan_rereview_20260715.md"


class ReviewedSourceBuildError(ValueError):
    """Raised when reviewed source evidence is incomplete or inconsistent."""


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ReviewedSourceBuildError(f"Review root must be an object: {path}")
    return payload


def _selected_entry(role: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("status") != "pass":
        raise ReviewedSourceBuildError(f"Selected source review is not pass: {role}")
    entry = payload.get("dataset_entry")
    if not isinstance(entry, Mapping) or entry.get("dataset_id") != EXPECTED_DATASET_IDS[role]:
        raise ReviewedSourceBuildError(f"Selected source identity changed: {role}")
    return json.loads(json.dumps(entry))


def build_reviewed_payload(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    selected = {
        role: _load(project_root / relative)
        for role, relative in SELECTED_REVIEWS.items()
    }
    target = _selected_entry("primary_visual_target", selected["primary_visual_target"])
    generic = _selected_entry("generic_visual_source", selected["generic_visual_source"])
    coal = _selected_entry("coal_visual_source", selected["coal_visual_source"])
    rgbt = _selected_entry("primary_rgbt_dataset", selected["primary_rgbt_dataset"])
    methane = _selected_entry("methane_dataset", selected["methane_dataset"])
    if generic.get("pretraining_conditions") != ["generic_matched"]:
        raise ReviewedSourceBuildError("Generic source pretraining role changed")
    if coal.get("pretraining_conditions") != ["single_coal", "multi_coal"]:
        raise ReviewedSourceBuildError("Coal source pretraining roles changed")
    if (
        selected["generic_visual_source"].get("eligibility_decision", {}).get(
            "generic_matched_source"
        )
        != "eligible"
    ):
        raise ReviewedSourceBuildError("COCO generic-source eligibility is not pass")
    if (
        selected["coal_visual_source"].get("eligibility_decision", {}).get(
            "confirmatory_multi_coal_source"
        )
        != "eligible"
    ):
        raise ReviewedSourceBuildError("DsLMF+ coal-source eligibility is not pass")
    if (
        selected["primary_rgbt_dataset"].get("eligibility_decision", {}).get(
            "primary_rgbt_dataset"
        )
        != "eligible"
    ):
        raise ReviewedSourceBuildError("LLVIP RGB-T eligibility is not pass")

    candidate_dispositions = {}
    review_inputs = []
    for relative in sorted(set(SELECTED_REVIEWS.values()) | set(TRANSFER_CANDIDATE_REVIEWS)):
        path = project_root / relative
        review = _load(path)
        dataset_id = str(review.get("dataset_entry", {}).get("dataset_id", ""))
        if not dataset_id:
            raise ReviewedSourceBuildError(f"Candidate review lacks dataset identity: {relative}")
        review_inputs.append(
            {"path": relative, "sha256": sha256_file(path), "dataset_id": dataset_id}
        )
        if relative in TRANSFER_CANDIDATE_REVIEWS:
            disposition = review.get("eligibility_decision", {}).get(
                "confirmatory_multi_coal_source"
            )
            if disposition not in {"eligible", "rejected"}:
                raise ReviewedSourceBuildError(
                    f"Transfer candidate disposition is incomplete: {dataset_id}"
                )
            candidate_dispositions[dataset_id] = disposition
    eligible_coal = sorted(
        dataset_id
        for dataset_id, disposition in candidate_dispositions.items()
        if disposition == "eligible"
    )
    if eligible_coal != ["dslmfplus_coal_miner_v1"]:
        raise ReviewedSourceBuildError("Eligible non-target coal-source set changed")

    authority = project_root / AUTHORITY_RELATIVE
    authority_text = authority.read_text(encoding="utf-8-sig")
    if "C-TRANSFER" not in authority_text or "PLAN_REVIEW_PASS" not in authority_text:
        raise ReviewedSourceBuildError("Fallback authority is not the reviewed plan decision")
    return {
        "schema_version": 1,
        "status": "reviewed",
        "reviewed_at": datetime.now().astimezone().isoformat(),
        "visual_direction_id": "dsdpm66_from_public_pretraining_v1",
        "roles": {
            "primary_visual_target": target,
            "primary_visual_sources": [generic, coal],
            "primary_rgbt_dataset": rgbt,
            "methane_dataset": methane,
        },
        "claim_fallbacks": {
            "C-TRANSFER": {
                "status": "not_applicable",
                "reason_code": "fewer_than_two_eligible_non_target_coal_sources",
                "eligible_non_target_coal_dataset_ids": eligible_coal,
                "candidate_review_paths": TRANSFER_CANDIDATE_REVIEWS,
                "authority_path": AUTHORITY_RELATIVE,
                "authority_sha256": sha256_file(authority),
            }
        },
        "review_inputs": review_inputs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_reviewed_payload(args.project_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
