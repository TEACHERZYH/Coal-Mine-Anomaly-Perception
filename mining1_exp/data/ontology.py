from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

import yaml


PathLike = Union[str, Path]
MAPPING_STATUSES = {"compatible", "ignored", "unmapped", "reject"}
EVENT_SEMANTICS = {
    "observable_presence",
    "annotated_anomaly",
    "threshold_risk",
    "not_event_eligible",
}
NEGATIVE_SEMANTICS = {
    "exhaustive_verified_absence",
    "unverified_missing_annotation",
    "not_applicable",
}
ONTOLOGY_ENTRY_FIELDS = {
    "dataset_id",
    "source_label",
    "canonical_concept_id",
    "mapping_status",
    "annotation_policy",
    "event_semantics",
    "negative_semantics",
    "allowed_tasks",
    "evidence_reference",
    "reviewer_decision",
}


class OntologyValidationError(ValueError):
    """Raised when an ontology decision lacks annotation-level evidence."""


def load_ontology_lock(path: PathLike) -> Dict[str, Any]:
    lock_path = Path(path)
    with lock_path.open("r", encoding="utf-8-sig") as handle:
        payload = yaml.safe_load(handle)
    return validate_ontology_lock(payload)


def validate_ontology_lock(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise OntologyValidationError("ontology lock root must be a mapping")
    version = payload.get("ontology_version")
    entries = payload.get("entries")
    if not isinstance(version, str) or not version.strip():
        raise OntologyValidationError("ontology_version is required")
    if not isinstance(entries, list) or not entries:
        raise OntologyValidationError("ontology entries must be a non-empty list")

    keys = set()
    validated: List[Dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict) or not ONTOLOGY_ENTRY_FIELDS.issubset(entry):
            raise OntologyValidationError("ontology entry is missing required fields")
        key = (version, entry["dataset_id"], entry["source_label"])
        if key in keys:
            raise OntologyValidationError(f"duplicate ontology key: {key}")
        keys.add(key)
        if entry["mapping_status"] not in MAPPING_STATUSES:
            raise OntologyValidationError("unsupported mapping_status")
        if entry["event_semantics"] not in EVENT_SEMANTICS:
            raise OntologyValidationError("unsupported event_semantics")
        if entry["negative_semantics"] not in NEGATIVE_SEMANTICS:
            raise OntologyValidationError("unsupported negative_semantics")
        if not isinstance(entry["allowed_tasks"], list):
            raise OntologyValidationError("allowed_tasks must be a list")
        if entry["mapping_status"] == "compatible":
            if not str(entry["canonical_concept_id"] or "").strip():
                raise OntologyValidationError("compatible mappings require a concept ID")
            if not str(entry["annotation_policy"]).strip():
                raise OntologyValidationError("compatible mappings require annotation policy")
            if not str(entry["evidence_reference"]).strip():
                raise OntologyValidationError("name-only ontology mapping is forbidden")
        if "confirmatory_f1" in entry["allowed_tasks"] and entry[
            "negative_semantics"
        ] != "exhaustive_verified_absence":
            raise OntologyValidationError(
                "confirmatory_f1 requires verified negative semantics"
            )
        validated.append(dict(entry))
    return {"ontology_version": version, "entries": validated}
