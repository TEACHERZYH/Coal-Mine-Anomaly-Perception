from __future__ import annotations

import csv
from pathlib import Path
import re
from typing import Any, Dict, Mapping

import yaml

from .provenance import sha256_file
from .workflow_common import WorkflowExecutionError, write_json_artifact


REQUIRED_PROTOCOL_SECTIONS = {
    "clean_room",
    "confirmatory_endpoints",
    "data",
    "efficiency",
    "environment",
    "episodes",
    "evaluation",
    "evidence",
    "locks",
    "models",
    "pilot",
    "project",
    "remote_execution",
    "resources",
    "seeds",
    "statistics",
    "training",
}
EXPECTED_ENDPOINTS = {
    "V2",
    "T1",
    "S1",
    "E2_E3",
    "E3_RELIABILITY",
    "E3_GRAPH",
    "E3_MEMORY",
    "R1",
}
EXPECTED_ARTIFACT_NAMES = {
    "source_registry",
    "archive_manifest",
    "file_manifest",
    "ontology_lock",
    "split_manifest",
    "fewshot_manifest",
    "graph_eligibility",
    "fusion_eligibility",
    "test_seal",
    "episode_skeleton_manifest",
    "episode_predictions",
    "run_bundle",
    "step_review",
    "environment_ready",
    "prediction_lock",
    "test_release",
    "metrics_bundle",
    "gate",
    "efficiency_trace",
    "evidence_package",
}


def _load_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise WorkflowExecutionError(f"Expected a YAML mapping: {path}")
    return payload


def _count_tbd(value: Any) -> int:
    if isinstance(value, dict):
        return sum(_count_tbd(item) for item in value.values())
    if isinstance(value, list):
        return sum(_count_tbd(item) for item in value)
    return int(isinstance(value, str) and "TBD" in value)


def validate_protocol(
    root: Path,
    row: Mapping[str, str],
    arguments: Mapping[str, Any],
) -> Dict[str, Any]:
    if arguments.get("template") is not True:
        raise WorkflowExecutionError("E000 requires --template")
    source = root / "configs/protocol_lock.template.yaml"
    protocol = _load_yaml(source)
    missing = sorted(REQUIRED_PROTOCOL_SECTIONS - set(protocol))
    if missing:
        raise WorkflowExecutionError(f"Protocol sections are missing: {missing}")
    endpoints = protocol.get("confirmatory_endpoints")
    if not isinstance(endpoints, dict) or set(endpoints) != EXPECTED_ENDPOINTS:
        raise WorkflowExecutionError("Confirmatory endpoint definitions do not match minimal_v2")
    if protocol.get("status") != "template_not_locked" or protocol.get("schema_version") != 3:
        raise WorkflowExecutionError("Protocol template identity is invalid")
    text = source.read_text(encoding="utf-8-sig")
    if re.search(r"F:[\\/]2026[\\/]mining(?:[\\/]|$)", text, re.I):
        raise WorkflowExecutionError("Protocol template contains the old project path")
    output = root / "evidence/gates/protocol_template_validation.json"
    payload = {
        "schema_version": 1,
        "step_id": row["step_id"],
        "status": "pass",
        "protocol_schema_version": protocol["schema_version"],
        "protocol_status": protocol["status"],
        "required_section_count": len(REQUIRED_PROTOCOL_SECTIONS),
        "endpoint_count": len(endpoints),
        "endpoint_ids": sorted(endpoints),
        "template_placeholder_count": _count_tbd(protocol),
        "template_sha256": sha256_file(source),
        "performance_claims_authorized": False,
    }
    write_json_artifact(output, payload)
    return {
        "status": "pass",
        "output_paths": [output.relative_to(root).as_posix()],
        "inputs": [{"path": source.relative_to(root).as_posix(), "sha256": sha256_file(source)}],
        "details": {"endpoint_count": len(endpoints)},
    }


def validate_artifact_contract(
    root: Path,
    row: Mapping[str, str],
    arguments: Mapping[str, Any],
) -> Dict[str, Any]:
    del arguments
    source = root / "configs/artifact_contract.template.yaml"
    contract = _load_yaml(source)
    artifacts = contract.get("artifacts")
    if not isinstance(artifacts, dict):
        raise WorkflowExecutionError("Artifact contract lacks an artifacts mapping")
    missing = sorted(EXPECTED_ARTIFACT_NAMES - set(artifacts))
    if missing:
        raise WorkflowExecutionError(f"Artifact definitions are missing: {missing}")
    if contract.get("schema_version") != 2 or contract.get("status") != "template_not_locked":
        raise WorkflowExecutionError("Artifact contract template identity is invalid")
    required_fields = artifacts.get("step_review", {}).get("required_fields", {})
    for field in (
        "step_id",
        "attempt",
        "status",
        "reviewed_at",
        "output_artifacts",
        "acceptance_checks",
        "findings",
        "advance_allowed",
    ):
        if field not in required_fields:
            raise WorkflowExecutionError(f"Step-review contract lacks {field}")
    output = root / "evidence/gates/artifact_contract_validation.json"
    payload = {
        "schema_version": 1,
        "step_id": row["step_id"],
        "status": "pass",
        "contract_schema_version": contract["schema_version"],
        "artifact_definition_count": len(artifacts),
        "required_core_artifacts": sorted(EXPECTED_ARTIFACT_NAMES),
        "step_review_required_fields": sorted(required_fields),
        "contract_sha256": sha256_file(source),
        "performance_claims_authorized": False,
    }
    write_json_artifact(output, payload)
    return {
        "status": "pass",
        "output_paths": [output.relative_to(root).as_posix()],
        "inputs": [{"path": source.relative_to(root).as_posix(), "sha256": sha256_file(source)}],
        "details": {"artifact_definition_count": len(artifacts)},
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def dry_run_matrix(
    root: Path,
    row: Mapping[str, str],
    arguments: Mapping[str, Any],
) -> Dict[str, Any]:
    del arguments
    matrix_path = root / "configs/experiment_matrix.template.csv"
    crosswalk_path = root / "configs/family_step_crosswalk.csv"
    matrix = _read_csv(matrix_path)
    crosswalk = _read_csv(crosswalk_path)
    families = [item["family_id"] for item in matrix]
    if len(matrix) != 25 or len(set(families)) != 25:
        raise WorkflowExecutionError("Minimal matrix must contain 25 unique families")
    trained = sum(int(item["planned_trained_runs"]) for item in matrix)
    if trained > 39:
        raise WorkflowExecutionError("Minimal matrix exceeds 39 trained or fitted runs")
    if {item["family_id"] for item in crosswalk} != set(families):
        raise WorkflowExecutionError("Family-to-step crosswalk does not cover the matrix exactly")
    conditional = {
        "E2-MEAN",
        "E2-LOGIT",
        "E3-SHUFFLE",
        "E3-NOREL",
        "E3-NOGRAPH",
        "E3-NOMEM",
        "E3-FULL",
        "C1-EFF",
    }
    for item in crosswalk:
        if item["family_id"] in conditional and "conditional" not in item["acceptance_scope"]:
            if item["family_id"] != "C1-EFF":
                raise WorkflowExecutionError(
                    f"Conditional family lacks an explicit branch: {item['family_id']}"
                )
    output = root / "evidence/gates/minimal_matrix_dry_run.json"
    payload = {
        "schema_version": 1,
        "step_id": row["step_id"],
        "status": "pass",
        "family_count": len(matrix),
        "trained_or_fitted_run_count": trained,
        "conditional_family_ids": sorted(conditional),
        "crosswalk_family_count": len(crosswalk),
        "matrix_sha256": sha256_file(matrix_path),
        "crosswalk_sha256": sha256_file(crosswalk_path),
        "performance_claims_authorized": False,
    }
    write_json_artifact(output, payload)
    return {
        "status": "pass",
        "output_paths": [output.relative_to(root).as_posix()],
        "inputs": [
            {"path": matrix_path.relative_to(root).as_posix(), "sha256": sha256_file(matrix_path)},
            {
                "path": crosswalk_path.relative_to(root).as_posix(),
                "sha256": sha256_file(crosswalk_path),
            },
        ],
        "details": {"family_count": len(matrix), "trained_or_fitted_runs": trained},
    }
