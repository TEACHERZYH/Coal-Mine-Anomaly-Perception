from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

import mining1_exp.implementation_review as review_module
from mining1_exp.implementation_review import (
    ImplementationReviewError,
    collect_implementation_evidence,
    implementation_source_scope,
    scan_implementation_residue,
    validate_implementation_review,
)
from mining1_exp.provenance import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_current_implementation_evidence_and_residue_are_reviewable() -> None:
    evidence = collect_implementation_evidence(PROJECT_ROOT)
    assert evidence["status"] == "pass"
    assert [item["step_id"] for item in evidence["steps"]] == [
        "I000",
        "I020",
        "I040",
        "I050",
        "I070",
        "I080",
    ]
    residue = scan_implementation_residue(PROJECT_ROOT)
    assert residue["status"] == "pass", residue["errors"]
    assert residue["scanned_file_count"] >= 40
    scope = implementation_source_scope(PROJECT_ROOT)
    assert scope["status"] == "pass"
    assert scope["file_count"] >= residue["scanned_file_count"]
    assert scope["plan_step_count"] == 78
    assert len(scope["scope_sha256"]) == 64
    assert len(scope["plan_contract_sha256"]) == 64


def test_residue_scan_rejects_removed_code_old_paths_and_test_pool_fit(
    tmp_path: Path,
) -> None:
    source = tmp_path / "mining1_exp" / "models" / "bad.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
import sys
sys.path.append(r'F:\\2026\\mining\\legacy')
REMOVED = 'yolov10'
class MethaneMLP:
    pass
import legacy_project_module
model.fit(values, pool='D_e_te')
""".strip()
        + "\n",
        encoding="ascii",
    )
    residue = scan_implementation_residue(tmp_path)
    assert residue["status"] == "fail"
    assert any("old project path" in error for error in residue["errors"])
    assert any("removed implementation yolov10" in error for error in residue["errors"])
    assert any("removed implementation class MethaneMLP" in error for error in residue["errors"])
    assert any("sys.path mutation" in error for error in residue["errors"])
    assert any(
        "unapproved import root legacy_project_module" in error
        for error in residue["errors"]
    )
    assert any("test-pool fit or selection" in error for error in residue["errors"])


def _valid_review_payload() -> dict:
    return {
        "schema_version": 1,
        "step_id": "I090",
        "status": "pass",
        "reviewer": "test",
        "reviewed_at": "2026-07-15T00:00:00+00:00",
        "implementation_evidence": {"status": "pass"},
        "residue_scan": {"status": "pass"},
        "static_plan_validation": {
            "status": "pass",
            "counts": {
                "families": 25,
                "trained_or_fitted_runs": 39,
                "steps": 78,
                "claims": 9,
            },
        },
        "test_suite": {"status": "pass", "passed": 1},
        "source_scope": {
            "status": "pass",
            "file_count": 1,
            "scope_sha256": "0" * 64,
            "manifest": [{"path": "x", "sha256": "1" * 64}],
            "plan_step_count": 78,
            "plan_contract_sha256": "2" * 64,
        },
        "acceptance_checks": [{"id": "review", "status": "pass"}],
        "scope_boundary": {
            "full_local_dataset_extractions": 0,
            "remote_connections": 0,
            "slurm_jobs_created": 0,
            "performance_claims_authorized": False,
        },
        "failure_reference": ["evidence/implementation/I090_attempt_01_rejected.json"],
        "findings": ["reviewed"],
        "advance_allowed": True,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("advance_allowed", False),
        ("performance_claims_authorized", True),
    ),
)
def test_review_validation_fails_closed_on_decision_boundaries(
    field: str,
    value: bool,
) -> None:
    payload = deepcopy(_valid_review_payload())
    if field == "performance_claims_authorized":
        payload["scope_boundary"][field] = value
    else:
        payload[field] = value
    with pytest.raises(ImplementationReviewError):
        validate_implementation_review(payload)


def test_review_creation_reports_the_write_once_artifact_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = {
        "status": "pass",
        "steps": [{"step_id": step_id} for step_id in review_module.IMPLEMENTATION_STEPS],
    }
    scope = deepcopy(_valid_review_payload()["source_scope"])
    monkeypatch.setattr(review_module, "collect_implementation_evidence", lambda _: evidence)
    monkeypatch.setattr(
        review_module,
        "scan_implementation_residue",
        lambda _: {"status": "pass", "scanned_file_count": 1, "errors": []},
    )
    monkeypatch.setattr(review_module, "implementation_source_scope", lambda _: scope)
    monkeypatch.setattr(
        review_module,
        "_run_json_command",
        lambda *_: {
            "status": "pass",
            "check_count": 1,
            "counts": {
                "families": 25,
                "trained_or_fitted_runs": 39,
                "steps": 78,
                "claims": 9,
            },
        },
    )
    monkeypatch.setattr(
        review_module,
        "run_full_test_suite",
        lambda _: {
            "status": "pass",
            "passed": 1,
            "runtime_seconds": 0.0,
            "returncode": 0,
            "summary_line": "1 passed in 0.00s",
        },
    )
    output = tmp_path / "I090.json"
    result = review_module.run_implementation_review(PROJECT_ROOT, output)
    assert result["mode"] == "created"
    assert result["review_sha256"] == sha256_file(output)
