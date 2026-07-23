from __future__ import annotations

import json
from pathlib import Path
import shutil

from mining1_exp.workflow_spec import (
    dry_run_matrix,
    validate_artifact_contract,
    validate_protocol,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _spec_root(tmp_path: Path) -> Path:
    configs = tmp_path / "configs"
    configs.mkdir()
    for name in (
        "protocol_lock.template.yaml",
        "artifact_contract.template.yaml",
        "experiment_matrix.template.csv",
        "family_step_crosswalk.csv",
    ):
        shutil.copy2(PROJECT_ROOT / "configs" / name, configs / name)
    return tmp_path


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def test_static_workflow_handlers_validate_the_frozen_specifications(
    tmp_path: Path,
) -> None:
    root = _spec_root(tmp_path)
    protocol = validate_protocol(root, {"step_id": "E000"}, {"template": True})
    artifact = validate_artifact_contract(root, {"step_id": "E002"}, {})
    matrix = dry_run_matrix(root, {"step_id": "E004"}, {})
    assert protocol["status"] == artifact["status"] == matrix["status"] == "pass"

    protocol_payload = _load(root / "evidence/gates/protocol_template_validation.json")
    artifact_payload = _load(root / "evidence/gates/artifact_contract_validation.json")
    matrix_payload = _load(root / "evidence/gates/minimal_matrix_dry_run.json")
    assert protocol_payload["endpoint_count"] == 8
    assert protocol_payload["performance_claims_authorized"] is False
    assert artifact_payload["artifact_definition_count"] == 54
    assert matrix_payload["family_count"] == 25
    assert matrix_payload["trained_or_fitted_run_count"] <= 39
