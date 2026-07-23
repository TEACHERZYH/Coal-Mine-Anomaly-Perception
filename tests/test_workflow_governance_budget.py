from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from mining1_exp.workflow_common import load_json
from mining1_exp.workflow_governance import (
    _build_family_resource_budget,
    _experiment_steps_contract_sha256,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolved_protocol() -> dict:
    protocol = yaml.safe_load(
        (PROJECT_ROOT / "configs/protocol_lock.template.yaml").read_text(
            encoding="utf-8-sig"
        )
    )
    protocol["training"]["visible"]["max_updates"] = 10000
    protocol["training"]["rgbt"]["max_updates"] = 5000
    protocol["training"]["methane"]["max_updates"] = 5000
    protocol["training"]["episode"]["max_updates"] = 5000
    protocol["training"]["matched_pretraining"]["optimizer_updates"] = 10000
    return protocol


def _pilot() -> dict:
    return load_json(PROJECT_ROOT / "evidence/pilot/resource_pilot.json")


def _write_step_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_resource_budget_covers_frozen_minimal_matrix() -> None:
    rows, summaries = _build_family_resource_budget(
        PROJECT_ROOT, _resolved_protocol(), _pilot()
    )
    assert len(rows) == 23
    assert sum(row["run_count"] for row in rows) == 39
    assert {summary["package_id"] for summary in summaries} == {
        "V2",
        "T1",
        "S1",
        "E2_E3",
        "R1",
        "C1",
    }
    required = {
        "family_id",
        "run_count",
        "pilot_seconds_per_update",
        "projected_gpu_hours",
        "projected_storage_gib",
        "priority",
        "cap_decision",
    }
    assert all(required.issubset(row) for row in rows)
    assert all(summary["cap_decision"] == "within_frozen_package_cap" for summary in summaries)
    assert all(summary["projected_gpu_hours"] <= summary["gpu_hour_cap"] for summary in summaries)


def test_resource_budget_keeps_cpu_and_reuse_families_explicit() -> None:
    rows, _ = _build_family_resource_budget(
        PROJECT_ROOT, _resolved_protocol(), _pilot()
    )
    by_family = {row["family_id"]: row for row in rows}
    assert by_family["S1-HGB"]["execution_resource"] == "cpu_fit"
    assert by_family["S1-HGB"]["projected_gpu_hours"] == 0.0
    assert by_family["E2-LOGIT"]["execution_resource"] == "cpu_fit"
    assert by_family["R1-GEN"]["execution_resource"] == "reuse_or_nontraining"
    assert by_family["C1-EFF"]["run_count"] == 0


def test_resource_budget_requires_selected_precision_benchmark() -> None:
    pilot = _pilot()
    pilot["benchmarks"] = [
        row for row in pilot["benchmarks"] if row["family"] != "episode"
    ]
    with pytest.raises(RuntimeError, match="missing for episode"):
        _build_family_resource_budget(PROJECT_ROOT, _resolved_protocol(), pilot)


def test_resource_budget_rejects_projection_over_package_cap() -> None:
    pilot = deepcopy(_pilot())
    pilot["resource_cap_basis"]["package_caps_gpu_hours"]["V2"] = 0.001
    with pytest.raises(RuntimeError, match="exceeds the frozen GPU-hour cap for V2"):
        _build_family_resource_budget(PROJECT_ROOT, _resolved_protocol(), pilot)


def test_experiment_steps_contract_hash_ignores_mutable_status(
    tmp_path: Path,
) -> None:
    source = PROJECT_ROOT / "plans/experiment_steps.csv"
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    original = tmp_path / "original.csv"
    changed = tmp_path / "status_changed.csv"
    _write_step_rows(original, rows)
    changed_rows = deepcopy(rows)
    changed_rows[0]["status"] = "pass"
    _write_step_rows(changed, changed_rows)
    assert _experiment_steps_contract_sha256(original) == (
        _experiment_steps_contract_sha256(changed)
    )


def test_experiment_steps_contract_hash_detects_contract_change(
    tmp_path: Path,
) -> None:
    source = PROJECT_ROOT / "plans/experiment_steps.csv"
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    original = tmp_path / "original.csv"
    changed = tmp_path / "contract_changed.csv"
    _write_step_rows(original, rows)
    changed_rows = deepcopy(rows)
    changed_rows[0]["acceptance"] += "; changed contract"
    _write_step_rows(changed, changed_rows)
    assert _experiment_steps_contract_sha256(original) != (
        _experiment_steps_contract_sha256(changed)
    )
