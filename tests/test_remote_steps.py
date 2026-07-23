from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from mining1_exp import remote_steps
from mining1_exp.provenance import sha256_file
from mining1_exp.workflow_common import WorkflowExecutionError


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_remote_dispatcher_covers_exactly_the_locked_sbatch_steps() -> None:
    with (PROJECT_ROOT / "plans/experiment_steps.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    expected = {
        row["step_id"]
        for row in rows
        if row["command_entry"].startswith("sbatch slurm/")
    }
    assert remote_steps.REMOTE_STEP_IDS == expected


def _write_remote_fixture(root: Path, run_root: Path) -> None:
    (root / "plans").mkdir(parents=True)
    (root / "configs").mkdir()
    with (root / "plans/experiment_steps.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "step_id",
                "phase",
                "gate",
                "depends_on",
                "site",
                "resource",
                "command_entry",
                "inputs",
                "outputs",
                "acceptance",
                "failure_action",
                "status",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "step_id": "E024",
                "phase": "DATA",
                "gate": "G1",
                "depends_on": "none",
                "site": "remote_compute",
                "resource": "cpu_sbatch",
                "command_entry": "sbatch slurm/verify_archives.sbatch",
                "inputs": "fixture",
                "outputs": "evidence/out.json",
                "acceptance": "pass",
                "failure_action": "repair",
                "status": "not_run",
            }
        )
    decision = {
        "status": "locked",
        "selected_remote_python": "/data/home/xinxi-zhyh/envs/mining1/bin/python",
    }
    decision_path = root / "configs/environment_decision.lock.json"
    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    run_root.mkdir(parents=True)
    probe = {
        "selected_remote_python": decision["selected_remote_python"],
        "environment_fingerprint_sha256": "a" * 64,
        "package_inventory_sha256": "b" * 64,
        "environment_decision_sha256": sha256_file(decision_path),
        "compute_node": "cu01",
        "slurm_job_id": "12345",
        "slurm_array_task_id": None,
        "resource_kind": "cpu",
    }
    (run_root / "slurm_environment_probe.json").write_text(
        json.dumps(probe), encoding="utf-8"
    )


def test_remote_dispatcher_validates_context_and_writes_immutable_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "project"
    run_root = root / "runs/slurm/E024/12345"
    _write_remote_fixture(root, run_root)
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("MINING1_STEP_ID", "E024")
    monkeypatch.setenv("MINING1_RESOURCE_KIND", "cpu")
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
    monkeypatch.setattr(
        remote_steps.sys,
        "executable",
        "/data/home/xinxi-zhyh/envs/mining1/bin/python",
    )

    def fake_handler(*args, **kwargs):
        output = root / "evidence/out.json"
        output.parent.mkdir()
        output.write_text('{"status":"pass"}\n', encoding="ascii")
        return {"status": "pass", "output_paths": ["evidence/out.json"]}

    monkeypatch.setattr(remote_steps, "_dispatch_remote_handler", fake_handler)
    created = remote_steps.execute_remote_step(
        step_id="E024", run_root=run_root, config_path=None, project_root=root
    )
    assert created["status"] == "pass"
    receipt = json.loads(
        (run_root / "step_execution_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["compute_node"] == "cu01"
    assert receipt["remote_closeout_status"] == "pending_local_post_job_closeout"

    monkeypatch.setattr(
        remote_steps,
        "_dispatch_remote_handler",
        lambda *args, **kwargs: pytest.fail("completed run must not execute twice"),
    )
    reused = remote_steps.execute_remote_step(
        step_id="E024", run_root=run_root, config_path=None, project_root=root
    )
    assert reused["mode"] == "validated_existing"


def test_remote_dispatcher_rejects_login_or_non_slurm_execution(tmp_path: Path) -> None:
    with pytest.raises(WorkflowExecutionError, match="does not exist"):
        remote_steps.execute_remote_step(
            step_id="E064",
            run_root=tmp_path / "missing",
            config_path=None,
            project_root=PROJECT_ROOT,
        )


def test_e200_multi_coal_array_task_writes_reviewed_not_applicable_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "project"
    run_root = root / "runs/slurm/E200/999_6"
    (root / "plans").mkdir(parents=True)
    (root / "configs").mkdir()
    with (root / "plans/experiment_steps.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "step_id",
                "phase",
                "gate",
                "depends_on",
                "site",
                "resource",
                "command_entry",
                "inputs",
                "outputs",
                "acceptance",
                "failure_action",
                "status",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "step_id": "E069",
                "phase": "LOCK",
                "gate": "G1",
                "depends_on": "none",
                "site": "local",
                "resource": "manual",
                "command_entry": "closed",
                "inputs": "fixture",
                "outputs": "fixture",
                "acceptance": "pass",
                "failure_action": "repair",
                "status": "pass",
            }
        )
        writer.writerow(
            {
                "step_id": "E200",
                "phase": "TRANSFER",
                "gate": "G3",
                "depends_on": "E069",
                "site": "remote_compute",
                "resource": "gpu_sbatch",
                "command_entry": "sbatch slurm/train_v2_pretrain_array.sbatch",
                "inputs": "fixture",
                "outputs": "runs/V2/pretrain/",
                "acceptance": "fixture",
                "failure_action": "retry only under frozen failure rule",
                "status": "not_run",
            }
        )
    review_dir = root / "evidence/step_reviews"
    review_dir.mkdir(parents=True)
    (review_dir / "E069.json").write_text(
        json.dumps({"status": "pass", "advance_allowed": True}), encoding="ascii"
    )
    source_decision = root / "evidence/data/dataset_source_decision.json"
    source_decision.parent.mkdir(parents=True)
    source_decision.write_text(
        json.dumps(
            {
                "status": "pass",
                "claim_eligibility": {
                    "C-TRANSFER": {
                        "status": "not_applicable",
                        "reason_code": "fewer_than_two_eligible_non_target_coal_sources",
                        "eligible_non_target_coal_dataset_ids": ["dslmfplus_coal_miner_v1"],
                        "authority_path": "notes/minimal_experiment_plan_rereview_20260715.md",
                        "authority_sha256": "a" * 64,
                        "candidate_review_hashes": [
                            {"path": "evidence/data/source_reviews/dslmfplus.json", "sha256": "b" * 64}
                        ],
                    }
                },
            }
        ),
        encoding="ascii",
    )
    decision = {
        "status": "locked",
        "selected_remote_python": "/data/home/xinxi-zhyh/envs/mining1/bin/python",
    }
    decision_path = root / "configs/environment_decision.lock.json"
    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    run_root.mkdir(parents=True)
    (run_root / "slurm_environment_probe.json").write_text(
        json.dumps(
            {
                "selected_remote_python": decision["selected_remote_python"],
                "environment_fingerprint_sha256": "c" * 64,
                "package_inventory_sha256": "d" * 64,
                "environment_decision_sha256": sha256_file(decision_path),
                "compute_node": "gpu01",
                "slurm_job_id": "999",
                "slurm_array_task_id": "6",
                "resource_kind": "gpu",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SLURM_JOB_ID", "999")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "6")
    monkeypatch.setenv("MINING1_STEP_ID", "E200")
    monkeypatch.setenv("MINING1_RESOURCE_KIND", "gpu")
    monkeypatch.setattr(
        remote_steps.sys,
        "executable",
        "/data/home/xinxi-zhyh/envs/mining1/bin/python",
    )

    result = remote_steps.execute_remote_step(
        step_id="E200", run_root=run_root, config_path=None, project_root=root
    )

    assert result["status"] == "pass"
    receipt = json.loads(
        (run_root / "V2-PRE-MULTI_not_applicable.json").read_text(encoding="utf-8")
    )
    assert receipt["status"] == "not_applicable"
    assert receipt["family_id"] == "V2-PRE-MULTI"
    assert receipt["fallback"]["reason_code"] == (
        "fewer_than_two_eligible_non_target_coal_sources"
    )
