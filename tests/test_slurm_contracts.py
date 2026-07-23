from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

from mining1_exp.governance.immutable import GovernanceContractError
from mining1_exp.governance.slurm_contracts import (
    validate_locked_step_runner,
    validate_sbatch_script,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _remote_sbatch_rows() -> list[dict[str, str]]:
    with (PROJECT_ROOT / "plans" / "experiment_steps.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if row["command_entry"].startswith("sbatch slurm/")]


def test_every_minimal_plan_sbatch_entry_exists_and_has_locked_resources() -> None:
    rows = _remote_sbatch_rows()
    assert len(rows) == 20
    seen_paths = set()
    for row in rows:
        relative = row["command_entry"].split(maxsplit=1)[1]
        script = PROJECT_ROOT / relative
        assert script.is_file(), f"missing Slurm entry: {relative}"
        resource_kind = "gpu" if row["resource"] == "gpu_sbatch" else "cpu"
        directives = validate_sbatch_script(script, resource_kind=resource_kind)
        assert f'MINING1_STEP_ID="{row["step_id"]}"' in script.read_text(
            encoding="utf-8"
        )
        assert directives["output"] != directives["error"]
        if row["step_id"] == "E066":
            assert directives["gres"] == "gpu:2"
        seen_paths.add(relative)
    assert len(seen_paths) == len(rows)


def test_e024_streaming_hash_job_does_not_overallocate_cpu_or_memory() -> None:
    text = (PROJECT_ROOT / "slurm/verify_archives.sbatch").read_text(
        encoding="utf-8-sig"
    )
    assert "#SBATCH --cpus-per-task=1" in text
    assert "#SBATCH --mem=4G" in text


def test_e026_materialization_job_uses_bounded_cpu_and_memory() -> None:
    text = (PROJECT_ROOT / "slurm/build_file_manifest.sbatch").read_text(
        encoding="utf-8-sig"
    )
    assert "#SBATCH --cpus-per-task=8" in text
    assert "#SBATCH --mem=32G" in text


def test_e034_dedup_job_uses_bounded_parallel_cpu_and_memory() -> None:
    text = (PROJECT_ROOT / "slurm/audit_groups_dedup.sbatch").read_text(
        encoding="utf-8-sig"
    )
    assert "#SBATCH --cpus-per-task=8" in text
    assert "#SBATCH --mem=16G" in text


def test_sbatch_entrypoints_do_not_depend_on_preserved_executable_mode() -> None:
    entrypoints = [
        PROJECT_ROOT / row["command_entry"].split(maxsplit=1)[1]
        for row in _remote_sbatch_rows()
    ]
    assert entrypoints
    expected = 'exec bash "$PROJECT_ROOT/slurm/run_locked_step.sh"'
    direct = 'exec "$PROJECT_ROOT/slurm/run_locked_step.sh"'
    for entrypoint in entrypoints:
        text = entrypoint.read_text(encoding="utf-8")
        assert expected in text, entrypoint.name
        assert direct not in text, entrypoint.name


def test_locked_runner_uses_remote_decision_and_propagates_exit_and_signal() -> None:
    runner = PROJECT_ROOT / "slurm" / "run_locked_step.sh"
    validate_locked_step_runner(runner)
    content = runner.read_text(encoding="utf-8")
    assert "/data/home/xinxi-zhyh/xinxi-zhyh/envs/mining1-py39-cu121/bin/python" in content
    assert "SLURM_JOB_ID:?" in content
    assert "compute_node.txt" in content
    assert "slurm_environment_probe.json" in content
    assert 'cd "$PROJECT_ROOT"' in content
    assert 'export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"' in content
    assert content.index('cd "$PROJECT_ROOT"') < content.index("-m mining1_exp.cli")
    assert '"$RUN_ROOT/environment.json"' not in content
    assert "180.209.128.66" not in content
    assert "zhangl@" not in content

    help_result = subprocess.run(
        [sys.executable, "-m", "mining1_exp.cli", "execute-step", "--help"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0
    blocked_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mining1_exp.cli",
            "execute-step",
            "--step",
            "E064",
            "--run-root",
            str(PROJECT_ROOT / "runs" / "synthetic"),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert blocked_result.returncode != 0
    assert json.loads(blocked_result.stderr)["status"] == "fail"


def test_monitor_and_account_closeout_are_account_wide_and_fail_closed() -> None:
    monitor = (
        PROJECT_ROOT / "tools" / "remote" / "monitor_jobs.ps1"
    ).read_text(encoding="utf-8")
    closeout = (
        PROJECT_ROOT / "tools" / "remote" / "account_wide_closeout.ps1"
    ).read_text(encoding="utf-8")
    for content in (monitor, closeout):
        assert "xinxi-zhyh@211.87.115.228" in content
        assert "180.209.128.66" not in content
        assert "squeue" in content
        assert "sacct" in content
        assert "tmux" in content
        assert "screen" in content
        assert "ConnectTimeout=8" in content
    assert "nvidia-smi" not in monitor
    assert "RemoteRunRoot must be an absolute path" in monitor
    assert "AddMinutes(30)" in monitor
    assert "gpu_telemetry_artifacts" in monitor
    assert "gpu_telemetry_excerpt" in monitor
    assert "tmux_and_screen_sessions" in monitor
    assert monitor.count("& ssh") == 1
    assert "account_scope = 'all_user_owned_work'" in closeout
    assert "UnsynchronizedArtifacts" in closeout
    assert "RetainedResourcesWithReason" in closeout
    assert "CurrentProjectJobPattern" in closeout
    assert "active job for another named project" in closeout
    assert "$jobName -notmatch $CurrentProjectJobPattern" in closeout
    assert "provider_stop_action" in closeout
    assert "final_billing_state" in closeout
    assert "Logging out or closing SSH is not billing-stop evidence" in closeout
    assert closeout.count("& ssh") == 3
    assert "scancel -u" not in closeout


def test_malformed_sbatch_and_bare_python_are_rejected(tmp_path: Path) -> None:
    malformed = tmp_path / "bad.sbatch"
    malformed.write_text(
        "#!/bin/bash\n#SBATCH --partition=3090\nset -euo pipefail\npython -m x\n",
        encoding="utf-8",
    )
    with pytest.raises(GovernanceContractError, match="directives are missing"):
        validate_sbatch_script(malformed, resource_kind="gpu")

    valid_source = (PROJECT_ROOT / "slurm" / "smoke_minimal.sbatch").read_text(
        encoding="utf-8"
    )
    bad_nodes = tmp_path / "bad-nodes.sbatch"
    bad_nodes.write_text(
        valid_source.replace("#SBATCH --nodes=1", "#SBATCH --nodes=2"),
        encoding="utf-8",
    )
    with pytest.raises(GovernanceContractError, match="exactly one node"):
        validate_sbatch_script(bad_nodes, resource_kind="gpu")

    bad_log = tmp_path / "bad-log.sbatch"
    bad_log.write_text(
        valid_source.replace(
            "/data/home/xinxi-zhyh/xinxi-zhyh/logs/mining1/%x_%j.out",
            "/tmp/%x_%j.out",
        ),
        encoding="utf-8",
    )
    with pytest.raises(GovernanceContractError, match="outside the project account"):
        validate_sbatch_script(bad_log, resource_kind="gpu")

    bare_python = tmp_path / "bare-python.sbatch"
    bare_python.write_text(valid_source + "\npython -m forbidden\n", encoding="utf-8")
    with pytest.raises(GovernanceContractError, match="bare Python"):
        validate_sbatch_script(bare_python, resource_kind="gpu")


def test_all_package_sources_parse_with_the_locked_remote_python39_grammar() -> None:
    for source in (PROJECT_ROOT / "mining1_exp").rglob("*.py"):
        ast.parse(
            source.read_text(encoding="utf-8"),
            filename=str(source),
            feature_version=(3, 9),
        )
