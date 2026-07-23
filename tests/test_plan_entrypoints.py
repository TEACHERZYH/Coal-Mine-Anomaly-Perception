from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import subprocess

import pytest

from mining1_exp.cli import build_parser
from mining1_exp.workflow_registry import PLAN_CLI_COMMANDS, SLURM_STEP_IDS


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _plan_rows() -> list[dict[str, str]]:
    with (PROJECT_ROOT / "plans/experiment_steps.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        return list(csv.DictReader(handle))


def _parser_commands() -> frozenset[str]:
    parser = build_parser()
    action = next(
        item for item in parser._actions if isinstance(item, argparse._SubParsersAction)
    )
    return frozenset(action.choices)


def test_plan_cli_commands_exactly_match_the_registered_workflow_surface() -> None:
    commands = set()
    pattern = re.compile(r"-Module mining1_exp\.cli\s+([^\s]+)")
    for row in _plan_rows():
        match = pattern.search(row["command_entry"])
        if match:
            commands.add(match.group(1))
    assert commands == PLAN_CLI_COMMANDS
    assert PLAN_CLI_COMMANDS.issubset(_parser_commands())


def test_every_plan_cli_command_exposes_help() -> None:
    for command in sorted(PLAN_CLI_COMMANDS):
        with pytest.raises(SystemExit) as exc_info:
            build_parser().parse_args([command, "--help"])
        assert exc_info.value.code == 0


def test_slurm_scripts_exactly_match_the_remote_dispatch_surface() -> None:
    found = set()
    pattern = re.compile(r'MINING1_STEP_ID="([^"]+)"')
    for path in (PROJECT_ROOT / "slurm").glob("*.sbatch"):
        content = path.read_text(encoding="utf-8-sig")
        match = pattern.search(content)
        if path.name == "setup_remote_environment.sbatch":
            assert match is None
            assert "mining1_P030_env" in content
            assert "runs/slurm/P030" in content
            continue
        assert match is not None, path
        found.add(match.group(1))
    assert found == SLURM_STEP_IDS


def test_locked_runner_accepts_the_bom_prefixed_decision() -> None:
    text = (PROJECT_ROOT / "slurm/run_locked_step.sh").read_text(encoding="utf-8-sig")
    assert 'read_text(encoding="utf-8-sig")' in text


def test_selected_python_wrapper_forwards_downstream_decision_option(
) -> None:
    wrapper = PROJECT_ROOT / "tools/preflight/run_selected_python.ps1"
    text = wrapper.read_text(encoding="utf-8-sig")
    assert "[string]$EnvironmentDecisionPath" in text
    assert "[string]$DecisionPath" not in text
    assert "@RemainingArguments" in text

    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(wrapper),
            "-Module",
            "pytest",
            "--decision",
            "wrapper-forwarding-probe",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "unrecognized arguments: --decision" in combined
    assert "Selected local Python is missing or invalid" not in combined
