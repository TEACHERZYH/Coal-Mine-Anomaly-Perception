from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import tarfile
from unittest.mock import patch

import pytest

from mining1_exp.remote_environment import (
    REMOTE_HOST,
    STAGED_FILES,
    _parse_remote_result,
    _remote_submission_script,
    _run_remote_setup,
    _stage_archive,
    _validate_compute_receipt,
)
from mining1_exp.provenance import sha256_file
from mining1_exp.workflow_common import WorkflowExecutionError
from tools.preflight.verify_locked_environment import _import_module_captured


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_p030_archive_and_submission_are_bounded_to_compute_setup() -> None:
    archive = _stage_archive(PROJECT_ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as handle:
        assert tuple(sorted(handle.getnames())) == tuple(sorted(STAGED_FILES))
    script = _remote_submission_script(archive)
    assert script.count("sbatch --parsable") == 1
    assert "setup_remote_environment.sbatch" in script
    assert "--partition=cu" not in script
    assert "pip install" not in script
    assert "211.87.115.228" not in script
    assert "180.209.128.66" not in script


def test_p030_ssh_stdin_is_lf_only_bytes() -> None:
    with patch("mining1_exp.remote_environment.subprocess.run") as run:
        run.return_value = __import__("subprocess").CompletedProcess(
            args=["ssh"], returncode=0, stdout=b"ok\n", stderr=b""
        )
        completed = _run_remote_setup("set -euo pipefail\r\nprintf 'ok\\n'\r\n")

    sent = run.call_args.kwargs["input"]
    assert isinstance(sent, bytes)
    assert b"\r" not in sent
    assert sent.startswith(b"set -euo pipefail\n")
    assert completed.stdout == "ok\n"


def test_environment_import_output_is_captured(capsys: pytest.CaptureFixture[str]) -> None:
    sentinel = object()

    def noisy_import(module_name: str) -> object:
        print(f"created settings for {module_name}")
        return sentinel

    with patch(
        "tools.preflight.verify_locked_environment.importlib.import_module",
        side_effect=noisy_import,
    ):
        module, diagnostics = _import_module_captured("ultralytics")

    assert module is sentinel
    assert diagnostics == {"stdout": "created settings for ultralytics"}
    assert capsys.readouterr().out == ""


def test_p030_result_parser_and_compute_receipt_are_fail_closed(tmp_path: Path) -> None:
    decision = tmp_path / "decision.json"
    decision.write_text('{"status":"locked"}', encoding="ascii")
    receipt = {
        "schema_version": 2,
        "scope": "remote",
        "status": "pass",
        "ready_at": "2026-07-15T00:00:00+00:00",
        "host": REMOTE_HOST,
        "decision_sha256": sha256_file(decision),
        "strategy": "new_isolated",
        "selected_python": "/data/home/xinxi-zhyh/envs/mining1/bin/python",
        "environment_specification_sha256": "a" * 64,
        "package_inventory_sha256": "b" * 64,
        "slurm_job_id": "12345",
        "compute_node": "cu01",
        "remote_closeout_reference": None,
    }
    raw = json.dumps(receipt).encode("utf-8")
    stdout = "\n".join(
        (
            "P030_JOB_ID=12345",
            "P030_STATE=COMPLETED",
            "P030_EXIT_CODE=0:0",
            "P030_RECEIPT_B64=" + base64.b64encode(raw).decode("ascii"),
        )
    )
    job_id, parsed, parsed_raw = _parse_remote_result(stdout)
    assert parsed_raw == raw
    _validate_compute_receipt(parsed, decision, job_id)

    parsed["compute_node"] = "mu01"
    with pytest.raises(WorkflowExecutionError, match="violates"):
        _validate_compute_receipt(parsed, decision, job_id)
