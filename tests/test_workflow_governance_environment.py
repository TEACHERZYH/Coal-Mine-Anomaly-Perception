from __future__ import annotations

import json
from pathlib import Path

import pytest

from mining1_exp.provenance import sha256_file
from mining1_exp.workflow_governance import _environment_values
from mining1_exp.workflow_governance import _load_or_create_precision_feasibility


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _verification(scope: str, selected_python: str, python_version: str, packages: dict) -> dict:
    return {
        "schema_version": 1,
        "scope": scope,
        "status": "pass",
        "selected_python": selected_python,
        "python_version": python_version,
        "torch_cuda_runtime": "12.8" if scope == "local" else "12.1",
        "packages": packages,
    }


def _prepare_environment_root(tmp_path: Path) -> Path:
    local_python = r"C:\Users\zyh\miniconda3\envs\dl_env\python.exe"
    remote_python = "/data/home/xinxi-zhyh/xinxi-zhyh/envs/mining1-py39-cu121/bin/python"
    decision = {
        "local_strategy": "install_compatible_missing",
        "remote_strategy": "new_isolated",
        "selected_local_python": local_python,
        "selected_remote_python": remote_python,
        "selected_remote_compiler": "/usr/bin/gcc 11.4.1 and /usr/bin/g++ 11.4.1",
        "local_environment_specification": "env/local.lock.txt",
        "remote_environment_specification": "env/remote.lock.txt",
    }
    decision_path = tmp_path / "configs/environment_decision.lock.json"
    _write_json(decision_path, decision)
    local_verification = _verification(
        "local", local_python, "3.10.20", {"torch": "2.7.1+cu128"}
    )
    remote_packages = {
        "torch": "2.5.1+cu121",
        "ultralytics": "8.4.87",
        "numpy": "1.26.4",
        "scipy": "1.13.1",
        "pandas": "2.2.3",
        "pyarrow": "19.0.1",
    }
    remote_verification = _verification(
        "remote", remote_python, "3.9.18", remote_packages
    )
    _write_json(
        tmp_path / "evidence/preimplementation/local_environment_ready.json",
        {
            "selected_python": local_python,
            "verification_output_excerpt": [json.dumps(local_verification)],
            "package_inventory": ["serialized pip freeze is not the verified package map"],
        },
    )
    remote_ready_path = tmp_path / "evidence/preimplementation/remote_environment_ready.json"
    _write_json(
        remote_ready_path,
        {
            "selected_python": remote_python,
            "verification_output_excerpt": [json.dumps(remote_verification)],
            "package_inventory": ["serialized pip freeze is not the verified package map"],
        },
    )
    _write_json(
        tmp_path / "evidence/preimplementation/E067_remote_driver_repair.json",
        {
            "status": "pass",
            "step_id": "E067",
            "probe_kind": "nvidia_driver_metadata",
            "job_state": "COMPLETED",
            "exit_code": "0:0",
            "remote_host": "xinxi-zhyh@211.87.115.228",
            "selected_remote_python": remote_python,
            "python_version": "3.9.18",
            "torch_cuda_runtime": "12.1",
            "driver_version": "535.183.01",
            "decision_sha256": sha256_file(decision_path),
            "remote_ready_receipt_sha256": sha256_file(remote_ready_path),
            "cuda_available": True,
            "training_performed": False,
            "dataset_accessed": False,
            "stderr_bytes": 0,
        },
    )
    return tmp_path


def test_environment_values_parse_current_ready_receipt_schema(tmp_path: Path) -> None:
    root = _prepare_environment_root(tmp_path)
    values = _environment_values(root)
    assert values["local_python_version"] == "3.10.20"
    assert values["remote_python_version"] == "3.9.18"
    assert values["remote_pytorch_version"] == "2.5.1+cu121"
    assert values["remote_cuda_runtime"] == "12.1"
    assert values["remote_cuda_driver"] == "535.183.01"
    assert values["remote_compiler"] == "/usr/bin/gcc 11.4.1 and /usr/bin/g++ 11.4.1"


def test_environment_values_require_driver_repair_when_ready_receipt_omits_driver(
    tmp_path: Path,
) -> None:
    root = _prepare_environment_root(tmp_path)
    (root / "evidence/preimplementation/E067_remote_driver_repair.json").unlink()
    with pytest.raises(RuntimeError, match="driver repair receipt is missing"):
        _environment_values(root)


def test_environment_values_reject_driver_repair_runtime_conflict(tmp_path: Path) -> None:
    root = _prepare_environment_root(tmp_path)
    path = root / "evidence/preimplementation/E067_remote_driver_repair.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["torch_cuda_runtime"] = "11.8"
    _write_json(path, payload)
    with pytest.raises(RuntimeError, match="torch_cuda_runtime does not match"):
        _environment_values(root)


def _precision_protocol() -> dict:
    return {
        "seeds": {"statistics": 12011},
        "data": {
            "precision_feasibility": {"simulation_repetitions": 10000},
            "independent_group_floor_hard_min": 20,
            "positive_group_floor_per_claimed_class_hard_min": 10,
        },
    }


def test_precision_feasibility_retry_reuses_identical_write_once_evidence(
    tmp_path: Path,
) -> None:
    first = _load_or_create_precision_feasibility(tmp_path, _precision_protocol())
    second = _load_or_create_precision_feasibility(tmp_path, _precision_protocol())
    assert first == second
    assert second["simulation_repetitions"] == 10000


def test_precision_feasibility_retry_rejects_content_drift(tmp_path: Path) -> None:
    _load_or_create_precision_feasibility(tmp_path, _precision_protocol())
    path = tmp_path / "evidence/g1/precision_feasibility.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["endpoint_results"][0]["success_probability"] = 0.0
    _write_json(path, payload)
    with pytest.raises(RuntimeError, match="differs from the frozen simulation"):
        _load_or_create_precision_feasibility(tmp_path, _precision_protocol())
