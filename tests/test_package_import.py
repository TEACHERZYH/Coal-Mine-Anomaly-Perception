from __future__ import annotations

import importlib
import io
import json
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OLD_PROJECT_ROOT = Path("F:/2026/mining").resolve()


def test_package_import_graph_is_clean_room() -> None:
    package = importlib.import_module("mining1_exp")
    imported = [
        module
        for name, module in sys.modules.items()
        if name == "mining1_exp" or name.startswith("mining1_exp.")
    ]
    assert package.__version__ == "0.1.0.dev0"
    for module in imported:
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        source = Path(module_file).resolve()
        assert source == PROJECT_ROOT or PROJECT_ROOT in source.parents
        assert source != OLD_PROJECT_ROOT and OLD_PROJECT_ROOT not in source.parents


def test_package_sources_do_not_reference_old_project() -> None:
    forbidden = str(OLD_PROJECT_ROOT).replace("/", "\\").lower() + "\\"
    for source in (PROJECT_ROOT / "mining1_exp").rglob("*.py"):
        text = source.read_text(encoding="utf-8").replace("/", "\\").lower()
        assert forbidden not in text


def test_structured_config_snapshot_and_hashes(tmp_path: Path) -> None:
    from mining1_exp.config import ConfigError, resolved_config_snapshot

    config_path = tmp_path / "config.yaml"
    config_path.write_text("beta: 2\nalpha: [1, 3]\n", encoding="utf-8")
    snapshot = resolved_config_snapshot(config_path)
    assert snapshot["resolved"] == {"beta": 2, "alpha": [1, 3]}
    assert len(snapshot["source_sha256"]) == 64
    assert len(snapshot["resolved_sha256"]) == 64
    invalid_path = tmp_path / "config.txt"
    invalid_path.write_text("alpha=1", encoding="utf-8")
    with pytest.raises(ConfigError):
        resolved_config_snapshot(invalid_path)


def test_canonical_hash_is_order_independent() -> None:
    from mining1_exp.provenance import canonical_json_sha256

    assert canonical_json_sha256({"a": 1, "b": 2}) == canonical_json_sha256(
        {"b": 2, "a": 1}
    )


def test_atomic_receipt_contains_input_hash(tmp_path: Path) -> None:
    from mining1_exp.provenance import build_receipt, sha256_file, write_receipt

    source = tmp_path / "input.json"
    source.write_text('{"value":1}\n', encoding="utf-8")
    receipt = build_receipt(
        step_id="I000_TEST",
        status="pass",
        command="pytest tests/test_package_import.py",
        inputs=[source],
        details={"purpose": "unit-test"},
    )
    target = tmp_path / "receipt.json"
    digest = write_receipt(target, receipt)
    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert loaded["inputs"][0]["sha256"] == sha256_file(source)
    assert digest.sha256 == sha256_file(target)
    assert not list(tmp_path.glob("*.tmp"))


def test_json_logging() -> None:
    from mining1_exp.logging_utils import configure_logging

    stream = io.StringIO()
    logger = configure_logging(stream=stream)
    logger.info("ready", extra={"step_id": "I000"})
    payload = json.loads(stream.getvalue())
    assert payload["level"] == "INFO"
    assert payload["message"] == "ready"
    assert payload["step_id"] == "I000"


def test_cli_help_version_and_nonzero_failure(tmp_path: Path) -> None:
    help_result = subprocess.run(
        [sys.executable, "-m", "mining1_exp.cli", "--help"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0
    assert "validate-config" in help_result.stdout

    version_result = subprocess.run(
        [sys.executable, "-m", "mining1_exp.cli", "version"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert version_result.returncode == 0
    assert json.loads(version_result.stdout)["version"] == "0.1.0.dev0"

    failure_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mining1_exp.cli",
            "validate-config",
            "--config",
            str(tmp_path / "missing.yaml"),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert failure_result.returncode != 0
    assert json.loads(failure_result.stderr)["status"] == "fail"
