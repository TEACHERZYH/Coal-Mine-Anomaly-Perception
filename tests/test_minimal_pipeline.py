from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd
import pytest

from mining1_exp.minimal_pipeline import (
    MinimalPipelineError,
    _write_parquet_once,
    run_minimal_pipeline,
    validate_minimal_pipeline_run,
)
from mining1_exp.provenance import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _file_hashes(root: Path) -> dict[str, str]:
    return {
        path.name: sha256_file(path)
        for path in sorted(root.iterdir())
        if path.is_file()
    }


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, object]]:
    run_root = tmp_path_factory.mktemp("i080-complete") / "integration"
    return run_root, run_minimal_pipeline(run_root)


def test_minimal_pipeline_runs_end_to_end_and_is_idempotent(
    completed_run: tuple[Path, dict[str, object]],
) -> None:
    run_root, created = completed_run
    assert created["status"] == "pass"
    assert created["mode"] == "created"
    assert created["artifact_count"] == 10

    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["synthetic_only"] is True
    assert summary["max_groups_per_pool"] <= 2
    assert set(summary["pool_group_counts"]) == {
        "D_b_tr",
        "D_b_prob",
        "D_e_tr",
        "D_e_te",
    }
    assert summary["full_local_dataset_extractions"] == 0
    assert summary["remote_connections"] == 0
    assert summary["slurm_jobs_created"] == 0
    assert summary["fit_and_inference_samples_disjoint"] is True
    assert summary["methane_boundary"]["fit_rows"] == 20
    assert summary["methane_boundary"]["inference_rows"] == 4
    assert summary["methane_boundary"]["fit_input_sha256"] != summary[
        "methane_boundary"
    ]["inference_input_sha256"]
    assert summary["episode_boundary"]["fit_input_sha256"] != summary[
        "episode_boundary"
    ]["inference_input_sha256"]
    assert {
        "methane_hgb",
        "methane_gru_one_step",
        "episode_reliability_graph_one_step",
        "episode_memory_and_event_evaluation",
    }.issubset(summary["models_exercised"])

    split = pd.read_parquet(run_root / "split_manifest.parquet")
    assert split.groupby("pool")["raw_group_id"].nunique().max() <= 2
    branch = pd.read_parquet(run_root / "branch_predictions.parquet")
    episode = pd.read_parquet(run_root / "episode_predictions.parquet")
    forbidden = ("truth", "label", "target", "test_metric")
    assert not any(token in column.lower() for column in branch for token in forbidden)
    assert not any(token in column.lower() for column in episode for token in forbidden)
    assert branch.duplicated(["record_id", "concept_id", "modality"]).sum() == 0
    assert episode["skeleton_item_id"].is_unique
    assert episode["skeleton_item_id"].str.fullmatch(r"[0-9a-f]{64}").all()

    first_hashes = _file_hashes(run_root)
    existing = run_minimal_pipeline(run_root)
    assert existing["status"] == "pass"
    assert existing["mode"] == "validated_existing"
    assert _file_hashes(run_root) == first_hashes
    assert validate_minimal_pipeline_run(run_root)["receipt_sha256"] == created[
        "receipt_sha256"
    ]


def test_execute_step_cli_dispatches_only_the_reviewed_integration(tmp_path: Path) -> None:
    run_root = tmp_path / "cli-run"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mining1_exp.cli",
            "execute-step",
            "--step",
            "I080",
            "--run-root",
            str(run_root),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "pass"
    assert payload["step_id"] == "I080"
    assert validate_minimal_pipeline_run(run_root)["status"] == "pass"

    unsupported_root = tmp_path / "unsupported"
    blocked = subprocess.run(
        [
            sys.executable,
            "-m",
            "mining1_exp.cli",
            "execute-step",
            "--step",
            "E064",
            "--run-root",
            str(unsupported_root),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert blocked.returncode == 2
    assert json.loads(blocked.stderr)["status"] == "fail"
    assert not unsupported_root.exists()


def test_minimal_pipeline_fails_closed_on_dirty_or_tampered_roots(
    tmp_path: Path, completed_run: tuple[Path, dict[str, object]]
) -> None:
    dirty = tmp_path / "dirty"
    dirty.mkdir()
    (dirty / "untracked.txt").write_text("do not overwrite", encoding="ascii")
    with pytest.raises(MinimalPipelineError, match="must be empty"):
        run_minimal_pipeline(dirty)
    assert (dirty / "untracked.txt").read_text(encoding="ascii") == "do not overwrite"
    failure = json.loads((dirty / "failure.json").read_text(encoding="utf-8"))
    assert failure["status"] == "failed"
    assert failure["retry_rule"] == "repair_then_use_fresh_run_root"

    completed = tmp_path / "completed"
    shutil.copytree(completed_run[0], completed)
    untracked_directory = completed / "untracked"
    untracked_directory.mkdir()
    with pytest.raises(MinimalPipelineError, match="directory or link"):
        validate_minimal_pipeline_run(completed)
    untracked_directory.rmdir()
    summary_path = completed / "summary.json"
    receipt_path = completed / "integration_receipt.json"
    original_summary = summary_path.read_bytes()
    original_receipt = receipt_path.read_bytes()
    summary_path.write_bytes(summary_path.read_bytes() + b"\n")
    with pytest.raises(MinimalPipelineError, match="hash drift"):
        validate_minimal_pipeline_run(completed)
    summary_path.write_bytes(original_summary)
    receipt_path.write_bytes(original_receipt)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["remote_connections"] = 1
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["summary_sha256"] = sha256_file(summary_path)
    for artifact in receipt["artifacts"]:
        if artifact["path"] == "summary.json":
            artifact["bytes"] = summary_path.stat().st_size
            artifact["sha256"] = sha256_file(summary_path)
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(MinimalPipelineError, match="local synthetic boundary"):
        validate_minimal_pipeline_run(completed)


def test_minimal_pipeline_source_has_no_dataset_or_remote_side_effect_path() -> None:
    source = (PROJECT_ROOT / "mining1_exp" / "minimal_pipeline.py").read_text(
        encoding="utf-8"
    ).lower()
    for forbidden in (
        "g:\\datasets",
        "180.209.128.66",
        "211.87.115.228",
        "subprocess",
        "requests",
        "urllib",
        "shutil.unpack_archive",
        "zipfile",
        "tarfile",
        "sbatch",
        "ssh ",
    ):
        assert forbidden not in source


def test_parquet_outputs_are_write_once(tmp_path: Path) -> None:
    target = tmp_path / "artifact.parquet"
    target.write_bytes(b"existing evidence")
    with pytest.raises(MinimalPipelineError, match="refusing to overwrite"):
        _write_parquet_once(pd.DataFrame({"value": [1]}), target)
    assert target.read_bytes() == b"existing evidence"
    assert not list(tmp_path.glob(".*.stage"))
