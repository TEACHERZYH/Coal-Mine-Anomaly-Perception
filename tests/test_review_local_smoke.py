from __future__ import annotations

import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

from mining1_exp.minimal_pipeline import run_minimal_pipeline
from mining1_exp.provenance import sha256_file
from mining1_exp.review_local_smoke import review_e062


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


@pytest.fixture(scope="module")
def completed_smoke(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("e062-complete") / "project"
    rows = []
    for modality in ("visible", "infrared", "methane"):
        sample = root / f"state/minimal_sample/{modality}.bin"
        sample.parent.mkdir(parents=True, exist_ok=True)
        sample.write_bytes(modality.encode("ascii"))
        rows.append(
            {
                "dataset_id": "fixture",
                "record_id": modality,
                "raw_group_id": f"group-{modality}",
                "modality": modality,
                "sample_path": sample.relative_to(root).as_posix(),
                "sha256": sha256_file(sample),
            }
        )
    manifest_path = root / "evidence/data/minimal_sample_manifest.parquet"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(manifest_path, index=False)
    rgbt_path = root / "data/locked/rgbt_input_lock.json"
    _write_json(
        rgbt_path,
        {
            "status": "pass",
            "source_modality_to_branch_role": {
                "visible": "visible",
                "infrared": "thermal",
            },
        },
    )
    bundle = root / "evidence/smoke/local_module_bundle"
    pipeline = run_minimal_pipeline(bundle)
    bundle_receipt = bundle / "integration_receipt.json"
    smoke_path = root / "evidence/smoke/local_smoke.json"
    smoke = {
        "schema_version": 1,
        "step_id": "E062",
        "status": "pass",
        "actual_sample_record_count": 3,
        "actual_sample_modalities": ["infrared", "methane", "visible"],
        "actual_sample_branch_roles": ["thermal", "visible"],
        "minimal_sample_manifest_sha256": sha256_file(manifest_path),
        "rgbt_input_lock_sha256": sha256_file(rgbt_path),
        "module_pipeline_mode": pipeline["mode"],
        "module_pipeline_receipt_sha256": sha256_file(bundle_receipt),
        "full_local_dataset_extractions": 0,
        "remote_connections": 0,
        "slurm_jobs_created": 0,
        "performance_claims_authorized": False,
    }
    _write_json(smoke_path, smoke)
    _write_json(
        root / "evidence/command_receipts/E062.json",
        {
            "step_id": "E062",
            "status": "pass",
            "command": "smoke-local",
            "arguments": {"minimal": True},
            "inputs": [
                {
                    "path": "evidence/data/minimal_sample_manifest.parquet",
                    "sha256": sha256_file(manifest_path),
                },
                {
                    "path": "data/locked/rgbt_input_lock.json",
                    "sha256": sha256_file(rgbt_path),
                },
            ],
            "outputs": [
                {
                    "path": "evidence/smoke/local_smoke.json",
                    "kind": "file",
                    "bytes": smoke_path.stat().st_size,
                    "sha256": sha256_file(smoke_path),
                }
            ],
            "details": {"actual_sample_record_count": 3, "full_extractions": 0},
        },
    )
    return root


def test_review_e062_accepts_closed_local_smoke(completed_smoke: Path) -> None:
    result = review_e062(completed_smoke)
    assert result["status"] == "pass"
    assert result["error_count"] == 0
    assert result["diagnostics"]["module_artifact_count"] == 10
    assert result["diagnostics"]["locked_branch_roles"] == ["thermal", "visible"]


def test_review_e062_rejects_tampered_branch_coverage(
    tmp_path: Path, completed_smoke: Path
) -> None:
    root = tmp_path / "tampered"
    shutil.copytree(completed_smoke, root)
    smoke_path = root / "evidence/smoke/local_smoke.json"
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    smoke["actual_sample_branch_roles"] = ["visible"]
    _write_json(smoke_path, smoke)
    result = review_e062(root)
    assert result["status"] == "fail"
    assert any("actual_sample_branch_roles" in error for error in result["errors"])
