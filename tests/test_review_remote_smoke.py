from __future__ import annotations

import json
from pathlib import Path
import shutil

from mining1_exp.review_remote_smoke import (
    REQUIRED_RELATIVE_PATHS,
    review_e064,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _copy_review_fixture(target: Path) -> None:
    for relative in REQUIRED_RELATIVE_PATHS:
        source = PROJECT_ROOT / relative
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    source_bundle = PROJECT_ROOT / "runs/slurm/E064/19644/synthetic_contract_pipeline"
    target_bundle = target / "runs/slurm/E064/19644/synthetic_contract_pipeline"
    shutil.copytree(source_bundle, target_bundle)


def test_review_e064_accepts_synced_remote_smoke() -> None:
    result = review_e064(PROJECT_ROOT)
    assert result["status"] == "pass"
    assert result["error_count"] == 0
    assert result["diagnostics"]["module_artifact_count"] == 10
    assert result["diagnostics"]["final_billing_state"] == "verified_nonbilling"


def test_review_e064_rejects_tampered_cuda_coverage(tmp_path: Path) -> None:
    root = tmp_path / "tampered"
    _copy_review_fixture(root)
    smoke_path = root / "evidence/smoke/remote_smoke.json"
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    smoke["cuda_device_count"] = 0
    smoke_path.write_text(json.dumps(smoke, sort_keys=True) + "\n", encoding="utf-8")
    result = review_e064(root)
    assert result["status"] == "fail"
    assert any("cuda_device_count" in error for error in result["errors"])
