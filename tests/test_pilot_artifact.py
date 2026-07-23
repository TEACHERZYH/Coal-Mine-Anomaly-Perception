from __future__ import annotations

import json
from pathlib import Path

from mining1_exp.pilot_artifact import normalize_resource_pilot_metadata
from mining1_exp.provenance import canonical_json_sha256, sha256_file


def test_normalize_resource_pilot_metadata_preserves_measurements(tmp_path: Path) -> None:
    source = tmp_path / "raw.json"
    target = tmp_path / "normalized.json"
    payload = {
        "status": "pass",
        "slurm_job_id": "19653",
        "cuda_device_count": 2,
        "cuda_device_names": ["A100", "A100"],
        "benchmarks": [{"family": "visible", "wall_time_seconds": 1.0}],
        "scaling_trial": {"compare_gpu_counts": [1, 2]},
        "decisions": {
            "gpu_hour_caps": {
                "T1": 24.0,
                "S1": 24.0,
                "V2": 160.0,
                "E2_E3": 48.0,
                "R1": 12.0,
                "C1": 8.0,
            }
        },
    }
    source.write_text(json.dumps(payload), encoding="utf-8")
    source_sha256 = sha256_file(source)
    normalized = normalize_resource_pilot_metadata(source, target)
    assert normalized["decisions"]["gpu_hour_caps"]["G2"] == 48.0
    assert normalized["metadata_repair"]["source_sha256"] == source_sha256
    assert normalized["metadata_repair"]["measurements_reused_without_rerun"] is True
    original_measurements = {
        key: payload[key]
        for key in (
            "slurm_job_id",
            "cuda_device_count",
            "cuda_device_names",
            "benchmarks",
            "scaling_trial",
        )
    }
    normalized_measurements = {
        key: normalized[key] for key in original_measurements
    }
    assert canonical_json_sha256(original_measurements) == canonical_json_sha256(
        normalized_measurements
    )
