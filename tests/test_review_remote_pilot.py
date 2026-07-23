from __future__ import annotations

import json
from pathlib import Path

import yaml

from mining1_exp.pilot_artifact import normalize_resource_pilot_metadata
from mining1_exp.review_remote_pilot import review_resource_pilot


def _record(family: str, precision: str, batch: int, speed: float, memory: int) -> dict:
    return {
        "status": "pass",
        "family": family,
        "precision": precision,
        "micro_batch_size": batch,
        "warmup_updates": 100,
        "timed_updates": 300,
        "wall_time_seconds": 2.0,
        "updates_per_second": speed / batch,
        "examples_per_second": speed,
        "peak_memory_bytes": memory,
        "numerically_finite": True,
        "gradient_scaler_enabled": precision == "amp_fp16",
        "deterministic_repeat_hash": ("a" if precision == "fp32" else "b") * 64,
    }


def test_review_resource_pilot_checks_measurements_and_gate_caps(tmp_path: Path) -> None:
    root = tmp_path
    (root / "configs").mkdir()
    (root / "evidence/failures/E066").mkdir(parents=True)
    (root / "evidence/pilot").mkdir(parents=True)
    (root / "configs/protocol_lock.template.yaml").write_text(
        yaml.safe_dump(
            {
                "pilot": {
                    "scaling_trial": {"compare_gpu_counts": [1, 2]},
                    "forbidden_outputs": [
                        "test_metrics",
                        "test_predictions",
                        "condition_ranking",
                        "claim_direction",
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    batches = {"visible": 4, "rgbt": 2, "methane": 32, "episode": 64}
    benchmarks = []
    for family, batch in batches.items():
        benchmarks.append(_record(family, "fp32", batch, 100.0, 1000))
        benchmarks.append(_record(family, "amp_fp16", batch, 90.0, 800))
    scaling_records = []
    for gpu_count, speed, digest in ((1, 100.0, "c" * 64), (2, 50.0, "d" * 64)):
        for repeat in (0, 1):
            scaling_records.append(
                {
                    **_record("visible", "amp_fp16", 4, speed, 800),
                    "gpu_count": gpu_count,
                    "repeat_index": repeat,
                    "examples_per_update": 4 * gpu_count,
                    "deterministic_repeat_hash": digest,
                }
            )
    raw = {
        "schema_version": 1,
        "step_id": "E066",
        "status": "pass",
        "slurm_job_id": "19653",
        "cuda_device_count": 2,
        "cuda_device_names": ["A100", "A100"],
        "benchmarks": benchmarks,
        "scaling_trial": {
            "status": "pass",
            "compare_gpu_counts": [1, 2],
            "records": scaling_records,
            "per_device_batch": 4,
            "two_gpu_throughput_speedup": 0.5,
            "two_gpu_parallel_efficiency": 0.25,
            "reproducibility_pass": True,
        },
        "decisions": {
            "precision": "amp_fp16",
            "micro_batch_size": batches,
            "gradient_accumulation": {"visible": 16, "rgbt": 32, "methane": 4, "episode": 1},
            "max_array_concurrency": 1,
            "multi_gpu_training_enabled": False,
            "gpu_hour_caps": {"T1": 24.0, "S1": 24.0, "V2": 160.0, "E2_E3": 48.0, "R1": 12.0, "C1": 8.0},
        },
        "model_or_condition_ranking_performed": False,
        "performance_claims_authorized": False,
    }
    source = root / "evidence/failures/E066/raw.json"
    source.write_text(json.dumps(raw), encoding="utf-8")
    target = root / "evidence/pilot/resource_pilot.json"
    normalize_resource_pilot_metadata(source, target)
    review = review_resource_pilot(root, target)
    assert review["status"] == "pass"
    assert review["benchmark_count"] == 8
    assert review["two_gpu_speedup"] == 0.5
    assert review["gpu_hour_cap_keys"] == ["G2", "G3", "G4", "G5", "G6_G7", "pilot"]
