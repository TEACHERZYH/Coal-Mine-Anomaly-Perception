from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Optional, Sequence

import yaml

from .governance.immutable import write_once_json
from .provenance import canonical_json_sha256, sha256_file
from .workflow_common import WorkflowExecutionError


_FAMILIES = ("visible", "rgbt", "methane", "episode")
_PRECISIONS = ("fp32", "amp_fp16")
_EFFECTIVE_BATCH = {"visible": 64, "rgbt": 64, "methane": 128, "episode": 64}
_CAPS = {
    "pilot": 8.0,
    "G2": 48.0,
    "G3": 160.0,
    "G4": 12.0,
    "G5": 48.0,
    "G6_G7": 8.0,
}
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise WorkflowExecutionError(message)


def _measurement_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: payload[key]
        for key in (
            "slurm_job_id",
            "cuda_device_count",
            "cuda_device_names",
            "benchmarks",
            "scaling_trial",
        )
    }


def review_resource_pilot(project_root: Path, pilot_path: Path) -> Dict[str, Any]:
    payload = json.loads(pilot_path.read_text(encoding="utf-8-sig"))
    protocol = yaml.safe_load(
        (project_root / "configs/protocol_lock.template.yaml").read_text(
            encoding="utf-8-sig"
        )
    )
    _require(payload.get("status") == "pass", "Pilot status is not pass")
    _require(payload.get("step_id") == "E066", "Pilot step ID is not E066")
    _require(payload.get("cuda_device_count") == 2, "Pilot did not receive two GPUs")
    names = payload.get("cuda_device_names")
    _require(isinstance(names, list) and len(names) == 2, "Pilot GPU names are incomplete")

    benchmarks = payload.get("benchmarks")
    _require(isinstance(benchmarks, list) and len(benchmarks) == 8, "Pilot needs 8 benchmarks")
    benchmark_map: Dict[tuple[str, str], Mapping[str, Any]] = {}
    for record in benchmarks:
        key = (str(record.get("family")), str(record.get("precision")))
        _require(key not in benchmark_map, f"Duplicate benchmark: {key}")
        _require(key[0] in _FAMILIES and key[1] in _PRECISIONS, f"Unexpected benchmark: {key}")
        _require(record.get("status") == "pass", f"Benchmark status failed: {key}")
        _require(record.get("numerically_finite") is True, f"Non-finite benchmark: {key}")
        _require(
            record.get("gradient_scaler_enabled") is (key[1] == "amp_fp16"),
            f"Gradient-scaler mismatch: {key}",
        )
        for field in ("wall_time_seconds", "examples_per_second", "updates_per_second"):
            value = float(record.get(field, 0.0))
            _require(math.isfinite(value) and value > 0.0, f"Invalid {field}: {key}")
        _require(int(record.get("peak_memory_bytes", 0)) > 0, f"Invalid memory: {key}")
        _require(_HEX64.fullmatch(str(record.get("deterministic_repeat_hash"))) is not None, f"Invalid hash: {key}")
        benchmark_map[key] = record
    _require(
        set(benchmark_map) == {(family, precision) for family in _FAMILIES for precision in _PRECISIONS},
        "Benchmark Cartesian product is incomplete",
    )

    decisions = payload.get("decisions")
    _require(isinstance(decisions, dict), "Pilot decisions are absent")
    visible_fp32 = benchmark_map[("visible", "fp32")]
    visible_amp = benchmark_map[("visible", "amp_fp16")]
    expected_precision = (
        "amp_fp16"
        if bool(visible_amp["numerically_finite"])
        and (
            float(visible_amp["updates_per_second"])
            >= float(visible_fp32["updates_per_second"])
            or int(visible_amp["peak_memory_bytes"])
            < int(visible_fp32["peak_memory_bytes"])
        )
        else "fp32"
    )
    _require(decisions.get("precision") == expected_precision, "Precision decision drifted")
    micro = decisions.get("micro_batch_size")
    accumulation = decisions.get("gradient_accumulation")
    _require(isinstance(micro, dict) and isinstance(accumulation, dict), "Batch decisions are absent")
    for family in _FAMILIES:
        selected = int(micro.get(family, 0))
        _require(selected > 0, f"Invalid micro-batch: {family}")
        _require(
            selected == min(
                int(benchmark_map[(family, precision)]["micro_batch_size"])
                for precision in _PRECISIONS
            ),
            f"Micro-batch does not match measured feasibility: {family}",
        )
        _require(
            selected * int(accumulation.get(family, 0)) == _EFFECTIVE_BATCH[family],
            f"Effective batch drifted: {family}",
        )
    _require(decisions.get("max_array_concurrency") == 1, "Array concurrency is not conservative")
    _require(decisions.get("multi_gpu_training_enabled") is False, "Multi-GPU training was enabled")
    _require(decisions.get("gpu_hour_caps") == _CAPS, "GPU-hour cap keys or values drifted")

    scaling = payload.get("scaling_trial")
    frozen_scaling = protocol["pilot"]["scaling_trial"]
    _require(isinstance(scaling, dict) and scaling.get("status") == "pass", "Scaling trial failed")
    _require(scaling.get("compare_gpu_counts") == frozen_scaling["compare_gpu_counts"], "Scaling GPU counts drifted")
    records = scaling.get("records")
    _require(isinstance(records, list) and len(records) == 4, "Scaling trial needs four records")
    grouped: Dict[int, list[Mapping[str, Any]]] = {1: [], 2: []}
    for record in records:
        gpu_count = int(record.get("gpu_count", 0))
        _require(gpu_count in grouped, "Unexpected scaling GPU count")
        grouped[gpu_count].append(record)
        _require(record.get("numerically_finite") is True, "Scaling record is non-finite")
        _require(float(record.get("examples_per_second", 0.0)) > 0.0, "Scaling throughput is invalid")
        _require(int(record.get("examples_per_update", 0)) == int(scaling["per_device_batch"]) * gpu_count, "Scaling global batch drifted")
    means: Dict[int, float] = {}
    for gpu_count, selected in grouped.items():
        _require(sorted(int(item["repeat_index"]) for item in selected) == [0, 1], "Scaling repeat indices drifted")
        _require(len({str(item["deterministic_repeat_hash"]) for item in selected}) == 1, "Scaling repeat hashes differ")
        means[gpu_count] = sum(float(item["examples_per_second"]) for item in selected) / 2.0
    speedup = means[2] / means[1]
    _require(math.isclose(float(scaling["two_gpu_throughput_speedup"]), speedup, rel_tol=1e-12), "Scaling speedup is inconsistent")
    _require(math.isclose(float(scaling["two_gpu_parallel_efficiency"]), speedup / 2.0, rel_tol=1e-12), "Parallel efficiency is inconsistent")
    _require(scaling.get("reproducibility_pass") is True, "Scaling reproducibility failed")
    _require(speedup < 1.0, "Reviewer expected the measured two-GPU slowdown")

    _require(payload.get("model_or_condition_ranking_performed") is False, "Model ranking was performed")
    _require(payload.get("performance_claims_authorized") is False, "Performance claims were authorized")
    for key in protocol["pilot"]["forbidden_outputs"]:
        _require(key not in payload, f"Forbidden pilot output present: {key}")

    repair = payload.get("metadata_repair")
    _require(isinstance(repair, dict) and repair.get("status") == "pass", "Metadata repair receipt is absent")
    source = Path(str(repair["source_path"]))
    if not source.is_absolute():
        source = project_root / source
    _require(source.is_file(), "Metadata-repair source artifact is absent")
    _require(sha256_file(source) == repair["source_sha256"], "Metadata-repair source hash drifted")
    source_payload = json.loads(source.read_text(encoding="utf-8-sig"))
    source_measurement_hash = canonical_json_sha256(_measurement_payload(source_payload))
    current_measurement_hash = canonical_json_sha256(_measurement_payload(payload))
    _require(source_measurement_hash == current_measurement_hash, "Metadata repair changed measurements")
    _require(current_measurement_hash == repair["measurement_payload_sha256"], "Measurement receipt hash drifted")

    return {
        "schema_version": 1,
        "step_id": "E066",
        "status": "pass",
        "review_role": "independent_reconciliation",
        "pilot_path": pilot_path.relative_to(project_root).as_posix(),
        "pilot_sha256": sha256_file(pilot_path),
        "source_pilot_sha256": repair["source_sha256"],
        "measurement_payload_sha256": current_measurement_hash,
        "benchmark_count": 8,
        "scaling_record_count": 4,
        "chosen_precision": expected_precision,
        "one_gpu_examples_per_second": means[1],
        "two_gpu_examples_per_second": means[2],
        "two_gpu_speedup": speedup,
        "multi_gpu_training_enabled": False,
        "gpu_hour_cap_keys": sorted(_CAPS),
        "model_or_condition_ranking_performed": False,
        "performance_claims_authorized": False,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Independently review E066")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    review = review_resource_pilot(args.project_root, args.pilot)
    write_once_json(args.output, review)
    print(json.dumps(review, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
