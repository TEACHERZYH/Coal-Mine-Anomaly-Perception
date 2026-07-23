from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from .data.episodes import validate_fusion_eligibility_lock
from .governance.efficiency import (
    EXPECTED_FAMILY_BY_MODULE,
    EXPECTED_GATE_BY_MODULE,
    MODULE_IDS,
    derive_efficiency_summary,
    validate_efficiency_completion,
    validate_efficiency_scope,
    validate_efficiency_trace,
)
from .governance.immutable import write_once_bytes
from .models.methane import MethaneGRU
from .provenance import canonical_json_sha256, sha256_file
from .train.episode import load_episode_model
from .training_data import load_frozen_protocol
from .workflow_common import (
    WorkflowExecutionError,
    load_json,
    write_parquet_artifact,
)


@dataclass(frozen=True)
class Representative:
    module_id: str
    family_id: str
    train_seed: int
    checkpoint: Path
    checkpoint_sha256: str
    policy_sha256: Optional[str]


@dataclass(frozen=True)
class Benchmark:
    representative: Representative
    model: nn.Module
    invoke: Callable[[], Any]
    input_shape: str
    parameter_count: int


def select_representative(
    records: Sequence[Mapping[str, Any]],
    *,
    family_id: str,
    expected_count: int,
    selection_pool: str,
) -> Dict[str, Any]:
    candidates = [dict(item) for item in records if item.get("family_id") == family_id]
    if len(candidates) != expected_count:
        raise WorkflowExecutionError(
            f"{family_id} efficiency selection requires {expected_count} locked candidates"
        )
    for item in candidates:
        if item.get("status") != "pass" or item.get("selection_pool") != selection_pool:
            raise WorkflowExecutionError(f"{family_id} efficiency candidate is not eligible")
        metric = float(item.get("selection_metric_value", np.nan))
        if not np.isfinite(metric) or int(item.get("train_seed", 0)) <= 0:
            raise WorkflowExecutionError(f"{family_id} efficiency validation evidence is invalid")
    ranked = sorted(
        candidates,
        key=lambda item: (
            -float(item["selection_metric_value"]),
            int(item["train_seed"]),
            int(item.get("subset_seed") or 0),
        ),
    )
    return ranked[len(ranked) // 2]


def _read_manifests(paths: Sequence[Path]) -> list[Dict[str, Any]]:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise WorkflowExecutionError(f"missing locked manifest: {missing[0]}")
    return [load_json(path) for path in paths]


def _representative_for_module(project_root: Path, module_id: str) -> Representative:
    family = EXPECTED_FAMILY_BY_MODULE[module_id]
    if module_id == "thermal_inference":
        paths = [project_root / "runs/T1/T1-THERM/seed-1701/run_manifest.json"]
        selected = select_representative(
            _read_manifests(paths),
            family_id=family,
            expected_count=1,
            selection_pool="D_b_sel",
        )
    elif module_id == "visible_inference":
        paths = sorted(
            (project_root / "runs/V2/finetune/V2-A-10-MULTI").glob(
                "seed-*/run_manifest.json"
            )
        )
        selected = select_representative(
            _read_manifests(paths),
            family_id=family,
            expected_count=3,
            selection_pool="D_b_sel",
        )
    elif module_id == "sensor_inference":
        paths = sorted(
            (project_root / "runs/S1/gru/S1-GRU").glob("seed-*/run_manifest.json")
        )
        selected = select_representative(
            _read_manifests(paths),
            family_id=family,
            expected_count=3,
            selection_pool="D_b_sel",
        )
    elif module_id == "graph_fusion":
        paths = sorted(
            (project_root / "runs/E2_E3/E3-FULL").glob("seed-*/run_manifest.json")
        )
        selected = select_representative(
            _read_manifests(paths),
            family_id=family,
            expected_count=3,
            selection_pool="D_e_sel",
        )
    else:
        raise WorkflowExecutionError(f"Unknown efficiency module: {module_id}")
    checkpoint = project_root / str(selected.get("checkpoint_path", ""))
    checkpoint_hash = str(selected.get("checkpoint_sha256", ""))
    if not checkpoint.is_file() or sha256_file(checkpoint) != checkpoint_hash:
        raise WorkflowExecutionError(f"Efficiency checkpoint drifted for {module_id}")
    policy_hash = None
    if module_id == "graph_fusion":
        policy_path = (
            project_root
            / f"runs/E2_E3/policies/E3-FULL-seed-{int(selected['train_seed'])}.json"
        )
        policy = load_json(policy_path)
        policy_hash = str(policy.get("policy_sha256", ""))
        canonical = dict(policy)
        canonical.pop("policy_sha256", None)
        if policy.get("status") != "pass" or canonical_json_sha256(canonical) != policy_hash:
            raise WorkflowExecutionError("E3-FULL efficiency policy hash drifted")
    return Representative(
        module_id=module_id,
        family_id=family,
        train_seed=int(selected["train_seed"]),
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_hash,
        policy_sha256=policy_hash,
    )


def _gate_receipts(project_root: Path) -> Dict[str, Dict[str, Any]]:
    receipts = {}
    for gate_id in {"G2", "G3", "G5"}:
        path = project_root / f"evidence/gates/{gate_id}.json"
        payload = load_json(path)
        receipts[gate_id] = {"status": payload.get("status"), "sha256": sha256_file(path)}
    return receipts


def _graph_population_nonempty(project_root: Path) -> bool:
    lock = validate_fusion_eligibility_lock(
        pd.read_parquet(project_root / "data/locked/fusion_eligibility_lock.parquet")
    )
    return bool(lock["graph_primary_eligible"].astype(bool).any())


def _build_scope(
    project_root: Path,
    *,
    protocol_sha256: str,
    gate_receipts: Mapping[str, Mapping[str, Any]],
    graph_nonempty: bool,
) -> tuple[pd.DataFrame, Dict[str, Representative]]:
    rows = []
    representatives = {}
    for module_id in MODULE_IDS:
        gate_id = EXPECTED_GATE_BY_MODULE[module_id]
        gate = gate_receipts[gate_id]
        reason = None
        representative = None
        if gate["status"] != "pass":
            reason = f"source_gate_{gate_id}_{gate['status']}"
        elif module_id == "graph_fusion" and not graph_nonempty:
            reason = "graph_primary_population_empty"
        else:
            try:
                representative = _representative_for_module(project_root, module_id)
            except (FileNotFoundError, WorkflowExecutionError) as exc:
                if "requires" not in str(exc) and "missing" not in str(exc).lower():
                    raise
                reason = f"locked_checkpoint_unavailable:{type(exc).__name__}"
        if representative is not None:
            representatives[module_id] = representative
        rows.append(
            {
                "module_id": module_id,
                "family_id": EXPECTED_FAMILY_BY_MODULE[module_id],
                "eligibility_status": "measured" if representative else "not_applicable",
                "source_gate_id": gate_id,
                "source_gate_sha256": gate["sha256"],
                "checkpoint_sha256": (
                    representative.checkpoint_sha256 if representative else None
                ),
                "exclusion_reason": reason,
                "protocol_sha256": protocol_sha256,
            }
        )
    scope = pd.DataFrame.from_records(rows)
    validate_efficiency_scope(
        scope,
        graph_primary_population_nonempty=graph_nonempty,
        gate_receipts=gate_receipts,
    )
    return scope, representatives


def _detection_benchmark(
    representative: Representative, device: torch.device
) -> Benchmark:
    from ultralytics import YOLO

    model = YOLO(str(representative.checkpoint)).model.to(device).eval()
    sample = torch.linspace(
        0.0, 1.0, 3 * 640 * 640, device=device, dtype=torch.float32
    ).reshape(1, 3, 640, 640)
    return Benchmark(
        representative=representative,
        model=model,
        invoke=lambda: model(sample),
        input_shape=json.dumps(list(sample.shape), separators=(",", ":")),
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
    )


def _sensor_benchmark(
    representative: Representative,
    device: torch.device,
    protocol: Mapping[str, Any],
) -> Benchmark:
    checkpoint = torch.load(representative.checkpoint, map_location="cpu", weights_only=False)
    feature_names = tuple(checkpoint["feature_names"])
    model = MethaneGRU(feature_names)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device).eval()
    methane = protocol["data"]["methane"]
    history_steps = int(methane["history_seconds"]) // int(methane["stride_seconds"])
    sample = torch.zeros((1, history_steps, len(feature_names)), device=device)
    return Benchmark(
        representative=representative,
        model=model,
        invoke=lambda: model.probabilities(sample),
        input_shape=json.dumps(list(sample.shape), separators=(",", ":")),
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
    )


def _graph_benchmark(
    representative: Representative, device: torch.device
) -> Benchmark:
    model, checkpoint = load_episode_model(representative.checkpoint, device=device)
    node_count = len(checkpoint["node_ids"])
    concept_count = len(checkpoint["concept_ids"])
    quality_count = len(checkpoint["quality_feature_names"])
    pair_count = len(checkpoint["pair_feature_names"])
    edge_pairs = [
        (source, target)
        for source in range(node_count)
        for target in range(node_count)
        if source != target
    ]
    edge_index = torch.tensor(edge_pairs, dtype=torch.long, device=device).T.contiguous()
    steps = 32
    probabilities = torch.full((steps, node_count), 0.5, device=device)
    quality = torch.zeros((steps, node_count, quality_count), device=device)
    availability = torch.ones((steps, node_count), dtype=torch.bool, device=device)
    concepts = torch.zeros((steps, concept_count), device=device)
    concepts[:, 0] = 1.0
    pair_features = torch.ones((steps, len(edge_pairs), pair_count), device=device)
    edge_validity = torch.ones((steps, len(edge_pairs)), dtype=torch.bool, device=device)

    def invoke() -> Any:
        return model(
            probabilities,
            quality,
            availability,
            concepts,
            edge_index=edge_index,
            pair_features=pair_features,
            edge_validity=edge_validity,
            abstention_threshold=0.10,
        )

    shape = [1, steps, node_count, quality_count, len(edge_pairs), pair_count]
    return Benchmark(
        representative=representative,
        model=model,
        invoke=invoke,
        input_shape=json.dumps(shape, separators=(",", ":")),
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
    )


def _gpu_metadata() -> Dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip() or "0"
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            visible,
            "--query-gpu=name,uuid,clocks.sm,driver_version",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    if result.returncode != 0 or len(result.stdout.strip().splitlines()) != 1:
        raise WorkflowExecutionError("E400 could not capture locked GPU metadata")
    name, uuid, clock, driver = [value.strip() for value in result.stdout.split(",", 3)]
    return {
        "device_identifier": name,
        "gpu_uuid": uuid,
        "gpu_clock_mhz": float(clock),
        "cuda_driver": driver,
    }


def _concurrent_gpu_jobs() -> int:
    result = subprocess.run(
        ["squeue", "-h", "-u", os.environ.get("USER", ""), "-t", "RUNNING", "-o", "%i|%b"],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    if result.returncode != 0:
        raise WorkflowExecutionError("E400 could not audit concurrent Slurm jobs")
    current = os.environ.get("SLURM_JOB_ID", "")
    count = 0
    for line in result.stdout.splitlines():
        job_id, _, gres = line.partition("|")
        if job_id.split("_", 1)[0] == current or gres in {"", "(null)", "N/A"}:
            continue
        count += 1
    return count


def _cpu_affinity() -> str:
    getter = getattr(os, "sched_getaffinity", None)
    if getter is None:
        return "unavailable"
    return ",".join(str(value) for value in sorted(getter(0)))


def _measure_benchmark(
    benchmark: Benchmark,
    *,
    precision: str,
    common: Mapping[str, Any],
) -> tuple[list[Dict[str, Any]], int]:
    enabled_amp = precision == "amp_fp16"
    if precision not in {"fp32", "amp_fp16"}:
        raise WorkflowExecutionError(f"Unsupported locked efficiency precision: {precision}")
    rows = []
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for phase, count in (("warmup", 50), ("timed", 200)):
            for iteration in range(1, count + 1):
                torch.cuda.synchronize()
                started = time.perf_counter_ns()
                with torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=enabled_amp
                ):
                    benchmark.invoke()
                torch.cuda.synchronize()
                duration = time.perf_counter_ns() - started
                if duration <= 0:
                    raise WorkflowExecutionError("E400 measured a nonpositive duration")
                rows.append(
                    {
                        **common,
                        "measurement_id": canonical_json_sha256(
                            {
                                "module_id": benchmark.representative.module_id,
                                "checkpoint_sha256": benchmark.representative.checkpoint_sha256,
                                "allocation_id": common["allocation_id"],
                            }
                        ),
                        "module_id": benchmark.representative.module_id,
                        "variant_id": (
                            f"{benchmark.representative.family_id}-seed-"
                            f"{benchmark.representative.train_seed}"
                        ),
                        "model_family_id": benchmark.representative.family_id,
                        "representative_seed": benchmark.representative.train_seed,
                        "checkpoint_sha256": benchmark.representative.checkpoint_sha256,
                        "policy_sha256": benchmark.representative.policy_sha256,
                        "repeat_id": 1,
                        "phase": phase,
                        "iteration_index": iteration,
                        "duration_ns": duration,
                        "precision": precision,
                        "batch_size": 1,
                        "input_shape": benchmark.input_shape,
                        "captured_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
    return rows, int(torch.cuda.max_memory_allocated())


def measure_efficiency(
    project_root: Path, run_root: Path, config_path: Optional[Path]
) -> Dict[str, Any]:
    del config_path
    probe = run_root / "slurm_environment_probe.json"
    if not probe.is_file() or not torch.cuda.is_available():
        raise WorkflowExecutionError("E400 requires its Slurm probe and CUDA")
    protocol_path = project_root / "configs/protocol_lock.pretest.yaml"
    protocol = load_frozen_protocol(project_root)
    efficiency = protocol["efficiency"]
    if (
        int(efficiency["batch_size"]) != 1
        or int(efficiency["warmup_iterations"]) != 50
        or int(efficiency["timed_iterations"]) != 200
        or int(efficiency["independent_repeats"]) != 1
    ):
        raise WorkflowExecutionError("E400 iteration contract drifted")
    environment_path = project_root / "evidence/preimplementation/remote_environment_ready.json"
    environment = load_json(environment_path)
    if environment.get("status") != "pass" or environment.get("scope") != "remote":
        raise WorkflowExecutionError("E400 remote environment receipt is invalid")
    protocol_hash = sha256_file(protocol_path)
    gates = _gate_receipts(project_root)
    graph_nonempty = _graph_population_nonempty(project_root)
    scope, representatives = _build_scope(
        project_root,
        protocol_sha256=protocol_hash,
        gate_receipts=gates,
        graph_nonempty=graph_nonempty,
    )
    if not representatives:
        raise WorkflowExecutionError("E400 must not be submitted when no module is eligible")
    concurrent = _concurrent_gpu_jobs()
    if concurrent != 0:
        raise WorkflowExecutionError("E400 forbids concurrent user GPU jobs")
    device = torch.device("cuda:0")
    gpu = _gpu_metadata()
    job_id = os.environ.get("SLURM_JOB_ID", "")
    allocation_id = os.environ.get("SLURM_ARRAY_JOB_ID", job_id)
    common = {
        "environment_ready_sha256": sha256_file(environment_path),
        "protocol_sha256": protocol_hash,
        "slurm_job_id": job_id,
        "allocation_id": allocation_id,
        "compute_node": platform.node(),
        "device_kind": "cuda",
        **gpu,
        "cpu_affinity": _cpu_affinity(),
        "cuda_synchronized_before_and_after": True,
        "concurrent_user_gpu_job_count": 0,
        "interference_audit_status": "pass",
    }
    benchmarks = []
    for module_id, representative in representatives.items():
        if module_id in {"visible_inference", "thermal_inference"}:
            benchmark = _detection_benchmark(representative, device)
        elif module_id == "sensor_inference":
            benchmark = _sensor_benchmark(representative, device, protocol)
        else:
            benchmark = _graph_benchmark(representative, device)
        benchmarks.append(benchmark)
    rows = []
    peak_memory = {}
    parameter_count = {}
    checkpoint_bytes = {}
    for benchmark in benchmarks:
        measured, peak = _measure_benchmark(
            benchmark, precision=str(efficiency["precision"]), common=common
        )
        rows.extend(measured)
        module_id = benchmark.representative.module_id
        peak_memory[module_id] = peak
        parameter_count[module_id] = benchmark.parameter_count
        checkpoint_bytes[module_id] = benchmark.representative.checkpoint.stat().st_size
    trace = pd.DataFrame.from_records(rows)
    validate_efficiency_trace(trace, scope)
    scope_path = project_root / "results/C1/efficiency_scope.csv"
    write_once_bytes(scope_path, scope.to_csv(index=False).encode("utf-8"))
    trace_path = project_root / "evidence/g6/efficiency_trace.parquet"
    write_parquet_artifact(trace_path, trace)
    summary = derive_efficiency_summary(
        trace,
        trace_sha256=sha256_file(trace_path),
        derivation_code_sha256=sha256_file(Path(__file__)),
        peak_memory_bytes=peak_memory,
        parameter_count=parameter_count,
        checkpoint_bytes=checkpoint_bytes,
    )
    validate_efficiency_completion(scope, trace, summary)
    summary_path = project_root / "evidence/g6/efficiency_summary.parquet"
    write_parquet_artifact(summary_path, summary)
    result_path = project_root / "results/C1/efficiency.parquet"
    write_once_bytes(result_path, summary_path.read_bytes())
    return {
        "status": "pass",
        "output_paths": [
            scope_path.relative_to(project_root).as_posix(),
            trace_path.relative_to(project_root).as_posix(),
            summary_path.relative_to(project_root).as_posix(),
            result_path.relative_to(project_root).as_posix(),
        ],
        "details": {
            "measured_modules": sorted(representatives),
            "not_applicable_modules": sorted(set(MODULE_IDS).difference(representatives)),
            "timed_iterations_per_module": 200,
        },
    }
