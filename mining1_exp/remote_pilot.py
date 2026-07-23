from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields, is_dataclass
import json
import math
import os
from pathlib import Path
import platform
import time
from typing import Any, Callable, Dict, Mapping, Optional

import torch
from torch import nn
import yaml

from .governance.immutable import write_once_json
from .minimal_pipeline import run_minimal_pipeline
from .models.episode_fusion import ReliabilityGraphFusion
from .models.methane import MethaneGRU
from .models.yolo_adapter import build_yolov8n_model
from .provenance import sha256_file
from .training_data import load_frozen_protocol
from .workflow_common import WorkflowExecutionError


@dataclass(frozen=True)
class _PilotStep:
    output: Any
    loss: torch.Tensor


def _require_cuda() -> int:
    if not torch.cuda.is_available():
        raise WorkflowExecutionError("Remote GPU step requires CUDA")
    count = int(torch.cuda.device_count())
    if count <= 0:
        raise WorkflowExecutionError("CUDA reports no visible GPU")
    return count


def _tensor_digest(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if torch.is_tensor(item):
            digest.update(item.detach().float().cpu().contiguous().numpy().tobytes())
        elif is_dataclass(item) and not isinstance(item, type):
            for field in fields(item):
                visit(getattr(item, field.name))
        elif isinstance(item, Mapping):
            for key in sorted(item):
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return digest.hexdigest()


def _tensor_loss(value: Any) -> torch.Tensor:
    tensors = []

    def visit(item: Any) -> None:
        if torch.is_tensor(item) and item.is_floating_point():
            tensors.append(item.float().square().mean())
        elif is_dataclass(item) and not isinstance(item, type):
            for field in fields(item):
                visit(getattr(item, field.name))
        elif isinstance(item, Mapping):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    if not tensors:
        raise WorkflowExecutionError("Pilot model produced no floating-point tensor")
    return torch.stack(tensors).sum()


def _prepare_primary_cuda_memory_stats(device: torch.device) -> int:
    if device.type != "cuda":
        raise WorkflowExecutionError("Pilot memory accounting requires a CUDA device")
    device_index = (
        int(device.index) if device.index is not None else int(torch.cuda.current_device())
    )
    torch.cuda.set_device(device_index)
    torch.cuda.empty_cache()
    # Both locked PyTorch builds require the current-device form of this API.
    torch.cuda.reset_peak_memory_stats()
    return device_index


def _build_grad_scaler(precision: str) -> torch.amp.GradScaler:
    if precision not in {"fp32", "amp_fp16"}:
        raise WorkflowExecutionError(f"Unsupported pilot precision: {precision}")
    return torch.amp.GradScaler("cuda", enabled=precision == "amp_fp16")


def _emit_pilot_event(event: str, **details: Any) -> None:
    print(
        json.dumps({"pilot_event": event, **details}, sort_keys=True),
        flush=True,
    )


def remote_smoke(
    project_root: Path,
    run_root: Path,
    config_path: Optional[Path],
) -> Dict[str, Any]:
    del config_path
    gpu_count = _require_cuda()
    synthetic_root = run_root / "synthetic_contract_pipeline"
    minimal = run_minimal_pipeline(synthetic_root)
    device = torch.device("cuda:0")
    torch.manual_seed(1701)
    torch.cuda.manual_seed_all(1701)
    model = build_yolov8n_model().to(device).eval()
    sample = torch.linspace(0.0, 1.0, 3 * 64 * 64, device=device).reshape(1, 3, 64, 64)
    with torch.inference_mode():
        output = model(sample)
    torch.cuda.synchronize(device)
    if not math.isfinite(float(_tensor_loss(output).detach().cpu())):
        raise WorkflowExecutionError("Remote smoke produced a non-finite YOLO output")
    probe = run_root / "slurm_environment_probe.json"
    if not probe.is_file():
        raise WorkflowExecutionError("Remote smoke lacks its Slurm environment probe")
    target = project_root / "evidence/smoke/remote_smoke.json"
    write_once_json(
        target,
        {
            "schema_version": 1,
            "step_id": "E064",
            "status": "pass",
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "compute_node": os.environ.get("SLURMD_NODENAME") or platform.node(),
            "cuda_device_count": gpu_count,
            "cuda_device_names": [torch.cuda.get_device_name(index) for index in range(gpu_count)],
            "gpu_tensor_path_exercised": True,
            "yolov8n_output_sha256": _tensor_digest(output),
            "minimal_pipeline_status": minimal.get("status"),
            "minimal_pipeline_receipt": str(
                (synthetic_root / "integration_receipt.json").relative_to(project_root)
            ).replace("\\", "/"),
            "minimal_pipeline_receipt_sha256": sha256_file(
                synthetic_root / "integration_receipt.json"
            ),
            "environment_probe_sha256": sha256_file(probe),
            "training_performed": False,
            "performance_claims_authorized": False,
        },
    )
    return {
        "status": "pass",
        "output_paths": [target.relative_to(project_root).as_posix()],
        "details": {"cuda_device_count": gpu_count},
    }


def _benchmark(
    factory: Callable[[torch.device], tuple[nn.Module, Callable[[], Any]]],
    *,
    device: torch.device,
    precision: str,
    warmup: int,
    timed: int,
    examples_per_update: int,
) -> Dict[str, Any]:
    if examples_per_update <= 0:
        raise WorkflowExecutionError("Pilot examples_per_update must be positive")
    _prepare_primary_cuda_memory_stats(device)
    torch.manual_seed(1701)
    torch.cuda.manual_seed_all(1701)
    model, forward = factory(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-4)
    use_amp = precision == "amp_fp16"
    scaler = _build_grad_scaler(precision)

    def one() -> Any:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            step = forward()
            if isinstance(step, _PilotStep):
                output = step.output
                loss = step.loss.float()
            else:
                output = step
                loss = _tensor_loss(output)
        if not torch.isfinite(loss):
            raise WorkflowExecutionError("Pilot loss is not finite")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return output

    for _ in range(warmup):
        one()
    torch.cuda.synchronize()
    started = time.perf_counter()
    last = None
    for _ in range(timed):
        last = one()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "status": "pass",
        "precision": precision,
        "warmup_updates": warmup,
        "timed_updates": timed,
        "wall_time_seconds": elapsed,
        "updates_per_second": timed / elapsed,
        "examples_per_update": examples_per_update,
        "examples_per_second": timed * examples_per_update / elapsed,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "numerically_finite": True,
        "gradient_scaler_enabled": use_amp,
        "deterministic_repeat_hash": _tensor_digest(last),
    }


def _detection_factory(batch: int) -> Callable[[torch.device], tuple[nn.Module, Callable[[], Any]]]:
    def factory(device: torch.device) -> tuple[nn.Module, Callable[[], Any]]:
        model = build_yolov8n_model().to(device)
        sample = torch.rand(batch, 3, 640, 640, device=device)
        return model, lambda: model(sample)

    return factory


def _methane_factory(batch: int) -> Callable[[torch.device], tuple[nn.Module, Callable[[], Any]]]:
    def factory(device: torch.device) -> tuple[nn.Module, Callable[[], Any]]:
        names = ("ch4_value", "missing_mask", "temperature")
        model = MethaneGRU(names, hidden_size=64, layers=1, dropout=0.20).to(device)
        sample = torch.rand(batch, 30, len(names), device=device)
        return model, lambda: model(sample)

    return factory


def _episode_factory(batch: int) -> Callable[[torch.device], tuple[nn.Module, Callable[[], Any]]]:
    def factory(device: torch.device) -> tuple[nn.Module, Callable[[], Any]]:
        model = ReliabilityGraphFusion(
            node_count=3,
            quality_feature_names=("quality_signal", "missing_mask"),
            concept_dim=4,
            pair_feature_names=("pair_alignment",),
            graph_layers=2,
            use_graph=False,
            use_reliability=True,
        ).to(device)
        probabilities = torch.rand(batch, 3, device=device).clamp(0.05, 0.95)
        quality = torch.rand(batch, 3, 2, device=device)
        availability = torch.ones(batch, 3, dtype=torch.bool, device=device)
        concept = torch.rand(batch, 4, device=device)
        targets = (torch.arange(batch, device=device) % 2).float()
        reliability_targets = torch.full((batch, 3), 0.75, device=device)

        def forward() -> _PilotStep:
            output = model(
                probabilities,
                quality,
                availability,
                concept,
                abstention_threshold=0.50,
            )
            with torch.autocast(device_type="cuda", enabled=False):
                loss = torch.nn.functional.binary_cross_entropy(
                    output.probability.float(),
                    targets,
                ) + torch.nn.functional.mse_loss(
                    output.reliability.float(),
                    reliability_targets,
                )
            return _PilotStep(output=output, loss=loss)

        return model, forward

    return factory


def _detection_scaling_factory(
    per_device_batch: int,
    gpu_count: int,
) -> Callable[[torch.device], tuple[nn.Module, Callable[[], Any]]]:
    if per_device_batch <= 0 or gpu_count not in {1, 2}:
        raise WorkflowExecutionError("Detection scaling requires one or two GPUs and a positive batch")

    def factory(device: torch.device) -> tuple[nn.Module, Callable[[], Any]]:
        model: nn.Module = build_yolov8n_model().to(device)
        if gpu_count == 2:
            model = nn.DataParallel(model, device_ids=[0, 1], output_device=0)
        sample = torch.rand(per_device_batch * gpu_count, 3, 640, 640, device=device)
        return model, lambda: model(sample)

    return factory


def _summarize_scaling_results(
    records: list[Dict[str, Any]],
    compare_gpu_counts: tuple[int, ...],
) -> Dict[str, Any]:
    summaries: Dict[str, Any] = {}
    for gpu_count in compare_gpu_counts:
        selected = [item for item in records if int(item["gpu_count"]) == gpu_count]
        if len(selected) != 2:
            raise WorkflowExecutionError("Each GPU-count scaling trial requires two repeats")
        hashes = [str(item["deterministic_repeat_hash"]) for item in selected]
        summaries[str(gpu_count)] = {
            "gpu_count": gpu_count,
            "repeat_count": 2,
            "deterministic_repeat_pass": len(set(hashes)) == 1,
            "mean_examples_per_second": sum(
                float(item["examples_per_second"]) for item in selected
            )
            / 2.0,
            "mean_wall_time_seconds": sum(
                float(item["wall_time_seconds"]) for item in selected
            )
            / 2.0,
            "peak_memory_bytes_per_primary_device": max(
                int(item["peak_memory_bytes"]) for item in selected
            ),
        }
    one = summaries.get("1")
    two = summaries.get("2")
    if one is None or two is None:
        raise WorkflowExecutionError("Frozen scaling trial must compare one and two GPUs")
    speedup = float(two["mean_examples_per_second"]) / float(
        one["mean_examples_per_second"]
    )
    reproducibility_pass = all(
        bool(item["deterministic_repeat_pass"]) for item in summaries.values()
    )
    return {
        "status": "pass",
        "compare_gpu_counts": list(compare_gpu_counts),
        "summaries": summaries,
        "two_gpu_throughput_speedup": speedup,
        "two_gpu_parallel_efficiency": speedup / 2.0,
        "reproducibility_pass": reproducibility_pass,
        "measured_speedup_gt_one": speedup > 1.0,
    }


def _run_detection_scaling_trial(
    *,
    device: torch.device,
    precision: str,
    per_device_batch: int,
    compare_gpu_counts: tuple[int, ...],
    available_gpu_count: int,
    warmup: int,
    timed: int,
) -> Dict[str, Any]:
    if compare_gpu_counts != (1, 2) or available_gpu_count < 2:
        raise WorkflowExecutionError(
            "E066 requires the frozen one-versus-two-GPU scaling allocation"
        )
    records: list[Dict[str, Any]] = []
    for gpu_count in compare_gpu_counts:
        for repeat_index in range(2):
            _emit_pilot_event(
                "scaling_repeat_start",
                gpu_count=gpu_count,
                precision=precision,
                repeat_index=repeat_index,
            )
            try:
                result = _benchmark(
                    _detection_scaling_factory(per_device_batch, gpu_count),
                    device=device,
                    precision=precision,
                    warmup=warmup,
                    timed=timed,
                    examples_per_update=per_device_batch * gpu_count,
                )
            except Exception as exc:
                raise WorkflowExecutionError(
                    "Detection scaling failed for "
                    f"gpu_count={gpu_count}, precision={precision}, "
                    f"repeat_index={repeat_index}: {exc}"
                ) from exc
            result["gpu_count"] = gpu_count
            result["repeat_index"] = repeat_index
            records.append(result)
            _emit_pilot_event(
                "scaling_repeat_complete",
                gpu_count=gpu_count,
                precision=precision,
                repeat_index=repeat_index,
                numerically_finite=result["numerically_finite"],
            )
    summary = _summarize_scaling_results(records, compare_gpu_counts)
    summary["records"] = records
    summary["precision"] = precision
    summary["per_device_batch"] = per_device_batch
    return summary


def _run_with_one_oom_fallback(
    factory_builder: Callable[[int], Callable[[torch.device], tuple[nn.Module, Callable[[], Any]]]],
    *,
    initial_batch: int,
    device: torch.device,
    precision: str,
    warmup: int,
    timed: int,
) -> tuple[int, Dict[str, Any]]:
    batch = int(initial_batch)
    for attempt in range(2):
        try:
            return batch, _benchmark(
                factory_builder(batch),
                device=device,
                precision=precision,
                warmup=warmup,
                timed=timed,
                examples_per_update=batch,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if attempt or batch <= 1:
                raise
            batch = max(1, batch // 2)
    raise WorkflowExecutionError("Pilot OOM fallback did not resolve")


def _resource_cap_decisions() -> tuple[Dict[str, float], Dict[str, Any]]:
    package_caps = {
        "T1": 24.0,
        "S1": 24.0,
        "V2": 160.0,
        "E2_E3": 48.0,
        "R1": 12.0,
        "C1": 8.0,
    }
    gate_caps = {
        "pilot": 8.0,
        "G2": package_caps["T1"] + package_caps["S1"],
        "G3": package_caps["V2"],
        "G4": package_caps["R1"],
        "G5": package_caps["E2_E3"],
        "G6_G7": package_caps["C1"],
    }
    return gate_caps, {
        "package_caps_gpu_hours": package_caps,
        "gate_aggregation": {
            "pilot": "two_complete_two_gpu_two_hour_attempts",
            "G2": "T1_plus_S1",
            "G3": "V2",
            "G4": "R1",
            "G5": "E2_E3",
            "G6_G7": "C1",
        },
    }


def resource_pilot(
    project_root: Path,
    run_root: Path,
    config_path: Optional[Path],
) -> Dict[str, Any]:
    del config_path
    gpu_count = _require_cuda()
    protocol_path = project_root / "configs/protocol_lock.template.yaml"
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8-sig"))
    pilot = protocol["pilot"]
    device = torch.device("cuda:0")
    warmup = int(pilot["scaling_trial"]["warmup_updates"])
    timed = int(pilot["scaling_trial"]["timed_updates"])
    family_builders = {
        "visible": _detection_factory,
        "rgbt": _detection_factory,
        "methane": _methane_factory,
        "episode": _episode_factory,
    }
    benchmarks = []
    chosen_batches = {}
    precision_candidates = ("fp32", "amp_fp16")
    for family, builder in family_builders.items():
        family_results = []
        final_batch = None
        for precision in precision_candidates:
            _emit_pilot_event(
                "family_benchmark_start",
                family=family,
                precision=precision,
            )
            try:
                batch, result = _run_with_one_oom_fallback(
                    builder,
                    initial_batch=int(pilot["initial_batch_size"][family]),
                    device=device,
                    precision=precision,
                    warmup=warmup,
                    timed=timed,
                )
            except Exception as exc:
                raise WorkflowExecutionError(
                    f"Pilot benchmark failed for family={family}, "
                    f"precision={precision}: {exc}"
                ) from exc
            result["family"] = family
            result["micro_batch_size"] = batch
            family_results.append(result)
            final_batch = batch if final_batch is None else min(final_batch, batch)
            _emit_pilot_event(
                "family_benchmark_complete",
                family=family,
                precision=precision,
                micro_batch_size=batch,
                numerically_finite=result["numerically_finite"],
            )
        benchmarks.extend(family_results)
        effective_batch = {
            "visible": 64,
            "rgbt": 64,
            "methane": 128,
            "episode": 64,
        }[family]
        divisors = [
            candidate
            for candidate in range(1, int(final_batch) + 1)
            if effective_batch % candidate == 0
        ]
        chosen_batches[family] = max(divisors)
    visible = [item for item in benchmarks if item["family"] == "visible"]
    fp32 = next(item for item in visible if item["precision"] == "fp32")
    amp = next(item for item in visible if item["precision"] == "amp_fp16")
    chosen_precision = (
        "amp_fp16"
        if amp["numerically_finite"]
        and (
            amp["updates_per_second"] >= fp32["updates_per_second"]
            or amp["peak_memory_bytes"] < fp32["peak_memory_bytes"]
        )
        else "fp32"
    )
    compare_gpu_counts = tuple(
        int(value) for value in pilot["scaling_trial"]["compare_gpu_counts"]
    )
    scaling_trial = _run_detection_scaling_trial(
        device=device,
        precision=chosen_precision,
        per_device_batch=chosen_batches["visible"],
        compare_gpu_counts=compare_gpu_counts,
        available_gpu_count=gpu_count,
        warmup=warmup,
        timed=timed,
    )
    effective = {
        "visible": 64,
        "rgbt": 64,
        "methane": 128,
        "episode": 64,
    }
    accumulation = {
        family: int(effective[family] // batch)
        for family, batch in chosen_batches.items()
    }
    if any(
        chosen_batches[family] * accumulation[family] != effective[family]
        for family in effective
    ):
        raise WorkflowExecutionError("Pilot decisions changed a locked effective batch size")
    gpu_hour_caps, resource_cap_basis = _resource_cap_decisions()
    decisions = {
        "precision": chosen_precision,
        "workers": int(pilot["initial_workers"]),
        "micro_batch_size": chosen_batches,
        "gradient_accumulation": accumulation,
        "checkpoint_interval_updates": 500,
        "fixed_cpu_affinity": "slurm_assigned_cpus",
        "max_array_concurrency": 1,
        "multi_gpu_training_enabled": False,
        "multi_gpu_decision_reason": (
            "single_gpu_default_even_after_scaling_probe_because_equal_price_cards_"
            "require_superlinear_speedup_for_lower_gpu_cost_and_no_deadline_need_is_locked"
        ),
        "gpu_hour_caps": gpu_hour_caps,
    }
    target = project_root / "evidence/pilot/resource_pilot.json"
    write_once_json(
        target,
        {
            "schema_version": 1,
            "step_id": "E066",
            "status": "pass",
            "result_role": "diagnostic_only",
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "cuda_device_count": gpu_count,
            "cuda_device_names": [torch.cuda.get_device_name(index) for index in range(gpu_count)],
            "benchmarks": benchmarks,
            "scaling_trial": scaling_trial,
            "decisions": decisions,
            "resource_cap_basis": resource_cap_basis,
            "decision_inputs_only": list(pilot["decision_inputs_only"]),
            "forbidden_decisions": list(pilot["forbidden_decisions"]),
            "forbidden_outputs": list(pilot["forbidden_outputs"]),
            "model_or_condition_ranking_performed": False,
            "performance_claims_authorized": False,
            "environment_probe_sha256": sha256_file(
                run_root / "slurm_environment_probe.json"
            ),
        },
    )
    return {
        "status": "pass",
        "output_paths": [target.relative_to(project_root).as_posix()],
        "details": {
            "chosen_precision": chosen_precision,
            "max_array_concurrency": decisions["max_array_concurrency"],
        },
    }
