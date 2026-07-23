from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from mining1_exp.models.episode_fusion import FusionOutput
from mining1_exp import remote_pilot
from mining1_exp.remote_pilot import (
    _resource_cap_decisions,
    _summarize_scaling_results,
    _tensor_digest,
    _tensor_loss,
)


def _fusion_output(probability: float) -> FusionOutput:
    return FusionOutput(
        probability=torch.tensor([probability], dtype=torch.float32),
        reliability=torch.tensor([[0.8, 0.7]], dtype=torch.float32),
        reliability_weights=torch.tensor([[0.6, 0.4]], dtype=torch.float32),
        abstained=torch.tensor([False]),
        edge_weights=torch.tensor([[[0.0, 0.5], [0.5, 0.0]]], dtype=torch.float32),
        message_norms=torch.tensor([[0.2, 0.3]], dtype=torch.float32),
    )


def test_pilot_tensor_walkers_cover_fusion_dataclass_fields() -> None:
    first = _fusion_output(0.25)
    second = _fusion_output(0.75)
    assert float(_tensor_loss(first)) > 0.0
    assert _tensor_digest(first) != _tensor_digest(second)


def _scaling_record(gpu_count: int, throughput: float, digest: str) -> dict:
    return {
        "gpu_count": gpu_count,
        "examples_per_second": throughput,
        "wall_time_seconds": 2.0,
        "peak_memory_bytes": 1024,
        "deterministic_repeat_hash": digest,
    }


def test_scaling_summary_requires_stable_one_and_two_gpu_repeats() -> None:
    result = _summarize_scaling_results(
        [
            _scaling_record(1, 10.0, "one"),
            _scaling_record(1, 12.0, "one"),
            _scaling_record(2, 18.0, "two"),
            _scaling_record(2, 20.0, "two"),
        ],
        (1, 2),
    )
    assert result["reproducibility_pass"] is True
    assert result["measured_speedup_gt_one"] is True
    assert result["two_gpu_throughput_speedup"] == 19.0 / 11.0

    unstable = _summarize_scaling_results(
        [
            _scaling_record(1, 10.0, "one-a"),
            _scaling_record(1, 10.0, "one-b"),
            _scaling_record(2, 20.0, "two"),
            _scaling_record(2, 20.0, "two"),
        ],
        (1, 2),
    )
    assert unstable["reproducibility_pass"] is False


def test_prepare_primary_cuda_memory_stats_uses_current_device_api(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        remote_pilot.torch.cuda,
        "set_device",
        lambda index: calls.append(("set", index)),
    )
    monkeypatch.setattr(
        remote_pilot.torch.cuda,
        "empty_cache",
        lambda: calls.append(("empty",)),
    )
    monkeypatch.setattr(
        remote_pilot.torch.cuda,
        "reset_peak_memory_stats",
        lambda: calls.append(("reset",)),
    )
    assert remote_pilot._prepare_primary_cuda_memory_stats(torch.device("cuda:0")) == 0
    assert calls == [("set", 0), ("empty",), ("reset",)]


def test_build_grad_scaler_enables_only_amp_fp16(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        remote_pilot.torch.amp,
        "GradScaler",
        lambda device, enabled: calls.append((device, enabled)) or enabled,
    )
    assert remote_pilot._build_grad_scaler("fp32") is False
    assert remote_pilot._build_grad_scaler("amp_fp16") is True
    assert calls == [("cuda", False), ("cuda", True)]


def test_episode_pilot_uses_finite_main_and_reliability_losses() -> None:
    model, forward = remote_pilot._episode_factory(4)(torch.device("cpu"))
    model.train()
    step = forward()
    assert isinstance(step, remote_pilot._PilotStep)
    assert torch.isfinite(step.loss)
    assert step.output.probability.shape == (4,)
    assert step.output.reliability.shape == (4, 3)


def test_resource_cap_decisions_match_frozen_protocol_gate_keys() -> None:
    caps, basis = _resource_cap_decisions()
    assert caps == {
        "pilot": 8.0,
        "G2": 48.0,
        "G3": 160.0,
        "G4": 12.0,
        "G5": 48.0,
        "G6_G7": 8.0,
    }
    assert basis["gate_aggregation"]["G2"] == "T1_plus_S1"


def test_resource_pilot_records_scaling_without_enabling_multi_gpu(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "project"
    protocol_path = root / "configs/protocol_lock.template.yaml"
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_text(
        yaml.safe_dump(
            {
                "pilot": {
                    "initial_workers": 2,
                    "initial_batch_size": {
                        "visible": 4,
                        "rgbt": 2,
                        "methane": 32,
                        "episode": 64,
                    },
                    "scaling_trial": {
                        "warmup_updates": 1,
                        "timed_updates": 1,
                        "compare_gpu_counts": [1, 2],
                    },
                    "decision_inputs_only": ["throughput"],
                    "forbidden_decisions": ["architecture"],
                    "forbidden_outputs": ["test_metrics"],
                }
            }
        ),
        encoding="utf-8",
    )
    run_root = root / "runs/slurm/E066/fixture"
    run_root.mkdir(parents=True)
    (run_root / "slurm_environment_probe.json").write_text("{}", encoding="ascii")

    monkeypatch.setattr(remote_pilot, "_require_cuda", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: f"gpu-{index}")
    monkeypatch.setattr(
        remote_pilot,
        "_run_with_one_oom_fallback",
        lambda builder, **kwargs: (
            kwargs["initial_batch"],
            {
                "status": "pass",
                "precision": kwargs["precision"],
                "updates_per_second": 2.0,
                "examples_per_second": 8.0,
                "peak_memory_bytes": 1024,
                "numerically_finite": True,
                "deterministic_repeat_hash": "a" * 64,
            },
        ),
    )
    monkeypatch.setattr(
        remote_pilot,
        "_run_detection_scaling_trial",
        lambda **kwargs: {
            "status": "pass",
            "compare_gpu_counts": [1, 2],
            "reproducibility_pass": True,
            "measured_speedup_gt_one": True,
            "two_gpu_throughput_speedup": 1.5,
            "two_gpu_parallel_efficiency": 0.75,
            "records": [],
        },
    )
    result = remote_pilot.resource_pilot(root, run_root, None)
    assert result["status"] == "pass"
    payload = json.loads((root / "evidence/pilot/resource_pilot.json").read_text())
    assert payload["scaling_trial"]["compare_gpu_counts"] == [1, 2]
    assert payload["decisions"]["multi_gpu_training_enabled"] is False
    assert payload["decisions"]["max_array_concurrency"] == 1
