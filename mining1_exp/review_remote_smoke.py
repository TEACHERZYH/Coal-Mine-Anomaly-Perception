from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Optional, Sequence

from .minimal_pipeline import MinimalPipelineError, validate_minimal_pipeline_run


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_RELATIVE_PATHS = (
    "configs/environment_decision.lock.json",
    "evidence/smoke/remote_smoke.json",
    "evidence/smoke/E064_submission.json",
    "evidence/smoke/E064_resource_cost_estimate.json",
    "evidence/smoke/E064_actual_resource_usage.json",
    "evidence/remote_sync/E064_result_sync.json",
    "evidence/remote_closeout/E064-job19644-complete.json",
    "runs/slurm/E064/19644/step_execution_receipt.json",
    "runs/slurm/E064/19644/slurm_environment_probe.json",
    "runs/slurm/E064/19644/exit_code.txt",
    "runs/slurm/E064/19644/slurm_stdout.log",
    "runs/slurm/E064/19644/slurm_stderr.log",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"review input is not an object: {path}")
    return payload


def _expect_equal(errors: list[str], name: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        errors.append(f"{name}: observed={observed!r}, expected={expected!r}")


def _result(
    root: Path,
    errors: Sequence[str],
    paths: Sequence[Path],
    diagnostics: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "step_id": "E064",
        "reviewer": "independent_remote_smoke_reconciler",
        "status": "pass" if not errors else "fail",
        "error_count": len(errors),
        "errors": list(errors),
        "bindings": {
            path.relative_to(root).as_posix(): _sha256_file(path)
            for path in paths
            if path.is_file()
        },
        "diagnostics": dict(diagnostics),
        "training_performed": False,
        "model_outcomes_used": False,
        "performance_claims_authorized": False,
    }


def review_e064(project_root: Path) -> Dict[str, Any]:
    root = Path(project_root).resolve()
    paths = tuple(root / relative for relative in REQUIRED_RELATIVE_PATHS)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    decision, smoke, submission, estimate, usage, sync, closeout, receipt, probe = (
        _load(path) for path in paths[:9]
    )
    exit_path, stdout_path, stderr_path = paths[9:]
    errors: list[str] = []
    job_id = "19644"
    run_root = root / f"runs/slurm/E064/{job_id}"
    synthetic_root = run_root / "synthetic_contract_pipeline"
    synthetic_receipt_path = synthetic_root / "integration_receipt.json"
    if not synthetic_receipt_path.is_file():
        raise FileNotFoundError(synthetic_receipt_path)

    for name, payload in {
        "remote_smoke": smoke,
        "submission": submission,
        "actual_resource_usage": usage,
        "result_sync": sync,
        "step_execution_receipt": receipt,
    }.items():
        _expect_equal(errors, f"{name}.step_id", payload.get("step_id"), "E064")
        _expect_equal(errors, f"{name}.status", payload.get("status"), "pass")
    for name, payload in {
        "remote_smoke": smoke,
        "submission": submission,
        "actual_resource_usage": usage,
        "result_sync": sync,
        "step_execution_receipt": receipt,
        "slurm_environment_probe": probe,
    }.items():
        _expect_equal(errors, f"{name}.slurm_job_id", str(payload.get("slurm_job_id")), job_id)

    _expect_equal(errors, "decision.status", decision.get("status"), "locked")
    _expect_equal(
        errors,
        "probe.selected_remote_python",
        probe.get("selected_remote_python"),
        decision.get("selected_remote_python"),
    )
    _expect_equal(errors, "probe.environment_decision_sha256", probe.get("environment_decision_sha256"), _sha256_file(paths[0]))
    _expect_equal(errors, "probe.resource_kind", probe.get("resource_kind"), "gpu")
    _expect_equal(errors, "probe.compute_node", probe.get("compute_node"), "gpu03")
    if str(probe.get("compute_node", "")).lower() == "mu01":
        errors.append("remote smoke ran on the login node")
    if probe.get("framework_version") != "2.5.1+cu121" or probe.get("cuda_runtime") != "12.1":
        errors.append("remote framework or CUDA runtime drifted from the locked environment")

    _expect_equal(errors, "remote_smoke.compute_node", smoke.get("compute_node"), probe.get("compute_node"))
    _expect_equal(errors, "remote_smoke.environment_probe_sha256", smoke.get("environment_probe_sha256"), _sha256_file(paths[8]))
    _expect_equal(errors, "remote_smoke.cuda_device_count", smoke.get("cuda_device_count"), 1)
    names = smoke.get("cuda_device_names")
    if names != ["NVIDIA GeForce RTX 3090"]:
        errors.append(f"unexpected remote CUDA device names: {names!r}")
    if smoke.get("gpu_tensor_path_exercised") is not True:
        errors.append("GPU tensor path was not exercised")
    if not SHA256_PATTERN.fullmatch(str(smoke.get("yolov8n_output_sha256", ""))):
        errors.append("YOLOv8n output digest is invalid")
    _expect_equal(errors, "remote_smoke.minimal_pipeline_status", smoke.get("minimal_pipeline_status"), "pass")
    _expect_equal(
        errors,
        "remote_smoke.minimal_pipeline_receipt",
        smoke.get("minimal_pipeline_receipt"),
        f"runs/slurm/E064/{job_id}/synthetic_contract_pipeline/integration_receipt.json",
    )
    _expect_equal(errors, "remote_smoke.minimal_pipeline_receipt_sha256", smoke.get("minimal_pipeline_receipt_sha256"), _sha256_file(synthetic_receipt_path))
    _expect_equal(errors, "remote_smoke.training_performed", smoke.get("training_performed"), False)
    _expect_equal(errors, "remote_smoke.performance_claims_authorized", smoke.get("performance_claims_authorized"), False)

    expected_output = {
        "path": "evidence/smoke/remote_smoke.json",
        "kind": "file",
        "bytes": paths[1].stat().st_size,
        "sha256": _sha256_file(paths[1]),
    }
    _expect_equal(errors, "step_execution_receipt.outputs", receipt.get("outputs"), [expected_output])
    _expect_equal(errors, "step_execution_receipt.compute_node", receipt.get("compute_node"), probe.get("compute_node"))
    _expect_equal(errors, "step_execution_receipt.resource_kind", receipt.get("resource_kind"), "gpu")
    _expect_equal(errors, "step_execution_receipt.plan_resource", receipt.get("plan_resource"), "gpu_sbatch")
    _expect_equal(errors, "step_execution_receipt.plan_site", receipt.get("plan_site"), "remote_compute")

    try:
        pipeline = validate_minimal_pipeline_run(synthetic_root)
    except MinimalPipelineError as exc:
        errors.append(f"synthetic module pipeline validation failed: {exc}")
        pipeline = {"status": "fail", "artifact_count": 0}
    if exit_path.read_text(encoding="ascii").strip() != "0":
        errors.append("Slurm child exit code is not zero")
    if stderr_path.stat().st_size != 0:
        errors.append("Slurm stderr is not empty")
    stdout_text = stdout_path.read_text(encoding="utf-8-sig")
    if '"status": "pass"' not in stdout_text or '"step_id": "E064"' not in stdout_text:
        errors.append("Slurm stdout lacks the successful E064 receipt")

    _expect_equal(errors, "submission.partition", submission.get("partition"), "3090")
    _expect_equal(errors, "submission.compute_node", submission.get("compute_node"), "gpu03")
    _expect_equal(errors, "submission.duplicate_job_count", submission.get("duplicate_job_count"), 0)
    _expect_equal(errors, "actual_resource_usage.slurm_state", usage.get("slurm_state"), "COMPLETED")
    _expect_equal(errors, "actual_resource_usage.exit_code", usage.get("exit_code"), "0:0")
    elapsed = int(usage.get("elapsed_seconds", -1))
    if elapsed <= 0 or elapsed > 1800:
        errors.append("remote smoke elapsed time is outside the frozen limit")
    actual_cost = float(usage.get("fixed_price_cost_estimate_rmb", {}).get("gpu_plus_cpu", -1.0))
    upper_cost = float(estimate.get("cost_upper_bound_rmb", {}).get("gpu_plus_cpu", -1.0))
    if actual_cost < 0.0 or upper_cost < 0.0 or actual_cost > upper_cost:
        errors.append("actual linear cost exceeds the recorded upper bound")

    synced = sync.get("synchronized_artifacts")
    expected_synced = {
        "evidence/smoke/remote_smoke.json": _sha256_file(paths[1]),
        f"runs/slurm/E064/{job_id}/step_execution_receipt.json": _sha256_file(paths[7]),
        f"runs/slurm/E064/{job_id}/slurm_environment_probe.json": _sha256_file(paths[8]),
        f"runs/slurm/E064/{job_id}/exit_code.txt": _sha256_file(exit_path),
        f"runs/slurm/E064/{job_id}/slurm_stdout.log": _sha256_file(stdout_path),
        f"runs/slurm/E064/{job_id}/slurm_stderr.log": _sha256_file(stderr_path),
    }
    _expect_equal(errors, "result_sync.synchronized_artifacts", synced, expected_synced)
    _expect_equal(errors, "result_sync.unsynchronized_required_artifact_count", sync.get("verification", {}).get("unsynchronized_required_artifact_count"), 0)

    _expect_equal(errors, "closeout.trigger_step_id", closeout.get("trigger_step_id"), "E064")
    _expect_equal(errors, "closeout.final_billing_state", closeout.get("final_billing_state"), "verified_nonbilling")
    for key in (
        "slurm_jobs_and_allocations",
        "tmux_and_screen_sessions",
        "relevant_processes",
        "transfers_and_monitors",
        "unsynchronized_artifacts",
    ):
        _expect_equal(errors, f"closeout.{key}", closeout.get(key), [])

    diagnostics = {
        "slurm_job_id": job_id,
        "compute_node": smoke.get("compute_node"),
        "cuda_device_names": names,
        "elapsed_seconds": elapsed,
        "module_artifact_count": pipeline.get("artifact_count"),
        "stderr_bytes": stderr_path.stat().st_size,
        "actual_linear_cost_rmb": actual_cost,
        "cost_upper_bound_rmb": upper_cost,
        "final_billing_state": closeout.get("final_billing_state"),
    }
    return _result(root, errors, (*paths, synthetic_receipt_path), diagnostics)


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"immutable review output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Independently reconcile E064 remote smoke")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = review_e064(args.project_root)
    if args.output is not None:
        output = args.output if args.output.is_absolute() else args.project_root / args.output
        _write_once(output, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
