from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, Mapping, Optional

from .governance.immutable import write_once_json
from .provenance import sha256_file
from .workflow_common import (
    WorkflowExecutionError,
    describe_output,
    plan_by_id,
    require_completed_dependencies,
    utc_now,
    validate_output_digests,
)


REMOTE_STEP_IDS = frozenset(
    {
        "E024",
        "E026",
        "E034",
        "E059",
        "E064",
        "E066",
        "E100",
        "E111",
        "E120",
        "E122",
        "E123",
        "E200",
        "E202",
        "E205",
        "E220",
        "E221",
        "E300",
        "E303",
        "E305",
        "E400",
    }
)
ARRAY_STEP_IDS = frozenset({"E100", "E122", "E200", "E202", "E303"})
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _load_object(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise WorkflowExecutionError(f"Required remote-step artifact is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise WorkflowExecutionError(f"Remote-step artifact is not an object: {path}")
    return payload


def _require_remote_plan_step(project_root: Path, step_id: str) -> Dict[str, str]:
    if step_id not in REMOTE_STEP_IDS:
        raise WorkflowExecutionError(f"Unsupported remote step: {step_id}")
    steps = plan_by_id(project_root)
    row = steps.get(step_id)
    if row is None or row.get("site") != "remote_compute":
        raise WorkflowExecutionError(f"Step is not a locked remote-compute action: {step_id}")
    expected_resource = "gpu_sbatch" if os.environ.get("MINING1_RESOURCE_KIND") == "gpu" else "cpu_sbatch"
    if row.get("resource") != expected_resource:
        raise WorkflowExecutionError(
            f"Remote resource contract mismatch for {step_id}: {row.get('resource')}"
        )
    require_completed_dependencies(project_root, row)
    return dict(row)


def _validate_slurm_context(
    project_root: Path, run_root: Path, step_id: str
) -> Dict[str, Any]:
    if not run_root.is_dir():
        raise WorkflowExecutionError(f"Slurm run root does not exist: {run_root}")
    allowed_root = (project_root / "runs/slurm" / step_id).resolve()
    resolved_run = run_root.resolve()
    if allowed_root not in resolved_run.parents:
        raise WorkflowExecutionError(
            f"Slurm run root is outside the locked step namespace: {run_root}"
        )
    job_id = os.environ.get("SLURM_JOB_ID", "").strip()
    resource_kind = os.environ.get("MINING1_RESOURCE_KIND", "").strip()
    environment_step = os.environ.get("MINING1_STEP_ID", "").strip()
    if not job_id or resource_kind not in {"cpu", "gpu"} or environment_step != step_id:
        raise WorkflowExecutionError("Remote step requires a matching Slurm job context")
    raw_array_index = os.environ.get("SLURM_ARRAY_TASK_ID")
    if step_id in ARRAY_STEP_IDS:
        if raw_array_index is None or not raw_array_index.isdigit():
            raise WorkflowExecutionError(f"Remote array index is missing for {step_id}")
    elif raw_array_index is not None:
        raise WorkflowExecutionError(f"Non-array step received an array index: {step_id}")

    decision_path = project_root / "configs/environment_decision.lock.json"
    decision = _load_object(decision_path)
    selected_python = str(decision.get("selected_remote_python", ""))
    if decision.get("status") != "locked" or not selected_python.startswith("/"):
        raise WorkflowExecutionError("Remote environment decision is not locked")
    if os.path.realpath(selected_python) != os.path.realpath(sys.executable):
        raise WorkflowExecutionError("Remote step is not using the selected Python")

    probe_path = run_root / "slurm_environment_probe.json"
    probe = _load_object(probe_path)
    if (
        str(probe.get("slurm_job_id")) != job_id
        or str(probe.get("resource_kind")) != resource_kind
        or str(probe.get("selected_remote_python")) != selected_python
    ):
        raise WorkflowExecutionError("Slurm environment probe does not match the active job")
    compute_node = str(probe.get("compute_node", "")).strip()
    if not compute_node or compute_node.lower() == "mu01":
        raise WorkflowExecutionError("Remote step must execute on a Slurm compute node")
    if str(probe.get("environment_decision_sha256")) != sha256_file(decision_path):
        raise WorkflowExecutionError("Slurm probe environment-decision hash drifted")
    for field in ("environment_fingerprint_sha256", "package_inventory_sha256"):
        if not SHA256_PATTERN.fullmatch(str(probe.get(field, ""))):
            raise WorkflowExecutionError(f"Slurm probe lacks a valid {field}")
    if step_id in ARRAY_STEP_IDS and str(probe.get("slurm_array_task_id")) != raw_array_index:
        raise WorkflowExecutionError("Slurm probe array index does not match the active task")
    return {
        "job_id": job_id,
        "array_task_id": raw_array_index,
        "resource_kind": resource_kind,
        "compute_node": compute_node,
        "probe_path": probe_path,
        "probe_sha256": sha256_file(probe_path),
    }


def _array_index() -> int:
    raw = os.environ.get("SLURM_ARRAY_TASK_ID")
    if raw is None or not raw.isdigit():
        raise WorkflowExecutionError("A nonnegative Slurm array index is required")
    return int(raw)


def _training_result(result: Mapping[str, Any], output_path: str) -> Dict[str, Any]:
    if result.get("status") != "pass":
        raise WorkflowExecutionError("Remote training handler did not pass")
    return {
        "status": "pass",
        "output_paths": [output_path],
        "details": {
            "family_id": result.get("family_id"),
            "train_seed": result.get("train_seed"),
        },
    }


def _not_applicable_result(
    project_root: Path,
    run_root: Path,
    *,
    step_id: str,
    family_id: str,
    condition: str,
    train_seed: int,
    subset_seed: Optional[int],
    fallback: Mapping[str, Any],
) -> Dict[str, Any]:
    path = run_root / f"{family_id}_not_applicable.json"
    payload = {
        "schema_version": 1,
        "step_id": step_id,
        "status": "not_applicable",
        "family_id": family_id,
        "condition": condition,
        "train_seed": train_seed,
        "subset_seed": subset_seed,
        "fallback": dict(fallback),
        "source_decision_path": "evidence/data/dataset_source_decision.json",
        "source_decision_sha256": sha256_file(
            project_root / "evidence/data/dataset_source_decision.json"
        ),
        "created_at": utc_now(),
    }
    if path.is_file():
        existing = _load_object(path)
        if existing != payload:
            raise WorkflowExecutionError(f"not_applicable receipt drifted: {path}")
    else:
        write_once_json(path, payload)
    return {
        "status": "pass",
        "output_paths": [path.relative_to(project_root).as_posix()],
        "details": {
            "workflow_status": "not_applicable",
            "family_id": family_id,
            "train_seed": train_seed,
            "subset_seed": subset_seed,
            "reason_code": fallback.get("reason_code"),
        },
    }


def _dispatch_remote_handler(
    step_id: str,
    project_root: Path,
    run_root: Path,
    config_path: Optional[Path],
) -> Dict[str, Any]:
    if step_id in {"E024", "E026", "E034"}:
        from .remote_data_steps import audit_groups_dedup, build_file_manifest, verify_archives

        return {
            "E024": verify_archives,
            "E026": build_file_manifest,
            "E034": audit_groups_dedup,
        }[step_id](project_root, run_root, config_path)
    if step_id == "E059":
        from .methane_materialization import materialize_methane_windows
        from .review_methane_materialization import (
            REVIEW_PATH,
            review_methane_materialization,
        )

        result = materialize_methane_windows(project_root, run_root, config_path)
        review = review_methane_materialization(project_root)
        result["output_paths"].append(REVIEW_PATH)
        result["details"]["independent_review_sha256"] = review["output_sha256"]
        return result
    if step_id in {"E064", "E066"}:
        from .remote_pilot import remote_smoke, resource_pilot

        return {"E064": remote_smoke, "E066": resource_pilot}[step_id](
            project_root, run_root, config_path
        )
    if step_id in {"E100", "E200", "E202"}:
        from .train.detection import (
            t1_spec,
            train_detection_run,
            v2_finetune_spec,
            v2_pretrain_spec,
        )
        from .training_data import transfer_claim_not_applicable

        spec = {
            "E100": t1_spec,
            "E200": v2_pretrain_spec,
            "E202": v2_finetune_spec,
        }[step_id](_array_index())
        if step_id in {"E200", "E202"} and spec.condition == "multi_coal":
            fallback = transfer_claim_not_applicable(project_root)
            if fallback is not None:
                return _not_applicable_result(
                    project_root,
                    run_root,
                    step_id=step_id,
                    family_id=spec.family_id,
                    condition=spec.condition,
                    train_seed=spec.train_seed,
                    subset_seed=spec.subset_seed,
                    fallback=fallback,
                )
        result = train_detection_run(project_root, run_root, spec)
        return _training_result(result, spec.run_directory)
    if step_id in {"E111", "E205"}:
        from .predict.detection import predict_detection_package

        package = "T1" if step_id == "E111" else "V2"
        return predict_detection_package(project_root, run_root, package)
    if step_id == "E120":
        from .train.methane import fit_s1_baselines

        return fit_s1_baselines(project_root, run_root, config_path)
    if step_id == "E122":
        from .train.methane import s1_gru_seed, train_s1_gru

        index = _array_index()
        result = train_s1_gru(project_root, run_root, index)
        output = f"runs/S1/gru/S1-GRU/seed-{s1_gru_seed(index)}"
        return _training_result(result, output)
    if step_id == "E123":
        from .predict.methane import predict_s1_sealed

        return predict_s1_sealed(project_root, run_root, config_path)
    if step_id == "E220":
        from .robustness import build_locked_corruptions

        return build_locked_corruptions(project_root, run_root, config_path)
    if step_id == "E221":
        from .robustness import predict_r1_sealed

        return predict_r1_sealed(project_root, run_root, config_path)
    if step_id == "E300":
        from .predict.episode import predict_episode_branches

        return predict_episode_branches(project_root, run_root, config_path)
    if step_id == "E303":
        from .train.episode import train_episode_run

        return train_episode_run(
            project_root, run_root, config_path, array_index=_array_index()
        )
    if step_id == "E305":
        from .predict.episode_sealed import predict_episode_sealed

        return predict_episode_sealed(project_root, run_root, config_path)
    if step_id == "E400":
        from .efficiency import measure_efficiency

        return measure_efficiency(project_root, run_root, config_path)
    raise WorkflowExecutionError(f"No remote handler is registered for {step_id}")


def execute_remote_step(
    *,
    step_id: str,
    run_root: Path,
    config_path: Optional[Path],
    project_root: Path,
) -> Dict[str, Any]:
    root = project_root.resolve()
    run = run_root.resolve()
    context = _validate_slurm_context(root, run, step_id)
    row = _require_remote_plan_step(root, step_id)
    if config_path is not None:
        resolved_config = config_path.resolve()
        if not resolved_config.is_file() or root not in resolved_config.parents:
            raise WorkflowExecutionError("Remote step config must be an existing project file")
    receipt_path = run / "step_execution_receipt.json"
    if receipt_path.is_file():
        receipt = _load_object(receipt_path)
        if receipt.get("status") != "pass" or receipt.get("step_id") != step_id:
            raise WorkflowExecutionError("Existing remote step receipt is invalid")
        validate_output_digests(root, receipt.get("outputs", []))
        return {
            "status": "pass",
            "step_id": step_id,
            "mode": "validated_existing",
            "receipt_sha256": sha256_file(receipt_path),
        }

    result = _dispatch_remote_handler(step_id, root, run, config_path)
    if not isinstance(result, dict) or result.get("status") != "pass":
        raise WorkflowExecutionError(f"Remote handler did not pass: {step_id}")
    output_paths = result.get("output_paths")
    if not isinstance(output_paths, list) or not output_paths:
        raise WorkflowExecutionError(f"Remote handler returned no outputs: {step_id}")
    normalized_paths = [str(value).rstrip("/") for value in output_paths]
    if len(set(normalized_paths)) != len(normalized_paths):
        raise WorkflowExecutionError(f"Remote handler returned duplicate outputs: {step_id}")
    outputs = [describe_output(root, value) for value in normalized_paths]
    receipt = {
        "schema_version": 1,
        "step_id": step_id,
        "status": "pass",
        "plan_site": row["site"],
        "plan_resource": row["resource"],
        "slurm_job_id": context["job_id"],
        "slurm_array_task_id": context["array_task_id"],
        "resource_kind": context["resource_kind"],
        "compute_node": context["compute_node"],
        "environment_probe_path": context["probe_path"].relative_to(root).as_posix(),
        "environment_probe_sha256": context["probe_sha256"],
        "outputs": outputs,
        "details": result.get("details", {}),
        "remote_closeout_status": "pending_local_post_job_closeout",
        "created_at": utc_now(),
    }
    write_once_json(receipt_path, receipt)
    return {
        "status": "pass",
        "step_id": step_id,
        "mode": "created",
        "receipt_sha256": sha256_file(receipt_path),
        "outputs": outputs,
    }
