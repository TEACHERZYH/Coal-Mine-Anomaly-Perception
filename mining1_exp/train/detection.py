from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

import pandas as pd
import torch
import yaml

from ..governance.immutable import write_once_bytes, write_once_json
from ..provenance import (
    canonical_json_sha256,
    sha256_file,
    validate_source_lock_amendment,
)
from ..training_data import (
    build_yolo_view,
    dataset_ids_for_role,
    load_frozen_protocol,
    source_detection_records,
    target_detection_records,
)
from ..workflow_common import WorkflowExecutionError, load_json


TRAINING_SEEDS = (1701, 2903, 4219)
SUBSET_SEEDS = (5171, 6197, 7331)
V2_PRETRAIN_CONDITIONS = (
    ("V2-PRE-GEN", "generic_matched"),
    ("V2-PRE-SINGLE", "single_coal"),
    ("V2-PRE-MULTI", "multi_coal"),
)
V2_FINETUNE_CONDITIONS = (
    ("V2-A-10-SCR", "scratch", None),
    ("V2-A-10-GEN", "generic_matched", "V2-PRE-GEN"),
    ("V2-A-10-SINGLE", "single_coal", "V2-PRE-SINGLE"),
    ("V2-A-10-MULTI", "multi_coal", "V2-PRE-MULTI"),
)


@dataclass(frozen=True)
class DetectionRunSpec:
    step_id: str
    package_id: str
    family_id: str
    condition: str
    train_seed: int
    subset_seed: Optional[int]
    modality: Optional[str]
    parent_family_id: Optional[str]
    run_directory: str


def _indexed(items: tuple[Any, ...], index: int, label: str) -> Any:
    if index < 0 or index >= len(items):
        raise WorkflowExecutionError(f"{label} array index is outside the frozen range")
    return items[index]


def t1_spec(array_index: int) -> DetectionRunSpec:
    family_id, modality = _indexed(
        (("T1-VIS", "visible"), ("T1-THERM", "thermal")),
        array_index,
        "E100",
    )
    return DetectionRunSpec(
        step_id="E100",
        package_id="T1",
        family_id=family_id,
        condition="frozen_feature_generator",
        train_seed=1701,
        subset_seed=None,
        modality=modality,
        parent_family_id=None,
        run_directory=f"runs/T1/{family_id}/seed-1701",
    )


def v2_pretrain_spec(array_index: int) -> DetectionRunSpec:
    condition_index, seed_index = divmod(array_index, len(TRAINING_SEEDS))
    family_id, condition = _indexed(
        V2_PRETRAIN_CONDITIONS, condition_index, "E200 condition"
    )
    seed = _indexed(TRAINING_SEEDS, seed_index, "E200 seed")
    return DetectionRunSpec(
        step_id="E200",
        package_id="V2",
        family_id=family_id,
        condition=condition,
        train_seed=seed,
        subset_seed=None,
        modality="visible",
        parent_family_id=None,
        run_directory=f"runs/V2/pretrain/{family_id}/seed-{seed}",
    )


def v2_finetune_spec(array_index: int) -> DetectionRunSpec:
    condition_index, seed_index = divmod(array_index, len(TRAINING_SEEDS))
    family_id, condition, parent = _indexed(
        V2_FINETUNE_CONDITIONS, condition_index, "E202 condition"
    )
    seed = _indexed(TRAINING_SEEDS, seed_index, "E202 seed")
    subset_seed = _indexed(SUBSET_SEEDS, seed_index, "E202 subset seed")
    return DetectionRunSpec(
        step_id="E202",
        package_id="V2",
        family_id=family_id,
        condition=condition,
        train_seed=seed,
        subset_seed=subset_seed,
        modality="visible",
        parent_family_id=parent,
        run_directory=f"runs/V2/finetune/{family_id}/seed-{seed}-subset-{subset_seed}",
    )


def _pilot_decisions(project_root: Path) -> Dict[str, Any]:
    payload = load_json(project_root / "evidence/pilot/resource_pilot.json")
    decisions = payload.get("decisions")
    if payload.get("status") != "pass" or not isinstance(decisions, dict):
        raise WorkflowExecutionError("Resource pilot decisions are unavailable")
    required = {"precision", "workers", "micro_batch_size", "gradient_accumulation"}
    if not required.issubset(decisions):
        raise WorkflowExecutionError("Resource pilot lacks detection training decisions")
    return decisions


def _prepare_view(project_root: Path, spec: DetectionRunSpec) -> Dict[str, Any]:
    if spec.step_id == "E100":
        dataset_id = dataset_ids_for_role(project_root, "primary_rgbt_dataset")[0]
        records = target_detection_records(
            project_root, dataset_id=dataset_id, modality=spec.modality
        )
        view_id = f"T1-{spec.modality}-seed-{spec.train_seed}"
    elif spec.step_id == "E200":
        records = source_detection_records(
            project_root, condition=spec.condition, train_seed=spec.train_seed
        )
        view_id = f"V2-pre-{spec.condition}-seed-{spec.train_seed}"
    elif spec.step_id == "E202":
        dataset_id = dataset_ids_for_role(project_root, "primary_visual_target")[0]
        records = target_detection_records(
            project_root,
            dataset_id=dataset_id,
            subset_seed=spec.subset_seed,
            modality="visible",
        )
        view_id = (
            f"V2-target10-{spec.condition}-seed-{spec.train_seed}-"
            f"subset-{spec.subset_seed}"
        )
    else:
        raise WorkflowExecutionError(f"Unsupported detection training step: {spec.step_id}")
    return build_yolo_view(project_root, view_id=view_id, records=records)


def _parent_checkpoint(project_root: Path, spec: DetectionRunSpec) -> Optional[Path]:
    if spec.parent_family_id is None:
        return None
    path = (
        project_root
        / "runs/V2/pretrain"
        / spec.parent_family_id
        / f"seed-{spec.train_seed}"
        / "run_manifest.json"
    )
    manifest = load_json(path)
    checkpoint = project_root / str(manifest.get("checkpoint_path", ""))
    if (
        manifest.get("status") != "pass"
        or manifest.get("family_id") != spec.parent_family_id
        or int(manifest.get("train_seed", -1)) != spec.train_seed
        or not checkpoint.is_file()
        or sha256_file(checkpoint) != manifest.get("checkpoint_sha256")
    ):
        raise WorkflowExecutionError(f"Parent pretraining checkpoint is invalid: {path}")
    return checkpoint


def _fresh_yolo(yolo_factory: Callable[..., Any], seed: int) -> Any:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return yolo_factory("yolov8n.yaml", task="detect")


def _transfer_backbone_neck(model: Any, checkpoint_path: Path) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    parent = checkpoint.get("ema") or checkpoint.get("model")
    if parent is None or not hasattr(parent, "state_dict"):
        raise WorkflowExecutionError("Parent checkpoint lacks a model state dictionary")
    source_state = parent.float().state_dict()
    target_state = model.model.state_dict()
    module_indices = []
    for key in target_state:
        parts = key.split(".")
        if len(parts) > 1 and parts[0] == "model" and parts[1].isdigit():
            module_indices.append(int(parts[1]))
    if not module_indices:
        raise WorkflowExecutionError("Cannot identify the YOLO detection head")
    head_index = max(module_indices)
    head_prefix = f"model.{head_index}."
    head_before = canonical_json_sha256(
        {
            key: hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
            for key, value in target_state.items()
            if key.startswith(head_prefix)
        }
    )
    transfer = {
        key: value
        for key, value in source_state.items()
        if key in target_state
        and not key.startswith(head_prefix)
        and target_state[key].shape == value.shape
    }
    if not transfer:
        raise WorkflowExecutionError("No compatible backbone/neck weights were transferred")
    incompatible = model.model.load_state_dict(transfer, strict=False)
    head_after = canonical_json_sha256(
        {
            key: hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
            for key, value in model.model.state_dict().items()
            if key.startswith(head_prefix)
        }
    )
    if head_before != head_after:
        raise WorkflowExecutionError("Target detection head changed during transfer")
    return {
        "parent_checkpoint_path": checkpoint_path.as_posix(),
        "parent_checkpoint_sha256": sha256_file(checkpoint_path),
        "transferred_tensor_count": len(transfer),
        "target_head_index": head_index,
        "target_head_initialization_sha256": head_after,
        "missing_key_count": len(incompatible.missing_keys),
        "unexpected_key_count": len(incompatible.unexpected_keys),
    }


def _training_parameters(
    project_root: Path,
    spec: DetectionRunSpec,
    view: Mapping[str, Any],
) -> Dict[str, Any]:
    protocol = load_frozen_protocol(project_root)
    decisions = _pilot_decisions(project_root)
    family_key = "rgbt" if spec.step_id == "E100" else "visible"
    training = protocol["training"][family_key]
    if spec.step_id == "E200":
        max_updates = int(protocol["training"]["matched_pretraining"]["optimizer_updates"])
    else:
        max_updates = int(training["max_updates"])
    micro = int(decisions["micro_batch_size"][family_key])
    accumulation = int(decisions["gradient_accumulation"][family_key])
    workers = int(decisions["workers"])
    if min(max_updates, micro, accumulation) <= 0 or workers < 0:
        raise WorkflowExecutionError("Pilot-derived training parameters are invalid")
    batches_per_epoch = max(1, int(math.ceil(int(view["train_count"]) / micro)))
    optimizer_updates_per_epoch = max(1, int(math.ceil(batches_per_epoch / accumulation)))
    epochs = int(math.ceil(max_updates / optimizer_updates_per_epoch))
    return {
        "optimizer": "SGD",
        "lr0": float(training["learning_rate"]),
        "batch": micro,
        "nbs": int(training["effective_batch_size"]),
        "workers": workers,
        "amp": str(decisions["precision"]).lower()
        in {"amp", "amp_fp16", "fp16", "mixed_fp16"},
        "epochs": epochs,
        "max_updates": max_updates,
        "gradient_accumulation": accumulation,
        "imgsz": int(protocol["models"]["visible"]["input_size"]),
    }


def _copy_metrics(run_directory: Path) -> None:
    source = run_directory / "results.csv"
    target = run_directory / "metrics_per_epoch.csv"
    if not source.is_file():
        raise WorkflowExecutionError("Ultralytics did not write results.csv")
    write_once_bytes(target, source.read_bytes())


def _selection_metric(train_result: Any, run_directory: Path) -> float:
    values = getattr(train_result, "results_dict", None)
    if isinstance(values, Mapping):
        for key in ("metrics/mAP50-95(B)", "metrics/mAP50-95"):
            if key in values:
                return float(values[key])
    frame = pd.read_csv(run_directory / "results.csv")
    candidates = [column for column in frame.columns if "mAP50-95" in column]
    if not candidates:
        raise WorkflowExecutionError("Validation mAP50-95 is absent from training results")
    return float(pd.to_numeric(frame[candidates[0]], errors="raise").iloc[-1])


def train_detection_run(
    project_root: Path,
    run_root: Path,
    spec: DetectionRunSpec,
    *,
    yolo_factory: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    run_directory = project_root / spec.run_directory
    final_manifest = run_directory / "run_manifest.json"
    if final_manifest.is_file():
        manifest = load_json(final_manifest)
        checkpoint = project_root / str(manifest.get("checkpoint_path", ""))
        if (
            manifest.get("status") != "pass"
            or manifest.get("family_id") != spec.family_id
            or not checkpoint.is_file()
            or sha256_file(checkpoint) != manifest.get("checkpoint_sha256")
        ):
            raise WorkflowExecutionError(f"Completed detection run drifted: {spec.run_directory}")
        return manifest
    if yolo_factory is None:
        from ultralytics import YOLO

        yolo_factory = YOLO
    view = _prepare_view(project_root, spec)
    parameters = _training_parameters(project_root, spec, view)
    protocol_path = project_root / "configs/protocol_lock.pretest.yaml"
    source_lock_amendment = validate_source_lock_amendment(project_root)
    data_yaml = project_root / str(view["dataset_yaml_path"])
    parent = _parent_checkpoint(project_root, spec)
    transfer = None
    model = _fresh_yolo(yolo_factory, spec.train_seed)
    if parent is not None:
        transfer = _transfer_backbone_neck(model, parent)
    contract = {
        "schema_version": 1,
        "step_id": spec.step_id,
        "family_id": spec.family_id,
        "condition": spec.condition,
        "train_seed": spec.train_seed,
        "subset_seed": spec.subset_seed,
        "modality": spec.modality,
        "protocol_sha256": sha256_file(protocol_path),
        "source_lock_amendment": source_lock_amendment,
        "dataset_yaml_sha256": sha256_file(data_yaml),
        "source_signature_sha256": view["source_signature_sha256"],
        "parameters": parameters,
        "transfer": transfer,
    }
    contract_hash = canonical_json_sha256(contract)
    contract_path = run_directory / "run_contract.json"
    if run_directory.exists():
        existing = load_json(contract_path) if contract_path.is_file() else None
        last = run_directory / "weights/last.pt"
        retry = run_directory / "retry_once.marker"
        if existing != contract or not last.is_file() or retry.exists():
            raise WorkflowExecutionError(
                f"Partial run cannot resume under the exact frozen contract: {spec.run_directory}"
            )
        write_once_bytes(retry, b"one_exact_hash_resume\n")
        model = yolo_factory(str(last))
        train_kwargs = {"resume": True}
    else:
        run_directory.parent.mkdir(parents=True, exist_ok=True)
        run_directory.mkdir()
        write_once_json(contract_path, contract)
        train_kwargs = {
            "data": str(data_yaml),
            "project": str(run_directory.parent),
            "name": run_directory.name,
            "exist_ok": True,
            "optimizer": parameters["optimizer"],
            "lr0": parameters["lr0"],
            "batch": parameters["batch"],
            "nbs": parameters["nbs"],
            "workers": parameters["workers"],
            "amp": parameters["amp"],
            "epochs": parameters["epochs"],
            "imgsz": parameters["imgsz"],
            "seed": spec.train_seed,
            "deterministic": True,
            "patience": 0,
            "save": True,
            "val": True,
            "device": "0",
            "verbose": True,
        }
    started = datetime.now(timezone.utc).isoformat()
    result = model.train(**train_kwargs)
    finished = datetime.now(timezone.utc).isoformat()
    best = run_directory / "weights/best.pt"
    if not best.is_file():
        best = run_directory / "weights/last.pt"
    if not best.is_file():
        raise WorkflowExecutionError("Detection training produced no checkpoint")
    _copy_metrics(run_directory)
    selection_value = _selection_metric(result, run_directory)
    environment_probe = run_root / "slurm_environment_probe.json"
    if not environment_probe.is_file():
        raise WorkflowExecutionError("Slurm environment probe is missing")
    run_id = canonical_json_sha256(
        {
            "contract_sha256": contract_hash,
            "checkpoint_sha256": sha256_file(best),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        }
    )
    manifest = {
        "schema_version": 1,
        "status": "pass",
        "run_id": run_id,
        "package_id": spec.package_id,
        "family_id": spec.family_id,
        "condition": spec.condition,
        "train_seed": spec.train_seed,
        "subset_seed": spec.subset_seed,
        "modality": spec.modality,
        "selection_pool": "source_validation" if spec.step_id == "E200" else "D_b_sel",
        "selection_metric": "map50_95",
        "selection_metric_value": selection_value,
        "checkpoint_path": best.relative_to(project_root).as_posix(),
        "checkpoint_sha256": sha256_file(best),
        "run_contract_path": contract_path.relative_to(project_root).as_posix(),
        "run_contract_sha256": sha256_file(contract_path),
        "dataset_yaml_path": data_yaml.relative_to(project_root).as_posix(),
        "dataset_yaml_sha256": sha256_file(data_yaml),
        "environment_probe_path": environment_probe.relative_to(project_root).as_posix(),
        "environment_probe_sha256": sha256_file(environment_probe),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "started_at": started,
        "finished_at": finished,
        "remote_closeout_status": "pending_local_post_job_closeout",
    }
    write_once_json(final_manifest, manifest)
    write_once_bytes(
        run_directory / "execution.log",
        (
            f"status=pass\nfamily_id={spec.family_id}\nrun_id={run_id}\n"
            f"checkpoint_sha256={manifest['checkpoint_sha256']}\n"
        ).encode("ascii"),
    )
    return manifest
