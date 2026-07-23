from __future__ import annotations

from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..data.episodes import validate_fusion_eligibility_lock
from ..episode_data import EpisodeArrays, load_episode_arrays, training_node_ids
from ..evaluate.event_metrics import evaluate_episode_events
from ..governance.immutable import write_once_bytes, write_once_json
from ..models.episode_fusion import (
    CalibratedLogitFusion,
    ReliabilityGraphFusion,
    RobustQualityScaler,
    reliability_brier_loss,
    reliability_supervision,
)
from ..provenance import canonical_json_sha256, sha256_file
from ..training_data import load_frozen_protocol
from ..workflow_common import WorkflowExecutionError, load_json


TRAINABLE_FAMILIES = (
    "E2-LOGIT",
    "E3-NOREL",
    "E3-NOGRAPH",
    "E3-FULL",
)


@dataclass(frozen=True)
class EpisodeRunSpec:
    family_id: str
    train_seed: int
    array_index: int


def episode_run_spec(array_index: int, protocol: Mapping[str, Any]) -> EpisodeRunSpec:
    seeds = tuple(int(value) for value in protocol["seeds"]["episode"])
    expected = len(TRAINABLE_FAMILIES) * len(seeds)
    if array_index < 0 or array_index >= expected:
        raise WorkflowExecutionError("E303 array index is outside the frozen matrix")
    family_index, seed_index = divmod(array_index, len(seeds))
    return EpisodeRunSpec(
        family_id=TRAINABLE_FAMILIES[family_index],
        train_seed=seeds[seed_index],
        array_index=array_index,
    )


def _graph_population_nonempty(project_root: Path) -> bool:
    eligibility = validate_fusion_eligibility_lock(
        pd.read_parquet(
            project_root / "data/locked/fusion_eligibility_lock.parquet"
        )
    )
    return bool(eligibility["graph_primary_eligible"].astype(bool).any())


def _model_for_spec(
    spec: EpisodeRunSpec,
    arrays: EpisodeArrays,
    *,
    graph_enabled: bool,
) -> nn.Module:
    if spec.family_id == "E2-LOGIT":
        return CalibratedLogitFusion(len(arrays.node_ids))
    use_graph = graph_enabled and spec.family_id != "E3-NOGRAPH"
    use_reliability = spec.family_id != "E3-NOREL"
    return ReliabilityGraphFusion(
        node_count=len(arrays.node_ids),
        quality_feature_names=arrays.quality_feature_names,
        concept_dim=len(arrays.concept_ids),
        pair_feature_names=arrays.pair_feature_names,
        graph_layers=2,
        use_graph=use_graph,
        use_reliability=use_reliability,
    )


def _tensor_batch(
    arrays: EpisodeArrays,
    indices: np.ndarray,
    *,
    scaled_quality: np.ndarray,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    labels = (
        arrays.labels[indices]
        if arrays.labels is not None
        else np.zeros(len(indices), dtype=np.float32)
    )
    return {
        "probabilities": torch.from_numpy(arrays.probabilities[indices]).to(device),
        "quality": torch.from_numpy(scaled_quality[indices].astype(np.float32)).to(device),
        "availability": torch.from_numpy(arrays.availability[indices]).to(device),
        "concept_embedding": torch.from_numpy(arrays.concept_embedding[indices]).to(device),
        "labels": torch.from_numpy(labels).to(device),
        "edge_index": torch.from_numpy(arrays.edge_index).to(device),
        "pair_features": torch.from_numpy(arrays.pair_features[indices]).to(device),
        "edge_validity": torch.from_numpy(arrays.edge_validity[indices]).to(device),
    }


def _forward(
    model: nn.Module,
    batch: Mapping[str, torch.Tensor],
    *,
    use_graph: bool,
) -> tuple[torch.Tensor, torch.Tensor, Optional[Any]]:
    if isinstance(model, CalibratedLogitFusion):
        probability, abstained = model(
            batch["probabilities"], batch["availability"]
        )
        return probability, abstained, None
    output = model(
        batch["probabilities"],
        batch["quality"],
        batch["availability"],
        batch["concept_embedding"],
        edge_index=batch["edge_index"] if use_graph else None,
        pair_features=batch["pair_features"] if use_graph else None,
        edge_validity=batch["edge_validity"] if use_graph else None,
        abstention_threshold=0.10,
    )
    return output.probability, output.abstained, output


def _selection_event_macro_f1(
    arrays: EpisodeArrays, probabilities: np.ndarray, abstained: np.ndarray
) -> float:
    if arrays.labels is None:
        raise WorkflowExecutionError("Checkpoint selection labels are unavailable")
    states = np.where(
        abstained,
        "abstain",
        np.where(probabilities >= 0.5, "alarm", "normal"),
    )
    values = []
    for concept_id in arrays.concept_ids:
        mask = arrays.metadata["concept_id"].astype(str).to_numpy() == concept_id
        if not mask.any():
            continue
        metrics = evaluate_episode_events(
            truth_event=arrays.labels[mask].astype(np.int64),
            effective_states=states[mask].tolist(),
            episode_ids=arrays.metadata.loc[mask, "episode_id"].astype(str).tolist(),
            step_indices=arrays.metadata.loc[mask, "step_index"].astype(int).tolist(),
        )
        values.append(float(metrics["event_f1"]))
    if not values:
        raise WorkflowExecutionError("D_e_sel has no evaluable episode concepts")
    return float(np.mean(values))


def _predict_arrays(
    model: nn.Module,
    arrays: EpisodeArrays,
    *,
    scaled_quality: np.ndarray,
    use_graph: bool,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities = []
    abstained = []
    with torch.inference_mode():
        for start in range(0, len(arrays.metadata), 2048):
            indices = np.arange(start, min(start + 2048, len(arrays.metadata)))
            batch = _tensor_batch(
                arrays, indices, scaled_quality=scaled_quality, device=device
            )
            output, rejected, _ = _forward(model, batch, use_graph=use_graph)
            probabilities.append(output.cpu().numpy())
            abstained.append(rejected.cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(abstained)


def _pilot_decisions(project_root: Path) -> Dict[str, Any]:
    payload = load_json(project_root / "evidence/pilot/resource_pilot.json")
    decisions = payload.get("decisions", payload)
    micro_batch = int(decisions["micro_batch_size"]["episode"])
    accumulation = int(decisions["gradient_accumulation"]["episode"])
    if micro_batch * accumulation != 64:
        raise WorkflowExecutionError("Episode pilot changed the effective batch size")
    return {
        "micro_batch": micro_batch,
        "accumulation": accumulation,
        "precision": str(decisions.get("precision", "fp32")),
    }


def train_episode_run(
    project_root: Path,
    run_root: Path,
    config_path: Optional[Path],
    *,
    array_index: Optional[int] = None,
) -> Dict[str, Any]:
    del config_path
    probe = run_root / "slurm_environment_probe.json"
    if not probe.is_file():
        raise WorkflowExecutionError("E303 lacks its Slurm environment probe")
    protocol = load_frozen_protocol(project_root)
    if array_index is None:
        raw_index = os.environ.get("SLURM_ARRAY_TASK_ID")
        if raw_index is None:
            raise WorkflowExecutionError("E303 requires a Slurm array index")
        array_index = int(raw_index)
    spec = episode_run_spec(int(array_index), protocol)
    graph_enabled = _graph_population_nonempty(project_root)
    output_root = project_root / f"runs/E2_E3/{spec.family_id}/seed-{spec.train_seed}"
    if spec.family_id == "E3-NOGRAPH" and not graph_enabled:
        payload = {
            "schema_version": 1,
            "step_id": "E303",
            "family_id": spec.family_id,
            "train_seed": spec.train_seed,
            "status": "accepted_not_applicable",
            "reason": "graph_eligible_population_empty_at_E058",
            "graph_eligibility_sha256": sha256_file(
                project_root / "data/locked/fusion_eligibility_lock.parquet"
            ),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": str(array_index),
        }
        path = output_root / "run_manifest.json"
        write_once_json(path, payload)
        return {
            "status": "pass",
            "output_paths": [path.relative_to(project_root).as_posix()],
            "details": {"workflow_status": "accepted_not_applicable"},
        }
    nodes = training_node_ids(project_root)
    train = load_episode_arrays(
        project_root, pool="D_e_tr", include_labels=True, node_ids=nodes
    )
    select = load_episode_arrays(
        project_root,
        pool="D_e_sel",
        include_labels=True,
        node_ids=nodes,
        concept_ids=train.concept_ids,
    )
    scaler = RobustQualityScaler(train.quality_feature_names).fit(
        train.quality, pool="D_e_tr"
    )
    train_quality, _ = scaler.transform(train.quality)
    select_quality, _ = scaler.transform(select.quality)
    torch.manual_seed(spec.train_seed)
    np.random.seed(spec.train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(spec.train_seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = _model_for_spec(spec, train, graph_enabled=graph_enabled).to(device)
    use_graph = isinstance(model, ReliabilityGraphFusion) and model.use_graph
    training = protocol["training"]["episode"]
    if str(training["optimizer"]).lower() != "adamw":
        raise WorkflowExecutionError("Episode optimizer drifted from AdamW")
    max_updates = int(training["max_updates"])
    if max_updates <= 0:
        raise WorkflowExecutionError("Episode max_updates must be positive")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"])
    )
    pilot = _pilot_decisions(project_root)
    micro_batch = pilot["micro_batch"]
    accumulation = pilot["accumulation"]
    generator = np.random.default_rng(spec.train_seed)
    checkpoints = set(
        np.linspace(1, max_updates, num=min(6, max_updates), dtype=int).tolist()
    )
    history = []
    best_metric = -np.inf
    best_update = -1
    best_state: Optional[Dict[str, torch.Tensor]] = None
    loss_function = nn.BCELoss()
    model.train()
    for update in range(1, max_updates + 1):
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        for _ in range(accumulation):
            indices = generator.integers(0, len(train.metadata), size=micro_batch)
            batch = _tensor_batch(
                train, indices, scaled_quality=train_quality, device=device
            )
            probability, _, fusion_output = _forward(
                model, batch, use_graph=use_graph
            )
            loss = loss_function(probability, batch["labels"])
            if (
                fusion_output is not None
                and isinstance(model, ReliabilityGraphFusion)
                and model.use_reliability
            ):
                targets = reliability_supervision(
                    batch["labels"],
                    batch["probabilities"],
                    batch["availability"],
                    pool="D_e_tr",
                )
                loss = loss + 0.20 * reliability_brier_loss(
                    fusion_output.reliability,
                    targets,
                    batch["availability"],
                )
            (loss / accumulation).backward()
            loss_total += float(loss.detach().cpu()) / accumulation
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        if update in checkpoints:
            select_probability, select_abstained = _predict_arrays(
                model,
                select,
                scaled_quality=select_quality,
                use_graph=use_graph,
                device=device,
            )
            metric = _selection_event_macro_f1(
                select, select_probability, select_abstained
            )
            history.append(
                {
                    "optimizer_update": update,
                    "training_loss": loss_total,
                    "selection_event_macro_f1": metric,
                }
            )
            if metric > best_metric:
                best_metric = metric
                best_update = update
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            model.train()
    if best_state is None:
        raise WorkflowExecutionError("Episode training selected no checkpoint")
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint = output_root / "best.pt"
    checkpoint_payload = {
        "schema_version": 1,
        "family_id": spec.family_id,
        "train_seed": spec.train_seed,
        "node_ids": list(train.node_ids),
        "concept_ids": list(train.concept_ids),
        "quality_feature_names": list(train.quality_feature_names),
        "pair_feature_names": list(train.pair_feature_names),
        "quality_scaler": scaler,
        "use_graph": use_graph,
        "use_reliability": (
            bool(model.use_reliability)
            if isinstance(model, ReliabilityGraphFusion)
            else False
        ),
        "model_state_dict": best_state,
        "selection_pool": "D_e_sel",
        "selection_metric": "provisional_event_macro_f1_at_0.5_no_memory",
        "selection_metric_value": float(best_metric),
        "selected_optimizer_update": int(best_update),
        "protocol_sha256": sha256_file(
            project_root / "configs/protocol_lock.pretest.yaml"
        ),
    }
    buffer = io.BytesIO()
    torch.save(checkpoint_payload, buffer)
    write_once_bytes(checkpoint, buffer.getvalue())
    metrics_path = output_root / "metrics_per_epoch.csv"
    write_once_bytes(
        metrics_path,
        pd.DataFrame(history).to_csv(index=False).encode("utf-8"),
    )
    manifest = {
        "schema_version": 1,
        "step_id": "E303",
        "status": "pass",
        "run_id": canonical_json_sha256(
            {
                "family_id": spec.family_id,
                "train_seed": spec.train_seed,
                "checkpoint_sha256": sha256_file(checkpoint),
            }
        ),
        "family_id": spec.family_id,
        "train_seed": spec.train_seed,
        "selection_pool": "D_e_sel",
        "selection_metric": checkpoint_payload["selection_metric"],
        "selection_metric_value": float(best_metric),
        "selected_optimizer_update": int(best_update),
        "checkpoint_path": checkpoint.relative_to(project_root).as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint),
        "use_graph": use_graph,
        "use_reliability": checkpoint_payload["use_reliability"],
        "max_optimizer_updates": max_updates,
        "effective_batch_size": micro_batch * accumulation,
        "environment_probe_sha256": sha256_file(probe),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": str(array_index),
        "test_labels_opened": False,
        "remote_closeout_status": "pending_local_post_job_closeout",
    }
    manifest_path = output_root / "run_manifest.json"
    write_once_json(manifest_path, manifest)
    return {
        "status": "pass",
        "output_paths": [
            checkpoint.relative_to(project_root).as_posix(),
            metrics_path.relative_to(project_root).as_posix(),
            manifest_path.relative_to(project_root).as_posix(),
        ],
        "details": {
            "family_id": spec.family_id,
            "train_seed": spec.train_seed,
            "selection_metric_value": float(best_metric),
            "test_labels_opened": False,
        },
    }


def load_episode_model(
    checkpoint_path: Path, *, device: torch.device
) -> tuple[nn.Module, Dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    family_id = str(checkpoint.get("family_id"))
    node_count = len(checkpoint["node_ids"])
    if family_id == "E2-LOGIT":
        model: nn.Module = CalibratedLogitFusion(node_count)
    elif family_id in {"E3-NOREL", "E3-NOGRAPH", "E3-FULL"}:
        model = ReliabilityGraphFusion(
            node_count=node_count,
            quality_feature_names=checkpoint["quality_feature_names"],
            concept_dim=len(checkpoint["concept_ids"]),
            pair_feature_names=checkpoint["pair_feature_names"],
            graph_layers=2,
            use_graph=bool(checkpoint["use_graph"]),
            use_reliability=bool(checkpoint["use_reliability"]),
        )
    else:
        raise WorkflowExecutionError(f"Unknown episode checkpoint family: {family_id}")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval(), checkpoint


def infer_episode_model(
    model: nn.Module,
    checkpoint: Mapping[str, Any],
    arrays: EpisodeArrays,
    *,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    scaler = checkpoint["quality_scaler"]
    scaled_quality, _ = scaler.transform(arrays.quality)
    use_graph = bool(checkpoint.get("use_graph"))
    probabilities = []
    base_abstained = []
    answer_confidence = []
    reliabilities = []
    edge_traces = []
    with torch.inference_mode():
        for start in range(0, len(arrays.metadata), 2048):
            indices = np.arange(start, min(start + 2048, len(arrays.metadata)))
            batch = _tensor_batch(
                arrays, indices, scaled_quality=scaled_quality, device=device
            )
            output, rejected, fusion_output = _forward(
                model, batch, use_graph=use_graph
            )
            probabilities.append(output.cpu().numpy())
            base_abstained.append(rejected.cpu().numpy())
            if fusion_output is None:
                safe = np.where(
                    arrays.availability[indices],
                    arrays.probabilities[indices],
                    np.nan,
                )
                confidence = np.nanmax(np.abs(safe - 0.5) * 2.0, axis=1)
                answer_confidence.append(np.nan_to_num(confidence, nan=0.0))
                reliabilities.append(
                    arrays.availability[indices].astype(np.float32)
                )
                edge_traces.append(
                    np.zeros((len(indices), 0, 0), dtype=np.float32)
                )
            else:
                reliability = fusion_output.reliability.cpu().numpy()
                answer_confidence.append(reliability.max(axis=1))
                reliabilities.append(reliability)
                edge_traces.append(fusion_output.edge_weights.cpu().numpy())
    return {
        "probability": np.concatenate(probabilities),
        "base_abstained": np.concatenate(base_abstained),
        "answer_confidence": np.concatenate(answer_confidence),
        "reliability": np.concatenate(reliabilities),
        "edge_trace": np.concatenate(edge_traces),
    }
