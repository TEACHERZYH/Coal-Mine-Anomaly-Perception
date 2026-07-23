from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from ..data.episodes import validate_fusion_eligibility_lock
from ..episode_data import EpisodeArrays, load_episode_arrays, training_node_ids
from ..evaluate.event_metrics import evaluate_episode_events, select_episode_policy
from ..governance.immutable import write_once_json
from ..governance.prediction_lock import build_prediction_lock, close_prediction_lock
from ..models.episode_fusion import (
    EventMemoryPolicy,
    MemoryPolicyConfig,
    derange_temporal_steps,
    derange_values_within_strata,
    locked_episode_policy_grid,
    mask_only_control,
    mean_fusion,
    validate_same_checkpoint,
)
from ..provenance import canonical_json_sha256, sha256_file
from ..train.episode import (
    TRAINABLE_FAMILIES,
    infer_episode_model,
    load_episode_model,
)
from ..training_data import load_frozen_protocol
from ..workflow_common import WorkflowExecutionError, write_parquet_artifact


def _graph_population_nonempty(project_root: Path) -> bool:
    frame = validate_fusion_eligibility_lock(
        pd.read_parquet(
            project_root / "data/locked/fusion_eligibility_lock.parquet"
        )
    )
    return bool(frame["graph_primary_eligible"].astype(bool).any())


def _mean_inference(arrays: EpisodeArrays) -> Dict[str, np.ndarray]:
    probabilities = torch.from_numpy(arrays.probabilities)
    availability = torch.from_numpy(arrays.availability)
    fused, abstained = mean_fusion(probabilities, availability)
    safe = np.where(arrays.availability, arrays.probabilities, np.nan)
    confidence = np.nanmax(np.abs(safe - 0.5) * 2.0, axis=1)
    return {
        "probability": fused.numpy(),
        "base_abstained": abstained.numpy(),
        "answer_confidence": np.nan_to_num(confidence, nan=0.0),
        "reliability": arrays.availability.astype(np.float32),
        "edge_trace": np.zeros((len(arrays.metadata), 0, 0), dtype=np.float32),
    }


def _aggregate_policy_metrics(
    arrays: EpisodeArrays,
    states: Sequence[str],
) -> Dict[str, float]:
    if arrays.labels is None:
        raise WorkflowExecutionError("Policy fitting requires D_e_pol labels")
    per_concept = []
    for concept_id in arrays.concept_ids:
        mask = arrays.metadata["concept_id"].astype(str).to_numpy() == concept_id
        if not mask.any():
            continue
        per_concept.append(
            evaluate_episode_events(
                truth_event=arrays.labels[mask].astype(np.int64),
                effective_states=np.asarray(states, dtype=object)[mask].tolist(),
                episode_ids=arrays.metadata.loc[mask, "episode_id"].astype(str).tolist(),
                step_indices=arrays.metadata.loc[mask, "step_index"].astype(int).tolist(),
            )
        )
    if not per_concept:
        raise WorkflowExecutionError("Policy pool has no eligible concepts")
    truth_events = sum(int(item["truth_event_count"]) for item in per_concept)
    missed = sum(int(item["missed_event_count"]) for item in per_concept)
    matched_delays = [
        float(value)
        for item in per_concept
        for value in item["detection_delays_steps"]
    ]
    episodes = sum(int(item["episode_count"]) for item in per_concept)
    false_alarms = sum(int(item["false_alarm_count"]) for item in per_concept)
    flips = sum(int(item["state_flip_count"]) for item in per_concept)
    return {
        "answer_coverage": float(
            sum(int(item["nonabstained_step_count"]) for item in per_concept)
            / sum(int(item["eligible_step_count"]) for item in per_concept)
        ),
        "false_alarms_per_100_episodes": float(false_alarms * 100.0 / episodes),
        "event_miss_rate": float(missed / truth_events) if truth_events else 0.0,
        "event_macro_f1": float(
            np.mean(
                [
                    float(item["event_f1"])
                    for item in per_concept
                    if item["event_f1"] is not None
                ]
                or [0.0]
            )
        ),
        "mean_detection_delay_steps": float(np.mean(matched_delays))
        if matched_delays
        else float(32.0),
        "mean_state_flips_per_episode": float(flips / episodes),
    }


def _apply_policy(
    arrays: EpisodeArrays,
    inference: Mapping[str, np.ndarray],
    *,
    policy: Mapping[str, Any],
    use_memory: bool,
) -> tuple[list[str], list[Dict[str, Any]], np.ndarray]:
    config = MemoryPolicyConfig(
        beta=float(policy["beta"]),
        memory_k=int(policy["memory_k"]),
        low_threshold=float(policy["low_threshold"]),
        high_threshold=float(policy["high_threshold"]),
        alarm_threshold=float(policy["alarm_threshold"]),
    )
    threshold = float(policy["abstention_threshold"])
    abstained = np.asarray(inference["base_abstained"], dtype=bool) | (
        np.asarray(inference["answer_confidence"], dtype=np.float64) < threshold
    )
    states = np.empty(len(arrays.metadata), dtype=object)
    traces: list[Optional[Dict[str, Any]]] = [None] * len(arrays.metadata)
    grouping = arrays.metadata.groupby(["episode_id", "concept_id"], sort=True).groups
    for _, raw_indices in grouping.items():
        indices = np.asarray(list(raw_indices), dtype=np.int64)
        ordered = indices[
            np.argsort(
                arrays.metadata.loc[indices, "step_index"].to_numpy(dtype=np.int64),
                kind="stable",
            )
        ]
        policy_trace = EventMemoryPolicy(config).run(
            np.asarray(inference["probability"])[ordered],
            abstained[ordered],
            alarm_allowed=True,
            use_memory=use_memory,
        )
        for index, item in zip(ordered, policy_trace):
            states[index] = item.state
            traces[index] = {
                "memory": item.memory,
                "high_count": item.high_count,
                "alarm_count": item.alarm_count,
                "use_memory": use_memory,
            }
    if any(value is None for value in traces):
        raise WorkflowExecutionError("Episode policy did not cover every row")
    return states.astype(str).tolist(), [dict(value) for value in traces], abstained


def _fit_policy(
    project_root: Path,
    *,
    lock_id: str,
    model_hash: str,
    arrays: EpisodeArrays,
    inference: Mapping[str, np.ndarray],
    protocol: Mapping[str, Any],
) -> tuple[Dict[str, Any], str, list[str]]:
    candidates = locked_episode_policy_grid()
    rows = []
    for complexity_rank, item in enumerate(candidates.to_dict("records")):
        states, _, _ = _apply_policy(
            arrays, inference, policy=item, use_memory=True
        )
        rows.append(
            {
                **item,
                **_aggregate_policy_metrics(arrays, states),
                "complexity_rank": complexity_rank,
            }
        )
    table = pd.DataFrame.from_records(rows)
    selection = select_episode_policy(
        table,
        pool="D_e_pol",
        target_answer_coverage=float(protocol["episodes"]["target_answer_coverage"]),
        max_coverage_deviation=float(
            protocol["episodes"]["policy_calibration"][
                "max_target_answer_coverage_absolute_deviation"
            ]
        ),
        max_false_alarms_per_100_episodes=float(
            protocol["episodes"]["policy_constraints"][
                "max_false_alarms_per_100_episodes"
            ]
        ),
        max_event_miss_rate=float(
            protocol["episodes"]["policy_constraints"]["max_event_miss_rate"]
        ),
    )
    candidate_path = project_root / f"runs/E2_E3/policies/{lock_id}_candidates.parquet"
    write_parquet_artifact(candidate_path, table)
    selected_id = selection.selected_id
    if selection.status != "pass" or selected_id is None:
        diagnostic = table.loc[
            table["candidate_id"] == selection.diagnostic_id
        ].iloc[0].to_dict()
        policy_payload = {
            "schema_version": 1,
            "status": selection.status,
            "lock_id": lock_id,
            "model_hash": model_hash,
            "selection_pool": "D_e_pol",
            "test_information_used": False,
            "candidate_store_sha256": sha256_file(candidate_path),
            "selection_status": selection.status,
            "selection_reason": selection.reason,
            "selected_policy": None,
            "diagnostic_candidate": {
                key: diagnostic[key]
                for key in (
                    "candidate_id",
                    "answer_coverage",
                    "false_alarms_per_100_episodes",
                    "event_miss_rate",
                    "event_macro_f1",
                    "mean_detection_delay_steps",
                    "mean_state_flips_per_episode",
                )
            },
        }
        policy_hash = canonical_json_sha256(policy_payload)
        policy_payload["policy_sha256"] = policy_hash
        policy_path = project_root / f"runs/E2_E3/policies/{lock_id}.json"
        write_once_json(policy_path, policy_payload)
        raise WorkflowExecutionError(
            f"Episode policy is infeasible for {lock_id}: {selection.diagnostic_id}"
        )
    selected = table.loc[table["candidate_id"] == selected_id].iloc[0].to_dict()
    policy_payload = {
        "schema_version": 1,
        "status": "pass",
        "lock_id": lock_id,
        "model_hash": model_hash,
        "selection_pool": "D_e_pol",
        "test_information_used": False,
        "candidate_store_sha256": sha256_file(candidate_path),
        "selection_status": selection.status,
        "selection_reason": selection.reason,
        "selected_policy": {
            key: selected[key]
            for key in (
                "candidate_id",
                "beta",
                "memory_k",
                "low_threshold",
                "high_threshold",
                "alarm_threshold",
                "abstention_threshold",
                "answer_coverage",
                "false_alarms_per_100_episodes",
                "event_miss_rate",
                "event_macro_f1",
                "mean_detection_delay_steps",
                "mean_state_flips_per_episode",
            )
        },
    }
    policy_hash = canonical_json_sha256(policy_payload)
    policy_payload["policy_sha256"] = policy_hash
    policy_path = project_root / f"runs/E2_E3/policies/{lock_id}.json"
    write_once_json(policy_path, policy_payload)
    states, _, _ = _apply_policy(
        arrays,
        inference,
        policy=policy_payload["selected_policy"],
        use_memory=True,
    )
    return policy_payload, policy_hash, states


def _load_trainable_manifests(
    project_root: Path, protocol: Mapping[str, Any], *, graph_enabled: bool
) -> list[Dict[str, Any]]:
    seeds = [int(value) for value in protocol["seeds"]["episode"]]
    manifests = []
    for family in TRAINABLE_FAMILIES:
        if family == "E3-NOGRAPH" and not graph_enabled:
            continue
        for seed in seeds:
            path = project_root / f"runs/E2_E3/{family}/seed-{seed}/run_manifest.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            checkpoint = project_root / str(payload.get("checkpoint_path", ""))
            if (
                payload.get("status") != "pass"
                or payload.get("selection_pool") != "D_e_sel"
                or int(payload.get("train_seed", -1)) != seed
                or not checkpoint.is_file()
                or sha256_file(checkpoint) != payload.get("checkpoint_sha256")
            ):
                raise WorkflowExecutionError(f"Episode run manifest drifted: {path}")
            manifests.append({**payload, "checkpoint": checkpoint})
    return manifests


def _prediction_rows(
    arrays: EpisodeArrays,
    inference: Mapping[str, np.ndarray],
    *,
    family_id: str,
    train_seed: Optional[int],
    model_hash: str,
    policy_hash: str,
    policy: Mapping[str, Any],
    use_memory: bool,
    control_type: Optional[str] = None,
) -> list[Dict[str, Any]]:
    states, memory, abstained = _apply_policy(
        arrays, inference, policy=policy, use_memory=use_memory
    )
    run_id = canonical_json_sha256(
        {
            "family_id": family_id,
            "train_seed": train_seed,
            "model_hash": model_hash,
            "policy_hash": policy_hash,
            "control_type": control_type,
        }
    )
    rows = []
    reliability = np.asarray(inference["reliability"])
    edge_trace = np.asarray(inference["edge_trace"])
    for index, item in enumerate(arrays.metadata.itertuples(index=False)):
        rows.append(
            {
                "run_id": run_id,
                "family_id": family_id,
                "train_seed": train_seed,
                "control_type": control_type,
                "episode_id": str(item.episode_id),
                "step_index": int(item.step_index),
                "concept_id": str(item.concept_id),
                "score_calibrated": np.float32(inference["probability"][index]),
                "state_prediction": states[index],
                "abstained": bool(abstained[index]),
                "reliability_json": json.dumps(
                    {
                        node: float(value)
                        for node, value in zip(
                            arrays.node_ids, reliability[index].tolist()
                        )
                    },
                    sort_keys=True,
                ),
                "edge_trace_json": json.dumps(edge_trace[index].tolist()),
                "memory_state_json": json.dumps(memory[index], sort_keys=True),
                "model_hash": model_hash,
                "policy_hash": policy_hash,
            }
        )
    return rows


def _recompute_quality(arrays: EpisodeArrays, probabilities: np.ndarray) -> EpisodeArrays:
    quality = arrays.quality.copy()
    quality[:, :, 0] = np.where(
        arrays.availability, np.abs(probabilities - 0.5), np.nan
    )
    quality[:, :, 1] = arrays.availability.astype(np.float32)
    return replace(arrays, probabilities=probabilities.astype(np.float32), quality=quality)


def _mask_only_arrays(
    train: EpisodeArrays, test: EpisodeArrays
) -> EpisodeArrays:
    if train.labels is None:
        raise WorkflowExecutionError("Mask-only control requires D_e_tr labels")
    probabilities = np.empty_like(test.probabilities)
    for concept_id in test.concept_ids:
        train_mask = train.metadata["concept_id"].astype(str).to_numpy() == concept_id
        test_mask = test.metadata["concept_id"].astype(str).to_numpy() == concept_id
        prior = float(np.mean(train.labels[train_mask]))
        probabilities[test_mask] = mask_only_control(
            test.probabilities[test_mask],
            test.availability[test_mask],
            [prior] * len(test.node_ids),
            prior_fit_pool="D_e_tr",
        )
    return _recompute_quality(test, probabilities)


def _temporal_shuffle_arrays(arrays: EpisodeArrays, *, seed: int) -> EpisodeArrays:
    fields = {
        "probabilities": arrays.probabilities.copy(),
        "quality": arrays.quality.copy(),
        "availability": arrays.availability.copy(),
        "pair_features": arrays.pair_features.copy(),
        "edge_validity": arrays.edge_validity.copy(),
    }
    for _, raw_indices in arrays.metadata.groupby(
        ["episode_id", "concept_id"], sort=True
    ).groups.items():
        indices = np.asarray(list(raw_indices), dtype=np.int64)
        ordered = indices[
            np.argsort(
                arrays.metadata.loc[indices, "step_index"].to_numpy(dtype=np.int64),
                kind="stable",
            )
        ]
        if len(ordered) < 2:
            continue
        _, permutation = derange_temporal_steps(
            arrays.probabilities[ordered], seed=seed
        )
        source = ordered[permutation]
        for name, values in fields.items():
            values[ordered] = getattr(arrays, name)[source]
    return replace(arrays, **fields)


def _score_shuffle_arrays(arrays: EpisodeArrays, *, seed: int) -> EpisodeArrays:
    probabilities = arrays.probabilities.copy()
    concepts = arrays.metadata["concept_id"].astype(str).to_numpy()
    for node_index, node_id in enumerate(arrays.node_ids):
        available_indices = np.flatnonzero(arrays.availability[:, node_index])
        values = probabilities[available_indices, node_index]
        strata = [
            ("D_e_te", concepts[index], node_id, True)
            for index in available_indices
        ]
        shuffled, _ = derange_values_within_strata(values, strata, seed=seed)
        probabilities[available_indices, node_index] = shuffled
    return _recompute_quality(arrays, probabilities)


def _pair_alignment_shuffle_arrays(
    arrays: EpisodeArrays, *, seed: int
) -> EpisodeArrays:
    """Break cross-node pairing while preserving every node marginal."""
    probabilities = arrays.probabilities.copy()
    concepts = arrays.metadata["concept_id"].astype(str).to_numpy()
    availability_counts = arrays.availability.sum(axis=0)
    anchor_index = int(np.argmax(availability_counts))
    for node_index, node_id in enumerate(arrays.node_ids):
        if node_index == anchor_index:
            continue
        available_indices = np.flatnonzero(arrays.availability[:, node_index])
        values = probabilities[available_indices, node_index]
        strata = [
            ("D_e_te", concepts[index], node_id, True)
            for index in available_indices
        ]
        shuffled, _ = derange_values_within_strata(values, strata, seed=seed)
        probabilities[available_indices, node_index] = shuffled
    return _recompute_quality(arrays, probabilities)


def predict_episode_sealed(
    project_root: Path,
    run_root: Path,
    config_path: Optional[Path],
) -> Dict[str, Any]:
    del config_path
    probe = run_root / "slurm_environment_probe.json"
    if not probe.is_file():
        raise WorkflowExecutionError("E305 lacks its Slurm environment probe")
    seal_path = project_root / "data/seals/episode_test_seal.json"
    if not seal_path.is_file():
        raise WorkflowExecutionError("E305 episode test seal is missing")
    protocol = load_frozen_protocol(project_root)
    graph_enabled = _graph_population_nonempty(project_root)
    manifests = _load_trainable_manifests(
        project_root, protocol, graph_enabled=graph_enabled
    )
    nodes = training_node_ids(project_root)
    policy_arrays = load_episode_arrays(
        project_root, pool="D_e_pol", include_labels=True, node_ids=nodes
    )
    test_arrays = load_episode_arrays(
        project_root,
        pool="D_e_te",
        include_labels=False,
        node_ids=nodes,
        concept_ids=policy_arrays.concept_ids,
    )
    train_arrays = load_episode_arrays(
        project_root,
        pool="D_e_tr",
        include_labels=True,
        node_ids=nodes,
        concept_ids=policy_arrays.concept_ids,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main_rows: list[Dict[str, Any]] = []
    control_rows: list[Dict[str, Any]] = []
    model_hashes: Dict[str, str] = {}
    policy_hashes: Dict[str, str] = {}
    prediction_paths: Dict[str, Path] = {}
    checkpoint_equalities = []
    mean_hash = canonical_json_sha256(
        {"family_id": "E2-MEAN", "rule": "available_branch_arithmetic_mean", "version": 1}
    )
    mean_policy_inference = _mean_inference(policy_arrays)
    mean_policy, mean_policy_hash, _ = _fit_policy(
        project_root,
        lock_id="E2-MEAN",
        model_hash=mean_hash,
        arrays=policy_arrays,
        inference=mean_policy_inference,
        protocol=protocol,
    )
    main_rows.extend(
        _prediction_rows(
            test_arrays,
            _mean_inference(test_arrays),
            family_id="E2-MEAN",
            train_seed=None,
            model_hash=mean_hash,
            policy_hash=mean_policy_hash,
            policy=mean_policy["selected_policy"],
            use_memory=True,
        )
    )
    model_hashes["E2-MEAN"] = mean_hash
    policy_hashes["E2-MEAN"] = mean_policy_hash
    loaded_full: list[tuple[Dict[str, Any], Any, Dict[str, Any], Dict[str, Any], str]] = []
    for manifest in manifests:
        lock_id = f"{manifest['family_id']}-seed-{int(manifest['train_seed'])}"
        model_hash = str(manifest["checkpoint_sha256"])
        model, checkpoint = load_episode_model(manifest["checkpoint"], device=device)
        if tuple(checkpoint["node_ids"]) != tuple(test_arrays.node_ids) or tuple(
            checkpoint["concept_ids"]
        ) != tuple(test_arrays.concept_ids):
            raise WorkflowExecutionError(f"Episode checkpoint axes drifted: {lock_id}")
        policy_inference = infer_episode_model(
            model, checkpoint, policy_arrays, device=device
        )
        policy_payload, policy_hash, _ = _fit_policy(
            project_root,
            lock_id=lock_id,
            model_hash=model_hash,
            arrays=policy_arrays,
            inference=policy_inference,
            protocol=protocol,
        )
        test_inference = infer_episode_model(
            model, checkpoint, test_arrays, device=device
        )
        main_rows.extend(
            _prediction_rows(
                test_arrays,
                test_inference,
                family_id=str(manifest["family_id"]),
                train_seed=int(manifest["train_seed"]),
                model_hash=model_hash,
                policy_hash=policy_hash,
                policy=policy_payload["selected_policy"],
                use_memory=True,
            )
        )
        model_hashes[lock_id] = model_hash
        policy_hashes[lock_id] = policy_hash
        if manifest["family_id"] == "E3-FULL":
            loaded_full.append(
                (manifest, model, checkpoint, policy_payload, policy_hash)
            )
    for manifest, model, checkpoint, policy_payload, full_policy_hash in loaded_full:
        seed = int(manifest["train_seed"])
        model_hash = str(manifest["checkpoint_sha256"])
        full_test = infer_episode_model(model, checkpoint, test_arrays, device=device)
        no_memory_id = f"E3-NOMEM-seed-{seed}"
        no_memory_policy_hash = canonical_json_sha256(
            {"base_policy_hash": full_policy_hash, "control": "no_memory", "use_memory": False}
        )
        validate_same_checkpoint(model_hash, model_hash)
        main_rows.extend(
            _prediction_rows(
                test_arrays,
                full_test,
                family_id="E3-NOMEM",
                train_seed=seed,
                model_hash=model_hash,
                policy_hash=no_memory_policy_hash,
                policy=policy_payload["selected_policy"],
                use_memory=False,
                control_type="no_memory",
            )
        )
        model_hashes[no_memory_id] = model_hash
        policy_hashes[no_memory_id] = no_memory_policy_hash
        checkpoint_equalities.append(
            {"control": no_memory_id, "full_model_hash": model_hash, "control_model_hash": model_hash}
        )
        controls = {
            "mask_only": _mask_only_arrays(train_arrays, test_arrays),
            "temporal_order_shuffle": _temporal_shuffle_arrays(test_arrays, seed=seed),
            "branch_score_shuffle": _score_shuffle_arrays(test_arrays, seed=seed),
        }
        if graph_enabled:
            controls["pair_alignment_shuffle"] = _pair_alignment_shuffle_arrays(
                test_arrays, seed=seed
            )
        for control_name, control_arrays in controls.items():
            control_id = f"E3-{control_name.upper()}-seed-{seed}"
            control_inference = infer_episode_model(
                model, checkpoint, control_arrays, device=device
            )
            control_policy_hash = canonical_json_sha256(
                {
                    "base_policy_hash": full_policy_hash,
                    "control": control_name,
                    "checkpoint_sha256": model_hash,
                }
            )
            validate_same_checkpoint(model_hash, model_hash)
            control_rows.extend(
                _prediction_rows(
                    test_arrays,
                    control_inference,
                    family_id=(
                        "E3-SHUFFLE"
                        if control_name == "pair_alignment_shuffle"
                        else "E3-FULL"
                    ),
                    train_seed=seed,
                    model_hash=model_hash,
                    policy_hash=control_policy_hash,
                    policy=policy_payload["selected_policy"],
                    use_memory=True,
                    control_type=control_name,
                )
            )
            model_hashes[control_id] = model_hash
            policy_hashes[control_id] = control_policy_hash
            checkpoint_equalities.append(
                {"control": control_id, "full_model_hash": model_hash, "control_model_hash": model_hash}
            )
    main = pd.DataFrame.from_records(main_rows).sort_values(
        ["run_id", "episode_id", "step_index", "concept_id"]
    )
    controls_frame = pd.DataFrame.from_records(control_rows).sort_values(
        ["run_id", "episode_id", "step_index", "concept_id"]
    )
    main_path = project_root / "predictions/locked/episodes.parquet"
    controls_path = project_root / "predictions/locked/episode_shortcut_controls.parquet"
    write_parquet_artifact(main_path, main)
    write_parquet_artifact(controls_path, controls_frame)
    for lock_id in model_hashes:
        prediction_paths[lock_id] = (
            controls_path
            if any(
                token in lock_id
                for token in ("MASK_ONLY", "TEMPORAL", "BRANCH_SCORE", "PAIR_ALIGNMENT")
            )
            else main_path
        )
    lock = build_prediction_lock(
        scope="episode",
        seal_hash=sha256_file(seal_path),
        protocol_hash=sha256_file(project_root / "configs/protocol_lock.pretest.yaml"),
        source_hash=sha256_file(project_root / "evidence/data/dataset_source_decision.json"),
        matrix_hash=sha256_file(project_root / "configs/experiment_matrix.template.csv"),
        required_prediction_families=sorted(model_hashes),
        prediction_paths=prediction_paths,
        model_hashes=model_hashes,
        calibrator_and_policy_hashes=policy_hashes,
    )
    lock["checkpoint_equality_assertions"] = checkpoint_equalities
    lock["main_prediction_path"] = main_path.relative_to(project_root).as_posix()
    lock["shortcut_prediction_path"] = controls_path.relative_to(project_root).as_posix()
    lock["test_labels_opened"] = False
    lock_path = project_root / "predictions/locked/episodes_prediction_lock.json"
    close_prediction_lock(lock_path, lock)
    return {
        "status": "pass",
        "output_paths": [
            main_path.relative_to(project_root).as_posix(),
            controls_path.relative_to(project_root).as_posix(),
            lock_path.relative_to(project_root).as_posix(),
        ],
        "details": {
            "main_prediction_count": len(main),
            "shortcut_prediction_count": len(controls_frame),
            "locked_family_count": len(model_hashes),
            "test_labels_opened": False,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "environment_probe_sha256": sha256_file(probe),
        },
    }
