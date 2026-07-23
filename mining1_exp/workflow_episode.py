from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import secrets
from typing import Any, Dict, Iterable, Mapping, Sequence

import pandas as pd
import yaml

from .data.episodes import (
    make_skeleton_item_id,
    validate_episode_skeleton_manifest,
    validate_fusion_eligibility_lock,
)
from .data.manifests import read_file_manifest, read_split_manifest
from .data.ontology import validate_ontology_lock
from .data.seals import DENIED_LABEL_ROLES, validate_test_seal
from .governance.immutable import write_once_json
from .governance.prediction_lock import assert_truth_free_prediction
from .governance.release import write_test_release
from .methane_data import METHANE_CONCEPT_ID
from .provenance import canonical_json_sha256, sha256_file
from .workflow_common import (
    hash_existing_inputs,
    load_json,
    utc_now,
    write_json_artifact,
    write_parquet_artifact,
)


EPISODE_POOLS = ("D_e_tr", "D_e_sel", "D_e_pol", "D_e_te")
TEMPLATE_FAMILIES = (
    "abrupt",
    "gradual",
    "intermittent",
    "evidence_delay",
    "evidence_conflict",
    "modality_missingness",
)


def _protocol(root: Path) -> Dict[str, Any]:
    path = root / "configs/protocol_lock.pretest.yaml"
    if not path.is_file():
        path = root / "configs/protocol_lock.template.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise RuntimeError("Protocol must be a mapping")
    return payload


def _concept_mapping(root: Path) -> Dict[tuple[str, str], str]:
    ontology = validate_ontology_lock(
        yaml.safe_load(
            (root / "data/locked/ontology_lock.yaml").read_text(encoding="utf-8-sig")
        )
    )
    return {
        (str(item["dataset_id"]), str(item["source_label"])): str(
            item["canonical_concept_id"]
        )
        for item in ontology["entries"]
        if item["mapping_status"] == "compatible"
    }


def _record_concepts(
    item: Any,
    mapping: Mapping[tuple[str, str], str],
) -> tuple[str, ...]:
    summary = json.loads(str(item.label_summary_json))
    if isinstance(summary.get("class_ids"), list):
        labels = [str(value) for value in summary["class_ids"]]
    elif isinstance(summary.get("class_counts"), dict):
        labels = [str(value) for value in summary["class_counts"]]
    else:
        labels = [str(value) for value in summary if not str(value).startswith("_")]
    concepts = {
        mapping.get((str(item.dataset_id), label), label)
        for label in labels
        if label.strip()
    }
    return tuple(sorted(concepts))


def _positive_flag(item: Any) -> bool:
    summary = json.loads(str(item.label_summary_json))
    counts = summary.get("class_counts")
    if isinstance(counts, dict):
        return any(float(value) > 0 for value in counts.values())
    classes = summary.get("class_ids")
    if isinstance(classes, list):
        return bool(classes)
    return any(bool(value) for key, value in summary.items() if not str(key).startswith("_"))


def _modality_node(modality: str) -> tuple[str, str]:
    normalized = modality.strip().lower()
    if normalized in {"infrared", "ir", "thermal", "thermal_ir"}:
        return "thermal", "thermal"
    if normalized in {"methane", "sensor", "time_series"}:
        return "methane", "methane"
    return "visible", "visible"


def _stable_rank(seed: int, *parts: str) -> str:
    return hashlib.sha256(
        "|".join((str(seed), *(str(value) for value in parts))).encode("utf-8")
    ).hexdigest()


def _episode_assignments(
    records: pd.DataFrame,
    *,
    seed: int,
    pool: str,
    template_family: str,
    max_steps: int,
    ordered_groups: Sequence[str],
) -> Dict[str, tuple[str, int]]:
    assignments: Dict[str, tuple[str, int]] = {}
    group_records = {
        str(group_id): sorted(
            {
                str(value)
                for value in group["step_identity"]
            }
        )
        for group_id, group in records.groupby("raw_group_id", sort=True)
    }
    if set(ordered_groups) != set(group_records) or len(ordered_groups) != len(
        group_records
    ):
        raise RuntimeError("Template ordering does not close every raw group exactly once")
    episode_index = 0
    used_steps = 0
    for group_id in ordered_groups:
        group_size = len(group_records[group_id])
        if used_steps and used_steps + group_size > max_steps:
            episode_index += 1
            used_steps = 0
        episode_id = hashlib.sha256(
            f"episode|{pool}|{seed}|{template_family}|{episode_index}".encode(
                "utf-8"
            )
        ).hexdigest()[:24]
        assignments[group_id] = (episode_id, used_steps)
        used_steps += min(group_size, max_steps)
        if used_steps >= max_steps:
            episode_index += 1
            used_steps = 0
    return assignments


def _alternate(left: Sequence[str], right: Sequence[str]) -> list[str]:
    output: list[str] = []
    for index in range(max(len(left), len(right))):
        if index < len(left):
            output.append(str(left[index]))
        if index < len(right):
            output.append(str(right[index]))
    return output


def _ordered_template_groups(
    records: pd.DataFrame,
    *,
    seed: int,
    pool: str,
    template_family: str,
) -> list[str]:
    truth = records.groupby("raw_group_id", sort=True)["positive"].max().astype(bool)
    positive = sorted(
        truth.index[truth].astype(str),
        key=lambda value: _stable_rank(seed, pool, template_family, "positive", value),
    )
    negative = sorted(
        truth.index[~truth].astype(str),
        key=lambda value: _stable_rank(seed, pool, template_family, "negative", value),
    )
    if template_family in {"abrupt", "evidence_delay"}:
        return [*negative, *positive]
    if template_family == "gradual":
        # Interleave after a negative lead-in so positive density rises over the sequence.
        lead = max(0, len(negative) - len(positive))
        return [*negative[:lead], *_alternate(negative[lead:], positive)]
    if template_family == "intermittent":
        return _alternate(negative, positive)
    return sorted(
        [*negative, *positive],
        key=lambda value: _stable_rank(seed, pool, template_family, value),
    )


def _template_family_assignments(
    records: pd.DataFrame, *, seed: int, pool: str
) -> Dict[str, str]:
    assignments: Dict[str, str] = {}
    grouped = records.groupby("raw_group_id", sort=True)
    for raw_group_id, group in grouped:
        identity_truth = group.groupby(
            ["step_identity", "concept_id"], sort=False
        )["positive"].agg(["min", "max"])
        natural_conflict = bool((identity_truth["min"] != identity_truth["max"]).any())
        branch_count = int(group["modality"].nunique())
        if natural_conflict:
            candidates = ("evidence_conflict",)
        elif branch_count >= 2:
            candidates = (
                "abrupt",
                "gradual",
                "intermittent",
                "evidence_delay",
                "modality_missingness",
            )
        else:
            candidates = ("abrupt", "gradual", "intermittent")
        rank = int(_stable_rank(seed, pool, str(raw_group_id))[:16], 16)
        assignments[str(raw_group_id)] = candidates[rank % len(candidates)]
    return assignments


def _apply_frozen_template_masks(skeleton: pd.DataFrame) -> pd.DataFrame:
    keep = pd.Series(True, index=skeleton.index)
    keys = ["episode_id", "concept_id"]
    for (episode_id, concept_id), group in skeleton.groupby(keys, sort=True):
        family = str(group["template_family"].iloc[0])
        if group["template_family"].nunique() != 1:
            raise RuntimeError("One episode-concept contains multiple template families")
        if family == "modality_missingness":
            affected_steps = {
                int(value)
                for offset, value in enumerate(sorted(set(group["step_index"])))
                if offset % 4 == 1
            }
        elif family == "evidence_delay":
            positive_steps = sorted(
                set(
                    group.loc[
                        group["event_position_role"] == "positive_observation",
                        "step_index",
                    ].astype(int)
                )
            )
            affected_steps = set(positive_steps[:2])
        else:
            affected_steps = set()
        for step in affected_steps:
            candidates = group.loc[group["step_index"].astype(int) == step]
            if candidates["node_id"].nunique() < 2:
                continue
            drop_index = max(
                candidates.index,
                key=lambda index: _stable_rank(
                    int(candidates.loc[index, "episode_seed"]),
                    str(episode_id),
                    str(concept_id),
                    str(step),
                    str(candidates.loc[index, "node_id"]),
                ),
            )
            keep.loc[drop_index] = False
    result = skeleton.loc[keep].copy()
    if result.empty:
        raise RuntimeError("Frozen template masks removed every episode node")
    return result.reset_index(drop=True)


def _not_applicable(
    root: Path,
    *,
    step_id: str,
    output: str,
    reason: str,
) -> Dict[str, Any]:
    eligibility = root / "data/locked/fusion_eligibility_lock.parquet"
    payload = {
        "schema_version": 1,
        "step_id": step_id,
        "status": "accepted_not_applicable",
        "reason": reason,
        "fusion_eligibility_sha256": sha256_file(eligibility),
        "slurm_submission_created": False,
        "blocked_claims": ["C-FUSION", "C-MEMORY", "C-GRAPH"],
        "created_at": utc_now(),
    }
    write_once_json(root / output, payload)
    return {
        "status": "pass",
        "output_paths": [output],
        "inputs": hash_existing_inputs(
            root, ["data/locked/fusion_eligibility_lock.parquet"]
        ),
        "details": {"workflow_status": "accepted_not_applicable", "reason": reason},
    }


def _fusion_population_nonempty(root: Path) -> bool:
    frame = validate_fusion_eligibility_lock(
        pd.read_parquet(root / "data/locked/fusion_eligibility_lock.parquet")
    )
    return bool(frame["fusion_primary_eligible"].astype(bool).any())


def build_episode_skeletons(
    root: Path, row: Mapping[str, str], arguments: Mapping[str, Any]
) -> Dict[str, Any]:
    del arguments
    split_path = root / "data/locked/split_manifest.parquet"
    file_path = root / "data/locked/file_manifest.parquet"
    graph_path = root / "data/locked/graph_eligibility_lock.json"
    split = read_split_manifest(split_path)
    files = read_file_manifest(file_path)
    episode_split = split.loc[split["pool"].isin(EPISODE_POOLS)].copy()
    if episode_split.empty or set(episode_split["pool"]) != set(EPISODE_POOLS):
        raise RuntimeError("All four episode pools must be populated before E058")
    joined = files.merge(
        episode_split[["dataset_id", "record_id", "raw_group_id", "pool"]],
        on=["dataset_id", "record_id", "raw_group_id"],
        validate="one_to_one",
    )
    source_decision = load_json(root / "evidence/data/dataset_source_decision.json")
    rgbt_entry = source_decision.get("roles", {}).get("primary_rgbt_dataset", {})
    rgbt_ids = {
        str(item.get("dataset_id"))
        for item in (rgbt_entry if isinstance(rgbt_entry, list) else [rgbt_entry])
        if isinstance(item, Mapping) and str(item.get("dataset_id", "")).strip()
    }
    ontology = validate_ontology_lock(
        yaml.safe_load(
            (root / "data/locked/ontology_lock.yaml").read_text(encoding="utf-8-sig")
        )
    )
    mapping = _concept_mapping(root)
    dataset_concepts: Dict[str, set[str]] = {}
    dataset_negative_verified: Dict[str, bool] = {}
    for entry in ontology["entries"]:
        if entry["mapping_status"] != "compatible":
            continue
        dataset_id = str(entry["dataset_id"])
        dataset_concepts.setdefault(dataset_id, set()).add(
            str(entry["canonical_concept_id"])
        )
        verified = entry["negative_semantics"] in {
            "exhaustive_verified_absence",
            "explicit_negative_label",
        }
        dataset_negative_verified[dataset_id] = (
            dataset_negative_verified.get(dataset_id, True) and verified
        )
    expanded = []
    detection_records = joined.loc[
        ~joined["modality"].astype(str).str.lower().isin({"methane", "sensor", "time_series"})
    ].copy()
    feature_rows = []
    for item in detection_records.itertuples(index=False):
        modality, node_id = _modality_node(str(item.modality))
        summary = json.loads(str(item.label_summary_json))
        if isinstance(summary.get("class_ids"), list):
            source_labels = {str(value) for value in summary["class_ids"]}
        elif isinstance(summary.get("class_counts"), dict):
            source_labels = {
                str(label)
                for label, count in summary["class_counts"].items()
                if float(count) > 0
            }
        else:
            source_labels = {
                str(box.get("source_label"))
                for box in summary.get("boxes", [])
                if str(box.get("source_label", "")).strip()
            }
        present = {
            mapping[(str(item.dataset_id), label)]
            for label in source_labels
            if (str(item.dataset_id), label) in mapping
        }
        concepts = dataset_concepts.get(str(item.dataset_id), set())
        if not concepts:
            continue
        negatives_verified = bool(summary.get("negative_annotation_verified")) or bool(
            dataset_negative_verified.get(str(item.dataset_id))
        )
        if not negatives_verified:
            raise RuntimeError(
                f"Episode background semantics are not verified: {item.dataset_id}/{item.record_id}"
            )
        feature_rows.append(
            {
                "dataset_id": str(item.dataset_id),
                "record_id": str(item.record_id),
                "raw_group_id": str(item.raw_group_id),
                "pair_id": str(item.pair_id) if pd.notna(item.pair_id) else None,
                "modality": modality,
                "feature_kind": "image_path",
                "relative_path": str(item.relative_path),
                "history_json": None,
                "source_feature_sha256": str(item.sha256),
            }
        )
        for concept_id in sorted(concepts):
            step_identity = str(item.pair_id).strip() if pd.notna(item.pair_id) else ""
            if not step_identity:
                step_identity = str(item.record_id)
            expanded.append(
                {
                    "dataset_id": str(item.dataset_id),
                    "record_id": str(item.record_id),
                    "raw_group_id": str(item.raw_group_id),
                    "pair_id": str(item.pair_id) if pd.notna(item.pair_id) else "",
                    "pool": str(item.pool),
                    "concept_id": concept_id,
                    "node_id": node_id,
                    "modality": modality,
                    "step_identity": step_identity,
                    "step_order": str(item.timestamp_or_order),
                    "positive": concept_id in present,
                }
            )
    methane_feature_path = root / "data/locked/episode_methane_features.parquet"
    methane_truth_path = root / "data/locked/episode_methane_truth.parquet"
    if methane_feature_path.is_file() or methane_truth_path.is_file():
        if not methane_feature_path.is_file() or not methane_truth_path.is_file():
            raise RuntimeError("Episode methane feature/truth partitions are not aligned")
        methane_features = pd.read_parquet(methane_feature_path)
        methane_truth = pd.read_parquet(methane_truth_path)
        methane = methane_features.merge(
            methane_truth[["window_id", "event_truth"]],
            on="window_id",
            validate="one_to_one",
        )
        if set(methane["pool"].astype(str)) != set(EPISODE_POOLS):
            raise RuntimeError("Episode methane windows do not cover all four episode pools")
        for item in methane.itertuples(index=False):
            feature_rows.append(
                {
                    "dataset_id": str(item.dataset_id),
                    "record_id": str(item.window_id),
                    "raw_group_id": str(item.raw_group_id),
                    "pair_id": None,
                    "modality": "methane",
                    "feature_kind": "causal_history_json",
                    "relative_path": None,
                    "history_json": str(item.history_json),
                    "source_feature_sha256": str(item.feature_manifest_hash),
                }
            )
            expanded.append(
                {
                    "dataset_id": str(item.dataset_id),
                    "record_id": str(item.window_id),
                    "raw_group_id": str(item.raw_group_id),
                    "pair_id": "",
                    "pool": str(item.pool),
                    "concept_id": METHANE_CONCEPT_ID,
                    "node_id": "methane",
                    "modality": "methane",
                    "step_identity": str(item.window_id),
                    "step_order": pd.Timestamp(item.history_end).isoformat(),
                    "positive": bool(int(item.event_truth)),
                }
            )
    expanded_frame = pd.DataFrame(expanded)
    if expanded_frame.empty:
        raise RuntimeError("No ontology-compatible episode records are available")
    feature_manifest = pd.DataFrame.from_records(feature_rows)
    if feature_manifest["record_id"].astype(str).duplicated().any():
        raise RuntimeError("Episode branch feature record IDs are not globally unique")
    if set(expanded_frame["record_id"].astype(str)) - set(
        feature_manifest["record_id"].astype(str)
    ):
        raise RuntimeError("Episode skeleton contains records without truth-free branch features")
    feature_manifest_output = "data/locked/episode_branch_feature_manifest.parquet"
    write_parquet_artifact(root / feature_manifest_output, feature_manifest)
    protocol = _protocol(root)
    seeds = [int(value) for value in protocol["seeds"]["episode"]]
    max_steps = int(protocol["episodes"]["main_length_steps"])
    nonce = secrets.token_bytes(32)
    template_lock = {
        "schema_version": 1,
        "status": "locked",
        "episode_seeds": seeds,
        "main_length_steps": max_steps,
        "target_event_prevalence": float(
            protocol["episodes"]["main_event_prevalence"]
        ),
        "template_families": list(TEMPLATE_FAMILIES),
        "repeat_design": "all_locked_raw_groups_replayed_once_per_episode_seed",
        "within_seed_group_reuse": False,
        "ordering_rules": {
            "abrupt": "background_then_positive",
            "gradual": "negative_lead_in_then_increasing_positive_density",
            "intermittent": "alternating_background_and_positive",
            "evidence_delay": "background_then_positive_with_secondary_node_delay",
            "evidence_conflict": "natural_cross_node_annotation_disagreement_only",
            "modality_missingness": "deterministic_one_node_mask_every_fourth_step",
        },
        "evidence_delay_steps": 2,
        "skeleton_item_id_salt": "independent_sealed_random_128bit_salt_per_repeat_row",
        "model_outcomes_used": False,
        "test_outcomes_used": False,
        "nonce_sha256": hashlib.sha256(nonce).hexdigest(),
        "source_split_sha256": sha256_file(split_path),
        "episode_branch_feature_manifest_sha256": sha256_file(
            root / feature_manifest_output
        ),
        "created_at": utc_now(),
    }
    template_output = "data/locked/episode_template_lock.json"
    write_once_json(root / template_output, template_lock)
    template_hash = sha256_file(root / template_output)
    skeleton_rows = []
    source_split_hash = sha256_file(split_path)
    for pool in EPISODE_POOLS:
        pool_rows = expanded_frame.loc[expanded_frame["pool"] == pool].copy()
        for seed in seeds:
            family_by_group = _template_family_assignments(
                pool_rows, seed=seed, pool=pool
            )
            for family in TEMPLATE_FAMILIES:
                family_groups = {
                    group_id
                    for group_id, assigned_family in family_by_group.items()
                    if assigned_family == family
                }
                if not family_groups:
                    continue
                family_rows = pool_rows.loc[
                    pool_rows["raw_group_id"].astype(str).isin(family_groups)
                ].copy()
                ordered_groups = _ordered_template_groups(
                    family_rows,
                    seed=seed,
                    pool=pool,
                    template_family=family,
                )
                assignments = _episode_assignments(
                    family_rows,
                    seed=seed,
                    pool=pool,
                    template_family=family,
                    max_steps=max_steps,
                    ordered_groups=ordered_groups,
                )
                for raw_group_id, group in family_rows.groupby(
                    "raw_group_id", sort=True
                ):
                    episode_id, offset = assignments[str(raw_group_id)]
                    identity_order = (
                        group[["step_identity", "step_order"]]
                        .drop_duplicates("step_identity")
                        .sort_values(["step_order", "step_identity"], kind="stable")
                    )
                    identities = {
                        str(value): index
                        for index, value in enumerate(identity_order["step_identity"])
                    }
                    step_truth = (
                        group.groupby(["step_identity", "concept_id"], sort=False)[
                            "positive"
                        ]
                        .max()
                        .astype(bool)
                        .to_dict()
                    )
                    template_id = hashlib.sha256(
                        f"template|{pool}|{seed}|{episode_id}|{family}".encode(
                            "utf-8"
                        )
                    ).hexdigest()[:24]
                    for item in group.itertuples(index=False):
                        step_index = offset + identities[str(item.step_identity)]
                        if step_index >= max_steps:
                            continue
                        positive = bool(
                            step_truth[(str(item.step_identity), str(item.concept_id))]
                        )
                        skeleton_rows.append(
                            {
                                "skeleton_item_id": make_skeleton_item_id(
                                    nonce,
                                    record_id=str(item.record_id),
                                    concept_id=str(item.concept_id),
                                    node_id=str(item.node_id),
                                    instance_salt=secrets.token_bytes(16),
                                ),
                                "episode_id": episode_id,
                                "step_index": int(step_index),
                                "record_id": str(item.record_id),
                                "raw_group_id": str(item.raw_group_id),
                                "concept_id": str(item.concept_id),
                                "node_id": str(item.node_id),
                                "pool": pool,
                                "episode_seed": seed,
                                "generator_family_id": (
                                    "S1-GRU"
                                    if item.modality == "methane"
                                    else (
                                        "T1-THERM"
                                        if item.modality == "thermal"
                                        else (
                                            "T1-VIS"
                                            if item.dataset_id in rgbt_ids
                                            else "V2-A-10-MULTI"
                                        )
                                    )
                                ),
                                "template_instance_id": template_id,
                                "template_family": family,
                                "event_position_role": (
                                    "positive_observation" if positive else "background"
                                ),
                                "source_split_hash": source_split_hash,
                                "template_lock_hash": template_hash,
                            }
                        )
    skeleton_frame = pd.DataFrame(skeleton_rows)
    if skeleton_frame.empty:
        raise RuntimeError("Episode template compiler produced no skeleton rows")
    skeleton = validate_episode_skeleton_manifest(
        _apply_frozen_template_masks(skeleton_frame)
    )
    skeleton_output = "data/locked/episode_skeleton_manifest.parquet"
    write_parquet_artifact(root / skeleton_output, skeleton)
    concept_steps = skeleton.drop_duplicates(
        ["episode_id", "step_index", "concept_id"]
    ).copy()
    concept_steps["event_truth"] = (
        concept_steps["event_position_role"] == "positive_observation"
    ).astype("int64")
    template_instances = (
        concept_steps.groupby(
            [
                "template_instance_id",
                "template_family",
                "episode_id",
                "pool",
                "episode_seed",
            ],
            sort=True,
        )
        .agg(
            concept_step_count=("concept_id", "size"),
            event_prevalence=("event_truth", "mean"),
        )
        .reset_index()
    )
    template_instances["template_lock_hash"] = template_hash
    template_instances_output = "data/locked/episode_template_instances.parquet"
    write_parquet_artifact(root / template_instances_output, template_instances)
    test_rows = skeleton.loc[skeleton["pool"] == "D_e_te"].copy()
    if test_rows.empty:
        raise RuntimeError("Episode test skeleton is empty")
    sealed_test_output = "data/seals/episode_test_skeleton_full.parquet"
    write_parquet_artifact(root / sealed_test_output, test_rows)
    projection_columns = [
        "skeleton_item_id",
        "record_id",
        "concept_id",
        "node_id",
        "generator_family_id",
    ]
    projection = skeleton.loc[:, projection_columns].copy()
    projection_output = "data/seals/episode_skeleton_inference_projection.parquet"
    write_parquet_artifact(root / projection_output, projection)
    acl = sorted(
        {
            "deny:trainer",
            "deny:selector",
            "deny:policy_fitter",
            "deny:test_predictor",
            "deny:human_reviewer",
        }
    )
    candidate = {
        "scope": "episode_test_skeleton",
        "schema_hash": canonical_json_sha256(sorted(skeleton.columns)),
        "full_manifest_hash": sha256_file(root / sealed_test_output),
        "inference_projection_hash": sha256_file(root / projection_output),
        "row_count": int(len(test_rows)),
        "raw_group_count": int(test_rows["raw_group_id"].nunique()),
        "acl_hash": canonical_json_sha256(acl),
        "created_at": utc_now(),
    }
    candidate_output = "data/seals/episode_test_skeleton_candidate.json"
    write_once_json(root / candidate_output, candidate)
    graph = load_json(graph_path)
    eligibility_source = skeleton.merge(
        feature_manifest[["record_id", "pair_id"]],
        on="record_id",
        validate="many_to_one",
    )
    eligibility_source["modality"] = eligibility_source[
        "generator_family_id"
    ].map(
        {
            "S1-GRU": "methane",
            "T1-THERM": "thermal",
            "T1-VIS": "visible",
            "V2-A-10-MULTI": "visible",
        }
    )
    if eligibility_source["modality"].isna().any():
        raise RuntimeError("Episode skeleton contains an unknown branch generator")
    positive_floor = int(
        protocol["data"]["positive_group_floor_per_claimed_class_hard_min"]
    )
    eligibility_rows = []
    for concept_id in sorted(set(skeleton["concept_id"].astype(str))):
        per_pool: Dict[str, Dict[str, int]] = {}
        for pool in EPISODE_POOLS:
            subset = eligibility_source.loc[
                (eligibility_source["concept_id"].astype(str) == concept_id)
                & (eligibility_source["pool"].astype(str) == pool)
            ].copy()
            step_columns = ["episode_id", "step_index", "concept_id"]
            step_summary = (
                subset.groupby(step_columns, sort=True)
                .agg(
                    node_count=("node_id", "nunique"),
                    modality_count=("modality", "nunique"),
                    positive=(
                        "event_position_role",
                        lambda values: bool((values == "positive_observation").all()),
                    ),
                    raw_group_count=("raw_group_id", "nunique"),
                    observed_pair_count=(
                        "pair_id",
                        lambda values: int(
                            pd.Series(values).dropna().astype(str).replace("", pd.NA).nunique()
                        ),
                    ),
                )
                .reset_index()
                if not subset.empty
                else pd.DataFrame()
            )
            if step_summary.empty:
                per_pool[pool] = {
                    "branch_types": [],
                    "multibranch_steps": 0,
                    "positive_multibranch_event_count": 0,
                    "positive_multibranch_groups": 0,
                    "positive_observed_pair_groups": 0,
                }
                continue
            multibranch = step_summary.loc[step_summary["modality_count"] >= 2]
            positive_multi = multibranch.loc[multibranch["positive"].astype(bool)]
            positive_keys = subset.merge(
                positive_multi[step_columns], on=step_columns, how="inner"
            )
            observed_keys = positive_keys.loc[
                positive_keys["pair_id"].notna()
                & positive_keys["pair_id"].astype(str).str.strip().ne("")
            ]
            if not observed_keys.empty:
                paired_rows = observed_keys.groupby(
                    [*step_columns, "pair_id"], sort=False
                ).filter(lambda values: values["modality"].nunique() >= 2)
            else:
                paired_rows = observed_keys
            per_pool[pool] = {
                "branch_types": sorted(set(subset["modality"].astype(str))),
                "multibranch_steps": int(len(multibranch)),
                "positive_multibranch_event_count": int(len(positive_multi)),
                "positive_multibranch_groups": int(
                    positive_keys["raw_group_id"].nunique()
                ),
                "positive_observed_pair_groups": int(
                    paired_rows["raw_group_id"].nunique()
                ),
            }
        eligible_branch_types = sorted(
            set.intersection(
                *(set(value["branch_types"]) for value in per_pool.values())
            )
        )
        branch_count = len(eligible_branch_types)
        group_count = min(
            value["positive_multibranch_groups"] for value in per_pool.values()
        )
        multibranch_steps = min(
            value["multibranch_steps"] for value in per_pool.values()
        )
        observed_pair_groups = min(
            value["positive_observed_pair_groups"] for value in per_pool.values()
        )
        eligible = (
            branch_count >= 2
            and multibranch_steps > 0
            and group_count >= positive_floor
        )
        reasons = []
        if branch_count < 2:
            reasons.append("fewer_than_two_branch_types_in_at_least_one_pool")
        if multibranch_steps == 0:
            reasons.append("no_aligned_multibranch_step_in_at_least_one_pool")
        if group_count < positive_floor:
            reasons.append("positive_multibranch_group_floor_not_met_in_every_pool")
        graph_concepts = set(graph.get("compatible_concept_ids", []))
        graph_primary_eligible = (
            eligible
            and bool(graph.get("graph_eligible"))
            and concept_id in graph_concepts
            and observed_pair_groups >= positive_floor
        )
        exclusion_reason = None if eligible else "|".join(reasons)
        eligibility_rows.append(
            {
                "concept_id": concept_id,
                "eligible_branch_type_count": branch_count,
                "eligible_branch_types": eligible_branch_types,
                "episode_skeleton_candidate_hash": sha256_file(
                    root / candidate_output
                ),
                "exclusion_reason": exclusion_reason,
                "fusion_primary_eligible": eligible,
                "graph_primary_eligible": graph_primary_eligible,
                "independent_multibranch_positive_group_count": group_count,
                "multibranch_event_count_by_pool": {
                    pool: int(values["positive_multibranch_event_count"])
                    for pool, values in per_pool.items()
                },
                "ontology_lock_hash": sha256_file(
                    root / "data/locked/ontology_lock.yaml"
                ),
                "split_manifest_hash": sha256_file(split_path),
                "minimum_multibranch_step_count": multibranch_steps,
                "minimum_observed_pair_positive_group_count": observed_pair_groups,
                "positive_group_floor": positive_floor,
                "per_pool_counts_json": json.dumps(per_pool, sort_keys=True),
                "model_outcomes_used": False,
                "skeleton_manifest_sha256": sha256_file(root / skeleton_output),
                "graph_eligibility_sha256": sha256_file(graph_path),
            }
        )
    eligibility = validate_fusion_eligibility_lock(
        pd.DataFrame(eligibility_rows), positive_group_floor=positive_floor
    )
    eligibility_output = "data/locked/fusion_eligibility_lock.parquet"
    write_parquet_artifact(root / eligibility_output, eligibility)
    return {
        "status": "pass",
        "output_paths": [
            template_output,
            template_instances_output,
            feature_manifest_output,
            skeleton_output,
            eligibility_output,
            candidate_output,
            projection_output,
            sealed_test_output,
        ],
        "inputs": hash_existing_inputs(
            root,
            [
                "data/locked/file_manifest.parquet",
                "data/locked/split_manifest.parquet",
                "data/locked/ontology_lock.yaml",
                "data/locked/rgbt_input_lock.json",
                "data/locked/methane_role_lock.json",
                "data/locked/graph_eligibility_lock.json",
            ],
        ),
        "details": {
            "row_count": len(skeleton),
            "test_row_count": len(test_rows),
            "fusion_eligible_concept_count": int(
                eligibility["fusion_primary_eligible"].sum()
            ),
        },
    }


class _UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _component_ids(skeleton: pd.DataFrame) -> Dict[str, str]:
    graph = _UnionFind()
    for item in skeleton[["episode_id", "raw_group_id"]].drop_duplicates().itertuples(
        index=False
    ):
        graph.union(f"episode:{item.episode_id}", f"group:{item.raw_group_id}")
    result = {}
    for episode_id in sorted(set(skeleton["episode_id"].astype(str))):
        root_id = graph.find(f"episode:{episode_id}")
        result[episode_id] = hashlib.sha256(root_id.encode("utf-8")).hexdigest()[:24]
    return result


def _partition_write(
    root: Path,
    base: str,
    frame: pd.DataFrame,
    *,
    pool_column: str = "pool",
    drop_pool_column: bool = False,
) -> list[str]:
    outputs = []
    for pool, group in frame.groupby(pool_column, sort=True):
        relative = f"{base}/{pool}.parquet"
        payload = group.drop(columns=[pool_column]) if drop_pool_column else group
        write_parquet_artifact(root / relative, payload.reset_index(drop=True))
        outputs.append(relative)
    return outputs


def build_and_audit_episode_features(
    root: Path, row: Mapping[str, str], arguments: Mapping[str, Any]
) -> Dict[str, Any]:
    del arguments
    if not _fusion_population_nonempty(root):
        return _not_applicable(
            root,
            step_id=row["step_id"],
            output="evidence/episode/E301_not_applicable.json",
            reason="fusion_eligible_population_empty_at_E058",
        )
    skeleton_path = root / "data/locked/episode_skeleton_manifest.parquet"
    skeleton = validate_episode_skeleton_manifest(pd.read_parquet(skeleton_path))
    prediction_files = sorted((root / "predictions/episode_branches").glob("*.parquet"))
    if not prediction_files:
        raise RuntimeError("E300 episode branch predictions are missing")
    predictions = pd.concat(
        [pd.read_parquet(path) for path in prediction_files], ignore_index=True
    )
    for path in prediction_files:
        assert_truth_free_prediction(path)
    required_prediction = {
        "skeleton_item_id",
        "record_id",
        "concept_id",
        "node_id",
        "modality",
        "calibrated_probability",
        "available",
        "generator_family_id",
        "model_hash",
        "calibrator_or_policy_hash",
        "skeleton_manifest_hash",
        "skeleton_inference_projection_hash",
    }
    if not required_prediction.issubset(predictions.columns):
        raise RuntimeError("Episode branch prediction schema is incomplete")
    if predictions["skeleton_item_id"].duplicated().any():
        raise RuntimeError("Episode branch predictions are not one row per skeleton item")
    projection_path = root / "data/seals/episode_skeleton_inference_projection.parquet"
    projection = pd.read_parquet(projection_path)
    projection_hash = sha256_file(projection_path)
    if set(predictions["skeleton_item_id"].astype(str)) != set(
        projection["skeleton_item_id"].astype(str)
    ):
        raise RuntimeError("Episode branch predictions do not close the inference projection")
    if set(predictions["skeleton_inference_projection_hash"].astype(str)) != {
        projection_hash
    }:
        raise RuntimeError("Episode branch predictions bind the wrong inference projection")
    eligibility = pd.read_parquet(
        root / "data/locked/fusion_eligibility_lock.parquet"
    )
    skeleton_hashes = set(eligibility["skeleton_manifest_sha256"].astype(str))
    if len(skeleton_hashes) != 1 or set(
        predictions["skeleton_manifest_hash"].astype(str)
    ) != skeleton_hashes:
        raise RuntimeError("Episode branch predictions bind the wrong skeleton manifest")
    if set(predictions["skeleton_item_id"]) != set(skeleton["skeleton_item_id"]):
        raise RuntimeError("Episode branch predictions do not close the skeleton")
    joined = skeleton.merge(
        predictions,
        on=[
            "skeleton_item_id",
            "record_id",
            "concept_id",
            "node_id",
            "generator_family_id",
        ],
        validate="one_to_one",
    )
    components = _component_ids(skeleton)
    branch_features = pd.read_parquet(
        root / "data/locked/episode_branch_feature_manifest.parquet"
    )
    if branch_features["record_id"].astype(str).duplicated().any():
        raise RuntimeError("Episode records must have globally unique record_id values")
    record_meta = branch_features.set_index(["record_id"])[
        ["dataset_id", "pair_id"]
    ].to_dict("index")
    features = joined[
        [
            "episode_id",
            "step_index",
            "record_id",
            "concept_id",
            "node_id",
            "modality",
            "calibrated_probability",
            "available",
            "pool",
        ]
    ].copy()
    features["quality_vector_json"] = [
        json.dumps(
            {
                "available": bool(available),
                "probability_margin": abs(float(probability) - 0.5),
            },
            sort_keys=True,
        )
        for probability, available in features[
            ["calibrated_probability", "available"]
        ].itertuples(index=False, name=None)
    ]
    feature_outputs = _partition_write(
        root,
        "data/locked/episode_features",
        features,
        drop_pool_column=True,
    )
    audit_rows = []
    label_rows = []
    graph_eligibility = load_json(root / "data/locked/graph_eligibility_lock.json")
    graph_lock_hash = sha256_file(root / "data/locked/graph_eligibility_lock.json")
    for item in joined.itertuples(index=False):
        metadata = record_meta[str(item.record_id)]
        pair_id = metadata["pair_id"]
        has_pair = pd.notna(pair_id) and bool(str(pair_id).strip())
        edge_provenance = (
            "observed_pair"
            if has_pair and graph_eligibility.get("graph_eligible")
            else "no_edge"
        )
        source_component_id = components[str(item.episode_id)]
        audit_rows.append(
            {
                "episode_id": str(item.episode_id),
                "step_index": int(item.step_index),
                "record_id": str(item.record_id),
                "episode_seed": int(item.episode_seed),
                "raw_group_id": str(item.raw_group_id),
                "source_component_id": source_component_id,
                "concept_id": str(item.concept_id),
                "node_id": str(item.node_id),
                "pair_id": str(pair_id) if has_pair else None,
                "edge_provenance": edge_provenance,
                "pool": str(item.pool),
                "dataset_id": str(metadata["dataset_id"]),
                "generator_family_id": str(item.generator_family_id),
                "template_instance_id": str(item.template_instance_id),
                "template_family": str(item.template_family),
                "feature_source_hash": str(item.model_hash),
                "audit_source_hash": sha256_file(skeleton_path),
            }
        )
    audit = pd.DataFrame(audit_rows).drop_duplicates(
        ["episode_id", "step_index", "concept_id", "node_id"]
    )
    for key, group in joined.groupby(
        ["episode_id", "step_index", "concept_id", "pool"], sort=True
    ):
        if group["event_position_role"].nunique() != 1:
            raise RuntimeError(f"Episode concept-step truth is inconsistent: {key}")
        if group["template_instance_id"].nunique() != 1:
            raise RuntimeError(f"Episode concept-step template is inconsistent: {key}")
        concept_id = str(key[2])
        role = str(group["event_position_role"].iloc[0])
        event_truth = int(role == "positive_observation")
        event_semantics = (
            "threshold_risk" if concept_id == METHANE_CONCEPT_ID else "observable_presence"
        )
        rule = (
            "future_threshold_risk_from_locked_methane_rule"
            if event_semantics == "threshold_risk"
            else "observable_presence_or_over_locked_node_annotations"
        )
        label_rows.append(
            {
                "episode_id": str(key[0]),
                "step_index": int(key[1]),
                "source_component_id": components[str(key[0])],
                "concept_id": concept_id,
                "event_truth": event_truth,
                "state_truth": "alarm" if event_truth else "normal",
                "evaluation_attributes_json": json.dumps({}, sort_keys=True),
                "event_semantics": event_semantics,
                "raw_annotation_hash": canonical_json_sha256(
                    {
                        "record_ids": sorted(group["record_id"].astype(str)),
                        "concept_id": concept_id,
                        "event_position_role": role,
                    }
                ),
                "generator_family_id": "|".join(
                    sorted(set(group["generator_family_id"].astype(str)))
                ),
                "template_instance_id": str(group["template_instance_id"].iloc[0]),
                "label_rule_hash": canonical_json_sha256(
                    {"rule": rule, "version": 1}
                ),
                "pool": str(key[3]),
            }
        )
    labels = pd.DataFrame(label_rows)
    audit_outputs = _partition_write(root, "data/locked/episode_feature_audit", audit)
    label_outputs = _partition_write(root, "data/locked/episode_labels", labels)
    edge_rows = []
    for key, group in audit.loc[audit["edge_provenance"] == "observed_pair"].groupby(
        ["episode_id", "step_index", "concept_id", "pair_id"], dropna=False
    ):
        nodes = sorted(set(group["node_id"].astype(str)))
        for source in nodes:
            for target in nodes:
                if source == target:
                    continue
                edge_rows.append(
                    {
                        "episode_id": str(key[0]),
                        "step_index": int(key[1]),
                        "concept_id": str(key[2]),
                        "source_node_id": source,
                        "target_node_id": target,
                        "modality_pair_embedding_id": "__to__".join(sorted((source, target))),
                        "edge_contract_hash": canonical_json_sha256(
                            {"source": source, "target": target, "provenance": "observed_pair"}
                        ),
                        "graph_eligibility_lock_hash": graph_lock_hash,
                        "pool": str(group["pool"].iloc[0]),
                    }
                )
    edges = pd.DataFrame(edge_rows)
    edge_outputs = []
    if not edges.empty:
        edge_outputs = _partition_write(
            root,
            "data/locked/episode_graph_edges",
            edges,
            drop_pool_column=True,
        )
    acl = sorted(DENIED_LABEL_ROLES)
    test_labels = root / "data/locked/episode_labels/D_e_te.parquet"
    candidate = {
        "scope": "episode",
        "stage": "candidate",
        "skeleton_manifest_hash": sha256_file(skeleton_path),
        "skeleton_candidate_seal_hash": sha256_file(
            root / "data/seals/episode_test_skeleton_candidate.json"
        ),
        "feature_manifest_hash": canonical_json_sha256(
            {path: sha256_file(root / path) for path in sorted(feature_outputs)}
        ),
        "label_manifest_hash": sha256_file(test_labels),
        "label_row_count": int(len(pd.read_parquet(test_labels))),
        "raw_group_count": int(
            audit.loc[audit["pool"] == "D_e_te", "raw_group_id"].nunique()
        ),
        "label_acl": acl,
        "acl_hash": canonical_json_sha256(acl),
        "artifact_contract_hash": sha256_file(
            root / "configs/artifact_contract.template.yaml"
        ),
        "created_at": utc_now(),
    }
    candidate_output = "data/seals/episode_test_seal_candidate.json"
    write_once_json(root / candidate_output, candidate)
    audit_receipt = {
        "schema_version": 1,
        "status": "pass",
        "feature_files": {path: sha256_file(root / path) for path in feature_outputs},
        "audit_files": {path: sha256_file(root / path) for path in audit_outputs},
        "label_files": {path: sha256_file(root / path) for path in label_outputs},
        "edge_files": {path: sha256_file(root / path) for path in edge_outputs},
        "forbidden_tensor_columns_present": False,
        "test_labels_denied_to_predictor": True,
        "created_at": utc_now(),
    }
    audit_output = "evidence/episode/independence_audit.json"
    write_json_artifact(root / audit_output, audit_receipt)
    return {
        "status": "pass",
        "output_paths": [
            "data/locked/episode_features",
            "data/locked/episode_feature_audit",
            "data/locked/episode_labels",
            candidate_output,
            audit_output,
            *( ["data/locked/episode_graph_edges"] if edge_outputs else [] ),
        ],
        "inputs": hash_existing_inputs(
            root,
            [
                "data/locked/episode_skeleton_manifest.parquet",
                "data/locked/fusion_eligibility_lock.parquet",
                "data/locked/episode_branch_feature_manifest.parquet",
                "predictions/episode_branches",
            ],
        ),
        "details": {
            "feature_row_count": len(features),
            "label_row_count": len(labels),
            "edge_row_count": len(edges),
        },
    }


def seal_episode_test(
    root: Path, row: Mapping[str, str], arguments: Mapping[str, Any]
) -> Dict[str, Any]:
    del arguments
    if not _fusion_population_nonempty(root):
        return _not_applicable(
            root,
            step_id=row["step_id"],
            output="evidence/episode/E302_not_applicable.json",
            reason="fusion_eligible_population_empty_at_E058",
        )
    candidate_path = root / "data/seals/episode_test_seal_candidate.json"
    skeleton_candidate = root / "data/seals/episode_test_skeleton_candidate.json"
    candidate = load_json(candidate_path)
    protocol_path = root / "configs/protocol_lock.pretest.yaml"
    source_path = root / "evidence/data/dataset_source_decision.json"
    payload = {
        "scope": "episode",
        "stage": "final",
        "split_hash": sha256_file(root / "data/locked/split_manifest.parquet"),
        "feature_manifest_hash": candidate["feature_manifest_hash"],
        "label_manifest_hash": candidate["label_manifest_hash"],
        "feature_acl": ["allow:test_predictor", "allow:evaluator"],
        "label_acl": list(candidate["label_acl"]),
        "artifact_contract_hash": candidate["artifact_contract_hash"],
        "sealed_at": utc_now(),
        "status": "sealed",
        "candidate_seal_hash": sha256_file(candidate_path),
        "episode_skeleton_candidate_seal_hash": sha256_file(skeleton_candidate),
        "protocol_hash": sha256_file(protocol_path),
        "source_hash": sha256_file(source_path),
        "matrix_hash": sha256_file(root / "configs/experiment_matrix.template.csv"),
    }
    validate_test_seal(payload)
    output = "data/seals/episode_test_seal.json"
    write_once_json(root / output, payload)
    return {
        "status": "pass",
        "output_paths": [output],
        "inputs": hash_existing_inputs(
            root,
            [
                "data/seals/episode_test_seal_candidate.json",
                "data/seals/episode_test_skeleton_candidate.json",
                "configs/protocol_lock.pretest.yaml",
            ],
        ),
        "details": {"scope": "episode", "stage": "final"},
    }


def release_episode_test(
    root: Path, row: Mapping[str, str], arguments: Mapping[str, Any]
) -> Dict[str, Any]:
    del arguments
    if not _fusion_population_nonempty(root):
        return _not_applicable(
            root,
            step_id=row["step_id"],
            output="evidence/episode/E306_not_applicable.json",
            reason="fusion_eligible_population_empty_at_E058",
        )
    lock = root / "predictions/locked/episodes_prediction_lock.json"
    payload = load_json(lock)
    validate_prediction_lock(payload)
    prediction_paths = {
        family: root / path for family, path in payload.get("prediction_paths", {}).items()
    }
    verify_prediction_lock(lock, prediction_paths)
    seal = root / "data/seals/episode_test_seal.json"
    protocol = root / "configs/protocol_lock.pretest.yaml"
    output = "data/releases/episode_test_release.json"
    write_test_release(
        output_path=root / output,
        scope="episode",
        seal_path=seal,
        prediction_lock_path=lock,
        protocol_hash=sha256_file(protocol),
        source_hash=sha256_file(root / "evidence/data/dataset_source_decision.json"),
        matrix_hash=sha256_file(root / "configs/experiment_matrix.template.csv"),
        evaluator_identity="mining1_independent_evaluator",
        evaluator_label_acl=["allow:evaluator:mining1_independent_evaluator"],
        signer_identity="mining1_release_authority",
    )
    return {
        "status": "pass",
        "output_paths": [output],
        "inputs": hash_existing_inputs(
            root,
            [
                "predictions/locked/episodes_prediction_lock.json",
                "data/seals/episode_test_seal.json",
                "configs/protocol_lock.pretest.yaml",
            ],
        ),
        "details": {"scope": "episode"},
    }
