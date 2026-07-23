from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import pandas as pd

from .manifests import ManifestValidationError, POOLS, validate_split_manifest


FEWSHOT_COLUMNS = {
    "dataset_id",
    "direction_id",
    "ratio_percent",
    "subset_seed",
    "raw_group_id",
    "included",
    "class_group_counts_json",
    "parent_manifest_hash",
}

TEMPORAL_POOL_ORDER = (
    "D_b_tr",
    "D_e_tr",
    "D_b_sel",
    "D_e_sel",
    "D_b_prob",
    "D_e_pol",
    "D_b_te",
    "D_e_te",
)
TEMPORAL_FAMILY_POOL_ORDER = {
    "branch": ("D_b_tr", "D_b_sel", "D_b_prob", "D_b_te"),
    "episode": ("D_e_tr", "D_e_sel", "D_e_pol", "D_e_te"),
}


def _unit_interval(preimage: str) -> float:
    value = int(hashlib.sha256(preimage.encode("utf-8")).hexdigest(), 16)
    return value / float(2**256)


def assign_group_pools(
    records: pd.DataFrame,
    *,
    pool_weights: Mapping[str, float],
    seed: int,
    split_version: str,
    ontology_hash: str,
    dedup_report_hash: str,
) -> pd.DataFrame:
    required = {"dataset_id", "record_id", "raw_group_id"}
    missing = sorted(required.difference(records.columns))
    if missing:
        raise ManifestValidationError(f"split input is missing columns: {missing}")
    if not pool_weights or not set(pool_weights).issubset(POOLS):
        raise ManifestValidationError("split pool weights use unknown or empty pools")
    if any(weight <= 0 for weight in pool_weights.values()):
        raise ManifestValidationError("split pool weights must be positive")

    total = float(sum(pool_weights.values()))
    cumulative = []
    running = 0.0
    for pool, weight in sorted(pool_weights.items()):
        running += float(weight) / total
        cumulative.append((running, pool))

    group_to_pool: Dict[Tuple[str, str], str] = {}
    groups = records[["dataset_id", "raw_group_id"]].drop_duplicates()
    for dataset_id, raw_group_id in groups.itertuples(index=False, name=None):
        if pd.isna(raw_group_id) or not str(raw_group_id).strip():
            raise ManifestValidationError("raw_group_id must exist before splitting")
        score = _unit_interval(f"{seed}|{dataset_id}|{raw_group_id}")
        selected = cumulative[-1][1]
        for boundary, pool in cumulative:
            if score < boundary:
                selected = pool
                break
        group_to_pool[(str(dataset_id), str(raw_group_id))] = selected

    result = records[["dataset_id", "record_id", "raw_group_id"]].copy()
    result["pool"] = [
        group_to_pool[(str(dataset), str(group))]
        for dataset, group in result[["dataset_id", "raw_group_id"]].itertuples(
            index=False, name=None
        )
    ]
    result["split_seed"] = int(seed)
    result["split_version"] = split_version
    result["ontology_hash"] = ontology_hash
    result["dedup_report_hash"] = dedup_report_hash
    return validate_split_manifest(result)


def _balanced_pool_quotas(
    component_count: int,
    pool_weights: Mapping[str, float],
    *,
    seed: int,
    dataset_id: str,
) -> Dict[str, int]:
    pools = sorted(pool_weights)
    if component_count < len(pools):
        raise ManifestValidationError(
            f"dataset {dataset_id} has {component_count} independent components for {len(pools)} pools"
        )
    total = float(sum(pool_weights.values()))
    exact = {
        pool: component_count * float(pool_weights[pool]) / total for pool in pools
    }
    quotas = {pool: int(math.floor(exact[pool])) for pool in pools}
    remainder_order = sorted(
        pools,
        key=lambda pool: (
            -(exact[pool] - quotas[pool]),
            hashlib.sha256(
                f"{seed}|{dataset_id}|quota-remainder|{pool}".encode("utf-8")
            ).hexdigest(),
        ),
    )
    for pool in remainder_order[: component_count - sum(quotas.values())]:
        quotas[pool] += 1
    for empty_pool in sorted(
        (pool for pool in pools if quotas[pool] == 0),
        key=lambda pool: hashlib.sha256(
            f"{seed}|{dataset_id}|quota-empty|{pool}".encode("utf-8")
        ).hexdigest(),
    ):
        donors = [pool for pool in pools if quotas[pool] > 1]
        if not donors:
            raise ManifestValidationError(
                f"dataset {dataset_id} cannot populate every required pool"
            )
        donor = min(
            donors,
            key=lambda pool: (
                -(quotas[pool] - exact[pool]),
                -quotas[pool],
                hashlib.sha256(
                    f"{seed}|{dataset_id}|quota-donor|{pool}".encode("utf-8")
                ).hexdigest(),
            ),
        )
        quotas[donor] -= 1
        quotas[empty_pool] = 1
    if sum(quotas.values()) != component_count or any(value <= 0 for value in quotas.values()):
        raise ManifestValidationError(f"dataset {dataset_id} has invalid balanced quotas")
    return quotas


def assign_temporal_cohort_pools(
    component_records: pd.DataFrame,
    *,
    pool_weights: Mapping[str, float],
    seed: int,
    dataset_id: str,
    group_duration_seconds: int,
    purge_gap_seconds: int,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    required = {"component_id", "dataset_id", "timestamp_or_order"}
    missing = sorted(required.difference(component_records.columns))
    if missing:
        raise ManifestValidationError(f"temporal component records are missing columns: {missing}")
    if set(pool_weights) != POOLS or any(float(value) <= 0 for value in pool_weights.values()):
        raise ManifestValidationError("temporal cohort assignment requires positive eight-pool weights")
    if group_duration_seconds <= 0 or purge_gap_seconds <= 0:
        raise ManifestValidationError("temporal cohort duration and purge gap must be positive")
    records = component_records[
        ["component_id", "dataset_id", "timestamp_or_order"]
    ].copy()
    records["component_id"] = records["component_id"].astype(str)
    records["dataset_id"] = records["dataset_id"].astype(str)
    if records.empty or set(records["dataset_id"]) != {str(dataset_id)}:
        raise ManifestValidationError("temporal component records do not match the selected dataset")
    if records["component_id"].str.strip().eq("").any() or records["timestamp_or_order"].isna().any():
        raise ManifestValidationError("temporal component records contain an empty identifier or time")
    try:
        timestamps = pd.to_datetime(records["timestamp_or_order"], utc=True, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ManifestValidationError("temporal cohort timestamps are not parseable UTC times") from exc
    records["cohort_epoch_seconds"] = timestamps.astype("int64") // 1_000_000_000
    component_cohort_counts = records.groupby("component_id")["cohort_epoch_seconds"].nunique()
    if int(component_cohort_counts.max()) != 1:
        raise ManifestValidationError("one temporal dedup component spans multiple time cohorts")
    component_cohort = {
        str(component_id): int(frame["cohort_epoch_seconds"].iloc[0])
        for component_id, frame in records.groupby("component_id", sort=True)
    }
    cohort_components = {
        int(cohort): sorted(set(frame["component_id"].astype(str)))
        for cohort, frame in records.groupby("cohort_epoch_seconds", sort=True)
    }
    cohorts = sorted(cohort_components)
    if any(
        right - left != group_duration_seconds
        for left, right in zip(cohorts, cohorts[1:])
    ):
        raise ManifestValidationError("temporal source cohorts are not one continuous ordered series")
    quotas = _balanced_pool_quotas(
        len(cohorts), pool_weights, seed=seed, dataset_id=f"{dataset_id}:temporal_cohorts"
    )
    assignment: Dict[str, str] = {}
    pool_blocks = []
    cursor = 0
    for ordinal, pool in enumerate(TEMPORAL_POOL_ORDER):
        count = int(quotas[pool])
        selected = cohorts[cursor : cursor + count]
        if len(selected) != count or not selected:
            raise ManifestValidationError(f"temporal pool {pool} has an invalid cohort block")
        components = [component for cohort in selected for component in cohort_components[cohort]]
        for component in components:
            if component in assignment:
                raise ManifestValidationError("temporal component received conflicting pool assignments")
            assignment[component] = pool
        pool_blocks.append(
            {
                "pool": pool,
                "ordinal": ordinal,
                "cohort_count": count,
                "component_count": len(components),
                "first_cohort_start": pd.Timestamp(selected[0], unit="s", tz="UTC").isoformat(),
                "last_cohort_start": pd.Timestamp(selected[-1], unit="s", tz="UTC").isoformat(),
                "first_cohort_epoch_seconds": selected[0],
                "last_cohort_epoch_seconds": selected[-1],
                "block_end_exclusive_epoch_seconds": selected[-1] + group_duration_seconds,
            }
        )
        cursor += count
    if cursor != len(cohorts) or set(assignment) != set(component_cohort):
        raise ManifestValidationError("temporal cohort assignment did not consume every component")
    block_by_pool = {str(item["pool"]): item for item in pool_blocks}
    family_gap_evidence = []
    for family, pools in TEMPORAL_FAMILY_POOL_ORDER.items():
        for earlier, later in zip(pools, pools[1:]):
            gap_seconds = int(
                block_by_pool[later]["first_cohort_epoch_seconds"]
                - block_by_pool[earlier]["block_end_exclusive_epoch_seconds"]
            )
            passed = gap_seconds >= purge_gap_seconds
            family_gap_evidence.append(
                {
                    "family": family,
                    "from_pool": earlier,
                    "to_pool": later,
                    "gap_seconds": gap_seconds,
                    "required_gap_seconds": purge_gap_seconds,
                    "passed": passed,
                }
            )
            if not passed:
                raise ManifestValidationError(
                    f"temporal {family} gap {earlier}->{later} is {gap_seconds}s, below {purge_gap_seconds}s"
                )
    component_pool_counts = {
        pool: sum(value == pool for value in assignment.values()) for pool in sorted(POOLS)
    }
    diagnostics = {
        "algorithm": "chronological_cohort_contiguous_alternating_role_family_v1",
        "dataset_id": str(dataset_id),
        "temporal_pool_order": list(TEMPORAL_POOL_ORDER),
        "family_pool_order": {
            family: list(pools) for family, pools in TEMPORAL_FAMILY_POOL_ORDER.items()
        },
        "group_duration_seconds": int(group_duration_seconds),
        "purge_gap_seconds_required": int(purge_gap_seconds),
        "cohort_count": len(cohorts),
        "component_count": len(component_cohort),
        "cohort_pool_counts": {pool: int(quotas[pool]) for pool in sorted(POOLS)},
        "component_pool_counts": component_pool_counts,
        "pool_blocks": pool_blocks,
        "family_gap_evidence": family_gap_evidence,
        "minimum_family_gap_seconds": min(
            int(item["gap_seconds"]) for item in family_gap_evidence
        ),
        "same_cohort_cross_pool_count": 0,
        "non_contiguous_pool_block_count": 0,
    }
    return assignment, diagnostics


def assign_balanced_component_pools(
    component_membership: pd.DataFrame,
    *,
    pool_weights: Mapping[str, float],
    seed: int,
    max_cross_component_search_nodes: int = 100_000,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    required = {"component_id", "dataset_id"}
    missing = sorted(required.difference(component_membership.columns))
    if missing:
        raise ManifestValidationError(f"component membership is missing columns: {missing}")
    if not pool_weights or set(pool_weights) != POOLS:
        raise ManifestValidationError("balanced component assignment requires all eight pools")
    if any(float(weight) <= 0 for weight in pool_weights.values()):
        raise ManifestValidationError("balanced component pool weights must be positive")
    membership = component_membership[["component_id", "dataset_id"]].copy()
    membership["component_id"] = membership["component_id"].astype(str)
    membership["dataset_id"] = membership["dataset_id"].astype(str)
    if membership.empty or membership.apply(lambda column: column.str.strip().eq("").any()).any():
        raise ManifestValidationError("component membership contains an empty identifier")
    membership = membership.drop_duplicates().sort_values(
        ["component_id", "dataset_id"], kind="mergesort"
    )
    components_by_dataset = {
        str(dataset_id): sorted(set(frame["component_id"]))
        for dataset_id, frame in membership.groupby("dataset_id", sort=True)
    }
    datasets_by_component = {
        str(component_id): sorted(set(frame["dataset_id"]))
        for component_id, frame in membership.groupby("component_id", sort=True)
    }
    quotas = {
        dataset_id: _balanced_pool_quotas(
            len(components), pool_weights, seed=seed, dataset_id=dataset_id
        )
        for dataset_id, components in components_by_dataset.items()
    }
    remaining = {dataset_id: dict(values) for dataset_id, values in quotas.items()}
    assignment: Dict[str, str] = {}
    cross_components = sorted(
        (
            component_id
            for component_id, datasets in datasets_by_component.items()
            if len(datasets) > 1
        ),
        key=lambda component_id: hashlib.sha256(
            f"{seed}|cross-component|{component_id}".encode("utf-8")
        ).hexdigest(),
    )
    search_nodes = 0

    def solve_cross(unassigned: Tuple[str, ...]) -> bool:
        nonlocal search_nodes
        search_nodes += 1
        if search_nodes > max_cross_component_search_nodes:
            raise ManifestValidationError("cross-dataset component assignment search limit exceeded")
        if not unassigned:
            return True
        candidate_details = []
        for component_id in unassigned:
            datasets = datasets_by_component[component_id]
            feasible = [
                pool
                for pool in sorted(pool_weights)
                if all(remaining[dataset_id][pool] > 0 for dataset_id in datasets)
            ]
            candidate_details.append((len(feasible), component_id, feasible))
        _, component_id, feasible_pools = min(
            candidate_details,
            key=lambda item: (
                item[0],
                hashlib.sha256(
                    f"{seed}|cross-mrv|{item[1]}".encode("utf-8")
                ).hexdigest(),
            ),
        )
        if not feasible_pools:
            return False
        datasets = datasets_by_component[component_id]
        ordered_pools = sorted(
            feasible_pools,
            key=lambda pool: (
                -sum(
                    remaining[dataset_id][pool] / quotas[dataset_id][pool]
                    for dataset_id in datasets
                ),
                hashlib.sha256(
                    f"{seed}|cross-pool|{component_id}|{pool}".encode("utf-8")
                ).hexdigest(),
            ),
        )
        next_unassigned = tuple(value for value in unassigned if value != component_id)
        for pool in ordered_pools:
            assignment[component_id] = pool
            for dataset_id in datasets:
                remaining[dataset_id][pool] -= 1
            if solve_cross(next_unassigned):
                return True
            for dataset_id in datasets:
                remaining[dataset_id][pool] += 1
            assignment.pop(component_id, None)
        return False

    if not solve_cross(tuple(cross_components)):
        raise ManifestValidationError("cross-dataset components cannot satisfy balanced pool quotas")

    for dataset_id, components in sorted(components_by_dataset.items()):
        local_components = [value for value in components if value not in assignment]
        slots = []
        for pool in sorted(pool_weights):
            for ordinal in range(remaining[dataset_id][pool]):
                slots.append(
                    (
                        hashlib.sha256(
                            f"{seed}|{dataset_id}|slot|{pool}|{ordinal}".encode("utf-8")
                        ).hexdigest(),
                        pool,
                    )
                )
        ranked_components = sorted(
            local_components,
            key=lambda component_id: hashlib.sha256(
                f"{seed}|{dataset_id}|component|{component_id}".encode("utf-8")
            ).hexdigest(),
        )
        ranked_slots = [pool for _, pool in sorted(slots)]
        if len(ranked_components) != len(ranked_slots):
            raise ManifestValidationError(
                f"dataset {dataset_id} balanced slot count does not match local components"
            )
        for component_id, pool in zip(ranked_components, ranked_slots):
            if component_id in assignment:
                raise ManifestValidationError("component received conflicting pool assignments")
            assignment[component_id] = pool
    if set(assignment) != set(datasets_by_component):
        raise ManifestValidationError("not every component received a balanced pool assignment")

    dataset_pool_counts = {
        dataset_id: {
            pool: sum(assignment[component_id] == pool for component_id in components)
            for pool in sorted(pool_weights)
        }
        for dataset_id, components in components_by_dataset.items()
    }
    if dataset_pool_counts != quotas:
        raise ManifestValidationError("balanced component assignment does not match quotas")
    diagnostics = {
        "algorithm": "largest_remainder_minimum_one_per_pool_with_cross_component_csp_v1",
        "dataset_component_counts": {
            dataset_id: len(components)
            for dataset_id, components in components_by_dataset.items()
        },
        "dataset_pool_component_counts": dataset_pool_counts,
        "cross_dataset_component_count": len(cross_components),
        "cross_component_search_nodes": search_nodes,
        "global_component_count": len(datasets_by_component),
        "global_pool_component_counts": {
            pool: sum(value == pool for value in assignment.values())
            for pool in sorted(pool_weights)
        },
    }
    return assignment, diagnostics


def assert_no_duplicate_cross_pool(
    split_manifest: pd.DataFrame, duplicate_pairs: pd.DataFrame
) -> None:
    split = validate_split_manifest(split_manifest)
    lookup = {
        (str(row.dataset_id), str(row.record_id)): str(row.pool)
        for row in split.itertuples(index=False)
    }
    required = {
        "left_dataset_id",
        "left_record_id",
        "right_dataset_id",
        "right_record_id",
    }
    if not required.issubset(duplicate_pairs.columns):
        raise ManifestValidationError("duplicate pair schema is incomplete")
    for row in duplicate_pairs.itertuples(index=False):
        left = (str(row.left_dataset_id), str(row.left_record_id))
        right = (str(row.right_dataset_id), str(row.right_record_id))
        if left not in lookup or right not in lookup:
            raise ManifestValidationError("duplicate pair references an unknown record")
        if lookup[left] != lookup[right]:
            raise ManifestValidationError(
                f"duplicate records cross pools: {left}={lookup[left]}, {right}={lookup[right]}"
            )


def build_fewshot_manifest(
    split_manifest: pd.DataFrame,
    *,
    direction_id: str,
    subset_seeds: Sequence[int],
    group_classes: Mapping[Tuple[str, str], Iterable[str]],
    ratio_percent: int = 10,
    parent_manifest_hash: str,
) -> pd.DataFrame:
    if ratio_percent != 10:
        raise ManifestValidationError("only the preregistered 10 percent ratio is allowed")
    normalized_seeds = [int(seed) for seed in subset_seeds]
    if len(normalized_seeds) != 3 or len(set(normalized_seeds)) != 3:
        raise ManifestValidationError("exactly three distinct few-shot subset seeds are required")
    split = validate_split_manifest(split_manifest)
    training_groups = split.loc[split["pool"] == "D_b_tr", ["dataset_id", "raw_group_id"]].drop_duplicates()
    rows = []
    for dataset_id, dataset_groups in training_groups.groupby("dataset_id"):
        groups = sorted(str(value) for value in dataset_groups["raw_group_id"])
        if not groups:
            continue
        selected_count = max(1, int(math.ceil(len(groups) * ratio_percent / 100.0)))
        for subset_seed in normalized_seeds:
            ranked = sorted(
                groups,
                key=lambda group: hashlib.sha256(
                    f"{subset_seed}|{dataset_id}|{direction_id}|{group}".encode("utf-8")
                ).hexdigest(),
            )
            included = set(ranked[:selected_count])
            counts: Dict[str, int] = {}
            for group in included:
                for class_id in set(group_classes.get((str(dataset_id), group), ())):
                    counts[str(class_id)] = counts.get(str(class_id), 0) + 1
            counts_json = json.dumps(counts, ensure_ascii=True, sort_keys=True)
            for group in groups:
                rows.append(
                    {
                        "dataset_id": str(dataset_id),
                        "direction_id": direction_id,
                        "ratio_percent": 10,
                        "subset_seed": int(subset_seed),
                        "raw_group_id": group,
                        "included": group in included,
                        "class_group_counts_json": counts_json,
                        "parent_manifest_hash": parent_manifest_hash,
                    }
                )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ManifestValidationError("no D_b_tr groups are available for few-shot sampling")
    key = ["dataset_id", "direction_id", "ratio_percent", "subset_seed", "raw_group_id"]
    if result.duplicated(key).any():
        raise ManifestValidationError("few-shot primary key is not unique")
    if not result["parent_manifest_hash"].str.fullmatch(r"[0-9a-f]{64}").all():
        raise ManifestValidationError("few-shot parent manifest hash is invalid")
    return result


def assert_shared_fewshot_groups(*condition_manifests: pd.DataFrame) -> None:
    if len(condition_manifests) < 2:
        raise ManifestValidationError("at least two condition manifests are required")

    def signature(
        frame: pd.DataFrame,
    ) -> Dict[Tuple[str, str, int], Tuple[Tuple[str, ...], Tuple[str, ...]]]:
        result: Dict[Tuple[str, str, int], Tuple[Tuple[str, ...], Tuple[str, ...]]] = {}
        for key, group in frame.groupby(["dataset_id", "direction_id", "subset_seed"]):
            all_groups = tuple(sorted(str(value) for value in group["raw_group_id"]))
            included_groups = tuple(
                sorted(str(value) for value in group.loc[group["included"], "raw_group_id"])
            )
            result[(str(key[0]), str(key[1]), int(key[2]))] = (
                all_groups,
                included_groups,
            )
        return result

    reference = signature(condition_manifests[0])
    if any(signature(frame) != reference for frame in condition_manifests[1:]):
        raise ManifestValidationError("few-shot conditions do not share identical raw groups")


def causal_window_masks(
    timestamps: pd.Series,
    *,
    anchor: pd.Timestamp,
    history_seconds: int,
    horizon_seconds: int,
) -> Tuple[pd.Series, pd.Series]:
    if history_seconds <= 0 or horizon_seconds <= 0:
        raise ManifestValidationError("history and horizon must be positive")
    values = pd.to_datetime(timestamps, utc=True, errors="raise")
    anchor_utc = pd.Timestamp(anchor)
    if anchor_utc.tzinfo is None:
        anchor_utc = anchor_utc.tz_localize("UTC")
    else:
        anchor_utc = anchor_utc.tz_convert("UTC")
    history_start = anchor_utc - pd.Timedelta(seconds=history_seconds)
    forecast_end = anchor_utc + pd.Timedelta(seconds=horizon_seconds)
    history = (values > history_start) & (values <= anchor_utc)
    forecast = (values > anchor_utc) & (values <= forecast_end)
    if (history & forecast).any():
        raise ManifestValidationError("causal history overlaps the prediction horizon")
    return history, forecast
