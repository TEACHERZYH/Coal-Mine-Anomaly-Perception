from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import pandas as pd
import yaml


POOLS = {
    "D_b_tr",
    "D_b_sel",
    "D_b_prob",
    "D_b_te",
    "D_e_tr",
    "D_e_sel",
    "D_e_pol",
    "D_e_te",
}
SPLIT_COLUMNS = [
    "dataset_id",
    "record_id",
    "raw_group_id",
    "pool",
    "split_seed",
    "split_version",
    "ontology_hash",
    "dedup_report_hash",
]
SPLIT_VERSION = "minimal_v3_component_balanced_with_methane_temporal_cohorts"
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class _DisjointSet:
    def __init__(self, values: Iterable[Tuple[str, str]]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: Tuple[str, str]) -> Tuple[str, str]:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: Tuple[str, str], right: Tuple[str, str]) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        self.parent[second] = first


def _expect_equal(errors: list[str], name: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        errors.append(f"{name}: observed={observed!r}, expected={expected!r}")


def _balanced_quotas(
    component_count: int,
    weights: Mapping[str, float],
    *,
    seed: int,
    dataset_id: str,
) -> Dict[str, int]:
    pools = sorted(weights)
    if component_count < len(pools):
        raise ValueError(
            f"dataset {dataset_id} has {component_count} components for {len(pools)} pools"
        )
    total = float(sum(weights.values()))
    exact = {pool: component_count * float(weights[pool]) / total for pool in pools}
    quotas = {pool: int(math.floor(exact[pool])) for pool in pools}
    order = sorted(
        pools,
        key=lambda pool: (
            -(exact[pool] - quotas[pool]),
            hashlib.sha256(
                f"{seed}|{dataset_id}|quota-remainder|{pool}".encode("utf-8")
            ).hexdigest(),
        ),
    )
    for pool in order[: component_count - sum(quotas.values())]:
        quotas[pool] += 1
    empty_order = sorted(
        (pool for pool in pools if quotas[pool] == 0),
        key=lambda pool: hashlib.sha256(
            f"{seed}|{dataset_id}|quota-empty|{pool}".encode("utf-8")
        ).hexdigest(),
    )
    for empty_pool in empty_order:
        donors = [pool for pool in pools if quotas[pool] > 1]
        if not donors:
            raise ValueError(f"dataset {dataset_id} cannot populate all pools")
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
    return quotas


def _combined_weights(protocol: Mapping[str, Any]) -> Dict[str, float]:
    data = protocol["data"]
    family = data["target_role_family_weights"]
    branch = data["branch_pool_weights"]
    episode = data["episode_pool_weights"]
    weights = {
        **{pool: float(family["branch"]) * float(value) for pool, value in branch.items()},
        **{pool: float(family["episode"]) * float(value) for pool, value in episode.items()},
    }
    if set(weights) != POOLS or abs(sum(weights.values()) - 1.0) > 1.0e-9:
        raise ValueError("protocol does not define normalized eight-pool weights")
    return weights


def _review_temporal_cohorts(
    *,
    errors: list[str],
    joined: pd.DataFrame,
    dataset_id: str,
    disjoint: _DisjointSet,
    component_pool: Mapping[Tuple[str, str], str],
    weights: Mapping[str, float],
    seed: int,
    group_duration_seconds: int,
    purge_gap_seconds: int,
) -> Dict[str, Any]:
    temporal = joined.loc[joined["dataset_id"].astype(str) == dataset_id].copy()
    if temporal.empty:
        errors.append("selected methane dataset has no records")
        return {}
    if temporal["timestamp_or_order"].isna().any():
        errors.append("selected methane dataset has missing cohort timestamps")
        return {}
    try:
        temporal["cohort_epoch_seconds"] = (
            pd.to_datetime(temporal["timestamp_or_order"], utc=True, errors="raise").astype(
                "int64"
            )
            // 1_000_000_000
        )
    except (TypeError, ValueError):
        errors.append("selected methane dataset has unparseable UTC cohort timestamps")
        return {}
    temporal["component"] = [
        disjoint.find((dataset_id, str(group)))
        for group in temporal["raw_group_id_manifest"].astype(str)
    ]
    component_cohort_counts = temporal.groupby("component")["cohort_epoch_seconds"].nunique()
    if int(component_cohort_counts.max()) != 1:
        errors.append("one methane dedup component spans multiple time cohorts")
        return {}
    cohort_components = {
        int(cohort): set(frame["component"])
        for cohort, frame in temporal.groupby("cohort_epoch_seconds", sort=True)
    }
    cohorts = sorted(cohort_components)
    if any(
        right - left != group_duration_seconds
        for left, right in zip(cohorts, cohorts[1:])
    ):
        errors.append("methane source cohorts are not one continuous ordered series")
    try:
        quotas = _balanced_quotas(
            len(cohorts),
            weights,
            seed=seed,
            dataset_id=f"{dataset_id}:temporal_cohorts",
        )
    except ValueError as exc:
        errors.append(str(exc))
        return {}
    expected_component_pool: Dict[Tuple[str, str], str] = {}
    pool_blocks = []
    cursor = 0
    same_cohort_cross_pool_count = 0
    for ordinal, pool in enumerate(TEMPORAL_POOL_ORDER):
        count = int(quotas[pool])
        selected = cohorts[cursor : cursor + count]
        if len(selected) != count or not selected:
            errors.append(f"methane temporal pool {pool} has an invalid cohort block")
            return {}
        components = [component for cohort in selected for component in cohort_components[cohort]]
        for cohort in selected:
            observed = {component_pool[component] for component in cohort_components[cohort]}
            if len(observed) != 1:
                same_cohort_cross_pool_count += 1
            if observed != {pool}:
                errors.append(
                    f"methane cohort {cohort} pool assignment is {sorted(observed)}, expected {pool}"
                )
        for component in components:
            expected_component_pool[component] = pool
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
    if cursor != len(cohorts):
        errors.append("methane temporal assignment does not consume every cohort")
    methane_components = set().union(*cohort_components.values())
    if set(expected_component_pool) != methane_components:
        errors.append("methane temporal assignment does not consume every component")
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
                errors.append(
                    f"methane {family} gap {earlier}->{later} is {gap_seconds}s, below {purge_gap_seconds}s"
                )
    component_pool_counts = {
        pool: sum(value == pool for value in expected_component_pool.values())
        for pool in sorted(POOLS)
    }
    return {
        "algorithm": "chronological_cohort_contiguous_alternating_role_family_v1",
        "dataset_id": dataset_id,
        "temporal_pool_order": list(TEMPORAL_POOL_ORDER),
        "family_pool_order": {
            family: list(pools) for family, pools in TEMPORAL_FAMILY_POOL_ORDER.items()
        },
        "group_duration_seconds": int(group_duration_seconds),
        "purge_gap_seconds_required": int(purge_gap_seconds),
        "cohort_count": len(cohorts),
        "component_count": len(methane_components),
        "cohort_pool_counts": {pool: int(quotas[pool]) for pool in sorted(POOLS)},
        "component_pool_counts": component_pool_counts,
        "pool_blocks": pool_blocks,
        "family_gap_evidence": family_gap_evidence,
        "minimum_family_gap_seconds": min(
            int(item["gap_seconds"]) for item in family_gap_evidence
        ),
        "same_cohort_cross_pool_count": same_cohort_cross_pool_count,
        "non_contiguous_pool_block_count": 0,
    }


def _result(
    root: Path,
    errors: Sequence[str],
    paths: Sequence[Path],
    diagnostics: Mapping[str, Any],
) -> Dict[str, Any]:
    bindings = {
        path.relative_to(root).as_posix(): _sha256_file(path)
        for path in paths
        if path.is_file()
    }
    return {
        "schema_version": 1,
        "step_id": "E038",
        "reviewer": "independent_local_split_reconciler",
        "status": "pass" if not errors else "fail",
        "error_count": len(errors),
        "errors": list(errors),
        "bindings": bindings,
        "diagnostics": dict(diagnostics),
        "model_outcomes_used": False,
        "performance_claims_authorized": False,
    }


def review_e038(project_root: Path) -> Dict[str, Any]:
    root = Path(project_root).resolve()
    manifest_path = root / "data/locked/file_manifest.parquet"
    ontology_path = root / "data/locked/ontology_lock.yaml"
    dedup_path = root / "data/locked/dedup_report.parquet"
    split_path = root / "data/locked/split_manifest.parquet"
    protocol_path = root / "configs/protocol_lock.template.yaml"
    decision_path = root / "evidence/data/dataset_source_decision.json"
    receipt_path = root / "evidence/command_receipts/E038.json"
    paths = (
        manifest_path,
        ontology_path,
        dedup_path,
        split_path,
        protocol_path,
        decision_path,
        receipt_path,
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = pd.read_parquet(manifest_path)
    dedup = pd.read_parquet(dedup_path)
    split = pd.read_parquet(split_path)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8-sig"))
    decision = json.loads(decision_path.read_text(encoding="utf-8-sig"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
    errors: list[str] = []
    diagnostics: Dict[str, Any] = {}
    required_manifest = {
        "dataset_id",
        "record_id",
        "raw_group_id",
        "pair_id",
        "modality",
        "timestamp_or_order",
    }
    missing_manifest = sorted(required_manifest - set(manifest.columns))
    if missing_manifest:
        errors.append(f"manifest missing columns: {missing_manifest}")
    _expect_equal(errors, "split_columns", list(split.columns), SPLIT_COLUMNS)
    if errors:
        return _result(root, errors, paths, diagnostics)

    key_columns = ["dataset_id", "record_id"]
    if manifest.duplicated(key_columns).any():
        errors.append("file manifest keys are not unique")
    if split.duplicated(key_columns).any():
        errors.append("split manifest keys are not unique")
    manifest_keys = {
        (str(row.dataset_id), str(row.record_id))
        for row in manifest[key_columns].itertuples(index=False)
    }
    split_keys = {
        (str(row.dataset_id), str(row.record_id))
        for row in split[key_columns].itertuples(index=False)
    }
    _expect_equal(errors, "record_key_set", split_keys, manifest_keys)
    if split_keys != manifest_keys:
        return _result(root, errors, paths, diagnostics)
    joined = manifest.merge(
        split,
        on=key_columns,
        how="inner",
        suffixes=("_manifest", "_split"),
        validate="one_to_one",
    )
    raw_group_mismatch_count = int(
        (
            joined["raw_group_id_manifest"].astype(str)
            != joined["raw_group_id_split"].astype(str)
        ).sum()
    )
    _expect_equal(errors, "raw_group_mismatch_count", raw_group_mismatch_count, 0)
    if not set(split["pool"].astype(str)).issubset(POOLS):
        errors.append("split contains an unknown pool")

    seed = int(protocol["seeds"]["split"])
    ontology_hash = _sha256_file(ontology_path)
    dedup_hash = _sha256_file(dedup_path)
    _expect_equal(errors, "split_seed_values", sorted(set(split["split_seed"])), [seed])
    _expect_equal(
        errors, "split_version_values", sorted(set(split["split_version"])), [SPLIT_VERSION]
    )
    _expect_equal(
        errors, "ontology_hash_values", sorted(set(split["ontology_hash"])), [ontology_hash]
    )
    _expect_equal(
        errors, "dedup_hash_values", sorted(set(split["dedup_report_hash"])), [dedup_hash]
    )

    group_keys = {
        (str(row.dataset_id), str(row.raw_group_id))
        for row in manifest[["dataset_id", "raw_group_id"]].itertuples(index=False)
    }
    disjoint = _DisjointSet(group_keys)
    record_group = {
        (str(row.dataset_id), str(row.record_id)): (
            str(row.dataset_id),
            str(row.raw_group_id),
        )
        for row in manifest.itertuples(index=False)
    }
    split_lookup = {
        (str(row.dataset_id), str(row.record_id)): str(row.pool)
        for row in split.itertuples(index=False)
    }
    for position, edge in enumerate(dedup.to_dict(orient="records")):
        left = (str(edge["left_dataset_id"]), str(edge["left_record_id"]))
        right = (str(edge["right_dataset_id"]), str(edge["right_record_id"]))
        if left not in record_group or right not in record_group:
            errors.append(f"dedup edge[{position}] references an unknown record")
            continue
        if split_lookup.get(left) != split_lookup.get(right):
            errors.append(f"dedup edge[{position}] crosses pools")
        disjoint.union(record_group[left], record_group[right])
    group_pool_counts = split.groupby(["dataset_id", "raw_group_id"], dropna=False)[
        "pool"
    ].nunique()
    if not group_pool_counts.empty and int(group_pool_counts.max()) != 1:
        errors.append("one raw group appears in multiple pools")
    group_pool = {
        (str(row.dataset_id), str(row.raw_group_id)): str(row.pool)
        for row in split[["dataset_id", "raw_group_id", "pool"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    component_members: Dict[Tuple[str, str], list[Tuple[str, str]]] = {}
    for group in group_keys:
        component_members.setdefault(disjoint.find(group), []).append(group)
    for component, members in component_members.items():
        pools = {group_pool[member] for member in members}
        if len(pools) != 1:
            errors.append(f"dedup component {component!r} crosses pools")
    component_pool = {
        component: group_pool[members[0]]
        for component, members in component_members.items()
    }
    cross_dataset_component_count = sum(
        len({dataset_id for dataset_id, _ in members}) > 1
        for members in component_members.values()
    )
    global_pool_component_counts = {
        pool: sum(value == pool for value in component_pool.values())
        for pool in sorted(POOLS)
    }
    components_by_dataset: Dict[str, set[Tuple[str, str]]] = {}
    for group in group_keys:
        components_by_dataset.setdefault(group[0], set()).add(disjoint.find(group))
    weights = _combined_weights(protocol)
    methane_dataset_id = str(decision["roles"]["methane_dataset"]["dataset_id"])
    methane_entry = decision["roles"]["methane_dataset"]
    group_duration_seconds = int(
        methane_entry.get("adapter_contract", {})
        .get("methane", {})
        .get("group_duration_seconds", 0)
    )
    purge_gap_seconds = int(protocol["data"]["methane"]["purge_gap_seconds"])
    temporal_diagnostics = _review_temporal_cohorts(
        errors=errors,
        joined=joined,
        dataset_id=methane_dataset_id,
        disjoint=disjoint,
        component_pool=component_pool,
        weights=weights,
        seed=seed,
        group_duration_seconds=group_duration_seconds,
        purge_gap_seconds=purge_gap_seconds,
    )
    dataset_pool_component_counts = {}
    expected_dataset_pool_component_counts = {}
    weight_deviation = {}
    for dataset_id, components in sorted(components_by_dataset.items()):
        observed = {
            pool: sum(component_pool[component] == pool for component in components)
            for pool in sorted(POOLS)
        }
        dataset_pool_component_counts[dataset_id] = observed
        if dataset_id == methane_dataset_id:
            expected = temporal_diagnostics.get("component_pool_counts", {})
        else:
            try:
                expected = _balanced_quotas(
                    len(components), weights, seed=seed, dataset_id=dataset_id
                )
            except ValueError as exc:
                errors.append(str(exc))
                continue
        expected_dataset_pool_component_counts[dataset_id] = expected
        _expect_equal(errors, f"dataset_pool_component_counts.{dataset_id}", observed, expected)
        weight_deviation[dataset_id] = {
            pool: observed[pool] / len(components) - weights[pool]
            for pool in sorted(POOLS)
        }

    required_roles = {
        "primary_visual_target": str(
            decision["roles"]["primary_visual_target"]["dataset_id"]
        ),
        "primary_rgbt_dataset": str(
            decision["roles"]["primary_rgbt_dataset"]["dataset_id"]
        ),
        "methane_dataset": methane_dataset_id,
    }
    required_role_pool_coverage = {}
    for role, dataset_id in sorted(required_roles.items()):
        observed = sorted(
            set(split.loc[split["dataset_id"].astype(str) == dataset_id, "pool"].astype(str))
        )
        missing = sorted(POOLS - set(observed))
        required_role_pool_coverage[role] = {
            "dataset_id": dataset_id,
            "observed_pools": observed,
            "missing_pools": missing,
        }
        if missing:
            errors.append(f"required role {role}/{dataset_id} lacks pools {missing}")

    rgbt_id = required_roles["primary_rgbt_dataset"]
    rgbt = joined.loc[joined["dataset_id"].astype(str) == rgbt_id].copy()
    pair_errors = 0
    if rgbt.empty or rgbt["pair_id"].isna().any() or rgbt["pair_id"].astype(str).eq("").any():
        errors.append("paired RGB-T role has empty or missing pair IDs")
    else:
        for _, pair in rgbt.groupby("pair_id", sort=False):
            if len(pair) != 2:
                pair_errors += 1
                continue
            if pair["raw_group_id_manifest"].astype(str).nunique() != 1:
                pair_errors += 1
            if pair["pool"].astype(str).nunique() != 1:
                pair_errors += 1
            if set(pair["modality"].astype(str).str.lower()) != {"visible", "infrared"}:
                pair_errors += 1
        if pair_errors:
            errors.append(f"paired RGB-T integrity failures: {pair_errors}")

    pool_record_counts = {
        str(pool): int(count) for pool, count in split.groupby("pool", sort=True).size().items()
    }
    pool_raw_group_counts = {
        str(pool): int(frame[["dataset_id", "raw_group_id"]].drop_duplicates().shape[0])
        for pool, frame in split.groupby("pool", sort=True)
    }
    expected_details = {
        "record_count": len(split),
        "raw_group_count": len(group_keys),
        "dedup_component_count": len(component_members),
        "pool_record_counts": pool_record_counts,
        "pool_raw_group_counts": pool_raw_group_counts,
        "required_role_pool_coverage": required_role_pool_coverage,
        "split_version": SPLIT_VERSION,
        "split_seed": seed,
    }
    for name, expected in expected_details.items():
        _expect_equal(errors, f"receipt.details.{name}", receipt.get("details", {}).get(name), expected)
    receipt_balance = receipt.get("details", {}).get("balance_diagnostics", {})
    _expect_equal(
        errors,
        "receipt.balance.dataset_component_counts",
        receipt_balance.get("dataset_component_counts"),
        {dataset_id: len(values) for dataset_id, values in components_by_dataset.items()},
    )
    _expect_equal(
        errors,
        "receipt.balance.dataset_pool_component_counts",
        receipt_balance.get("dataset_pool_component_counts"),
        dataset_pool_component_counts,
    )
    _expect_equal(
        errors,
        "receipt.balance.global_component_count",
        receipt_balance.get("global_component_count"),
        len(component_members),
    )
    _expect_equal(
        errors,
        "receipt.balance.algorithm",
        receipt_balance.get("algorithm"),
        "generic_balanced_v1_plus_methane_temporal_cohort_v1",
    )
    _expect_equal(
        errors,
        "receipt.balance.cross_dataset_component_count",
        receipt_balance.get("cross_dataset_component_count"),
        cross_dataset_component_count,
    )
    _expect_equal(
        errors,
        "receipt.balance.global_pool_component_counts",
        receipt_balance.get("global_pool_component_counts"),
        global_pool_component_counts,
    )
    _expect_equal(
        errors,
        "receipt.balance.temporal_cohort_diagnostics",
        receipt_balance.get("temporal_cohort_diagnostics"),
        temporal_diagnostics,
    )
    search_nodes = receipt_balance.get("cross_component_search_nodes")
    if (
        isinstance(search_nodes, bool)
        or not isinstance(search_nodes, int)
        or not 1 <= search_nodes <= 100_000
    ):
        errors.append(
            "receipt.balance.cross_component_search_nodes: expected an integer in [1, 100000]"
        )
    _expect_equal(errors, "receipt.step_id", receipt.get("step_id"), "E038")
    _expect_equal(errors, "receipt.status", receipt.get("status"), "pass")
    _expect_equal(errors, "receipt.command", receipt.get("command"), "create-splits")
    _expect_equal(errors, "receipt.arguments", receipt.get("arguments"), {"grouped": True})
    expected_inputs = {
        "data/locked/file_manifest.parquet": _sha256_file(manifest_path),
        "data/locked/ontology_lock.yaml": ontology_hash,
        "data/locked/dedup_report.parquet": dedup_hash,
        "evidence/data/dataset_source_decision.json": _sha256_file(decision_path),
    }
    observed_inputs = {
        str(item.get("path")): str(item.get("sha256")) for item in receipt.get("inputs", [])
    }
    _expect_equal(errors, "receipt.inputs", observed_inputs, expected_inputs)
    expected_output = {
        "path": "data/locked/split_manifest.parquet",
        "kind": "file",
        "bytes": split_path.stat().st_size,
        "sha256": _sha256_file(split_path),
    }
    _expect_equal(errors, "receipt.outputs", receipt.get("outputs"), [expected_output])
    diagnostics.update(
        {
            "record_count": len(split),
            "raw_group_count": len(group_keys),
            "dedup_component_count": len(component_members),
            "dedup_edge_count": len(dedup),
            "cross_dataset_component_count": cross_dataset_component_count,
            "global_pool_component_counts": global_pool_component_counts,
            "pair_count": int(rgbt["pair_id"].nunique()) if not rgbt.empty else 0,
            "pair_integrity_error_count": pair_errors,
            "dataset_pool_component_counts": dataset_pool_component_counts,
            "expected_dataset_pool_component_counts": expected_dataset_pool_component_counts,
            "component_weight_deviation": weight_deviation,
            "temporal_cohort_diagnostics": temporal_diagnostics,
            "required_role_pool_coverage": required_role_pool_coverage,
        }
    )
    return _result(root, errors, paths, diagnostics)


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"immutable review output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Independently reconcile E038 split artifacts")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = review_e038(args.project_root)
    if args.output is not None:
        output = args.output if args.output.is_absolute() else args.project_root / args.output
        _write_once(output, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
