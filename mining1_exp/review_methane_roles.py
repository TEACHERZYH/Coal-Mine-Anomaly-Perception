from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import pandas as pd
import yaml


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
FAMILY_POOL_ORDER = {
    "branch": ("D_b_tr", "D_b_sel", "D_b_prob", "D_b_te"),
    "episode": ("D_e_tr", "D_e_sel", "D_e_pol", "D_e_te"),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
        "step_id": "E050",
        "reviewer": "independent_local_methane_role_reconciler",
        "status": "pass" if not errors else "fail",
        "error_count": len(errors),
        "errors": list(errors),
        "bindings": {
            path.relative_to(root).as_posix(): _sha256_file(path)
            for path in paths
            if path.is_file()
        },
        "diagnostics": dict(diagnostics),
        "model_outcomes_used": False,
        "performance_claims_authorized": False,
    }


def review_e050(project_root: Path) -> Dict[str, Any]:
    root = Path(project_root).resolve()
    manifest_path = root / "data/locked/file_manifest.parquet"
    split_path = root / "data/locked/split_manifest.parquet"
    decision_path = root / "evidence/data/dataset_source_decision.json"
    protocol_path = root / "configs/protocol_lock.template.yaml"
    lock_path = root / "data/locked/methane_role_lock.json"
    receipt_path = root / "evidence/command_receipts/E050.json"
    paths = (
        manifest_path,
        split_path,
        decision_path,
        protocol_path,
        lock_path,
        receipt_path,
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = pd.read_parquet(manifest_path)
    split = pd.read_parquet(split_path)
    decision = json.loads(decision_path.read_text(encoding="utf-8-sig"))
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8-sig"))
    lock = json.loads(lock_path.read_text(encoding="utf-8-sig"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
    errors: list[str] = []
    methane_entry = decision["roles"]["methane_dataset"]
    dataset_id = str(methane_entry["dataset_id"])
    rows = manifest.loc[manifest["dataset_id"].astype(str) == dataset_id].copy()
    if rows.empty or rows["timestamp_or_order"].isna().any():
        errors.append("selected methane dataset has no complete ordered records")
        return _result(root, errors, paths, {"dataset_id": dataset_id})
    joined = rows.merge(
        split[["dataset_id", "record_id", "pool"]],
        on=["dataset_id", "record_id"],
        how="left",
        validate="one_to_one",
    )
    if joined["pool"].isna().any():
        errors.append("selected methane records are missing from the split")
    try:
        joined["cohort_epoch_seconds"] = (
            pd.to_datetime(joined["timestamp_or_order"], utc=True, errors="raise").astype(
                "int64"
            )
            // 1_000_000_000
        )
    except (TypeError, ValueError):
        errors.append("selected methane records have unparseable UTC timestamps")
        return _result(root, errors, paths, {"dataset_id": dataset_id})
    group_duration = int(
        methane_entry.get("adapter_contract", {})
        .get("methane", {})
        .get("group_duration_seconds", 0)
    )
    methane_contract = protocol["data"]["methane"]
    history = int(methane_contract["history_seconds"])
    horizon = int(methane_contract["horizon_seconds"])
    purge = int(methane_contract["purge_gap_seconds"])
    if group_duration <= 0:
        errors.append("methane group duration is not positive")
    if purge < history + horizon:
        errors.append("methane purge gap does not cover history plus horizon")
    cohort_pool_counts = joined.groupby("cohort_epoch_seconds")["pool"].nunique()
    same_cohort_cross_pool_count = int((cohort_pool_counts != 1).sum())
    if same_cohort_cross_pool_count:
        errors.append(f"methane same-cohort cross-pool count is {same_cohort_cross_pool_count}")
    cohort_pools = (
        joined[["cohort_epoch_seconds", "pool"]]
        .drop_duplicates()
        .sort_values("cohort_epoch_seconds", kind="mergesort")
    )
    cohort_epochs = sorted(set(int(value) for value in joined["cohort_epoch_seconds"]))
    continuity_errors = sum(
        right - left != group_duration for left, right in zip(cohort_epochs, cohort_epochs[1:])
    )
    if continuity_errors:
        errors.append(f"methane source continuity error count is {continuity_errors}")
    observed_order = []
    for pool in cohort_pools["pool"].astype(str):
        if not observed_order or observed_order[-1] != pool:
            observed_order.append(pool)
    if observed_order != list(TEMPORAL_POOL_ORDER):
        errors.append(
            f"methane chronological pool order is {observed_order}, expected {list(TEMPORAL_POOL_ORDER)}"
        )
    pool_rows = []
    pool_continuity_errors = 0
    for pool in TEMPORAL_POOL_ORDER:
        pool_frame = joined.loc[joined["pool"].astype(str) == pool].copy()
        if pool_frame.empty:
            errors.append(f"methane chronological pool is empty: {pool}")
            continue
        values = sorted(set(int(value) for value in pool_frame["cohort_epoch_seconds"]))
        local_errors = sum(
            right - left != group_duration for left, right in zip(values, values[1:])
        )
        pool_continuity_errors += local_errors
        if local_errors:
            errors.append(f"methane pool {pool} is not one contiguous time block")
        pool_rows.append(
            {
                "pool": pool,
                "record_count": len(pool_frame),
                "raw_group_count": int(pool_frame["raw_group_id"].nunique()),
                "cohort_count": len(values),
                "first_order": pd.Timestamp(values[0], unit="s", tz="UTC").isoformat(),
                "last_order": pd.Timestamp(values[-1], unit="s", tz="UTC").isoformat(),
                "first_epoch_seconds": values[0],
                "last_epoch_seconds": values[-1],
                "block_end_exclusive_epoch_seconds": values[-1] + group_duration,
                "record_hash": _canonical_hash(
                    sorted(str(value) for value in pool_frame["record_id"])
                ),
            }
        )
    block_by_pool = {str(item["pool"]): item for item in pool_rows}
    family_gaps = []
    for family, pools in FAMILY_POOL_ORDER.items():
        for earlier, later in zip(pools, pools[1:]):
            if earlier not in block_by_pool or later not in block_by_pool:
                continue
            gap = int(
                block_by_pool[later]["first_epoch_seconds"]
                - block_by_pool[earlier]["block_end_exclusive_epoch_seconds"]
            )
            passed = gap >= purge
            family_gaps.append(
                {
                    "family": family,
                    "from_pool": earlier,
                    "to_pool": later,
                    "gap_seconds": gap,
                    "required_gap_seconds": purge,
                    "passed": passed,
                }
            )
            if not passed:
                errors.append(f"methane {family} purge gap {earlier}->{later} is too short")
    minimum_gap = min((int(item["gap_seconds"]) for item in family_gaps), default=-1)
    expected_lock = {
        "schema_version": 2,
        "step_id": "E050",
        "status": "pass",
        "dataset_id": dataset_id,
        "history_seconds": history,
        "horizon_seconds": horizon,
        "stride_seconds": int(methane_contract["stride_seconds"]),
        "purge_gap_seconds": purge,
        "group_duration_seconds": group_duration,
        "imputation_fit_pool": str(methane_contract["imputation_fit_pool"]),
        "normalization_fit_pool": str(methane_contract["normalization_fit_pool"]),
        "chronological_pool_order": list(TEMPORAL_POOL_ORDER),
        "chronological_pool_evidence": pool_rows,
        "family_purge_gap_evidence": family_gaps,
        "minimum_family_purge_gap_seconds": minimum_gap,
        "same_cohort_cross_pool_count": same_cohort_cross_pool_count,
        "continuous_pool_block_error_count": pool_continuity_errors,
        "future_values_allowed_in_features": False,
        "file_manifest_sha256": _sha256_file(manifest_path),
        "split_manifest_sha256": _sha256_file(split_path),
    }
    _expect_equal(errors, "methane_role_lock", lock, expected_lock)
    _expect_equal(errors, "receipt.step_id", receipt.get("step_id"), "E050")
    _expect_equal(errors, "receipt.status", receipt.get("status"), "pass")
    _expect_equal(errors, "receipt.command", receipt.get("command"), "lock-methane-roles")
    _expect_equal(errors, "receipt.arguments", receipt.get("arguments"), {})
    expected_inputs = {
        "data/locked/file_manifest.parquet": _sha256_file(manifest_path),
        "data/locked/split_manifest.parquet": _sha256_file(split_path),
    }
    observed_inputs = {
        str(item.get("path")): str(item.get("sha256")) for item in receipt.get("inputs", [])
    }
    _expect_equal(errors, "receipt.inputs", observed_inputs, expected_inputs)
    expected_output = {
        "path": "data/locked/methane_role_lock.json",
        "kind": "file",
        "bytes": lock_path.stat().st_size,
        "sha256": _sha256_file(lock_path),
    }
    _expect_equal(errors, "receipt.outputs", receipt.get("outputs"), [expected_output])
    _expect_equal(
        errors,
        "receipt.details",
        receipt.get("details"),
        {
            "dataset_id": dataset_id,
            "pool_count": len(pool_rows),
            "minimum_family_purge_gap_seconds": minimum_gap,
            "cohort_count": len(cohort_epochs),
            "record_count": len(joined),
        },
    )
    diagnostics = {
        "dataset_id": dataset_id,
        "record_count": len(joined),
        "cohort_count": len(cohort_epochs),
        "observed_pool_order": observed_order,
        "same_cohort_cross_pool_count": same_cohort_cross_pool_count,
        "source_continuity_error_count": continuity_errors,
        "pool_continuity_error_count": pool_continuity_errors,
        "minimum_family_purge_gap_seconds": minimum_gap,
        "family_purge_gap_evidence": family_gaps,
        "canonical_data_files_opened": 0,
    }
    return _result(root, errors, paths, diagnostics)


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"immutable review output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Independently reconcile E050 methane roles")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = review_e050(args.project_root)
    if args.output is not None:
        output = args.output if args.output.is_absolute() else args.project_root / args.output
        _write_once(output, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
