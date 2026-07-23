from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import pandas as pd
import yaml


THERMAL_SOURCE_MODALITIES = {"thermal", "infrared"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        "step_id": "E046",
        "reviewer": "independent_local_rgbt_input_reconciler",
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


def review_e046(project_root: Path) -> Dict[str, Any]:
    root = Path(project_root).resolve()
    manifest_path = root / "data/locked/file_manifest.parquet"
    split_path = root / "data/locked/split_manifest.parquet"
    decision_path = root / "evidence/data/dataset_source_decision.json"
    protocol_path = root / "configs/protocol_lock.template.yaml"
    lock_path = root / "data/locked/rgbt_input_lock.json"
    receipt_path = root / "evidence/command_receipts/E046.json"
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
    dataset_id = str(decision["roles"]["primary_rgbt_dataset"]["dataset_id"])
    rows = manifest.loc[manifest["dataset_id"].astype(str) == dataset_id].copy()
    if rows.empty:
        errors.append("selected RGB-T dataset has no records")
        return _result(root, errors, paths, {"dataset_id": dataset_id})
    if rows["pair_id"].isna().any() or rows["pair_id"].astype(str).str.strip().eq("").any():
        errors.append("selected RGB-T dataset has empty pair IDs")
    split_rows = split.loc[split["dataset_id"].astype(str) == dataset_id].copy()
    joined = rows.merge(
        split_rows[["dataset_id", "record_id", "pool"]],
        on=["dataset_id", "record_id"],
        how="left",
        validate="one_to_one",
    )
    if joined["pool"].isna().any():
        errors.append("selected RGB-T records are missing from the split")
    valid_pairs = []
    observed_modalities: set[str] = set()
    signature_counts: Dict[str, int] = {}
    pair_errors = 0
    for pair_id, pair in joined.groupby("pair_id", sort=True):
        modalities = set(pair["modality"].astype(str).str.lower())
        thermal_members = modalities & THERMAL_SOURCE_MODALITIES
        if len(pair) != 2 or "visible" not in modalities or len(thermal_members) != 1:
            continue
        if pair["raw_group_id"].astype(str).nunique() != 1 or pair["pool"].astype(str).nunique() != 1:
            pair_errors += 1
            continue
        valid_pairs.append(str(pair_id))
        observed_modalities.update(modalities)
        signature = ",".join(sorted(modalities))
        signature_counts[signature] = signature_counts.get(signature, 0) + 1
    if not valid_pairs:
        errors.append("no valid visible plus infrared-or-thermal pair remains")
    if pair_errors:
        errors.append(f"valid pair provenance or pool errors: {pair_errors}")
    valid_rows = joined.loc[joined["pair_id"].astype(str).isin(valid_pairs)].copy()
    required_branch_roles = ["thermal", "visible"]
    source_mapping = {
        modality: ("visible" if modality == "visible" else "thermal")
        for modality in sorted(observed_modalities)
    }
    physical_temperature = bool(
        protocol["data"]["rgbt"]["physical_temperature_claim_enabled"]
    )
    expected_lock = {
        "schema_version": 2,
        "step_id": "E046",
        "status": "pass",
        "dataset_id": dataset_id,
        "required_modalities": required_branch_roles,
        "required_branch_roles": required_branch_roles,
        "observed_source_modalities": sorted(observed_modalities),
        "source_modality_to_branch_role": source_mapping,
        "pair_modality_signature_counts": signature_counts,
        "physical_temperature_claim_enabled": physical_temperature,
        "valid_pair_count": len(valid_pairs),
        "raw_group_count": int(valid_rows["raw_group_id"].nunique()),
        "pool_counts": {
            str(key): int(value) for key, value in split_rows["pool"].value_counts().items()
        },
        "file_manifest_sha256": _sha256_file(manifest_path),
        "split_manifest_sha256": _sha256_file(split_path),
        "model_outcomes_used": False,
    }
    _expect_equal(errors, "rgbt_input_lock", lock, expected_lock)
    _expect_equal(errors, "receipt.step_id", receipt.get("step_id"), "E046")
    _expect_equal(errors, "receipt.status", receipt.get("status"), "pass")
    _expect_equal(errors, "receipt.command", receipt.get("command"), "audit-rgbt-inputs")
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
        "path": "data/locked/rgbt_input_lock.json",
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
            "valid_pair_count": len(valid_pairs),
            "observed_source_modalities": sorted(observed_modalities),
            "physical_temperature_claim_enabled": physical_temperature,
        },
    )
    diagnostics = {
        "dataset_id": dataset_id,
        "record_count": len(rows),
        "valid_pair_count": len(valid_pairs),
        "pair_error_count": pair_errors,
        "independent_raw_group_count": int(valid_rows["raw_group_id"].nunique()),
        "observed_source_modalities": sorted(observed_modalities),
        "required_branch_roles": required_branch_roles,
        "pair_modality_signature_counts": signature_counts,
        "physical_temperature_claim_enabled": physical_temperature,
    }
    return _result(root, errors, paths, diagnostics)


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"immutable review output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Independently reconcile E046 RGB-T inputs")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = review_e046(args.project_root)
    if args.output is not None:
        output = args.output if args.output.is_absolute() else args.project_root / args.output
        _write_once(output, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
