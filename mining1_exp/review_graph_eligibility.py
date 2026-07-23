from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import pandas as pd
import yaml


THERMAL_SOURCE_MODALITIES = {"infrared", "thermal"}


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
        "step_id": "E054",
        "reviewer": "independent_local_graph_eligibility_reconciler",
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


def review_e054(project_root: Path) -> Dict[str, Any]:
    root = Path(project_root).resolve()
    manifest_path = root / "data/locked/file_manifest.parquet"
    split_path = root / "data/locked/split_manifest.parquet"
    decision_path = root / "evidence/data/dataset_source_decision.json"
    protocol_path = root / "configs/protocol_lock.template.yaml"
    ontology_path = root / "data/locked/ontology_lock.yaml"
    rgbt_path = root / "data/locked/rgbt_input_lock.json"
    graph_path = root / "data/locked/graph_eligibility_lock.json"
    receipt_path = root / "evidence/command_receipts/E054.json"
    paths = (
        manifest_path,
        split_path,
        decision_path,
        protocol_path,
        ontology_path,
        rgbt_path,
        graph_path,
        receipt_path,
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = pd.read_parquet(manifest_path)
    split = pd.read_parquet(split_path)
    decision = json.loads(decision_path.read_text(encoding="utf-8-sig"))
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8-sig"))
    ontology = yaml.safe_load(ontology_path.read_text(encoding="utf-8-sig"))
    rgbt = json.loads(rgbt_path.read_text(encoding="utf-8-sig"))
    graph = json.loads(graph_path.read_text(encoding="utf-8-sig"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
    errors: list[str] = []

    dataset_id = str(decision["roles"]["primary_rgbt_dataset"]["dataset_id"])
    rows = manifest.loc[manifest["dataset_id"].astype(str) == dataset_id].copy()
    split_rows = split.loc[split["dataset_id"].astype(str) == dataset_id].copy()
    if rows.empty:
        errors.append("selected RGB-T dataset has no records")
        return _result(root, errors, paths, {"dataset_id": dataset_id})
    if rows["pair_id"].isna().any() or rows["pair_id"].astype(str).str.strip().eq("").any():
        errors.append("selected RGB-T dataset has empty pair IDs")
    joined = rows.merge(
        split_rows[["dataset_id", "record_id", "pool"]],
        on=["dataset_id", "record_id"],
        how="left",
        validate="one_to_one",
    )
    if joined["pool"].isna().any():
        errors.append("selected RGB-T records are missing from the split")

    valid_pair_ids: list[str] = []
    pair_contract_error_count = 0
    observed_modalities: set[str] = set()
    for pair_id, pair in joined.groupby("pair_id", sort=True):
        modalities = set(pair["modality"].astype(str).str.lower())
        thermal_members = modalities & THERMAL_SOURCE_MODALITIES
        shape_valid = len(pair) == 2 and "visible" in modalities and len(thermal_members) == 1
        provenance_valid = (
            pair["raw_group_id"].astype(str).nunique() == 1
            and pair["pool"].astype(str).nunique() == 1
        )
        if not shape_valid or not provenance_valid:
            pair_contract_error_count += 1
            continue
        valid_pair_ids.append(str(pair_id))
        observed_modalities.update(modalities)
    valid_rows = joined.loc[joined["pair_id"].astype(str).isin(valid_pair_ids)].copy()
    if len(valid_rows) != len(rows):
        pair_contract_error_count += max(0, (len(rows) - len(valid_rows)) // 2)
    if pair_contract_error_count:
        errors.append(f"independent pair contract errors: {pair_contract_error_count}")

    source_mapping = {
        modality: ("visible" if modality == "visible" else "thermal")
        for modality in sorted(observed_modalities)
    }
    branch_roles = sorted(set(source_mapping.values()))
    raw_group_count = int(valid_rows["raw_group_id"].astype(str).nunique())
    floor = int(protocol["data"]["independent_group_floor_hard_min"])
    physical_temperature = bool(
        protocol["data"]["rgbt"]["physical_temperature_claim_enabled"]
    )
    compatible_concepts = sorted(
        {
            str(entry["canonical_concept_id"])
            for entry in ontology["entries"]
            if str(entry.get("dataset_id")) == dataset_id
            and entry.get("mapping_status") == "compatible"
            and str(entry.get("canonical_concept_id") or "").strip()
        }
    )
    pair_contract_pass = (
        pair_contract_error_count == 0
        and len(valid_rows) == len(rows)
        and branch_roles == ["thermal", "visible"]
    )
    eligible = pair_contract_pass and raw_group_count >= floor and bool(compatible_concepts)
    reasons: list[str] = []
    if not pair_contract_pass:
        reasons.append("observed_pairing_or_modality_contract_failed")
    if raw_group_count < floor:
        reasons.append("independent_group_floor_not_met")
    if not compatible_concepts:
        reasons.append("no_compatible_concept_mapping")

    expected_rgbt_fields = {
        "status": "pass",
        "dataset_id": dataset_id,
        "observed_source_modalities": sorted(observed_modalities),
        "required_branch_roles": ["thermal", "visible"],
        "source_modality_to_branch_role": source_mapping,
        "physical_temperature_claim_enabled": physical_temperature,
        "valid_pair_count": len(valid_pair_ids),
        "raw_group_count": raw_group_count,
        "file_manifest_sha256": _sha256_file(manifest_path),
        "split_manifest_sha256": _sha256_file(split_path),
    }
    for key, expected in expected_rgbt_fields.items():
        _expect_equal(errors, f"rgbt_input_lock.{key}", rgbt.get(key), expected)

    expected_graph = {
        "schema_version": 1,
        "step_id": "E054",
        "status": "pass",
        "graph_eligible": eligible,
        "eligibility_reasons": reasons or ["observed_pairing_and_group_floor_pass"],
        "dataset_id": dataset_id,
        "observed_modalities": sorted(observed_modalities),
        "branch_roles": ["thermal", "visible"],
        "source_modality_to_branch_role": source_mapping,
        "physical_temperature_claim_enabled": physical_temperature,
        "independent_raw_group_count": raw_group_count,
        "independent_group_floor": floor,
        "compatible_concept_ids": compatible_concepts,
        "virtual_edges_counted_as_observed": False,
        "model_outcomes_used": False,
        "rgbt_input_lock_sha256": _sha256_file(rgbt_path),
        "ontology_lock_sha256": _sha256_file(ontology_path),
    }
    _expect_equal(errors, "graph_eligibility_lock", graph, expected_graph)
    _expect_equal(errors, "receipt.step_id", receipt.get("step_id"), "E054")
    _expect_equal(errors, "receipt.status", receipt.get("status"), "pass")
    _expect_equal(errors, "receipt.command", receipt.get("command"), "audit-graph-eligibility")
    _expect_equal(errors, "receipt.arguments", receipt.get("arguments"), {})
    expected_inputs = {
        "data/locked/rgbt_input_lock.json": _sha256_file(rgbt_path),
        "data/locked/ontology_lock.yaml": _sha256_file(ontology_path),
    }
    observed_inputs = {
        str(item.get("path")): str(item.get("sha256")) for item in receipt.get("inputs", [])
    }
    _expect_equal(errors, "receipt.inputs", observed_inputs, expected_inputs)
    expected_output = {
        "path": "data/locked/graph_eligibility_lock.json",
        "kind": "file",
        "bytes": graph_path.stat().st_size,
        "sha256": _sha256_file(graph_path),
    }
    _expect_equal(errors, "receipt.outputs", receipt.get("outputs"), [expected_output])
    _expect_equal(errors, "receipt.details", receipt.get("details"), {"graph_eligible": eligible})

    diagnostics = {
        "dataset_id": dataset_id,
        "record_count": len(rows),
        "valid_pair_count": len(valid_pair_ids),
        "pair_contract_error_count": pair_contract_error_count,
        "observed_source_modalities": sorted(observed_modalities),
        "branch_roles": branch_roles,
        "independent_raw_group_count": raw_group_count,
        "independent_group_floor": floor,
        "compatible_concept_ids": compatible_concepts,
        "graph_eligible": eligible,
        "physical_temperature_claim_enabled": physical_temperature,
        "virtual_edges_counted_as_observed": False,
    }
    return _result(root, errors, paths, diagnostics)


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"immutable review output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Independently reconcile E054 graph eligibility"
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = review_e054(args.project_root)
    if args.output is not None:
        output = args.output if args.output.is_absolute() else args.project_root / args.output
        _write_once(output, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
