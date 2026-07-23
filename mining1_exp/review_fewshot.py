from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import pandas as pd
import yaml


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
PRIMARY_KEY = [
    "dataset_id",
    "direction_id",
    "ratio_percent",
    "subset_seed",
    "raw_group_id",
]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _label_classes(value: str) -> set[str]:
    summary = json.loads(str(value))
    if not isinstance(summary, dict):
        raise ValueError("label summary must be a JSON object")
    if isinstance(summary.get("class_ids"), list):
        return {str(item) for item in summary["class_ids"]}
    if isinstance(summary.get("class_counts"), dict):
        return {str(item) for item in summary["class_counts"]}
    return {str(item) for item in summary if not str(item).startswith("_")}


def _result(
    root: Path,
    errors: Sequence[str],
    paths: Sequence[Path],
    diagnostics: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "step_id": "E042",
        "reviewer": "independent_local_fewshot_reconciler",
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


def review_e042(project_root: Path) -> Dict[str, Any]:
    root = Path(project_root).resolve()
    manifest_path = root / "data/locked/file_manifest.parquet"
    split_path = root / "data/locked/split_manifest.parquet"
    fewshot_path = root / "data/locked/fewshot_manifest.parquet"
    protocol_path = root / "configs/protocol_lock.template.yaml"
    decision_path = root / "evidence/data/dataset_source_decision.json"
    receipt_path = root / "evidence/command_receipts/E042.json"
    paths = (
        manifest_path,
        split_path,
        fewshot_path,
        protocol_path,
        decision_path,
        receipt_path,
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = pd.read_parquet(manifest_path)
    split = pd.read_parquet(split_path)
    fewshot = pd.read_parquet(fewshot_path)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8-sig"))
    decision = json.loads(decision_path.read_text(encoding="utf-8-sig"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
    errors: list[str] = []
    diagnostics: Dict[str, Any] = {}

    required_manifest = {"dataset_id", "raw_group_id", "label_summary_json"}
    required_split = {"dataset_id", "raw_group_id", "pool"}
    missing_manifest = sorted(required_manifest - set(manifest.columns))
    missing_split = sorted(required_split - set(split.columns))
    missing_fewshot = sorted(FEWSHOT_COLUMNS - set(fewshot.columns))
    if missing_manifest:
        errors.append(f"file manifest missing columns: {missing_manifest}")
    if missing_split:
        errors.append(f"split manifest missing columns: {missing_split}")
    if missing_fewshot:
        errors.append(f"fewshot manifest missing columns: {missing_fewshot}")
    if errors:
        return _result(root, errors, paths, diagnostics)

    if fewshot.empty:
        errors.append("fewshot manifest is empty")
    if fewshot.duplicated(PRIMARY_KEY).any():
        errors.append("fewshot primary key is not unique")
    if not pd.api.types.is_bool_dtype(fewshot["included"]):
        errors.append("fewshot included column is not boolean")

    direction = str(decision["visual_direction_id"])
    seeds = [int(value) for value in protocol["seeds"]["fewshot_subset"]]
    split_hash = _sha256_file(split_path)
    if len(seeds) != 3 or len(set(seeds)) != 3:
        errors.append("protocol does not contain exactly three distinct fewshot seeds")
    if set(fewshot["direction_id"].astype(str)) != {direction}:
        errors.append("fewshot direction does not match the frozen source decision")
    if set(fewshot["ratio_percent"].astype(int)) != {10}:
        errors.append("fewshot ratio is not the frozen 10 percent")
    if set(fewshot["subset_seed"].astype(int)) != set(seeds):
        errors.append("fewshot seeds do not match the frozen protocol")
    if set(fewshot["parent_manifest_hash"].astype(str)) != {split_hash}:
        errors.append("fewshot parent hash does not match the formal split")

    training_groups = {
        (str(row.dataset_id), str(row.raw_group_id))
        for row in split.loc[
            split["pool"].astype(str) == "D_b_tr", ["dataset_id", "raw_group_id"]
        ]
        .drop_duplicates()
        .itertuples(index=False)
    }
    observed_groups = {
        (str(row.dataset_id), str(row.raw_group_id))
        for row in fewshot[["dataset_id", "raw_group_id"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    if observed_groups != training_groups:
        errors.append("fewshot candidate groups are not exactly the D_b_tr groups")

    group_classes: Dict[Tuple[str, str], set[str]] = {}
    try:
        for row in manifest.itertuples(index=False):
            key = (str(row.dataset_id), str(row.raw_group_id))
            group_classes.setdefault(key, set()).update(
                _label_classes(str(row.label_summary_json))
            )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"invalid label summary: {exc}")

    per_dataset_seed: Dict[str, Any] = {}
    datasets = sorted({dataset_id for dataset_id, _ in training_groups})
    for dataset_id in datasets:
        groups = sorted(group for dataset, group in training_groups if dataset == dataset_id)
        expected_count = max(1, int(math.ceil(len(groups) * 0.10)))
        for seed in seeds:
            key = f"{dataset_id}|{seed}"
            rows = fewshot.loc[
                (fewshot["dataset_id"].astype(str) == dataset_id)
                & (fewshot["subset_seed"].astype(int) == seed)
            ]
            row_groups = set(rows["raw_group_id"].astype(str))
            if row_groups != set(groups):
                errors.append(f"{key}: candidate group set mismatch")
            ranked = sorted(
                groups,
                key=lambda group: hashlib.sha256(
                    f"{seed}|{dataset_id}|{direction}|{group}".encode("utf-8")
                ).hexdigest(),
            )
            expected_included = set(ranked[:expected_count])
            observed_included = set(
                rows.loc[rows["included"].astype(bool), "raw_group_id"].astype(str)
            )
            if observed_included != expected_included:
                errors.append(f"{key}: included group set mismatch")
            class_counts: Dict[str, int] = {}
            for group in expected_included:
                for class_id in group_classes.get((dataset_id, group), set()):
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
            expected_counts_json = json.dumps(class_counts, ensure_ascii=True, sort_keys=True)
            observed_counts = set(rows["class_group_counts_json"].astype(str))
            if observed_counts != {expected_counts_json}:
                errors.append(f"{key}: class group counts mismatch")
            per_dataset_seed[key] = {
                "candidate_group_count": len(groups),
                "included_group_count": len(observed_included),
                "expected_included_group_count": expected_count,
                "class_group_counts": class_counts,
            }

    expected_inputs = {
        "data/locked/split_manifest.parquet": split_hash,
        "data/locked/file_manifest.parquet": _sha256_file(manifest_path),
        "evidence/data/dataset_source_decision.json": _sha256_file(decision_path),
    }
    observed_inputs = {
        str(item.get("path")): str(item.get("sha256")) for item in receipt.get("inputs", [])
    }
    if observed_inputs != expected_inputs:
        errors.append("command receipt inputs do not match E042 inputs")
    expected_output = {
        "path": "data/locked/fewshot_manifest.parquet",
        "kind": "file",
        "bytes": fewshot_path.stat().st_size,
        "sha256": _sha256_file(fewshot_path),
    }
    if receipt.get("outputs") != [expected_output]:
        errors.append("command receipt output does not match the fewshot artifact")
    if receipt.get("step_id") != "E042" or receipt.get("status") != "pass":
        errors.append("command receipt step or status is invalid")
    if receipt.get("command") != "create-fewshot":
        errors.append("command receipt command is invalid")
    if receipt.get("arguments") != {"pairs": 3, "ratio": 10}:
        errors.append("command receipt arguments are invalid")
    expected_details = {
        "subset_seed_count": int(fewshot["subset_seed"].nunique()),
        "included_group_count": int(
            fewshot.loc[fewshot["included"].astype(bool), "raw_group_id"].nunique()
        ),
    }
    if receipt.get("details") != expected_details:
        errors.append("command receipt details do not reconcile")

    diagnostics.update(
        {
            "direction_id": direction,
            "ratio_percent": 10,
            "subset_seeds": seeds,
            "dataset_count": len(datasets),
            "candidate_dataset_group_count": len(training_groups),
            "included_dataset_group_count_across_seeds": int(
                fewshot.loc[fewshot["included"].astype(bool), ["dataset_id", "raw_group_id"]]
                .drop_duplicates()
                .shape[0]
            ),
            "per_dataset_seed": per_dataset_seed,
        }
    )
    return _result(root, errors, paths, diagnostics)


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"immutable review output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Independently reconcile E042 fewshot artifacts")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = review_e042(args.project_root)
    if args.output is not None:
        output = args.output if args.output.is_absolute() else args.project_root / args.output
        _write_once(output, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
