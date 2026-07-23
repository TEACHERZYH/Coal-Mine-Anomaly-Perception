from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import pandas as pd


EXPECTED_ARTIFACTS = {
    "branch_predictions.parquet",
    "duplicate_report.parquet",
    "episode_predictions.parquet",
    "execution.log",
    "fewshot_manifest.parquet",
    "file_manifest.parquet",
    "methane_predictions.parquet",
    "metrics.json",
    "split_manifest.parquet",
    "summary.json",
}
EXPECTED_MODELS = (
    "visible_yolov8n_adapter_synthetic_backend",
    "thermal_yolov8n_adapter_synthetic_backend",
    "concept_max_aggregation_and_platt_calibration",
    "methane_persistence_rule",
    "methane_hgb",
    "methane_gru_one_step",
    "episode_mean",
    "episode_calibrated_logit_one_step",
    "episode_reliability_graph_one_step",
    "episode_memory_and_event_evaluation",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expect_equal(errors: list[str], name: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        errors.append(f"{name}: observed={observed!r}, expected={expected!r}")


def _check_probability_table(
    errors: list[str],
    frame: pd.DataFrame,
    *,
    name: str,
    required: set[str],
    probability_columns: Sequence[str],
    primary_key: Sequence[str],
) -> None:
    missing = required - set(frame.columns)
    if missing:
        errors.append(f"{name} missing columns: {sorted(missing)}")
        return
    if frame.empty:
        errors.append(f"{name} is empty")
    if frame.duplicated(list(primary_key)).any():
        errors.append(f"{name} primary key is not unique")
    for column in probability_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not values.between(0.0, 1.0).all():
            errors.append(f"{name}.{column} is not a finite probability")


def _result(
    root: Path,
    errors: Sequence[str],
    paths: Sequence[Path],
    diagnostics: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "step_id": "E062",
        "reviewer": "independent_local_smoke_reconciler",
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


def review_e062(project_root: Path) -> Dict[str, Any]:
    root = Path(project_root).resolve()
    manifest_path = root / "evidence/data/minimal_sample_manifest.parquet"
    rgbt_path = root / "data/locked/rgbt_input_lock.json"
    smoke_path = root / "evidence/smoke/local_smoke.json"
    bundle = root / "evidence/smoke/local_module_bundle"
    bundle_receipt_path = bundle / "integration_receipt.json"
    summary_path = bundle / "summary.json"
    command_receipt_path = root / "evidence/command_receipts/E062.json"
    required_paths = (
        manifest_path,
        rgbt_path,
        smoke_path,
        bundle_receipt_path,
        summary_path,
        command_receipt_path,
    )
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = pd.read_parquet(manifest_path)
    rgbt = json.loads(rgbt_path.read_text(encoding="utf-8-sig"))
    smoke = json.loads(smoke_path.read_text(encoding="utf-8-sig"))
    bundle_receipt = json.loads(bundle_receipt_path.read_text(encoding="utf-8-sig"))
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    command_receipt = json.loads(command_receipt_path.read_text(encoding="utf-8-sig"))
    errors: list[str] = []

    required_manifest = {
        "dataset_id",
        "record_id",
        "raw_group_id",
        "modality",
        "sample_path",
        "sha256",
    }
    if manifest.empty or not required_manifest.issubset(manifest.columns):
        errors.append("minimal sample manifest is empty or incomplete")
    sample_hash_errors = 0
    source_modalities: set[str] = set()
    if required_manifest.issubset(manifest.columns):
        for item in manifest.itertuples(index=False):
            sample = (root / str(item.sample_path)).resolve()
            if (
                root not in sample.parents
                or not sample.is_file()
                or sample.is_symlink()
                or _sha256_file(sample) != str(item.sha256)
            ):
                sample_hash_errors += 1
            source_modalities.add(str(item.modality).lower())
    if sample_hash_errors:
        errors.append(f"minimal sample path or hash errors: {sample_hash_errors}")

    mapping_raw = rgbt.get("source_modality_to_branch_role")
    mapping = (
        {
            str(source).lower(): str(branch).lower()
            for source, branch in mapping_raw.items()
        }
        if isinstance(mapping_raw, dict)
        else {}
    )
    branch_modalities = source_modalities - {"methane"}
    unmapped = branch_modalities - set(mapping)
    branch_roles = {mapping[item] for item in branch_modalities if item in mapping}
    if rgbt.get("status") != "pass" or unmapped:
        errors.append(f"RGB-T source mapping is not closed: {sorted(unmapped)}")
    if "methane" not in source_modalities or not {"visible", "thermal"}.issubset(
        branch_roles
    ):
        errors.append("minimal sample does not cover visible, thermal, and methane modules")

    expected_smoke = {
        "schema_version": 1,
        "step_id": "E062",
        "status": "pass",
        "actual_sample_record_count": len(manifest),
        "actual_sample_modalities": sorted(source_modalities),
        "actual_sample_branch_roles": sorted(branch_roles),
        "minimal_sample_manifest_sha256": _sha256_file(manifest_path),
        "rgbt_input_lock_sha256": _sha256_file(rgbt_path),
        "module_pipeline_receipt_sha256": _sha256_file(bundle_receipt_path),
        "full_local_dataset_extractions": 0,
        "remote_connections": 0,
        "slurm_jobs_created": 0,
        "performance_claims_authorized": False,
    }
    for key, expected in expected_smoke.items():
        _expect_equal(errors, f"local_smoke.{key}", smoke.get(key), expected)
    if smoke.get("module_pipeline_mode") not in {"created", "validated_existing"}:
        errors.append("local_smoke.module_pipeline_mode is invalid")

    _expect_equal(errors, "bundle_receipt.schema_version", bundle_receipt.get("schema_version"), 1)
    _expect_equal(errors, "bundle_receipt.step_id", bundle_receipt.get("step_id"), "I080")
    _expect_equal(errors, "bundle_receipt.status", bundle_receipt.get("status"), "pass")
    records = bundle_receipt.get("artifacts")
    recorded_names = (
        [str(item.get("path", "")) for item in records]
        if isinstance(records, list)
        else []
    )
    if len(recorded_names) != len(set(recorded_names)) or set(recorded_names) != EXPECTED_ARTIFACTS:
        errors.append("module bundle receipt does not close the exact artifact set")
    artifact_paths: list[Path] = []
    if isinstance(records, list):
        for record in records:
            relative = Path(str(record.get("path", "")))
            artifact = bundle / relative
            artifact_paths.append(artifact)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or len(relative.parts) != 1
                or not artifact.is_file()
                or artifact.is_symlink()
                or artifact.stat().st_size != record.get("bytes")
                or _sha256_file(artifact) != record.get("sha256")
            ):
                errors.append(f"module bundle artifact drift: {relative.as_posix()}")
    actual_bundle_names = {
        path.name
        for path in bundle.iterdir()
        if path.is_file() and path.name != ".I080.lock"
    }
    if actual_bundle_names != EXPECTED_ARTIFACTS | {"integration_receipt.json"}:
        errors.append("module bundle has missing or untracked files")

    summary_expectations = {
        "schema_version": 1,
        "step_id": "I080",
        "status": "pass",
        "synthetic_only": True,
        "fit_and_inference_samples_disjoint": True,
        "full_local_dataset_extractions": 0,
        "remote_connections": 0,
        "slurm_jobs_created": 0,
        "models_exercised": list(EXPECTED_MODELS),
    }
    for key, expected in summary_expectations.items():
        _expect_equal(errors, f"summary.{key}", summary.get(key), expected)
    _expect_equal(
        errors,
        "bundle_receipt.summary_sha256",
        bundle_receipt.get("summary_sha256"),
        _sha256_file(summary_path),
    )
    for boundary_name in ("methane_boundary", "episode_boundary"):
        boundary = summary.get(boundary_name)
        if (
            not isinstance(boundary, dict)
            or boundary.get("fit_and_inference_disjoint") is not True
            or int(boundary.get("fit_rows", 0)) <= 0
            or int(boundary.get("inference_rows", 0)) <= 0
            or boundary.get("fit_input_sha256") == boundary.get("inference_input_sha256")
        ):
            errors.append(f"{boundary_name} does not prove disjoint nonempty inputs")
    pool_counts = summary.get("pool_group_counts")
    if (
        not isinstance(pool_counts, dict)
        or set(pool_counts) != {"D_b_tr", "D_b_prob", "D_e_tr", "D_e_te"}
        or any(int(value) <= 0 or int(value) > 2 for value in pool_counts.values())
    ):
        errors.append("module smoke pool coverage exceeds or misses the frozen bound")

    branch = pd.read_parquet(bundle / "branch_predictions.parquet")
    _check_probability_table(
        errors,
        branch,
        name="branch_predictions",
        required={
            "record_id",
            "concept_id",
            "modality",
            "step_score_raw",
            "step_probability",
            "model_probability_mean",
        },
        probability_columns=("step_score_raw", "step_probability", "model_probability_mean"),
        primary_key=("record_id", "concept_id", "modality"),
    )
    methane = pd.read_parquet(bundle / "methane_predictions.parquet")
    _check_probability_table(
        errors,
        methane,
        name="methane_predictions",
        required={"window_id", "rule_probability", "hgb_probability", "gru_probability"},
        probability_columns=("rule_probability", "hgb_probability", "gru_probability"),
        primary_key=("window_id",),
    )
    episode = pd.read_parquet(bundle / "episode_predictions.parquet")
    _check_probability_table(
        errors,
        episode,
        name="episode_predictions",
        required={
            "skeleton_item_id",
            "mean_probability",
            "logit_probability",
            "graph_probability",
            "memory_probability",
        },
        probability_columns=(
            "mean_probability",
            "logit_probability",
            "graph_probability",
            "memory_probability",
        ),
        primary_key=("skeleton_item_id",),
    )

    _expect_equal(errors, "command_receipt.step_id", command_receipt.get("step_id"), "E062")
    _expect_equal(errors, "command_receipt.status", command_receipt.get("status"), "pass")
    _expect_equal(errors, "command_receipt.command", command_receipt.get("command"), "smoke-local")
    _expect_equal(errors, "command_receipt.arguments", command_receipt.get("arguments"), {"minimal": True})
    expected_inputs = {
        "evidence/data/minimal_sample_manifest.parquet": _sha256_file(manifest_path),
        "data/locked/rgbt_input_lock.json": _sha256_file(rgbt_path),
    }
    observed_inputs = {
        str(item.get("path")): str(item.get("sha256"))
        for item in command_receipt.get("inputs", [])
    }
    _expect_equal(errors, "command_receipt.inputs", observed_inputs, expected_inputs)
    expected_output = {
        "path": "evidence/smoke/local_smoke.json",
        "kind": "file",
        "bytes": smoke_path.stat().st_size,
        "sha256": _sha256_file(smoke_path),
    }
    _expect_equal(errors, "command_receipt.outputs", command_receipt.get("outputs"), [expected_output])
    _expect_equal(
        errors,
        "command_receipt.details",
        command_receipt.get("details"),
        {"actual_sample_record_count": len(manifest), "full_extractions": 0},
    )

    diagnostics = {
        "minimal_sample_record_count": len(manifest),
        "minimal_sample_group_count": int(manifest["raw_group_id"].nunique()),
        "source_modalities": sorted(source_modalities),
        "locked_branch_roles": sorted(branch_roles),
        "sample_hash_error_count": sample_hash_errors,
        "module_artifact_count": len(recorded_names),
        "models_exercised_count": len(summary.get("models_exercised", [])),
        "full_local_dataset_extractions": smoke.get("full_local_dataset_extractions"),
        "remote_connections": smoke.get("remote_connections"),
        "slurm_jobs_created": smoke.get("slurm_jobs_created"),
    }
    return _result(root, errors, (*required_paths, *artifact_paths), diagnostics)


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"immutable review output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Independently reconcile E062 local smoke")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = review_e062(args.project_root)
    if args.output is not None:
        output = args.output if args.output.is_absolute() else args.project_root / args.output
        _write_once(output, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
