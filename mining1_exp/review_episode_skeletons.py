from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Optional, Sequence

import pandas as pd
import yaml


EPISODE_POOLS = ("D_e_tr", "D_e_sel", "D_e_pol", "D_e_te")
TEMPLATE_FAMILIES = (
    "abrupt",
    "gradual",
    "intermittent",
    "evidence_delay",
    "evidence_conflict",
    "modality_missingness",
)
SKELETON_COLUMNS = (
    "skeleton_item_id",
    "episode_id",
    "step_index",
    "record_id",
    "raw_group_id",
    "concept_id",
    "node_id",
    "pool",
    "episode_seed",
    "generator_family_id",
    "template_instance_id",
    "template_family",
    "event_position_role",
    "source_split_hash",
    "template_lock_hash",
)
PROJECTION_COLUMNS = (
    "skeleton_item_id",
    "record_id",
    "concept_id",
    "node_id",
    "generator_family_id",
)
FEATURE_COLUMNS = (
    "dataset_id",
    "record_id",
    "raw_group_id",
    "pair_id",
    "modality",
    "feature_kind",
    "relative_path",
    "history_json",
    "source_feature_sha256",
)
ELIGIBILITY_REQUIRED_COLUMNS = {
    "concept_id",
    "eligible_branch_type_count",
    "eligible_branch_types",
    "episode_skeleton_candidate_hash",
    "exclusion_reason",
    "fusion_primary_eligible",
    "graph_primary_eligible",
    "independent_multibranch_positive_group_count",
    "multibranch_event_count_by_pool",
    "ontology_lock_hash",
    "split_manifest_hash",
}
FORBIDDEN_PROJECTION_COLUMNS = {
    "episode_id",
    "step_index",
    "raw_group_id",
    "pair_id",
    "pool",
    "episode_seed",
    "dataset_id",
    "template_instance_id",
    "event_position_role",
    "event_truth",
    "state_truth",
    "test_label",
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _expect_equal(errors: list[str], name: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        errors.append(f"{name}: observed={observed!r}, expected={expected!r}")


def _as_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _result(
    root: Path,
    errors: Sequence[str],
    paths: Sequence[Path],
    diagnostics: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "step_id": "E058",
        "reviewer": "independent_local_episode_skeleton_reconciler",
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


def _independent_eligibility(
    skeleton: pd.DataFrame,
    features: pd.DataFrame,
    graph: Mapping[str, Any],
    positive_floor: int,
    candidate_hash: str,
    ontology_hash: str,
    split_hash: str,
    skeleton_hash: str,
    graph_hash: str,
) -> Dict[str, Dict[str, Any]]:
    source = skeleton.merge(
        features[["record_id", "pair_id"]],
        on="record_id",
        validate="many_to_one",
    )
    generator_modalities = {
        "S1-GRU": "methane",
        "T1-THERM": "thermal",
        "T1-VIS": "visible",
        "V2-A-10-MULTI": "visible",
    }
    source["modality"] = source["generator_family_id"].map(generator_modalities)
    expected: Dict[str, Dict[str, Any]] = {}
    for concept_id in sorted(set(source["concept_id"].astype(str))):
        per_pool: Dict[str, Dict[str, Any]] = {}
        for pool in EPISODE_POOLS:
            subset = source.loc[
                source["concept_id"].astype(str).eq(concept_id)
                & source["pool"].astype(str).eq(pool)
            ].copy()
            steps = ["episode_id", "step_index", "concept_id"]
            summary = (
                subset.groupby(steps, sort=True)
                .agg(
                    modality_count=("modality", "nunique"),
                    positive=(
                        "event_position_role",
                        lambda values: bool((values == "positive_observation").all()),
                    ),
                )
                .reset_index()
            )
            multibranch = summary.loc[summary["modality_count"] >= 2]
            positive_multi = multibranch.loc[multibranch["positive"].astype(bool)]
            positive_rows = subset.merge(positive_multi[steps], on=steps, how="inner")
            observed_rows = positive_rows.loc[
                positive_rows["pair_id"].notna()
                & positive_rows["pair_id"].astype(str).str.strip().ne("")
            ]
            if not observed_rows.empty:
                paired_rows = observed_rows.groupby(
                    [*steps, "pair_id"], sort=False
                ).filter(lambda values: values["modality"].nunique() >= 2)
            else:
                paired_rows = observed_rows
            per_pool[pool] = {
                "branch_types": sorted(set(subset["modality"].astype(str))),
                "multibranch_steps": int(len(multibranch)),
                "positive_multibranch_event_count": int(len(positive_multi)),
                "positive_multibranch_groups": int(
                    positive_rows["raw_group_id"].astype(str).nunique()
                ),
                "positive_observed_pair_groups": int(
                    paired_rows["raw_group_id"].astype(str).nunique()
                ),
            }
        common_branches = sorted(
            set.intersection(*(set(item["branch_types"]) for item in per_pool.values()))
        )
        group_count = min(
            int(item["positive_multibranch_groups"]) for item in per_pool.values()
        )
        multibranch_steps = min(
            int(item["multibranch_steps"]) for item in per_pool.values()
        )
        observed_pair_groups = min(
            int(item["positive_observed_pair_groups"]) for item in per_pool.values()
        )
        eligible = (
            len(common_branches) >= 2
            and multibranch_steps > 0
            and group_count >= positive_floor
        )
        reasons: list[str] = []
        if len(common_branches) < 2:
            reasons.append("fewer_than_two_branch_types_in_at_least_one_pool")
        if multibranch_steps == 0:
            reasons.append("no_aligned_multibranch_step_in_at_least_one_pool")
        if group_count < positive_floor:
            reasons.append("positive_multibranch_group_floor_not_met_in_every_pool")
        graph_primary = (
            eligible
            and bool(graph.get("graph_eligible"))
            and concept_id in set(graph.get("compatible_concept_ids", []))
            and observed_pair_groups >= positive_floor
        )
        expected[concept_id] = {
            "concept_id": concept_id,
            "eligible_branch_type_count": len(common_branches),
            "eligible_branch_types": common_branches,
            "episode_skeleton_candidate_hash": candidate_hash,
            "exclusion_reason": None if eligible else "|".join(reasons),
            "fusion_primary_eligible": eligible,
            "graph_primary_eligible": graph_primary,
            "independent_multibranch_positive_group_count": group_count,
            "multibranch_event_count_by_pool": {
                pool: int(item["positive_multibranch_event_count"])
                for pool, item in per_pool.items()
            },
            "ontology_lock_hash": ontology_hash,
            "split_manifest_hash": split_hash,
            "minimum_multibranch_step_count": multibranch_steps,
            "minimum_observed_pair_positive_group_count": observed_pair_groups,
            "positive_group_floor": positive_floor,
            "per_pool_counts": per_pool,
            "model_outcomes_used": False,
            "skeleton_manifest_sha256": skeleton_hash,
            "graph_eligibility_sha256": graph_hash,
        }
    return expected


def review_e058(project_root: Path) -> Dict[str, Any]:
    root = Path(project_root).resolve()
    relative_paths = (
        "configs/protocol_lock.template.yaml",
        "data/locked/file_manifest.parquet",
        "data/locked/split_manifest.parquet",
        "data/locked/ontology_lock.yaml",
        "data/locked/rgbt_input_lock.json",
        "data/locked/methane_role_lock.json",
        "data/locked/graph_eligibility_lock.json",
        "data/locked/episode_template_lock.json",
        "data/locked/episode_template_instances.parquet",
        "data/locked/episode_branch_feature_manifest.parquet",
        "data/locked/episode_skeleton_manifest.parquet",
        "data/locked/fusion_eligibility_lock.parquet",
        "data/seals/episode_test_skeleton_candidate.json",
        "data/seals/episode_skeleton_inference_projection.parquet",
        "data/seals/episode_test_skeleton_full.parquet",
        "evidence/command_receipts/E058.json",
    )
    paths = tuple(root / item for item in relative_paths)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    by_name = {path.relative_to(root).as_posix(): path for path in paths}
    protocol = yaml.safe_load(
        by_name["configs/protocol_lock.template.yaml"].read_text(encoding="utf-8-sig")
    )
    split = pd.read_parquet(by_name["data/locked/split_manifest.parquet"])
    graph = json.loads(
        by_name["data/locked/graph_eligibility_lock.json"].read_text(
            encoding="utf-8-sig"
        )
    )
    template = json.loads(
        by_name["data/locked/episode_template_lock.json"].read_text(
            encoding="utf-8-sig"
        )
    )
    instances = pd.read_parquet(
        by_name["data/locked/episode_template_instances.parquet"]
    )
    features = pd.read_parquet(
        by_name["data/locked/episode_branch_feature_manifest.parquet"]
    )
    skeleton = pd.read_parquet(
        by_name["data/locked/episode_skeleton_manifest.parquet"]
    )
    eligibility = pd.read_parquet(
        by_name["data/locked/fusion_eligibility_lock.parquet"]
    )
    candidate = json.loads(
        by_name["data/seals/episode_test_skeleton_candidate.json"].read_text(
            encoding="utf-8-sig"
        )
    )
    projection = pd.read_parquet(
        by_name["data/seals/episode_skeleton_inference_projection.parquet"]
    )
    full_test = pd.read_parquet(
        by_name["data/seals/episode_test_skeleton_full.parquet"]
    )
    receipt = json.loads(
        by_name["evidence/command_receipts/E058.json"].read_text(
            encoding="utf-8-sig"
        )
    )
    errors: list[str] = []

    _expect_equal(errors, "feature_columns", list(features.columns), list(FEATURE_COLUMNS))
    if features["record_id"].astype(str).duplicated().any():
        errors.append("episode feature record IDs are not unique")
    if any(token in str(column).lower() for column in features.columns for token in ("truth", "label", "score", "prediction")):
        errors.append("episode feature manifest contains truth or model-outcome columns")
    _expect_equal(errors, "skeleton_columns", list(skeleton.columns), list(SKELETON_COLUMNS))
    if skeleton.duplicated(["episode_id", "step_index", "concept_id", "node_id"]).any():
        errors.append("episode skeleton primary key is not unique")
    if skeleton["skeleton_item_id"].astype(str).duplicated().any():
        errors.append("skeleton_item_id is not unique")
    if not skeleton["skeleton_item_id"].astype(str).str.fullmatch(SHA256_PATTERN).all():
        errors.append("skeleton_item_id is not SHA-256")
    _expect_equal(errors, "episode_pools", sorted(set(skeleton["pool"])), sorted(EPISODE_POOLS))
    _expect_equal(
        errors,
        "episode_seeds",
        sorted(set(int(value) for value in skeleton["episode_seed"])),
        sorted(int(value) for value in protocol["seeds"]["episode"]),
    )
    if int(skeleton["step_index"].min()) < 0 or int(skeleton["step_index"].max()) >= int(
        protocol["episodes"]["main_length_steps"]
    ):
        errors.append("episode step index is outside the locked length")
    if not set(skeleton["template_family"]).issubset(set(TEMPLATE_FAMILIES)):
        errors.append("episode skeleton contains an unknown template family")
    if skeleton.groupby(["pool", "episode_seed", "raw_group_id"])["episode_id"].nunique().max() > 1:
        errors.append("a raw group is assigned to multiple episodes within one pool and seed")
    split_hash = _sha256_file(by_name["data/locked/split_manifest.parquet"])
    template_hash = _sha256_file(by_name["data/locked/episode_template_lock.json"])
    _expect_equal(errors, "skeleton.source_split_hash", set(skeleton["source_split_hash"]), {split_hash})
    _expect_equal(errors, "skeleton.template_lock_hash", set(skeleton["template_lock_hash"]), {template_hash})
    if skeleton.groupby("template_instance_id")["pool"].nunique().max() > 1:
        errors.append("template instances cross episode pools")

    feature_source = features.merge(
        split[["dataset_id", "record_id", "raw_group_id", "pool"]],
        on=["dataset_id", "record_id", "raw_group_id"],
        how="left",
        validate="one_to_one",
    )
    skeleton_source = skeleton.merge(
        feature_source[["record_id", "raw_group_id", "pool"]],
        on="record_id",
        how="left",
        suffixes=("", "_source"),
        validate="many_to_one",
    )
    if skeleton_source["pool_source"].isna().any():
        errors.append("skeleton records are missing from the locked split")
    if not skeleton_source["raw_group_id"].astype(str).equals(
        skeleton_source["raw_group_id_source"].astype(str)
    ) or not skeleton_source["pool"].astype(str).equals(
        skeleton_source["pool_source"].astype(str)
    ):
        errors.append("skeleton record pool or raw-group provenance differs from split")

    concept_steps = skeleton.drop_duplicates(
        ["episode_id", "step_index", "concept_id"]
    ).copy()
    concept_steps["event_truth"] = (
        concept_steps["event_position_role"] == "positive_observation"
    ).astype("int64")
    expected_instances = (
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
    expected_instances["template_lock_hash"] = template_hash
    sort_columns = ["template_instance_id", "episode_id"]
    try:
        pd.testing.assert_frame_equal(
            instances.sort_values(sort_columns).reset_index(drop=True),
            expected_instances.sort_values(sort_columns).reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=0,
            atol=1e-15,
        )
    except AssertionError as exc:
        errors.append(f"template instance reconciliation failed: {str(exc).splitlines()[0]}")

    expected_test = skeleton.loc[skeleton["pool"].astype(str).eq("D_e_te")].reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(full_test.reset_index(drop=True), expected_test, check_dtype=False)
    except AssertionError as exc:
        errors.append(f"full test skeleton mismatch: {str(exc).splitlines()[0]}")
    _expect_equal(errors, "projection_columns", list(projection.columns), list(PROJECTION_COLUMNS))
    if set(projection.columns) & FORBIDDEN_PROJECTION_COLUMNS:
        errors.append("inference projection contains forbidden hidden-skeleton columns")
    expected_projection = skeleton.loc[:, list(PROJECTION_COLUMNS)].reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            projection.reset_index(drop=True), expected_projection, check_dtype=False
        )
    except AssertionError as exc:
        errors.append(f"inference projection mismatch: {str(exc).splitlines()[0]}")

    acl = sorted(
        {
            "deny:trainer",
            "deny:selector",
            "deny:policy_fitter",
            "deny:test_predictor",
            "deny:human_reviewer",
        }
    )
    expected_candidate = {
        "scope": "episode_test_skeleton",
        "schema_hash": _canonical_json_sha256(sorted(skeleton.columns)),
        "full_manifest_hash": _sha256_file(
            by_name["data/seals/episode_test_skeleton_full.parquet"]
        ),
        "inference_projection_hash": _sha256_file(
            by_name["data/seals/episode_skeleton_inference_projection.parquet"]
        ),
        "row_count": int(len(expected_test)),
        "raw_group_count": int(expected_test["raw_group_id"].astype(str).nunique()),
        "acl_hash": _canonical_json_sha256(acl),
    }
    for key, value in expected_candidate.items():
        _expect_equal(errors, f"candidate.{key}", candidate.get(key), value)
    forbidden_candidate_fields = {
        "event_counts",
        "event_rates",
        "event_positions",
        "template_counts",
        "per_concept_rates",
        "label_samples",
    }
    if set(candidate) & forbidden_candidate_fields:
        errors.append("candidate receipt exposes hidden event or label distributions")

    missing_eligibility = sorted(ELIGIBILITY_REQUIRED_COLUMNS - set(eligibility.columns))
    if missing_eligibility:
        errors.append(f"fusion eligibility is missing fields: {missing_eligibility}")
    obsolete = set(eligibility.columns) & {
        "fusion_eligible",
        "graph_eligible",
        "distinct_branch_type_count",
        "independent_raw_group_count",
    }
    if obsolete:
        errors.append(f"fusion eligibility contains obsolete fields: {sorted(obsolete)}")
    candidate_hash = _sha256_file(
        by_name["data/seals/episode_test_skeleton_candidate.json"]
    )
    expected_eligibility = _independent_eligibility(
        skeleton,
        features,
        graph,
        int(protocol["data"]["positive_group_floor_per_claimed_class_hard_min"]),
        candidate_hash,
        _sha256_file(by_name["data/locked/ontology_lock.yaml"]),
        split_hash,
        _sha256_file(by_name["data/locked/episode_skeleton_manifest.parquet"]),
        _sha256_file(by_name["data/locked/graph_eligibility_lock.json"]),
    )
    for item in eligibility.to_dict("records"):
        concept_id = str(item["concept_id"])
        expected = expected_eligibility.get(concept_id)
        if expected is None:
            errors.append(f"unexpected eligibility concept: {concept_id}")
            continue
        observed = dict(item)
        observed["eligible_branch_types"] = _as_string_list(
            observed["eligible_branch_types"]
        )
        observed["multibranch_event_count_by_pool"] = {
            str(key): int(value)
            for key, value in observed["multibranch_event_count_by_pool"].items()
        }
        observed["per_pool_counts"] = json.loads(observed.pop("per_pool_counts_json"))
        for key, value in expected.items():
            _expect_equal(errors, f"eligibility.{concept_id}.{key}", observed.get(key), value)
    _expect_equal(
        errors,
        "eligibility_concepts",
        sorted(str(value) for value in eligibility["concept_id"]),
        sorted(expected_eligibility),
    )

    _expect_equal(errors, "receipt.step_id", receipt.get("step_id"), "E058")
    _expect_equal(errors, "receipt.status", receipt.get("status"), "pass")
    _expect_equal(errors, "receipt.command", receipt.get("command"), "build-episode-skeletons")
    _expect_equal(errors, "receipt.arguments", receipt.get("arguments"), {})
    expected_input_paths = (
        "data/locked/file_manifest.parquet",
        "data/locked/split_manifest.parquet",
        "data/locked/ontology_lock.yaml",
        "data/locked/rgbt_input_lock.json",
        "data/locked/methane_role_lock.json",
        "data/locked/graph_eligibility_lock.json",
    )
    expected_inputs = [
        {
            "path": item,
            "kind": "file",
            "bytes": by_name[item].stat().st_size,
            "sha256": _sha256_file(by_name[item]),
        }
        for item in expected_input_paths
    ]
    _expect_equal(errors, "receipt.inputs", receipt.get("inputs"), expected_inputs)
    expected_output_paths = (
        "data/locked/episode_template_lock.json",
        "data/locked/episode_template_instances.parquet",
        "data/locked/episode_branch_feature_manifest.parquet",
        "data/locked/episode_skeleton_manifest.parquet",
        "data/locked/fusion_eligibility_lock.parquet",
        "data/seals/episode_test_skeleton_candidate.json",
        "data/seals/episode_skeleton_inference_projection.parquet",
        "data/seals/episode_test_skeleton_full.parquet",
    )
    expected_outputs = [
        {
            "path": item,
            "kind": "file",
            "bytes": by_name[item].stat().st_size,
            "sha256": _sha256_file(by_name[item]),
        }
        for item in expected_output_paths
    ]
    _expect_equal(errors, "receipt.outputs", receipt.get("outputs"), expected_outputs)
    expected_details = {
        "row_count": int(len(skeleton)),
        "test_row_count": int(len(expected_test)),
        "fusion_eligible_concept_count": int(
            eligibility["fusion_primary_eligible"].astype(bool).sum()
        ),
    }
    _expect_equal(errors, "receipt.details", receipt.get("details"), expected_details)

    _expect_equal(
        errors,
        "template.episode_seeds",
        template.get("episode_seeds"),
        [int(value) for value in protocol["seeds"]["episode"]],
    )
    _expect_equal(
        errors,
        "template.main_length_steps",
        template.get("main_length_steps"),
        int(protocol["episodes"]["main_length_steps"]),
    )
    _expect_equal(
        errors,
        "template.template_families",
        template.get("template_families"),
        list(TEMPLATE_FAMILIES),
    )
    _expect_equal(errors, "template.model_outcomes_used", template.get("model_outcomes_used"), False)
    _expect_equal(errors, "template.test_outcomes_used", template.get("test_outcomes_used"), False)
    _expect_equal(errors, "template.source_split_sha256", template.get("source_split_sha256"), split_hash)
    _expect_equal(
        errors,
        "template.feature_manifest_sha256",
        template.get("episode_branch_feature_manifest_sha256"),
        _sha256_file(by_name["data/locked/episode_branch_feature_manifest.parquet"]),
    )

    eligibility_summary = {
        str(item["concept_id"]): {
            "fusion_primary_eligible": bool(item["fusion_primary_eligible"]),
            "graph_primary_eligible": bool(item["graph_primary_eligible"]),
            "eligible_branch_type_count": int(item["eligible_branch_type_count"]),
            "independent_multibranch_positive_group_count": int(
                item["independent_multibranch_positive_group_count"]
            ),
            "exclusion_reason": item["exclusion_reason"],
        }
        for item in eligibility.to_dict("records")
    }
    diagnostics = {
        "skeleton_row_count": int(len(skeleton)),
        "test_row_count": int(len(expected_test)),
        "feature_row_count": int(len(features)),
        "template_instance_count": int(len(instances)),
        "episode_count": int(skeleton["episode_id"].astype(str).nunique()),
        "raw_group_count": int(skeleton["raw_group_id"].astype(str).nunique()),
        "projection_column_count": int(len(projection.columns)),
        "eligibility": eligibility_summary,
        "test_full_rows_exposed_to_predictor": False,
        "full_local_dataset_extractions": 0,
    }
    return _result(root, errors, paths, diagnostics)


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"immutable review output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Independently reconcile E058 episode skeletons and seals"
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = review_e058(args.project_root)
    if args.output is not None:
        output = args.output if args.output.is_absolute() else args.project_root / args.output
        _write_once(output, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
