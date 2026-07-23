from __future__ import annotations

from copy import deepcopy
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from mining1_exp.data.corruptions import (
    CorruptionContractError,
    apply_corruption,
    build_corruption_record,
    validate_corruption_manifest,
)
from mining1_exp.data.episodes import (
    EpisodeContractError,
    make_skeleton_item_id,
    select_model_features,
    validate_episode_skeleton_manifest,
    validate_fusion_eligibility_lock,
)
from mining1_exp.data.groups import assign_raw_group_ids, audit_duplicates
from mining1_exp.data.inventory import bounded_file_inventory, read_archive_manifest
from mining1_exp.data.manifests import (
    ARCHIVE_COLUMNS,
    FILE_COLUMNS,
    ManifestValidationError,
    SPLIT_COLUMNS,
    assert_fit_pool,
    read_file_manifest,
    validate_file_manifest,
    validate_methane_windows,
    validate_split_manifest,
    write_parquet_atomic,
)
from mining1_exp.data.ontology import (
    ONTOLOGY_ENTRY_FIELDS,
    OntologyValidationError,
    validate_ontology_lock,
)
from mining1_exp.data.seals import (
    DENIED_LABEL_ROLES,
    SealValidationError,
    TEST_RELEASE_REQUIRED_FIELDS,
    TEST_SEAL_REQUIRED_FIELDS,
    build_candidate_test_seal,
    build_test_release,
    validate_test_release,
)
from mining1_exp.data.splits import (
    FEWSHOT_COLUMNS,
    assert_no_duplicate_cross_pool,
    assert_shared_fewshot_groups,
    assign_balanced_component_pools,
    assign_group_pools,
    assign_temporal_cohort_pools,
    build_fewshot_manifest,
    causal_window_masks,
)
from mining1_exp.data.corruptions import CORRUPTION_COLUMNS
from mining1_exp.data.episodes import EPISODE_SKELETON_COLUMNS


def _sha(character: str) -> str:
    return character * 64


def _split_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["split_seed"] = 2026
    frame["split_version"] = "v1"
    frame["ontology_hash"] = _sha("a")
    frame["dedup_report_hash"] = _sha("b")
    return validate_split_manifest(frame)


def test_schema_constants_match_the_frozen_artifact_contract() -> None:
    contract_path = Path(__file__).resolve().parents[1] / "configs" / "artifact_contract.template.yaml"
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))["artifacts"]
    assert set(contract["archive_manifest"]["required_columns"]) == ARCHIVE_COLUMNS
    assert set(contract["file_manifest"]["required_columns"]) == FILE_COLUMNS
    assert set(contract["split_manifest"]["required_columns"]) == SPLIT_COLUMNS
    assert set(contract["fewshot_manifest"]["required_columns"]) == FEWSHOT_COLUMNS
    assert set(contract["corruption_manifest"]["required_columns"]) == CORRUPTION_COLUMNS
    assert set(contract["test_seal"]["required_fields"]) == TEST_SEAL_REQUIRED_FIELDS
    assert set(contract["test_release"]["required_fields"]) == TEST_RELEASE_REQUIRED_FIELDS
    assert set(contract["episode_skeleton_manifest"]["required_columns"]) == EPISODE_SKELETON_COLUMNS
    assert set(contract["ontology_lock"]["required_fields"]) == {
        "ontology_version",
        *ONTOLOGY_ENTRY_FIELDS,
    }


def test_archive_manifest_and_bounded_inventory_do_not_extract(tmp_path: Path) -> None:
    manifest_path = tmp_path / "archive_manifest.csv"
    pd.DataFrame(
        [
            {
                "dataset_id": "visual_a",
                "archive_id": "archive_1",
                "source_url_or_doi": "doi:10.example/source",
                "access_date": "2026-07-15",
                "license_id": "research-only",
                "version": "v1",
                "local_or_remote_path": str(tmp_path / "source.zip"),
                "byte_size": 12,
                "sha256": _sha("a"),
                "acquisition_route": "local_existing",
                "verification_status": "pass",
            }
        ]
    ).to_csv(manifest_path, index=False)
    loaded = read_archive_manifest(manifest_path)
    assert loaded.loc[0, "archive_id"] == "archive_1"

    inventory_root = tmp_path / "inventory"
    inventory_root.mkdir()
    for index in range(3):
        (inventory_root / f"file_{index}.bin").write_bytes(bytes([index]))
    inventory = bounded_file_inventory(inventory_root, max_files=2)
    assert inventory["file_count"] == 2
    assert inventory["truncated"] is True
    assert inventory["archives_extracted"] is False
    assert all(Path(item["relative_path"]).name.startswith("file_") for item in inventory["files"])


def test_raw_groups_precede_file_manifest_and_parquet_roundtrip(tmp_path: Path) -> None:
    source = pd.DataFrame(
        {
            "dataset_id": ["rgbt", "rgbt"],
            "record_id": ["visible_1", "thermal_1"],
            "archive_id": ["archive_1", "archive_1"],
            "relative_path": ["visible/1.png", "thermal/1.png"],
            "modality": ["visible", "thermal"],
            "pair_id": ["pair_1", "pair_1"],
            "sequence_id": [None, None],
            "timestamp_or_order": ["1", "1"],
            "label_summary_json": ['{"person":1}', '{"person":1}'],
            "byte_size": [10, 11],
            "sha256": [_sha("a"), _sha("b")],
            "source_sequence": ["sequence_1", "sequence_1"],
        }
    )
    grouped = assign_raw_group_ids(source, ["dataset_id", "source_sequence", "pair_id"])
    assert grouped["raw_group_id"].nunique() == 1
    validated = validate_file_manifest(grouped, paired_dataset_ids={"rgbt"})
    target = write_parquet_atomic(validated, tmp_path / "file_manifest.parquet")
    loaded = read_file_manifest(target)
    pd.testing.assert_frame_equal(
        loaded.sort_index(axis=1),
        validated.sort_index(axis=1),
        check_dtype=False,
    )
    assert not list(tmp_path.glob("*.tmp"))


def test_ontology_requires_annotation_evidence_and_verified_negatives() -> None:
    entry = {
        "dataset_id": "visual_a",
        "source_label": "helmet",
        "canonical_concept_id": "ppe_helmet",
        "mapping_status": "compatible",
        "annotation_policy": "exhaustive per image",
        "event_semantics": "observable_presence",
        "negative_semantics": "exhaustive_verified_absence",
        "allowed_tasks": ["confirmatory_f1"],
        "evidence_reference": "dataset-card section 3",
        "reviewer_decision": "accept",
    }
    payload = {"ontology_version": "v1", "entries": [entry]}
    assert validate_ontology_lock(payload)["entries"][0]["canonical_concept_id"] == "ppe_helmet"

    name_only = deepcopy(payload)
    name_only["entries"][0]["evidence_reference"] = ""
    with pytest.raises(OntologyValidationError, match="name-only"):
        validate_ontology_lock(name_only)

    unverified_negative = deepcopy(payload)
    unverified_negative["entries"][0]["negative_semantics"] = "unverified_missing_annotation"
    with pytest.raises(OntologyValidationError, match="verified negative"):
        validate_ontology_lock(unverified_negative)


def test_duplicate_audit_detects_noncontiguous_exact_and_near_pairs() -> None:
    records = pd.DataFrame(
        [
            {"dataset_id": "d1", "record_id": "r1", "sha256": _sha("a"), "phash": "f0", "modality": "visible"},
            {"dataset_id": "d1", "record_id": "r2", "sha256": _sha("b"), "phash": "00", "modality": "visible"},
            {"dataset_id": "d2", "record_id": "r3", "sha256": _sha("a"), "phash": "ff", "modality": "visible"},
            {"dataset_id": "d2", "record_id": "r4", "sha256": _sha("c"), "phash": "01", "modality": "visible"},
        ]
    )
    report = audit_duplicates(
        records,
        phash_column="phash",
        metadata_columns=["modality"],
        hamming_threshold=1,
    )
    exact = report.loc[report["duplicate_type"] == "exact"]
    near = report.loc[report["duplicate_type"] == "near"]
    assert set(zip(exact["left_record_id"], exact["right_record_id"])) == {("r1", "r3")}
    assert set(zip(near["left_record_id"], near["right_record_id"])) == {("r2", "r4")}

    split = _split_frame(
        [
            {"dataset_id": "d1", "record_id": "r1", "raw_group_id": "g1", "pool": "D_b_tr"},
            {"dataset_id": "d2", "record_id": "r3", "raw_group_id": "g2", "pool": "D_b_te"},
            {"dataset_id": "d1", "record_id": "r2", "raw_group_id": "g3", "pool": "D_b_tr"},
            {"dataset_id": "d2", "record_id": "r4", "raw_group_id": "g4", "pool": "D_b_tr"},
        ]
    )
    with pytest.raises(ManifestValidationError, match="cross pools"):
        assert_no_duplicate_cross_pool(split, report)


def test_group_split_is_deterministic_and_group_disjoint() -> None:
    records = pd.DataFrame(
        [
            {"dataset_id": "d1", "record_id": f"r{group}_{item}", "raw_group_id": f"g{group}"}
            for group in range(40)
            for item in range(2)
        ]
    )
    kwargs = {
        "pool_weights": {"D_b_tr": 0.6, "D_b_sel": 0.2, "D_b_te": 0.2},
        "seed": 2026,
        "split_version": "v1",
        "ontology_hash": _sha("a"),
        "dedup_report_hash": _sha("b"),
    }
    first = assign_group_pools(records, **kwargs)
    reversed_kwargs = dict(kwargs)
    reversed_kwargs["pool_weights"] = {
        "D_b_te": 0.2,
        "D_b_sel": 0.2,
        "D_b_tr": 0.6,
    }
    second = assign_group_pools(
        records.sample(frac=1, random_state=7), **reversed_kwargs
    )
    columns = ["dataset_id", "record_id", "raw_group_id", "pool"]
    pd.testing.assert_frame_equal(
        first[columns].sort_values(["dataset_id", "record_id"]).reset_index(drop=True),
        second[columns].sort_values(["dataset_id", "record_id"]).reset_index(drop=True),
    )
    assert first.groupby(["dataset_id", "raw_group_id"])["pool"].nunique().max() == 1


def test_balanced_component_split_populates_every_pool_deterministically() -> None:
    weights = {
        "D_b_tr": 0.30,
        "D_b_sel": 0.075,
        "D_b_prob": 0.05,
        "D_b_te": 0.075,
        "D_e_tr": 0.25,
        "D_e_sel": 0.075,
        "D_e_pol": 0.075,
        "D_e_te": 0.10,
    }
    membership = pd.DataFrame(
        [
            {"dataset_id": dataset_id, "component_id": f"{dataset_id}-c{index:02d}"}
            for dataset_id, count in (("d1", 26), ("d2", 17))
            for index in range(count)
        ]
    )
    first, diagnostics = assign_balanced_component_pools(
        membership, pool_weights=weights, seed=13007
    )
    second, _ = assign_balanced_component_pools(
        membership.sample(frac=1, random_state=7),
        pool_weights=dict(reversed(list(weights.items()))),
        seed=13007,
    )
    assert first == second
    assert diagnostics["cross_dataset_component_count"] == 0
    assert diagnostics["dataset_component_counts"] == {"d1": 26, "d2": 17}
    for counts in diagnostics["dataset_pool_component_counts"].values():
        assert set(counts) == set(weights)
        assert all(value >= 1 for value in counts.values())


def test_balanced_component_split_keeps_cross_dataset_component_in_one_pool() -> None:
    weights = {pool: 1.0 for pool in (
        "D_b_tr", "D_b_sel", "D_b_prob", "D_b_te",
        "D_e_tr", "D_e_sel", "D_e_pol", "D_e_te",
    )}
    rows = [{"dataset_id": dataset_id, "component_id": "shared"} for dataset_id in ("d1", "d2")]
    rows.extend(
        {"dataset_id": dataset_id, "component_id": f"{dataset_id}-c{index}"}
        for dataset_id in ("d1", "d2")
        for index in range(8)
    )
    assignment, diagnostics = assign_balanced_component_pools(
        pd.DataFrame(rows), pool_weights=weights, seed=13007
    )
    assert assignment["shared"] in weights
    assert diagnostics["cross_dataset_component_count"] == 1
    assert all(
        set(counts) == set(weights) and all(value >= 1 for value in counts.values())
        for counts in diagnostics["dataset_pool_component_counts"].values()
    )


def test_balanced_component_split_rejects_fewer_components_than_pools() -> None:
    weights = {pool: 1.0 for pool in (
        "D_b_tr", "D_b_sel", "D_b_prob", "D_b_te",
        "D_e_tr", "D_e_sel", "D_e_pol", "D_e_te",
    )}
    membership = pd.DataFrame(
        [{"dataset_id": "d1", "component_id": f"c{index}"} for index in range(7)]
    )
    with pytest.raises(ManifestValidationError, match="independent components"):
        assign_balanced_component_pools(membership, pool_weights=weights, seed=13007)


def test_temporal_cohort_split_is_contiguous_purged_and_deterministic() -> None:
    weights = {
        "D_b_tr": 0.30,
        "D_b_sel": 0.075,
        "D_b_prob": 0.05,
        "D_b_te": 0.075,
        "D_e_tr": 0.25,
        "D_e_sel": 0.075,
        "D_e_pol": 0.075,
        "D_e_te": 0.10,
    }
    rows = []
    for cohort in range(16):
        timestamp = (pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=cohort)).isoformat()
        for sensor in ("MM263", "MM264", "MM256"):
            rows.append(
                {
                    "dataset_id": "methane",
                    "component_id": f"{sensor}-{cohort}",
                    "timestamp_or_order": timestamp,
                }
            )
    records = pd.DataFrame(rows)
    first, diagnostics = assign_temporal_cohort_pools(
        records,
        pool_weights=weights,
        seed=13007,
        dataset_id="methane",
        group_duration_seconds=86400,
        purge_gap_seconds=600,
    )
    second, _ = assign_temporal_cohort_pools(
        records.sample(frac=1, random_state=7),
        pool_weights=dict(reversed(list(weights.items()))),
        seed=13007,
        dataset_id="methane",
        group_duration_seconds=86400,
        purge_gap_seconds=600,
    )
    assert first == second
    assert diagnostics["cohort_count"] == 16
    assert diagnostics["component_count"] == 48
    assert diagnostics["same_cohort_cross_pool_count"] == 0
    assert diagnostics["minimum_family_gap_seconds"] >= 600
    for cohort in range(16):
        assert len({first[f"{sensor}-{cohort}"] for sensor in ("MM263", "MM264", "MM256")}) == 1


def test_temporal_cohort_split_rejects_noncontinuous_source() -> None:
    weights = {pool: 1.0 for pool in (
        "D_b_tr", "D_b_sel", "D_b_prob", "D_b_te",
        "D_e_tr", "D_e_sel", "D_e_pol", "D_e_te",
    )}
    records = pd.DataFrame(
        [
            {
                "dataset_id": "methane",
                "component_id": f"c{index}",
                "timestamp_or_order": (
                    pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=index + (1 if index >= 4 else 0))
                ).isoformat(),
            }
            for index in range(8)
        ]
    )
    with pytest.raises(ManifestValidationError, match="continuous ordered series"):
        assign_temporal_cohort_pools(
            records,
            pool_weights=weights,
            seed=13007,
            dataset_id="methane",
            group_duration_seconds=86400,
            purge_gap_seconds=600,
        )


def test_fewshot_uses_three_paired_ten_percent_group_subsets() -> None:
    split = _split_frame(
        [
            {
                "dataset_id": "d1",
                "record_id": f"r{index}",
                "raw_group_id": f"g{index:02d}",
                "pool": "D_b_tr" if index < 20 else "D_b_te",
            }
            for index in range(21)
        ]
    )
    classes = {("d1", f"g{index:02d}"): ["person"] for index in range(20)}
    manifest = build_fewshot_manifest(
        split,
        direction_id="visible_to_thermal",
        subset_seeds=[2026, 2027, 2028],
        group_classes=classes,
        parent_manifest_hash=_sha("c"),
    )
    assert set(manifest["subset_seed"]) == {2026, 2027, 2028}
    assert manifest.groupby("subset_seed")["included"].sum().to_dict() == {
        2026: 2,
        2027: 2,
        2028: 2,
    }
    assert "g20" not in set(manifest["raw_group_id"])
    assert_shared_fewshot_groups(manifest, manifest.copy())

    changed = manifest.copy()
    seed_rows = changed.index[changed["subset_seed"] == 2026]
    included_index = next(index for index in seed_rows if bool(changed.at[index, "included"]))
    excluded_index = next(index for index in seed_rows if not bool(changed.at[index, "included"]))
    changed.at[included_index, "included"] = False
    changed.at[excluded_index, "included"] = True
    with pytest.raises(ManifestValidationError, match="identical raw groups"):
        assert_shared_fewshot_groups(manifest, changed)
    with pytest.raises(ManifestValidationError, match="exactly three"):
        build_fewshot_manifest(
            split,
            direction_id="visible_to_thermal",
            subset_seeds=[2026, 2027],
            group_classes=classes,
            parent_manifest_hash=_sha("c"),
        )


def test_methane_windows_are_causal_purged_and_fit_pool_limited() -> None:
    timestamps = pd.Series(
        pd.to_datetime(
            ["2026-01-01T00:00:00Z", "2026-01-01T00:05:00Z", "2026-01-01T00:10:00Z"]
        )
    )
    history, forecast = causal_window_masks(
        timestamps,
        anchor=pd.Timestamp("2026-01-01T00:05:00Z"),
        history_seconds=300,
        horizon_seconds=300,
    )
    assert history.tolist() == [False, True, False]
    assert forecast.tolist() == [False, False, True]

    windows = pd.DataFrame(
        [
            {
                "dataset_id": "methane",
                "window_id": "w1",
                "raw_group_id": "g1",
                "sensor_group_id": "s1",
                "history_start": "2026-01-01T00:00:00Z",
                "history_end": "2026-01-01T00:05:00Z",
                "forecast_end": "2026-01-01T00:10:00Z",
                "pool": "D_b_tr",
            },
            {
                "dataset_id": "methane",
                "window_id": "w2",
                "raw_group_id": "g2",
                "sensor_group_id": "s1",
                "history_start": "2026-01-01T00:20:00Z",
                "history_end": "2026-01-01T00:25:00Z",
                "forecast_end": "2026-01-01T00:30:00Z",
                "pool": "D_b_te",
            },
        ]
    )
    validate_methane_windows(windows, purge_gap_seconds=600)
    overlapping = windows.copy()
    overlapping.loc[1, "history_start"] = "2026-01-01T00:15:00Z"
    with pytest.raises(ManifestValidationError, match="purge gap"):
        validate_methane_windows(overlapping, purge_gap_seconds=600)
    assert_fit_pool("D_b_tr", {"D_b_tr", "D_b_sel"})
    with pytest.raises(ManifestValidationError, match="not authorized"):
        assert_fit_pool("D_b_te", {"D_b_tr", "D_b_sel"})


def test_corruptions_are_deterministic_and_cannot_receive_labels() -> None:
    image = np.arange(12 * 12 * 3, dtype=np.uint8).reshape(12, 12, 3)
    signature = inspect.signature(apply_corruption)
    assert set(signature.parameters) == {
        "image",
        "record_id",
        "corruption_type",
        "severity",
        "operator_version",
    }
    assert not {"label", "truth", "boxes", "prediction"}.intersection(
        signature.parameters
    )

    hashes = set()
    for corruption_type in ("low_light", "dust_fog_proxy"):
        for severity in (1, 2, 3):
            first = apply_corruption(
                image,
                record_id="record_1",
                corruption_type=corruption_type,
                severity=severity,
                operator_version="v1",
            )
            labels = ["helmet", "person", "none"]
            labels.reverse()
            second = apply_corruption(
                image,
                record_id="record_1",
                corruption_type=corruption_type,
                severity=severity,
                operator_version="v1",
            )
            assert first.tobytes() == second.tobytes()
            row = build_corruption_record(
                image=image,
                record_id="record_1",
                raw_group_id="group_1",
                corruption_type=corruption_type,
                severity=severity,
                operator_version="v1",
                generator_hash=_sha("a"),
                source_feature_hash=_sha("b"),
            )
            hashes.add(row["corrupted_feature_hash"])
            assert row["label_independent"] is True
    assert len(hashes) == 6
    with pytest.raises(CorruptionContractError, match="forbidden"):
        validate_corruption_manifest(
            pd.DataFrame(
                [
                    {
                        **row,
                        "label": "helmet",
                    }
                ]
            )
        )


def test_test_seal_release_is_append_only_and_labels_are_denied() -> None:
    seal = build_candidate_test_seal(
        scope="branch",
        split_hash=_sha("a"),
        feature_manifest_hash=_sha("b"),
        label_manifest_hash=_sha("c"),
        artifact_contract_hash=_sha("d"),
        feature_acl=["allow:test_predictor"],
        label_acl=sorted(DENIED_LABEL_ROLES),
    )
    original = deepcopy(seal)
    release = build_test_release(
        scope="branch",
        seal_hash=_sha("e"),
        prediction_lock_hash=_sha("f"),
        protocol_hash=_sha("1"),
        source_hash=_sha("2"),
        matrix_hash=_sha("3"),
        evaluator_identity="independent-evaluator",
        evaluator_label_acl=["allow:evaluator:independent-evaluator"],
        signer_identity="release-authority",
        sealed_payload=seal,
    )
    assert release["authorization_status"] == "released"
    validate_test_release(release)
    tampered = {**release, "evaluator_identity": "other-evaluator"}
    with pytest.raises(SealValidationError, match="does not bind"):
        validate_test_release(tampered)
    assert seal == original
    with pytest.raises(SealValidationError, match="not denied"):
        build_candidate_test_seal(
            scope="branch",
            split_hash=_sha("a"),
            feature_manifest_hash=_sha("b"),
            label_manifest_hash=_sha("c"),
            artifact_contract_hash=_sha("d"),
            feature_acl=["allow:test_predictor"],
            label_acl=sorted(DENIED_LABEL_ROLES - {"deny:test_predictor"}),
        )


def test_episode_skeleton_id_is_join_only_and_truth_free() -> None:
    nonce = b"0123456789abcdef"
    first = make_skeleton_item_id(
        nonce, record_id="record_1", concept_id="ppe", node_id="visible"
    )
    repeated = make_skeleton_item_id(
        nonce, record_id="record_1", concept_id="ppe", node_id="visible"
    )
    second = make_skeleton_item_id(
        nonce, record_id="record_2", concept_id="ppe", node_id="visible"
    )
    repeated_instance = make_skeleton_item_id(
        nonce,
        record_id="record_1",
        concept_id="ppe",
        node_id="visible",
        instance_salt=b"fedcba9876543210",
    )
    assert first == repeated
    assert len(first) == 64
    assert first != second
    assert first != repeated_instance

    skeleton = pd.DataFrame(
        [
            {
                "skeleton_item_id": first,
                "episode_id": "episode_1",
                "step_index": 0,
                "record_id": "record_1",
                "raw_group_id": "group_1",
                "concept_id": "ppe",
                "node_id": "visible",
                "pool": "D_e_tr",
                "episode_seed": 2026,
                "generator_family_id": "family_1",
                "template_instance_id": "template_1",
                "template_family": "abrupt",
                "event_position_role": "background",
                "source_split_hash": _sha("a"),
                "template_lock_hash": _sha("b"),
            }
        ]
    )
    validate_episode_skeleton_manifest(skeleton)
    leaked = skeleton.assign(truth=[1])
    with pytest.raises(EpisodeContractError, match="forbidden"):
        validate_episode_skeleton_manifest(leaked)

    frame = pd.DataFrame(
        {
            "skeleton_item_id": [first],
            "record_id": ["record_1"],
            "score": [0.75],
            "temperature": [0.2],
        }
    )
    selected = select_model_features(frame, ["score", "temperature"])
    assert selected.columns.tolist() == ["score", "temperature"]
    with pytest.raises(EpisodeContractError, match="forbidden"):
        select_model_features(frame, ["score", "record_id"])


def _valid_fusion_eligibility_row() -> dict:
    return {
        "concept_id": "worker_presence",
        "eligible_branch_type_count": 2,
        "eligible_branch_types": ["thermal", "visible"],
        "episode_skeleton_candidate_hash": "a" * 64,
        "exclusion_reason": None,
        "fusion_primary_eligible": True,
        "graph_primary_eligible": True,
        "independent_multibranch_positive_group_count": 10,
        "multibranch_event_count_by_pool": {
            "D_e_tr": 10,
            "D_e_sel": 10,
            "D_e_pol": 10,
            "D_e_te": 10,
        },
        "ontology_lock_hash": "b" * 64,
        "split_manifest_hash": "c" * 64,
        "model_outcomes_used": False,
    }


def test_fusion_eligibility_lock_matches_frozen_runtime_contract() -> None:
    frame = pd.DataFrame([_valid_fusion_eligibility_row()])

    validated = validate_fusion_eligibility_lock(
        frame, positive_group_floor=10
    )

    assert validated.loc[0, "fusion_primary_eligible"]
    assert validated.loc[0, "eligible_branch_type_count"] == 2


def test_fusion_eligibility_lock_rejects_obsolete_aliases() -> None:
    row = _valid_fusion_eligibility_row()
    row["fusion_eligible"] = True

    with pytest.raises(EpisodeContractError, match="obsolete or forbidden"):
        validate_fusion_eligibility_lock(pd.DataFrame([row]))


def test_fusion_eligibility_lock_rejects_inconsistent_branch_count() -> None:
    row = _valid_fusion_eligibility_row()
    row["eligible_branch_type_count"] = 1

    with pytest.raises(EpisodeContractError, match="does not match"):
        validate_fusion_eligibility_lock(pd.DataFrame([row]))
