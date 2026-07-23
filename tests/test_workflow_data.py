from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pandas as pd
import pytest
import yaml

from mining1_exp.provenance import sha256_file
import mining1_exp.workflow_data as workflow_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_zip(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as handle:
        for name, data in members.items():
            handle.writestr(name, data)


def _source_entry(
    dataset_id: str,
    archive: Path,
    records: list[dict],
) -> dict:
    license_path = archive.with_suffix(".LICENSE.txt")
    license_path.write_text("test-only permissive fixture\n", encoding="ascii")
    if dataset_id == "methane":
        adapter_contract = {
            "schema_version": 1,
            "kind": "methane_csv",
            "archive_subdir": ".",
            "methane": {
                "csv_globs": ["**/*.csv"],
                "timestamp_column": "timestamp",
                "value_column": "value",
                "sensor_group_column": "sensor",
                "feature_columns": [],
                "group_duration_seconds": 3600,
                "timezone": "UTC",
            },
        }
    else:
        modality_rule = {"constant": "visible"}
        if dataset_id == "rgbt":
            modality_rule = {
                "regex": {
                    "source": "relative_path",
                    "pattern": r"g1/(?P<modality>visible|thermal)\.jpg",
                    "template": "{modality}",
                }
            }
        adapter_contract = {
            "schema_version": 1,
            "kind": "yolo_detection",
            "archive_subdir": ".",
            "record_id_rule": {
                "regex": {
                    "source": "relative_path",
                    "pattern": r"g1/(?P<record>[^/]+)\.jpg",
                    "template": "{record}",
                }
            },
            "raw_group_rule": {"constant": "g1"},
            "modality_rule": modality_rule,
            "pair_id_rule": {"null": True},
            "sequence_id_rule": {"null": True},
            "timestamp_rule": {"null": True},
            "yolo": {
                "image_globs": ["**/*.jpg"],
                "image_root": "g1",
                "label_root": "g1",
                "class_names": ["person"],
                "missing_label_policy": "empty",
            },
        }
    return {
        "dataset_id": dataset_id,
        "dataset_version": "test-v1",
        "source_location": "local",
        "source_path_or_url": str(archive),
        "archive_sha256": sha256_file(archive),
        "license_id": "test-license",
        "license_evidence_path": str(license_path),
        "transfer_route": "local_existing_upload",
        "selection_evidence": "fixture metadata only",
        "adapter_contract": adapter_contract,
        "minimal_sample_records": records,
    }


def _reviewed_sources(tmp_path: Path) -> dict:
    archives = tmp_path / "archives"
    target = archives / "target.zip"
    source_a = archives / "source-a.zip"
    source_b = archives / "source-b.zip"
    source_c = archives / "source-c.zip"
    rgbt = archives / "rgbt.zip"
    methane = archives / "methane.zip"
    _write_zip(target, {"g1/target.jpg": b"target"})
    _write_zip(source_a, {"g1/source-a.jpg": b"source-a"})
    _write_zip(source_b, {"g1/source-b.jpg": b"source-b"})
    _write_zip(source_c, {"g1/source-c.jpg": b"source-c"})
    _write_zip(rgbt, {"g1/visible.jpg": b"visible", "g1/thermal.jpg": b"thermal"})
    _write_zip(methane, {"g1/methane.csv": b"timestamp,value\n0,0.1\n"})
    return {
        "schema_version": 1,
        "status": "reviewed",
        "visual_direction_id": "targetA_from_sources",
        "roles": {
            "primary_visual_target": _source_entry(
                "target",
                target,
                [
                    {
                        "record_id": "target-1",
                        "raw_group_id": "g1",
                        "member_path": "g1/target.jpg",
                        "modality": "visible",
                    }
                ],
            ),
            "primary_visual_sources": [
                {
                    **_source_entry(
                        "source-a",
                        source_a,
                        [
                            {
                                "record_id": "source-a-1",
                                "raw_group_id": "g1",
                                "member_path": "g1/source-a.jpg",
                                "modality": "visible",
                            }
                        ],
                    ),
                    "pretraining_conditions": ["generic_matched"],
                },
                {
                    **_source_entry(
                        "source-b",
                        source_b,
                        [
                            {
                                "record_id": "source-b-1",
                                "raw_group_id": "g1",
                                "member_path": "g1/source-b.jpg",
                                "modality": "visible",
                            }
                        ],
                    ),
                    "pretraining_conditions": ["single_coal", "multi_coal"],
                },
                {
                    **_source_entry(
                        "source-c",
                        source_c,
                        [
                            {
                                "record_id": "source-c-1",
                                "raw_group_id": "g1",
                                "member_path": "g1/source-c.jpg",
                                "modality": "visible",
                            }
                        ],
                    ),
                    "pretraining_conditions": ["multi_coal"],
                },
            ],
            "primary_rgbt_dataset": _source_entry(
                "rgbt",
                rgbt,
                [
                    {
                        "record_id": "rgbt-v-1",
                        "raw_group_id": "g1",
                        "member_path": "g1/visible.jpg",
                        "modality": "visible",
                    },
                    {
                        "record_id": "rgbt-t-1",
                        "raw_group_id": "g1",
                        "member_path": "g1/thermal.jpg",
                        "modality": "thermal",
                    },
                ],
            ),
            "methane_dataset": _source_entry(
                "methane",
                methane,
                [
                    {
                        "record_id": "methane-1",
                        "raw_group_id": "g1",
                        "member_path": "g1/methane.csv",
                        "modality": "methane",
                        "sample_row_limit": 2,
                    }
                ],
            ),
        },
    }


def test_local_and_mocked_remote_inventory_are_bounded(
    tmp_path: Path,
    monkeypatch,
) -> None:
    search_root = tmp_path / "datasets"
    _write_zip(search_root / "candidate.zip", {"x.txt": b"x"})
    (search_root / "LICENSE.txt").write_text("fixture\n", encoding="ascii")
    local_root = tmp_path / "local-project"
    result = workflow_data.inventory_local_datasets(
        local_root, {"step_id": "E010"}, {"root": str(search_root)}
    )
    assert result["details"]["candidate_count"] == 1
    frame = pd.read_parquet(local_root / "evidence/data/local_dataset_inventory.parquet")
    assert len(frame) == 2
    assert not frame["fully_extracted_locally"].any()

    remote_payload = json.dumps(
        {
            "hostname": "mu01",
            "entries": [
                {
                    "path": "/data/home/xinxi-zhyh/xinxi-zhyh/datasets/a.zip",
                    "byte_size": 1,
                    "sha256": "0" * 64,
                    "kind": "candidate_file",
                }
            ],
        },
        sort_keys=True,
    )
    monkeypatch.setattr(
        workflow_data,
        "run_ssh_script",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=remote_payload + "\n__SQUEUE__\n", stderr="", returncode=0
        ),
    )
    remote_root = tmp_path / "remote-project"
    result = workflow_data.inventory_remote_datasets(
        remote_root, {"step_id": "E014"}, {"bounded": True}
    )
    assert result["details"]["created_compute_allocation"] is False
    payload = json.loads(
        (remote_root / "evidence/data/remote_dataset_inventory.json").read_text()
    )
    assert payload["max_files"] == 512
    assert payload["created_compute_allocation"] is False


def test_reviewed_sources_build_only_the_selected_minimal_members(tmp_path: Path) -> None:
    root = tmp_path / "project"
    reviewed = _reviewed_sources(tmp_path)
    reviewed_path = root / "evidence/data/dataset_source_candidates.reviewed.json"
    reviewed_path.parent.mkdir(parents=True)
    reviewed_path.write_text(json.dumps(reviewed), encoding="utf-8")
    decision = workflow_data.decide_dataset_sources(
        root, {"step_id": "E016"}, {}
    )
    assert decision["details"]["selected_entry_count"] == 6
    result = workflow_data.build_minimal_sample(
        root, {"step_id": "E020"}, {"max_groups": 2}
    )
    assert result["details"]["full_extractions"] == 0
    frame = pd.read_parquet(root / "evidence/data/minimal_sample_manifest.parquet")
    assert len(frame) == 7
    assert set(frame["modality"]) >= {"visible", "thermal", "methane"}
    assert frame.groupby("dataset_id")["raw_group_id"].nunique().max() <= 2
    methane_sample = root / frame.loc[frame["dataset_id"] == "methane", "sample_path"].iloc[0]
    assert len(methane_sample.read_text(encoding="utf-8").splitlines()) <= 3


def test_minimal_sample_candidates_are_bounded_by_reviewed_group_order() -> None:
    records = [
        {
            "record_id": "a-visible",
            "raw_group_id": "group-a",
            "member_path": "a/visible.jpg",
            "modality": "visible",
        },
        {
            "record_id": "a-thermal",
            "raw_group_id": "group-a",
            "member_path": "a/thermal.jpg",
            "modality": "thermal",
        },
        {
            "record_id": "b-visible",
            "raw_group_id": "group-b",
            "member_path": "b/visible.jpg",
            "modality": "visible",
        },
        {
            "record_id": "c-visible",
            "raw_group_id": "group-c",
            "member_path": "c/visible.jpg",
            "modality": "visible",
        },
    ]
    bounded, candidates, selected = workflow_data._bounded_minimal_sample_records(
        "fixture", records, 2
    )
    assert candidates == ["group-a", "group-b", "group-c"]
    assert selected == ["group-a", "group-b"]
    assert [record["record_id"] for record in bounded] == [
        "a-visible",
        "a-thermal",
        "b-visible",
    ]


def _write_smoke_fixture(root: Path, mapping: dict[str, str]) -> None:
    rows = []
    for modality in ("visible", "infrared", "methane"):
        sample = root / f"state/minimal_sample/{modality}.bin"
        sample.parent.mkdir(parents=True, exist_ok=True)
        sample.write_bytes(modality.encode("ascii"))
        rows.append(
            {
                "dataset_id": "fixture",
                "record_id": modality,
                "raw_group_id": f"group-{modality}",
                "modality": modality,
                "sample_path": sample.relative_to(root).as_posix(),
                "sha256": sha256_file(sample),
            }
        )
    manifest = root / "evidence/data/minimal_sample_manifest.parquet"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(manifest, index=False)
    lock = root / "data/locked/rgbt_input_lock.json"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(
        json.dumps(
            {
                "status": "pass",
                "source_modality_to_branch_role": mapping,
            }
        ),
        encoding="utf-8",
    )


def test_smoke_local_uses_locked_infrared_to_thermal_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "project"
    _write_smoke_fixture(root, {"visible": "visible", "infrared": "thermal"})

    def fake_pipeline(bundle: Path) -> dict[str, str]:
        bundle.mkdir(parents=True)
        (bundle / "integration_receipt.json").write_text("{}", encoding="ascii")
        return {"mode": "synthetic_minimal_integration"}

    monkeypatch.setattr(workflow_data, "run_minimal_pipeline", fake_pipeline)
    result = workflow_data.smoke_local(
        root, {"step_id": "E062"}, {"minimal": True}
    )
    payload = json.loads((root / "evidence/smoke/local_smoke.json").read_text())
    assert result["status"] == "pass"
    assert payload["actual_sample_modalities"] == ["infrared", "methane", "visible"]
    assert payload["actual_sample_branch_roles"] == ["thermal", "visible"]


def test_smoke_local_rejects_missing_locked_thermal_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "project"
    _write_smoke_fixture(root, {"visible": "visible", "infrared": "infrared"})
    monkeypatch.setattr(
        workflow_data,
        "run_minimal_pipeline",
        lambda bundle: pytest.fail("pipeline must not run without thermal coverage"),
    )
    with pytest.raises(
        workflow_data.WorkflowExecutionError,
        match="thermal-branch",
    ):
        workflow_data.smoke_local(root, {"step_id": "E062"}, {"minimal": True})


def test_reviewed_source_fallback_requires_complete_candidate_evidence(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    reviewed = _reviewed_sources(tmp_path)
    reviewed["roles"]["primary_visual_sources"] = reviewed["roles"][
        "primary_visual_sources"
    ][:2]
    reviewed_path = root / "evidence/data/dataset_source_candidates.reviewed.json"
    reviewed_path.parent.mkdir(parents=True)
    reviewed_path.write_text(json.dumps(reviewed), encoding="utf-8")
    with pytest.raises(workflow_data.WorkflowExecutionError, match="reviewed C-TRANSFER"):
        workflow_data.decide_dataset_sources(root, {"step_id": "E016"}, {})

    authority = root / "notes/minimal_experiment_plan_rereview_20260715.md"
    authority.parent.mkdir(parents=True)
    authority.write_text("fewer than two eligible coal sources removes C-TRANSFER\n")
    review_dir = root / "evidence/data/source_reviews"
    review_dir.mkdir(parents=True)
    candidate_reviews = {
        "source-b.json": ("source-b", "eligible"),
        "source-c.json": ("source-c", "rejected"),
    }
    review_paths = []
    for name, (dataset_id, disposition) in candidate_reviews.items():
        path = review_dir / name
        path.write_text(
            json.dumps(
                {
                    "dataset_entry": {"dataset_id": dataset_id},
                    "eligibility_decision": {
                        "confirmatory_multi_coal_source": disposition
                    },
                }
            ),
            encoding="utf-8",
        )
        review_paths.append(path.relative_to(root).as_posix())
    reviewed["claim_fallbacks"] = {
        "C-TRANSFER": {
            "status": "not_applicable",
            "reason_code": "fewer_than_two_eligible_non_target_coal_sources",
            "eligible_non_target_coal_dataset_ids": ["source-b"],
            "candidate_review_paths": review_paths,
            "authority_path": authority.relative_to(root).as_posix(),
            "authority_sha256": sha256_file(authority),
        }
    }
    reviewed_path.write_text(json.dumps(reviewed), encoding="utf-8")
    result = workflow_data.decide_dataset_sources(root, {"step_id": "E016"}, {})
    assert result["details"]["selected_entry_count"] == 5
    decision = json.loads(
        (root / "evidence/data/dataset_source_decision.json").read_text()
    )
    assert decision["claim_eligibility"]["C-TRANSFER"]["status"] == "not_applicable"


def test_transfer_fallback_is_forbidden_when_two_coal_sources_are_eligible(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    reviewed = _reviewed_sources(tmp_path)
    reviewed["claim_fallbacks"] = {
        "C-TRANSFER": {
            "status": "not_applicable",
            "reason_code": "fewer_than_two_eligible_non_target_coal_sources",
        }
    }
    reviewed_path = root / "evidence/data/dataset_source_candidates.reviewed.json"
    reviewed_path.parent.mkdir(parents=True)
    reviewed_path.write_text(json.dumps(reviewed), encoding="utf-8")
    with pytest.raises(workflow_data.WorkflowExecutionError, match="fallback is forbidden"):
        workflow_data.decide_dataset_sources(root, {"step_id": "E016"}, {})


def test_stage_datasets_stages_every_reviewed_archive_part(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "project"
    reviewed = _reviewed_sources(tmp_path)
    companion = tmp_path / "archives/source-a-annotations.zip"
    _write_zip(companion, {"annotations/a.txt": b"label"})
    source_a = reviewed["roles"]["primary_visual_sources"][0]
    source_a["companion_archives"] = [
        {
            "archive_part_id": "annotations",
            "source_path_or_url": str(companion),
            "archive_sha256": sha256_file(companion),
        }
    ]
    reviewed_path = root / "evidence/data/dataset_source_candidates.reviewed.json"
    reviewed_path.parent.mkdir(parents=True)
    reviewed_path.write_text(json.dumps(reviewed), encoding="utf-8")
    workflow_data.decide_dataset_sources(root, {"step_id": "E016"}, {})
    monkeypatch.setattr(
        workflow_data,
        "_stage_local_archive",
        lambda local, remote, expected: (expected, "uploaded_resumable_atomic"),
    )
    result = workflow_data.stage_datasets(
        root,
        {"step_id": "E022"},
        {"decision": "evidence/data/dataset_source_decision.json"},
    )
    assert result["details"] == {
        "archive_count": 7,
        "dataset_count": 6,
        "full_local_extractions": 0,
        "uploaded_archive_bytes_upper_bound": sum(
            Path(value["source_path_or_url"]).stat().st_size
            + sum(Path(part["source_path_or_url"]).stat().st_size for part in value.get("companion_archives", []))
            for values in reviewed["roles"].values()
            for value in (values if isinstance(values, list) else [values])
        ),
        "resumable_upload": True,
        "remote_hash_partition": "cu",
    }
    receipt = json.loads(
        (root / "evidence/data/staging_receipts.json").read_text(encoding="utf-8")
    )
    source_parts = {
        item["archive_part_id"]
        for item in receipt["archives"]
        if item["dataset_id"] == "source-a"
    }
    assert source_parts == {"primary", "annotations"}
    assert receipt["transfer_policy"] == {
        "protocol": "openssh_sftp_reput_to_temporary_then_atomic_promote",
        "resume_supported": True,
        "remote_hash_partition": "cu",
        "login_node_full_archive_hashes": 0,
    }


def test_resumable_staging_hashes_on_compute_before_atomic_promotion(
    tmp_path: Path, monkeypatch
) -> None:
    local = tmp_path / "archive.zip"
    local.write_bytes(b"archive")
    expected = hashlib.sha256(b"archive").hexdigest()
    remote = "/data/home/xinxi-zhyh/xinxi-zhyh/datasets/mining1/a/archive.zip"
    sizes = iter([None, None, local.stat().st_size])
    scripts = []
    uploads = []
    monkeypatch.setattr(workflow_data, "_remote_file_size", lambda path: next(sizes))
    monkeypatch.setattr(workflow_data, "_remote_sha256", lambda path: expected)
    monkeypatch.setattr(
        workflow_data, "run_sftp_reput", lambda source, target: uploads.append(target)
    )
    monkeypatch.setattr(
        workflow_data, "run_ssh_script", lambda script, **kwargs: scripts.append(script)
    )

    observed, action = workflow_data._stage_local_archive(local, remote, expected)

    assert observed == expected
    assert action == "uploaded_resumable_atomic"
    assert uploads == [f"{remote}.upload.part"]
    assert any("mkdir -p" in script for script in scripts)
    assert any("umask 077" in script and ": >" in script for script in scripts)
    assert any("mv --" in script and ".upload.part" in script for script in scripts)


def test_resumable_staging_reuses_verified_complete_remote(
    tmp_path: Path, monkeypatch
) -> None:
    local = tmp_path / "archive.zip"
    local.write_bytes(b"archive")
    expected = hashlib.sha256(b"archive").hexdigest()
    remote = "/data/home/xinxi-zhyh/xinxi-zhyh/datasets/mining1/a/archive.zip"
    monkeypatch.setattr(
        workflow_data, "_remote_file_size", lambda path: local.stat().st_size
    )
    monkeypatch.setattr(workflow_data, "_remote_sha256", lambda path: expected)

    assert workflow_data._stage_local_archive(local, remote, expected) == (
        expected,
        "reused_verified_remote",
    )


def _manifest_row(
    dataset_id: str,
    record_id: str,
    raw_group_id: str,
    modality: str,
    pair_id: str,
    order: int | str,
) -> dict:
    digest = hashlib.sha256(f"{dataset_id}|{record_id}".encode()).hexdigest()
    return {
        "dataset_id": dataset_id,
        "record_id": record_id,
        "archive_id": f"{dataset_id}-archive",
        "relative_path": f"{dataset_id}/{record_id}.dat",
        "modality": modality,
        "raw_group_id": raw_group_id,
        "pair_id": pair_id,
        "sequence_id": raw_group_id,
        "timestamp_or_order": order,
        "label_summary_json": json.dumps(
            {"class_ids": ["worker_presence"]}, sort_keys=True
        ),
        "byte_size": 1,
        "sha256": digest,
    }


def _ontology_entries(dataset_ids: list[str]) -> list[dict]:
    return [
        {
            "dataset_id": dataset_id,
            "source_label": "person",
            "canonical_concept_id": "worker_presence",
            "mapping_status": "compatible",
            "annotation_policy": "fixture boxes are exhaustive",
            "event_semantics": "observable_presence",
            "negative_semantics": "exhaustive_verified_absence",
            "allowed_tasks": ["confirmatory_f1"],
            "evidence_reference": f"fixture:{dataset_id}",
            "reviewer_decision": "accept for contract test",
        }
        for dataset_id in dataset_ids
    ]


def test_grouped_locks_cover_split_fewshot_rgbt_methane_and_graph(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    (root / "data/locked").mkdir(parents=True)
    (root / "evidence/data").mkdir(parents=True)
    (root / "configs").mkdir(parents=True)
    (root / "configs/protocol_lock.template.yaml").write_bytes(
        (PROJECT_ROOT / "configs/protocol_lock.template.yaml").read_bytes()
    )
    rows = []
    for index in range(80):
        group = f"v{index:03d}"
        rows.append(_manifest_row("target", f"target-{index}", group, "visible", "", index))
    for index in range(40):
        group = f"r{index:03d}"
        pair = f"pair-{index:03d}"
        rows.append(_manifest_row("rgbt", f"rgbt-v-{index}", group, "visible", pair, index))
        rows.append(_manifest_row("rgbt", f"rgbt-t-{index}", group, "infrared", pair, index))
    for index in range(60):
        cohort = index // 3
        group = f"m{index:03d}"
        record_id = f"methane-{index}"
        source = root / f"data/canonical/methane/{record_id}.parquet"
        source.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "__timestamp_utc": pd.date_range(
                    "2026-01-01", periods=31, freq="30s", tz="UTC"
                ),
                "value": [0.2 if step < 20 else 1.2 for step in range(31)],
                "sensor": [["MM263", "MM264", "MM256"][index % 3]] * 31,
            }
        ).to_parquet(source, index=False)
        manifest_row = _manifest_row(
            "methane",
            record_id,
            group,
            "methane",
            "",
            (pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=cohort)).isoformat(),
        )
        manifest_row.update(
            {
                "relative_path": source.relative_to(root).as_posix(),
                "byte_size": source.stat().st_size,
                "sha256": sha256_file(source),
            }
        )
        rows.append(manifest_row)
    frame = pd.DataFrame(rows)
    frame["timestamp_or_order"] = frame["timestamp_or_order"].astype(str)
    frame.to_parquet(root / "data/locked/file_manifest.parquet", index=False)
    pd.DataFrame(
        columns=[
            "left_dataset_id",
            "left_record_id",
            "right_dataset_id",
            "right_record_id",
            "duplicate_type",
            "phash_hamming_distance",
        ]
    ).to_parquet(root / "data/locked/dedup_report.parquet", index=False)
    roles = {
        "primary_visual_target": {"dataset_id": "target"},
        "primary_visual_sources": [
            {"dataset_id": "source-a"},
            {"dataset_id": "source-b"},
        ],
        "primary_rgbt_dataset": {"dataset_id": "rgbt"},
        "methane_dataset": {
            "dataset_id": "methane",
            "adapter_contract": {
                "schema_version": 1,
                "kind": "methane_csv",
                "archive_subdir": ".",
                "methane": {
                    "csv_globs": ["**/*.csv"],
                    "timestamp_column": "timestamp",
                    "value_column": "value",
                    "sensor_group_column": "sensor",
                    "feature_columns": [],
                    "group_duration_seconds": 86400,
                    "timezone": "UTC",
                },
            },
        },
    }
    (root / "evidence/data/dataset_source_decision.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "roles": roles,
                "visual_direction_id": "target_from_sources",
            }
        ),
        encoding="utf-8",
    )
    reviewed_ontology = {
        "ontology_version": "fixture-v1",
        "entries": _ontology_entries(["target", "source-a", "source-b", "rgbt", "methane"]),
    }
    (root / "evidence/data/ontology_mapping.reviewed.yaml").write_text(
        yaml.safe_dump(reviewed_ontology, sort_keys=True), encoding="utf-8"
    )
    workflow_data.lock_ontology(root, {"step_id": "E030"}, {})
    split_result = workflow_data.create_splits(
        root, {"step_id": "E038"}, {"grouped": True}
    )
    workflow_data.create_fewshot(
        root, {"step_id": "E042"}, {"ratio": 10, "pairs": 3}
    )
    rgbt = workflow_data.audit_rgbt_inputs(root, {"step_id": "E046"}, {})
    methane = workflow_data.lock_methane_roles(root, {"step_id": "E050"}, {})
    graph = workflow_data.audit_graph_eligibility(root, {"step_id": "E054"}, {})
    assert rgbt["details"]["valid_pair_count"] == 40
    assert rgbt["details"]["observed_source_modalities"] == ["infrared", "visible"]
    assert rgbt["details"]["physical_temperature_claim_enabled"] is False
    assert methane["details"]["pool_count"] == 8
    assert methane["details"]["minimum_family_purge_gap_seconds"] >= 600
    assert graph["details"]["graph_eligible"] is True
    fewshot = pd.read_parquet(root / "data/locked/fewshot_manifest.parquet")
    assert fewshot["subset_seed"].nunique() == 3
    split = pd.read_parquet(root / "data/locked/split_manifest.parquet")
    assert set(split["pool"]) == {
        "D_b_tr",
        "D_b_sel",
        "D_b_prob",
        "D_b_te",
        "D_e_tr",
        "D_e_sel",
        "D_e_pol",
        "D_e_te",
    }
    assert split.groupby(["dataset_id", "raw_group_id"])["pool"].nunique().max() == 1
    assert split_result["details"]["raw_group_count"] == int(
        split[["dataset_id", "raw_group_id"]].drop_duplicates().shape[0]
    )
    assert sum(split_result["details"]["pool_record_counts"].values()) == len(split)
    assert sum(split_result["details"]["pool_raw_group_counts"].values()) == int(
        split[["dataset_id", "raw_group_id"]].drop_duplicates().shape[0]
    )


def test_methane_role_lock_rejects_nonchronological_pool_order_before_window_build(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    (root / "data/locked").mkdir(parents=True)
    (root / "evidence/data").mkdir(parents=True)
    (root / "configs").mkdir(parents=True)
    rows = []
    for index in range(8):
        rows.append(
            _manifest_row(
                "methane",
                f"methane-{index}",
                f"g{index}",
                "methane",
                "",
                (pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=index)).isoformat(),
            )
        )
    manifest = pd.DataFrame(rows)
    manifest.to_parquet(root / "data/locked/file_manifest.parquet", index=False)
    split = manifest[["dataset_id", "record_id", "raw_group_id"]].copy()
    split["pool"] = [
        "D_e_te",
        "D_b_te",
        "D_e_pol",
        "D_b_prob",
        "D_e_sel",
        "D_b_sel",
        "D_e_tr",
        "D_b_tr",
    ]
    split["split_seed"] = 13007
    split["split_version"] = "fixture"
    split["ontology_hash"] = "a" * 64
    split["dedup_report_hash"] = "b" * 64
    split.to_parquet(root / "data/locked/split_manifest.parquet", index=False)
    decision = {
        "roles": {
            "methane_dataset": {
                "dataset_id": "methane",
                "adapter_contract": {
                    "methane": {"group_duration_seconds": 86400}
                },
            }
        }
    }
    (root / "evidence/data/dataset_source_decision.json").write_text(
        json.dumps(decision), encoding="utf-8"
    )
    protocol = {
        "data": {
            "methane": {
                "history_seconds": 300,
                "horizon_seconds": 300,
                "stride_seconds": 30,
                "purge_gap_seconds": 600,
            }
        }
    }
    (root / "configs/protocol_lock.template.yaml").write_text(
        yaml.safe_dump(protocol), encoding="utf-8"
    )

    with pytest.raises(workflow_data.WorkflowExecutionError, match="chronological pool order"):
        workflow_data.lock_methane_roles(root, {"step_id": "E050"}, {})
