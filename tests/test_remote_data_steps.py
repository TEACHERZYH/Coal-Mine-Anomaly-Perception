from __future__ import annotations

import json
import hashlib
from pathlib import Path
import random
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from mining1_exp.remote_data_steps import (
    _candidate_near_pairs,
    _image_phash,
    audit_groups_dedup,
    build_file_manifest,
    verify_archives,
)
import mining1_exp.remote_data_steps as remote_data_steps
from mining1_exp.workflow_common import WorkflowExecutionError
from mining1_exp.workflow_data import _dedup_assignment_manifest


def _write_archive(tmp_path: Path, name: str) -> Path:
    source = tmp_path / f"source-{name}"
    image_path = source / "dataset/images/g1/frame001.jpg"
    label_path = source / "dataset/labels/g1/frame001.txt"
    image_path.parent.mkdir(parents=True)
    label_path.parent.mkdir(parents=True)
    Image.new("RGB", (16, 12), color=(20, 40, 60)).save(image_path)
    label_path.write_text("0 0.5 0.5 0.25 0.5\n", encoding="ascii")
    archive = tmp_path / f"{name}.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.write(image_path, "dataset/images/g1/frame001.jpg")
        handle.write(label_path, "dataset/labels/g1/frame001.txt")
    return archive


def _adapter() -> dict:
    path_rule = {
        "regex": {
            "source": "relative_path",
            "pattern": r"images/(?P<group>[^/]+)/(?P<record>[^/]+)\.jpg",
            "template": "{group}-{record}",
        }
    }
    return {
        "schema_version": 1,
        "kind": "yolo_detection",
        "archive_subdir": "dataset",
        "record_id_rule": path_rule,
        "raw_group_rule": {
            "regex": {
                "source": "relative_path",
                "pattern": r"images/(?P<group>[^/]+)/(?P<record>[^/]+)\.jpg",
                "template": "{group}",
            }
        },
        "modality_rule": {"constant": "visible"},
        "pair_id_rule": {"null": True},
        "sequence_id_rule": {"null": True},
        "timestamp_rule": {"null": True},
        "yolo": {
            "image_globs": ["images/**/*.jpg"],
            "image_root": "images",
            "label_root": "labels",
            "class_names": ["person"],
            "missing_label_policy": "error",
        },
    }


def _write_ontology(root: Path, dataset_labels: dict[str, str]) -> None:
    entries = []
    for dataset_id, source_label in sorted(dataset_labels.items()):
        entries.append(
            {
                "dataset_id": dataset_id,
                "source_label": source_label,
                "canonical_concept_id": "worker_presence",
                "mapping_status": "compatible",
                "annotation_policy": "synthetic test policy",
                "event_semantics": "observable_presence",
                "negative_semantics": "exhaustive_verified_absence",
                "allowed_tasks": ["confirmatory_f1"],
                "evidence_reference": "synthetic-test-evidence",
                "reviewer_decision": "accept",
            }
        )
    path = root / "data/locked/ontology_lock.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"ontology_version": "test-v1", "entries": entries}),
        encoding="utf-8",
    )


def test_remote_manifest_and_dedup_steps_are_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "project"
    (root / "evidence/data").mkdir(parents=True)
    archives = {
        dataset_id: _write_archive(tmp_path, dataset_id)
        for dataset_id in ("source-a", "source-b")
    }
    roles = {
        "primary_visual_sources": [
            {
                "dataset_id": dataset_id,
                "dataset_version": "v1",
                "adapter_contract": _adapter(),
            }
            for dataset_id in sorted(archives)
        ]
    }
    (root / "evidence/data/dataset_source_decision.json").write_text(
        json.dumps({"status": "pass", "roles": roles}), encoding="utf-8"
    )
    verified = []
    for dataset_id, archive in sorted(archives.items()):
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        verified.append(
            {
                "dataset_id": dataset_id,
                "archive_id": f"{dataset_id}-{digest[:16]}",
                "remote_path": str(archive),
                "sha256": digest,
            }
        )
    (root / "evidence/data/remote_archive_verification.json").write_text(
        json.dumps({"status": "pass", "archives": verified}), encoding="utf-8"
    )
    built = build_file_manifest(root, tmp_path / "run", None)
    assert built["status"] == "pass"
    manifest = pd.read_parquet(root / "data/locked/file_manifest.parquet")
    assert len(manifest) == 2
    _write_ontology(root, {dataset_id: "person" for dataset_id in archives})
    result = audit_groups_dedup(root, tmp_path / "run", None)
    assert result["status"] == "pass"
    report = pd.read_parquet(root / "data/locked/dedup_report.parquet")
    assert len(report) == 1
    assert report.iloc[0]["duplicate_kind"] == "exact_sha256"
    assert report.iloc[0]["resolution"] == "merge_connected_raw_groups_before_split"


def test_remote_verification_groups_archive_parts_by_dataset(
    tmp_path: Path, monkeypatch
) -> None:
    remote_home = tmp_path / "remote-home"
    root = remote_home / "project"
    evidence = root / "evidence/data"
    evidence.mkdir(parents=True)
    images = remote_home / "datasets/images.zip"
    labels = remote_home / "datasets/labels.zip"
    images.parent.mkdir(parents=True)
    images.write_bytes(b"images")
    labels.write_bytes(b"labels")
    image_hash = hashlib.sha256(images.read_bytes()).hexdigest()
    label_hash = hashlib.sha256(labels.read_bytes()).hexdigest()
    license_path = evidence / "fixture.license"
    license_path.write_text("fixture license\n", encoding="ascii")
    license_hash = hashlib.sha256(license_path.read_bytes()).hexdigest()
    entry = {
        "dataset_id": "source-a",
        "dataset_version": "v1",
        "source_path_or_url": str(images),
        "archive_sha256": image_hash,
        "transfer_route": "remote_existing",
        "license_id": "fixture",
        "adapter_contract": _adapter(),
        "companion_archives": [
            {
                "archive_part_id": "labels",
                "source_path_or_url": str(labels),
                "archive_sha256": label_hash,
            }
        ],
    }
    (evidence / "dataset_source_decision.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "roles": {"primary_visual_sources": [entry]},
                "license_snapshots": [
                    {
                        "dataset_id": "source-a",
                        "snapshot_path": license_path.relative_to(root).as_posix(),
                        "sha256": license_hash,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (evidence / "staging_receipts.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "archives": [
                    {
                        "dataset_id": "source-a",
                        "planned_role": "primary_visual_sources",
                        "archive_part_id": "primary",
                        "remote_path": str(images),
                        "sha256": image_hash,
                    },
                    {
                        "dataset_id": "source-a",
                        "planned_role": "primary_visual_sources",
                        "archive_part_id": "labels",
                        "remote_path": str(labels),
                        "sha256": label_hash,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(remote_data_steps, "REMOTE_HOME", remote_home)
    result = verify_archives(root, tmp_path / "run", None)
    assert result["details"] == {"archive_count": 2, "dataset_count": 1}
    payload = json.loads(
        (evidence / "remote_archive_verification.json").read_text(encoding="utf-8")
    )
    assert len(payload["archives"]) == 1
    assert {part["archive_part_id"] for part in payload["archives"][0]["archives"]} == {
        "primary",
        "labels",
    }


def test_dedup_assignment_merges_cross_dataset_groups_without_rewriting_ids() -> None:
    manifest = pd.DataFrame(
        [
            {"dataset_id": "a", "record_id": "a1", "raw_group_id": "ga"},
            {"dataset_id": "b", "record_id": "b1", "raw_group_id": "gb"},
            {"dataset_id": "b", "record_id": "b2", "raw_group_id": "gc"},
        ]
    )
    duplicates = pd.DataFrame(
        [
            {
                "left_dataset_id": "a",
                "left_record_id": "a1",
                "right_dataset_id": "b",
                "right_record_id": "b1",
            }
        ]
    )
    assignment, original = _dedup_assignment_manifest(manifest, duplicates)
    assert assignment.iloc[0]["raw_group_id"] == assignment.iloc[1]["raw_group_id"]
    assert assignment.iloc[2]["raw_group_id"] != assignment.iloc[1]["raw_group_id"]
    assert original.equals(manifest)


def test_near_duplicate_candidates_cover_locked_hamming_threshold() -> None:
    left = 0
    right = sum(1 << bit for bit in range(4))
    pairs = list(_candidate_near_pairs({0: left, 1: right}))
    assert pairs == [(0, 1, 4)]


def test_bktree_near_duplicate_candidates_match_bruteforce() -> None:
    generator = random.Random(13007)
    hashes = {index: generator.getrandbits(64) for index in range(200)}
    hashes[200] = hashes[0]
    hashes[201] = hashes[1] ^ sum(1 << bit for bit in range(4))
    observed = set(_candidate_near_pairs(hashes))
    expected = {
        (left, right, bin(hashes[left] ^ hashes[right]).count("1"))
        for left in hashes
        for right in hashes
        if left < right and bin(hashes[left] ^ hashes[right]).count("1") <= 4
    }
    assert observed == expected


def test_dedup_report_rejects_low_information_same_phash_collisions(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "project"
    (root / "data/locked").mkdir(parents=True)
    rows = []
    for index, value in enumerate((80, 110, 140), start=1):
        image = root / f"data/canonical/d{index}/image.png"
        image.parent.mkdir(parents=True)
        Image.new("L", (32, 32), color=value).save(image)
        rows.append(
            {
                "dataset_id": f"d{index}",
                "record_id": f"r{index}",
                "archive_id": f"a{index}",
                "relative_path": image.relative_to(root).as_posix(),
                "modality": "visible",
                "raw_group_id": f"g{index}",
                "pair_id": "",
                "sequence_id": "",
                "timestamp_or_order": index,
                "label_summary_json": json.dumps(
                    {
                        "class_ids": ["person"],
                        "boxes": [],
                        "negative_annotation_verified": True,
                    },
                    sort_keys=True,
                ),
                "byte_size": image.stat().st_size,
                "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            }
        )
    pd.DataFrame(rows).to_parquet(
        root / "data/locked/file_manifest.parquet", index=False
    )
    _write_ontology(root, {"d1": "person", "d2": "person", "d3": "person"})
    monkeypatch.setenv("MINING1_DEDUP_WORKERS", "2")
    result = audit_groups_dedup(root, tmp_path / "run", None)
    assert result["status"] == "pass"
    report = pd.read_parquet(root / "data/locked/dedup_report.parquet")
    assert report.empty
    summary = json.loads((root / "evidence/data/dedup_summary.json").read_text())
    assert summary["report_semantics"].startswith("minimal_deterministic")
    assert summary["affected_raw_group_count"] == 0
    assert summary["dedup_component_count"] == 3
    assert summary["near_rejected_structure_count"] == 3
    assert summary["phash_worker_count"] == 2


def test_dedup_report_confirms_structured_brightness_shift(tmp_path: Path) -> None:
    root = tmp_path / "project"
    (root / "data/locked").mkdir(parents=True)
    y, x = np.indices((64, 64))
    base = (20 + x + y + 20 * ((x // 8 + y // 8) % 2)).astype(np.uint8)
    variants = [base, (base.astype(np.int16) + 30).astype(np.uint8)]
    rows = []
    for index, values in enumerate(variants, start=1):
        image = root / f"data/canonical/d{index}/image.png"
        image.parent.mkdir(parents=True)
        Image.fromarray(values, mode="L").save(image)
        rows.append(
            {
                "dataset_id": f"d{index}",
                "record_id": f"r{index}",
                "archive_id": f"a{index}",
                "relative_path": image.relative_to(root).as_posix(),
                "modality": "visible",
                "raw_group_id": f"g{index}",
                "pair_id": "",
                "sequence_id": "",
                "timestamp_or_order": index,
                "label_summary_json": json.dumps(
                    {
                        "class_ids": ["person"],
                        "boxes": [],
                        "negative_annotation_verified": True,
                    },
                    sort_keys=True,
                ),
                "byte_size": image.stat().st_size,
                "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            }
        )
    pd.DataFrame(rows).to_parquet(root / "data/locked/file_manifest.parquet", index=False)
    _write_ontology(root, {"d1": "person", "d2": "person"})
    result = audit_groups_dedup(root, tmp_path / "run", None)
    assert result["status"] == "pass"
    report = pd.read_parquet(root / "data/locked/dedup_report.parquet")
    assert len(report) == 1
    assert report.iloc[0]["duplicate_kind"] == "near_phash64"
    assert report.iloc[0]["near_confirmation_method"].startswith("phash64_hamming4")
    assert float(report.iloc[0]["thumbnail_intensity_correlation"]) >= 0.995


def test_exact_duplicate_coordinate_variation_is_audited_not_rejected(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    (root / "data/locked").mkdir(parents=True)
    first = root / "data/canonical/d1/first.png"
    second = root / "data/canonical/d1/second.png"
    first.parent.mkdir(parents=True)
    Image.new("L", (32, 32), color=100).save(first)
    second.write_bytes(first.read_bytes())
    rows = []
    for record_id, group_id, path, x1 in (
        ("r1", "g1", first, 0.1),
        ("r2", "g2", second, 0.2),
    ):
        label = {
            "class_ids": ["person"],
            "boxes": [{"source_label": "person", "x1": x1}],
            "negative_annotation_verified": True,
        }
        rows.append(
            {
                "dataset_id": "d1",
                "record_id": record_id,
                "archive_id": "a1",
                "relative_path": path.relative_to(root).as_posix(),
                "modality": "visible",
                "raw_group_id": group_id,
                "pair_id": "",
                "sequence_id": "",
                "timestamp_or_order": "",
                "label_summary_json": json.dumps(label, sort_keys=True),
                "byte_size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    pd.DataFrame(rows).to_parquet(root / "data/locked/file_manifest.parquet", index=False)
    _write_ontology(root, {"d1": "person"})
    result = audit_groups_dedup(root, tmp_path / "run", None)
    assert result["status"] == "pass"
    report = pd.read_parquet(root / "data/locked/dedup_report.parquet")
    assert len(report) == 1
    assert bool(report.iloc[0]["label_summary_conflict"]) is True
    assert bool(report.iloc[0]["semantic_label_conflict"]) is False
    summary = json.loads((root / "evidence/data/dedup_summary.json").read_text())
    assert summary["exact_label_summary_conflict_group_count"] == 1
    assert summary["exact_semantic_conflict_group_count"] == 0


def test_exact_duplicate_semantic_conflict_remains_fatal(tmp_path: Path) -> None:
    root = tmp_path / "project"
    (root / "data/locked").mkdir(parents=True)
    first = root / "data/canonical/d1/first.png"
    second = root / "data/canonical/d1/second.png"
    first.parent.mkdir(parents=True)
    Image.new("L", (32, 32), color=100).save(first)
    second.write_bytes(first.read_bytes())
    labels = (
        {
            "class_ids": ["person"],
            "boxes": [{"source_label": "person", "x1": 0.1}],
            "negative_annotation_verified": True,
        },
        {
            "class_ids": [],
            "boxes": [],
            "negative_annotation_verified": True,
        },
    )
    rows = []
    for index, (path, label) in enumerate(zip((first, second), labels), start=1):
        rows.append(
            {
                "dataset_id": "d1",
                "record_id": f"r{index}",
                "archive_id": "a1",
                "relative_path": path.relative_to(root).as_posix(),
                "modality": "visible",
                "raw_group_id": f"g{index}",
                "pair_id": "",
                "sequence_id": "",
                "timestamp_or_order": "",
                "label_summary_json": json.dumps(label, sort_keys=True),
                "byte_size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    pd.DataFrame(rows).to_parquet(root / "data/locked/file_manifest.parquet", index=False)
    _write_ontology(root, {"d1": "person"})
    with pytest.raises(WorkflowExecutionError, match="ontology-level semantics"):
        audit_groups_dedup(root, tmp_path / "run", None)


def test_phash_is_stable_to_uniform_brightness_shift(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("L", (32, 32), color=80).save(first)
    Image.new("L", (32, 32), color=140).save(second)
    assert _image_phash(first) == _image_phash(second)
