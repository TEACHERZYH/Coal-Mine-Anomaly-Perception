from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import Image
import yaml

from mining1_exp.training_data import build_yolo_view, target_detection_records


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_target_fewshot_view_is_group_locked_and_ontology_remapped(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    (root / "data/locked").mkdir(parents=True)
    image_root = root / "data/canonical/target/images"
    image_root.mkdir(parents=True)
    rows = []
    split_rows = []
    for index, (group, pool) in enumerate(
        (("g1", "D_b_tr"), ("g2", "D_b_tr"), ("g3", "D_b_sel"))
    ):
        path = image_root / f"image-{index}.jpg"
        Image.new("RGB", (16, 12), color=(index * 20, 30, 40)).save(path)
        record_id = f"record-{index}"
        rows.append(
            {
                "dataset_id": "target",
                "record_id": record_id,
                "archive_id": "target-v1",
                "relative_path": path.relative_to(root).as_posix(),
                "modality": "visible",
                "raw_group_id": group,
                "pair_id": None,
                "sequence_id": None,
                "timestamp_or_order": None,
                "label_summary_json": json.dumps(
                    {
                        "image_width": 16,
                        "image_height": 12,
                        "boxes": [
                            {
                                "source_label": "person",
                                "x_center_normalized": 0.5,
                                "y_center_normalized": 0.5,
                                "width_normalized": 0.25,
                                "height_normalized": 0.5,
                            }
                        ],
                    },
                    sort_keys=True,
                ),
                "byte_size": path.stat().st_size,
                "sha256": _sha(path),
            }
        )
        split_rows.append(
            {
                "dataset_id": "target",
                "record_id": record_id,
                "raw_group_id": group,
                "pool": pool,
                "split_seed": 13007,
                "split_version": "fixture-v1",
                "ontology_hash": "a" * 64,
                "dedup_report_hash": "b" * 64,
            }
        )
    pd.DataFrame(rows).to_parquet(root / "data/locked/file_manifest.parquet", index=False)
    pd.DataFrame(split_rows).to_parquet(root / "data/locked/split_manifest.parquet", index=False)
    pd.DataFrame(
        [
            {
                "dataset_id": "target",
                "direction_id": "fixture",
                "ratio_percent": 10,
                "subset_seed": 5171,
                "raw_group_id": group,
                "included": group == "g1",
                "class_group_counts_json": "{}",
                "parent_manifest_hash": "c" * 64,
            }
            for group in ("g1", "g2")
        ]
    ).to_parquet(root / "data/locked/fewshot_manifest.parquet", index=False)
    ontology = {
        "ontology_version": "fixture-v1",
        "entries": [
            {
                "dataset_id": "target",
                "source_label": "person",
                "canonical_concept_id": "worker_presence",
                "mapping_status": "compatible",
                "annotation_policy": "fixture boxes are exhaustive",
                "event_semantics": "observable_presence",
                "negative_semantics": "exhaustive_verified_absence",
                "allowed_tasks": ["confirmatory_f1"],
                "evidence_reference": "fixture:target",
                "reviewer_decision": "accept for contract test",
            }
        ],
    }
    (root / "data/locked/ontology_lock.yaml").write_text(
        yaml.safe_dump(ontology, sort_keys=True), encoding="utf-8"
    )
    selected = target_detection_records(
        root, dataset_id="target", subset_seed=5171
    )
    assert set(selected["record_id"]) == {"record-0", "record-2"}
    receipt = build_yolo_view(root, view_id="target-10-seed5171", records=selected)
    assert receipt["train_count"] == 1
    assert receipt["validation_count"] == 1
    assert receipt["concepts"] == ["worker_presence"]
    manifest = pd.read_parquet(root / receipt["view_manifest_path"])
    label = root / manifest.iloc[0]["label_path"]
    assert label.read_text(encoding="ascii").startswith("0 ")


def test_target_records_map_locked_infrared_source_to_thermal_branch(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    (root / "data/locked").mkdir(parents=True)
    files = []
    splits = []
    for index, (source_modality, pool) in enumerate(
        (
            ("visible", "D_b_tr"),
            ("infrared", "D_b_tr"),
            ("visible", "D_b_sel"),
            ("infrared", "D_b_sel"),
        )
    ):
        record_id = f"record-{index}"
        group_id = f"group-{index // 2}"
        files.append(
            {
                "dataset_id": "rgbt",
                "record_id": record_id,
                "archive_id": "fixture-v1",
                "relative_path": f"data/canonical/rgbt/{record_id}.jpg",
                "modality": source_modality,
                "raw_group_id": group_id,
                "pair_id": f"pair-{index // 2}",
                "sequence_id": None,
                "timestamp_or_order": index,
                "label_summary_json": "{}",
                "byte_size": 1,
                "sha256": "a" * 64,
            }
        )
        splits.append(
            {
                "dataset_id": "rgbt",
                "record_id": record_id,
                "raw_group_id": group_id,
                "pool": pool,
                "split_seed": 13007,
                "split_version": "fixture-v1",
                "ontology_hash": "b" * 64,
                "dedup_report_hash": "c" * 64,
            }
        )
    pd.DataFrame(files).to_parquet(root / "data/locked/file_manifest.parquet", index=False)
    pd.DataFrame(splits).to_parquet(root / "data/locked/split_manifest.parquet", index=False)
    (root / "data/locked/rgbt_input_lock.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "dataset_id": "rgbt",
                "source_modality_to_branch_role": {
                    "visible": "visible",
                    "infrared": "thermal",
                },
            }
        ),
        encoding="utf-8",
    )
    thermal = target_detection_records(root, dataset_id="rgbt", modality="thermal")
    visible = target_detection_records(root, dataset_id="rgbt", modality="visible")
    assert set(thermal["modality"]) == {"infrared"}
    assert set(thermal["branch_modality"]) == {"thermal"}
    assert set(visible["modality"]) == {"visible"}
