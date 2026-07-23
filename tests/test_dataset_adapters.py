from __future__ import annotations

import json
from pathlib import Path
import subprocess
import zipfile

import pandas as pd
from PIL import Image
import pytest

from mining1_exp.data.adapters import (
    AdapterContractError,
    _seven_zip_executable,
    apply_rule,
    adapter_contract_from_entry,
    extract_archive_immutable,
    materialize_dataset,
    read_7z_member_bounded,
    validate_adapter_contract,
)


def _rule(pattern: str, template: str) -> dict:
    return {
        "regex": {
            "source": "relative_path",
            "pattern": pattern,
            "template": template,
        }
    }


def _write_detection_zip(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    image_path = source / "dataset/images/g1/frame001.jpg"
    label_path = source / "dataset/labels/g1/frame001.txt"
    image_path.parent.mkdir(parents=True)
    label_path.parent.mkdir(parents=True)
    Image.new("RGB", (16, 12), color=(20, 40, 60)).save(image_path)
    label_path.write_text("0 0.5 0.5 0.25 0.5\n", encoding="ascii")
    archive = tmp_path / "detection.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.write(image_path, "dataset/images/g1/frame001.jpg")
        handle.write(label_path, "dataset/labels/g1/frame001.txt")
    return archive


def _detection_contract() -> dict:
    return {
        "schema_version": 1,
        "kind": "yolo_detection",
        "archive_subdir": "dataset",
        "record_id_rule": _rule(
            r"images/(?P<group>[^/]+)/(?P<record>[^/]+)\.jpg",
            "{group}-{record}",
        ),
        "raw_group_rule": _rule(
            r"images/(?P<group>[^/]+)/(?P<record>[^/]+)\.jpg", "{group}"
        ),
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


def test_yolo_adapter_materializes_a_valid_manifest(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    frame = materialize_dataset(
        project_root=root,
        dataset_id="visual-fixture",
        archive_id="visual-fixture-v1",
        archive_path=_write_detection_zip(tmp_path),
        contract=_detection_contract(),
    )
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["record_id"] == "g1-frame001"
    assert row["raw_group_id"] == "g1"
    assert row["modality"] == "visible"
    assert (root / row["relative_path"]).is_file()
    summary = json.loads(row["label_summary_json"])
    assert summary["class_ids"] == ["person"]
    assert summary["image_width"] == 16
    assert summary["image_height"] == 12
    assert summary["boxes"][0]["source_class_id"] == 0


def test_filename_range_lookup_assigns_official_scenarios() -> None:
    contract = _detection_contract()
    contract["raw_group_rule"] = {
        "filename_range_lookup": {
            "source": "relative_path",
            "pattern": r"images/(?:train|val)/(?P<record>(?:[0-9]+|[pP][0-9]+-[0-9]+))[.]jpg",
            "record_group": "record",
            "ranges": [
                {"value": "scenario-01", "start": "0000001", "end": "0000852"},
                {"value": "scenario-02", "start": "p29-125", "end": "p30-912"},
            ],
        }
    }
    rule = validate_adapter_contract(contract)["raw_group_rule"]
    assert apply_rule(rule, relative_path="images/train/0000100.jpg") == "scenario-01"
    assert apply_rule(rule, relative_path="images/val/p30-1.jpg") == "scenario-02"
    with pytest.raises(AdapterContractError, match="matched 0 groups"):
        apply_rule(rule, relative_path="images/train/p99-1.jpg")


def test_filename_range_lookup_rejects_overlapping_intervals() -> None:
    contract = _detection_contract()
    contract["raw_group_rule"] = {
        "filename_range_lookup": {
            "pattern": r"images/train/(?P<record>[0-9]+)[.]jpg",
            "record_group": "record",
            "ranges": [
                {"value": "scenario-01", "start": "1", "end": "10"},
                {"value": "scenario-02", "start": "10", "end": "20"},
            ],
        }
    }
    with pytest.raises(AdapterContractError, match="overlapping ranges"):
        validate_adapter_contract(contract)


def test_yolo_adapter_requires_and_drops_reviewed_ambiguous_stems(tmp_path: Path) -> None:
    source = tmp_path / "ambiguous-source"
    for stem in ("0000001", "0000011"):
        image = source / f"dataset/images/train/{stem}.jpg"
        label = source / f"dataset/labels/train/{stem}.txt"
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 12), color=(20, 40, 60)).save(image)
        label.write_text("0 0.5 0.5 0.25 0.5\n", encoding="ascii")
    archive = tmp_path / "ambiguous.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for item in sorted(source.rglob("*")):
            if item.is_file():
                handle.write(item, item.relative_to(source).as_posix())
    contract = _detection_contract()
    contract["raw_group_rule"] = {"constant": "scenario-01"}
    contract["yolo"]["excluded_image_stems"] = ["0000011"]
    root = tmp_path / "ambiguous-project"
    root.mkdir()
    frame = materialize_dataset(
        project_root=root,
        dataset_id="ambiguous",
        archive_id="ambiguous-v1",
        archive_path=archive,
        contract=contract,
    )
    assert frame["record_id"].tolist() == ["train-0000001"]

    contract["yolo"]["excluded_image_stems"] = ["0000099"]
    missing_root = tmp_path / "missing-exclusion-project"
    missing_root.mkdir()
    with pytest.raises(AdapterContractError, match="did not observe reviewed excluded"):
        materialize_dataset(
            project_root=missing_root,
            dataset_id="missing-exclusion",
            archive_id="missing-exclusion-v1",
            archive_path=archive,
            contract=contract,
        )


def test_yolo_adapter_merges_separately_hashed_image_and_label_archives(
    tmp_path: Path,
) -> None:
    source = tmp_path / "split-source"
    image_path = source / "dataset/images/g1/frame001.jpg"
    label_path = source / "dataset/labels/g1/frame001.txt"
    image_path.parent.mkdir(parents=True)
    label_path.parent.mkdir(parents=True)
    Image.new("RGB", (16, 12), color=(20, 40, 60)).save(image_path)
    label_path.write_text("0 0.5 0.5 0.25 0.5\n", encoding="ascii")
    images = tmp_path / "images.zip"
    labels = tmp_path / "labels.zip"
    with zipfile.ZipFile(images, "w") as handle:
        handle.write(image_path, "dataset/images/g1/frame001.jpg")
    with zipfile.ZipFile(labels, "w") as handle:
        handle.write(label_path, "dataset/labels/g1/frame001.txt")
    root = tmp_path / "project"
    root.mkdir()
    frame = materialize_dataset(
        project_root=root,
        dataset_id="multi-archive-fixture",
        archive_id="multi-archive-fixture-v1",
        archive_paths=[images, labels],
        contract=_detection_contract(),
    )
    assert len(frame) == 1
    assert json.loads(frame.iloc[0]["label_summary_json"])["class_ids"] == ["person"]


def test_coco_adapter_filters_classes_and_records_reviewed_drops(tmp_path: Path) -> None:
    source = tmp_path / "coco-source"
    for name in ("a.jpg", "b.jpg"):
        path = source / "train2017" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (20, 10), color=(20, 40, 60)).save(path)
    payload = {
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 20, "height": 10, "license": 4},
            {"id": 2, "file_name": "b.jpg", "width": 20, "height": 10, "license": 3},
        ],
        "categories": [{"id": 1, "name": "person"}, {"id": 3, "name": "car"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [2, 1, 4, 5]},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [0, 0, 0, 2]},
            {
                "id": 3,
                "image_id": 1,
                "category_id": 1,
                "bbox": [1, 1, 2, 2],
                "iscrowd": 1,
            },
            {"id": 4, "image_id": 1, "category_id": 3, "bbox": [1, 1, 2, 2]},
            {"id": 5, "image_id": 2, "category_id": 3, "bbox": [1, 1, 2, 2]},
            {"id": 6, "image_id": 2, "category_id": 1, "bbox": [1, 1, 2, 2]},
        ],
    }
    annotation = source / "annotations/instances_train2017.json"
    annotation.parent.mkdir(parents=True)
    annotation.write_text(json.dumps(payload), encoding="utf-8")
    images = tmp_path / "coco-images.zip"
    annotations = tmp_path / "coco-annotations.zip"
    with zipfile.ZipFile(images, "w") as handle:
        for path in sorted((source / "train2017").iterdir()):
            handle.write(path, f"train2017/{path.name}")
    with zipfile.ZipFile(annotations, "w") as handle:
        handle.write(annotation, "annotations/instances_train2017.json")
    contract = {
        "schema_version": 1,
        "kind": "coco_detection",
        "archive_subdir": ".",
        "record_id_rule": _rule(r"train2017/(?P<record>[^/]+)[.]jpg", "{record}"),
        "raw_group_rule": {"constant": "coco-train2017"},
        "modality_rule": {"constant": "visible"},
        "pair_id_rule": {"null": True},
        "sequence_id_rule": {"null": True},
        "timestamp_rule": {"null": True},
        "coco": {
            "annotation_file": "annotations/instances_train2017.json",
            "image_root": "train2017",
            "class_names": ["person"],
            "invalid_bbox_policy": "drop",
            "iscrowd_policy": "drop",
            "image_selection": "selected_category_presence",
            "allowed_license_ids": [4],
        },
    }
    root = tmp_path / "coco-project"
    root.mkdir()
    frame = materialize_dataset(
        project_root=root,
        dataset_id="coco-person",
        archive_id="coco-person-v1",
        archive_paths=[images, annotations],
        contract=contract,
    )
    assert len(frame) == 1
    summary = json.loads(frame.iloc[0]["label_summary_json"])
    assert summary["class_ids"] == ["person"]
    assert len(summary["boxes"]) == 1
    assert summary["dropped_invalid_bbox_count"] == 1
    assert summary["dropped_iscrowd_count"] == 1


def test_voc_adapter_filters_classes_and_records_reviewed_drops(tmp_path: Path) -> None:
    source = tmp_path / "voc-source/dataset"
    image = source / "JPEGImages/1.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (20, 10), color=(20, 40, 60)).save(image)
    annotation = source / "Annotations/1.xml"
    annotation.parent.mkdir(parents=True)
    annotation.write_text(
        """<annotation>
<filename>C:\\released-dataset\\JPEGImages\\1.jpg</filename><size><width>20</width><height>10</height></size>
<object><name>coal_miner</name><difficult>0</difficult><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>10</xmax><ymax>9</ymax></bndbox></object>
<object><name>coal_miner</name><difficult>0</difficult><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>1</xmax><ymax>2</ymax></bndbox></object>
<object><name>machinery</name><difficult>0</difficult><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>3</xmax><ymax>3</ymax></bndbox></object>
<object><name>coal_miner</name><difficult>1</difficult><bndbox><xmin>2</xmin><ymin>2</ymin><xmax>4</xmax><ymax>4</ymax></bndbox></object>
</annotation>""",
        encoding="utf-8",
    )
    archive = tmp_path / "voc.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.write(image, "dataset/JPEGImages/1.jpg")
        handle.write(annotation, "dataset/Annotations/1.xml")
    contract = {
        "schema_version": 1,
        "kind": "voc_detection",
        "archive_subdir": "dataset",
        "record_id_rule": _rule(r"JPEGImages/(?P<record>[^/]+)[.]jpg", "{record}"),
        "raw_group_rule": {"constant": "scene-1"},
        "modality_rule": {"constant": "visible"},
        "pair_id_rule": {"null": True},
        "sequence_id_rule": {"null": True},
        "timestamp_rule": {"null": True},
        "voc": {
            "image_globs": ["JPEGImages/*.jpg"],
            "image_root": "JPEGImages",
            "annotation_root": "Annotations",
            "class_names": ["coal_miner"],
            "missing_annotation_policy": "error",
            "invalid_bbox_policy": "drop",
            "difficult_policy": "drop",
            "image_selection": "all",
            "coordinate_convention": "voc_xyxy",
        },
    }
    root = tmp_path / "voc-project"
    root.mkdir()
    frame = materialize_dataset(
        project_root=root,
        dataset_id="voc-coal",
        archive_id="voc-coal-v1",
        archive_path=archive,
        contract=contract,
    )
    assert len(frame) == 1
    summary = json.loads(frame.iloc[0]["label_summary_json"])
    assert summary["class_ids"] == ["coal_miner"]
    assert len(summary["boxes"]) == 1
    assert summary["dropped_invalid_bbox_count"] == 1
    assert summary["dropped_difficult_count"] == 1
    assert summary["ignored_unselected_class_count"] == 1


def test_voc_adapter_supports_paired_modalities_with_shared_flat_annotations(
    tmp_path: Path,
) -> None:
    source = tmp_path / "llvip-source/LLVIP"
    annotation_root = source / "Annotations"
    annotation_root.mkdir(parents=True)
    for split, record in (("train", "010001"), ("test", "020001")):
        for modality, color in (("visible", (20, 40, 60)), ("infrared", (40, 40, 40))):
            image = source / modality / split / f"{record}.jpg"
            image.parent.mkdir(parents=True)
            Image.new("RGB", (20, 10), color=color).save(image)
        (annotation_root / f"{record}.xml").write_text(
            f"""<annotation>
<filename>{record}.jpg</filename><size><width>20</width><height>10</height></size>
<object><name>person</name><difficult>0</difficult><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>10</xmax><ymax>9</ymax></bndbox></object>
</annotation>""",
            encoding="utf-8",
        )
    archive = tmp_path / "llvip.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for item in sorted(source.rglob("*")):
            if item.is_file():
                handle.write(item, item.relative_to(source.parent).as_posix())
    contract = {
        "schema_version": 1,
        "kind": "voc_detection",
        "archive_subdir": ".",
        "record_id_rule": _rule(
            r"LLVIP/(?P<modality>visible|infrared)/(?P<split>train|test)/(?P<record>[0-9]{6})[.]jpg",
            "{modality}-{record}",
        ),
        "raw_group_rule": _rule(
            r"LLVIP/(?:visible|infrared)/(?:train|test)/(?P<group>[0-9]{2})[0-9]{4}[.]jpg",
            "capture-{group}",
        ),
        "modality_rule": _rule(
            r"LLVIP/(?P<modality>visible|infrared)/(?:train|test)/[0-9]{6}[.]jpg",
            "{modality}",
        ),
        "pair_id_rule": _rule(
            r"LLVIP/(?:visible|infrared)/(?:train|test)/(?P<record>[0-9]{6})[.]jpg",
            "pair-{record}",
        ),
        "sequence_id_rule": _rule(
            r"LLVIP/(?:visible|infrared)/(?:train|test)/(?P<group>[0-9]{2})[0-9]{4}[.]jpg",
            "capture-{group}",
        ),
        "timestamp_rule": {"null": True},
        "voc": {
            "image_globs": ["LLVIP/visible/**/*.jpg", "LLVIP/infrared/**/*.jpg"],
            "image_root": "LLVIP",
            "annotation_root": "LLVIP/Annotations",
            "annotation_path_rule": "image_stem",
            "class_names": ["person"],
            "missing_annotation_policy": "error",
            "invalid_bbox_policy": "drop",
            "difficult_policy": "drop",
            "image_selection": "all",
            "coordinate_convention": "voc_xyxy",
        },
    }
    root = tmp_path / "llvip-project"
    root.mkdir()
    frame = materialize_dataset(
        project_root=root,
        dataset_id="llvip",
        archive_id="llvip-v1",
        archive_path=archive,
        contract=contract,
    )

    assert len(frame) == 4
    assert set(frame["modality"]) == {"visible", "infrared"}
    assert frame.groupby("pair_id")["modality"].nunique().to_dict() == {
        "pair-010001": 2,
        "pair-020001": 2,
    }
    assert set(frame["raw_group_id"]) == {"capture-01", "capture-02"}


def test_archive_traversal_and_unreviewed_adapter_are_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.txt", b"forbidden")
    with pytest.raises(AdapterContractError, match="safe relative path"):
        extract_archive_immutable(archive, tmp_path / "canonical")
    with pytest.raises(AdapterContractError, match="lacks a reviewed adapter_contract"):
        adapter_contract_from_entry({"dataset_id": "missing"})


def test_7z_archive_is_listed_read_and_extracted_with_bounded_members(
    tmp_path: Path,
) -> None:
    try:
        executable = _seven_zip_executable()
    except AdapterContractError:
        pytest.skip("7z executable is unavailable")
    source = tmp_path / "seven-source"
    member = source / "nested/sample.txt"
    member.parent.mkdir(parents=True)
    member.write_bytes(b"bounded-seven-zip\n")
    archive = tmp_path / "sample.7z"
    created = subprocess.run(
        [executable, "a", "-bd", "-bb0", str(archive), str(source / "*")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert created.returncode == 0
    assert read_7z_member_bounded(
        archive, "nested/sample.txt", max_bytes=1024
    ) == b"bounded-seven-zip\n"
    with pytest.raises(AdapterContractError, match="exceeds 4 bytes"):
        read_7z_member_bounded(archive, "nested/sample.txt", max_bytes=4)
    extracted = extract_archive_immutable(archive, tmp_path / "seven-extracted")
    assert (extracted / "nested/sample.txt").read_bytes() == b"bounded-seven-zip\n"


def test_methane_adapter_writes_deterministic_chronological_blocks(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "methane.csv"
    pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01T00:00:00",
                "2026-01-01T00:30:00",
                "2026-01-01T01:00:00",
            ],
            "sensor": ["S-1", "S-1", "S-1"],
            "value": [0.1, 0.2, 0.3],
            "temperature": [20.0, 20.1, 20.2],
        }
    ).to_csv(csv_path, index=False)
    archive = tmp_path / "methane.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.write(csv_path, "source/methane.csv")
    contract = {
        "schema_version": 1,
        "kind": "methane_csv",
        "archive_subdir": "source",
        "methane": {
            "csv_globs": ["*.csv"],
            "timestamp_column": "timestamp",
            "value_column": "value",
            "sensor_group_column": "sensor",
            "feature_columns": ["temperature"],
            "group_duration_seconds": 3600,
            "timezone": "UTC",
        },
    }
    assert validate_adapter_contract(contract)["kind"] == "methane_csv"
    root = tmp_path / "project"
    root.mkdir()
    frame = materialize_dataset(
        project_root=root,
        dataset_id="methane-fixture",
        archive_id="methane-fixture-v1",
        archive_path=archive,
        contract=contract,
    )
    assert len(frame) == 2
    assert frame["raw_group_id"].nunique() == 2
    assert set(frame["modality"]) == {"methane"}


def test_wide_methane_adapter_streams_target_sensor_blocks(tmp_path: Path) -> None:
    archive = tmp_path / "wide-methane.zip"
    wide = (
        "year,month,day,hour,minute,second,MM263,MM264,MM256,TEMP\n"
        "2026,1,1,0,0,0,0.2,0.3,0.4,20.0\n"
        "2026,1,1,0,0,1,0.2,0.3,0.4,20.1\n"
        "2026,1,1,1,0,0,1.2,1.3,1.4,20.2\n"
    )
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("methane_data.csv", wide)
    contract = {
        "schema_version": 1,
        "kind": "methane_csv",
        "archive_subdir": ".",
        "methane": {
            "input_layout": "wide",
            "csv_globs": ["*.csv"],
            "timestamp_column": "timestamp",
            "timestamp_components": {
                "year": "year",
                "month": "month",
                "day": "day",
                "hour": "hour",
                "minute": "minute",
                "second": "second",
            },
            "value_column": "target_value",
            "sensor_group_column": "sensor_group_id",
            "target_sensor_columns": ["MM263", "MM264", "MM256"],
            "feature_columns": ["MM263", "MM264", "MM256", "TEMP"],
            "group_duration_seconds": 3600,
            "timezone": "UTC",
            "chunk_rows": 2,
            "target_missing_policy": "drop",
        },
    }
    root = tmp_path / "project"
    root.mkdir()
    frame = materialize_dataset(
        project_root=root,
        dataset_id="wide-methane",
        archive_id="wide-methane-v1",
        archive_path=archive,
        contract=contract,
    )
    assert len(frame) == 6
    assert frame["raw_group_id"].nunique() == 6
    first = pd.read_parquet(root / frame.sort_values("record_id").iloc[0]["relative_path"])
    assert {"__timestamp_utc", "sensor_group_id", "target_value", "TEMP"}.issubset(
        first.columns
    )
    assert first["sensor_group_id"].nunique() == 1
    for relative in frame["relative_path"]:
        block = pd.read_parquet(root / relative)
        assert "__timestamp_utc" in block.columns
        assert block["__timestamp_utc"].is_monotonic_increasing
