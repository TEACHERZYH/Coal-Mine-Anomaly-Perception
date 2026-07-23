from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle

import pandas as pd
from PIL import Image
import torch
import yaml

from mining1_exp.predict.detection import predict_detection_package
from mining1_exp.models.yolo_adapter import BinaryProbabilityCalibrator
from mining1_exp.provenance import sha256_file
from mining1_exp.robustness import build_locked_corruptions, predict_r1_sealed
from mining1_exp.workflow_governance import (
    create_branch_test_seal,
    finalize_branch_test_seal,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Boxes:
    def __init__(self, class_index: int) -> None:
        self.xyxy = torch.tensor([[1.0, 1.0, 7.0, 7.0]])
        self.conf = torch.tensor([0.9])
        self.cls = torch.tensor([float(class_index)])

    def __len__(self) -> int:
        return 1


class _Result:
    def __init__(self, class_index: int) -> None:
        self.boxes = _Boxes(class_index)


class _FakeYolo:
    def __init__(self, checkpoint: str) -> None:
        self.checkpoint = checkpoint

    def predict(self, *, source: str, **_: object) -> list[_Result]:
        class_index = 0 if "worker" in Path(source).stem else 1
        return [_Result(class_index)]


def _prepare_project(root: Path) -> Path:
    for relative in (
        "configs",
        "data/locked",
        "data/sealed",
        "data/seals",
        "evidence/data",
        "runs/T1/visible",
        "runs/T1/thermal",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    protocol = {
        "models": {"visible": {"input_size": 64}},
        "evaluation": {
            "calibration_group_cv_folds": 2,
            "calibration_candidates": ["temperature"],
            "robustness": {
                "corruption_types": ["low_light", "dust_fog_proxy"],
                "severity_levels": [1, 2, 3],
            },
        },
    }
    (root / "configs/protocol_lock.pretest.yaml").write_text(
        yaml.safe_dump(protocol), encoding="utf-8"
    )
    (root / "configs/experiment_matrix.template.csv").write_text(
        "family_id\nT1-VIS\nT1-THERM\n", encoding="utf-8"
    )
    (root / "configs/artifact_contract.template.yaml").write_text(
        "schema_version: 1\n", encoding="utf-8"
    )
    (root / "evidence/data/dataset_source_decision.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "roles": {
                    "primary_rgbt_dataset": {"dataset_id": "rgbt"},
                    "primary_visual_target": {"dataset_id": "rgbt"},
                },
            }
        ),
        encoding="utf-8",
    )
    ontology_entries = []
    for source_label, concept in (
        ("person", "worker_presence"),
        ("helmet", "helmet_presence"),
    ):
        ontology_entries.append(
            {
                "dataset_id": "rgbt",
                "source_label": source_label,
                "canonical_concept_id": concept,
                "mapping_status": "compatible",
                "annotation_policy": "fixture boxes are exhaustive",
                "event_semantics": "observable_presence",
                "negative_semantics": "exhaustive_verified_absence",
                "allowed_tasks": ["confirmatory_f1"],
                "evidence_reference": "fixture:rgbt",
                "reviewer_decision": "accept for contract test",
            }
        )
    (root / "data/locked/ontology_lock.yaml").write_text(
        yaml.safe_dump({"ontology_version": "fixture-v1", "entries": ontology_entries}),
        encoding="utf-8",
    )
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
    files = []
    splits = []
    image_root = root / "data/canonical/rgbt/images"
    image_root.mkdir(parents=True)
    index = 0
    for _, source_modality in (("visible", "visible"), ("thermal", "infrared")):
        for pool, group_count in (("D_b_prob", 4), ("D_b_te", 2)):
            for group in range(group_count):
                for token, source_label in (("worker", "person"), ("helmet", "helmet")):
                    image = image_root / f"{source_modality}-{pool}-{group}-{token}.png"
                    Image.new("RGB", (8, 8), color=(index % 255, 20, 30)).save(image)
                    record_id = f"record-{index}"
                    raw_group_id = f"{source_modality}-{pool}-group-{group}"
                    files.append(
                        {
                            "dataset_id": "rgbt",
                            "record_id": record_id,
                            "archive_id": "fixture-v1",
                            "relative_path": image.relative_to(root).as_posix(),
                            "modality": source_modality,
                            "raw_group_id": raw_group_id,
                            "pair_id": f"pair-{index}",
                            "sequence_id": None,
                            "timestamp_or_order": index,
                            "label_summary_json": json.dumps(
                                {
                                    "negative_annotation_verified": True,
                                    "image_width": 8,
                                    "image_height": 8,
                                    "boxes": [
                                        {
                                            "source_label": source_label,
                                            "x_center_normalized": 0.5,
                                            "y_center_normalized": 0.5,
                                            "width_normalized": 0.5,
                                            "height_normalized": 0.5,
                                        }
                                    ],
                                },
                                sort_keys=True,
                            ),
                            "byte_size": image.stat().st_size,
                            "sha256": _sha(image),
                        }
                    )
                    splits.append(
                        {
                            "dataset_id": "rgbt",
                            "record_id": record_id,
                            "raw_group_id": raw_group_id,
                            "pool": pool,
                            "split_seed": 13007,
                            "split_version": "fixture-v1",
                            "ontology_hash": "a" * 64,
                            "dedup_report_hash": "b" * 64,
                        }
                    )
                    index += 1
    pd.DataFrame(files).to_parquet(root / "data/locked/file_manifest.parquet", index=False)
    pd.DataFrame(splits).to_parquet(root / "data/locked/split_manifest.parquet", index=False)
    pd.DataFrame(
        [
            {
                "dataset_id": "methane",
                "window_id": "methane-test-0",
                "raw_group_id": "methane-test-group",
                "sensor_group_id": "MM263",
                "pool": "D_b_te",
                "history_end": pd.Timestamp("2026-01-01T00:00:00Z"),
                "history_json": json.dumps({"feature_names": ["ch4"], "values": [[0.2]]}),
            }
        ]
    ).to_parquet(root / "data/locked/branch_methane_test_features.parquet", index=False)
    pd.DataFrame(
        [
            {
                "dataset_id": "methane",
                "window_id": "methane-test-0",
                "raw_group_id": "methane-test-group",
                "sensor_group_id": "MM263",
                "pool": "D_b_te",
                "timestamp_seconds": 0,
                "raw_value": 0.2,
                "event_truth": 0,
            }
        ]
    ).to_parquet(root / "data/sealed/branch_methane_truth.parquet", index=False)
    create_branch_test_seal(root, {"step_id": "E060"}, {"candidate": True})
    finalize_branch_test_seal(
        root,
        {"step_id": "E061"},
        {"protocol": "configs/protocol_lock.pretest.yaml"},
    )
    for family, modality in (("T1-VIS", "visible"), ("T1-THERM", "thermal")):
        run = root / f"runs/T1/{modality}"
        checkpoint = run / "best.pt"
        checkpoint.write_bytes(family.encode("ascii"))
        (run / "run_manifest.json").write_text(
            json.dumps(
                {
                    "status": "pass",
                    "run_id": f"run-{modality}",
                    "family_id": family,
                    "modality": modality,
                    "checkpoint_path": checkpoint.relative_to(root).as_posix(),
                    "checkpoint_sha256": sha256_file(checkpoint),
                }
            ),
            encoding="utf-8",
        )
    run_root = root / "job"
    run_root.mkdir()
    (run_root / "slurm_environment_probe.json").write_text("{}", encoding="utf-8")
    return run_root


def test_detection_prediction_is_complete_and_truth_free(tmp_path: Path) -> None:
    run_root = _prepare_project(tmp_path)
    result = predict_detection_package(
        tmp_path,
        run_root,
        "T1",
        yolo_factory=_FakeYolo,
    )
    detections = pd.read_parquet(tmp_path / "predictions/locked/T1/detections.parquet")
    concepts = pd.read_parquet(tmp_path / "predictions/locked/T1/concepts.parquet")
    lock = json.loads(
        (tmp_path / "predictions/locked/T1/prediction_lock.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "pass"
    assert detections["run_id"].nunique() == 2
    assert len(concepts) == 2 * 2 * 2 * 2
    assert not concepts.duplicated(["run_id", "record_id", "concept_id", "modality"]).any()
    assert set(lock["required_prediction_families"]) == {"T1-VIS", "T1-THERM"}
    features = pd.read_parquet(tmp_path / "data/locked/branch_test_features.parquet")
    assert "label_summary_json" not in features.columns
    assert len(pd.read_parquet(tmp_path / "data/sealed/branch_detection_truth.parquet")) == 8
    assert len(pd.read_parquet(tmp_path / "data/sealed/branch_concept_truth.parquet")) == 16
    assert not any("truth" in column or "label" in column for column in detections.columns)
    assert not any("truth" in column or "label" in column for column in concepts.columns)


def test_corruption_builder_closes_locked_two_by_three_grid(tmp_path: Path) -> None:
    run_root = _prepare_project(tmp_path)
    result = build_locked_corruptions(tmp_path, run_root, None)
    manifest = pd.read_parquet(tmp_path / "data/locked/corruption_manifest.parquet")
    assert result["status"] == "pass"
    assert len(manifest) == 4 * 2 * 3
    assert set(manifest["corruption_type"]) == {"low_light", "dust_fog_proxy"}
    assert set(manifest["severity"]) == {1, 2, 3}
    assert manifest["label_independent"].all()


def _prepare_v2_r1_inputs(root: Path) -> None:
    selected = []
    calibrator_store = {}
    seeds = (1701, 2903, 4219)
    subset_seeds = (5171, 6197, 7331)
    calibrators = {
        concept: BinaryProbabilityCalibrator(
            "temperature", fitted_pool="D_b_prob", _model=1.0
        )
        for concept in ("worker_presence", "helmet_presence")
    }
    calibrator_payload = pickle.dumps({"calibrators": calibrators}, protocol=4)
    for family in (
        "V2-A-10-SCR",
        "V2-A-10-GEN",
        "V2-A-10-SINGLE",
        "V2-A-10-MULTI",
    ):
        for seed, subset_seed in zip(seeds, subset_seeds):
            run = root / f"runs/V2/finetune/{family}/seed-{seed}-subset-{subset_seed}"
            run.mkdir(parents=True, exist_ok=True)
            checkpoint = run / "best.pt"
            checkpoint.write_bytes(f"{family}:{seed}:{subset_seed}".encode("ascii"))
            selected.append(
                {
                    "family_id": family,
                    "train_seed": seed,
                    "subset_seed": subset_seed,
                    "checkpoint_path": checkpoint.relative_to(root).as_posix(),
                    "checkpoint_sha256": sha256_file(checkpoint),
                }
            )
            if family in {"V2-A-10-GEN", "V2-A-10-MULTI"}:
                lock_id = f"{family}-seed-{seed}-subset-{subset_seed}"
                calibrator_store[lock_id] = calibrator_payload
    selection_path = root / "runs/V2/checkpoint_selection.json"
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(
        json.dumps({"status": "pass", "selected": selected}), encoding="utf-8"
    )
    store_path = root / "predictions/locked/V2/calibrators.pkl"
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store_path.write_bytes(pickle.dumps(calibrator_store, protocol=4))


def test_r1_prediction_reuses_six_locked_v2_runs_without_truth(tmp_path: Path) -> None:
    run_root = _prepare_project(tmp_path)
    build_locked_corruptions(tmp_path, run_root, None)
    _prepare_v2_r1_inputs(tmp_path)
    result = predict_r1_sealed(tmp_path, run_root, None, yolo_factory=_FakeYolo)
    predictions = pd.read_parquet(tmp_path / "predictions/locked/R1.parquet")
    lock = json.loads(
        (tmp_path / "predictions/locked/R1/prediction_lock.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["status"] == "pass"
    assert predictions["run_id"].nunique() == 6
    assert set(predictions["family_id"]) == {"R1-GEN", "R1-MULTI"}
    assert set(predictions["corruption_type"]) == {"low_light", "dust_fog_proxy"}
    assert set(predictions["severity"]) == {1, 2, 3}
    assert len(lock["required_prediction_families"]) == 6
    assert not any("truth" in column or "label" in column for column in predictions.columns)
