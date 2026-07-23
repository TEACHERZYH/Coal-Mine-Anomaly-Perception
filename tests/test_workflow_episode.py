from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from mining1_exp import workflow_data, workflow_episode
from mining1_exp.governance.prediction_lock import (
    build_prediction_lock,
    close_prediction_lock,
)
from mining1_exp.episode_data import load_episode_arrays
from mining1_exp.models.yolo_adapter import BinaryProbabilityCalibrator
from mining1_exp.predict.episode import predict_episode_branches
from mining1_exp.provenance import sha256_file
from mining1_exp.review_episode_skeletons import review_e058
from mining1_exp.train.episode import (
    infer_episode_model,
    load_episode_model,
    train_episode_run,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _Boxes:
    def __init__(self) -> None:
        self.xyxy = torch.tensor([[1.0, 1.0, 7.0, 7.0]])
        self.conf = torch.tensor([0.8])
        self.cls = torch.tensor([0.0])

    def __len__(self) -> int:
        return 1


class _Result:
    def __init__(self) -> None:
        self.boxes = _Boxes()


class _FakeYolo:
    def __init__(self, checkpoint: str) -> None:
        self.checkpoint = checkpoint

    def predict(self, **_: object) -> list[_Result]:
        return [_Result()]


def _manifest_row(index: int, modality: str) -> dict:
    group = f"group-{index:04d}"
    record = f"rgbt-{modality}-{index:04d}"
    return {
        "dataset_id": "rgbt",
        "record_id": record,
        "archive_id": "rgbt-archive",
        "relative_path": f"rgbt/{record}.dat",
        "modality": modality,
        "raw_group_id": group,
        "pair_id": f"pair-{index:04d}",
        "sequence_id": group,
        "timestamp_or_order": str(index),
        "label_summary_json": json.dumps(
            {"class_ids": ["worker_presence"] if index % 2 == 0 else []}, sort_keys=True
        ),
        "byte_size": 1,
        "sha256": hashlib.sha256(record.encode("utf-8")).hexdigest(),
    }


def _prepare_root(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    for relative in ("configs", "data/locked", "evidence/data"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    for relative in (
        "configs/protocol_lock.template.yaml",
        "configs/artifact_contract.template.yaml",
        "configs/experiment_matrix.template.csv",
    ):
        target = root / relative
        target.write_bytes((PROJECT_ROOT / relative).read_bytes())
    rows = [
        _manifest_row(index, modality)
        for index in range(400)
        for modality in ("visible", "thermal")
    ]
    for cohort in range(8):
        timestamp = (pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=cohort)).isoformat()
        record = f"methane-{cohort:02d}"
        rows.append(
            {
                "dataset_id": "methane",
                "record_id": record,
                "archive_id": "methane-archive",
                "relative_path": f"methane/{record}.parquet",
                "modality": "methane",
                "raw_group_id": f"methane-group-{cohort:02d}",
                "pair_id": "",
                "sequence_id": "MM263",
                "timestamp_or_order": timestamp,
                "label_summary_json": json.dumps({"class_ids": []}, sort_keys=True),
                "byte_size": 1,
                "sha256": hashlib.sha256(record.encode("utf-8")).hexdigest(),
            }
        )
    pd.DataFrame(rows).to_parquet(
        root / "data/locked/file_manifest.parquet", index=False
    )
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
    ontology = {
        "ontology_version": "episode-fixture-v1",
        "entries": [
            {
                "dataset_id": "rgbt",
                "source_label": "worker_presence",
                "canonical_concept_id": "worker_presence",
                "mapping_status": "compatible",
                "annotation_policy": "fixture boxes are exhaustive",
                "event_semantics": "observable_presence",
                "negative_semantics": "exhaustive_verified_absence",
                "allowed_tasks": ["confirmatory_f1"],
                "evidence_reference": "fixture:rgbt",
                "reviewer_decision": "accept for contract test",
            },
            *[
                {
                    "dataset_id": dataset_id,
                    "source_label": "worker_presence",
                    "canonical_concept_id": "worker_presence",
                    "mapping_status": "compatible",
                    "annotation_policy": "fixture boxes are exhaustive",
                    "event_semantics": "observable_presence",
                    "negative_semantics": "exhaustive_verified_absence",
                    "allowed_tasks": ["confirmatory_f1"],
                    "evidence_reference": f"fixture:{dataset_id}",
                    "reviewer_decision": "accept for contract test",
                }
                for dataset_id in ("source-a", "source-b", "methane")
            ],
        ],
    }
    (root / "evidence/data/ontology_mapping.reviewed.yaml").write_text(
        yaml.safe_dump(ontology, sort_keys=True), encoding="utf-8"
    )
    (root / "evidence/data/dataset_source_decision.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "roles": {
                    "primary_visual_target": {"dataset_id": "rgbt"},
                    "primary_visual_sources": [
                        {"dataset_id": "source-a"},
                        {"dataset_id": "source-b"},
                    ],
                    "primary_rgbt_dataset": {"dataset_id": "rgbt"},
                    "methane_dataset": {
                        "dataset_id": "methane",
                        "adapter_contract": {
                            "methane": {"group_duration_seconds": 86400}
                        },
                    },
                },
                "visual_direction_id": "fixture",
            }
        ),
        encoding="utf-8",
    )
    workflow_data.lock_ontology(root, {"step_id": "E030"}, {})
    workflow_data.create_splits(root, {"step_id": "E038"}, {"grouped": True})
    graph = {
        "schema_version": 1,
        "status": "pass",
        "graph_eligible": True,
        "compatible_concept_ids": ["worker_presence"],
    }
    (root / "data/locked/graph_eligibility_lock.json").write_text(
        json.dumps(graph, sort_keys=True), encoding="utf-8"
    )
    for name in ("rgbt_input_lock.json", "methane_role_lock.json"):
        (root / "data/locked" / name).write_text(
            json.dumps({"status": "pass"}, sort_keys=True), encoding="utf-8"
        )
    return root


def _write_branch_predictions(root: Path) -> None:
    skeleton = pd.read_parquet(
        root / "data/locked/episode_skeleton_manifest.parquet"
    )
    modalities = {
        "T1-VIS": "visible",
        "T1-THERM": "thermal",
        "S1-GRU": "methane",
        "V2-A-10-MULTI": "visible",
    }
    frame = skeleton[
        [
            "skeleton_item_id",
            "record_id",
            "concept_id",
            "node_id",
            "generator_family_id",
            "pool",
        ]
    ].copy()
    frame["modality"] = frame["generator_family_id"].map(modalities)
    frame["step_score_raw"] = 0.75
    frame["calibrated_probability"] = 0.70
    frame["available"] = True
    frame["model_hash"] = "a" * 64
    frame["calibrator_or_policy_hash"] = "b" * 64
    frame["skeleton_manifest_hash"] = sha256_file(
        root / "data/locked/episode_skeleton_manifest.parquet"
    )
    frame["skeleton_inference_projection_hash"] = sha256_file(
        root / "data/seals/episode_skeleton_inference_projection.parquet"
    )
    output = root / "predictions/episode_branches"
    output.mkdir(parents=True)
    for pool, group in frame.groupby("pool"):
        group.drop(columns=["pool"]).to_parquet(output / f"{pool}.parquet", index=False)


def _write_t1_generator_locks(root: Path) -> None:
    calibrator = BinaryProbabilityCalibrator("temperature").fit(
        [0.1, 0.9],
        [0, 1],
        pool="D_b_prob",
        negatives_verified=True,
    )
    model_hashes = {}
    policy_hashes = {}
    payloads = {}
    lock_ids = []
    for family, modality in (("T1-VIS", "visible"), ("T1-THERM", "thermal")):
        run = root / f"runs/T1/{family}/seed-1701"
        run.mkdir(parents=True, exist_ok=True)
        checkpoint = run / "best.pt"
        checkpoint.write_bytes(family.encode("utf-8"))
        checkpoint_hash = sha256_file(checkpoint)
        manifest = {
            "status": "pass",
            "run_id": f"run-{family}",
            "family_id": family,
            "train_seed": 1701,
            "subset_seed": None,
            "modality": modality,
            "selection_pool": "D_b_sel",
            "selection_metric": "map50_95",
            "selection_metric_value": 0.5,
            "checkpoint_path": checkpoint.relative_to(root).as_posix(),
            "checkpoint_sha256": checkpoint_hash,
        }
        (run / "run_manifest.json").write_text(
            json.dumps(manifest, sort_keys=True), encoding="utf-8"
        )
        lock_id = f"{family}-seed-1701"
        lock_ids.append(lock_id)
        model_hashes[lock_id] = checkpoint_hash
        policy_hashes[lock_id] = hashlib.sha256(
            f"policy-{family}".encode("utf-8")
        ).hexdigest()
        payloads[lock_id] = pickle.dumps(
            {"schema_version": 1, "calibrators": {"worker_presence": calibrator}},
            protocol=4,
        )
    output_root = root / "predictions/locked/T1"
    output_root.mkdir(parents=True, exist_ok=True)
    dummy = output_root / "concepts.parquet"
    pd.DataFrame({"record_id": ["dummy"], "score_calibrated": [0.5]}).to_parquet(
        dummy, index=False
    )
    store = output_root / "calibrators.pkl"
    store.write_bytes(pickle.dumps(payloads, protocol=4))
    lock = build_prediction_lock(
        scope="branch",
        seal_hash="a" * 64,
        protocol_hash="b" * 64,
        source_hash="c" * 64,
        matrix_hash="d" * 64,
        required_prediction_families=lock_ids,
        prediction_paths={lock_id: dummy for lock_id in lock_ids},
        model_hashes=model_hashes,
        calibrator_and_policy_hashes=policy_hashes,
    )
    lock["calibrator_store_path"] = store.relative_to(root).as_posix()
    lock["calibrator_store_sha256"] = sha256_file(store)
    close_prediction_lock(output_root / "prediction_lock.json", lock)


def _write_e058_receipt(root: Path, built: dict) -> None:
    receipt_path = root / "evidence/command_receipts/E058.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for relative in built["output_paths"]:
        path = root / relative
        outputs.append(
            {
                "path": relative,
                "kind": "file",
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "step_id": "E058",
                "status": "pass",
                "command": "build-episode-skeletons",
                "arguments": {},
                "inputs": built["inputs"],
                "outputs": outputs,
                "details": built["details"],
            }
        ),
        encoding="utf-8",
    )


def test_episode_skeleton_features_and_seal_are_separated(tmp_path: Path) -> None:
    root = _prepare_root(tmp_path)
    built = workflow_episode.build_episode_skeletons(
        root, {"step_id": "E058"}, {}
    )
    _write_e058_receipt(root, built)
    independent_review = review_e058(root)
    assert independent_review["status"] == "pass"
    assert independent_review["error_count"] == 0
    assert built["details"]["fusion_eligible_concept_count"] == 1
    eligibility = pd.read_parquet(
        root / "data/locked/fusion_eligibility_lock.parquet"
    )
    assert {
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
    }.issubset(eligibility.columns)
    assert "fusion_eligible" not in eligibility.columns
    assert "graph_eligible" not in eligibility.columns
    assert eligibility["fusion_primary_eligible"].astype(bool).all()
    skeleton = pd.read_parquet(
        root / "data/locked/episode_skeleton_manifest.parquet"
    )
    assert skeleton["skeleton_item_id"].is_unique
    assert set(skeleton["pool"]) == {"D_e_tr", "D_e_sel", "D_e_pol", "D_e_te"}
    assert set(skeleton["event_position_role"]) == {"positive_observation", "background"}
    assert {
        "abrupt",
        "gradual",
        "intermittent",
        "evidence_delay",
        "modality_missingness",
    }.issubset(set(skeleton["template_family"]))
    assert (
        skeleton.groupby(["pool", "raw_group_id"])["episode_seed"].nunique().min()
        == 3
    )
    assert (
        skeleton.groupby(["pool", "episode_seed", "raw_group_id"])["episode_id"]
        .nunique()
        .max()
        == 1
    )
    missingness = skeleton.loc[
        skeleton["template_family"] == "modality_missingness"
    ]
    assert (
        missingness.groupby(["episode_id", "step_index", "concept_id"])["node_id"]
        .nunique()
        .eq(1)
        .any()
    )
    assert (root / "data/locked/episode_template_instances.parquet").is_file()
    projection = pd.read_parquet(
        root / "data/seals/episode_skeleton_inference_projection.parquet"
    )
    assert set(projection.columns) == {
        "skeleton_item_id",
        "record_id",
        "concept_id",
        "node_id",
        "generator_family_id",
    }
    _write_branch_predictions(root)
    features = workflow_episode.build_and_audit_episode_features(
        root, {"step_id": "E301"}, {}
    )
    assert features["details"]["feature_row_count"] == len(skeleton)
    feature_columns = set(
        pd.read_parquet(root / "data/locked/episode_features/D_e_te.parquet").columns
    )
    assert "pool" not in feature_columns
    assert "event_truth" not in feature_columns
    label_columns = set(
        pd.read_parquet(root / "data/locked/episode_labels/D_e_te.parquet").columns
    )
    assert "event_truth" in label_columns
    train_arrays = load_episode_arrays(root, pool="D_e_tr", include_labels=True)
    assert train_arrays.probabilities.shape[1] == 2
    assert train_arrays.labels is not None
    assert train_arrays.edge_validity.any()
    test_arrays = load_episode_arrays(
        root,
        pool="D_e_te",
        include_labels=False,
        node_ids=train_arrays.node_ids,
        concept_ids=train_arrays.concept_ids,
    )
    assert test_arrays.labels is None
    with pytest.raises(RuntimeError, match="test labels"):
        load_episode_arrays(root, pool="D_e_te", include_labels=True)
    (root / "configs/protocol_lock.pretest.yaml").write_text(
        yaml.safe_dump(
            {
                "models": {
                    "visible": {"input_size": 64},
                    "methane": {"hidden_size": 64, "layers": 1, "dropout": 0.2},
                }
            }
        ),
        encoding="utf-8",
    )
    sealed = workflow_episode.seal_episode_test(root, {"step_id": "E302"}, {})
    assert sealed["details"] == {"scope": "episode", "stage": "final"}
    seal = json.loads((root / "data/seals/episode_test_seal.json").read_text())
    assert seal["candidate_seal_hash"] == sha256_file(
        root / "data/seals/episode_test_seal_candidate.json"
    )
    assert seal["episode_skeleton_candidate_seal_hash"] == sha256_file(
        root / "data/seals/episode_test_skeleton_candidate.json"
    )


def test_e300_predicts_only_from_truth_free_projection(tmp_path: Path) -> None:
    root = _prepare_root(tmp_path)
    workflow_episode.build_episode_skeletons(root, {"step_id": "E058"}, {})
    (root / "configs/protocol_lock.pretest.yaml").write_text(
        yaml.safe_dump(
            {
                "models": {
                    "visible": {"input_size": 64},
                    "methane": {"hidden_size": 64, "layers": 1, "dropout": 0.2},
                }
            }
        ),
        encoding="utf-8",
    )
    _write_t1_generator_locks(root)
    run_root = root / "runs/slurm/E300/fixture"
    run_root.mkdir(parents=True)
    (run_root / "slurm_environment_probe.json").write_text(
        json.dumps({"status": "pass"}), encoding="utf-8"
    )
    result = predict_episode_branches(
        root,
        run_root,
        None,
        yolo_factory=_FakeYolo,
    )
    assert result["status"] == "pass"
    predictions = pd.read_parquet(root / "predictions/episode_branches/all.parquet")
    projection = pd.read_parquet(
        root / "data/seals/episode_skeleton_inference_projection.parquet"
    )
    assert len(predictions) == len(projection)
    assert predictions["skeleton_item_id"].is_unique
    assert set(predictions["skeleton_item_id"]) == set(projection["skeleton_item_id"])
    assert not {
        "episode_id",
        "step_index",
        "pool",
        "template_family",
        "event_position_role",
        "event_truth",
    }.intersection(predictions.columns)
    selection = json.loads(
        (root / "evidence/episode/E300_generator_selection.json").read_text()
    )
    assert selection["full_skeleton_opened"] is False
    assert selection["test_information_used"] is False


def test_e303_full_model_trains_without_opening_test_labels(tmp_path: Path) -> None:
    root = _prepare_root(tmp_path)
    workflow_episode.build_episode_skeletons(root, {"step_id": "E058"}, {})
    _write_branch_predictions(root)
    workflow_episode.build_and_audit_episode_features(root, {"step_id": "E301"}, {})
    protocol = {
        "seeds": {"episode": [9103, 9137, 9161]},
        "training": {
            "episode": {
                "optimizer": "adamw",
                "learning_rate": 0.001,
                "effective_batch_size": 64,
                "max_updates": 2,
            }
        },
    }
    (root / "configs/protocol_lock.pretest.yaml").write_text(
        yaml.safe_dump(protocol), encoding="utf-8"
    )
    pilot = root / "evidence/pilot/resource_pilot.json"
    pilot.parent.mkdir(parents=True, exist_ok=True)
    pilot.write_text(
        json.dumps(
            {
                "decisions": {
                    "precision": "fp32",
                    "micro_batch_size": {"episode": 8},
                    "gradient_accumulation": {"episode": 8},
                }
            }
        ),
        encoding="utf-8",
    )
    run_root = root / "runs/slurm/E303/fixture"
    run_root.mkdir(parents=True)
    (run_root / "slurm_environment_probe.json").write_text(
        json.dumps({"status": "pass"}), encoding="utf-8"
    )
    result = train_episode_run(root, run_root, None, array_index=9)
    assert result["status"] == "pass"
    manifest = json.loads(
        (root / "runs/E2_E3/E3-FULL/seed-9103/run_manifest.json").read_text()
    )
    assert manifest["test_labels_opened"] is False
    checkpoint_path = root / manifest["checkpoint_path"]
    device = torch.device("cpu")
    model, checkpoint = load_episode_model(checkpoint_path, device=device)
    arrays = load_episode_arrays(
        root,
        pool="D_e_te",
        include_labels=False,
        node_ids=checkpoint["node_ids"],
        concept_ids=checkpoint["concept_ids"],
    )
    inference = infer_episode_model(
        model, checkpoint, arrays, device=device
    )
    assert len(inference["probability"]) == len(arrays.metadata)
    assert np.isfinite(inference["probability"]).all()


def test_e058_independent_review_rejects_tampered_candidate(tmp_path: Path) -> None:
    root = _prepare_root(tmp_path)
    built = workflow_episode.build_episode_skeletons(
        root, {"step_id": "E058"}, {}
    )
    _write_e058_receipt(root, built)
    path = root / "data/seals/episode_test_skeleton_candidate.json"
    candidate = json.loads(path.read_text(encoding="utf-8"))
    candidate["row_count"] += 1
    path.write_text(json.dumps(candidate), encoding="utf-8")

    review = review_e058(root)

    assert review["status"] == "fail"
    assert any("candidate.row_count" in error for error in review["errors"])


def test_e058_independent_review_rejects_tampered_eligibility(tmp_path: Path) -> None:
    root = _prepare_root(tmp_path)
    built = workflow_episode.build_episode_skeletons(
        root, {"step_id": "E058"}, {}
    )
    _write_e058_receipt(root, built)
    path = root / "data/locked/fusion_eligibility_lock.parquet"
    eligibility = pd.read_parquet(path)
    eligibility.loc[:, "independent_multibranch_positive_group_count"] = 0
    eligibility.to_parquet(path, index=False)

    review = review_e058(root)

    assert review["status"] == "fail"
    assert any(
        "independent_multibranch_positive_group_count" in error
        for error in review["errors"]
    )
