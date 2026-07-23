from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from mining1_exp.data.seals import build_candidate_test_seal
from mining1_exp.governance.artifacts import (
    canonical_run_id,
    validate_run_bundle,
)
from mining1_exp.governance.efficiency import (
    MODULE_IDS,
    derive_efficiency_summary,
    validate_efficiency_completion,
    validate_efficiency_scope,
    validate_efficiency_trace,
)
from mining1_exp.governance.gates import build_gate, validate_gate, write_gate
from mining1_exp.governance.immutable import (
    GovernanceContractError,
    write_once_json,
)
from mining1_exp.governance.prediction_lock import (
    assert_truth_free_prediction,
    build_prediction_lock,
    close_prediction_lock,
    verify_prediction_lock,
)
from mining1_exp.governance.release import (
    build_invalidation,
    validate_label_access,
    write_invalidation,
    write_test_release,
)
from mining1_exp.efficiency import _build_scope
from mining1_exp.governance.remote import (
    validate_remote_closeout,
    validate_remote_closeout_index,
    validate_remote_monitor_receipt,
)
from mining1_exp.provenance import sha256_file
from mining1_exp import workflow_governance


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def test_write_once_is_idempotent_only_for_identical_content(tmp_path: Path) -> None:
    path = tmp_path / "immutable.json"
    first = write_once_json(path, {"value": 1})
    second = write_once_json(path, {"value": 1})
    assert first.created is True
    assert second.created is False
    before = path.read_bytes()
    with pytest.raises(GovernanceContractError, match="immutable artifact differs"):
        write_once_json(path, {"value": 2})
    assert path.read_bytes() == before


def _close_fixture_prediction_lock(tmp_path: Path) -> tuple[Path, Path, dict[str, Path]]:
    seal_path = tmp_path / "branch_test_seal.json"
    seal_payload = build_candidate_test_seal(
        scope="branch",
        split_hash=SHA_A,
        feature_manifest_hash=SHA_B,
        label_manifest_hash=SHA_C,
        artifact_contract_hash=SHA_D,
        feature_acl=["allow:test_predictor"],
        label_acl=[
            "deny:trainer",
            "deny:selector",
            "deny:calibrator",
            "deny:policy_fitter",
            "deny:test_predictor",
            "deny:human_reviewer",
        ],
    )
    seal_payload.update(
        {
            "stage": "final",
            "candidate_seal_hash": SHA_A,
            "protocol_hash": SHA_A,
            "source_hash": SHA_B,
            "matrix_hash": SHA_C,
        }
    )
    write_once_json(seal_path, seal_payload)
    predictions = {}
    for family, score in (("T1-VIS", 0.7), ("T1-THERM", 0.6)):
        path = tmp_path / f"{family}.parquet"
        pd.DataFrame(
            [{"record_id": "r1", "concept_id": "helmet", "probability": score}]
        ).to_parquet(path, index=False)
        predictions[family] = path
    payload = build_prediction_lock(
        scope="branch",
        seal_hash=sha256_file(seal_path),
        protocol_hash=SHA_A,
        source_hash=SHA_B,
        matrix_hash=SHA_C,
        required_prediction_families=list(predictions),
        prediction_paths=predictions,
        model_hashes={family: SHA_D for family in predictions},
        calibrator_and_policy_hashes={family: SHA_A for family in predictions},
    )
    lock_path = tmp_path / "prediction_lock.json"
    close_prediction_lock(lock_path, payload)
    return seal_path, lock_path, predictions


def test_prediction_lock_release_acl_and_invalidation_are_append_only(
    tmp_path: Path,
) -> None:
    seal_path, lock_path, predictions = _close_fixture_prediction_lock(tmp_path)
    assert set(verify_prediction_lock(lock_path, predictions)) == set(predictions)
    seal_hash = sha256_file(seal_path)
    lock_hash = sha256_file(lock_path)

    with pytest.raises(GovernanceContractError, match="remain sealed"):
        validate_label_access(
            role="evaluator",
            principal_identity="independent_evaluator_v1",
            seal_hash=seal_hash,
            prediction_lock_hash=lock_hash,
            release_payload=None,
        )
    with pytest.raises(GovernanceContractError, match="does not match both"):
        write_test_release(
            output_path=tmp_path / "mismatched_release.json",
            scope="branch",
            seal_path=seal_path,
            prediction_lock_path=lock_path,
            protocol_hash=SHA_D,
            source_hash=SHA_B,
            matrix_hash=SHA_C,
            evaluator_identity="independent_evaluator_v1",
            evaluator_label_acl=["allow:evaluator:independent_evaluator_v1"],
            signer_identity="release_authority_v1",
        )
    with pytest.raises(GovernanceContractError, match="only the independent evaluator"):
        write_test_release(
            output_path=tmp_path / "overbroad_release.json",
            scope="branch",
            seal_path=seal_path,
            prediction_lock_path=lock_path,
            protocol_hash=SHA_A,
            source_hash=SHA_B,
            matrix_hash=SHA_C,
            evaluator_identity="independent_evaluator_v1",
            evaluator_label_acl=[
                "allow:evaluator:independent_evaluator_v1",
                "allow:trainer",
            ],
            signer_identity="release_authority_v1",
        )
    release_path = tmp_path / "test_release.json"
    write_test_release(
        output_path=release_path,
        scope="branch",
        seal_path=seal_path,
        prediction_lock_path=lock_path,
        protocol_hash=SHA_A,
        source_hash=SHA_B,
        matrix_hash=SHA_C,
        evaluator_identity="independent_evaluator_v1",
        evaluator_label_acl=["allow:evaluator:independent_evaluator_v1"],
        signer_identity="release_authority_v1",
    )
    release = json.loads(release_path.read_text(encoding="utf-8"))
    validate_label_access(
        role="evaluator",
        principal_identity="independent_evaluator_v1",
        seal_hash=seal_hash,
        prediction_lock_hash=lock_hash,
        release_payload=release,
    )
    tampered_release = {**release, "evaluator_identity": "different_evaluator"}
    with pytest.raises(GovernanceContractError, match="does not bind"):
        validate_label_access(
            role="evaluator",
            principal_identity="different_evaluator",
            seal_hash=seal_hash,
            prediction_lock_hash=lock_hash,
            release_payload=tampered_release,
        )
    with pytest.raises(GovernanceContractError, match="never authorized"):
        validate_label_access(
            role="trainer",
            principal_identity="independent_evaluator_v1",
            seal_hash=seal_hash,
            prediction_lock_hash=lock_hash,
            release_payload=release,
        )
    assert sha256_file(seal_path) == seal_hash
    assert sha256_file(lock_path) == lock_hash

    pd.DataFrame(
        [{"record_id": "r1", "concept_id": "helmet", "probability": 0.9}]
    ).to_parquet(predictions["T1-VIS"], index=False)
    with pytest.raises(GovernanceContractError, match="prediction hash drift"):
        verify_prediction_lock(lock_path, predictions)
    invalidation = build_invalidation(
        scope="branch",
        seal_hash=seal_hash,
        prediction_lock_hash=lock_hash,
        reason="prediction artifact hash changed after lock close",
        observed_prediction_hashes={
            "T1-VIS": sha256_file(predictions["T1-VIS"])
        },
        signer_identity="invalidation_authority_v1",
    )
    write_invalidation(tmp_path / "invalidation.json", invalidation)
    assert sha256_file(seal_path) == seal_hash
    assert sha256_file(lock_path) == lock_hash


def test_branch_release_can_be_scoped_to_s1_only(tmp_path: Path) -> None:
    root = tmp_path / "project"
    for directory in (
        root / "configs",
        root / "data/locked",
        root / "data/sealed",
        root / "data/seals",
        root / "evidence/data",
        root / "predictions/locked/S1",
    ):
        directory.mkdir(parents=True)
    (root / "configs/protocol_lock.pretest.yaml").write_text("locked: true\n", encoding="utf-8")
    (root / "configs/experiment_matrix.template.csv").write_text("family_id\nS1-GRU\n", encoding="utf-8")
    (root / "evidence/data/dataset_source_decision.json").write_text(
        json.dumps({"status": "pass"}, sort_keys=True), encoding="utf-8"
    )

    branch_features = root / "data/locked/branch_test_features.parquet"
    methane_features = root / "data/locked/branch_methane_test_features.parquet"
    detection_truth = root / "data/sealed/branch_detection_truth.parquet"
    concept_truth = root / "data/sealed/branch_concept_truth.parquet"
    methane_truth = root / "data/sealed/branch_methane_truth.parquet"
    pd.DataFrame([{"record_id": "r1", "feature_value": 1.0}]).to_parquet(
        branch_features, index=False
    )
    pd.DataFrame([{"window_id": "w1", "pool": "D_b_te", "feature_value": 1.0}]).to_parquet(
        methane_features, index=False
    )
    pd.DataFrame(
        [{"record_id": "r1", "raw_group_id": "g1", "concept_id": "worker", "x1": 0, "y1": 0, "x2": 1, "y2": 1}]
    ).to_parquet(detection_truth, index=False)
    pd.DataFrame(
        [{"record_id": "r1", "raw_group_id": "g1", "concept_id": "worker", "concept_truth": 1}]
    ).to_parquet(concept_truth, index=False)
    pd.DataFrame([{"window_id": "w1", "pool": "D_b_te", "event_truth": 1}]).to_parquet(
        methane_truth, index=False
    )
    (root / "data/locked/branch_test_feature_manifest.json").write_text(
        json.dumps(
            {
                "detection_features": {"path": "data/locked/branch_test_features.parquet", "sha256": sha256_file(branch_features), "row_count": 1},
                "methane_features": {"path": "data/locked/branch_methane_test_features.parquet", "sha256": sha256_file(methane_features), "row_count": 1},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (root / "data/sealed/branch_test_truth_manifest.json").write_text(
        json.dumps(
            {
                "detection_truth": {"path": "data/sealed/branch_detection_truth.parquet", "sha256": sha256_file(detection_truth), "row_count": 1},
                "concept_truth": {"path": "data/sealed/branch_concept_truth.parquet", "sha256": sha256_file(concept_truth), "row_count": 1},
                "methane_truth": {"path": "data/sealed/branch_methane_truth.parquet", "sha256": sha256_file(methane_truth), "row_count": 1},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    protocol_hash = sha256_file(root / "configs/protocol_lock.pretest.yaml")
    source_hash = sha256_file(root / "evidence/data/dataset_source_decision.json")
    matrix_hash = sha256_file(root / "configs/experiment_matrix.template.csv")
    seal_path = root / "data/seals/branch_test_seal.json"
    seal_payload = build_candidate_test_seal(
        scope="branch",
        split_hash=SHA_A,
        feature_manifest_hash=sha256_file(root / "data/locked/branch_test_feature_manifest.json"),
        label_manifest_hash=sha256_file(root / "data/sealed/branch_test_truth_manifest.json"),
        artifact_contract_hash=SHA_D,
        feature_acl=["allow:test_predictor"],
        label_acl=[
            "deny:trainer",
            "deny:selector",
            "deny:calibrator",
            "deny:policy_fitter",
            "deny:test_predictor",
            "deny:human_reviewer",
        ],
    )
    seal_payload.update(
        {
            "stage": "final",
            "candidate_seal_hash": SHA_A,
            "protocol_hash": protocol_hash,
            "source_hash": source_hash,
            "matrix_hash": matrix_hash,
        }
    )
    seal_hash = write_once_json(seal_path, seal_payload).artifact.sha256
    prediction_path = root / "predictions/locked/S1.parquet"
    pd.DataFrame([{"window_id": "w1", "score_calibrated": 0.8}]).to_parquet(
        prediction_path, index=False
    )
    lock_payload = build_prediction_lock(
        scope="branch",
        seal_hash=seal_hash,
        protocol_hash=protocol_hash,
        source_hash=source_hash,
        matrix_hash=matrix_hash,
        required_prediction_families=["S1-GRU"],
        prediction_paths={"S1-GRU": prediction_path},
        model_hashes={"S1-GRU": SHA_A},
        calibrator_and_policy_hashes={"S1-GRU": SHA_B},
    )
    lock_payload["prediction_paths"] = {
        "S1-GRU": prediction_path.relative_to(root).as_posix()
    }
    close_prediction_lock(root / "predictions/locked/S1/prediction_lock.json", lock_payload)

    result = workflow_governance.release_branch_test(
        root,
        {"step_id": "E206"},
        {"packages": ["S1"]},
    )

    assert result["status"] == "pass"
    assert result["details"]["packages"] == ["S1"]
    assert result["details"]["prediction_lock_count"] == 1
    assert not (root / "predictions/locked/T1/prediction_lock.json").exists()
    merged = json.loads((root / "predictions/locked/branch_prediction_lock.json").read_text())
    assert merged["required_prediction_families"] == ["S1-GRU"]


def test_gate_review_digest_accepts_not_applicable_dependency(tmp_path: Path) -> None:
    review_dir = tmp_path / "evidence/step_reviews"
    review_dir.mkdir(parents=True)
    review_path = review_dir / "E112.json"
    review_path.write_text(
        json.dumps(
            {"schema_version": 1, "step_id": "E112", "status": "not_applicable", "advance_allowed": True},
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    digest = workflow_governance._review_digest(tmp_path, "E112")

    assert digest["status"] == "pass"
    assert digest["review_status"] == "not_applicable"
    assert digest["review_sha256"] == sha256_file(review_path)


def test_prediction_truth_scan_and_lock_family_closure_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = tmp_path / "bad.parquet"
    pd.DataFrame([{"record_id": "r1", "event_truth": 1}]).to_parquet(
        bad, index=False
    )
    with pytest.raises(GovernanceContractError, match="truth fields"):
        assert_truth_free_prediction(bad)
    metric_leak = tmp_path / "metric-leak.parquet"
    pd.DataFrame([{"record_id": "r1", "map50": 0.9}]).to_parquet(
        metric_leak, index=False
    )
    with pytest.raises(GovernanceContractError, match="truth fields"):
        assert_truth_free_prediction(metric_leak)
    safe = tmp_path / "schema-only.parquet"
    pd.DataFrame([{"record_id": "r1", "probability": 0.5}]).to_parquet(
        safe, index=False
    )
    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("prediction lock must not load the full Parquet table")
        ),
    )
    assert assert_truth_free_prediction(safe) == ("record_id", "probability")
    seal = tmp_path / "seal.json"
    write_once_json(seal, {"sealed": True})
    with pytest.raises(GovernanceContractError, match="does not close"):
        build_prediction_lock(
            scope="branch",
            seal_hash=sha256_file(seal),
            protocol_hash=SHA_A,
            source_hash=SHA_B,
            matrix_hash=SHA_C,
            required_prediction_families=["A", "B"],
            prediction_paths={"A": bad},
            model_hashes={"A": SHA_A, "B": SHA_B},
            calibrator_and_policy_hashes={"A": SHA_A, "B": SHA_B},
        )


def _rehash_run_bundle(root: Path) -> None:
    names = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "artifact_sha256.txt"
    )
    (root / "artifact_sha256.txt").write_text(
        "".join(f"{sha256_file(root / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )


def _build_run_bundle(root: Path) -> None:
    root.mkdir()
    manifest = {
        "family_id": "T1-VIS",
        "protocol_hash": SHA_A,
        "artifact_contract_hash": SHA_B,
        "code_hash": SHA_C,
        "environment_ready_sha256": SHA_D,
        "package_inventory_sha256": SHA_A,
        "command_receipt_sha256": SHA_B,
        "parent_checkpoint_hashes": {"initialization": SHA_B},
        "data_and_split_hashes": {"data": SHA_C, "split": SHA_D},
        "seeds": {"training": 1701},
        "host_and_slurm_job_id": {
            "host": "xinxi-zhyh@211.87.115.228",
            "slurm_job_id": "12345",
            "allocation_id": "12345",
            "compute_node": "gpu01",
        },
        "started_at_and_finished_at": {
            "started_at": "2026-07-15T00:00:00+00:00",
            "finished_at": "2026-07-15T01:00:00+00:00",
        },
        "exit_code": 0,
        "status": "pass",
        "remote_closeout_reference": {
            "path": "evidence/closeout/12345.json",
            "sha256": SHA_D,
            "remote_action_id": "action-E100-12345",
        },
    }
    manifest["run_id"] = canonical_run_id(manifest)
    (root / "run_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    (root / "config_resolved.yaml").write_text(
        yaml.safe_dump({"family_id": "T1-VIS"}), encoding="utf-8"
    )
    (root / "environment.json").write_text(
        json.dumps(
            {
                "selected_remote_python": (
                    "/data/home/xinxi-zhyh/xinxi-zhyh/envs/"
                    "mining1-py39-cu121/bin/python"
                ),
                "python_version": "3.9.19",
                "framework_version": "torch-2.5.1",
                "cuda_runtime": "12.1",
                "environment_ready_sha256": SHA_D,
                "package_inventory_sha256": SHA_A,
                "environment_fingerprint_sha256": SHA_C,
                "slurm_job_id": "12345",
                "allocation_id": "12345",
                "compute_node": "gpu01",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (root / "data_hashes.json").write_text(
        json.dumps({"data": SHA_C, "split": SHA_D}, sort_keys=True),
        encoding="utf-8",
    )
    pd.DataFrame([{"epoch": 1, "loss": 0.5}]).to_csv(
        root / "metrics_per_epoch.csv", index=False
    )
    (root / "execution.log").write_text("epoch 1 complete\n", encoding="utf-8")
    for name in ("last_checkpoint", "best_checkpoint"):
        (root / name).write_bytes(name.encode("ascii"))
    (root / "calibrator.json").write_text('{"kind":"temperature"}', encoding="utf-8")
    pd.DataFrame([{"record_id": "r1", "probability": 0.5}]).to_parquet(
        root / "predictions.parquet", index=False
    )
    _rehash_run_bundle(root)


def test_run_bundle_hash_closure_and_pre_release_metric_boundary(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _build_run_bundle(root)
    result = validate_run_bundle(
        root,
        selected_trainable_run=True,
        probability_calibration_applies=True,
        scheduled_prediction_family=True,
        label_release_authorized=False,
        expected_selected_remote_python=(
            "/data/home/xinxi-zhyh/xinxi-zhyh/envs/"
            "mining1-py39-cu121/bin/python"
        ),
        expected_environment_ready_sha256=SHA_D,
        expected_environment_fingerprint_sha256=SHA_C,
        expected_package_inventory_sha256=SHA_A,
    )
    assert result["status"] == "pass"
    assert result["verified_file_count"] == 10

    (root / "untracked.txt").write_text("not in artifact manifest", encoding="utf-8")
    with pytest.raises(GovernanceContractError, match="does not close"):
        validate_run_bundle(
            root,
            selected_trainable_run=True,
            probability_calibration_applies=True,
            scheduled_prediction_family=True,
            label_release_authorized=False,
            expected_selected_remote_python=(
                "/data/home/xinxi-zhyh/xinxi-zhyh/envs/"
                "mining1-py39-cu121/bin/python"
            ),
            expected_environment_ready_sha256=SHA_D,
            expected_environment_fingerprint_sha256=SHA_C,
            expected_package_inventory_sha256=SHA_A,
        )
    (root / "untracked.txt").unlink()

    (root / "metrics.json").write_text('{"map50":0.5}', encoding="utf-8")
    with pytest.raises(GovernanceContractError, match="before matching label release"):
        validate_run_bundle(
            root,
            selected_trainable_run=True,
            probability_calibration_applies=True,
            scheduled_prediction_family=True,
            label_release_authorized=False,
            expected_selected_remote_python=(
                "/data/home/xinxi-zhyh/xinxi-zhyh/envs/"
                "mining1-py39-cu121/bin/python"
            ),
            expected_environment_ready_sha256=SHA_D,
            expected_environment_fingerprint_sha256=SHA_C,
            expected_package_inventory_sha256=SHA_A,
        )


def test_run_bundle_rejects_environment_and_data_hash_drift(tmp_path: Path) -> None:
    environment_root = tmp_path / "environment-drift"
    _build_run_bundle(environment_root)
    environment = json.loads(
        (environment_root / "environment.json").read_text(encoding="utf-8")
    )
    environment["environment_ready_sha256"] = SHA_C
    (environment_root / "environment.json").write_text(
        json.dumps(environment, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(GovernanceContractError, match="ready hash drifted"):
        validate_run_bundle(
            environment_root,
            selected_trainable_run=True,
            probability_calibration_applies=True,
            scheduled_prediction_family=True,
            label_release_authorized=False,
            expected_selected_remote_python=(
                "/data/home/xinxi-zhyh/xinxi-zhyh/envs/"
                "mining1-py39-cu121/bin/python"
            ),
            expected_environment_ready_sha256=SHA_D,
            expected_environment_fingerprint_sha256=SHA_C,
            expected_package_inventory_sha256=SHA_A,
        )

    data_root = tmp_path / "data-drift"
    _build_run_bundle(data_root)
    (data_root / "data_hashes.json").write_text(
        json.dumps({"data": SHA_A, "split": SHA_D}, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(GovernanceContractError, match="does not match"):
        validate_run_bundle(
            data_root,
            selected_trainable_run=True,
            probability_calibration_applies=True,
            scheduled_prediction_family=True,
            label_release_authorized=False,
            expected_selected_remote_python=(
                "/data/home/xinxi-zhyh/xinxi-zhyh/envs/"
                "mining1-py39-cu121/bin/python"
            ),
            expected_environment_ready_sha256=SHA_D,
            expected_environment_fingerprint_sha256=SHA_C,
            expected_package_inventory_sha256=SHA_A,
        )


def test_failed_run_bundle_preserves_failure_and_retry_contract(tmp_path: Path) -> None:
    root = tmp_path / "failed-run"
    _build_run_bundle(root)
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    manifest["status"] = "fail"
    manifest["exit_code"] = 143
    manifest["run_id"] = canonical_run_id(manifest)
    (root / "run_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    pd.DataFrame(columns=["epoch", "loss"]).to_csv(
        root / "metrics_per_epoch.csv", index=False
    )
    failure = {
        "error_type": "InterruptedError",
        "error": "Slurm SIGTERM received",
        "last_safe_checkpoint": "last_checkpoint",
        "retry_eligible": True,
        "frozen_retry_rule": "resume_exact_checkpoint_once",
        "failed_at": "2026-07-15T00:59:00+00:00",
        "exit_code": 143,
    }
    (root / "failure.json").write_text(
        json.dumps(failure, sort_keys=True), encoding="utf-8"
    )
    _rehash_run_bundle(root)
    result = validate_run_bundle(
        root,
        selected_trainable_run=False,
        probability_calibration_applies=True,
        scheduled_prediction_family=True,
        label_release_authorized=False,
        expected_selected_remote_python=(
            "/data/home/xinxi-zhyh/xinxi-zhyh/envs/"
            "mining1-py39-cu121/bin/python"
        ),
        expected_environment_ready_sha256=SHA_D,
        expected_environment_fingerprint_sha256=SHA_C,
        expected_package_inventory_sha256=SHA_A,
    )
    assert result["status"] == "fail"
    broken_failure = {**failure, "frozen_retry_rule": ""}
    (root / "failure.json").write_text(
        json.dumps(broken_failure, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(GovernanceContractError, match="frozen retry rule"):
        validate_run_bundle(
            root,
            selected_trainable_run=False,
            probability_calibration_applies=True,
            scheduled_prediction_family=True,
            label_release_authorized=False,
            expected_selected_remote_python=(
                "/data/home/xinxi-zhyh/xinxi-zhyh/envs/"
                "mining1-py39-cu121/bin/python"
            ),
            expected_environment_ready_sha256=SHA_D,
            expected_environment_fingerprint_sha256=SHA_C,
            expected_package_inventory_sha256=SHA_A,
        )


def test_gate_pass_forbids_waivers_and_remote_pass_requires_closeout(
    tmp_path: Path,
) -> None:
    gate = build_gate(
        gate_id="LOCAL_IMPLEMENTATION",
        status="pass",
        protocol_hash=SHA_A,
        code_hash=SHA_B,
        experiment_matrix_hash=SHA_C,
        artifact_contract_hash=SHA_D,
        required_steps=["I050"],
        input_artifacts=[{"path": "input.json", "sha256": SHA_A}],
        output_artifacts=[{"path": "output.json", "sha256": SHA_B}],
        checks=[{"id": "I050", "status": "pass"}],
        failures=[],
        waivers=[],
        allowed_next_steps=["I070"],
    )
    validate_gate(gate)
    first = write_gate(tmp_path / "gate.json", gate)
    second = write_gate(tmp_path / "gate.json", gate)
    assert first.created and not second.created
    with pytest.raises(GovernanceContractError, match="failures or waivers"):
        validate_gate({**gate, "waivers": ["skip test"]})
    with pytest.raises(GovernanceContractError, match="requires closeout"):
        build_gate(
            **{
                **{key: value for key, value in gate.items() if key not in {"created_at"}},
                "gate_id": "REMOTE_GATE",
                "host": "xinxi-zhyh@211.87.115.228",
                "remote_closeout_artifact": None,
            }
        )
    with pytest.raises(GovernanceContractError, match="validated closeout check"):
        build_gate(
            **{
                **{key: value for key, value in gate.items() if key not in {"created_at"}},
                "gate_id": "REMOTE_GATE",
                "host": "xinxi-zhyh@211.87.115.228",
                "remote_closeout_artifact": "evidence/closeout/E064.json",
            }
        )
    remote_gate = build_gate(
        **{
            **{key: value for key, value in gate.items() if key not in {"created_at"}},
            "gate_id": "REMOTE_GATE",
            "host": "xinxi-zhyh@211.87.115.228",
            "checks": [
                {"id": "I050", "status": "pass"},
                {"id": "remote_closeout", "status": "pass"},
            ],
            "output_artifacts": [
                *gate["output_artifacts"],
                {"path": "evidence/closeout/E064.json", "sha256": SHA_C},
            ],
            "remote_closeout_artifact": "evidence/closeout/E064.json",
        }
    )
    validate_gate(remote_gate)
    failed_gate = {
        **gate,
        "status": "fail",
        "checks": [{"id": "I050", "status": "fail"}],
        "failures": ["I050 contract failed"],
    }
    validate_gate(failed_gate)
    with pytest.raises(GovernanceContractError, match="passing checks"):
        validate_gate({**failed_gate, "status": "pass", "failures": []})
    with pytest.raises(GovernanceContractError, match="local gate cannot list"):
        validate_gate({**gate, "slurm_job_ids": ["12345"]})


def _efficiency_scope() -> pd.DataFrame:
    families = {
        "visible_inference": "V2-A-10-MULTI",
        "thermal_inference": "T1-THERM",
        "sensor_inference": "S1-GRU",
        "graph_fusion": "E3-FULL",
    }
    gates = {
        "visible_inference": "G3",
        "thermal_inference": "G2",
        "sensor_inference": "G2",
        "graph_fusion": "G5",
    }
    rows = []
    for module_id in MODULE_IDS:
        measured = module_id != "graph_fusion"
        rows.append(
            {
                "module_id": module_id,
                "family_id": families[module_id],
                "eligibility_status": "measured" if measured else "not_applicable",
                "source_gate_id": gates[module_id],
                "source_gate_sha256": SHA_A,
                "checkpoint_sha256": SHA_B if measured else None,
                "exclusion_reason": None if measured else "empty graph-primary population",
                "protocol_sha256": SHA_C,
            }
        )
    return pd.DataFrame(rows)


def _efficiency_trace(scope: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for module_id in scope.loc[scope["eligibility_status"] == "measured", "module_id"]:
        device_kind = "cpu" if module_id == "sensor_inference" else "cuda"
        for phase, count in (("warmup", 50), ("timed", 200)):
            for iteration in range(1, count + 1):
                rows.append(
                    {
                        "measurement_id": "measurement_1",
                        "module_id": module_id,
                        "variant_id": "primary",
                        "model_family_id": str(
                            scope.loc[
                                scope["module_id"] == module_id, "family_id"
                            ].iloc[0]
                        ),
                        "representative_seed": 1701,
                        "checkpoint_sha256": SHA_B,
                        "policy_sha256": None,
                        "environment_ready_sha256": SHA_D,
                        "protocol_sha256": SHA_C,
                        "slurm_job_id": "12345",
                        "allocation_id": "allocation_12345",
                        "compute_node": "gpu01",
                        "repeat_id": 1,
                        "phase": phase,
                        "iteration_index": iteration,
                        "duration_ns": 1_000_000 + iteration,
                        "device_kind": device_kind,
                        "device_identifier": "cpu0" if device_kind == "cpu" else "cuda:0",
                        "gpu_uuid": None if device_kind == "cpu" else "GPU-test",
                        "gpu_clock_mhz": None if device_kind == "cpu" else 1500.0,
                        "cuda_driver": None if device_kind == "cpu" else "550.54",
                        "cpu_affinity": "0-7",
                        "precision": "fp32",
                        "batch_size": 1,
                        "input_shape": f"locked:{module_id}",
                        "cuda_synchronized_before_and_after": True,
                        "concurrent_user_gpu_job_count": 0,
                        "interference_audit_status": "pass",
                        "captured_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
    return pd.DataFrame(rows)


def test_efficiency_scope_trace_and_summary_close_without_substitution() -> None:
    scope = _efficiency_scope()
    gate_receipts = {
        "G2": {"status": "pass", "sha256": SHA_A},
        "G3": {"status": "pass", "sha256": SHA_A},
        "G5": {"status": "blocked", "sha256": SHA_A},
    }
    validate_efficiency_scope(
        scope,
        graph_primary_population_nonempty=False,
        gate_receipts=gate_receipts,
    )
    trace = _efficiency_trace(scope)
    validate_efficiency_trace(trace, scope)
    measured = set(scope.loc[scope["eligibility_status"] == "measured", "module_id"])
    summary = derive_efficiency_summary(
        trace,
        trace_sha256=SHA_A,
        derivation_code_sha256=SHA_B,
        peak_memory_bytes={module: 1024 for module in measured},
        parameter_count={module: 100 for module in measured},
        checkpoint_bytes={module: 2048 for module in measured},
    )
    validate_efficiency_completion(scope, trace, summary)
    assert set(summary["module_id"]) == measured
    assert (summary["timed_iteration_count_per_repeat"] == 200).all()
    tampered_summary = summary.copy()
    tampered_summary.loc[tampered_summary.index[0], "latency_p95_ns"] += 1
    with pytest.raises(GovernanceContractError, match="p95 is not derived"):
        validate_efficiency_completion(scope, trace, tampered_summary)

    interfered = trace.copy()
    timed_index = interfered.index[interfered["phase"] == "timed"][0]
    interfered.loc[timed_index, "concurrent_user_gpu_job_count"] = 1
    with pytest.raises(GovernanceContractError, match="concurrent interference"):
        validate_efficiency_trace(interfered, scope)
    family_substitution = trace.copy()
    family_substitution.loc[
        family_substitution["module_id"] == "visible_inference", "model_family_id"
    ] = "replacement_family"
    with pytest.raises(GovernanceContractError, match="substituted a model family"):
        validate_efficiency_trace(family_substitution, scope)
    substituted = scope.copy()
    graph_index = substituted.index[substituted["module_id"] == "graph_fusion"][0]
    substituted.loc[graph_index, "checkpoint_sha256"] = SHA_D
    with pytest.raises(GovernanceContractError, match="cannot substitute"):
        validate_efficiency_scope(
            substituted,
            graph_primary_population_nonempty=False,
            gate_receipts=gate_receipts,
        )
    family_drift = scope.copy()
    family_drift.loc[
        family_drift["module_id"] == "visible_inference", "family_id"
    ] = "replacement_family"
    with pytest.raises(GovernanceContractError, match="substituted a model family"):
        validate_efficiency_scope(
            family_drift,
            graph_primary_population_nonempty=False,
            gate_receipts=gate_receipts,
        )
    duplicate_variant = pd.concat(
        [trace, trace.loc[trace["module_id"] == "visible_inference"].assign(variant_id="alternate")],
        ignore_index=True,
    )
    with pytest.raises(GovernanceContractError, match="one preregistered variant"):
        validate_efficiency_trace(duplicate_variant, scope)
    blocked_g2 = {**gate_receipts, "G2": {"status": "blocked", "sha256": SHA_A}}
    with pytest.raises(GovernanceContractError, match="lacks a passing source gate"):
        validate_efficiency_scope(
            scope,
            graph_primary_population_nonempty=False,
            gate_receipts=blocked_g2,
        )


def test_efficiency_scope_marks_missing_locked_manifests_not_applicable(tmp_path: Path) -> None:
    run_root = tmp_path / "runs/S1/gru/S1-GRU"
    for seed, metric in ((1701, 0.8), (2903, 0.9), (4219, 0.7)):
        seed_root = run_root / f"seed-{seed}"
        seed_root.mkdir(parents=True)
        checkpoint = seed_root / "best.pt"
        checkpoint.write_bytes(f"checkpoint-{seed}".encode("utf-8"))
        manifest = {
            "status": "pass",
            "family_id": "S1-GRU",
            "train_seed": seed,
            "selection_pool": "D_b_sel",
            "selection_metric_value": metric,
            "checkpoint_path": str(checkpoint.relative_to(tmp_path)).replace("\\", "/"),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        }
        (seed_root / "run_manifest.json").write_text(
            json.dumps(manifest, sort_keys=True),
            encoding="utf-8",
        )

    scope, representatives = _build_scope(
        tmp_path,
        protocol_sha256=SHA_C,
        gate_receipts={
            "G2": {"status": "pass", "sha256": SHA_A},
            "G3": {"status": "pass", "sha256": SHA_A},
            "G5": {"status": "pass", "sha256": SHA_A},
        },
        graph_nonempty=False,
    )

    by_module = scope.set_index("module_id")
    assert set(representatives) == {"sensor_inference"}
    assert by_module.loc["sensor_inference", "eligibility_status"] == "measured"
    assert by_module.loc["thermal_inference", "eligibility_status"] == "not_applicable"
    assert str(by_module.loc["thermal_inference", "exclusion_reason"]).startswith(
        "locked_checkpoint_unavailable"
    )
    assert by_module.loc["visible_inference", "eligibility_status"] == "not_applicable"
    assert by_module.loc["graph_fusion", "eligibility_status"] == "not_applicable"


def test_remote_monitor_closeout_and_index_require_30_minute_and_billing_evidence() -> None:
    monitor = {
        "step_id": "E100",
        "monitored_at": "2026-07-15T00:00:00+00:00",
        "host": "xinxi-zhyh@211.87.115.228",
        "slurm_job_ids": ["12345"],
        "gpu_expected": True,
        "squeue_state": {
            "records": [{"job_id": "12345", "state": "RUNNING"}]
        },
        "sacct_state": {"records": []},
        "relevant_processes": [],
        "tmux_and_screen_sessions": [],
        "gpu_telemetry_artifacts": [
            {"path": "/runs/E100/telemetry.json", "sha256": SHA_A}
        ],
        "gpu_telemetry_excerpt": ["{\"gpu_utilization\":87}"],
        "recent_log_artifacts": [],
        "checkpoint_artifacts": [],
        "result_artifacts": [],
        "workflow_state": "running",
        "remote_error": None,
        "next_check_due_at": "2026-07-15T00:30:00+00:00",
    }
    validate_remote_monitor_receipt(monitor)
    with pytest.raises(GovernanceContractError, match="30 minutes"):
        validate_remote_monitor_receipt(
            {**monitor, "next_check_due_at": "2026-07-15T00:05:00+00:00"}
        )
    completed_monitor = {
        **monitor,
        "squeue_state": {"records": []},
        "sacct_state": {
            "records": [
                {"job_id": "12345", "state": "COMPLETED", "exit_code": "0:0"}
            ]
        },
        "workflow_state": "completed",
        "next_check_due_at": None,
    }
    validate_remote_monitor_receipt(completed_monitor)
    with pytest.raises(GovernanceContractError, match="complete sacct evidence"):
        validate_remote_monitor_receipt(
            {**completed_monitor, "sacct_state": {"records": []}}
        )
    with pytest.raises(GovernanceContractError, match="failed sacct"):
        validate_remote_monitor_receipt(
            {
                **completed_monitor,
                "sacct_state": {
                    "records": [
                        {"job_id": "12345", "state": "FAILED", "exit_code": "1:0"}
                    ]
                },
            }
        )

    closeout = {
        "trigger_step_id": "E100",
        "remote_action_id": "action_1",
        "remote_action_at": "2026-07-15T00:55:00+00:00",
        "trigger_kind": "remote_action_end",
        "closed_at": "2026-07-15T01:00:00+00:00",
        "host": "xinxi-zhyh@211.87.115.228",
        "account_scope": "all_user_owned_work",
        "slurm_jobs_and_allocations": [],
        "interactive_sessions": [],
        "tmux_and_screen_sessions": [],
        "relevant_processes": [],
        "transfers_and_monitors": [],
        "unsynchronized_artifacts": [],
        "retained_resources_with_reason": [],
        "cancellation_or_release_actions": [],
        "provider_stop_action": None,
        "final_billing_state": "verified_nonbilling",
    }
    validate_remote_closeout(closeout, require_verified=True)
    with pytest.raises(GovernanceContractError, match="unverified billing"):
        validate_remote_closeout(
            {**closeout, "final_billing_state": "unverified"}, require_verified=True
        )
    with pytest.raises(GovernanceContractError, match="active resources"):
        validate_remote_closeout(
            {**closeout, "relevant_processes": ["python -m mining1_exp.cli"]},
            require_verified=True,
        )

    index = {
        "reconciled_at": "2026-07-15T01:05:00+00:00",
        "remote_command_ledger_sha256": SHA_A,
        "remote_action_ids": ["action_1"],
        "remote_action_timestamps": {
            "action_1": "2026-07-15T00:55:00+00:00"
        },
        "closeout_receipts": [
            {
                "path": "closeout.json",
                "sha256": SHA_B,
                "remote_action_id": "action_1",
                "closed_at": "2026-07-15T01:00:00+00:00",
                "final_billing_state": "verified_nonbilling",
            }
        ],
        "action_to_receipt": {"action_1": ["closeout.json"]},
        "latest_remote_activity_at": "2026-07-15T00:55:00+00:00",
        "latest_closeout_at": "2026-07-15T01:00:00+00:00",
        "uncovered_remote_actions": [],
        "final_billing_state": "verified_nonbilling",
    }
    validate_remote_closeout_index(index)
    with pytest.raises(GovernanceContractError, match="wrong closeout"):
        validate_remote_closeout_index(
            {
                **index,
                "remote_action_ids": ["action_1", "action_2"],
                "remote_action_timestamps": {
                    "action_1": "2026-07-15T00:55:00+00:00",
                    "action_2": "2026-07-15T00:56:00+00:00",
                },
                "action_to_receipt": {
                    "action_1": ["closeout.json"],
                    "action_2": ["closeout.json"],
                },
                "latest_remote_activity_at": "2026-07-15T00:56:00+00:00",
            }
        )
    with pytest.raises(GovernanceContractError, match="predates"):
        validate_remote_closeout_index(
            {
                **index,
                "latest_closeout_at": "2026-07-15T00:50:00+00:00",
                "closeout_receipts": [
                    {
                        **index["closeout_receipts"][0],
                        "closed_at": "2026-07-15T00:50:00+00:00",
                    }
                ],
            }
        )
