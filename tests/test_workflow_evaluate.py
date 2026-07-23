from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from mining1_exp import workflow_evaluate


def test_select_v2_checkpoints_uses_only_locked_validation_cells(tmp_path: Path) -> None:
    root = tmp_path / "project"
    families = (
        "V2-A-10-SCR",
        "V2-A-10-GEN",
        "V2-A-10-SINGLE",
        "V2-A-10-MULTI",
    )
    seeds = ((1701, 5171), (2903, 6197), (4219, 7331))
    for family in families:
        for train_seed, subset_seed in seeds:
            run_id = f"{family}_{train_seed}_{subset_seed}"
            directory = root / "runs/V2/finetune" / run_id
            directory.mkdir(parents=True)
            checkpoint = directory / "best.pt"
            checkpoint.write_bytes(run_id.encode("utf-8"))
            manifest = {
                "run_id": run_id,
                "family_id": family,
                "train_seed": train_seed,
                "subset_seed": subset_seed,
                "selection_pool": "D_b_sel",
                "selection_metric": "map50_95",
                "selection_metric_value": 0.25,
                "checkpoint_path": checkpoint.relative_to(root).as_posix(),
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            }
            (directory / "run_manifest.json").write_text(
                json.dumps(manifest, sort_keys=True), encoding="utf-8"
            )
    result = workflow_evaluate.select_checkpoints(
        root,
        {"step_id": "E204"},
        {"family": "V2"},
    )
    assert result["details"]["selected_repeat_cells"] == 12
    payload = json.loads((root / "runs/V2/checkpoint_selection.json").read_text())
    assert payload["test_information_used"] is False
    assert {item["selection_pool"] for item in payload["selected"]} == {"D_b_sel"}
    assert result["details"]["resolved_repeat_cells"] == 12


def test_select_v2_checkpoints_accepts_reviewed_multi_not_applicable(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    families = (
        "V2-A-10-SCR",
        "V2-A-10-GEN",
        "V2-A-10-SINGLE",
    )
    seeds = ((1701, 5171), (2903, 6197), (4219, 7331))
    for family in families:
        for task_id, (train_seed, subset_seed) in enumerate(seeds):
            run_id = f"{family}_{train_seed}_{subset_seed}"
            directory = root / "runs/V2/finetune" / family / f"seed-{train_seed}-subset-{subset_seed}"
            directory.mkdir(parents=True)
            manifest = {
                "run_id": run_id,
                "family_id": family,
                "train_seed": train_seed,
                "subset_seed": subset_seed,
                "selection_pool": "D_b_sel",
                "selection_metric": "map50_95",
                "selection_metric_value": 0.25,
                "checkpoint_path": f"runs/V2/finetune/{family}/seed-{train_seed}-subset-{subset_seed}/weights/best.pt",
                "checkpoint_sha256": hashlib.sha256(run_id.encode("utf-8")).hexdigest(),
                "slurm_array_task_id": str(task_id),
            }
            (directory / "run_manifest.json").write_text(
                json.dumps(manifest, sort_keys=True), encoding="utf-8"
            )
    for task_id, (train_seed, subset_seed) in enumerate(seeds, start=9):
        receipt = root / "evidence/remote_runs/E202/19777/slurm_run" / str(19780 + task_id) / "V2-A-10-MULTI_not_applicable.json"
        receipt.parent.mkdir(parents=True)
        receipt.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "step_id": "E202",
                    "status": "not_applicable",
                    "family_id": "V2-A-10-MULTI",
                    "condition": "multi_coal",
                    "train_seed": train_seed,
                    "subset_seed": subset_seed,
                    "fallback": {
                        "reason_code": "fewer_than_two_eligible_non_target_coal_sources"
                    },
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    result = workflow_evaluate.select_checkpoints(
        root,
        {"step_id": "E204"},
        {"family": "V2"},
    )
    payload = json.loads((root / "runs/V2/checkpoint_selection.json").read_text())
    assert result["details"] == {
        "selected_repeat_cells": 9,
        "not_applicable_repeat_cells": 3,
        "resolved_repeat_cells": 12,
    }
    assert {item["family_id"] for item in payload["selected"]} == set(families)
    assert {item["family_id"] for item in payload["not_applicable"]} == {
        "V2-A-10-MULTI"
    }
    assert {item["checkpoint_availability"] for item in payload["selected"]} == {
        "remote_manifest_bound"
    }


def test_detection_ap_is_exact_for_perfect_predictions() -> None:
    truth = pd.DataFrame(
        [
            {
                "record_id": "r1",
                "raw_group_id": "g1",
                "concept_id": "worker",
                "x1": 0.0,
                "y1": 0.0,
                "x2": 10.0,
                "y2": 10.0,
            },
            {
                "record_id": "r2",
                "raw_group_id": "g2",
                "concept_id": "worker",
                "x1": 5.0,
                "y1": 5.0,
                "x2": 15.0,
                "y2": 15.0,
            },
        ]
    )
    predictions = truth.copy()
    predictions["score_calibrated"] = [0.9, 0.8]
    ap, true_positive, false_positive, false_negative = workflow_evaluate._class_ap(
        predictions,
        truth,
        concept_id="worker",
        iou_threshold=0.5,
    )
    assert ap == pytest.approx(1.0)
    assert (true_positive, false_positive, false_negative) == (2, 0, 0)
    map50, map50_95 = workflow_evaluate._map_values(predictions, truth)
    assert map50 == pytest.approx(1.0)
    assert map50_95 == pytest.approx(1.0)


def test_ece_uses_all_probability_bins() -> None:
    assert workflow_evaluate._ece([0.1, 0.9], [0, 1], bins=2) == pytest.approx(0.1)


def test_metric_row_returns_a_complete_serializable_record() -> None:
    row = workflow_evaluate._metric_row(
        package="S1",
        family="S1-GRU",
        run_id="run-1",
        repeat_key="train_seed=1701",
        metric_name="event_macro_f1",
        value=0.75,
        result_level="repeat",
        group_id="__all__",
        population_hash="a" * 64,
        prediction_lock_hash="b" * 64,
        label_manifest_hash="c" * 64,
        diagnostics={"truth_event_count": 2},
    )
    assert row["metric_value"] == pytest.approx(0.75)
    assert row["metric_definition_hash"]
    assert json.loads(row["diagnostic_counts_json"])["truth_event_count"] == 2


def test_repeat_key_skips_all_null_seed_columns() -> None:
    frame = pd.DataFrame(
        [{"run_id": "baseline", "train_seed": None, "subset_seed": None}]
    )
    assert workflow_evaluate._repeat_key(frame, "baseline") == "baseline"


def test_episode_metrics_cover_protocol_and_isolate_shortcut_controls(
    tmp_path: Path,
) -> None:
    label_rows = []
    for episode_id, truth in (("episode-a", (0, 1, 0)), ("episode-b", (0, 1, 1))):
        for step_index, event_truth in enumerate(truth):
            label_rows.append(
                {
                    "episode_id": episode_id,
                    "step_index": step_index,
                    "concept_id": "hazard",
                    "event_truth": event_truth,
                    "source_component_id": f"component-{episode_id}",
                }
            )
    labels = pd.DataFrame(label_rows)

    def predictions(
        run_id: str,
        family_id: str,
        states: tuple[str, ...],
        *,
        control_type=None,
        model_hash: str = "d" * 64,
    ) -> pd.DataFrame:
        rows = []
        for index, label in labels.iterrows():
            rows.append(
                {
                    "run_id": run_id,
                    "family_id": family_id,
                    "train_seed": 9103,
                    "control_type": control_type,
                    "episode_id": label["episode_id"],
                    "step_index": int(label["step_index"]),
                    "concept_id": label["concept_id"],
                    "score_calibrated": 0.9 if int(label["event_truth"]) else 0.1,
                    "state_prediction": states[index],
                    "abstained": states[index] == "abstain",
                    "model_hash": model_hash,
                    "policy_hash": "e" * 64,
                }
            )
        return pd.DataFrame(rows)

    main = predictions(
        "full-9103",
        "E3-FULL",
        ("normal", "alarm", "normal", "normal", "prewarning", "alarm"),
    )
    pair = predictions(
        "pair-9103",
        "E3-SHUFFLE",
        ("normal", "normal", "normal", "normal", "normal", "normal"),
        control_type="pair_alignment_shuffle",
    )
    mask = predictions(
        "mask-9103",
        "E3-FULL",
        ("normal", "attention", "normal", "normal", "attention", "normal"),
        control_type="mask_only",
    )
    (tmp_path / "predictions/locked").mkdir(parents=True)
    (tmp_path / "data/locked/episode_labels").mkdir(parents=True)
    main.to_parquet(tmp_path / "predictions/locked/episodes.parquet", index=False)
    pd.concat([pair, mask], ignore_index=True).to_parquet(
        tmp_path / "predictions/locked/episode_shortcut_controls.parquet", index=False
    )
    labels.to_parquet(
        tmp_path / "data/locked/episode_labels/D_e_te.parquet", index=False
    )

    metrics, shortcut = workflow_evaluate._episode_metrics(tmp_path, "f" * 64)
    full_repeat = metrics.loc[
        (metrics["family_id"] == "E3-FULL")
        & (metrics["result_level"] == "repeat")
    ]
    assert {
        "event_macro_f1",
        "event_precision",
        "event_recall",
        "false_alarms_per_100_episodes",
        "event_miss_rate",
        "detection_delay_steps",
        "state_flips",
        "expected_calibration_error",
        "answer_coverage",
        "abstention_rate",
        "selective_risk",
    }.issubset(set(full_repeat["metric_name"]))
    assert "E3-SHUFFLE" in set(metrics["family_id"])
    assert len(shortcut["controls"]) == 2
    assert all(
        control["same_full_checkpoint"] for control in shortcut["controls"].values()
    )
    diagnostics = json.loads(full_repeat.iloc[0]["diagnostic_counts_json"])
    assert diagnostics["eligible_step_rows"] == 6
    assert diagnostics["evaluated_episode_count"] == 2
