from __future__ import annotations

import pandas as pd
import pytest

from mining1_exp.evaluate.event_metrics import (
    EventMetricError,
    evaluate_episode_events,
    evaluate_methane_events,
    extract_events,
    macro_event_f1,
    select_episode_policy,
    select_s1_threshold,
)


def test_event_extraction_contiguity_and_merge_are_explicit() -> None:
    events = extract_events(
        [True, True, False, True, False, True],
        [0, 1, 2, 3, 4, 5],
        continuity_gap=1,
        merge_gap=2,
        event_prefix="event",
    )
    assert [(event.start, event.end) for event in events] == [(0.0, 5.0)]


def test_methane_matching_is_one_to_one_and_duplicate_alarm_is_false_alarm() -> None:
    result = evaluate_methane_events(
        raw_positions_seconds=[100, 110, 120, 130],
        raw_values=[2.0, 2.0, 0.0, 0.0],
        risk_threshold=1.0,
        prediction_positions_seconds=[80, 90, 100, 110],
        prediction_scores=[0.9, 0.0, 0.9, 0.0],
        score_threshold=0.5,
        raw_continuity_seconds=10,
        prediction_continuity_seconds=5,
        event_merge_gap_seconds=0,
        horizon_seconds=30,
        valid_observed_sensor_hours=2.0,
    )
    assert result["truth_event_count"] == 1
    assert result["prediction_event_count"] == 2
    assert result["matched_event_count"] == 1
    assert result["false_alarm_count"] == 1
    assert result["median_lead_time_seconds"] == 20.0
    assert result["lead_time_iqr_seconds"] == 0.0
    assert result["event_miss_rate"] == 0.0


def test_ongoing_early_alarm_is_not_relabelled_as_detection() -> None:
    result = evaluate_methane_events(
        raw_positions_seconds=[100, 110, 120],
        raw_values=[2.0, 2.0, 0.0],
        risk_threshold=1.0,
        prediction_positions_seconds=[60, 65, 70, 75],
        prediction_scores=[0.9, 0.9, 0.9, 0.9],
        score_threshold=0.5,
        raw_continuity_seconds=10,
        prediction_continuity_seconds=5,
        event_merge_gap_seconds=0,
        horizon_seconds=30,
        valid_observed_sensor_hours=1.0,
    )
    assert result["matched_event_count"] == 0
    assert result["missed_event_count"] == 1
    assert result["false_alarm_count"] == 1
    assert result["event_f1"] == 0.0


def test_episode_abstention_breaks_events_and_coverage_is_not_hidden() -> None:
    result = evaluate_episode_events(
        truth_event=[0, 1, 1, 0, 0],
        effective_states=["prewarning", "prewarning", "abstain", "normal", "normal"],
        episode_ids=["episode_0"] * 5,
        step_indices=[0, 1, 2, 3, 4],
    )
    assert result["matched_event_count"] == 0
    assert result["missed_event_count"] == 1
    assert result["false_alarm_count"] == 1
    assert result["answer_coverage"] == pytest.approx(0.8)
    assert result["abstention_rate"] == pytest.approx(0.2)

    all_abstain = evaluate_episode_events(
        truth_event=[0, 1, 1],
        effective_states=["abstain", "abstain", "abstain"],
        episode_ids=["episode_0"] * 3,
        step_indices=[0, 1, 2],
    )
    assert all_abstain["answer_coverage"] == 0.0
    assert all_abstain["event_f1"] == 0.0
    assert all_abstain["missed_event_count"] == 1


def test_no_truth_units_stay_explicit_and_are_excluded_from_macro_f1() -> None:
    no_truth = evaluate_episode_events(
        truth_event=[0, 0, 0],
        effective_states=["normal", "prewarning", "normal"],
        episode_ids=["episode_0"] * 3,
        step_indices=[0, 1, 2],
    )
    with_truth = evaluate_episode_events(
        truth_event=[0, 1, 0],
        effective_states=["normal", "prewarning", "normal"],
        episode_ids=["episode_0"] * 3,
        step_indices=[0, 1, 2],
    )
    assert no_truth["eligible_for_macro_f1"] is False
    assert no_truth["event_f1"] is None
    assert no_truth["false_alarm_count"] == 1
    assert macro_event_f1([no_truth, with_truth]) == 1.0
    empty = evaluate_episode_events(
        truth_event=[0, 0, 0],
        effective_states=["normal", "normal", "normal"],
        episode_ids=["episode_0"] * 3,
        step_indices=[0, 1, 2],
    )
    assert empty["truth_event_count"] == 0
    assert empty["prediction_event_count"] == 0
    assert empty["event_f1"] is None
    with pytest.raises(EventMetricError, match="no eligible"):
        macro_event_f1([no_truth])


def test_episode_boundaries_prevent_cross_episode_events_and_state_flips() -> None:
    result = evaluate_episode_events(
        truth_event=[1, 1],
        effective_states=["alarm", "alarm"],
        episode_ids=["episode_a", "episode_b"],
        step_indices=[0, 0],
    )
    assert result["episode_count"] == 2
    assert result["truth_event_count"] == 2
    assert result["prediction_event_count"] == 2
    assert result["matched_event_count"] == 2
    assert result["state_flip_count"] == 0
    assert result["mean_state_flips_per_episode"] == 0.0

    with pytest.raises(EventMetricError, match="strictly increasing"):
        evaluate_episode_events(
            truth_event=[0, 1],
            effective_states=["normal", "alarm"],
            episode_ids=["episode_a", "episode_a"],
            step_indices=[1, 0],
        )


def test_infeasible_threshold_and_policy_return_blocked_diagnostic_only() -> None:
    s1 = select_s1_threshold(
        pd.DataFrame(
            [
                {"candidate_id": "a", "threshold": 0.5, "false_alarms_per_hour": 2.0, "event_macro_f1": 0.8, "event_miss_rate": 0.1},
                {"candidate_id": "b", "threshold": 0.6, "false_alarms_per_hour": 1.0, "event_macro_f1": 0.7, "event_miss_rate": 0.2},
            ]
        ),
        pool="D_b_sel",
        max_false_alarms_per_hour=0.5,
    )
    assert s1.status == "diagnostic_only_blocked"
    assert s1.selected_id is None
    assert s1.diagnostic_id == "b"

    policy = select_episode_policy(
        pd.DataFrame(
            [
                {"candidate_id": "p", "beta": 0.70, "memory_k": 2, "low_threshold": 0.30, "high_threshold": 0.60, "alarm_threshold": 0.80, "abstention_threshold": 0.10, "answer_coverage": 0.5, "false_alarms_per_100_episodes": 10.0, "event_miss_rate": 0.5, "event_macro_f1": 0.4, "mean_detection_delay_steps": 3.0, "mean_state_flips_per_episode": 5.0, "complexity_rank": 0}
            ]
        ),
        pool="D_e_pol",
        target_answer_coverage=0.9,
        max_coverage_deviation=0.02,
        max_false_alarms_per_100_episodes=5.0,
        max_event_miss_rate=0.2,
    )
    assert policy.status == "diagnostic_only_blocked"
    assert policy.selected_id is None
