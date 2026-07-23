from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from ..models.episode_fusion import (
    POLICY_ABSTENTION_THRESHOLDS,
    POLICY_ALARM_THRESHOLD_CANDIDATES,
    POLICY_BETA_CANDIDATES,
    POLICY_HYSTERESIS_PAIRS,
    POLICY_MEMORY_K_CANDIDATES,
)


class EventMetricError(ValueError):
    """Raised when an event metric or policy selection violates its protocol."""


@dataclass(frozen=True)
class EventInterval:
    event_id: str
    start: float
    end: float


@dataclass(frozen=True)
class EventMatch:
    prediction_event_id: str
    truth_event_id: str
    prediction_onset: float
    truth_start: float
    truth_end: float


@dataclass(frozen=True)
class EventCounts:
    truth_event_count: int
    prediction_event_count: int
    matched_event_count: int
    missed_event_count: int
    false_alarm_count: int
    event_precision: Optional[float]
    event_recall: Optional[float]
    event_f1: Optional[float]
    event_miss_rate: Optional[float]
    eligible_for_macro_f1: bool


@dataclass(frozen=True)
class SelectionResult:
    status: str
    selected_id: Optional[str]
    diagnostic_id: str
    reason: str


def extract_events(
    positive: Sequence[bool],
    positions: Sequence[float],
    *,
    continuity_gap: float,
    merge_gap: float = 0.0,
    event_prefix: str,
) -> list[EventInterval]:
    flags = np.asarray(positive, dtype=bool)
    coordinates = np.asarray(positions, dtype=np.float64)
    if flags.ndim != 1 or coordinates.shape != flags.shape or len(flags) == 0:
        raise EventMetricError("event flags and positions must be aligned non-empty vectors")
    if not np.isfinite(coordinates).all() or np.any(np.diff(coordinates) <= 0):
        raise EventMetricError("event positions must be finite and strictly increasing")
    if continuity_gap <= 0 or merge_gap < 0:
        raise EventMetricError("event continuity and merge gaps are invalid")

    positive_positions = coordinates[flags]
    if positive_positions.size == 0:
        return []
    contiguous: list[tuple[float, float]] = []
    start = end = float(positive_positions[0])
    for position in positive_positions[1:]:
        position_value = float(position)
        if position_value - end <= continuity_gap:
            end = position_value
        else:
            contiguous.append((start, end))
            start = end = position_value
    contiguous.append((start, end))

    merged: list[tuple[float, float]] = []
    for interval_start, interval_end in contiguous:
        if merged and interval_start - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], interval_end)
        else:
            merged.append((interval_start, interval_end))
    return [
        EventInterval(f"{event_prefix}_{index}", interval_start, interval_end)
        for index, (interval_start, interval_end) in enumerate(merged)
    ]


def match_events_one_to_one(
    truth_events: Sequence[EventInterval],
    prediction_events: Sequence[EventInterval],
    *,
    truth_window_lookback: float,
) -> tuple[list[EventMatch], set[str], set[str]]:
    if truth_window_lookback < 0:
        raise EventMetricError("truth match-window lookback must be nonnegative")
    truth = sorted(truth_events, key=lambda event: (event.end, event.event_id))
    predictions = sorted(
        prediction_events, key=lambda event: (event.start, event.event_id)
    )
    unmatched_truth = {event.event_id for event in truth}
    matched_predictions = set()
    matches = []
    for prediction in predictions:
        candidates = [
            event
            for event in truth
            if event.event_id in unmatched_truth
            and event.start - truth_window_lookback <= prediction.start <= event.end
        ]
        if not candidates:
            continue
        selected = min(candidates, key=lambda event: (event.end, event.event_id))
        unmatched_truth.remove(selected.event_id)
        matched_predictions.add(prediction.event_id)
        matches.append(
            EventMatch(
                prediction.event_id,
                selected.event_id,
                prediction.start,
                selected.start,
                selected.end,
            )
        )
    unmatched_predictions = {
        event.event_id for event in predictions
    }.difference(matched_predictions)
    return matches, unmatched_truth, unmatched_predictions


def summarize_event_counts(
    truth_events: Sequence[EventInterval],
    prediction_events: Sequence[EventInterval],
    matches: Sequence[EventMatch],
) -> EventCounts:
    truth_count = len(truth_events)
    prediction_count = len(prediction_events)
    matched_count = len(matches)
    missed_count = truth_count - matched_count
    false_alarm_count = prediction_count - matched_count
    if truth_count == 0:
        precision = (
            float(matched_count / prediction_count) if prediction_count else None
        )
        recall = None
        f1 = None
        eligible = False
    else:
        precision = float(matched_count / prediction_count) if prediction_count else 0.0
        recall = float(matched_count / truth_count)
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        eligible = True
    return EventCounts(
        truth_event_count=truth_count,
        prediction_event_count=prediction_count,
        matched_event_count=matched_count,
        missed_event_count=missed_count,
        false_alarm_count=false_alarm_count,
        event_precision=precision,
        event_recall=recall,
        event_f1=f1,
        event_miss_rate=(float(missed_count / truth_count) if truth_count else None),
        eligible_for_macro_f1=eligible,
    )


def evaluate_methane_events(
    *,
    raw_positions_seconds: Sequence[float],
    raw_values: Sequence[float],
    risk_threshold: float,
    prediction_positions_seconds: Sequence[float],
    prediction_scores: Sequence[float],
    score_threshold: float,
    raw_continuity_seconds: float,
    prediction_continuity_seconds: float,
    event_merge_gap_seconds: float,
    horizon_seconds: float,
    valid_observed_sensor_hours: float,
) -> dict[str, Any]:
    raw = np.asarray(raw_values, dtype=np.float64)
    scores = np.asarray(prediction_scores, dtype=np.float64)
    if not np.isfinite(raw).all() or not np.isfinite(scores).all():
        raise EventMetricError("methane truth values and prediction scores must be finite")
    if risk_threshold <= 0 or not 0 <= score_threshold <= 1:
        raise EventMetricError("methane thresholds are invalid")
    if valid_observed_sensor_hours <= 0 or horizon_seconds < 0:
        raise EventMetricError("methane exposure hours and horizon are invalid")
    truth_events = extract_events(
        raw >= risk_threshold,
        raw_positions_seconds,
        continuity_gap=raw_continuity_seconds,
        merge_gap=event_merge_gap_seconds,
        event_prefix="truth",
    )
    prediction_events = extract_events(
        scores >= score_threshold,
        prediction_positions_seconds,
        continuity_gap=prediction_continuity_seconds,
        merge_gap=event_merge_gap_seconds,
        event_prefix="prediction",
    )
    matches, unmatched_truth, unmatched_predictions = match_events_one_to_one(
        truth_events,
        prediction_events,
        truth_window_lookback=horizon_seconds,
    )
    counts = summarize_event_counts(truth_events, prediction_events, matches)
    lead_times = [match.truth_start - match.prediction_onset for match in matches]
    lead_q25 = float(np.quantile(lead_times, 0.25)) if lead_times else None
    lead_q75 = float(np.quantile(lead_times, 0.75)) if lead_times else None
    return {
        **asdict(counts),
        "false_alarms_per_hour": counts.false_alarm_count
        / valid_observed_sensor_hours,
        "lead_times_seconds": lead_times,
        "median_lead_time_seconds": (
            float(np.median(lead_times)) if lead_times else None
        ),
        "lead_time_q25_seconds": lead_q25,
        "lead_time_q75_seconds": lead_q75,
        "lead_time_iqr_seconds": (
            lead_q75 - lead_q25 if lead_times else None
        ),
        "valid_observed_sensor_hours": float(valid_observed_sensor_hours),
        "unmatched_truth_event_ids": sorted(unmatched_truth),
        "unmatched_prediction_event_ids": sorted(unmatched_predictions),
        "matches": [asdict(match) for match in matches],
    }


def evaluate_episode_events(
    *,
    truth_event: Sequence[int],
    effective_states: Sequence[str],
    episode_ids: Sequence[str],
    step_indices: Sequence[int],
) -> dict[str, Any]:
    truth = np.asarray(truth_event, dtype=np.int64)
    states = [str(value) for value in effective_states]
    raw_episode_ids = list(episode_ids)
    steps = np.asarray(step_indices, dtype=np.float64)
    if (
        truth.ndim != 1
        or len(truth) != len(states)
        or len(truth) != len(raw_episode_ids)
        or steps.shape != truth.shape
        or len(truth) == 0
    ):
        raise EventMetricError(
            "episode truth, states, IDs, and step indices must be aligned non-empty vectors"
        )
    if not set(truth).issubset({0, 1}):
        raise EventMetricError("episode truth must be binary")
    if not set(states).issubset({"normal", "attention", "prewarning", "alarm", "abstain"}):
        raise EventMetricError("episode contains an unknown effective state")
    if not np.isfinite(steps).all() or np.any(steps < 0) or np.any(steps != np.floor(steps)):
        raise EventMetricError("episode step indices must be finite nonnegative integers")
    if any(not isinstance(value, str) or not value.strip() for value in raw_episode_ids):
        raise EventMetricError("episode IDs must be non-empty strings")

    normalized_ids = np.asarray([value.strip() for value in raw_episode_ids], dtype=object)
    ordered_ids = list(dict.fromkeys(normalized_ids.tolist()))
    truth_events: list[EventInterval] = []
    prediction_events: list[EventInterval] = []
    matches: list[EventMatch] = []
    unmatched_truth: set[str] = set()
    unmatched_predictions: set[str] = set()
    state_flips_per_episode: list[int] = []
    for episode_order, episode_id in enumerate(ordered_ids):
        row_ids = np.flatnonzero(normalized_ids == episode_id)
        episode_steps = steps[row_ids]
        if np.any(np.diff(episode_steps) <= 0):
            raise EventMetricError(
                "step indices must be strictly increasing within each episode"
            )
        episode_states = [states[index] for index in row_ids]
        episode_truth_events = extract_events(
            truth[row_ids] == 1,
            episode_steps,
            continuity_gap=1.0,
            event_prefix=f"truth_e{episode_order}",
        )
        predicted_positive = np.asarray(
            [state in {"prewarning", "alarm"} for state in episode_states], dtype=bool
        )
        episode_prediction_events = extract_events(
            predicted_positive,
            episode_steps,
            continuity_gap=1.0,
            event_prefix=f"prediction_e{episode_order}",
        )
        episode_matches, episode_unmatched_truth, episode_unmatched_predictions = (
            match_events_one_to_one(
                episode_truth_events,
                episode_prediction_events,
                truth_window_lookback=0.0,
            )
        )
        truth_events.extend(episode_truth_events)
        prediction_events.extend(episode_prediction_events)
        matches.extend(episode_matches)
        unmatched_truth.update(episode_unmatched_truth)
        unmatched_predictions.update(episode_unmatched_predictions)
        state_flips_per_episode.append(
            sum(
                left != right
                for left, right in zip(episode_states, episode_states[1:])
            )
        )

    counts = summarize_event_counts(truth_events, prediction_events, matches)
    delays = [match.prediction_onset - match.truth_start for match in matches]
    nonabstained = sum(state != "abstain" for state in states)
    answer_coverage = nonabstained / len(states)
    episode_count = len(ordered_ids)
    state_flip_count = sum(state_flips_per_episode)
    return {
        **asdict(counts),
        "false_alarms_per_100_episodes": counts.false_alarm_count
        * 100.0
        / episode_count,
        "detection_delays_steps": delays,
        "mean_detection_delay_steps": float(np.mean(delays)) if delays else None,
        "episode_count": int(episode_count),
        "eligible_step_count": len(states),
        "nonabstained_step_count": nonabstained,
        "abstained_step_count": len(states) - nonabstained,
        "answer_coverage": answer_coverage,
        "abstention_rate": 1.0 - answer_coverage,
        "state_flip_count": state_flip_count,
        "state_flips_per_episode": state_flips_per_episode,
        "mean_state_flips_per_episode": float(np.mean(state_flips_per_episode)),
        "unmatched_truth_event_ids": sorted(unmatched_truth),
        "unmatched_prediction_event_ids": sorted(unmatched_predictions),
        "matches": [asdict(match) for match in matches],
    }


def macro_event_f1(unit_results: Iterable[dict[str, Any]]) -> float:
    values = [
        float(result["event_f1"])
        for result in unit_results
        if result.get("eligible_for_macro_f1") and result.get("event_f1") is not None
    ]
    if not values:
        raise EventMetricError("macro event F1 has no eligible truth-bearing units")
    return float(np.mean(values))


def select_s1_threshold(
    candidate_metrics: pd.DataFrame,
    *,
    pool: str,
    max_false_alarms_per_hour: float,
) -> SelectionResult:
    if pool != "D_b_sel":
        raise EventMetricError("S1 threshold selection may use only D_b_sel")
    required = {
        "candidate_id",
        "threshold",
        "false_alarms_per_hour",
        "event_macro_f1",
        "event_miss_rate",
    }
    missing = sorted(required.difference(candidate_metrics.columns))
    if missing or candidate_metrics.empty:
        raise EventMetricError(f"S1 candidate metrics are incomplete: {missing}")
    table = candidate_metrics.copy()
    table["candidate_id"] = table["candidate_id"].astype(str)
    if table["candidate_id"].str.strip().eq("").any() or table["candidate_id"].duplicated().any():
        raise EventMetricError("S1 candidate IDs must be non-empty and unique")
    for column in required.difference({"candidate_id"}):
        table[column] = pd.to_numeric(table[column], errors="raise")
    numeric_values = table[list(required.difference({"candidate_id"}))].to_numpy(
        dtype=np.float64
    )
    if (
        not np.isfinite(numeric_values).all()
        or not np.isfinite(max_false_alarms_per_hour)
        or max_false_alarms_per_hour < 0
    ):
        raise EventMetricError("S1 candidate metrics must be finite and nonnegative where required")
    threshold_grid = np.round(np.arange(0.10, 0.9001, 0.05), 2)
    if not table["threshold"].map(
        lambda value: bool(np.isclose(float(value), threshold_grid).any())
    ).all():
        raise EventMetricError("S1 candidate threshold is outside the locked grid")
    if (
        (table["false_alarms_per_hour"] < 0).any()
        or ((table["event_macro_f1"] < 0) | (table["event_macro_f1"] > 1)).any()
        or ((table["event_miss_rate"] < 0) | (table["event_miss_rate"] > 1)).any()
    ):
        raise EventMetricError("S1 candidate metrics are outside valid ranges")
    feasible = table.loc[
        table["false_alarms_per_hour"] <= max_false_alarms_per_hour
    ]
    diagnostic = table.sort_values(
        ["false_alarms_per_hour", "event_miss_rate", "threshold", "candidate_id"],
        ascending=[True, True, False, True],
    ).iloc[0]
    if feasible.empty:
        return SelectionResult(
            "diagnostic_only_blocked",
            None,
            str(diagnostic["candidate_id"]),
            "no candidate satisfies the locked false-alarm constraint",
        )
    selected = feasible.sort_values(
        ["event_macro_f1", "event_miss_rate", "threshold", "candidate_id"],
        ascending=[False, True, False, True],
    ).iloc[0]
    return SelectionResult(
        "pass",
        str(selected["candidate_id"]),
        str(diagnostic["candidate_id"]),
        "selected by feasibility, event macro F1, miss rate, and higher threshold",
    )


def select_episode_policy(
    candidate_metrics: pd.DataFrame,
    *,
    pool: str,
    target_answer_coverage: float,
    max_coverage_deviation: float,
    max_false_alarms_per_100_episodes: float,
    max_event_miss_rate: float,
) -> SelectionResult:
    if pool != "D_e_pol":
        raise EventMetricError("episode policy selection may use only D_e_pol")
    required = {
        "candidate_id",
        "beta",
        "memory_k",
        "low_threshold",
        "high_threshold",
        "alarm_threshold",
        "abstention_threshold",
        "answer_coverage",
        "false_alarms_per_100_episodes",
        "event_miss_rate",
        "event_macro_f1",
        "mean_detection_delay_steps",
        "mean_state_flips_per_episode",
        "complexity_rank",
    }
    missing = sorted(required.difference(candidate_metrics.columns))
    if missing or candidate_metrics.empty:
        raise EventMetricError(f"episode policy candidates are incomplete: {missing}")
    table = candidate_metrics.copy()
    table["candidate_id"] = table["candidate_id"].astype(str)
    if table["candidate_id"].str.strip().eq("").any() or table["candidate_id"].duplicated().any():
        raise EventMetricError("episode policy candidate IDs must be non-empty and unique")
    numeric = required.difference({"candidate_id"})
    for column in numeric:
        table[column] = pd.to_numeric(table[column], errors="raise")
    numeric_values = table[list(numeric)].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric_values).all():
        raise EventMetricError("episode policy metrics must be finite")
    locked_beta = table["beta"].map(
        lambda value: bool(np.isclose(float(value), POLICY_BETA_CANDIDATES).any())
    )
    locked_memory_k = table["memory_k"].isin(POLICY_MEMORY_K_CANDIDATES) & (
        table["memory_k"] == np.floor(table["memory_k"])
    )
    locked_hysteresis = table.apply(
        lambda row: any(
            np.isclose(float(row["low_threshold"]), low)
            and np.isclose(float(row["high_threshold"]), high)
            for low, high in POLICY_HYSTERESIS_PAIRS
        ),
        axis=1,
    )
    locked_alarm = table["alarm_threshold"].map(
        lambda value: bool(
            np.isclose(float(value), POLICY_ALARM_THRESHOLD_CANDIDATES).any()
        )
    )
    locked_abstention = table["abstention_threshold"].map(
        lambda value: bool(
            np.isclose(float(value), POLICY_ABSTENTION_THRESHOLDS).any()
        )
    )
    if not (
        locked_beta
        & locked_memory_k
        & locked_hysteresis
        & locked_alarm
        & locked_abstention
    ).all():
        raise EventMetricError("episode policy candidate is outside the locked grid")
    constraints = np.asarray(
        [
            target_answer_coverage,
            max_coverage_deviation,
            max_false_alarms_per_100_episodes,
            max_event_miss_rate,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(constraints).all():
        raise EventMetricError("episode policy constraints must be finite")
    if not 0 <= target_answer_coverage <= 1 or not 0 <= max_coverage_deviation <= 1:
        raise EventMetricError("episode policy coverage targets are invalid")
    if max_false_alarms_per_100_episodes < 0 or not 0 <= max_event_miss_rate <= 1:
        raise EventMetricError("episode policy constraints are invalid")
    if (
        ((table["answer_coverage"] < 0) | (table["answer_coverage"] > 1)).any()
        or (table["false_alarms_per_100_episodes"] < 0).any()
        or ((table["event_miss_rate"] < 0) | (table["event_miss_rate"] > 1)).any()
        or ((table["event_macro_f1"] < 0) | (table["event_macro_f1"] > 1)).any()
        or (table["mean_detection_delay_steps"] < 0).any()
        or (table["mean_state_flips_per_episode"] < 0).any()
        or (table["complexity_rank"] < 0).any()
        or (table["complexity_rank"] != np.floor(table["complexity_rank"])).any()
    ):
        raise EventMetricError("episode policy metrics are outside valid ranges")
    coverage_ok = (
        table["answer_coverage"] - target_answer_coverage
    ).abs() <= max_coverage_deviation
    constraints_ok = (
        table["false_alarms_per_100_episodes"]
        <= max_false_alarms_per_100_episodes
    ) & (table["event_miss_rate"] <= max_event_miss_rate)
    feasible = table.loc[coverage_ok & constraints_ok]
    diagnostic = table.sort_values(
        ["event_miss_rate", "false_alarms_per_100_episodes", "candidate_id"],
        ascending=[True, True, True],
    ).iloc[0]
    if feasible.empty:
        return SelectionResult(
            "diagnostic_only_blocked",
            None,
            str(diagnostic["candidate_id"]),
            "no policy satisfies coverage and event constraints",
        )
    selected = feasible.sort_values(
        [
            "event_macro_f1",
            "mean_detection_delay_steps",
            "mean_state_flips_per_episode",
            "complexity_rank",
            "candidate_id",
        ],
        ascending=[False, True, True, True, True],
    ).iloc[0]
    return SelectionResult(
        "pass",
        str(selected["candidate_id"]),
        str(diagnostic["candidate_id"]),
        "selected by constraints, macro F1, delay, state flips, and simplicity",
    )
