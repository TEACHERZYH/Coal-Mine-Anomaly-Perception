from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mining1_exp.episode_data import EpisodeArrays
from mining1_exp.predict import episode_sealed
from mining1_exp.predict.episode_sealed import (
    _pair_alignment_shuffle_arrays,
    _score_shuffle_arrays,
)
from mining1_exp.workflow_common import WorkflowExecutionError


def _episode_arrays() -> EpisodeArrays:
    count = 6
    probabilities = np.asarray(
        [
            [0.05, 0.15, 0.25],
            [0.10, 0.20, 0.30],
            [0.35, 0.45, 0.55],
            [0.40, 0.50, 0.60],
            [0.65, 0.75, 0.85],
            [0.70, 0.80, 0.90],
        ],
        dtype=np.float32,
    )
    availability = np.ones_like(probabilities, dtype=bool)
    quality = np.stack(
        [np.abs(probabilities - 0.5), availability.astype(np.float32)], axis=-1
    )
    return EpisodeArrays(
        pool="D_e_te",
        metadata=pd.DataFrame(
            {
                "episode_id": ["episode-1"] * count,
                "step_index": list(range(count)),
                "concept_id": ["worker_presence"] * count,
            }
        ),
        probabilities=probabilities,
        quality=quality,
        availability=availability,
        concept_embedding=np.ones((count, 1), dtype=np.float32),
        labels=None,
        edge_index=np.asarray([[0, 1], [1, 2]], dtype=np.int64),
        pair_features=np.ones((count, 2, 1), dtype=np.float32),
        edge_validity=np.ones((count, 2), dtype=bool),
        node_ids=("anchor", "branch-b", "branch-c"),
        concept_ids=("worker_presence",),
        quality_feature_names=("margin", "available"),
        pair_feature_names=("observed_pair",),
    )


def test_pair_alignment_shuffle_is_distinct_and_preserves_node_marginals() -> None:
    arrays = _episode_arrays()
    pair_shuffle = _pair_alignment_shuffle_arrays(arrays, seed=9103)
    score_shuffle = _score_shuffle_arrays(arrays, seed=9103)

    np.testing.assert_array_equal(
        pair_shuffle.probabilities[:, 0], arrays.probabilities[:, 0]
    )
    for node_index in (1, 2):
        assert np.all(
            pair_shuffle.probabilities[:, node_index]
            != arrays.probabilities[:, node_index]
        )
        np.testing.assert_array_equal(
            np.sort(pair_shuffle.probabilities[:, node_index]),
            np.sort(arrays.probabilities[:, node_index]),
        )
    for node_index in range(3):
        assert np.all(
            score_shuffle.probabilities[:, node_index]
            != arrays.probabilities[:, node_index]
        )
    assert not np.array_equal(
        pair_shuffle.probabilities, score_shuffle.probabilities
    )
    np.testing.assert_array_equal(pair_shuffle.availability, arrays.availability)
    np.testing.assert_array_equal(pair_shuffle.edge_validity, arrays.edge_validity)


def test_infeasible_episode_policy_preserves_diagnostic_evidence(
    tmp_path, monkeypatch
) -> None:
    arrays = _episode_arrays()
    arrays = EpisodeArrays(
        **{
            **arrays.__dict__,
            "pool": "D_e_pol",
            "labels": np.asarray([0, 0, 1, 1, 0, 0], dtype=np.int64),
        }
    )
    candidate = pd.DataFrame.from_records(
        [
            {
                "candidate_id": "candidate-1",
                "beta": 0.70,
                "memory_k": 2,
                "low_threshold": 0.30,
                "high_threshold": 0.60,
                "alarm_threshold": 0.80,
                "abstention_threshold": 0.10,
            }
        ]
    )
    monkeypatch.setattr(episode_sealed, "locked_episode_policy_grid", lambda: candidate)
    monkeypatch.setattr(
        episode_sealed,
        "_apply_policy",
        lambda *args, **kwargs: (["normal"] * 6, [{}] * 6, np.zeros(6, dtype=bool)),
    )
    monkeypatch.setattr(
        episode_sealed,
        "_aggregate_policy_metrics",
        lambda *args, **kwargs: {
            "answer_coverage": 0.50,
            "false_alarms_per_100_episodes": 20.0,
            "event_miss_rate": 1.0,
            "event_macro_f1": 0.0,
            "mean_detection_delay_steps": 32.0,
            "mean_state_flips_per_episode": 0.0,
        },
    )
    protocol = {
        "episodes": {
            "target_answer_coverage": 0.90,
            "policy_calibration": {
                "max_target_answer_coverage_absolute_deviation": 0.02
            },
            "policy_constraints": {
                "max_false_alarms_per_100_episodes": 10.0,
                "max_event_miss_rate": 0.20,
            },
        }
    }

    with pytest.raises(WorkflowExecutionError, match="infeasible"):
        episode_sealed._fit_policy(
            tmp_path,
            lock_id="blocked",
            model_hash="model-hash",
            arrays=arrays,
            inference={},
            protocol=protocol,
        )

    candidate_path = tmp_path / "runs/E2_E3/policies/blocked_candidates.parquet"
    receipt_path = tmp_path / "runs/E2_E3/policies/blocked.json"
    assert candidate_path.is_file()
    payload = pd.read_json(receipt_path, typ="series")
    assert payload["status"] == "diagnostic_only_blocked"
    assert payload["selected_policy"] is None
    assert payload["diagnostic_candidate"]["candidate_id"] == "candidate-1"
    assert payload["test_information_used"] is False
