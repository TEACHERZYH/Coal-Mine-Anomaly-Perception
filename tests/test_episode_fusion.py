from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from mining1_exp.evaluate.event_metrics import (
    EventMetricError,
    evaluate_episode_events,
    select_episode_policy,
    select_s1_threshold,
)
from mining1_exp.models.episode_fusion import (
    CalibratedLogitFusion,
    EventMemoryPolicy,
    MemoryPolicyConfig,
    ModelContractError,
    RobustQualityScaler,
    ReliabilityGraphFusion,
    assert_fusion_feature_names,
    derange_values_within_strata,
    derange_temporal_steps,
    locked_episode_policy_grid,
    mask_only_control,
    mean_fusion,
    reliability_brier_loss,
    reliability_supervision,
    validate_observed_pair_edges,
    validate_same_checkpoint,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _edge_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"source_node": 0, "target_node": 1, "edge_provenance": "observed_pair"},
            {"source_node": 1, "target_node": 0, "edge_provenance": "observed_pair"},
        ]
    )


def test_fusion_constants_match_the_frozen_protocol() -> None:
    protocol = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "protocol_lock.template.yaml").read_text(
            encoding="utf-8"
        )
    )
    episode = protocol["models"]["episode"]
    assert episode["no_train_baselines"] == ["mean"]
    assert episode["trainable_baselines"] == ["calibrated_logit"]
    assert episode["ablations"] == ["no_reliability", "no_graph", "no_memory"]
    assert episode["mechanism_controls"] == ["pair_alignment_shuffle_inference"]
    assert episode["graph_layers"] == 2
    assert episode["probability_clip_epsilon"] == pytest.approx(1.0e-6)
    assert episode["inference_policy_controls_require_same_checkpoint_hash_as_full"] is True
    policy = protocol["episodes"]["policy_calibration"]
    grid = locked_episode_policy_grid()
    assert len(grid) == 136
    assert sorted(grid["beta"].unique()) == policy["beta_candidates"]
    assert sorted(grid["memory_k"].unique()) == policy["memory_k_candidates"]
    assert sorted(grid["alarm_threshold"].unique()) == policy["alarm_threshold_candidates"]
    assert grid["policy_id"].is_unique


def test_mean_and_calibrated_logit_baselines_respect_availability() -> None:
    probabilities = torch.tensor([[0.2, 0.8], [0.7, float("nan")]])
    availability = torch.tensor([[True, True], [True, False]])
    mean_probability, mean_abstained = mean_fusion(probabilities, availability)
    assert torch.allclose(mean_probability, torch.tensor([0.5, 0.7]))
    assert not mean_abstained.any()

    logit = CalibratedLogitFusion(node_count=2)
    logit_probability, logit_abstained = logit(probabilities, availability)
    assert logit_probability.shape == (2,)
    assert torch.isfinite(logit_probability).all()
    assert not logit_abstained.any()

    empty_probability, empty_abstained = mean_fusion(
        torch.full((1, 2), float("nan")), torch.zeros(1, 2, dtype=torch.bool)
    )
    assert empty_abstained.tolist() == [True]
    assert empty_probability.tolist() == [0.5]


def test_full_graph_reliability_and_structural_ablations_forward() -> None:
    torch.manual_seed(1701)
    edge_index = validate_observed_pair_edges(_edge_frame(), node_count=2)
    probabilities = torch.tensor([[0.8, 0.3], [0.6, float("nan")]])
    quality = torch.tensor(
        [
            [[0.2, 0.5, 1.0], [0.8, 0.1, 1.0]],
            [[0.4, 0.3, 1.0], [float("nan"), float("nan"), float("nan")]],
        ]
    )
    availability = torch.tensor([[True, True], [True, False]])
    concept_embedding = torch.tensor([[0.1, 0.2], [0.2, 0.3]])
    pair_features = torch.tensor(
        [
            [[0.5, 0.1], [0.5, 0.1]],
            [[0.4, 0.2], [0.4, 0.2]],
        ]
    )
    scaler = RobustQualityScaler(
        ["quality_mean", "quality_spread", "quality_valid"]
    ).fit(quality.numpy(), pool="D_e_tr")
    scaled_quality, finite_mask = scaler.transform(quality.numpy())
    assert scaled_quality.shape == quality.shape
    assert finite_mask.shape == quality.shape
    assert np.isfinite(scaled_quality).all()
    with pytest.raises(ModelContractError, match="only on D_e_tr"):
        RobustQualityScaler(
            ["quality_mean", "quality_spread", "quality_valid"]
        ).fit(quality.numpy(), pool="D_e_te")
    full = ReliabilityGraphFusion(
        node_count=2,
        quality_feature_names=["quality_mean", "quality_spread", "quality_valid"],
        concept_dim=2,
        pair_feature_names=["pair_time_delta", "pair_quality"],
        graph_layers=2,
        use_graph=True,
        use_reliability=True,
    )
    output = full(
        probabilities,
        quality,
        availability,
        concept_embedding,
        edge_index=edge_index,
        pair_features=pair_features,
        abstention_threshold=0.10,
    )
    assert output.probability.shape == (2,)
    assert torch.isfinite(output.probability).all()
    assert torch.allclose(output.reliability_weights.sum(dim=1), torch.ones(2))
    assert output.edge_weights.shape == (2, 2, 2)
    assert output.message_norms.shape == (2, 2, 2)
    assert torch.count_nonzero(output.message_norms[0]) > 0

    targets = reliability_supervision(
        torch.tensor([1.0, 0.0]), probabilities, availability, pool="D_e_tr"
    )
    loss = reliability_brier_loss(output.reliability, targets, availability)
    (output.probability.sum() + loss).backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in full.parameters()
    )
    with pytest.raises(ModelContractError, match="only in D_e_tr"):
        reliability_supervision(
            torch.tensor([1.0, 0.0]), probabilities, availability, pool="D_e_te"
        )

    no_reliability = ReliabilityGraphFusion(
        node_count=2,
        quality_feature_names=["quality_mean", "quality_spread", "quality_valid"],
        concept_dim=2,
        pair_feature_names=["pair_time_delta", "pair_quality"],
        use_graph=True,
        use_reliability=False,
    )
    no_rel_output = no_reliability(
        probabilities,
        quality,
        availability,
        concept_embedding,
        edge_index=edge_index,
        pair_features=pair_features,
        abstention_threshold=0.10,
    )
    assert torch.allclose(no_rel_output.reliability_weights[0], torch.tensor([0.5, 0.5]))

    no_graph = ReliabilityGraphFusion(
        node_count=2,
        quality_feature_names=["quality_mean", "quality_spread", "quality_valid"],
        concept_dim=2,
        pair_feature_names=["pair_time_delta", "pair_quality"],
        use_graph=False,
        use_reliability=True,
    )
    no_graph_output = no_graph(
        probabilities,
        quality,
        availability,
        concept_embedding,
        abstention_threshold=0.10,
    )
    assert no_graph_output.edge_weights.numel() == 0
    assert sum(parameter.numel() for parameter in no_graph.parameters()) == sum(
        parameter.numel() for parameter in full.parameters()
    )

    all_missing = full(
        torch.full((1, 2), float("nan")),
        torch.full((1, 2, 3), float("nan")),
        torch.zeros(1, 2, dtype=torch.bool),
        torch.tensor([[0.1, 0.2]]),
        edge_index=edge_index,
        pair_features=torch.zeros(1, 2, 2),
        abstention_threshold=0.1,
    )
    assert all_missing.abstained.tolist() == [True]
    assert all_missing.probability.tolist() == [0.5]


def test_graph_and_feature_contracts_reject_shortcuts_and_virtual_edges() -> None:
    assert assert_fusion_feature_names(["quality_mean", "quality_std"]) == (
        "quality_mean",
        "quality_std",
    )
    with pytest.raises(ModelContractError, match="forbidden"):
        assert_fusion_feature_names(["quality_mean", "step_index"])
    with pytest.raises(ModelContractError, match="forbidden"):
        assert_fusion_feature_names(["quality_mean", "source_component_id"])
    with pytest.raises(ModelContractError, match="forbidden"):
        ReliabilityGraphFusion(
            node_count=2,
            quality_feature_names=["quality_mean"],
            concept_dim=2,
            pair_feature_names=["event_truth"],
        )
    with pytest.raises(ModelContractError, match="forbidden"):
        validate_observed_pair_edges(_edge_frame().assign(pair_id="raw_pair"), node_count=2)
    with pytest.raises(ModelContractError, match="only observed_pair"):
        validate_observed_pair_edges(
            _edge_frame().assign(edge_provenance="virtual_compatibility"),
            node_count=2,
        )
    with pytest.raises(ModelContractError, match="both directed"):
        validate_observed_pair_edges(_edge_frame().iloc[[0]], node_count=2)
    with pytest.raises(ModelContractError, match="non-self"):
        validate_observed_pair_edges(
            pd.DataFrame(
                [
                    {
                        "source_node": 0,
                        "target_node": 0,
                        "edge_provenance": "observed_pair",
                    }
                ]
            ),
            node_count=2,
        )


def test_event_memory_hysteresis_abstention_and_checkpoint_control() -> None:
    with pytest.raises(ModelContractError, match="beta is outside"):
        MemoryPolicyConfig(
            beta=0.5,
            memory_k=2,
            low_threshold=0.3,
            high_threshold=0.6,
            alarm_threshold=0.8,
        )
    policy = EventMemoryPolicy(
        MemoryPolicyConfig(
            beta=0.7,
            memory_k=2,
            low_threshold=0.3,
            high_threshold=0.6,
            alarm_threshold=0.8,
        )
    )
    trace = policy.run(
        [1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        [False, False, True, False, False, False],
        alarm_allowed=False,
    )
    assert [row.state for row in trace] == [
        "attention",
        "attention",
        "abstain",
        "attention",
        "prewarning",
        "prewarning",
    ]
    assert trace[2].memory == trace[1].memory
    assert trace[2].high_count == trace[1].high_count

    no_memory = policy.run(
        [1.0, 0.0], [False, False], alarm_allowed=False, use_memory=False
    )
    assert no_memory[-1].memory == 0.0
    assert no_memory[-1].state == "normal"
    validate_same_checkpoint("a" * 64, "a" * 64)
    with pytest.raises(ModelContractError, match="reuse"):
        validate_same_checkpoint("a" * 64, "b" * 64)


def test_pair_alignment_shuffle_is_stratified_deterministic_derangement() -> None:
    values = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    strata = [
        ("D_e_te", "helmet", "visible", True),
        ("D_e_te", "helmet", "visible", True),
        ("D_e_te", "helmet", "visible", True),
        ("D_e_te", "helmet", "thermal", True),
        ("D_e_te", "helmet", "thermal", True),
        ("D_e_te", "helmet", "thermal", True),
    ]
    shuffled, permutation = derange_values_within_strata(values, strata, seed=9103)
    repeated, repeated_permutation = derange_values_within_strata(
        values, strata, seed=9103
    )
    assert np.array_equal(shuffled, repeated)
    assert np.array_equal(permutation, repeated_permutation)
    assert np.all(permutation != np.arange(len(values)))
    for key in {tuple(value) for value in strata}:
        indices = [index for index, value in enumerate(strata) if tuple(value) == key]
        assert sorted(shuffled[indices]) == sorted(values[indices])
    with pytest.raises(ModelContractError, match="at least two"):
        derange_values_within_strata(
            [0.1], [("D_e_te", "helmet", "visible", True)], seed=9103
        )
    with pytest.raises(ModelContractError, match="pool, concept, modality"):
        derange_values_within_strata([0.1], [("only",)], seed=9103)
    with pytest.raises(ModelContractError, match="not preregistered"):
        derange_values_within_strata(values, strata, seed=1)

    masks = np.asarray([[True, False], [True, True]])
    mask_only = mask_only_control(
        np.asarray([[0.9, np.nan], [0.2, 0.8]]),
        masks,
        [0.25, 0.75],
        prior_fit_pool="D_e_tr",
    )
    assert mask_only[0, 0] == pytest.approx(0.25)
    assert np.isnan(mask_only[0, 1])
    assert mask_only[1, 1] == pytest.approx(0.75)
    with pytest.raises(ModelContractError, match="only on D_e_tr"):
        mask_only_control(
            np.asarray([[0.9, np.nan], [0.2, 0.8]]),
            masks,
            [0.25, 0.75],
            prior_fit_pool="D_e_te",
        )

    steps = np.arange(12).reshape(4, 3)
    shuffled_steps, step_permutation = derange_temporal_steps(steps, seed=1701)
    assert np.all(step_permutation != np.arange(4))
    assert sorted(map(tuple, shuffled_steps)) == sorted(map(tuple, steps))


def test_event_smoke_and_non_test_policy_selection_boundaries() -> None:
    result = evaluate_episode_events(
        truth_event=[0, 1, 1, 0],
        effective_states=["normal", "prewarning", "prewarning", "normal"],
        episode_ids=["episode_0"] * 4,
        step_indices=[0, 1, 2, 3],
    )
    assert result["matched_event_count"] == 1
    assert result["event_f1"] == 1.0
    assert result["answer_coverage"] == 1.0

    s1_candidates = pd.DataFrame(
        [
            {"candidate_id": "t0.5", "threshold": 0.5, "false_alarms_per_hour": 0.4, "event_macro_f1": 0.7, "event_miss_rate": 0.2},
            {"candidate_id": "t0.6", "threshold": 0.6, "false_alarms_per_hour": 0.4, "event_macro_f1": 0.7, "event_miss_rate": 0.2},
        ]
    )
    selected_s1 = select_s1_threshold(
        s1_candidates, pool="D_b_sel", max_false_alarms_per_hour=0.5
    )
    assert selected_s1.selected_id == "t0.6"
    with pytest.raises(EventMetricError, match="only D_b_sel"):
        select_s1_threshold(
            s1_candidates, pool="D_b_te", max_false_alarms_per_hour=0.5
        )

    episode_candidates = pd.DataFrame(
        [
            {"candidate_id": "simple", "beta": 0.70, "memory_k": 2, "low_threshold": 0.30, "high_threshold": 0.60, "alarm_threshold": 0.80, "abstention_threshold": 0.10, "answer_coverage": 0.90, "false_alarms_per_100_episodes": 2.0, "event_miss_rate": 0.1, "event_macro_f1": 0.8, "mean_detection_delay_steps": 1.0, "mean_state_flips_per_episode": 2.0, "complexity_rank": 0},
            {"candidate_id": "complex", "beta": 0.85, "memory_k": 3, "low_threshold": 0.40, "high_threshold": 0.70, "alarm_threshold": 0.80, "abstention_threshold": 0.90, "answer_coverage": 0.90, "false_alarms_per_100_episodes": 2.0, "event_miss_rate": 0.1, "event_macro_f1": 0.8, "mean_detection_delay_steps": 1.0, "mean_state_flips_per_episode": 2.0, "complexity_rank": 1},
        ]
    )
    selected_policy = select_episode_policy(
        episode_candidates,
        pool="D_e_pol",
        target_answer_coverage=0.90,
        max_coverage_deviation=0.02,
        max_false_alarms_per_100_episodes=5.0,
        max_event_miss_rate=0.2,
    )
    assert selected_policy.selected_id == "simple"
    outside_grid = episode_candidates.copy()
    outside_grid.loc[0, "beta"] = 0.5
    with pytest.raises(EventMetricError, match="outside the locked grid"):
        select_episode_policy(
            outside_grid,
            pool="D_e_pol",
            target_answer_coverage=0.90,
            max_coverage_deviation=0.02,
            max_false_alarms_per_100_episodes=5.0,
            max_event_miss_rate=0.2,
        )
