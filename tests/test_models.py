from __future__ import annotations

import inspect
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
import yaml

from mining1_exp.models.methane import (
    CAUSAL_STATISTICS,
    CAUSAL_SUBWINDOWS_SECONDS,
    CausalStandardizer,
    MethaneGRU,
    MethaneHGB,
    PersistenceRiskRule,
    assert_causal_feature_names,
    build_causal_stat_features,
)
from mining1_exp.models.rgbt_branches import (
    IndependentT1Branches,
    THERMAL_SINGLE_CHANNEL_TRANSFORM,
    build_branch_batch,
)
from mining1_exp.models.yolo_adapter import (
    BinaryProbabilityCalibrator,
    ModelContractError,
    YoloV8nAdapter,
    aggregate_concept_scores,
    build_yolov8n_model,
    select_calibrator_group_cv,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _walk_tensors(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_tensors(item)


class _TinyDetector(nn.Module):
    def __init__(self, channels: int = 2) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, channels, kernel_size=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(images).mean(dim=(2, 3))


def test_model_constants_match_the_frozen_protocol() -> None:
    protocol = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "protocol_lock.template.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert protocol["models"]["visible"]["primary_backbone"] == "yolov8n"
    assert protocol["models"]["visible"]["external_baselines"] == []
    assert protocol["models"]["rgbt"]["single_modalities"] == [
        "visible_only",
        "thermal_only",
    ]
    assert protocol["models"]["rgbt"]["trainable_fusion_models"] == []
    assert protocol["models"]["methane"]["deterministic_baselines"] == [
        "persistence_or_threshold_rule"
    ]
    assert protocol["models"]["methane"]["classical_baselines"] == [
        "hist_gradient_boosting"
    ]
    assert protocol["models"]["methane"]["neural_models"] == ["gru"]
    assert protocol["models"]["methane"]["hidden_size"] == 64
    assert protocol["models"]["methane"]["layers"] == 1
    assert protocol["models"]["methane"]["dropout"] == pytest.approx(0.20)
    assert THERMAL_SINGLE_CHANNEL_TRANSFORM == "replicate_3ch"


def test_actual_yolov8n_yaml_forward_is_finite_without_weights() -> None:
    model = build_yolov8n_model().eval()
    adapter = YoloV8nAdapter(model).eval()
    images = torch.zeros(1, 3, 64, 64, dtype=torch.float32)
    with torch.no_grad():
        output = adapter(images)
    tensors = list(_walk_tensors(output))
    assert tensors
    assert all(torch.isfinite(tensor).all() for tensor in tensors)
    assert sum(parameter.numel() for parameter in model.parameters()) == 3_157_200
    with pytest.raises(ModelContractError, match="only the frozen"):
        YoloV8nAdapter(_TinyDetector(), model_family="yolov10n")
    with pytest.raises(ModelContractError, match="three channels"):
        adapter(torch.zeros(1, 1, 64, 64))


def test_t1_visible_and_thermal_batches_are_independent_and_id_free() -> None:
    visible_image = np.full((64, 64, 3), 128, dtype=np.uint8)
    thermal_image = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    thermal_image = (thermal_image % 256).astype(np.uint8)
    visible_batch = build_branch_batch(
        [visible_image], record_ids=["visible_record"], modality="visible_only"
    )
    thermal_batch = build_branch_batch(
        [thermal_image], record_ids=["thermal_record"], modality="thermal_only"
    )
    assert visible_batch.model_tensor().shape == (1, 3, 64, 64)
    assert thermal_batch.model_tensor().shape == (1, 3, 64, 64)
    assert torch.equal(
        thermal_batch.images[:, 0], thermal_batch.images[:, 1]
    ) and torch.equal(thermal_batch.images[:, 1], thermal_batch.images[:, 2])
    assert visible_batch.record_ids == ("visible_record",)
    assert visible_batch.model_tensor().dtype == torch.float32

    visible = YoloV8nAdapter(_TinyDetector())
    thermal = YoloV8nAdapter(_TinyDetector())
    branches = IndependentT1Branches(visible, thermal, training_seed=1701)
    assert branches.forward_visible(visible_batch).shape == (1, 2)
    assert branches.forward_thermal(thermal_batch).shape == (1, 2)
    assert "forward" not in IndependentT1Branches.__dict__
    with pytest.raises(ModelContractError, match="BCHW tensor"):
        visible(visible_batch)
    with pytest.raises(ModelContractError, match="independent"):
        IndependentT1Branches(visible, visible)


def test_post_nms_concept_aggregation_emits_explicit_zero_rows() -> None:
    detections = pd.DataFrame(
        [
            {"record_id": "r1", "concept_id": "helmet", "score_raw": 0.4},
            {"record_id": "r1", "concept_id": "helmet", "score_raw": 0.9},
            {"record_id": "r1", "concept_id": "person", "score_raw": 0.6},
        ]
    )
    concepts = aggregate_concept_scores(
        detections,
        record_ids=["r1", "r2"],
        concept_ids=["helmet", "person"],
        modality="visible",
        post_nms=True,
    )
    assert len(concepts) == 4
    lookup = concepts.set_index(["record_id", "concept_id"])["step_score_raw"]
    assert lookup.loc[("r1", "helmet")] == pytest.approx(0.9)
    assert lookup.loc[("r2", "helmet")] == 0.0
    assert lookup.loc[("r2", "person")] == 0.0
    with pytest.raises(ModelContractError, match="post-NMS"):
        aggregate_concept_scores(
            detections,
            record_ids=["r1", "r2"],
            concept_ids=["helmet", "person"],
            modality="visible",
            post_nms=False,
        )


def test_all_locked_calibrators_fit_only_verified_D_b_prob() -> None:
    scores = np.asarray([0.01, 0.05, 0.15, 0.30, 0.65, 0.80, 0.95, 0.99])
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    for method in ("temperature", "platt", "isotonic"):
        calibrator = BinaryProbabilityCalibrator(method).fit(
            scores,
            labels,
            pool="D_b_prob",
            negatives_verified=True,
        )
        probabilities = calibrator.predict(scores)
        assert probabilities.shape == scores.shape
        assert np.isfinite(probabilities).all()
        assert ((probabilities >= 0) & (probabilities <= 1)).all()

    with pytest.raises(ModelContractError, match="only on D_b_prob"):
        BinaryProbabilityCalibrator("platt").fit(
            scores, labels, pool="D_b_te", negatives_verified=True
        )
    with pytest.raises(ModelContractError, match="verified negative"):
        BinaryProbabilityCalibrator("platt").fit(
            scores, labels, pool="D_b_prob", negatives_verified=False
        )

    group_scores = np.tile(np.asarray([0.10, 0.90]), 6)
    group_labels = np.tile(np.asarray([0, 1]), 6)
    raw_groups = [f"group_{index}" for index in range(6) for _ in range(2)]
    selection = select_calibrator_group_cv(
        group_scores,
        group_labels,
        raw_groups,
        pool="D_b_prob",
        negatives_verified=True,
        folds=3,
    )
    assert selection.method in {"temperature", "platt", "isotonic"}
    assert set(selection.cv_brier_by_method) == {
        "temperature",
        "platt",
        "isotonic",
    }
    assert selection.fold_count == 3
    assert selection.calibrator.fitted_pool == "D_b_prob"


def test_causal_statistics_and_preprocessing_exclude_future_and_ids() -> None:
    history = np.arange(2 * 10 * 2, dtype=np.float64).reshape(2, 10, 2)
    history[0, 2, 1] = np.nan
    features, names = build_causal_stat_features(
        history,
        sample_period_seconds=30,
    )
    assert features.shape == (
        2,
        len(CAUSAL_SUBWINDOWS_SECONDS) * 2 * len(CAUSAL_STATISTICS),
    )
    assert not {
        "future_values",
        "labels",
        "record_ids",
    }.intersection(inspect.signature(build_causal_stat_features).parameters)
    standardizer = CausalStandardizer(names).fit(features, pool="D_b_tr")
    transformed = standardizer.transform(features)
    assert transformed.shape == features.shape
    assert np.isfinite(transformed).all()
    with pytest.raises(ModelContractError, match="only on D_b_tr"):
        CausalStandardizer(names).fit(features, pool="D_b_te")
    with pytest.raises(ModelContractError, match="forbidden"):
        assert_causal_feature_names(["methane", "future_value"])
    with pytest.raises(ModelContractError, match="forbidden"):
        assert_causal_feature_names(["methane", "record_id"])


def test_rule_hgb_and_gru_follow_the_locked_model_boundaries() -> None:
    history = np.asarray(
        [
            [[0.2], [0.4], [0.8]],
            [[0.8], [1.0], [1.4]],
        ],
        dtype=np.float64,
    )
    rule = PersistenceRiskRule(concentration_threshold=1.0, transition_scale=0.2)
    rule_scores = rule.score(history)
    assert 0 < rule_scores[0] < rule_scores[1] < 1

    random = np.random.default_rng(1701)
    features = random.normal(size=(24, 4))
    labels = np.asarray([0, 1] * 12)
    hgb = MethaneHGB(
        ["history_mean", "history_std", "history_last", "history_slope"],
        random_state=1701,
        max_iter=10,
    ).fit(features, labels, pool="D_b_tr")
    probabilities = hgb.predict_proba(features)
    assert probabilities.shape == (24,)
    assert np.isfinite(probabilities).all()
    with pytest.raises(ModelContractError, match="single fit"):
        hgb.fit(features, labels, pool="D_b_tr")

    torch.manual_seed(1701)
    gru = MethaneGRU(
        ["sensor_0", "sensor_1"], hidden_size=64, layers=1, dropout=0.20
    )
    sequence = torch.randn(4, 8, 2, requires_grad=True)
    logits = gru(sequence)
    assert logits.shape == (4,)
    assert torch.isfinite(logits).all()
    logits.sum().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in gru.parameters()
    )
    assert gru.hidden_size == 64
    assert gru.layers == 1
    assert gru.dropout_probability == pytest.approx(0.20)

    buffer = io.BytesIO()
    gru.eval()
    expected = gru(sequence.detach())
    torch.save(gru.state_dict(), buffer)
    buffer.seek(0)
    restored = MethaneGRU(["sensor_0", "sensor_1"])
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    restored.eval()
    assert torch.allclose(expected, restored(sequence.detach()))
    with pytest.raises(ModelContractError, match="forbidden"):
        MethaneGRU(["sensor_0", "dataset_id"])
