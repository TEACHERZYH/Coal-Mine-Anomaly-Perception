from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import product
import re
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from .yolo_adapter import ModelContractError


PROBABILITY_EPSILON = 1.0e-6
FUSION_TRAIN_POOL = "D_e_tr"
POLICY_POOL = "D_e_pol"
PAIR_SHUFFLE_SEEDS = {9103, 9137, 9161}
RISK_STATES = {"normal", "attention", "prewarning", "alarm", "abstain"}
POLICY_BETA_CANDIDATES = (0.70, 0.85)
POLICY_MEMORY_K_CANDIDATES = (2, 3)
POLICY_HYSTERESIS_PAIRS = ((0.30, 0.60), (0.40, 0.70))
POLICY_ALARM_THRESHOLD_CANDIDATES = (0.80,)
POLICY_ABSTENTION_THRESHOLDS = tuple(
    float(value) for value in np.round(np.arange(0.10, 0.9001, 0.05), 2)
)
FORBIDDEN_FUSION_FEATURE_TOKENS = {
    "episode_id",
    "episode_seed",
    "step_index",
    "skeleton_item_id",
    "record_id",
    "dataset_id",
    "raw_group_id",
    "source_component_id",
    "template_instance_id",
    "generator_family_id",
    "pair_id",
    "node_id",
    "event_position_role",
    "event_truth",
    "state_truth",
    "test_label",
    "pool",
    "label",
    "truth",
    "future",
}
FORBIDDEN_EDGE_COLUMNS = {
    "episode_id",
    "episode_seed",
    "step_index",
    "skeleton_item_id",
    "record_id",
    "raw_group_id",
    "source_component_id",
    "dataset_id",
    "generator_family_id",
    "pair_id",
    "pool",
    "template_instance_id",
    "event_position_role",
    "event_truth",
    "state_truth",
    "test_label",
    "label",
    "truth",
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _is_locked_float(value: float, candidates: Sequence[float]) -> bool:
    return bool(np.isfinite(value) and np.isclose(float(value), candidates).any())


def locked_episode_policy_grid() -> pd.DataFrame:
    rows = []
    for beta, memory_k, thresholds, alarm, abstention in product(
        POLICY_BETA_CANDIDATES,
        POLICY_MEMORY_K_CANDIDATES,
        POLICY_HYSTERESIS_PAIRS,
        POLICY_ALARM_THRESHOLD_CANDIDATES,
        POLICY_ABSTENTION_THRESHOLDS,
    ):
        low, high = thresholds
        rows.append(
            {
                "policy_id": (
                    f"b{beta:.2f}_k{memory_k}_l{low:.2f}_h{high:.2f}"
                    f"_a{alarm:.2f}_u{abstention:.2f}"
                ),
                "beta": beta,
                "memory_k": memory_k,
                "low_threshold": low,
                "high_threshold": high,
                "alarm_threshold": alarm,
                "abstention_threshold": abstention,
            }
        )
    return pd.DataFrame(rows)


def assert_fusion_feature_names(feature_names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(str(name) for name in feature_names)
    if not names or len(set(names)) != len(names):
        raise ModelContractError("fusion feature names must be non-empty and unique")
    forbidden = sorted(
        name
        for name in names
        if any(token in name.lower() for token in FORBIDDEN_FUSION_FEATURE_TOKENS)
    )
    if forbidden:
        raise ModelContractError(f"forbidden fusion feature names: {forbidden}")
    return names


@dataclass
class RobustQualityScaler:
    feature_names: Sequence[str]
    nonfinite_fill: float = 0.0
    fitted_pool: Optional[str] = None
    medians_: Optional[np.ndarray] = None
    scales_: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.feature_names = assert_fusion_feature_names(self.feature_names)
        if not np.isfinite(self.nonfinite_fill):
            raise ModelContractError("quality nonfinite fill must be finite")

    def fit(self, quality: np.ndarray, *, pool: str) -> "RobustQualityScaler":
        if pool != FUSION_TRAIN_POOL:
            raise ModelContractError("quality scaling may fit only on D_e_tr")
        if self.fitted_pool is not None:
            raise ModelContractError("quality scaler is already fitted")
        values = np.asarray(quality, dtype=np.float64)
        if values.ndim != 3 or values.shape[0] == 0 or values.shape[2] != len(
            self.feature_names
        ):
            raise ModelContractError("quality scaler input has the wrong shape")
        medians = np.zeros(values.shape[1:], dtype=np.float64)
        scales = np.ones(values.shape[1:], dtype=np.float64)
        for node in range(values.shape[1]):
            for feature in range(values.shape[2]):
                column = values[:, node, feature]
                finite = column[np.isfinite(column)]
                if finite.size:
                    median = float(np.median(finite))
                    iqr = float(np.quantile(finite, 0.75) - np.quantile(finite, 0.25))
                    medians[node, feature] = median
                    scales[node, feature] = iqr if iqr > 0 else 1.0
        self.medians_ = medians
        self.scales_ = scales
        self.fitted_pool = pool
        return self

    def transform(self, quality: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.fitted_pool != FUSION_TRAIN_POOL:
            raise ModelContractError("quality scaler is not fitted on D_e_tr")
        values = np.asarray(quality, dtype=np.float64)
        if values.ndim != 3 or values.shape[1:] != self.medians_.shape:
            raise ModelContractError("quality scaler input has the wrong shape")
        finite_mask = np.isfinite(values)
        filled = np.where(finite_mask, values, self.medians_)
        scaled = (filled - self.medians_) / self.scales_
        scaled = np.where(finite_mask, scaled, self.nonfinite_fill)
        if not np.isfinite(scaled).all():
            raise ModelContractError("quality scaling produced non-finite values")
        return scaled, finite_mask


def validate_observed_pair_edges(
    edge_frame: pd.DataFrame, *, node_count: int
) -> torch.Tensor:
    required = {"source_node", "target_node", "edge_provenance"}
    missing = sorted(required.difference(edge_frame.columns))
    if missing:
        raise ModelContractError(f"graph edge frame is missing columns: {missing}")
    forbidden = sorted(FORBIDDEN_EDGE_COLUMNS.intersection(edge_frame.columns))
    if forbidden:
        raise ModelContractError(f"graph edge frame contains forbidden columns: {forbidden}")
    if edge_frame.empty:
        raise ModelContractError("confirmatory graph requires observed-pair edges")
    if set(edge_frame["edge_provenance"]) != {"observed_pair"}:
        raise ModelContractError("confirmatory graph accepts only observed_pair edges")
    source = pd.to_numeric(edge_frame["source_node"], errors="raise").astype("int64")
    target = pd.to_numeric(edge_frame["target_node"], errors="raise").astype("int64")
    if ((source < 0) | (source >= node_count) | (target < 0) | (target >= node_count)).any():
        raise ModelContractError("graph edge endpoint is outside the node set")
    if (source == target).any():
        raise ModelContractError("confirmatory graph edges must be non-self")
    pairs = set(zip(source.tolist(), target.tolist()))
    if any((target_node, source_node) not in pairs for source_node, target_node in pairs):
        raise ModelContractError("each observed undirected pair requires both directed edges")
    if len(pairs) != len(edge_frame):
        raise ModelContractError("graph edge endpoints must be unique")
    return torch.tensor([source.tolist(), target.tolist()], dtype=torch.long)


def _validate_probability_inputs(
    probabilities: torch.Tensor, availability: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if not torch.is_tensor(probabilities) or probabilities.ndim != 2:
        raise ModelContractError("fusion probabilities must be a batch x node tensor")
    if not torch.is_tensor(availability) or availability.shape != probabilities.shape:
        raise ModelContractError("fusion availability must align with probabilities")
    available = availability.to(dtype=torch.bool)
    if not probabilities.is_floating_point():
        raise ModelContractError("fusion probabilities must be floating point")
    finite_or_missing = torch.isfinite(probabilities) | ~available
    if not finite_or_missing.all():
        raise ModelContractError("available fusion probabilities must be finite")
    safe = torch.where(available, probabilities, torch.full_like(probabilities, 0.5))
    if ((safe < 0) | (safe > 1)).any():
        raise ModelContractError("available fusion probabilities must lie in [0, 1]")
    return safe, available


def clipped_logit(
    probabilities: torch.Tensor, epsilon: float = PROBABILITY_EPSILON
) -> torch.Tensor:
    if not 0 < epsilon < 0.5:
        raise ModelContractError("probability epsilon must lie in (0, 0.5)")
    clipped = probabilities.clamp(epsilon, 1.0 - epsilon)
    return torch.log(clipped / (1.0 - clipped))


def mean_fusion(
    probabilities: torch.Tensor, availability: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    safe, available = _validate_probability_inputs(probabilities, availability)
    weights = available.to(dtype=safe.dtype)
    denominator = weights.sum(dim=1)
    abstained = denominator == 0
    fused = (safe * weights).sum(dim=1) / denominator.clamp_min(1.0)
    fused = torch.where(abstained, torch.full_like(fused, 0.5), fused)
    return fused, abstained


class CalibratedLogitFusion(nn.Module):
    def __init__(self, node_count: int) -> None:
        super().__init__()
        if node_count < 2:
            raise ModelContractError("calibrated-logit fusion requires at least two nodes")
        self.node_count = int(node_count)
        self.raw_weights = nn.Parameter(torch.zeros(self.node_count))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(
        self, probabilities: torch.Tensor, availability: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        safe, available = _validate_probability_inputs(probabilities, availability)
        if safe.shape[1] != self.node_count:
            raise ModelContractError("calibrated-logit input has the wrong node count")
        positive_weights = torch.nn.functional.softplus(self.raw_weights)
        masked_weights = positive_weights[None, :] * available.to(dtype=safe.dtype)
        denominator = masked_weights.sum(dim=1)
        abstained = denominator <= PROBABILITY_EPSILON
        fused_logit = (
            masked_weights * clipped_logit(safe)
        ).sum(dim=1) / denominator.clamp_min(PROBABILITY_EPSILON) + self.bias
        probabilities_out = torch.sigmoid(fused_logit)
        probabilities_out = torch.where(
            abstained, torch.full_like(probabilities_out, 0.5), probabilities_out
        )
        return probabilities_out, abstained


class ReliabilityEstimator(nn.Module):
    def __init__(
        self, node_count: int, quality_dim: int, concept_dim: int, hidden_dim: int
    ) -> None:
        super().__init__()
        if min(node_count, quality_dim, concept_dim, hidden_dim) <= 0:
            raise ModelContractError("reliability dimensions must be positive")
        self.node_count = int(node_count)
        input_dim = int(quality_dim + concept_dim)
        self.estimators = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(self.node_count)
            ]
        )

    def forward(
        self,
        quality: torch.Tensor,
        concept_embedding: torch.Tensor,
        availability: torch.Tensor,
    ) -> torch.Tensor:
        if quality.ndim != 3 or quality.shape[:2] != availability.shape:
            raise ModelContractError("quality tensor must align with batch and nodes")
        if concept_embedding.ndim != 2 or concept_embedding.shape[0] != quality.shape[0]:
            raise ModelContractError("concept embedding must align with the batch")
        if quality.shape[1] != self.node_count:
            raise ModelContractError("quality tensor has the wrong node count")
        available = availability.to(dtype=torch.bool)
        if not (torch.isfinite(quality) | ~available.unsqueeze(-1)).all():
            raise ModelContractError("available quality features must be finite")
        if not torch.isfinite(concept_embedding).all():
            raise ModelContractError("concept embedding must be finite")
        safe_quality = torch.where(
            available.unsqueeze(-1), quality, torch.zeros_like(quality)
        )
        outputs = []
        for node_index, estimator in enumerate(self.estimators):
            inputs = torch.cat(
                [safe_quality[:, node_index], concept_embedding], dim=-1
            )
            outputs.append(torch.sigmoid(estimator(inputs).squeeze(-1)))
        reliability = torch.stack(outputs, dim=1)
        return reliability * available.to(dtype=reliability.dtype)


class EdgeConditionedGraphLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_feature_dim: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.score = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.message = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.normalization = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        states: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        availability: torch.Tensor,
        edge_validity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if states.ndim != 3 or edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ModelContractError("graph states or edge index have an invalid shape")
        if edge_features.ndim != 3 or edge_features.shape[:2] != (
            states.shape[0],
            edge_index.shape[1],
        ):
            raise ModelContractError("graph edge features do not align with edges")
        if not torch.isfinite(states).all() or not torch.isfinite(edge_features).all():
            raise ModelContractError("graph tensors must be finite")
        if edge_validity.shape != edge_features.shape[:2]:
            raise ModelContractError("graph edge validity does not align with edges")
        source_all = edge_index[0].to(device=states.device)
        target_all = edge_index[1].to(device=states.device)
        if (source_all == target_all).any():
            raise ModelContractError("graph message passing rejects self edges")
        batch_size, node_count, _ = states.shape
        if source_all.numel() and (
            source_all.min() < 0
            or target_all.min() < 0
            or source_all.max() >= node_count
            or target_all.max() >= node_count
        ):
            raise ModelContractError("graph edge endpoint is outside the node tensor")
        aggregate = torch.zeros_like(states)
        edge_weights = states.new_zeros((batch_size, edge_index.shape[1]))
        for target_node in range(node_count):
            edge_ids = torch.nonzero(target_all == target_node, as_tuple=False).flatten()
            if not edge_ids.numel():
                continue
            sources = source_all[edge_ids]
            source_states = states[:, sources, :]
            target_states = states[:, target_node, :].unsqueeze(1).expand_as(source_states)
            score_inputs = torch.cat(
                [source_states, target_states, edge_features[:, edge_ids, :]], dim=-1
            )
            scores = self.score(score_inputs).squeeze(-1)
            valid_sources = availability[:, sources].to(dtype=torch.bool) & edge_validity[
                :, edge_ids
            ].to(dtype=torch.bool)
            masked_scores = torch.where(valid_sources, scores, torch.full_like(scores, -torch.inf))
            maxima = masked_scores.max(dim=1, keepdim=True).values
            maxima = torch.where(torch.isfinite(maxima), maxima, torch.zeros_like(maxima))
            exponentials = torch.where(
                valid_sources, torch.exp(scores - maxima), torch.zeros_like(scores)
            )
            alpha = exponentials / exponentials.sum(dim=1, keepdim=True).clamp_min(
                PROBABILITY_EPSILON
            )
            messages = self.message(source_states) * alpha.unsqueeze(-1)
            aggregate[:, target_node, :] = messages.sum(dim=1)
            edge_weights[:, edge_ids] = alpha
        updated = self.normalization(states + aggregate)
        return updated, edge_weights, aggregate.norm(dim=-1)


class NodeIndependentLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_feature_dim: int) -> None:
        super().__init__()
        self.edge_feature_dim = int(edge_feature_dim)
        self.score = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.message = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.normalization = nn.LayerNorm(hidden_dim)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        zero_edges = states.new_zeros((*states.shape[:2], self.edge_feature_dim))
        gate = torch.sigmoid(
            self.score(torch.cat([states, states, zero_edges], dim=-1))
        )
        return self.normalization(states + gate * self.message(states))


@dataclass
class FusionOutput:
    probability: torch.Tensor
    reliability: torch.Tensor
    reliability_weights: torch.Tensor
    abstained: torch.Tensor
    edge_weights: torch.Tensor
    message_norms: torch.Tensor


class ReliabilityGraphFusion(nn.Module):
    def __init__(
        self,
        *,
        node_count: int,
        quality_feature_names: Sequence[str],
        concept_dim: int,
        pair_feature_names: Sequence[str],
        hidden_dim: int = 16,
        modality_embedding_dim: int = 4,
        graph_layers: int = 2,
        use_graph: bool = True,
        use_reliability: bool = True,
    ) -> None:
        super().__init__()
        self.quality_feature_names = assert_fusion_feature_names(quality_feature_names)
        if node_count < 2 or graph_layers != 2:
            raise ModelContractError("fusion requires at least two nodes and two graph layers")
        if min(concept_dim, hidden_dim, modality_embedding_dim) <= 0:
            raise ModelContractError("fusion dimensions must be positive")
        self.node_count = int(node_count)
        self.quality_dim = len(self.quality_feature_names)
        self.concept_dim = int(concept_dim)
        self.pair_feature_names = assert_fusion_feature_names(pair_feature_names)
        self.pair_feature_dim = len(self.pair_feature_names)
        self.hidden_dim = int(hidden_dim)
        self.use_graph = bool(use_graph)
        self.use_reliability = bool(use_reliability)
        self.modality_embedding = nn.Parameter(
            torch.empty(self.node_count, modality_embedding_dim)
        )
        nn.init.normal_(self.modality_embedding, mean=0.0, std=0.02)
        node_input_dim = 1 + self.quality_dim + modality_embedding_dim + 1
        self.node_encoder = nn.Linear(node_input_dim, self.hidden_dim)
        self.reliability_estimator = ReliabilityEstimator(
            self.node_count,
            self.quality_dim,
            self.concept_dim,
            self.hidden_dim,
        )
        edge_dim = self.pair_feature_dim + 4
        if self.use_graph:
            self.layers = nn.ModuleList(
                [EdgeConditionedGraphLayer(self.hidden_dim, edge_dim) for _ in range(2)]
            )
        else:
            self.layers = nn.ModuleList(
                [
                    NodeIndependentLayer(self.hidden_dim, edge_dim)
                    for _ in range(2)
                ]
            )
        self.output_head = nn.Linear(self.hidden_dim, 1)

    def forward(
        self,
        probabilities: torch.Tensor,
        quality: torch.Tensor,
        availability: torch.Tensor,
        concept_embedding: torch.Tensor,
        *,
        edge_index: Optional[torch.Tensor] = None,
        pair_features: Optional[torch.Tensor] = None,
        edge_validity: Optional[torch.Tensor] = None,
        abstention_threshold: float,
    ) -> FusionOutput:
        safe_probabilities, available = _validate_probability_inputs(
            probabilities, availability
        )
        batch_size, node_count = safe_probabilities.shape
        if node_count != self.node_count:
            raise ModelContractError("fusion probability tensor has the wrong node count")
        if quality.shape != (batch_size, node_count, self.quality_dim):
            raise ModelContractError("fusion quality tensor has the wrong shape")
        if concept_embedding.shape != (batch_size, self.concept_dim):
            raise ModelContractError("fusion concept embedding has the wrong shape")
        if not _is_locked_float(abstention_threshold, POLICY_ABSTENTION_THRESHOLDS):
            raise ModelContractError("abstention threshold is outside the locked grid")
        if not (torch.isfinite(quality) | ~available.unsqueeze(-1)).all():
            raise ModelContractError("available quality features must be finite")
        safe_quality = torch.where(
            available.unsqueeze(-1), quality, torch.zeros_like(quality)
        )
        logits = clipped_logit(safe_probabilities) * available.to(
            dtype=safe_probabilities.dtype
        )
        modality = self.modality_embedding[None, :, :].expand(batch_size, -1, -1)
        node_inputs = torch.cat(
            [
                logits.unsqueeze(-1),
                safe_quality,
                modality,
                available.to(dtype=safe_probabilities.dtype).unsqueeze(-1),
            ],
            dim=-1,
        )
        states = torch.relu(self.node_encoder(node_inputs))
        edge_weight_layers = []
        message_norm_layers = []
        if self.use_graph:
            if edge_index is None or pair_features is None:
                raise ModelContractError("graph-enabled fusion requires edge tensors")
            if pair_features.shape != (
                batch_size,
                edge_index.shape[1],
                self.pair_feature_dim,
            ):
                raise ModelContractError("pair feature tensor has the wrong shape")
            if edge_validity is None:
                edge_validity = torch.ones(
                    (batch_size, edge_index.shape[1]),
                    dtype=torch.bool,
                    device=probabilities.device,
                )
            elif edge_validity.shape != (batch_size, edge_index.shape[1]):
                raise ModelContractError("edge validity tensor has the wrong shape")
            source = edge_index[0].to(device=probabilities.device)
            target = edge_index[1].to(device=probabilities.device)
            dynamic_edges = torch.cat(
                [
                    pair_features,
                    available[:, source].to(dtype=probabilities.dtype).unsqueeze(-1),
                    available[:, target].to(dtype=probabilities.dtype).unsqueeze(-1),
                    (safe_probabilities[:, source] - safe_probabilities[:, target])
                    .abs()
                    .unsqueeze(-1),
                    (safe_probabilities[:, source] * safe_probabilities[:, target]).unsqueeze(-1),
                ],
                dim=-1,
            )
            for layer in self.layers:
                states, edge_weights, message_norms = layer(
                    states, edge_index, dynamic_edges, available, edge_validity
                )
                edge_weight_layers.append(edge_weights)
                message_norm_layers.append(message_norms)
        else:
            if edge_index is not None or pair_features is not None or edge_validity is not None:
                raise ModelContractError("no-graph fusion must not receive edge tensors")
            for layer in self.layers:
                states = layer(states)

        if self.use_reliability:
            reliability = self.reliability_estimator(
                safe_quality, concept_embedding, available
            )
        else:
            reliability = available.to(dtype=probabilities.dtype)
        reliability_mass = reliability.sum(dim=1)
        weights = reliability / reliability_mass.unsqueeze(-1).clamp_min(
            PROBABILITY_EPSILON
        )
        max_reliability = reliability.max(dim=1).values
        abstained = (
            ~available.any(dim=1)
            | (reliability_mass <= PROBABILITY_EPSILON)
            | (max_reliability < abstention_threshold)
        )
        pooled = (weights.unsqueeze(-1) * states).sum(dim=1)
        fused_probability = torch.sigmoid(self.output_head(pooled).squeeze(-1))
        fused_probability = torch.where(
            abstained,
            torch.full_like(fused_probability, 0.5),
            fused_probability,
        )
        if edge_weight_layers:
            edge_weights_out = torch.stack(edge_weight_layers, dim=1)
            message_norms_out = torch.stack(message_norm_layers, dim=1)
        else:
            edge_weights_out = probabilities.new_zeros((batch_size, 0, 0))
            message_norms_out = probabilities.new_zeros((batch_size, 0, node_count))
        return FusionOutput(
            probability=fused_probability,
            reliability=reliability,
            reliability_weights=weights,
            abstained=abstained,
            edge_weights=edge_weights_out,
            message_norms=message_norms_out,
        )


def reliability_supervision(
    truth: torch.Tensor,
    branch_probabilities: torch.Tensor,
    availability: torch.Tensor,
    *,
    pool: str,
) -> torch.Tensor:
    if pool != FUSION_TRAIN_POOL:
        raise ModelContractError("reliability targets may be built only in D_e_tr")
    safe, available = _validate_probability_inputs(branch_probabilities, availability)
    if truth.ndim != 1 or truth.shape[0] != safe.shape[0]:
        raise ModelContractError("reliability truth must align with the batch")
    if not torch.isfinite(truth).all() or ((truth < 0) | (truth > 1)).any():
        raise ModelContractError("reliability truth must be binary or probabilistic")
    targets = 1.0 - (truth.unsqueeze(-1) - safe).abs()
    return targets * available.to(dtype=targets.dtype)


def reliability_brier_loss(
    predicted: torch.Tensor, targets: torch.Tensor, availability: torch.Tensor
) -> torch.Tensor:
    if predicted.shape != targets.shape or predicted.shape != availability.shape:
        raise ModelContractError("reliability loss tensors must align")
    weights = availability.to(dtype=predicted.dtype)
    denominator = weights.sum().clamp_min(PROBABILITY_EPSILON)
    return (((predicted - targets) ** 2) * weights).sum() / denominator


@dataclass(frozen=True)
class MemoryPolicyConfig:
    beta: float
    memory_k: int
    low_threshold: float
    high_threshold: float
    alarm_threshold: float

    def __post_init__(self) -> None:
        if not _is_locked_float(self.beta, POLICY_BETA_CANDIDATES):
            raise ModelContractError("memory beta is outside the locked grid")
        if self.memory_k not in POLICY_MEMORY_K_CANDIDATES:
            raise ModelContractError("memory K is outside the locked grid")
        if not any(
            np.isclose(self.low_threshold, low)
            and np.isclose(self.high_threshold, high)
            for low, high in POLICY_HYSTERESIS_PAIRS
        ):
            raise ModelContractError("hysteresis thresholds are outside the locked grid")
        if not _is_locked_float(
            self.alarm_threshold, POLICY_ALARM_THRESHOLD_CANDIDATES
        ):
            raise ModelContractError("alarm threshold is outside the locked grid")


@dataclass(frozen=True)
class MemoryTraceRow:
    step: int
    probability: float
    memory: float
    high_count: int
    alarm_count: int
    state: str
    abstained: bool


class EventMemoryPolicy:
    def __init__(self, config: MemoryPolicyConfig) -> None:
        self.config = config

    def run(
        self,
        probabilities: Iterable[float],
        abstained: Iterable[bool],
        *,
        alarm_allowed: bool,
        use_memory: bool = True,
    ) -> list[MemoryTraceRow]:
        scores = [float(value) for value in probabilities]
        abstentions = [bool(value) for value in abstained]
        if len(scores) != len(abstentions) or not scores:
            raise ModelContractError("memory scores and abstentions must be aligned")
        if any(not np.isfinite(value) or not 0 <= value <= 1 for value in scores):
            raise ModelContractError("memory probabilities must lie in [0, 1]")
        memory = 0.0
        high_count = 0
        alarm_count = 0
        internal_state = "normal"
        trace = []
        for step, (score, is_abstained) in enumerate(zip(scores, abstentions)):
            if is_abstained:
                trace.append(
                    MemoryTraceRow(
                        step,
                        score,
                        memory,
                        high_count,
                        alarm_count,
                        "abstain",
                        True,
                    )
                )
                continue
            if use_memory:
                memory = self.config.beta * memory + (1.0 - self.config.beta) * score
            else:
                memory = score
            high_count = high_count + 1 if memory >= self.config.high_threshold else 0
            alarm_count = (
                alarm_count + 1
                if alarm_allowed and memory >= self.config.alarm_threshold
                else 0
            )
            if internal_state == "alarm" and memory >= self.config.low_threshold:
                next_state = "alarm"
            elif internal_state == "prewarning" and memory >= self.config.low_threshold:
                next_state = (
                    "alarm"
                    if alarm_allowed and alarm_count >= self.config.memory_k
                    else "prewarning"
                )
            elif alarm_allowed and alarm_count >= self.config.memory_k:
                next_state = "alarm"
            elif high_count >= self.config.memory_k:
                next_state = "prewarning"
            elif memory >= self.config.low_threshold:
                next_state = "attention"
            else:
                next_state = "normal"
            internal_state = next_state
            trace.append(
                MemoryTraceRow(
                    step,
                    score,
                    memory,
                    high_count,
                    alarm_count,
                    next_state,
                    False,
                )
            )
        return trace


def validate_same_checkpoint(full_hash: str, control_hash: str) -> None:
    if SHA256_PATTERN.fullmatch(str(full_hash)) is None or SHA256_PATTERN.fullmatch(
        str(control_hash)
    ) is None:
        raise ModelContractError("checkpoint hashes must be SHA-256 strings")
    if full_hash != control_hash:
        raise ModelContractError("inference control must reuse the full checkpoint hash")


def mask_only_control(
    probabilities: np.ndarray,
    availability: np.ndarray,
    d_e_tr_concept_priors: Sequence[float],
    *,
    prior_fit_pool: str,
) -> np.ndarray:
    if prior_fit_pool != FUSION_TRAIN_POOL:
        raise ModelContractError("mask-only priors may be fitted only on D_e_tr")
    scores = np.asarray(probabilities, dtype=np.float64)
    masks = np.asarray(availability, dtype=bool)
    priors = np.asarray(d_e_tr_concept_priors, dtype=np.float64)
    if scores.ndim != 2 or masks.shape != scores.shape or priors.shape != (
        scores.shape[1],
    ):
        raise ModelContractError("mask-only control inputs have incompatible shapes")
    if not np.isfinite(priors).all() or ((priors < 0) | (priors > 1)).any():
        raise ModelContractError("mask-only priors must be frozen probabilities")
    output = np.broadcast_to(priors, scores.shape).copy()
    output[~masks] = np.nan
    return output


def derange_temporal_steps(
    step_features: np.ndarray, *, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(step_features)
    if values.ndim < 1 or values.shape[0] < 2:
        raise ModelContractError("temporal shuffle requires at least two steps")
    digest = hashlib.sha256(f"temporal|{int(seed)}|{values.shape[0]}".encode("utf-8")).hexdigest()
    offset = int(digest[:16], 16) % (values.shape[0] - 1) + 1
    permutation = np.roll(np.arange(values.shape[0]), -offset)
    if np.any(permutation == np.arange(values.shape[0])):
        raise ModelContractError("temporal shuffle failed to derange steps")
    return values[permutation].copy(), permutation


def derange_values_within_strata(
    values: Sequence[float],
    strata_keys: Sequence[Sequence[object]],
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if int(seed) not in PAIR_SHUFFLE_SEEDS:
        raise ModelContractError("pair shuffle seed is not preregistered")
    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 1 or len(source) != len(strata_keys) or len(source) == 0:
        raise ModelContractError("shuffle values and strata must be aligned vectors")
    if not np.isfinite(source).all() or ((source < 0) | (source > 1)).any():
        raise ModelContractError("pair shuffle accepts only finite branch probabilities")
    groups: dict[tuple[str, ...], list[int]] = {}
    for index, raw_key in enumerate(strata_keys):
        if len(raw_key) != 4:
            raise ModelContractError(
                "pair shuffle strata must be pool, concept, modality, availability"
            )
        pool, concept, modality, is_available = raw_key
        if str(pool) != "D_e_te":
            raise ModelContractError("pair shuffle is restricted to D_e_te inference")
        if not str(concept).strip() or not str(modality).strip() or not isinstance(
            is_available, (bool, np.bool_)
        ):
            raise ModelContractError("pair shuffle stratum values are invalid")
        key = tuple(str(value) for value in raw_key)
        groups.setdefault(key, []).append(index)
    output = source.copy()
    permutation = np.arange(len(source), dtype=np.int64)
    for key, indices in sorted(groups.items()):
        if len(indices) < 2:
            raise ModelContractError("pair shuffle requires at least two rows per stratum")
        digest = hashlib.sha256(
            f"{int(seed)}|{'|'.join(key)}".encode("utf-8")
        ).hexdigest()
        offset = int(digest[:16], 16) % (len(indices) - 1) + 1
        sources = indices[offset:] + indices[:offset]
        output[np.asarray(indices)] = source[np.asarray(sources)]
        permutation[np.asarray(indices)] = np.asarray(sources)
    if np.any(permutation == np.arange(len(source))):
        raise ModelContractError("pair shuffle failed to produce a derangement")
    return output, permutation
