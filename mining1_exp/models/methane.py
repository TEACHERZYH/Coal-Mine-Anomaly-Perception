from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Optional, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import torch
from torch import nn

from .yolo_adapter import ModelContractError


METHANE_FIT_POOL = "D_b_tr"
CAUSAL_SUBWINDOWS_SECONDS = (60, 180, 300)
CAUSAL_STATISTICS = (
    "last",
    "mean",
    "std",
    "min",
    "max",
    "linear_slope",
    "q25",
    "median",
    "q75",
    "missing_fraction",
)
FORBIDDEN_FEATURE_TOKENS = {
    "dataset_id",
    "record_id",
    "raw_group_id",
    "future",
    "forecast",
    "post_window",
    "label",
    "truth",
    "target",
    "event_outcome",
}


def assert_causal_feature_names(feature_names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(str(name) for name in feature_names)
    if not names or len(set(names)) != len(names):
        raise ModelContractError("methane feature names must be non-empty and unique")
    forbidden = sorted(
        name
        for name in names
        if any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    )
    if forbidden:
        raise ModelContractError(f"forbidden methane feature names: {forbidden}")
    return names


def _window_statistics(values: np.ndarray, sample_period_seconds: int) -> list[float]:
    finite_mask = np.isfinite(values)
    finite_values = values[finite_mask]
    missing_fraction = 1.0 - float(finite_mask.mean())
    if finite_values.size == 0:
        return [math.nan] * 9 + [missing_fraction]
    finite_positions = np.flatnonzero(finite_mask).astype(np.float64) * sample_period_seconds
    slope = 0.0
    if finite_values.size >= 2 and np.ptp(finite_positions) > 0:
        slope = float(np.polyfit(finite_positions, finite_values, 1)[0])
    return [
        float(finite_values[-1]),
        float(np.mean(finite_values)),
        float(np.std(finite_values)),
        float(np.min(finite_values)),
        float(np.max(finite_values)),
        slope,
        float(np.quantile(finite_values, 0.25)),
        float(np.quantile(finite_values, 0.50)),
        float(np.quantile(finite_values, 0.75)),
        missing_fraction,
    ]


def build_causal_stat_features(
    history_values: np.ndarray,
    *,
    sample_period_seconds: int,
    subwindows_seconds: Sequence[int] = CAUSAL_SUBWINDOWS_SECONDS,
) -> tuple[np.ndarray, tuple[str, ...]]:
    values = np.asarray(history_values, dtype=np.float64)
    if values.ndim == 2:
        values = values[None, :, :]
    if values.ndim != 3 or values.shape[1] == 0 or values.shape[2] == 0:
        raise ModelContractError("methane history must have shape batch x time x sensors")
    if sample_period_seconds <= 0:
        raise ModelContractError("sample period must be positive")
    windows = tuple(int(window) for window in subwindows_seconds)
    if windows != tuple(sorted(set(windows))) or any(window <= 0 for window in windows):
        raise ModelContractError("causal subwindows must be unique positive ascending values")
    required_steps = math.ceil(max(windows) / sample_period_seconds)
    if values.shape[1] < required_steps:
        raise ModelContractError("history does not cover the largest causal subwindow")

    feature_names = tuple(
        f"sensor_{sensor}__history_{window}s__{statistic}"
        for window in windows
        for sensor in range(values.shape[2])
        for statistic in CAUSAL_STATISTICS
    )
    assert_causal_feature_names(feature_names)
    rows = []
    for batch_index in range(values.shape[0]):
        features = []
        for window in windows:
            step_count = math.ceil(window / sample_period_seconds)
            window_values = values[batch_index, -step_count:, :]
            for sensor in range(values.shape[2]):
                features.extend(
                    _window_statistics(window_values[:, sensor], sample_period_seconds)
                )
        rows.append(features)
    return np.asarray(rows, dtype=np.float64), feature_names


@dataclass
class CausalStandardizer:
    feature_names: Sequence[str]
    fitted_pool: Optional[str] = None
    medians_: Optional[np.ndarray] = None
    means_: Optional[np.ndarray] = None
    scales_: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.feature_names = assert_causal_feature_names(self.feature_names)

    def fit(self, features: np.ndarray, *, pool: str) -> "CausalStandardizer":
        if pool != METHANE_FIT_POOL:
            raise ModelContractError("methane preprocessing may fit only on D_b_tr")
        if self.fitted_pool is not None:
            raise ModelContractError("methane preprocessing is already fitted")
        values = _validate_feature_matrix(features, len(self.feature_names), allow_nan=True)
        medians = []
        for column in values.T:
            finite = column[np.isfinite(column)]
            medians.append(float(np.median(finite)) if finite.size else 0.0)
        self.medians_ = np.asarray(medians, dtype=np.float64)
        imputed = np.where(np.isfinite(values), values, self.medians_)
        self.means_ = np.mean(imputed, axis=0)
        scales = np.std(imputed, axis=0)
        self.scales_ = np.where(scales > 0, scales, 1.0)
        self.fitted_pool = pool
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.fitted_pool != METHANE_FIT_POOL:
            raise ModelContractError("methane preprocessing is not fitted on D_b_tr")
        values = _validate_feature_matrix(features, len(self.feature_names), allow_nan=True)
        imputed = np.where(np.isfinite(values), values, self.medians_)
        result = (imputed - self.means_) / self.scales_
        if not np.isfinite(result).all():
            raise ModelContractError("methane preprocessing produced non-finite values")
        return result


def _validate_feature_matrix(
    features: np.ndarray, expected_columns: int, *, allow_nan: bool
) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != expected_columns:
        raise ModelContractError("methane feature matrix has the wrong shape")
    if np.isinf(values).any() or (not allow_nan and np.isnan(values).any()):
        raise ModelContractError("methane features contain invalid values")
    return values


class PersistenceRiskRule:
    def __init__(
        self,
        concentration_threshold: float,
        transition_scale: float,
        *,
        missing_value: Optional[float] = None,
    ) -> None:
        if concentration_threshold <= 0 or transition_scale <= 0:
            raise ModelContractError("rule threshold and transition scale must be positive")
        if missing_value is not None and not np.isfinite(missing_value):
            raise ModelContractError("rule missing-value fallback must be finite")
        self.concentration_threshold = float(concentration_threshold)
        self.transition_scale = float(transition_scale)
        self.missing_value = None if missing_value is None else float(missing_value)

    def score(self, history_values: np.ndarray, *, sensor_index: int = 0) -> np.ndarray:
        values = np.asarray(history_values, dtype=np.float64)
        if values.ndim == 2:
            values = values[:, :, None]
        if values.ndim != 3 or not 0 <= sensor_index < values.shape[2]:
            raise ModelContractError("rule history has an invalid shape or sensor index")
        last_values = []
        for row in values[:, :, sensor_index]:
            finite = row[np.isfinite(row)]
            if finite.size == 0:
                if self.missing_value is None:
                    raise ModelContractError("rule requires one observed historical value")
                last_values.append(self.missing_value)
            else:
                last_values.append(float(finite[-1]))
        logits = (
            np.asarray(last_values) - self.concentration_threshold
        ) / self.transition_scale
        logits = np.clip(logits, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-logits))


class MethaneHGB:
    def __init__(
        self,
        feature_names: Sequence[str],
        *,
        random_state: int = 1701,
        max_iter: int = 100,
    ) -> None:
        self.feature_names = assert_causal_feature_names(feature_names)
        self.model = HistGradientBoostingClassifier(
            random_state=int(random_state), max_iter=int(max_iter)
        )
        self.fit_count = 0

    def fit(self, features: np.ndarray, labels: Iterable[int], *, pool: str) -> "MethaneHGB":
        if pool != METHANE_FIT_POOL:
            raise ModelContractError("HGB may fit only on D_b_tr")
        if self.fit_count:
            raise ModelContractError("HGB is frozen after its single fit")
        values = _validate_feature_matrix(features, len(self.feature_names), allow_nan=False)
        targets = np.asarray(list(labels), dtype=np.int64)
        if targets.shape != (values.shape[0],) or set(targets) != {0, 1}:
            raise ModelContractError("HGB requires aligned binary labels with both classes")
        self.model.fit(values, targets)
        self.fit_count = 1
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.fit_count != 1:
            raise ModelContractError("HGB has not completed its single fit")
        values = _validate_feature_matrix(features, len(self.feature_names), allow_nan=False)
        return self.model.predict_proba(values)[:, 1]


class MethaneGRU(nn.Module):
    def __init__(
        self,
        input_feature_names: Sequence[str],
        *,
        hidden_size: int = 64,
        layers: int = 1,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.input_feature_names = assert_causal_feature_names(input_feature_names)
        if hidden_size <= 0 or layers != 1 or not 0 <= dropout < 1:
            raise ModelContractError(
                "GRU requires positive hidden size, one layer, and dropout in [0, 1)"
            )
        self.hidden_size = int(hidden_size)
        self.layers = int(layers)
        self.dropout_probability = float(dropout)
        self.gru = nn.GRU(
            input_size=len(self.input_feature_names),
            hidden_size=self.hidden_size,
            num_layers=self.layers,
            batch_first=True,
            dropout=0.0,
        )
        self.output_dropout = nn.Dropout(self.dropout_probability)
        self.classifier = nn.Linear(self.hidden_size, 1)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(history) or history.ndim != 3:
            raise ModelContractError("GRU history must be a batch x time x feature tensor")
        if history.shape[2] != len(self.input_feature_names) or history.shape[1] == 0:
            raise ModelContractError("GRU history has the wrong feature or time dimension")
        if not history.is_floating_point() or not torch.isfinite(history).all():
            raise ModelContractError("GRU history must contain finite floating-point values")
        _, hidden = self.gru(history)
        representation = self.output_dropout(hidden[-1])
        return self.classifier(representation).squeeze(-1)

    def probabilities(self, history: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self(history))
