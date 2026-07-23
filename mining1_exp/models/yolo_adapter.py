from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GroupKFold
import torch
from torch import nn


YOLO_MODEL_FAMILY = "yolov8n"
CALIBRATION_POOL = "D_b_prob"
CALIBRATION_METHODS = {"temperature", "platt", "isotonic"}


class ModelContractError(ValueError):
    """Raised when a branch model violates the frozen implementation contract."""


def build_yolov8n_model() -> nn.Module:
    from ultralytics import YOLO

    model = YOLO("yolov8n.yaml", task="detect").model
    if not isinstance(model, nn.Module):
        raise ModelContractError("Ultralytics did not return a torch detection module")
    return model


class YoloV8nAdapter(nn.Module):
    def __init__(
        self,
        backend: Optional[nn.Module] = None,
        *,
        model_family: str = YOLO_MODEL_FAMILY,
    ) -> None:
        super().__init__()
        if model_family != YOLO_MODEL_FAMILY:
            raise ModelContractError("only the frozen YOLOv8n family is supported")
        self.model_family = model_family
        self.backend = backend if backend is not None else build_yolov8n_model()
        if not isinstance(self.backend, nn.Module):
            raise ModelContractError("YOLO backend must be a torch module")

    def forward(self, images: torch.Tensor) -> Any:
        if not torch.is_tensor(images) or images.ndim != 4:
            raise ModelContractError("YOLO input must be a BCHW tensor")
        if images.shape[1] != 3:
            raise ModelContractError("YOLOv8n input must contain exactly three channels")
        if images.shape[2] % 32 or images.shape[3] % 32:
            raise ModelContractError("YOLO input height and width must be divisible by 32")
        if not images.is_floating_point() or not torch.isfinite(images).all():
            raise ModelContractError("YOLO input must contain finite floating-point values")
        if images.numel() and (images.min() < 0 or images.max() > 1):
            raise ModelContractError("YOLO input must be normalized to [0, 1]")
        return self.backend(images)


def aggregate_concept_scores(
    detections: pd.DataFrame,
    *,
    record_ids: Sequence[str],
    concept_ids: Sequence[str],
    modality: str,
    post_nms: bool,
) -> pd.DataFrame:
    if not post_nms:
        raise ModelContractError("concept aggregation requires post-NMS detections")
    records = tuple(dict.fromkeys(str(value) for value in record_ids))
    concepts = tuple(dict.fromkeys(str(value) for value in concept_ids))
    if not records or not concepts or not str(modality).strip():
        raise ModelContractError("record IDs, concept IDs, and modality are required")
    required = {"record_id", "concept_id", "score_raw"}
    missing = sorted(required.difference(detections.columns))
    if missing:
        raise ModelContractError(f"detection table is missing columns: {missing}")
    if not detections.empty:
        unknown_records = set(detections["record_id"].astype(str)).difference(records)
        unknown_concepts = set(detections["concept_id"].astype(str)).difference(concepts)
        if unknown_records or unknown_concepts:
            raise ModelContractError("detections reference an unknown record or concept")
        scores = pd.to_numeric(detections["score_raw"], errors="raise")
        if (~np.isfinite(scores)).any() or ((scores < 0) | (scores > 1)).any():
            raise ModelContractError("detection scores must be finite probabilities")
        scored = detections.assign(
            record_id=detections["record_id"].astype(str),
            concept_id=detections["concept_id"].astype(str),
            score_raw=scores.astype(float),
        )
        maxima = scored.groupby(["record_id", "concept_id"])["score_raw"].max()
    else:
        maxima = pd.Series(dtype=float)

    rows = []
    for record_id in records:
        for concept_id in concepts:
            key = (record_id, concept_id)
            score = float(maxima.loc[key]) if key in maxima.index else 0.0
            rows.append(
                {
                    "record_id": record_id,
                    "concept_id": concept_id,
                    "modality": modality,
                    "step_score_raw": score,
                }
            )
    return pd.DataFrame(rows)


def _as_binary_arrays(
    scores: Iterable[float], labels: Iterable[int]
) -> tuple[np.ndarray, np.ndarray]:
    score_values = np.asarray(list(scores), dtype=np.float64)
    label_values = np.asarray(list(labels), dtype=np.int64)
    if score_values.ndim != 1 or label_values.ndim != 1 or len(score_values) != len(label_values):
        raise ModelContractError("calibration scores and labels must be aligned vectors")
    if len(score_values) < 2 or not np.isfinite(score_values).all():
        raise ModelContractError("calibration requires finite non-empty scores")
    if ((score_values < 0) | (score_values > 1)).any():
        raise ModelContractError("calibration scores must lie in [0, 1]")
    if set(label_values) != {0, 1}:
        raise ModelContractError("calibration requires both binary classes")
    return score_values, label_values


def _logit(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(scores, 1.0e-6, 1.0 - 1.0e-6)
    return np.log(clipped / (1.0 - clipped))


@dataclass
class BinaryProbabilityCalibrator:
    method: str
    fitted_pool: Optional[str] = None
    _model: Any = None

    def __post_init__(self) -> None:
        if self.method not in CALIBRATION_METHODS:
            raise ModelContractError(f"unsupported calibration method: {self.method}")

    def fit(
        self,
        scores: Iterable[float],
        labels: Iterable[int],
        *,
        pool: str,
        negatives_verified: bool,
    ) -> "BinaryProbabilityCalibrator":
        if pool != CALIBRATION_POOL:
            raise ModelContractError("detection calibration may fit only on D_b_prob")
        if not negatives_verified:
            raise ModelContractError("calibration requires verified negative semantics")
        score_values, label_values = _as_binary_arrays(scores, labels)
        logits = _logit(score_values)
        if self.method == "temperature":
            def objective(log_temperature: float) -> float:
                temperature = math.exp(log_temperature)
                probabilities = 1.0 / (1.0 + np.exp(-logits / temperature))
                probabilities = np.clip(probabilities, 1.0e-9, 1.0 - 1.0e-9)
                return float(
                    -np.mean(
                        label_values * np.log(probabilities)
                        + (1 - label_values) * np.log(1.0 - probabilities)
                    )
                )

            result = minimize_scalar(objective, bounds=(-5.0, 5.0), method="bounded")
            if not result.success:
                raise ModelContractError("temperature calibration optimization failed")
            self._model = float(math.exp(float(result.x)))
        elif self.method == "platt":
            model = LogisticRegression(random_state=0, solver="lbfgs")
            model.fit(logits.reshape(-1, 1), label_values)
            self._model = model
        else:
            model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            model.fit(score_values, label_values)
            self._model = model
        self.fitted_pool = pool
        return self

    def predict(self, scores: Iterable[float]) -> np.ndarray:
        if self._model is None or self.fitted_pool != CALIBRATION_POOL:
            raise ModelContractError("calibrator is not fitted on the locked pool")
        values = np.asarray(list(scores), dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ModelContractError("calibration prediction scores must lie in [0, 1]")
        if self.method == "temperature":
            probabilities = 1.0 / (1.0 + np.exp(-_logit(values) / float(self._model)))
        elif self.method == "platt":
            probabilities = self._model.predict_proba(_logit(values).reshape(-1, 1))[:, 1]
        else:
            probabilities = self._model.predict(values)
        return np.clip(np.asarray(probabilities, dtype=np.float64), 0.0, 1.0)


@dataclass
class CalibrationSelection:
    method: str
    calibrator: BinaryProbabilityCalibrator
    cv_brier_by_method: dict[str, float]
    fold_count: int


def select_calibrator_group_cv(
    scores: Iterable[float],
    labels: Iterable[int],
    raw_group_ids: Iterable[str],
    *,
    pool: str,
    negatives_verified: bool,
    folds: int,
    methods: Sequence[str] = ("temperature", "platt", "isotonic"),
) -> CalibrationSelection:
    if pool != CALIBRATION_POOL:
        raise ModelContractError("calibrator selection may use only D_b_prob")
    if not negatives_verified:
        raise ModelContractError("calibrator selection requires verified negatives")
    score_values, label_values = _as_binary_arrays(scores, labels)
    groups = np.asarray([str(value) for value in raw_group_ids], dtype=object)
    if groups.shape != score_values.shape or any(not value.strip() for value in groups):
        raise ModelContractError("calibration raw groups must align with scores")
    if folds < 2 or len(set(groups)) < folds:
        raise ModelContractError("calibration group CV has insufficient distinct groups")
    ordered_methods = tuple(dict.fromkeys(str(method) for method in methods))
    if not ordered_methods or not set(ordered_methods).issubset(CALIBRATION_METHODS):
        raise ModelContractError("calibration candidate methods are invalid")

    splitter = GroupKFold(n_splits=int(folds))
    losses: dict[str, float] = {}
    for method in ordered_methods:
        fold_losses = []
        for train_indices, validation_indices in splitter.split(
            score_values, label_values, groups
        ):
            if set(groups[train_indices]).intersection(groups[validation_indices]):
                raise ModelContractError("calibration raw group leaked across CV folds")
            calibrator = BinaryProbabilityCalibrator(method).fit(
                score_values[train_indices],
                label_values[train_indices],
                pool=pool,
                negatives_verified=negatives_verified,
            )
            predictions = calibrator.predict(score_values[validation_indices])
            fold_losses.append(
                float(brier_score_loss(label_values[validation_indices], predictions))
            )
        losses[method] = float(np.mean(fold_losses))
    method_order = {method: index for index, method in enumerate(ordered_methods)}
    selected_method = min(
        ordered_methods, key=lambda method: (losses[method], method_order[method])
    )
    selected = BinaryProbabilityCalibrator(selected_method).fit(
        score_values,
        label_values,
        pool=pool,
        negatives_verified=negatives_verified,
    )
    return CalibrationSelection(
        method=selected_method,
        calibrator=selected,
        cv_brier_by_method=losses,
        fold_count=int(folds),
    )
