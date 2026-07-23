"""Prediction entry points implemented after model contracts pass."""

from .methane import predict_s1_sealed
from .detection import predict_detection_package

__all__ = ["predict_detection_package", "predict_s1_sealed"]
