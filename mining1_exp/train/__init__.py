"""Frozen training entry points for the clean-room experiment package."""

from .detection import (
    DetectionRunSpec,
    t1_spec,
    train_detection_run,
    v2_finetune_spec,
    v2_pretrain_spec,
)
from .methane import fit_s1_baselines, train_s1_gru

__all__ = [
    "DetectionRunSpec",
    "fit_s1_baselines",
    "t1_spec",
    "train_detection_run",
    "train_s1_gru",
    "v2_finetune_spec",
    "v2_pretrain_spec",
]
