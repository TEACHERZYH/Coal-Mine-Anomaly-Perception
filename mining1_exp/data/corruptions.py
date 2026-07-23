from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


CORRUPTION_TYPES = {"low_light", "dust_fog_proxy"}
CORRUPTION_SEVERITIES = {1, 2, 3}
CORRUPTION_COLUMNS = {
    "record_id",
    "raw_group_id",
    "corruption_type",
    "severity",
    "operator_version",
    "generator_hash",
    "parameter_json",
    "source_feature_hash",
    "corrupted_feature_hash",
    "label_independent",
}
FORBIDDEN_COLUMNS = {
    "box",
    "boxes",
    "label",
    "labels",
    "truth",
    "prediction",
    "saliency",
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class CorruptionContractError(ValueError):
    """Raised when a corruption can depend on labels or is not reproducible."""


def corruption_parameters(corruption_type: str, severity: int) -> Dict[str, float]:
    if corruption_type not in CORRUPTION_TYPES:
        raise CorruptionContractError(f"unsupported corruption type: {corruption_type}")
    if int(severity) not in CORRUPTION_SEVERITIES:
        raise CorruptionContractError("corruption severity must be 1, 2, or 3")
    if corruption_type == "low_light":
        return {"gamma": {1: 1.3, 2: 1.7, 3: 2.1}[int(severity)]}
    return {
        "mean_transmission": {1: 0.80, 2: 0.65, 3: 0.50}[int(severity)],
        "airlight": 0.90,
        "transmission_noise_sd": 0.04,
    }


def apply_corruption(
    image: np.ndarray,
    *,
    record_id: str,
    corruption_type: str,
    severity: int,
    operator_version: str,
) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise CorruptionContractError("corruption input must be a uint8 NumPy array")
    if image.ndim not in {2, 3} or image.size == 0:
        raise CorruptionContractError("corruption input must be a non-empty image array")
    if not str(record_id).strip() or not str(operator_version).strip():
        raise CorruptionContractError("record_id and operator_version are required")
    parameters = corruption_parameters(corruption_type, int(severity))
    values = image.astype(np.float32) / 255.0
    if corruption_type == "low_light":
        corrupted = np.power(values, parameters["gamma"])
    else:
        seed_text = f"{record_id}|{corruption_type}|{int(severity)}|{operator_version}"
        seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16)
        raw_field = np.random.default_rng(seed).normal(0.0, 1.0, size=image.shape[:2])
        sigma = max(1.0, max(image.shape[:2]) / 12.0)
        smooth = gaussian_filter(raw_field, sigma=sigma, mode="reflect")
        smooth = (smooth - float(np.mean(smooth))) / max(float(np.std(smooth)), 1.0e-6)
        transmission = np.clip(
            parameters["mean_transmission"]
            + parameters["transmission_noise_sd"] * smooth,
            0.05,
            1.0,
        )
        if values.ndim == 3:
            transmission = transmission[:, :, None]
        corrupted = (
            values * transmission
            + parameters["airlight"] * (1.0 - transmission)
        )
    return np.ascontiguousarray(
        np.clip(np.rint(np.clip(corrupted, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)
    )


def array_sha256(image: np.ndarray) -> str:
    header = json.dumps(
        {"dtype": str(image.dtype), "shape": list(image.shape)},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(header + b"\0" + np.ascontiguousarray(image).tobytes()).hexdigest()


def build_corruption_record(
    *,
    image: np.ndarray,
    record_id: str,
    raw_group_id: str,
    corruption_type: str,
    severity: int,
    operator_version: str,
    generator_hash: str,
    source_feature_hash: str,
) -> Dict[str, Any]:
    corrupted = apply_corruption(
        image,
        record_id=record_id,
        corruption_type=corruption_type,
        severity=severity,
        operator_version=operator_version,
    )
    row = {
        "record_id": record_id,
        "raw_group_id": raw_group_id,
        "corruption_type": corruption_type,
        "severity": int(severity),
        "operator_version": operator_version,
        "generator_hash": generator_hash,
        "parameter_json": json.dumps(
            corruption_parameters(corruption_type, int(severity)),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ),
        "source_feature_hash": source_feature_hash,
        "corrupted_feature_hash": array_sha256(corrupted),
        "label_independent": True,
    }
    validate_corruption_manifest(pd.DataFrame([row]))
    return row


def validate_corruption_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(CORRUPTION_COLUMNS.difference(frame.columns))
    if missing:
        raise CorruptionContractError(f"corruption manifest is missing columns: {missing}")
    forbidden = sorted(FORBIDDEN_COLUMNS.intersection(frame.columns))
    if forbidden:
        raise CorruptionContractError(f"corruption manifest contains forbidden fields: {forbidden}")
    result = frame.copy()
    key = ["record_id", "corruption_type", "severity"]
    if result.duplicated(key, keep=False).any():
        raise CorruptionContractError("corruption manifest primary key is not unique")
    if not set(result["corruption_type"]).issubset(CORRUPTION_TYPES):
        raise CorruptionContractError("corruption manifest contains an unknown type")
    severity = pd.to_numeric(result["severity"], errors="raise").astype("int64")
    if not set(severity).issubset(CORRUPTION_SEVERITIES):
        raise CorruptionContractError("corruption manifest severity must be 1, 2, or 3")
    result["severity"] = severity
    for field in ("generator_hash", "source_feature_hash", "corrupted_feature_hash"):
        if not result[field].astype(str).str.fullmatch(SHA256_PATTERN).all():
            raise CorruptionContractError(f"{field} must contain SHA-256 strings")
    if not result["label_independent"].map(lambda value: value is True).all():
        raise CorruptionContractError("every corruption row must be label independent")
    if (result["operator_version"].astype(str).str.strip() == "").any():
        raise CorruptionContractError("operator_version is required")
    for value in result["parameter_json"]:
        parsed = json.loads(str(value))
        if not isinstance(parsed, Mapping):
            raise CorruptionContractError("parameter_json must encode an object")
    return result
