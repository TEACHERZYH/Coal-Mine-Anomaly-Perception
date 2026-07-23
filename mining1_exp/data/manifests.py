from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile
from typing import Iterable, Optional, Sequence, Set, Union

import pandas as pd


PathLike = Union[str, Path]
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
POOLS = {
    "D_b_tr",
    "D_b_sel",
    "D_b_prob",
    "D_b_te",
    "D_e_tr",
    "D_e_sel",
    "D_e_pol",
    "D_e_te",
}
ARCHIVE_ROUTES = {
    "local_existing",
    "remote_existing",
    "domestic_remote_download",
    "local_download_upload",
}

ARCHIVE_COLUMNS = {
    "dataset_id",
    "archive_id",
    "source_url_or_doi",
    "access_date",
    "license_id",
    "version",
    "local_or_remote_path",
    "byte_size",
    "sha256",
    "acquisition_route",
    "verification_status",
}
FILE_COLUMNS = {
    "dataset_id",
    "record_id",
    "archive_id",
    "relative_path",
    "modality",
    "raw_group_id",
    "pair_id",
    "sequence_id",
    "timestamp_or_order",
    "label_summary_json",
    "byte_size",
    "sha256",
}
FILE_FEATURE_COLUMNS = FILE_COLUMNS.difference({"label_summary_json"})
SPLIT_COLUMNS = {
    "dataset_id",
    "record_id",
    "raw_group_id",
    "pool",
    "split_seed",
    "split_version",
    "ontology_hash",
    "dedup_report_hash",
}


class ManifestValidationError(ValueError):
    """Raised when an artifact manifest violates its frozen schema."""


def _require_columns(frame: pd.DataFrame, required: Set[str], name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ManifestValidationError(f"{name} is missing columns: {missing}")


def _require_unique(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    if frame.duplicated(list(columns), keep=False).any():
        raise ManifestValidationError(f"{name} primary key is not unique: {columns}")


def _require_sha256(series: pd.Series, name: str) -> None:
    invalid = ~series.astype(str).str.fullmatch(SHA256_PATTERN)
    if invalid.any():
        raise ManifestValidationError(f"{name} contains invalid SHA-256 values")


def validate_archive_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    _require_columns(result, ARCHIVE_COLUMNS, "archive_manifest")
    _require_unique(result, ["dataset_id", "archive_id"], "archive_manifest")
    sizes = pd.to_numeric(result["byte_size"], errors="raise")
    if (sizes < 0).any():
        raise ManifestValidationError("archive byte_size must be nonnegative")
    result["byte_size"] = sizes.astype("int64")
    _require_sha256(result["sha256"], "archive_manifest.sha256")
    if not set(result["acquisition_route"]).issubset(ARCHIVE_ROUTES):
        raise ManifestValidationError("archive_manifest has an unsupported acquisition_route")
    if not set(result["verification_status"]).issubset({"pass", "reject", "blocked"}):
        raise ManifestValidationError("archive_manifest has an invalid verification_status")
    return result


def validate_file_manifest(
    frame: pd.DataFrame, paired_dataset_ids: Iterable[str] = ()
) -> pd.DataFrame:
    result = frame.copy()
    _require_columns(result, FILE_COLUMNS, "file_manifest")
    _require_unique(result, ["dataset_id", "record_id"], "file_manifest")
    _require_unique(result, ["dataset_id", "relative_path"], "file_manifest")
    if result["raw_group_id"].isna().any() or (
        result["raw_group_id"].astype(str).str.strip() == ""
    ).any():
        raise ManifestValidationError("raw_group_id must be generated before split")
    sizes = pd.to_numeric(result["byte_size"], errors="raise")
    if (sizes < 0).any():
        raise ManifestValidationError("file byte_size must be nonnegative")
    result["byte_size"] = sizes.astype("int64")
    _require_sha256(result["sha256"], "file_manifest.sha256")
    for value in result["label_summary_json"]:
        parsed = json.loads(str(value))
        if not isinstance(parsed, dict):
            raise ManifestValidationError("label_summary_json must encode an object")
    paired = set(paired_dataset_ids)
    if paired:
        mask = result["dataset_id"].isin(paired)
        if result.loc[mask, "pair_id"].isna().any() or (
            result.loc[mask, "pair_id"].astype(str).str.strip() == ""
        ).any():
            raise ManifestValidationError("declared paired datasets require pair_id")
    return result


def validate_split_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    _require_columns(result, SPLIT_COLUMNS, "split_manifest")
    _require_unique(result, ["dataset_id", "record_id"], "split_manifest")
    if not set(result["pool"]).issubset(POOLS):
        raise ManifestValidationError("split_manifest contains an unknown pool")
    group_pool_counts = result.groupby(
        ["dataset_id", "raw_group_id"], dropna=False
    )["pool"].nunique()
    if (group_pool_counts > 1).any():
        raise ManifestValidationError("one raw group appears in multiple pools")
    _require_sha256(result["ontology_hash"], "split_manifest.ontology_hash")
    _require_sha256(result["dedup_report_hash"], "split_manifest.dedup_report_hash")
    return result


def read_file_manifest(path: PathLike) -> pd.DataFrame:
    return validate_file_manifest(pd.read_parquet(Path(path)))


def read_file_feature_manifest(path: PathLike) -> pd.DataFrame:
    """Read model-facing file metadata without materializing label summaries."""
    result = pd.read_parquet(Path(path), columns=sorted(FILE_FEATURE_COLUMNS))
    _require_columns(result, FILE_FEATURE_COLUMNS, "file_feature_manifest")
    _require_unique(result, ["dataset_id", "record_id"], "file_feature_manifest")
    _require_unique(result, ["dataset_id", "relative_path"], "file_feature_manifest")
    if result["raw_group_id"].isna().any() or (
        result["raw_group_id"].astype(str).str.strip() == ""
    ).any():
        raise ManifestValidationError("raw_group_id must be generated before split")
    _require_sha256(result["sha256"], "file_feature_manifest.sha256")
    return result


def read_split_manifest(path: PathLike) -> pd.DataFrame:
    return validate_split_manifest(pd.read_parquet(Path(path)))


def write_parquet_atomic(frame: pd.DataFrame, path: PathLike) -> Path:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False)
        if target.exists():
            if not target.is_file() or target.read_bytes() != temporary.read_bytes():
                raise FileExistsError(f"immutable Parquet artifact differs: {target}")
            return target
        try:
            os.link(str(temporary), str(target))
        except FileExistsError:
            if not target.is_file() or target.read_bytes() != temporary.read_bytes():
                raise FileExistsError(f"immutable Parquet artifact race differs: {target}")
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)
    return target


def validate_methane_windows(
    frame: pd.DataFrame, purge_gap_seconds: int
) -> pd.DataFrame:
    required = {
        "dataset_id",
        "window_id",
        "raw_group_id",
        "sensor_group_id",
        "history_start",
        "history_end",
        "forecast_end",
        "pool",
    }
    _require_columns(frame, required, "methane_window_manifest")
    result = frame.copy()
    for column in ("history_start", "history_end", "forecast_end"):
        result[column] = pd.to_datetime(result[column], utc=True, errors="raise")
    if not (
        (result["history_start"] < result["history_end"])
        & (result["history_end"] < result["forecast_end"])
    ).all():
        raise ManifestValidationError("methane windows must be strictly causal")
    if purge_gap_seconds < 0:
        raise ManifestValidationError("purge_gap_seconds must be nonnegative")
    for _, group in result.groupby("sensor_group_id"):
        latest_forecast_by_pool = {}
        for row in group.sort_values("history_start").itertuples(index=False):
            for previous_pool, previous_end in latest_forecast_by_pool.items():
                if previous_pool == row.pool:
                    continue
                gap = (row.history_start - previous_end).total_seconds()
                if gap < purge_gap_seconds:
                    raise ManifestValidationError(
                        "cross-pool methane windows violate purge gap"
                    )
            current_end = latest_forecast_by_pool.get(row.pool)
            if current_end is None or row.forecast_end > current_end:
                latest_forecast_by_pool[row.pool] = row.forecast_end
    return result


def assert_fit_pool(pool: str, allowed_pools: Iterable[str]) -> None:
    allowed = set(allowed_pools)
    if pool not in allowed:
        raise ManifestValidationError(f"pool {pool} is not authorized for fitting")
