from __future__ import annotations

import hashlib
import hmac
import json
import re
from typing import Any, Mapping, Sequence

import pandas as pd


FORBIDDEN_MODEL_COLUMNS = {
    "episode_id",
    "record_id",
    "raw_group_id",
    "pool",
    "template_instance_id",
    "pair_id",
    "event_position_role",
    "label",
    "truth",
    "skeleton_item_id",
}
EPISODE_POOLS = {"D_e_tr", "D_e_sel", "D_e_pol", "D_e_te"}
EPISODE_SKELETON_COLUMNS = {
    "skeleton_item_id",
    "episode_id",
    "step_index",
    "record_id",
    "raw_group_id",
    "concept_id",
    "node_id",
    "pool",
    "episode_seed",
    "generator_family_id",
    "template_instance_id",
    "template_family",
    "event_position_role",
    "source_split_hash",
    "template_lock_hash",
}
FORBIDDEN_SKELETON_COLUMNS = {
    "model_score",
    "model_prediction",
    "validation_metric",
    "test_metric",
    "model_saliency",
    "reliability_score",
    "label",
    "truth",
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
FUSION_ELIGIBILITY_REQUIRED_COLUMNS = {
    "concept_id",
    "eligible_branch_type_count",
    "eligible_branch_types",
    "episode_skeleton_candidate_hash",
    "exclusion_reason",
    "fusion_primary_eligible",
    "graph_primary_eligible",
    "independent_multibranch_positive_group_count",
    "multibranch_event_count_by_pool",
    "ontology_lock_hash",
    "split_manifest_hash",
}
FORBIDDEN_FUSION_ELIGIBILITY_COLUMNS = {
    "model_score",
    "validation_metric",
    "test_metric",
    "claim_direction",
    "fusion_eligible",
    "graph_eligible",
    "distinct_branch_type_count",
    "independent_raw_group_count",
}


class EpisodeContractError(ValueError):
    """Raised when join-only or truth-bearing fields enter model features."""


def make_skeleton_item_id(
    nonce: bytes,
    *,
    record_id: str,
    concept_id: str,
    node_id: str,
    instance_salt: bytes = b"",
) -> str:
    if len(nonce) < 16:
        raise EpisodeContractError("sealed skeleton nonce must contain at least 16 bytes")
    if instance_salt and len(instance_salt) < 16:
        raise EpisodeContractError("skeleton instance salt must contain at least 16 bytes")
    identity = json.dumps(
        {
            "record_id": record_id,
            "concept_id": concept_id,
            "node_id": node_id,
            "instance_salt_sha256": hashlib.sha256(instance_salt).hexdigest()
            if instance_salt
            else None,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hmac.new(nonce, identity, hashlib.sha256).hexdigest()


def validate_episode_skeleton_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(EPISODE_SKELETON_COLUMNS.difference(frame.columns))
    if missing:
        raise EpisodeContractError(f"episode skeleton is missing columns: {missing}")
    forbidden = sorted(FORBIDDEN_SKELETON_COLUMNS.intersection(frame.columns))
    if forbidden:
        raise EpisodeContractError(f"episode skeleton contains forbidden columns: {forbidden}")
    result = frame.copy()
    key = ["episode_id", "step_index", "concept_id", "node_id"]
    if result.duplicated(key, keep=False).any():
        raise EpisodeContractError("episode skeleton primary key is not unique")
    if result["skeleton_item_id"].duplicated(keep=False).any():
        raise EpisodeContractError("skeleton_item_id must be unique")
    if not result["skeleton_item_id"].astype(str).str.fullmatch(SHA256_PATTERN).all():
        raise EpisodeContractError("skeleton_item_id must be a SHA-256 string")
    for field in ("source_split_hash", "template_lock_hash"):
        if not result[field].astype(str).str.fullmatch(SHA256_PATTERN).all():
            raise EpisodeContractError(f"{field} must contain SHA-256 strings")
    result["step_index"] = pd.to_numeric(result["step_index"], errors="raise").astype("int64")
    result["episode_seed"] = pd.to_numeric(result["episode_seed"], errors="raise").astype("int64")
    if (result["step_index"] < 0).any() or (result["episode_seed"] <= 0).any():
        raise EpisodeContractError("step_index and episode_seed are outside their allowed range")
    if not set(result["pool"]).issubset(EPISODE_POOLS):
        raise EpisodeContractError("episode skeleton contains an unknown pool")
    assignment_counts = result.groupby(
        ["pool", "episode_seed", "raw_group_id"], dropna=False
    )["episode_id"].nunique()
    if (assignment_counts > 1).any():
        raise EpisodeContractError(
            "one raw group is assigned to multiple episodes within a pool and seed"
        )
    return result


def _normalize_string_list(value: Any, field: str) -> list[str]:
    if isinstance(value, str) or isinstance(value, Mapping):
        raise EpisodeContractError(f"{field} must be a list of strings")
    try:
        normalized = [str(item) for item in value]
    except TypeError as exc:
        raise EpisodeContractError(f"{field} must be a list of strings") from exc
    if any(not item.strip() for item in normalized):
        raise EpisodeContractError(f"{field} contains an empty value")
    if normalized != sorted(set(normalized)):
        raise EpisodeContractError(f"{field} must be sorted and unique")
    return normalized


def validate_fusion_eligibility_lock(
    frame: pd.DataFrame,
    *,
    positive_group_floor: int | None = None,
) -> pd.DataFrame:
    missing = sorted(FUSION_ELIGIBILITY_REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise EpisodeContractError(
            f"fusion eligibility lock is missing columns: {missing}"
        )
    forbidden = sorted(FORBIDDEN_FUSION_ELIGIBILITY_COLUMNS.intersection(frame.columns))
    if forbidden:
        raise EpisodeContractError(
            f"fusion eligibility lock contains obsolete or forbidden columns: {forbidden}"
        )
    result = frame.copy()
    if result.empty:
        raise EpisodeContractError("fusion eligibility lock is empty")
    if result["concept_id"].astype(str).duplicated(keep=False).any():
        raise EpisodeContractError("fusion eligibility concept IDs are not unique")
    if not pd.api.types.is_bool_dtype(result["fusion_primary_eligible"]):
        raise EpisodeContractError("fusion_primary_eligible must be boolean")
    if not pd.api.types.is_bool_dtype(result["graph_primary_eligible"]):
        raise EpisodeContractError("graph_primary_eligible must be boolean")
    if (
        result["graph_primary_eligible"].astype(bool)
        & ~result["fusion_primary_eligible"].astype(bool)
    ).any():
        raise EpisodeContractError(
            "graph_primary_eligible cannot be true when fusion is ineligible"
        )
    for field in (
        "episode_skeleton_candidate_hash",
        "ontology_lock_hash",
        "split_manifest_hash",
    ):
        if not result[field].astype(str).str.fullmatch(SHA256_PATTERN).all():
            raise EpisodeContractError(f"{field} must contain SHA-256 strings")
    for field in (
        "eligible_branch_type_count",
        "independent_multibranch_positive_group_count",
    ):
        numeric = pd.to_numeric(result[field], errors="raise")
        if (numeric < 0).any() or (numeric % 1 != 0).any():
            raise EpisodeContractError(f"{field} must contain nonnegative integers")
        result[field] = numeric.astype("int64")
    for index, item in result.iterrows():
        branch_types = _normalize_string_list(
            item["eligible_branch_types"], "eligible_branch_types"
        )
        if int(item["eligible_branch_type_count"]) != len(branch_types):
            raise EpisodeContractError(
                "eligible_branch_type_count does not match eligible_branch_types"
            )
        pool_counts = item["multibranch_event_count_by_pool"]
        if not isinstance(pool_counts, Mapping) or set(pool_counts) != EPISODE_POOLS:
            raise EpisodeContractError(
                "multibranch_event_count_by_pool must map all four episode pools"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in pool_counts.values()
        ):
            raise EpisodeContractError(
                "multibranch_event_count_by_pool must contain nonnegative integers"
            )
        eligible = bool(item["fusion_primary_eligible"])
        reason = item["exclusion_reason"]
        reason_missing = reason is None or (isinstance(reason, float) and pd.isna(reason))
        if eligible:
            if len(branch_types) < 2:
                raise EpisodeContractError(
                    "fusion eligibility requires at least two branch types"
                )
            if positive_group_floor is not None and int(
                item["independent_multibranch_positive_group_count"]
            ) < int(positive_group_floor):
                raise EpisodeContractError(
                    "fusion eligibility does not meet the locked positive-group floor"
                )
            if not reason_missing:
                raise EpisodeContractError(
                    "eligible fusion rows must not carry an exclusion reason"
                )
        elif reason_missing or not str(reason).strip():
            raise EpisodeContractError(
                "ineligible fusion rows require an exclusion reason"
            )
    if "model_outcomes_used" in result.columns and result[
        "model_outcomes_used"
    ].astype(bool).any():
        raise EpisodeContractError(
            "fusion eligibility cannot use model outcomes"
        )
    return result


def select_model_features(frame: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    if not feature_columns:
        raise EpisodeContractError("model feature columns are required")
    forbidden = sorted(set(feature_columns).intersection(FORBIDDEN_MODEL_COLUMNS))
    if forbidden:
        raise EpisodeContractError(f"forbidden model feature columns: {forbidden}")
    missing = sorted(set(feature_columns).difference(frame.columns))
    if missing:
        raise EpisodeContractError(f"missing model feature columns: {missing}")
    return frame.loc[:, list(feature_columns)].copy()
