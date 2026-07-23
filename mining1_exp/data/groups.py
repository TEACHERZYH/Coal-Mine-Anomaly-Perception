from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import itertools
import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from .manifests import ManifestValidationError, SHA256_PATTERN


def assign_raw_group_ids(
    frame: pd.DataFrame, identity_columns: Sequence[str]
) -> pd.DataFrame:
    if not identity_columns:
        raise ManifestValidationError("raw group identity columns are required")
    missing = [column for column in identity_columns if column not in frame.columns]
    if missing:
        raise ManifestValidationError(f"missing raw group identity columns: {missing}")
    if frame[list(identity_columns)].isna().any().any():
        raise ManifestValidationError("raw group identity cannot contain null values")

    result = frame.copy()
    identifiers: List[str] = []
    for row in result[list(identity_columns)].itertuples(index=False, name=None):
        preimage = json.dumps(
            list(row), ensure_ascii=True, separators=(",", ":"), sort_keys=False
        ).encode("utf-8")
        identifiers.append("grp_" + hashlib.sha256(preimage).hexdigest()[:24])
    result["raw_group_id"] = identifiers
    return result


def _hamming_distance(left: int, right: int) -> int:
    return bin(left ^ right).count("1")


@dataclass
class _BKNode:
    value: int
    payloads: List[Mapping[str, Any]] = field(default_factory=list)
    children: Dict[int, "_BKNode"] = field(default_factory=dict)

    def search(self, value: int, threshold: int) -> List[Tuple[int, Mapping[str, Any]]]:
        distance = _hamming_distance(self.value, value)
        matches = [(distance, payload) for payload in self.payloads] if distance <= threshold else []
        for child_distance, child in self.children.items():
            if distance - threshold <= child_distance <= distance + threshold:
                matches.extend(child.search(value, threshold))
        return matches

    def insert(self, value: int, payload: Mapping[str, Any]) -> None:
        distance = _hamming_distance(self.value, value)
        if distance == 0:
            self.payloads.append(payload)
            return
        child = self.children.get(distance)
        if child is None:
            self.children[distance] = _BKNode(value=value, payloads=[payload])
        else:
            child.insert(value, payload)


def audit_duplicates(
    frame: pd.DataFrame,
    *,
    phash_column: Optional[str] = None,
    metadata_columns: Sequence[str] = (),
    hamming_threshold: int = 0,
) -> pd.DataFrame:
    required = {"dataset_id", "record_id", "sha256"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ManifestValidationError(f"duplicate audit is missing columns: {missing}")
    if not frame["sha256"].astype(str).str.fullmatch(SHA256_PATTERN).all():
        raise ManifestValidationError("duplicate audit requires valid SHA-256 values")
    if hamming_threshold < 0:
        raise ManifestValidationError("hamming_threshold must be nonnegative")
    if phash_column is not None and phash_column not in frame.columns:
        raise ManifestValidationError(f"pHash column not found: {phash_column}")
    for column in metadata_columns:
        if column not in frame.columns:
            raise ManifestValidationError(f"duplicate metadata column not found: {column}")

    rows = frame.sort_values(["dataset_id", "record_id"]).to_dict("records")
    pairs: List[Dict[str, Any]] = []
    seen_pairs = set()
    exact_groups: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        exact_groups.setdefault(str(row["sha256"]), []).append(row)
    for digest in sorted(exact_groups):
        duplicate_rows = exact_groups[digest]
        for left, right in itertools.combinations(duplicate_rows, 2):
            key = tuple(sorted(((left["dataset_id"], left["record_id"]), (right["dataset_id"], right["record_id"]))))
            seen_pairs.add(key)
            pairs.append(_pair_row(left, right, "exact", 0))

    if phash_column is not None:
        trees: Dict[Tuple[Any, ...], _BKNode] = {}
        for row in rows:
            raw_phash = row.get(phash_column)
            if raw_phash is None or pd.isna(raw_phash) or str(raw_phash).strip() == "":
                continue
            try:
                phash = int(str(raw_phash), 16)
            except ValueError as exc:
                raise ManifestValidationError(f"invalid pHash: {raw_phash}") from exc
            metadata_key = tuple(row[column] for column in metadata_columns)
            tree = trees.get(metadata_key)
            if tree is None:
                trees[metadata_key] = _BKNode(value=phash, payloads=[row])
                continue
            for distance, previous in tree.search(phash, hamming_threshold):
                key = tuple(sorted(((previous["dataset_id"], previous["record_id"]), (row["dataset_id"], row["record_id"]))))
                if key in seen_pairs or previous["sha256"] == row["sha256"]:
                    continue
                seen_pairs.add(key)
                pairs.append(_pair_row(previous, row, "near", distance))
            tree.insert(phash, row)

    columns = [
        "left_dataset_id",
        "left_record_id",
        "right_dataset_id",
        "right_record_id",
        "duplicate_type",
        "phash_hamming_distance",
    ]
    return pd.DataFrame(pairs, columns=columns).sort_values(columns[:4]).reset_index(drop=True)


def _pair_row(
    left: Mapping[str, Any], right: Mapping[str, Any], duplicate_type: str, distance: int
) -> Dict[str, Any]:
    first, second = sorted(
        (left, right), key=lambda row: (str(row["dataset_id"]), str(row["record_id"]))
    )
    return {
        "left_dataset_id": first["dataset_id"],
        "left_record_id": first["record_id"],
        "right_dataset_id": second["dataset_id"],
        "right_record_id": second["record_id"],
        "duplicate_type": duplicate_type,
        "phash_hamming_distance": int(distance),
    }
