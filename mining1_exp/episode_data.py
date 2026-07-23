from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .data.episodes import validate_fusion_eligibility_lock
from .models.episode_fusion import assert_fusion_feature_names
from .workflow_common import WorkflowExecutionError


EPISODE_POOLS = {"D_e_tr", "D_e_sel", "D_e_pol", "D_e_te"}
LABELED_NON_TEST_POOLS = {"D_e_tr", "D_e_sel", "D_e_pol"}
QUALITY_FEATURE_NAMES = assert_fusion_feature_names(
    ("probability_margin", "branch_available")
)
PAIR_FEATURE_NAMES = assert_fusion_feature_names(("observed_pair_indicator",))
KEY_COLUMNS = ("episode_id", "step_index", "concept_id")


@dataclass(frozen=True)
class EpisodeArrays:
    pool: str
    metadata: pd.DataFrame
    probabilities: np.ndarray
    quality: np.ndarray
    availability: np.ndarray
    concept_embedding: np.ndarray
    labels: Optional[np.ndarray]
    edge_index: np.ndarray
    pair_features: np.ndarray
    edge_validity: np.ndarray
    node_ids: tuple[str, ...]
    concept_ids: tuple[str, ...]
    quality_feature_names: tuple[str, ...]
    pair_feature_names: tuple[str, ...]


def fusion_eligible_concepts(project_root: Path) -> tuple[str, ...]:
    path = project_root / "data/locked/fusion_eligibility_lock.parquet"
    frame = validate_fusion_eligibility_lock(pd.read_parquet(path))
    concepts = tuple(
        sorted(
            frame.loc[
                frame["fusion_primary_eligible"].astype(bool), "concept_id"
            ].astype(str)
        )
    )
    if not concepts:
        raise WorkflowExecutionError("Fusion-eligible episode population is empty")
    return concepts


def training_node_ids(project_root: Path) -> tuple[str, ...]:
    path = project_root / "data/locked/episode_features/D_e_tr.parquet"
    frame = pd.read_parquet(path, columns=["concept_id", "node_id"])
    concepts = set(fusion_eligible_concepts(project_root))
    nodes = tuple(
        sorted(
            frame.loc[
                frame["concept_id"].astype(str).isin(concepts), "node_id"
            ].astype(str)
        )
    )
    nodes = tuple(dict.fromkeys(nodes))
    if len(nodes) < 2:
        raise WorkflowExecutionError(
            "Fusion training pool does not expose at least two branch nodes"
        )
    return nodes


def _read_features(project_root: Path, pool: str) -> pd.DataFrame:
    path = project_root / f"data/locked/episode_features/{pool}.parquet"
    if not path.is_file():
        raise WorkflowExecutionError(f"Episode feature partition is missing: {pool}")
    frame = pd.read_parquet(path)
    required = {
        *KEY_COLUMNS,
        "record_id",
        "node_id",
        "modality",
        "calibrated_probability",
        "available",
        "quality_vector_json",
    }
    if not required.issubset(frame.columns):
        raise WorkflowExecutionError(f"Episode feature schema is incomplete: {pool}")
    forbidden = {
        "pool",
        "event_truth",
        "state_truth",
        "raw_group_id",
        "source_component_id",
        "pair_id",
        "dataset_id",
        "generator_family_id",
        "template_instance_id",
        "template_family",
    }
    leaked = sorted(forbidden.intersection(frame.columns))
    if leaked:
        raise WorkflowExecutionError(f"Episode model features expose audit fields: {leaked}")
    return frame


def _quality_vector(value: object, *, available: bool, probability: float) -> np.ndarray:
    payload = json.loads(str(value))
    allowed = {"available", "probability_margin"}
    if not isinstance(payload, dict) or set(payload) != allowed:
        raise WorkflowExecutionError("Episode quality-vector contract drifted")
    margin = float(payload["probability_margin"])
    payload_available = bool(payload["available"])
    if payload_available != available or not np.isclose(
        margin, abs(float(probability) - 0.5)
    ):
        raise WorkflowExecutionError("Episode quality vector does not match branch output")
    return np.asarray([margin, float(available)], dtype=np.float32)


def load_episode_arrays(
    project_root: Path,
    *,
    pool: str,
    include_labels: bool,
    node_ids: Optional[Sequence[str]] = None,
    concept_ids: Optional[Sequence[str]] = None,
) -> EpisodeArrays:
    pool = str(pool)
    if pool not in EPISODE_POOLS:
        raise WorkflowExecutionError(f"Unknown episode pool: {pool}")
    if include_labels and pool not in LABELED_NON_TEST_POOLS:
        raise WorkflowExecutionError("Episode test labels are never available to model code")
    concepts = tuple(concept_ids or fusion_eligible_concepts(project_root))
    if not concepts or len(set(concepts)) != len(concepts):
        raise WorkflowExecutionError("Episode concept axis is empty or duplicated")
    nodes = tuple(node_ids or training_node_ids(project_root))
    if len(nodes) < 2 or len(set(nodes)) != len(nodes):
        raise WorkflowExecutionError("Episode node axis is invalid")
    features = _read_features(project_root, pool)
    features = features.loc[
        features["concept_id"].astype(str).isin(concepts)
    ].copy()
    if features.empty:
        raise WorkflowExecutionError(f"No fusion-eligible episode rows exist in {pool}")
    unknown_nodes = sorted(set(features["node_id"].astype(str)) - set(nodes))
    if unknown_nodes:
        raise WorkflowExecutionError(
            f"Episode partition contains nodes absent from D_e_tr: {unknown_nodes}"
        )
    if features.duplicated([*KEY_COLUMNS, "node_id"]).any():
        raise WorkflowExecutionError("Episode feature node keys are duplicated")
    metadata = (
        features.loc[:, list(KEY_COLUMNS)]
        .drop_duplicates()
        .sort_values(list(KEY_COLUMNS), kind="stable")
        .reset_index(drop=True)
    )
    key_to_index = {
        tuple(value): index
        for index, value in enumerate(metadata.itertuples(index=False, name=None))
    }
    node_to_index = {value: index for index, value in enumerate(nodes)}
    concept_to_index = {value: index for index, value in enumerate(concepts)}
    count = len(metadata)
    probabilities = np.full((count, len(nodes)), np.nan, dtype=np.float32)
    availability = np.zeros((count, len(nodes)), dtype=bool)
    quality = np.full(
        (count, len(nodes), len(QUALITY_FEATURE_NAMES)), np.nan, dtype=np.float32
    )
    for item in features.itertuples(index=False):
        key = (str(item.episode_id), int(item.step_index), str(item.concept_id))
        row_index = key_to_index[key]
        node_index = node_to_index[str(item.node_id)]
        probability = float(item.calibrated_probability)
        available = bool(item.available)
        if available and (not np.isfinite(probability) or not 0 <= probability <= 1):
            raise WorkflowExecutionError("Available episode probability is invalid")
        probabilities[row_index, node_index] = probability if available else np.nan
        availability[row_index, node_index] = available
        quality[row_index, node_index] = _quality_vector(
            item.quality_vector_json,
            available=available,
            probability=probability,
        )
    if not availability.any(axis=1).all():
        raise WorkflowExecutionError("Episode concept-step has no available branch")
    concept_embedding = np.zeros((count, len(concepts)), dtype=np.float32)
    for index, concept in enumerate(metadata["concept_id"].astype(str)):
        concept_embedding[index, concept_to_index[concept]] = 1.0
    labels: Optional[np.ndarray] = None
    if include_labels:
        label_path = project_root / f"data/locked/episode_labels/{pool}.parquet"
        labels_frame = pd.read_parquet(label_path)
        required_labels = {*KEY_COLUMNS, "event_truth"}
        if not required_labels.issubset(labels_frame.columns):
            raise WorkflowExecutionError("Episode label schema is incomplete")
        selected_labels = labels_frame.loc[
            labels_frame["concept_id"].astype(str).isin(concepts),
            [*KEY_COLUMNS, "event_truth"],
        ]
        merged = metadata.merge(
            selected_labels,
            on=list(KEY_COLUMNS),
            validate="one_to_one",
        )
        if len(merged) != len(metadata):
            raise WorkflowExecutionError("Episode labels do not close feature keys")
        labels = pd.to_numeric(merged["event_truth"], errors="raise").to_numpy(
            dtype=np.float32
        )
        if not set(labels.tolist()).issubset({0.0, 1.0}):
            raise WorkflowExecutionError("Episode labels are not binary")
    edge_pairs = [
        (source, target)
        for source in range(len(nodes))
        for target in range(len(nodes))
        if source != target
    ]
    edge_index = np.asarray(edge_pairs, dtype=np.int64).T
    pair_features = np.zeros(
        (count, len(edge_pairs), len(PAIR_FEATURE_NAMES)), dtype=np.float32
    )
    edge_validity = np.zeros((count, len(edge_pairs)), dtype=bool)
    edge_lookup = {pair: index for index, pair in enumerate(edge_pairs)}
    edge_path = project_root / f"data/locked/episode_graph_edges/{pool}.parquet"
    if edge_path.is_file():
        edges = pd.read_parquet(edge_path)
        required_edges = {
            *KEY_COLUMNS,
            "source_node_id",
            "target_node_id",
            "edge_contract_hash",
        }
        if not required_edges.issubset(edges.columns):
            raise WorkflowExecutionError("Episode graph-edge schema is incomplete")
        for item in edges.itertuples(index=False):
            key = (str(item.episode_id), int(item.step_index), str(item.concept_id))
            if key not in key_to_index:
                continue
            source = node_to_index.get(str(item.source_node_id))
            target = node_to_index.get(str(item.target_node_id))
            if source is None or target is None or source == target:
                raise WorkflowExecutionError("Episode graph edge has an invalid endpoint")
            edge_id = edge_lookup[(source, target)]
            pair_features[key_to_index[key], edge_id, 0] = 1.0
            edge_validity[key_to_index[key], edge_id] = True
    return EpisodeArrays(
        pool=pool,
        metadata=metadata,
        probabilities=probabilities,
        quality=quality,
        availability=availability,
        concept_embedding=concept_embedding,
        labels=labels,
        edge_index=edge_index,
        pair_features=pair_features,
        edge_validity=edge_validity,
        node_ids=nodes,
        concept_ids=concepts,
        quality_feature_names=tuple(QUALITY_FEATURE_NAMES),
        pair_feature_names=tuple(PAIR_FEATURE_NAMES),
    )
