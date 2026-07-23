from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, Iterable, Mapping, NamedTuple, Optional

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn

from .data.adapters import (
    adapter_contract_from_entry,
    combine_file_manifests,
    materialize_dataset,
)
from .data.manifests import read_file_manifest, validate_file_manifest
from .data.ontology import load_ontology_lock
from .data.source_contract import (
    DatasetSourceContractError,
    archive_parts_from_entry,
    archive_set_sha256,
)
from .provenance import canonical_json_sha256, sha256_file
from .workflow_common import (
    WorkflowExecutionError,
    load_json,
    write_json_artifact,
    write_parquet_artifact,
)


REMOTE_HOME = Path("/data/home/xinxi-zhyh/xinxi-zhyh")
SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")
NEAR_PHASH_THRESHOLD = 4
NEAR_ASPECT_RATIO_DELTA_MAX = 0.02
NEAR_THUMBNAIL_STD_MIN = 5.0
NEAR_INTENSITY_CORRELATION_MIN = 0.995
NEAR_GRADIENT_CORRELATION_MIN = 0.990
NEAR_AFFINE_RMSE_MAX = 0.030
NEAR_CONFIRMATION_METHOD = "phash64_hamming4_thumbnail16_ncc_gradient_affine_v1"


def _role_entries(roles: Mapping[str, Any]) -> list[Dict[str, Any]]:
    entries = []
    for role, value in sorted(roles.items()):
        values = value if isinstance(value, list) else [value]
        for item in values:
            if not isinstance(item, Mapping):
                raise WorkflowExecutionError(f"Dataset role {role} has a non-object entry")
            entry = dict(item)
            entry["planned_role"] = role
            entries.append(entry)
    return entries


def _require_remote_path(value: Any) -> Path:
    path = Path(str(value))
    try:
        path.relative_to(REMOTE_HOME)
    except ValueError as exc:
        raise WorkflowExecutionError(f"Remote data path is outside the account home: {path}") from exc
    if not path.is_file():
        raise WorkflowExecutionError(f"Remote data archive is absent: {path}")
    return path


def _verified_archive_parts(archive: Mapping[str, Any]) -> list[Dict[str, Any]]:
    parts = archive.get("archives")
    if parts is None:
        parts = [
            {
                "archive_part_id": "primary",
                "remote_path": archive.get("remote_path"),
                "sha256": archive.get("sha256"),
                "byte_size": archive.get("byte_size"),
            }
        ]
    if not isinstance(parts, list) or not parts:
        raise WorkflowExecutionError("Verified dataset has no archive parts")
    normalized = []
    seen = set()
    for part in parts:
        if not isinstance(part, Mapping):
            raise WorkflowExecutionError("Verified archive part must be an object")
        part_id = str(part.get("archive_part_id", ""))
        if SAFE_ID.fullmatch(part_id) is None or part_id in seen:
            raise WorkflowExecutionError(f"Invalid verified archive_part_id: {part_id}")
        seen.add(part_id)
        normalized.append(
            {
                "archive_part_id": part_id,
                "remote_path": str(part.get("remote_path", "")),
                "sha256": str(part.get("sha256", "")),
                "byte_size": part.get("byte_size"),
            }
        )
    return sorted(normalized, key=lambda value: value["archive_part_id"])


def verify_archives(
    project_root: Path,
    run_root: Path,
    config_path: Optional[Path],
) -> Dict[str, Any]:
    del run_root, config_path
    decision_path = project_root / "evidence/data/dataset_source_decision.json"
    staging_path = project_root / "evidence/data/staging_receipts.json"
    decision = load_json(decision_path)
    staging = load_json(staging_path)
    if decision.get("status") != "pass" or staging.get("status") != "pass":
        raise WorkflowExecutionError("Dataset decision and staging receipt must both pass")
    entries = {
        str(item["dataset_id"]): item for item in _role_entries(decision.get("roles", {}))
    }
    snapshots = {
        str(item["dataset_id"]): item for item in decision.get("license_snapshots", [])
    }
    expected_parts: Dict[tuple[str, str], Dict[str, str]] = {}
    for dataset_id, entry in entries.items():
        try:
            parts = archive_parts_from_entry(entry)
        except DatasetSourceContractError as exc:
            raise WorkflowExecutionError(str(exc)) from exc
        for part in parts:
            expected_parts[(dataset_id, part["archive_part_id"])] = part
    staged = staging.get("archives")
    if not isinstance(staged, list) or len(staged) != len(expected_parts):
        raise WorkflowExecutionError("Staging receipt does not cover every selected archive part")
    staged_by_key: Dict[tuple[str, str], Mapping[str, Any]] = {}
    for receipt in staged:
        if not isinstance(receipt, Mapping):
            raise WorkflowExecutionError("Staging receipt archive entry must be an object")
        key = (
            str(receipt.get("dataset_id", "")),
            str(receipt.get("archive_part_id", "primary")),
        )
        if key in staged_by_key or key not in expected_parts:
            raise WorkflowExecutionError(f"Unexpected or duplicate staged archive part: {key}")
        staged_by_key[key] = receipt
    verified = []
    for dataset_id, entry in sorted(entries.items()):
        if dataset_id not in snapshots:
            raise WorkflowExecutionError(f"Unreviewed staged dataset: {dataset_id}")
        snapshot = snapshots[dataset_id]
        license_path = project_root / str(snapshot["snapshot_path"])
        if not license_path.is_file() or sha256_file(license_path) != snapshot["sha256"]:
            raise WorkflowExecutionError(f"License snapshot hash mismatch: {dataset_id}")
        adapter = adapter_contract_from_entry(entry)
        archive_parts = []
        for key, expected_part in sorted(expected_parts.items()):
            if key[0] != dataset_id:
                continue
            receipt = staged_by_key[key]
            if str(receipt.get("planned_role", "")) != str(entry["planned_role"]):
                raise WorkflowExecutionError(
                    f"Staged archive role drift: {dataset_id}:{key[1]}"
                )
            archive = _require_remote_path(receipt.get("remote_path"))
            observed = sha256_file(archive)
            expected = expected_part["archive_sha256"]
            if observed != expected or observed != str(receipt.get("sha256", "")):
                raise WorkflowExecutionError(
                    f"Remote archive hash mismatch: {dataset_id}:{key[1]}"
                )
            archive_parts.append(
                {
                    "archive_part_id": key[1],
                    "remote_path": archive.as_posix(),
                    "byte_size": archive.stat().st_size,
                    "sha256": observed,
                }
            )
        try:
            archive_set_hash = archive_set_sha256(archive_parts)
        except DatasetSourceContractError as exc:
            raise WorkflowExecutionError(str(exc)) from exc
        archive_id = f"{dataset_id}-{archive_set_hash[:16]}"
        if SAFE_ID.fullmatch(archive_id) is None:
            raise WorkflowExecutionError(f"Dataset cannot form a safe archive ID: {dataset_id}")
        verified.append(
            {
                "dataset_id": dataset_id,
                "dataset_version": str(entry["dataset_version"]),
                "archive_id": archive_id,
                "archive_set_sha256": archive_set_hash,
                "archives": archive_parts,
                "license_id": str(entry["license_id"]),
                "license_snapshot_path": str(snapshot["snapshot_path"]),
                "license_snapshot_sha256": str(snapshot["sha256"]),
                "adapter_contract_sha256": canonical_json_sha256(adapter),
                "planned_role": str(entry["planned_role"]),
                "verification_status": "pass",
            }
        )
    output = project_root / "evidence/data/remote_archive_verification.json"
    write_json_artifact(
        output,
        {
            "schema_version": 1,
            "step_id": "E024",
            "status": "pass",
            "decision_sha256": sha256_file(decision_path),
            "staging_receipt_sha256": sha256_file(staging_path),
            "archive_count": len(expected_parts),
            "dataset_count": len(verified),
            "archives": verified,
            "licenses_verified": True,
            "checksums_verified": True,
            "training_performed": False,
        },
    )
    return {
        "status": "pass",
        "output_paths": [output.relative_to(project_root).as_posix()],
        "details": {
            "archive_count": len(expected_parts),
            "dataset_count": len(verified),
        },
    }


def _per_dataset_manifest_paths(project_root: Path, dataset_id: str) -> tuple[Path, Path]:
    base = project_root / "data/canonical" / dataset_id
    return base / "adapter_file_manifest.parquet", base / "adapter_receipt.json"


def _materialize_or_validate(
    project_root: Path,
    archive: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    dataset_id = str(archive["dataset_id"])
    manifest_path, receipt_path = _per_dataset_manifest_paths(project_root, dataset_id)
    adapter = adapter_contract_from_entry(entry)
    contract_hash = canonical_json_sha256(adapter)
    archive_parts = _verified_archive_parts(archive)
    try:
        archive_set_hash = archive_set_sha256(archive_parts)
    except DatasetSourceContractError as exc:
        raise WorkflowExecutionError(str(exc)) from exc
    declared_set_hash = str(archive.get("archive_set_sha256", archive_set_hash))
    if declared_set_hash != archive_set_hash:
        raise WorkflowExecutionError(f"Verified archive-set hash drift: {dataset_id}")
    if manifest_path.is_file() and receipt_path.is_file():
        receipt = load_json(receipt_path)
        if (
            receipt.get("status") != "pass"
            or receipt.get("archive_set_sha256") != archive_set_hash
            or receipt.get("adapter_contract_sha256") != contract_hash
            or receipt.get("manifest_sha256") != sha256_file(manifest_path)
        ):
            raise WorkflowExecutionError(f"Canonical dataset receipt drift: {dataset_id}")
        frame = validate_file_manifest(pd.read_parquet(manifest_path))
        if int(receipt.get("record_count", -1)) != len(frame):
            raise WorkflowExecutionError(f"Canonical dataset count drift: {dataset_id}")
        return frame, receipt
    canonical_root = project_root / "data/canonical" / dataset_id
    if canonical_root.exists():
        raise WorkflowExecutionError(
            f"Incomplete canonical dataset requires manual failure review: {dataset_id}"
        )
    frame = materialize_dataset(
        project_root=project_root,
        dataset_id=dataset_id,
        archive_id=str(archive["archive_id"]),
        archive_paths=[Path(part["remote_path"]) for part in archive_parts],
        contract=adapter,
    )
    write_parquet_artifact(manifest_path, frame)
    receipt = {
        "schema_version": 1,
        "step_id": "E026",
        "status": "pass",
        "dataset_id": dataset_id,
        "archive_id": str(archive["archive_id"]),
        "archive_set_sha256": archive_set_hash,
        "archives": [
            {
                "archive_part_id": part["archive_part_id"],
                "sha256": part["sha256"],
            }
            for part in archive_parts
        ],
        "adapter_contract_sha256": contract_hash,
        "manifest_path": manifest_path.relative_to(project_root).as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "record_count": len(frame),
        "raw_group_count": int(frame["raw_group_id"].nunique()),
        "modality_counts": {
            str(key): int(value) for key, value in frame["modality"].value_counts().items()
        },
    }
    write_json_artifact(receipt_path, receipt)
    return frame, receipt


def build_file_manifest(
    project_root: Path,
    run_root: Path,
    config_path: Optional[Path],
) -> Dict[str, Any]:
    del run_root, config_path
    verification_path = project_root / "evidence/data/remote_archive_verification.json"
    decision_path = project_root / "evidence/data/dataset_source_decision.json"
    verification = load_json(verification_path)
    decision = load_json(decision_path)
    if verification.get("status") != "pass" or decision.get("status") != "pass":
        raise WorkflowExecutionError("E026 requires pass archive verification and source decision")
    entries = {
        str(item["dataset_id"]): item for item in _role_entries(decision.get("roles", {}))
    }
    frames = []
    receipts = []
    for archive in verification.get("archives", []):
        dataset_id = str(archive["dataset_id"])
        if dataset_id not in entries:
            raise WorkflowExecutionError(f"Verified archive has no source entry: {dataset_id}")
        frame, receipt = _materialize_or_validate(project_root, archive, entries[dataset_id])
        frames.append(frame)
        receipts.append(receipt)
    combined = combine_file_manifests(frames)
    output = project_root / "data/locked/file_manifest.parquet"
    write_parquet_artifact(output, combined)
    registry = project_root / "data/locked/canonical_dataset_registry.json"
    write_json_artifact(
        registry,
        {
            "schema_version": 1,
            "step_id": "E026",
            "status": "pass",
            "archive_verification_sha256": sha256_file(verification_path),
            "dataset_count": int(combined["dataset_id"].nunique()),
            "record_count": len(combined),
            "raw_group_count": int(
                combined[["dataset_id", "raw_group_id"]].drop_duplicates().shape[0]
            ),
            "file_manifest_path": output.relative_to(project_root).as_posix(),
            "file_manifest_sha256": sha256_file(output),
            "datasets": receipts,
            "training_performed": False,
        },
    )
    return {
        "status": "pass",
        "output_paths": [
            output.relative_to(project_root).as_posix(),
            registry.relative_to(project_root).as_posix(),
        ],
        "details": {
            "dataset_count": int(combined["dataset_id"].nunique()),
            "record_count": len(combined),
        },
    }


class _ImageFingerprint(NamedTuple):
    phash: int
    width: int
    height: int
    thumbnail: bytes


def _image_fingerprint(path: Path) -> _ImageFingerprint:
    with Image.open(path) as image:
        width, height = image.size
        grayscale = image.convert("L")
        values = np.asarray(
            grayscale.resize((32, 32), Image.Resampling.LANCZOS),
            dtype=np.float64,
        )
        thumbnail = np.asarray(
            grayscale.resize((16, 16), Image.Resampling.LANCZOS),
            dtype=np.uint8,
        ).tobytes()
    low_frequency = dctn(values, type=2, norm="ortho")[:8, :8]
    median = float(np.median(low_frequency.ravel()[1:]))
    bits = low_frequency > median
    result = 0
    for bit in bits.ravel():
        result = (result << 1) | int(bit)
    return _ImageFingerprint(result, int(width), int(height), thumbnail)


def _image_phash(path: Path) -> int:
    return _image_fingerprint(path).phash


def _vector_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - float(left.mean())
    right_centered = right - float(right.mean())
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator <= 1.0e-12:
        return -1.0
    return float(np.dot(left_centered, right_centered) / denominator)


def _affine_rmse(source: np.ndarray, target: np.ndarray) -> float:
    centered = source - float(source.mean())
    denominator = float(np.dot(centered, centered))
    if denominator <= 1.0e-12:
        return float("inf")
    slope = float(np.dot(centered, target - float(target.mean())) / denominator)
    intercept = float(target.mean()) - slope * float(source.mean())
    residual = target - (slope * source + intercept)
    return float(np.sqrt(np.mean(np.square(residual))) / 255.0)


def _near_duplicate_metrics(
    left: _ImageFingerprint, right: _ImageFingerprint
) -> Dict[str, Any]:
    left_values = np.frombuffer(left.thumbnail, dtype=np.uint8).astype(np.float64)
    right_values = np.frombuffer(right.thumbnail, dtype=np.uint8).astype(np.float64)
    left_grid = left_values.reshape(16, 16)
    right_grid = right_values.reshape(16, 16)
    left_gradient = np.concatenate(
        (np.diff(left_grid, axis=0).ravel(), np.diff(left_grid, axis=1).ravel())
    )
    right_gradient = np.concatenate(
        (np.diff(right_grid, axis=0).ravel(), np.diff(right_grid, axis=1).ravel())
    )
    left_ratio = float(left.width) / float(left.height)
    right_ratio = float(right.width) / float(right.height)
    aspect_delta = abs(left_ratio - right_ratio) / max(left_ratio, right_ratio)
    min_std = min(float(left_values.std()), float(right_values.std()))
    intensity_correlation = _vector_correlation(left_values, right_values)
    gradient_correlation = _vector_correlation(left_gradient, right_gradient)
    affine_rmse = max(
        _affine_rmse(left_values, right_values),
        _affine_rmse(right_values, left_values),
    )
    confirmed = (
        aspect_delta <= NEAR_ASPECT_RATIO_DELTA_MAX
        and min_std >= NEAR_THUMBNAIL_STD_MIN
        and intensity_correlation >= NEAR_INTENSITY_CORRELATION_MIN
        and gradient_correlation >= NEAR_GRADIENT_CORRELATION_MIN
        and affine_rmse <= NEAR_AFFINE_RMSE_MAX
    )
    return {
        "confirmed": bool(confirmed),
        "thumbnail_min_std": min_std,
        "thumbnail_intensity_correlation": intensity_correlation,
        "thumbnail_gradient_correlation": gradient_correlation,
        "thumbnail_affine_rmse": affine_rmse,
        "aspect_ratio_relative_delta": aspect_delta,
    }


_POPCOUNT_16 = tuple(bin(value).count("1") for value in range(1 << 16))


def _hamming64(left: int, right: int) -> int:
    value = int(left) ^ int(right)
    return sum(_POPCOUNT_16[(value >> shift) & 0xFFFF] for shift in (0, 16, 32, 48))


class _HammingBKTree:
    def __init__(self) -> None:
        self._root: Optional[Dict[str, Any]] = None

    def add(self, value: int) -> None:
        candidate = int(value)
        if self._root is None:
            self._root = {"value": candidate, "children": {}}
            return
        node = self._root
        while True:
            distance = _hamming64(candidate, int(node["value"]))
            if distance == 0:
                return
            children = node["children"]
            if distance not in children:
                children[distance] = {"value": candidate, "children": {}}
                return
            node = children[distance]

    def query(self, value: int, threshold: int) -> list[tuple[int, int]]:
        if self._root is None:
            return []
        matches = []
        stack = [self._root]
        while stack:
            node = stack.pop()
            distance = _hamming64(int(value), int(node["value"]))
            if distance <= threshold:
                matches.append((int(node["value"]), distance))
            lower = max(0, distance - threshold)
            upper = distance + threshold
            for edge, child in node["children"].items():
                if lower <= int(edge) <= upper:
                    stack.append(child)
        return sorted(matches, key=lambda item: (item[1], item[0]))


def _candidate_near_pairs(
    hashes: Mapping[int, int], threshold: int = NEAR_PHASH_THRESHOLD
) -> Iterable[tuple[int, int, int]]:
    tree = _HammingBKTree()
    indices_by_value: Dict[int, list[int]] = {}
    for index in sorted(hashes):
        value = int(hashes[index])
        for other_value, distance in tree.query(value, threshold):
            for other_index in indices_by_value[other_value]:
                yield other_index, index, distance
        if value not in indices_by_value:
            tree.add(value)
            indices_by_value[value] = []
        indices_by_value[value].append(index)


def _parallel_image_fingerprints(
    project_root: Path, manifest: pd.DataFrame
) -> tuple[Dict[int, _ImageFingerprint], int]:
    items = []
    for index, row in manifest.iterrows():
        path = project_root / str(row["relative_path"])
        if path.suffix.lower() in {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}:
            items.append((int(index), path))
    if not items:
        return {}, 0
    requested = os.environ.get("MINING1_DEDUP_WORKERS") or os.environ.get(
        "SLURM_CPUS_PER_TASK", "1"
    )
    try:
        worker_count = int(requested)
    except (TypeError, ValueError):
        worker_count = 1
    worker_count = max(1, min(worker_count, 8, len(items)))

    def fingerprint_item(item: tuple[int, Path]) -> tuple[int, _ImageFingerprint]:
        index, path = item
        return index, _image_fingerprint(path)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = executor.map(fingerprint_item, items)
        return dict(results), worker_count


def _parallel_image_phashes(
    project_root: Path, manifest: pd.DataFrame
) -> tuple[Dict[int, int], int]:
    fingerprints, worker_count = _parallel_image_fingerprints(project_root, manifest)
    return {index: item.phash for index, item in fingerprints.items()}, worker_count


def _label_signature(value: Any) -> str:
    payload = json.loads(str(value))
    if not isinstance(payload, dict):
        raise WorkflowExecutionError("Label summary must encode an object")
    for field in ("annotation_path", "canonical_yolo_label_path"):
        payload.pop(field, None)
    return canonical_json_sha256(payload)


def _compatible_ontology_mapping(ontology: Mapping[str, Any]) -> Dict[tuple[str, str], str]:
    mapping: Dict[tuple[str, str], str] = {}
    for entry in ontology["entries"]:
        if entry["mapping_status"] != "compatible":
            continue
        key = (str(entry["dataset_id"]), str(entry["source_label"]))
        mapping[key] = str(entry["canonical_concept_id"])
    return mapping


def _semantic_label_signature(
    dataset_id: str,
    value: Any,
    ontology_mapping: Mapping[tuple[str, str], str],
) -> str:
    payload = json.loads(str(value))
    if not isinstance(payload, dict):
        raise WorkflowExecutionError("Label summary must encode an object")
    if "boxes" not in payload and "class_ids" not in payload:
        return _label_signature(value)
    boxes = payload.get("boxes") or []
    source_labels = {str(item) for item in payload.get("class_ids") or []}
    for box in boxes:
        if isinstance(box, Mapping) and box.get("source_label") is not None:
            source_labels.add(str(box["source_label"]))
    canonical_concepts = []
    for source_label in sorted(source_labels):
        key = (str(dataset_id), source_label)
        if key not in ontology_mapping:
            raise WorkflowExecutionError(
                f"No compatible ontology mapping for {dataset_id}/{source_label}"
            )
        canonical_concepts.append(ontology_mapping[key])
    semantic = {
        "canonical_concepts": sorted(set(canonical_concepts)),
        "observable_presence": bool(boxes),
        "negative_annotation_verified": payload.get("negative_annotation_verified") is True,
    }
    if semantic["observable_presence"] and not semantic["canonical_concepts"]:
        raise WorkflowExecutionError(
            f"Positive annotation has no compatible ontology concept: {dataset_id}"
        )
    return canonical_json_sha256(semantic)


def audit_groups_dedup(
    project_root: Path,
    run_root: Path,
    config_path: Optional[Path],
) -> Dict[str, Any]:
    del run_root, config_path
    manifest_path = project_root / "data/locked/file_manifest.parquet"
    ontology_path = project_root / "data/locked/ontology_lock.yaml"
    manifest = read_file_manifest(manifest_path).reset_index(drop=True)
    ontology = load_ontology_lock(ontology_path)
    ontology_mapping = _compatible_ontology_mapping(ontology)
    group_keys = sorted(
        {
            (str(item.dataset_id), str(item.raw_group_id))
            for item in manifest[["dataset_id", "raw_group_id"]]
            .drop_duplicates()
            .itertuples(index=False)
        }
    )
    parent = {key: key for key in group_keys}

    def find(key: tuple[str, str]) -> tuple[str, str]:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(left: tuple[str, str], right: tuple[str, str]) -> bool:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return False
        first, second = sorted((left_root, right_root))
        parent[second] = first
        return True

    def row_group(index: int) -> tuple[str, str]:
        row = manifest.loc[index]
        return str(row["dataset_id"]), str(row["raw_group_id"])

    rows = []

    def append_resolution_edge(
        left_index: int,
        right_index: int,
        duplicate_kind: str,
        distance: int,
        confirmation: Mapping[str, Any],
    ) -> None:
        left = manifest.loc[left_index]
        right = manifest.loc[right_index]
        left_key = (
            str(left["dataset_id"]),
            str(left["raw_group_id"]),
            str(left["record_id"]),
        )
        right_key = (
            str(right["dataset_id"]),
            str(right["raw_group_id"]),
            str(right["record_id"]),
        )
        if right_key < left_key:
            left, right = right, left
        rows.append(
            {
                "left_dataset_id": str(left["dataset_id"]),
                "left_record_id": str(left["record_id"]),
                "left_raw_group_id": str(left["raw_group_id"]),
                "right_dataset_id": str(right["dataset_id"]),
                "right_record_id": str(right["record_id"]),
                "right_raw_group_id": str(right["raw_group_id"]),
                "duplicate_kind": duplicate_kind,
                "perceptual_hamming_distance": int(distance),
                "near_confirmation_method": str(confirmation["method"]),
                "thumbnail_min_std": float(confirmation["thumbnail_min_std"]),
                "thumbnail_intensity_correlation": float(
                    confirmation["thumbnail_intensity_correlation"]
                ),
                "thumbnail_gradient_correlation": float(
                    confirmation["thumbnail_gradient_correlation"]
                ),
                "thumbnail_affine_rmse": float(
                    confirmation["thumbnail_affine_rmse"]
                ),
                "aspect_ratio_relative_delta": float(
                    confirmation["aspect_ratio_relative_delta"]
                ),
                "label_summary_conflict": _label_signature(left["label_summary_json"])
                != _label_signature(right["label_summary_json"]),
                "semantic_label_conflict": _semantic_label_signature(
                    str(left["dataset_id"]),
                    left["label_summary_json"],
                    ontology_mapping,
                )
                != _semantic_label_signature(
                    str(right["dataset_id"]),
                    right["label_summary_json"],
                    ontology_mapping,
                ),
                "resolution": "merge_connected_raw_groups_before_split",
            }
        )

    exact_sha_group_count = 0
    exact_record_pair_count = 0
    exact_label_summary_conflict_group_count = 0
    exact_semantic_conflict_group_count = 0
    for _, group in manifest.groupby("sha256", sort=True):
        indices = sorted(int(index) for index in group.index)
        if len(indices) < 2:
            continue
        exact_sha_group_count += 1
        exact_record_pair_count += len(indices) * (len(indices) - 1) // 2
        signatures = {
            _label_signature(manifest.loc[index, "label_summary_json"])
            for index in indices
        }
        if len(signatures) != 1:
            exact_label_summary_conflict_group_count += 1
        semantic_signatures = {
            _semantic_label_signature(
                str(manifest.loc[index, "dataset_id"]),
                manifest.loc[index, "label_summary_json"],
                ontology_mapping,
            )
            for index in indices
        }
        if len(semantic_signatures) != 1:
            exact_semantic_conflict_group_count += 1
            first, second = indices[:2]
            raise WorkflowExecutionError(
                "Exact duplicate has conflicting ontology-level semantics: "
                f"{manifest.loc[first, 'dataset_id']}/{manifest.loc[first, 'record_id']} and "
                f"{manifest.loc[second, 'dataset_id']}/{manifest.loc[second, 'record_id']}"
            )
        representatives: Dict[tuple[str, str], int] = {}
        for index in indices:
            representatives.setdefault(row_group(index), index)
        ordered = sorted(representatives)
        for other_group in ordered[1:]:
            if union(ordered[0], other_group):
                append_resolution_edge(
                    representatives[ordered[0]],
                    representatives[other_group],
                    "exact_sha256",
                    0,
                    {
                        "method": "exact_sha256",
                        "thumbnail_min_std": 0.0,
                        "thumbnail_intensity_correlation": 1.0,
                        "thumbnail_gradient_correlation": 1.0,
                        "thumbnail_affine_rmse": 0.0,
                        "aspect_ratio_relative_delta": 0.0,
                    },
                )

    fingerprints, worker_count = _parallel_image_fingerprints(project_root, manifest)
    image_hashes = {index: item.phash for index, item in fingerprints.items()}
    groups_by_phash: Dict[int, Dict[tuple[str, str], int]] = {}
    for index in sorted(image_hashes):
        groups_by_phash.setdefault(int(image_hashes[index]), {}).setdefault(
            row_group(index), index
        )
    same_phash_group_pair_candidate_count = 0
    near_unique_hash_pair_count = 0
    near_group_pair_comparison_count = 0
    near_structurally_confirmed_pair_count = 0
    near_rejected_modality_count = 0
    near_rejected_structure_count = 0

    def consider_near_pair(left_index: int, right_index: int, distance: int) -> None:
        nonlocal near_group_pair_comparison_count
        nonlocal near_structurally_confirmed_pair_count
        nonlocal near_rejected_modality_count
        nonlocal near_rejected_structure_count
        left_group = row_group(left_index)
        right_group = row_group(right_index)
        if left_group == right_group or find(left_group) == find(right_group):
            return
        near_group_pair_comparison_count += 1
        left_modality = str(manifest.loc[left_index, "modality"]).strip().lower()
        right_modality = str(manifest.loc[right_index, "modality"]).strip().lower()
        if left_modality != right_modality:
            near_rejected_modality_count += 1
            return
        metrics = _near_duplicate_metrics(
            fingerprints[left_index], fingerprints[right_index]
        )
        if not metrics["confirmed"]:
            near_rejected_structure_count += 1
            return
        left_semantic = _semantic_label_signature(
            str(manifest.loc[left_index, "dataset_id"]),
            manifest.loc[left_index, "label_summary_json"],
            ontology_mapping,
        )
        right_semantic = _semantic_label_signature(
            str(manifest.loc[right_index, "dataset_id"]),
            manifest.loc[right_index, "label_summary_json"],
            ontology_mapping,
        )
        if left_semantic != right_semantic:
            raise WorkflowExecutionError(
                "Structurally confirmed near duplicate has conflicting ontology-level semantics: "
                f"{manifest.loc[left_index, 'dataset_id']}/{manifest.loc[left_index, 'record_id']} and "
                f"{manifest.loc[right_index, 'dataset_id']}/{manifest.loc[right_index, 'record_id']}"
            )
        near_structurally_confirmed_pair_count += 1
        if union(left_group, right_group):
            append_resolution_edge(
                left_index,
                right_index,
                "near_phash64",
                distance,
                {"method": NEAR_CONFIRMATION_METHOD, **metrics},
            )

    for value in sorted(groups_by_phash):
        representatives = groups_by_phash[value]
        ordered = [(group, representatives[group]) for group in sorted(representatives)]
        same_phash_group_pair_candidate_count += len(ordered) * (len(ordered) - 1) // 2
        for (_, left_index), (_, right_index) in combinations(ordered, 2):
            consider_near_pair(left_index, right_index, 0)

    unique_values = sorted(groups_by_phash)
    unique_hashes = {index: value for index, value in enumerate(unique_values)}
    for left_hash_index, right_hash_index, distance in _candidate_near_pairs(
        unique_hashes, threshold=NEAR_PHASH_THRESHOLD
    ):
        if distance == 0:
            continue
        near_unique_hash_pair_count += 1
        left_representatives = groups_by_phash[unique_values[left_hash_index]]
        right_representatives = groups_by_phash[unique_values[right_hash_index]]
        for left_group in sorted(left_representatives):
            for right_group in sorted(right_representatives):
                consider_near_pair(
                    left_representatives[left_group],
                    right_representatives[right_group],
                    distance,
                )
    columns = [
        "left_dataset_id",
        "left_record_id",
        "left_raw_group_id",
        "right_dataset_id",
        "right_record_id",
        "right_raw_group_id",
        "duplicate_kind",
        "perceptual_hamming_distance",
        "near_confirmation_method",
        "thumbnail_min_std",
        "thumbnail_intensity_correlation",
        "thumbnail_gradient_correlation",
        "thumbnail_affine_rmse",
        "aspect_ratio_relative_delta",
        "label_summary_conflict",
        "semantic_label_conflict",
        "resolution",
    ]
    report = pd.DataFrame.from_records(rows, columns=columns)
    if not report.empty:
        report = report.sort_values(
            [
                "duplicate_kind",
                "left_dataset_id",
                "left_raw_group_id",
                "left_record_id",
                "right_dataset_id",
                "right_raw_group_id",
                "right_record_id",
            ],
            kind="mergesort",
        ).reset_index(drop=True)
    output = project_root / "data/locked/dedup_report.parquet"
    write_parquet_artifact(output, report)
    component_sizes: Dict[tuple[str, str], int] = {}
    for key in group_keys:
        component_sizes[find(key)] = component_sizes.get(find(key), 0) + 1
    summary = project_root / "evidence/data/dedup_summary.json"
    write_json_artifact(
        summary,
        {
            "schema_version": 1,
            "step_id": "E034",
            "status": "pass",
            "file_manifest_sha256": sha256_file(manifest_path),
            "ontology_lock_sha256": sha256_file(ontology_path),
            "pair_count": len(report),
            "exact_pair_count": int((report["duplicate_kind"] == "exact_sha256").sum()),
            "near_pair_count": int((report["duplicate_kind"] == "near_phash64").sum()),
            "report_semantics": "minimal_deterministic_spanning_edges_per_duplicate_component",
            "raw_group_count": len(group_keys),
            "dedup_component_count": len(component_sizes),
            "affected_raw_group_count": sum(
                size for size in component_sizes.values() if size > 1
            ),
            "exact_sha_group_count": exact_sha_group_count,
            "exact_record_pair_count": exact_record_pair_count,
            "exact_label_summary_conflict_group_count": exact_label_summary_conflict_group_count,
            "exact_semantic_conflict_group_count": exact_semantic_conflict_group_count,
            "label_summary_conflict_edge_count": int(report["label_summary_conflict"].sum()),
            "semantic_label_conflict_edge_count": int(report["semantic_label_conflict"].sum()),
            "semantic_conflict_policy": "retain_coordinate_detail_variation_only_when_ontology_semantics_match; fail_exact_or_confirmed_near_semantic_conflict",
            "phash_record_count": len(image_hashes),
            "unique_phash_count": len(unique_values),
            "same_phash_group_pair_candidate_count": same_phash_group_pair_candidate_count,
            "near_unique_hash_pair_count": near_unique_hash_pair_count,
            "near_group_pair_comparison_count": near_group_pair_comparison_count,
            "near_structurally_confirmed_pair_count": near_structurally_confirmed_pair_count,
            "near_rejected_modality_count": near_rejected_modality_count,
            "near_rejected_structure_count": near_rejected_structure_count,
            "near_duplicate_method": NEAR_CONFIRMATION_METHOD,
            "near_confirmation": {
                "same_modality_required": True,
                "aspect_ratio_relative_delta_max": NEAR_ASPECT_RATIO_DELTA_MAX,
                "thumbnail_min_std": NEAR_THUMBNAIL_STD_MIN,
                "thumbnail_intensity_correlation_min": NEAR_INTENSITY_CORRELATION_MIN,
                "thumbnail_gradient_correlation_min": NEAR_GRADIENT_CORRELATION_MIN,
                "thumbnail_affine_rmse_max": NEAR_AFFINE_RMSE_MAX,
            },
            "candidate_index": "bktree_hamming64",
            "perceptual_hamming_threshold": NEAR_PHASH_THRESHOLD,
            "phash_worker_count": worker_count,
            "resolution": "merge_connected_raw_groups_before_split",
            "model_outcomes_used": False,
        },
    )
    return {
        "status": "pass",
        "output_paths": [
            output.relative_to(project_root).as_posix(),
            summary.relative_to(project_root).as_posix(),
        ],
        "details": {"duplicate_pair_count": len(report)},
    }
