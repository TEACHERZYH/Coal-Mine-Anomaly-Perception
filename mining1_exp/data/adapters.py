from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import tempfile
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence
import zipfile
from xml.etree import ElementTree as ET

import pandas as pd
from PIL import Image

from .manifests import validate_file_manifest


SUPPORTED_ADAPTER_KINDS = {
    "yolo_detection",
    "coco_detection",
    "voc_detection",
    "methane_csv",
}
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")


class AdapterContractError(ValueError):
    """Raised when a reviewed dataset adapter cannot be executed deterministically."""


def _safe_relative(value: Any, field: str, *, allow_dot: bool = False) -> str:
    if value is None:
        raise AdapterContractError(f"{field} must be a safe relative path")
    text = str(value).replace("\\", "/").strip()
    path = PurePosixPath(text)
    if (not text or path.is_absolute() or ".." in path.parts) and not (
        allow_dot and text == "."
    ):
        raise AdapterContractError(f"{field} must be a safe relative path")
    return text


def _require_string(value: Any, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise AdapterContractError(f"{field} must be a non-empty string")
    return text


def _filename_range_key(value: Any, field: str) -> tuple[int, int, int]:
    text = _require_string(value, field)
    if re.fullmatch(r"[0-9]+", text):
        return (0, int(text), 0)
    match = re.fullmatch(r"[pP]([0-9]+)-([0-9]+)", text)
    if match is None:
        raise AdapterContractError(
            f"{field} must be digits or a p<major>-<minor> filename stem"
        )
    return (1, int(match.group(1)), int(match.group(2)))


def _validate_filename_range_lookup(
    value: Any,
    field: str,
    *,
    allowed_sources: Sequence[str],
) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AdapterContractError(f"{field}.filename_range_lookup must be an object")
    source = str(value.get("source", "relative_path"))
    if source not in allowed_sources:
        raise AdapterContractError(
            f"{field}.filename_range_lookup has an unsupported source"
        )
    pattern = _require_string(
        value.get("pattern"), f"{field}.filename_range_lookup.pattern"
    )
    record_group = _require_string(
        value.get("record_group", "record"),
        f"{field}.filename_range_lookup.record_group",
    )
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        raise AdapterContractError(
            f"{field}.filename_range_lookup.pattern is invalid"
        ) from exc
    if record_group not in compiled.groupindex:
        raise AdapterContractError(
            f"{field}.filename_range_lookup.record_group is not a named regex group"
        )
    ranges = value.get("ranges")
    if not isinstance(ranges, list) or not ranges:
        raise AdapterContractError(
            f"{field}.filename_range_lookup.ranges must be a non-empty list"
        )
    normalized_ranges = []
    sortable_ranges: list[tuple[tuple[int, int, int], tuple[int, int, int], str]] = []
    for index, item in enumerate(ranges):
        item_field = f"{field}.filename_range_lookup.ranges[{index}]"
        if not isinstance(item, Mapping):
            raise AdapterContractError(f"{item_field} must be an object")
        group_id = _require_string(item.get("value"), f"{item_field}.value")
        if SAFE_ID.fullmatch(group_id) is None:
            raise AdapterContractError(f"{item_field}.value must be a safe identifier")
        start = _require_string(item.get("start"), f"{item_field}.start")
        end = _require_string(item.get("end"), f"{item_field}.end")
        start_key = _filename_range_key(start, f"{item_field}.start")
        end_key = _filename_range_key(end, f"{item_field}.end")
        if start_key[0] != end_key[0] or start_key > end_key:
            raise AdapterContractError(
                f"{item_field} has incompatible or descending endpoints"
            )
        normalized_ranges.append({"value": group_id, "start": start, "end": end})
        sortable_ranges.append((start_key, end_key, group_id))
    for namespace in (0, 1):
        previous_end: Optional[tuple[int, int, int]] = None
        for start_key, end_key, _ in sorted(
            (item for item in sortable_ranges if item[0][0] == namespace),
            key=lambda item: item[0],
        ):
            if previous_end is not None and start_key <= previous_end:
                raise AdapterContractError(
                    f"{field}.filename_range_lookup contains overlapping ranges"
                )
            previous_end = end_key
    return {
        "source": source,
        "pattern": pattern,
        "record_group": record_group,
        "ranges": normalized_ranges,
    }


def _validate_rule(
    value: Any,
    field: str,
    *,
    nullable: bool,
    allowed_sources: Sequence[str] = ("relative_path",),
) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AdapterContractError(f"{field} must be an object")
    rule = dict(value)
    modes = [
        name
        for name in ("constant", "regex", "filename_range_lookup", "null")
        if name in rule
    ]
    if len(modes) != 1:
        raise AdapterContractError(
            f"{field} must define exactly one of constant, regex, "
            "filename_range_lookup, or null"
        )
    mode = modes[0]
    if mode == "null":
        if not nullable or rule["null"] is not True:
            raise AdapterContractError(f"{field} cannot use a null rule")
        return {"null": True}
    if mode == "constant":
        return {"constant": _require_string(rule["constant"], f"{field}.constant")}
    if mode == "filename_range_lookup":
        return {
            "filename_range_lookup": _validate_filename_range_lookup(
                rule["filename_range_lookup"],
                field,
                allowed_sources=allowed_sources,
            )
        }
    regex = rule["regex"]
    if not isinstance(regex, Mapping):
        raise AdapterContractError(f"{field}.regex must be an object")
    source = str(regex.get("source", "relative_path"))
    if source not in allowed_sources:
        raise AdapterContractError(f"{field}.regex has an unsupported source")
    pattern = _require_string(regex.get("pattern"), f"{field}.regex.pattern")
    template = _require_string(regex.get("template"), f"{field}.regex.template")
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        raise AdapterContractError(f"{field}.regex.pattern is invalid") from exc
    groups = set(compiled.groupindex)
    referenced = set(re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", template))
    if not referenced or not referenced.issubset(groups):
        raise AdapterContractError(
            f"{field}.regex.template must reference named regex groups only"
        )
    return {
        "regex": {
            "source": source,
            "pattern": pattern,
            "template": template,
        }
    }


def _validate_class_names(value: Any, field: str) -> Dict[int, str]:
    if isinstance(value, list):
        result = {index: _require_string(name, f"{field}[{index}]") for index, name in enumerate(value)}
    elif isinstance(value, Mapping):
        result = {}
        for key, name in value.items():
            try:
                class_id = int(key)
            except (TypeError, ValueError) as exc:
                raise AdapterContractError(f"{field} keys must be integer class IDs") from exc
            if class_id < 0 or class_id in result:
                raise AdapterContractError(f"{field} contains an invalid class ID")
            result[class_id] = _require_string(name, f"{field}.{class_id}")
    else:
        raise AdapterContractError(f"{field} must be a list or object")
    if not result:
        raise AdapterContractError(f"{field} cannot be empty")
    if len(result.values()) != len(set(result.values())):
        raise AdapterContractError(f"{field} contains duplicate class names")
    return dict(sorted(result.items()))


def _validate_selected_class_names(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise AdapterContractError(f"{field} must be a non-empty list")
    names = [_require_string(item, f"{field}[]") for item in value]
    if len(names) != len(set(names)):
        raise AdapterContractError(f"{field} contains duplicate class names")
    return names


def validate_adapter_contract(payload: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise AdapterContractError("adapter_contract must be an object")
    contract = dict(payload)
    if contract.get("schema_version") != 1:
        raise AdapterContractError("adapter_contract.schema_version must be 1")
    kind = str(contract.get("kind", ""))
    if kind not in SUPPORTED_ADAPTER_KINDS:
        raise AdapterContractError(f"Unsupported adapter kind: {kind}")
    normalized: Dict[str, Any] = {
        "schema_version": 1,
        "kind": kind,
        "archive_subdir": _safe_relative(
            contract.get("archive_subdir", "."),
            "adapter_contract.archive_subdir",
            allow_dot=True,
        ),
    }
    if kind in {"yolo_detection", "coco_detection", "voc_detection"}:
        normalized["record_id_rule"] = _validate_rule(
            contract.get("record_id_rule"), "adapter_contract.record_id_rule", nullable=False
        )
        normalized["raw_group_rule"] = _validate_rule(
            contract.get("raw_group_rule"), "adapter_contract.raw_group_rule", nullable=False
        )
        normalized["modality_rule"] = _validate_rule(
            contract.get("modality_rule"), "adapter_contract.modality_rule", nullable=False
        )
        for name in ("pair_id_rule", "sequence_id_rule", "timestamp_rule"):
            normalized[name] = _validate_rule(
                contract.get(name, {"null": True}),
                f"adapter_contract.{name}",
                nullable=True,
            )
    if kind == "yolo_detection":
        section = contract.get("yolo")
        if not isinstance(section, Mapping):
            raise AdapterContractError("adapter_contract.yolo must be an object")
        globs = section.get("image_globs")
        if not isinstance(globs, list) or not globs:
            raise AdapterContractError("adapter_contract.yolo.image_globs cannot be empty")
        excluded_image_stems = section.get("excluded_image_stems", [])
        if not isinstance(excluded_image_stems, list):
            raise AdapterContractError(
                "adapter_contract.yolo.excluded_image_stems must be a list"
            )
        normalized["yolo"] = {
            "image_globs": [
                _safe_relative(item, "adapter_contract.yolo.image_globs[]") for item in globs
            ],
            "image_root": _safe_relative(section.get("image_root"), "adapter_contract.yolo.image_root"),
            "label_root": _safe_relative(section.get("label_root"), "adapter_contract.yolo.label_root"),
            "class_names": _validate_class_names(
                section.get("class_names"), "adapter_contract.yolo.class_names"
            ),
            "missing_label_policy": str(section.get("missing_label_policy", "error")),
            "excluded_image_stems": sorted(
                {
                    _require_string(
                        item, "adapter_contract.yolo.excluded_image_stems[]"
                    ).casefold()
                    for item in excluded_image_stems
                }
            ),
        }
        if any(
            SAFE_ID.fullmatch(item) is None
            for item in normalized["yolo"]["excluded_image_stems"]
        ):
            raise AdapterContractError(
                "adapter_contract.yolo.excluded_image_stems contains an unsafe stem"
            )
        if normalized["yolo"]["missing_label_policy"] not in {"empty", "error"}:
            raise AdapterContractError(
                "adapter_contract.yolo.missing_label_policy must be empty or error"
            )
    elif kind == "coco_detection":
        section = contract.get("coco")
        if not isinstance(section, Mapping):
            raise AdapterContractError("adapter_contract.coco must be an object")
        allowed_license_ids = section.get("allowed_license_ids")
        if allowed_license_ids is not None:
            if not isinstance(allowed_license_ids, list) or not allowed_license_ids:
                raise AdapterContractError(
                    "adapter_contract.coco.allowed_license_ids must be a non-empty list"
                )
            try:
                allowed_license_ids = [int(value) for value in allowed_license_ids]
            except (TypeError, ValueError) as exc:
                raise AdapterContractError(
                    "adapter_contract.coco.allowed_license_ids must contain integers"
                ) from exc
            if len(allowed_license_ids) != len(set(allowed_license_ids)):
                raise AdapterContractError(
                    "adapter_contract.coco.allowed_license_ids contains duplicates"
                )
        normalized["coco"] = {
            "annotation_file": _safe_relative(
                section.get("annotation_file"), "adapter_contract.coco.annotation_file"
            ),
            "image_root": _safe_relative(
                section.get("image_root"), "adapter_contract.coco.image_root"
            ),
            "class_names": _validate_selected_class_names(
                section.get("class_names"), "adapter_contract.coco.class_names"
            ),
            "invalid_bbox_policy": str(section.get("invalid_bbox_policy", "error")),
            "iscrowd_policy": str(section.get("iscrowd_policy", "drop")),
            "image_selection": str(section.get("image_selection", "all")),
            "allowed_license_ids": allowed_license_ids,
        }
        if normalized["coco"]["invalid_bbox_policy"] not in {"drop", "error"}:
            raise AdapterContractError(
                "adapter_contract.coco.invalid_bbox_policy must be drop or error"
            )
        if normalized["coco"]["iscrowd_policy"] not in {"drop", "include"}:
            raise AdapterContractError(
                "adapter_contract.coco.iscrowd_policy must be drop or include"
            )
        if normalized["coco"]["image_selection"] not in {
            "all",
            "selected_category_presence",
        }:
            raise AdapterContractError(
                "adapter_contract.coco.image_selection is unsupported"
            )
    elif kind == "voc_detection":
        section = contract.get("voc")
        if not isinstance(section, Mapping):
            raise AdapterContractError("adapter_contract.voc must be an object")
        globs = section.get("image_globs")
        if not isinstance(globs, list) or not globs:
            raise AdapterContractError("adapter_contract.voc.image_globs cannot be empty")
        normalized["voc"] = {
            "image_globs": [
                _safe_relative(item, "adapter_contract.voc.image_globs[]")
                for item in globs
            ],
            "image_root": _safe_relative(
                section.get("image_root"), "adapter_contract.voc.image_root"
            ),
            "annotation_root": _safe_relative(
                section.get("annotation_root"), "adapter_contract.voc.annotation_root"
            ),
            "annotation_path_rule": str(
                section.get("annotation_path_rule", "image_relative")
            ),
            "class_names": _validate_selected_class_names(
                section.get("class_names"), "adapter_contract.voc.class_names"
            ),
            "missing_annotation_policy": str(
                section.get("missing_annotation_policy", "error")
            ),
            "invalid_bbox_policy": str(section.get("invalid_bbox_policy", "error")),
            "difficult_policy": str(section.get("difficult_policy", "include")),
            "image_selection": str(section.get("image_selection", "all")),
            "coordinate_convention": str(
                section.get("coordinate_convention", "voc_xyxy")
            ),
        }
        if normalized["voc"]["missing_annotation_policy"] not in {"empty", "error"}:
            raise AdapterContractError(
                "adapter_contract.voc.missing_annotation_policy must be empty or error"
            )
        if normalized["voc"]["annotation_path_rule"] not in {
            "image_relative",
            "image_stem",
        }:
            raise AdapterContractError(
                "adapter_contract.voc.annotation_path_rule is unsupported"
            )
        if normalized["voc"]["invalid_bbox_policy"] not in {"drop", "error"}:
            raise AdapterContractError(
                "adapter_contract.voc.invalid_bbox_policy must be drop or error"
            )
        if normalized["voc"]["difficult_policy"] not in {"drop", "error", "include"}:
            raise AdapterContractError(
                "adapter_contract.voc.difficult_policy is unsupported"
            )
        if normalized["voc"]["image_selection"] not in {
            "all",
            "selected_category_presence",
        }:
            raise AdapterContractError(
                "adapter_contract.voc.image_selection is unsupported"
            )
        if normalized["voc"]["coordinate_convention"] != "voc_xyxy":
            raise AdapterContractError(
                "adapter_contract.voc.coordinate_convention must be voc_xyxy"
            )
    else:
        section = contract.get("methane")
        if not isinstance(section, Mapping):
            raise AdapterContractError("adapter_contract.methane must be an object")
        globs = section.get("csv_globs")
        if not isinstance(globs, list) or not globs:
            raise AdapterContractError("adapter_contract.methane.csv_globs cannot be empty")
        feature_columns = section.get("feature_columns", [])
        if not isinstance(feature_columns, list):
            raise AdapterContractError("adapter_contract.methane.feature_columns must be a list")
        normalized_features = [
            _require_string(item, "adapter_contract.methane.feature_columns[]")
            for item in feature_columns
        ]
        if len(normalized_features) != len(set(normalized_features)):
            raise AdapterContractError("adapter_contract.methane.feature_columns contains duplicates")
        duration = int(section.get("group_duration_seconds", 0))
        if duration <= 0:
            raise AdapterContractError(
                "adapter_contract.methane.group_duration_seconds must be positive"
            )
        input_layout = str(section.get("input_layout", "long"))
        if input_layout not in {"long", "wide"}:
            raise AdapterContractError(
                "adapter_contract.methane.input_layout must be long or wide"
            )
        normalized["methane"] = {
            "input_layout": input_layout,
            "csv_globs": [
                _safe_relative(item, "adapter_contract.methane.csv_globs[]") for item in globs
            ],
            "timestamp_column": _require_string(
                section.get("timestamp_column"), "adapter_contract.methane.timestamp_column"
            ),
            "value_column": _require_string(
                section.get("value_column"), "adapter_contract.methane.value_column"
            ),
            "sensor_group_column": _require_string(
                section.get("sensor_group_column"),
                "adapter_contract.methane.sensor_group_column",
            ),
            "feature_columns": normalized_features,
            "group_duration_seconds": duration,
            "timezone": _require_string(
                section.get("timezone", "UTC"), "adapter_contract.methane.timezone"
            ),
        }
        if input_layout == "wide":
            components = section.get("timestamp_components")
            required_components = {"year", "month", "day", "hour", "minute", "second"}
            if not isinstance(components, Mapping) or set(components) != required_components:
                raise AdapterContractError(
                    "adapter_contract.methane.timestamp_components must define "
                    "year, month, day, hour, minute, and second"
                )
            targets = section.get("target_sensor_columns")
            if not isinstance(targets, list) or not targets:
                raise AdapterContractError(
                    "adapter_contract.methane.target_sensor_columns cannot be empty"
                )
            normalized_targets = [
                _require_string(item, "adapter_contract.methane.target_sensor_columns[]")
                for item in targets
            ]
            if len(normalized_targets) != len(set(normalized_targets)):
                raise AdapterContractError(
                    "adapter_contract.methane.target_sensor_columns contains duplicates"
                )
            chunk_rows = int(section.get("chunk_rows", 100_000))
            if chunk_rows <= 0 or chunk_rows > 1_000_000:
                raise AdapterContractError(
                    "adapter_contract.methane.chunk_rows must be in [1, 1000000]"
                )
            missing_policy = str(section.get("target_missing_policy", "drop"))
            if missing_policy not in {"drop", "error"}:
                raise AdapterContractError(
                    "adapter_contract.methane.target_missing_policy must be drop or error"
                )
            normalized["methane"].update(
                {
                    "timestamp_components": {
                        key: _require_string(
                            components[key],
                            f"adapter_contract.methane.timestamp_components.{key}",
                        )
                        for key in sorted(required_components)
                    },
                    "target_sensor_columns": normalized_targets,
                    "chunk_rows": chunk_rows,
                    "target_missing_policy": missing_policy,
                }
            )
            output_names = {
                normalized["methane"]["timestamp_column"],
                normalized["methane"]["value_column"],
                normalized["methane"]["sensor_group_column"],
            }
            if output_names.intersection(normalized_features):
                raise AdapterContractError(
                    "Wide methane output columns cannot overlap feature_columns"
                )
    return normalized


def adapter_contract_from_entry(entry: Mapping[str, Any]) -> Dict[str, Any]:
    if "adapter_contract" not in entry:
        raise AdapterContractError(
            f"Dataset {entry.get('dataset_id')} lacks a reviewed adapter_contract"
        )
    return validate_adapter_contract(entry["adapter_contract"])


def apply_rule(rule: Mapping[str, Any], *, relative_path: str, **values: Any) -> Optional[str]:
    if rule.get("null") is True:
        return None
    if "constant" in rule:
        return str(rule["constant"])
    if "filename_range_lookup" in rule:
        lookup = rule["filename_range_lookup"]
        source = str(lookup["source"])
        source_value = (
            relative_path if source == "relative_path" else str(values.get(source, ""))
        )
        match = re.fullmatch(str(lookup["pattern"]), source_value)
        if match is None:
            raise AdapterContractError(
                f"Adapter filename range regex did not match {source}: {source_value}"
            )
        record = match.group(str(lookup["record_group"]))
        key = _filename_range_key(record, "adapter filename range record")
        matches = []
        for item in lookup["ranges"]:
            start = _filename_range_key(item["start"], "adapter filename range start")
            end = _filename_range_key(item["end"], "adapter filename range end")
            if start <= key <= end:
                matches.append(str(item["value"]))
        if len(matches) != 1:
            raise AdapterContractError(
                f"Adapter filename range record matched {len(matches)} groups: {record}"
            )
        return matches[0]
    regex = rule["regex"]
    source = str(regex["source"])
    source_value = relative_path if source == "relative_path" else str(values.get(source, ""))
    match = re.fullmatch(str(regex["pattern"]), source_value)
    if match is None:
        raise AdapterContractError(f"Adapter regex did not match {source}: {source_value}")
    rendered = str(regex["template"]).format(**match.groupdict()).strip()
    if not rendered:
        raise AdapterContractError("Adapter regex produced an empty identifier")
    return rendered


def _destination_for_member(root: Path, name: str) -> Path:
    relative = _safe_relative(name, "archive member")
    destination = (root / PurePosixPath(relative)).resolve()
    resolved_root = root.resolve()
    if resolved_root not in destination.parents and destination != resolved_root:
        raise AdapterContractError(f"Archive member escapes extraction root: {name}")
    return destination


def _copy_stream(source: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)


def _seven_zip_executable() -> str:
    configured = os.environ.get("MINING1_7Z_EXECUTABLE", "").strip()
    candidates = [configured, shutil.which("7zz"), shutil.which("7z")]
    if os.name == "nt":
        candidates.extend(
            [
                r"C:\Program Files\7-Zip\7z.exe",
                r"C:\Program Files (x86)\7-Zip\7z.exe",
            ]
        )
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate))
    raise AdapterContractError(
        "7z archive support requires 7zz/7z or MINING1_7Z_EXECUTABLE"
    )


def _seven_zip_entries(archive: Path) -> list[Dict[str, Any]]:
    result = subprocess.run(
        [_seven_zip_executable(), "l", "-slt", "-ba", "-sccUTF-8", str(archive)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip()[-500:]
        raise AdapterContractError(f"Cannot list 7z archive: {archive}: {detail}")
    entries = []
    current: Dict[str, str] = {}
    for line in [*result.stdout.splitlines(), ""]:
        if not line.strip():
            if current:
                path = _safe_relative(current.get("Path"), "7z archive member")
                attributes = current.get("Attributes", "")
                if (
                    "Symbolic Link" in current
                    or "Hard Link" in current
                    or attributes.lower().startswith("l")
                ):
                    raise AdapterContractError(f"Link 7z member is forbidden: {path}")
                if current.get("Encrypted", "-") == "+":
                    raise AdapterContractError(f"Encrypted 7z member is forbidden: {path}")
                size = int(current.get("Size", "0"))
                if size < 0:
                    raise AdapterContractError(f"7z member has an invalid size: {path}")
                entries.append(
                    {
                        "path": path,
                        "size": size,
                        "is_dir": attributes.upper().startswith("D"),
                    }
                )
                current = {}
            continue
        if " = " in line:
            key, value = line.split(" = ", 1)
            current[key] = value
    if not entries:
        raise AdapterContractError(f"7z archive contains no members: {archive}")
    paths = [entry["path"].casefold() for entry in entries]
    if len(paths) != len(set(paths)):
        raise AdapterContractError(f"7z archive has duplicate member paths: {archive}")
    return entries


def read_7z_member_bounded(archive: Path, member: str, *, max_bytes: int) -> bytes:
    normalized = _safe_relative(member, "7z archive member")
    entries = {entry["path"]: entry for entry in _seven_zip_entries(archive)}
    if normalized not in entries or entries[normalized]["is_dir"]:
        raise AdapterContractError(f"7z member is absent or a directory: {member}")
    declared_size = int(entries[normalized]["size"])
    if declared_size > max_bytes:
        raise AdapterContractError(
            f"7z member exceeds {max_bytes} bytes: {normalized}"
        )
    result = subprocess.run(
        [
            _seven_zip_executable(),
            "x",
            "-so",
            "-bd",
            "-bb0",
            "-sccUTF-8",
            str(archive),
            normalized,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()[-500:]
        raise AdapterContractError(f"Cannot read 7z member {normalized}: {detail}")
    if len(result.stdout) != declared_size:
        raise AdapterContractError(
            f"7z member size drift for {normalized}: {len(result.stdout)} != {declared_size}"
        )
    return result.stdout


def _extract_7z_into(source: Path, stage: Path) -> None:
    _seven_zip_entries(source)
    with tempfile.TemporaryDirectory(prefix=".mining1-7z-", dir=stage.parent) as temporary:
        isolated = Path(temporary)
        result = subprocess.run(
            [
                _seven_zip_executable(),
                "x",
                "-y",
                "-bd",
                "-bb0",
                "-sccUTF-8",
                f"-o{isolated}",
                str(source),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.strip()[-500:]
            raise AdapterContractError(f"Cannot extract 7z archive: {source}: {detail}")
        for item in sorted(isolated.rglob("*")):
            if item.is_symlink():
                raise AdapterContractError(f"Extracted 7z link is forbidden: {item}")
            relative = item.relative_to(isolated).as_posix()
            destination = _destination_for_member(stage, relative)
            if item.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            elif item.is_file():
                with item.open("rb") as source_stream:
                    _copy_stream(source_stream, destination)


def _extract_archive_into(source: Path, stage: Path) -> None:
    lowered = source.name.lower()
    if lowered.endswith(".zip"):
        with zipfile.ZipFile(source) as handle:
            for info in handle.infolist():
                mode = (info.external_attr >> 16) & 0o170000
                if mode == 0o120000:
                    raise AdapterContractError(
                        f"Symbolic-link ZIP member is forbidden: {info.filename}"
                    )
                member_target = _destination_for_member(stage, info.filename)
                if info.is_dir():
                    member_target.mkdir(parents=True, exist_ok=True)
                    continue
                with handle.open(info, "r") as source_stream:
                    _copy_stream(source_stream, member_target)
    elif lowered.endswith(".7z"):
        _extract_7z_into(source, stage)
    elif any(
        lowered.endswith(suffix)
        for suffix in (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")
    ):
        with tarfile.open(source, mode="r:*") as handle:
            for info in handle:
                if info.issym() or info.islnk() or info.isdev():
                    raise AdapterContractError(
                        f"Link or device TAR member is forbidden: {info.name}"
                    )
                member_target = _destination_for_member(stage, info.name)
                if info.isdir():
                    member_target.mkdir(parents=True, exist_ok=True)
                    continue
                if not info.isfile():
                    continue
                source_stream = handle.extractfile(info)
                if source_stream is None:
                    raise AdapterContractError(f"Cannot read TAR member: {info.name}")
                with source_stream:
                    _copy_stream(source_stream, member_target)
    else:
        with source.open("rb") as source_stream:
            _copy_stream(source_stream, stage / source.name)


def extract_archives_immutable(archives: Sequence[Path], destination: Path) -> Path:
    sources = [archive.resolve() for archive in archives]
    if not sources:
        raise AdapterContractError("Dataset archive set cannot be empty")
    if len(sources) != len(set(sources)):
        raise AdapterContractError("Dataset archive set contains duplicate paths")
    for source in sources:
        if not source.is_file():
            raise AdapterContractError(f"Dataset archive is not a file: {source}")
    target = destination.resolve()
    if target.exists():
        raise AdapterContractError(f"Canonical dataset root already exists: {target}")
    stage = target.with_name(f".{target.name}.{os.getpid()}.stage")
    if stage.exists():
        raise AdapterContractError(f"Dataset extraction stage already exists: {stage}")
    stage.mkdir(parents=True)
    try:
        for source in sources:
            _extract_archive_into(source, stage)
        if not any(item.is_file() for item in stage.rglob("*")):
            raise AdapterContractError("Dataset archive set extracted no files")
        target.parent.mkdir(parents=True, exist_ok=True)
        stage.replace(target)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return target


def extract_archive_immutable(archive: Path, destination: Path) -> Path:
    return extract_archives_immutable([archive], destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_relative(project_root: Path, path: Path) -> str:
    resolved_root = project_root.resolve()
    resolved = path.resolve()
    if resolved_root not in resolved.parents:
        raise AdapterContractError(f"Canonical artifact escapes project root: {path}")
    return resolved.relative_to(resolved_root).as_posix()


def _common_record_fields(
    *,
    project_root: Path,
    dataset_id: str,
    archive_id: str,
    path: Path,
    adapter_relative_path: str,
    contract: Mapping[str, Any],
    label_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    values = {
        "record_id": apply_rule(
            contract["record_id_rule"], relative_path=adapter_relative_path
        ),
        "raw_group_id": apply_rule(
            contract["raw_group_rule"], relative_path=adapter_relative_path
        ),
        "modality": apply_rule(
            contract["modality_rule"], relative_path=adapter_relative_path
        ),
        "pair_id": apply_rule(
            contract["pair_id_rule"], relative_path=adapter_relative_path
        ),
        "sequence_id": apply_rule(
            contract["sequence_id_rule"], relative_path=adapter_relative_path
        ),
        "timestamp_or_order": apply_rule(
            contract["timestamp_rule"], relative_path=adapter_relative_path
        ),
    }
    if any(not values[name] for name in ("record_id", "raw_group_id", "modality")):
        raise AdapterContractError("Required adapter identifiers cannot be empty")
    return {
        "dataset_id": dataset_id,
        "record_id": values["record_id"],
        "archive_id": archive_id,
        "relative_path": _project_relative(project_root, path),
        "modality": values["modality"],
        "raw_group_id": values["raw_group_id"],
        "pair_id": values["pair_id"],
        "sequence_id": values["sequence_id"],
        "timestamp_or_order": values["timestamp_or_order"],
        "label_summary_json": json.dumps(label_summary, sort_keys=True),
        "byte_size": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _parse_yolo_label(
    label_path: Path,
    class_names: Mapping[int, str],
    *,
    missing_label_policy: str,
) -> list[Dict[str, Any]]:
    if not label_path.is_file():
        if missing_label_policy == "empty":
            return []
        raise AdapterContractError(f"YOLO label is missing: {label_path}")
    boxes = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise AdapterContractError(
                f"YOLO label must have five fields: {label_path}:{line_number}"
            )
        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = (float(value) for value in parts[1:])
        except ValueError as exc:
            raise AdapterContractError(
                f"YOLO label contains a non-numeric field: {label_path}:{line_number}"
            ) from exc
        if class_id not in class_names:
            raise AdapterContractError(f"YOLO class ID is not reviewed: {class_id}")
        if not all(0.0 <= value <= 1.0 for value in (x_center, y_center, width, height)):
            raise AdapterContractError(f"YOLO coordinates are outside [0,1]: {label_path}")
        boxes.append(
            {
                "source_class_id": class_id,
                "source_label": class_names[class_id],
                "x_center_normalized": x_center,
                "y_center_normalized": y_center,
                "width_normalized": width,
                "height_normalized": height,
            }
        )
    return boxes


def _build_yolo_manifest(
    *,
    project_root: Path,
    dataset_id: str,
    archive_id: str,
    extracted_root: Path,
    contract: Mapping[str, Any],
) -> pd.DataFrame:
    adapter_root = (extracted_root / str(contract["archive_subdir"])).resolve()
    if not adapter_root.is_dir() or extracted_root.resolve() not in {
        adapter_root,
        *adapter_root.parents,
    }:
        raise AdapterContractError("YOLO archive_subdir is absent or unsafe")
    section = contract["yolo"]
    image_root = (adapter_root / str(section["image_root"])).resolve()
    label_root = (adapter_root / str(section["label_root"])).resolve()
    images: set[Path] = set()
    for pattern in section["image_globs"]:
        images.update(
            item.resolve()
            for item in adapter_root.glob(str(pattern))
            if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
        )
    if not images:
        raise AdapterContractError(f"YOLO adapter found no images for {dataset_id}")
    rows = []
    excluded_image_stems = set(section.get("excluded_image_stems", []))
    observed_excluded_stems: set[str] = set()
    for image_path in sorted(images):
        if image_root not in image_path.parents:
            raise AdapterContractError(f"YOLO image is outside image_root: {image_path}")
        image_relative = image_path.relative_to(image_root)
        image_stem = image_relative.stem.casefold()
        if image_stem in excluded_image_stems:
            observed_excluded_stems.add(image_stem)
            continue
        label_path = label_root / image_relative.with_suffix(".txt")
        boxes = _parse_yolo_label(
            label_path,
            section["class_names"],
            missing_label_policy=str(section["missing_label_policy"]),
        )
        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except Exception as exc:
            raise AdapterContractError(f"Cannot decode image: {image_path}") from exc
        if width <= 0 or height <= 0:
            raise AdapterContractError(f"Image dimensions are invalid: {image_path}")
        relative = image_path.relative_to(adapter_root).as_posix()
        rows.append(
            _common_record_fields(
                project_root=project_root,
                dataset_id=dataset_id,
                archive_id=archive_id,
                path=image_path,
                adapter_relative_path=relative,
                contract=contract,
                label_summary={
                    "annotation_format": "yolo_normalized_xywh",
                    "annotation_path": _project_relative(project_root, label_path)
                    if label_path.is_file()
                    else None,
                    "image_width": int(width),
                    "image_height": int(height),
                    "class_ids": sorted({box["source_label"] for box in boxes}),
                    "boxes": boxes,
                    "negative_annotation_verified": bool(label_path.is_file()),
                },
            )
        )
    missing_exclusions = excluded_image_stems - observed_excluded_stems
    if missing_exclusions:
        raise AdapterContractError(
            "YOLO adapter did not observe reviewed excluded image stems: "
            f"{sorted(missing_exclusions)}"
        )
    return pd.DataFrame.from_records(rows)


def _load_coco(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdapterContractError(f"COCO annotation is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise AdapterContractError("COCO annotation root must be an object")
    for field in ("images", "annotations", "categories"):
        if not isinstance(payload.get(field), list):
            raise AdapterContractError(f"COCO annotation lacks list field: {field}")
    return payload


def _write_absolute_yolo_label(
    path: Path,
    boxes: Iterable[Mapping[str, Any]],
    class_index: Mapping[str, int],
    *,
    width: int,
    height: int,
) -> None:
    lines = []
    for box in boxes:
        source_label = str(box["source_label"])
        if source_label not in class_index:
            raise AdapterContractError(f"Detection box uses an unreviewed class: {source_label}")
        x1, y1, x2, y2 = (float(box[name]) for name in ("x1", "y1", "x2", "y2"))
        x_center = ((x1 + x2) / 2.0) / width
        y_center = ((y1 + y2) / 2.0) / height
        normalized_width = (x2 - x1) / width
        normalized_height = (y2 - y1) / height
        if not all(
            0.0 <= value <= 1.0
            for value in (x_center, y_center, normalized_width, normalized_height)
        ):
            raise AdapterContractError("Detection bbox exceeds reviewed image bounds")
        lines.append(
            f"{class_index[source_label]} {x_center:.10f} {y_center:.10f} "
            f"{normalized_width:.10f} {normalized_height:.10f}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    data = ("\n".join(lines) + ("\n" if lines else "")).encode("ascii")
    with path.open("xb") as handle:
        handle.write(data)


def _build_coco_manifest(
    *,
    project_root: Path,
    dataset_id: str,
    archive_id: str,
    extracted_root: Path,
    contract: Mapping[str, Any],
) -> pd.DataFrame:
    adapter_root = (extracted_root / str(contract["archive_subdir"])).resolve()
    section = contract["coco"]
    annotation_path = adapter_root / str(section["annotation_file"])
    image_root = (adapter_root / str(section["image_root"])).resolve()
    payload = _load_coco(annotation_path)
    categories = {}
    for category in payload["categories"]:
        category_id = int(category["id"])
        if category_id in categories:
            raise AdapterContractError(f"Duplicate COCO category ID: {category_id}")
        categories[category_id] = _require_string(
            category.get("name"), f"COCO category {category_id} name"
        )
    if len(categories.values()) != len(set(categories.values())):
        raise AdapterContractError("COCO category names are not unique")
    selected_names = list(section["class_names"])
    name_to_category = {name: category_id for category_id, name in categories.items()}
    missing_names = sorted(set(selected_names) - set(name_to_category))
    if missing_names:
        raise AdapterContractError(f"COCO selected categories are absent: {missing_names}")
    selected_category_ids = {name_to_category[name] for name in selected_names}
    class_index = {name: index for index, name in enumerate(selected_names)}
    image_ids = [int(image["id"]) for image in payload["images"]]
    if len(image_ids) != len(set(image_ids)):
        raise AdapterContractError("Duplicate COCO image ID")
    known_image_ids = set(image_ids)
    annotations: Dict[int, list[Mapping[str, Any]]] = {}
    for annotation in payload["annotations"]:
        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])
        if image_id not in known_image_ids:
            raise AdapterContractError("COCO annotations reference unknown images")
        if category_id not in categories:
            raise AdapterContractError(f"COCO annotation uses unknown category: {category_id}")
        annotations.setdefault(image_id, []).append(annotation)
    label_root = extracted_root / "canonical_yolo_labels"
    rows = []
    for image in sorted(payload["images"], key=lambda item: int(item["id"])):
        image_id = int(image["id"])
        image_license_id = int(image.get("license", -1))
        if (
            section["allowed_license_ids"] is not None
            and image_license_id not in section["allowed_license_ids"]
        ):
            continue
        selected_annotations = [
            annotation
            for annotation in annotations.get(image_id, [])
            if int(annotation["category_id"]) in selected_category_ids
        ]
        if (
            section["image_selection"] == "selected_category_presence"
            and not selected_annotations
        ):
            continue
        file_name = _safe_relative(image.get("file_name"), "COCO image.file_name")
        image_path = (image_root / PurePosixPath(file_name)).resolve()
        if image_root not in image_path.parents or not image_path.is_file():
            raise AdapterContractError(f"COCO image is absent or unsafe: {file_name}")
        try:
            with Image.open(image_path) as decoded:
                decoded_width, decoded_height = decoded.size
        except Exception as exc:
            raise AdapterContractError(f"Cannot decode COCO image: {file_name}") from exc
        width = int(image.get("width", decoded_width))
        height = int(image.get("height", decoded_height))
        if (
            width <= 0
            or height <= 0
            or (width, height) != (decoded_width, decoded_height)
        ):
            raise AdapterContractError(f"COCO image dimensions drift: {file_name}")
        boxes = []
        dropped_invalid = 0
        dropped_crowd = 0
        for annotation in selected_annotations:
            if int(annotation.get("iscrowd", 0)) != 0 and section["iscrowd_policy"] == "drop":
                dropped_crowd += 1
                continue
            values = annotation.get("bbox")
            valid = isinstance(values, list) and len(values) == 4
            if valid:
                try:
                    x, y, box_width, box_height = (float(value) for value in values)
                except (TypeError, ValueError):
                    valid = False
            if valid:
                valid = (
                    all(math.isfinite(value) for value in (x, y, box_width, box_height))
                    and 0.0 <= x < x + box_width <= width
                    and 0.0 <= y < y + box_height <= height
                )
            if not valid:
                if section["invalid_bbox_policy"] == "drop":
                    dropped_invalid += 1
                    continue
                raise AdapterContractError(
                    f"COCO bbox is invalid: annotation {annotation.get('id')}"
                )
            category_id = int(annotation["category_id"])
            boxes.append(
                {
                    "source_class_id": category_id,
                    "source_label": categories[category_id],
                    "x1": x,
                    "y1": y,
                    "x2": x + box_width,
                    "y2": y + box_height,
                }
            )
        if section["image_selection"] == "selected_category_presence" and not boxes:
            continue
        label_path = label_root / PurePosixPath(file_name).with_suffix(".txt")
        _write_absolute_yolo_label(
            label_path,
            boxes,
            class_index,
            width=width,
            height=height,
        )
        relative = image_path.relative_to(adapter_root).as_posix()
        rows.append(
            _common_record_fields(
                project_root=project_root,
                dataset_id=dataset_id,
                archive_id=archive_id,
                path=image_path,
                adapter_relative_path=relative,
                contract=contract,
                label_summary={
                    "annotation_format": "coco_xywh_absolute",
                    "annotation_path": _project_relative(project_root, annotation_path),
                    "canonical_yolo_label_path": _project_relative(
                        project_root, label_path
                    ),
                    "image_width": width,
                    "image_height": height,
                    "image_license_id": image_license_id,
                    "class_ids": sorted({box["source_label"] for box in boxes}),
                    "boxes": boxes,
                    "negative_annotation_verified": True,
                    "dropped_invalid_bbox_count": dropped_invalid,
                    "dropped_iscrowd_count": dropped_crowd,
                },
            )
        )
    return pd.DataFrame.from_records(rows)


def _build_voc_manifest(
    *,
    project_root: Path,
    dataset_id: str,
    archive_id: str,
    extracted_root: Path,
    contract: Mapping[str, Any],
) -> pd.DataFrame:
    adapter_root = (extracted_root / str(contract["archive_subdir"])).resolve()
    if not adapter_root.is_dir() or extracted_root.resolve() not in {
        adapter_root,
        *adapter_root.parents,
    }:
        raise AdapterContractError("VOC archive_subdir is absent or unsafe")
    section = contract["voc"]
    image_root = (adapter_root / str(section["image_root"])).resolve()
    annotation_root = (adapter_root / str(section["annotation_root"])).resolve()
    selected_names = list(section["class_names"])
    selected_set = set(selected_names)
    class_index = {name: index for index, name in enumerate(selected_names)}
    images: set[Path] = set()
    for pattern in section["image_globs"]:
        images.update(
            item.resolve()
            for item in adapter_root.glob(str(pattern))
            if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
        )
    if not images:
        raise AdapterContractError(f"VOC adapter found no images for {dataset_id}")
    label_root = extracted_root / "canonical_yolo_labels"
    rows = []
    for image_path in sorted(images):
        if image_root not in image_path.parents:
            raise AdapterContractError(f"VOC image is outside image_root: {image_path}")
        image_relative = image_path.relative_to(image_root)
        if section["annotation_path_rule"] == "image_stem":
            annotation_path = annotation_root / f"{image_path.stem}.xml"
        else:
            annotation_path = annotation_root / image_relative.with_suffix(".xml")
        try:
            with Image.open(image_path) as decoded:
                width, height = decoded.size
        except Exception as exc:
            raise AdapterContractError(f"Cannot decode VOC image: {image_path}") from exc
        if width <= 0 or height <= 0:
            raise AdapterContractError(f"VOC image dimensions are invalid: {image_path}")
        boxes = []
        dropped_invalid = 0
        dropped_difficult = 0
        ignored_class = 0
        annotation_verified = annotation_path.is_file()
        if not annotation_verified:
            if section["missing_annotation_policy"] == "error":
                raise AdapterContractError(f"VOC annotation is missing: {annotation_path}")
        else:
            try:
                xml_root = ET.parse(annotation_path).getroot()
            except (OSError, ET.ParseError) as exc:
                raise AdapterContractError(
                    f"VOC annotation is unreadable: {annotation_path}"
                ) from exc
            declared_name = (xml_root.findtext("filename") or "").strip()
            declared_basename = PurePosixPath(declared_name.replace("\\", "/")).name
            if declared_name and declared_basename.casefold() != image_path.name.casefold():
                raise AdapterContractError(
                    f"VOC filename does not match image: {annotation_path}"
                )
            size_node = xml_root.find("size")
            if size_node is not None:
                try:
                    declared_width = int(size_node.findtext("width", "0"))
                    declared_height = int(size_node.findtext("height", "0"))
                except ValueError as exc:
                    raise AdapterContractError(
                        f"VOC size is non-numeric: {annotation_path}"
                    ) from exc
                if (declared_width, declared_height) != (width, height):
                    raise AdapterContractError(
                        f"VOC image dimensions drift: {annotation_path}"
                    )
            for object_node in xml_root.findall("object"):
                class_name = _require_string(
                    object_node.findtext("name"), f"VOC object name in {annotation_path}"
                )
                if class_name not in selected_set:
                    ignored_class += 1
                    continue
                try:
                    difficult = int((object_node.findtext("difficult") or "0").strip())
                except ValueError as exc:
                    raise AdapterContractError(
                        f"VOC difficult flag is non-numeric: {annotation_path}"
                    ) from exc
                if difficult != 0:
                    if section["difficult_policy"] == "drop":
                        dropped_difficult += 1
                        continue
                    if section["difficult_policy"] == "error":
                        raise AdapterContractError(
                            f"VOC difficult object is forbidden: {annotation_path}"
                        )
                bbox = object_node.find("bndbox")
                valid = bbox is not None
                if valid:
                    try:
                        x1, y1, x2, y2 = (
                            float(bbox.findtext(name, "nan"))
                            for name in ("xmin", "ymin", "xmax", "ymax")
                        )
                    except ValueError:
                        valid = False
                if valid:
                    valid = (
                        all(math.isfinite(value) for value in (x1, y1, x2, y2))
                        and 0.0 <= x1 < x2 <= width
                        and 0.0 <= y1 < y2 <= height
                    )
                if not valid:
                    if section["invalid_bbox_policy"] == "drop":
                        dropped_invalid += 1
                        continue
                    raise AdapterContractError(f"VOC bbox is invalid: {annotation_path}")
                boxes.append(
                    {
                        "source_class_id": class_index[class_name],
                        "source_label": class_name,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    }
                )
        if section["image_selection"] == "selected_category_presence" and not boxes:
            continue
        label_path = label_root / image_relative.with_suffix(".txt")
        _write_absolute_yolo_label(
            label_path, boxes, class_index, width=width, height=height
        )
        relative = image_path.relative_to(adapter_root).as_posix()
        rows.append(
            _common_record_fields(
                project_root=project_root,
                dataset_id=dataset_id,
                archive_id=archive_id,
                path=image_path,
                adapter_relative_path=relative,
                contract=contract,
                label_summary={
                    "annotation_format": "voc_xyxy_absolute",
                    "annotation_path": _project_relative(project_root, annotation_path)
                    if annotation_verified
                    else None,
                    "canonical_yolo_label_path": _project_relative(
                        project_root, label_path
                    ),
                    "image_width": width,
                    "image_height": height,
                    "class_ids": sorted({box["source_label"] for box in boxes}),
                    "boxes": boxes,
                    "negative_annotation_verified": annotation_verified,
                    "dropped_invalid_bbox_count": dropped_invalid,
                    "dropped_difficult_count": dropped_difficult,
                    "ignored_unselected_class_count": ignored_class,
                },
            )
        )
    return pd.DataFrame.from_records(rows)


def _parse_timestamps(values: pd.Series, timezone_name: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="raise")
    if getattr(parsed.dt, "tz", None) is None:
        parsed = parsed.dt.tz_localize(
            timezone_name, ambiguous="raise", nonexistent="raise"
        )
    return parsed.dt.tz_convert("UTC")


def _safe_component(value: Any, field: str) -> str:
    text = str(value).strip()
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", text)
    if not text or not safe or not SAFE_ID.fullmatch(safe):
        raise AdapterContractError(f"{field} cannot form a safe path component")
    return safe


def _methane_manifest_row(
    *,
    project_root: Path,
    dataset_id: str,
    archive_id: str,
    extracted_root: Path,
    section: Mapping[str, Any],
    sensor: Any,
    block_epoch: int,
    block: pd.DataFrame,
) -> Dict[str, Any]:
    sensor_column = str(section["sensor_group_column"])
    safe_sensor = _safe_component(sensor, "methane sensor group")
    block = block.sort_values("__timestamp_utc").copy()
    if block["__timestamp_utc"].duplicated().any():
        raise AdapterContractError(
            f"Methane sensor block has duplicate timestamps: {sensor}/{block_epoch}"
        )
    block_start = pd.Timestamp(int(block_epoch), unit="s", tz="UTC")
    output = extracted_root / "normalized_methane" / safe_sensor / f"{int(block_epoch)}.parquet"
    if output.exists():
        raise AdapterContractError(f"Methane block would overwrite an existing record: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    export_columns = [
        "__timestamp_utc",
        sensor_column,
        section["value_column"],
        *section["feature_columns"],
        "__source_csv",
    ]
    block[export_columns].to_parquet(output, index=False)
    raw_group_id = f"{dataset_id}:{safe_sensor}:{int(block_epoch)}"
    return {
        "dataset_id": dataset_id,
        "record_id": raw_group_id,
        "archive_id": archive_id,
        "relative_path": _project_relative(project_root, output),
        "modality": "methane",
        "raw_group_id": raw_group_id,
        "pair_id": None,
        "sequence_id": f"{dataset_id}:{safe_sensor}",
        "timestamp_or_order": block_start.isoformat(),
        "label_summary_json": json.dumps(
            {
                "annotation_format": "continuous_sensor_series",
                "measurement_count": len(block),
                "sensor_group_column": sensor_column,
                "timestamp_column": "__timestamp_utc",
                "value_column": section["value_column"],
                "future_values_in_features": False,
            },
            sort_keys=True,
        ),
        "byte_size": output.stat().st_size,
        "sha256": _sha256(output),
    }


def _wide_methane_timestamps(frame: pd.DataFrame, section: Mapping[str, Any]) -> pd.Series:
    components = {
        key: pd.to_numeric(frame[column], errors="raise")
        for key, column in section["timestamp_components"].items()
    }
    parsed = pd.to_datetime(pd.DataFrame(components), errors="raise")
    return _parse_timestamps(parsed, str(section["timezone"]))


def _build_wide_methane_manifest(
    *,
    project_root: Path,
    dataset_id: str,
    archive_id: str,
    extracted_root: Path,
    adapter_root: Path,
    csv_paths: Sequence[Path],
    section: Mapping[str, Any],
) -> pd.DataFrame:
    targets = list(section["target_sensor_columns"])
    features = list(section["feature_columns"])
    input_columns = {
        *section["timestamp_components"].values(),
        *targets,
        *features,
    }
    duration = int(section["group_duration_seconds"])
    sensor_column = str(section["sensor_group_column"])
    value_column = str(section["value_column"])
    rows = []
    for csv_path in csv_paths:
        header = pd.read_csv(csv_path, nrows=0)
        missing = sorted(input_columns.difference(header.columns))
        if missing:
            raise AdapterContractError(f"Wide methane CSV lacks columns {missing}: {csv_path}")
        pending: Optional[pd.DataFrame] = None
        last_timestamp: Optional[pd.Timestamp] = None
        reader = pd.read_csv(
            csv_path,
            usecols=sorted(input_columns),
            chunksize=int(section["chunk_rows"]),
        )
        for chunk in reader:
            chunk["__timestamp_utc"] = _wide_methane_timestamps(chunk, section)
            if not chunk["__timestamp_utc"].is_monotonic_increasing:
                raise AdapterContractError(f"Wide methane CSV is not chronological: {csv_path}")
            if last_timestamp is not None and chunk["__timestamp_utc"].iloc[0] <= last_timestamp:
                raise AdapterContractError(
                    f"Wide methane CSV has duplicate or decreasing timestamps: {csv_path}"
                )
            last_timestamp = chunk["__timestamp_utc"].iloc[-1]
            for column in sorted(set(targets).union(features)):
                chunk[column] = pd.to_numeric(chunk[column], errors="raise")
            chunk["__source_csv"] = csv_path.relative_to(adapter_root).as_posix()
            epoch = chunk["__timestamp_utc"].astype("int64") // 1_000_000_000
            chunk["__block_epoch"] = (epoch // duration) * duration
            if pending is not None:
                chunk = pd.concat([pending, chunk], ignore_index=True)
            block_epochs = sorted(int(value) for value in chunk["__block_epoch"].unique())
            for block_epoch in block_epochs[:-1]:
                wide_block = chunk.loc[chunk["__block_epoch"] == block_epoch]
                for sensor in targets:
                    target_block = wide_block.loc[wide_block[sensor].notna()].copy()
                    if target_block.empty:
                        continue
                    if section["target_missing_policy"] == "error" and len(target_block) != len(
                        wide_block
                    ):
                        raise AdapterContractError(
                            f"Wide methane target contains missing values: {sensor}/{block_epoch}"
                        )
                    target_block[sensor_column] = sensor
                    target_block[value_column] = target_block[sensor]
                    rows.append(
                        _methane_manifest_row(
                            project_root=project_root,
                            dataset_id=dataset_id,
                            archive_id=archive_id,
                            extracted_root=extracted_root,
                            section=section,
                            sensor=sensor,
                            block_epoch=block_epoch,
                            block=target_block,
                        )
                    )
            pending = chunk.loc[chunk["__block_epoch"] == block_epochs[-1]].copy()
        if pending is None or pending.empty:
            continue
        block_epoch = int(pending["__block_epoch"].iloc[0])
        for sensor in targets:
            target_block = pending.loc[pending[sensor].notna()].copy()
            if target_block.empty:
                continue
            if section["target_missing_policy"] == "error" and len(target_block) != len(pending):
                raise AdapterContractError(
                    f"Wide methane target contains missing values: {sensor}/{block_epoch}"
                )
            target_block[sensor_column] = sensor
            target_block[value_column] = target_block[sensor]
            rows.append(
                _methane_manifest_row(
                    project_root=project_root,
                    dataset_id=dataset_id,
                    archive_id=archive_id,
                    extracted_root=extracted_root,
                    section=section,
                    sensor=sensor,
                    block_epoch=block_epoch,
                    block=target_block,
                )
            )
    if not rows:
        raise AdapterContractError("Wide methane adapter produced no chronological blocks")
    return pd.DataFrame.from_records(rows)


def _build_methane_manifest(
    *,
    project_root: Path,
    dataset_id: str,
    archive_id: str,
    extracted_root: Path,
    contract: Mapping[str, Any],
) -> pd.DataFrame:
    adapter_root = (extracted_root / str(contract["archive_subdir"])).resolve()
    section = contract["methane"]
    csv_paths: set[Path] = set()
    for pattern in section["csv_globs"]:
        csv_paths.update(item.resolve() for item in adapter_root.glob(str(pattern)) if item.is_file())
    if not csv_paths:
        raise AdapterContractError(f"Methane adapter found no CSV files for {dataset_id}")
    if section["input_layout"] == "wide":
        return _build_wide_methane_manifest(
            project_root=project_root,
            dataset_id=dataset_id,
            archive_id=archive_id,
            extracted_root=extracted_root,
            adapter_root=adapter_root,
            csv_paths=sorted(csv_paths),
            section=section,
        )
    required = {
        section["timestamp_column"],
        section["value_column"],
        section["sensor_group_column"],
        *section["feature_columns"],
    }
    frames = []
    for csv_path in sorted(csv_paths):
        frame = pd.read_csv(csv_path)
        missing = sorted(set(required) - set(frame.columns))
        if missing:
            raise AdapterContractError(f"Methane CSV lacks columns {missing}: {csv_path}")
        selected = frame[list(required)].copy()
        selected["__source_csv"] = csv_path.relative_to(adapter_root).as_posix()
        frames.append(selected)
    values = pd.concat(frames, ignore_index=True)
    values["__timestamp_utc"] = _parse_timestamps(
        values[section["timestamp_column"]], str(section["timezone"])
    )
    values[section["value_column"]] = pd.to_numeric(
        values[section["value_column"]], errors="raise"
    )
    for column in section["feature_columns"]:
        values[column] = pd.to_numeric(values[column], errors="raise")
    if values[section["value_column"]].isna().any():
        raise AdapterContractError("Methane value column contains missing values")
    sensor_column = str(section["sensor_group_column"])
    if values[sensor_column].isna().any():
        raise AdapterContractError("Methane sensor group contains missing values")
    duration = int(section["group_duration_seconds"])
    epoch_seconds = values["__timestamp_utc"].astype("int64") // 1_000_000_000
    values["__block_epoch"] = (epoch_seconds // duration) * duration
    rows = []
    for (sensor, block_epoch), block in values.groupby(
        [sensor_column, "__block_epoch"], sort=True
    ):
        rows.append(
            _methane_manifest_row(
                project_root=project_root,
                dataset_id=dataset_id,
                archive_id=archive_id,
                extracted_root=extracted_root,
                section=section,
                sensor=sensor,
                block_epoch=int(block_epoch),
                block=block,
            )
        )
    if not rows:
        raise AdapterContractError("Methane adapter produced no chronological blocks")
    return pd.DataFrame.from_records(rows)


def materialize_dataset(
    *,
    project_root: Path,
    dataset_id: str,
    archive_id: str,
    archive_path: Optional[Path] = None,
    archive_paths: Optional[Sequence[Path]] = None,
    contract: Mapping[str, Any],
) -> pd.DataFrame:
    if not SAFE_ID.fullmatch(dataset_id):
        raise AdapterContractError(f"dataset_id is unsafe: {dataset_id}")
    if not SAFE_ID.fullmatch(archive_id):
        raise AdapterContractError(f"archive_id is unsafe: {archive_id}")
    if archive_paths is not None and archive_path is not None:
        raise AdapterContractError("Provide archive_path or archive_paths, not both")
    sources = list(archive_paths) if archive_paths is not None else [archive_path]
    if not sources or any(source is None for source in sources):
        raise AdapterContractError("Dataset archive set cannot be empty")
    normalized = validate_adapter_contract(contract)
    destination = project_root.resolve() / "data" / "canonical" / dataset_id
    extracted_root = extract_archives_immutable(
        [Path(source) for source in sources if source is not None], destination
    )
    try:
        if normalized["kind"] == "yolo_detection":
            frame = _build_yolo_manifest(
                project_root=project_root,
                dataset_id=dataset_id,
                archive_id=archive_id,
                extracted_root=extracted_root,
                contract=normalized,
            )
        elif normalized["kind"] == "coco_detection":
            frame = _build_coco_manifest(
                project_root=project_root,
                dataset_id=dataset_id,
                archive_id=archive_id,
                extracted_root=extracted_root,
                contract=normalized,
            )
        elif normalized["kind"] == "voc_detection":
            frame = _build_voc_manifest(
                project_root=project_root,
                dataset_id=dataset_id,
                archive_id=archive_id,
                extracted_root=extracted_root,
                contract=normalized,
            )
        else:
            frame = _build_methane_manifest(
                project_root=project_root,
                dataset_id=dataset_id,
                archive_id=archive_id,
                extracted_root=extracted_root,
                contract=normalized,
            )
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    if frame.empty:
        raise AdapterContractError(f"Dataset adapter produced no records: {dataset_id}")
    return validate_file_manifest(frame)


def combine_file_manifests(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    materialized = [frame.copy() for frame in frames]
    if not materialized:
        raise AdapterContractError("No dataset manifests were provided")
    combined = pd.concat(materialized, ignore_index=True)
    return validate_file_manifest(combined)
