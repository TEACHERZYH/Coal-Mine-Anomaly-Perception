from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
from typing import Any, Dict, Iterable, Mapping, Sequence
import zipfile

import pandas as pd
import yaml

from .data.manifests import (
    read_file_manifest,
    read_split_manifest,
    validate_file_manifest,
    validate_split_manifest,
)
from .data.adapters import (
    AdapterContractError,
    adapter_contract_from_entry,
    read_7z_member_bounded,
)
from .data.source_contract import (
    DatasetSourceContractError,
    archive_parts_from_entry,
)
from .data.ontology import validate_ontology_lock
from .data.splits import (
    TEMPORAL_FAMILY_POOL_ORDER,
    TEMPORAL_POOL_ORDER,
    assert_no_duplicate_cross_pool,
    assign_balanced_component_pools,
    assign_temporal_cohort_pools,
    build_fewshot_manifest,
)
from .governance.immutable import write_once_bytes
from .minimal_pipeline import run_minimal_pipeline
from .provenance import canonical_json_sha256, sha256_file
from .remote_transport import CURRENT_REMOTE_HOST, run_sftp_reput, run_ssh_script
from .workflow_common import (
    WorkflowExecutionError,
    load_json,
    write_json_artifact,
    write_parquet_artifact,
)


CANDIDATE_SUFFIXES = {
    ".zip",
    ".tar",
    ".tgz",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
    ".rar",
    ".csv",
    ".json",
    ".parquet",
    ".yaml",
    ".yml",
}
LICENSE_NAMES = {"license", "license.txt", "copying", "readme", "readme.md"}
MAX_MINIMAL_SAMPLE_MEMBER_BYTES = 16 * 1024 * 1024
MAX_MINIMAL_SAMPLE_CSV_ROWS = 10_000
REQUIRED_DATASET_ROLES = {
    "primary_visual_target",
    "primary_visual_sources",
    "primary_rgbt_dataset",
    "methane_dataset",
}
ALLOWED_TRANSFER_ROUTES = {
    "local_existing_upload",
    "remote_existing",
    "domestic_remote_download",
}
REMOTE_DATA_ROOT = "/data/home/xinxi-zhyh/xinxi-zhyh/datasets/mining1"
TRANSFER_CLAIM_ID = "C-TRANSFER"
TRANSFER_SOURCE_FALLBACK_REASON = "fewer_than_two_eligible_non_target_coal_sources"


def _walk_bounded(root: Path, *, max_depth: int, max_files: int) -> list[Path]:
    found: list[Path] = []
    base_depth = len(root.parts)
    for directory, child_directories, files in os.walk(root):
        path = Path(directory)
        depth = len(path.parts) - base_depth
        if depth >= max_depth:
            child_directories[:] = []
        child_directories.sort()
        for name in sorted(files):
            candidate = path / name
            lowered = name.lower()
            if candidate.suffix.lower() in CANDIDATE_SUFFIXES or lowered in LICENSE_NAMES:
                found.append(candidate)
                if len(found) >= max_files:
                    return found
    return found


def _nearby_license(path: Path, root: Path) -> str:
    for parent in (path.parent, path.parent.parent):
        if root not in parent.parents and parent != root:
            continue
        for candidate in sorted(parent.iterdir()) if parent.is_dir() else ():
            if candidate.is_file() and candidate.name.lower() in LICENSE_NAMES:
                return str(candidate.resolve())
    return ""


def inventory_local_datasets(
    root: Path,
    row: Mapping[str, str],
    arguments: Mapping[str, Any],
) -> Dict[str, Any]:
    search_root = Path(str(arguments.get("root", ""))).resolve()
    if not search_root.is_dir():
        raise WorkflowExecutionError(f"Local dataset root is unavailable: {search_root}")
    output = root / "evidence/data/local_dataset_inventory.parquet"
    paths = _walk_bounded(search_root, max_depth=4, max_files=1024)
    records = []
    for path in paths:
        is_license = path.name.lower() in LICENSE_NAMES
        digest = sha256_file(path)
        records.append(
            {
                "candidate_id": hashlib.sha256(
                    str(path.relative_to(search_root)).encode("utf-8")
                ).hexdigest(),
                "location": "local",
                "path": str(path.resolve()),
                "relative_path": path.relative_to(search_root).as_posix(),
                "kind": "license_evidence" if is_license else "candidate_file",
                "suffix": path.suffix.lower(),
                "byte_size": int(path.stat().st_size),
                "sha256": digest,
                "license_evidence_path": str(path.resolve())
                if is_license
                else _nearby_license(path, search_root),
                "fully_extracted_locally": False,
            }
        )
    columns = [
        "candidate_id",
        "location",
        "path",
        "relative_path",
        "kind",
        "suffix",
        "byte_size",
        "sha256",
        "license_evidence_path",
        "fully_extracted_locally",
    ]
    frame = pd.DataFrame.from_records(records, columns=columns)
    if not frame.empty and frame["candidate_id"].duplicated().any():
        raise WorkflowExecutionError("Local inventory candidate IDs are not unique")
    write_parquet_artifact(output, frame)
    return {
        "status": "pass",
        "output_paths": [output.relative_to(root).as_posix()],
        "inputs": [{"path": str(search_root), "kind": "bounded_dataset_root"}],
        "details": {
            "candidate_count": int((frame["kind"] == "candidate_file").sum()),
            "license_evidence_count": int((frame["kind"] == "license_evidence").sum()),
            "max_depth": 4,
            "max_files": 1024,
            "full_extractions": 0,
        },
    }


REMOTE_INVENTORY_SCRIPT = r"""
set -euo pipefail
python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path

roots = [
    Path('/data/home/xinxi-zhyh/xinxi-zhyh/datasets'),
    Path('/data/home/xinxi-zhyh/xinxi-zhyh/cache'),
]
suffixes = {'.zip','.tar','.tgz','.gz','.bz2','.xz','.7z','.rar','.csv','.json','.parquet','.yaml','.yml'}
license_names = {'license','license.txt','copying','readme','readme.md'}
rows = []
for root in roots:
    if not root.is_dir():
        continue
    root_depth = len(root.parts)
    for directory, child_directories, files in os.walk(root):
        path = Path(directory)
        if len(path.parts) - root_depth >= 3:
            child_directories[:] = []
        child_directories.sort()
        for name in sorted(files):
            candidate = path / name
            if candidate.suffix.lower() not in suffixes and name.lower() not in license_names:
                continue
            digest = hashlib.sha256()
            with candidate.open('rb') as handle:
                while True:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
            rows.append({
                'path': str(candidate),
                'byte_size': candidate.stat().st_size,
                'sha256': digest.hexdigest(),
                'kind': 'license_evidence' if name.lower() in license_names else 'candidate_file',
            })
            if len(rows) >= 512:
                break
        if len(rows) >= 512:
            break
    if len(rows) >= 512:
        break
print(json.dumps({'hostname': os.uname().nodename, 'entries': rows}, sort_keys=True))
PY
printf '__SQUEUE__\n'
squeue -h -u "$USER" -o '%i|%P|%j|%T|%M|%D|%R'
"""


def inventory_remote_datasets(
    root: Path,
    row: Mapping[str, str],
    arguments: Mapping[str, Any],
) -> Dict[str, Any]:
    if arguments.get("bounded") is not True:
        raise WorkflowExecutionError("E014 requires --bounded")
    output = root / "evidence/data/remote_dataset_inventory.json"
    result = run_ssh_script(REMOTE_INVENTORY_SCRIPT, timeout_seconds=180)
    marker = "\n__SQUEUE__\n"
    if marker not in result.stdout:
        raise WorkflowExecutionError("Remote inventory lacks the Slurm audit marker")
    inventory_text, queue_text = result.stdout.split(marker, 1)
    inventory = json.loads(inventory_text.strip().splitlines()[-1])
    if inventory.get("hostname") != "mu01" or not isinstance(inventory.get("entries"), list):
        raise WorkflowExecutionError("Remote inventory identity or schema is invalid")
    payload = {
        "schema_version": 1,
        "step_id": row["step_id"],
        "status": "pass",
        "remote_host": CURRENT_REMOTE_HOST,
        "login_node": inventory["hostname"],
        "bounded_roots": [
            "/data/home/xinxi-zhyh/xinxi-zhyh/datasets",
            "/data/home/xinxi-zhyh/xinxi-zhyh/cache",
        ],
        "max_depth": 3,
        "max_files": 512,
        "entries": inventory["entries"],
        "user_slurm_jobs_observed": [
            line for line in queue_text.splitlines() if line.strip()
        ],
        "created_compute_allocation": False,
        "full_extractions": 0,
    }
    write_json_artifact(output, payload)
    return {
        "status": "pass",
        "output_paths": [output.relative_to(root).as_posix()],
        "inputs": [{"remote_host": CURRENT_REMOTE_HOST, "scope": "bounded_inventory"}],
        "details": {
            "candidate_count": sum(
                item.get("kind") == "candidate_file" for item in inventory["entries"]
            ),
            "created_compute_allocation": False,
        },
    }


def _forbidden_decision_keys(value: Any, prefix: str = "") -> list[str]:
    forbidden = []
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if any(token in str(key).lower() for token in ("metric", "prediction", "pilot_performance")):
                forbidden.append(path)
            forbidden.extend(_forbidden_decision_keys(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            forbidden.extend(_forbidden_decision_keys(item, f"{prefix}[{index}]"))
    return forbidden


def _role_entries(roles: Mapping[str, Any]) -> list[Dict[str, Any]]:
    entries: list[Dict[str, Any]] = []
    for role in sorted(REQUIRED_DATASET_ROLES):
        value = roles[role]
        values = value if isinstance(value, list) else [value]
        if role != "primary_visual_sources" and len(values) != 1:
            raise WorkflowExecutionError(f"Role {role} must select exactly one dataset")
        if role == "primary_visual_sources" and len(values) < 2:
            raise WorkflowExecutionError("At least two primary visual sources are required")
        for item in values:
            if not isinstance(item, dict):
                raise WorkflowExecutionError(f"Role {role} contains a non-object selection")
            entry = dict(item)
            entry["planned_role"] = role
            entries.append(entry)
    return entries


def _reviewed_transfer_claim_availability(
    root: Path,
    reviewed: Mapping[str, Any],
    multi_coal_dataset_ids: Sequence[str],
) -> Dict[str, Any]:
    eligible_ids = sorted(str(value) for value in multi_coal_dataset_ids)
    if len(eligible_ids) >= 2:
        if TRANSFER_CLAIM_ID in reviewed.get("claim_fallbacks", {}):
            raise WorkflowExecutionError(
                "C-TRANSFER fallback is forbidden when at least two coal sources are eligible"
            )
        return {
            "status": "eligible",
            "eligible_non_target_coal_dataset_ids": eligible_ids,
        }

    fallbacks = reviewed.get("claim_fallbacks")
    fallback = fallbacks.get(TRANSFER_CLAIM_ID) if isinstance(fallbacks, Mapping) else None
    if not isinstance(fallback, Mapping):
        raise WorkflowExecutionError(
            "Multi-coal pretraining requires at least two sources or a reviewed C-TRANSFER fallback"
        )
    if (
        fallback.get("status") != "not_applicable"
        or fallback.get("reason_code") != TRANSFER_SOURCE_FALLBACK_REASON
    ):
        raise WorkflowExecutionError("C-TRANSFER fallback status or reason is invalid")
    declared_eligible = fallback.get("eligible_non_target_coal_dataset_ids")
    if not isinstance(declared_eligible, list) or sorted(map(str, declared_eligible)) != eligible_ids:
        raise WorkflowExecutionError("C-TRANSFER fallback eligible-source list is inconsistent")

    review_paths = fallback.get("candidate_review_paths")
    if not isinstance(review_paths, list) or len(review_paths) < 2:
        raise WorkflowExecutionError("C-TRANSFER fallback lacks complete candidate reviews")
    reviewed_candidates: Dict[str, str] = {}
    review_hashes = []
    resolved_root = root.resolve()
    for relative in review_paths:
        path = (root / str(relative)).resolve()
        if resolved_root not in path.parents or not path.is_file():
            raise WorkflowExecutionError("C-TRANSFER candidate review path is invalid")
        payload = load_json(path)
        dataset_id = str(payload.get("dataset_entry", {}).get("dataset_id", ""))
        disposition = str(
            payload.get("eligibility_decision", {}).get(
                "confirmatory_multi_coal_source", ""
            )
        )
        if not dataset_id or disposition not in {"eligible", "rejected"}:
            raise WorkflowExecutionError("C-TRANSFER candidate review is incomplete")
        if dataset_id in reviewed_candidates:
            raise WorkflowExecutionError("C-TRANSFER candidate reviews contain duplicate datasets")
        reviewed_candidates[dataset_id] = disposition
        review_hashes.append(
            {"path": path.relative_to(resolved_root).as_posix(), "sha256": sha256_file(path)}
        )
    reviewed_eligible = sorted(
        dataset_id
        for dataset_id, disposition in reviewed_candidates.items()
        if disposition == "eligible"
    )
    if reviewed_eligible != eligible_ids:
        raise WorkflowExecutionError(
            "C-TRANSFER fallback candidate dispositions do not match selected coal sources"
        )

    authority_relative = str(fallback.get("authority_path", ""))
    authority = (root / authority_relative).resolve()
    if resolved_root not in authority.parents or not authority.is_file():
        raise WorkflowExecutionError("C-TRANSFER fallback authority is invalid")
    authority_hash = sha256_file(authority)
    if fallback.get("authority_sha256") != authority_hash:
        raise WorkflowExecutionError("C-TRANSFER fallback authority hash is invalid")
    return {
        "status": "not_applicable",
        "reason_code": TRANSFER_SOURCE_FALLBACK_REASON,
        "eligible_non_target_coal_dataset_ids": eligible_ids,
        "candidate_review_hashes": review_hashes,
        "authority_path": authority.relative_to(resolved_root).as_posix(),
        "authority_sha256": authority_hash,
    }


def decide_dataset_sources(
    root: Path,
    row: Mapping[str, str],
    arguments: Mapping[str, Any],
) -> Dict[str, Any]:
    del arguments
    reviewed_path = root / "evidence/data/dataset_source_candidates.reviewed.json"
    reviewed = load_json(reviewed_path)
    if reviewed.get("status") != "reviewed":
        raise WorkflowExecutionError("Dataset source candidates lack reviewed status")
    roles = reviewed.get("roles")
    if not isinstance(roles, dict) or set(roles) != REQUIRED_DATASET_ROLES:
        raise WorkflowExecutionError("Reviewed dataset roles are incomplete or contain extras")
    forbidden = _forbidden_decision_keys(reviewed)
    if forbidden:
        raise WorkflowExecutionError(f"Dataset decision uses forbidden outcome fields: {forbidden}")
    entries = _role_entries(roles)
    dataset_ids = [str(item.get("dataset_id", "")) for item in entries]
    if len(dataset_ids) != len(set(dataset_ids)):
        raise WorkflowExecutionError("One dataset cannot occupy multiple frozen source entries")
    source_entries = roles["primary_visual_sources"]
    if not isinstance(source_entries, list):
        raise WorkflowExecutionError("Primary visual sources must be a reviewed list")
    memberships: Dict[str, list[str]] = {
        "generic_matched": [],
        "single_coal": [],
        "multi_coal": [],
    }
    for source in source_entries:
        conditions = source.get("pretraining_conditions")
        if not isinstance(conditions, list) or not conditions:
            raise WorkflowExecutionError(
                f"Visual source {source.get('dataset_id')} lacks pretraining_conditions"
            )
        normalized_conditions = [str(value) for value in conditions]
        if len(set(normalized_conditions)) != len(normalized_conditions) or not set(
            normalized_conditions
        ).issubset(memberships):
            raise WorkflowExecutionError("Visual source has invalid pretraining conditions")
        for condition in normalized_conditions:
            memberships[condition].append(str(source.get("dataset_id")))
    if len(memberships["generic_matched"]) != 1:
        raise WorkflowExecutionError("Exactly one generic-matched source must be frozen")
    if len(memberships["single_coal"]) != 1:
        raise WorkflowExecutionError("Exactly one single-coal source must be frozen")
    if memberships["single_coal"][0] not in memberships["multi_coal"]:
        raise WorkflowExecutionError("The single-coal source must also belong to multi-coal")
    transfer_claim = _reviewed_transfer_claim_availability(
        root, reviewed, memberships["multi_coal"]
    )
    direction_id = str(reviewed.get("visual_direction_id", "")).strip()
    if not direction_id or re.fullmatch(r"[A-Za-z0-9._-]+", direction_id) is None:
        raise WorkflowExecutionError("Reviewed dataset sources lack a safe visual_direction_id")
    license_snapshots = []
    snapshot_outputs = []
    for item in entries:
        required = {
            "dataset_id",
            "dataset_version",
            "source_location",
            "source_path_or_url",
            "archive_sha256",
            "license_id",
            "license_evidence_path",
            "transfer_route",
            "selection_evidence",
        }
        missing = sorted(required - set(item))
        if missing:
            raise WorkflowExecutionError(
                f"Dataset source {item.get('dataset_id')} lacks fields: {missing}"
            )
        if re.fullmatch(r"[A-Za-z0-9._-]+", str(item["dataset_id"])) is None:
            raise WorkflowExecutionError("Dataset source has an unsafe dataset_id")
        if not re.fullmatch(r"[0-9a-f]{64}", str(item["archive_sha256"])):
            raise WorkflowExecutionError("Dataset source archive SHA-256 is invalid")
        if item["transfer_route"] not in ALLOWED_TRANSFER_ROUTES:
            raise WorkflowExecutionError("Dataset source transfer route is unsupported")
        try:
            archive_parts = archive_parts_from_entry(item)
        except DatasetSourceContractError as exc:
            raise WorkflowExecutionError(str(exc)) from exc
        if any(
            part["transfer_route"] not in ALLOWED_TRANSFER_ROUTES
            for part in archive_parts
        ):
            raise WorkflowExecutionError("Dataset source transfer route is unsupported")
        if not str(item["license_id"]).strip() or not str(item["license_evidence_path"]).strip():
            raise WorkflowExecutionError("Dataset source lacks explicit license evidence")
        adapter_contract_from_entry(item)
        license_source = Path(str(item["license_evidence_path"]))
        if not license_source.is_absolute():
            license_source = root / license_source
        license_source = license_source.resolve()
        if not license_source.is_file():
            raise WorkflowExecutionError(
                f"Dataset license evidence is not a local snapshot: {item['dataset_id']}"
            )
        license_data = license_source.read_bytes()
        if not license_data:
            raise WorkflowExecutionError(
                f"Dataset license evidence is empty: {item['dataset_id']}"
            )
        license_hash = hashlib.sha256(license_data).hexdigest()
        safe_dataset = re.sub(r"[^A-Za-z0-9._-]", "_", str(item["dataset_id"]))
        snapshot = (
            root
            / "evidence/data/license_snapshots"
            / f"{safe_dataset}.{license_hash[:16]}.license"
        )
        write_once_bytes(snapshot, license_data)
        snapshot_relative = snapshot.relative_to(root).as_posix()
        snapshot_outputs.append(snapshot_relative)
        license_snapshots.append(
            {
                "dataset_id": str(item["dataset_id"]),
                "license_id": str(item["license_id"]),
                "source_reference": str(item["license_evidence_path"]),
                "snapshot_path": snapshot_relative,
                "sha256": license_hash,
                "byte_size": len(license_data),
            }
        )
    output = root / "evidence/data/dataset_source_decision.json"
    payload = {
        "schema_version": 1,
        "step_id": row["step_id"],
        "status": "pass",
        "roles": roles,
        "visual_direction_id": direction_id,
        "selected_entry_count": len(entries),
        "selected_archive_part_count": sum(
            len(archive_parts_from_entry(item)) for item in entries
        ),
        "pretraining_source_memberships": memberships,
        "claim_eligibility": {TRANSFER_CLAIM_ID: transfer_claim},
        "license_snapshots": license_snapshots,
        "reviewed_candidates_sha256": sha256_file(reviewed_path),
        "selection_forbidden_inputs": [
            "model_predictions",
            "validation_metrics",
            "test_metrics",
            "pilot_performance",
        ],
        "performance_claims_authorized": False,
    }
    write_json_artifact(output, payload)
    return {
        "status": "pass",
        "output_paths": [output.relative_to(root).as_posix(), *snapshot_outputs],
        "inputs": [
            {"path": reviewed_path.relative_to(root).as_posix(), "sha256": sha256_file(reviewed_path)}
        ],
        "details": {"selected_entry_count": len(entries)},
    }


def _safe_remote_filename(dataset_id: str, source: str) -> str:
    name = Path(source).name
    if not name or re.fullmatch(r"[A-Za-z0-9._-]+", name) is None:
        raise WorkflowExecutionError(f"Unsafe archive filename for {dataset_id}: {name}")
    safe_dataset = re.sub(r"[^A-Za-z0-9._-]", "_", dataset_id)
    return str(PurePosixPath(REMOTE_DATA_ROOT) / safe_dataset / name)


def _remote_sha256(path: str) -> str:
    if not path.startswith("/data/home/xinxi-zhyh/xinxi-zhyh/"):
        raise WorkflowExecutionError(f"Remote archive path is outside project home: {path}")
    result = run_ssh_script(
        "set -euo pipefail\n"
        f"test -f '{path}'\n"
        "command -v srun >/dev/null\n"
        "srun --quiet --partition=cu --nodes=1 --ntasks=1 --cpus-per-task=1 "
        "--mem=2G --time=00:45:00 --kill-on-bad-exit=1 "
        f"bash -lc \"sha256sum '{path}'\" | awk '{{print $1}}'\n",
        timeout_seconds=3600,
    )
    digest = result.stdout.strip().splitlines()[-1]
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise WorkflowExecutionError(f"Remote SHA-256 output is invalid for {path}")
    return digest


def _remote_file_size(path: str) -> int | None:
    if not path.startswith("/data/home/xinxi-zhyh/xinxi-zhyh/"):
        raise WorkflowExecutionError(f"Remote archive path is outside project home: {path}")
    result = run_ssh_script(
        "set -euo pipefail\n"
        f"if test -f '{path}'; then stat -c '%s' '{path}'; else printf '%s\\n' absent; fi\n"
    )
    value = result.stdout.strip().splitlines()[-1]
    if value == "absent":
        return None
    try:
        size = int(value)
    except ValueError as exc:
        raise WorkflowExecutionError(f"Remote size output is invalid for {path}") from exc
    if size < 0:
        raise WorkflowExecutionError(f"Remote size is negative for {path}")
    return size


def _stage_local_archive(local_path: Path, remote_path: str, expected: str) -> tuple[str, str]:
    local_size = local_path.stat().st_size
    existing_size = _remote_file_size(remote_path)
    if existing_size is not None:
        if existing_size != local_size:
            raise WorkflowExecutionError(
                f"Existing remote archive has an unexpected size: {remote_path}"
            )
        observed = _remote_sha256(remote_path)
        if observed != expected:
            raise WorkflowExecutionError(
                f"Existing remote archive has an unexpected hash: {remote_path}"
            )
        return observed, "reused_verified_remote"

    parent = str(PurePosixPath(remote_path).parent)
    temporary = f"{remote_path}.upload.part"
    run_ssh_script(f"set -euo pipefail\nmkdir -p '{parent}'\n")
    partial_size = _remote_file_size(temporary)
    if partial_size is None:
        run_ssh_script(
            "set -euo pipefail\n"
            "umask 077\n"
            f": > '{temporary}'\n"
        )
        partial_size = 0
    if partial_size is not None and partial_size > local_size:
        raise WorkflowExecutionError(
            f"Resumable remote partial is larger than its local source: {temporary}"
        )
    run_sftp_reput(local_path, temporary)
    uploaded_size = _remote_file_size(temporary)
    if uploaded_size != local_size:
        raise WorkflowExecutionError(
            f"Resumable remote upload is incomplete: {temporary}"
        )
    observed = _remote_sha256(temporary)
    if observed != expected:
        raise WorkflowExecutionError(
            f"Staged archive hash mismatch before promotion: {temporary}"
        )
    run_ssh_script(
        "set -euo pipefail\n"
        f"test ! -e '{remote_path}'\n"
        f"mv -- '{temporary}' '{remote_path}'\n"
        f"test \"$(stat -c '%s' '{remote_path}')\" = '{local_size}'\n"
    )
    return observed, "uploaded_resumable_atomic"


def stage_datasets(
    root: Path,
    row: Mapping[str, str],
    arguments: Mapping[str, Any],
) -> Dict[str, Any]:
    decision_path = (root / str(arguments.get("decision", ""))).resolve()
    if root not in decision_path.parents or not decision_path.is_file():
        raise WorkflowExecutionError("Dataset decision path is invalid")
    decision = load_json(decision_path)
    if decision.get("status") != "pass" or not isinstance(decision.get("roles"), dict):
        raise WorkflowExecutionError("Dataset source decision is not pass")
    entries = _role_entries(decision["roles"])
    receipts = []
    remote_destinations = set()
    total_archive_bytes = 0
    uploaded_archive_bytes_upper_bound = 0
    for item in entries:
        try:
            archive_parts = archive_parts_from_entry(item)
        except DatasetSourceContractError as exc:
            raise WorkflowExecutionError(str(exc)) from exc
        for part in archive_parts:
            route = part["transfer_route"]
            source = part["source_path_or_url"]
            expected = part["archive_sha256"]
            if route == "remote_existing":
                remote_path = source
                observed = _remote_sha256(remote_path)
                transfer_action = "reused_reviewed_remote"
            elif route == "local_existing_upload":
                local_path = Path(source).resolve()
                if not local_path.is_file() or sha256_file(local_path) != expected:
                    raise WorkflowExecutionError(
                        "Local archive is absent or hash-mismatched: "
                        f"{item['dataset_id']}:{part['archive_part_id']}"
                    )
                remote_path = _safe_remote_filename(str(item["dataset_id"]), source)
                if remote_path in remote_destinations:
                    raise WorkflowExecutionError(
                        f"Archive parts collide at remote destination: {remote_path}"
                    )
                total_archive_bytes += local_path.stat().st_size
                observed, transfer_action = _stage_local_archive(
                    local_path, remote_path, expected
                )
                if transfer_action == "uploaded_resumable_atomic":
                    uploaded_archive_bytes_upper_bound += local_path.stat().st_size
            else:
                raise WorkflowExecutionError(
                    "Domestic remote download requires a separately reviewed URL-fetch receipt"
                )
            if remote_path in remote_destinations:
                raise WorkflowExecutionError(
                    f"Archive parts collide at remote destination: {remote_path}"
                )
            remote_destinations.add(remote_path)
            if observed != expected:
                raise WorkflowExecutionError(
                    f"Staged archive hash mismatch for {item['dataset_id']}:"
                    f"{part['archive_part_id']}"
                )
            receipts.append(
                {
                    "dataset_id": item["dataset_id"],
                    "planned_role": item["planned_role"],
                    "archive_part_id": part["archive_part_id"],
                    "transfer_route": route,
                    "source": source,
                    "remote_path": remote_path,
                    "sha256": observed,
                    "byte_size": (
                        Path(source).resolve().stat().st_size
                        if route == "local_existing_upload"
                        else None
                    ),
                    "transfer_action": transfer_action,
                    "license_id": item["license_id"],
                    "license_evidence_path": item["license_evidence_path"],
                }
            )
    output = root / "evidence/data/staging_receipts.json"
    payload = {
        "schema_version": 1,
        "step_id": row["step_id"],
        "status": "pass",
        "remote_host": CURRENT_REMOTE_HOST,
        "decision_sha256": sha256_file(decision_path),
        "archives": receipts,
        "archive_count": len(receipts),
        "dataset_count": len(entries),
        "full_local_extractions": 0,
        "transfer_policy": {
            "protocol": "openssh_sftp_reput_to_temporary_then_atomic_promote",
            "resume_supported": True,
            "remote_hash_partition": "cu",
            "login_node_full_archive_hashes": 0,
        },
        "resource_and_cost_estimate": {
            "total_archive_bytes": total_archive_bytes,
            "uploaded_archive_bytes_upper_bound_this_attempt": (
                uploaded_archive_bytes_upper_bound
            ),
            "hash_jobs_maximum": len(receipts),
            "hash_job_cpus": 1,
            "hash_job_time_limit_hours_each": 0.75,
            "cpu_price_rmb_per_core_hour": 0.01,
            "cpu_hash_cost_upper_rmb": round(len(receipts) * 0.75 * 0.01, 4),
            "storage_tb_decimal": round(total_archive_bytes / 1_000_000_000_000, 6),
            "storage_price_rmb_per_tb_day": 2.5,
            "storage_cost_rmb_per_day": round(
                total_archive_bytes / 1_000_000_000_000 * 2.5, 4
            ),
            "gpu_count": 0,
            "unpriced_items": ["network_transfer", "provider_rounding"],
        },
    }
    write_json_artifact(output, payload)
    return {
        "status": "pass",
        "output_paths": [output.relative_to(root).as_posix()],
        "inputs": [
            {"path": decision_path.relative_to(root).as_posix(), "sha256": sha256_file(decision_path)}
        ],
        "details": {
            "archive_count": len(receipts),
            "dataset_count": len(entries),
            "full_local_extractions": 0,
            "uploaded_archive_bytes_upper_bound": uploaded_archive_bytes_upper_bound,
            "resumable_upload": True,
            "remote_hash_partition": "cu",
        },
    }


def _bounded_minimal_sample_records(
    dataset_id: str,
    selected: Any,
    max_groups: int,
) -> tuple[list[Dict[str, Any]], list[str], list[str]]:
    if not isinstance(selected, list) or not selected:
        raise WorkflowExecutionError(
            f"Dataset {dataset_id} lacks reviewed minimal_sample_records"
        )
    required = {"record_id", "raw_group_id", "member_path", "modality"}
    validated: list[Dict[str, Any]] = []
    candidate_groups: list[str] = []
    for record in selected:
        if not isinstance(record, dict) or not required.issubset(record):
            raise WorkflowExecutionError("Minimal-sample record schema is incomplete")
        group_id = str(record.get("raw_group_id", ""))
        if not group_id:
            raise WorkflowExecutionError(
                f"Dataset {dataset_id} has an empty minimal-sample raw group"
            )
        validated.append(dict(record))
        if group_id not in candidate_groups:
            candidate_groups.append(group_id)
    selected_groups = candidate_groups[:max_groups]
    selected_group_set = set(selected_groups)
    bounded = [
        record
        for record in validated
        if str(record["raw_group_id"]) in selected_group_set
    ]
    return bounded, candidate_groups, selected_groups


def build_minimal_sample(
    root: Path,
    row: Mapping[str, str],
    arguments: Mapping[str, Any],
) -> Dict[str, Any]:
    max_groups = int(arguments.get("max_groups", 0))
    if max_groups <= 0 or max_groups > 2:
        raise WorkflowExecutionError("E020 permits at most two raw groups per dataset")
    decision_path = root / "evidence/data/dataset_source_decision.json"
    decision = load_json(decision_path)
    if decision.get("status") != "pass":
        raise WorkflowExecutionError("Dataset source decision is not pass")
    records = []
    selection_audit = []
    for item in _role_entries(decision["roles"]):
        try:
            archive_parts = {
                part["archive_part_id"]: part for part in archive_parts_from_entry(item)
            }
        except DatasetSourceContractError as exc:
            raise WorkflowExecutionError(str(exc)) from exc
        selected, candidate_groups, selected_groups = _bounded_minimal_sample_records(
            str(item["dataset_id"]), item.get("minimal_sample_records"), max_groups
        )
        selection_audit.append(
            {
                "dataset_id": str(item["dataset_id"]),
                "candidate_group_count": len(candidate_groups),
                "candidate_groups": candidate_groups,
                "selected_group_count": len(selected_groups),
                "selected_groups": selected_groups,
                "selection_rule": "first_occurrence_in_reviewed_candidate_order",
            }
        )
        for record in selected:
            member = str(record["member_path"]).replace("\\", "/")
            archive_part_id = str(record.get("archive_part_id", "primary"))
            if archive_part_id not in archive_parts:
                raise WorkflowExecutionError(
                    f"Unknown minimal-sample archive part: {item['dataset_id']}:"
                    f"{archive_part_id}"
                )
            archive_part = archive_parts[archive_part_id]
            archive = Path(
                str(
                    archive_part.get("minimal_sample_local_path")
                    or archive_part["source_path_or_url"]
                )
            ).resolve()
            if (
                not archive.is_file()
                or sha256_file(archive) != archive_part["archive_sha256"]
            ):
                raise WorkflowExecutionError(
                    "Minimal-sample source is absent or hash-mismatched: "
                    f"{item['dataset_id']}:{archive_part_id}"
                )
            pure = PurePosixPath(member)
            if pure.is_absolute() or ".." in pure.parts:
                raise WorkflowExecutionError(f"Unsafe archive member path: {member}")
            row_limit = record.get("sample_row_limit")
            if row_limit is not None:
                if str(record["modality"]) != "methane" or pure.suffix.lower() != ".csv":
                    raise WorkflowExecutionError(
                        "sample_row_limit is permitted only for methane CSV records"
                    )
                try:
                    row_limit = int(row_limit)
                except (TypeError, ValueError) as exc:
                    raise WorkflowExecutionError("sample_row_limit must be an integer") from exc
                if row_limit <= 0 or row_limit > MAX_MINIMAL_SAMPLE_CSV_ROWS:
                    raise WorkflowExecutionError(
                        f"sample_row_limit must be in [1, {MAX_MINIMAL_SAMPLE_CSV_ROWS}]"
                    )
            data = _read_selected_member(archive, member, sample_row_limit=row_limit)
            suffix = PurePosixPath(member).suffix or archive.suffix
            safe_record = re.sub(r"[^A-Za-z0-9._-]", "_", str(record["record_id"]))
            safe_dataset = re.sub(r"[^A-Za-z0-9._-]", "_", str(item["dataset_id"]))
            target = root / "state/minimal_sample" / safe_dataset / f"{safe_record}{suffix}"
            write_once_bytes(target, data)
            records.append(
                {
                    "dataset_id": str(item["dataset_id"]),
                    "planned_role": item["planned_role"],
                    "record_id": str(record["record_id"]),
                    "raw_group_id": str(record["raw_group_id"]),
                    "modality": str(record["modality"]),
                    "source_archive": str(archive),
                    "source_archive_part_id": archive_part_id,
                    "source_member": member,
                    "sample_path": target.relative_to(root).as_posix(),
                    "byte_size": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "sample_only": True,
                    "sample_row_limit": row_limit,
                }
            )
    frame = pd.DataFrame.from_records(records)
    if frame.empty or frame.duplicated(["dataset_id", "record_id"]).any():
        raise WorkflowExecutionError("Minimal sample manifest is empty or has duplicate records")
    role_set = set(frame["planned_role"])
    if not REQUIRED_DATASET_ROLES.issubset(role_set):
        raise WorkflowExecutionError("Minimal sample does not cover every required dataset role")
    output = root / "evidence/data/minimal_sample_manifest.parquet"
    write_parquet_artifact(output, frame)
    return {
        "status": "pass",
        "output_paths": [output.relative_to(root).as_posix()],
        "inputs": [
            {"path": decision_path.relative_to(root).as_posix(), "sha256": sha256_file(decision_path)}
        ],
        "details": {
            "record_count": len(frame),
            "dataset_count": int(frame["dataset_id"].nunique()),
            "max_groups_per_dataset": max_groups,
            "full_extractions": 0,
            "selection_audit": selection_audit,
        },
    }


def _read_bounded_lines(stream: Any, *, data_row_limit: int) -> bytes:
    result = bytearray()
    for _ in range(data_row_limit + 1):
        line = stream.readline(MAX_MINIMAL_SAMPLE_MEMBER_BYTES + 1)
        if not line:
            break
        result.extend(line)
        if len(result) > MAX_MINIMAL_SAMPLE_MEMBER_BYTES:
            raise WorkflowExecutionError("Bounded CSV sample exceeds the byte-size limit")
    if not result:
        raise WorkflowExecutionError("Bounded CSV sample is empty")
    return bytes(result)


def _read_selected_member(
    archive: Path,
    member: str,
    *,
    sample_row_limit: Optional[int] = None,
) -> bytes:
    lowered = archive.name.lower()
    if lowered.endswith(".zip"):
        with zipfile.ZipFile(archive) as handle:
            info = handle.getinfo(member)
            if info.is_dir():
                raise WorkflowExecutionError(f"Minimal-sample member is a directory: {member}")
            if sample_row_limit is not None:
                with handle.open(info) as stream:
                    return _read_bounded_lines(stream, data_row_limit=sample_row_limit)
            if info.file_size > MAX_MINIMAL_SAMPLE_MEMBER_BYTES:
                raise WorkflowExecutionError(
                    f"Minimal-sample member exceeds {MAX_MINIMAL_SAMPLE_MEMBER_BYTES} bytes: {member}"
                )
            return handle.read(info)
    if lowered.endswith(".7z"):
        if sample_row_limit is not None:
            raise WorkflowExecutionError("sample_row_limit is unsupported for 7z members")
        try:
            return read_7z_member_bounded(
                archive, member, max_bytes=MAX_MINIMAL_SAMPLE_MEMBER_BYTES
            )
        except AdapterContractError as exc:
            raise WorkflowExecutionError(str(exc)) from exc
    if any(
        lowered.endswith(suffix)
        for suffix in (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")
    ):
        with tarfile.open(archive, mode="r:*") as handle:
            info = handle.getmember(member)
            if not info.isfile():
                raise WorkflowExecutionError(f"Minimal-sample member is not a file: {member}")
            stream = handle.extractfile(info)
            if stream is None:
                raise WorkflowExecutionError(f"Cannot read minimal-sample member: {member}")
            if sample_row_limit is not None:
                with stream:
                    return _read_bounded_lines(stream, data_row_limit=sample_row_limit)
            if info.size > MAX_MINIMAL_SAMPLE_MEMBER_BYTES:
                raise WorkflowExecutionError(
                    f"Minimal-sample member exceeds {MAX_MINIMAL_SAMPLE_MEMBER_BYTES} bytes: {member}"
                )
            return stream.read()
    if member not in {"", archive.name}:
        raise WorkflowExecutionError(
            f"Non-archive source must select its own filename: {archive.name}"
        )
    if sample_row_limit is not None:
        with archive.open("rb") as stream:
            return _read_bounded_lines(stream, data_row_limit=sample_row_limit)
    if archive.stat().st_size > MAX_MINIMAL_SAMPLE_MEMBER_BYTES:
        raise WorkflowExecutionError(
            f"Minimal-sample source exceeds {MAX_MINIMAL_SAMPLE_MEMBER_BYTES} bytes: {archive}"
        )
    return archive.read_bytes()


def lock_ontology(
    root: Path,
    row: Mapping[str, str],
    arguments: Mapping[str, Any],
) -> Dict[str, Any]:
    del arguments
    reviewed_path = root / "evidence/data/ontology_mapping.reviewed.yaml"
    payload = yaml.safe_load(reviewed_path.read_text(encoding="utf-8-sig"))
    validated = validate_ontology_lock(payload)
    if any("TBD" in json.dumps(entry, sort_keys=True) for entry in validated["entries"]):
        raise WorkflowExecutionError("Reviewed ontology still contains a TBD token")
    decision = load_json(root / "evidence/data/dataset_source_decision.json")
    dataset_ids = {str(item["dataset_id"]) for item in _role_entries(decision["roles"])}
    mapped_ids = {str(item["dataset_id"]) for item in validated["entries"]}
    if not dataset_ids.issubset(mapped_ids):
        raise WorkflowExecutionError("Ontology review does not cover every selected dataset")
    output = root / "data/locked/ontology_lock.yaml"
    data = yaml.safe_dump(validated, allow_unicode=True, sort_keys=True).encode("utf-8")
    write_once_bytes(output, data)
    return {
        "status": "pass",
        "output_paths": [output.relative_to(root).as_posix()],
        "inputs": [
            {"path": reviewed_path.relative_to(root).as_posix(), "sha256": sha256_file(reviewed_path)}
        ],
        "details": {"entry_count": len(validated["entries"])},
    }


def _dedup_assignment_manifest(
    file_manifest: pd.DataFrame, duplicates: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = [
        (str(item.dataset_id), str(item.raw_group_id))
        for item in file_manifest[["dataset_id", "raw_group_id"]]
        .drop_duplicates()
        .itertuples(index=False)
    ]
    parent = {key: key for key in keys}

    def find(key: tuple[str, str]) -> tuple[str, str]:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(left: tuple[str, str], right: tuple[str, str]) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        parent[second] = first

    record_group = {
        (str(item.dataset_id), str(item.record_id)): (
            str(item.dataset_id),
            str(item.raw_group_id),
        )
        for item in file_manifest.itertuples(index=False)
    }
    for item in duplicates.itertuples(index=False):
        left_record = (str(item.left_dataset_id), str(item.left_record_id))
        right_record = (str(item.right_dataset_id), str(item.right_record_id))
        if left_record not in record_group or right_record not in record_group:
            raise WorkflowExecutionError("Dedup report references an unknown record")
        union(record_group[left_record], record_group[right_record])
    components: Dict[tuple[str, str], list[tuple[str, str]]] = {}
    for key in keys:
        components.setdefault(find(key), []).append(key)
    component_ids = {}
    component_datasets = {}
    for members in components.values():
        digest = canonical_json_sha256(sorted([list(member) for member in members]))
        datasets = sorted({member[0] for member in members})
        assignment_dataset = (
            datasets[0] if len(datasets) == 1 else f"__cross_dataset_dedup__{digest[:16]}"
        )
        for member in members:
            component_ids[member] = f"dedup-{digest[:24]}"
            component_datasets[member] = assignment_dataset
    original = file_manifest[["dataset_id", "record_id", "raw_group_id"]].copy()
    assignment = pd.DataFrame(
        {
            "dataset_id": [
                component_datasets[(str(item.dataset_id), str(item.raw_group_id))]
                for item in original.itertuples(index=False)
            ],
            "record_id": [f"assignment-{index:012d}" for index in range(len(original))],
            "raw_group_id": [
                component_ids[(str(item.dataset_id), str(item.raw_group_id))]
                for item in original.itertuples(index=False)
            ],
        }
    )
    return assignment, original


def create_splits(
    root: Path,
    row: Mapping[str, str],
    arguments: Mapping[str, Any],
) -> Dict[str, Any]:
    if arguments.get("grouped") is not True:
        raise WorkflowExecutionError("E038 requires --grouped")
    file_path = root / "data/locked/file_manifest.parquet"
    ontology_path = root / "data/locked/ontology_lock.yaml"
    dedup_path = root / "data/locked/dedup_report.parquet"
    decision_path = root / "evidence/data/dataset_source_decision.json"
    protocol = yaml.safe_load(
        (root / "configs/protocol_lock.template.yaml").read_text(encoding="utf-8-sig")
    )
    decision = load_json(decision_path)
    file_manifest = read_file_manifest(file_path)
    duplicates = pd.read_parquet(dedup_path)
    branch_weights = protocol["data"]["branch_pool_weights"]
    episode_weights = protocol["data"]["episode_pool_weights"]
    family_weights = protocol["data"]["target_role_family_weights"]
    expected_branch = {"D_b_tr", "D_b_sel", "D_b_prob", "D_b_te"}
    expected_episode = {"D_e_tr", "D_e_sel", "D_e_pol", "D_e_te"}
    expected_pools = expected_branch | expected_episode
    if set(branch_weights) != expected_branch or set(episode_weights) != expected_episode:
        raise WorkflowExecutionError("Eight target-role leaf-pool weights are incomplete")
    if set(family_weights) != {"branch", "episode"}:
        raise WorkflowExecutionError("Branch and episode role-family weights are incomplete")
    if abs(sum(float(value) for value in family_weights.values()) - 1.0) > 1.0e-9:
        raise WorkflowExecutionError("Target role-family weights must sum to one")
    weights = {
        **{
            pool: float(family_weights["branch"]) * float(weight)
            for pool, weight in branch_weights.items()
        },
        **{
            pool: float(family_weights["episode"]) * float(weight)
            for pool, weight in episode_weights.items()
        },
    }
    assignment_manifest, original_keys = _dedup_assignment_manifest(
        file_manifest, duplicates
    )
    component_membership = pd.DataFrame(
        {
            "component_id": assignment_manifest["raw_group_id"].astype(str),
            "dataset_id": original_keys["dataset_id"].astype(str),
        }
    ).drop_duplicates()
    methane_entry = decision["roles"]["methane_dataset"]
    methane_dataset_id = str(methane_entry["dataset_id"])
    methane_components = set(
        component_membership.loc[
            component_membership["dataset_id"].astype(str) == methane_dataset_id,
            "component_id",
        ].astype(str)
    )
    if not methane_components:
        raise WorkflowExecutionError("Selected methane dataset has no dedup components")
    component_dataset_counts = component_membership.groupby("component_id")["dataset_id"].nunique()
    if any(int(component_dataset_counts.get(component, 0)) != 1 for component in methane_components):
        raise WorkflowExecutionError("A methane dedup component crosses dataset boundaries")
    generic_membership = component_membership.loc[
        ~component_membership["component_id"].astype(str).isin(methane_components)
    ].copy()
    generic_pool, generic_diagnostics = assign_balanced_component_pools(
        generic_membership,
        pool_weights=weights,
        seed=int(protocol["seeds"]["split"]),
    )
    record_components = original_keys.copy()
    record_components["component_id"] = assignment_manifest["raw_group_id"].astype(str).to_numpy()
    temporal_records = record_components.merge(
        file_manifest[["dataset_id", "record_id", "timestamp_or_order"]],
        on=["dataset_id", "record_id"],
        validate="one_to_one",
    )
    temporal_records = temporal_records.loc[
        temporal_records["dataset_id"].astype(str) == methane_dataset_id,
        ["component_id", "dataset_id", "timestamp_or_order"],
    ]
    methane_contract = methane_entry.get("adapter_contract", {}).get("methane", {})
    group_duration_seconds = int(methane_contract.get("group_duration_seconds", 0))
    purge_gap_seconds = int(protocol["data"]["methane"]["purge_gap_seconds"])
    temporal_pool, temporal_diagnostics = assign_temporal_cohort_pools(
        temporal_records,
        pool_weights=weights,
        seed=int(protocol["seeds"]["split"]),
        dataset_id=methane_dataset_id,
        group_duration_seconds=group_duration_seconds,
        purge_gap_seconds=purge_gap_seconds,
    )
    component_pool = {**generic_pool, **temporal_pool}
    if len(component_pool) != len(generic_pool) + len(temporal_pool):
        raise WorkflowExecutionError("Generic and temporal component assignments overlap")
    dataset_component_counts = dict(generic_diagnostics["dataset_component_counts"])
    dataset_component_counts[methane_dataset_id] = len(temporal_pool)
    dataset_pool_component_counts = dict(
        generic_diagnostics["dataset_pool_component_counts"]
    )
    dataset_pool_component_counts[methane_dataset_id] = temporal_diagnostics[
        "component_pool_counts"
    ]
    balance_diagnostics = {
        "algorithm": "generic_balanced_v1_plus_methane_temporal_cohort_v1",
        "dataset_component_counts": dataset_component_counts,
        "dataset_pool_component_counts": dataset_pool_component_counts,
        "cross_dataset_component_count": int(
            generic_diagnostics["cross_dataset_component_count"]
        ),
        "cross_component_search_nodes": int(
            generic_diagnostics["cross_component_search_nodes"]
        ),
        "global_component_count": len(component_pool),
        "global_pool_component_counts": {
            pool: sum(value == pool for value in component_pool.values())
            for pool in sorted(expected_pools)
        },
        "temporal_cohort_diagnostics": temporal_diagnostics,
    }
    split = original_keys.copy()
    split["pool"] = [
        component_pool[str(component_id)]
        for component_id in assignment_manifest["raw_group_id"]
    ]
    split["split_seed"] = int(protocol["seeds"]["split"])
    split["split_version"] = "minimal_v3_component_balanced_with_methane_temporal_cohorts"
    split["ontology_hash"] = sha256_file(ontology_path)
    split["dedup_report_hash"] = sha256_file(dedup_path)
    split = validate_split_manifest(split)
    assert_no_duplicate_cross_pool(split, duplicates)
    if set(split["pool"]) != expected_pools:
        raise WorkflowExecutionError("Full manifest does not populate all eight leaf pools")
    required_role_datasets = {
        "primary_visual_target": str(
            decision["roles"]["primary_visual_target"]["dataset_id"]
        ),
        "primary_rgbt_dataset": str(
            decision["roles"]["primary_rgbt_dataset"]["dataset_id"]
        ),
        "methane_dataset": str(decision["roles"]["methane_dataset"]["dataset_id"]),
    }
    role_pool_coverage = {}
    for role, dataset_id in sorted(required_role_datasets.items()):
        observed = set(
            split.loc[split["dataset_id"].astype(str) == dataset_id, "pool"].astype(str)
        )
        missing = sorted(expected_pools - observed)
        role_pool_coverage[role] = {
            "dataset_id": dataset_id,
            "observed_pools": sorted(observed),
            "missing_pools": missing,
        }
        if missing:
            raise WorkflowExecutionError(
                f"Required role {role}/{dataset_id} lacks split pools: {missing}"
            )
    output = root / "data/locked/split_manifest.parquet"
    write_parquet_artifact(output, split)
    pool_record_counts = {
        str(pool): int(count)
        for pool, count in split.groupby("pool", sort=True).size().items()
    }
    pool_raw_group_counts = {
        str(pool): int(
            frame[["dataset_id", "raw_group_id"]].drop_duplicates().shape[0]
        )
        for pool, frame in split.groupby("pool", sort=True)
    }
    return {
        "status": "pass",
        "output_paths": [output.relative_to(root).as_posix()],
        "inputs": [
            {"path": file_path.relative_to(root).as_posix(), "sha256": sha256_file(file_path)},
            {"path": ontology_path.relative_to(root).as_posix(), "sha256": sha256_file(ontology_path)},
            {"path": dedup_path.relative_to(root).as_posix(), "sha256": sha256_file(dedup_path)},
            {"path": decision_path.relative_to(root).as_posix(), "sha256": sha256_file(decision_path)},
        ],
        "details": {
            "record_count": len(split),
            "raw_group_count": int(
                split[["dataset_id", "raw_group_id"]].drop_duplicates().shape[0]
            ),
            "dedup_component_count": int(
                assignment_manifest[["dataset_id", "raw_group_id"]]
                .drop_duplicates()
                .shape[0]
            ),
            "pool_record_counts": pool_record_counts,
            "pool_raw_group_counts": pool_raw_group_counts,
            "balance_diagnostics": balance_diagnostics,
            "required_role_pool_coverage": role_pool_coverage,
            "split_version": "minimal_v3_component_balanced_with_methane_temporal_cohorts",
            "split_seed": int(protocol["seeds"]["split"]),
        },
    }


def create_fewshot(
    root: Path,
    row: Mapping[str, str],
    arguments: Mapping[str, Any],
) -> Dict[str, Any]:
    if int(arguments.get("ratio", 0)) != 10 or int(arguments.get("pairs", 0)) != 3:
        raise WorkflowExecutionError("E042 requires one 10 percent ratio and three pairs")
    split_path = root / "data/locked/split_manifest.parquet"
    file_path = root / "data/locked/file_manifest.parquet"
    decision_path = root / "evidence/data/dataset_source_decision.json"
    split = read_split_manifest(split_path)
    file_manifest = read_file_manifest(file_path)
    decision = load_json(decision_path)
    protocol = yaml.safe_load(
        (root / "configs/protocol_lock.template.yaml").read_text(encoding="utf-8-sig")
    )
    group_classes: Dict[tuple[str, str], set[str]] = {}
    for item in file_manifest.itertuples(index=False):
        summary = json.loads(str(item.label_summary_json))
        if isinstance(summary.get("class_ids"), list):
            classes = {str(value) for value in summary["class_ids"]}
        elif isinstance(summary.get("class_counts"), dict):
            classes = {str(value) for value in summary["class_counts"]}
        else:
            classes = {str(value) for value in summary if not str(value).startswith("_")}
        key = (str(item.dataset_id), str(item.raw_group_id))
        group_classes.setdefault(key, set()).update(classes)
    fewshot = build_fewshot_manifest(
        split,
        direction_id=str(decision["visual_direction_id"]),
        subset_seeds=protocol["seeds"]["fewshot_subset"],
        group_classes=group_classes,
        ratio_percent=10,
        parent_manifest_hash=sha256_file(split_path),
    )
    output = root / "data/locked/fewshot_manifest.parquet"
    write_parquet_artifact(output, fewshot)
    return {
        "status": "pass",
        "output_paths": [output.relative_to(root).as_posix()],
        "inputs": [
            {"path": split_path.relative_to(root).as_posix(), "sha256": sha256_file(split_path)},
            {"path": file_path.relative_to(root).as_posix(), "sha256": sha256_file(file_path)},
            {"path": decision_path.relative_to(root).as_posix(), "sha256": sha256_file(decision_path)},
        ],
        "details": {
            "subset_seed_count": int(fewshot["subset_seed"].nunique()),
            "included_group_count": int(fewshot.loc[fewshot["included"], "raw_group_id"].nunique()),
        },
    }


def audit_rgbt_inputs(
    root: Path,
    row: Mapping[str, str],
    arguments: Mapping[str, Any],
) -> Dict[str, Any]:
    del arguments
    file_path = root / "data/locked/file_manifest.parquet"
    split_path = root / "data/locked/split_manifest.parquet"
    decision_path = root / "evidence/data/dataset_source_decision.json"
    file_manifest = read_file_manifest(file_path)
    split = read_split_manifest(split_path)
    decision = load_json(decision_path)
    rgbt = decision["roles"]["primary_rgbt_dataset"]
    dataset_id = str(rgbt["dataset_id"])
    rows = file_manifest.loc[file_manifest["dataset_id"].astype(str) == dataset_id].copy()
    if rows.empty or rows["pair_id"].isna().any():
        raise WorkflowExecutionError("Selected RGB-T dataset lacks paired records")
    required_branch_roles = {"visible", "thermal"}
    thermal_source_modalities = {"thermal", "infrared"}
    valid_pairs = []
    observed_source_modalities = set()
    pair_signature_counts: Dict[str, int] = {}
    for pair_id, pair_rows in rows.groupby("pair_id"):
        modalities = set(pair_rows["modality"].astype(str).str.lower())
        thermal_members = modalities & thermal_source_modalities
        if len(pair_rows) != 2 or "visible" not in modalities or len(thermal_members) != 1:
            continue
        groups = set(pair_rows["raw_group_id"].astype(str))
        if len(groups) != 1:
            raise WorkflowExecutionError(f"RGB-T pair spans raw groups: {pair_id}")
        valid_pairs.append(str(pair_id))
        observed_source_modalities.update(modalities)
        signature = ",".join(sorted(modalities))
        pair_signature_counts[signature] = pair_signature_counts.get(signature, 0) + 1
    if not valid_pairs:
        raise WorkflowExecutionError("No valid visible-thermal pair remains after audit")
    lookup = split.loc[split["dataset_id"].astype(str) == dataset_id]
    valid_rows = rows.loc[rows["pair_id"].astype(str).isin(valid_pairs)].merge(
        lookup[["dataset_id", "record_id", "pool"]],
        on=["dataset_id", "record_id"],
        validate="one_to_one",
    )
    if int(valid_rows.groupby("pair_id")["pool"].nunique().max()) != 1:
        raise WorkflowExecutionError("A valid RGB-T pair crosses split pools")
    protocol = yaml.safe_load(
        (root / "configs/protocol_lock.template.yaml").read_text(encoding="utf-8-sig")
    )
    physical_temperature_claim_enabled = bool(
        protocol["data"]["rgbt"]["physical_temperature_claim_enabled"]
    )
    source_modality_to_branch_role = {
        modality: ("visible" if modality == "visible" else "thermal")
        for modality in sorted(observed_source_modalities)
    }
    payload = {
        "schema_version": 2,
        "step_id": row["step_id"],
        "status": "pass",
        "dataset_id": dataset_id,
        "required_modalities": sorted(required_branch_roles),
        "required_branch_roles": sorted(required_branch_roles),
        "observed_source_modalities": sorted(observed_source_modalities),
        "source_modality_to_branch_role": source_modality_to_branch_role,
        "pair_modality_signature_counts": pair_signature_counts,
        "physical_temperature_claim_enabled": physical_temperature_claim_enabled,
        "valid_pair_count": len(valid_pairs),
        "raw_group_count": int(valid_rows["raw_group_id"].nunique()),
        "pool_counts": {str(key): int(value) for key, value in lookup["pool"].value_counts().items()},
        "file_manifest_sha256": sha256_file(file_path),
        "split_manifest_sha256": sha256_file(split_path),
        "model_outcomes_used": False,
    }
    output = root / "data/locked/rgbt_input_lock.json"
    write_json_artifact(output, payload)
    return {
        "status": "pass",
        "output_paths": [output.relative_to(root).as_posix()],
        "inputs": [
            {"path": file_path.relative_to(root).as_posix(), "sha256": sha256_file(file_path)},
            {"path": split_path.relative_to(root).as_posix(), "sha256": sha256_file(split_path)},
        ],
        "details": {
            "valid_pair_count": len(valid_pairs),
            "observed_source_modalities": sorted(observed_source_modalities),
            "physical_temperature_claim_enabled": physical_temperature_claim_enabled,
        },
    }


def lock_methane_roles(
    root: Path,
    row: Mapping[str, str],
    arguments: Mapping[str, Any],
) -> Dict[str, Any]:
    del arguments
    file_path = root / "data/locked/file_manifest.parquet"
    split_path = root / "data/locked/split_manifest.parquet"
    decision_path = root / "evidence/data/dataset_source_decision.json"
    file_manifest = read_file_manifest(file_path)
    split = read_split_manifest(split_path)
    decision = load_json(decision_path)
    methane = decision["roles"]["methane_dataset"]
    dataset_id = str(methane["dataset_id"])
    rows = file_manifest.loc[file_manifest["dataset_id"].astype(str) == dataset_id].copy()
    if rows.empty or rows["timestamp_or_order"].isna().any():
        raise WorkflowExecutionError("Methane dataset lacks ordered records")
    joined = rows.merge(
        split[["dataset_id", "record_id", "pool"]],
        on=["dataset_id", "record_id"],
        validate="one_to_one",
    )
    protocol = yaml.safe_load(
        (root / "configs/protocol_lock.template.yaml").read_text(encoding="utf-8-sig")
    )
    methane_contract = protocol["data"]["methane"]
    history_seconds = int(methane_contract["history_seconds"])
    horizon_seconds = int(methane_contract["horizon_seconds"])
    purge_gap_seconds = int(methane_contract["purge_gap_seconds"])
    if purge_gap_seconds < history_seconds + horizon_seconds:
        raise WorkflowExecutionError("Methane purge gap does not cover history plus horizon")
    group_duration_seconds = int(
        methane.get("adapter_contract", {})
        .get("methane", {})
        .get("group_duration_seconds", 0)
    )
    if group_duration_seconds <= 0:
        raise WorkflowExecutionError("Methane group duration is not frozen")
    try:
        joined["cohort_epoch_seconds"] = (
            pd.to_datetime(joined["timestamp_or_order"], utc=True, errors="raise").astype("int64")
            // 1_000_000_000
        )
    except (TypeError, ValueError) as exc:
        raise WorkflowExecutionError("Methane record times are not parseable UTC timestamps") from exc
    cohort_pool_counts = joined.groupby("cohort_epoch_seconds")["pool"].nunique()
    if int(cohort_pool_counts.max()) != 1:
        raise WorkflowExecutionError("One methane time cohort crosses split pools")
    cohort_pools = (
        joined[["cohort_epoch_seconds", "pool"]]
        .drop_duplicates()
        .sort_values("cohort_epoch_seconds", kind="mergesort")
    )
    cohort_epochs = [int(value) for value in cohort_pools["cohort_epoch_seconds"]]
    if any(
        right - left != group_duration_seconds
        for left, right in zip(cohort_epochs, cohort_epochs[1:])
    ):
        raise WorkflowExecutionError("Methane source cohorts are not one continuous ordered series")
    observed_pool_order = []
    for pool in cohort_pools["pool"].astype(str):
        if not observed_pool_order or observed_pool_order[-1] != pool:
            observed_pool_order.append(pool)
    if observed_pool_order != list(TEMPORAL_POOL_ORDER):
        raise WorkflowExecutionError(
            f"Methane chronological pool order is invalid: {observed_pool_order}"
        )
    pool_rows = []
    for pool in TEMPORAL_POOL_ORDER:
        pool_frame = joined.loc[joined["pool"].astype(str) == pool].copy()
        if pool_frame.empty:
            raise WorkflowExecutionError(f"Methane chronological pool is empty: {pool}")
        cohort_values = sorted(set(int(value) for value in pool_frame["cohort_epoch_seconds"]))
        if any(
            right - left != group_duration_seconds
            for left, right in zip(cohort_values, cohort_values[1:])
        ):
            raise WorkflowExecutionError(f"Methane pool {pool} is not a contiguous time block")
        pool_rows.append(
            {
                "pool": str(pool),
                "record_count": len(pool_frame),
                "raw_group_count": int(pool_frame["raw_group_id"].nunique()),
                "cohort_count": len(cohort_values),
                "first_order": pd.Timestamp(cohort_values[0], unit="s", tz="UTC").isoformat(),
                "last_order": pd.Timestamp(cohort_values[-1], unit="s", tz="UTC").isoformat(),
                "first_epoch_seconds": cohort_values[0],
                "last_epoch_seconds": cohort_values[-1],
                "block_end_exclusive_epoch_seconds": cohort_values[-1]
                + group_duration_seconds,
                "record_hash": canonical_json_sha256(
                    sorted(str(value) for value in pool_frame["record_id"])
                ),
            }
        )
    block_by_pool = {str(item["pool"]): item for item in pool_rows}
    family_gap_evidence = []
    for family, pools in TEMPORAL_FAMILY_POOL_ORDER.items():
        for earlier, later in zip(pools, pools[1:]):
            gap_seconds = int(
                block_by_pool[later]["first_epoch_seconds"]
                - block_by_pool[earlier]["block_end_exclusive_epoch_seconds"]
            )
            passed = gap_seconds >= purge_gap_seconds
            family_gap_evidence.append(
                {
                    "family": family,
                    "from_pool": earlier,
                    "to_pool": later,
                    "gap_seconds": gap_seconds,
                    "required_gap_seconds": purge_gap_seconds,
                    "passed": passed,
                }
            )
            if not passed:
                raise WorkflowExecutionError(
                    f"Methane {family} purge gap {earlier}->{later} is too short"
                )
    payload = {
        "schema_version": 2,
        "step_id": row["step_id"],
        "status": "pass",
        "dataset_id": dataset_id,
        "history_seconds": history_seconds,
        "horizon_seconds": horizon_seconds,
        "stride_seconds": int(methane_contract["stride_seconds"]),
        "purge_gap_seconds": purge_gap_seconds,
        "group_duration_seconds": group_duration_seconds,
        "imputation_fit_pool": "D_b_tr",
        "normalization_fit_pool": "D_b_tr",
        "chronological_pool_order": list(TEMPORAL_POOL_ORDER),
        "chronological_pool_evidence": pool_rows,
        "family_purge_gap_evidence": family_gap_evidence,
        "minimum_family_purge_gap_seconds": min(
            int(item["gap_seconds"]) for item in family_gap_evidence
        ),
        "same_cohort_cross_pool_count": 0,
        "continuous_pool_block_error_count": 0,
        "future_values_allowed_in_features": False,
        "file_manifest_sha256": sha256_file(file_path),
        "split_manifest_sha256": sha256_file(split_path),
    }
    output = root / "data/locked/methane_role_lock.json"
    write_json_artifact(output, payload)
    return {
        "status": "pass",
        "output_paths": [output.relative_to(root).as_posix()],
        "inputs": [
            {"path": file_path.relative_to(root).as_posix(), "sha256": sha256_file(file_path)},
            {"path": split_path.relative_to(root).as_posix(), "sha256": sha256_file(split_path)},
        ],
        "details": {
            "dataset_id": dataset_id,
            "pool_count": len(pool_rows),
            "minimum_family_purge_gap_seconds": min(
                int(item["gap_seconds"]) for item in family_gap_evidence
            ),
            "cohort_count": len(cohort_epochs),
            "record_count": len(joined),
        },
    }


def audit_graph_eligibility(
    root: Path,
    row: Mapping[str, str],
    arguments: Mapping[str, Any],
) -> Dict[str, Any]:
    del arguments
    rgbt_path = root / "data/locked/rgbt_input_lock.json"
    ontology_path = root / "data/locked/ontology_lock.yaml"
    rgbt = load_json(rgbt_path)
    ontology = validate_ontology_lock(
        yaml.safe_load(ontology_path.read_text(encoding="utf-8-sig"))
    )
    protocol = yaml.safe_load(
        (root / "configs/protocol_lock.template.yaml").read_text(encoding="utf-8-sig")
    )
    floor = int(protocol["data"]["independent_group_floor_hard_min"])
    compatible = {
        str(item["canonical_concept_id"])
        for item in ontology["entries"]
        if item["mapping_status"] == "compatible"
        and str(item["dataset_id"]) == str(rgbt["dataset_id"])
    }
    eligible = (
        rgbt.get("status") == "pass"
        and int(rgbt.get("raw_group_count", 0)) >= floor
        and bool(compatible)
    )
    reasons = []
    if int(rgbt.get("raw_group_count", 0)) < floor:
        reasons.append("independent_group_floor_not_met")
    if not compatible:
        reasons.append("no_compatible_concept_mapping")
    payload = {
        "schema_version": 1,
        "step_id": row["step_id"],
        "status": "pass",
        "graph_eligible": eligible,
        "eligibility_reasons": reasons or ["observed_pairing_and_group_floor_pass"],
        "dataset_id": rgbt["dataset_id"],
        "observed_modalities": rgbt["observed_source_modalities"],
        "branch_roles": rgbt["required_branch_roles"],
        "source_modality_to_branch_role": rgbt["source_modality_to_branch_role"],
        "physical_temperature_claim_enabled": bool(
            rgbt["physical_temperature_claim_enabled"]
        ),
        "independent_raw_group_count": int(rgbt["raw_group_count"]),
        "independent_group_floor": floor,
        "compatible_concept_ids": sorted(compatible),
        "virtual_edges_counted_as_observed": False,
        "model_outcomes_used": False,
        "rgbt_input_lock_sha256": sha256_file(rgbt_path),
        "ontology_lock_sha256": sha256_file(ontology_path),
    }
    output = root / "data/locked/graph_eligibility_lock.json"
    write_json_artifact(output, payload)
    return {
        "status": "pass",
        "output_paths": [output.relative_to(root).as_posix()],
        "inputs": [
            {"path": rgbt_path.relative_to(root).as_posix(), "sha256": sha256_file(rgbt_path)},
            {"path": ontology_path.relative_to(root).as_posix(), "sha256": sha256_file(ontology_path)},
        ],
        "details": {"graph_eligible": eligible},
    }


def smoke_local(
    root: Path,
    row: Mapping[str, str],
    arguments: Mapping[str, Any],
) -> Dict[str, Any]:
    if arguments.get("minimal") is not True:
        raise WorkflowExecutionError("E062 requires --minimal")
    manifest_path = root / "evidence/data/minimal_sample_manifest.parquet"
    manifest = pd.read_parquet(manifest_path)
    required = {"dataset_id", "record_id", "raw_group_id", "modality", "sample_path", "sha256"}
    if manifest.empty or not required.issubset(manifest.columns):
        raise WorkflowExecutionError("Minimal sample manifest is empty or incomplete")
    validated = 0
    modalities = set()
    for item in manifest.itertuples(index=False):
        path = (root / str(item.sample_path)).resolve()
        if root not in path.parents or not path.is_file() or sha256_file(path) != str(item.sha256):
            raise WorkflowExecutionError(f"Minimal sample hash mismatch: {item.record_id}")
        validated += 1
        modalities.add(str(item.modality).lower())
    rgbt_lock_path = root / "data/locked/rgbt_input_lock.json"
    rgbt_lock = load_json(rgbt_lock_path)
    source_mapping = rgbt_lock.get("source_modality_to_branch_role")
    if rgbt_lock.get("status") != "pass" or not isinstance(source_mapping, dict):
        raise WorkflowExecutionError("E062 requires a passed RGB-T input lock")
    normalized_mapping = {
        str(source).lower(): str(branch).lower()
        for source, branch in source_mapping.items()
    }
    branch_modalities = modalities - {"methane"}
    unmapped = branch_modalities - set(normalized_mapping)
    if unmapped:
        raise WorkflowExecutionError(
            "Minimal sample contains RGB-T modalities absent from the locked mapping: "
            + ", ".join(sorted(unmapped))
        )
    branch_roles = {normalized_mapping[modality] for modality in branch_modalities}
    if "methane" not in modalities or not {"visible", "thermal"}.issubset(branch_roles):
        raise WorkflowExecutionError(
            "Minimal sample lacks locked visible, thermal-branch, or methane coverage"
        )
    bundle = root / "evidence/smoke/local_module_bundle"
    pipeline = run_minimal_pipeline(bundle)
    receipt_path = bundle / "integration_receipt.json"
    payload = {
        "schema_version": 1,
        "step_id": row["step_id"],
        "status": "pass",
        "actual_sample_record_count": validated,
        "actual_sample_modalities": sorted(modalities),
        "actual_sample_branch_roles": sorted(branch_roles),
        "minimal_sample_manifest_sha256": sha256_file(manifest_path),
        "rgbt_input_lock_sha256": sha256_file(rgbt_lock_path),
        "module_pipeline_mode": pipeline["mode"],
        "module_pipeline_receipt_sha256": sha256_file(receipt_path),
        "full_local_dataset_extractions": 0,
        "remote_connections": 0,
        "slurm_jobs_created": 0,
        "performance_claims_authorized": False,
    }
    output = root / "evidence/smoke/local_smoke.json"
    write_json_artifact(output, payload)
    return {
        "status": "pass",
        "output_paths": [output.relative_to(root).as_posix()],
        "inputs": [
            {"path": manifest_path.relative_to(root).as_posix(), "sha256": sha256_file(manifest_path)},
            {"path": rgbt_lock_path.relative_to(root).as_posix(), "sha256": sha256_file(rgbt_lock_path)},
        ],
        "details": {"actual_sample_record_count": validated, "full_extractions": 0},
    }
