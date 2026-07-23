from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Union

from .governance.immutable import write_once_bytes, write_once_json
from .provenance import canonical_json_sha256, describe_artifact, sha256_file


PathLike = Union[str, Path]
COMPLETED_STATUSES = {"pass", "not_applicable"}
PLAN_COMMAND_PATTERN = re.compile(r"-Module mining1_exp\.cli\s+([^\s]+)")


class WorkflowExecutionError(RuntimeError):
    """Raised when a frozen workflow command cannot execute safely."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: PathLike) -> Dict[str, Any]:
    target = Path(path)
    payload = json.loads(target.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise WorkflowExecutionError(f"Expected a JSON object: {target}")
    return payload


def load_plan(project_root: PathLike) -> list[dict[str, str]]:
    path = Path(project_root) / "plans/experiment_steps.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(not row.get("step_id") for row in rows):
        raise WorkflowExecutionError("Experiment plan is empty or malformed")
    if len({row["step_id"] for row in rows}) != len(rows):
        raise WorkflowExecutionError("Experiment plan contains duplicate step IDs")
    return rows


def _command_from_row(row: Mapping[str, str]) -> Optional[str]:
    match = PLAN_COMMAND_PATTERN.search(row["command_entry"])
    return match.group(1) if match else None


def resolve_plan_step(
    project_root: PathLike,
    command: str,
    arguments: Mapping[str, Any],
) -> dict[str, str]:
    candidates = [row for row in load_plan(project_root) if _command_from_row(row) == command]
    if command == "close-gate":
        gate = str(arguments.get("gate", ""))
        candidates = [row for row in candidates if f"--gate {gate}" in row["command_entry"]]
    elif command == "evaluate":
        package = str(arguments.get("package", ""))
        candidates = [
            row for row in candidates if f"--package {package}" in row["command_entry"]
        ]
    if len(candidates) != 1:
        raise WorkflowExecutionError(
            f"Command {command!r} resolves to {len(candidates)} frozen plan steps"
        )
    return dict(candidates[0])


def plan_by_id(project_root: PathLike) -> dict[str, dict[str, str]]:
    return {row["step_id"]: row for row in load_plan(project_root)}


def require_completed_dependencies(project_root: PathLike, row: Mapping[str, str]) -> None:
    steps = plan_by_id(project_root)
    for dependency in row["depends_on"].split("|"):
        if dependency == "none":
            continue
        status = steps.get(dependency, {}).get("status")
        if status not in COMPLETED_STATUSES:
            raise WorkflowExecutionError(
                f"Dependency {dependency} is not complete for {row['step_id']}: {status}"
            )
        review_path = Path(project_root) / "evidence/step_reviews" / f"{dependency}.json"
        review = load_json(review_path)
        if (
            review.get("status") != status
            or review.get("advance_allowed") is not True
        ):
            raise WorkflowExecutionError(
                f"Dependency review does not authorize advancement: {dependency}"
            )


def require_selected_local_python(project_root: PathLike) -> Dict[str, Any]:
    decision = load_json(Path(project_root) / "configs/environment_decision.lock.json")
    selected = Path(str(decision.get("selected_local_python", ""))).resolve()
    running = Path(os.path.realpath(os.sys.executable)).resolve()
    if selected != running:
        raise WorkflowExecutionError(
            f"Selected local Python mismatch: selected={selected}, running={running}"
        )
    if decision.get("status") != "locked":
        raise WorkflowExecutionError("Environment decision is not locked")
    return decision


def _directory_manifest(path: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": item.relative_to(path).as_posix(),
            "bytes": item.stat().st_size,
            "sha256": sha256_file(item),
        }
        for item in sorted(path.rglob("*"))
        if item.is_file() and "__pycache__" not in item.parts
    ]


def describe_output(project_root: PathLike, relative_path: str) -> Dict[str, Any]:
    root = Path(project_root).resolve()
    target = (root / relative_path).resolve()
    if root not in target.parents and target != root:
        raise WorkflowExecutionError(f"Output escapes project root: {relative_path}")
    if target.is_file():
        digest = describe_artifact(target).to_dict()
        digest["path"] = target.relative_to(root).as_posix()
        digest["kind"] = "file"
        return digest
    if target.is_dir():
        manifest = _directory_manifest(target)
        if not manifest:
            raise WorkflowExecutionError(f"Output directory is empty: {relative_path}")
        return {
            "path": target.relative_to(root).as_posix() + "/",
            "kind": "directory",
            "file_count": len(manifest),
            "bytes": sum(item["bytes"] for item in manifest),
            "sha256": canonical_json_sha256(manifest),
        }
    raise WorkflowExecutionError(f"Required output does not exist: {relative_path}")


def validate_output_digests(project_root: PathLike, outputs: Sequence[Mapping[str, Any]]) -> None:
    for expected in outputs:
        observed = describe_output(project_root, str(expected["path"]).rstrip("/"))
        for field in ("kind", "bytes", "sha256"):
            if observed.get(field) != expected.get(field):
                raise WorkflowExecutionError(
                    f"Output digest drift for {expected['path']}: {field}"
                )
        if expected.get("kind") == "directory" and observed.get("file_count") != expected.get(
            "file_count"
        ):
            raise WorkflowExecutionError(
                f"Output directory file-count drift: {expected['path']}"
            )


def hash_existing_inputs(project_root: PathLike, paths: Iterable[str]) -> list[Dict[str, Any]]:
    root = Path(project_root).resolve()
    digests = []
    for relative in paths:
        target = (root / relative).resolve()
        if target.exists():
            digests.append(describe_output(root, target.relative_to(root).as_posix()))
    return digests


def write_json_artifact(path: PathLike, payload: Mapping[str, Any]) -> Dict[str, Any]:
    result = write_once_json(path, payload)
    return result.artifact.to_dict()


def write_parquet_artifact(path: PathLike, frame: Any) -> Dict[str, Any]:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{os.getpid()}.stage"
    if temporary.exists():
        raise WorkflowExecutionError(f"Parquet staging path exists: {temporary}")
    try:
        frame.to_parquet(temporary, index=False)
        result = write_once_bytes(target, temporary.read_bytes())
    finally:
        temporary.unlink(missing_ok=True)
    return result.artifact.to_dict()


def _acquire_lock(project_root: Path, step_id: str) -> Path:
    lock = project_root / "state/locks" / f"{step_id}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise WorkflowExecutionError(f"Workflow lock already exists: {lock}") from exc
    with os.fdopen(descriptor, "w", encoding="ascii") as handle:
        handle.write(f"pid={os.getpid()}\n")
    return lock


def write_failure_receipt(
    project_root: PathLike,
    row: Mapping[str, str],
    command: str,
    error: Exception,
) -> Path:
    root = Path(project_root).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = root / "evidence/failures" / row["step_id"] / f"{stamp}.failure.json"
    payload = {
        "schema_version": 1,
        "step_id": row["step_id"],
        "status": "fail",
        "command": command,
        "error_type": type(error).__name__,
        "error": str(error),
        "last_safe_checkpoint": None,
        "retry_requires_frozen_rule_review": True,
        "recorded_at": utc_now(),
    }
    write_once_json(path, payload)
    return path


def execute_with_contract(
    project_root: PathLike,
    command: str,
    arguments: Mapping[str, Any],
    handler: Any,
) -> Dict[str, Any]:
    root = Path(project_root).resolve()
    row = resolve_plan_step(root, command, arguments)
    require_selected_local_python(root)
    require_completed_dependencies(root, row)
    receipt_path = root / "evidence/command_receipts" / f"{row['step_id']}.json"
    if receipt_path.is_file():
        receipt = load_json(receipt_path)
        if receipt.get("status") != "pass" or receipt.get("step_id") != row["step_id"]:
            raise WorkflowExecutionError(f"Existing command receipt is invalid: {receipt_path}")
        validate_output_digests(root, receipt.get("outputs", []))
        return {
            "status": "pass",
            "step_id": row["step_id"],
            "mode": "validated_existing",
            "receipt_sha256": sha256_file(receipt_path),
        }
    lock = _acquire_lock(root, row["step_id"])
    try:
        result = handler(root, row, dict(arguments))
        if result.get("status") != "pass":
            raise WorkflowExecutionError(f"Handler did not pass: {row['step_id']}")
        output_paths = result.get("output_paths")
        if not isinstance(output_paths, list) or not output_paths:
            raise WorkflowExecutionError(f"Handler returned no outputs: {row['step_id']}")
        outputs = [describe_output(root, str(path)) for path in output_paths]
        receipt = {
            "schema_version": 1,
            "step_id": row["step_id"],
            "status": "pass",
            "command": command,
            "arguments": dict(sorted(arguments.items())),
            "plan_contract_sha256": canonical_json_sha256(
                {key: value for key, value in row.items() if key != "status"}
            ),
            "inputs": result.get("inputs", []),
            "outputs": outputs,
            "details": result.get("details", {}),
            "created_at": utc_now(),
        }
        write_once_json(receipt_path, receipt)
        return {
            "status": "pass",
            "step_id": row["step_id"],
            "mode": "created",
            "receipt_sha256": sha256_file(receipt_path),
            "outputs": outputs,
        }
    except Exception as exc:
        failure = write_failure_receipt(root, row, command, exc)
        raise WorkflowExecutionError(
            f"{row['step_id']} failed; retained {failure.relative_to(root).as_posix()}: {exc}"
        ) from exc
    finally:
        lock.unlink(missing_ok=True)
