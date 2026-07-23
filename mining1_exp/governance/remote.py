from __future__ import annotations

from datetime import datetime
import re
from typing import Any, Mapping

from .immutable import GovernanceContractError, require_sha256, require_timestamp


REMOTE_HOST = "xinxi-zhyh@211.87.115.228"


def _require_list(payload: Mapping[str, Any], field: str) -> list[Any]:
    value = payload.get(field)
    if not isinstance(value, list):
        raise GovernanceContractError(f"{field} must be a list")
    return value


def _parse_time(value: Any, field: str) -> datetime:
    normalized = require_timestamp(value, field)
    parse_value = re.sub(r"(\.\d{6})\d+(Z|[+-]\d\d:\d\d)$", r"\1\2", normalized)
    return datetime.fromisoformat(parse_value.replace("Z", "+00:00"))


def _require_job_ids(values: list[Any], field: str) -> list[str]:
    normalized = [str(value) for value in values]
    if len(set(normalized)) != len(normalized) or any(
        re.fullmatch(r"\d+(?:_\d+)?", value) is None for value in normalized
    ):
        raise GovernanceContractError(f"{field} contains invalid or duplicate Slurm job IDs")
    return normalized


def _require_artifact_digests(payload: Mapping[str, Any], field: str) -> list[Any]:
    values = _require_list(payload, field)
    paths = []
    for value in values:
        if not isinstance(value, dict) or not {"path", "sha256"}.issubset(value):
            raise GovernanceContractError(f"{field} requires path and SHA-256 records")
        path = str(value["path"])
        if not path.strip():
            raise GovernanceContractError(f"{field} contains an empty path")
        require_sha256(value["sha256"], f"{field}.sha256")
        paths.append(path)
    if len(set(paths)) != len(paths):
        raise GovernanceContractError(f"{field} contains duplicate paths")
    return values


def validate_remote_monitor_receipt(payload: Mapping[str, Any]) -> None:
    required = {
        "step_id",
        "monitored_at",
        "host",
        "slurm_job_ids",
        "gpu_expected",
        "squeue_state",
        "sacct_state",
        "relevant_processes",
        "tmux_and_screen_sessions",
        "gpu_telemetry_artifacts",
        "gpu_telemetry_excerpt",
        "recent_log_artifacts",
        "checkpoint_artifacts",
        "result_artifacts",
        "workflow_state",
        "remote_error",
        "next_check_due_at",
    }
    if not required.issubset(payload) or not str(payload["step_id"]).strip():
        raise GovernanceContractError("remote monitor receipt is incomplete")
    if payload["host"] != REMOTE_HOST:
        raise GovernanceContractError("remote monitor references a forbidden host")
    monitored = _parse_time(payload["monitored_at"], "monitored_at")
    state = payload["workflow_state"]
    allowed = {"pending", "running", "completed", "failed", "blocked", "connection_failed"}
    if state not in allowed:
        raise GovernanceContractError("remote workflow state is invalid")
    job_ids = _require_job_ids(_require_list(payload, "slurm_job_ids"), "slurm_job_ids")
    if not isinstance(payload["gpu_expected"], bool):
        raise GovernanceContractError("remote monitor gpu_expected must be boolean")
    for field in (
        "relevant_processes",
        "tmux_and_screen_sessions",
        "gpu_telemetry_excerpt",
    ):
        values = _require_list(payload, field)
        if any(not str(value).strip() for value in values):
            raise GovernanceContractError(f"{field} contains an empty entry")
    for field in (
        "gpu_telemetry_artifacts",
        "recent_log_artifacts",
        "checkpoint_artifacts",
        "result_artifacts",
    ):
        _require_artifact_digests(payload, field)
    if not isinstance(payload["squeue_state"], dict) or not isinstance(
        payload["sacct_state"], dict
    ):
        raise GovernanceContractError("remote scheduler states must be mappings")
    if state in {"pending", "running"}:
        due = _parse_time(payload["next_check_due_at"], "next_check_due_at")
        delta_minutes = (due - monitored).total_seconds() / 60.0
        if not 29.0 <= delta_minutes <= 31.0:
            raise GovernanceContractError("running monitor cadence must be 30 minutes")
    elif payload["next_check_due_at"] is not None:
        raise GovernanceContractError("terminal monitor state cannot schedule another check")
    if state in {"connection_failed", "failed", "blocked"} and not str(
        payload["remote_error"] or ""
    ).strip():
        raise GovernanceContractError("failed or blocked monitor state requires an error")
    if state not in {"connection_failed", "failed", "blocked"} and str(
        payload["remote_error"] or ""
    ).strip():
        raise GovernanceContractError("healthy monitor state cannot retain a remote error")
    squeue_records = payload["squeue_state"].get("records")
    sacct_records = payload["sacct_state"].get("records")
    if not isinstance(squeue_records, list) or not isinstance(sacct_records, list):
        raise GovernanceContractError("remote scheduler states require parsed records")
    active_ids = {
        str(record.get("job_id"))
        for record in squeue_records
        if isinstance(record, dict) and str(record.get("job_id")) in job_ids
    }
    if state in {"pending", "running"} and job_ids and not active_ids:
        raise GovernanceContractError("active monitor state lacks a requested squeue job")
    if (
        state == "running"
        and payload["gpu_expected"]
        and (
            not payload["gpu_telemetry_artifacts"]
            or not payload["gpu_telemetry_excerpt"]
        )
    ):
        raise GovernanceContractError("running GPU job lacks compute-node telemetry evidence")
    if state in {"completed", "failed"} and job_ids:
        matching_records = [
            record
            for record in sacct_records
            if isinstance(record, dict) and str(record.get("job_id")) in job_ids
        ]
        matching_ids = [str(record.get("job_id")) for record in matching_records]
        if len(set(matching_ids)) != len(matching_ids):
            raise GovernanceContractError("terminal sacct evidence contains duplicate jobs")
        terminal_by_id = {
            str(record.get("job_id")): record
            for record in sacct_records
            if isinstance(record, dict) and str(record.get("job_id")) in job_ids
        }
        if set(terminal_by_id) != set(job_ids):
            raise GovernanceContractError("terminal monitor state lacks complete sacct evidence")
        completed = all(
            str(record.get("state")) == "COMPLETED"
            and str(record.get("exit_code")) == "0:0"
            for record in terminal_by_id.values()
        )
        if state == "completed" and not completed:
            raise GovernanceContractError("completed monitor state has failed sacct evidence")
        terminal_states = {
            "COMPLETED",
            "FAILED",
            "CANCELLED",
            "TIMEOUT",
            "OUT_OF_MEMORY",
            "NODE_FAIL",
            "PREEMPTED",
            "BOOT_FAIL",
            "DEADLINE",
        }
        if state == "failed" and (
            completed
            or any(str(record.get("state")) not in terminal_states for record in terminal_by_id.values())
        ):
            raise GovernanceContractError("failed monitor state lacks terminal failure evidence")


def validate_remote_closeout(payload: Mapping[str, Any], *, require_verified: bool) -> None:
    required = {
        "trigger_step_id",
        "remote_action_id",
        "remote_action_at",
        "trigger_kind",
        "closed_at",
        "host",
        "account_scope",
        "slurm_jobs_and_allocations",
        "interactive_sessions",
        "tmux_and_screen_sessions",
        "relevant_processes",
        "transfers_and_monitors",
        "unsynchronized_artifacts",
        "retained_resources_with_reason",
        "cancellation_or_release_actions",
        "provider_stop_action",
        "final_billing_state",
    }
    if not required.issubset(payload):
        raise GovernanceContractError("remote closeout receipt is incomplete")
    if payload["host"] != REMOTE_HOST or payload["account_scope"] != "all_user_owned_work":
        raise GovernanceContractError("remote closeout is not account-wide on the current host")
    if payload["trigger_kind"] not in {
        "remote_action_end",
        "connection_check_end",
        "final_reconcile",
    }:
        raise GovernanceContractError("remote closeout trigger kind is invalid")
    if not str(payload["trigger_step_id"]).strip():
        raise GovernanceContractError("remote closeout trigger step is required")
    action_at = _parse_time(payload["remote_action_at"], "remote_action_at")
    closed_at = _parse_time(payload["closed_at"], "closed_at")
    if closed_at < action_at:
        raise GovernanceContractError("remote closeout predates its remote action")
    if not str(payload["remote_action_id"]).strip():
        raise GovernanceContractError("remote closeout action ID is required")
    list_fields = (
        "slurm_jobs_and_allocations",
        "interactive_sessions",
        "tmux_and_screen_sessions",
        "relevant_processes",
        "transfers_and_monitors",
        "unsynchronized_artifacts",
        "retained_resources_with_reason",
        "cancellation_or_release_actions",
    )
    for field in list_fields:
        _require_list(payload, field)
    state = payload["final_billing_state"]
    if state not in {"verified_nonbilling", "verified_other_work_only", "unverified"}:
        raise GovernanceContractError("remote final billing state is invalid")
    if require_verified and state == "unverified":
        raise GovernanceContractError("remote step cannot pass with unverified billing state")
    if payload["unsynchronized_artifacts"] and state != "unverified":
        raise GovernanceContractError("billing cannot be verified with unsynchronized artifacts")
    if state == "verified_nonbilling":
        active_fields = (
            "slurm_jobs_and_allocations",
            "interactive_sessions",
            "tmux_and_screen_sessions",
            "relevant_processes",
            "transfers_and_monitors",
            "retained_resources_with_reason",
        )
        if any(payload[field] for field in active_fields):
            raise GovernanceContractError("nonbilling state still lists active resources")
    retained = payload["retained_resources_with_reason"]
    retained_names = []
    for record in retained:
        if (
            not isinstance(record, dict)
            or not str(record.get("resource", "")).strip()
            or not str(record.get("reason", "")).strip()
        ):
            raise GovernanceContractError("retained resource lacks an explicit reason")
        retained_names.append(str(record["resource"]))
    if len(set(retained_names)) != len(retained_names):
        raise GovernanceContractError("retained resources must be unique")
    if state == "verified_other_work_only":
        active_fields = (
            "slurm_jobs_and_allocations",
            "interactive_sessions",
            "tmux_and_screen_sessions",
            "relevant_processes",
            "transfers_and_monitors",
        )
        if not any(payload[field] for field in active_fields) or not retained:
            raise GovernanceContractError(
                "other-work state requires active resources and retained reasons"
            )
        active_job_ids = {
            str(value).split("|", 1)[0]
            for value in payload["slurm_jobs_and_allocations"]
        }
        if not active_job_ids.issubset(retained_names):
            raise GovernanceContractError("active Slurm jobs lack retained-resource reasons")
        for field, resource_name in (
            ("interactive_sessions", "interactive_sessions"),
            ("tmux_and_screen_sessions", "tmux_and_screen_sessions"),
            ("relevant_processes", "relevant_processes"),
            ("transfers_and_monitors", "transfers_and_monitors"),
        ):
            if payload[field] and resource_name not in retained_names:
                raise GovernanceContractError(f"{field} lacks a retained-resource reason")


def validate_remote_closeout_index(payload: Mapping[str, Any]) -> None:
    required = {
        "reconciled_at",
        "remote_command_ledger_sha256",
        "remote_action_ids",
        "remote_action_timestamps",
        "closeout_receipts",
        "action_to_receipt",
        "latest_remote_activity_at",
        "latest_closeout_at",
        "uncovered_remote_actions",
        "final_billing_state",
    }
    if not required.issubset(payload):
        raise GovernanceContractError("remote closeout index is incomplete")
    require_timestamp(payload["reconciled_at"], "reconciled_at")
    require_sha256(payload["remote_command_ledger_sha256"], "remote_command_ledger_sha256")
    actions = _require_list(payload, "remote_action_ids")
    if len(set(actions)) != len(actions) or any(
        not isinstance(value, str) or not value.strip() for value in actions
    ):
        raise GovernanceContractError("remote closeout action IDs must be unique and nonempty")
    receipts = _require_list(payload, "closeout_receipts")
    uncovered = _require_list(payload, "uncovered_remote_actions")
    mapping = payload["action_to_receipt"]
    if not isinstance(mapping, dict) or set(mapping) != set(actions):
        raise GovernanceContractError("remote closeout mapping does not cover every action")
    action_times = payload["remote_action_timestamps"]
    if not isinstance(action_times, dict) or set(action_times) != set(actions):
        raise GovernanceContractError("remote closeout timestamps do not cover every action")
    parsed_action_times = {
        str(action): _parse_time(value, f"remote_action_timestamps.{action}")
        for action, value in action_times.items()
    }
    if uncovered:
        raise GovernanceContractError("remote closeout index has uncovered actions")
    receipt_by_path = {}
    receipt_fields = {
        "path",
        "sha256",
        "remote_action_id",
        "closed_at",
        "final_billing_state",
    }
    for receipt in receipts:
        if not isinstance(receipt, dict) or not receipt_fields.issubset(receipt):
            raise GovernanceContractError("remote closeout receipt digest is invalid")
        path = str(receipt["path"])
        if not path.strip() or path in receipt_by_path:
            raise GovernanceContractError("remote closeout receipt paths must be unique")
        require_sha256(receipt["sha256"], "closeout_receipts.sha256")
        if receipt["remote_action_id"] not in parsed_action_times:
            raise GovernanceContractError("closeout receipt references an unknown action")
        if receipt["final_billing_state"] not in {
            "verified_nonbilling",
            "verified_other_work_only",
        }:
            raise GovernanceContractError("indexed closeout receipt is not billing-verified")
        receipt_by_path[path] = {
            **receipt,
            "parsed_closed_at": _parse_time(receipt["closed_at"], "closeout_receipts.closed_at"),
        }
    for action in actions:
        linked = mapping[action]
        if (
            not isinstance(linked, list)
            or not linked
            or len(set(str(value) for value in linked)) != len(linked)
        ):
            raise GovernanceContractError("each remote action requires one or more closeouts")
        for path in linked:
            receipt = receipt_by_path.get(str(path))
            if receipt is None or receipt["remote_action_id"] != action:
                raise GovernanceContractError("remote action maps to the wrong closeout receipt")
            if receipt["parsed_closed_at"] <= parsed_action_times[str(action)]:
                raise GovernanceContractError("remote action closeout predates the action")
    linked_paths = {
        str(path) for linked in mapping.values() for path in linked
    }
    if linked_paths != set(receipt_by_path):
        raise GovernanceContractError("remote closeout index has unlinked receipts")
    activity = _parse_time(payload["latest_remote_activity_at"], "latest_remote_activity_at")
    closeout = _parse_time(payload["latest_closeout_at"], "latest_closeout_at")
    reconciled = _parse_time(payload["reconciled_at"], "reconciled_at")
    if actions and activity != max(parsed_action_times.values()):
        raise GovernanceContractError("latest remote activity timestamp is inconsistent")
    if receipts and closeout != max(
        receipt["parsed_closed_at"] for receipt in receipt_by_path.values()
    ):
        raise GovernanceContractError("latest closeout timestamp is inconsistent")
    if closeout <= activity:
        raise GovernanceContractError("latest closeout predates remote activity")
    if reconciled < closeout:
        raise GovernanceContractError("remote closeout index was reconciled too early")
    if payload["final_billing_state"] not in {
        "verified_nonbilling",
        "verified_other_work_only",
    }:
        raise GovernanceContractError("remote closeout index is not billing-verified")
