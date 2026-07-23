from __future__ import annotations

import re
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .immutable import GovernanceContractError, require_sha256, require_timestamp


MODULE_IDS = (
    "visible_inference",
    "thermal_inference",
    "sensor_inference",
    "graph_fusion",
)
EXPECTED_FAMILY_BY_MODULE = {
    "visible_inference": "V2-A-10-MULTI",
    "thermal_inference": "T1-THERM",
    "sensor_inference": "S1-GRU",
    "graph_fusion": "E3-FULL",
}
EXPECTED_GATE_BY_MODULE = {
    "visible_inference": "G3",
    "thermal_inference": "G2",
    "sensor_inference": "G2",
    "graph_fusion": "G5",
}
SCOPE_REQUIRED_COLUMNS = {
    "module_id",
    "family_id",
    "eligibility_status",
    "source_gate_id",
    "source_gate_sha256",
    "checkpoint_sha256",
    "exclusion_reason",
    "protocol_sha256",
}
TRACE_REQUIRED_COLUMNS = {
    "measurement_id",
    "module_id",
    "variant_id",
    "model_family_id",
    "representative_seed",
    "checkpoint_sha256",
    "policy_sha256",
    "environment_ready_sha256",
    "protocol_sha256",
    "slurm_job_id",
    "allocation_id",
    "compute_node",
    "repeat_id",
    "phase",
    "iteration_index",
    "duration_ns",
    "device_kind",
    "device_identifier",
    "gpu_uuid",
    "gpu_clock_mhz",
    "cuda_driver",
    "cpu_affinity",
    "precision",
    "batch_size",
    "input_shape",
    "cuda_synchronized_before_and_after",
    "concurrent_user_gpu_job_count",
    "interference_audit_status",
    "captured_at",
}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if bool(pd.isna(value)):
            return True
    except (TypeError, ValueError):
        pass
    return not str(value).strip()


def validate_efficiency_scope(
    scope: pd.DataFrame,
    *,
    graph_primary_population_nonempty: bool,
    gate_receipts: Mapping[str, Mapping[str, Any]],
) -> None:
    missing = sorted(SCOPE_REQUIRED_COLUMNS.difference(scope.columns))
    if missing:
        raise GovernanceContractError(f"efficiency scope columns are missing: {missing}")
    if len(scope) != len(MODULE_IDS) or scope["module_id"].duplicated().any():
        raise GovernanceContractError("efficiency scope requires one row per module")
    if set(scope["module_id"]) != set(MODULE_IDS):
        raise GovernanceContractError("efficiency scope module set drifted")
    expected_gate_ids = set(EXPECTED_GATE_BY_MODULE.values())
    if not expected_gate_ids.issubset(gate_receipts):
        raise GovernanceContractError("efficiency scope lacks source gate receipts")
    for gate_id in expected_gate_ids:
        receipt = gate_receipts[gate_id]
        if not isinstance(receipt, Mapping) or receipt.get("status") not in {
            "pass",
            "fail",
            "blocked",
        }:
            raise GovernanceContractError("efficiency source gate receipt is invalid")
        require_sha256(receipt.get("sha256"), f"gate_receipts.{gate_id}.sha256")
    if scope["protocol_sha256"].nunique(dropna=False) != 1:
        raise GovernanceContractError("efficiency scope must share one protocol hash")
    for row in scope.to_dict("records"):
        if row["eligibility_status"] not in {"measured", "not_applicable"}:
            raise GovernanceContractError("efficiency eligibility status is invalid")
        module_id = str(row["module_id"])
        if row["family_id"] != EXPECTED_FAMILY_BY_MODULE[module_id]:
            raise GovernanceContractError("efficiency scope substituted a model family")
        if row["source_gate_id"] != EXPECTED_GATE_BY_MODULE[module_id]:
            raise GovernanceContractError("efficiency scope source gate drifted")
        require_sha256(row["source_gate_sha256"], "source_gate_sha256")
        gate_receipt = gate_receipts[str(row["source_gate_id"])]
        if row["source_gate_sha256"] != gate_receipt["sha256"]:
            raise GovernanceContractError("efficiency scope source gate hash drifted")
        require_sha256(row["protocol_sha256"], "protocol_sha256")
        if row["eligibility_status"] == "measured":
            if gate_receipt["status"] != "pass":
                raise GovernanceContractError("measured module lacks a passing source gate")
            require_sha256(row["checkpoint_sha256"], "checkpoint_sha256")
            if not _is_missing(row["exclusion_reason"]):
                raise GovernanceContractError("measured module cannot have an exclusion reason")
        else:
            if not _is_missing(row["checkpoint_sha256"]):
                raise GovernanceContractError("not-applicable module cannot substitute a checkpoint")
            if _is_missing(row["exclusion_reason"]):
                raise GovernanceContractError("not-applicable module requires a reason")
        if row["module_id"] == "graph_fusion" and row["eligibility_status"] == "measured":
            if not graph_primary_population_nonempty:
                raise GovernanceContractError("graph efficiency measurement lacks G5 eligibility")


def validate_efficiency_trace(trace: pd.DataFrame, scope: pd.DataFrame) -> None:
    missing = sorted(TRACE_REQUIRED_COLUMNS.difference(trace.columns))
    if missing:
        raise GovernanceContractError(f"efficiency trace columns are missing: {missing}")
    measured = set(
        scope.loc[scope["eligibility_status"] == "measured", "module_id"].astype(str)
    )
    not_applicable = set(MODULE_IDS).difference(measured)
    if set(trace["module_id"]) != measured:
        raise GovernanceContractError("efficiency trace does not match measured scope")
    if set(trace["module_id"]).intersection(not_applicable):
        raise GovernanceContractError("not-applicable module has measured trace rows")
    primary_key = [
        "measurement_id",
        "module_id",
        "variant_id",
        "repeat_id",
        "phase",
        "iteration_index",
    ]
    if trace.duplicated(primary_key).any():
        raise GovernanceContractError("efficiency trace primary key is not unique")
    if trace.empty:
        if measured:
            raise GovernanceContractError("measured efficiency scope has no trace")
        return
    for module_id, group in trace.groupby("module_id", sort=True):
        if group["measurement_id"].nunique() != 1 or group["variant_id"].nunique() != 1:
            raise GovernanceContractError(
                f"efficiency module {module_id} must have one preregistered variant"
            )
    scope_by_module = scope.set_index("module_id", drop=False)
    if not set(trace["phase"]).issubset({"warmup", "timed"}):
        raise GovernanceContractError("efficiency trace phase is invalid")
    numeric_positive = ("representative_seed", "repeat_id", "iteration_index", "duration_ns")
    for column in numeric_positive:
        values = pd.to_numeric(trace[column], errors="raise")
        if not np.isfinite(values).all() or (values <= 0).any():
            raise GovernanceContractError(f"efficiency {column} must be positive")
    if (pd.to_numeric(trace["batch_size"], errors="raise") != 1).any():
        raise GovernanceContractError("efficiency batch size must be one")
    if (pd.to_numeric(trace["concurrent_user_gpu_job_count"], errors="raise") < 0).any():
        raise GovernanceContractError("concurrent job counts must be nonnegative")
    for value in trace["captured_at"]:
        require_timestamp(value, "captured_at")
    for column in ("environment_ready_sha256", "protocol_sha256"):
        for value in trace[column].unique():
            require_sha256(value, column)
    for column in ("checkpoint_sha256", "policy_sha256"):
        for value in trace[column].dropna().unique():
            if str(value).strip():
                require_sha256(value, column)
    for column in ("measurement_id", "variant_id", "model_family_id", "slurm_job_id", "allocation_id", "compute_node", "device_identifier", "cpu_affinity", "precision", "input_shape"):
        if trace[column].map(lambda value: not str(value).strip()).any():
            raise GovernanceContractError(f"efficiency {column} contains empty values")
    if trace["slurm_job_id"].map(
        lambda value: re.fullmatch(r"\d+(?:_\d+)?", str(value)) is None
    ).any():
        raise GovernanceContractError("efficiency trace contains an invalid Slurm job ID")
    nonmissing_clocks = pd.to_numeric(trace["gpu_clock_mhz"].dropna(), errors="raise")
    if (nonmissing_clocks < 0).any():
        raise GovernanceContractError("efficiency GPU clock must be nonnegative")

    timed = trace.loc[trace["phase"] == "timed"]
    if (timed["concurrent_user_gpu_job_count"] != 0).any() or not (
        timed["interference_audit_status"] == "pass"
    ).all():
        raise GovernanceContractError("timed efficiency trace has concurrent interference")
    if not set(trace["interference_audit_status"]).issubset({"pass", "fail"}):
        raise GovernanceContractError("interference audit status is invalid")
    cuda_rows = trace.loc[trace["device_kind"] == "cuda"]
    if not set(trace["device_kind"]).issubset({"cpu", "cuda"}):
        raise GovernanceContractError("efficiency device kind is invalid")
    if not cuda_rows.empty:
        if cuda_rows["gpu_uuid"].map(_is_missing).any() or cuda_rows["cuda_driver"].map(_is_missing).any():
            raise GovernanceContractError("CUDA trace requires GPU UUID and driver")
        if not cuda_rows["cuda_synchronized_before_and_after"].map(
            lambda value: value is True or isinstance(value, np.bool_) and bool(value)
        ).all():
            raise GovernanceContractError("CUDA timing requires synchronization")

    group_columns = ["measurement_id", "module_id", "variant_id"]
    for _, group in trace.groupby(group_columns, sort=True):
        module_id = str(group["module_id"].iloc[0])
        scope_row = scope_by_module.loc[module_id]
        if set(group["model_family_id"]) != {str(scope_row["family_id"])}:
            raise GovernanceContractError("efficiency trace substituted a model family")
        if set(group["checkpoint_sha256"]) != {str(scope_row["checkpoint_sha256"])}:
            raise GovernanceContractError("efficiency trace substituted a checkpoint")
        if set(group["protocol_sha256"]) != {str(scope_row["protocol_sha256"])}:
            raise GovernanceContractError("efficiency trace protocol hash drifted from scope")
        if set(group["repeat_id"].astype(int)) != {1}:
            raise GovernanceContractError("each efficiency module variant requires repeat 1")
        for phase, expected_count in (("warmup", 50), ("timed", 200)):
            phase_rows = group.loc[group["phase"] == phase]
            if len(phase_rows) != expected_count:
                raise GovernanceContractError(
                    f"efficiency {phase} row count must be {expected_count}"
                )
            if set(phase_rows["iteration_index"].astype(int)) != set(
                range(1, expected_count + 1)
            ):
                raise GovernanceContractError("efficiency iteration index set drifted")
        for column in (
            "model_family_id",
            "representative_seed",
            "checkpoint_sha256",
            "policy_sha256",
            "environment_ready_sha256",
            "protocol_sha256",
            "slurm_job_id",
            "allocation_id",
            "compute_node",
            "device_kind",
            "device_identifier",
            "cpu_affinity",
            "precision",
            "batch_size",
            "input_shape",
        ):
            if group[column].nunique(dropna=False) != 1:
                raise GovernanceContractError(f"efficiency {column} drifted within a module")
    for column in ("allocation_id", "environment_ready_sha256", "protocol_sha256", "precision", "batch_size"):
        if trace[column].nunique(dropna=False) != 1:
            raise GovernanceContractError(f"compared modules do not share {column}")


def derive_efficiency_summary(
    trace: pd.DataFrame,
    *,
    trace_sha256: str,
    derivation_code_sha256: str,
    peak_memory_bytes: Mapping[str, int],
    parameter_count: Mapping[str, int],
    checkpoint_bytes: Mapping[str, int],
) -> pd.DataFrame:
    require_sha256(trace_sha256, "trace_sha256")
    require_sha256(derivation_code_sha256, "derivation_code_sha256")
    rows = []
    for keys, group in trace.groupby(["measurement_id", "module_id", "variant_id"], sort=True):
        measurement_id, module_id, variant_id = keys
        timed = group.loc[group["phase"] == "timed", "duration_ns"].to_numpy(dtype=np.int64)
        if len(timed) != 200:
            raise GovernanceContractError("summary requires exactly 200 timed rows")
        for mapping_name, mapping in {
            "peak_memory_bytes": peak_memory_bytes,
            "parameter_count": parameter_count,
            "checkpoint_bytes": checkpoint_bytes,
        }.items():
            if module_id not in mapping or int(mapping[module_id]) < 0:
                raise GovernanceContractError(f"{mapping_name} is missing a measured module")
        rows.append(
            {
                "measurement_id": measurement_id,
                "module_id": module_id,
                "variant_id": variant_id,
                "trace_sha256": trace_sha256,
                "repeat_count": 1,
                "timed_iteration_count_per_repeat": 200,
                "latency_p50_ns": int(np.quantile(timed, 0.50)),
                "latency_p95_ns": int(np.quantile(timed, 0.95)),
                "throughput_per_second": float(1.0e9 / np.mean(timed)),
                "peak_memory_bytes": int(peak_memory_bytes[module_id]),
                "parameter_count": int(parameter_count[module_id]),
                "checkpoint_bytes": int(checkpoint_bytes[module_id]),
                "environment_ready_sha256": str(group["environment_ready_sha256"].iloc[0]),
                "derivation_code_sha256": derivation_code_sha256,
            }
        )
    return pd.DataFrame(rows)


def validate_efficiency_completion(
    scope: pd.DataFrame, trace: pd.DataFrame, summary: pd.DataFrame
) -> None:
    required_summary = {
        "measurement_id",
        "module_id",
        "variant_id",
        "trace_sha256",
        "repeat_count",
        "timed_iteration_count_per_repeat",
        "latency_p50_ns",
        "latency_p95_ns",
        "throughput_per_second",
        "peak_memory_bytes",
        "parameter_count",
        "checkpoint_bytes",
        "environment_ready_sha256",
        "derivation_code_sha256",
    }
    missing = sorted(required_summary.difference(summary.columns))
    if missing:
        raise GovernanceContractError(f"efficiency summary columns are missing: {missing}")
    measured = set(scope.loc[scope["eligibility_status"] == "measured", "module_id"])
    if set(summary.get("module_id", pd.Series(dtype=str))) != measured:
        raise GovernanceContractError("efficiency summary does not close measured scope")
    expected = trace.loc[trace["phase"] == "timed"].groupby(
        ["measurement_id", "module_id", "variant_id"]
    ).size()
    summary_keys = set(
        tuple(value)
        for value in summary[["measurement_id", "module_id", "variant_id"]].itertuples(
            index=False, name=None
        )
    )
    if summary_keys != set(expected.index) or not (expected == 200).all():
        raise GovernanceContractError("efficiency summary keys or timed counts drifted")
    if summary.duplicated(["measurement_id", "module_id", "variant_id"]).any():
        raise GovernanceContractError("efficiency summary primary key is not unique")
    for field in ("trace_sha256", "derivation_code_sha256", "environment_ready_sha256"):
        if summary[field].nunique(dropna=False) != 1:
            raise GovernanceContractError(f"efficiency summary {field} drifted across modules")
    summary_by_key = summary.set_index(["measurement_id", "module_id", "variant_id"])
    for key, group in trace.groupby(["measurement_id", "module_id", "variant_id"], sort=True):
        row = summary_by_key.loc[key]
        timed_values = group.loc[group["phase"] == "timed", "duration_ns"].to_numpy(
            dtype=np.int64
        )
        if int(row["repeat_count"]) != 1 or int(row["timed_iteration_count_per_repeat"]) != 200:
            raise GovernanceContractError("efficiency summary repeat contract drifted")
        if int(row["latency_p50_ns"]) != int(np.quantile(timed_values, 0.50)):
            raise GovernanceContractError("efficiency p50 is not derived from timed trace")
        if int(row["latency_p95_ns"]) != int(np.quantile(timed_values, 0.95)):
            raise GovernanceContractError("efficiency p95 is not derived from timed trace")
        expected_throughput = float(1.0e9 / np.mean(timed_values))
        if not np.isclose(float(row["throughput_per_second"]), expected_throughput):
            raise GovernanceContractError("efficiency throughput is not trace-derived")
        if row["environment_ready_sha256"] != group["environment_ready_sha256"].iloc[0]:
            raise GovernanceContractError("efficiency summary environment hash drifted")
        for field in ("trace_sha256", "derivation_code_sha256"):
            require_sha256(row[field], field)
        for field in ("peak_memory_bytes", "parameter_count", "checkpoint_bytes"):
            if int(row[field]) < 0:
                raise GovernanceContractError(f"efficiency {field} must be nonnegative")
