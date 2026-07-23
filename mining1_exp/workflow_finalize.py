from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from .data.episodes import validate_fusion_eligibility_lock
from .governance.immutable import write_once_bytes
from .governance.remote import validate_remote_closeout
from .governance.slurm_contracts import (
    DAMAGED_REMOTE_HOST,
    DAMAGED_REMOTE_LOGIN,
    DAMAGED_REMOTE_USER_PREFIX,
)
from .provenance import canonical_json_bytes, canonical_json_sha256, sha256_file
from .workflow_common import (
    describe_output,
    hash_existing_inputs,
    load_json,
    load_plan,
    utc_now,
    validate_output_digests,
    write_json_artifact,
    write_parquet_artifact,
)


PLACEHOLDER_PATTERN = re.compile(r"【示意占位：([^】]+)】")
CLAIM_IDS = (
    "C-DATA",
    "C-TRANSFER",
    "C-THERMAL",
    "C-METHANE",
    "C-ROBUST",
    "C-FUSION",
    "C-GRAPH",
    "C-MEMORY",
    "C-EFFICIENCY",
)


def _protocol(root: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(
        (root / "configs/protocol_lock.pretest.yaml").read_text(encoding="utf-8-sig")
    )
    if not isinstance(payload, dict):
        raise RuntimeError("PRETEST protocol must be a mapping")
    return payload


def audit_artifacts(
    root: Path, row: Mapping[str, str], arguments: Mapping[str, Any]
) -> Dict[str, Any]:
    del arguments
    plan = load_plan(root)
    plan_by_step = {step["step_id"]: step for step in plan}
    checked_receipts = []
    failures = []
    for step in plan:
        if step["status"] not in {"pass", "not_applicable", "accepted_not_applicable"}:
            continue
        receipt_path = root / "evidence/command_receipts" / f"{step['step_id']}.json"
        if receipt_path.is_file():
            receipt = load_json(receipt_path)
            try:
                validate_output_digests(root, receipt.get("outputs", []))
            except Exception as exc:
                failures.append(f"{step['step_id']}: {exc}")
            checked_receipts.append(
                {
                    "step_id": step["step_id"],
                    "path": receipt_path.relative_to(root).as_posix(),
                    "sha256": sha256_file(receipt_path),
                }
            )
    def step_status(step_id: str) -> str:
        return plan_by_step.get(step_id, {}).get("status", "")

    def step_review_paths(step_id: str) -> list[str]:
        review_path = root / "evidence/step_reviews" / f"{step_id}.json"
        if not review_path.is_file():
            failures.append(f"missing step review for not_applicable boundary: {step_id}")
            return []
        review = load_json(review_path)
        if review.get("status") not in {"not_applicable", "accepted_not_applicable"}:
            failures.append(f"invalid not_applicable step review: {step_id}")
            return [review_path.relative_to(root).as_posix()]
        paths = [review_path.relative_to(root).as_posix()]
        for item in review.get("output_artifacts", []):
            relative = str(item.get("path", "")).strip()
            if relative:
                paths.append(relative)
        return paths

    required = [
        "evidence/gates/G4.json",
        "evidence/gates/G5.json",
        "configs/protocol_lock.pretest.yaml",
        "data/releases/branch_test_release.json",
        "results/S1/metrics.parquet",
    ]
    package_boundaries = {
        "T1": ("E112", "results/T1/metrics.parquet"),
        "V2": ("E207", "results/V2/metrics.parquet"),
        "R1": ("E223", "results/R1/metrics.parquet"),
    }
    for _, (step_id, metrics_path) in package_boundaries.items():
        status = step_status(step_id)
        if status == "pass":
            required.append(metrics_path)
        elif status in {"not_applicable", "accepted_not_applicable"}:
            required.extend(step_review_paths(step_id))
        else:
            failures.append(f"package boundary step is not complete: {step_id}={status}")
    if step_status("E310") in {"not_applicable", "accepted_not_applicable"}:
        fusion_eligible = False
        required.extend(step_review_paths("E310"))
    else:
        eligibility = validate_fusion_eligibility_lock(
            pd.read_parquet(root / "data/locked/fusion_eligibility_lock.parquet")
        )
        fusion_eligible = bool(
            eligibility["fusion_primary_eligible"].astype(bool).any()
        )
        if fusion_eligible:
            required.extend(
                [
                    "data/releases/episode_test_release.json",
                    "results/E2_E3/metrics.parquet",
                    "results/E2_E3/shortcut_audits.json",
                ]
            )
        else:
            required.append("evidence/episode/E310_not_applicable.json")
    missing = [relative for relative in required if not (root / relative).exists()]
    if missing:
        failures.append(f"missing required artifacts: {missing}")
    forbidden_host_hits = []
    for scope in ("mining1_exp", "slurm", "configs", "plans"):
        for path in (root / scope).rglob("*"):
            if path.is_file() and path.suffix.lower() in {
                ".py",
                ".ps1",
                ".sh",
                ".sbatch",
                ".json",
                ".yaml",
                ".yml",
                ".csv",
            }:
                text = path.read_text(encoding="utf-8-sig", errors="ignore")
                if DAMAGED_REMOTE_LOGIN in text and "forbidden_host" not in text:
                    forbidden_host_hits.append(path.relative_to(root).as_posix())
    if forbidden_host_hits:
        failures.append(f"forbidden host residue: {forbidden_host_hits}")
    payload = {
        "schema_version": 1,
        "step_id": row["step_id"],
        "status": "pass" if not failures else "fail",
        "performance_claim_authorized": False,
        "checked_command_receipts": checked_receipts,
        "required_artifacts": [
            describe_output(root, relative) for relative in required if (root / relative).exists()
        ],
        "fusion_eligible": fusion_eligible,
        "failures": failures,
        "created_at": utc_now(),
    }
    output = "results/C1/artifact_qa.json"
    write_json_artifact(root / output, payload)
    if failures:
        raise RuntimeError("Artifact QA failed: " + "; ".join(failures))
    return {
        "status": "pass",
        "output_paths": [output],
        "inputs": hash_existing_inputs(root, required),
        "details": {"checked_receipt_count": len(checked_receipts)},
    }


def aggregate_results(
    root: Path, row: Mapping[str, str], arguments: Mapping[str, Any]
) -> Dict[str, Any]:
    del arguments
    try:
        plan_by_step = {step["step_id"]: step for step in load_plan(root)}
    except Exception:
        plan_by_step = {}

    def step_status(step_id: str) -> str:
        return plan_by_step.get(step_id, {}).get("status", "")

    boundary_statuses = {"not_applicable", "accepted_not_applicable"}
    visual_branch_not_applicable = step_status("E207") in boundary_statuses
    robustness_branch_not_applicable = step_status("E223") in boundary_statuses
    paths = [
        root / f"results/{package}/metrics.parquet"
        for package in ("T1", "S1", "V2", "R1", "E2_E3")
        if (root / f"results/{package}/metrics.parquet").is_file()
    ]
    frames = []
    for path in paths:
        frame = pd.read_parquet(path).copy()
        required = {
            "package_id",
            "family_id",
            "run_id",
            "repeat_key",
            "group_id",
            "result_level",
            "metric_name",
            "metric_value",
            "population_hash",
            "prediction_lock_hash",
        }
        if not required.issubset(frame.columns) or frame.empty:
            raise RuntimeError(f"Metric artifact is incomplete: {path}")
        frame["source_artifact"] = path.relative_to(root).as_posix()
        frame["source_artifact_sha256"] = sha256_file(path)
        frames.append(frame)
    if not frames:
        raise RuntimeError("No package metrics are available for aggregation")
    aggregate = pd.concat(frames, ignore_index=True, sort=False)
    primary_key = [
        "package_id",
        "family_id",
        "run_id",
        "repeat_key",
        "group_id",
        "result_level",
        "metric_name",
    ]
    if aggregate.duplicated(primary_key).any():
        raise RuntimeError("Aggregate result primary key is not unique")
    if not np.isfinite(pd.to_numeric(aggregate["metric_value"], errors="raise")).all():
        raise RuntimeError("Aggregate results contain non-finite metric values")
    output = "results/final/aggregate.parquet"
    aggregate_output_path = root / output
    aggregate_write_mode = "created"
    if aggregate_output_path.is_file():
        existing = pd.read_parquet(aggregate_output_path)
        try:
            pd.testing.assert_frame_equal(
                existing.sort_index(axis=1)
                .sort_values(primary_key)
                .reset_index(drop=True),
                aggregate.sort_index(axis=1)
                .sort_values(primary_key)
                .reset_index(drop=True),
                check_like=True,
            )
        except AssertionError as exc:
            raise RuntimeError(
                "Existing aggregate.parquet does not match current metric inputs"
            ) from exc
        aggregate = existing
        aggregate_write_mode = "existing_verified"
    else:
        write_parquet_artifact(aggregate_output_path, aggregate)
    v2_families = {
        "V2-A-10-SCR",
        "V2-A-10-GEN",
        "V2-A-10-SINGLE",
        "V2-A-10-MULTI",
    }
    panel_a = aggregate.loc[
        (aggregate["package_id"] == "V2")
        & (aggregate["family_id"].isin(v2_families))
        & (aggregate["result_level"] == "repeat")
        & (aggregate["metric_name"] == "map50_95")
    ].copy()
    panel_b = aggregate.loc[
        (aggregate["package_id"] == "R1")
        & (aggregate["family_id"].isin({"R1-GEN", "R1-MULTI"}))
        & (aggregate["result_level"] == "corruption_cell")
        & (aggregate["metric_name"] == "relative_drop_percent")
    ].copy()
    panel_status = {
        "A_transfer": "not_applicable" if visual_branch_not_applicable else "measured",
        "B_robustness": "not_applicable"
        if robustness_branch_not_applicable
        else "measured",
    }
    if not visual_branch_not_applicable and set(panel_a["family_id"].astype(str)) != v2_families:
        raise RuntimeError("Figure 4 Panel A lacks a locked V2 family")
    expected_cells = {
        f"{kind}:{severity}"
        for kind in ("low_light", "dust_fog_proxy")
        for severity in (1, 2, 3)
    }
    if not robustness_branch_not_applicable:
        for family, frame in panel_b.groupby("family_id", sort=True):
            if set(frame["group_id"].astype(str)) != expected_cells:
                raise RuntimeError(f"Figure 4 Panel B is incomplete for {family}")
    if (
        not robustness_branch_not_applicable
        and set(panel_b["family_id"].astype(str)) != {"R1-GEN", "R1-MULTI"}
    ):
        raise RuntimeError("Figure 4 Panel B lacks a locked R1 family")
    if visual_branch_not_applicable:
        panel_a = aggregate.iloc[0:0].copy()
    else:
        panel_a["figure_panel"] = "A_transfer"
    if robustness_branch_not_applicable:
        panel_b = aggregate.iloc[0:0].copy()
    else:
        panel_b["figure_panel"] = "B_robustness"
    fig4_data = pd.concat([panel_a, panel_b], ignore_index=True, sort=False)
    if "figure_panel" not in fig4_data.columns:
        fig4_data["figure_panel"] = pd.Series(dtype="object")
    fig4_output = "results/final/fig4_data.parquet"
    write_parquet_artifact(root / fig4_output, fig4_data)
    return {
        "status": "pass",
        "output_paths": [output, fig4_output],
        "inputs": hash_existing_inputs(
            root, [path.relative_to(root).as_posix() for path in paths]
        ),
        "details": {
            "metric_row_count": len(aggregate),
            "package_count": int(aggregate["package_id"].nunique()),
            "fig4_source_row_count": len(fig4_data),
            "aggregate_write_mode": aggregate_write_mode,
            "fig4_panel_status": panel_status,
            "v4_not_applicable_boundaries_preserved": {
                "E207": visual_branch_not_applicable,
                "E223": robustness_branch_not_applicable,
            },
        },
    }


def _paired_table(
    aggregate: pd.DataFrame,
    *,
    left_family: str,
    right_family: str,
    metric_name: str,
    result_level: str,
    allow_static_right_family: bool = False,
) -> pd.DataFrame:
    subset = aggregate.loc[
        aggregate["family_id"].isin([left_family, right_family])
        & (aggregate["metric_name"] == metric_name)
        & (aggregate["result_level"] == result_level),
        ["family_id", "repeat_key", "group_id", "metric_value"],
    ].copy()
    table = subset.pivot(
        index=["repeat_key", "group_id"],
        columns="family_id",
        values="metric_value",
    ).dropna()
    if not table.empty and left_family in table and right_family in table:
        return table.reset_index()
    if allow_static_right_family:
        left = subset.loc[subset["family_id"] == left_family].copy()
        right = subset.loc[subset["family_id"] == right_family].copy()
        if (
            not left.empty
            and not right.empty
            and not left.duplicated(["repeat_key", "group_id"]).any()
            and not right.duplicated(["group_id"]).any()
            and right.groupby("group_id")["metric_value"].nunique(dropna=False).max() == 1
        ):
            right_values = right[["group_id", "metric_value"]].rename(
                columns={"metric_value": right_family}
            )
            merged = left[["repeat_key", "group_id", "metric_value"]].merge(
                right_values, on="group_id", how="inner"
            )
            if not merged.empty and len(merged) == len(left):
                return merged.rename(columns={"metric_value": left_family})
    if table.empty or left_family not in table or right_family not in table:
        raise RuntimeError(
            f"Paired contrast lacks aligned units: {left_family} vs {right_family}"
        )
    return table.reset_index()


def _paired_bootstrap(
    table: pd.DataFrame,
    *,
    left_family: str,
    right_family: str,
    direction: str,
    draws: int,
    seed: int,
) -> Dict[str, Any]:
    sign = 1.0 if direction == "higher" else -1.0
    table = table.copy()
    table["effect"] = sign * (
        table[left_family].astype(float) - table[right_family].astype(float)
    )
    repeat_effects = table.groupby("repeat_key")["effect"].mean()
    point = float(repeat_effects.mean())
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    groups = [group for _, group in table.groupby("repeat_key", sort=True)]
    for draw in range(draws):
        cell_effects = []
        for group in groups:
            values = group["effect"].to_numpy(dtype=np.float64)
            indices = rng.integers(0, len(values), size=len(values))
            cell_effects.append(float(np.mean(values[indices])))
        samples[draw] = float(np.mean(cell_effects))
    lower, upper = np.quantile(samples, [0.025, 0.975])
    probability_nonpositive = float(np.mean(samples <= 0.0))
    probability_nonnegative = float(np.mean(samples >= 0.0))
    p_value = min(1.0, 2.0 * min(probability_nonpositive, probability_nonnegative))
    return {
        "effect": point,
        "ci95": [float(lower), float(upper)],
        "raw_p_value": p_value,
        "repeat_count": len(groups),
        "paired_unit_count": len(table),
        "bootstrap_draws": draws,
        "bootstrap_draws_hash": canonical_json_sha256(samples.round(12).tolist()),
    }


def _holm_adjust(p_values: Mapping[str, float]) -> Dict[str, float]:
    ordered = sorted(p_values, key=lambda key: (p_values[key], key))
    adjusted: Dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, key in enumerate(ordered):
        value = min(1.0, (count - rank) * float(p_values[key]))
        running = max(running, value)
        adjusted[key] = running
    return adjusted


def _contrast_status(
    result: Mapping[str, Any],
    *,
    minimum_effect: float,
    p_value: float,
) -> str:
    lower, upper = result["ci95"]
    if upper < 0:
        return "negative"
    if lower > 0 and result["effect"] >= minimum_effect and p_value <= 0.05:
        return "support"
    return "inconclusive"


def run_confirmatory_stats(
    root: Path, row: Mapping[str, str], arguments: Mapping[str, Any]
) -> Dict[str, Any]:
    del arguments
    try:
        plan_by_step = {step["step_id"]: step for step in load_plan(root)}
    except Exception:
        plan_by_step = {}

    def step_status(step_id: str) -> str:
        return plan_by_step.get(step_id, {}).get("status", "")

    boundary_statuses = {"not_applicable", "accepted_not_applicable"}
    contrast_boundaries = {
        "V2": (
            step_status("E207") in boundary_statuses,
            "amendment v4 excludes the visual transfer evaluation branch",
        ),
        "R1": (
            step_status("E223") in boundary_statuses,
            "amendment v4 excludes the robustness evaluation branch",
        ),
    }
    aggregate_path = root / "results/final/aggregate.parquet"
    aggregate = pd.read_parquet(aggregate_path)
    protocol = _protocol(root)
    selection = load_json(root / "runs/S1/baselines/selection.json")
    baseline = selection.get("selected_family_id") or selection.get(
        "selected_or_diagnostic_family_id"
    )
    if not baseline:
        raise RuntimeError("S1 baseline selection lacks a selected family field")
    definitions = {
        "V2": (
            "V2-A-10-MULTI",
            "V2-A-10-GEN",
            "group_map50_95",
            "group",
            "higher",
            "V2",
        ),
        "S1": (
            "S1-GRU",
            str(baseline),
            "sensor_event_f1",
            "group",
            "higher",
            "S1",
        ),
        "E2_E3": (
            "E3-FULL",
            "E2-LOGIT",
            "concept_event_f1",
            "group",
            "higher",
            "E2_E3",
        ),
        "E3_RELIABILITY": (
            "E3-FULL",
            "E3-NOREL",
            "concept_event_f1",
            "group",
            "higher",
            "E3_RELIABILITY",
        ),
        "E3_GRAPH": (
            "E3-FULL",
            "E3-NOGRAPH",
            "concept_event_f1",
            "group",
            "higher",
            "E3_GRAPH",
        ),
        "E3_MEMORY": (
            "E3-FULL",
            "E3-NOMEM",
            "concept_event_f1",
            "group",
            "higher",
            "E3_MEMORY",
        ),
        "R1": (
            "R1-MULTI",
            "R1-GEN",
            "relative_drop_percent",
            "corruption_cell",
            "lower",
            "R1",
        ),
    }
    results: Dict[str, Any] = {}
    draws = int(protocol["statistics"]["bootstrap_draws"])
    seed = int(protocol["seeds"]["statistics"])
    for offset, (contrast_id, definition) in enumerate(definitions.items()):
        excluded, reason = contrast_boundaries.get(contrast_id, (False, ""))
        if excluded:
            results[contrast_id] = {
                "contrast_id": contrast_id,
                "status": "not_applicable",
                "reason": reason,
            }
            continue
        left, right, metric, level, direction, endpoint = definition
        try:
            table = _paired_table(
                aggregate,
                left_family=left,
                right_family=right,
                metric_name=metric,
                result_level=level,
                allow_static_right_family=contrast_id == "S1",
            )
        except RuntimeError:
            if contrast_id.startswith("E"):
                results[contrast_id] = {
                    "contrast_id": contrast_id,
                    "status": "not_applicable",
                    "reason": "PRETEST eligibility excluded one or both families",
                }
                continue
            raise
        result = _paired_bootstrap(
            table,
            left_family=left,
            right_family=right,
            direction=direction,
            draws=draws,
            seed=seed + offset,
        )
        minimum_pp = float(
            protocol["confirmatory_endpoints"][endpoint]["minimum_meaningful_effect_pp"]
        )
        minimum_effect = minimum_pp if metric == "relative_drop_percent" else minimum_pp / 100.0
        result.update(
            {
                "contrast_id": contrast_id,
                "left_family": left,
                "right_family": right,
                "metric_name": metric,
                "direction_favoring_left": direction,
                "minimum_meaningful_effect": minimum_effect,
                "multiplicity_family": (
                    "mechanism_claim_family"
                    if contrast_id in {"E3_RELIABILITY", "E3_GRAPH", "E3_MEMORY"}
                    else None
                ),
            }
        )
        results[contrast_id] = result
    mechanism = {
        key: result["raw_p_value"]
        for key, result in results.items()
        if result.get("multiplicity_family") == "mechanism_claim_family"
    }
    adjusted = _holm_adjust(mechanism) if mechanism else {}
    for contrast_id, result in results.items():
        if result.get("status") == "not_applicable":
            continue
        adjusted_p = adjusted.get(contrast_id, result["raw_p_value"])
        result["adjusted_p_value"] = adjusted_p
        result["decision_status"] = _contrast_status(
            result,
            minimum_effect=float(result["minimum_meaningful_effect"]),
            p_value=adjusted_p,
        )
    payload = {
        "schema_version": 1,
        "step_id": row["step_id"],
        "status": "pass",
        "confidence_level": float(protocol["statistics"]["confidence_level"]),
        "bootstrap_unit": "smallest_independent_raw_group_or_source_component",
        "bootstrap_draws": draws,
        "secondary_multiplicity": "holm",
        "contrasts": results,
        "negative_results_retained": True,
        "created_at": utc_now(),
    }
    output = "results/final/statistical_tests.json"
    write_json_artifact(root / output, payload)
    return {
        "status": "pass",
        "output_paths": [output],
        "inputs": hash_existing_inputs(
            root,
            [
                "results/final/aggregate.parquet",
                "runs/S1/baselines/selection.json",
                "configs/protocol_lock.pretest.yaml",
            ],
        ),
        "details": {"contrast_count": len(results)},
    }


def _gate_pass(root: Path, gate_id: str) -> bool:
    path = root / "evidence/gates" / f"{gate_id}.json"
    return path.is_file() and load_json(path).get("status") == "pass"


def _claim_status_from_decision(decision: str) -> str:
    return {
        "support": "supported",
        "inconclusive": "limited",
        "negative": "negative",
    }[decision]


def _claim_rows(root: Path, stats: Mapping[str, Any]) -> list[Dict[str, Any]]:
    contrasts = stats["contrasts"]
    rows = []
    thermal_evidence = [
        "evidence/step_reviews/E100.json",
        "evidence/reviews/E100_job19760_thermal_completion_review.json",
    ]
    if not all((root / path).exists() for path in thermal_evidence):
        thermal_evidence = ["results/T1/metrics.parquet"]
    e310_not_applicable = "evidence/episode/E310_not_applicable_amendment_v4.json"
    if not (root / e310_not_applicable).exists():
        e310_not_applicable = "evidence/episode/E310_not_applicable.json"

    def add(
        claim_id: str,
        status: str,
        gate_id: str,
        evidence: Sequence[str],
        wording: str,
    ) -> None:
        rows.append(
            {
                "claim_id": claim_id,
                "status": status,
                "gate_id": gate_id,
                "gate_sha256": sha256_file(root / f"evidence/gates/{gate_id}.json"),
                "evidence_paths_json": json.dumps(list(evidence), sort_keys=True),
                "permitted_wording": wording,
            }
        )

    add(
        "C-DATA",
        "supported" if _gate_pass(root, "G1") else "removed",
        "G1",
        ["data/locked/file_manifest.parquet", "data/locked/split_manifest.parquet"],
        "可复现的有边界公开异构证据空间",
    )
    transfer = contrasts["V2"]
    transfer_not_applicable = transfer.get("status") == "not_applicable"
    add(
        "C-TRANSFER",
        (
            "removed"
            if transfer_not_applicable
            else _claim_status_from_decision(transfer["decision_status"])
        ),
        "G3",
        (
            ["results/final/statistical_tests.json"]
            if transfer_not_applicable
            else ["results/V2/metrics.parquet", "results/final/statistical_tests.json"]
        ),
        "仅限预注册方向和 10% 目标组的预算匹配迁移效应",
    )
    add(
        "C-THERMAL",
        "supported" if _gate_pass(root, "G2") else "removed",
        "G2",
        thermal_evidence,
        "可见光和热分支输入有效性，不含融合优越性或设备温升",
    )
    methane = contrasts["S1"]
    add(
        "C-METHANE",
        _claim_status_from_decision(methane["decision_status"]),
        "G2",
        ["results/S1/metrics.parquet", "results/final/statistical_tests.json"],
        "purged 时间角色下的未来风险事件评分",
    )
    robust = contrasts["R1"]
    robust_not_applicable = robust.get("status") == "not_applicable"
    add(
        "C-ROBUST",
        (
            "removed"
            if robust_not_applicable
            else "limited"
            if robust.get("decision_status") != "negative"
            else "negative"
        ),
        "G4",
        (
            ["results/final/statistical_tests.json"]
            if robust_not_applicable
            else ["results/R1/metrics.parquet", "results/final/statistical_tests.json"]
        ),
        "仅限两类三等级确定性可见度代理退化",
    )
    fusion = contrasts.get("E2_E3", {})
    episode_not_applicable = fusion.get("status") == "not_applicable"
    fusion_status = (
        "removed"
        if episode_not_applicable
        else _claim_status_from_decision(fusion["decision_status"])
    )
    add(
        "C-FUSION",
        fusion_status,
        "G5",
        (
            [
                e310_not_applicable,
                "results/final/statistical_tests.json",
            ]
            if episode_not_applicable
            else [
                "results/E2_E3/metrics.parquet",
                "results/final/statistical_tests.json",
            ]
        ),
        "仅限 fusion-eligible multibranch concept-event 人群",
    )
    graph = contrasts.get("E3_GRAPH", {})
    graph_status = (
        "removed"
        if graph.get("status") == "not_applicable"
        else _claim_status_from_decision(graph["decision_status"])
    )
    add(
        "C-GRAPH",
        graph_status,
        "G5",
        (
            [
                "data/locked/graph_eligibility_lock.json",
                e310_not_applicable,
            ]
            if graph.get("status") == "not_applicable"
            else [
                "data/locked/graph_eligibility_lock.json",
                "results/E2_E3/metrics.parquet",
            ]
        ),
        "仅限 observed-pair 图资格子集，并受 Holm、shuffle 和 edge trace 约束",
    )
    memory = contrasts.get("E3_MEMORY", {})
    memory_status = (
        "removed"
        if memory.get("status") == "not_applicable"
        else _claim_status_from_decision(memory["decision_status"])
    )
    add(
        "C-MEMORY",
        memory_status,
        "G5",
        (
            [
                e310_not_applicable,
                "results/final/statistical_tests.json",
            ]
            if memory.get("status") == "not_applicable"
            else [
                "results/E2_E3/metrics.parquet",
                "results/final/statistical_tests.json",
            ]
        ),
        "同 checkpoint no-memory 控制下的事件记忆效应",
    )
    add(
        "C-EFFICIENCY",
        "limited" if _gate_pass(root, "G6") else "removed",
        "G6",
        ["results/C1/efficiency.parquet", "results/C1/efficiency_scope.csv"],
        "同一 3090 allocation 内的分模块离线测量，不代表现场端到端性能",
    )
    if {row["claim_id"] for row in rows} != set(CLAIM_IDS):
        raise RuntimeError("Claim adjudication does not cover the locked claim set")
    return rows


def _metric_values(
    aggregate: pd.DataFrame,
    family: str,
    metric: str,
    *,
    level: str = "repeat",
) -> np.ndarray:
    values = aggregate.loc[
        (aggregate["family_id"] == family)
        & (aggregate["metric_name"] == metric)
        & (aggregate["result_level"] == level),
        "metric_value",
    ].to_numpy(dtype=float)
    if not len(values):
        raise RuntimeError(f"Metric values are missing: {family}/{metric}/{level}")
    return values


def _mean_ci(values: np.ndarray) -> tuple[float, float, float, float]:
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    half = 1.96 * sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, sd, mean - half, mean + half


def _contrast(root_stats: Mapping[str, Any], contrast_id: str) -> Mapping[str, Any]:
    result = root_stats["contrasts"][contrast_id]
    if result.get("status") == "not_applicable":
        raise RuntimeError(f"Contrast is not applicable: {contrast_id}")
    return result


def _dataset_summary(root: Path, alias: str) -> tuple[int, int, int, str]:
    manifest = pd.read_parquet(root / "data/locked/file_manifest.parquet")
    aliases = {
        "DSLMF": ("dslmf",),
        "DSDPM": ("dsdpm",),
        "DRILL": ("drill", "钻场"),
        "BELT": ("belt",),
        "HELMET": ("helmet",),
        "CMUID": ("cmuid",),
        "HAZE": ("haze",),
        "LLVIP": ("llvip",),
        "KAIST": ("kaist",),
        "FLIR": ("flir",),
        "METHANE": ("methane",),
    }
    candidates = [
        value
        for value in sorted(set(manifest["dataset_id"].astype(str)))
        if any(token in value.lower() for token in aliases[alias])
    ]
    if len(candidates) != 1:
        return 0, 0, 0, "未进入最终锁定数据角色"
    dataset_id = candidates[0]
    rows = manifest.loc[manifest["dataset_id"].astype(str) == dataset_id]
    concepts = set()
    for value in rows["label_summary_json"]:
        payload = json.loads(str(value))
        concepts.update(str(item) for item in payload.get("class_ids", []))
        concepts.update(str(item) for item in payload.get("class_counts", {}))
    return len(rows), int(rows["raw_group_id"].nunique()), len(concepts), "已按 G1 锁定"


def _render_registry(
    root: Path,
    aggregate: pd.DataFrame,
    stats: Mapping[str, Any],
) -> Dict[str, str]:
    source_path = root / "manuscript/archive/paper.pre_experiment.md"
    if not source_path.is_file():
        source_path = root / "manuscript/paper.md"
    paper = source_path.read_text(encoding="utf-8-sig")
    protocol = _protocol(root)
    tokens = sorted(set(PLACEHOLDER_PATTERN.findall(paper)))
    rendered: Dict[str, str] = {}
    contrast_key = {
        "ABS-V-DELTA": "V2",
        "ABS-S-DELTA": "S1",
        "ABS-E-DELTA": "E2_E3",
        "S1-DELTA": "S1",
        "E3-DELTA-F1": "E2_E3",
    }
    endpoint_key = {
        "STAT-DELTA-V": "V2",
        "STAT-DELTA-S": "S1",
        "STAT-DELTA-E": "E2_E3",
        "STAT-DELTA-REL": "E3_RELIABILITY",
        "STAT-DELTA-G": "E3_GRAPH",
        "STAT-DELTA-M": "E3_MEMORY",
        "STAT-DELTA-R": "R1",
    }
    family_by_token = {
        "T1-VIS": "T1-VIS",
        "T1-THERM": "T1-THERM",
        "S1-RULE": "S1-RULE",
        "S1-HGB": "S1-HGB",
        "S1-GRU": "S1-GRU",
        "E2-MEAN": "E2-MEAN",
        "E2-LOGIT": "E2-LOGIT",
        "E3-SHUFFLE": "E3-SHUFFLE",
        "E3-NOREL": "E3-NOREL",
        "E3-NOGRAPH": "E3-NOGRAPH",
        "E3-NOMEM": "E3-NOMEM",
        "E3-FULL": "E3-FULL",
    }
    v2_family = {
        "V2-A-SCRATCH": "V2-A-10-SCR",
        "V2-A-GENERIC": "V2-A-10-GEN",
        "V2-A-SINGLE": "V2-A-10-SINGLE",
        "V2-A-MULTI": "V2-A-10-MULTI",
    }
    source_decision = load_json(root / "evidence/data/dataset_source_decision.json")
    for token in tokens:
        token_id = token.split("=", 1)[0]
        if token_id in contrast_key:
            result = stats["contrasts"][contrast_key[token_id]]
            if result.get("status") == "not_applicable":
                rendered[token] = "不适用（PRETEST 资格门未通过）"
                continue
            scale = 1.0 if result["metric_name"] == "relative_drop_percent" else 100.0
            effect = scale * float(result["effect"])
            lower, upper = [scale * float(value) for value in result["ci95"]]
            if token_id in {"S1-DELTA", "E3-DELTA-F1"}:
                unit = " 个百分点" if "个百分点" in token else " pp"
                rendered[token] = f"{effect:.1f}{unit}；95% CI [{lower:.1f}, {upper:.1f}]"
            else:
                unit = " 个百分点" if "个百分点" in token else " pp"
                rendered[token] = f"{effect:.1f}{unit}"
            continue
        if token_id in endpoint_key:
            value = float(
                protocol["confirmatory_endpoints"][endpoint_key[token_id]][
                    "minimum_meaningful_effect_pp"
                ]
            )
            rendered[token] = f"{value:.1f} 个百分点"
            continue
        if token_id == "D1-MIN-GROUPS":
            rendered[token] = (
                f"{int(protocol['data']['independent_group_floor'])} 组；"
                f"每概念阳性组 {int(protocol['data']['positive_group_floor_per_claimed_class'])}"
            )
            continue
        if token_id.startswith("D1-") and token_id.endswith("-SCALE"):
            alias = token_id.split("-")[1]
            records, groups, _, _ = _dataset_summary(root, alias)
            unit = "对" if alias in {"LLVIP", "KAIST", "FLIR"} else "样本"
            rendered[token] = f"{records} {unit}/{groups} 组"
            continue
        if token_id.startswith("D1-") and token_id.endswith("-DECISION"):
            alias = token_id.split("-")[1]
            _, _, concepts, decision = _dataset_summary(root, alias)
            rendered[token] = f"保留概念 {concepts}；结论 {decision}"
            continue
        if token_id in {"T1-VIS", "T1-THERM"}:
            family = family_by_token[token_id]
            try:
                map_value = float(np.mean(_metric_values(aggregate, family, "map50_95")))
                brier = float(
                    np.mean(_metric_values(aggregate, family, "concept_brier_score"))
                )
                ece = float(
                    np.mean(
                        _metric_values(
                            aggregate, family, "concept_expected_calibration_error"
                        )
                    )
                )
            except RuntimeError:
                rendered[token] = "未纳入 amendment v4 最终确认性聚合；保留输入有效性证据"
                continue
            rendered[token] = (
                f"mAP50-95 {100 * map_value:.1f}；Brier {brier:.3f}；ECE {ece:.3f}"
            )
            continue
        if token_id in {"S1-RULE", "S1-HGB", "S1-GRU"}:
            family = family_by_token[token_id]
            f1_values = _metric_values(aggregate, family, "event_macro_f1")
            f1_mean, f1_sd, f1_low, f1_high = _mean_ci(f1_values)
            false_alarm = float(
                np.mean(_metric_values(aggregate, family, "false_alarms_per_hour"))
            )
            miss = float(np.mean(_metric_values(aggregate, family, "event_miss_rate")))
            lead = float(
                np.mean(_metric_values(aggregate, family, "median_lead_time_seconds"))
            )
            brier = float(np.mean(_metric_values(aggregate, family, "brier_score")))
            ece = float(
                np.mean(_metric_values(aggregate, family, "expected_calibration_error"))
            )
            if token_id == "S1-GRU":
                delta = 100 * float(_contrast(stats, "S1")["effect"])
                rendered[token] = (
                    f"事件宏 F1 {100*f1_mean:.1f}±{100*f1_sd:.1f}%；"
                    f"误报/小时 {false_alarm:.2f}；漏报率 {100*miss:.1f}%；"
                    f"提前量中位数 {lead:.1f} s；Brier/ECE {brier:.3f}/{ece:.3f}；"
                    f"相对参考 Δ {delta:.1f}；95% CI [{100*f1_low:.1f}, {100*f1_high:.1f}]"
                )
            else:
                rendered[token] = (
                    f"事件宏 F1 {100*f1_mean:.1f}%；误报/小时 {false_alarm:.2f}；"
                    f"漏报率 {100*miss:.1f}%；提前量中位数 {lead:.1f} s；"
                    f"Brier/ECE {brier:.3f}/{ece:.3f}"
                )
            continue
        if token_id == "S1-BASE":
            selection = load_json(root / "runs/S1/baselines/selection.json")
            baseline = selection.get("selected_family_id") or selection.get(
                "selected_or_diagnostic_family_id"
            )
            if not baseline:
                raise RuntimeError("S1 baseline selection lacks a selected family field")
            values = _metric_values(aggregate, str(baseline), "event_macro_f1")
            mean, _, low, high = _mean_ci(values)
            false_alarm = float(
                np.mean(_metric_values(aggregate, str(baseline), "false_alarms_per_hour"))
            )
            miss = float(
                np.mean(_metric_values(aggregate, str(baseline), "event_miss_rate"))
            )
            rendered[token] = (
                f"方法 {baseline}；事件宏 F1 {100*mean:.1f}%；"
                f"误报/小时 {false_alarm:.2f}；漏报率 {100*miss:.1f}%；"
                f"95% CI [{100*low:.1f}, {100*high:.1f}]"
            )
            continue
        if token_id in family_by_token and token_id.startswith("E"):
            family = family_by_token[token_id]
            try:
                f1 = float(np.mean(_metric_values(aggregate, family, "event_macro_f1")))
            except RuntimeError:
                rendered[token] = "不适用（PRETEST 资格门未通过）"
                continue
            false_alarm = float(
                np.mean(
                    _metric_values(aggregate, family, "false_alarms_per_100_episodes")
                )
            )
            miss = float(np.mean(_metric_values(aggregate, family, "event_miss_rate")))
            coverage = float(
                np.mean(_metric_values(aggregate, family, "answer_coverage"))
            )
            try:
                delay = float(
                    np.mean(
                        _metric_values(
                            aggregate, family, "mean_detection_delay_steps"
                        )
                    )
                )
                delay_text = f"{delay:.1f} 步"
            except RuntimeError:
                delay_text = "无成功匹配"
            rendered[token] = (
                f"{100*f1:.1f}%；{false_alarm:.1f}；{100*miss:.1f}%；"
                f"{delay_text}；{100*coverage:.1f}%"
            )
            continue
        if token_id in v2_family:
            if stats["contrasts"]["V2"].get("status") == "not_applicable":
                rendered[token] = "不适用（amendment v4 已撤回视觉迁移正式主张）"
                continue
            values = _metric_values(aggregate, v2_family[token_id], "map50_95")
            mean, _, low, high = _mean_ci(values)
            rendered[token] = f"{100*mean:.1f}；95% CI [{100*low:.1f}, {100*high:.1f}]"
            continue
        if token_id in {"V2-A-SINGLE-DELTA", "V2-A-MULTI-DELTA"}:
            if stats["contrasts"]["V2"].get("status") == "not_applicable":
                rendered[token] = "不适用（amendment v4 已撤回视觉迁移正式主张）"
                continue
            family = (
                "V2-A-10-SINGLE" if "SINGLE" in token_id else "V2-A-10-MULTI"
            )
            value = float(np.mean(_metric_values(aggregate, family, "map50_95")))
            generic = float(
                np.mean(_metric_values(aggregate, "V2-A-10-GEN", "map50_95"))
            )
            rendered[token] = f"{100*(value-generic):.1f}"
            continue
        if token_id == "V2-A-BUDGET":
            if stats["contrasts"]["V2"].get("status") == "not_applicable":
                rendered[token] = "不适用（amendment v4 已撤回视觉迁移正式主张）"
                continue
            matched = protocol["training"]["matched_pretraining"]
            rendered[token] = (
                f"{int(matched['unique_source_images'])} 样本/"
                f"{int(matched['optimizer_updates'])} 更新"
            )
            continue
        if token_id == "V2-DIR-A":
            rendered[token] = str(source_decision["visual_direction_id"])
            continue
        if token_id == "ABS-E-DELAY":
            if stats["contrasts"]["E2_E3"].get("status") == "not_applicable":
                rendered[token] = "不适用（PRETEST 资格门未通过）"
                continue
            full = float(
                np.mean(_metric_values(aggregate, "E3-FULL", "mean_detection_delay_steps"))
            )
            logit = float(
                np.mean(_metric_values(aggregate, "E2-LOGIT", "mean_detection_delay_steps"))
            )
            unit = " 步" if "步" in token else " steps"
            rendered[token] = f"{full-logit:.1f}{unit}"
            continue
        if token_id in {"E3-DELTA-DELAY", "E3-DELTA-FA"}:
            if stats["contrasts"]["E2_E3"].get("status") == "not_applicable":
                rendered[token] = "不适用（PRETEST 资格门未通过）"
                continue
            metric = (
                "mean_detection_delay_steps"
                if token_id.endswith("DELAY")
                else "false_alarms_per_100_episodes"
            )
            full = float(np.mean(_metric_values(aggregate, "E3-FULL", metric)))
            logit = float(np.mean(_metric_values(aggregate, "E2-LOGIT", metric)))
            suffix = " 步" if token_id.endswith("DELAY") else "/100 个虚拟序列"
            rendered[token] = f"{full-logit:.1f}{suffix}"
            continue
        if token_id == "FIG4":
            if (
                stats["contrasts"]["V2"].get("status") == "not_applicable"
                and stats["contrasts"]["R1"].get("status") == "not_applicable"
            ):
                rendered[token] = "不适用（amendment v4 下 V2/R1 图源表为空）"
            else:
                rendered[token] = "由 G3/G4 锁定逐组结果生成的 Panel A 迁移与 Panel B 2×3 退化图"
            continue
        if token_id.startswith("EFF-"):
            rendered[token] = _render_efficiency_token(root, token_id)
            continue
        raise RuntimeError(f"No evidence renderer is defined for manuscript token: {token}")
    return rendered


def _render_efficiency_token(root: Path, token_id: str) -> str:
    measurements = pd.read_parquet(root / "results/C1/efficiency.parquet")
    scope_path = root / "results/C1/efficiency_scope.csv"
    if scope_path.is_file():
        scope = pd.read_csv(scope_path)
        frame = scope.merge(measurements, on="module_id", how="left", suffixes=("", "_measured"))
    else:
        frame = measurements.copy()
        if "eligibility_status" not in frame.columns:
            frame["eligibility_status"] = "measured"
    module_by_prefix = {
        "EFF-V": "visible_inference",
        "EFF-T": "thermal_inference",
        "EFF-S": "sensor_inference",
        "EFF-E": "graph_fusion",
    }
    if token_id == "EFF-HW":
        measured = frame.loc[frame["eligibility_status"] == "measured"]
        if measured.empty:
            return "无资格合格模块；未执行硬件计时"
        e400_review_path = root / "evidence/reviews/E400_job19960_completion_review.json"
        e400_review = load_json(e400_review_path) if e400_review_path.is_file() else {}
        slurm_state = e400_review.get("slurm_terminal_state", {})
        device = (
            measured["device_identifier"].iloc[0]
            if "device_identifier" in measured
            else f"Slurm 3090/{slurm_state.get('compute_node', 'unknown')}"
        )
        runtime = (
            measured["runtime_backend"].iloc[0]
            if "runtime_backend" in measured
            else "offline Python/PyTorch"
        )
        precision = (
            measured["precision"].iloc[0]
            if "precision" in measured
            else "protocol-bound"
        )
        return (
            f"GPU/CPU 型号 {device}；"
            f"运行时 {runtime}；"
            f"精度 {precision}"
        )
    prefix = token_id[:5]
    module_id = module_by_prefix[prefix]
    rows = frame.loc[frame["module_id"] == module_id]
    if rows.empty or rows["eligibility_status"].iloc[0] != "measured":
        return "不适用（资格门未通过）"
    row = rows.iloc[0]
    if token_id.endswith("-LAT"):
        if "median_latency_ms" in row:
            median_ms = float(row["median_latency_ms"])
            p95_ms = float(row["p95_latency_ms"])
        else:
            median_ms = float(row["latency_p50_ns"]) / 1_000_000.0
            p95_ms = float(row["latency_p95_ns"]) / 1_000_000.0
        return f"{median_ms:.1f}/{p95_ms:.1f} ms"
    if token_id.endswith("-MEM"):
        unit = "MiB" if module_id == "graph_fusion" else "GiB"
        if "peak_memory_mib" in row:
            value = float(row["peak_memory_mib"])
        else:
            value = float(row["peak_memory_bytes"]) / 2**20
        if unit == "GiB":
            value /= 1024.0
        return f"{value:.1f} {unit}"
    if token_id.endswith("-SIZE"):
        weight_size_bytes = (
            float(row["weight_size_bytes"])
            if "weight_size_bytes" in row
            else float(row["checkpoint_bytes"])
        )
        return (
            f"{float(row['parameter_count'])/1e6:.1f} M/"
            f"{weight_size_bytes/2**20:.1f} MiB"
        )
    raise RuntimeError(f"Unknown efficiency token: {token_id}")


def adjudicate_claims(
    root: Path, row: Mapping[str, str], arguments: Mapping[str, Any]
) -> Dict[str, Any]:
    del arguments
    stats_path = root / "results/final/statistical_tests.json"
    aggregate_path = root / "results/final/aggregate.parquet"
    stats = load_json(stats_path)
    aggregate = pd.read_parquet(aggregate_path)
    claims = pd.DataFrame(_claim_rows(root, stats))
    output = "results/final/claim_status.csv"
    target = root / output
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = claims.to_csv(index=False, lineterminator="\n").encode("utf-8")
    claim_write_mode = "created"
    if target.is_file():
        if target.read_bytes() != encoded:
            current_claims = pd.read_csv(target)
            stable_columns = [
                "claim_id",
                "status",
                "gate_id",
                "gate_sha256",
                "permitted_wording",
            ]
            if not current_claims[stable_columns].equals(claims[stable_columns]):
                raise RuntimeError("Existing claim_status.csv does not match current statistics")
            _atomic_replace_bytes(target, encoded)
            claim_write_mode = "updated_stale_v4_evidence_paths"
        else:
            claim_write_mode = "existing_verified"
    else:
        write_once_bytes(target, encoded)
    registry = _render_registry(root, aggregate, stats)
    registry_output = "results/final/manuscript_values.json"
    registry_payload = {
        "schema_version": 1,
        "status": "pass",
        "source_aggregate_sha256": sha256_file(aggregate_path),
        "source_statistics_sha256": sha256_file(stats_path),
        "values_by_full_token": registry,
        "created_at": utc_now(),
    }
    registry_path = root / registry_output
    registry_bytes = canonical_json_bytes(registry_payload) + b"\n"
    registry_write_mode = "created"
    if registry_path.is_file():
        if registry_path.read_bytes() != registry_bytes:
            _atomic_replace_bytes(registry_path, registry_bytes)
            registry_write_mode = "updated_stale_v4_rendering"
        else:
            registry_write_mode = "existing_verified"
    else:
        write_once_bytes(registry_path, registry_bytes)
    return {
        "status": "pass",
        "output_paths": [output, registry_output],
        "inputs": hash_existing_inputs(
            root,
            [
                "results/final/statistical_tests.json",
                "results/final/aggregate.parquet",
                "manuscript/paper.md",
            ],
        ),
        "details": {
            "claim_count": len(claims),
            "placeholder_value_count": len(registry),
            "claim_write_mode": claim_write_mode,
            "registry_write_mode": registry_write_mode,
        },
    }


def _atomic_replace_bytes(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.parent / f".{target.name}.{os.getpid()}.stage"
    if staged.exists():
        raise RuntimeError(f"Manuscript staging path already exists: {staged}")
    try:
        with staged.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staged, target)
    finally:
        staged.unlink(missing_ok=True)


def _editorial_cleanup(text: str, claim_statuses: Mapping[str, str]) -> str:
    text = re.sub(
        r"^> \*\*编辑占位说明（最终投稿前删除）\*\*：.*\r?\n\r?\n?",
        "",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    replacements = {
        "示意性结果叙述为": "锁定实验结果为",
        "The illustrative result sentence is as follows": (
            "The locked experimental results are as follows"
        ),
        "未通过前，其对应内容只保留示意占位": (
            "未通过时，对应主张不得进入正式结果"
        ),
        "当前版式占位为": "测试前锁定值为",
        (
            "当前尚未执行净室重建实验，因此本节所有以 `【示意占位` 开头的数字、"
            "方向和数据集保留结论仅用于完善论文结构，不进入证据判断；后续必须由"
            "锁定脚本从逐样本结果替换，并在替换前后保留编号映射。"
        ): (
            "本节数值均由锁定脚本从逐样本结果生成，并由编号映射、聚合结果和统计"
            "产物共同追溯。"
        ),
        (
            "表 3 给出 G1 数据审计结果的写作版式。每个占位项后续必须由新项目"
            "不可变清单、数据卡和划分哈希联合替换；"
        ): (
            "表 3 给出 G1 数据审计结果。每个数值由新项目不可变清单、数据卡和"
            "划分哈希联合支撑；"
        ),
        "只有真实差值及其区间替换后，正文才按": "正文依据锁定差值及其区间按",
        "表 6 给出事件级最小基线与机制对照的示意版式": (
            "表 6 给出事件级最小基线与机制对照的锁定结果"
        ),
        "事件级示意结果": "事件级锁定结果",
        "写作版式": "锁定结果格式",
        "`XX` 不表示零值或缺失值插补。": "所有数值均来自锁定结果或显式不适用回执。",
        "当前只保留 ": "锁定图数据说明为 ",
        "，不生成虚构曲线、点位或结果性图注。": "。",
        "结果结论的当前写作占位为": "锁定实验结果为",
        (
            "这些字段不是性能提升或显著性证据；资格为空时删除对应句，真实置信"
            "区间跨越零时改写为“未观察到明确差异”，真实结果为负时报告降低。"
        ): (
            "结论方向依据锁定效应及其置信区间判定；资格为空的主张标为不适用。"
        ),
        "才能用新结果替换全部示意占位": "才能将锁定结果写入正文",
    }
    for source, replacement in replacements.items():
        text = text.replace(source, replacement)

    if claim_statuses.get("C-FUSION") == "removed":
        text = re.sub(
            r"；若融合合格多分支事件人群非空，完整方法相对校准逻辑后融合的事件"
            r"宏 F1 变化为 .*?，平均检测延迟变化为 .*?(?=。不同公开数据)",
            "",
            text,
            count=1,
        )
        text = re.sub(
            r"; and, if the fusion-eligible multibranch event population is nonempty, "
            r"the full method changes event macro-F1 by .*? relative to calibrated "
            r"logistic late fusion(?=\. Public datasets)",
            "",
            text,
            count=1,
        )
        text = re.sub(
            r"；仅当融合合格多分支事件人群非空时，完整方法相对校准逻辑后融合的"
            r"事件宏 F1 和检测延迟变化分别为 .*?(?=。结论方向)",
            "",
            text,
            count=1,
        )
        lines = text.splitlines()
        lines = [
            (
                "融合资格门未通过，E2/E3 及其可靠性、图和记忆机制对照不进入正式结果。"
                if line.startswith("在融合合格多分支 concept-event 人群上")
                else line
            )
            for line in lines
        ]
        text = "\n".join(lines) + ("\n" if text.endswith("\n") else "")

    if PLACEHOLDER_PATTERN.search(text) or "示意" in text or "占位" in text:
        raise RuntimeError("Manuscript editorial placeholder residue remains after replacement")
    return text


def _resolved_registry_markdown(
    values: Mapping[str, str], registry_sha256: str, created_at: str
) -> bytes:
    lines = [
        "# 论文实验字段最终替换登记表",
        "",
        f"> 状态：`resolved`；来源登记哈希：`{registry_sha256}`；时间：`{created_at}`。",
        "",
        "| 完整字段 | 锁定替换值 |",
        "|---|---|",
    ]
    for token, value in sorted(values.items()):
        escaped_token = token.replace("|", "\\|")
        escaped_value = str(value).replace("|", "\\|")
        lines.append(f"| `{escaped_token}` | {escaped_value} |")
    lines.append("")
    return "\n".join(lines).encode("utf-8")


def _render_fig4(root: Path, created_at: str) -> list[str]:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    data_path = root / "results/final/fig4_data.parquet"
    frame = pd.read_parquet(data_path)
    panel_a = frame.loc[frame["figure_panel"] == "A_transfer"].copy()
    panel_b = frame.loc[frame["figure_panel"] == "B_robustness"].copy()
    if panel_a.empty or panel_b.empty:
        contract = {
            "schema_version": 1,
            "status": "scope_boundary_not_applicable",
            "core_conclusion": (
                "Amendment v4 removes the V2 transfer and R1 robustness claim families "
                "from the confirmatory scope; no locked Figure 4 quantitative panels are drawn."
            ),
            "archetype": "scope_boundary",
            "backend": "python_matplotlib",
            "panels": {
                "A": "not_applicable: V2 transfer claim removed under amendment v4",
                "B": "not_applicable: R1 robustness claim removed under amendment v4",
            },
            "review_risks": [
                "empty panels must not be described as negative quantitative results",
                "removed claims must not be restored through figure text",
                "source data remains the locked empty v4 fig4_data artifact",
            ],
            "source_data": "results/final/fig4_data.parquet",
            "source_data_sha256": sha256_file(data_path),
            "exports": ["svg", "pdf", "tiff_600dpi", "png_600dpi"],
            "created_at": created_at,
        }
        contract_path = root / "figures/fig4_contract.json"
        _atomic_replace_bytes(contract_path, canonical_json_bytes(contract) + b"\n")
        mpl.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
                "svg.fonttype": "none",
                "svg.hashsalt": "mining1-fig4-v4-scope-boundary",
                "pdf.fonttype": 42,
                "font.size": 8,
            }
        )
        fig, ax = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
        ax.axis("off")
        ax.text(
            0.02,
            0.80,
            "Figure 4 scope boundary",
            fontsize=12,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            0.02,
            0.58,
            "V2 transfer and R1 robustness panels are not applicable in amendment v4.",
            transform=ax.transAxes,
        )
        ax.text(
            0.02,
            0.42,
            "The locked source artifact is intentionally empty; no quantitative visual claim is restored.",
            transform=ax.transAxes,
        )
        ax.text(
            0.02,
            0.26,
            "See claim_status.csv: C-TRANSFER=removed and C-ROBUST=removed.",
            transform=ax.transAxes,
        )
        base = root / "figures/fig4_transfer_robustness"
        base.parent.mkdir(parents=True, exist_ok=True)
        creator = "mining1 locked Python workflow"
        fig.savefig(
            base.with_suffix(".svg"),
            bbox_inches="tight",
            metadata={"Creator": creator, "Date": None},
        )
        fig.savefig(
            base.with_suffix(".pdf"),
            bbox_inches="tight",
            metadata={"Creator": creator, "CreationDate": None, "ModDate": None},
        )
        fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
        fig.savefig(
            base.with_suffix(".tiff"),
            dpi=600,
            bbox_inches="tight",
            pil_kwargs={"compression": "tiff_lzw"},
        )
        plt.close(fig)
        outputs = [
            "figures/fig4_contract.json",
            "figures/fig4_transfer_robustness.svg",
            "figures/fig4_transfer_robustness.pdf",
            "figures/fig4_transfer_robustness.png",
            "figures/fig4_transfer_robustness.tiff",
        ]
        for relative in outputs:
            if not (root / relative).is_file() or (root / relative).stat().st_size == 0:
                raise RuntimeError(f"Figure 4 export is missing or empty: {relative}")
        return outputs
    contract = {
        "schema_version": 1,
        "status": "locked_result_figure",
        "core_conclusion": (
            "At the locked 10% target-group budget and controlled proxy corruptions, "
            "initialization source may alter target performance and stability."
        ),
        "archetype": "quantitative_grid",
        "backend": "python_matplotlib",
        "panels": {
            "A": "four locked V2 initializations with repeat-level 95% intervals",
            "B": "generic versus multi-coal relative drop in the locked 2 by 3 cells",
        },
        "review_risks": [
            "repeat units must not be treated as independent frames",
            "proxy corruptions must not be described as field conditions",
            "negative or null effects must remain visible",
        ],
        "source_data": "results/final/fig4_data.parquet",
        "source_data_sha256": sha256_file(data_path),
        "exports": ["svg", "pdf", "tiff_600dpi", "png_600dpi"],
        "created_at": created_at,
    }
    contract_path = root / "figures/fig4_contract.json"
    _atomic_replace_bytes(contract_path, canonical_json_bytes(contract) + b"\n")

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "svg.hashsalt": "mining1-fig4-v1",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    neutral = "#65717C"
    signal = "#2B7A9B"
    accent = "#B55245"

    family_order = [
        "V2-A-10-SCR",
        "V2-A-10-GEN",
        "V2-A-10-SINGLE",
        "V2-A-10-MULTI",
    ]
    family_labels = ["Scratch", "Generic", "Single-coal", "Multi-coal"]
    means = []
    intervals = []
    for index, family in enumerate(family_order):
        values = panel_a.loc[panel_a["family_id"] == family, "metric_value"].to_numpy(
            dtype=float
        )
        mean, _, low, high = _mean_ci(values)
        means.append(100.0 * mean)
        intervals.append((100.0 * (mean - low), 100.0 * (high - mean)))
        offsets = np.linspace(-0.07, 0.07, len(values)) if len(values) > 1 else [0.0]
        axes[0].scatter(
            np.asarray(offsets) + index,
            100.0 * values,
            s=16,
            color=neutral,
            alpha=0.65,
            linewidths=0,
            zorder=2,
        )
    axes[0].errorbar(
        np.arange(len(family_order)),
        means,
        yerr=np.asarray(intervals).T,
        fmt="o",
        color=signal,
        ecolor=signal,
        markersize=5,
        capsize=3,
        linewidth=1.2,
        zorder=3,
    )
    axes[0].set_xticks(np.arange(len(family_order)), family_labels, rotation=20, ha="right")
    axes[0].set_ylabel("mAP50-95 (%)")
    axes[0].set_title("a  Transfer at 10% target groups", loc="left", fontweight="bold")
    axes[0].grid(axis="y", color="#D9DEE2", linewidth=0.6, alpha=0.8)

    styles = {
        ("R1-GEN", "low_light"): (neutral, "-", "o", "Generic · low light"),
        ("R1-GEN", "dust_fog_proxy"): (
            neutral,
            "--",
            "s",
            "Generic · dust/fog proxy",
        ),
        ("R1-MULTI", "low_light"): (signal, "-", "o", "Multi-coal · low light"),
        ("R1-MULTI", "dust_fog_proxy"): (
            accent,
            "--",
            "s",
            "Multi-coal · dust/fog proxy",
        ),
    }
    for (family, corruption), (color, linestyle, marker, label) in styles.items():
        means_b = []
        errors_b = []
        for severity in (1, 2, 3):
            values = panel_b.loc[
                (panel_b["family_id"] == family)
                & (panel_b["group_id"] == f"{corruption}:{severity}"),
                "metric_value",
            ].to_numpy(dtype=float)
            mean, _, low, high = _mean_ci(values)
            means_b.append(mean)
            errors_b.append((mean - low, high - mean))
        axes[1].errorbar(
            (1, 2, 3),
            means_b,
            yerr=np.asarray(errors_b).T,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=4,
            capsize=2,
            linewidth=1.1,
            label=label,
        )
    axes[1].axhline(0.0, color="#AEB6BC", linewidth=0.7)
    axes[1].set_xticks((1, 2, 3))
    axes[1].set_xlabel("Proxy corruption severity")
    axes[1].set_ylabel("Relative mAP50-95 drop (%)")
    axes[1].set_title("b  Controlled robustness", loc="left", fontweight="bold")
    axes[1].grid(axis="y", color="#D9DEE2", linewidth=0.6, alpha=0.8)
    axes[1].legend(loc="best", fontsize=6)

    base = root / "figures/fig4_transfer_robustness"
    base.parent.mkdir(parents=True, exist_ok=True)
    creator = "mining1 locked Python workflow"
    fig.savefig(
        base.with_suffix(".svg"),
        bbox_inches="tight",
        metadata={"Creator": creator, "Date": None},
    )
    fig.savefig(
        base.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={"Creator": creator, "CreationDate": None, "ModDate": None},
    )
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(
        base.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
    outputs = [
        "figures/fig4_contract.json",
        "figures/fig4_transfer_robustness.svg",
        "figures/fig4_transfer_robustness.pdf",
        "figures/fig4_transfer_robustness.png",
        "figures/fig4_transfer_robustness.tiff",
    ]
    for relative in outputs:
        if not (root / relative).is_file() or (root / relative).stat().st_size == 0:
            raise RuntimeError(f"Figure 4 export is missing or empty: {relative}")
    return outputs


def update_manuscript(
    root: Path, row: Mapping[str, str], arguments: Mapping[str, Any]
) -> Dict[str, Any]:
    del row, arguments
    values_path = root / "results/final/manuscript_values.json"
    claims_path = root / "results/final/claim_status.csv"
    paper_path = root / "manuscript/paper.md"
    if not values_path.is_file() or not claims_path.is_file() or not paper_path.is_file():
        raise RuntimeError("Manuscript update inputs are incomplete")
    payload = load_json(values_path)
    values = payload.get("values_by_full_token")
    if not isinstance(values, dict) or not values:
        raise RuntimeError("Manuscript value registry is empty")
    if sha256_file(root / "results/final/aggregate.parquet") != payload.get(
        "source_aggregate_sha256"
    ):
        raise RuntimeError("Manuscript registry aggregate hash no longer resolves")
    if sha256_file(root / "results/final/statistical_tests.json") != payload.get(
        "source_statistics_sha256"
    ):
        raise RuntimeError("Manuscript registry statistics hash no longer resolves")

    claims = pd.read_csv(claims_path)
    claim_statuses = dict(zip(claims["claim_id"], claims["status"]))
    backup_path = root / "manuscript/archive/paper.pre_experiment.md"
    current = paper_path.read_bytes()
    if PLACEHOLDER_PATTERN.search(current.decode("utf-8-sig")):
        write_once_bytes(backup_path, current)
    elif not backup_path.is_file():
        raise RuntimeError("Placeholder-free manuscript lacks its immutable source backup")

    source = backup_path.read_text(encoding="utf-8-sig")
    source_tokens = PLACEHOLDER_PATTERN.findall(source)
    if set(source_tokens) != set(values):
        missing = sorted(set(source_tokens) - set(values))
        extra = sorted(set(values) - set(source_tokens))
        raise RuntimeError(f"Manuscript registry mismatch: missing={missing}, extra={extra}")
    updated = source
    for token in sorted(values, key=len, reverse=True):
        updated = updated.replace(f"【示意占位：{token}】", str(values[token]))
    updated = _editorial_cleanup(updated, claim_statuses)
    updated_bytes = updated.encode("utf-8")
    if current != updated_bytes:
        _atomic_replace_bytes(paper_path, updated_bytes)

    created_at = str(payload.get("created_at") or utc_now())
    figure_outputs = _render_fig4(root, created_at)
    registry_hash = sha256_file(values_path)
    resolved_registry = root / "notes/manuscript_placeholder_registry.resolved.md"
    _atomic_replace_bytes(
        resolved_registry,
        _resolved_registry_markdown(values, registry_hash, created_at),
    )
    docx_status_path = root / "manuscript/paper_draft.status.json"
    draft_docx = root / "manuscript/paper_draft.docx"
    docx_status = {
        "schema_version": 1,
        "status": "historical_non_authoritative",
        "authoritative_source": "manuscript/paper.md",
        "historical_file": "manuscript/paper_draft.docx",
        "historical_file_sha256": sha256_file(draft_docx) if draft_docx.is_file() else None,
        "reason": "predates the clean-room manuscript and must not be submitted as current",
        "created_at": created_at,
    }
    _atomic_replace_bytes(docx_status_path, canonical_json_bytes(docx_status) + b"\n")
    receipt_path = root / "manuscript/experiment_update_receipt.json"
    receipt = {
        "schema_version": 1,
        "status": "pass",
        "authoritative_source": "manuscript/paper.md",
        "source_backup_sha256": sha256_file(backup_path),
        "value_registry_sha256": registry_hash,
        "claim_status_sha256": sha256_file(claims_path),
        "updated_manuscript_sha256": sha256_file(paper_path),
        "resolved_token_count": len(values),
        "unresolved_token_count": 0,
        "submission_metadata_changed": False,
        "docx_status_sha256": sha256_file(docx_status_path),
        "figure_outputs": [describe_output(root, path) for path in figure_outputs],
        "created_at": created_at,
    }
    _atomic_replace_bytes(receipt_path, canonical_json_bytes(receipt) + b"\n")
    outputs = [
        "manuscript/paper.md",
        "manuscript/archive/paper.pre_experiment.md",
        "notes/manuscript_placeholder_registry.resolved.md",
        "manuscript/paper_draft.status.json",
        "manuscript/experiment_update_receipt.json",
    ] + figure_outputs
    return {
        "status": "pass",
        "output_paths": outputs,
        "inputs": hash_existing_inputs(
            root,
            [
                "results/final/manuscript_values.json",
                "results/final/claim_status.csv",
                "results/final/aggregate.parquet",
                "results/final/statistical_tests.json",
            ],
        ),
        "details": {"resolved_token_count": len(values), "authoritative_format": "md"},
    }


def _review_claim_map(root: Path, failures: list[str]) -> Dict[str, int]:
    path = root / "results/final/claim_status.csv"
    claims = pd.read_csv(path)
    required_columns = {
        "claim_id",
        "status",
        "gate_id",
        "gate_sha256",
        "evidence_paths_json",
        "permitted_wording",
    }
    if not required_columns.issubset(claims.columns):
        failures.append("claim map lacks required columns")
        return {"claim_count": len(claims), "evidence_path_count": 0}
    claim_ids = set(claims["claim_id"].astype(str))
    if claim_ids != set(CLAIM_IDS) or claims["claim_id"].duplicated().any():
        failures.append("claim map does not contain the nine unique locked claims")
    allowed = {"supported", "limited", "negative", "removed"}
    if not set(claims["status"].astype(str)).issubset(allowed):
        failures.append("claim map contains an invalid adjudication status")
    evidence_count = 0
    for record in claims.to_dict(orient="records"):
        gate_path = root / "evidence/gates" / f"{record['gate_id']}.json"
        if not gate_path.is_file() or sha256_file(gate_path) != str(record["gate_sha256"]):
            failures.append(f"claim gate hash does not resolve: {record['claim_id']}")
        try:
            evidence_paths = json.loads(str(record["evidence_paths_json"]))
        except json.JSONDecodeError:
            failures.append(f"claim evidence list is invalid: {record['claim_id']}")
            continue
        if not isinstance(evidence_paths, list) or not evidence_paths:
            failures.append(f"claim evidence list is empty: {record['claim_id']}")
            continue
        evidence_count += len(evidence_paths)
        for relative in evidence_paths:
            if not (root / str(relative)).exists():
                failures.append(
                    f"claim evidence does not resolve: {record['claim_id']}:{relative}"
                )
    return {"claim_count": len(claims), "evidence_path_count": evidence_count}


def _review_figure4(root: Path, failures: list[str]) -> Dict[str, Any]:
    outputs = [
        "figures/fig4_transfer_robustness.svg",
        "figures/fig4_transfer_robustness.pdf",
        "figures/fig4_transfer_robustness.png",
        "figures/fig4_transfer_robustness.tiff",
    ]
    missing = [path for path in outputs if not (root / path).is_file()]
    if missing:
        failures.append(f"Figure 4 exports are missing: {missing}")
        return {"exports": [], "pixel_check": "not_run"}
    contract = load_json(root / "figures/fig4_contract.json")
    data_path = root / "results/final/fig4_data.parquet"
    if contract.get("source_data_sha256") != sha256_file(data_path):
        failures.append("Figure 4 source-data hash does not resolve")
    if not (root / outputs[0]).read_text(encoding="utf-8", errors="ignore").lstrip().startswith(
        "<?xml"
    ):
        failures.append("Figure 4 SVG is not a valid XML export")
    if not (root / outputs[1]).read_bytes().startswith(b"%PDF"):
        failures.append("Figure 4 PDF signature is invalid")
    pixel_check = "pass"
    try:
        from PIL import Image

        with Image.open(root / outputs[2]) as image:
            if image.width < 2400 or image.height < 1000:
                failures.append("Figure 4 PNG resolution is below the locked export floor")
            extrema = image.convert("RGB").getextrema()
            if all(low == high for low, high in extrema):
                failures.append("Figure 4 PNG is visually blank")
        with Image.open(root / outputs[3]) as image:
            if image.width < 2400 or image.height < 1000:
                failures.append("Figure 4 TIFF resolution is below the locked export floor")
    except Exception as exc:
        pixel_check = "fail"
        failures.append(f"Figure 4 pixel QA failed: {exc}")
    return {
        "exports": [describe_output(root, path) for path in outputs],
        "pixel_check": pixel_check,
    }


def review_package(
    root: Path, row: Mapping[str, str], arguments: Mapping[str, Any]
) -> Dict[str, Any]:
    del arguments
    failures: list[str] = []
    required = [
        "results/final/aggregate.parquet",
        "results/final/fig4_data.parquet",
        "results/final/statistical_tests.json",
        "results/final/claim_status.csv",
        "results/final/manuscript_values.json",
        "results/C1/artifact_qa.json",
        "manuscript/paper.md",
        "manuscript/experiment_update_receipt.json",
        "manuscript/paper_draft.status.json",
        "notes/manuscript_placeholder_registry.resolved.md",
        "figures/fig4_contract.json",
    ] + [f"evidence/gates/G{index}.json" for index in range(7)]
    missing = [relative for relative in required if not (root / relative).is_file()]
    if missing:
        raise RuntimeError(f"Final review inputs are incomplete: {missing}")

    for gate_id in (f"G{index}" for index in range(7)):
        if not _gate_pass(root, gate_id):
            failures.append(f"evidence gate did not pass: {gate_id}")
    artifact_qa = load_json(root / "results/C1/artifact_qa.json")
    if artifact_qa.get("status") != "pass" or artifact_qa.get("failures"):
        failures.append("artifact QA is not clean")

    aggregate_path = root / "results/final/aggregate.parquet"
    aggregate = pd.read_parquet(aggregate_path)
    if aggregate.empty or not np.isfinite(aggregate["metric_value"].astype(float)).all():
        failures.append("final aggregate is empty or non-finite")
    plan_path = root / "plans/experiment_steps.csv"
    plan_statuses = (
        {step["step_id"]: step["status"] for step in load_plan(root)}
        if plan_path.is_file()
        else {}
    )
    expected_packages = {"S1"}
    if plan_statuses.get("E207") not in {"not_applicable", "accepted_not_applicable"}:
        expected_packages.add("V2")
    if plan_statuses.get("E223") not in {"not_applicable", "accepted_not_applicable"}:
        expected_packages.add("R1")
    if not expected_packages.issubset(set(aggregate["package_id"].astype(str))):
        failures.append("final aggregate lacks a mandatory evidence package")

    stats = load_json(root / "results/final/statistical_tests.json")
    if stats.get("negative_results_retained") is not True:
        failures.append("statistical package does not retain negative results")
    if not {"V2", "S1", "R1"}.issubset(set(stats.get("contrasts", {}))):
        failures.append("statistical package lacks a mandatory contrast")
    values = load_json(root / "results/final/manuscript_values.json")
    if values.get("source_aggregate_sha256") != sha256_file(aggregate_path):
        failures.append("manuscript value registry aggregate hash does not resolve")
    stats_path = root / "results/final/statistical_tests.json"
    if values.get("source_statistics_sha256") != sha256_file(stats_path):
        failures.append("manuscript value registry statistics hash does not resolve")

    claim_summary = _review_claim_map(root, failures)
    paper_path = root / "manuscript/paper.md"
    paper = paper_path.read_text(encoding="utf-8-sig")
    forbidden_fragments = (
        "【示意占位：",
        DAMAGED_REMOTE_HOST,
        DAMAGED_REMOTE_USER_PREFIX,
        "F:\\2026\\mining\\",
        "/data/zyh/",
    )
    for fragment in forbidden_fragments:
        if fragment in paper:
            failures.append(f"authoritative manuscript contains forbidden residue: {fragment}")
    if "示意" in paper or "占位" in paper or re.search(r"\bXX(?:\.X+)?\b", paper):
        failures.append("authoritative manuscript still contains editorial placeholder language")

    update_receipt = load_json(root / "manuscript/experiment_update_receipt.json")
    if update_receipt.get("updated_manuscript_sha256") != sha256_file(paper_path):
        failures.append("manuscript update receipt no longer matches the authoritative source")
    if update_receipt.get("submission_metadata_changed") is not False:
        failures.append("manuscript update improperly changed submission metadata")
    docx_status = load_json(root / "manuscript/paper_draft.status.json")
    if docx_status.get("status") != "historical_non_authoritative":
        failures.append("legacy DOCX is not explicitly excluded from the current package")
    historical_docx = root / "manuscript/paper_draft.docx"
    if historical_docx.is_file() and docx_status.get("historical_file_sha256") != sha256_file(
        historical_docx
    ):
        failures.append("legacy DOCX status hash does not resolve")

    figure_summary = _review_figure4(root, failures)
    if failures:
        raise RuntimeError("Final package review failed: " + "; ".join(failures))
    output = "evidence/reviews/final_review.json"
    payload = {
        "schema_version": 1,
        "step_id": row["step_id"],
        "status": "pass",
        "claim_support_review": "pass",
        "test_leakage_review": "pass_via_G1_G5_and_artifact_QA",
        "reproducibility_review": "pass",
        "presentation_review": "pass",
        "authoritative_manuscript": describe_output(root, "manuscript/paper.md"),
        "claim_summary": claim_summary,
        "figure4_summary": figure_summary,
        "submission_metadata": "pending_nonblocking_until_final_submission",
        "legacy_docx": "historical_non_authoritative",
        "failures": [],
        "created_at": utc_now(),
    }
    write_json_artifact(root / output, payload)
    return {
        "status": "pass",
        "output_paths": [output],
        "inputs": hash_existing_inputs(root, required),
        "details": {
            "claim_count": claim_summary["claim_count"],
            "figure_pixel_check": figure_summary["pixel_check"],
        },
    }


def finalize_closeout(
    root: Path, row: Mapping[str, str], arguments: Mapping[str, Any]
) -> Dict[str, Any]:
    del arguments
    audit_path = root / "evidence/closeout/remote_account_audit.json"
    if not audit_path.is_file():
        raise RuntimeError("Account-wide remote closeout audit is missing")
    remote_audit = load_json(audit_path)
    validate_remote_closeout(remote_audit, require_verified=True)
    if remote_audit.get("unsynchronized_artifacts"):
        raise RuntimeError("Final closeout still lists unsynchronized artifacts")
    gates = {}
    for gate_id in (f"G{index}" for index in range(8)):
        path = root / "evidence/gates" / f"{gate_id}.json"
        if not path.is_file() or load_json(path).get("status") != "pass":
            raise RuntimeError(f"Final closeout requires a passing {gate_id}")
        gates[gate_id] = sha256_file(path)
    plan = load_plan(root)
    incomplete = [
        item["step_id"]
        for item in plan
        if item["step_id"] != row["step_id"]
        and item["status"] not in {"pass", "not_applicable", "accepted_not_applicable"}
    ]
    if incomplete:
        raise RuntimeError(f"Final closeout has incomplete frozen-plan steps: {incomplete}")
    required_local = [
        "results/final/aggregate.parquet",
        "results/final/fig4_data.parquet",
        "results/final/statistical_tests.json",
        "results/final/claim_status.csv",
        "results/final/manuscript_values.json",
        "manuscript/paper.md",
        "manuscript/experiment_update_receipt.json",
        "evidence/reviews/final_review.json",
        "figures/fig4_transfer_robustness.svg",
        "figures/fig4_transfer_robustness.pdf",
        "figures/fig4_transfer_robustness.png",
        "figures/fig4_transfer_robustness.tiff",
    ]
    missing = [relative for relative in required_local if not (root / relative).is_file()]
    if missing:
        raise RuntimeError(f"Final artifacts are not fully synchronized locally: {missing}")
    output = "evidence/closeout/final_receipt.json"
    payload = {
        "schema_version": 1,
        "step_id": row["step_id"],
        "status": "pass",
        "remote_host": remote_audit["host"],
        "account_scope": remote_audit["account_scope"],
        "verified_billing_state": remote_audit["final_billing_state"],
        "remote_audit_sha256": sha256_file(audit_path),
        "gate_hashes": gates,
        "local_final_artifacts": [describe_output(root, path) for path in required_local],
        "all_frozen_steps_complete_before_E910": True,
        "submission_metadata": "pending_nonblocking_until_final_submission",
        "created_at": utc_now(),
    }
    write_json_artifact(root / output, payload)
    return {
        "status": "pass",
        "output_paths": [output],
        "inputs": hash_existing_inputs(root, [str(audit_path.relative_to(root)), *required_local]),
        "details": {
            "verified_billing_state": remote_audit["final_billing_state"],
            "local_artifact_count": len(required_local),
        },
    }
