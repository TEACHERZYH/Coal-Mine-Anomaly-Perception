from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import yaml

from mining1_exp.provenance import sha256_file
from mining1_exp.workflow_finalize import (
    _holm_adjust,
    adjudicate_claims,
    aggregate_results,
    audit_artifacts,
    review_package,
    run_confirmatory_stats,
    update_manuscript,
)


def _metric(
    package: str,
    family: str,
    repeat: str,
    group: str,
    level: str,
    name: str,
    value: float,
) -> dict[str, object]:
    return {
        "package_id": package,
        "family_id": family,
        "run_id": f"{family}-{repeat}",
        "repeat_key": repeat,
        "group_id": group,
        "result_level": level,
        "metric_name": name,
        "metric_value": value,
        "population_hash": "p" * 64,
        "prediction_lock_hash": "l" * 64,
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _build_metric_packages(root: Path) -> None:
    t1 = []
    for family, base in (("T1-VIS", 0.42), ("T1-THERM", 0.39)):
        for metric, value in (
            ("map50_95", base),
            ("concept_brier_score", 0.15),
            ("concept_expected_calibration_error", 0.04),
        ):
            t1.append(_metric("T1", family, "seed0", "all", "repeat", metric, value))

    s1 = []
    display_metrics = {
        "event_macro_f1": 0.60,
        "false_alarms_per_hour": 0.20,
        "event_miss_rate": 0.15,
        "median_lead_time_seconds": 8.0,
        "brier_score": 0.12,
        "expected_calibration_error": 0.03,
    }
    for family, bonus in (("S1-RULE", -0.03), ("S1-HGB", 0.0), ("S1-GRU", 0.05)):
        for repeat in ("seed1", "seed2", "seed3"):
            for metric, value in display_metrics.items():
                adjusted = value + bonus if metric == "event_macro_f1" else value
                s1.append(_metric("S1", family, repeat, "all", "repeat", metric, adjusted))
            for group_index in range(3):
                value = 0.60 + bonus + 0.01 * group_index
                s1.append(
                    _metric(
                        "S1",
                        family,
                        repeat,
                        f"methane-{group_index}",
                        "group",
                        "sensor_event_f1",
                        value,
                    )
                )

    v2 = []
    v2_values = {
        "V2-A-10-SCR": 0.31,
        "V2-A-10-GEN": 0.36,
        "V2-A-10-SINGLE": 0.38,
        "V2-A-10-MULTI": 0.41,
    }
    for family, base in v2_values.items():
        for repeat_index, repeat in enumerate(("seed1", "seed2", "seed3")):
            value = base + 0.005 * repeat_index
            v2.append(_metric("V2", family, repeat, "all", "repeat", "map50_95", value))
            for group_index in range(3):
                v2.append(
                    _metric(
                        "V2",
                        family,
                        repeat,
                        f"visual-{group_index}",
                        "group",
                        "group_map50_95",
                        value + 0.002 * group_index,
                    )
                )

    r1 = []
    for family, base in (("R1-GEN", 18.0), ("R1-MULTI", 13.0)):
        for repeat_index, repeat in enumerate(("seed1", "seed2", "seed3")):
            for corruption in ("low_light", "dust_fog_proxy"):
                for severity in (1, 2, 3):
                    value = base + 3.0 * severity + 0.2 * repeat_index
                    r1.append(
                        _metric(
                            "R1",
                            family,
                            repeat,
                            f"{corruption}:{severity}",
                            "corruption_cell",
                            "relative_drop_percent",
                            value,
                        )
                    )

    for package, rows in (("T1", t1), ("S1", s1), ("V2", v2), ("R1", r1)):
        path = root / "results" / package / "metrics.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(path, index=False)


def _build_contracts(root: Path) -> None:
    endpoint_ids = ("V2", "S1", "E2_E3", "E3_RELIABILITY", "E3_GRAPH", "E3_MEMORY", "R1")
    protocol = {
        "statistics": {"bootstrap_draws": 200, "confidence_level": 0.95},
        "seeds": {"statistics": 2026},
        "confirmatory_endpoints": {
            endpoint: {"minimum_meaningful_effect_pp": 1.0} for endpoint in endpoint_ids
        },
    }
    config = root / "configs/protocol_lock.pretest.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(yaml.safe_dump(protocol), encoding="utf-8")
    _write_json(
        root / "runs/S1/baselines/selection.json",
        {"selected_family_id": "S1-HGB"},
    )
    _write_json(
        root / "evidence/data/dataset_source_decision.json",
        {"visual_direction_id": "source-A-to-target-B"},
    )
    _write_json(root / "evidence/episode/E310_not_applicable.json", {"status": "pass"})
    _write_json(root / "data/locked/graph_eligibility_lock.json", {"eligible": False})
    for relative in (
        "data/locked/file_manifest.parquet",
        "data/locked/split_manifest.parquet",
        "results/C1/efficiency.parquet",
        "results/C1/efficiency_scope.csv",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"locked")
    for gate_index in range(7):
        _write_json(root / f"evidence/gates/G{gate_index}.json", {"status": "pass"})
    _write_json(root / "results/C1/artifact_qa.json", {"status": "pass", "failures": []})


def _build_manuscript(root: Path) -> None:
    paper = """> **编辑占位说明（最终投稿前删除）**：本文所有字段仅用于固定版式。

# 合成闭环测试稿

示意性结果叙述为：迁移差值 `【示意占位：ABS-V-DELTA=XX.X 个百分点】`；
时序差值 `【示意占位：ABS-S-DELTA=XX.X 个百分点】`；若融合资格存在，
融合差值 `【示意占位：ABS-E-DELTA=XX.X 个百分点】`，延迟
`【示意占位：ABS-E-DELAY=XX.X 步】`。

图数据：`【示意占位：FIG4=待真实逐样本结果生成】`。
"""
    path = root / "manuscript/paper.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(paper, encoding="utf-8")


def test_holm_adjust_is_monotonic() -> None:
    adjusted = _holm_adjust({"b": 0.03, "a": 0.01, "c": 0.20})
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"]
    assert all(0.0 <= value <= 1.0 for value in adjusted.values())


def test_finalize_chain_handles_episode_not_applicable(tmp_path: Path) -> None:
    _build_metric_packages(tmp_path)
    _build_contracts(tmp_path)
    _build_manuscript(tmp_path)

    aggregate_results(tmp_path, {"step_id": "E500"}, {})
    run_confirmatory_stats(tmp_path, {"step_id": "E510"}, {})
    adjudicate_claims(tmp_path, {"step_id": "E520"}, {})
    update_manuscript(tmp_path, {"step_id": "E530"}, {})
    result = review_package(tmp_path, {"step_id": "E540"}, {})

    paper = (tmp_path / "manuscript/paper.md").read_text(encoding="utf-8")
    assert "示意" not in paper
    assert "占位" not in paper
    assert "不适用（PRETEST 资格门未通过）" in paper
    claims = pd.read_csv(tmp_path / "results/final/claim_status.csv")
    assert len(claims) == 9
    assert claims.loc[claims["claim_id"] == "C-FUSION", "status"].item() == "removed"
    assert result["status"] == "pass"
    assert (tmp_path / "figures/fig4_transfer_robustness.svg").stat().st_size > 0
    update_receipt = json.loads(
        (tmp_path / "manuscript/experiment_update_receipt.json").read_text(encoding="utf-8")
    )
    assert update_receipt["updated_manuscript_sha256"] == sha256_file(
        tmp_path / "manuscript/paper.md"
    )


def test_aggregate_results_accepts_v4_visual_robustness_not_applicable(
    tmp_path: Path,
) -> None:
    fieldnames = [
        "step_id",
        "phase",
        "gate",
        "depends_on",
        "site",
        "resource",
        "command_entry",
        "inputs",
        "outputs",
        "acceptance",
        "failure_action",
        "status",
    ]
    plan_path = tmp_path / "plans/experiment_steps.csv"
    plan_path.parent.mkdir(parents=True)
    with plan_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in (
            {"step_id": "E207", "status": "not_applicable"},
            {"step_id": "E223", "status": "not_applicable"},
            {"step_id": "E500", "status": "not_run"},
        ):
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    s1_rows = [
        _metric("S1", "S1-GRU", "seed1", "all", "repeat", "event_macro_f1", 0.71),
        _metric("S1", "S1-GRU", "seed2", "all", "repeat", "event_macro_f1", 0.72),
        _metric("S1", "S1-HGB", "seed1", "all", "repeat", "event_macro_f1", 0.67),
    ]
    s1_path = tmp_path / "results/S1/metrics.parquet"
    s1_path.parent.mkdir(parents=True)
    pd.DataFrame(s1_rows).to_parquet(s1_path, index=False)

    result = aggregate_results(tmp_path, {"step_id": "E500"}, {})

    assert result["status"] == "pass"
    assert result["details"]["fig4_source_row_count"] == 0
    assert result["details"]["fig4_panel_status"] == {
        "A_transfer": "not_applicable",
        "B_robustness": "not_applicable",
    }
    aggregate = pd.read_parquet(tmp_path / "results/final/aggregate.parquet")
    fig4 = pd.read_parquet(tmp_path / "results/final/fig4_data.parquet")
    assert set(aggregate["package_id"]) == {"S1"}
    assert fig4.empty
    assert "figure_panel" in fig4.columns


def test_run_confirmatory_stats_accepts_v4_s1_only_boundaries(tmp_path: Path) -> None:
    fieldnames = [
        "step_id",
        "phase",
        "gate",
        "depends_on",
        "site",
        "resource",
        "command_entry",
        "inputs",
        "outputs",
        "acceptance",
        "failure_action",
        "status",
    ]
    plan_path = tmp_path / "plans/experiment_steps.csv"
    plan_path.parent.mkdir(parents=True)
    with plan_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in (
            {"step_id": "E207", "status": "not_applicable"},
            {"step_id": "E223", "status": "not_applicable"},
            {"step_id": "E510", "status": "not_run"},
        ):
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    protocol = {
        "statistics": {"bootstrap_draws": 20, "confidence_level": 0.95},
        "seeds": {"statistics": 2026},
        "confirmatory_endpoints": {
            endpoint: {"minimum_meaningful_effect_pp": 1.0}
            for endpoint in (
                "V2",
                "S1",
                "E2_E3",
                "E3_RELIABILITY",
                "E3_GRAPH",
                "E3_MEMORY",
                "R1",
            )
        },
    }
    config = tmp_path / "configs/protocol_lock.pretest.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(yaml.safe_dump(protocol), encoding="utf-8")
    _write_json(
        tmp_path / "runs/S1/baselines/selection.json",
        {"selected_or_diagnostic_family_id": "S1-RULE"},
    )

    rows = []
    for repeat in ("seed1", "seed2", "seed3"):
        for group_index in range(3):
            rows.append(
                _metric(
                    "S1",
                    "S1-GRU",
                    repeat,
                    f"methane-{group_index}",
                    "group",
                    "sensor_event_f1",
                    0.50 + 0.01 * group_index,
                )
            )
    for group_index in range(3):
        rows.append(
            _metric(
                "S1",
                "S1-RULE",
                "static-rule",
                f"methane-{group_index}",
                "group",
                "sensor_event_f1",
                0.60 + 0.01 * group_index,
            )
        )
    for metric, value in (
        ("event_macro_f1", 0.61),
        ("false_alarms_per_hour", 0.0),
        ("event_miss_rate", 0.2),
    ):
        rows.append(
            _metric(
                "S1",
                "S1-RULE",
                "static-rule",
                "all",
                "repeat",
                metric,
                value,
            )
        )
    aggregate = pd.DataFrame(rows)
    aggregate_path = tmp_path / "results/final/aggregate.parquet"
    aggregate_path.parent.mkdir(parents=True)
    aggregate.to_parquet(aggregate_path, index=False)

    result = run_confirmatory_stats(tmp_path, {"step_id": "E510"}, {})

    assert result["status"] == "pass"
    stats = json.loads((tmp_path / "results/final/statistical_tests.json").read_text())
    assert stats["contrasts"]["S1"]["decision_status"] in {
        "support",
        "inconclusive",
        "negative",
    }
    assert stats["contrasts"]["V2"]["status"] == "not_applicable"
    assert stats["contrasts"]["R1"]["status"] == "not_applicable"

    for gate_id in ("G1", "G2", "G3", "G4", "G5", "G6"):
        _write_json(tmp_path / f"evidence/gates/{gate_id}.json", {"status": "pass"})
    _write_json(
        tmp_path / "evidence/data/dataset_source_decision.json",
        {"visual_direction_id": "v4-not-applicable"},
    )
    (tmp_path / "results/C1").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "module_id": "visible_inference",
                "family_id": "V2-A-10-MULTI",
                "eligibility_status": "not_applicable",
            },
            {
                "module_id": "sensor_inference",
                "family_id": "S1-GRU",
                "eligibility_status": "measured",
            },
        ]
    ).to_csv(tmp_path / "results/C1/efficiency_scope.csv", index=False)
    pd.DataFrame(
        [
            {
                "module_id": "sensor_inference",
                "variant_id": "S1-GRU-seed-2903",
                "latency_p50_ns": 500_000.0,
                "latency_p95_ns": 700_000.0,
                "peak_memory_bytes": 2**20,
                "parameter_count": 23873,
                "checkpoint_bytes": 100104,
            }
        ]
    ).to_parquet(tmp_path / "results/C1/efficiency.parquet", index=False)
    pd.DataFrame({"figure_panel": pd.Series(dtype="object")}).to_parquet(
        tmp_path / "results/final/fig4_data.parquet", index=False
    )
    paper = tmp_path / "manuscript/paper.md"
    paper.parent.mkdir(parents=True, exist_ok=True)
    paper.write_text(
        "效率 `【示意占位：EFF-HW=待测】`；"
        "`【示意占位：EFF-S-LAT=待测】`；"
        "`【示意占位：EFF-S-MEM=待测】`；"
        "`【示意占位：EFF-S-SIZE=待测】`；"
        "`【示意占位：EFF-V-LAT=待测】`；"
        "`【示意占位：S1-BASE=待填】`；"
        "`【示意占位：T1-VIS=待填】`；"
        "`【示意占位：V2-A-GENERIC=待填】`；"
        "`【示意占位：FIG4=待填】`。",
        encoding="utf-8",
    )
    adjudicate_claims(tmp_path, {"step_id": "E520"}, {})
    claims = pd.read_csv(tmp_path / "results/final/claim_status.csv")
    status_by_claim = dict(zip(claims["claim_id"], claims["status"]))
    assert status_by_claim["C-TRANSFER"] == "removed"
    assert status_by_claim["C-METHANE"] == "negative"
    assert status_by_claim["C-ROBUST"] == "removed"
    registry = json.loads((tmp_path / "results/final/manuscript_values.json").read_text())
    values = registry["values_by_full_token"]
    assert values["EFF-S-LAT=待测"] == "0.5/0.7 ms"
    assert values["EFF-V-LAT=待测"] == "不适用（资格门未通过）"
    assert values["S1-BASE=待填"].startswith("方法 S1-RULE")
    assert values["T1-VIS=待填"].startswith("未纳入 amendment v4")
    assert values["V2-A-GENERIC=待填"].startswith("不适用")
    assert values["FIG4=待填"].startswith("不适用")

    update_manuscript(tmp_path, {"step_id": "E530"}, {})
    fig4_contract = json.loads(
        (tmp_path / "figures/fig4_contract.json").read_text(encoding="utf-8")
    )
    assert fig4_contract["status"] == "scope_boundary_not_applicable"
    assert (tmp_path / "figures/fig4_transfer_robustness.svg").stat().st_size > 0


def test_artifact_audit_accepts_v4_not_applicable_boundaries(tmp_path: Path) -> None:
    fieldnames = [
        "step_id",
        "phase",
        "gate",
        "depends_on",
        "site",
        "resource",
        "command_entry",
        "inputs",
        "outputs",
        "acceptance",
        "failure_action",
        "status",
    ]
    rows = [
        {"step_id": "E112", "status": "not_applicable"},
        {"step_id": "E124", "status": "pass"},
        {"step_id": "E207", "status": "not_applicable"},
        {"step_id": "E223", "status": "not_applicable"},
        {"step_id": "E310", "status": "not_applicable"},
        {
            "step_id": "E402",
            "status": "not_run",
            "command_entry": "powershell -NoProfile -ExecutionPolicy Bypass -File tools/preflight/run_selected_python.ps1 -Module mining1_exp.cli audit-artifacts",
        },
    ]
    plan_path = tmp_path / "plans/experiment_steps.csv"
    plan_path.parent.mkdir(parents=True)
    with plan_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    for relative in (
        "evidence/gates/G4.json",
        "evidence/gates/G5.json",
        "data/releases/branch_test_release.json",
    ):
        _write_json(tmp_path / relative, {"status": "pass"})
    (tmp_path / "configs").mkdir(parents=True)
    (tmp_path / "configs/protocol_lock.pretest.yaml").write_text(
        yaml.safe_dump({"status": "locked"}), encoding="utf-8"
    )
    s1_metrics = tmp_path / "results/S1/metrics.parquet"
    s1_metrics.parent.mkdir(parents=True)
    pd.DataFrame([{"package_id": "S1", "metric_name": "event_macro_f1"}]).to_parquet(
        s1_metrics, index=False
    )
    fusion_lock = pd.DataFrame(
        [
            {
                "concept_id": "c0",
                "eligible_branch_type_count": 0,
                "eligible_branch_types": [],
                "episode_skeleton_candidate_hash": "a" * 64,
                "exclusion_reason": "amendment_v4_not_applicable",
                "fusion_primary_eligible": False,
                "graph_primary_eligible": False,
                "independent_multibranch_positive_group_count": 0,
                "multibranch_event_count_by_pool": {
                    "D_b_tr": 0,
                    "D_b_sel": 0,
                    "D_e_dev": 0,
                    "D_e_test": 0,
                },
                "ontology_lock_hash": "b" * 64,
                "split_manifest_hash": "c" * 64,
            }
        ]
    )
    fusion_path = tmp_path / "data/locked/fusion_eligibility_lock.parquet"
    fusion_path.parent.mkdir(parents=True)
    fusion_lock.to_parquet(fusion_path, index=False)
    for step_id in ("E112", "E207", "E223", "E310"):
        fallback = tmp_path / f"evidence/fallback/{step_id}_not_applicable.json"
        _write_json(fallback, {"status": "not_applicable"})
        _write_json(
            tmp_path / f"evidence/step_reviews/{step_id}.json",
            {
                "status": "not_applicable",
                "advance_allowed": True,
                "output_artifacts": [
                    {"path": fallback.relative_to(tmp_path).as_posix()}
                ],
            },
        )

    result = audit_artifacts(tmp_path, {"step_id": "E402"}, {})

    qa = json.loads((tmp_path / "results/C1/artifact_qa.json").read_text())
    required_paths = {item["path"] for item in qa["required_artifacts"]}
    assert result["status"] == "pass"
    assert "results/S1/metrics.parquet" in required_paths
    assert "results/T1/metrics.parquet" not in required_paths
    assert "results/V2/metrics.parquet" not in required_paths
    assert "results/R1/metrics.parquet" not in required_paths
