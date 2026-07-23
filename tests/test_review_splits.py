import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from mining1_exp.review_dedup import REPORT_COLUMNS
from mining1_exp.review_splits import review_e038
from mining1_exp.workflow_data import create_splits


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _fixture(root: Path) -> None:
    (root / "data/locked").mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "evidence/data").mkdir(parents=True)
    (root / "evidence/command_receipts").mkdir(parents=True)
    rows = []
    for index in range(8):
        rows.append(
            {
                "dataset_id": "target",
                "record_id": f"r{index}",
                "archive_id": "a-target",
                "relative_path": f"data/canonical/target/r{index}.dat",
                "modality": "visible",
                "raw_group_id": f"g{index}",
                "pair_id": "",
                "sequence_id": "",
                "timestamp_or_order": str(index),
                "label_summary_json": json.dumps({"class_ids": []}, sort_keys=True),
                "byte_size": 1,
                "sha256": _sha(f"target-{index}"),
            }
        )
    for cohort in range(8):
        timestamp = (pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=cohort)).isoformat()
        for sensor in ("MM263", "MM264", "MM256"):
            rows.append(
                {
                    "dataset_id": "methane",
                    "record_id": f"{sensor}-{cohort}",
                    "archive_id": "a-methane",
                    "relative_path": f"data/canonical/methane/{sensor}-{cohort}.dat",
                    "modality": "methane",
                    "raw_group_id": f"{sensor}-g{cohort}",
                    "pair_id": "",
                    "sequence_id": sensor,
                    "timestamp_or_order": timestamp,
                    "label_summary_json": json.dumps({"class_ids": []}, sort_keys=True),
                    "byte_size": 1,
                    "sha256": _sha(f"methane-{sensor}-{cohort}"),
                }
            )
    for index in range(8):
        for modality in ("visible", "infrared"):
            rows.append(
                {
                    "dataset_id": "rgbt",
                    "record_id": f"{modality}-{index}",
                    "archive_id": "a-rgbt",
                    "relative_path": f"data/canonical/rgbt/{modality}-{index}.png",
                    "modality": modality,
                    "raw_group_id": f"g{index}",
                    "pair_id": f"pair-{index}",
                    "sequence_id": f"seq-{index}",
                    "timestamp_or_order": str(index),
                    "label_summary_json": json.dumps({"class_ids": []}, sort_keys=True),
                    "byte_size": 1,
                    "sha256": _sha(f"rgbt-{modality}-{index}"),
                }
            )
    pd.DataFrame(rows).to_parquet(root / "data/locked/file_manifest.parquet", index=False)
    pd.DataFrame(columns=REPORT_COLUMNS).to_parquet(
        root / "data/locked/dedup_report.parquet", index=False
    )
    (root / "data/locked/ontology_lock.yaml").write_text(
        yaml.safe_dump({"ontology_version": "fixture-v1", "entries": []}), encoding="utf-8"
    )
    decision = {
        "roles": {
            "primary_visual_target": {"dataset_id": "target"},
            "primary_rgbt_dataset": {"dataset_id": "rgbt"},
            "methane_dataset": {
                "dataset_id": "methane",
                "adapter_contract": {
                    "methane": {"group_duration_seconds": 86400}
                },
            },
        }
    }
    (root / "evidence/data/dataset_source_decision.json").write_text(
        json.dumps(decision), encoding="utf-8"
    )
    protocol = {
        "seeds": {"split": 13007},
        "data": {
            "target_role_family_weights": {"branch": 0.5, "episode": 0.5},
            "branch_pool_weights": {
                "D_b_tr": 0.60,
                "D_b_sel": 0.15,
                "D_b_prob": 0.10,
                "D_b_te": 0.15,
            },
            "episode_pool_weights": {
                "D_e_tr": 0.50,
                "D_e_sel": 0.15,
                "D_e_pol": 0.15,
                "D_e_te": 0.20,
            },
            "methane": {"purge_gap_seconds": 600},
        },
    }
    (root / "configs/protocol_lock.template.yaml").write_text(
        yaml.safe_dump(protocol, sort_keys=True), encoding="utf-8"
    )
    result = create_splits(root, {"step_id": "E038"}, {"grouped": True})
    output = root / "data/locked/split_manifest.parquet"
    receipt = {
        "schema_version": 1,
        "step_id": "E038",
        "status": "pass",
        "command": "create-splits",
        "arguments": {"grouped": True},
        "inputs": result["inputs"],
        "outputs": [
            {
                "path": "data/locked/split_manifest.parquet",
                "kind": "file",
                "bytes": output.stat().st_size,
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            }
        ],
        "details": result["details"],
    }
    (root / "evidence/command_receipts/E038.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )


def test_review_e038_accepts_balanced_group_disjoint_split(tmp_path: Path) -> None:
    _fixture(tmp_path)
    result = review_e038(tmp_path)
    assert result["status"] == "pass"
    assert result["error_count"] == 0
    assert result["diagnostics"]["pair_integrity_error_count"] == 0
    for role in result["diagnostics"]["required_role_pool_coverage"].values():
        assert role["missing_pools"] == []


def test_review_e038_rejects_tampered_component_pool(tmp_path: Path) -> None:
    _fixture(tmp_path)
    path = tmp_path / "data/locked/split_manifest.parquet"
    split = pd.read_parquet(path)
    target = split["dataset_id"].eq("target") & split["pool"].eq("D_b_te")
    split.loc[target, "pool"] = "D_b_tr"
    split.to_parquet(path, index=False)
    result = review_e038(tmp_path)
    assert result["status"] == "fail"
    assert any("lacks pools" in error for error in result["errors"])
    assert any("dataset_pool_component_counts" in error for error in result["errors"])


@pytest.mark.parametrize(
    ("field", "replacement", "error_fragment"),
    [
        ("algorithm", "unreviewed", "receipt.balance.algorithm"),
        (
            "cross_dataset_component_count",
            1,
            "receipt.balance.cross_dataset_component_count",
        ),
        (
            "global_pool_component_counts",
            {},
            "receipt.balance.global_pool_component_counts",
        ),
        (
            "cross_component_search_nodes",
            0,
            "receipt.balance.cross_component_search_nodes",
        ),
    ],
)
def test_review_e038_rejects_tampered_balance_diagnostics(
    tmp_path: Path,
    field: str,
    replacement: object,
    error_fragment: str,
) -> None:
    _fixture(tmp_path)
    receipt_path = tmp_path / "evidence/command_receipts/E038.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["details"]["balance_diagnostics"][field] = replacement
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    result = review_e038(tmp_path)

    assert result["status"] == "fail"
    assert any(error_fragment in error for error in result["errors"])


def test_review_e038_reports_record_key_mismatch_without_crashing(tmp_path: Path) -> None:
    _fixture(tmp_path)
    path = tmp_path / "data/locked/split_manifest.parquet"
    split = pd.read_parquet(path).iloc[:-1].copy()
    split.to_parquet(path, index=False)

    result = review_e038(tmp_path)

    assert result["status"] == "fail"
    assert any("record_key_set" in error for error in result["errors"])


def test_review_e038_rejects_methane_cohort_split_across_pools(tmp_path: Path) -> None:
    _fixture(tmp_path)
    path = tmp_path / "data/locked/split_manifest.parquet"
    split = pd.read_parquet(path)
    target = split["dataset_id"].eq("methane") & split["record_id"].eq("MM263-0")
    split.loc[target, "pool"] = "D_e_te"
    split.to_parquet(path, index=False)

    result = review_e038(tmp_path)

    assert result["status"] == "fail"
    assert any("methane cohort" in error for error in result["errors"])


def test_review_e038_rejects_tampered_temporal_diagnostics(tmp_path: Path) -> None:
    _fixture(tmp_path)
    receipt_path = tmp_path / "evidence/command_receipts/E038.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["details"]["balance_diagnostics"]["temporal_cohort_diagnostics"][
        "minimum_family_gap_seconds"
    ] = 0
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    result = review_e038(tmp_path)

    assert result["status"] == "fail"
    assert any("temporal_cohort_diagnostics" in error for error in result["errors"])
