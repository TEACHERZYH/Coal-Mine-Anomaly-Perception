import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from mining1_exp.provenance import sha256_file
from mining1_exp.review_methane_roles import review_e050
from mining1_exp.workflow_data import lock_methane_roles


POOL_ORDER = [
    "D_b_tr",
    "D_e_tr",
    "D_b_sel",
    "D_e_sel",
    "D_b_prob",
    "D_e_pol",
    "D_b_te",
    "D_e_te",
]


def _fixture(root: Path) -> None:
    (root / "data/locked").mkdir(parents=True)
    (root / "evidence/data").mkdir(parents=True)
    (root / "evidence/command_receipts").mkdir(parents=True)
    (root / "configs").mkdir(parents=True)
    rows = []
    for cohort, pool in enumerate(POOL_ORDER):
        timestamp = (pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=cohort)).isoformat()
        for sensor in ("MM263", "MM264", "MM256"):
            record_id = f"{sensor}-{cohort}"
            rows.append(
                {
                    "dataset_id": "methane",
                    "record_id": record_id,
                    "archive_id": "methane-archive",
                    "relative_path": f"intentionally-absent/{record_id}.parquet",
                    "modality": "methane",
                    "raw_group_id": record_id,
                    "pair_id": "",
                    "sequence_id": sensor,
                    "timestamp_or_order": timestamp,
                    "label_summary_json": json.dumps({"class_ids": []}),
                    "byte_size": 1,
                    "sha256": hashlib.sha256(record_id.encode("utf-8")).hexdigest(),
                    "pool": pool,
                }
            )
    frame = pd.DataFrame(rows)
    frame.drop(columns="pool").to_parquet(
        root / "data/locked/file_manifest.parquet", index=False
    )
    split = frame[["dataset_id", "record_id", "raw_group_id", "pool"]].copy()
    split["split_seed"] = 13007
    split["split_version"] = "fixture"
    split["ontology_hash"] = "a" * 64
    split["dedup_report_hash"] = "b" * 64
    split.to_parquet(root / "data/locked/split_manifest.parquet", index=False)
    decision = {
        "roles": {
            "methane_dataset": {
                "dataset_id": "methane",
                "adapter_contract": {
                    "methane": {"group_duration_seconds": 86400}
                },
            }
        }
    }
    (root / "evidence/data/dataset_source_decision.json").write_text(
        json.dumps(decision), encoding="utf-8"
    )
    protocol = {
        "data": {
            "methane": {
                "history_seconds": 300,
                "horizon_seconds": 300,
                "stride_seconds": 30,
                "purge_gap_seconds": 600,
                "imputation_fit_pool": "D_b_tr",
                "normalization_fit_pool": "D_b_tr",
            }
        }
    }
    (root / "configs/protocol_lock.template.yaml").write_text(
        yaml.safe_dump(protocol), encoding="utf-8"
    )
    result = lock_methane_roles(root, {"step_id": "E050"}, {})
    lock_path = root / "data/locked/methane_role_lock.json"
    receipt = {
        "schema_version": 1,
        "step_id": "E050",
        "status": "pass",
        "command": "lock-methane-roles",
        "arguments": {},
        "inputs": result["inputs"],
        "outputs": [
            {
                "path": "data/locked/methane_role_lock.json",
                "kind": "file",
                "bytes": lock_path.stat().st_size,
                "sha256": sha256_file(lock_path),
            }
        ],
        "details": result["details"],
    }
    (root / "evidence/command_receipts/E050.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )


def test_review_e050_accepts_temporal_lock_without_opening_canonical_data(
    tmp_path: Path,
) -> None:
    _fixture(tmp_path)

    result = review_e050(tmp_path)

    assert result["status"] == "pass"
    assert result["error_count"] == 0
    assert result["diagnostics"]["canonical_data_files_opened"] == 0
    assert result["diagnostics"]["same_cohort_cross_pool_count"] == 0
    assert result["diagnostics"]["minimum_family_purge_gap_seconds"] >= 600


def test_review_e050_rejects_tampered_purge_claim(tmp_path: Path) -> None:
    _fixture(tmp_path)
    path = tmp_path / "data/locked/methane_role_lock.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["minimum_family_purge_gap_seconds"] = 0
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = review_e050(tmp_path)

    assert result["status"] == "fail"
    assert any("methane_role_lock" in error for error in result["errors"])


def test_review_e050_rejects_same_cohort_crossing_pools(tmp_path: Path) -> None:
    _fixture(tmp_path)
    path = tmp_path / "data/locked/split_manifest.parquet"
    split = pd.read_parquet(path)
    split.loc[split["record_id"].eq("MM263-0"), "pool"] = "D_e_te"
    split.to_parquet(path, index=False)

    result = review_e050(tmp_path)

    assert result["status"] == "fail"
    assert any("same-cohort cross-pool" in error for error in result["errors"])
