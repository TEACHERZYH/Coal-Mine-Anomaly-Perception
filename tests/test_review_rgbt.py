import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from mining1_exp.provenance import sha256_file
from mining1_exp.review_rgbt import review_e046
from mining1_exp.workflow_data import audit_rgbt_inputs


POOLS = [
    "D_b_tr",
    "D_b_sel",
    "D_b_prob",
    "D_b_te",
    "D_e_tr",
    "D_e_sel",
    "D_e_pol",
    "D_e_te",
]


def _fixture(root: Path) -> None:
    (root / "data/locked").mkdir(parents=True)
    (root / "evidence/data").mkdir(parents=True)
    (root / "evidence/command_receipts").mkdir(parents=True)
    (root / "configs").mkdir(parents=True)
    rows = []
    for index, pool in enumerate(POOLS):
        for modality in ("visible", "infrared"):
            record_id = f"{modality}-{index}"
            rows.append(
                {
                    "dataset_id": "rgbt",
                    "record_id": record_id,
                    "archive_id": "rgbt-archive",
                    "relative_path": f"rgbt/{record_id}.png",
                    "modality": modality,
                    "raw_group_id": f"g{index}",
                    "pair_id": f"pair-{index}",
                    "sequence_id": f"seq-{index}",
                    "timestamp_or_order": str(index),
                    "label_summary_json": json.dumps({"class_ids": ["person"]}),
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
    (root / "evidence/data/dataset_source_decision.json").write_text(
        json.dumps({"roles": {"primary_rgbt_dataset": {"dataset_id": "rgbt"}}}),
        encoding="utf-8",
    )
    (root / "configs/protocol_lock.template.yaml").write_text(
        yaml.safe_dump(
            {"data": {"rgbt": {"physical_temperature_claim_enabled": False}}}
        ),
        encoding="utf-8",
    )
    result = audit_rgbt_inputs(root, {"step_id": "E046"}, {})
    lock_path = root / "data/locked/rgbt_input_lock.json"
    receipt = {
        "schema_version": 1,
        "step_id": "E046",
        "status": "pass",
        "command": "audit-rgbt-inputs",
        "arguments": {},
        "inputs": result["inputs"],
        "outputs": [
            {
                "path": "data/locked/rgbt_input_lock.json",
                "kind": "file",
                "bytes": lock_path.stat().st_size,
                "sha256": sha256_file(lock_path),
            }
        ],
        "details": result["details"],
    }
    (root / "evidence/command_receipts/E046.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )


def test_review_e046_accepts_observed_infrared_as_thermal_branch(tmp_path: Path) -> None:
    _fixture(tmp_path)

    result = review_e046(tmp_path)

    assert result["status"] == "pass"
    assert result["error_count"] == 0
    assert result["diagnostics"]["observed_source_modalities"] == ["infrared", "visible"]
    assert result["diagnostics"]["required_branch_roles"] == ["thermal", "visible"]
    assert result["diagnostics"]["physical_temperature_claim_enabled"] is False


def test_review_e046_rejects_tampered_modality_claim(tmp_path: Path) -> None:
    _fixture(tmp_path)
    path = tmp_path / "data/locked/rgbt_input_lock.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["observed_source_modalities"] = ["thermal", "visible"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = review_e046(tmp_path)

    assert result["status"] == "fail"
    assert any("rgbt_input_lock" in error for error in result["errors"])


def test_review_e046_rejects_pair_crossing_pools(tmp_path: Path) -> None:
    _fixture(tmp_path)
    path = tmp_path / "data/locked/split_manifest.parquet"
    split = pd.read_parquet(path)
    split.loc[split["record_id"].eq("infrared-0"), "pool"] = "D_e_te"
    split.to_parquet(path, index=False)

    result = review_e046(tmp_path)

    assert result["status"] == "fail"
    assert any("pair provenance or pool errors" in error for error in result["errors"])
