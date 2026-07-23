import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from mining1_exp.provenance import sha256_file
from mining1_exp.review_graph_eligibility import review_e054
from mining1_exp.workflow_data import audit_graph_eligibility, audit_rgbt_inputs


def _fixture(root: Path) -> None:
    (root / "data/locked").mkdir(parents=True)
    (root / "evidence/data").mkdir(parents=True)
    (root / "evidence/command_receipts").mkdir(parents=True)
    (root / "configs").mkdir(parents=True)
    rows = []
    for index in range(4):
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
                    "pool": "D_b_tr" if index < 2 else "D_e_tr",
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
            {
                "data": {
                    "independent_group_floor_hard_min": 3,
                    "rgbt": {"physical_temperature_claim_enabled": False},
                }
            }
        ),
        encoding="utf-8",
    )
    ontology = {
        "ontology_version": "fixture-v1",
        "entries": [
            {
                "dataset_id": "rgbt",
                "source_label": "person",
                "canonical_concept_id": "worker_presence",
                "mapping_status": "compatible",
                "annotation_policy": "fixture boxes are exhaustive",
                "event_semantics": "observable_presence",
                "negative_semantics": "exhaustive_verified_absence",
                "allowed_tasks": ["confirmatory_f1"],
                "evidence_reference": "fixture:rgbt",
                "reviewer_decision": "accept for contract test",
            }
        ],
    }
    (root / "data/locked/ontology_lock.yaml").write_text(
        yaml.safe_dump(ontology, sort_keys=True), encoding="utf-8"
    )
    audit_rgbt_inputs(root, {"step_id": "E046"}, {})
    result = audit_graph_eligibility(root, {"step_id": "E054"}, {})
    graph_path = root / "data/locked/graph_eligibility_lock.json"
    receipt = {
        "schema_version": 1,
        "step_id": "E054",
        "status": "pass",
        "command": "audit-graph-eligibility",
        "arguments": {},
        "inputs": result["inputs"],
        "outputs": [
            {
                "path": "data/locked/graph_eligibility_lock.json",
                "kind": "file",
                "bytes": graph_path.stat().st_size,
                "sha256": sha256_file(graph_path),
            }
        ],
        "details": result["details"],
    }
    (root / "evidence/command_receipts/E054.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )


def test_review_e054_reconstructs_graph_eligibility(tmp_path: Path) -> None:
    _fixture(tmp_path)

    result = review_e054(tmp_path)

    assert result["status"] == "pass"
    assert result["error_count"] == 0
    assert result["diagnostics"]["graph_eligible"] is True
    assert result["diagnostics"]["independent_raw_group_count"] == 4
    assert result["diagnostics"]["compatible_concept_ids"] == ["worker_presence"]


def test_review_e054_rejects_tampered_eligibility(tmp_path: Path) -> None:
    _fixture(tmp_path)
    path = tmp_path / "data/locked/graph_eligibility_lock.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["graph_eligible"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = review_e054(tmp_path)

    assert result["status"] == "fail"
    assert any("graph_eligibility_lock" in error for error in result["errors"])


def test_review_e054_rejects_pair_crossing_pools(tmp_path: Path) -> None:
    _fixture(tmp_path)
    path = tmp_path / "data/locked/split_manifest.parquet"
    split = pd.read_parquet(path)
    split.loc[split["record_id"].eq("infrared-0"), "pool"] = "D_e_te"
    split.to_parquet(path, index=False)

    result = review_e054(tmp_path)

    assert result["status"] == "fail"
    assert any("pair contract errors" in error for error in result["errors"])


def test_review_e054_rejects_removed_compatible_mapping(tmp_path: Path) -> None:
    _fixture(tmp_path)
    path = tmp_path / "data/locked/ontology_lock.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["entries"][0]["mapping_status"] = "reject"
    payload["entries"][0]["canonical_concept_id"] = None
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    result = review_e054(tmp_path)

    assert result["status"] == "fail"
    assert result["diagnostics"]["compatible_concept_ids"] == []
    assert result["diagnostics"]["graph_eligible"] is False
