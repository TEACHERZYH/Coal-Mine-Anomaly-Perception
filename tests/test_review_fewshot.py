import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from mining1_exp.review_fewshot import review_e042
from mining1_exp.workflow_data import create_fewshot


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _fixture(root: Path) -> None:
    (root / "data/locked").mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "evidence/data").mkdir(parents=True)
    (root / "evidence/command_receipts").mkdir(parents=True)
    manifest_rows = []
    split_rows = []
    for dataset_id in ("target", "source"):
        for index in range(21):
            group = f"g{index:02d}"
            record = f"{dataset_id}-r{index:02d}"
            manifest_rows.append(
                {
                    "dataset_id": dataset_id,
                    "record_id": record,
                    "archive_id": f"a-{dataset_id}",
                    "relative_path": f"data/{record}.dat",
                    "modality": "visible",
                    "raw_group_id": group,
                    "pair_id": "",
                    "sequence_id": "",
                    "timestamp_or_order": str(index),
                    "label_summary_json": json.dumps(
                        {"class_ids": ["person" if index % 2 else "miner"]},
                        sort_keys=True,
                    ),
                    "byte_size": 1,
                    "sha256": _sha(record),
                }
            )
            split_rows.append(
                {
                    "dataset_id": dataset_id,
                    "record_id": record,
                    "raw_group_id": group,
                    "pool": "D_b_tr" if index < 20 else "D_b_te",
                    "split_seed": 13007,
                    "split_version": "fixture-v1",
                    "ontology_hash": _sha("ontology"),
                    "dedup_report_hash": _sha("dedup"),
                }
            )
    pd.DataFrame(manifest_rows).to_parquet(
        root / "data/locked/file_manifest.parquet", index=False
    )
    split_path = root / "data/locked/split_manifest.parquet"
    pd.DataFrame(split_rows).to_parquet(split_path, index=False)
    decision = {"visual_direction_id": "source_to_target"}
    (root / "evidence/data/dataset_source_decision.json").write_text(
        json.dumps(decision), encoding="utf-8"
    )
    protocol = {"seeds": {"fewshot_subset": [5171, 6197, 7331]}}
    (root / "configs/protocol_lock.template.yaml").write_text(
        yaml.safe_dump(protocol), encoding="utf-8"
    )
    result = create_fewshot(
        root, {"step_id": "E042"}, {"ratio": 10, "pairs": 3}
    )
    output = root / "data/locked/fewshot_manifest.parquet"
    receipt = {
        "schema_version": 1,
        "step_id": "E042",
        "status": "pass",
        "command": "create-fewshot",
        "arguments": {"pairs": 3, "ratio": 10},
        "inputs": result["inputs"],
        "outputs": [
            {
                "path": "data/locked/fewshot_manifest.parquet",
                "kind": "file",
                "bytes": output.stat().st_size,
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            }
        ],
        "details": result["details"],
    }
    (root / "evidence/command_receipts/E042.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )


def test_review_e042_accepts_exact_three_seed_group_subsets(tmp_path: Path) -> None:
    _fixture(tmp_path)

    result = review_e042(tmp_path)

    assert result["status"] == "pass"
    assert result["error_count"] == 0
    assert result["diagnostics"]["dataset_count"] == 2
    for details in result["diagnostics"]["per_dataset_seed"].values():
        assert details["candidate_group_count"] == 20
        assert details["included_group_count"] == 2


def test_review_e042_rejects_tampered_included_group(tmp_path: Path) -> None:
    _fixture(tmp_path)
    path = tmp_path / "data/locked/fewshot_manifest.parquet"
    frame = pd.read_parquet(path)
    rows = frame.index[
        (frame["dataset_id"] == "target") & (frame["subset_seed"] == 5171)
    ]
    included = next(index for index in rows if bool(frame.at[index, "included"]))
    excluded = next(index for index in rows if not bool(frame.at[index, "included"]))
    frame.at[included, "included"] = False
    frame.at[excluded, "included"] = True
    frame.to_parquet(path, index=False)

    result = review_e042(tmp_path)

    assert result["status"] == "fail"
    assert any("included group set mismatch" in error for error in result["errors"])


def test_review_e042_rejects_tampered_class_counts(tmp_path: Path) -> None:
    _fixture(tmp_path)
    path = tmp_path / "data/locked/fewshot_manifest.parquet"
    frame = pd.read_parquet(path)
    frame.loc[frame["dataset_id"] == "target", "class_group_counts_json"] = "{}"
    frame.to_parquet(path, index=False)

    result = review_e042(tmp_path)

    assert result["status"] == "fail"
    assert any("class group counts mismatch" in error for error in result["errors"])


@pytest.mark.parametrize(
    ("field", "value", "fragment"),
    [
        ("arguments", {"pairs": 2, "ratio": 10}, "arguments"),
        ("details", {}, "details"),
        ("status", "fail", "step or status"),
    ],
)
def test_review_e042_rejects_tampered_receipt(
    tmp_path: Path, field: str, value: object, fragment: str
) -> None:
    _fixture(tmp_path)
    path = tmp_path / "evidence/command_receipts/E042.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt[field] = value
    path.write_text(json.dumps(receipt), encoding="utf-8")

    result = review_e042(tmp_path)

    assert result["status"] == "fail"
    assert any(fragment in error for error in result["errors"])
