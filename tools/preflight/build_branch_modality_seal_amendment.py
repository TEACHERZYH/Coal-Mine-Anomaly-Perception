from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pandas as pd

from mining1_exp.data.seals import (
    DENIED_LABEL_ROLES,
    build_candidate_test_seal,
    validate_test_seal,
)
from mining1_exp.governance.immutable import write_once_json
from mining1_exp.provenance import sha256_file
from mining1_exp.workflow_common import utc_now, write_parquet_artifact
from mining1_exp.workflow_governance import _branch_truth_rows, _source_hash


ROOT = Path(__file__).resolve().parents[2]
AMENDMENT_ID = "E100-E120-source-lock-amendment-v2"
ARTIFACTS = {
    "feature_store": "data/locked/branch_test_features.amendment_v2.parquet",
    "feature_manifest": "data/locked/branch_test_feature_manifest.amendment_v2.json",
    "detection_truth": "data/sealed/branch_detection_truth.amendment_v2.parquet",
    "concept_truth": "data/sealed/branch_concept_truth.amendment_v2.parquet",
    "truth_manifest": "data/sealed/branch_test_truth_manifest.amendment_v2.json",
    "candidate_seal": "data/seals/branch_test_seal_candidate.amendment_v2.json",
    "final_seal": "data/seals/branch_test_seal.amendment_v2.json",
}
RECEIPT = "evidence/amendments/branch_test_seal_v2_build.json"


def _artifact(root: Path, key: str) -> Path:
    return root / ARTIFACTS[key]


def build(root: Path = ROOT) -> dict[str, object]:
    root = root.resolve()
    receipt_path = root / RECEIPT
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
        for item in receipt.get("outputs", []):
            path = root / str(item["path"])
            if not path.is_file() or sha256_file(path) != item["sha256"]:
                raise RuntimeError("existing branch-seal amendment output drifted")
        return receipt
    features, detection_truth, concept_truth = _branch_truth_rows(root)
    source_modalities = set(features["modality"].astype(str).str.lower())
    if "infrared" not in source_modalities or "visible" not in source_modalities:
        raise RuntimeError("amended branch seal lacks visible or infrared source records")
    methane_features_path = root / "data/locked/branch_methane_test_features.parquet"
    methane_truth_path = root / "data/sealed/branch_methane_truth.parquet"
    methane_features = pd.read_parquet(methane_features_path)
    methane_truth = pd.read_parquet(methane_truth_path)
    if (
        methane_features.empty
        or methane_truth.empty
        or set(methane_features["pool"].astype(str)) != {"D_b_te"}
        or set(methane_truth["pool"].astype(str)) != {"D_b_te"}
    ):
        raise RuntimeError("branch methane test partitions are incomplete")
    write_parquet_artifact(_artifact(root, "feature_store"), features)
    write_parquet_artifact(_artifact(root, "detection_truth"), detection_truth)
    write_parquet_artifact(_artifact(root, "concept_truth"), concept_truth)
    feature_manifest = {
        "schema_version": 1,
        "scope": "branch",
        "pool": "D_b_te",
        "amendment_id": AMENDMENT_ID,
        "detection_features": {
            "path": ARTIFACTS["feature_store"],
            "sha256": sha256_file(_artifact(root, "feature_store")),
            "row_count": len(features),
        },
        "methane_features": {
            "path": methane_features_path.relative_to(root).as_posix(),
            "sha256": sha256_file(methane_features_path),
            "row_count": len(methane_features),
        },
    }
    write_once_json(_artifact(root, "feature_manifest"), feature_manifest)
    truth_manifest = {
        "schema_version": 1,
        "scope": "branch",
        "pool": "D_b_te",
        "amendment_id": AMENDMENT_ID,
        "detection_truth": {
            "path": ARTIFACTS["detection_truth"],
            "sha256": sha256_file(_artifact(root, "detection_truth")),
            "row_count": len(detection_truth),
        },
        "concept_truth": {
            "path": ARTIFACTS["concept_truth"],
            "sha256": sha256_file(_artifact(root, "concept_truth")),
            "row_count": len(concept_truth),
        },
        "methane_truth": {
            "path": methane_truth_path.relative_to(root).as_posix(),
            "sha256": sha256_file(methane_truth_path),
            "row_count": len(methane_truth),
        },
        "record_count": len(features),
        "raw_group_count": int(features["raw_group_id"].nunique()),
    }
    write_once_json(_artifact(root, "truth_manifest"), truth_manifest)
    candidate = build_candidate_test_seal(
        scope="branch",
        split_hash=sha256_file(root / "data/locked/split_manifest.parquet"),
        feature_manifest_hash=sha256_file(_artifact(root, "feature_manifest")),
        label_manifest_hash=sha256_file(_artifact(root, "truth_manifest")),
        artifact_contract_hash=sha256_file(
            root / "configs/artifact_contract.template.yaml"
        ),
        feature_acl=["allow:test_predictor", "allow:evaluator"],
        label_acl=sorted(DENIED_LABEL_ROLES),
    )
    candidate.update(
        {
            "amendment_id": AMENDMENT_ID,
            "feature_manifest_path": ARTIFACTS["feature_manifest"],
            "label_manifest_path": ARTIFACTS["truth_manifest"],
        }
    )
    validate_test_seal(candidate)
    write_once_json(_artifact(root, "candidate_seal"), candidate)
    final = deepcopy(candidate)
    final.update(
        {
            "stage": "final",
            "sealed_at": utc_now(),
            "candidate_seal_hash": sha256_file(_artifact(root, "candidate_seal")),
            "protocol_hash": sha256_file(root / "configs/protocol_lock.pretest.yaml"),
            "source_hash": _source_hash(root),
            "matrix_hash": sha256_file(root / "configs/experiment_matrix.template.csv"),
        }
    )
    validate_test_seal(final)
    write_once_json(_artifact(root, "final_seal"), final)
    outputs = [
        {
            "path": path,
            "sha256": sha256_file(root / path),
        }
        for path in ARTIFACTS.values()
    ]
    receipt = {
        "schema_version": 1,
        "amendment_id": AMENDMENT_ID,
        "status": "pass",
        "created_at": utc_now(),
        "base_seal_path": "data/seals/branch_test_seal.json",
        "base_seal_sha256": sha256_file(root / "data/seals/branch_test_seal.json"),
        "source_modalities": sorted(source_modalities),
        "detection_feature_count": len(features),
        "detection_truth_count": len(detection_truth),
        "concept_truth_count": len(concept_truth),
        "outputs": outputs,
        "test_labels_released": False,
        "performance_claims_authorized": False,
    }
    write_once_json(receipt_path, receipt)
    return receipt


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
