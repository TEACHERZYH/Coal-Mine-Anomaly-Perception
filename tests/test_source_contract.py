from __future__ import annotations

import json
from pathlib import Path

from mining1_exp.data.source_contract import (
    DatasetSourceContractError,
    archive_parts_from_entry,
    archive_set_sha256,
)
from mining1_exp.provenance import (
    canonical_json_sha256,
    sha256_file,
    validate_source_lock_amendment,
)
import pytest


def test_archive_parts_normalize_primary_and_companions() -> None:
    entry = {
        "source_path_or_url": "images.zip",
        "archive_sha256": "1" * 64,
        "transfer_route": "local_existing_upload",
        "companion_archives": [
            {
                "archive_part_id": "annotations",
                "source_path_or_url": "annotations.zip",
                "archive_sha256": "2" * 64,
            }
        ],
    }
    parts = archive_parts_from_entry(entry)
    assert [part["archive_part_id"] for part in parts] == ["primary", "annotations"]
    assert parts[1]["transfer_route"] == "local_existing_upload"
    assert archive_set_sha256(parts) == archive_set_sha256(list(reversed(parts)))


def test_archive_parts_reject_ambiguous_companion_ids() -> None:
    entry = {
        "source_path_or_url": "images.zip",
        "archive_sha256": "1" * 64,
        "transfer_route": "local_existing_upload",
        "companion_archives": [
            {
                "archive_part_id": "primary",
                "source_path_or_url": "annotations.zip",
                "archive_sha256": "2" * 64,
            }
        ],
    }
    with pytest.raises(DatasetSourceContractError, match="Duplicate or empty"):
        archive_parts_from_entry(entry)


def test_source_lock_amendment_binds_and_detects_source_drift(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "evidence/pretest").mkdir(parents=True)
    (tmp_path / "mining1_exp").mkdir()
    protocol = tmp_path / "configs/protocol_lock.pretest.yaml"
    protocol.write_text("schema_version: 1\n", encoding="utf-8")
    source = tmp_path / "mining1_exp/example.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    files = [{"path": "mining1_exp/example.py", "sha256": sha256_file(source)}]
    manifest = {
        "schema_version": 1,
        "files": files,
        "source_code_sha256": canonical_json_sha256(files),
    }
    manifest_path = tmp_path / "evidence/pretest/source_code_hashes.amendment_v2.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    amendment = {
        "schema_version": 1,
        "amendment_id": "fixture-v2",
        "status": "locked",
        "base_protocol_sha256": sha256_file(protocol),
        "source_manifest_path": manifest_path.relative_to(tmp_path).as_posix(),
        "source_manifest_sha256": sha256_file(manifest_path),
        "amended_source_code_sha256": manifest["source_code_sha256"],
    }
    (tmp_path / "configs/source_lock_amendment.v2.json").write_text(
        json.dumps(amendment), encoding="utf-8"
    )
    binding = validate_source_lock_amendment(tmp_path)
    assert binding is not None
    assert binding["source_code_sha256"] == manifest["source_code_sha256"]
    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source drifted"):
        validate_source_lock_amendment(tmp_path)
