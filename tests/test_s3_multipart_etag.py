from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tools.data.compute_s3_multipart_etag import compute_hashes


def test_compute_s3_multipart_etag_and_file_hashes(tmp_path: Path) -> None:
    path = tmp_path / "fixture.bin"
    payload = b"abcdefghij"
    path.write_bytes(payload)
    result = compute_hashes(path, part_bytes=4)
    digests = [hashlib.md5(payload[index : index + 4]).digest() for index in range(0, 10, 4)]
    expected_etag = f"{hashlib.md5(b''.join(digests)).hexdigest()}-3"
    assert result == {
        "schema_version": 1,
        "path": str(path.resolve()),
        "byte_size": 10,
        "part_bytes": 4,
        "part_count": 3,
        "md5": hashlib.md5(payload).hexdigest(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "s3_multipart_etag": expected_etag,
    }


def test_compute_s3_multipart_etag_rejects_empty_input(tmp_path: Path) -> None:
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        compute_hashes(path, part_bytes=4)
