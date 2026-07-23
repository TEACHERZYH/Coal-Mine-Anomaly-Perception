from __future__ import annotations

from pathlib import Path


def test_range_downloader_binds_resume_parts_and_validates_content_range() -> None:
    script = (
        Path(__file__).resolve().parents[1] / "tools/data/download_figshare_ranges.ps1"
    ).read_text(encoding="utf-8-sig")
    assert "schema_version = 2" in script
    assert "downloader_sha256" in script
    assert "different downloader contract" in script
    assert "range artifacts lack a downloader contract" in script
    assert "Assert-FragmentResponse" in script
    assert "response is not HTTP 206" in script
    assert "Content-Range mismatch" in script
