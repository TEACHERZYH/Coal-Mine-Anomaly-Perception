from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mining1_exp.remote_transport import (
    RemoteTransportError,
    run_sftp_reput,
)


def test_sftp_reput_uses_resume_batch_and_current_host(tmp_path, monkeypatch) -> None:
    source = tmp_path / "sample.bin"
    source.write_bytes(b"sample")
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["input"] = kwargs["input"].decode("utf-8")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr("mining1_exp.remote_transport.subprocess.run", fake_run)
    result = run_sftp_reput(
        source,
        "/data/home/xinxi-zhyh/xinxi-zhyh/datasets/mining1/sample.bin.part",
    )
    assert result.returncode == 0
    assert observed["command"][0:3] == ["sftp", "-b", "-"]
    assert observed["command"][-1] == "xinxi-zhyh@211.87.115.228"
    assert observed["input"].startswith('reput "')
    assert observed["input"].endswith('"/data/home/xinxi-zhyh/xinxi-zhyh/datasets/mining1/sample.bin.part"\n')


def test_sftp_reput_rejects_unsafe_remote_target(tmp_path) -> None:
    source = tmp_path / "sample.bin"
    source.write_bytes(b"sample")
    with pytest.raises(RemoteTransportError, match="outside the project home"):
        run_sftp_reput(source, "/tmp/sample.bin")
