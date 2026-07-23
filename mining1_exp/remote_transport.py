from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
from typing import Sequence, Union


PathLike = Union[str, Path]
CURRENT_REMOTE_HOST = "xinxi-zhyh@211.87.115.228"
HOST_PATTERN = re.compile(r"^[A-Za-z0-9._-]+@[A-Za-z0-9.:-]+$")


class RemoteTransportError(RuntimeError):
    """Raised when a bounded current-host transport action fails."""


@dataclass(frozen=True)
class RemoteCommandResult:
    returncode: int
    stdout: str
    stderr: str


def _require_current_host(host: str) -> str:
    if host != CURRENT_REMOTE_HOST or HOST_PATTERN.fullmatch(host) is None:
        raise RemoteTransportError(f"Remote host is not permitted: {host}")
    return host


def run_ssh_script(
    script: str,
    *,
    host: str = CURRENT_REMOTE_HOST,
    timeout_seconds: int = 90,
) -> RemoteCommandResult:
    _require_current_host(host)
    if timeout_seconds <= 0 or timeout_seconds > 3600:
        raise RemoteTransportError("SSH timeout is outside the bounded range")
    command = script.replace("\r\n", "\n").replace("\r", "\n").rstrip() + "\n"
    process = subprocess.Popen(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectionAttempts=1",
            "-o",
            "ConnectTimeout=8",
            "-o",
            "ServerAliveInterval=5",
            "-o",
            "ServerAliveCountMax=1",
            host,
            "bash",
            "-s",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        stdout_bytes, stderr_bytes = process.communicate(
            command.encode("utf-8"), timeout=timeout_seconds
        )
    except subprocess.TimeoutExpired as exc:
        process.kill()
        process.communicate()
        raise RemoteTransportError(
            f"SSH action exceeded {timeout_seconds} seconds"
        ) from exc
    result = RemoteCommandResult(
        returncode=int(process.returncode),
        stdout=stdout_bytes.decode("utf-8", errors="replace"),
        stderr=stderr_bytes.decode("utf-8", errors="replace"),
    )
    if result.returncode != 0:
        detail = "\n".join((result.stdout + result.stderr).splitlines()[-8:])
        raise RemoteTransportError(
            f"SSH action failed with exit code {result.returncode}: {detail}"
        )
    return result


def run_scp_upload(
    local_path: PathLike,
    remote_path: str,
    *,
    host: str = CURRENT_REMOTE_HOST,
    timeout_seconds: int = 1800,
) -> RemoteCommandResult:
    _require_current_host(host)
    source = Path(local_path).resolve()
    if not source.is_file():
        raise RemoteTransportError(f"Upload source is not a file: {source}")
    if not remote_path.startswith("/data/home/xinxi-zhyh/xinxi-zhyh/"):
        raise RemoteTransportError(f"Upload target is outside the project home: {remote_path}")
    completed = subprocess.run(
        [
            "scp",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectionAttempts=1",
            "-o",
            "ConnectTimeout=8",
            str(source),
            f"{host}:{remote_path}",
        ],
        check=False,
        capture_output=True,
        timeout=timeout_seconds,
    )
    result = RemoteCommandResult(
        returncode=int(completed.returncode),
        stdout=completed.stdout.decode("utf-8", errors="replace"),
        stderr=completed.stderr.decode("utf-8", errors="replace"),
    )
    if result.returncode != 0:
        detail = "\n".join((result.stdout + result.stderr).splitlines()[-8:])
        raise RemoteTransportError(
            f"SCP upload failed with exit code {result.returncode}: {detail}"
        )
    return result


def run_sftp_reput(
    local_path: PathLike,
    remote_path: str,
    *,
    host: str = CURRENT_REMOTE_HOST,
    timeout_seconds: int = 14_400,
) -> RemoteCommandResult:
    """Resume an upload into a caller-managed temporary remote path."""
    _require_current_host(host)
    source = Path(local_path).resolve()
    if not source.is_file():
        raise RemoteTransportError(f"Upload source is not a file: {source}")
    if not remote_path.startswith("/data/home/xinxi-zhyh/xinxi-zhyh/"):
        raise RemoteTransportError(f"Upload target is outside the project home: {remote_path}")
    if any(token in remote_path for token in ('"', "\r", "\n")):
        raise RemoteTransportError("Upload target contains an unsafe SFTP character")
    if timeout_seconds <= 0 or timeout_seconds > 14_400:
        raise RemoteTransportError("SFTP timeout is outside the bounded range")
    source_text = source.as_posix()
    if '"' in source_text or any(token in source_text for token in ("\r", "\n")):
        raise RemoteTransportError("Upload source contains an unsafe SFTP character")
    batch = f'reput "{source_text}" "{remote_path}"\n'
    try:
        completed = subprocess.run(
            [
                "sftp",
                "-b",
                "-",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectionAttempts=1",
                "-o",
                "ConnectTimeout=8",
                "-o",
                "ServerAliveInterval=15",
                "-o",
                "ServerAliveCountMax=4",
                host,
            ],
            input=batch.encode("utf-8"),
            check=False,
            capture_output=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise RemoteTransportError(
            f"SFTP upload exceeded {timeout_seconds} seconds"
        ) from exc
    result = RemoteCommandResult(
        returncode=int(completed.returncode),
        stdout=completed.stdout.decode("utf-8", errors="replace"),
        stderr=completed.stderr.decode("utf-8", errors="replace"),
    )
    if result.returncode != 0:
        detail = "\n".join((result.stdout + result.stderr).splitlines()[-8:])
        raise RemoteTransportError(
            f"Resumable SFTP upload failed with exit code {result.returncode}: {detail}"
        )
    return result


def command_exists(name: str, search_path: Sequence[str] = ("ssh", "scp")) -> bool:
    if name not in search_path:
        return False
    try:
        completed = subprocess.run(
            [name, "-V"], check=False, capture_output=True, timeout=10
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return completed.returncode in {0, 1}
