from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _inventory() -> list[str]:
    values = {
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    }
    return sorted(values, key=str.lower)


def build_receipt(
    *,
    decision_path: Path,
    verification_path: Path,
    setup_executed: bool,
) -> dict[str, Any]:
    decision = _load_object(decision_path)
    verification = _load_object(verification_path)
    if decision.get("status") != "locked" or verification.get("status") != "pass":
        raise ValueError("Remote environment decision and verification must pass")
    selected_python = str(decision.get("selected_remote_python", ""))
    if os.path.realpath(selected_python) != os.path.realpath(sys.executable):
        raise ValueError("Receipt writer is not running under the selected remote Python")
    project_root = decision_path.resolve().parent.parent
    specification = Path(str(decision["remote_environment_specification"]))
    if not specification.is_absolute():
        specification = project_root / specification
    if not specification.is_file():
        raise FileNotFoundError(f"Remote environment specification is missing: {specification}")
    inventory = _inventory()
    packages = verification.get("packages")
    if not isinstance(packages, dict) or not packages:
        raise ValueError("Remote verification package inventory is empty")
    job_id = os.environ.get("SLURM_JOB_ID", "").strip()
    compute_node = platform.node().strip()
    if not job_id or not compute_node or compute_node.lower() == "mu01":
        raise ValueError("Remote environment receipt requires a compute-node Slurm job")
    python_probe = json.dumps(
        {"executable": sys.executable, "version": platform.python_version()},
        sort_keys=True,
    )
    verification_line = json.dumps(
        verification, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    return {
        "schema_version": 2,
        "scope": "remote",
        "status": "pass",
        "ready_at": datetime.now(timezone.utc).isoformat(),
        "host": "xinxi-zhyh@211.87.115.228",
        "decision_sha256": _sha256_bytes(decision_path.read_bytes()),
        "strategy": decision["remote_strategy"],
        "selected_python": selected_python,
        "environment_specification": str(decision["remote_environment_specification"]),
        "specification_hash_mode": "file_content",
        "environment_specification_sha256": _sha256_bytes(specification.read_bytes()),
        "activation_command_sha256": _sha256_bytes(
            str(decision["remote_activation_command"]).encode("utf-8")
        ),
        "setup_command_sha256": _sha256_bytes(
            str(decision["remote_setup_command"]).encode("utf-8")
        ),
        "verification_command_sha256": _sha256_bytes(
            str(decision["remote_environment_verification_command"]).encode("utf-8")
        ),
        "package_inventory_sha256": _sha256_bytes(_canonical_json(inventory)),
        "setup_command_executed": bool(setup_executed),
        "verification_output_excerpt": [verification_line[:8000]],
        "python_probe": [python_probe],
        "package_inventory": [json.dumps(inventory, separators=(",", ":"))],
        "slurm_job_id": job_id,
        "compute_node": compute_node,
        "remote_closeout_reference": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--setup-executed", choices=("true", "false"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = build_receipt(
        decision_path=args.decision,
        verification_path=args.verification,
        setup_executed=args.setup_executed == "true",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(receipt, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    try:
        os.link(temporary, args.output)
    except FileExistsError as exc:
        raise FileExistsError(f"Immutable remote environment receipt exists: {args.output}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
