from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .governance.immutable import write_once_json
from .provenance import canonical_json_sha256, sha256_file
from .remote_pilot import _resource_cap_decisions
from .workflow_common import WorkflowExecutionError


_LEGACY_PACKAGE_CAPS = {
    "T1": 24.0,
    "S1": 24.0,
    "V2": 160.0,
    "E2_E3": 48.0,
    "R1": 12.0,
    "C1": 8.0,
}


def _measurement_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: deepcopy(payload[key])
        for key in (
            "slurm_job_id",
            "cuda_device_count",
            "cuda_device_names",
            "benchmarks",
            "scaling_trial",
        )
    }


def normalize_resource_pilot_metadata(source: Path, target: Path) -> Dict[str, Any]:
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    decisions = payload.get("decisions")
    if payload.get("status") != "pass" or not isinstance(decisions, dict):
        raise WorkflowExecutionError("Resource-pilot source is not a successful pilot artifact")
    if decisions.get("gpu_hour_caps") != _LEGACY_PACKAGE_CAPS:
        raise WorkflowExecutionError("Resource-pilot source lacks the reviewed legacy cap mapping")
    measurement = _measurement_payload(payload)
    measurement_sha256 = canonical_json_sha256(measurement)
    normalized = deepcopy(payload)
    gate_caps, basis = _resource_cap_decisions()
    normalized["decisions"]["gpu_hour_caps"] = gate_caps
    normalized["resource_cap_basis"] = basis
    normalized["metadata_repair"] = {
        "status": "pass",
        "reason": "aggregate reviewed package caps into frozen protocol gate keys",
        "source_path": source.as_posix(),
        "source_sha256": sha256_file(source),
        "measurement_payload_sha256": measurement_sha256,
        "changed_fields": [
            "decisions.gpu_hour_caps",
            "resource_cap_basis",
            "metadata_repair",
        ],
        "measurements_reused_without_rerun": True,
        "performance_claims_authorized": False,
    }
    if canonical_json_sha256(_measurement_payload(normalized)) != measurement_sha256:
        raise WorkflowExecutionError("Metadata normalization changed measured pilot telemetry")
    write_once_json(target, normalized)
    return normalized


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize reviewed E066 metadata")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    payload = normalize_resource_pilot_metadata(args.source, args.target)
    print(
        json.dumps(
            {
                "status": "pass",
                "source": args.source.as_posix(),
                "target": args.target.as_posix(),
                "target_sha256": sha256_file(args.target),
                "measurement_payload_sha256": payload["metadata_repair"][
                    "measurement_payload_sha256"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
