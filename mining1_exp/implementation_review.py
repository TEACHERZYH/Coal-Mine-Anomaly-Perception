from __future__ import annotations

import ast
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable, Mapping, Union

from .governance.immutable import write_once_json
from .provenance import canonical_json_sha256, sha256_file


PathLike = Union[str, Path]
IMPLEMENTATION_STEPS = ("I000", "I020", "I040", "I050", "I070", "I080")
REMOVED_FAMILY_IDS = {
    "V1-Y8",
    "V1-Y10",
    "V1-RTDETR",
    "T1-FUSION",
    "S1-CNN",
    "E2-FIXED",
    "E2-MLP",
    "E2-HMM",
    "E3-NOHYST",
    "E3-NOABS",
    "V2-DIR-B",
}
REMOVED_IMPLEMENTATION_TOKENS = {
    "evidence_package_performance",
    "yolov10",
    "rtdetr",
    "rt-detr",
    "methane_hmm",
    "methane_mlp",
    "trainable_rgbt_fusion",
    "rgbt_fusion",
    "fixed_weight_fusion",
    "methane_1d_cnn",
    "sensitivity_sweep",
}
REMOVED_CLASS_NAMES = {
    "evidencepackageperformance",
    "fixedweightfusion",
    "methane1dcnn",
    "methanehmm",
    "methanelogisticregression",
    "methanemlp",
    "rtdetr",
    "rgbtfusion",
    "sensitivitysweep",
    "trainablergbtfusion",
    "yolov10",
}
TEST_POOLS = {"D_b_te", "D_e_te"}
OLD_PROJECT_PATTERN = re.compile(r"f:[\\/]2026[\\/]mining(?:[\\/]|$)", re.I)
OLD_HOST = "180.209.128.66"
ALLOWED_OLD_HOST_GUARD = Path("mining1_exp/governance/slurm_contracts.py")
REVIEW_RULE_SOURCE = Path("mining1_exp/implementation_review.py")
ALLOWED_IMPORT_ROOTS = {
    "__future__",
    "argparse",
    "ast",
    "base64",
    "collections",
    "concurrent",
    "copy",
    "csv",
    "dataclasses",
    "datetime",
    "hashlib",
    "hmac",
    "importlib",
    "io",
    "itertools",
    "json",
    "logging",
    "matplotlib",
    "math",
    "mining1_exp",
    "numpy",
    "os",
    "pandas",
    "pathlib",
    "PIL",
    "pickle",
    "platform",
    "pyarrow",
    "re",
    "scipy",
    "secrets",
    "shutil",
    "sklearn",
    "subprocess",
    "sys",
    "tarfile",
    "tempfile",
    "time",
    "torch",
    "typing",
    "ultralytics",
    "uuid",
    "xml",
    "yaml",
    "zipfile",
}


class ImplementationReviewError(ValueError):
    """Raised when the implementation package is not ready to leave I090."""


def implementation_source_scope(project_root: PathLike) -> dict[str, Any]:
    root = Path(project_root).resolve()
    files: set[Path] = set()
    for directory in (
        "mining1_exp",
        "tests",
        "tools",
        "slurm",
        "configs",
        "notes",
        "manuscript",
        "env",
        "lockfiles",
    ):
        base = root / directory
        if base.is_dir():
            files.update(
                path.resolve()
                for path in base.rglob("*")
                if path.is_file() and "__pycache__" not in path.parts
            )
    for name in ("pyproject.toml", "AGENTS.md"):
        path = root / name
        if path.is_file():
            files.add(path.resolve())
    manifest = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(files)
    ]
    plan_rows = _read_csv(root / "plans/experiment_steps.csv")
    plan_contract = [
        {key: value for key, value in row.items() if key != "status"}
        for row in plan_rows
    ]
    return {
        "status": "pass",
        "file_count": len(manifest),
        "scope_sha256": canonical_json_sha256(manifest),
        "manifest": manifest,
        "plan_step_count": len(plan_contract),
        "plan_contract_sha256": canonical_json_sha256(plan_contract),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _is_sys_path_mutation(node: ast.Call) -> bool:
    function = node.func
    if not isinstance(function, ast.Attribute) or function.attr not in {
        "append",
        "extend",
        "insert",
    }:
        return False
    value = function.value
    return (
        isinstance(value, ast.Attribute)
        and value.attr == "path"
        and isinstance(value.value, ast.Name)
        and value.value.id == "sys"
    )


def _test_pool_fit_calls(tree: ast.AST) -> list[int]:
    lines = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function_name = ""
        if isinstance(node.func, ast.Name):
            function_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            function_name = node.func.attr
        if not any(
            token in function_name.lower()
            for token in ("fit", "train", "select", "calibr")
        ):
            continue
        for keyword in node.keywords:
            if (
                keyword.arg == "pool"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value in TEST_POOLS
            ):
                lines.append(int(getattr(node, "lineno", 0)))
    return lines


def _review_source_files(root: Path) -> list[Path]:
    files = list((root / "mining1_exp").rglob("*.py"))
    for directory, patterns in (
        (root / "slurm", ("*.sh", "*.sbatch")),
        (root / "tools" / "preflight", ("*.ps1",)),
        (root / "tools" / "remote", ("*.ps1",)),
    ):
        for pattern in patterns:
            files.extend(directory.glob(pattern))
    return sorted(set(path.resolve() for path in files if path.is_file()))


def scan_implementation_residue(project_root: PathLike) -> dict[str, Any]:
    root = Path(project_root).resolve()
    errors: list[str] = []
    files = _review_source_files(root)
    for path in files:
        relative = path.relative_to(root)
        text = path.read_text(encoding="utf-8-sig")
        lowered = text.lower()
        is_review_rule_source = relative == REVIEW_RULE_SOURCE
        if not is_review_rule_source and OLD_PROJECT_PATTERN.search(text):
            errors.append(f"old project path: {relative.as_posix()}")
        if OLD_HOST in text and relative not in {
            ALLOWED_OLD_HOST_GUARD,
            REVIEW_RULE_SOURCE,
        }:
            errors.append(f"old host outside rejection guard: {relative.as_posix()}")
        if not is_review_rule_source:
            for token in sorted(REMOVED_FAMILY_IDS):
                if token.lower() in lowered:
                    errors.append(f"removed family {token}: {relative.as_posix()}")
            for token in sorted(REMOVED_IMPLEMENTATION_TOKENS):
                if token in lowered:
                    errors.append(f"removed implementation {token}: {relative.as_posix()}")
        if path.suffix == ".py":
            try:
                tree = ast.parse(text, filename=str(path), feature_version=(3, 9))
            except SyntaxError as exc:
                errors.append(f"Python 3.9 syntax failure: {relative.as_posix()}:{exc.lineno}")
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    normalized_name = re.sub(r"[^a-z0-9]", "", node.name.lower())
                    if normalized_name in REMOVED_CLASS_NAMES:
                        errors.append(
                            "removed implementation class "
                            f"{node.name}: {relative.as_posix()}:{node.lineno}"
                        )
                if isinstance(node, ast.Call) and _is_sys_path_mutation(node):
                    errors.append(f"sys.path mutation: {relative.as_posix()}:{node.lineno}")
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        root_name = alias.name.split(".")[0]
                        if root_name not in ALLOWED_IMPORT_ROOTS:
                            errors.append(
                                "unapproved import root "
                                f"{root_name}: {relative.as_posix()}:{node.lineno}"
                            )
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.level == 0
                    and node.module
                ):
                    root_name = node.module.split(".")[0]
                    if root_name not in ALLOWED_IMPORT_ROOTS:
                        errors.append(
                            "unapproved import root "
                            f"{root_name}: {relative.as_posix()}:{node.lineno}"
                        )
            for line in _test_pool_fit_calls(tree):
                errors.append(f"test-pool fit or selection: {relative.as_posix()}:{line}")
    return {
        "status": "pass" if not errors else "fail",
        "scanned_file_count": len(files),
        "errors": errors,
        "allowed_old_host_guard": ALLOWED_OLD_HOST_GUARD.as_posix(),
    }


def collect_implementation_evidence(project_root: PathLike) -> dict[str, Any]:
    root = Path(project_root).resolve()
    plan = {row["step_id"]: row for row in _read_csv(root / "plans/experiment_steps.csv")}
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    for step_id in IMPLEMENTATION_STEPS:
        receipt_path = root / "evidence" / "implementation" / f"{step_id}_test_receipt.json"
        review_path = root / "evidence" / "step_reviews" / f"{step_id}.json"
        if plan.get(step_id, {}).get("status") != "pass":
            errors.append(f"implementation ledger is not pass: {step_id}")
        if not receipt_path.is_file() or not review_path.is_file():
            errors.append(f"implementation evidence is missing: {step_id}")
            continue
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
            review = json.loads(review_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"implementation evidence is unreadable: {step_id}: {exc}")
            continue
        if receipt.get("step_id") != step_id or receipt.get("status") != "pass":
            errors.append(f"implementation receipt state mismatch: {step_id}")
        checks = review.get("acceptance_checks")
        if (
            review.get("step_id") != step_id
            or review.get("status") != "pass"
            or review.get("advance_allowed") is not True
            or not isinstance(checks, list)
            or not checks
            or any(check.get("status") != "pass" for check in checks)
        ):
            errors.append(f"implementation review state mismatch: {step_id}")
        records.append(
            {
                "step_id": step_id,
                "receipt_path": receipt_path.relative_to(root).as_posix(),
                "receipt_sha256": sha256_file(receipt_path),
                "review_path": review_path.relative_to(root).as_posix(),
                "review_sha256": sha256_file(review_path),
            }
        )
    return {
        "status": "pass" if not errors else "fail",
        "steps": records,
        "errors": errors,
    }


def _run_json_command(command: list[str], root: Path, label: str) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if completed.returncode != 0:
        tail = "\n".join((completed.stdout + "\n" + completed.stderr).splitlines()[-20:])
        raise ImplementationReviewError(f"{label} failed:\n{tail}")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ImplementationReviewError(f"{label} did not emit one JSON object") from exc


def run_full_test_suite(project_root: PathLike) -> dict[str, Any]:
    root = Path(project_root).resolve()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    combined = completed.stdout + "\n" + completed.stderr
    match = re.search(r"(\d+) passed in ([0-9.]+)s", combined)
    if completed.returncode != 0 or match is None:
        tail = "\n".join(combined.splitlines()[-30:])
        raise ImplementationReviewError(f"full implementation regression failed:\n{tail}")
    return {
        "status": "pass",
        "passed": int(match.group(1)),
        "runtime_seconds": float(match.group(2)),
        "returncode": completed.returncode,
        "summary_line": match.group(0),
    }


def validate_implementation_review(payload: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "step_id",
        "status",
        "reviewer",
        "reviewed_at",
        "implementation_evidence",
        "residue_scan",
        "static_plan_validation",
        "test_suite",
        "source_scope",
        "acceptance_checks",
        "scope_boundary",
        "advance_allowed",
        "failure_reference",
        "findings",
    }
    if not required.issubset(payload):
        raise ImplementationReviewError("I090 review is missing required fields")
    if payload["schema_version"] != 1 or payload["step_id"] != "I090":
        raise ImplementationReviewError("I090 review identity is invalid")
    if payload["status"] != "pass":
        raise ImplementationReviewError("I090 review is not pass")
    for field in (
        "implementation_evidence",
        "residue_scan",
        "static_plan_validation",
        "test_suite",
        "source_scope",
    ):
        if payload[field].get("status") != "pass":
            raise ImplementationReviewError(f"I090 component is not pass: {field}")
    static_counts = payload["static_plan_validation"].get("counts", {})
    if static_counts != {
        "families": 25,
        "trained_or_fitted_runs": 39,
        "steps": 78,
        "claims": 9,
    }:
        raise ImplementationReviewError("I090 static plan counts are not minimal_v2")
    if int(payload["test_suite"].get("passed", 0)) <= 0:
        raise ImplementationReviewError("I090 has no passing full-regression evidence")
    source_scope = payload["source_scope"]
    if (
        not isinstance(source_scope, dict)
        or int(source_scope.get("file_count", 0)) <= 0
        or re.fullmatch(r"[0-9a-f]{64}", str(source_scope.get("scope_sha256", "")))
        is None
        or not isinstance(source_scope.get("manifest"), list)
        or len(source_scope["manifest"]) != source_scope["file_count"]
        or source_scope.get("plan_step_count") != 78
        or re.fullmatch(
            r"[0-9a-f]{64}", str(source_scope.get("plan_contract_sha256", ""))
        )
        is None
    ):
        raise ImplementationReviewError("I090 source scope is invalid")
    checks = payload["acceptance_checks"]
    if not isinstance(checks, list) or not checks or any(
        check.get("status") != "pass" for check in checks
    ):
        raise ImplementationReviewError("I090 acceptance checks are incomplete")
    boundary = payload["scope_boundary"]
    if (
        boundary.get("full_local_dataset_extractions") != 0
        or boundary.get("remote_connections") != 0
        or boundary.get("slurm_jobs_created") != 0
        or boundary.get("performance_claims_authorized") is not False
    ):
        raise ImplementationReviewError("I090 exceeded its local review boundary")
    if payload["advance_allowed"] is not True:
        raise ImplementationReviewError("I090 does not authorize plan advancement")
    if not isinstance(payload["failure_reference"], list) or not payload[
        "failure_reference"
    ]:
        raise ImplementationReviewError("I090 lacks rejected-attempt references")
    if not isinstance(payload["findings"], list) or not payload["findings"]:
        raise ImplementationReviewError("I090 lacks reviewer findings")
    try:
        reviewed_at = datetime.fromisoformat(
            str(payload["reviewed_at"]).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ImplementationReviewError("I090 reviewed_at is invalid") from exc
    if reviewed_at.tzinfo is None or reviewed_at.utcoffset() is None:
        raise ImplementationReviewError("I090 reviewed_at lacks a timezone")


def run_implementation_review(
    project_root: PathLike,
    output_path: PathLike,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    output = Path(output_path)
    if not output.is_absolute():
        output = root / output
    if output.exists():
        payload = json.loads(output.read_text(encoding="utf-8-sig"))
        validate_implementation_review(payload)
        current_scope = implementation_source_scope(root)
        if current_scope != payload["source_scope"]:
            raise ImplementationReviewError("I090 source scope changed after review")
        current_evidence = collect_implementation_evidence(root)
        if current_evidence != payload["implementation_evidence"]:
            raise ImplementationReviewError("I090 implementation evidence changed after review")
        current_plan = _run_json_command(
            [sys.executable, "-m", "tools.validate_minimal_plan"],
            root,
            "minimal plan validation",
        )
        if current_plan.get("status") != "pass":
            raise ImplementationReviewError("I090 current plan no longer validates")
        return {
            "status": "pass",
            "step_id": "I090",
            "mode": "validated_existing",
            "review_sha256": sha256_file(output),
        }
    decision = json.loads(
        (root / "configs/environment_decision.lock.json").read_text(
            encoding="utf-8-sig"
        )
    )
    if Path(decision["selected_local_python"]).resolve() != Path(sys.executable).resolve():
        raise ImplementationReviewError("I090 is not running under selected_local_python")
    evidence = collect_implementation_evidence(root)
    residue = scan_implementation_residue(root)
    source_scope = implementation_source_scope(root)
    static_plan = _run_json_command(
        [sys.executable, "-m", "tools.validate_minimal_plan"],
        root,
        "minimal plan validation",
    )
    tests = run_full_test_suite(root)
    components = {
        "implementation_evidence": evidence,
        "residue_scan": residue,
        "static_plan_validation": static_plan,
        "test_suite": tests,
        "source_scope": source_scope,
    }
    failed = [name for name, value in components.items() if value.get("status") != "pass"]
    if failed:
        raise ImplementationReviewError(f"I090 components failed: {failed}")
    payload = {
        "schema_version": 1,
        "step_id": "I090",
        "status": "pass",
        "reviewer": "codex_independent_implementation_review",
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        **components,
        "acceptance_checks": [
            {
                "id": "all_implementation_steps_reviewed",
                "status": "pass",
                "evidence": (
                    f"{len(evidence['steps'])}/{len(IMPLEMENTATION_STEPS)} "
                    "implementation steps have pass receipts and pass reviews."
                ),
            },
            {
                "id": "no_removed_family_or_old_code",
                "status": "pass",
                "evidence": (
                    "Residue scan passed across "
                    f"{residue['scanned_file_count']} implementation and execution files."
                ),
            },
            {
                "id": "source_scope_bound",
                "status": "pass",
                "evidence": (
                    f"Review binds {source_scope['file_count']} source, test, tool, "
                    "configuration, plan-support, and manuscript files with scope "
                    f"SHA-256 {source_scope['scope_sha256']}."
                ),
            },
            {
                "id": "no_test_pool_fitting",
                "status": "pass",
                "evidence": (
                    "AST scan found no fit, train, selection, or calibration call "
                    "with D_b_te or D_e_te as its pool."
                ),
            },
            {
                "id": "matrix_and_claim_contract",
                "status": "pass",
                "evidence": (
                    "Static plan validation preserves 25 families, at most 39 "
                    "trained/fitted runs, 78 steps, and 9 claims."
                ),
            },
            {
                "id": "full_regression",
                "status": "pass",
                "evidence": tests["summary_line"],
            },
            {
                "id": "local_review_boundary",
                "status": "pass",
                "evidence": "I090 performed source, artifact, plan, and local test review only.",
            },
        ],
        "scope_boundary": {
            "full_local_dataset_extractions": 0,
            "remote_connections": 0,
            "slurm_jobs_created": 0,
            "performance_claims_authorized": False,
        },
        "failure_reference": [
            "evidence/implementation/I090_attempt_01_rejected.json",
            "evidence/implementation/I090_attempt_02_rejected.json",
            "evidence/implementation/I090_attempt_03_rejected.json",
            "evidence/implementation/I090_attempt_04_rejected.json",
            "evidence/implementation/I090_attempt_05_postwrite_reporting_failure.json",
            "evidence/implementation/I090_attempt_05_partial_review.json",
        ],
        "findings": [
            "All six prerequisite implementation steps have pass receipts and reviews.",
            "No removed implementation family or legacy project dependency remains in scope.",
            "I090 authorizes progression only; it does not authorize performance claims.",
        ],
        "advance_allowed": True,
    }
    validate_implementation_review(payload)
    result = write_once_json(output, payload)
    return {
        "status": "pass",
        "step_id": "I090",
        "mode": "created",
        "review_sha256": result.artifact.sha256,
        "test_count": tests["passed"],
        "static_check_count": static_plan["check_count"],
        "residue_file_count": residue["scanned_file_count"],
    }
