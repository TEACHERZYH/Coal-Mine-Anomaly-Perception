from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import importlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


MODULE_NAMES = {
    "pyyaml": "yaml",
    "scikit-learn": "sklearn",
    "opencv-python": "cv2",
    "opencv-python-headless": "cv2",
    "onnxruntime-gpu": "onnxruntime",
    "protobuf": "google.protobuf",
}


def _normalized_path(value: str) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(value)))


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _import_module_captured(module_name: str) -> tuple[Any, dict[str, str]]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        module = importlib.import_module(module_name)
    diagnostics = {
        stream: value.getvalue().strip()[-2000:]
        for stream, value in (("stdout", stdout), ("stderr", stderr))
        if value.getvalue().strip()
    }
    return module, diagnostics


def _parse_lock(path: Path) -> dict[str, str]:
    packages: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            raise ValueError(f"Unpinned requirement at {path}:{line_number}: {line}")
        name, version = (part.strip() for part in line.split("==", 1))
        if not name or not version:
            raise ValueError(f"Malformed requirement at {path}:{line_number}: {line}")
        key = name.lower()
        if key in packages:
            raise ValueError(f"Duplicate requirement in {path}: {name}")
        packages[key] = version
    if not packages:
        raise ValueError(f"Environment lock contains no packages: {path}")
    return packages


def _expected_cuda(torch_version: str) -> str | None:
    if "+cu" not in torch_version:
        return None
    suffix = torch_version.rsplit("+cu", 1)[1]
    if not suffix.isdigit() or len(suffix) < 2:
        raise ValueError(f"Unsupported torch CUDA version suffix: {torch_version}")
    return f"{suffix[:-1]}.{suffix[-1]}"


def verify(scope: str, decision_path: Path) -> dict[str, Any]:
    decision = _load_json(decision_path)
    if decision.get("status") != "locked":
        raise ValueError("Environment decision must have status=locked")

    project_root = decision_path.resolve().parent.parent
    selected_python = str(decision[f"selected_{scope}_python"])
    if _normalized_path(sys.executable) != _normalized_path(selected_python):
        raise RuntimeError(
            f"Interpreter mismatch: running={sys.executable!r}, selected={selected_python!r}"
        )

    specification = Path(str(decision[f"{scope}_environment_specification"]))
    if not specification.is_absolute():
        specification = project_root / specification
    specification = specification.resolve()
    if not specification.is_file():
        raise FileNotFoundError(f"Environment specification not found: {specification}")

    expected = _parse_lock(specification)
    observed: dict[str, str] = {}
    imported_modules: dict[str, str] = {}
    import_diagnostics: dict[str, dict[str, str]] = {}
    for package_name, expected_version in expected.items():
        observed_version = importlib.metadata.version(package_name)
        observed[package_name] = observed_version
        if observed_version != expected_version:
            raise RuntimeError(
                f"Package version mismatch for {package_name}: "
                f"expected={expected_version}, observed={observed_version}"
            )
        module_name = MODULE_NAMES.get(package_name, package_name.replace("-", "_"))
        _, diagnostics = _import_module_captured(module_name)
        imported_modules[package_name] = module_name
        if diagnostics:
            import_diagnostics[package_name] = diagnostics

    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        capture_output=True,
        text=True,
    )
    if pip_check.returncode != 0:
        detail = (pip_check.stdout + "\n" + pip_check.stderr).strip()
        raise RuntimeError(f"pip check failed: {detail}")

    torch_runtime = None
    if "torch" in expected:
        torch_module, diagnostics = _import_module_captured("torch")
        if diagnostics:
            import_diagnostics.setdefault("torch", {}).update(diagnostics)
        torch_runtime = getattr(torch_module.version, "cuda", None)
        expected_runtime = _expected_cuda(expected["torch"])
        if expected_runtime is not None and torch_runtime != expected_runtime:
            raise RuntimeError(
                f"Torch CUDA runtime mismatch: expected={expected_runtime}, "
                f"observed={torch_runtime}"
            )

    return {
        "schema_version": 1,
        "status": "pass",
        "scope": scope,
        "selected_python": selected_python,
        "running_python": sys.executable,
        "python_version": sys.version.split()[0],
        "specification": str(specification),
        "package_count": len(observed),
        "packages": dict(sorted(observed.items())),
        "imported_modules": dict(sorted(imported_modules.items())),
        "import_diagnostics": dict(sorted(import_diagnostics.items())),
        "pip_check": (pip_check.stdout + pip_check.stderr).strip(),
        "torch_cuda_runtime": torch_runtime,
        "gpu_availability_checked": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("local", "remote"), required=True)
    parser.add_argument("--decision", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify(args.scope, args.decision)
    except Exception as exc:
        result = {
            "schema_version": 1,
            "status": "fail",
            "scope": args.scope,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        print(json.dumps(result, ensure_ascii=True, sort_keys=True))
        return 1
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
