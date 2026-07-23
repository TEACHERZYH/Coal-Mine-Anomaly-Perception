from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Sequence

from . import __version__
from .config import resolved_config_snapshot
from .provenance import build_receipt, write_receipt
from .workflow_registry import add_workflow_parsers


def _emit(payload: Dict[str, Any], stream: Any = sys.stdout) -> None:
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True), file=stream)


def _version(_: argparse.Namespace) -> int:
    _emit(
        {
            "package": "mining1_exp",
            "python": sys.version.split()[0],
            "version": __version__,
        }
    )
    return 0


def _validate_config(args: argparse.Namespace) -> int:
    snapshot = resolved_config_snapshot(args.config)
    result = {"status": "pass", "config_snapshot": snapshot}
    if args.receipt is not None:
        receipt = build_receipt(
            step_id="CLI_VALIDATE_CONFIG",
            status="pass",
            command="mining1_exp.cli validate-config",
            inputs=[args.config],
            config_snapshot=snapshot,
        )
        digest = write_receipt(args.receipt, receipt)
        result["receipt"] = digest.to_dict()
    _emit(result)
    return 0


def _execute_step(args: argparse.Namespace) -> int:
    if args.step == "I080":
        from .minimal_pipeline import run_minimal_pipeline

        result = run_minimal_pipeline(args.run_root)
    else:
        from .remote_steps import execute_remote_step

        result = execute_remote_step(
            step_id=args.step,
            run_root=args.run_root,
            config_path=args.config,
            project_root=Path.cwd(),
        )
    _emit(result)
    return 0


def _review_step(args: argparse.Namespace) -> int:
    if args.step != "I090":
        raise RuntimeError(f"No implementation review handler is registered for {args.step}")
    from .implementation_review import run_implementation_review

    output = args.output or Path("evidence/reviews/I090.json")
    _emit(run_implementation_review(Path.cwd(), output))
    return 0


def _workflow_step(args: argparse.Namespace) -> int:
    from .workflow import execute_workflow_command

    _emit(execute_workflow_command(args.command, args, Path.cwd()))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mining1_exp.cli",
        description="Clean-room experiment orchestration CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    version_parser = subparsers.add_parser("version", help="Print package versions.")
    version_parser.set_defaults(handler=_version)

    config_parser = subparsers.add_parser(
        "validate-config", help="Load a structured config and emit its hashes."
    )
    config_parser.add_argument("--config", type=Path, required=True)
    config_parser.add_argument("--receipt", type=Path)
    config_parser.set_defaults(handler=_validate_config)

    execute_parser = subparsers.add_parser(
        "execute-step",
        help="Execute one locked local or Slurm step through the registered dispatcher.",
    )
    execute_parser.add_argument("--step", required=True)
    execute_parser.add_argument("--run-root", type=Path, required=True)
    execute_parser.add_argument("--config", type=Path)
    execute_parser.set_defaults(handler=_execute_step)

    review_parser = subparsers.add_parser(
        "review-step",
        help="Run a fail-closed implementation review for an eligible ledger step.",
    )
    review_parser.add_argument("--step", required=True)
    review_parser.add_argument("--output", type=Path)
    review_parser.set_defaults(handler=_review_step)
    add_workflow_parsers(subparsers, _workflow_step)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except Exception as exc:
        _emit(
            {
                "status": "fail",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "command": args.command,
            },
            stream=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
