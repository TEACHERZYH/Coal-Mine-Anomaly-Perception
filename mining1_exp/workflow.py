from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, Mapping

from .workflow_common import execute_with_contract
from .workflow_registry import WORKFLOW_COMMAND_SPECS


HANDLER_TARGETS: Mapping[str, str] = {
    "adjudicate-claims": "mining1_exp.workflow_finalize:adjudicate_claims",
    "aggregate-results": "mining1_exp.workflow_finalize:aggregate_results",
    "apply-remote-environment": "mining1_exp.remote_environment:apply_remote_environment",
    "audit-artifacts": "mining1_exp.workflow_finalize:audit_artifacts",
    "audit-graph-eligibility": "mining1_exp.workflow_data:audit_graph_eligibility",
    "audit-rgbt-inputs": "mining1_exp.workflow_data:audit_rgbt_inputs",
    "build-and-audit-episode-features": (
        "mining1_exp.workflow_episode:build_and_audit_episode_features"
    ),
    "build-episode-skeletons": "mining1_exp.workflow_episode:build_episode_skeletons",
    "build-minimal-sample": "mining1_exp.workflow_data:build_minimal_sample",
    "close-gate": "mining1_exp.workflow_governance:close_gate",
    "create-branch-test-seal": "mining1_exp.workflow_governance:create_branch_test_seal",
    "create-fewshot": "mining1_exp.workflow_data:create_fewshot",
    "create-splits": "mining1_exp.workflow_data:create_splits",
    "decide-dataset-sources": "mining1_exp.workflow_data:decide_dataset_sources",
    "dry-run-matrix": "mining1_exp.workflow_spec:dry_run_matrix",
    "evaluate": "mining1_exp.workflow_evaluate:evaluate_package",
    "fill-pretest-lock": "mining1_exp.workflow_governance:fill_pretest_lock",
    "finalize-branch-test-seal": (
        "mining1_exp.workflow_governance:finalize_branch_test_seal"
    ),
    "finalize-closeout": "mining1_exp.workflow_finalize:finalize_closeout",
    "inventory-local-datasets": "mining1_exp.workflow_data:inventory_local_datasets",
    "inventory-remote-datasets": "mining1_exp.workflow_data:inventory_remote_datasets",
    "lock-methane-roles": "mining1_exp.workflow_data:lock_methane_roles",
    "lock-ontology": "mining1_exp.workflow_data:lock_ontology",
    "release-branch-test": "mining1_exp.workflow_governance:release_branch_test",
    "release-episode-test": "mining1_exp.workflow_episode:release_episode_test",
    "review-package": "mining1_exp.workflow_finalize:review_package",
    "run-confirmatory-stats": "mining1_exp.workflow_finalize:run_confirmatory_stats",
    "seal-episode-test": "mining1_exp.workflow_episode:seal_episode_test",
    "select-checkpoints": "mining1_exp.workflow_evaluate:select_checkpoints",
    "smoke-local": "mining1_exp.workflow_data:smoke_local",
    "stage-datasets": "mining1_exp.workflow_data:stage_datasets",
    "update-manuscript": "mining1_exp.workflow_finalize:update_manuscript",
    "validate-artifact-contract": "mining1_exp.workflow_spec:validate_artifact_contract",
    "validate-protocol": "mining1_exp.workflow_spec:validate_protocol",
}


def _load_handler(target: str) -> Any:
    module_name, function_name = target.split(":", 1)
    module = importlib.import_module(module_name)
    handler = getattr(module, function_name)
    if not callable(handler):
        raise TypeError(f"Workflow target is not callable: {target}")
    return handler


def resolve_workflow_handlers() -> Dict[str, Any]:
    return {name: _load_handler(target) for name, target in HANDLER_TARGETS.items()}


def execute_workflow_command(command: str, args: Any, project_root: Path) -> Dict[str, Any]:
    if command not in HANDLER_TARGETS:
        raise RuntimeError(f"No workflow handler is registered for {command}")
    arguments = {
        key: value
        for key, value in vars(args).items()
        if key not in {"command", "handler"} and value is not None
    }
    handler = _load_handler(HANDLER_TARGETS[command])
    return execute_with_contract(project_root, command, arguments, handler)


if set(HANDLER_TARGETS) != {spec.name for spec in WORKFLOW_COMMAND_SPECS}:
    raise RuntimeError("Workflow handler registry does not match command specifications")
