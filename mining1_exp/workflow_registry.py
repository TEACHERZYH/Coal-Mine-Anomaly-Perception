from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple


ArgumentDefinition = Tuple[Tuple[str, ...], Mapping[str, Any]]


@dataclass(frozen=True)
class WorkflowCommandSpec:
    name: str
    help: str
    arguments: Tuple[ArgumentDefinition, ...] = ()


def _argument(*flags: str, **kwargs: Any) -> ArgumentDefinition:
    return tuple(flags), dict(kwargs)


WORKFLOW_COMMAND_SPECS: Tuple[WorkflowCommandSpec, ...] = (
    WorkflowCommandSpec("adjudicate-claims", "Adjudicate every locked manuscript claim."),
    WorkflowCommandSpec("aggregate-results", "Aggregate provenance-preserving results."),
    WorkflowCommandSpec(
        "apply-remote-environment",
        "Apply and verify the locked remote environment through Slurm.",
    ),
    WorkflowCommandSpec("audit-artifacts", "Audit final artifact integrity and completeness."),
    WorkflowCommandSpec(
        "audit-graph-eligibility",
        "Freeze graph eligibility from pairing provenance.",
    ),
    WorkflowCommandSpec("audit-rgbt-inputs", "Audit visible and thermal branch inputs."),
    WorkflowCommandSpec(
        "build-and-audit-episode-features",
        "Build physically separated episode feature, edge, audit, and label stores.",
    ),
    WorkflowCommandSpec("build-episode-skeletons", "Freeze outcome-blind episode skeletons."),
    WorkflowCommandSpec(
        "build-minimal-sample",
        "Extract or construct only a bounded local module sample.",
        (_argument("--max-groups", type=int, required=True),),
    ),
    WorkflowCommandSpec(
        "close-gate",
        "Close one evidence gate after independent artifact validation.",
        (_argument("--gate", required=True),),
    ),
    WorkflowCommandSpec(
        "create-branch-test-seal",
        "Create the candidate branch-test seal.",
        (_argument("--candidate", action="store_true", required=True),),
    ),
    WorkflowCommandSpec(
        "create-fewshot",
        "Freeze paired raw-group few-shot subsets.",
        (
            _argument("--ratio", type=int, required=True),
            _argument("--pairs", type=int, required=True),
        ),
    ),
    WorkflowCommandSpec(
        "create-splits",
        "Create leakage-audited group-disjoint pools.",
        (_argument("--grouped", action="store_true", required=True),),
    ),
    WorkflowCommandSpec("decide-dataset-sources", "Select one licensed source per role."),
    WorkflowCommandSpec("dry-run-matrix", "Resolve the frozen minimal experiment matrix."),
    WorkflowCommandSpec(
        "evaluate",
        "Evaluate one released prediction package.",
        (_argument("--package", required=True),),
    ),
    WorkflowCommandSpec("fill-pretest-lock", "Fill and freeze the PRETEST protocol lock."),
    WorkflowCommandSpec(
        "finalize-branch-test-seal",
        "Finalize the branch-test seal against the PRETEST lock.",
        (_argument("--protocol", required=True),),
    ),
    WorkflowCommandSpec("finalize-closeout", "Finalize synchronized non-billing closeout."),
    WorkflowCommandSpec(
        "inventory-local-datasets",
        "Build a bounded local dataset-candidate inventory.",
        (_argument("--root", required=True),),
    ),
    WorkflowCommandSpec(
        "inventory-remote-datasets",
        "Build a bounded allocation-free remote dataset inventory.",
        (_argument("--bounded", action="store_true", required=True),),
    ),
    WorkflowCommandSpec("lock-methane-roles", "Freeze chronological methane data roles."),
    WorkflowCommandSpec("lock-ontology", "Freeze compatible labels and exclusions."),
    WorkflowCommandSpec(
        "release-branch-test",
        "Release branch labels after prediction locks.",
        (_argument("--packages", nargs="+", default=("T1", "S1", "V2", "R1")),),
    ),
    WorkflowCommandSpec("release-episode-test", "Release episode truth after prediction locks."),
    WorkflowCommandSpec("review-package", "Run the independent final package review."),
    WorkflowCommandSpec(
        "run-confirmatory-stats",
        "Run frozen paired intervals, effects, and multiplicity procedures.",
    ),
    WorkflowCommandSpec("seal-episode-test", "Finalize the episode-test seal."),
    WorkflowCommandSpec(
        "select-checkpoints",
        "Select checkpoints from validation evidence only.",
        (_argument("--family", required=True),),
    ),
    WorkflowCommandSpec(
        "smoke-local",
        "Run every module on the bounded local sample.",
        (_argument("--minimal", action="store_true", required=True),),
    ),
    WorkflowCommandSpec(
        "stage-datasets",
        "Reuse or transfer locked dataset archives according to source decisions.",
        (_argument("--decision", required=True),),
    ),
    WorkflowCommandSpec("update-manuscript", "Replace placeholders with linked evidence only."),
    WorkflowCommandSpec(
        "validate-artifact-contract",
        "Validate artifact schemas and cross-artifact constraints.",
    ),
    WorkflowCommandSpec(
        "validate-protocol",
        "Validate the protocol template and endpoint definitions.",
        (_argument("--template", action="store_true", required=True),),
    ),
)

PLAN_CLI_COMMANDS = frozenset(
    {"review-step", *(spec.name for spec in WORKFLOW_COMMAND_SPECS)}
)

SLURM_STEP_IDS = frozenset(
    {
        "E024",
        "E026",
        "E034",
        "E059",
        "E064",
        "E066",
        "E100",
        "E111",
        "E120",
        "E122",
        "E123",
        "E200",
        "E202",
        "E205",
        "E220",
        "E221",
        "E300",
        "E303",
        "E305",
        "E400",
    }
)


def add_workflow_parsers(subparsers: Any, handler: Any) -> None:
    for spec in WORKFLOW_COMMAND_SPECS:
        parser = subparsers.add_parser(spec.name, help=spec.help)
        for flags, kwargs in spec.arguments:
            parser.add_argument(*flags, **dict(kwargs))
        parser.set_defaults(handler=handler)


def command_names(specs: Sequence[WorkflowCommandSpec]) -> frozenset[str]:
    return frozenset(spec.name for spec in specs)
