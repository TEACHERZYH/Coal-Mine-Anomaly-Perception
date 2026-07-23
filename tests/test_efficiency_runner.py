from __future__ import annotations

import pytest

from mining1_exp.efficiency import select_representative
from mining1_exp.workflow_common import WorkflowExecutionError


def _candidate(seed: int, metric: float) -> dict:
    return {
        "status": "pass",
        "family_id": "S1-GRU",
        "train_seed": seed,
        "selection_pool": "D_b_sel",
        "selection_metric_value": metric,
    }


def test_efficiency_representative_is_median_validation_rank_then_seed() -> None:
    selected = select_representative(
        [_candidate(4219, 0.70), _candidate(1701, 0.80), _candidate(2903, 0.70)],
        family_id="S1-GRU",
        expected_count=3,
        selection_pool="D_b_sel",
    )
    assert selected["train_seed"] == 2903


def test_efficiency_representative_rejects_missing_repeat() -> None:
    with pytest.raises(WorkflowExecutionError, match="requires 3"):
        select_representative(
            [_candidate(1701, 0.80), _candidate(2903, 0.70)],
            family_id="S1-GRU",
            expected_count=3,
            selection_pool="D_b_sel",
        )
