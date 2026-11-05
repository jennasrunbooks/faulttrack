"""Tests for faulttrack.models: the rule baseline, candidates and final model."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV

from faulttrack.data import FAULTS, KINDS
from faulttrack.evaluation import grouped_cv
from faulttrack.features import FEATURE_NAMES
from faulttrack.models import (
    KIND_TO_FAULT,
    RuleBaseline,
    candidate_models,
    final_model,
    param_grids,
    tuned_search,
)


def _rows(kind_shifts: list[dict[str, float]]) -> pd.DataFrame:
    """Feature rows that are 0 everywhere except the given ``<kind>_max`` values."""
    rows = []
    for shifts in kind_shifts:
        row = dict.fromkeys(FEATURE_NAMES, 0.0)
        row.update({f"{k}_max": v for k, v in shifts.items()})
        rows.append(row)
    return pd.DataFrame(rows, columns=FEATURE_NAMES)


def test_rule_maps_every_kind_to_its_fault() -> None:
    assert set(KIND_TO_FAULT) == set(KINDS)
    assert set(KIND_TO_FAULT.values()) == set(FAULTS)
    assert KIND_TO_FAULT["diskio"] == "disk"
    assert KIND_TO_FAULT["latency"] == KIND_TO_FAULT["workload"] == "delay"
    assert KIND_TO_FAULT["error"] == "loss"


def test_rule_uses_the_largest_absolute_shift() -> None:
    X = _rows(
        [
            {"cpu": 2.0, "mem": -3.0},  # a drop counts by its size
            {"diskio": 1.0, "latency": 0.5},
            {"error": 4.0, "latency": 3.9},
            {"workload": -2.5, "socket": 2.4},
            {},  # all zero: the first kind (cpu) wins the tie
        ]
    )
    rule = RuleBaseline().fit(X, np.array(["cpu", "mem"]))
    assert rule.predict(X).tolist() == ["mem", "disk", "loss", "delay", "cpu"]


def test_rule_returns_codes_when_trained_on_codes() -> None:
    X = _rows([{"socket": 5.0}, {"mem": 1.0}])
    rule = RuleBaseline().fit(X, np.array([0, 1, 2]))
    assert rule.predict(X).tolist() == [FAULTS.index("socket"), FAULTS.index("mem")]
    # positional arrays in FEATURE_NAMES order work too
    assert rule.predict(X.to_numpy()).tolist() == rule.predict(X).tolist()


def test_rule_works_with_grouped_cv() -> None:
    rows = []
    for c, fault in enumerate(FAULTS):
        kind = next(k for k, f in KIND_TO_FAULT.items() if f == fault)
        for rep in range(5):
            row = dict.fromkeys(FEATURE_NAMES, 0.0)
            row[f"{kind}_max"] = 3.0
            rows.append({"case_id": f"c{c}_{rep}", "fault": fault, **row})
    df = pd.DataFrame(rows)
    result = grouped_cv(RuleBaseline(), df)
    assert result.mean == pytest.approx(1.0)
    cloned: Any = clone(RuleBaseline())
    assert cloned.get_params() == {"labels": FAULTS}


def test_candidates_are_seeded_and_grids_match() -> None:
    models = candidate_models()
    assert list(models) == ["dummy", "rule", "logreg", "tree", "forest", "xgboost"]
    for name in ("tree", "forest", "xgboost"):
        assert models[name].get_params()["random_state"] == 42
    assert models["logreg"].get_params()["model__random_state"] == 42
    for name, grid in param_grids().items():
        assert set(grid) <= set(models[name].get_params())
        n_combinations = int(np.prod([len(v) for v in grid.values()]))
        assert 10 <= n_combinations <= 40
        search = tuned_search(name)
        assert isinstance(search, GridSearchCV)
        assert search.scoring == "f1_macro"


def test_final_model_is_a_tuned_candidate() -> None:
    model = final_model()
    assert model.get_params()["random_state"] == 42
    # a fresh object each call, so train.py and the notebook can't share state
    assert final_model() is not model


def test_final_model_reproduces_the_notebook_headline() -> None:
    # The committed feature table (no data/raw/ needed). Pins the numbers the
    # notebook (section 6) and docs/decisions.md report for final_model().
    table = Path(__file__).parents[1] / "data" / "features" / "re2_features.parquet"
    result = grouped_cv(final_model(), pd.read_parquet(table))
    assert result.mean == pytest.approx(0.799, abs=5e-4)
    assert result.std == pytest.approx(0.028, abs=5e-4)
    assert result.min == pytest.approx(0.775, abs=5e-4)
    assert result.max == pytest.approx(0.850, abs=5e-4)
