"""Tests for faulttrack.evaluation on small synthetic feature tables (no data/raw/)."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier

from faulttrack.evaluation import (
    LABELS,
    grouped_cv,
    grouped_splits,
    leave_one_system_out,
    suspect_hit_rate,
    system_splits,
)
from faulttrack.features import DELAYS_S, FEATURE_NAMES

SYSTEMS = ("re2ob", "re2ss", "re2tt")


def _table(cases_per_cell: int = 4, seed: int = 0) -> pd.DataFrame:
    """A balanced synthetic feature table: 3 systems x 6 faults x N cases x 5 delays.

    Each fault raises one feature, so a tree model can learn it, plus noise.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for system in SYSTEMS:
        for f_idx, fault in enumerate(LABELS):
            for rep in range(1, cases_per_cell + 1):
                case_id = f"{system}/svc{f_idx}_{fault}/{rep}"
                for delay in DELAYS_S:
                    values = rng.normal(0.0, 1.0, len(FEATURE_NAMES))
                    values[f_idx] += 3.0
                    rows.append(
                        {
                            "case_id": case_id,
                            "system": system,
                            "service": f"svc{f_idx}",
                            "fault": fault,
                            "delay_s": delay,
                            "suspect_service": f"svc{f_idx}",
                            **dict(zip(FEATURE_NAMES, values, strict=True)),
                        }
                    )
    return pd.DataFrame(rows)


class _ColumnSpy(ClassifierMixin, BaseEstimator):
    """Records the columns it was fit on; predicts the first class."""

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "_ColumnSpy":
        self.columns_ = list(X.columns)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.classes_[0])


def test_no_case_in_both_train_and_test() -> None:
    df = _table()
    splits = list(grouped_splits(df, n_splits=5))
    assert len(splits) == 5
    cases = df["case_id"].to_numpy()
    for train, test in splits:
        assert not set(cases[train]) & set(cases[test])
    # every row is tested exactly once
    tested = np.concatenate([test for _, test in splits])
    assert sorted(tested) == list(range(len(df)))


def test_grouped_cv_has_one_score_per_fold_and_pools_every_row() -> None:
    df = _table()
    result = grouped_cv(DummyClassifier(strategy="most_frequent"), df)
    assert result.n_folds == 5
    assert len(result.y_pred) == len(df)
    assert set(result.delay_macro_f1) == set(DELAYS_S)
    assert all(len(v) == 5 for v in result.delay_macro_f1.values())


def test_model_is_cloned_and_fit_on_feature_names_only() -> None:
    df = _table()
    spy = _ColumnSpy()
    grouped_cv(spy, df)
    assert not hasattr(spy, "columns_")  # the caller's model is never fit
    fitted: list[list[str]] = []

    class _Recorder(_ColumnSpy):
        def fit(self, X: pd.DataFrame, y: np.ndarray) -> _ColumnSpy:
            fitted.append(list(X.columns))
            return super().fit(X, y)

    grouped_cv(_Recorder(), df)
    assert fitted == [FEATURE_NAMES] * 5


def test_leave_one_system_out_holds_out_exactly_one_system() -> None:
    df = _table()
    systems = df["system"].to_numpy()
    seen = []
    for held_out, train, test in system_splits(df):
        assert set(systems[test]) == {held_out}
        assert held_out not in set(systems[train])
        assert len(train) + len(test) == len(df)
        seen.append(held_out)
    assert seen == sorted(SYSTEMS)

    result = leave_one_system_out(RandomForestClassifier(random_state=42), df)
    assert list(result.macro_f1) == sorted(SYSTEMS)
    assert result.mean == pytest.approx(np.mean(list(result.macro_f1.values())))


def test_same_seed_and_model_give_identical_results() -> None:
    df = _table()
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    first = grouped_cv(model, df, seed=42)
    second = grouped_cv(model, df, seed=42)
    assert first.fold_macro_f1 == second.fold_macro_f1
    assert np.array_equal(first.y_pred, second.y_pred)
    assert first.to_dict() == second.to_dict()


def test_dummy_on_balanced_classes_scores_about_one_sixth() -> None:
    df = _table(cases_per_cell=5)
    result = grouped_cv(DummyClassifier(strategy="uniform", random_state=42), df)
    # "uniform" guesses at random; "most_frequent" would pick the class that is
    # rarest in each test fold (it's held out of train), so it scores below 1/6
    pooled_accuracy = float(np.mean(result.y_true == result.y_pred))
    assert pooled_accuracy == pytest.approx(1 / 6, abs=0.04)


def test_learnable_signal_scores_well_and_reports_cleanly() -> None:
    df = _table()
    result = grouped_cv(RandomForestClassifier(random_state=42), df)
    assert result.mean > 0.8
    summary = result.summary()
    assert summary.startswith("macro F1 ") and "± " in summary and "min " in summary
    as_dict = result.to_dict()
    json.dumps(as_dict)  # JSON-ready
    assert as_dict["labels"] == list(LABELS)
    assert np.asarray(as_dict["confusion_matrix"]).shape == (6, 6)
    assert set(as_dict["per_class"]) == set(LABELS)


def test_unknown_label_is_rejected() -> None:
    df = _table()
    df.loc[0, "fault"] = "network"
    with pytest.raises(ValueError, match="unknown fault labels"):
        grouped_cv(DummyClassifier(), df)


def test_suspect_hit_rate_counts_rows_and_merges_db_suffix() -> None:
    df = pd.DataFrame(
        {
            "service": ["carts", "carts-db", "orders", "orders"],
            "suspect_service": ["carts", "carts", "payment", None],
        }
    )
    assert suspect_hit_rate(df) == pytest.approx(0.5)
