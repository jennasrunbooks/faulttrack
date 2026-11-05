"""Model definitions for FaultTrack: the baselines, the candidates and the final model.

The notebook compares ``candidate_models()`` and tunes them with
``tuned_search()``; ``final_model()`` is the chosen model with its tuned
hyperparameters, the single definition ``train.py`` fits. Every model here is a
scikit-learn-style classifier that ``faulttrack.evaluation.grouped_cv`` can
score, and every random component uses ``random_state=SEED`` (42).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from faulttrack.data import FAULTS, KINDS
from faulttrack.features import FEATURE_NAMES

SEED = 42
"""Random seed for every model and search."""

KIND_TO_FAULT: dict[str, str] = {
    "cpu": "cpu",
    "mem": "mem",
    "diskio": "disk",
    "socket": "socket",
    "latency": "delay",
    "error": "loss",
    "workload": "delay",
}
"""The rule baseline's mapping from the most-shifted metric kind to a fault."""


class RuleBaseline(ClassifierMixin, BaseEstimator):
    """ "The kind with the biggest shift wins": a no-ML baseline.

    For each row, finds the metric kind with the largest ``|<kind>_max|`` (a
    log-compressed z-score) and predicts the fault in ``KIND_TO_FAULT``. On a
    tie, the kind that comes first in ``KINDS`` wins (so an all-zero row is
    ``cpu``). ``fit`` learns nothing; it only records the label encoding:
    string labels give string predictions, and integer labels are read as
    indices into ``labels`` (``FAULTS`` order, as ``grouped_cv`` encodes them).

    ``X`` is a DataFrame with the ``<kind>_max`` columns, or an array whose
    columns are in ``FEATURE_NAMES`` order.
    """

    def __init__(self, labels: Sequence[str] = FAULTS) -> None:
        self.labels = labels

    def fit(self, X: Any, y: Any) -> RuleBaseline:
        """Record the classes in ``y`` and whether they are names or codes."""
        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)
        self.encoded_ = bool(np.issubdtype(y_arr.dtype, np.integer))
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Return one fault per row (names or ``labels`` indices, as in ``fit``)."""
        frame = (
            X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=FEATURE_NAMES)
        )
        shifts = frame.loc[:, [f"{kind}_max" for kind in KINDS]].abs().to_numpy(float)
        faults = np.array([KIND_TO_FAULT[k] for k in KINDS])[shifts.argmax(axis=1)]
        if self.encoded_:
            codes = {label: i for i, label in enumerate(self.labels)}
            return np.array([codes[f] for f in faults], dtype=int)
        return faults


def candidate_models() -> dict[str, Any]:
    """The untuned models compared in the notebook, simplest first.

    Keys: ``dummy`` (most frequent class), ``rule`` (``RuleBaseline``),
    ``logreg`` (``StandardScaler`` + ``LogisticRegression`` in a ``Pipeline``),
    ``tree`` (decision tree), ``forest`` (random forest) and ``xgboost``. All use
    library defaults apart from ``random_state=42`` and settings needed to run.
    Models run single-threaded (``n_jobs=1``); searches parallelize over
    candidates instead, which avoids thread oversubscription.
    """
    return {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "rule": RuleBaseline(),
        "logreg": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=5000, random_state=SEED)),
            ]
        ),
        "tree": DecisionTreeClassifier(random_state=SEED),
        "forest": RandomForestClassifier(random_state=SEED, n_jobs=1),
        "xgboost": XGBClassifier(
            random_state=SEED, n_jobs=1, tree_method="hist", eval_metric="mlogloss"
        ),
    }


def param_grids() -> dict[str, dict[str, list[Any]]]:
    """The hyperparameter grids searched for each tunable candidate.

    Keys match ``candidate_models()``; parameter names use the ``Pipeline`` step
    prefix (``model__C``) where needed. Each grid has 13 to 36 combinations.
    """
    return {
        "logreg": {"model__C": [float(c) for c in np.logspace(-3, 3, 13)]},
        "tree": {
            "max_depth": [2, 3, 4, 5, 6, 8, 10, None],
            "min_samples_leaf": [1, 5, 10, 20],
        },
        "forest": {
            "n_estimators": [200, 500],
            "max_depth": [None, 8, 16],
            "min_samples_leaf": [1, 2, 5, 10],
        },
        "xgboost": {
            "learning_rate": [0.05, 0.1, 0.3],
            "max_depth": [3, 5, 7],
            "min_child_weight": [1, 5],
            "n_estimators": [100, 300],
        },
    }


def tuned_search(name: str, cv: Any = None) -> GridSearchCV:
    """A ``GridSearchCV`` over ``param_grids()[name]``, scored on macro F1.

    ``cv`` defaults to ``GroupKFold(5)``, which needs ``groups=``:
    ``grouped_cv`` passes the training fold's ``case_id``, so the search is
    nested and grouped. For tuning on the whole table, pass
    ``cv=list(faulttrack.evaluation.grouped_splits(df))`` instead.
    """
    return GridSearchCV(
        candidate_models()[name],
        param_grids()[name],
        cv=GroupKFold(n_splits=5) if cv is None else cv,
        scoring="f1_macro",
        n_jobs=-1,
    )


FINAL_MODEL_NAME = "forest"
"""Which candidate the notebook chose (section 6)."""

FINAL_PARAMS: dict[str, Any] = {
    "max_depth": None,
    "min_samples_leaf": 2,
    "n_estimators": 200,
}
"""Tuned hyperparameters of the final model, from whole-table grouped tuning."""


def final_model() -> Any:
    """The chosen model with its tuned hyperparameters (unfitted).

    This is the single definition ``train.py`` fits and the notebook scores.
    """
    model = candidate_models()[FINAL_MODEL_NAME]
    return model.set_params(**FINAL_PARAMS)
