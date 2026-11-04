"""Grouped cross-validation for FaultTrack. The only CV code in the project.

Every split here is grouped by ``case_id``: the 5 detection-delay rows of one
case always land on the same side of a split, so windows of one incident never
leak from train into test. The notebook, ``train.py`` and the plan line 7
experiments call these functions instead of writing their own CV loops.

Input frames are the feature table (``data/features/re2_features.parquet``) or
a subset of it: the identity columns ``case_id``, ``system``, ``fault`` (the
label) and, where present, ``delay_s`` and ``suspect_service``, plus every name
in ``FEATURE_NAMES``. Models only ever see the ``FEATURE_NAMES`` columns.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupKFold

from faulttrack.data import FAULTS
from faulttrack.features import FEATURE_NAMES

LABELS: tuple[str, ...] = FAULTS
"""Class order for encoded labels, per-class metrics and the confusion matrix."""

SEED = 42
"""Default random seed for fold assignment."""

Split = tuple[np.ndarray, np.ndarray]


@dataclass
class CVResult:
    """Cross-validation result: per-fold scores plus pooled out-of-fold output.

    ``fold_macro_f1`` and ``fold_accuracy`` hold one score per test fold. The
    mean, std (population std, ``ddof=0``), min and max are across folds.
    ``y_true`` and ``y_pred`` are fault labels for every row of the input frame
    that was in a test fold, in input order (``index`` holds the frame's index).
    ``per_class`` and ``confusion`` are computed on the pooled out-of-fold
    predictions, with classes in ``LABELS`` order (confusion rows are true
    labels, columns predicted). ``delay_macro_f1`` maps each detection delay
    (seconds) to its per-fold macro F1 scores, when the frame has ``delay_s``.
    """

    fold_macro_f1: list[float]
    fold_accuracy: list[float]
    y_true: np.ndarray
    y_pred: np.ndarray
    index: pd.Index
    labels: tuple[str, ...] = LABELS
    delay_macro_f1: dict[int, list[float]] = field(default_factory=dict)

    @property
    def n_folds(self) -> int:
        return len(self.fold_macro_f1)

    @property
    def mean(self) -> float:
        """Mean macro F1 across folds."""
        return float(np.mean(self.fold_macro_f1))

    @property
    def std(self) -> float:
        """Std of macro F1 across folds (``ddof=0``, as in the spike)."""
        return float(np.std(self.fold_macro_f1))

    @property
    def min(self) -> float:
        return float(np.min(self.fold_macro_f1))

    @property
    def max(self) -> float:
        return float(np.max(self.fold_macro_f1))

    @property
    def accuracy_mean(self) -> float:
        return float(np.mean(self.fold_accuracy))

    @property
    def accuracy_std(self) -> float:
        return float(np.std(self.fold_accuracy))

    @property
    def pooled_macro_f1(self) -> float:
        """Macro F1 on all out-of-fold predictions at once (not the headline)."""
        return float(
            f1_score(
                self.y_true,
                self.y_pred,
                labels=list(self.labels),
                average="macro",
                zero_division=0,  # pyright: ignore[reportArgumentType]
            )
        )

    @property
    def per_class(self) -> dict[str, dict[str, float]]:
        """Pooled precision, recall, F1 and support per class, in label order."""
        precision, recall, f1, support = precision_recall_fscore_support(
            self.y_true,
            self.y_pred,
            labels=list(self.labels),
            zero_division=0,  # pyright: ignore[reportArgumentType]
        )
        return {
            label: {
                "precision": float(np.asarray(precision)[i]),
                "recall": float(np.asarray(recall)[i]),
                "f1": float(np.asarray(f1)[i]),
                "support": int(np.asarray(support)[i]),
            }
            for i, label in enumerate(self.labels)
        }

    @property
    def confusion(self) -> np.ndarray:
        """Pooled confusion matrix: rows true, columns predicted, label order."""
        return confusion_matrix(self.y_true, self.y_pred, labels=list(self.labels))

    def delay_summary(self) -> dict[int, dict[str, float]]:
        """Per detection delay (seconds): macro F1 mean, std, min and max."""
        return {delay: _stats(scores) for delay, scores in self.delay_macro_f1.items()}

    def summary(self) -> str:
        """One line, for example ``macro F1 0.828 ± 0.039 (min 0.769, max 0.875)``."""
        return (
            f"macro F1 {self.mean:.3f} ± {self.std:.3f} "
            f"(min {self.min:.3f}, max {self.max:.3f})"
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready summary (plain floats, ints, lists and dicts)."""
        return {
            "n_folds": self.n_folds,
            "labels": list(self.labels),
            "macro_f1": _stats(self.fold_macro_f1),
            "accuracy": _stats(self.fold_accuracy),
            "fold_macro_f1": [float(v) for v in self.fold_macro_f1],
            "fold_accuracy": [float(v) for v in self.fold_accuracy],
            "pooled_macro_f1": self.pooled_macro_f1,
            "per_class": self.per_class,
            "confusion_matrix": self.confusion.tolist(),
            "macro_f1_by_delay": {
                str(delay): stats for delay, stats in self.delay_summary().items()
            },
        }


@dataclass
class LOSOResult:
    """Leave-one-system-out result: one ``CVResult`` per held-out system.

    Each per-system result has a single fold (train on the other systems, test
    on that one). ``mean`` and ``std`` (``ddof=0``) are across systems.
    """

    per_system: dict[str, CVResult]

    @property
    def macro_f1(self) -> dict[str, float]:
        """Macro F1 on each held-out system."""
        return {s: r.fold_macro_f1[0] for s, r in self.per_system.items()}

    @property
    def mean(self) -> float:
        return float(np.mean(list(self.macro_f1.values())))

    @property
    def std(self) -> float:
        return float(np.std(list(self.macro_f1.values())))

    def summary(self) -> str:
        """One line, for example ``LOSO macro F1 0.364 ± 0.112 (re2ob 0.521, ...)``."""
        parts = ", ".join(f"{s} {v:.3f}" for s, v in self.macro_f1.items())
        return f"LOSO macro F1 {self.mean:.3f} ± {self.std:.3f} ({parts})"

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready summary: per-system macro F1 and results, mean and std."""
        return {
            "macro_f1": self.macro_f1,
            "mean": self.mean,
            "std": self.std,
            "per_system": {s: r.to_dict() for s, r in self.per_system.items()},
        }


def grouped_splits(
    df: pd.DataFrame, n_splits: int = 5, seed: int = SEED
) -> Iterator[Split]:
    """Yield (train, test) positional indices from ``GroupKFold`` by ``case_id``.

    Groups are shuffled with ``seed`` before being assigned to folds, so fold
    membership is fixed for a given seed and table. No ``case_id`` is ever in
    both train and test.
    """
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    yield from splitter.split(df, groups=df["case_id"].to_numpy())


def system_splits(df: pd.DataFrame) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Yield (held-out system, train, test) positional indices, one per system.

    Systems come in sorted order. Each split tests on exactly one whole system
    and trains on all the others.
    """
    systems = df["system"].to_numpy()
    for system in sorted(pd.unique(systems)):
        test = systems == system
        yield str(system), np.flatnonzero(~test), np.flatnonzero(test)


def grouped_cv(
    model: Any,
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = SEED,
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> CVResult:
    """Grouped k-fold CV of ``model`` on the feature table ``df``.

    Folds come from ``grouped_splits`` (``GroupKFold`` by ``case_id``, shuffled
    with ``seed``). ``model`` is any scikit-learn-style classifier; it is cloned
    for every fold, fit on the ``feature_names`` columns only (``FEATURE_NAMES``
    by default, never the identity columns), and trained on integer labels in
    ``LABELS`` order so XGBoost works too. Returns a ``CVResult``.
    """
    return _cross_validate(
        model, df, list(grouped_splits(df, n_splits, seed)), feature_names
    )


def leave_one_system_out(
    model: Any,
    df: pd.DataFrame,
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> LOSOResult:
    """Train on all systems but one and test on the held-out one, for each system.

    Uses ``system_splits``; ``model`` is cloned per split and sees only the
    ``feature_names`` columns. Returns per-system macro F1 with the mean and std
    across systems (``LOSOResult``).
    """
    per_system = {
        system: _cross_validate(model, df, [(train, test)], feature_names)
        for system, train, test in system_splits(df)
    }
    return LOSOResult(per_system=per_system)


def suspect_hit_rate(df: pd.DataFrame) -> float:
    """Share of rows where the rule's ``suspect_service`` is the injected service.

    Each row is one alert window (case × detection delay), so this is the hit
    rate per alert, as in the spike. Names are compared after removing a
    trailing ``-db``, the same way ``data.kind_of`` merges a service with its
    database. A missing suspect counts as a miss. Returns a value in [0, 1].
    """
    if df.empty:
        raise ValueError("empty frame: no rows to score")
    suspect = df["suspect_service"].fillna("").astype(str).str.removesuffix("-db")
    injected = df["service"].astype(str).str.removesuffix("-db")
    return float((suspect == injected).mean())


def _cross_validate(
    model: Any,
    df: pd.DataFrame,
    splits: list[Split],
    feature_names: Sequence[str],
) -> CVResult:
    unknown = sorted(set(df["fault"]) - set(LABELS))
    if unknown:
        raise ValueError(f"unknown fault labels: {unknown}")
    x = df.loc[:, list(feature_names)]
    codes = {label: i for i, label in enumerate(LABELS)}
    y = np.array([codes[label] for label in df["fault"]], dtype=int)
    delays = df["delay_s"].to_numpy() if "delay_s" in df.columns else None
    pred = np.full(len(df), -1)
    fold_f1: list[float] = []
    fold_acc: list[float] = []
    delay_f1: dict[int, list[float]] = {}
    for train, test in splits:
        estimator: Any = clone(model)
        fitted = estimator.fit(x.iloc[train], y[train])
        pred[test] = np.asarray(fitted.predict(x.iloc[test])).astype(int)
        fold_f1.append(_macro_f1(y[test], pred[test]))
        fold_acc.append(float(accuracy_score(y[test], pred[test])))
        if delays is not None:
            for delay in sorted(pd.unique(delays[test])):
                rows = test[delays[test] == delay]
                score = _macro_f1(y[rows], pred[rows])
                delay_f1.setdefault(int(delay), []).append(score)
    tested = pred >= 0
    names = np.asarray(LABELS)
    return CVResult(
        fold_macro_f1=fold_f1,
        fold_accuracy=fold_acc,
        y_true=names[y[tested]],
        y_pred=names[pred[tested]],
        index=df.index[tested],
        delay_macro_f1=dict(sorted(delay_f1.items())),
    )


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro F1 over the classes present in ``y_true`` or ``y_pred``."""
    return float(
        f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,  # pyright: ignore[reportArgumentType]
        )
    )


def _stats(scores: Sequence[float]) -> dict[str, float]:
    values = np.asarray(scores, dtype=float)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }
