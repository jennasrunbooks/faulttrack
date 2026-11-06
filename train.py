"""Train the final FaultTrack model and write the artifacts the API serves.

Run it from the repo root with ``make train`` (``uv run python train.py``). It reads
the committed feature table, so it needs neither ``data/raw/`` nor the network.

It writes, under ``<output_dir>`` (the repo root by default):

* ``models/model.bin``: ``faulttrack.models.final_model()`` fitted on all rows
  (pickled; ``predict.py`` loads it).
* ``models/feature_spec.json``: feature order, classes, window settings and the
  model version.
* ``web/results.json``: grouped-CV, leave-one-system-out and root-cause-rule
  figures, the baselines, the runbook tree and sample held-out cases.

Every split and model uses seed 42, so two runs give the same ``results.json``
apart from ``generated_at``.
"""

from __future__ import annotations

import json
import pickle
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier, export_text

from faulttrack.build import FEATURES_FILE
from faulttrack.evaluation import (
    LABELS,
    CVResult,
    grouped_cv,
    leave_one_system_out,
    suspect_hit_rate,
)
from faulttrack.features import (
    BASELINE_S,
    FEATURE_NAMES,
    LOG_Z,
    POST_S,
    Z_FLOOR,
)
from faulttrack.models import (
    FINAL_MODEL_NAME,
    SEED,
    RuleBaseline,
    final_model,
)

ROOT = Path(__file__).resolve().parent
FEATURES_PATH = ROOT / "data" / "features" / FEATURES_FILE

SHORT_NAMES = {"forest": "rf", "xgboost": "xgb", "logreg": "lr", "tree": "dt"}
"""Short model names for ``model_version`` (for example ``2026.11.10-rf``)."""


def main(
    output_dir: Path = ROOT, features_path: Path = FEATURES_PATH
) -> dict[str, Any]:
    """Evaluate and fit the final model, write the artifacts, print a summary.

    ``features_path`` is the committed feature table (1,350 rows). The files go
    to ``<output_dir>/models`` and ``<output_dir>/web``; existing ones are
    overwritten. Returns the ``results.json`` content.
    """
    df = pd.read_parquet(features_path)
    now = datetime.now(UTC)

    cv = grouped_cv(final_model(), df)
    loso = leave_one_system_out(final_model(), df)
    hit_rate = suspect_hit_rate(df)

    model = final_model().fit(df[FEATURE_NAMES], df["fault"])
    version = f"{now:%Y.%m.%d}-{SHORT_NAMES.get(FINAL_MODEL_NAME, FINAL_MODEL_NAME)}"

    spec = {
        "feature_names": list(FEATURE_NAMES),
        "classes": [str(c) for c in model.classes_],
        "baseline_s": BASELINE_S,
        "post_s": POST_S,
        "z_floor": Z_FLOOR,
        "log_z": LOG_Z,
        "norm_counts": False,
        "model_name": FINAL_MODEL_NAME,
        "model_params": _json_safe(model.get_params()),
        "model_version": version,
    }
    results = _results(df, cv, loso, hit_rate, version, now)

    models_dir = Path(output_dir) / "models"
    web_dir = Path(output_dir) / "web"
    models_dir.mkdir(parents=True, exist_ok=True)
    web_dir.mkdir(parents=True, exist_ok=True)
    with open(models_dir / "model.bin", "wb") as fh:
        pickle.dump(model, fh)
    _write_json(models_dir / "feature_spec.json", spec)
    _write_json(web_dir / "results.json", results)

    print(f"model {version}: {len(df)} rows, {df['case_id'].nunique()} cases")
    print(f"grouped 5-fold CV: {cv.summary()}")
    print(f"{loso.summary()}")
    print(f"root-cause rule hit rate: {hit_rate:.3f}")
    print(f"wrote {models_dir} and {web_dir}")
    return results


def _results(
    df: pd.DataFrame, cv: CVResult, loso: Any, hit_rate: float, version: str, now: Any
) -> dict[str, Any]:
    """Build the ``results.json`` content (plain floats, ints, lists and dicts)."""
    dummy = grouped_cv(DummyClassifier(strategy="most_frequent"), df)
    rule = grouped_cv(RuleBaseline(), df)
    tree = grouped_cv(_runbook_tree(), df)
    auc = cv.roc_auc_ovr
    stats = cv.to_dict()["macro_f1"]
    tree_stats = tree.to_dict()["macro_f1"]
    by_system = {
        str(system): suspect_hit_rate(rows) for system, rows in df.groupby("system")
    }
    return {
        "model_version": version,
        "n_rows": len(df),
        "n_cases": len(set(df["case_id"].astype(str))),
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fault_cv_macro_f1_mean": stats["mean"],
        "fault_cv_macro_f1_std": stats["std"],
        "fault_cv_macro_f1_min": stats["min"],
        "fault_cv_macro_f1_max": stats["max"],
        "fault_cv_accuracy_mean": cv.accuracy_mean,
        "fault_cv_roc_auc_ovr": None if auc is None else auc["macro"],
        "per_class": cv.per_class,
        "classes": list(LABELS),
        "confusion_matrix": cv.confusion.tolist(),
        "f1_by_delay": {str(d): s for d, s in cv.delay_summary().items()},
        "baselines": {
            "dummy": dummy.to_dict()["macro_f1"],
            "rule": rule.to_dict()["macro_f1"],
        },
        "loso": {
            "macro_f1": loso.macro_f1,
            "mean": loso.mean,
            "std": loso.std,
        },
        "suspect_hit_rate": hit_rate,
        "suspect_hit_rate_by_system": by_system,
        "runbook_tree": export_text(
            _runbook_tree().fit(df[FEATURE_NAMES], df["fault"]),
            feature_names=list(FEATURE_NAMES),
        ),
        "runbook_tree_cv_macro_f1": {
            "mean": tree_stats["mean"],
            "std": tree_stats["std"],
        },
        "sample_cases": _sample_cases(df, cv),
    }


def _runbook_tree() -> DecisionTreeClassifier:
    """The depth-3 tree of the notebook's on-call runbook plot."""
    return DecisionTreeClassifier(max_depth=3, random_state=SEED)


def _sample_cases(df: pd.DataFrame, cv: CVResult) -> list[dict[str, Any]]:
    """One out-of-fold prediction per fault, at ``delay_s == 0``.

    For each fault, the first ``case_id`` in sorted order, whether the model got
    it right or not. ``probability`` is the model's probability of the class it
    predicted; ``suspect_hit`` says whether the rule named the injected service.
    """
    if cv.y_proba is None:
        raise ValueError("the final model has no predict_proba")
    oof = pd.DataFrame(
        {
            "true": cv.y_true,
            "pred": cv.y_pred,
            "prob": cv.y_proba[
                range(len(cv.y_pred)), [LABELS.index(p) for p in cv.y_pred]
            ],
        },
        index=cv.index,
    )
    rows: list[dict[str, Any]] = []
    for fault in LABELS:
        chosen = oof[(oof["true"] == fault) & (df.loc[oof.index, "delay_s"] == 0)]
        first = df.loc[chosen.index].sort_values("case_id").index[0]
        rows.append(
            {
                "case_id": str(df.loc[first, "case_id"]),
                "system": str(df.loc[first, "system"]),
                "true_fault": fault,
                "predicted_fault": str(oof.loc[first, "pred"]),
                "probability": float(oof.loc[first, "prob"]),
                "suspect_service": str(df.loc[first, "suspect_service"]),
                "injected_service": str(df.loc[first, "service"]),
                "suspect_hit": bool(suspect_hit_rate(df.loc[[first]]) == 1.0),
            }
        )
    return rows


def _json_safe(params: dict[str, Any]) -> dict[str, Any]:
    """Keep only the hyperparameters that are plain JSON values."""
    return {
        key: value
        for key, value in params.items()
        if value is None or isinstance(value, bool | int | float | str)
    }


def _write_json(path: Path, content: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(content, fh, indent=2)
        fh.write("\n")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as e:
        print(f"train.py failed: {e}", file=sys.stderr)
        raise SystemExit(1) from e
