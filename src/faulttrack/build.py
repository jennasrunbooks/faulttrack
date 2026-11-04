"""Build the committed feature table and case inventory (``make features``).

Run from the repo root after ``make data``::

    uv run python -m faulttrack.build            # write data/features/*.parquet
    uv run python -m faulttrack.build --check    # rebuild in memory, compare

Writes two files to ``data/features/``:

- ``re2_features.parquet``: one row per usable case and detection delay
  (0, 30, 60, 90, 120 s), with the identity columns ``case_id``, ``system``,
  ``service``, ``fault``, ``delay_s`` and ``suspect_service`` (the root-cause
  rule's pick, never a model input), then ``FEATURE_NAMES``.
- ``re2_inventory.parquet``: one row per case folder, skipped ones included,
  with its size, metric columns per kind, missing values and skip reason.

Rows are sorted (features by ``case_id`` and ``delay_s``, inventory by
``case_id``), so a rebuild gives an identical table.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from faulttrack.data import (
    KINDS,
    METRICS_FILE,
    Case,
    clean_metrics,
    find_cases,
    kind_of,
    read_inject_time,
    read_metrics,
)
from faulttrack.features import DELAYS_S, FEATURE_NAMES, featurize

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/features")
FEATURES_FILE = "re2_features.parquet"
INVENTORY_FILE = "re2_inventory.parquet"

ID_COLUMNS: list[str] = [
    "case_id",
    "system",
    "service",
    "fault",
    "delay_s",
    "suspect_service",
]
"""Identity columns of the feature table, before ``FEATURE_NAMES``. Never inputs."""

INVENTORY_COLUMNS: list[str] = [
    "case_id",
    "system",
    "service",
    "fault",
    "repetition",
    "status",
    "n_rows",
    "n_metric_cols",
    *[f"n_cols_{kind}" for kind in KINDS],
    "nan_pct",
    "inf_count",
    "inject_pos",
    "n_feature_rows",
]
"""Inventory columns, in order."""

STATUS_OK = "ok"


def case_feature_rows(
    case: Case, metrics: pd.DataFrame, inject_time: int, delays: Sequence[int]
) -> list[dict[str, Any]]:
    """Feature rows for one case, one per detection delay (seconds).

    ``metrics`` is the cleaned frame and ``inject_time`` is in Unix seconds;
    each row uses ``t_det = inject_time + delay``. A delay whose windows don't
    fit in the file is skipped with a message on stderr.
    """
    rows: list[dict[str, Any]] = []
    for delay in delays:
        try:
            result = featurize(metrics, inject_time + delay)
        except ValueError as e:
            print(f"skip {case.case_id} delay {delay}s: {e}", file=sys.stderr)
            continue
        rows.append(
            {
                "case_id": case.case_id,
                "system": case.system,
                "service": case.service,
                "fault": case.fault,
                "delay_s": int(delay),
                "suspect_service": result.suspect_service,
                **result.features,
            }
        )
    return rows


def inventory_row(
    case: Case,
    status: str,
    raw: pd.DataFrame | None = None,
    metrics: pd.DataFrame | None = None,
    inject_time: int | None = None,
    n_feature_rows: int = 0,
) -> dict[str, Any]:
    """One inventory row for a case.

    ``raw`` is the metrics file as read (for NaN and ``inf`` counts over its
    metric columns), ``metrics`` the cleaned frame (for row and column counts),
    ``inject_time`` Unix seconds. ``inject_pos`` is where the injection falls
    in the file's time range, from 0 (first sample) to 1 (last). Values that
    can't be computed (a skipped case) are missing.
    """
    row: dict[str, Any] = dict.fromkeys(INVENTORY_COLUMNS)
    row.update(
        case_id=case.case_id,
        system=case.system,
        service=case.service,
        fault=case.fault,
        repetition=case.repetition,
        status=status,
        n_feature_rows=n_feature_rows,
    )
    if raw is not None:
        value_cols = [
            str(c)
            for c in raw.columns
            if str(c) != "time" and not str(c).startswith("time.")
        ]
        values = raw[value_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        row["n_rows"] = len(raw)
        row["nan_pct"] = float(np.isnan(values).mean() * 100) if values.size else 0.0
        row["inf_count"] = int(np.isinf(values).sum())
    if metrics is not None:
        columns = [str(c) for c in metrics.columns if str(c) != "time"]
        mapped = [kind_of(c) for c in columns]
        row["n_metric_cols"] = len(columns)
        for kind in KINDS:
            row[f"n_cols_{kind}"] = sum(m is not None and m[1] == kind for m in mapped)
        if inject_time is not None and len(metrics) > 1:
            times = metrics["time"].to_numpy(dtype=float)
            t_min, t_max = float(times.min()), float(times.max())
            if t_max > t_min:
                row["inject_pos"] = (inject_time - t_min) / (t_max - t_min)
    return row


def build_tables(
    root: str | PathLike[str] = RAW_DIR, delays: Sequence[int] = DELAYS_S
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the feature table and the inventory from the raw RE2 cases.

    ``root`` is the folder ``make data`` fills (``data/raw``). Returns
    ``(features, inventory)``: the feature table has ``ID_COLUMNS`` then
    ``FEATURE_NAMES``, sorted by ``case_id`` and ``delay_s``; the inventory has
    ``INVENTORY_COLUMNS``, one row per case folder, sorted by ``case_id``.
    Unreadable cases are skipped with a message on stderr and get their reason
    in the inventory's ``status``.
    """
    feature_rows: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    for case in find_cases(root, include_unusable=True):
        metrics_path = case.path / METRICS_FILE
        if not metrics_path.is_file():
            print(f"skip {case.case_id}: no {METRICS_FILE}", file=sys.stderr)
            inventory.append(inventory_row(case, f"no {METRICS_FILE}"))
            continue
        try:
            raw = read_metrics(metrics_path)
            metrics = clean_metrics(raw)
            inject_time = read_inject_time(case.path)
        except (OSError, ValueError) as e:
            print(f"skip {case.case_id}: {e}", file=sys.stderr)
            inventory.append(inventory_row(case, f"unreadable: {e}"))
            continue
        rows = case_feature_rows(case, metrics, inject_time, delays)
        feature_rows.extend(rows)
        status = STATUS_OK if rows else "no feature rows: windows outside the file"
        inventory.append(
            inventory_row(case, status, raw, metrics, inject_time, len(rows))
        )
    if not inventory:
        raise ValueError(f"no RCAEval cases found under {root}; run `make data`")
    return _features_frame(feature_rows), _inventory_frame(inventory)


def _features_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=ID_COLUMNS + FEATURE_NAMES)
    frame = frame.astype({"delay_s": "int64"})
    frame = frame.sort_values(["case_id", "delay_s"], kind="mergesort")
    return frame.reset_index(drop=True)


def _inventory_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=INVENTORY_COLUMNS)
    counts = ["n_rows", "n_metric_cols", *[f"n_cols_{k}" for k in KINDS], "inf_count"]
    frame = frame.astype(
        {
            **dict.fromkeys(counts, "Int64"),
            "repetition": "int64",
            "n_feature_rows": "int64",
            "nan_pct": "float64",
            "inject_pos": "float64",
        }
    )
    frame = frame.sort_values("case_id", kind="mergesort")
    return frame.reset_index(drop=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; returns the exit code."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--raw", type=Path, default=RAW_DIR, help="raw data folder")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="output folder")
    parser.add_argument(
        "--check",
        action="store_true",
        help="rebuild in memory and compare with the files in --out; write nothing",
    )
    args = parser.parse_args(argv)
    features, inventory = build_tables(args.raw)
    features_path = args.out / FEATURES_FILE
    inventory_path = args.out / INVENTORY_FILE

    n_cases = features["case_id"].nunique()
    print(f"{len(features)} feature rows from {n_cases} cases")
    print(f"{len(inventory)} inventory rows; status counts:")
    print(inventory["status"].value_counts().to_string())
    print("cases per system x fault:")
    print(
        features.groupby(["system", "fault"])["case_id"]
        .nunique()
        .unstack(fill_value=0)
        .to_string()
    )

    if args.check:
        try:
            pd.testing.assert_frame_equal(features, pd.read_parquet(features_path))
            pd.testing.assert_frame_equal(inventory, pd.read_parquet(inventory_path))
        except (AssertionError, OSError) as e:
            print(f"check failed: {e}", file=sys.stderr)
            return 1
        print(f"check passed: {features_path} and {inventory_path} match a rebuild")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    features.to_parquet(features_path, index=False)
    inventory.to_parquet(inventory_path, index=False)
    print(f"wrote {features_path} and {inventory_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
