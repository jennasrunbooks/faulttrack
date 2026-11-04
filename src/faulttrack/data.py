"""RCAEval RE2 case discovery, loading and dataset quirks (spec Section 3.3).

This is the only module that knows the RCAEval folder layout. It returns a tidy
metrics frame (one ``time`` column in Unix seconds plus ``<service>_<metric>``
columns), the injection time and the label. It computes no features.

Layout (Zenodo copy)::

    <root>/RE2-XX/RE2-XX/<service>_<fault>/<n>/
        simple_metrics.csv   # the only metrics file the features use
        metrics.csv          # raw Prometheus export, not used
        inject_time.txt      # fault injection time, Unix seconds
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd

FAULTS: tuple[str, ...] = ("cpu", "mem", "disk", "socket", "delay", "loss")
"""Fault labels in RE2, as they appear in case folder names."""

KINDS: tuple[str, ...] = (
    "cpu",
    "mem",
    "diskio",
    "socket",
    "latency",
    "error",
    "workload",
)
"""Metric kinds the features group columns by, in feature order."""

METRICS_FILE = "simple_metrics.csv"
RAW_METRICS_FILE = "metrics.csv"
INJECT_TIME_FILE = "inject_time.txt"

# A file with more than this many container-/istio- columns is a raw Prometheus
# export (RE2 metrics.csv), whose column names the feature code doesn't parse.
_RAW_COLUMN_LIMIT = 20
_LABEL_RE = re.compile(r"^(?P<service>.+)_(?P<fault>[a-z]+)$")


@dataclass(frozen=True)
class Case:
    """One RCAEval case folder.

    ``system`` is the lowercased suite name without dashes (``re2ob``),
    ``case_id`` is ``<system>/<service>_<fault>/<repetition>`` and is the CV group.
    """

    system: str
    service: str
    fault: str
    repetition: int
    case_id: str
    path: Path


@dataclass(frozen=True)
class LoadedCase:
    """A cleaned case: metrics frame, injection time (Unix seconds) and label."""

    metrics: pd.DataFrame
    inject_time: int
    service: str
    fault: str


def parse_label(folder_name: str) -> tuple[str, str]:
    """Split a ``<service>_<fault>`` folder name into (service, fault).

    Service names keep their dataset spelling (``ts-order-service``,
    ``front-end``) so they match the metric column prefixes. Raises
    ``ValueError`` if the name has no ``_<fault>`` suffix.
    """
    m = _LABEL_RE.match(folder_name)
    if m is None:
        raise ValueError(f"not a <service>_<fault> folder name: {folder_name!r}")
    return m.group("service"), m.group("fault")


def _is_case_dir(path: Path) -> bool:
    return path.is_dir() and any(
        (path / name).is_file()
        for name in (METRICS_FILE, RAW_METRICS_FILE, INJECT_TIME_FILE)
    )


def find_cases(root: str | PathLike[str]) -> Iterator[Case]:
    """Yield one ``Case`` per case folder under ``root``, sorted by path.

    A case folder is ``<service>_<fault>/<n>/`` holding any of the RCAEval
    metrics or inject-time files. Folders without ``simple_metrics.csv``
    (raw-only cases, such as RE2-OB ``checkoutservice_cpu/4``) are skipped with
    a message on stderr. The system name comes from the folder above the label
    folder (``RE2-OB`` -> ``re2ob``).
    """
    root_path = Path(root)
    candidates = sorted(
        {
            p.parent
            for name in (METRICS_FILE, RAW_METRICS_FILE, INJECT_TIME_FILE)
            for p in root_path.rglob(name)
        }
    )
    for case_dir in candidates:
        if not _is_case_dir(case_dir) or not case_dir.name.isdigit():
            continue
        try:
            service, fault = parse_label(case_dir.parent.name)
        except ValueError:
            continue
        if not (case_dir / METRICS_FILE).is_file():
            print(f"skip {case_dir}: no {METRICS_FILE}", file=sys.stderr)
            continue
        system = case_dir.parent.parent.name.lower().replace("-", "")
        repetition = int(case_dir.name)
        yield Case(
            system=system,
            service=service,
            fault=fault,
            repetition=repetition,
            case_id=f"{system}/{service}_{fault}/{repetition}",
            path=case_dir,
        )


def read_metrics(source: str | PathLike[str] | IO[str] | IO[bytes]) -> pd.DataFrame:
    """Read a metrics CSV from a path or an open file-like object, unchanged.

    The API passes an uploaded file; training passes a path. Use
    ``clean_metrics`` on the result. Raises ``ValueError`` for an empty or
    unparseable file.
    """
    try:
        return pd.read_csv(source)
    except pd.errors.EmptyDataError as e:
        raise ValueError("metrics file is empty") from e
    except pd.errors.ParserError as e:
        raise ValueError(f"metrics file is not a valid CSV: {e}") from e


def clean_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the spec 3.3 quirks and return ``time`` plus metric columns.

    - drops repeated-header columns (``time.1``, ``time.2``, ...)
    - drops ``latency-50`` columns (p90 latency only)
    - raises ``ValueError`` for a raw Prometheus export (more than 20
      ``container-`` or ``istio-`` columns), a missing ``time`` column, or a
      ``time`` value that is empty or not a number
    - converts metric values to numbers (unparseable cells become NaN), then
      ``inf`` -> NaN, sorts by ``time``, forward-fills, and fills what's left
      with 0

    ``time`` is in Unix seconds. The input frame is not modified.
    """
    columns = [str(c) for c in frame.columns]
    n_raw = sum("container-" in c or "istio-" in c for c in columns)
    if n_raw > _RAW_COLUMN_LIMIT:
        raise ValueError(
            f"raw Prometheus metrics ({n_raw} container-/istio- columns); "
            f"expected {METRICS_FILE}"
        )
    if "time" not in columns:
        raise ValueError("metrics file has no 'time' column")
    keep = [
        c for c in columns if not c.startswith("time.") and not _is_ignored_metric(c)
    ]
    named = frame.set_axis(columns, axis="columns")
    out = pd.DataFrame(
        {c: pd.to_numeric(named[c], errors="coerce") for c in keep},
        index=named.index,
    )
    out = out.replace([np.inf, -np.inf], np.nan)
    if bool(out["time"].isna().to_numpy().any()):
        raise ValueError("'time' column has missing or non-numeric values")
    out = out.sort_values(by="time", kind="mergesort").ffill().fillna(0)
    return out.reset_index(drop=True)


def _is_ignored_metric(column: str) -> bool:
    return "_" in column and column.split("_", 1)[1].lower().startswith("latency-50")


def read_inject_time(case_dir: str | PathLike[str]) -> int:
    """Return the fault injection time (Unix seconds) from ``inject_time.txt``.

    Raises ``OSError`` if the file is missing and ``ValueError`` if it doesn't
    hold an integer.
    """
    with open(Path(case_dir) / INJECT_TIME_FILE, encoding="utf-8") as fh:
        return int(fh.read().strip())


def load_case(path: str | PathLike[str]) -> LoadedCase:
    """Load one case folder (``.../<service>_<fault>/<n>/``).

    Returns the cleaned ``simple_metrics.csv`` frame, the injection time in Unix
    seconds, and the label from the folder name. Raises ``OSError`` or
    ``ValueError`` if the case can't be read; callers skip such cases.
    """
    case_dir = Path(path)
    service, fault = parse_label(case_dir.parent.name)
    metrics = clean_metrics(read_metrics(case_dir / METRICS_FILE))
    return LoadedCase(
        metrics=metrics,
        inject_time=read_inject_time(case_dir),
        service=service,
        fault=fault,
    )


def kind_of(column: str) -> tuple[str, str] | None:
    """Map a metric column to (service, kind), or ``None`` if it isn't used.

    ``'cartservice_latency-90'`` -> ``('cartservice', 'latency')``. The service
    is the text before the first ``_``, with a ``-db`` suffix removed so a
    service and its database count as one service (``carts-db`` -> ``carts``).
    ``ts-`` and ``front-end`` names are kept as they are. ``latency-50``,
    ``time`` and unknown kinds return ``None``.
    """
    if "_" not in column:
        return None
    service, metric = column.split("_", 1)
    metric = metric.lower()
    if metric.startswith("latency-50"):
        return None
    for kind in KINDS:
        if (
            metric.startswith(kind)
            or (kind == "diskio" and "disk" in metric)
            or (kind == "error" and "err" in metric)
        ):
            return service.replace("-db", ""), kind
    return None
