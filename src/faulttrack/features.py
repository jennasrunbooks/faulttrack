"""Service-agnostic shift features (spec Section 4). The only feature code.

``featurize`` is a pure function of a cleaned metrics frame (from
``faulttrack.data.clean_metrics``), a detection time and the window settings.
It does no file I/O. ``make features``, ``train.py``, the notebook and
``predict.py`` all call it, so training and serving compute the same row.

Windows, both anchored at the detection time ``t_det`` (Unix seconds):

- baseline: ``[t_det - baseline_s, t_det)``
- post:     ``[t_det, t_det + post_s)``

Per column, the shift score is the post median minus the baseline median, over
a scale ``max(1.4826 * MAD, z_floor * |baseline median|)`` (the baseline std
when the MAD is 0), then log-compressed as ``sign(z) * log1p(|z|)``. Columns
are then grouped by metric kind, never by service name, so one feature vector
fits systems of any size.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from faulttrack.data import KINDS, kind_of

BASELINE_S = 300
"""Default baseline window length, seconds."""
POST_S = 120
"""Default post-detection window length, seconds."""
Z_FLOOR = 0.01
"""Default scale floor, as a fraction of |baseline median|."""
LOG_Z = True
"""Default: log-compress shift scores."""
DELAYS_S: tuple[int, ...] = (0, 30, 60, 90, 120)
"""Detection delays after injection used to build training rows, seconds."""

MIN_BASELINE_ROWS = 10
"""Fewest baseline samples ``featurize`` accepts."""
MIN_POST_ROWS = 5
"""Fewest post-window samples ``featurize`` accepts."""

_RAW_Z_CLIP = 1000.0
_SHIFT_THRESHOLD = 3.0  # "clearly shifted", in z units before log compression
_MAD_TO_SIGMA = 1.4826
_MAD_EPS = 1e-9


def feature_names(norm_counts: bool = False) -> list[str]:
    """Return the ordered feature names.

    With ``norm_counts``, the per-kind sum and count become a per-service mean
    (``<kind>_absmean``) and fraction (``<kind>_frac_gt3``), which don't grow
    with the number of services.
    """
    agg = ("absmean", "frac_gt3") if norm_counts else ("abssum", "n_gt3")
    names: list[str] = []
    for kind in KINDS:
        names += [f"{kind}_max", f"{kind}_2nd", f"{kind}_{agg[0]}", f"{kind}_{agg[1]}"]
    for kind in KINDS:
        names += [f"top_{kind}", f"top_{kind}_share"]
    return names


FEATURE_NAMES: list[str] = feature_names()
"""Ordered feature names for the default settings (the model's inputs)."""


@dataclass(frozen=True)
class Featurized:
    """One feature row plus the rule-based suspect service.

    ``features`` maps every name in ``feature_names(norm_counts)`` to its value,
    in that order. ``suspect_service`` is the service with the largest total
    |shift| across kinds. It comes from a rule, not the model. It is ``None``
    when no column maps to a known kind.
    """

    features: dict[str, float]
    suspect_service: str | None


def shift_scores(
    baseline: pd.DataFrame,
    post: pd.DataFrame,
    z_floor: float = Z_FLOOR,
    log_z: bool = LOG_Z,
) -> pd.Series:
    """Per-column shift of the post median from the baseline median.

    ``z = (median(post) - median(baseline)) / scale``, where ``scale`` is
    ``1.4826 * MAD(baseline)``, or the baseline std when the MAD is about 0,
    raised to at least ``z_floor * |median(baseline)|`` when ``z_floor > 0``.
    A column whose scale is 0 or undefined gets ``z = 0``.

    Returns ``sign(z) * log1p(|z|)`` when ``log_z`` (log-compressed z), else
    ``z`` clipped to +/-1000. Both frames have the same metric columns and no
    ``time`` column.
    """
    med = baseline.median()
    mad = (baseline - med).abs().median() * _MAD_TO_SIGMA
    scale = mad.where(mad > _MAD_EPS, baseline.std())
    if z_floor > 0:
        floor = z_floor * med.abs()
        scale = pd.Series(np.maximum(scale, floor), index=scale.index)
    scale = scale.replace(0, np.nan)
    z = ((post.median() - med) / scale).replace([np.inf, -np.inf], np.nan).fillna(0)
    if log_z:
        return pd.Series(np.sign(z) * np.log1p(z.abs()), index=z.index)
    return z.clip(-_RAW_Z_CLIP, _RAW_Z_CLIP)


def featurize(
    frame: pd.DataFrame,
    t_det: float,
    baseline_s: float = BASELINE_S,
    post_s: float = POST_S,
    z_floor: float = Z_FLOOR,
    log_z: bool = LOG_Z,
    norm_counts: bool = False,
) -> Featurized:
    """Compute one feature row for a detection time.

    ``frame`` is a cleaned metrics frame (``time`` in Unix seconds plus
    ``<service>_<metric>`` columns). ``t_det``, ``baseline_s`` and ``post_s``
    are in seconds. Shift scores are log-compressed z when ``log_z``.

    Per kind: ``<kind>_max`` is the shift with the largest magnitude (signed),
    ``<kind>_2nd`` the next one, ``<kind>_abssum`` the sum of |shift| and
    ``<kind>_n_gt3`` how many services shifted by more than 3 (on the same
    scale as the shift). Then, for the suspect service (largest total |shift|):
    ``top_<kind>`` is its shift and ``top_<kind>_share`` that shift's share of
    its total. A kind with no columns gets 0 everywhere.

    Raises ``ValueError`` if the baseline has fewer than 10 rows or the post
    window fewer than 5 (for example, ``t_det`` falls outside the file).
    """
    times = frame["time"]
    baseline = frame.loc[(times >= t_det - baseline_s) & (times < t_det)]
    post = frame.loc[(times >= t_det) & (times < t_det + post_s)]
    if len(baseline) < MIN_BASELINE_ROWS or len(post) < MIN_POST_ROWS:
        raise ValueError(
            f"not enough data around t_det={t_det}: {len(baseline)} baseline rows "
            f"(need {MIN_BASELINE_ROWS}), {len(post)} post rows (need {MIN_POST_ROWS})"
        )
    z = shift_scores(
        baseline.drop(columns="time"),
        post.drop(columns="time"),
        z_floor=z_floor,
        log_z=log_z,
    )

    by_kind: defaultdict[str, dict[str, float]] = defaultdict(dict)
    by_service: defaultdict[str, dict[str, float]] = defaultdict(dict)
    for column, value in z.items():
        mapped = kind_of(str(column))
        if mapped is None:
            continue
        service, kind = mapped
        by_kind[kind][service] = float(value)
        by_service[service][kind] = float(value)

    threshold = float(np.log1p(_SHIFT_THRESHOLD)) if log_z else _SHIFT_THRESHOLD
    features: dict[str, float] = {}
    for kind in KINDS:
        values = sorted(by_kind[kind].values(), key=abs, reverse=True) or [0.0]
        abs_values = np.abs(np.asarray(values, dtype=float))
        abs_sum = float(abs_values.sum())  # NumPy's pairwise sum, as in the spike
        n_shifted = int((abs_values > threshold).sum())
        features[f"{kind}_max"] = values[0]
        features[f"{kind}_2nd"] = values[1] if len(values) > 1 else 0.0
        if norm_counts:
            n_services = max(len(by_kind[kind]), 1)
            features[f"{kind}_absmean"] = abs_sum / n_services
            features[f"{kind}_frac_gt3"] = n_shifted / n_services
        else:
            features[f"{kind}_abssum"] = abs_sum
            features[f"{kind}_n_gt3"] = n_shifted

    suspect = suspect_service(by_service)
    profile = by_service.get(suspect, {}) if suspect is not None else {}
    total = sum(abs(v) for v in profile.values()) or 1.0
    for kind in KINDS:
        value = profile.get(kind, 0.0)
        features[f"top_{kind}"] = value
        features[f"top_{kind}_share"] = abs(value) / total
    return Featurized(features=features, suspect_service=suspect)


def suspect_service(by_service: dict[str, dict[str, float]]) -> str | None:
    """Return the service with the largest total |shift| across kinds.

    This is the root-cause rule (not a model). Ties go to the service seen
    first. Returns ``None`` for an empty mapping.
    """
    if not by_service:
        return None
    return max(by_service, key=lambda s: sum(abs(v) for v in by_service[s].values()))
