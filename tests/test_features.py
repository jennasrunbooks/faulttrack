"""Unit tests for faulttrack.features on small synthetic frames (no data/raw/)."""

import numpy as np
import pandas as pd
import pytest

from faulttrack.data import KINDS
from faulttrack.features import FEATURE_NAMES, feature_names, featurize, shift_scores

T0 = 1_700_000_000
T_DET = T0 + 600  # 600 s of history, then 600 s after detection
SERVICES = ("cartservice", "ts-order-service", "front-end")


def _frame(
    seed: int = 0, kinds: tuple[str, ...] = ("cpu", "mem", "socket")
) -> pd.DataFrame:
    """Noisy, stationary metrics at 1 s, from T0 to T0 + 1200 s."""
    rng = np.random.default_rng(seed)
    n = 1200
    data: dict[str, np.ndarray] = {"time": np.arange(T0, T0 + n)}
    for svc in SERVICES:
        for kind in kinds:
            data[f"{svc}_{kind}"] = 50.0 + rng.normal(0.0, 1.0, n)
    return pd.DataFrame(data)


def _step(frame: pd.DataFrame, column: str, size: float) -> pd.DataFrame:
    out = frame.copy()
    out.loc[out["time"] >= T_DET, column] += size
    return out


# ---------- names and shape ----------


def test_output_columns_equal_feature_names_in_order() -> None:
    result = featurize(_frame(), T_DET)
    assert list(result.features) == FEATURE_NAMES
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)) == 6 * len(KINDS)


def test_norm_counts_uses_its_own_names_in_order() -> None:
    result = featurize(_frame(), T_DET, norm_counts=True)
    assert list(result.features) == feature_names(norm_counts=True)
    assert "cpu_absmean" in result.features
    assert "cpu_frac_gt3" in result.features


def test_all_features_are_finite() -> None:
    result = featurize(_step(_frame(), "cartservice_cpu", 30.0), T_DET)
    assert all(np.isfinite(v) for v in result.features.values())


# ---------- behavior ----------


def test_step_in_one_cpu_column_gives_largest_cpu_max() -> None:
    result = featurize(_step(_frame(), "cartservice_cpu", 30.0), T_DET)
    f = result.features
    assert f["cpu_max"] > np.log1p(3)  # clearly shifted
    assert f["cpu_max"] > abs(f["cpu_2nd"])
    assert f["cpu_max"] > max(abs(f["mem_max"]), abs(f["socket_max"]))
    assert f["cpu_n_gt3"] == 1
    assert f["top_cpu"] == f["cpu_max"]


def test_suspect_service_is_the_one_with_the_step() -> None:
    stepped = _step(_frame(), "ts-order-service_mem", 25.0)
    assert featurize(stepped, T_DET).suspect_service == "ts-order-service"


def test_negative_step_keeps_its_sign() -> None:
    result = featurize(_step(_frame(), "front-end_socket", -30.0), T_DET)
    assert result.features["socket_max"] < -np.log1p(3)
    assert result.suspect_service == "front-end"


def test_missing_kind_gives_zeros() -> None:
    result = featurize(_frame(kinds=("cpu", "mem")), T_DET)
    for kind in ("diskio", "socket", "latency", "error", "workload"):
        for name in (f"{kind}_max", f"{kind}_2nd", f"{kind}_abssum", f"{kind}_n_gt3"):
            assert result.features[name] == 0
        assert result.features[f"top_{kind}"] == 0
        assert result.features[f"top_{kind}_share"] == 0


def test_top_shares_sum_to_one() -> None:
    result = featurize(_step(_frame(), "cartservice_cpu", 30.0), T_DET)
    shares = sum(result.features[f"top_{k}_share"] for k in KINDS)
    assert shares == pytest.approx(1.0)


def test_no_known_kind_gives_zero_row_and_no_suspect() -> None:
    frame = pd.DataFrame({"time": np.arange(T0, T0 + 1200), "cartservice_unknown": 1.0})
    result = featurize(frame, T_DET)
    assert result.suspect_service is None
    assert all(v == 0 for v in result.features.values())


def test_windows_are_anchored_at_detection_time() -> None:
    # The baseline is [t_det - 300, t_det), not the time before injection. A
    # step 400 s before t_det is already in the whole baseline, so the post
    # window doesn't shift against it.
    frame = _frame()
    early = frame.copy()
    early.loc[early["time"] >= T_DET - 400, "cartservice_cpu"] += 30.0
    assert abs(featurize(early, T_DET).features["cpu_max"]) < np.log1p(3)
    # The same step at t_det is a clear shift.
    assert featurize(_step(frame, "cartservice_cpu", 30.0), T_DET).features[
        "cpu_max"
    ] > np.log1p(3)


def test_window_lengths_select_expected_rows() -> None:
    frame = _frame()
    # Mark only the rows in [t_det, t_det + 120): a 120 s post window sees the
    # full step; one starting later wouldn't.
    marked = frame.copy()
    in_post = (marked["time"] >= T_DET) & (marked["time"] < T_DET + 120)
    marked.loc[in_post, "cartservice_cpu"] += 30.0
    assert featurize(marked, T_DET).features["cpu_max"] > np.log1p(3)
    assert abs(featurize(marked, T_DET + 120).features["cpu_max"]) < np.log1p(3)


def test_too_little_data_raises_value_error() -> None:
    frame = _frame()
    with pytest.raises(ValueError, match="not enough data"):
        featurize(frame, T0 + 5)  # fewer than 10 baseline rows
    with pytest.raises(ValueError, match="not enough data"):
        featurize(frame, T0 + 1198)  # fewer than 5 post rows


# ---------- shift_scores ----------


def _only(scores: pd.Series) -> float:
    """The single value of a one-column shift_scores result."""
    return float(scores.to_numpy()[0])


def test_flat_metric_does_not_blow_up() -> None:
    # A perfectly flat baseline has MAD 0 and std 0; the floor (1% of the
    # median) sets the scale, so a 1% rise is z = 1, not infinity.
    baseline = pd.DataFrame({"m": np.full(300, 1000.0)})
    post = pd.DataFrame({"m": np.full(120, 1010.0)})
    z = shift_scores(baseline, post)
    assert _only(z) == pytest.approx(np.log1p(1.0))


def test_flat_zero_metric_gives_zero() -> None:
    baseline = pd.DataFrame({"m": np.zeros(300)})
    post = pd.DataFrame({"m": np.full(120, 5.0)})
    assert _only(shift_scores(baseline, post)) == 0.0


def test_std_fallback_when_mad_is_zero() -> None:
    # Over half the baseline is one value, so MAD = 0; the std is used instead.
    base = np.full(300, 10.0)
    base[:100] = 20.0
    baseline = pd.DataFrame({"m": base})
    post = pd.DataFrame({"m": np.full(120, 10.0 + base.std(ddof=1))})
    z = shift_scores(baseline, post, z_floor=0.0, log_z=False)
    assert _only(z) == pytest.approx(1.0)


def test_mad_scale_and_log_compression() -> None:
    rng = np.random.default_rng(1)
    base = rng.normal(100.0, 2.0, 300)
    baseline = pd.DataFrame({"m": base})
    post = pd.DataFrame({"m": np.full(120, 160.0)})
    med = np.median(base)
    mad = np.median(np.abs(base - med)) * 1.4826
    raw = (160.0 - med) / mad
    assert _only(shift_scores(baseline, post, log_z=False)) == pytest.approx(raw)
    assert _only(shift_scores(baseline, post)) == pytest.approx(np.log1p(raw))


def test_raw_z_is_clipped() -> None:
    baseline = pd.DataFrame({"m": np.r_[np.full(299, 1.0), 1.001]})
    post = pd.DataFrame({"m": np.full(120, 1e9)})
    assert _only(shift_scores(baseline, post, z_floor=0.0, log_z=False)) == 1000.0
