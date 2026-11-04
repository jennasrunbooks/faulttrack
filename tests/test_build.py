"""Tests for faulttrack.build (``make features``) on synthetic case folders.

Each case is written to tmp_path in the RCAEval layout; no data/raw/, no network.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from faulttrack.build import (
    INVENTORY_COLUMNS,
    META_COLUMNS,
    SERVICE_COUNT_COLUMNS,
    STATUS_OK,
    build_tables,
    main,
)
from faulttrack.features import DELAYS_S, FEATURE_NAMES

T0 = 1_700_000_000
INJECT = T0 + 720  # 1,441 rows at 1 s with the injection at the midpoint, like RE2


def _write_case(
    root: Path, suite: str, label: str, rep: int, *, simple: bool = True
) -> None:
    case_dir = root / suite / suite / label / str(rep)
    case_dir.mkdir(parents=True)
    (case_dir / "inject_time.txt").write_text(f"{INJECT}\n")
    if not simple:
        (case_dir / "metrics.csv").write_text("time\n1\n")
        return
    rng = np.random.default_rng(rep)
    n = 1441
    time = np.arange(T0, T0 + n)
    service = label.rsplit("_", 1)[0]
    data: dict[str, np.ndarray] = {"time": time}
    for svc in (service, "other"):
        for kind in ("cpu", "mem", "socket"):
            data[f"{svc}_{kind}"] = 50.0 + rng.normal(0.0, 1.0, n)
    data[f"{service}_cpu"][time >= INJECT] += 30.0
    data["other_mem"][3] = np.inf
    pd.DataFrame(data).to_csv(case_dir / "simple_metrics.csv", index=False)


def _raw_root(tmp_path: Path) -> Path:
    root = tmp_path / "raw"
    _write_case(root, "RE2-SS", "carts_cpu", 2)
    _write_case(root, "RE2-OB", "cartservice_cpu", 1)
    _write_case(root, "RE2-OB", "checkoutservice_cpu", 4, simple=False)
    return root


def test_feature_table_has_id_columns_then_features_sorted(tmp_path: Path) -> None:
    features, _ = build_tables(_raw_root(tmp_path))
    assert list(features.columns) == META_COLUMNS + FEATURE_NAMES
    assert set(SERVICE_COUNT_COLUMNS).isdisjoint(FEATURE_NAMES)
    # 2 services ("<injected>", "other") with cpu/mem/socket; none with diskio
    row = features.iloc[0]
    assert row["n_services_cpu"] == 2 and row["n_services_diskio"] == 0
    assert len(features) == 2 * len(DELAYS_S)
    assert list(features["case_id"].unique()) == [
        "re2ob/cartservice_cpu/1",
        "re2ss/carts_cpu/2",
    ]
    assert list(features["delay_s"].iloc[:5]) == list(DELAYS_S)
    assert set(features["suspect_service"]) == {"cartservice", "carts"}


def test_inventory_lists_skipped_cases_with_their_reason(tmp_path: Path) -> None:
    _, inventory = build_tables(_raw_root(tmp_path))
    assert list(inventory.columns) == INVENTORY_COLUMNS
    assert len(inventory) == 3
    by_case = inventory.set_index("case_id")
    skipped = by_case.loc["re2ob/checkoutservice_cpu/4"]
    assert skipped["status"] == "no simple_metrics.csv"
    assert pd.isna(skipped["n_rows"])
    ok = by_case.loc["re2ob/cartservice_cpu/1"]
    assert ok["status"] == STATUS_OK
    assert ok["n_rows"] == 1441
    assert ok["n_untimed_rows"] == 0
    assert ok["n_metric_cols"] == 6
    assert ok["n_cols_cpu"] == 2
    assert ok["inf_count"] == 1
    assert ok["inject_pos"] == 0.5
    assert ok["n_feature_rows"] == len(DELAYS_S)


def test_rebuild_is_identical_and_check_mode_passes(tmp_path: Path) -> None:
    root = _raw_root(tmp_path)
    out = tmp_path / "features"
    assert main(["--raw", str(root), "--out", str(out)]) == 0
    first = pd.read_parquet(out / "re2_features.parquet")
    features, inventory = build_tables(root)
    pd.testing.assert_frame_equal(features, first)
    pd.testing.assert_frame_equal(
        inventory, pd.read_parquet(out / "re2_inventory.parquet")
    )
    assert main(["--raw", str(root), "--out", str(out), "--check"]) == 0
