"""Unit tests for faulttrack.data: case discovery, loading and the spec 3.3 quirks.

All inputs are small synthetic files written to tmp_path; no data/raw/, no network.
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from faulttrack.data import (
    Case,
    clean_metrics,
    find_cases,
    kind_of,
    load_case,
    parse_label,
    read_metrics,
)

T0 = 1_700_000_000


def _write_case(
    root: Path,
    suite: str,
    label: str,
    rep: int,
    *,
    simple: bool = True,
    raw: bool = False,
    inject: int = T0 + 5,
) -> Path:
    case_dir = root / suite / suite / label / str(rep)
    case_dir.mkdir(parents=True)
    if simple:
        frame = pd.DataFrame(
            {"time": range(T0, T0 + 10), f"{label.rsplit('_', 1)[0]}_cpu": 1.0}
        )
        frame.to_csv(case_dir / "simple_metrics.csv", index=False)
    if raw:
        (case_dir / "metrics.csv").write_text("time\n1\n")
    (case_dir / "inject_time.txt").write_text(f"{inject}\n")
    return case_dir


# ---------- find_cases ----------


def test_find_cases_yields_one_record_per_case(tmp_path: Path) -> None:
    _write_case(tmp_path, "RE2-OB", "cartservice_cpu", 1, raw=True)
    _write_case(tmp_path, "RE2-OB", "cartservice_cpu", 2)
    cases = list(find_cases(tmp_path))
    assert [c.case_id for c in cases] == [
        "re2ob/cartservice_cpu/1",
        "re2ob/cartservice_cpu/2",
    ]
    first = cases[0]
    assert first == Case(
        system="re2ob",
        service="cartservice",
        fault="cpu",
        repetition=1,
        case_id="re2ob/cartservice_cpu/1",
        path=tmp_path / "RE2-OB" / "RE2-OB" / "cartservice_cpu" / "1",
    )


def test_find_cases_skips_raw_only_case_with_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # spec 3.3: RE2-OB checkoutservice_cpu/4 has metrics.csv but no simple_metrics
    _write_case(tmp_path, "RE2-OB", "checkoutservice_cpu", 4, simple=False, raw=True)
    _write_case(tmp_path, "RE2-OB", "checkoutservice_cpu", 5)
    cases = list(find_cases(tmp_path))
    assert [c.repetition for c in cases] == [5]
    err = capsys.readouterr().err
    assert "checkoutservice_cpu/4" in err
    assert "simple_metrics.csv" in err


def test_find_cases_keeps_ts_and_front_end_names(tmp_path: Path) -> None:
    _write_case(tmp_path, "RE2-TT", "ts-order-service_delay", 1)
    _write_case(tmp_path, "RE2-SS", "front-end_loss", 3)
    by_system = {c.system: c for c in find_cases(tmp_path)}
    assert by_system["re2tt"].service == "ts-order-service"
    assert by_system["re2tt"].fault == "delay"
    assert by_system["re2ss"].service == "front-end"
    assert by_system["re2ss"].case_id == "re2ss/front-end_loss/3"


def test_parse_label_rejects_name_without_fault() -> None:
    with pytest.raises(ValueError):
        parse_label("cartservice")


# ---------- read_metrics / clean_metrics ----------


def _messy_csv() -> str:
    # unsorted time, a repeated time.1 header, inf and NaN gaps, latency-50
    return (
        "time,time.1,cart_cpu,cart_latency-50,cart_latency-90,ts-order-service_mem\n"
        f"{T0 + 2},{T0 + 2},inf,1,7,\n"
        f"{T0 + 0},{T0 + 0},,1,5,3\n"
        f"{T0 + 1},{T0 + 1},2,1,-inf,4\n"
        f"{T0 + 3},{T0 + 3},3,1,8,\n"
    )


def test_clean_metrics_applies_quirks() -> None:
    out = clean_metrics(read_metrics(io.StringIO(_messy_csv())))
    # time.1 and latency-50 dropped
    assert list(out.columns) == [
        "time",
        "cart_cpu",
        "cart_latency-90",
        "ts-order-service_mem",
    ]
    # sorted by time
    assert out["time"].tolist() == [T0, T0 + 1, T0 + 2, T0 + 3]
    # leading NaN -> 0, inf -> NaN -> forward-filled
    assert out["cart_cpu"].tolist() == [0.0, 2.0, 2.0, 3.0]
    assert out["cart_latency-90"].tolist() == [5.0, 5.0, 7.0, 8.0]
    assert out["ts-order-service_mem"].tolist() == [3.0, 4.0, 4.0, 4.0]
    assert not np.isinf(out.to_numpy()).any()
    assert not out.isna().any().any()


def test_clean_metrics_same_from_path_and_file_like(tmp_path: Path) -> None:
    path = tmp_path / "simple_metrics.csv"
    path.write_text(_messy_csv())
    from_path = clean_metrics(read_metrics(path))
    from_text = clean_metrics(read_metrics(io.StringIO(_messy_csv())))
    from_bytes = clean_metrics(read_metrics(io.BytesIO(_messy_csv().encode())))
    pd.testing.assert_frame_equal(from_path, from_text)
    pd.testing.assert_frame_equal(from_path, from_bytes)


def test_clean_metrics_rejects_raw_prometheus_export() -> None:
    cols = {"time": [T0, T0 + 1]}
    cols |= {f"svc{i}_container-cpu-usage": [0.0, 1.0] for i in range(21)}
    with pytest.raises(ValueError, match="raw Prometheus"):
        clean_metrics(pd.DataFrame(cols))


def test_clean_metrics_allows_few_container_columns() -> None:
    cols = {"time": [T0, T0 + 1]}
    cols |= {f"svc{i}_container-cpu-usage": [0.0, 1.0] for i in range(20)}
    assert clean_metrics(pd.DataFrame(cols)).shape == (2, 21)


def test_clean_metrics_requires_time_column() -> None:
    with pytest.raises(ValueError, match="time"):
        clean_metrics(pd.DataFrame({"cart_cpu": [1.0]}))


def test_clean_metrics_does_not_modify_input() -> None:
    raw = read_metrics(io.StringIO(_messy_csv()))
    before = raw.copy()
    clean_metrics(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_read_metrics_empty_file_raises_value_error() -> None:
    with pytest.raises(ValueError, match="empty"):
        read_metrics(io.StringIO(""))


# ---------- load_case ----------


def test_load_case_returns_metrics_inject_time_and_label(tmp_path: Path) -> None:
    case_dir = _write_case(tmp_path, "RE2-TT", "ts-auth-service_mem", 2, inject=T0 + 7)
    loaded = load_case(case_dir)
    assert loaded.inject_time == T0 + 7
    assert (loaded.service, loaded.fault) == ("ts-auth-service", "mem")
    assert list(loaded.metrics.columns) == ["time", "ts-auth-service_cpu"]
    assert len(loaded.metrics) == 10


def test_load_case_without_inject_time_raises_os_error(tmp_path: Path) -> None:
    case_dir = _write_case(tmp_path, "RE2-OB", "cartservice_cpu", 1)
    (case_dir / "inject_time.txt").unlink()
    with pytest.raises(OSError):
        load_case(case_dir)


# ---------- kind_of ----------


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        ("cartservice_cpu", ("cartservice", "cpu")),
        ("cartservice_mem", ("cartservice", "mem")),
        ("cartservice_diskio", ("cartservice", "diskio")),
        ("cartservice_socket", ("cartservice", "socket")),
        ("cartservice_latency-90", ("cartservice", "latency")),
        ("cartservice_error", ("cartservice", "error")),
        ("cartservice_workload", ("cartservice", "workload")),
        ("ts-order-service_cpu", ("ts-order-service", "cpu")),
        ("front-end_latency-90", ("front-end", "latency")),
        ("carts-db_mem", ("carts", "mem")),
        ("cartservice_latency-50", None),
        ("cartservice_unknown", None),
        ("time", None),
    ],
)
def test_kind_of(column: str, expected: tuple[str, str] | None) -> None:
    assert kind_of(column) == expected
