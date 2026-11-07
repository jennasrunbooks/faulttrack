"""API tests for predict.py on synthetic data (no data/raw/, no network).

The model is a ``DummyClassifier`` fitted on rows from ``featurize`` for a few
synthetic detection times, so nothing here depends on the real model or on the
committed feature table. Tests on the committed fixture case are a separate
follow-up (see the PR description).
"""

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier

import predict
from faulttrack.build import FEATURES_FILE
from faulttrack.data import clean_metrics, read_metrics
from faulttrack.features import (
    BASELINE_S,
    FEATURE_NAMES,
    LOG_Z,
    POST_S,
    Z_FLOOR,
    featurize,
)

CLASSES = ["cpu", "delay", "disk", "loss", "mem", "socket"]  # sorted, as sklearn does
T0 = 1_700_000_000
T_DET = T0 + 600
SERVICES = ("cartservice", "ts-order-service", "front-end")
CV_F1 = 0.8123
HIT_RATE = 0.7654


def _frame(shifted: str = "cartservice_cpu", step: float = 40.0) -> pd.DataFrame:
    """Noisy metrics at 1 s from T0 for 1200 s, with a step in one column."""
    rng = np.random.default_rng(0)
    n = 1200
    data: dict[str, Any] = {"time": np.arange(T0, T0 + n)}
    for svc in SERVICES:
        for kind in ("cpu", "mem", "socket"):
            data[f"{svc}_{kind}"] = 50.0 + rng.normal(0.0, 1.0, n)
    out = pd.DataFrame(data)
    out.loc[out["time"] >= T_DET, shifted] += step
    return out


@pytest.fixture(scope="module")
def csv_bytes() -> bytes:
    return _frame().to_csv(index=False).encode()


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> TestClient:
    tmp = tmp_path_factory.mktemp("artifacts")
    rows = [
        list(featurize(_frame(), T_DET + shift).features.values())
        for shift in (-60, -30, 0, 30, 60, 90)
    ]
    model = DummyClassifier(strategy="prior").fit(np.array(rows), CLASSES)
    (tmp / "model.bin").write_bytes(pickle.dumps(model))
    (tmp / "feature_spec.json").write_text(
        json.dumps(
            {
                "feature_names": FEATURE_NAMES,
                "classes": CLASSES,
                "baseline_s": BASELINE_S,
                "post_s": POST_S,
                "z_floor": Z_FLOOR,
                "log_z": LOG_Z,
                "model_version": "test-1",
            }
        )
    )
    results = tmp / "results.json"
    results.write_text(
        json.dumps({"fault_cv_macro_f1_mean": CV_F1, "suspect_hit_rate": HIT_RATE})
    )
    (tmp / "index.html").write_text("<!doctype html><title>test page</title>")
    return TestClient(predict.create_app(tmp, results))


def _post_case(client: TestClient, data: bytes, detection_time: float = T_DET) -> Any:
    return client.post(
        "/predict/case",
        files={"file": ("simple_metrics.csv", data, "text/csv")},
        data={"detection_time": str(detection_time)},
    )


def _vector() -> dict[str, float]:
    return dict(featurize(_frame(), T_DET).features)


# ---------- healthz and docs ----------


def test_healthz(client: TestClient) -> None:
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model_version": "test-1"}


def test_docs_say_suspect_service_is_a_rule(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    desc = schemas["CaseResponse"]["properties"]["suspect_service"]["description"]
    assert "RULE, not from the model" in desc
    assert "rule" in schemas["SuspectBlock"]["properties"]["source"]["description"]


# ---------- /predict ----------


def test_predict_valid_vector(client: TestClient) -> None:
    resp = client.post("/predict", json={"features": _vector()})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"fault"}  # no suspect service for a bare vector
    fault = body["fault"]
    assert fault["source"] == "model"
    assert fault["model_version"] == "test-1"
    assert fault["cv_macro_f1"] == round(CV_F1, 3)
    assert set(fault["probabilities"]) == set(CLASSES)
    assert sum(fault["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert fault["label"] in CLASSES
    assert len(fault["top_signals"]) == 3
    assert set(fault["top_signals"]) <= set(FEATURE_NAMES)


def test_predict_missing_feature_is_422(client: TestClient) -> None:
    vector = _vector()
    del vector["cpu_max"]
    resp = client.post("/predict", json={"features": vector})
    assert resp.status_code == 422
    assert resp.json()["detail"]["missing_features"] == ["cpu_max"]


def test_predict_unknown_feature_is_422(client: TestClient) -> None:
    resp = client.post("/predict", json={"features": {**_vector(), "bogus": 1.0}})
    assert resp.status_code == 422
    assert resp.json()["detail"]["unknown_features"] == ["bogus"]


def test_predict_non_numeric_feature_is_422(client: TestClient) -> None:
    resp = client.post("/predict", json={"features": {**_vector(), "cpu_max": "x"}})
    assert resp.status_code == 422


# ---------- /predict/case ----------


def test_predict_case_response_shape_and_labels(
    client: TestClient, csv_bytes: bytes
) -> None:
    resp = _post_case(client, csv_bytes)
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"fault", "suspect_service"}
    fault, suspect = body["fault"], body["suspect_service"]
    assert fault["source"] == "model"
    assert fault["cv_macro_f1"] == round(CV_F1, 3)
    assert set(fault["probabilities"]) == set(CLASSES)
    assert suspect["source"] == "rule"
    assert suspect["hit_rate"] == round(HIT_RATE, 3)
    assert "rule" not in suspect["name"]
    assert suspect["name"] == "cartservice"  # the column with the step
    assert isinstance(suspect["rule"], str) and suspect["rule"]


def test_predict_case_features_equal_library_row(
    client: TestClient,
    csv_bytes: bytes,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The row built inside /predict/case equals featurize() on the same file."""
    path = tmp_path / "simple_metrics.csv"
    path.write_bytes(csv_bytes)
    expected = featurize(clean_metrics(read_metrics(path)), T_DET).features

    seen: list[dict[str, float]] = []
    real = predict._fault_block

    def spy(bundle: predict.Bundle, row: dict[str, float]) -> predict.FaultBlock:
        seen.append(dict(row))
        return real(bundle, row)

    monkeypatch.setattr(predict, "_fault_block", spy)
    assert _post_case(client, csv_bytes).status_code == 200
    assert seen == [expected]


def test_oversized_upload_is_413(client: TestClient) -> None:
    big = b"time,a_cpu\n" + b"1,2\n" * (predict.MAX_UPLOAD_BYTES // 4 + 10)
    assert len(big) > predict.MAX_UPLOAD_BYTES
    assert _post_case(client, big).status_code == 413


def test_window_outside_file_is_422(client: TestClient, csv_bytes: bytes) -> None:
    # featurize needs at least 10 baseline rows; the file starts at T0.
    assert _post_case(client, csv_bytes, T0 + 5).status_code == 422
    # And a detection time after the end leaves no post window.
    assert _post_case(client, csv_bytes, T0 + 5000).status_code == 422


def test_no_metric_columns_is_400(client: TestClient) -> None:
    data = b"time,foo,bar\n" + b"".join(b"%d,1,2\n" % (T0 + i) for i in range(1200))
    assert _post_case(client, data).status_code == 400


def test_raw_prometheus_file_is_400(client: TestClient) -> None:
    cols = ",".join(f"container-x{i}_cpu" for i in range(25))
    row = ",".join(["1"] * 25)
    data = f"time,{cols}\n{T0},{row}\n".encode()
    assert _post_case(client, data).status_code == 400


def test_empty_file_is_422(client: TestClient) -> None:
    assert _post_case(client, b"").status_code == 422


def test_missing_detection_time_is_422(client: TestClient, csv_bytes: bytes) -> None:
    resp = client.post(
        "/predict/case", files={"file": ("simple_metrics.csv", csv_bytes)}
    )
    assert resp.status_code == 422


# ---------- committed fixture case ----------

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
TABLE = ROOT / "data" / "features" / FEATURES_FILE
FIXTURE_CASE_ID = "re2ob/productcatalogservice_cpu/1"


def _fixture_post(client: TestClient) -> Any:
    inject_time = int((FIXTURES / "inject_time.txt").read_text().strip())
    return _post_case(
        client, (FIXTURES / "simple_metrics.csv").read_bytes(), inject_time
    )


def _fixture_row() -> pd.Series:
    table = pd.read_parquet(TABLE)
    rows = table[(table["case_id"] == FIXTURE_CASE_ID) & (table["delay_s"] == 0)]
    assert len(rows) == 1
    return rows.iloc[0]


def test_predict_case_on_fixture(client: TestClient) -> None:
    resp = _fixture_post(client)
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"fault", "suspect_service"}
    assert body["fault"]["source"] == "model"
    assert body["fault"]["cv_macro_f1"] == round(CV_F1, 3)
    assert set(body["fault"]["probabilities"]) == set(CLASSES)
    suspect = body["suspect_service"]
    assert suspect["source"] == "rule"
    assert suspect["hit_rate"] == round(HIT_RATE, 3)
    assert suspect["name"] == _fixture_row()["suspect_service"]


def test_predict_case_parity_with_feature_table(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Features built inside /predict/case equal the fixture's training row."""
    seen: list[dict[str, float]] = []
    real = predict._fault_block

    def spy(bundle: predict.Bundle, row: dict[str, float]) -> predict.FaultBlock:
        seen.append(dict(row))
        return real(bundle, row)

    monkeypatch.setattr(predict, "_fault_block", spy)
    assert _fixture_post(client).status_code == 200
    assert len(seen) == 1
    expected = _fixture_row()
    assert list(seen[0]) == FEATURE_NAMES
    for name in FEATURE_NAMES:
        assert seen[0][name] == pytest.approx(
            float(expected.to_dict()[name]), abs=1e-9
        ), name


# ---------- startup ----------


def test_missing_model_file_fails_with_its_name(
    client: TestClient, tmp_path: Path
) -> None:
    results = tmp_path / "results.json"
    results.write_text("{}")
    with pytest.raises(RuntimeError, match=r"model\.bin"):
        predict.create_app(tmp_path, results)


def test_missing_spec_and_results_fail_with_their_names(tmp_path: Path) -> None:
    (tmp_path / "model.bin").write_bytes(b"x")
    with pytest.raises(RuntimeError, match=r"feature_spec\.json"):
        predict.create_app(tmp_path, tmp_path / "results.json")
    (tmp_path / "feature_spec.json").write_text("{}")
    with pytest.raises(RuntimeError, match=r"results\.json"):
        predict.create_app(tmp_path, tmp_path / "results.json")


def test_results_without_metric_keys_fails_naming_the_key(
    client: TestClient, tmp_path: Path
) -> None:
    # Reuse the valid model and spec from the client fixture's directory.
    app_dir = tmp_path / "m"
    app_dir.mkdir()
    model = DummyClassifier(strategy="prior").fit(
        np.zeros((len(CLASSES), len(FEATURE_NAMES))), CLASSES
    )
    (app_dir / "model.bin").write_bytes(pickle.dumps(model))
    (app_dir / "feature_spec.json").write_text(
        json.dumps(
            {
                "feature_names": FEATURE_NAMES,
                "classes": CLASSES,
                "baseline_s": BASELINE_S,
                "post_s": POST_S,
                "z_floor": Z_FLOOR,
                "log_z": LOG_Z,
                "model_version": "t",
            }
        )
    )
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"suspect_hit_rate": 0.5}))
    with pytest.raises(RuntimeError, match="fault_cv_macro_f1_mean"):
        predict.create_app(app_dir, results)


def test_missing_index_html_fails_with_its_name(tmp_path: Path) -> None:
    app_dir = tmp_path / "m"
    app_dir.mkdir()
    model = DummyClassifier(strategy="prior").fit(
        np.zeros((len(CLASSES), len(FEATURE_NAMES))), CLASSES
    )
    (app_dir / "model.bin").write_bytes(pickle.dumps(model))
    (app_dir / "feature_spec.json").write_text(
        json.dumps(
            {
                "feature_names": FEATURE_NAMES,
                "classes": CLASSES,
                "baseline_s": BASELINE_S,
                "post_s": POST_S,
                "z_floor": Z_FLOOR,
                "log_z": LOG_Z,
                "model_version": "t",
            }
        )
    )
    web = tmp_path / "web"
    web.mkdir()
    results = web / "results.json"
    results.write_text(
        json.dumps({"fault_cv_macro_f1_mean": CV_F1, "suspect_hit_rate": HIT_RATE})
    )
    with pytest.raises(RuntimeError, match=r"index\.html"):
        predict.create_app(app_dir, results)


# ---------- results page ----------


def test_index_is_html(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "test page" in resp.text


def test_results_json_returns_the_fixture_values(client: TestClient) -> None:
    resp = client.get("/results.json")
    assert resp.status_code == 200
    assert resp.json() == {
        "fault_cv_macro_f1_mean": CV_F1,
        "suspect_hit_rate": HIT_RATE,
    }


@pytest.mark.parametrize(
    "path", ["/model.bin", "/feature_spec.json", "/index.html", "/web/", "/nope"]
)
def test_nothing_else_is_served(client: TestClient, path: str) -> None:
    assert client.get(path).status_code == 404


def test_web_files_are_not_in_the_openapi_schema(client: TestClient) -> None:
    paths = set(client.get("/openapi.json").json()["paths"])
    assert paths == {"/healthz", "/predict", "/predict/case"}


# ---------- committed artifacts ----------


def test_predict_case_with_committed_artifacts() -> None:
    """The real model, spec and results.json, paths taken from the repo root."""
    spec = json.loads((ROOT / "models" / "feature_spec.json").read_text())
    results = json.loads((ROOT / "web" / "results.json").read_text())
    app = predict.create_app(ROOT / "models", ROOT / "web" / "results.json")
    with TestClient(app) as real:
        assert real.get("/").status_code == 200
        assert real.get("/results.json").json() == results
        resp = _fixture_post(real)
        assert resp.status_code == 200
        body = resp.json()
    fault, suspect = body["fault"], body["suspect_service"]
    assert fault["label"] in spec["classes"]
    assert fault["source"] == "model"
    assert fault["cv_macro_f1"] == round(results["fault_cv_macro_f1_mean"], 3)
    assert fault["model_version"] == spec["model_version"]
    assert suspect["source"] == "rule"
    assert suspect["hit_rate"] == round(results["suspect_hit_rate"], 3)
    print("fixture response:", json.dumps(body))  # shown with pytest -s, for the PR


def test_import_does_not_build_the_app() -> None:
    assert "_app" in vars(predict)
    with pytest.raises(AttributeError):
        predict.__getattr__("nope")
