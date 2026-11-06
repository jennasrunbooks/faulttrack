"""FaultTrack API (spec Section 6).

Two answers, kept separate and labeled:

- ``fault`` comes from the trained classifier (``"source": "model"``), reported
  with ``cv_macro_f1``.
- ``suspect_service`` comes from a rule, not the model (``"source": "rule"``):
  the service with the largest total shift across metric kinds. It is reported
  with ``hit_rate``.

Both metrics are read from ``web/results.json`` at startup, never hardcoded.

``create_app`` loads ``models/model.bin``, ``models/feature_spec.json``,
``web/results.json`` and ``web/index.html`` once and raises ``RuntimeError``
naming the file if one is missing. Only ``/`` (``index.html``) and
``/results.json`` are served from ``web/``, as two explicit routes: nothing else
in that directory is reachable. ``app`` (for ``uvicorn predict:app``) is built on first access, so
importing this module, for example from tests, doesn't need the artifacts.
"""

from __future__ import annotations

import io
import json
import math
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from faulttrack.data import clean_metrics, kind_of, read_metrics
from faulttrack.features import feature_names, featurize

MODEL_FILE = "model.bin"
SPEC_FILE = "feature_spec.json"
INDEX_FILE = "index.html"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
"""Largest accepted ``simple_metrics.csv`` upload."""
TOP_SIGNALS = 3
"""How many feature names to list in ``fault.top_signals``."""

# Keys in web/results.json that train.py (plan line 8) must write.
RESULTS_FAULT_KEY = "fault_cv_macro_f1_mean"
RESULTS_SUSPECT_KEY = "suspect_hit_rate"

SUSPECT_RULE = (
    "most anomalous service: the largest total shift across metric kinds "
    "in the detection window"
)


# ---------- response and request models ----------


class FeatureVector(BaseModel):
    """A feature row. Names and meaning are listed in ``feature_spec.json``."""

    features: dict[str, float] = Field(
        description="Value for every feature in feature_spec.json, by name. "
        "No missing and no extra names."
    )


class FaultBlock(BaseModel):
    """The fault-type answer. It comes from the trained classifier."""

    label: str = Field(description="Most likely fault kind (the model's prediction).")
    probabilities: dict[str, float] = Field(
        description="The model's probability for each fault kind."
    )
    top_signals: list[str] = Field(
        description="Features that drove this prediction most: importance times "
        "|value| for tree models, |value| otherwise."
    )
    source: Literal["model"] = Field(
        description="Always 'model': this answer is a trained classifier's output."
    )
    model_version: str
    cv_macro_f1: float = Field(
        description="The model's macro F1 in grouped 5-fold cross-validation "
        "across all three systems (mean over folds), from results.json."
    )


class SuspectBlock(BaseModel):
    """The root-cause answer. It comes from a rule, not from the model."""

    name: str | None = Field(
        description="Service the rule points at. This is NOT a model prediction."
    )
    source: Literal["rule"] = Field(
        description="Always 'rule': a fixed rule, no machine learning involved."
    )
    rule: str = Field(description="Plain-text description of the rule.")
    hit_rate: float = Field(
        description="Share of RE2 detection windows (5 per case) where the rule "
        "names the injected service, from results.json."
    )


class PredictResponse(BaseModel):
    """Response of ``POST /predict``: the fault block only."""

    fault: FaultBlock


class CaseResponse(BaseModel):
    """Response of ``POST /predict/case``: two separate, labeled answers."""

    fault: FaultBlock
    suspect_service: SuspectBlock = Field(
        description="Root-cause guess from a RULE, not from the model. Its "
        "accuracy is hit_rate, which is separate from the model's cv_macro_f1."
    )


class Health(BaseModel):
    status: Literal["ok"]
    model_version: str


# ---------- loading ----------


@dataclass(frozen=True)
class Bundle:
    """Everything loaded once at startup."""

    model: Any
    features: list[str]
    classes: list[str]
    model_version: str
    baseline_s: float
    post_s: float
    z_floor: float
    log_z: bool
    norm_counts: bool
    cv_macro_f1: float
    hit_rate: float


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing required file: {path}")
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as e:
        raise RuntimeError(f"cannot read {path}: {e}") from e
    if not isinstance(data, dict):
        raise RuntimeError(f"{path} must hold a JSON object")
    return data


def _require(data: dict[str, Any], key: str, path: Path) -> Any:
    if key not in data:
        raise RuntimeError(f"{path} has no '{key}' key")
    return data[key]


def load_bundle(model_dir: Path, results_path: Path) -> Bundle:
    """Load and cross-check the model, its feature spec and the results file.

    Raises ``RuntimeError`` with a message naming the file if anything is
    missing, unreadable or inconsistent.
    """
    model_path = model_dir / MODEL_FILE
    spec_path = model_dir / SPEC_FILE
    if not model_path.is_file():
        raise RuntimeError(f"missing required file: {model_path}")
    spec = _read_json(spec_path)
    results = _read_json(results_path)

    try:
        with open(model_path, "rb") as fh:
            model = pickle.load(fh)  # our own artifact, written by train.py
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError) as e:
        raise RuntimeError(f"cannot load {model_path}: {e}") from e

    features = [str(f) for f in _require(spec, "feature_names", spec_path)]
    classes = [str(c) for c in _require(spec, "classes", spec_path)]
    norm_counts = bool(spec.get("norm_counts", False))
    if features != feature_names(norm_counts):
        raise RuntimeError(
            f"{spec_path}: 'feature_names' doesn't match faulttrack.features "
            f"(norm_counts={norm_counts}); regenerate it with train.py"
        )
    model_classes = [str(c) for c in getattr(model, "classes_", [])]
    if model_classes != classes:
        raise RuntimeError(
            f"{model_path} classes {model_classes} differ from {spec_path} "
            f"classes {classes}"
        )
    metrics: dict[str, float] = {}
    for key in (RESULTS_FAULT_KEY, RESULTS_SUSPECT_KEY):
        value = _require(results, key, results_path)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise RuntimeError(f"{results_path}: '{key}' must be a number")
        metrics[key] = float(value)

    return Bundle(
        model=model,
        features=features,
        classes=classes,
        model_version=str(_require(spec, "model_version", spec_path)),
        baseline_s=float(_require(spec, "baseline_s", spec_path)),
        post_s=float(_require(spec, "post_s", spec_path)),
        z_floor=float(_require(spec, "z_floor", spec_path)),
        log_z=bool(_require(spec, "log_z", spec_path)),
        norm_counts=norm_counts,
        cv_macro_f1=metrics[RESULTS_FAULT_KEY],
        hit_rate=metrics[RESULTS_SUSPECT_KEY],
    )


# ---------- prediction ----------


def _fault_block(bundle: Bundle, row: dict[str, float]) -> FaultBlock:
    values = np.array([[row[name] for name in bundle.features]], dtype=float)
    model = bundle.model
    x: Any = (
        pd.DataFrame(values, columns=bundle.features)
        if hasattr(model, "feature_names_in_")
        else values
    )
    proba = np.asarray(model.predict_proba(x))[0]
    probabilities = {
        c: round(float(p), 4) for c, p in zip(bundle.classes, proba, strict=True)
    }
    label = bundle.classes[int(np.argmax(proba))]

    weight = np.ones(len(bundle.features))
    importances = getattr(model, "feature_importances_", None)
    if importances is not None and len(importances) == len(bundle.features):
        weight = np.asarray(importances, dtype=float)
    score = weight * np.abs(values[0])
    order = np.argsort(-score, kind="stable")[:TOP_SIGNALS]
    top = [bundle.features[i] for i in order if score[i] > 0]

    return FaultBlock(
        label=label,
        probabilities=probabilities,
        top_signals=top,
        source="model",
        model_version=bundle.model_version,
        cv_macro_f1=bundle.cv_macro_f1,
    )


def create_app(
    model_dir: Path = Path("models"),
    results_path: Path = Path("web/results.json"),
    web_dir: Path | None = None,
) -> FastAPI:
    """Build the API. Loads the artifacts once; raises ``RuntimeError`` naming
    the file if any is missing, so a broken deploy fails at startup.

    ``web_dir`` holds ``index.html`` and defaults to ``results_path``'s folder.
    """
    results_path = Path(results_path)
    index_path = (Path(web_dir) if web_dir is not None else results_path.parent) / (
        INDEX_FILE
    )
    bundle = load_bundle(Path(model_dir), results_path)
    if not index_path.is_file():
        raise RuntimeError(f"missing required file: {index_path}")
    app = FastAPI(
        title="FaultTrack",
        version=bundle.model_version,
        description=(
            "Classifies the fault behind an alert from per-service Prometheus "
            "metrics. **Two separate answers:** `fault` is the output of a "
            "trained classifier (reported with its cross-validated macro F1); "
            "`suspect_service` comes from a **rule, not the model** (reported "
            "with its own hit rate)."
        ),
    )

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(index_path, media_type="text/html")

    @app.get("/results.json", include_in_schema=False)
    def results() -> FileResponse:
        return FileResponse(results_path, media_type="application/json")

    @app.get("/healthz", response_model=Health, summary="Liveness check")
    def healthz() -> Health:
        return Health(status="ok", model_version=bundle.model_version)

    @app.post(
        "/predict",
        response_model=PredictResponse,
        summary="Fault type from a feature vector (model only)",
    )
    def predict(body: FeatureVector) -> PredictResponse:
        missing = [f for f in bundle.features if f not in body.features]
        unknown = [f for f in body.features if f not in set(bundle.features)]
        if missing or unknown:
            raise HTTPException(
                status_code=422,
                detail={"missing_features": missing, "unknown_features": unknown},
            )
        bad = [f for f, v in body.features.items() if not math.isfinite(v)]
        if bad:
            raise HTTPException(status_code=422, detail={"non_finite_features": bad})
        return PredictResponse(fault=_fault_block(bundle, body.features))

    @app.post(
        "/predict/case",
        response_model=CaseResponse,
        summary="Fault type (model) and suspect service (rule) from a raw case",
        responses={
            400: {"description": "No usable metric columns (e.g. a raw metrics.csv)"},
            413: {"description": "Upload larger than 5 MB"},
            422: {"description": "Bad CSV, or the windows fall outside the file"},
        },
    )
    def predict_case(
        file: Annotated[
            UploadFile, File(description="simple_metrics.csv for one case.")
        ],
        detection_time: Annotated[
            float,
            Form(
                ge=0,
                description="Detection time, Unix seconds. The file needs at least 10 "
                "samples in the 300 s before it and 5 in the 120 s from it.",
            ),
        ],
    ) -> CaseResponse:
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"upload larger than {MAX_UPLOAD_BYTES // 1024 // 1024} MB",
            )
        try:
            raw = read_metrics(io.BytesIO(data))
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        try:
            frame = clean_metrics(raw)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        if not any(kind_of(str(c)) for c in frame.columns if c != "time"):
            raise HTTPException(
                status_code=400,
                detail="no usable metric columns; expected simple_metrics.csv "
                "with <service>_<metric> columns",
            )
        try:
            result = featurize(
                frame,
                detection_time,
                baseline_s=bundle.baseline_s,
                post_s=bundle.post_s,
                z_floor=bundle.z_floor,
                log_z=bundle.log_z,
                norm_counts=bundle.norm_counts,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        return CaseResponse(
            fault=_fault_block(bundle, result.features),
            suspect_service=SuspectBlock(
                name=result.suspect_service,
                source="rule",
                rule=SUSPECT_RULE,
                hit_rate=bundle.hit_rate,
            ),
        )

    return app


_app: FastAPI | None = None


def __getattr__(name: str) -> FastAPI:
    """Build ``app`` on first access (``uvicorn predict:app``), not on import."""
    global _app
    if name != "app":
        raise AttributeError(f"module 'predict' has no attribute {name!r}")
    if _app is None:
        _app = create_app()
    return _app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_app(), host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
