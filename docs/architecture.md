# Architecture

This page covers how FaultTrack is put together:
- the data pipeline
- feature computation
- training
- serving
- deployment
- CI

What to build and why is in [`product-spec.md`](../product-spec.md), and working rules are in [`AGENTS.md`](../AGENTS.md). Agents and reviewers should read this page before changing code that crosses module boundaries.

> **Status:** this is the target design for the graded build (spec Section 10). Feature names, model settings and the results figures listed here are fixed in `models/feature_spec.json` and `web/results.json` once `train.py` exists. Where this page and those files disagree, the files win, and this page gets corrected.

## Overview

```mermaid
flowchart LR
    subgraph offline["Offline (devcontainer)"]
        Z[("Zenodo RE2 zips")] -.->|make data| RAW["data/raw/"]
        RAW -.->|make features| FT["data/features/<br/>re2_features.parquet"]
        FT --> NB["notebook.ipynb"]
        FT --> TR["train.py"]
        TR --> ART["models/model.bin<br/>models/feature_spec.json<br/>web/results.json"]
    end
    subgraph lib["src/faulttrack (shared)"]
        DATA["data.py"]
        FEAT["features.py"]
    end
    RAW -.-> DATA
    DATA --> FEAT
    FEAT -.-> FT
    subgraph online["Online (container on Fly.io)"]
        API["predict.py<br/>FastAPI"]
        WEB["web/ results page"]
    end
    ART --> API
    FEAT --> API
    ART --> WEB
    API --> WEB
```

The dashed steps need the 4 GB raw download. Everything from `re2_features.parquet` onward runs from committed files.

## Module boundaries

| Layer | Rule |
|---|---|
| `src/faulttrack/data.py` | The only code that knows the RCAEval folder layout and its quirks (spec 3.3). It returns a tidy metrics `DataFrame` (one `time` column + `<service>_<kind>` columns), the injection time, and the label. No feature logic |
| `src/faulttrack/features.py` | The only code that computes features. It's a pure function of (metrics `DataFrame`, detection time, settings) and does no file I/O. It's used by `make features`, `train.py`, the notebook and `predict.py` |
| `src/faulttrack/evaluation.py` | The only CV code; always grouped by `case_id`. `grouped_cv`, `leave_one_system_out`, `grouped_permutation_importance` and `suspect_hit_rate` are used by the notebook, `train.py` and the experiments. Models see only `FEATURE_NAMES`. A `GridSearchCV` / `RandomizedSearchCV` passed to `grouped_cv` gets `groups=` (the training fold's `case_id`), so tuning inside it is nested and grouped |
| `src/faulttrack/models.py` | The only model definitions: `RuleBaseline`, `candidate_models()`, the tuning grids and `tuned_search()`, and `final_model()`, the chosen model with its tuned hyperparameters. The notebook scores `final_model()` and `train.py` fits it; neither defines a model of its own |
| `src/faulttrack/build.py` | `make features`: walks every case through `data.py` and `features.py` and writes the feature table and the case inventory. Its only logic of its own is row bookkeeping |
| `notebook.ipynb` | Reads `re2_features.parquet`. It may plot raw cases for EDA through `data.py`, but never re-implements features |
| `train.py` | Reads `re2_features.parquet`, fits the final model with the settings chosen in the notebook, and writes the artifacts. It doesn't download anything |
| `predict.py` | Loads the artifacts once at startup. It handles HTTP only and calls `features.py` for raw cases. It has no training code |
| `spike/` | Go/no-go scripts. Graded code never imports from it |

**Tests:**
- **Unit tests** cover `data.py` quirks and `features.py` math against small synthetic frames.
- **The parity test** runs one real fixture case (`tests/fixtures/`) through `features.py` the way `/predict/case` does, and asserts the result equals that case's row in `re2_features.parquet`.
- **API tests** use FastAPI's `TestClient`.
- No test needs `data/raw/` or the network.

## Data pipeline

| Stage | Code | Input | Output |
|---|---|---|---|
| Download | `make data` → `spike/download_rcaeval.sh` | Zenodo record 14590730 | `data/raw/RE2-{OB,SS,TT}/…/<service>_<fault>/<n>/` with `simple_metrics.csv`, `inject_time.txt` (logs and traces aren't extracted) |
| Discover cases | `data.find_cases()` | `data/raw/` | One entry per case: system, service, fault, repetition, path. Cases without `simple_metrics.csv` are skipped with a stderr message (none in RE2; `checkoutservice_cpu/multi-source-data/` in RE2-OB isn't a numbered case and is ignored) |
| Load | `data.load_case()` = `read_metrics()` + `clean_metrics()` + `inject_time.txt` + label | One case folder | Metrics frame. `inf` becomes NaN, then forward-fill, then 0. Rows with an empty `time` are dropped (two RE2-OB files end with about 256 of them); non-numeric `time` text is an error. The repeated `time.1` header column is dropped, and `latency-50` is ignored. A raw Prometheus file (more than 20 `container-`/`istio-` columns) raises `ValueError`. `predict.py` cleans an uploaded CSV with the same `read_metrics()` + `clean_metrics()` |
| Featurize | `features.featurize()` | Metrics frame + detection time | One feature row |
| Build table | `make features` → `faulttrack.build` | All cases × 5 detection delays | `data/features/re2_features.parquet` (1,350 rows: 270 cases × 5) and `data/features/re2_inventory.parquet` (one row per case folder, skipped ones with their reason) |

## Feature computation

For one case and one detection time `t_det` (the injection time plus a detection delay of 0, 30, 60, 90 or 120 s):

```mermaid
flowchart LR
    B["Baseline window<br/>[t_det − 300 s, t_det)"] --> Z["Shift per column<br/>z = (median_post − median_base) / scale<br/>scale = max(1.4826·MAD, 0.01·|median_base|)<br/>then sign(z)·log1p(|z|)"]
    P["Post window<br/>[t_det, t_det + 120 s)"] --> Z
    Z --> K["Group columns by metric kind<br/>cpu, mem, diskio, socket,<br/>latency, error, workload"]
    K --> AGG["Per kind: max, 2nd-highest,<br/>sum of |z|, count above threshold"]
    Z --> TOP["Most-shifted service:<br/>its z per kind + each kind's share"]
    AGG --> ROW["Feature row"]
    TOP --> ROW
```

**Design notes:**
- **The features are service-agnostic.** Columns are grouped by metric *kind*, never by service name, so one model covers systems with 11, 15 or 68 services. A service and its database (`carts`, `carts-db`) count as one service, keeping the larger shift of the two (see `docs/decisions.md`).
- **Scale floor and log compression.** Without them, near-constant metrics (memory) get huge z-scores that hide everything else. In the go/no-go spike, adding them took logistic regression on RE2-OB from 0.54 to 0.85 macro F1.
- **Missing kinds are 0, not NaN.** Some services don't export `diskio`, `latency` or `error`. A kind with no columns gets 0 for every aggregate. This is a known blind spot for disk faults: in RE2-OB, only 2 of the 5 disk-fault services expose `diskio`.
- **Row identity columns:** `case_id`, `system`, `service`, `fault` (label), `delay_s` and `suspect_service` (the root-cause rule's pick), plus the metadata counts `n_services_<kind>`. These are never model inputs. `case_id` is the CV group.

## Training

```mermaid
flowchart TD
    FTP["re2_features.parquet"] --> SPLIT["GroupKFold(5) by case_id"]
    SPLIT --> CV["Cross-validate the chosen model<br/>(settings from the notebook)"]
    CV --> RES["web/results.json<br/>macro F1 mean ± std, per-class metrics,<br/>confusion matrix, LOSO table"]
    FTP --> FIT["Refit on all rows"]
    FIT --> BIN["models/model.bin"]
    FIT --> SPEC["models/feature_spec.json<br/>feature order, classes, window settings,<br/>model_version"]
```

| Artifact | Contents | Used by |
|---|---|---|
| `models/model.bin` | The fitted classifier, pickled | `predict.py` |
| `models/feature_spec.json` | Ordered feature names, class labels, the baseline and post window lengths, the scale floor, `model_version` (for example `2026.11.10-rf`) | `predict.py` (validates input and builds the vector in order), parity test |
| `web/results.json` | Evaluation figures written by `train.py` | the results page at `/` |

**Evaluation rules (spec Section 5):**
- **Splits:** grouped CV by case only. Each case contributes 5 rows, so a row-level split would leak.
- **Headline:** grouped 5-fold CV across all three systems, reported as mean ± std (min/max).
- **Generalization:** leave-one-system-out is a separate section.

## Serving

```mermaid
sequenceDiagram
    participant C as Client (reviewer, curl, /docs)
    participant A as predict.py
    participant F as faulttrack.features
    participant M as model.bin
    C->>A: POST /predict/case (simple_metrics.csv + detection time)
    A->>A: validate (size limit, Pydantic, required columns)
    A->>F: featurize(frame, t_det, settings from feature_spec.json)
    F-->>A: feature row
    A->>M: predict_proba(row in spec order)
    M-->>A: class probabilities
    A->>A: suspect service = most anomalous service (rule, not the model)
    A-->>C: fault {label, probabilities, top_signals, source: model, cv_macro_f1} + suspect_service {name, source: rule, hit_rate}
```

| Endpoint | Input | Output | Errors |
|---|---|---|---|
| `GET /` | none | Static results page (`web/index.html` + `results.json`) | none |
| `POST /predict` | JSON feature vector (names from `feature_spec.json`) | `fault` block only (spec Section 6) | 422: missing or unknown features |
| `POST /predict/case` | Multipart: `simple_metrics.csv` + `detection_time` (Unix seconds) | `fault` block (model) + `suspect_service` block (rule), each with its source and its accuracy | 413: file too large · 422: bad input, or the window falls outside the file · 400: no usable metric columns (for example a raw `metrics.csv`) |
| `GET /healthz` | none | `{"status": "ok", "model_version": …}` | none |
| `GET /docs` | none | Swagger UI | none |

**Runtime behavior:**
- **Startup:** the model and spec load once at startup. If either is missing, the app exits instead of serving errors.
- **State:** the app holds no state and writes nothing to disk. Uploads are read in memory, subject to the size limit.
- **Port:** the app listens on `$PORT` (default 8080).

## Root-cause rule

`suspect_service` comes from a **rule, not the model**: it's the service with the largest total shift across metric kinds in the detection window. No ML is involved.

**Its accuracy is reported separately from the classifier's:**
- **Rule:** hit rate, the share of cases where the rule names the injected service. It's 0.900 per alert window on the committed feature table (go/no-go spike: 0.853).
- **Classifier:** macro F1 for the fault type. The final model scores 0.799 ± 0.028 with grouped 5-fold CV (go/no-go spike: 0.83).

**Where they appear:**
- **Results files:** `train.py` computes both and writes them to `web/results.json`.
- **API response:** each answer carries its own `source` (`"model"` or `"rule"`) and its own metric (`cv_macro_f1` or `hit_rate`).
- **README Results table:** both appear side by side.

A trained ranking model is stretch work.

## Deployment

```mermaid
flowchart LR
    SRC["repo at a commit"] --> IMG["Dockerfile<br/>1. deps layer: uv sync --frozen --no-dev --no-install-project<br/>2. COPY src/, predict.py, models/, web/<br/>3. uv sync --frozen --no-dev<br/>non-root user, CMD uvicorn on $PORT"]
    IMG -->|fly deploy, maintainer| FLY["Fly.io machine<br/>shared CPU, 512 MB<br/>auto-stop, min 0"]
    DNS["faulttrack.jennasrunbooks.com<br/>(fly certs + DNS record)"] --> FLY
```

**Cold starts:** the machine stops when idle, so the first request after a quiet period takes a few seconds while it starts. Later requests are fast.

## CI

**When it runs:** `.github/workflows/ci.yaml` runs on every PR and on every push to `main`.

**Jobs:** a `changes` job decides which of `lint`, `test` and `docker` run. It's a shell step: `git diff --name-only` against the PR base, matched with `grep -E`, with no third-party filter action. On PRs, a job is skipped when none of its files changed. Pushes to `main`, `workflow_dispatch`, and PRs with the `full-ci` label run everything.

**Gate:** `ci-ok` is the single required check. It passes when every other job either passed or was skipped. Path rules are in the spec's PR CI design table.

**Dependabot:** `dependabot.yaml` covers `uv`, `github-actions` and `docker`. Its PRs go through the same CI.

## Where things are decided

| Question | Answer lives in |
|---|---|
| What to build, by when | `product-spec.md` |
| Why a modelling or evaluation choice was made | `docs/decisions.md` |
| Exact feature list and model settings | `models/feature_spec.json` |
| Current results | `web/results.json`, the README "Results" section |
| Who may change what | `docs/permissions.md`, `docs/agent-team.md` |