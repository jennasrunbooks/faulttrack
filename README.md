# FaultTrack

[![CI](https://github.com/jennasrunbooks/faulttrack/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/jennasrunbooks/faulttrack/actions/workflows/ci.yaml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

FaultTrack looks at the metrics around an alert and tells on-call what kind of fault it probably is (CPU, memory, disk, socket, network delay or packet loss) and which service it most likely started in, before anyone opens a dashboard.

Live demo: **https://faulttrack.jennasrunbooks.com** (API docs at [`/docs`](https://faulttrack.jennasrunbooks.com/docs))

Reviewing for the [2026 ML Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) course? Start with [REVIEWING.md](REVIEWING.md).

> **Draft.** Sections marked _TBD_ are filled in as the plan lines in [product-spec.md](product-spec.md) complete. Final numbers come from the fresh-clone run on Nov 15.

---

## Problem

When an alert fires in a microservice system, the first minutes of an incident go to two questions:

- **What kind of problem is this?** Resource exhaustion (CPU, memory, disk, sockets) or a network problem (added delay, dropped packets)?
- **Where did it start?** A network fault on one service slows down every service that sends requests to it, so the symptoms show up in many places at once.

The answers decide who gets paged and which runbook to open. Today an on-call engineer works them out by clicking through dashboards.

FaultTrack makes that first guess automatically. It takes the per-service metrics in a window around the alert, and returns:

- **a fault type** with class probabilities
- **the suspected root-cause service.** This comes from a simple rule (the most anomalous service), not from the model. It's labeled that way in every response, with its own hit rate.
- **the signals that drove the prediction**

## How it's used

1. **An alert fires.** Today that means a manual step: export the alert window's metrics as a CSV. A live Alertmanager webhook is planned stretch work.
2. **The CSV goes to `POST /predict/case`** along with the detection time.
3. **FaultTrack replies** with two separately labeled answers:
   - **From the classifier:** the fault type (for example `loss`), with probabilities, the top contributing signals and the model's cross-validated macro F1.
   - **From a rule, not the model:** the suspect service (for example `cartservice`), with the rule's hit rate.
4. **On-call opens the matching runbook.** In this example that's the packet-loss runbook, starting at `cartservice`. The depth-3 decision tree on the results page is a readable version of the same logic.

**Who it's for:** on-call SREs and platform engineers triaging incidents in Kubernetes-style microservice systems.

**What it doesn't claim:** reliable accuracy on a system it was never trained on. See [Generalizing to an unseen system](#generalizing-to-an-unseen-system).

## Data

**RCAEval RE2** (Pham et al., *RCAEval: A Benchmark for Root Cause Analysis of Microservice Systems with Telemetry Data*, WWW 2025 Companion).

- **Source:** Zenodo, [doi:10.5281/zenodo.14590730](https://doi.org/10.5281/zenodo.14590730), licensed **CC BY 4.0**.
- **Systems:** three microservice systems, Online Boutique, Sock Shop and Train Ticket.
- **Design:** 6 fault types × 5 services × 3 repetitions per system = **270 fault-injection cases**.
- **What's used:** each case has per-service metrics at 1-second resolution for 24 minutes, with the fault injected at the midpoint. FaultTrack uses the metrics only (CPU, memory, disk I/O, sockets, latency, errors, workload), not the logs or traces.

**What's in this repo:**
- **[`data/features/re2_features.parquet`](data/features/):** the derived feature table (Parquet), about 1,350 rows (5 detection delays per case). It's committed, so you can run the notebook and the training script without downloading anything. Credit and the changes made are in [`ATTRIBUTION.md`](data/features/ATTRIBUTION.md).
- **The raw data** (about 4 GB of zips) isn't committed. To rebuild the table from scratch, see [`data/README.md`](data/README.md): `make data && make features`.

## Approach

**Features.** For each metric column, FaultTrack measures how far the metric moved after the alert compared with the 5 minutes before it. The shift is scaled by the baseline's spread, with a floor for near-constant metrics, and then log-compressed. The columns are grouped by metric *kind* rather than by service, so one model works across systems with 11, 15 or 68 services. On top of the per-kind aggregates, the features include the most-shifted service's profile. Details are in [docs/architecture.md](docs/architecture.md#feature-computation).

**Models.** The candidates, compared on the same grouped folds and each tuned:

- **baselines:** a dummy (majority-class) baseline, and a rule baseline where the biggest-shifting metric kind wins
- **models:** logistic regression, a decision tree, a random forest, and XGBoost

**Evaluation.** Each case contributes 5 rows, one per detection delay, so cross-validation is **grouped by case**. A random split would put windows from the same case in both train and test. Scores are reported as macro F1, mean ± std across folds.

## EDA highlights

_TBD (plan line 5)._ Planned highlights:

- **Fault signatures:** how each fault type shows up in the raw metrics.
- **The disk telemetry gap:** only some services export disk I/O, so disk faults often show up through memory and CPU instead.
- **Why scaling mattered:** in the go/no-go spike, logistic regression on Online Boutique went from 0.54 to 0.85 macro F1 with the same features once the scale floor and log compression were added.
- **Feature importance.**

## Results

_TBD (plan lines 6 and 8)._ Headline: grouped 5-fold CV across all three systems.

| Model | Macro F1 (mean ± std) | Min / max across folds |
|---|---|---|
| Dummy (majority class) | _TBD_ | |
| Rule baseline | _TBD_ | |
| Logistic regression | _TBD_ | |
| Decision tree | _TBD_ | |
| Random forest | _TBD_ | |
| XGBoost | _TBD_ | |
| **Final model** | _TBD_ | |

**Also reported:**
- the confusion matrix and per-class precision and recall
- how F1 changes with detection delay
- the suspect-service rule's hit rate. This is reported separately because it's a rule, not the model (see below)

_TBD._

**Fault type and suspect service are separate results.** The fault type comes from the trained classifier. The suspect service comes from a rule: the most anomalous service, with no ML involved.

| Answer | Comes from | Metric | Value |
|---|---|---|---|
| Fault type | Trained classifier | Macro F1, grouped 5-fold CV across 3 systems | _TBD_ (go/no-go spike: 0.83) |
| Suspect service | Rule: most anomalous service, not the model | Hit rate: share of cases where it names the injected service | _TBD_ (go/no-go spike: 0.85) |

### Generalizing to an unseen system

_TBD (plan line 7)._ This section trains on two systems and tests on the third, rotating through all three, and explains why that score is much lower than the headline. It's a domain-shift finding, reported here as a limitation rather than hidden.

## Setup

You need **[uv](https://docs.astral.sh/uv/getting-started/installation/)**. It installs Python 3.12 if you don't have it, then creates the virtual environment from the locked dependencies.

```bash
git clone https://github.com/jennasrunbooks/faulttrack.git
cd faulttrack
uv sync                      # creates .venv from pyproject.toml + uv.lock
```

**Running commands:** prefix them with `uv run` (as below), or activate the environment first with `source .venv/bin/activate`.

## Run it

```bash
uv run jupyter lab notebook.ipynb        # EDA, model selection, tuning
uv run python train.py                   # trains the final model: models/ + web/results.json
uv run uvicorn predict:app --port 8080   # serves the API: http://localhost:8080/docs
uv run pytest                            # tests
```

**Example request** (from a second terminal):

```bash
curl -s localhost:8080/healthz

# The fixture is one real RE2 case. Its injection time is the detection time
# (a detection delay of 0 s).
curl -s -F file=@tests/fixtures/simple_metrics.csv \
     -F detection_time="$(cat tests/fixtures/inject_time.txt)" \
     localhost:8080/predict/case | uv run python -m json.tool
```

The response has two separately labeled answers (shape shown; values come from the trained model and `web/results.json`):

```json
{
  "fault": {"label": "...", "probabilities": {"...": 0.0}, "top_signals": ["..."],
            "source": "model", "model_version": "...", "cv_macro_f1": 0.0},
  "suspect_service": {"name": "...", "source": "rule", "rule": "most anomalous service: ...",
                      "hit_rate": 0.0}
}
```

`tests/fixtures/README.md` names the source case and its true fault, so you can check the prediction. To call the live service instead, replace `localhost:8080` with `https://faulttrack.jennasrunbooks.com`.

## Docker

```bash
docker build -t faulttrack .        # or: make docker-build
docker run --rm -p 8080:8080 faulttrack   # or: make docker-run
```

The image holds the runtime dependencies, `src/`, `predict.py`, `models/` and `web/` only, and runs as a non-root user. It listens on `$PORT` (default 8080). Check it with the same `curl` commands as above:

```bash
curl -s localhost:8080/healthz
curl -s -F file=@tests/fixtures/simple_metrics.csv \
     -F detection_time="$(cat tests/fixtures/inject_time.txt)" \
     localhost:8080/predict/case
```

Or open http://localhost:8080/docs.

## Deploy

FaultTrack runs on [Fly.io](https://fly.io) as one small machine that stops when idle. The first request after a quiet period takes a few seconds.

**Configuration:** [`fly.toml`](fly.toml). **Commands:** _TBD (plan line 11)._

```bash
fly launch --no-deploy        # first time only: creates the app from fly.toml
fly deploy                    # builds the Dockerfile and deploys
fly certs add faulttrack.jennasrunbooks.com
```

**Screenshots** of the deployed service in use: [docs/images/](docs/images/) (_TBD_).

## Repository layout

```text
notebook.ipynb   train.py   predict.py   Dockerfile   fly.toml   Makefile
pyproject.toml   uv.lock    .python-version
src/faulttrack/  shared data loading + feature code (used by training and serving)
data/            raw data instructions; committed feature table
models/          trained model + feature spec
web/             results page served at /
tests/           feature parity + API tests
docs/            architecture, decisions, screenshots
```

More: [docs/architecture.md](docs/architecture.md) covers how the pieces fit, and [product-spec.md](product-spec.md) covers scope and plan.

## Rubric map

| Criterion | Where |
|---|---|
| Problem description | [Problem](#problem), [How it's used](#how-its-used) |
| EDA | [notebook.ipynb](notebook.ipynb) §2–3, [EDA highlights](#eda-highlights) |
| Model training | [notebook.ipynb](notebook.ipynb) §4–5, [Results](#results) |
| Exporting notebook to script | [train.py](train.py) |
| Reproducibility | Committed [feature table](data/features/), [Setup](#setup), [Run it](#run-it) |
| Model deployment | [predict.py](predict.py) (FastAPI) |
| Dependency and environment management | [pyproject.toml](pyproject.toml), [uv.lock](uv.lock), [Setup](#setup) |
| Containerization | [Dockerfile](Dockerfile), [Docker](#docker) |
| Cloud deployment | [fly.toml](fly.toml), [Deploy](#deploy), live URL, [screenshots](docs/images/) |

## License and credit

- **Code:** [Apache-2.0](LICENSE).
- **Data:** RCAEval by Pham et al., [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), [doi:10.5281/zenodo.14590730](https://doi.org/10.5281/zenodo.14590730). The committed feature table is derived from it; see [ATTRIBUTION.md](data/features/ATTRIBUTION.md).