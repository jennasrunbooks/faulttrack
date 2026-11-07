# FaultTrack

[![CI](https://github.com/jennasrunbooks/faulttrack/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/jennasrunbooks/faulttrack/actions/workflows/ci.yaml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

FaultTrack looks at the metrics around an alert and tells on-call what kind of fault it probably is (CPU, memory, disk, socket, network delay or packet loss) and which service it most likely started in, before anyone opens a dashboard.

Live demo: **https://faulttrack.jennasrunbooks.com** (API docs at [`/docs`](https://faulttrack.jennasrunbooks.com/docs))

Reviewing for the [2026 ML Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) course? Start with [REVIEWING.md](REVIEWING.md).

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
- **Systems:** three microservice systems, Online Boutique (OB), Sock Shop (SS) and Train Ticket (TT).
- **Design:** 6 fault types × 5 services × 3 repetitions per system = **270 fault-injection cases**, exactly 45 per fault type.
- **What's used:** each case has per-service metrics at 1-second resolution for 24 minutes, with the fault injected at the midpoint. FaultTrack uses the metrics only (CPU, memory, disk I/O, sockets, latency, errors, workload), not the logs or traces.

**What's in this repo:**
- **[`data/features/re2_features.parquet`](data/features/):** the derived feature table (Parquet), 1,350 rows (5 detection delays per case). It's committed, so you can run the notebook and the training script without downloading anything. Credit and the changes made are in [`ATTRIBUTION.md`](data/features/ATTRIBUTION.md).
- **The raw data** (about 4 GB of zips) isn't committed. To rebuild the table from scratch, see [`data/README.md`](data/README.md): `make data && make features`.

## Approach

**Features.** For each metric column, FaultTrack measures how far the metric moved after the alert compared with the 5 minutes before it. The shift is scaled by the baseline's spread, with a floor for near-constant metrics, and then log-compressed. The columns are grouped by metric *kind* rather than by service, so one model works across systems with 11 or 68 services. On top of the per-kind aggregates, the features include the most-shifted service's profile. Details are in [docs/architecture.md](docs/architecture.md#feature-computation).

**Models.** The candidates, compared on the same grouped folds:

- **baselines:** a dummy (majority-class) baseline, and a rule baseline where the biggest-shifting metric kind wins
- **models:** logistic regression, a decision tree, a random forest, and XGBoost, each tuned with a small grid

**Evaluation.** Each case contributes 5 rows, one per detection delay, so cross-validation is **grouped by case**. A random split would put windows from the same case in both train and test. Scores are reported as macro F1, mean ± std (min/max) across folds.

## EDA highlights

From [notebook.ipynb](notebook.ipynb) sections 2–3 (section 6.5 for the model's own importance):

- **Fault signatures** ([§2.5](notebook.ipynb)): latency shifts clearly (z above 3) in at least 92% of rows for *every* fault, so it says "something is wrong" but not what. CPU and memory carry the type: a clear CPU shift appears in 96–100% of `cpu`, `mem`, `disk` and `socket` rows but only 45% of `delay` rows. The medians leave two pairs that look alike, `mem` vs. `disk` and `delay` vs. `loss`.
- **The disk telemetry gap** ([§2.3](notebook.ipynb)): only 7 of 11 services export disk I/O in OB and SS, against 67 of 68 in TT. For disk faults, disk I/O clearly shifts in 27% of OB cases and 7% of SS cases, against 100% of TT cases, while the memory shift is large everywhere (median `mem_max` 4.7–8.4). In OB and SS a disk fault looks like a memory fault, which is where the model's 18 disk → mem errors come from.
- **Why scaling mattered** ([§2.7](notebook.ipynb)): on the fixture case, a plain z-score reaches 1000 for a latency column whose baseline is near 0, and near-constant columns crowd the top of the ranking. With the scale floor and log compression the same column scores 6.8, and `recommendationservice_mem` drops from rank 5 to rank 36. In the go/no-go spike, this change alone took logistic regression on OB from 0.54 to 0.85 macro F1 ([docs/decisions.md](docs/decisions.md)); the notebook doesn't re-run that ablation.
- **Feature importance** ([§3](notebook.ipynb), [§6.5](notebook.ipynb)): memory features carry the most mutual information with the label (`top_mem` 1.06 and `mem_max` 1.04 nats, out of a maximum of 1.79). The six `error` features carry almost none (at most 0.04) and have exactly 0 permutation importance, because only a few services export errors at all.
- **The systems differ in size** ([§2.3](notebook.ipynb), [§3.1](notebook.ipynb)): 11 services in OB and SS (after merging each service with its `-db`) and 68 in TT. Features that add up over services grow with it: the median `cpu_abssum` is about 10 in OB and SS and 34 in TT. This sets up the [unseen-system gap](#generalizing-to-an-unseen-system).

## Results

**The fault-type classifier scores macro F1 0.80 on systems it was trained on (grouped 5-fold CV across all three systems), and about 0.57 on a system it has never seen (leave-one-system-out; see [Generalizing to an unseen system](#generalizing-to-an-unseen-system)).** Grouped CV is the headline; every number in this section is grouped 5-fold CV by `case_id` (fold seed 42) unless it says otherwise.

| Model | Tuning | Macro F1 (mean ± std across 5 folds) | Min / max across folds |
|---|---|---|---|
| Dummy (most frequent class) | none | 0.030 ± 0.002 | 0.028 / 0.033 |
| Rule baseline (biggest shift wins) | none | 0.277 ± 0.038 | 0.212 / 0.318 |
| Logistic regression | nested | 0.724 ± 0.024 | 0.691 / 0.754 |
| Decision tree | nested | 0.676 ± 0.049 | 0.600 / 0.741 |
| Random forest | nested | 0.800 ± 0.030 | 0.778 / 0.859 |
| XGBoost | nested | 0.772 ± 0.018 | 0.757 / 0.803 |
| **Final model** (random forest, `train.py`) | fixed hyperparameters | **0.799 ± 0.028** | **0.775 / 0.850** |

**Which numbers are which:** the four tuned rows are **nested** grouped CV from [notebook §5](notebook.ipynb) (each outer fold runs its own grouped grid search on its training cases only). The final-model row is the shipped model with fixed hyperparameters, as scored by `train.py` and written to [`web/results.json`](web/results.json).

**Final model choice.** The rule, fixed before the results: take the highest nested mean, but prefer a simpler model if it's within one fold std of the best ([docs/decisions.md](docs/decisions.md)).
- The random forest has the highest nested mean (0.800 ± 0.030). XGBoost (0.772) is the only other model within one std, and it's the more complex of the two. Logistic regression and the tree are more than one std behind.
- Its hyperparameters (`n_estimators=200`, `max_depth=None`, `min_samples_leaf=2`, `random_state=42`) come from whole-table tuning on the same grouped folds. Because the same folds choose and score them, the nested 0.800 ± 0.030 is the unbiased estimate; the two agree to 0.001.
- **Tuning barely matters:** no model moved by more than 0.013 between its library defaults and its nested tuned score.
- **The second decimal is uncertain.** Re-running the final model with fold seeds 0–4 gives means from 0.787 to 0.823 (mean 0.803). Differences between models under about 0.03 aren't meaningful.

**Also measured on the final model** (pooled out-of-fold predictions, all 1,350 rows; [notebook §6](notebook.ipynb)):

- **One-vs-rest ROC AUC:** 0.974 macro (per fold 0.975 ± 0.005). Every class is above 0.95; `loss` is lowest (0.951).
- **Per class:** `mem` and `disk` are easiest (F1 0.916 and 0.919; disk precision 0.961). `cpu` 0.791, `socket` 0.755, `delay` 0.749 and `loss` 0.721.
- **Main confusions:** delay ↔ loss (56 delay rows predicted as loss, 54 the other way), because both show up mainly as latency and `error` is rarely exported. Then cpu ↔ socket (40 and 47), because a socket fault also raises CPU on the same service. Then disk → mem (18), the telemetry gap above. The model rarely mixes a resource fault (cpu, mem, disk, socket) with a network fault (delay, loss): 18 of 1,350 rows.
- **By detection delay:** macro F1 stays flat as the alert fires later, so the model doesn't depend on catching the fault at the moment of injection.

  | Detection delay | 0 s | 30 s | 60 s | 90 s | 120 s |
  |---|---|---|---|---|---|
  | Macro F1 (mean ± std across folds) | 0.787 ± 0.049 | 0.797 ± 0.051 | 0.805 ± 0.053 | 0.797 ± 0.023 | 0.806 ± 0.034 |

- **On-call runbook:** a depth-3 decision tree scores 0.570 ± 0.067. It's much weaker than the forest, but readable as three questions; it's on the results page and in [notebook §6.6](notebook.ipynb).

### Fault type vs. suspect service

**These are two separate answers.** The fault type comes from the trained classifier. The suspect service comes from a rule: the service with the largest total shift across metric kinds, with no ML involved.

| Answer | Comes from | Metric | Value |
|---|---|---|---|
| Fault type | Trained classifier (random forest) | Macro F1, grouped 5-fold CV across 3 systems (mean ± std) | 0.799 ± 0.028 |
| Suspect service | Rule: most anomalous service, not the model | Hit rate: share of the 1,350 alert windows where it names the injected service | 0.900 |

**The rule's hit rate per system:** OB 0.880, SS 0.973, TT 0.847. The rule isn't trained, so it has no folds and no seen/unseen gap. The API reads both metrics from `web/results.json` and returns them as `fault.cv_macro_f1` and `suspect_service.hit_rate`.

### Generalizing to an unseen system

The headline mixes all three systems: every test fold has other cases from the same systems in its training data. A team running FaultTrack on its own cluster gives it a system it has never seen. **Leave-one-system-out (LOSO)** measures that: train on two systems (900 rows) and test on all 450 rows of the third, once per system ([notebook §8](notebook.ipynb)).

| Model | LOSO: OB held out | LOSO: SS held out | LOSO: TT held out | LOSO mean ± spread across 3 systems | Grouped CV (headline, mean ± std across 5 folds) |
|---|---|---|---|---|---|
| **Random forest (final model)** | 0.655 | 0.643 | 0.407 | **0.568 ± 0.114** | 0.799 ± 0.028 |
| Logistic regression (untuned) | 0.583 | 0.606 | 0.303 | 0.497 ± 0.138 | 0.724 ± 0.027 |
| Rule baseline | 0.199 | 0.279 | 0.288 | 0.255 ± 0.040 | 0.277 ± 0.038 |

**About the ±:** a LOSO mean is over only three numbers, one per held-out system, and the ± is their population std (`ddof=0`, the same convention as the fold std). It shows how far apart the three systems are, not a confidence interval.

**Where the gap comes from.** OB and SS (11 services each) lose about 0.11 when held out (their out-of-fold rows score 0.769 and 0.759 under grouped CV). TT loses 0.489: it's the easiest system once seen (0.896) and the hardest unseen (0.407). Two causes:

- **System size.** TT has 68 services against 11. The `<kind>_abssum` and `<kind>_n_gt3` features add up over services, so a tree trained on OB and SS splits them at values that fit 11 services. On TT that shows up mainly as cpu → socket errors (54 of 75 CPU rows).
- **Domain shift.** The same fault shows up in different metrics. In TT a memory fault also shifts disk I/O (median `diskio_max` 6.91, against 0.07 and 0.32 in OB and SS), so a model that learned "memory plus disk I/O means disk" calls all 75 TT memory rows `disk`. Delay ↔ loss is confused on every held-out system (190 rows in all, against 110 under grouped CV).

**Two attempts to close the gap,** both on the committed table and neither adopted ([docs/decisions.md](docs/decisions.md)):

| Final model | LOSO mean ± spread across 3 systems | Grouped CV, mean ± std across 5 folds (fold seed 42) |
|---|---|---|
| Headline features | 0.568 ± 0.114 | 0.799 ± 0.028 |
| `--norm-counts` (sums and counts divided by the number of services) | 0.600 ± 0.095 | 0.775 ± 0.023 |
| Relative features (each feature standardized within its own system, no labels) | 0.571 ± 0.120 | 0.799 ± 0.028 |

- **`--norm-counts`** raises LOSO at every forest seed (0–4) by 0.013 to 0.039, on average from 0.572 to 0.596, and costs 0.005 on grouped CV on average over fold seeds 0–4. It cuts TT's cpu → socket errors from 54 to 11 rows but doesn't touch mem → disk (74 of 75). The gain is small and leaves LOSO about 0.20 below grouped CV.
- **Relative features** don't help the final model (0.571 against 0.568). They help logistic regression (0.497 → 0.533), but not up to the forest.

**What still generalizes:** on an unseen system the model mixes a resource fault (cpu, mem, disk, socket) with a network fault (delay, loss) in only 31 of 1,350 rows, and it stays well above the rule on every held-out system. What it loses is which fault *within* each group.

**Limitation.** The headline, macro F1 0.799 ± 0.028, is measured on the same three systems the model was trained on. On a system it has never seen, the score is lower: 0.568 ± 0.114 across the three held-out systems, and 0.407 on Train Ticket, the system least like the others. On a new cluster, trust the group the label falls in (resource: cpu, mem, disk, socket; or network: delay, loss) more than the exact fault, and retrain on labeled incidents from that cluster before relying on it. The suspect service comes from an untrained rule, so this gap doesn't apply to it (hit rate 0.847–0.973 per system).

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

[`tests/fixtures/README.md`](tests/fixtures/README.md) names the source case and its true fault, so you can check the prediction. To call the live service instead, replace `localhost:8080` with `https://faulttrack.jennasrunbooks.com`.

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

FaultTrack runs on [Fly.io](https://fly.io) as one `shared-cpu-1x` machine with 512 MB in `iad`. **The machine stops when idle** and starts again on the next request, so the first request after a quiet period takes about 10 seconds.

**Configuration:** [`fly.toml`](fly.toml), committed. Fly builds the repo's `Dockerfile` on its remote builder. The commands, run from the repo root with the `fly` CLI logged in:

```bash
fly apps create faulttrack                      # once; already done for this repo
fly deploy --ha=false                           # build and deploy one machine (or: make deploy)
fly certs add faulttrack.jennasrunbooks.com     # custom domain: prints the DNS record to add
# DNS: CNAME record, host "faulttrack", value "faulttrack.fly.dev."
fly certs check faulttrack.jennasrunbooks.com   # wait until the certificate is issued
fly status                                      # one machine, region iad
```

`--ha=false` keeps it to a single machine; on a first deploy Fly would otherwise create two. Redeploy with `fly deploy --ha=false` after any change to the image (`src/`, `predict.py`, `models/`, `web/`, `Dockerfile`, `uv.lock`).

**Screenshots** of the deployed service in use: [docs/images/](docs/images/README.md).

## Repository layout

```text
README.md  REVIEWING.md  product-spec.md  AGENTS.md  LICENSE
notebook.ipynb  train.py  predict.py          course deliverables
Dockerfile  .dockerignore  fly.toml  Makefile
pyproject.toml  uv.lock  .python-version      dependencies (uv)
src/faulttrack/        shared data loading, features, evaluation and models (used by training and serving)
data/                  raw-data instructions; data/features/ holds the committed feature table
models/                trained model (model.bin) + feature spec
web/                   results page served at /, and results.json
tests/                 parity, API, docs and unit tests; tests/fixtures/ holds one small real case
spike/                 go/no-go scripts (exploratory, not graded); also the download script for `make data`
docs/                  architecture, decisions, permissions, agent team, images/ (screenshots),
                       proposed-workflows/ (CI changes for a human to apply)
.devcontainer/  .github/
```

More: [docs/architecture.md](docs/architecture.md) covers how the pieces fit, and [product-spec.md](product-spec.md) covers scope and plan.

## Rubric map

| Criterion | Where |
|---|---|
| Problem description | [Problem](#problem), [How it's used](#how-its-used) |
| EDA | [notebook.ipynb](notebook.ipynb) §2–3, [EDA highlights](#eda-highlights) |
| Model training | [notebook.ipynb](notebook.ipynb) §4–6 and §8, [Results](#results) |
| Exporting notebook to script | [train.py](train.py) |
| Reproducibility | Committed [feature table](data/features/), [Setup](#setup), [Run it](#run-it) |
| Model deployment | [predict.py](predict.py) (FastAPI) |
| Dependency and environment management | [pyproject.toml](pyproject.toml), [uv.lock](uv.lock), [Setup](#setup) |
| Containerization | [Dockerfile](Dockerfile), [Docker](#docker) |
| Cloud deployment | [fly.toml](fly.toml), [Deploy](#deploy), live URL, [screenshots](docs/images/README.md) |

## License and credit

- **Code:** [Apache-2.0](LICENSE).
- **Data:** RCAEval by Pham et al., [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), [doi:10.5281/zenodo.14590730](https://doi.org/10.5281/zenodo.14590730). The committed feature table is derived from it; see [ATTRIBUTION.md](data/features/ATTRIBUTION.md).
