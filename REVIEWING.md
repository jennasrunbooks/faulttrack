# Reviewing FaultTrack for ML Zoomcamp 2026

This guide is for peer reviewers of the ML Zoomcamp 2026 midterm project. It maps each of the **9 rubric criteria (16 points)** to where the evidence lives and how to check it.

**You don't need** a cloud account, Docker (unless you're checking containerization), or the 4 GB raw dataset. The feature table the notebook and `train.py` use is committed to the repo.

**Reviewing the submitted version.** The commit submitted on 16 November 2026 is tagged `midterm-submission` and published as a [release](https://github.com/jennasrunbooks/faulttrack/releases/tag/midterm-submission). To browse it, open [the tag](https://github.com/jennasrunbooks/faulttrack/tree/midterm-submission). To run it, clone the repo and check out the tag:

```bash
git clone https://github.com/jennasrunbooks/faulttrack.git
cd faulttrack
git checkout midterm-submission
```

## Start here (2 min)

**What it is:**
- **Input:** the Prometheus metrics around an alert in a microservice system.
- **Output:** the likely fault type (`cpu`, `mem`, `disk`, `socket`, `delay` or `loss`) and the service it probably started in.
- **Training data:** RCAEval RE2 (CC BY 4.0), 270 fault-injection cases across 3 systems.
- **Serving:** FastAPI, in Docker, deployed to Fly.io.

**Three ways to see it:**

| Way | Needs | Time |
|---|---|---|
| Read the README and the notebook on GitHub, then try the live API at https://faulttrack.jennasrunbooks.com/docs | A browser | about 10 min |
| Clone and run the notebook, `train.py` and `predict.py` locally | Python 3.12 + [uv](https://docs.astral.sh/uv/getting-started/installation/) (uv installs Python 3.12 itself if needed) | about 20 min |
| Build and run the Docker image | Docker | about 10 min extra |

**Cold start:** the live API scales to zero, so the first request after a quiet period takes about 10 seconds.

## Review paths

| Path | Time | Requirements | What to do |
|---|---|---|---|
| **No-install tour** | ~10 min | Browser | Follow the numbered steps below |
| **Local** | ~20 min | uv | See "Local path" below |
| **Docker** | ~10 min | Docker | See "Docker path" below |
| **Full rebuild from raw data** (optional) | 45+ min, ~4 GB download | uv, `curl`, `unzip` | `make data && make features`, then the local path. Not needed for grading |

**No-install tour:**

1. **Problem and results:** read [README.md](README.md), especially "Problem", "Results" and "Generalizing to an unseen system".
2. **Notebook:** open [notebook.ipynb](notebook.ipynb) on GitHub. It's committed with outputs, so it renders without running.
3. **Live API:** open https://faulttrack.jennasrunbooks.com/docs and use **Try it out** on `POST /predict/case`. Upload [tests/fixtures/simple_metrics.csv](tests/fixtures/simple_metrics.csv) (download it first), and set `detection_time` to the number in [tests/fixtures/inject_time.txt](tests/fixtures/inject_time.txt). [tests/fixtures/README.md](tests/fixtures/README.md) names the case's true fault, so you can check the answer.
4. **Results page:** open https://faulttrack.jennasrunbooks.com/ for the results summary, confusion matrix and decision tree.

**Local path:**

```bash
git clone https://github.com/jennasrunbooks/faulttrack.git
cd faulttrack
uv sync                                  # creates .venv from uv.lock
uv run jupyter lab notebook.ipynb        # optional: Restart & Run All
uv run python train.py                   # trains, writes models/ and web/results.json
uv run uvicorn predict:app --port 8080   # then open http://localhost:8080/docs
uv run pytest                            # parity + API tests
```

**Calling the API from a second terminal:**

```bash
curl -s localhost:8080/healthz
curl -s -F file=@tests/fixtures/simple_metrics.csv \
     -F detection_time="$(cat tests/fixtures/inject_time.txt)" \
     localhost:8080/predict/case | uv run python -m json.tool
```

**Docker path:**

```bash
docker build -t faulttrack .
docker run --rm -p 8080:8080 faulttrack
# then the same curl commands as above
```

## Criterion by criterion

Points are the rubric maximums.

### 1. Problem description (2)

- **Asks:** the problem is described, along with how the solution will be used.
- **Evidence:** [README.md](README.md) "Problem" and "How it's used" sections; [product-spec.md](product-spec.md) Section 1.
- **Verify:** the README says who uses it (on-call SREs), when (as soon as an alert fires), and what they get (a fault type and a suspect service before they open a dashboard).

### 2. EDA (2)

- **Asks:** extensive EDA: ranges, missing values, analysis of the target, and feature importance.
- **Evidence:** [notebook.ipynb](notebook.ipynb) sections 2–3:
  - class balance per system
  - metric ranges and missing kinds per system (for example, the disk telemetry gap)
  - per-fault signatures (2.5) and one raw case plotted around the injection time (2.6)
  - the effect of the scale floor and log compression (2.7)
  - mutual information and correlation (3)
- **Verify:** look for the target distribution, a missing-values summary, and a feature importance table or plot.

### 3. Model training (3)

- **Asks:** multiple models, both linear and tree-based, with their parameters tuned.
- **Evidence:** [notebook.ipynb](notebook.ipynb) sections 4–6, and README "Results":
  - **baselines:** dummy and rule-based
  - **models:** logistic regression, decision tree, random forest, XGBoost
  - **tuning:** each model's grid search is scored with nested grouped 5-fold CV (section 5)
  - **final choice:** the selection rule and the final model's evaluation (section 6)
- **Verify:** confirm the CV uses `GroupKFold` by `case_id`. There are 5 windows per case, so this prevents leakage. Confirm results are shown as mean ± std across folds, and that the final choice is justified.

### 4. Exporting notebook to script (1)

- **Asks:** the training logic is exported to a separate script.
- **Evidence:** [train.py](train.py).
- **Verify:** `uv run python train.py` finishes and writes `models/model.bin`, `models/feature_spec.json` and `web/results.json`.

### 5. Reproducibility (1)

- **Asks:** the notebook and the training script can be re-executed without errors, and the data is accessible.
- **Evidence:**
  - [data/features/re2_features.parquet](data/features/) is committed, with [ATTRIBUTION.md](data/features/ATTRIBUTION.md).
  - Raw-data instructions are in [data/README.md](data/README.md) (Zenodo link, md5 checksums, `make data`).
  - Random seeds are fixed.
- **Verify:** run the local path. The grouped 5-fold CV macro F1 that `train.py` prints should match the "Final model" row of the README's "Results" table.

### 6. Model deployment (1)

- **Asks:** the model is deployed with Flask, FastAPI, BentoML or similar.
- **Evidence:** [predict.py](predict.py) (FastAPI).
- **Verify:** start it locally and open `/docs`, or use the live URL.

### 7. Dependency and environment management (2)

- **Asks:** a dependency file, a virtual environment, and a README explaining how to install the dependencies and activate the environment.
- **Evidence:**
  - [pyproject.toml](pyproject.toml), [uv.lock](uv.lock) (exact pinned versions) and [.python-version](.python-version)
  - README "Setup"
- **Verify:** `uv sync` creates `.venv` with the locked versions. The README explains installing uv, running `uv sync`, and either activating the environment (`source .venv/bin/activate`) or running commands through `uv run`.

### 8. Containerization (2)

- **Asks:** a Dockerfile, and a README describing how to build and run it.
- **Evidence:** [Dockerfile](Dockerfile); README "Docker".
- **Verify:** `docker build` and `docker run` (Docker path above), then `curl localhost:8080/healthz`.

### 9. Cloud deployment (2)

- **Asks:** code for deploying to the cloud, plus a URL for testing, or a video or screenshot of testing it.
- **Evidence:**
  - [fly.toml](fly.toml)
  - README "Deploy", with the exact `fly` commands
  - live URL https://faulttrack.jennasrunbooks.com
  - screenshots in [docs/images/](docs/images/)
- **Verify:** open `/docs` on the live URL and run one prediction. If the URL is down when you review, the screenshots show the same calls.

## Numbers, measured not guessed

These are filled in from a fresh-clone run on Nov 15 (plan line 13) and match `web/results.json`.

| What | Command | Result |
|---|---|---|
| Grouped 5-fold CV, final model, 3 systems (macro F1) | `uv run python train.py` | _mean ± std (min/max)_ |
| Dummy baseline (macro F1) | notebook §4 | _value_ |
| Leave-one-system-out (macro F1, mean over held-out systems) | notebook §6 | _value_ |
| Suspect-service rule hit rate (a rule, not the model) | notebook §6 | _value_ |
| Tests | `uv run pytest` | _N passed_ |
| Fresh clone to running API | local path above | _minutes_ |

## Known limits

- **Unseen systems.** Accuracy on a system the model wasn't trained on is much lower than the headline. The README's "Generalizing to an unseen system" section shows this, with the analysis behind it. It's a real finding about domain shift, not a hidden flaw.
- **Disk faults** sometimes have no direct disk telemetry in the data, so they're detected through memory and CPU side effects.
- **The suspect service** comes from a simple rule (the most anomalous service), not the trained model. The API response labels it `"source": "rule"` and reports its own hit rate next to the classifier's macro F1.

## How this was built

FaultTrack was built by its author working with AI coding agents. The agents worked on branches and opened PRs; the author reviewed and merged every one. The process and its guardrails are in [docs/agent-team.md](docs/agent-team.md) and [docs/permissions.md](docs/permissions.md), and every PR is visible in the repo history.