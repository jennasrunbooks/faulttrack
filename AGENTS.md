# AGENTS.md — FaultTrack

Guidelines for AI coding agents (Claude Code as `jksprattler-agent`) and humans working in this repo.

- **`product-spec.md`** is the source of truth for *what* to build and *when*.
- **This file** is *how* to work here.
- **If the two disagree,** stop and ask. Don't pick one silently.

## Before your first edit on any issue

This file is loaded automatically at the start of every session. The other docs are not, so before you change any file:

1. **Read the issue, including its comments:** `gh issue view <n> --comments`. Comments can change the scope after the issue was written.
2. **Read [docs/agent-team.md](docs/agent-team.md)** if this is your first issue in this clone. It covers roles, the issue → PR flow, and the rules for parallel agents.
3. **Read the spec sections the issue lists,** plus [product-spec.md §10](product-spec.md#10-plan-to-16-november-2026) for the plan line and its target date.
4. **Read the doc for the area you're touching** (see the table below). For example, read `docs/architecture.md` before changing code in `src/faulttrack/`.
5. **Start from an up-to-date `main`:** `git switch main && git pull`, then `git switch -c ws-<n>-<short-desc>`.

In the PR, say which of these you read. If the issue, this file and the spec disagree, comment on the issue and stop; don't guess.

## What FaultTrack is

FaultTrack is a DataTalksClub ML Zoomcamp 2026 midterm project.

**What it does:**
- **Input:** the per-service Prometheus metrics in a window around an alert.
- **Prediction:** which kind of fault it probably is: `cpu`, `mem`, `disk`, `socket`, `delay` or `loss`.
- **Root cause:** which service the fault most likely started in.

**How it's built:**
- **Data:** RCAEval RE2 (CC BY 4.0, 270 cases across 3 microservice systems).
- **Model:** a tuned tree model trained on service-agnostic shift features.
- **Serving:** FastAPI, in Docker, deployed to Fly.io at `faulttrack.jennasrunbooks.com`.

**Deadline: 16 November 2026 (Mon), 17:00.** The course grades the repo against a 16-point rubric (spec Section 9). Every change should move a rubric line or a plan line (spec Section 10) forward. Nothing else is in scope until after submission.

## Where to read more

This file is the entry point. Before starting a workstream, read the doc for the area you're working in.

| Doc | Read it when |
|---|---|
| [product-spec.md](product-spec.md) | Always. Read the plan line (Section 10) your task names, plus the sections it points to |
| [docs/agent-team.md](docs/agent-team.md) | Before your first issue: roles, how a plan line becomes a merged PR, rules for parallel agents |
| [docs/architecture.md](docs/architecture.md) | Code that crosses module boundaries: data → features → training → serving, artifacts, endpoints |
| [docs/permissions.md](docs/permissions.md) | Anything touching tokens, CI permissions, the container, hosting, or the accepted gaps |
| [REVIEWING.md](REVIEWING.md) | Changing anything a peer reviewer runs or reads; keep its commands and links working |
| [docs/decisions.md](docs/decisions.md) | Before changing evaluation, features or the model choice: what was decided, and why |
| [data/README.md](data/README.md) | Data download, checksums, extraction, the case layout |
| [data/features/ATTRIBUTION.md](data/features/ATTRIBUTION.md) | Touching the committed feature table |
| [README.md](README.md) | Writing anything a reviewer will run; the README is graded |
| [docs/proposed-workflows/](docs/proposed-workflows/) | You need a CI or Dependabot change (you can't edit `.github/`, see Hard rules) |

## Stack

| Area | Stack | Location |
|---|---|---|
| Shared library | Python 3.12, pandas, NumPy; package managed with `uv` (src layout, hatchling build) | `src/faulttrack/` |
| Modelling | scikit-learn, XGBoost; Jupyter for the notebook | `notebook.ipynb`, `train.py` |
| Serving | FastAPI, Pydantic v2, uvicorn | `predict.py`, `web/` |
| Tests | pytest, httpx (FastAPI `TestClient`) | `tests/` |
| Lint / types | Ruff (lint + format), basedpyright (a Pyright fork from PyPI that ships Node as a wheel, so nothing is downloaded at run time; reads `[tool.pyright]`, the same settings Pylance uses in VS Code) | settings in `pyproject.toml` |
| Container | Docker (service image) | `Dockerfile` |
| Hosting | Fly.io (scale-to-zero, 512 MB) | `fly.toml` |
| CI | GitHub Actions (path-aware jobs + `ci-ok` gate), Dependabot | `.github/` (human-edited only) |
| Dev environment | Devcontainer (rootless, `--cap-drop=ALL`, no Docker, no deploy credentials) | `.devcontainer/` |

## Repository map

```text
product-spec.md  AGENTS.md  README.md  REVIEWING.md  LICENSE  Makefile
pyproject.toml  uv.lock  .python-version   # uv only; no requirements.txt, no pip
notebook.ipynb             # course deliverable: prep, EDA, feature importance, model selection + tuning
train.py                   # course deliverable: trains the final model, writes models/ + web/results.json
predict.py                 # course deliverable: FastAPI app (/, /predict, /predict/case, /healthz, /docs)
src/faulttrack/
  data.py                  # case discovery, loading, dataset quirks (spec 3.3)
  features.py              # shared feature code: the ONLY place features are computed
data/raw/                  # gitignored; filled by `make data`
data/features/             # re2_features.parquet + re2_inventory.parquet (committed) + ATTRIBUTION.md
models/                    # model.bin + feature_spec.json (committed, small)
web/                       # static results page served at /
tests/                     # parity, API smoke, unit; tests/fixtures/ holds one small real case
spike/                     # go/no-go scripts; exploratory, not graded; excluded from ruff + typecheck; don't import from here
scripts/gh/                # maintainer-run gh scripts (labels, milestone, issues); agents don't run them
docs/                      # architecture.md, permissions.md, agent-team.md, decisions.md, images/, proposed-workflows/
.devcontainer/  Dockerfile  fly.toml  .github/
```

## Commands

Run everything from the repo root inside the devcontainer. Prefer `make` targets over ad-hoc commands.

| Task | Command |
|---|---|
| Install / sync the environment | `uv sync` |
| Unit tests (fast; run before every commit) | `make test` (`uv run pytest`) |
| Lint + format check | `make lint` (`uv run ruff check .` and `uv run ruff format --check .`) |
| Type check | `make typecheck` (`uv run basedpyright`) |
| Auto-fix lint and formatting | `uv run ruff check --fix .` and `uv run ruff format .` |
| Download raw data (about 4 GB of zips; metrics only are kept) | `make data` |
| Rebuild the committed feature table | `make features` |
| Train the final model | `make train` (`uv run python train.py`) |
| Re-run the notebook top to bottom | `uv run jupyter nbconvert --to notebook --execute --inplace notebook.ipynb` |
| Serve locally | `make serve` (`uv run uvicorn predict:app --port 8080`) |
| Add a dependency | `uv add <pkg>` (runtime) or `uv add --dev <pkg>` (notebook, plots, tests) |

**Adding commands:** if a `make` target doesn't exist yet, add it in the same PR instead of documenting a one-off command.

**What only runs on the host or in CI:**
- **Docker:** `make docker-build` and `make docker-run` need Docker, which the devcontainer deliberately doesn't have. They run on the maintainer's Mac and in CI's `docker` job. Don't try to install Docker.
- **Deploys:** `make deploy` (`fly deploy`) is for the maintainer only.

## Hard rules

These are enforced by the devcontainer (no Docker, no cloud credentials), the agent's fine-grained token (one repo, no Workflows permission), Claude Code hooks in `.claude/`, and branch protection on `main` (the `ci-ok` check is required). Don't try to work around a block. Report it.

1. **Never deploy or change hosting.**
   - No `fly` commands, no DNS changes, no certificate changes.
   - No creating, moving or deleting tags or releases. In particular, never touch the `midterm-submission` tag.
   - You have no Fly.io, Docker registry or cloud credentials, and that is intended. Don't ask for them or look for them in the environment.
2. **Never commit secrets or large or raw data.**
   - Don't read or print `.env` or `.env.*` (`.env.example` is fine).
   - Never commit `data/raw/`, zips, `.venv*`, `.vscode/`, or credentials or tokens of any kind.
   - The only committed data is `data/features/re2_features.parquet` and `data/features/re2_inventory.parquet` (with `ATTRIBUTION.md`) and the small fixture under `tests/fixtures/`.
3. **Don't edit `.github/` or `.claude/`** (workflows or `dependabot.yaml`).
   - The agent token has no Workflows permission, so pushes that touch `.github/workflows/` are rejected.
   - Instead, put the proposed file in `docs/proposed-workflows/<name>.yaml`, and say in the PR that a human needs to move it.
4. **Never commit to `main`.**
   - Work on a branch, open a PR, and wait for review.
   - Never merge your own PR, and never change branch protection, environments or repo settings.
   - **Keep your branch up to date by merging, never by rebasing.** When `main` moves or the PR shows conflicts:
     1. Run `git fetch origin && git merge origin/main`.
     2. Resolve the conflicts.
     3. Run `make lint` and `make test`.
     4. Commit, then `git push` (a normal push).
   - PRs are squash-merged. Never force-push.
5. **Write all code fresh for this repo.**
   - The course gives 0 points for plagiarism.
   - Don't copy code from other projects, other students, prior course cohorts, or the maintainer's projects for other courses.
   - `spike/` is this repo's own exploratory code. Port from it into `src/faulttrack/` and clean it up, but don't import from `spike/` in graded code.
6. **One feature implementation.**
   - Features are computed only in `src/faulttrack/features.py`, and `train.py`, `predict.py` and the notebook all import it.
   - Any change to it must keep the parity test green. The parity test checks that the API's features for the fixture case equal the training row exactly.
   - Don't re-implement feature logic inline in the notebook.
7. **Evaluation must be grouped by case.**
   - There are 5 windows per case, so use `GroupKFold` (or leave-one-system-out) with `groups=case_id`.
   - Never use random row-level splits (`train_test_split`, plain `KFold`, `cross_val_score` without groups). They leak windows of the same case into both train and test.
   - Report macro F1 as **mean ± std (min/max) across folds**.
   - The headline is grouped 5-fold CV across all 3 systems. Leave-one-system-out is the separate "Generalizing to an unseen system" section.
8. **Keep it reproducible for a reviewer on a fresh clone.**
   - Fix random seeds (`random_state=42` everywhere it applies).
   - Pin dependencies through `uv.lock`, and change them only with `uv add` or `uv remove`. Commit `pyproject.toml` and `uv.lock` together.
   - Tests must not need `data/raw/` or the network.
9. **Don't download and run binaries or installers from the internet.** Install tools only through `uv` or the devcontainer definition. If a task needs a tool the container doesn't have, stop and ask.
10. **No stretch work before submission.** Items in spec Section 12 (Helm/kind, Alertmanager webhook, raw-metric features, AWS hosting, logs/traces, CD) wait until after the Nov 16 deadline.

## Conventions

**Python**
- Python 3.12. Ruff settings are in `pyproject.toml` (line length 88; rules E4, E7, E9, F, I, B, UP). Imports go one per line, sorted by Ruff.
- `make typecheck` (`uv run basedpyright`) must show no errors in the files you changed. It reads `[tool.pyright]` with `typeCheckingMode = "standard"`, the same settings Pylance uses, so it matches what the maintainer sees in VS Code.
- Type hints on public functions in `src/faulttrack/`. Docstrings say what a function expects and returns, including units: seconds, and z-scores after log compression.
- Catch narrow exceptions (`OSError`, `ValueError`, `KeyError`), never a bare `except`. When a dataset case can't be read, skip it with a message on stderr. Never fill it silently.
- Use `with open(...)` for files and `pathlib` for paths. No hardcoded absolute paths.

**Data and features**
- Labels come from the case folder name (`<service>_<fault>`). Build features from `simple_metrics.csv` only. Raw `metrics.csv` is stretch work.
- Handle the known quirks in `data.py` (spec 3.3), not in callers: `ts-` and `front-end` naming, the `inf` then forward-fill then 0 sequence, the `time.1` column, and the missing `simple_metrics.csv` in `checkoutservice_cpu/4`.
- Feature names are stable. If you rename or add one, regenerate `re2_features.parquet` and `models/feature_spec.json` in the same PR, and say so in the PR.

**Notebook**
- `notebook.ipynb` is graded. Commit it **with outputs**, after a clean top-to-bottom run (the `nbconvert --execute` command above).
- Sections follow spec Section 9: data prep, EDA, feature importance, model selection and tuning, then the final model.
- Every plot has a title and axis labels. Every result cell has a short markdown sentence saying what it shows.
- Read the committed `data/features/re2_features.parquet`, so reviewers don't need the raw data.

**API**
- Pydantic models validate every request body. Responses match the example in spec Section 6.
- Keep the two answers separate and labeled. `fault` comes from the model (`"source": "model"`, with `cv_macro_f1`). `suspect_service` comes from the most-anomalous-service rule (`"source": "rule"`, with `hit_rate`). Read both metrics from `web/results.json`, never hardcode them. Never describe the suspect service as a model prediction in field descriptions, `/docs`, the README or the results page.
- `predict.py` loads the model once at startup and listens on `$PORT` (default 8080).
- Keep `/docs` (Swagger UI) working; reviewers use it.

**Writing (README, docs, PR text, notebook markdown)**
- Plain, specific language in US English (license, color, generalize). Don't use "robust". State the numbers, the fold spread and the limitation.
- Credit RCAEval (the paper and DOI `10.5281/zenodo.14590730`) wherever the data is described.
- Don't mention the maintainer's other projects in the README.

**Files**
- Use `.yaml`, not `.yml`.
- Name AWS resources `ft-<component>-…` if any are ever added (stretch only).

## Git workflow

1. **One plan line or task per branch and PR.** Start from an up-to-date `main`.
2. **Branch names:**
   - plan work: `ws-<n>-<short-desc>`, where `ws` means workstream and `<n>` is the plan line number in spec Section 10 (e.g. `ws-4-features`)
   - anything else: `fix/<area>-<short-desc>`, `chore/<area>-<short-desc>` or `docs/<short-desc>`
   - areas: `data`, `features`, `notebook`, `train`, `api`, `docker`, `deploy`, `ci`, `docs`
3. **Commit messages** follow Conventional Commits, with the area as the scope. For example: `feat(features): add per-kind abssum and n_gt3 aggregates`.
4. **Before every commit:** `make lint`, `make typecheck` and `make test` pass. Don't commit failing tests, and don't skip them to get green.
   - **QA on high-risk issues (plan lines 4, 6, 7):** before opening the PR, run the `qa-reviewer` subagent (`.claude/agents/qa-reviewer.md`). Fix everything it marks **BLOCK**, re-run it until the verdict is PASS or PASS WITH NOTES, and paste the final report into the PR under `## QA review`. If you think a BLOCK is wrong, say why in the PR and let the maintainer decide; don't work around it.
5. **Push the branch and open a PR** against `main` that:
   - names the plan line it advances (e.g. "Plan line 4: shared library") and the rubric criterion, if any
   - summarizes the change and links the spec section it implements
   - lists how it was tested: the commands and their results, including metrics as mean ± std where the change affects them
   - calls out any change to features, the evaluation setup, dependencies or committed data
   - lists any spec change you think is needed (see below)
   - lists the docs you read before starting (see "Before your first edit on any issue")
   - for plan lines 4, 6 and 7: includes the `qa-reviewer` report under `## QA review`
6. **Merging:** squash merge is the default, and a human merges.

**The spec:** agents don't edit `product-spec.md`. The maintainer owns it and marks the plan's Status column. Propose changes in the PR description.

**Dependabot PRs:** the maintainer reviews and merges them. Don't merge, rebase or modify them.

## When to stop and ask

- The spec is ambiguous, contradicts itself, or contradicts this file.
- A task needs credentials, Docker, a deploy, a `.github/` change, or a new runtime dependency that isn't in spec Section 8.
- A change would move a headline metric by more than about 0.05 macro F1, or would change the evaluation setup. Report it before building on it, and label the PR `metrics-change` with before/after mean ± std.
- The work for a plan line won't fit its target date. Say so early; the spec says what to cut first.
- You've hit the same failure twice. Explain what you tried instead of trying a third variation.

## Instruction files

This file is the only project instructions file. Don't add a `CLAUDE.md`, `.claude/CLAUDE.md` or `CLAUDE.local.md` anywhere in the repo: Claude Code loads `AGENTS.md` only when none of those exist. `.claude/settings.local.json` is personal and gitignored.