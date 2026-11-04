# Agent team

FaultTrack is built by one human (Jenna) and a small team of AI agents. Each agent has a fixed role, its own workspace, and only the access that role needs.

- **[`AGENTS.md`](../AGENTS.md)** holds the rules every implementer follows.
- **[`product-spec.md`](../product-spec.md)** holds what to build and when.
- **This file** covers who does what, and how a plan line becomes a merged PR.

**What the course allows.** ML Zoomcamp doesn't forbid AI assistance, but the project is graded as Jenna's work. She reviews every PR, and she must be able to explain every line: in the README, to peer reviewers, and in the learning-in-public posts. Agents speed up the typing. The decisions stay with her, and so does the understanding of the results.

## Roles

| Role | Who | Does | Can't |
|---|---|---|---|
| **Product owner and reviewer** | Jenna | Owns the spec and the plan's Status column. Sets priorities, reviews and merges PRs. Runs everything outside the devcontainer: Docker builds, `fly deploy`, DNS and certificates, branch protection, `.github/` files, Dependabot merges, the `midterm-submission` tag, and the submission on the course platform | Skip CI or review on her own PRs |
| **Planner and second reviewer** | Claude in the claude.ai project "Zoomcamp \| ML" | Drafts and revises the spec. Turns plan lines into GitHub issues with acceptance criteria, and gives Jenna the `gh` commands to create them. Writes the `scripts/gh/` scripts (labels, milestone, issues, repo variables and secrets). Writes helper scripts like the go/no-go spike. Reviews PRs against the spec and the rubric. Checks reported metrics against the evaluation rules. Keeps the plan on schedule: flags slips early and says what to cut | Push code, approve or merge PRs, deploy |
| **Implementers** | Claude Code in the devcontainer, as machine user `jksprattler-agent` | One issue on one branch, ending in a PR with test evidence. Fixes review comments on the same branch | Merge, edit `.github/`, edit the spec, read `.env`, use Docker, deploy, touch tags |
| **Gatekeepers** | Branch protection on `main`, the required `ci-ok` check, the fine-grained token scope, and the devcontainer (no Docker, no credentials) | Enforce the rules whatever an agent decides | Be bypassed by an agent (blocks are reported, not worked around) |

## Workstreams

The plan lines in spec Section 10 group into four workstreams. The *Owner* column says who does the work, not who merges: Jenna merges everything.

| Workstream | Plan lines | Owner | Main files |
|---|---|---|---|
| **Data and features** | 3 (scaffold), 4 (shared library) | Implementer. Jenna does the `.github/` and branch-protection parts of line 3 | `pyproject.toml`, `Makefile`, `src/faulttrack/`, `data/`, `tests/` |
| **Modelling** | 5, 6 (notebook), 7 (generalization), 8 (`train.py`) | Implementer drafts; Jenna reviews every result | `notebook.ipynb`, `train.py`, `models/`, `web/results.json`, `docs/decisions.md` |
| **Serving** | 9 (`predict.py` + tests), 10 (containerization) | Implementer writes the code; Jenna builds and runs the image on the Mac | `predict.py`, `web/`, `tests/`, `Dockerfile` |
| **Delivery** | 11 (deploy), 12 (README), 13 (reproducibility check), 14 (submit) | Jenna, with an implementer drafting the README and `fly.toml` | `README.md`, `fly.toml`, `docs/images/` |

### What can run in parallel

Most of the plan is a chain: the features feed the notebook, the notebook picks the model, and the model feeds the API. Parallel work is possible only at the points below. Outside them, run one implementer at a time.

| When | Parallel tracks | Why it's safe |
|---|---|---|
| After line 4 merges (Nov 5) | **A:** notebook EDA (line 5) · **B:** `predict.py` skeleton with Pydantic models, `/healthz` and a stub model, plus API tests (start of line 9) | A only touches `notebook.ipynb`; B only touches `predict.py` and `tests/test_api.py`. Both read the merged `features.py` and don't change it |
| After line 6 picks the model (Nov 9) | **A:** generalization analysis (line 7) · **B:** `train.py` (line 8) | A adds notebook sections; B writes `train.py` and `models/`. Only B regenerates `models/` |
| After line 8 (Nov 10) | **A:** finish `predict.py` with the real model (line 9) · **B:** README draft (line 12) | Different files; the README draft links to results that already exist |

## How a plan line becomes a merged PR

1. **Spec.** Every piece of work starts as a line in spec Section 10, with the sections it depends on.
2. **Issue.** The planner turns the line into a GitHub issue, using the template below. Large lines (such as 6) can be split into two or three issues. Each issue title starts with its plan line (for example `ws-4: shared library and feature table`), and the issue goes in the "Midterm submission" milestone. Labels come from `scripts/gh/labels.sh`, plus GitHub's defaults for issue type:
   - **one area:** `data`, `features`, `notebook`, `train`, `api`, `docker`, `deploy` or `ci` (`documentation` covers docs work)
   - **ML labels where they apply:** `evaluation`, `experiment`, `metrics-change`
   - **routing:** `needs-human`, `needs-decision`, `blocked`, `do-not-merge` (makes `ci-ok` fail, which blocks the merge), `stretch`, `full-ci`
   - **colors:** areas are pastel, ML labels mid-tone, routing labels bright, so the labels that need action stand out
   - **Dependabot:** `dependencies`, `security`
3. **Assignment.** Jenna assigns the issue and starts an implementer in a fresh clone, on the branch `ws-<n>-<short-desc>`.
4. **Implementation.** The implementer reads `AGENTS.md`, the issue and the spec sections it names. It implements, then runs `make lint` and `make test`, plus the notebook run if the notebook changed. It stops and asks whenever a decision belongs to Jenna (the list is in `AGENTS.md` under "When to stop and ask").
5. **PR.** The implementer opens a PR that says `Closes #<issue>` and lists:
   - the plan line and rubric criterion it advances
   - decisions made and alternatives rejected
   - test evidence: commands and results, plus metrics as mean ± std where they changed
   - anything it couldn't verify, such as Docker or the live URL
   - proposed spec changes, if any
6. **Review.** CI runs (path-aware, with `ci-ok` as the gate). Jenna and the planner review: the planner checks it against the spec and the evaluation rules, and Jenna checks she can explain it. The implementer pushes fixes to the same branch.
7. **Merge.** Jenna squash-merges, marks the plan line `Done (date)` in the spec if the line is complete, and records any decision in `docs/decisions.md`.
8. **Next issue.** The implementer starts its next issue in a new session, so stale context doesn't carry over.

### Issue template

```markdown
**Plan line:** <n>: <workstream> (target <date>)
**Rubric:** <criterion> (<points>), or "none"
**Spec sections:** <links>

## Scope
- ...

## Out of scope
- ...

## Acceptance criteria
- [ ] `make lint` and `make test` pass
- [ ] ...  (concrete, checkable: a file exists, a test asserts X, a metric is reported as mean ± std)

## Hand-offs to Jenna
- ...  (anything needing Docker, Fly.io, DNS, .github/, or a decision)
```

## Rules for parallel agents

- **Run agents in parallel where the plan allows** (see "What can run in parallel"), each in **its own clone** of the repo, on its own branch, in its own devcontainer.
  - **Don't use git worktrees.** The `dev-agent` scripts start one devcontainer per clone directory and mount only that folder, so a worktree, which shares one `.git` with its parent clone, doesn't fit.
  - **Each clone has its own `.venv`,** so `uv sync` in one agent's container can't break another's.
  - **Clones on the host:** clones are long-lived, one per track. Each new issue is a new branch in that track's clone.

    | Clone | Track | Plan lines | Owns |
    |---|---|---|---|
    | `faulttrack` | Maintainer (Jenna, host only) | 2, 3 (`.github/` part), 10 (image build), 11, 13, 14 | `.github/`, settings, Docker builds, `fly deploy`, DNS, the tag |
    | `faulttrack-pipeline` | Pipeline: scaffold, library, training script | 3, 4, 8 | `pyproject.toml`, `Makefile`, `src/faulttrack/`, `data/`, `train.py`, `models/` |
    | `faulttrack-nb` | Notebook | 5, 6, 7 | `notebook.ipynb`, `docs/decisions.md` |
    | `faulttrack-api` | Serving | 9, 10 (code) | `predict.py`, `web/`, `tests/test_api.py`, `Dockerfile`, `fly.toml` |
    | `faulttrack-docs` | Reviewer-facing docs | 12 | `README.md`, `REVIEWING.md`, `docs/images/` |

    The parallel windows above map onto these clones. On Nov 5, `-nb` and `-api` run together. On Nov 9, `-nb` (line 7) and `-pipeline` (line 8) run together. On Nov 10, `-api` and `-docs` run together. Each clone is created once with `git clone … ~/src/faulttrack-<track>`, and each issue is a new `git switch -c ws-<n>-<desc>` from an up-to-date `main`.
- **One issue per branch,** one branch per agent. Keep branches current by merging `origin/main`, never by rebasing (see `AGENTS.md`).
- **One agent at a time on `notebook.ipynb`.** Notebook merge conflicts are painful to resolve by hand. If two tracks need the notebook, they take turns.
- **Generated files have one owner per change.** Only the PR that changes `features.py` regenerates `data/features/re2_features.parquet`. Only the PR that changes training regenerates `models/` and `web/results.json`. Other PRs never commit changes to these files.
- **Human-only files stay with humans:**
  - `.github/`, `product-spec.md`, tags, and repo settings
  - Agents propose `.github/` changes in `docs/proposed-workflows/` and spec changes in the PR description.
- **Report blocks; don't work around them.** A rejected push, a missing tool or a needed credential goes into the PR or issue as a question.
- **Shared decisions go into the repo** (`docs/decisions.md`), not into an agent's session. The next agent starts with no memory of the last one.

## Access

| Who | GitHub | Fly.io / DNS | Data | Merge |
|---|---|---|---|---|
| Jenna | Admin on `jennasrunbooks/faulttrack` | Full (from the Mac host) | Downloads `data/raw/` | Yes, after CI and review |
| Planner | None directly; works from the spec and from PR content Jenna shares or links | None | None | No |
| Implementers | Fine-grained PAT for `jksprattler-agent`, this repo only: **Contents** read/write, **Pull requests** read/write, **Issues** read/write (to comment on their issues), **Metadata** read. No **Workflows**, no **Administration** | None | `make data` inside the devcontainer if a task needs it; tests and the notebook use committed data only | No |

**How the token reaches the agent:**
- **Source:** `GH_TOKEN` and `GIT_AUTHOR_*` come into the devcontainer from the host environment (`${localEnv:…}`).
- **Mount:** only the `faulttrack/` folder is mounted.
- **Container hardening:** the container runs rootless with `--cap-drop=ALL` and `--security-opt=no-new-privileges`.

## Evidence

The course rewards a repo a reviewer can follow, and this setup leaves a trail:

- **Tasks:** GitHub issues are the task tracker. Each one links to its plan line.
- **Changes:** agent PRs record each change, with its decisions and test evidence.
- **Checks:** CI runs on every PR (lint, tests, image build) and gates merges through `ci-ok`.
- **Decisions:** `docs/decisions.md` records the go/no-go and later choices, with dates.
- **Progress:** the spec's plan table shows status against the Nov 16 deadline.