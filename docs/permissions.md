# Permissions

This page covers who and what can change FaultTrack, and what each control does and doesn't stop. It covers four actors:

- **the coding agents** that build the repo (development time)
- **the CI pipeline**
- **the running service** (runtime)
- **the maintainer's hosting access**

Roles and the PR flow are in [`agent-team.md`](agent-team.md). The rules agents follow are in [`AGENTS.md`](../AGENTS.md).

FaultTrack's attack surface is small. There's no database, no user accounts, no cloud resources beyond one Fly.io app, and no secrets in the service. Most controls therefore protect the **repo**: the code reviewers grade, and the submitted commit.

## Accepted gaps

These are known, deliberate trade-offs for a two-week course project. Each row says how to close the gap if the project lives on.

| Gap | Where | Why it's accepted | How to close it |
|---|---|---|---|
| Agents can push to any branch except `main`, and can open as many PRs as they like | PAT scope (Contents: write) | Branch-per-issue work needs it, and nothing reaches `main` without review and CI | Add a ruleset restricting the agent to `ws-*`, `fix/*`, `chore/*` and `docs/*` branch names |
| An agent could weaken a test to make CI green | Agent PRs | CI can't judge test intent. Human review checks test changes, and `AGENTS.md` forbids skipping or weakening tests | CODEOWNERS on `tests/`, requiring maintainer review |
| No Claude Code guard hooks or deny rules | `.claude/` | The devcontainer and token already remove the dangerous capabilities: no Docker, no deploy credentials, no Workflows permission | Add `.claude/settings.json` deny rules for `fly`, `git push --force`, `git tag` and reads of `.env*` |
| The devcontainer has open outbound network | Devcontainer | `uv sync` needs PyPI, and `make data` needs Zenodo | A network allowlist (PyPI, Zenodo, GitHub) |
| `model.bin` is a pickle, and loading a pickle runs code | `predict.py` startup | The file comes only from this repo through reviewed PRs, and the API never loads user-supplied models | Serialize to a non-executable format, such as XGBoost JSON or skops |
| The public API has no authentication or rate limit | Fly.io app | It's a stateless demo with no data to protect. Scale-to-zero and Fly's machine limits cap the cost | A Fly.io concurrency limit, a request rate limit in the app, or an API key for `/predict*` |
| Issues: write on the agent token | PAT scope | Lets agents comment progress and questions on their own issues | Drop it, and have agents ask questions in PR comments only |

These gaps are reviewed again after the Nov 16 deadline.

## Development-time agent

| Layer | Where it's set | What it stops | What it doesn't stop |
|---|---|---|---|
| **Fine-grained PAT** (`jksprattler-agent`) | GitHub → the machine user's settings: this repo only. Contents RW, Pull requests RW, Issues RW, Metadata R | Access to any other repo. Pushes touching `.github/workflows/` (no Workflows permission). Changing settings, rulesets or environments (no Administration) | Pushing branches, opening PRs, commenting |
| **Branch protection on `main`** | Repo settings | Direct pushes and force pushes to `main`. Merging without the `ci-ok` check and an approving review | What's inside a PR. That's what review is for |
| **Devcontainer** | `.devcontainer/` | Docker (none installed), host files outside `faulttrack/` (only that folder is mounted), privilege escalation (`--cap-drop=ALL`, `--security-opt=no-new-privileges`, rootless) | Outbound network, see Accepted gaps |
| **No deploy credentials** | The maintainer's host only | `fly deploy`, DNS and certificate changes, registry pushes. The tokens aren't in the container | none |
| **`AGENTS.md` hard rules** | Repo | Sets expectations: no tags, no `.github/` edits, no copied code, grouped CV only, no stretch work before submission | Nothing by itself. It's guidance, backed by the layers above and by review |
| **Human review** | Every PR | Changes Jenna can't explain, weakened tests, leaked data or secrets, rubric regressions | Slow mistakes that look plausible. The fresh-clone check (plan line 13) is the backstop |

**How the token gets in.** `GH_TOKEN` and `GIT_AUTHOR_*` come from the host environment through `${localEnv:…}` in `devcontainer.json`. They are never written to a file in the repo.

**Adding to the rules.**
- Any new capability an agent needs goes through an issue. Examples: a new tool, network access, or a permission on the token.
- Jenna makes the change. Agents never widen their own access.
- Proposed `.github/` changes go in [`proposed-workflows/`](proposed-workflows/).

## GitHub repository settings

- **`main`:** pull request required (1 approval), the `ci-ok` status check required, no force pushes, no deletions.
- **Merge method:** squash only.
- **Tags:** `midterm-submission` is created by Jenna on the submitted commit and published as a release. A tag ruleset restricts creating, updating and deleting `midterm-submission` (only Jenna, as repository admin, can bypass it).
- **Security:** Dependabot alerts and security updates are on. Secret scanning and push protection are on (free for public repos).

## CI

**PR CI permissions:** `permissions: contents: read` at the top of `ci.yaml`. PR CI uses no secrets, so a PR from a fork or a Dependabot PR can't exfiltrate anything.

**Actions:** every action is pinned to a full-length commit SHA, and Dependabot keeps the pins current. The repo settings enforce this:
- **Allowed sources:** actions created by GitHub, actions from Marketplace verified creators, actions in `jennasrunbooks` repos, and an explicit allowlist. The allowlist currently has `astral-sh/setup-uv@*`; add other actions there as they're needed.
- **SHA pins:** actions must be pinned to a full-length commit SHA, so a workflow using a tag like `@v4` fails to start.
- **`GITHUB_TOKEN`:** read-only by default, and Actions can't create or approve pull requests.
- **Fork PRs:** workflows need approval for first-time contributors.

**Image builds:** the `docker` job builds the image but never pushes it. There's no registry credential in CI for the graded build.

**Stretch (CD):**
- A deploy workflow gets `FLY_API_TOKEN` as an environment secret.
- The workflow runs on release tags only, behind a required-approval environment.
- That workflow is the only place a deploy token would exist outside the maintainer's host.

## Runtime service

| Control | What it does |
|---|---|
| No secrets | The container needs no API keys, tokens or credentials, so there's nothing to leak |
| No outbound calls | The app never calls another service. It reads the uploaded file and the bundled model only |
| Input validation | Pydantic models validate JSON bodies. `/predict/case` enforces an upload size limit and the required columns, and rejects the request when the window falls outside the file |
| No writes | Uploads stay in memory, and nothing is written to disk or kept between requests |
| Non-root container user | The Dockerfile creates and switches to an unprivileged user |
| Minimal image | The image holds runtime dependencies only (`--no-dev`): no notebook, no test tools, no raw data |

## Hosting (Fly.io and DNS)

- **Fly.io org, app and API token:** Jenna's account, used from the Mac host only.
- **DNS for `jennasrunbooks.com`:** changed by Jenna only. The `faulttrack` record points at the Fly app, and `fly certs add` issues the certificate.
- **Cost guardrails:** one shared-CPU machine, 512 MB, auto-stop, `min_machines_running = 0`.

## Data and license obligations

- **Licenses:** RCAEval is CC BY 4.0. The committed `re2_features.parquet` is a derived work, and `data/features/ATTRIBUTION.md` credits the source and describes the changes. Code is Apache-2.0.
- **What stays out of the repo:** raw data (`data/raw/`) and the zips are never committed. The data contains no personal information. It's synthetic microservice telemetry from fault-injection experiments.