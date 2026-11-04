# Proposed workflows

This folder is where coding agents propose changes to GitHub Actions workflows and other `.github/` files, such as `dependabot.yaml`. A human reviews each proposal and moves it into production.

**Why this exists.** The agent's token (`jksprattler-agent`) has no **Workflows** permission, and `AGENTS.md` doesn't let agents edit `.github/`, so a push that touches `.github/workflows/` is rejected. This keeps agents from changing the checks that gate their own PRs.

## For agents: proposing a change

1. **Write the full file** here, named the way it would be named under `.github/`:

   | Proposal | Production path |
   |---|---|
   | `docs/proposed-workflows/ci.yaml` | `.github/workflows/ci.yaml` |
   | `docs/proposed-workflows/dependabot.yaml` | `.github/dependabot.yaml` |

   To change an existing workflow, copy the current file here and edit the copy. Don't send a diff or a fragment.

2. **Start the file with a comment header:**

   ```yaml
   # PROPOSED: move to .github/workflows/ci.yaml
   # Issue: #<n>   Plan line: <n>
   # Why: <one or two lines>
   # Secrets/permissions needed: <none | list>
   ```

3. **Follow the CI rules** in `product-spec.md` (PR CI design):
   - **Path-aware jobs.** A `changes` job decides which jobs run. Don't put a top-level `paths:` filter on the workflow.
   - **One required check.** `ci-ok` stays the only required check, and every new job is added to its `needs:`.
   - **Least privilege.** Set `permissions:` at the top, usually `contents: read`.
   - **Pinned actions.** Pin third-party actions to a commit SHA, with the version in a trailing comment.
   - **File extension:** `.yaml`, not `.yml`.

4. **Say so in the PR description:** "Needs a human to move `docs/proposed-workflows/<file>` to `<production path>`." Also list any secrets, environments or branch-protection changes it needs.

## For the maintainer: promoting a proposal

1. Review the proposal in the agent's PR.
2. On your own branch, move it (`git mv docs/proposed-workflows/ci.yaml .github/workflows/ci.yaml`) and remove the `PROPOSED` header.
3. Add any secrets or environments it needs in the repo settings, then open a PR.
4. Once CI passes on that PR, merge it.
5. Update branch protection if the required checks changed.

After it's promoted, the proposal file is deleted, so nothing in this folder is ever live.