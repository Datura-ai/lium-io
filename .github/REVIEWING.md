# Reviewing pull requests in `lium-io`

`.github/CODEOWNERS` names who is asked to review each path (primary first, backup second). This page says how fast,
what may merge on one approval, how stacked PRs are handled, and how PRs opened by the Lium loop are treated.
Ticket: [DAH-3022](https://app.notion.com/p/CODEOWNERS-review-SLA-across-repos-3d3b8bfdbde981d6b409f8a90f24d530).

## 1. Response time

- A code owner answers a review request within **1 business day**: approve, request changes, or say when they will look.
- **P0** (`[P0]` in the title or the `P0` label): **same day**.
- Either owner on the CODEOWNERS line may answer; the backup picks it up when the primary is out. The author does not
  ping a third person unless both owners are unavailable.

## 2. What may merge after ONE approval and green required checks

Branch protection requires a pull request and one approving review; with the code-owner rule on, that approval comes
from a code owner of every touched path. For the classes below that single approval is the whole process — the approver may tick "auto-merge" so the PR lands when the
checks pass, and nobody waits for a second opinion:

- **(a) Docs-only** — Markdown/MDX, `docs/**`, `llms*.txt`, comments and docstrings. No executable path changes.
- **(b) Provably dead code** — the PR body shows the identical test run before and after (same passed/skipped counts)
  and a repository-wide search for every removed symbol with zero remaining references.
- **(c) Dependency bumps** — only the manifest and lock file change and CI passes.
- **(d) CI-only workflow changes** — `.github/workflows/**` only, approved by an infra owner (@jam6099 / @pixel29913).

Everything else needs the same single approval, but the approver reads the change and the Verification section, and
the author waits for it — no auto-merge on runtime behaviour.

## 3. Stacked PRs

- The base PR is reviewed and merged first.
- A stacked PR says `Stacked on #<base>` in its first line and is reviewed for its own diff only (GitHub's compare
  against the base branch).
- After the base merges, the stacked PR is rebased onto `main`, checks re-run, and it is reviewed again only if
  the rebase changed it.
- A stacked PR is never merged before its base.

## 4. PRs opened by the Lium loop

The loop's PRs carry the `loop` label and follow exactly the rules above — same response time, same classes, same one
human approval. The loop never approves, never merges, never enables auto-merge on its own PRs. It answers review
comments and bot findings on its PRs itself; a `loop` PR that fails the class-(2) bar is a normal PR.
