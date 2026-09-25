# lium-core

Shared library for the Lium platform, published to PyPI as [`lium-core`](https://pypi.org/project/lium-core/).
Imported from `Datura-ai/lium-core` with its history (DAH-3135); that repository is archived once this lands and PyPI's trusted publisher points here.

- `lium_core.shared_config` — `SharedConfigClient`, `SharedConfig` and the defaults every service agrees on.
  Consumers: the validator and miner in this repository (`neurons/validators`, `neurons/miners`), and the
  lium backend, portal backend and support bot in `Datura-ai/lium-platform`. Every consumer installs the
  PyPI release, not this directory: the validator lock pins `lium-core` 0.1.8, the miner lock 0.1.6.

## Develop

```bash
cd packages/lium-core
pip install -e . pytest
pytest -q tests
```

CI: `.github/workflows/lium-core-ci.yml` runs the tests and builds the wheel on every change under this directory.

## Release

Bump `version` in `pyproject.toml`, merge, then someone on the tag ruleset's bypass list tags that commit
`lium-core-vX.Y.Z` with the same version and pushes the tag. `.github/workflows/lium-core-release.yml` builds from this directory, refuses a tag whose version is not the
one in `pyproject.toml`, runs `twine check`, and then waits in the **`pypi` environment** until its required reviewer
approves the run (Actions → the run → **Review deployments**). The upload is PyPI trusted publishing with PEP 740
attestations: pypi.org's publisher for `lium-core` names this repository, this workflow file and the `pypi`
environment, so the token exists only inside the approved job; no PyPI token is stored in this repository or on
anyone's machine. A `workflow_dispatch` run only builds, whatever ref it is started from. Every file of a release
shows a *Provenance* link on pypi.org; `https://pypi.org/integrity/lium-core/X.Y.Z/<filename>/provenance` returns
the signed statement.

Who may tag: the `lium-core-release-tags` ruleset (`.github/rulesets/lium-core-release-tags.json`) lets only its
bypass list create, move or delete a `lium-core-v*` tag. The list is the release team: 10954604 (taiberium),
4623096 (arhangel66), 231022467 (jam6099), 248050668 (pixel29913) and the loop's account 114649324 (`surcyf123`).
A tag alone publishes nothing: a human reviewer of the `pypi` environment approves each upload. Add a person by appending
`{ "actor_id": <user-id>, "actor_type": "User", "bypass_mode": "always" }` and re-applying with
`gh api "repos/$R/rulesets/<id>" --method PUT --input .github/rulesets/lium-core-release-tags.json`. Until an admin
applies this file, any account with write access, the loop's account included, can create a `lium-core-v*` tag. The `pypi` environment
requires a human reviewer, blocks self-review and does not allow admin bypass. Both guards are repository settings a
repository admin applies once:

```bash
R=Datura-ai/lium-io
# the environment, with self-approval blocked; required reviewers: one or more humans, by GitHub user id
# (`gh api users/<login> --jq .id`), in place of <human-id>. Never 114649324 (surcyf123, the loop's account): the
# publish job refuses to run while it is listed. The PUT replaces the whole reviewers list. can_admins_bypass false:
# the publish job refuses to run while admin bypass is on.
gh api -X PUT "repos/$R/environments/pypi" --input - <<'JSON'
{ "reviewers": [ { "type": "User", "id": <human-id> } ],
  "prevent_self_review": true,
  "can_admins_bypass": false,
  "deployment_branch_policy": { "protected_branches": false, "custom_branch_policies": true } }
JSON
# only lium-core-v* tags may enter it
gh api -X POST "repos/$R/environments/pypi/deployment-branch-policies" -f name='lium-core-v*' -f type=tag
# the tag ruleset (bypass list = GitHub user ids; `gh api users/<login> --jq .id`)
gh api "repos/$R/rulesets" --method POST --input .github/rulesets/lium-core-release-tags.json
```

Check: `gh api "repos/$R/environments/pypi" --jq '.protection_rules[]|.type'` → `required_reviewers`,
`branch_policy`;
`gh api "repos/$R/environments/pypi" --jq '.protection_rules[]|select(.type=="required_reviewers")|.prevent_self_review'`
→ `true`; `gh api "repos/$R/environments/pypi" --jq .can_admins_bypass` → `false`;
`gh api "repos/$R/environments/pypi" --jq '.protection_rules[]|select(.type=="required_reviewers")|.reviewers[]|.type+" "+(.reviewer.id|tostring)+" "+.reviewer.login'`
→ one `User <id> <login>` line per human, and no `114649324`;
`gh api "repos/$R/rulesets?targets=tag" --jq '.[]|.name+" "+.enforcement'` →
`lium-core-release-tags active`. Other publishers in this repository that later use the same environment add their
tag pattern with one more `deployment-branch-policies` call.

Self-approval is blocked (`prevent_self_review: true`): GitHub refuses an approval from the account that started the
run. That is all it blocks: any one listed required reviewer other than the run's starter can approve. So the list
holds humans only, at least one, and never 114649324 (`surcyf123`, the loop's account, which never approves a
release). List people by user id, not teams: the publish job cannot read team membership, so it refuses a team
reviewer. Whoever pushes the tag starts the run and so cannot approve it: list at least one reviewer who does
not push release tags.

Admin bypass: the `pypi` environment must not allow it. With `can_admins_bypass: true`, any repository admin can start a
waiting publish job ("Start all waiting jobs") without a reviewer's approval. So the publish job refuses to run unless
`can_admins_bypass` is `false` (a missing field counts as on), and the `PUT` above sets `"can_admins_bypass": false`.
Leaving the field out of a `PUT` does not keep the current value: the API's documented default is `true`. The same switch
is Settings → Environments → `pypi` → "Allow administrators to bypass configured protection rules".

**Order — it matters.** (1) Create the `pypi` environment with its reviewers (humans only, at least one, not
114649324), self-approval blocked, admin bypass off and the policy, as above, **before the workflow change merges**: a workflow that
names an environment that does not exist makes GitHub create it with no protection, and the first tagged run would
start the publish job with no click; only the guard step below stops it.
(2) Register the `pypi` publisher on pypi.org
(Manage → Publishing → Add a new publisher → GitHub: owner `Datura-ai`, repository `lium-io`, workflow
`lium-core-release.yml`, environment `pypi`). (3) Merge. (4) Apply the tag ruleset. (5) Proof release: someone on the bypass list pushes the tag, and a human reviewer approves the publish job. (6) **Delete
the old publishers** on pypi.org: `Datura-ai/lium-io · lium-core-release.yml · release` and the archived
`Datura-ai/lium-core · release.yml · release`. Until they are gone a branch whose edited `lium-core-release.yml`
keeps `environment: release` (no reviewer, no branch policy), run by hand, still uploads — any account
with write access can do that. The publish job checks step (1) itself: it reads
`repos/$R/environments/pypi` back and stops unless the environment has at least one required reviewer, 114649324
is not among them, every reviewer is a user, `prevent_self_review` is `true`, and `can_admins_bypass` is `false` (an unauthenticated read for a
public repository; the job holds `actions: read` for it). It cannot tell a human from another machine account; the
admin lists humans. It cannot check step (6) — pypi.org's side is the owner's click. With no `pypi` publisher registered, PyPI rejects the `pypi`-environment
token (the current publisher is bound to `release`), so if (3) runs before (2), a tag pushed after (3) and before (2)
fails closed — that is the only state that does.
