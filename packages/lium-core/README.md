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

Bump `version` in `pyproject.toml`, merge, then tag that commit `lium-core-vX.Y.Z` with the same version and push the
tag. `.github/workflows/lium-core-release.yml` builds from this directory, refuses a tag whose version is not the
one in `pyproject.toml`, runs `twine check`, and then waits in the **`pypi` environment** until its required reviewer
approves the run (Actions → the run → **Review deployments**). The upload is PyPI trusted publishing with PEP 740
attestations: pypi.org's publisher for `lium-core` names this repository, this workflow file and the `pypi`
environment, so the token exists only inside the approved job; no PyPI token is stored in this repository or on
anyone's machine. A `workflow_dispatch` run only builds. Every file of a release shows a *Provenance* link on
pypi.org; `https://pypi.org/integrity/lium-core/X.Y.Z/<filename>/provenance` returns the signed statement.

Who may tag: the `lium-core-release-tags` ruleset (`.github/rulesets/lium-core-release-tags.json`) lets only its
bypass list create, move or delete a `lium-core-v*` tag. Both guards are repository settings a repository admin
applies once:

```bash
R=Datura-ai/lium-io
# the environment, with the owner as the one required reviewer (GitHub user id 114649324 = surcyf123);
# only lium-core-v* tags may enter it
gh api -X PUT "repos/$R/environments/pypi" --input - <<'JSON'
{ "reviewers": [ { "type": "User", "id": 114649324 } ],
  "prevent_self_review": false,
  "deployment_branch_policy": { "protected_branches": false, "custom_branch_policies": true } }
JSON
gh api -X POST "repos/$R/environments/pypi/deployment-branch-policies" -f name='lium-core-v*' -f type=tag
# the tag ruleset (bypass list = GitHub user ids; `gh api users/<login> --jq .id`)
gh api "repos/$R/rulesets" --method POST --input .github/rulesets/lium-core-release-tags.json
```

Check: `gh api "repos/$R/environments/pypi" --jq '.protection_rules[]|.type'` → `required_reviewers`,
`branch_policy`; `gh api "repos/$R/rulesets?targets=tag" --jq '.[]|.name+" "+.enforcement'` →
`lium-core-release-tags active`. Other publishers in this repository that later use the same environment add their
tag pattern with one more `deployment-branch-policies` call.
