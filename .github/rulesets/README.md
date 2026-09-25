# Release guards: the `dockerhub-push` and `dockerhub-dev` environments and the `release-tags` ruleset

Two Docker Hub tokens, each an environment secret, so no branch can read the prod token:

- `dockerhub-push` holds the prod token (`DOCKERHUB_PAT`, `DOCKERHUB_USERNAME`). Its deployment
  policy allows only `main` and the release tags `executor-v*`, `miner-v*`, `validator-v*`,
  `watchtower-v*`. The jobs that use it are `executor_cd_prod.yml`, `miner_cd_prod.yml`,
  `validator_cd_prod.yml` and `watchtower_image.yml`. A run from any other ref stops before the
  job's first step with "Branch … is not allowed to deploy to dockerhub-push" and never sees the
  token. A manual run of `watchtower_image.yml` from `main` still publishes without a tag.
- `dockerhub-dev` holds the dev token (`DOCKERHUB_DEV_PAT`, `DOCKERHUB_DEV_USERNAME`) and allows
  every branch, so `executor_cd_dev.yml`, `miner_cd_dev.yml` and `validator_cd_dev.yml` still
  build and push `:dev` from a feature branch.

In every job the token is set only on the inline "Log in to Docker Hub" step, and the
`neurons/*/docker*publish.sh` scripts only push, using that login.
That login is saved on the runner, so every later step of the same job, those scripts included,
can use it. What keeps a branch away from the prod token is the `dockerhub-push` deployment
policy: a prod job runs only from `main` or a release tag, so the scripts it runs are the
reviewed ones. A run from any other branch gets only the dev token.
Anything outside this repository that runs these scripts (an external staging pipeline) has to
log in itself with the dev token; it cannot read either environment secret here.

The tag ruleset `release-tags` (`release-tags.json`) restricts who may create, move or delete
the four release tag patterns, so only the release role can start a tag-triggered production
image push.

Both settings are repository administration, done once by a repository admin with the commands
below. Create `dockerhub-dev` and its secrets before this change merges: the dev workflows read
only `DOCKERHUB_DEV_PAT` and fail at the login step until it exists. Until `dockerhub-push` is
configured, GitHub auto-creates it on the first run with no policy and no secrets, and the
repository-level prod secrets keep resolving.

## 1. Environment (admin, once)

UI: Settings → Environments → New environment → `dockerhub-push` → Deployment branches and tags →
"Selected branches and tags" → add `main` (branch) and `executor-v*`, `miner-v*`, `validator-v*`,
`watchtower-v*` (tag) → Environment secrets → add `DOCKERHUB_PAT` and `DOCKERHUB_USERNAME`.

Same thing from a shell (`gh` authenticated as a repository admin):

```bash
R=Datura-ai/lium-io
gh api -X PUT "repos/$R/environments/dockerhub-push" \
  -F 'deployment_branch_policy[protected_branches]=false' \
  -F 'deployment_branch_policy[custom_branch_policies]=true'
gh api -X POST "repos/$R/environments/dockerhub-push/deployment-branch-policies" -f name=main -f type=branch
for t in 'executor-v*' 'miner-v*' 'validator-v*' 'watchtower-v*'; do
  gh api -X POST "repos/$R/environments/dockerhub-push/deployment-branch-policies" -f "name=$t" -f type=tag
done
gh secret set DOCKERHUB_PAT      -R "$R" --env dockerhub-push          # paste the value when prompted
gh secret set DOCKERHUB_USERNAME -R "$R" --env dockerhub-push --body daturaai
# only after the two environment secrets exist:
gh secret delete DOCKERHUB_PAT      -R "$R"
gh secret delete DOCKERHUB_USERNAME -R "$R"
```

Check: `gh api "repos/$R/environments/dockerhub-push/deployment-branch-policies" --jq '.branch_policies[]|.type+" "+.name'`
lists the five refs; `gh secret list -R "$R" --env dockerhub-push` lists the two names.

Optional: "Required reviewers" on the environment makes every production image push a
two-person action.

### Dev environment

Docker Hub → Account settings → Personal access tokens (or the organisation's access tokens):
create a token with Read & Write only, no Delete, used only as the dev token. Then:

```bash
gh api -X PUT "repos/$R/environments/dockerhub-dev"
gh secret set DOCKERHUB_DEV_PAT      -R "$R" --env dockerhub-dev       # paste the dev token
gh secret set DOCKERHUB_DEV_USERNAME -R "$R" --env dockerhub-dev --body daturaai
```

Only the prod token is fenced. Docker Hub scopes a token per repository, not per tag, and the
`dev` images share their repositories with `latest` (`daturaai/compute-subnet-executor:dev` next
to `:latest`). So until the `dev` images move to their own repositories, the dev token, which
any branch can use, can still push `:latest`. Moving them means changing the dev compose files
and the dev hosts too; that is a separate change.

## 2. Tag ruleset (admin, once)

```bash
gh api "repos/$R/rulesets" --method POST --input .github/rulesets/release-tags.json
```

`release-tags.json` targets the four release tag patterns with the rules `creation`, `update`
and `deletion`; the bypass list is the release role, given as GitHub user ids
(`gh api users/<login> --jq .id`): `4623096` = `arhangel66`, `10954604` = `taiberium`,
`231022467` = `jam6099`, `248050668` = `pixel29913` (the members who push release tags today).
The automation account `surcyf123` is left off on purpose: it opens pull requests but never cuts
a release, and a tag it pushed would ship an image to every executor.
Add a person by appending `{ "actor_id": <id>, "actor_type": "User", "bypass_mode": "always" }`.

Check: `gh api "repos/$R/rulesets?targets=tag" --jq '.[]|.name+" "+.enforcement'` → `release-tags active`.
A push of `executor-v*` by anyone outside the bypass list is refused by GitHub before any workflow runs.

## What is not covered here

The credentials are still a long-lived token. Replacing it with a Docker Hub OIDC connection
(short-lived tokens, nothing stored in GitHub) is a separate change to the same seven workflows.
