# Release guards: the `dockerhub-push` environment and the `release-tags` ruleset

Every job in this repository that logs in to Docker Hub runs in the GitHub environment
`dockerhub-push` (`executor_cd_prod.yml`, `executor_cd_dev.yml`, `miner_cd_prod.yml`,
`miner_cd_dev.yml`, `validator_cd_prod.yml`, `validator_cd_dev.yml`, `watchtower_image.yml`).
The environment holds the Docker Hub credentials and a deployment policy that names the only
refs allowed to use them: `main` and the release tags `executor-v*`, `miner-v*`, `validator-v*`,
`watchtower-v*`. A `workflow_dispatch` from any other branch stops before the job's first step
with "Branch … is not allowed to deploy to dockerhub-push" and never sees the secrets.

The tag ruleset `release-tags` (`release-tags.json`) restricts who may create, move or delete
those four tag patterns, so only the release role can start a production image push.

Both settings are repository administration. The workflow files reference the environment;
configuring it and applying the ruleset is done once by a repository admin with the commands
below. Until that is done, GitHub auto-creates the environment on the first run with no policy
and no secrets, and the repository-level secrets keep resolving — nothing breaks in between.

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

## 2. Tag ruleset (admin, once)

```bash
gh api "repos/$R/rulesets" --method POST --input .github/rulesets/release-tags.json
```

`release-tags.json` targets the four release tag patterns with the rules `creation`, `update`
and `deletion`; the bypass list is the release role, given as GitHub user ids
(`gh api users/<login> --jq .id`): `114649324` = `surcyf123`, `4623096` = `arhangel66`.
Add a person by appending `{ "actor_id": <id>, "actor_type": "User", "bypass_mode": "always" }`.

Check: `gh api "repos/$R/rulesets?targets=tag" --jq '.[]|.name+" "+.enforcement'` → `release-tags active`.
A push of `executor-v*` by anyone outside the bypass list is refused by GitHub before any workflow runs.

## What is not covered here

The credentials are still a long-lived token. Replacing it with a Docker Hub OIDC connection
(short-lived tokens, nothing stored in GitHub) is a separate change to the same seven workflows.
