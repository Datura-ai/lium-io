# Release guards: the `dockerhub-push` environment and the `release-tags` ruleset

Every job in this repository that logs in to Docker Hub runs in the GitHub environment
`dockerhub-push` (`executor_cd_prod.yml`, `executor_cd_dev.yml`, `miner_cd_prod.yml`,
`miner_cd_dev.yml`, `validator_cd_prod.yml`, `validator_cd_dev.yml`, `watchtower_image.yml`).
The login is Docker Hub OIDC (`../DOCKERHUB_OIDC.md`): Docker trusts the subject
`repo:Datura-ai/lium-io:environment:dockerhub-push`, and the environment holds no secret. Its
deployment policy names the only refs allowed to log in: `main` and the release tags
`executor-v*`, `miner-v*`, `validator-v*`, `watchtower-v*`. A `workflow_dispatch` from any other
branch stops before the job's first step with "Branch … is not allowed to deploy to
dockerhub-push" and never gets a Docker Hub token.

The tag ruleset `release-tags` (`release-tags.json`) restricts who may create, move or delete
those four tag patterns, so only the release role can start a production image push.

Both settings are repository administration. The workflow files reference the environment;
configuring it and applying the ruleset is done once by a repository admin with the commands
below. Until that is done, GitHub auto-creates the environment on the first run with no policy:
the login works, but a run from any branch gets a push token.

## 1. Environment (admin, once)

UI: Settings → Environments → New environment → `dockerhub-push` → Deployment branches and tags →
"Selected branches and tags" → add `main` (branch) and `executor-v*`, `miner-v*`, `validator-v*`,
`watchtower-v*` (tag). Add no secrets: the login needs none.

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
```

Check: `gh api "repos/$R/environments/dockerhub-push/deployment-branch-policies" --jq '.branch_policies[]|.type+" "+.name'`
lists the five refs.

Optional: "Required reviewers" on the environment makes every production image push a
two-person action.

## 2. Tag ruleset (admin, once)

```bash
gh api "repos/$R/rulesets" --method POST --input .github/rulesets/release-tags.json
```

`release-tags.json` targets the four release tag patterns with the rules `creation`, `update`
and `deletion`; the bypass list is the release role, given as GitHub user ids
(`gh api users/<login> --jq .id`): `4623096` = `arhangel66`, `10954604` = `taiberium`,
`231022467` = `jam6099` (the members who push release tags today). The automation account
`surcyf123` is left off on purpose: it opens pull requests but never cuts a release, and a tag it
pushed would ship an image to every executor.
Add a person by appending `{ "actor_id": <id>, "actor_type": "User", "bypass_mode": "always" }`.

Check: `gh api "repos/$R/rulesets?targets=tag" --jq '.[]|.name+" "+.enforcement'` → `release-tags active`.
A push of `executor-v*` by anyone outside the bypass list is refused by GitHub before any workflow runs.
