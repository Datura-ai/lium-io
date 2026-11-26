# Release guards: the `dockerhub-push` environment and the `release-tags` ruleset

Every job in this repository that logs in to Docker Hub runs in the GitHub environment
`dockerhub-push` (`executor_cd_prod.yml`, `miner_cd_prod.yml`,
`miner_cd_dev.yml`, `validator_cd_prod.yml`, `validator_cd_dev.yml`, `watchtower_image.yml`).
The login is Docker Hub OIDC (`../DOCKERHUB_OIDC.md`): Docker trusts the subject
`repo:Datura-ai/lium-io:environment:dockerhub-push`, and the environment holds no secret. Its
deployment policy names the only refs allowed to log in: `main` and the release tags
`executor-v*`, `miner-v*`, `validator-v*`, `watchtower-v*`. A `workflow_dispatch` from any other
branch stops before the job's first step with "Branch … is not allowed to deploy to
dockerhub-push" and never gets a Docker Hub token.

The one exception is `executor_cd_dev.yml`, the test build developers dispatch from feature
branches: it runs in a second environment, `dockerhub-push-dev`, with no branch policy.
Docker trusts that subject only for `daturaai/compute-subnet-executor-dev` and
`…-executor-runner-dev` (ruleset `lium-io-dev`, `../DOCKERHUB_OIDC.md`), so a branch build
gets a token that cannot push a release image, whatever its workflow file says. A dispatch from
`main` takes the same path: the release repositories' `dev` tag, which the dev and staging stacks
pull, is written only by lium-io-deployment's staging executor publish (DAH-6670). `miner_cd_dev.yml` and
`validator_cd_dev.yml` are dispatched from `main` only.

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

The dev environment is created empty, with no branch policy and no secrets: its protection is
the Docker Hub ruleset's repository list, not GitHub.

```bash
gh api -X PUT "repos/$R/environments/dockerhub-push-dev"
```

Check: `gh api "repos/$R/environments/dockerhub-push-dev" --jq '.deployment_branch_policy'` → `null`
(every branch). Never add the dev subject to the `lium-io-publish` Docker ruleset.

## 2. Tag ruleset (admin, once)

```bash
gh api "repos/$R/rulesets" --method POST --input .github/rulesets/release-tags.json
```

`release-tags.json` targets the four release tag patterns with the rules `creation`, `update`
and `deletion`; the bypass list is the release role, given as GitHub user ids
(`gh api users/<login> --jq .id`): `4623096` = `arhangel66`, `10954604` = `taiberium`,
`231022467` = `jam6099`, `248050668` = `pixel29913` (the members who push release tags today). The automation account
`surcyf123` is left off on purpose: it opens pull requests but never cuts a release, and a tag it
pushed would ship an image to every executor.
Add a person by appending `{ "actor_id": <id>, "actor_type": "User", "bypass_mode": "always" }`.

Check: `gh api "repos/$R/rulesets?targets=tag" --jq '.[]|.name+" "+.enforcement'` → `release-tags active`.
A push of `executor-v*` by anyone outside the bypass list is refused by GitHub before any workflow runs.

## `pinned-actions-required-workflow`: the pinned-actions check always comes from main (organization owner, once)

`pinned-actions.yml` judges whether a PR or a merge-queue group may land, so the copy that runs must be the one on
`main`, not the one the candidate carries. A repository workflow cannot guarantee that on `pull_request` or
`merge_group` (both run the candidate's copy), so an organization ruleset requires it instead:
`pinned-actions-required-workflow.json` ("Require workflows to pass before merging" on the default branch of `lium-io`
and `lium`, workflow `.github/workflows/pinned-actions.yml` from `Datura-ai/lium-io` at `refs/heads/main`). GitHub then
runs main's copy on every PR and every queued group, whatever the candidate changes in its own workflow files. The rule
exists at organization level only, so an organization owner applies it:

Roll it out in three separate steps, each run on its own; never paste them as one script. Activating too early blocks
ordinary PRs in two ways: a required run that never reports leaves every PR pending, and a required run that fails
blocks every PR whose tree still has a mutable reference.

1. Precondition: the default branch of each target repository passes the check with zero refusals. `lium` does not yet
   (its own pinning PR, Datura-ai/lium#458, must merge first). Until both default branches pass, stop here.

2. Apply the ruleset in evaluate mode. It reports the required run but blocks nothing:

   ```bash
   jq '.enforcement = "evaluate"' .github/rulesets/pinned-actions-required-workflow.json |
     gh api "orgs/Datura-ai/rulesets" --method POST --input -
   ```

   Open a test PR in `lium`, the repository that is not the workflow's source, with any harmless change. Its checks
   must list `pinned-actions` from `Datura-ai/lium-io` with the result **success**. A pending check means the run never
   reports there (GitHub runs a ruleset workflow on `pull_request` outside its source repository only because the job has
   a condition, `if: true` in the workflow; GitHub's "Troubleshooting rules", Supported directory). A failure means that
   tree still has a mutable reference. In either case stay in evaluate mode and fix the cause first. Do the same on a test
   PR in `lium-io`.

3. Only after both test PRs show success, activate:

   ```bash
   id=$(gh api "orgs/Datura-ai/rulesets" --jq '.[]|select(.name=="pinned-actions-required-workflow")|.id')
   gh api "orgs/Datura-ai/rulesets/$id" --method PUT --input .github/rulesets/pinned-actions-required-workflow.json
   ```

If the organization's plan has no evaluate mode, the POST in step 2 refuses it. Do not apply the file active to find out
whether the run reports. Instead, apply it with the repository condition narrowed to `lium-io` (the workflow's source,
where the run is known to report) and confirm success on a `lium-io` test PR. Add `lium` to the condition only once step 1
holds for `lium` and the ruleset's own run on `lium-io` has shown success. Then check a `lium` test PR straight away, and
remove `lium` from the condition again if its result is not success.

Check: `gh api "orgs/Datura-ai/rulesets" --jq '.[]|.name+" "+.enforcement'` → `pinned-actions-required-workflow active`.

**A retargeted PR cannot merge unscanned, but it waits for a push.** A workflow required by a ruleset runs on its own
trigger list and ignores the `types` in the file, so a retarget (a stacked PR whose base PR merged) starts no required run.
A PR opened against another branch never got a required run (the ruleset covers the default branch only), so after the
retarget the required workflow shows as expected and the PR cannot merge. Strict mode forces a branch update only when
`main` is not yet an ancestor of the head; when the head already contains `main`, nothing forces one. In both cases start
the required run yourself: push a commit (or use "Update branch"), or close and reopen the PR, which runs it on the same
head. After a retarget, a required result exists only if a run already happened on that head while the PR targeted `main`.
The required run scans the PR's head commit, not its merge ref: a retarget changes the base but not the head, so an existing
result still describes the commit under review. The same ruleset also sets "Require branches to be up to date before merging" (strict), which
GitHub checks on the branch itself: a PR can merge only when `main` is an ancestor of its head, and then the merged tree is
exactly the head's tree. Bringing a branch up to date is a push, which starts the required run on the new head. The
`pinned-actions` status context only carries the strict rule. A PR's own copy of the workflow can report that context
too, but that grants nothing: the merge still needs the required run of `main`'s copy on the current head. In `lium-io`
the merge queue goes further: every queued group is checked on its own merge commit.
