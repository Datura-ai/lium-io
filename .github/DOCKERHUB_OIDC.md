# Docker Hub login by OIDC (no stored token)

The seven image-publishing workflows (`executor_cd_prod.yml`, `executor_cd_dev.yml`,
`miner_cd_prod.yml`, `miner_cd_dev.yml`, `validator_cd_prod.yml`, `validator_cd_dev.yml`,
`watchtower_image.yml`) log in to Docker Hub with `docker/login-action@v4` (4.5.0 or later) and a
Docker Hub **OIDC connection**: GitHub issues a signed ID token for the job, Docker checks its
subject claim against the connection's rulesets and answers with a short-lived access token. No
Docker Hub token is stored in GitHub, so there is nothing to rotate or leak.

Per job: `permissions: id-token: write` (plus `contents: read` for the checkout), and the step

```yaml
- name: Log in to Docker Hub (OIDC, no stored token)
  uses: docker/login-action@v4
  env:
    DOCKERHUB_OIDC_CONNECTIONID: ${{ vars.DOCKERHUB_OIDC_CONNECTIONID }}
  with:
    username: daturaai
```

`username` is the Docker organization name (only organization accounts can sign in with OIDC);
`DOCKERHUB_OIDC_CONNECTIONID` is a repository **variable** (Settings → Secrets and variables →
Actions → Variables), not a secret — the id is not sensitive. The `docker_publish.sh` scripts under
`neurons/*/` log in themselves only when a caller passes `DOCKERHUB_PAT`, so a caller in another
repository that still holds a token keeps working.

## Setting up the connection (Docker Home, organization owner)

Docker Home → organization `daturaai` → **Identity & auth** → **OIDC connections** → **Create OIDC
connection** (GitHub is the only supported provider). Rulesets (1 to 5 per connection), each with a
subject-claim rule, resources and a scope:

| Label | Subject claim | Resources (Docker Hub repositories) | Scope |
|---|---|---|---|
| `lium-io-publish` | `repo:Datura-ai/lium-io:environment:dockerhub-push` | `daturaai/compute-subnet-executor`, `…-executor-runner`, `…-miner`, `…-miner-runner`, `…-validator`, `…-validator-runner`, `daturaai/lium-watchtower` | image push |

Why the environment and not the tag: GitHub's default subject for a job that names an environment is
`repo:<org>/<repo>:environment:<name>` — the ref does not appear in it. The environment
`dockerhub-push` is where GitHub's deployment policy already limits which refs (`main`, the release
tags) may run these jobs, so Docker trusts the environment and GitHub decides who reaches it. A
per-tag-pattern mapping (`…:ref:refs/tags/executor-v*` → executor images only) would need either
jobs without the environment or a customised subject template for the whole repository, which
would also change the subject the PyPI trusted publisher in `lium-core-release.yml` relies on.

Copy the connection id into the repository variable:

```bash
gh variable set DOCKERHUB_OIDC_CONNECTIONID -R Datura-ai/lium-io --body "<connection id>"
```

Check: dispatch `executor_cd_dev.yml` from `main`; the login step ends with `Login Succeeded`.
Failures are listed on the connection's Edit page (Failures table).

Once every repository that pushes `daturaai/*` images from CI has its own ruleset here, the
organization access token used by CI can be deleted (Docker Home → Access tokens), and the GitHub
secrets `DOCKERHUB_PAT` / `DOCKERHUB_USERNAME` go last.

Sources: Docker docs "Create and manage OIDC connections" and "OIDC connections rulesets and
subject claims" (docs.docker.com/security/authentication/oidc-connections/); `docker/login-action`
README, "Docker Hub with OIDC"; GitHub docs "OpenID Connect" (the `sub` claim formats).
