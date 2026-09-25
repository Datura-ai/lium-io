"""The prod Docker Hub token is readable only in ``dockerhub-push`` jobs, never by a repository
script, and the ``dev`` builds keep working from any branch with a separate dev token.

A repository secret is readable by any workflow file on any branch; an environment secret only by a
job that names the environment, from a ref the environment's deployment policy allows.
``dockerhub-push`` allows ``main`` and the release tags; ``dockerhub-dev`` allows every branch and
holds ``DOCKERHUB_DEV_PAT``, so a ``*_cd_dev.yml`` run from a feature branch still pushes ``:dev``.

A secret in a job-level ``env:`` is in the environment of every step, including the ones that run
``neurons/*/docker*publish.sh`` from the checked-out ref. So the token is only ever set on the one
login step, whose command is inline in the workflow, and the publish scripts only push with the
login that step already did. These tests fail when a job reads a Docker Hub token outside its
environment, when the token is set anywhere but a login step, when a script logs in or reads a
token, when a workflow that reads a token can be started by a pull request, or when the tag ruleset
payload drifts from the prod workflows' tag triggers.
"""

import json
import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
WORKFLOWS_DIR = REPO / ".github" / "workflows"
WORKFLOWS = sorted([*WORKFLOWS_DIR.glob("*.yml"), *WORKFLOWS_DIR.glob("*.yaml")])
DEV_WORKFLOWS = sorted(WORKFLOWS_DIR.glob("*_cd_dev.yml"))
PUBLISH_SCRIPTS = sorted((REPO / "neurons").glob("*/docker*publish.sh"))
RULESET = REPO / ".github" / "rulesets" / "release-tags.json"
PROD_ENVIRONMENT = "dockerhub-push"
DEV_ENVIRONMENT = "dockerhub-dev"
PROD_SECRET = "secrets.DOCKERHUB_PAT"
DEV_SECRET = "secrets.DOCKERHUB_DEV_PAT"
LOGIN_RUN = 'echo "$DOCKERHUB_PAT" | docker login -u "$DOCKERHUB_USERNAME" --password-stdin'
PULL_REQUEST_TRIGGERS = {"pull_request", "pull_request_target", "workflow_run"}

DOCKER_HUB_REGISTRIES = {"", "docker.io", "registry-1.docker.io", "index.docker.io"}


def _triggers(workflow: dict) -> dict:
    on = workflow.get("on", workflow.get(True))  # PyYAML reads the bare key ``on`` as True
    if isinstance(on, str):
        return {on: None}
    if isinstance(on, list):
        return dict.fromkeys(on)
    return on or {}


def logs_in_to_docker_hub(job: dict) -> bool:
    for step in job.get("steps") or []:
        if str(step.get("uses", "")).startswith("docker/login-action"):
            registry = str((step.get("with") or {}).get("registry", "")).strip()
            if registry in DOCKER_HUB_REGISTRIES:
                return True
        if "docker login" in str(step.get("run", "")):
            return True
    return False


def token_problems(workflow_text: str) -> list[str]:
    """Every place a workflow exposes a Docker Hub token beyond its environment's login step."""
    workflow = yaml.safe_load(workflow_text)
    problems = []
    if "DOCKERHUB" in json.dumps(workflow.get("env") or {}):
        problems.append("workflow env")
    reads_a_token = False
    for job_id, job in (workflow.get("jobs") or {}).items():
        text = json.dumps(job)
        reads_prod, reads_dev = PROD_SECRET in text, DEV_SECRET in text
        reads_a_token |= reads_prod or reads_dev
        environment = job.get("environment")
        if reads_prod and environment != PROD_ENVIRONMENT:
            problems.append(f"{job_id}: prod token outside {PROD_ENVIRONMENT}")
        if reads_dev and environment != DEV_ENVIRONMENT:
            problems.append(f"{job_id}: dev token outside {DEV_ENVIRONMENT}")
        if logs_in_to_docker_hub(job) and environment not in (PROD_ENVIRONMENT, DEV_ENVIRONMENT):
            problems.append(f"{job_id}: Docker Hub login outside both environments")
        if "DOCKERHUB" in json.dumps(job.get("env") or {}):
            problems.append(f"{job_id}: token in job env")
        for step in job.get("steps") or []:
            step_text = json.dumps(step)
            if "secrets.DOCKERHUB" not in step_text:
                continue
            if step.get("uses") or str(step.get("run", "")).strip() != LOGIN_RUN:
                problems.append(f"{job_id}: token set on a step that is not the inline login")
    if reads_a_token and PULL_REQUEST_TRIGGERS & set(_triggers(workflow)):
        problems.append("a pull request can start a workflow that reads a token")
    return problems


def script_problems(script_text: str) -> bool:
    return bool(re.search(r"docker login|DOCKERHUB_", script_text))


PROD_JOB = (
    "on:\n  push:\n    tags: ['executor-v*']\njobs:\n  deploy:\n    runs-on: ubuntu-latest\n"
    f"    environment: {PROD_ENVIRONMENT}\n    steps:\n"
    "      - uses: actions/checkout@v4\n"
    "      - name: Log in to Docker Hub\n        env:\n"
    "          DOCKERHUB_PAT: ${{ secrets.DOCKERHUB_PAT }}\n"
    "          DOCKERHUB_USERNAME: ${{ secrets.DOCKERHUB_USERNAME }}\n"
    f"        run: '{LOGIN_RUN}'\n"
    "      - run: cd neurons/executor && ./docker_publish.sh\n"
)


def test_the_checker_accepts_the_prod_job_shape() -> None:
    assert token_problems(PROD_JOB) == []
    dev_job = PROD_JOB.replace(PROD_ENVIRONMENT, DEV_ENVIRONMENT).replace(
        "secrets.DOCKERHUB_PAT", "secrets.DOCKERHUB_DEV_PAT"
    )
    assert token_problems(dev_job) == []


def test_the_checker_flags_the_token_handed_to_a_script() -> None:
    """Negative control: the shape before this change, the token in the job env for every step."""
    job_env = PROD_JOB.replace(
        "    steps:\n", "    env:\n      DOCKERHUB_PAT: ${{ secrets.DOCKERHUB_PAT }}\n    steps:\n", 1
    )
    assert "deploy: token in job env" in token_problems(job_env)
    script_step = PROD_JOB.replace(f"        run: '{LOGIN_RUN}'\n", "        run: ./docker_publish.sh\n")
    assert token_problems(script_step) == ["deploy: token set on a step that is not the inline login"]


def test_the_checker_flags_a_token_outside_its_environment() -> None:
    """Negative controls: the prod token in a dev job, and a job with no environment at all."""
    prod_token_in_dev = PROD_JOB.replace(PROD_ENVIRONMENT, DEV_ENVIRONMENT)
    assert token_problems(prod_token_in_dev) == [f"deploy: prod token outside {PROD_ENVIRONMENT}"]
    no_environment = PROD_JOB.replace(f"    environment: {PROD_ENVIRONMENT}\n", "")
    assert token_problems(no_environment) == [
        f"deploy: prod token outside {PROD_ENVIRONMENT}",
        "deploy: Docker Hub login outside both environments",
    ]
    oidc_login = (
        "on: workflow_dispatch\njobs:\n  deploy:\n    runs-on: ubuntu-latest\n    steps:\n"
        "      - uses: docker/login-action@v4\n        with:\n          username: daturaai\n"
    )
    assert token_problems(oidc_login) == ["deploy: Docker Hub login outside both environments"]
    assert token_problems(oidc_login.replace("username: daturaai", "registry: ghcr.io")) == []


def test_the_checker_flags_a_pull_request_trigger() -> None:
    for trigger in ("pull_request_target", "pull_request"):
        on_pr = PROD_JOB.replace("on:\n", f"on:\n  {trigger}:\n", 1)
        assert token_problems(on_pr) == ["a pull request can start a workflow that reads a token"]


@pytest.mark.parametrize("workflow", WORKFLOWS, ids=lambda p: p.name)
def test_no_workflow_exposes_a_docker_hub_token(workflow: Path) -> None:
    assert token_problems(workflow.read_text()) == []


@pytest.mark.parametrize("workflow", DEV_WORKFLOWS, ids=lambda p: p.name)
def test_dev_builds_use_the_dev_token_from_any_branch(workflow: Path) -> None:
    parsed = yaml.safe_load(workflow.read_text())
    assert set(_triggers(parsed)) == {"workflow_dispatch"}
    (job,) = parsed["jobs"].values()
    assert job["environment"] == DEV_ENVIRONMENT
    assert job["env"]["TAG"] == "dev"
    text = workflow.read_text()
    assert DEV_SECRET in text and PROD_SECRET not in text


def test_the_checker_flags_a_script_that_logs_in() -> None:
    assert script_problems(f"#!/bin/bash\n{LOGIN_RUN}\ndocker push x\n")
    assert not script_problems('#!/bin/bash\ndocker push "$IMAGE_NAME"\n')


@pytest.mark.parametrize("script", PUBLISH_SCRIPTS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_publish_scripts_only_push(script: Path) -> None:
    script_text = script.read_text()
    assert "docker push" in script_text
    assert not script_problems(script_text)


def test_release_tag_ruleset_covers_every_tag_trigger_of_the_prod_workflows() -> None:
    ruleset = json.loads(RULESET.read_text())
    assert ruleset["target"] == "tag" and ruleset["enforcement"] == "active"
    assert {r["type"] for r in ruleset["rules"]} == {"creation", "update", "deletion"}
    assert ruleset["bypass_actors"] and all(
        a["actor_type"] == "User" and isinstance(a["actor_id"], int)
        for a in ruleset["bypass_actors"]
    )
    publish_tag_refs: set[str] = set()
    for workflow in WORKFLOWS:
        parsed = yaml.safe_load(workflow.read_text())
        push = _triggers(parsed).get("push") or {}
        jobs = (parsed.get("jobs") or {}).values()
        if any(job.get("environment") == PROD_ENVIRONMENT for job in jobs):
            publish_tag_refs.update(f"refs/tags/{t}" for t in push.get("tags") or [])
    assert publish_tag_refs == set(ruleset["conditions"]["ref_name"]["include"])
    assert publish_tag_refs == {
        "refs/tags/executor-v*",
        "refs/tags/miner-v*",
        "refs/tags/validator-v*",
        "refs/tags/watchtower-v*",
    }
