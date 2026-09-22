"""Every Docker Hub login in this repository runs inside the ``dockerhub-push`` environment, and the
release tags the publish workflows react to are the ones the ``release-tags`` ruleset guards.

A repository secret is readable by any workflow file on any branch; an environment secret only by a
job that names the environment, from a ref the environment's deployment policy allows. These tests
keep a new or edited publish job — one that reads the token or logs in to Docker Hub, by ``docker
login`` or ``docker/login-action`` — outside that environment, keep the shell trace off around the
``docker login`` line in scripts that run with ``set -x``, and keep the tag ruleset payload in step
with the workflows' tag triggers. The discriminator is the login, not the secret's name, so the
tests hold once the login moves to OIDC and no workflow names the secret any more.
"""

import json
import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
WORKFLOWS = sorted((REPO / ".github" / "workflows").glob("*.yml"))
PUBLISH_SCRIPTS = sorted((REPO / "neurons").glob("*/docker*publish.sh"))
RULESET = REPO / ".github" / "rulesets" / "release-tags.json"
ENVIRONMENT = "dockerhub-push"
LOGIN_LINE = 'echo "$DOCKERHUB_PAT" | docker login'


DOCKER_HUB_REGISTRIES = {"", "docker.io", "registry-1.docker.io", "index.docker.io"}


def publishes_to_docker_hub(job: dict) -> bool:
    """True when the job reads the Docker Hub token or logs in to Docker Hub (``docker login`` or ``docker/login-action``)."""
    if "secrets.DOCKERHUB_PAT" in json.dumps(job):
        return True
    for step in job.get("steps") or []:
        if str(step.get("uses", "")).startswith("docker/login-action"):
            registry = str((step.get("with") or {}).get("registry", "")).strip()
            if registry in DOCKER_HUB_REGISTRIES:
                return True
        if "docker login" in str(step.get("run", "")):
            return True
    return False


def jobs_reading_the_token_outside_the_environment(workflow_text: str) -> list[str]:
    """Job ids that publish to Docker Hub without ``environment: dockerhub-push``."""
    workflow = yaml.safe_load(workflow_text)
    return [
        job_id
        for job_id, job in (workflow.get("jobs") or {}).items()
        if publishes_to_docker_hub(job) and job.get("environment") != ENVIRONMENT
    ]


def login_traced(script_text: str) -> bool:
    """True when the script runs with xtrace on and the login line is not preceded by ``set +x``."""
    if not re.search(r"^set .*x", script_text, re.MULTILINE):
        return False
    before_login = script_text.split(LOGIN_LINE, 1)[0]
    return "set +x" not in before_login


def test_the_checker_flags_a_job_that_reads_the_token_without_the_environment():
    """Negative control: the shape every publish job had before the environment existed."""
    text = (
        "on: workflow_dispatch\njobs:\n  deploy:\n    runs-on: ubuntu-latest\n"
        "    env:\n      DOCKERHUB_PAT: ${{ secrets.DOCKERHUB_PAT }}\n    steps: []\n"
    )
    assert jobs_reading_the_token_outside_the_environment(text) == ["deploy"]
    assert (
        jobs_reading_the_token_outside_the_environment(
            text.replace("    env:", f"    environment: {ENVIRONMENT}\n    env:")
        )
        == []
    )


def test_the_checker_flags_a_docker_hub_login_step_without_the_environment():
    """Negative control: an OIDC login (no secret named) still has to run inside the environment."""
    text = (
        "on: workflow_dispatch\njobs:\n  deploy:\n    runs-on: ubuntu-latest\n    steps:\n"
        "      - uses: docker/login-action@v4\n        with:\n          username: daturaai\n"
    )
    assert jobs_reading_the_token_outside_the_environment(text) == ["deploy"]
    other_registry = text.replace("          username: daturaai\n", "          registry: ghcr.io\n")
    assert jobs_reading_the_token_outside_the_environment(other_registry) == []
    assert (
        jobs_reading_the_token_outside_the_environment(
            text.replace("    steps:", f"    environment: {ENVIRONMENT}\n    steps:")
        )
        == []
    )


@pytest.mark.parametrize("workflow", WORKFLOWS, ids=lambda p: p.name)
def test_every_job_that_reads_the_docker_hub_token_runs_in_the_environment(workflow):
    assert jobs_reading_the_token_outside_the_environment(workflow.read_text()) == []


def test_the_checker_flags_a_traced_login():
    """Negative control: ``set -eux`` and the login line, which ``bash -x`` prints expanded."""
    traced = f"#!/bin/bash\nset -eux -o pipefail\n{LOGIN_LINE} -u u --password-stdin\n"
    assert login_traced(traced)
    assert not login_traced(traced.replace(LOGIN_LINE, f"{{ set +x; }} 2>/dev/null\n{LOGIN_LINE}"))
    assert not login_traced(traced.replace("set -eux", "set -eu"))


@pytest.mark.parametrize("script", PUBLISH_SCRIPTS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_no_publish_script_traces_the_login(script):
    text = script.read_text()
    assert LOGIN_LINE in text, f"{script}: the login moved; update LOGIN_LINE"
    assert not login_traced(text)


def test_release_tag_ruleset_covers_every_tag_trigger_of_the_publish_workflows():
    ruleset = json.loads(RULESET.read_text())
    assert ruleset["target"] == "tag" and ruleset["enforcement"] == "active"
    assert {r["type"] for r in ruleset["rules"]} == {"creation", "update", "deletion"}
    assert ruleset["bypass_actors"] and all(
        a["actor_type"] == "User" and isinstance(a["actor_id"], int)
        for a in ruleset["bypass_actors"]
    )
    triggered = set()
    for workflow in WORKFLOWS:
        parsed = yaml.safe_load(workflow.read_text())
        push = (parsed.get("on") or parsed.get(True) or {}).get("push") or {}
        if any(publishes_to_docker_hub(job) for job in (parsed.get("jobs") or {}).values()):
            triggered.update(f"refs/tags/{t}" for t in push.get("tags") or [])
    assert triggered == set(ruleset["conditions"]["ref_name"]["include"])
    assert triggered == {
        "refs/tags/executor-v*",
        "refs/tags/miner-v*",
        "refs/tags/validator-v*",
        "refs/tags/watchtower-v*",
    }
