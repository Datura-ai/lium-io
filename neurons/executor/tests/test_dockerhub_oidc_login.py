"""Docker Hub login in CI is OIDC: ``docker/login-action`` with the organization name, an ``id-token: write``
job, the connection id from a repository variable — and no stored token anywhere in the workflows.

The publish scripts keep a ``docker login`` for a caller that still passes ``DOCKERHUB_PAT`` (the staging
publish in the deployment repository sources them) and skip it when no token is passed, without ever
printing the value under ``set -x``.
"""

import os
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
WORKFLOWS = sorted((REPO / ".github" / "workflows").glob("*.yml"))
PUBLISH_SCRIPTS = sorted((REPO / "neurons").glob("*/docker*publish.sh"))
LOGIN_ACTION = "docker/login-action@"
ORG = "daturaai"
CONNECTION_ID = "${{ vars.DOCKERHUB_OIDC_CONNECTIONID }}"

STUB_DOCKER = """#!/bin/bash
# records every call; `push` answers like the real client, `login` swallows stdin
echo "docker $*" >> "$STUB_LOG"
case "$1" in
  login) cat > /dev/null; echo "Login Succeeded" ;;
  push) echo "latest: digest: sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef size: 1" ;;
  *) : ;;
esac
"""


def docker_hub_login_problems(workflow_text: str) -> list[str]:
    """Every problem with a workflow's Docker Hub login: a login step without OIDC (a `password`, a wrong
    username, no connection id), a login job without `id-token: write`, a stored token still referenced."""
    problems = []
    if "secrets.DOCKERHUB_PAT" in workflow_text or "secrets.DOCKERHUB_USERNAME" in workflow_text:
        problems.append("names the stored Docker Hub token")
    for job_id, job in (yaml.safe_load(workflow_text).get("jobs") or {}).items():
        # a `registry:` other than Docker Hub (the staging workflows log in to ghcr.io with GITHUB_TOKEN) is not a Docker Hub login
        login_steps = [
            s
            for s in job.get("steps") or []
            if str(s.get("uses", "")).startswith(LOGIN_ACTION)
            and (s.get("with") or {}).get("registry", "docker.io")
            in ("docker.io", "registry-1.docker.io")
        ]
        if not login_steps:
            continue
        if (job.get("permissions") or {}).get("id-token") != "write":
            problems.append(f"{job_id}: no `id-token: write`")
        for step in login_steps:
            with_ = step.get("with") or {}
            if "password" in with_:
                problems.append(f"{job_id}: login step carries a password")
            if with_.get("username") != ORG:
                problems.append(f"{job_id}: username is not the organization name")
            if (step.get("env") or {}).get("DOCKERHUB_OIDC_CONNECTIONID") != CONNECTION_ID:
                problems.append(f"{job_id}: no DOCKERHUB_OIDC_CONNECTIONID from vars")
    return problems


def test_the_checker_flags_the_token_login_shape():
    """Negative control: the login every workflow had before — a repository secret and no id-token."""
    before = (
        "on: workflow_dispatch\njobs:\n  deploy:\n    runs-on: ubuntu-latest\n"
        "    env:\n      DOCKERHUB_PAT: ${{ secrets.DOCKERHUB_PAT }}\n    steps: []\n"
    )
    assert docker_hub_login_problems(before) == ["names the stored Docker Hub token"]
    password_login = (
        "on: workflow_dispatch\njobs:\n  deploy:\n    runs-on: ubuntu-latest\n    steps:\n"
        "      - uses: docker/login-action@v4\n        with:\n          username: daturaai\n          password: ${{ secrets.X }}\n"
    )
    assert docker_hub_login_problems(password_login) == [
        "deploy: no `id-token: write`",
        "deploy: login step carries a password",
        "deploy: no DOCKERHUB_OIDC_CONNECTIONID from vars",
    ]
    ghcr = password_login.replace("username: daturaai", "registry: ghcr.io\n          username: x")
    assert docker_hub_login_problems(ghcr) == []


@pytest.mark.parametrize("workflow", WORKFLOWS, ids=lambda p: p.name)
def test_every_docker_hub_login_is_oidc_and_no_workflow_names_the_token(workflow):
    assert docker_hub_login_problems(workflow.read_text()) == []


def test_the_seven_publish_workflows_log_in_with_the_action():
    """The inventory of pushers: each one carries the OIDC login step (a pusher without it would push unauthenticated and fail)."""
    with_login = {w.name for w in WORKFLOWS if LOGIN_ACTION in w.read_text()}
    assert with_login >= {
        "executor_cd_prod.yml",
        "executor_cd_dev.yml",
        "miner_cd_prod.yml",
        "miner_cd_dev.yml",
        "validator_cd_prod.yml",
        "validator_cd_dev.yml",
        "watchtower_image.yml",
    }


def _run_publish(
    script: Path, tmp_path: Path, env_extra: dict
) -> tuple[subprocess.CompletedProcess, list[str]]:
    """Run a publish script in a scratch dir with a stub `docker` and a stub `docker_build.sh` / `docker_runner_build.sh`."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    stub = bin_dir / "docker"
    stub.write_text(STUB_DOCKER)
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    for build in ("docker_build.sh", "docker_runner_build.sh"):
        (tmp_path / build).write_text("IMAGE_NAME=daturaai/stub:test\n")
    log = tmp_path / "calls.log"
    log.write_text("")
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "STUB_LOG": str(log),
        "IMAGE_NAME": "daturaai/stub:test",
        "DOCKERHUB_USERNAME": ORG,
        **env_extra,
    }
    proc = subprocess.run(
        ["bash", str(script)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30
    )
    return proc, [line for line in log.read_text().splitlines() if line]


@pytest.mark.parametrize("script", PUBLISH_SCRIPTS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_publish_script_skips_the_login_when_no_token_is_passed(script, tmp_path):
    proc, calls = _run_publish(script, tmp_path, {})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not any(c.startswith("docker login") for c in calls), calls
    assert "docker push daturaai/stub:test" in calls


@pytest.mark.parametrize("script", PUBLISH_SCRIPTS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_publish_script_logs_in_once_with_a_passed_token_and_never_prints_it(script, tmp_path):
    proc, calls = _run_publish(script, tmp_path, {"DOCKERHUB_PAT": "FAKE-TOKEN-VALUE"})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert [c for c in calls if c.startswith("docker login")] == [
        f"docker login -u {ORG} --password-stdin"
    ]
    assert "FAKE-TOKEN-VALUE" not in proc.stdout + proc.stderr, (
        "the token reached the log (set -x trace)"
    )


def test_a_traced_login_prints_the_token(tmp_path):
    """Negative control for the trace assertion: the pre-change shape of the miner script leaks under `set -x`."""
    traced = tmp_path / "traced.sh"
    traced.write_text(
        '#!/bin/bash\nset -eux\nsource ./docker_build.sh\necho "$DOCKERHUB_PAT" | docker login -u "$DOCKERHUB_USERNAME" --password-stdin\ndocker push "$IMAGE_NAME"\n'
    )
    proc, calls = _run_publish(traced, tmp_path, {"DOCKERHUB_PAT": "FAKE-TOKEN-VALUE"})
    assert proc.returncode == 0
    assert "FAKE-TOKEN-VALUE" in proc.stderr
