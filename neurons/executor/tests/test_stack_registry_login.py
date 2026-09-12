"""DAH-3376: the standard stack passes the host's `docker login` to Watchtower and the runner.

Anonymous Docker Hub pulls are counted per source IP; on a site where many nodes share one
egress the quota is exhausted and Watchtower's pull of the runner never succeeds. Both
containers therefore mount the host's Docker config directory read-only: Watchtower reads
``$DOCKER_CONFIG/config.json``, the runner's docker CLI reads ``/root/.docker/config.json``.
"""

from pathlib import Path

import pytest
import yaml

EXECUTOR_DIR = Path(__file__).resolve().parents[1]
COMPOSE_PATHS = [EXECUTOR_DIR / "docker-compose.yml", EXECUTOR_DIR / "docker-compose.dev.yml"]


def _volumes(service: dict) -> dict[str, str]:
    """Map host path -> '<container path>:<mode>' for the service's short-syntax volumes."""
    mapping: dict[str, str] = {}
    for entry in service["volumes"]:
        host, _, rest = entry.partition(":")
        mapping[host] = rest
    return mapping


def _env(service: dict) -> dict[str, str]:
    """Map NAME -> value for the service's environment, list or map syntax."""
    env = service.get("environment", {})
    if isinstance(env, dict):
        return {key: str(value) for key, value in env.items()}
    return dict(entry.partition("=")[::2] for entry in env)


@pytest.mark.parametrize("compose_path", COMPOSE_PATHS, ids=lambda p: p.name)
def test_both_services_read_the_same_host_docker_login_read_only(compose_path):
    """Regression: one of the two services stops mounting the host's Docker config (Watchtower pulls
    anonymously again and the shared-egress quota bites), the mount goes read-write (a container can rewrite
    the host's credentials), or the mount lands where the process does not read its config — Watchtower
    reads `$DOCKER_CONFIG/config.json`, the runner's CLI reads `/root/.docker/config.json` unless
    DOCKER_CONFIG says otherwise — so the login is mounted but never used. Every assertion is a relation
    between two parts of the file, not the value of one."""
    compose = yaml.safe_load(compose_path.read_text())
    watchtower, runner = compose["services"]["watchtower"], compose["services"]["executor-runner"]

    login_mounts = {}
    for name, service in (("watchtower", watchtower), ("executor-runner", runner)):
        docker_config = [
            (host, rest) for host, rest in _volumes(service).items() if host.endswith("/.docker")
        ]
        assert len(docker_config) == 1, f"{name} mounts {len(docker_config)} docker-config dirs"
        ((host, rest),) = docker_config
        target, _, mode = rest.rpartition(":")
        assert mode == "ro", f"{name} could rewrite the host's credentials ({host} -> {rest})"
        login_mounts[name] = (host, target)

    assert login_mounts["watchtower"][0] == login_mounts["executor-runner"][0], (
        "the two services read different logins"
    )
    # each process must find the file where it looks for it
    assert _env(watchtower)["DOCKER_CONFIG"] == login_mounts["watchtower"][1]
    assert _env(runner).get("DOCKER_CONFIG", "/root/.docker") == login_mounts["executor-runner"][1]
    # a `currentContext` in the host's config.json must not redirect the runner's CLI away from the daemon
    # whose socket is mounted
    socket_target = _volumes(runner)["/var/run/docker.sock"].partition(":")[0]
    assert _env(runner).get("DOCKER_HOST") == f"unix://{socket_target}"
