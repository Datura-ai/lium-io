"""DAH-3376: the standard stack passes the host's `docker login` to Watchtower and the runner.

Anonymous Docker Hub pulls are counted per source IP; on a site where many nodes share one
egress the quota is exhausted and Watchtower's pull of the runner never succeeds. Both
containers therefore mount the host's Docker config directory read-only: Watchtower reads
``$DOCKER_CONFIG/config.json``, the runner's docker CLI reads ``/root/.docker/config.json``.
"""

from pathlib import Path

import yaml

COMPOSE_PATH = Path(__file__).resolve().parents[1] / "docker-compose.yml"
HOST_DOCKER_CONFIG = "$HOME/.docker"


def _volumes(service: dict) -> dict[str, str]:
    """Map host path -> '<container path>:<mode>' for the service's short-syntax volumes."""
    mapping: dict[str, str] = {}
    for entry in service["volumes"]:
        host, _, rest = entry.partition(":")
        mapping[host] = rest
    return mapping


def test_watchtower_reads_host_docker_config_read_only():
    compose = yaml.safe_load(COMPOSE_PATH.read_text())
    watchtower = compose["services"]["watchtower"]

    volumes = _volumes(watchtower)
    assert volumes[HOST_DOCKER_CONFIG] == "/config:ro"
    assert "DOCKER_CONFIG=/config" in watchtower["environment"]


def test_runner_reads_host_docker_config_read_only():
    compose = yaml.safe_load(COMPOSE_PATH.read_text())
    runner = compose["services"]["executor-runner"]

    volumes = _volumes(runner)
    assert volumes[HOST_DOCKER_CONFIG] == "/root/.docker:ro"
    # the runner's own pieces are untouched: socket, wallets, .env, watchtower label
    assert volumes["/var/run/docker.sock"] == "/var/run/docker.sock"
    assert volumes["./.env"] == "/root/executor/.env"
    assert "com.centurylinklabs.watchtower.enable=true" in runner["labels"]
