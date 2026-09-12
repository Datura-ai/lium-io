"""DAH-3419: the standard stack's updater pulls the runner by a signed digest, not by tag.

99 of 493 nodes sat on an old runner because a Docker Hub mirror on their site kept
serving an old ``runner:latest`` and the tag-based updater (``nickfedor/watchtower``)
took the mirror's answer as current. The updater is now ``../../watchtower`` in this
repository, published as ``daturaai/lium-watchtower``: it reads the digest the validator
signed and pulls ``runner@sha256:…``. These tests hold the compose files to that.
"""

import re
import tomllib
from pathlib import Path

import pytest
import yaml

EXECUTOR_DIR = Path(__file__).resolve().parents[1]
COMPOSE_PATHS = [EXECUTOR_DIR / "docker-compose.yml", EXECUTOR_DIR / "docker-compose.dev.yml"]
WATCHTOWER_PYPROJECT = EXECUTOR_DIR.parents[1] / "watchtower" / "pyproject.toml"
UPDATER_REPOSITORY = "daturaai/lium-watchtower"
# Updaters that resolve a tag through the host's Docker daemon, mirrors included.
TAG_BASED_UPDATERS = ("nickfedor/watchtower", "containrrr/watchtower")
DOCKER_SOCKET_BIND = "/var/run/docker.sock:/var/run/docker.sock"


def _services(compose_path: Path) -> dict:
    return yaml.safe_load(compose_path.read_text())["services"]


def _watchtower_version() -> str:
    with WATCHTOWER_PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)["project"]["version"]


@pytest.mark.parametrize("compose_path", COMPOSE_PATHS, ids=lambda p: p.name)
def test_no_service_updates_by_tag(compose_path):
    """Regression: a tag-based updater comes back into the file. Every image in the file
    is checked, not only the service named ``watchtower``."""
    tag_based = [
        f"{name}: {service['image']}"
        for name, service in _services(compose_path).items()
        if service.get("image", "").split(":")[0].split("@")[0] in TAG_BASED_UPDATERS
    ]
    assert not tag_based, f"pulls by tag through the daemon, so a stale mirror wins: {tag_based}"


@pytest.mark.parametrize("compose_path", COMPOSE_PATHS, ids=lambda p: p.name)
def test_updater_is_lium_watchtower_at_the_source_tree_version(compose_path):
    """Regression: the updater's version in ``watchtower/pyproject.toml`` is bumped and
    published, and the compose files keep pulling the old build; or the compose file
    names a moving tag (``latest``, ``staging``) that a mirror can serve stale. The tag
    is the version from ``pyproject.toml``; the dev stack adds ``-staging``, which is
    the build with the staging digest endpoint baked in."""
    image = _services(compose_path)["watchtower"]["image"]
    repository, _, tag = image.rpartition(":")
    assert repository == UPDATER_REPOSITORY, f"unexpected updater image {image!r}"
    expected_suffix = "-staging" if compose_path.name == "docker-compose.dev.yml" else ""
    assert tag == _watchtower_version() + expected_suffix, (
        f"{compose_path.name} pulls {tag!r}; watchtower/pyproject.toml is {_watchtower_version()!r}"
    )


@pytest.mark.parametrize("compose_path", COMPOSE_PATHS, ids=lambda p: p.name)
def test_updater_has_the_docker_socket_and_only_read_only_binds_beside_it(compose_path):
    """Regression: the socket bind is dropped (the updater cannot recreate the runner and
    logs a connection error every interval), or a host directory is mounted writable
    into a container that only reads (a Docker config for logins, lium-io#1354, is
    read-only)."""
    volumes = _services(compose_path)["watchtower"].get("volumes", [])
    assert DOCKER_SOCKET_BIND in volumes, volumes
    writable = [v for v in volumes if v != DOCKER_SOCKET_BIND and not v.endswith(":ro")]
    assert not writable, f"writable host binds on the updater: {writable}"


@pytest.mark.parametrize("compose_path", COMPOSE_PATHS, ids=lambda p: p.name)
def test_runner_is_named_by_tag_so_the_updater_decides_the_digest(compose_path):
    """Regression: the runner is pinned by digest in the compose file. ``docker compose
    up -d`` would then recreate the runner from that pin on every host restart, undoing
    the updater's work. The compose file names the tag; the digest comes from the
    validator-signed endpoint at run time."""
    image = _services(compose_path)["executor-runner"]["image"]
    assert "@sha256:" not in image, f"runner pinned in compose: {image!r}"
    assert re.fullmatch(r"daturaai/compute-subnet-executor-runner:(latest|dev)", image), image
