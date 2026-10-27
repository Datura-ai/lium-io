"""DAH-3419: the standard stack's updater pulls the runner by a signed digest, not by tag.

99 of 493 nodes sat on an old runner because a Docker Hub mirror on their site kept
serving an old ``runner:latest`` and the tag-based updater (``nickfedor/watchtower``)
took the mirror's answer as current. The updater is now ``../../watchtower`` in this
repository, published as ``daturaai/lium-watchtower``: it reads the digest the validator
signed and pulls ``runner@sha256:…``. These tests hold the compose files to that.
"""

from pathlib import Path

import pytest
import yaml

EXECUTOR_DIR = Path(__file__).resolve().parents[1]
COMPOSE_PATHS = [EXECUTOR_DIR / "docker-compose.yml", EXECUTOR_DIR / "docker-compose.dev.yml"]
# Updaters that resolve a tag through the host's Docker daemon, mirrors included.
TAG_BASED_UPDATERS = ("nickfedor/watchtower", "containrrr/watchtower")


def _services(compose_path: Path) -> dict:
    return yaml.safe_load(compose_path.read_text())["services"]


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


