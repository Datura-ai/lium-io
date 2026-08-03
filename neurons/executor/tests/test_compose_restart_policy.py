from pathlib import Path

import pytest
import yaml


EXECUTOR_DIR = Path(__file__).resolve().parents[1]
COMPOSE_PATHS = [
    EXECUTOR_DIR / "docker-compose.app.yml",
    EXECUTOR_DIR / "docker-compose.app.dev.yml",
]


@pytest.mark.parametrize("compose_path", COMPOSE_PATHS, ids=lambda path: path.name)
def test_database_restarts_after_host_reboot(compose_path: Path) -> None:
    compose = yaml.safe_load(compose_path.read_text())

    assert compose["services"]["db"]["restart"] == "always"
