import ast
from pathlib import Path

import yaml


EXECUTOR_DIR = Path(__file__).resolve().parents[1]
MONITOR_PATH = EXECUTOR_DIR / "src" / "monitor.py"
COMPOSE_PATHS = [
    EXECUTOR_DIR / "docker-compose.app.yml",
    EXECUTOR_DIR / "docker-compose.app.dev.yml",
]


def test_monitor_does_not_import_or_initialize_nvml():
    tree = ast.parse(MONITOR_PATH.read_text())

    for node in ast.walk(tree):
        assert not (
            isinstance(node, ast.Import)
            and any(alias.name == "pynvml" for alias in node.names)
        )
        assert not (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.startswith("pynvml")
        )
        assert not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "nvmlInit"
        )


def test_monitor_service_does_not_reserve_gpu_devices():
    for compose_path in COMPOSE_PATHS:
        compose = yaml.safe_load(compose_path.read_text())
        monitor_service = compose["services"]["monitor"]

        devices = (
            monitor_service.get("deploy", {})
            .get("resources", {})
            .get("reservations", {})
            .get("devices")
        )

        assert devices is None, f"{compose_path.name} monitor service reserves GPU devices"
