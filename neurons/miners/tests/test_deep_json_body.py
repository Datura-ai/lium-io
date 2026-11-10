import os
import subprocess
import sys

import pytest

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))

# Runs in a child process: a C stack overflow kills the child, not the test run.
CHILD_SCRIPT = """
import sys
from unittest.mock import patch

patch("lium_core.shared_config.client.SharedConfigClient._fetch", return_value=None).start()

import asyncio

import miner

depth = 100_000
deep_body = ('{"k":' * depth + "1" + "}" * depth).encode()
scope = {
    "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST", "scheme": "http",
    "path": "/executors", "raw_path": b"/executors", "query_string": b"", "root_path": "",
    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(deep_body)).encode())],
    "client": ("127.0.0.1", 1), "server": ("127.0.0.1", 8000),
}
statuses = []


async def receive():
    return {"type": "http.request", "body": deep_body, "more_body": False}


async def send(message):
    if message["type"] == "http.response.start":
        statuses.append(message["status"])


try:
    asyncio.run(miner.app(scope, receive, send))
except Exception:
    statuses.append(500)
print(sys.getrecursionlimit(), statuses[0])
"""


def test_miner_answers_a_deeply_nested_body_with_4xx(tmp_path):
    # Arrange
    (tmp_path / "tmp").mkdir()  # the test SQLite URI is relative: tmp/test_miner.db
    env = {**os.environ, "PYTHONPATH": SRC_DIR}

    # Act
    result = subprocess.run(
        [sys.executable, "-c", CHILD_SCRIPT], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120
    )

    # Assert
    assert result.returncode == 0, result.stderr[-2000:]
    recursion_limit, status_code = result.stdout.split()[-2:]
    assert recursion_limit == "10000"
    assert 400 <= int(status_code) < 500


@pytest.mark.parametrize("entry_point", ["miner", "connector", "cli"])
def test_every_entry_point_caps_the_recursion_limit(tmp_path, entry_point):
    # Arrange
    (tmp_path / "tmp").mkdir()
    script = (
        'from unittest.mock import patch\n'
        'patch("lium_core.shared_config.client.SharedConfigClient._fetch", return_value=None).start()\n'
        f"import sys, {entry_point}, py_ecc\n"
        "print(sys.getrecursionlimit())\n"
    )

    # Act
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": SRC_DIR},
        capture_output=True,
        text=True,
        timeout=120,
    )

    # Assert
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.split()[-1] == "10000"


@pytest.mark.parametrize("entry_point", ["miner", "connector", "cli"])
def test_startup_survives_a_deeply_nested_shared_config_response(tmp_path, entry_point):
    # Arrange: the first /v1/shared-config fetch runs while core.config is imported, before the entry point's own cap
    (tmp_path / "tmp").mkdir()
    script = (
        "from unittest.mock import patch\n"
        "import requests\n"
        "depth = 100_000\n"
        "response = requests.Response()\n"
        "response.status_code = 200\n"
        "response._content = ('{\"k\":' * depth + '1' + '}' * depth).encode()\n"
        "patch('requests.get', return_value=response).start()\n"
        f"import sys, {entry_point}\n"
        "print(sys.getrecursionlimit())\n"
    )

    # Act
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": SRC_DIR},
        capture_output=True,
        text=True,
        timeout=120,
    )

    # Assert
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.split()[-1] == "10000"
