"""Event-loop isolation for metrics endpoints (DAH-2391).

A hung or slow metrics collection (psutil / pynvml / docker stats / df inside a
renter container) must never block the event loop: /ping has to stay fast, and
excess concurrent metrics calls have to be rejected instead of piling up.
"""

import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from dependencies.auth import verify_allowed_hotkey_signature, verify_ping_signature
from routes import apis
from routes.apis import apis_router
from services import hardware_service


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(apis_router)
    app.dependency_overrides[verify_allowed_hotkey_signature] = lambda: None
    app.dependency_overrides[verify_ping_signature] = lambda: None
    with TestClient(app) as test_client:
        yield test_client


def test_ping_stays_fast_while_metrics_call_is_slow(client, monkeypatch):
    # Arrange: a metrics collection that blocks its worker thread for seconds
    release = threading.Event()

    def slow_metrics():
        release.wait(5.0)
        return {"cpu": 0.0}

    monkeypatch.setattr(apis, "get_system_metrics", slow_metrics)

    with ThreadPoolExecutor(max_workers=1) as pool:
        metrics_future = pool.submit(client.post, "/hardware_utilization")
        time.sleep(0.3)  # let the metrics request reach the worker thread

        # Act: ping while the metrics call is still blocked
        started = time.monotonic()
        response = client.post("/ping")
        elapsed = time.monotonic() - started

        release.set()
        metrics_response = metrics_future.result()

    # Assert
    assert response.status_code == 200
    assert response.json() == {"status": "pong"}
    assert elapsed < 1.0
    assert metrics_response.status_code == 200


def test_metrics_requests_over_limit_are_rejected_with_503(client, monkeypatch):
    # Arrange: fill every metrics slot with a blocked collection
    release = threading.Event()
    entered = threading.Semaphore(0)

    def blocking_metrics():
        entered.release()
        release.wait(10.0)
        return {"cpu": 0.0}

    monkeypatch.setattr(apis, "get_system_metrics", blocking_metrics)

    with ThreadPoolExecutor(max_workers=apis.METRICS_MAX_CONCURRENT) as pool:
        futures = [
            pool.submit(client.post, "/hardware_utilization")
            for _ in range(apis.METRICS_MAX_CONCURRENT)
        ]
        for _ in range(apis.METRICS_MAX_CONCURRENT):
            assert entered.acquire(timeout=5)

        # Act: one request over the limit
        response = client.post("/hardware_utilization")

        release.set()
        occupied = [future.result() for future in futures]

    # Assert
    assert response.status_code == 503
    assert all(r.status_code == 200 for r in occupied)


def test_metrics_call_timeout_returns_503(client, monkeypatch):
    # Arrange: collection outlives the (shrunk) timeout
    monkeypatch.setattr(apis, "METRICS_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(apis, "get_system_metrics", lambda: time.sleep(2.0))

    # Act
    response = client.post("/hardware_utilization")

    # Assert
    assert response.status_code == 503


def test_filesystem_usage_df_timeout_returns_stale_zeros(monkeypatch):
    # Arrange: df subprocess never returns within the timeout
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="docker exec", timeout=hardware_service.DF_TIMEOUT_SECONDS)

    monkeypatch.setattr(hardware_service.subprocess, "run", fake_run)

    # Act
    result = hardware_service._get_filesystem_usage("container_x", "/")

    # Assert
    assert result == {
        "total": 0,
        "used": 0,
        "available": 0,
        "utilization": 0.0,
        "mount_point": "/",
        "stale": True,
    }


def test_filesystem_usage_parses_df_output(monkeypatch):
    # Arrange: a successful df -k run
    captured = {}

    class FakeResult:
        returncode = 0
        stdout = b"Filesystem 1K-blocks Used Available Use% Mounted on\noverlay 100 40 60 40% /\n"
        stderr = b""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["timeout"] = kwargs.get("timeout")
        return FakeResult()

    monkeypatch.setattr(hardware_service.subprocess, "run", fake_run)

    # Act
    result = hardware_service._get_filesystem_usage("container_x", "/")

    # Assert
    assert result == {"total": 100, "used": 40, "available": 60, "utilization": 40.0, "mount_point": "/"}
    assert captured["cmd"] == ["docker", "exec", "-u", "root", "container_x", "df", "-k", "/"]
    assert captured["timeout"] == hardware_service.DF_TIMEOUT_SECONDS
