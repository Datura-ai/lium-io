import subprocess
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from middlewares.miner import request_timeout_seconds
from vast_api.config import VastSettings
from vast_api.errors import ApiFailure, ApiRefusal
from vast_api.schemas import SetupRequest
from vast_api.vast_client import VastClient
from vast_api.wiring import attach_vast_api

# routes are mounted in the executor app behind MinerMiddleware; these tests
# exercise the router itself, so bodies carry unverified auth fields
SIGNED = {"data_to_sign": "d", "signature": "s"}


def _build_client(tmp_path):
    settings = VastSettings(RUNS_DIR=str(tmp_path / "runs"))
    app = FastAPI()
    attach_vast_api(app, settings)
    return app, TestClient(app)


def _host_with_key(key: str = "supersecretkey123"):
    host = Mock()
    host.run.return_value = Mock(returncode=0, stdout=key + "\n")
    return host


def test_healthz_open_get(tmp_path):
    _, client = _build_client(tmp_path)

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert response.json()["version"]


def test_mutating_body_requires_auth_fields():
    # the middleware skips GETs; every mutating body must carry the signature contract
    with pytest.raises(ValidationError):
        SetupRequest()


def test_setup_without_auth_fields_is_422(tmp_path):
    _, client = _build_client(tmp_path)

    response = client.post("/vast/setup", json={})

    assert response.status_code == 422


def test_vast_paths_get_stretched_middleware_timeout():
    assert request_timeout_seconds("/vast/list") == 180
    assert request_timeout_seconds("/upload_ssh_key") == 30


def test_setup_returns_202_and_run_id(tmp_path):
    app, client = _build_client(tmp_path)
    app.state.vast_manager = Mock(setup=Mock(return_value="setup-20260811-093015"))

    response = client.post("/vast/setup", json=SIGNED)

    assert response.status_code == 202
    assert response.json() == {"run_id": "setup-20260811-093015"}


def test_refusal_returns_409_error_envelope(tmp_path):
    app, client = _build_client(tmp_path)
    app.state.vast_manager = Mock(setup=Mock(side_effect=ApiRefusal("run_in_progress", "busy")))

    response = client.post("/vast/setup", json=SIGNED)

    assert response.status_code == 409
    assert response.json() == {"error": {"code": "run_in_progress", "detail": "busy"}}


def _wire_market_mocks(app, gpu_mem_mib: int = 1):
    manager = app.state.vast_manager
    manager.vast = Mock()
    manager.vast.get_machines.return_value = [{"id": 147139}]
    manager.vast.search_offers.return_value = [{"id": 47437865}]
    manager.host = Mock()
    manager.host.run.return_value = Mock(stdout="")  # no persisted numeric_machine_id
    manager.host.gpu_stats.return_value = [{"idx": 0, "mem_mib": gpu_mem_mib, "util_pct": 0}]
    manager.host.gpu_compute_apps.return_value = []
    return manager


def test_list_calls_unlist_then_list(tmp_path):
    app, client = _build_client(tmp_path)
    manager = _wire_market_mocks(app)

    response = client.post("/vast/list", json={**SIGNED, "price_gpu_usd": 1.5})

    assert response.status_code == 200
    assert response.json()["listed"] is True
    call_names = [name for name, _, _ in manager.vast.mock_calls]
    assert call_names.index("unlist_machine") < call_names.index("list_machine")


def test_list_gpu_busy_refused_without_force(tmp_path):
    app, client = _build_client(tmp_path)
    manager = _wire_market_mocks(app, gpu_mem_mib=70000)

    response = client.post("/vast/list", json={**SIGNED, "price_gpu_usd": 1.5})

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "gpu_busy"
    manager.vast.unlist_machine.assert_not_called()


def test_list_gpu_busy_overridden_with_force(tmp_path):
    app, client = _build_client(tmp_path)
    manager = _wire_market_mocks(app, gpu_mem_mib=70000)

    response = client.post("/vast/list", json={**SIGNED, "price_gpu_usd": 1.5, "force": True})

    assert response.status_code == 200
    manager.vast.unlist_machine.assert_called_once()


def test_account_key_never_leaks_on_http_error(tmp_path, monkeypatch):
    settings = VastSettings(RUNS_DIR=str(tmp_path / "runs"))
    vast = VastClient(settings, _host_with_key())
    mock_get = Mock(return_value=Mock(ok=False, status_code=401))
    monkeypatch.setattr("vast_api.vast_client.requests.get", mock_get)

    with pytest.raises(ApiFailure) as exc_info:
        vast.get_machines()

    assert exc_info.value.code == "vast_api_failed"
    assert "supersecretkey123" not in str(exc_info.value)
    args, kwargs = mock_get.call_args
    assert "supersecretkey123" not in args[0]  # key never in the URL
    assert "params" not in kwargs
    assert kwargs["headers"]["Authorization"] == "Bearer supersecretkey123"


def test_missing_host_key_is_vast_key_missing(tmp_path):
    settings = VastSettings(RUNS_DIR=str(tmp_path / "runs"))
    host = Mock()
    host.run.return_value = Mock(returncode=1, stdout="")
    vast = VastClient(settings, host)

    with pytest.raises(ApiFailure) as exc_info:
        vast.get_machines()

    assert exc_info.value.code == "vast_key_missing"


def test_setup_refused_while_rental_running(tmp_path):
    app, client = _build_client(tmp_path)
    manager = app.state.vast_manager
    manager.vast = Mock()
    manager.vast.get_machines.return_value = [{"id": 147139, "current_rentals_running": 1}]
    manager.host = Mock()
    manager.host.run.return_value = Mock(stdout="")

    response = client.post("/vast/setup", json=SIGNED)

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "rental_running"


def test_setup_force_overrides_rental_rail(tmp_path):
    app, client = _build_client(tmp_path)
    manager = app.state.vast_manager
    manager.vast = Mock()
    manager.vast.get_machines.return_value = [{"id": 147139, "current_rentals_running": 1}]
    manager.host = Mock()
    manager.host.run.return_value = Mock(stdout="")
    manager.docker_ops = Mock()

    response = client.post("/vast/setup", json={**SIGNED, "force": True})

    assert response.status_code == 202
    assert response.json()["run_id"].startswith("setup-")


def test_delete_refused_when_machine_unresolved(tmp_path):
    app, client = _build_client(tmp_path)
    manager = app.state.vast_manager
    manager.vast = Mock()
    manager.vast.get_machines.return_value = [
        {"id": 1, "public_ipaddr": "10.0.0.1"},
        {"id": 2, "public_ipaddr": ""},
    ]
    manager.host = Mock()
    manager.host.run.return_value = Mock(stdout="")
    manager.host.local_ips.return_value = ["192.0.2.7"]
    manager.docker_ops = Mock()

    response = client.request("DELETE", "/vast", json=SIGNED)

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "machine_unresolved"
    manager.docker_ops.delete_vast_uns.assert_not_called()


def test_delete_proceeds_when_account_has_no_machines(tmp_path):
    app, client = _build_client(tmp_path)
    manager = app.state.vast_manager
    manager.vast = Mock()
    manager.vast.get_machines.return_value = []
    manager.docker_ops = Mock()
    manager.docker_ops.delete_vast_uns.return_value = True

    response = client.request("DELETE", "/vast", json=SIGNED)

    assert response.status_code == 200
    assert response.json() == {
        "container_removed": True,
        "state_removed": False,
        "vast_record_deleted": False,
    }


def test_search_offers_cli_line_has_any_flags(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))  # ~/.vast_api_key lands in tmp
    settings = VastSettings(RUNS_DIR=str(tmp_path / "runs"))
    vast = VastClient(settings, _host_with_key("k123"))
    captured = {}

    def fake_run(cmd, capture_output, text, timeout):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="[]", stderr="")

    monkeypatch.setattr("vast_api.vast_client.subprocess.run", fake_run)

    offers = vast.search_offers(147139)

    assert offers == []
    assert "machine_id=147139 rentable=any rented=any verified=any" in captured["cmd"]
    assert "--api-key" not in captured["cmd"]  # key must never reach argv
    assert (tmp_path / ".vast_api_key").read_text().strip() == "k123"
    assert (tmp_path / ".vast_api_key").stat().st_mode & 0o777 == 0o600


def test_list_offers_never_appear_fails_listing_not_confirmed(tmp_path, monkeypatch):
    app, client = _build_client(tmp_path)
    manager = _wire_market_mocks(app)
    manager.vast.search_offers.return_value = []
    monkeypatch.setattr("vast_api.service.time.sleep", Mock())

    response = client.post("/vast/list", json={**SIGNED, "price_gpu_usd": 1.5})

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "listing_not_confirmed"


def test_failure_returns_500_error_envelope(tmp_path):
    app, client = _build_client(tmp_path)
    app.state.vast_manager = Mock(status=Mock(side_effect=ApiFailure("vast_api_failed", "boom")))

    response = client.get("/vast/status")

    assert response.status_code == 500
    assert response.json() == {"error": {"code": "vast_api_failed", "detail": "boom"}}


def _wire_delete_mocks(app, machine: dict):
    manager = app.state.vast_manager
    manager.vast = Mock()
    manager.vast.get_machines.return_value = [machine]
    manager.host = Mock()
    manager.host.run.return_value = Mock(stdout="")
    manager.docker_ops = Mock()
    manager.docker_ops.delete_vast_uns.return_value = True
    return manager


def test_delete_refused_while_rental_running(tmp_path):
    app, client = _build_client(tmp_path)
    manager = _wire_delete_mocks(app, {"id": 147139, "current_rentals_running": 1})

    response = client.request("DELETE", "/vast", json=SIGNED)

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "rental_running"
    manager.docker_ops.delete_vast_uns.assert_not_called()


def test_delete_rental_running_overridden_with_force(tmp_path):
    app, client = _build_client(tmp_path)
    manager = _wire_delete_mocks(app, {"id": 147139, "current_rentals_running": 1})

    response = client.request("DELETE", "/vast", json={**SIGNED, "force": True})

    assert response.status_code == 200
    manager.docker_ops.delete_vast_uns.assert_called_once()


def test_delete_purge_removes_state_and_vast_record(tmp_path):
    app, client = _build_client(tmp_path)
    manager = _wire_delete_mocks(app, {"id": 147139})

    response = client.request("DELETE", "/vast", json={**SIGNED, "purge": True})

    assert response.status_code == 200
    assert response.json()["state_removed"] is True
    assert response.json()["vast_record_deleted"] is True
    manager.vast.delete_machine.assert_called_once_with(147139)


def test_delete_plain_keeps_state_and_vast_record(tmp_path):
    app, client = _build_client(tmp_path)
    manager = _wire_delete_mocks(app, {"id": 147139})

    response = client.request("DELETE", "/vast", json=SIGNED)

    assert response.status_code == 200
    assert response.json()["state_removed"] is False
    assert response.json()["vast_record_deleted"] is False
    manager.vast.delete_machine.assert_not_called()
    rm_calls = [c for c in manager.host.run.call_args_list if c.args[0][:2] == ["rm", "-rf"]]
    assert rm_calls == []


def test_setup_fresh_box_without_machines_still_202(tmp_path):
    app, client = _build_client(tmp_path)
    manager = app.state.vast_manager
    manager.vast = Mock()
    manager.vast.get_machines.return_value = []
    manager.host = Mock()
    manager.docker_ops = Mock()

    response = client.post("/vast/setup", json=SIGNED)

    assert response.status_code == 202


def test_list_refused_while_run_in_progress(tmp_path):
    app, client = _build_client(tmp_path)
    manager = _wire_market_mocks(app)
    manager.runs.create("setup", {}, ["a"])  # a live run doc in the real store

    response = client.post("/vast/list", json={**SIGNED, "price_gpu_usd": 1.5})

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "run_in_progress"
    manager.vast.unlist_machine.assert_not_called()


def test_list_run_in_progress_overridden_with_force(tmp_path):
    app, client = _build_client(tmp_path)
    manager = _wire_market_mocks(app)
    manager.runs.create("setup", {}, ["a"])

    response = client.post("/vast/list", json={**SIGNED, "price_gpu_usd": 1.5, "force": True})

    assert response.status_code == 200
    manager.vast.unlist_machine.assert_called_once()


def test_unlist_force_bypasses_run_in_progress(tmp_path):
    # the product's critical path: a Lium rent arriving mid-run must still unlist
    app, client = _build_client(tmp_path)
    manager = _wire_market_mocks(app)
    manager.runs.create("setup", {}, ["a"])

    refused = client.post("/vast/unlist", json=SIGNED)
    forced = client.post("/vast/unlist", json={**SIGNED, "force": True})

    assert refused.status_code == 409
    assert refused.json()["error"]["code"] == "run_in_progress"
    assert forced.status_code == 200
    manager.vast.unlist_machine.assert_called_once()
