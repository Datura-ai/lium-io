import json
import sys
import time
from unittest.mock import Mock

import bittensor
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

import vast_api.router
from core.config import settings
from middlewares.miner import request_timeout_seconds, trusted_hotkeys
from vast_api.config import VastSettings
from vast_api.errors import ApiFailure, ApiRefusal
from vast_api.schemas import SetupRequest
from vast_api.wiring import attach_vast_api

# routes are mounted in the executor app behind MinerMiddleware; these tests
# exercise the router itself, so bodies carry unverified auth fields
SIGNED = {"data_to_sign": "d", "signature": "s"}
SETUP = {**SIGNED, "machine_key": "mk-test"}

# throwaway keypair for the sensitive-GET header auth; tests that GET must
# monkeypatch vast_api.router.trusted_hotkeys to trust it
GET_KEYPAIR = bittensor.Keypair.create_from_uri("//VastGetTester")


def _trust_get_keypair(monkeypatch):
    # the package __init__ rebinds `router` to the APIRouter, shadowing the
    # submodule for attribute walks — patch the real module object instead
    monkeypatch.setattr(
        sys.modules["vast_api.router"], "trusted_hotkeys", lambda path: [GET_KEYPAIR.ss58_address]
    )


def _signed_get_headers(path: str, timestamp: int | None = None) -> dict[str, str]:
    timestamp = int(time.time()) if timestamp is None else timestamp
    message = json.dumps({"path": path, "timestamp": timestamp}, sort_keys=True)
    # no 0x prefix on purpose — exercises the normalization path
    signature = GET_KEYPAIR.sign(message).hex()
    return {"x-signature": signature, "x-timestamp": str(timestamp)}


def _build_client(tmp_path):
    settings = VastSettings(RUNS_DIR=str(tmp_path / "runs"))
    app = FastAPI()
    attach_vast_api(app, settings)
    return app, TestClient(app)


def test_healthz_open_get(tmp_path):
    _, client = _build_client(tmp_path)

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert response.json()["version"]


def test_setup_body_requires_auth_fields_and_machine_key():
    # the middleware skips GETs; the mutating body must carry the signature
    # contract AND the backend-minted machine_key (plan-key-split)
    with pytest.raises(ValidationError):
        SetupRequest(data_to_sign="d", signature="s")  # no machine_key
    with pytest.raises(ValidationError):
        SetupRequest(machine_key="mk")  # no auth fields


def test_setup_without_machine_key_is_422(tmp_path):
    _, client = _build_client(tmp_path)

    response = client.post("/vast/setup", json=SIGNED)

    assert response.status_code == 422


def test_vast_paths_get_stretched_middleware_timeout():
    assert request_timeout_seconds("/vast/setup") == 180
    assert request_timeout_seconds("/upload_ssh_key") == 30


def test_vast_admin_hotkey_trusted_only_on_vast_paths():
    assert settings.VAST_ADMIN_HOTKEY in trusted_hotkeys("/vast/setup")
    assert settings.VAST_ADMIN_HOTKEY not in trusted_hotkeys("/upload_ssh_key")
    assert settings.MINER_HOTKEY_SS58_ADDRESS in trusted_hotkeys("/upload_ssh_key")


def test_setup_returns_202_and_run_id(tmp_path):
    app, client = _build_client(tmp_path)
    app.state.vast_manager = Mock(setup=Mock(return_value="setup-20260811-093015"))

    response = client.post("/vast/setup", json=SETUP)

    assert response.status_code == 202
    assert response.json() == {"run_id": "setup-20260811-093015"}


def test_refusal_returns_409_error_envelope(tmp_path):
    app, client = _build_client(tmp_path)
    app.state.vast_manager = Mock(setup=Mock(side_effect=ApiRefusal("run_in_progress", "busy")))

    response = client.post("/vast/setup", json=SETUP)

    assert response.status_code == 409
    assert response.json() == {"error": {"code": "run_in_progress", "detail": "busy"}}


def test_failure_returns_500_error_envelope(tmp_path, monkeypatch):
    app, client = _build_client(tmp_path)
    app.state.vast_manager = Mock(status=Mock(side_effect=ApiFailure("host_command_failed", "boom")))
    _trust_get_keypair(monkeypatch)

    response = client.get("/vast/status", headers=_signed_get_headers("/vast/status"))

    assert response.status_code == 500
    assert response.json() == {"error": {"code": "host_command_failed", "detail": "boom"}}


def test_sensitive_get_without_headers_is_422(tmp_path):
    _, client = _build_client(tmp_path)

    for path in ("/vast/status", "/vast/runs", "/vast/runs/some-run"):
        assert client.get(path).status_code == 422


def test_sensitive_get_wrong_signature_is_401(tmp_path, monkeypatch):
    _, client = _build_client(tmp_path)
    _trust_get_keypair(monkeypatch)
    headers = _signed_get_headers("/vast/status")
    headers["x-signature"] = "00" * 64

    response = client.get("/vast/status", headers=headers)

    assert response.status_code == 401


def test_sensitive_get_stale_timestamp_is_401(tmp_path, monkeypatch):
    _, client = _build_client(tmp_path)
    _trust_get_keypair(monkeypatch)
    headers = _signed_get_headers("/vast/status", timestamp=int(time.time()) - 301)

    response = client.get("/vast/status", headers=headers)

    assert response.status_code == 401


def test_sensitive_get_signature_bound_to_path(tmp_path, monkeypatch):
    # a signature minted for /vast/status must not open /vast/runs
    _, client = _build_client(tmp_path)
    _trust_get_keypair(monkeypatch)

    response = client.get("/vast/runs", headers=_signed_get_headers("/vast/status"))

    assert response.status_code == 401


def test_sensitive_get_valid_signature_is_200(tmp_path, monkeypatch):
    _, client = _build_client(tmp_path)
    _trust_get_keypair(monkeypatch)

    response = client.get("/vast/runs", headers=_signed_get_headers("/vast/runs"))

    assert response.status_code == 200
    assert response.json() == []


def test_run_doc_args_never_contain_machine_key(tmp_path, monkeypatch):
    # run docs are readable via GET /vast/runs — the key must not land there
    app, client = _build_client(tmp_path)
    manager = app.state.vast_manager
    manager.docker_ops = Mock()
    manager.docker_ops.get_vast_uns.return_value = None
    manager.host = Mock()
    _trust_get_keypair(monkeypatch)

    response = client.post("/vast/setup", json={**SETUP, "machine_id": 147063})

    assert response.status_code == 202
    runs = client.get("/vast/runs", headers=_signed_get_headers("/vast/runs")).json()
    assert runs and "mk-test" not in str(runs)
    assert runs[0]["args"] == {"machine_id": 147063}


def _wire_rental_mocks(app, rental_containers: list[str]):
    manager = app.state.vast_manager
    manager.docker_ops = Mock()
    manager.docker_ops.get_vast_uns.return_value = Mock()  # vast-uns exists
    manager.docker_ops.vast_rental_containers.return_value = rental_containers
    manager.docker_ops.delete_vast_uns.return_value = True
    manager.host = Mock()
    return manager


def test_setup_refused_while_rental_running(tmp_path):
    app, client = _build_client(tmp_path)
    _wire_rental_mocks(app, ["C.12345678"])

    response = client.post("/vast/setup", json=SETUP)

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "rental_running"


def test_setup_force_overrides_rental_rail(tmp_path):
    app, client = _build_client(tmp_path)
    _wire_rental_mocks(app, ["C.12345678"])

    response = client.post("/vast/setup", json={**SETUP, "force": True})

    assert response.status_code == 202
    assert response.json()["run_id"].startswith("setup-")


def test_setup_fresh_box_without_vast_uns_still_202(tmp_path):
    app, client = _build_client(tmp_path)
    manager = app.state.vast_manager
    manager.docker_ops = Mock()
    manager.docker_ops.get_vast_uns.return_value = None
    manager.host = Mock()

    response = client.post("/vast/setup", json=SETUP)

    assert response.status_code == 202


def _set_renting_config(monkeypatch, port_range: str | None, port_mappings: str | None):
    monkeypatch.setattr("vast_api.service.executor_settings.RENTING_PORT_RANGE", port_range)
    monkeypatch.setattr("vast_api.service.executor_settings.RENTING_PORT_MAPPINGS", port_mappings)


def test_setup_refused_on_renting_port_range_overlap(tmp_path, monkeypatch):
    app, client = _build_client(tmp_path)
    _set_renting_config(monkeypatch, "39990-40010", None)

    response = client.post("/vast/setup", json=SETUP)

    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "port_range_overlap"
    assert "40000" in error["detail"]


def test_setup_refused_on_renting_port_mappings_overlap(tmp_path, monkeypatch):
    # the EXTERNAL side of a mapping is what the host publishes
    app, client = _build_client(tmp_path)
    _set_renting_config(monkeypatch, None, "[[46681, 40100], [46682, 56682]]")

    response = client.post("/vast/setup", json=SETUP)

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "port_range_overlap"


def test_setup_disjoint_renting_ports_pass(tmp_path, monkeypatch):
    app, client = _build_client(tmp_path)
    manager = app.state.vast_manager
    manager.docker_ops = Mock()
    manager.docker_ops.get_vast_uns.return_value = None
    manager.host = Mock()
    _set_renting_config(monkeypatch, "20000-30000", None)

    response = client.post("/vast/setup", json=SETUP)

    assert response.status_code == 202


def test_setup_unset_renting_ports_pass(tmp_path, monkeypatch):
    app, client = _build_client(tmp_path)
    manager = app.state.vast_manager
    manager.docker_ops = Mock()
    manager.docker_ops.get_vast_uns.return_value = None
    manager.host = Mock()
    _set_renting_config(monkeypatch, None, None)

    response = client.post("/vast/setup", json=SETUP)

    assert response.status_code == 202


def test_setup_with_never_started_vast_uns_still_202(tmp_path):
    # created = the container never ran (e.g. failed port bind at docker start):
    # the nested dockerd never lived, so no rental can exist — recovery must not
    # demand force
    app, client = _build_client(tmp_path)
    manager = app.state.vast_manager
    manager.docker_ops = Mock()
    manager.docker_ops.get_vast_uns.return_value = Mock(status="created")
    manager.host = Mock()

    response = client.post("/vast/setup", json=SETUP)

    assert response.status_code == 202
    manager.docker_ops.vast_rental_containers.assert_not_called()


def test_setup_unreachable_nested_docker_is_500_not_silent_pass(tmp_path):
    # vast-uns exists but nested docker cannot answer — the rail cannot prove
    # "no rental", so it must refuse by failure rather than proceed
    app, client = _build_client(tmp_path)
    manager = _wire_rental_mocks(app, [])
    manager.docker_ops.vast_rental_containers.side_effect = ApiFailure(
        "host_command_failed", "nested docker ps failed"
    )

    response = client.post("/vast/setup", json=SETUP)

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "host_command_failed"


def test_delete_refused_while_rental_running(tmp_path):
    app, client = _build_client(tmp_path)
    manager = _wire_rental_mocks(app, ["C.12345678"])

    response = client.request("DELETE", "/vast", json=SIGNED)

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "rental_running"
    manager.docker_ops.delete_vast_uns.assert_not_called()


def test_delete_rental_running_overridden_with_force(tmp_path):
    app, client = _build_client(tmp_path)
    manager = _wire_rental_mocks(app, ["C.12345678"])

    response = client.request("DELETE", "/vast", json={**SIGNED, "force": True})

    assert response.status_code == 200
    manager.docker_ops.delete_vast_uns.assert_called_once()


def test_delete_refused_while_run_in_progress(tmp_path):
    app, client = _build_client(tmp_path)
    manager = _wire_rental_mocks(app, [])
    manager.runs.create("setup", {}, ["a"])  # a live run doc in the real store

    response = client.request("DELETE", "/vast", json=SIGNED)

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "run_in_progress"
    manager.docker_ops.delete_vast_uns.assert_not_called()


def test_delete_purge_removes_state(tmp_path):
    app, client = _build_client(tmp_path)
    manager = _wire_rental_mocks(app, [])

    response = client.request("DELETE", "/vast", json={**SIGNED, "purge": True})

    assert response.status_code == 200
    assert response.json() == {"container_removed": True, "state_removed": True}
    rm_calls = [c for c in manager.host.run.call_args_list if c.args[0][:2] == ["rm", "-rf"]]
    assert len(rm_calls) == 1
    # the nested dockerd data-root goes too, or old contracts resurrect on re-setup
    manager.host.purge_data_root.assert_called_once()


def test_delete_plain_keeps_state(tmp_path):
    app, client = _build_client(tmp_path)
    manager = _wire_rental_mocks(app, [])

    response = client.request("DELETE", "/vast", json=SIGNED)

    assert response.status_code == 200
    assert response.json() == {"container_removed": True, "state_removed": False}
    rm_calls = [c for c in manager.host.run.call_args_list if c.args[0][:2] == ["rm", "-rf"]]
    assert rm_calls == []
    manager.host.purge_data_root.assert_not_called()


def _wire_status_mocks(app):
    manager = app.state.vast_manager
    manager.host = Mock()
    manager.host.run.return_value = Mock(stdout="147063\n")
    manager.host.gpu_stats.return_value = [{"idx": 0, "mem_mib": 1, "util_pct": 0}]
    manager.docker_ops = Mock()
    manager.docker_ops.get_vast_uns.return_value = Mock()
    manager.docker_ops.vast_rental_containers.return_value = ["C.777"]
    manager.docker_ops.executor_health.return_value = "healthy"
    manager.docker_ops.filler_running.return_value = False
    manager.docker_ops.exec_in_uns.return_value = (0, "active\n")
    return manager


def test_status_reports_local_sections_only(tmp_path, monkeypatch):
    app, client = _build_client(tmp_path)
    _wire_status_mocks(app)
    _trust_get_keypair(monkeypatch)

    response = client.get("/vast/status", headers=_signed_get_headers("/vast/status"))

    assert response.status_code == 200
    doc = response.json()
    assert doc["machine_id"] == 147063
    assert doc["machine_id_error"] is None
    assert doc["rental_containers"] == ["C.777"]
    assert doc["box"]["kaalia_active"] is True
    assert "vast" not in doc and "offers" not in doc  # account-key sections are gone


def test_status_machine_id_absent_is_null_without_error(tmp_path, monkeypatch):
    app, client = _build_client(tmp_path)
    manager = _wire_status_mocks(app)
    manager.host.run.return_value = Mock(stdout="")  # cat rc!=0 → empty output
    _trust_get_keypair(monkeypatch)

    response = client.get("/vast/status", headers=_signed_get_headers("/vast/status"))

    assert response.status_code == 200
    doc = response.json()
    assert doc["machine_id"] is None
    assert doc["machine_id_error"] is None


def test_status_broken_machine_id_read_reports_error_slot(tmp_path, monkeypatch):
    # a read that raises (e.g. nsenter timeout) must not 500 the whole status
    app, client = _build_client(tmp_path)
    manager = _wire_status_mocks(app)
    manager.host.run.side_effect = ApiFailure("host_command_failed", "cat timed out after 300s")
    _trust_get_keypair(monkeypatch)

    response = client.get("/vast/status", headers=_signed_get_headers("/vast/status"))

    assert response.status_code == 200
    doc = response.json()
    assert doc["machine_id"] is None
    assert doc["machine_id_error"] == "host_command_failed: cat timed out after 300s"
