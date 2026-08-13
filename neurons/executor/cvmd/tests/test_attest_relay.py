"""DAH-2675 — the attest relay route and its scope.

What is pinned here:

- the route is open to any authorized key and closed to strangers;
- a key in the new `attest` scope can request a quote and NOTHING else that mutates;
- the relay refuses by name (wrong state, no forward, agent unreachable) and otherwise
  passes the agent's answer through verbatim, status and all;
- the dial-address rule that makes a loopback-bound forward reachable from the host.

The agent itself is faked at the `agent_relay.relay_attest` seam: these tests are about the
route's decisions, and the real dial function is three lines of urllib whose behavior is
pinned by `dial_address` and `_decode` tests below.
"""

import json

import pytest
from bittensor.sp_core import Keypair
from cvmd import agent_relay
from cvmd.agent_relay import AgentRelayError, dial_address
from cvmd.app import create_app
from cvmd.auth.clients import AuthorizedClientsError, Scope, load_authorized_clients
from cvmd.config import Config
from cvmd.cvm.instance import Instance, PortReport
from cvmd.cvm.manager import KIND_RENTER
from cvmd.state.machine import NodeState
from fastapi.testclient import TestClient
from tests.conftest import signed_request

ATTEST_URI = "//Ferdie"
NONCE = "ab" * 32
AGENT_ANSWER = {
    "version": "0.1.0",
    "report_data": "cd" * 64,
    "tls_public_key": "ee" * 32,
    "gpu_uuids": ["GPU-1"],
    "gpu_uuid_digest": "ff" * 32,
    "quote": '{"quote": "raw"}',
}


@pytest.fixture
def attest_key() -> Keypair:
    return Keypair.create_from_uri(ATTEST_URI)


@pytest.fixture
def clients_file_with_attest(tmp_path, validator_key, platform_key, attest_key):
    path = tmp_path / "authorized_clients.json"
    path.write_text(
        json.dumps(
            [
                {"hotkey": validator_key.ss58_address, "scope": "validation"},
                {"hotkey": platform_key.ss58_address, "scope": "renter"},
                {"hotkey": attest_key.ss58_address, "scope": "attest"},
            ]
        )
    )
    return path


@pytest.fixture
def app(clients_file_with_attest, state_dir):
    config = Config(authorized_clients=clients_file_with_attest, state_dir=state_dir)
    return create_app(config)


@pytest.fixture
def client(app) -> TestClient:
    return TestClient(app)


def make_renter(app, *, ports: list[PortReport] | None = None) -> Instance:
    """Put the app under a renter CVM, the shape reconciliation would have adopted."""
    default_ports = [
        PortReport(protocol="tcp", address="127.0.0.1", host_port=18451, guest_port=8451)
    ]
    instance = Instance(
        instance_id="cvm-renter-1",
        kind=KIND_RENTER,
        artifact_id="renter-abc",
        vm_dir="/nonexistent/vm",
        supervisor_pid=1,
        created_at="2026-08-13T00:00:00+00:00",
        qemu="10.1.0",
        os_image_hash="a" * 64,
        compose_hash="b" * 64,
        ports=default_ports if ports is None else ports,
    )
    app.state.instances.set(instance)
    app.state.store.transition(NodeState.RENTER_RUNNING)
    return instance


def attest(client, keypair, payload: dict):
    return signed_request(client, keypair, "POST", "/v1/attest", body=json.dumps(payload).encode())


class TestScope:
    def test_a_stranger_is_refused_before_anything_else(self, client, stranger_key, app):
        make_renter(app)
        assert attest(client, stranger_key, {"nonce": NONCE}).status_code == 401

    def test_every_authorized_scope_may_ask(
        self, client, app, validator_key, platform_key, attest_key, monkeypatch
    ):
        make_renter(app)
        monkeypatch.setattr(agent_relay, "relay_attest", lambda **kwargs: (200, AGENT_ANSWER))
        for keypair in (validator_key, platform_key, attest_key):
            assert attest(client, keypair, {"nonce": NONCE}).status_code == 200

    def test_the_attest_scope_holds_no_launch_or_teardown_right(self, client, app, attest_key):
        launch = signed_request(
            client,
            attest_key,
            "POST",
            "/v1/cvm",
            body=json.dumps(
                {
                    "kind": "validation",
                    "qemu": "10.1.0",
                    "os_image_hash": "a" * 64,
                    "compose_hash": "b" * 64,
                }
            ).encode(),
        )
        assert launch.status_code == 403

        teardown = signed_request(client, attest_key, "DELETE", "/v1/cvm")
        assert teardown.status_code == 403

    def test_the_attest_scope_still_reads_state(self, client, attest_key):
        assert signed_request(client, attest_key, "GET", "/v1/state").status_code == 200

    def test_the_clients_file_accepts_the_new_scope(self, clients_file_with_attest):
        clients = load_authorized_clients(clients_file_with_attest)
        assert len(clients) == 3

    def test_an_unknown_scope_still_refuses_startup(self, tmp_path, validator_key):
        path = tmp_path / "authorized_clients.json"
        path.write_text(json.dumps([{"hotkey": validator_key.ss58_address, "scope": "observer"}]))
        with pytest.raises(AuthorizedClientsError, match="unknown scope"):
            load_authorized_clients(path)

    def test_attest_is_a_scope_value(self):
        assert Scope("attest") is Scope.ATTEST


class TestRefusals:
    def test_a_malformed_nonce_is_422(self, client, app, validator_key):
        make_renter(app)
        answer = attest(client, validator_key, {"nonce": "not-hex"})
        assert answer.status_code == 422

    def test_an_unknown_field_is_422(self, client, app, validator_key):
        make_renter(app)
        answer = attest(client, validator_key, {"nonce": NONCE, "verify_tls": False})
        assert answer.status_code == 422

    def test_an_idle_node_has_no_agent_to_attest(self, client, validator_key):
        answer = attest(client, validator_key, {"nonce": NONCE})
        assert answer.status_code == 409
        assert "no renter CVM" in answer.json()["detail"]

    def test_a_validation_cvm_is_not_a_renter(self, client, app, validator_key):
        instance = Instance(
            instance_id="cvm-validation-1",
            kind="validation",
            artifact_id="validation-abc",
            vm_dir="/nonexistent/vm",
            supervisor_pid=1,
            created_at="2026-08-13T00:00:00+00:00",
            qemu="10.1.0",
            os_image_hash="a" * 64,
            compose_hash="b" * 64,
        )
        app.state.instances.set(instance)
        app.state.store.transition(NodeState.VALIDATION_RUNNING)
        answer = attest(client, validator_key, {"nonce": NONCE})
        assert answer.status_code == 409

    def test_a_cvm_without_the_agent_forward_is_502_by_name(self, client, app, validator_key):
        make_renter(
            app,
            ports=[PortReport(protocol="tcp", address="0.0.0.0", host_port=2201, guest_port=2200)],
        )
        answer = attest(client, validator_key, {"nonce": NONCE})
        assert answer.status_code == 502
        assert "no port forward" in answer.json()["detail"]

    def test_an_unreachable_agent_is_502_with_a_fixed_detail(
        self, client, app, validator_key, monkeypatch
    ):
        make_renter(app)

        def refuse(**kwargs):
            raise AgentRelayError("connection refused: secret internals")

        monkeypatch.setattr(agent_relay, "relay_attest", refuse)
        answer = attest(client, validator_key, {"nonce": NONCE})
        assert answer.status_code == 502
        # The caller's copy is the authored constant, never the exception's text.
        assert "secret internals" not in answer.json()["detail"]
        assert "could not be reached" in answer.json()["detail"]


class TestRelay:
    def test_the_agent_answer_passes_through_verbatim(
        self, client, app, validator_key, monkeypatch
    ):
        make_renter(app)
        seen = {}

        def fake_relay(**kwargs):
            seen.update(kwargs)
            return 200, AGENT_ANSWER

        monkeypatch.setattr(agent_relay, "relay_attest", fake_relay)
        answer = attest(client, validator_key, {"nonce": NONCE})
        assert answer.status_code == 200
        assert answer.json() == AGENT_ANSWER
        assert seen["nonce"] == NONCE
        assert seen["host_port"] == 18451
        assert seen["address"] == "127.0.0.1"

    def test_the_caller_may_name_a_different_agent_port(
        self, client, app, validator_key, monkeypatch
    ):
        make_renter(
            app,
            ports=[
                PortReport(protocol="tcp", address="127.0.0.1", host_port=19000, guest_port=9000)
            ],
        )
        seen = {}

        def fake_relay(**kwargs):
            seen.update(kwargs)
            return 200, AGENT_ANSWER

        monkeypatch.setattr(agent_relay, "relay_attest", fake_relay)
        answer = attest(client, validator_key, {"nonce": NONCE, "agent_port": 9000})
        assert answer.status_code == 200
        assert seen["host_port"] == 19000

    def test_an_agent_error_status_is_an_answer_not_a_relay_failure(
        self, client, app, validator_key, monkeypatch
    ):
        make_renter(app)
        monkeypatch.setattr(
            agent_relay,
            "relay_attest",
            lambda **kwargs: (503, {"detail": "no quote from the guest agent"}),
        )
        answer = attest(client, validator_key, {"nonce": NONCE})
        assert answer.status_code == 503
        assert answer.json()["detail"] == "no quote from the guest agent"


class TestDialRules:
    def test_a_loopback_bind_is_dialed_at_loopback(self):
        assert dial_address("127.0.0.1") == "127.0.0.1"

    def test_a_wildcard_bind_is_dialed_at_loopback(self):
        assert dial_address("0.0.0.0") == "127.0.0.1"

    def test_an_empty_bind_defaults_like_the_port_parser(self):
        assert dial_address("") == "127.0.0.1"

    def test_an_explicit_bind_is_dialed_as_written(self):
        assert dial_address("10.0.0.7") == "10.0.0.7"

    def test_a_non_object_agent_answer_is_a_relay_error(self):
        with pytest.raises(AgentRelayError, match="not an object"):
            agent_relay._decode(b"[1, 2]")

    def test_a_non_json_agent_answer_is_a_relay_error(self):
        with pytest.raises(AgentRelayError, match="not JSON"):
            agent_relay._decode(b"<html>gateway timeout</html>")
