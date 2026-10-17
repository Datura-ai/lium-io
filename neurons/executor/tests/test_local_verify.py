"""liumd phase 1: `POST /verify` — one validator-signed intent, the suite run locally, one result.

Fake GPU: the matmul and VerifyX scripts are replaced by small Python files that print what the
real ones print (`RESULT_JSON: …`, a cipher line) after an optional sleep, so parallelism and the
deadline are measured for real. Fake docker/ports: the mocked docker client returns a container
publishing one host port of the renting range.
"""

from __future__ import annotations

import secrets
import sys
import textwrap
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import bittensor
import pytest
import services.local_verify_service as lvs
from fastapi import FastAPI
from fastapi.testclient import TestClient
from middlewares.miner import MinerMiddleware
from payloads.verify import (
    MatmulStep,
    VerifyIntentBody,
    VerifySteps,
    VerifyXStep,
)
from routes.apis import apis_router
from services.local_verify_service import (
    NonceCache,
    canonical_intent_message,
)

from core.config import settings
from routes import apis as apis_module

EXECUTOR_UUID = "exec-0001"


FAKE_MATMUL = textwrap.dedent(
    """
    import argparse, json, os, sys, time
    p = argparse.ArgumentParser()
    for name in ("--dim_n", "--dim_k", "--seed", "--cipher_text"):
        p.add_argument(name)
    a = p.parse_args()
    time.sleep(float(os.environ.get("FAKE_SLEEP", "0")))
    print("UUID:  fake-uuid")
    print("RESULT_JSON: " + json.dumps({"uuid": "fake-uuid", "metrics": {"tflops": 1.0},
          "sealed": "cafe" + a.cipher_text, "seed": a.seed,
          "device": os.environ.get("CUDA_VISIBLE_DEVICES")}))
    sys.exit(int(os.environ.get("FAKE_EXIT", "0")))
    """
)

FAKE_VERIFYX = textwrap.dedent(
    """
    import argparse, os, time
    p = argparse.ArgumentParser()
    p.add_argument("--seed"); p.add_argument("--cipher_text")
    a = p.parse_args()
    time.sleep(float(os.environ.get("FAKE_SLEEP", "0")))
    print(("ab" * 40) + a.cipher_text)
    """
)


@pytest.fixture(scope="module")
def validator_keypair():
    return bittensor.Keypair.create_from_uri("//LiumTestValidator")


@pytest.fixture()
def fake_scripts(tmp_path, monkeypatch):
    matmul = tmp_path / "decrypt_challenge.py"
    matmul.write_text(FAKE_MATMUL)
    verifyx = tmp_path / "verifyx_executor.py"
    verifyx.write_text(FAKE_VERIFYX)
    monkeypatch.setattr(lvs, "MATMUL_SCRIPT", matmul)
    monkeypatch.setattr(lvs, "VERIFYX_SCRIPT", verifyx)
    monkeypatch.setattr(
        lvs, "LIBVERIFYX_PATH", str(verifyx)
    )  # any readable file: the digest rides along
    monkeypatch.setattr(lvs, "LIBINSPECTOR_PATH", str(matmul))
    monkeypatch.setattr(lvs, "INSPECTOR_SCRIPT", matmul)
    return tmp_path


@pytest.fixture()
def fake_docker():
    # conftest replaced the docker module with a MagicMock; give from_env() a client with one
    # running container that publishes host port 40001 and a sysbox runtime.
    container = SimpleNamespace(
        name="pod-1",
        status="running",
        attrs={"Config": {"Image": "img:1"}, "Created": "2026-09-09T00:00:00Z"},
        ports={"22/tcp": [{"HostIp": "0.0.0.0", "HostPort": "40001"}]},
    )
    client = MagicMock()
    client.info.return_value = {
        "ServerVersion": "27.0",
        "DockerRootDir": "/",
        "Runtimes": {"runc": {}, "sysbox-runc": {}},
        "DefaultRuntime": "runc",
    }
    client.containers.list.return_value = [container]
    sys.modules["docker"].from_env.return_value = client
    return client


def _body(**overrides) -> VerifyIntentBody:
    now = int(time.time())
    fields = dict(
        nonce=secrets.token_hex(16),
        issued_at=now,
        expires_at=now + 120,
        executor_uuid=EXECUTOR_UUID,
        miner_hotkey=settings.MINER_HOTKEY_SS58_ADDRESS,
        deadline_s=60,
        parallel_gpu=False,
        steps=VerifySteps(
            matmul=MatmulStep(dim_n=1900, dim_k=2000000, seed=7, cipher_text="c0ffee"),
            verifyx=VerifyXStep(seed=9, cipher_text="deadbeef"),
            docker=True,
            ports=True,
            inspector=True,
        ),
    )
    fields.update(overrides)
    return VerifyIntentBody(**fields)


def _signed(body: VerifyIntentBody, keypair) -> dict:
    wire = body.model_dump(by_alias=True)
    wire["signature"] = "0x" + keypair.sign(canonical_intent_message(wire)).hex()
    return wire


# --- the service: fake GPU, fake docker -------------------------------------------------------


# --- intent hygiene -----------------------------------------------------------------------------


# --- the route: auth, flag, replay -----------------------------------------------------------


def _app(validator_keypair, monkeypatch) -> FastAPI:
    monkeypatch.setattr(
        "dependencies.auth.VALIDATOR_HOTKEYS_SS58", {"current": validator_keypair.ss58_address}
    )
    monkeypatch.setattr(settings, "EXECUTOR_LOCAL_VERIFY_ENABLED", True)
    monkeypatch.setattr(settings, "RENTING_PORT_RANGE", "40000-40009")
    # The configured miner and the shared portal hotkey are two different keys here, so the
    # "portal hotkey binds nothing" refusal is tested against a real difference, not conftest's.
    monkeypatch.setattr(
        settings, "MINER_HOTKEY_SS58_ADDRESS", bittensor.Keypair.create_from_uri("//Miner").ss58_address
    )
    monkeypatch.setattr(
        settings, "DEFAULT_MINER_HOTKEY", bittensor.Keypair.create_from_uri("//Portal").ss58_address
    )
    monkeypatch.setattr(apis_module, "_local_verify_service", None)
    monkeypatch.setattr(apis_module, "_local_verify_nonces", NonceCache())
    app = FastAPI()
    app.add_middleware(MinerMiddleware)
    app.include_router(apis_router)
    return app


def _from_peer(app, host: str, port: int = 51234):
    """The app as seen from a TCP peer at `host` (starlette 0.37's TestClient cannot set one)."""

    async def wrapped(scope, receive, send):
        if scope["type"] == "http":
            scope = {**scope, "client": (host, port)}
        await app(scope, receive, send)

    return wrapped


@pytest.fixture()
def client(validator_keypair, monkeypatch, fake_scripts, fake_docker):
    # The peer the validator's SSH tunnel presents: sshd's direct-tcpip channel lands on loopback.
    return TestClient(_from_peer(_app(validator_keypair, monkeypatch), "127.0.0.1"))


def test_a_network_peer_is_403_before_the_body_is_read_and_burns_nothing(
    validator_keypair, monkeypatch, fake_scripts, fake_docker
):
    """The result is unsigned, so it must travel inside the validator's pinned SSH channel: the
    miner's port-forward (a network peer, here the docker gateway) is refused, a valid intent
    included, and its nonce stays unclaimed for the tunnel to use."""
    app = _app(validator_keypair, monkeypatch)
    from_network = TestClient(_from_peer(app, "172.18.0.1"))
    intent = _signed(_body(steps=VerifySteps(inspector=True)), validator_keypair)
    refused = from_network.post("/verify", json=intent)
    assert refused.status_code == 403 and "loopback" in refused.text
    assert from_network.post("/verify", data=b"not even json").status_code == 403
    # The QEMU slirp gateway a CVM sees every outside connection from, and the IPv6 forms.
    assert TestClient(_from_peer(app, "10.0.2.2")).post("/verify", json=intent).status_code == 403
    assert TestClient(_from_peer(app, "::ffff:10.0.2.2")).post("/verify", json=intent).status_code == 403
    assert TestClient(_from_peer(app, "testclient")).post("/verify", json=intent).status_code == 403
    # IPv6 loopback is the tunnel too; the nonce is still unclaimed after every refusal above.
    assert TestClient(_from_peer(app, "::1")).post("/verify", json=intent).status_code == 200


def test_wrong_key_or_tampered_field_is_401(client, validator_keypair):
    stranger = bittensor.Keypair.create_from_uri("//Stranger")
    assert client.post("/verify", json=_signed(_body(), stranger)).status_code == 401

    tampered = _signed(_body(), validator_keypair)
    tampered["steps"]["matmul"]["cipher_text"] = "ffff"
    assert client.post("/verify", json=tampered).status_code == 401

    swapped_uuid = _signed(_body(), validator_keypair)
    swapped_uuid["executor_uuid"] = "someone-else"
    assert client.post("/verify", json=swapped_uuid).status_code == 401


def test_an_intent_for_another_miners_executor_is_401_and_not_burnt(client, validator_keypair):
    """Regression: a validator intent captured on provider A's wire and relayed to provider B's
    executor (same validator signature, B's flag on) must not run B's GPU suite — B would answer
    the real validator 409 busy meanwhile. The executor knows only its miner, so the signed
    `miner_hotkey` is what binds the intent; the portal hotkey every executor trusts does not."""
    other_miner = bittensor.Keypair.create_from_uri("//OtherMiner").ss58_address
    relayed = client.post("/verify", json=_signed(_body(miner_hotkey=other_miner), validator_keypair))
    assert relayed.status_code == 401 and "another miner" in relayed.text
    portal = client.post(
        "/verify", json=_signed(_body(miner_hotkey=settings.DEFAULT_MINER_HOTKEY), validator_keypair)
    )
    assert portal.status_code == 401
    # refused before the nonce is claimed: the same nonce, correctly addressed, still runs
    body = _body(steps=VerifySteps(inspector=True))
    foreign = _signed(body.model_copy(update={"miner_hotkey": other_miner}), validator_keypair)
    assert client.post("/verify", json=foreign).status_code == 401
    assert client.post("/verify", json=_signed(body, validator_keypair)).status_code == 200


def test_replayed_nonce_is_refused(client, validator_keypair):
    intent = _signed(_body(steps=VerifySteps(inspector=True)), validator_keypair)
    assert client.post("/verify", json=intent).status_code == 200
    replay = client.post("/verify", json=intent)
    assert replay.status_code == 409 and "nonce" in replay.text


