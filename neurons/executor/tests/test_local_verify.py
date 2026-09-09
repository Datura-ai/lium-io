"""liumd phase 1: `POST /verify` — one validator-signed intent, the suite run locally, one result.

Fake GPU: the matmul and VerifyX scripts are replaced by small Python files that print what the
real ones print (`RESULT_JSON: …`, a cipher line) after an optional sleep, so parallelism and the
deadline are measured for real. Fake docker/ports: the mocked docker client returns a container
publishing one host port of the renting range.
"""

from __future__ import annotations

import asyncio
import json
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
    CAPABILITY,
    SCHEMA,
    MatmulStep,
    VerifyIntentBody,
    VerifySteps,
    VerifyXStep,
)
from routes.apis import apis_router
from services.local_verify_service import (
    BusyError,
    LocalVerifyService,
    NonceCache,
    canonical_intent_message,
    check_intent_window,
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
          "sealed": "cafe" + a.cipher_text, "device": os.environ.get("CUDA_VISIBLE_DEVICES")}))
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


def _service(**overrides) -> LocalVerifyService:
    kwargs = dict(
        executor_version="test",
        max_deadline_s=60,
        port_range="40000-40009",
        port_mappings=None,
        ssh_port=2200,
        python=sys.executable,
    )
    kwargs.update(overrides)
    return LocalVerifyService(**kwargs)


def _body(**overrides) -> VerifyIntentBody:
    now = int(time.time())
    fields = dict(
        nonce=secrets.token_hex(16),
        issued_at=now,
        expires_at=now + 120,
        executor_uuid=EXECUTOR_UUID,
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


def test_all_steps_run_and_return_raw_evidence(fake_scripts, fake_docker):
    result = asyncio.run(_service().run(_body()))

    assert result.schema_id == SCHEMA
    assert result.executor_uuid == EXECUTOR_UUID
    assert result.signer == "none" and result.signature is None
    assert not result.deadline_hit
    assert {name: s.status for name, s in result.steps.items()} == {
        "matmul": "ok",
        "verifyx": "ok",
        "docker": "ok",
        "ports": "ok",
        "inspector": "ok",
    }

    matmul = result.steps["matmul"]
    # The same stdout the validator parses over SSH, with its digest as evidence.
    assert "RESULT_JSON:" in matmul.stdout and '"sealed": "cafec0ffee"' in matmul.stdout
    assert matmul.exit_status == 0
    assert matmul.stdout_sha256 == lvs._sha256_text(matmul.stdout)

    verifyx = result.steps["verifyx"]
    assert verifyx.stdout.strip().endswith("deadbeef")
    assert verifyx.data["lib_sha256"] == lvs.sha256_of_file(
        str(fake_scripts / "verifyx_executor.py")
    )

    docker = result.steps["docker"].data
    assert docker["sysbox_runtime"] is True and docker["runtimes"] == ["runc", "sysbox-runc"]
    assert docker["containers"][0]["name"] == "pod-1"
    assert docker["disk"]["total_bytes"] > 0

    ports = result.steps["ports"].data
    assert (
        ports["configured"] == 10 and ports["published_by_docker"] == [40001] and ports["free"] == 9
    )

    inspector = result.steps["inspector"].data
    assert inspector["lib_present"] and inspector["script_present"] and inspector["lib_sha256"]


def test_gpu_steps_run_side_by_side_only_when_asked(fake_scripts, fake_docker, monkeypatch):
    monkeypatch.setenv("FAKE_SLEEP", "0.6")
    body = _body(steps=VerifySteps(matmul=_body().steps.matmul, verifyx=_body().steps.verifyx))

    serial = asyncio.run(_service().run(body))
    parallel = asyncio.run(_service().run(body.model_copy(update={"parallel_gpu": True})))

    assert serial.steps["matmul"].status == parallel.steps["matmul"].status == "ok"
    assert serial.elapsed_ms >= 1100, serial.elapsed_ms  # verifyx then matmul, the pipeline's order
    assert parallel.elapsed_ms < 1100, parallel.elapsed_ms


def test_deadline_returns_what_finished_and_marks_the_rest(fake_scripts, fake_docker, monkeypatch):
    monkeypatch.setenv("FAKE_SLEEP", "3")
    started = time.perf_counter()
    result = asyncio.run(_service(max_deadline_s=1).run(_body(parallel_gpu=True)))

    assert time.perf_counter() - started < 2.5
    assert result.deadline_hit
    assert (
        result.steps["matmul"].status == "timeout" and result.steps["verifyx"].status == "timeout"
    )
    # The fact steps do not wait for the GPU ones.
    assert result.steps["docker"].status == "ok" and result.steps["ports"].status == "ok"


def test_failed_script_is_evidence_not_an_exception(fake_scripts, fake_docker, monkeypatch):
    monkeypatch.setenv("FAKE_EXIT", "3")
    result = asyncio.run(_service().run(_body()))
    matmul = result.steps["matmul"]
    assert matmul.status == "failed" and matmul.exit_status == 3
    assert "RESULT_JSON:" in matmul.stdout  # what it printed before exiting is kept
    assert result.steps["verifyx"].status == "ok"


def test_all_cards_pins_one_run_per_device(fake_scripts, fake_docker):
    body = _body(
        steps=VerifySteps(
            matmul=MatmulStep(
                dim_n=1900, dim_k=2000000, seed=7, cipher_text="c0ffee", devices=[0, 1]
            )
        )
    )
    result = asyncio.run(_service().run(body))
    cards = result.steps["matmul"].data["per_card"]
    assert [c["card_index"] for c in cards] == [0, 1]
    assert all(c["status"] == "ok" for c in cards)
    assert [
        json.loads(c["stdout"].splitlines()[-1].split("RESULT_JSON: ")[1])["device"] for c in cards
    ] == ["0", "1"]


def test_steps_not_asked_for_are_skipped(fake_scripts, fake_docker):
    result = asyncio.run(_service().run(_body(steps=VerifySteps(docker=True))))
    assert result.steps["docker"].status == "ok"
    assert {n for n, s in result.steps.items() if s.status == "skipped"} == {
        "matmul",
        "verifyx",
        "ports",
        "inspector",
    }


def test_second_concurrent_intent_is_refused_as_busy(fake_scripts, fake_docker, monkeypatch):
    monkeypatch.setenv("FAKE_SLEEP", "0.5")
    service = _service()

    async def two_at_once():
        first = asyncio.ensure_future(service.run(_body()))
        await asyncio.sleep(0.05)
        with pytest.raises(BusyError):
            await service.run(_body())
        return await first

    assert asyncio.run(two_at_once()).steps["matmul"].status == "ok"


def test_docker_failure_is_one_failed_step(fake_scripts, fake_docker):
    fake_docker.info.side_effect = RuntimeError("daemon down")
    result = asyncio.run(_service().run(_body(steps=VerifySteps(docker=True, ports=True))))
    assert (
        result.steps["docker"].status == "failed" and "daemon down" in result.steps["docker"].error
    )
    assert result.steps["ports"].status == "ok"


# --- intent hygiene -----------------------------------------------------------------------------


def test_canonical_message_ignores_key_order_and_signature():
    a = {"nonce": "n", "steps": {"docker": True, "ports": False}, "issued_at": 1, "signature": "x"}
    b = {"signature": "y", "issued_at": 1, "steps": {"ports": False, "docker": True}, "nonce": "n"}
    assert canonical_intent_message(a) == canonical_intent_message(b)
    assert "signature" not in canonical_intent_message(a)


def test_intent_window():
    now = 1_000_000
    assert check_intent_window(_body(issued_at=now, expires_at=now + 120), now, 120) is None
    assert (
        check_intent_window(_body(issued_at=now, expires_at=now - 1), now, 120) == "intent expired"
    )
    assert (
        check_intent_window(_body(issued_at=now - 121, expires_at=now + 120), now, 120)
        == "issued_at outside the accepted window"
    )
    assert (
        check_intent_window(_body(issued_at=now, expires_at=now + 3600), now, 120)
        == "expiry too far from issued_at"
    )


def test_nonce_cache_refuses_replay_until_expiry():
    cache = NonceCache()
    assert cache.claim("a", expires_at=200, now=100)
    assert not cache.claim("a", expires_at=200, now=150)
    assert cache.claim("a", expires_at=400, now=201)  # expired entries are forgotten


# --- the route: auth, flag, replay -----------------------------------------------------------


@pytest.fixture()
def client(validator_keypair, monkeypatch, fake_scripts, fake_docker):
    monkeypatch.setattr("dependencies.auth.VALIDATOR_HOTKEY_SS58", validator_keypair.ss58_address)
    monkeypatch.setattr(settings, "EXECUTOR_LOCAL_VERIFY_ENABLED", True)
    monkeypatch.setattr(settings, "RENTING_PORT_RANGE", "40000-40009")
    monkeypatch.setattr(apis_module, "_local_verify_service", None)
    monkeypatch.setattr(apis_module, "_local_verify_nonces", NonceCache())
    app = FastAPI()
    app.add_middleware(MinerMiddleware)
    app.include_router(apis_router)
    return TestClient(app)


def test_version_advertises_the_capability_only_when_enabled(client, monkeypatch):
    assert client.get("/version").json()["capabilities"] == [CAPABILITY]
    monkeypatch.setattr(settings, "EXECUTOR_LOCAL_VERIFY_ENABLED", False)
    assert client.get("/version").json()["capabilities"] == []


def test_flag_off_is_404(client, validator_keypair, monkeypatch):
    monkeypatch.setattr(settings, "EXECUTOR_LOCAL_VERIFY_ENABLED", False)
    assert client.post("/verify", json=_signed(_body(), validator_keypair)).status_code == 404


def test_signed_intent_runs_the_suite(client, validator_keypair):
    response = client.post("/verify", json=_signed(_body(), validator_keypair))
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["schema"] == SCHEMA and result["executor_uuid"] == EXECUTOR_UUID
    assert (
        result["steps"]["matmul"]["status"] == "ok"
        and "RESULT_JSON:" in result["steps"]["matmul"]["stdout"]
    )
    assert result["steps"]["ports"]["data"]["published_by_docker"] == [40001]
    assert result["signer"] == "none"


def test_wrong_key_or_tampered_field_is_401(client, validator_keypair):
    stranger = bittensor.Keypair.create_from_uri("//Stranger")
    assert client.post("/verify", json=_signed(_body(), stranger)).status_code == 401

    tampered = _signed(_body(), validator_keypair)
    tampered["steps"]["matmul"]["cipher_text"] = "ffff"
    assert client.post("/verify", json=tampered).status_code == 401

    swapped_uuid = _signed(_body(), validator_keypair)
    swapped_uuid["executor_uuid"] = "someone-else"
    assert client.post("/verify", json=swapped_uuid).status_code == 401


def test_replayed_nonce_is_refused(client, validator_keypair):
    intent = _signed(_body(steps=VerifySteps(inspector=True)), validator_keypair)
    assert client.post("/verify", json=intent).status_code == 200
    replay = client.post("/verify", json=intent)
    assert replay.status_code == 409 and "nonce" in replay.text


def test_expired_or_skewed_intent_is_401(client, validator_keypair):
    now = int(time.time())
    expired = _signed(_body(issued_at=now - 300, expires_at=now - 10), validator_keypair)
    assert client.post("/verify", json=expired).status_code == 401
    skewed = _signed(_body(issued_at=now + 600, expires_at=now + 700), validator_keypair)
    assert client.post("/verify", json=skewed).status_code == 401


def test_malformed_intent_is_422_before_any_signature_check(client):
    assert client.post("/verify", json={"nonce": "short"}).status_code == 422
    assert client.post("/verify", json=[1, 2]).status_code == 422
