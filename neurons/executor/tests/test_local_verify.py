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
import threading
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
    DeviceChallenge,
    DockerFacts,
    InspectorFacts,
    MatmulStep,
    PortFacts,
    VerifyIntentBody,
    VerifyResult,
    VerifySteps,
    VerifyXData,
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


def _loop_thread_id() -> int:
    """The thread asyncio.run is on in these tests (the main thread)."""
    return threading.main_thread().ident


def _validator_module(name: str):
    """Load one of the validator's leaf modules by path (no validator package on this side)."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "validators" / "src" / "services" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"validator_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

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
    assert verifyx.data == VerifyXData(
        lib_sha256=lvs.sha256_of_file(str(fake_scripts / "verifyx_executor.py"))
    )

    docker = result.steps["docker"].data
    assert isinstance(docker, DockerFacts)
    assert docker.sysbox_runtime is True and docker.runtimes == ["runc", "sysbox-runc"]
    assert docker.containers[0].name == "pod-1"
    assert docker.disk.total_bytes > 0

    ports = result.steps["ports"].data
    assert isinstance(ports, PortFacts)
    assert ports.configured == 10 and ports.published_by_docker == [40001] and ports.free == 9

    inspector = result.steps["inspector"].data
    assert isinstance(inspector, InspectorFacts)
    assert inspector.lib_present and inspector.script_present and inspector.lib_sha256

    # The wire shape the validator reads is the typed models' field names, one round trip through JSON.
    wire = json.loads(result.model_dump_json(by_alias=True))
    assert wire["steps"]["ports"]["data"]["published_by_docker"] == [40001]
    assert set(wire["steps"]["docker"]["data"]) == set(DockerFacts.model_fields)
    assert VerifyResult.model_validate(wire).steps["docker"].data == docker


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


def test_deadline_keeps_the_finished_gpu_sibling_when_run_side_by_side(
    fake_scripts, fake_docker, monkeypatch
):
    """parallel_gpu: VerifyX finishes at once, the matmul outlives the deadline. The finished
    step's evidence must come back `ok`; only the unfinished one is `timeout`."""
    slow_matmul = fake_scripts / "decrypt_challenge.py"
    slow_matmul.write_text(FAKE_MATMUL.replace('os.environ.get("FAKE_SLEEP", "0")', '"3"'))
    started = time.perf_counter()
    result = asyncio.run(_service(max_deadline_s=1).run(_body(parallel_gpu=True)))

    assert time.perf_counter() - started < 2.5
    assert result.deadline_hit
    assert result.steps["verifyx"].status == "ok", result.steps["verifyx"]
    assert ("ab" * 40) in result.steps["verifyx"].stdout
    assert result.steps["matmul"].status == "timeout"


def test_failed_script_is_evidence_not_an_exception(fake_scripts, fake_docker, monkeypatch):
    monkeypatch.setenv("FAKE_EXIT", "3")
    result = asyncio.run(_service().run(_body()))
    matmul = result.steps["matmul"]
    assert matmul.status == "failed" and matmul.exit_status == 3
    assert "RESULT_JSON:" in matmul.stdout  # what it printed before exiting is kept
    assert result.steps["verifyx"].status == "ok"


def test_all_cards_pins_one_run_per_device_with_its_own_challenge(fake_scripts, fake_docker):
    body = _body(
        steps=VerifySteps(
            matmul=MatmulStep(
                dim_n=1900,
                dim_k=2000000,
                seed=7,
                cipher_text="c0ffee",
                devices=[
                    DeviceChallenge(index=0, seed=70, cipher_text="card0"),
                    DeviceChallenge(index=1, seed=71, cipher_text="card1"),
                ],
            )
        )
    )
    result = asyncio.run(_service().run(body))
    cards = result.steps["matmul"].data.per_card
    assert [c.card_index for c in cards] == [0, 1]
    assert all(c.status == "ok" for c in cards)
    printed = [json.loads(c.stdout.splitlines()[-1].split("RESULT_JSON: ")[1]) for c in cards]
    assert [p["device"] for p in printed] == ["0", "1"]
    # Each card ran ITS challenge: the sealed output carries that card's cipher text and seed, not the step's.
    assert [p["sealed"] for p in printed] == ["cafecard0", "cafecard1"]
    assert [p["seed"] for p in printed] == ["70", "71"]


@pytest.mark.parametrize(
    "devices",
    [
        # two cards, one challenge: one real run could answer for both
        [DeviceChallenge(index=0, seed=1, cipher_text="same"), DeviceChallenge(index=1, seed=2, cipher_text="same")],
        # a card's challenge equal to the step's own
        [DeviceChallenge(index=0, seed=1, cipher_text="c")],
        # the same card twice
        [DeviceChallenge(index=0, seed=1, cipher_text="a"), DeviceChallenge(index=0, seed=2, cipher_text="b")],
    ],
)
def test_a_shared_or_repeated_card_challenge_is_refused(devices):
    with pytest.raises(ValueError):
        MatmulStep(dim_n=1, dim_k=1, seed=1, cipher_text="c", devices=devices)


def test_a_shared_card_challenge_is_a_422_on_the_route(client, validator_keypair):
    """The model_validator's refusal reaches the wire as a JSON 422 (its `ctx` carries a ValueError
    object that FastAPI cannot serialise by itself)."""
    body = _body().model_dump(by_alias=True)
    body["steps"] = {
        "matmul": {
            "dim_n": 1, "dim_k": 1, "seed": 1, "cipher_text": "c",
            "devices": [
                {"index": 0, "seed": 1, "cipher_text": "same"},
                {"index": 1, "seed": 2, "cipher_text": "same"},
            ],
        }
    }
    body["signature"] = "0x" + validator_keypair.sign(canonical_intent_message(body)).hex()
    response = client.post("/verify", json=body)
    assert response.status_code == 422
    assert "own cipher_text" in json.dumps(response.json())


def test_no_configured_ports_means_the_validators_default_range(fake_docker):
    """Both settings unset: the validator counts 20000–65535 (port_utils.DEFAULT_PORT_RANGE), so
    the facts must count the same, not zero."""
    port_utils = _validator_module("port_utils")  # the validator's own definition, same repo
    assert lvs.DEFAULT_PORT_RANGE == port_utils.DEFAULT_PORT_RANGE
    assert lvs.parse_port_range(None, None) == port_utils.get_all_ports(None, None, 0)
    assert lvs.parse_port_range(None, None) == [(p, p) for p in range(20000, 65536)]
    assert lvs.parse_port_range("", "") == [(p, p) for p in range(20000, 65536)]
    facts = lvs._port_facts(None, None, ssh_port=22)
    assert facts.configured == 65536 - 20000 and facts.sampled == lvs.PORT_SAMPLE_MAX
    assert lvs.parse_port_range("40000-40001", None) == [(40000, 40000), (40001, 40001)]


def test_the_verifyx_library_is_hashed_off_the_event_loop(fake_scripts, fake_docker, monkeypatch):
    on_loop: list[bool] = []
    pool_thread: list[str] = []
    real = lvs.sha256_of_file

    def observed(path):
        on_loop.append(_loop_thread_id() == threading.get_ident())
        pool_thread.append(threading.current_thread().name)
        return real(path)

    monkeypatch.setattr(lvs, "sha256_of_file", observed)
    result = asyncio.run(_service().run(_body(steps=VerifySteps(verifyx=VerifyXStep(seed=1, cipher_text="d")))))
    assert result.steps["verifyx"].data.lib_sha256 == real(str(fake_scripts / "verifyx_executor.py"))
    assert on_loop == [False]
    assert pool_thread[0].startswith("local-verify-facts")  # the facts pool, not asyncio's default


def test_a_deadline_during_the_library_digest_does_not_start_the_next_gpu_step(
    fake_scripts, fake_docker, monkeypatch, tmp_path
):
    """Serial order, VerifyX instant, its library digest 2 s, the matmul 3 s, deadline 1 s: the
    cancellation lands in the digest await. It must end the GPU group there — the matmul never
    starts and `run` returns at the deadline. (A step that swallowed the cancellation would run the
    matmul afterwards and hold the answer for it: ≥ 5 s and a matmul that ran.)"""
    mark = tmp_path / "matmul_ran"
    slow_matmul = fake_scripts / "decrypt_challenge.py"
    slow_matmul.write_text(
        FAKE_MATMUL.replace('time.sleep(float(os.environ.get("FAKE_SLEEP", "0")))', f'open({str(mark)!r}, "w").close(); time.sleep(3)')
    )
    real = lvs.sha256_of_file

    def slow_digest(path):
        time.sleep(2)
        return real(path)

    monkeypatch.setattr(lvs, "sha256_of_file", slow_digest)
    started = time.perf_counter()
    result = asyncio.run(_service(max_deadline_s=1).run(_body()))  # serial: verifyx, then matmul
    assert time.perf_counter() - started < 2.5
    assert result.deadline_hit
    assert result.steps["verifyx"].status == "timeout" and result.steps["matmul"].status == "timeout"
    assert not mark.exists()  # the matmul never started


def test_a_slow_digest_neither_blocks_the_loop_nor_loses_the_run(fake_scripts, fake_docker, monkeypatch):
    """The digest starts before the script and is awaited after it: a slow one (here 0.4 s, the
    script instant) delays the answer by its remainder only and the run keeps its output."""
    real = lvs.sha256_of_file

    def slow(path):
        time.sleep(0.4)
        return real(path)

    monkeypatch.setattr(lvs, "sha256_of_file", slow)
    started = time.perf_counter()
    result = asyncio.run(_service().run(_body(steps=VerifySteps(verifyx=VerifyXStep(seed=1, cipher_text="d")))))
    assert result.steps["verifyx"].status == "ok" and result.steps["verifyx"].data.lib_sha256 == real(
        str(fake_scripts / "verifyx_executor.py")
    )
    assert time.perf_counter() - started < 3


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


def test_busy_is_409_and_does_not_burn_the_nonce(client, validator_keypair, monkeypatch):
    """A second intent while one runs is refused as busy WITHOUT claiming its nonce, so the same
    signed intent is accepted once the executor is free (the validator need not re-sign)."""
    monkeypatch.setenv("FAKE_SLEEP", "0.8")
    first = _signed(_body(steps=VerifySteps(verifyx=_body().steps.verifyx)), validator_keypair)
    second = _signed(_body(steps=VerifySteps(inspector=True)), validator_keypair)
    with client:  # the TestClient's own loop, so the two requests share one service instance

        async def scenario():
            loop = asyncio.get_running_loop()
            running = loop.run_in_executor(None, lambda: client.post("/verify", json=first))
            await asyncio.sleep(0.2)
            while_busy = await loop.run_in_executor(
                None, lambda: client.post("/verify", json=second)
            )
            return await running, while_busy

        done, while_busy = client.portal.call(scenario)
    assert done.status_code == 200
    assert while_busy.status_code == 409 and "already running" in while_busy.text
    after = client.post("/verify", json=second)
    assert after.status_code == 200, after.text


def test_expired_or_skewed_intent_is_401(client, validator_keypair):
    now = int(time.time())
    expired = _signed(_body(issued_at=now - 300, expires_at=now - 10), validator_keypair)
    assert client.post("/verify", json=expired).status_code == 401
    skewed = _signed(_body(issued_at=now + 600, expires_at=now + 700), validator_keypair)
    assert client.post("/verify", json=skewed).status_code == 401


def test_malformed_intent_is_422_before_any_signature_check(client, validator_keypair):
    assert client.post("/verify", json={"nonce": "short"}).status_code == 422
    assert client.post("/verify", json=[1, 2]).status_code == 422
    # Another schema version is refused, not run under v1 semantics — even correctly signed.
    other_schema = _body().model_dump(by_alias=True)
    other_schema["schema"] = "lium.local_verify/2"
    other_schema["signature"] = (
        "0x" + validator_keypair.sign(canonical_intent_message(other_schema)).hex()
    )
    assert client.post("/verify", json=other_schema).status_code == 422
    # The matmul fan-out is bounded: more devices than any host has, or a negative index, is 422.
    with pytest.raises(ValueError):
        MatmulStep(
            dim_n=1, dim_k=1, seed=1, cipher_text="c",
            devices=[DeviceChallenge(index=i, seed=i, cipher_text=f"card{i}") for i in range(65)],
        )
    with pytest.raises(ValueError):
        DeviceChallenge(index=-1, seed=1, cipher_text="x")


def test_canonical_message_is_the_shared_datura_definition():
    """One definition for signer and verifier (the validator client imports the same function)."""
    from datura.requests.validator_requests import (
        LOCAL_VERIFY_CAPABILITY,
        LOCAL_VERIFY_SCHEMA,
        local_verify_signing_blob,
    )

    assert canonical_intent_message is local_verify_signing_blob
    assert SCHEMA == LOCAL_VERIFY_SCHEMA and CAPABILITY == LOCAL_VERIFY_CAPABILITY
