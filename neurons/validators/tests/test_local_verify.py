"""liumd phase 1 (DAH-2834): the matmul and VerifyX from one signed `POST /verify`, judged by the
same functions the SSH path uses, with the SSH path as the fallback for every other outcome.

Fake executor: an in-process aiohttp server that checks the intent signature the way the executor
does (canonical JSON, validator hotkey), refuses a replayed nonce, and answers with stub script
output after a configurable per-step sleep. Fake SSH: an `ssh_client.run` that sleeps one RTT per
command. The timing test at the end is the e2e-stack measurement quoted in the PR body.
"""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import bittensor
import pytest
import services.matrix_validation_service as mvs
import services.verifyx_validation_service as vvs
from aiohttp import web
from aiohttp.test_utils import TestServer
from datura.requests.miner_requests import ExecutorSSHInfo
from neurons.validators.src.services.task.checks.capability import CapabilityCheck
from neurons.validators.src.services.task.checks.local_verify import (
    STEP_WALL_CLOCK_CAP_MS,
    LocalVerifyCheck,
    LocalVerifyOutcome,
)
from neurons.validators.src.services.task.checks.verifyx import VerifyXCheck
from neurons.validators.src.services.task.pipeline_factory import PipelineFactory
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse
from services.local_verify_client import (
    CAPABILITY,
    EXECUTOR_DEADLINE_MARGIN_SECONDS,
    EXECUTOR_DEADLINE_MAX_SECONDS,
    EXECUTOR_VERSION_MAX_CHARS,
    MAX_ANSWER_BYTES,
    MAX_CAPABILITIES,
    MAX_CAPABILITY_CHARS,
    SCHEMA,
    LocalVerifyClient,
    LocalVerifyUnavailable,
    build_intent,
    canonical_intent_message,
    executor_deadline_s,
    parse_answer,
    sign_intent,
)

from core.config import settings
from tests.helpers import build_context_config, build_services, build_state, make_context

SPECS = {"gpu": {"count": 1, "details": [{"uuid": "GPU-1", "name": "H100", "capacity": 81559}]}}
EXECUTOR_UUID = "executor-123"


# --- fakes -------------------------------------------------------------------------------------


@pytest.fixture
def keypair():
    return bittensor.Keypair.create_from_uri("//LiumTestValidator")


@pytest.fixture
def local_verify_on(monkeypatch):
    monkeypatch.setattr(settings, "VALIDATOR_LOCAL_VERIFY_ENABLED", True)
    monkeypatch.setattr(settings, "LOCAL_VERIFY_TIMEOUT_SECONDS", 5)
    monkeypatch.setattr(settings, "LOCAL_VERIFY_CONNECT_TIMEOUT_SECONDS", 2)


def matmul_service(monkeypatch, *, sealed: str | None = "challenge") -> mvs.ValidationService:
    """A ValidationService over a fake libdmcompverify: the challenge cipher is `deadbeef`; the
    unseal returns the uuid of the challenge most recently generated (`"challenge"` — what a real
    card proves), a literal wrong uuid, or "" (None — authentication failure)."""
    wrapper = MagicMock(name="DMCompVerifyWrapper")
    wrapper.DMCompVerify_new.return_value = "ptr"
    wrapper.getCipherText.return_value = "deadbeef"
    wrapper._has_sealed = True
    last_uuid: list[str] = []
    wrapper.generateChallenge.side_effect = lambda ptr, seed, info, uuid: last_uuid.append(uuid)

    def unseal(ptr, blob):
        if sealed is None:
            return ""
        proven = last_uuid[-1] if sealed == "challenge" else sealed
        return json.dumps({"uuid": proven, "metrics": {"tflops": 42.0}})

    wrapper.unsealResult.side_effect = unseal
    monkeypatch.setattr(mvs, "DMCompVerifyWrapper", lambda *_a, **_kw: wrapper)
    return mvs.ValidationService()


class _FakeVerifyXValidator:
    """Stands in for libverifyx: the challenge is `vx-<seed>`; a response is accepted when it is
    the challenge echoed back with `-ok` and yields a passing payload, else rejected."""

    def __init__(self, lib_name, seed):
        self.seed = seed

    def generate_challenge(self, challenge_input):
        return f"vx{self.seed}".ljust(vvs.MIN_CIPHER_LEN, "0")

    def verify_response(self, response):
        if not response.endswith("-ok"):
            raise RuntimeError("cipher rejected")
        return {"ok": True}


@pytest.fixture
def verifyx_service(monkeypatch):
    monkeypatch.setattr(vvs, "VerifyXValidator", _FakeVerifyXValidator)
    monkeypatch.setattr(vvs, "sha256_from_path", lambda _p: "lib-sha")
    monkeypatch.setattr(
        vvs,
        "_perform_verification_checks",
        lambda payload: {
            "success": True,
            "ram": {},
            "network": {"download_speed": 900.0, "upload_speed": 500.0, "success": True},
        },
    )
    return vvs.VerifyXValidationService()


def matmul_stdout(proven_uuid: str) -> str:
    # What decrypt_challenge.py prints; the sealed blob is what the validator trusts.
    return (
        "UUID:  "
        + proven_uuid
        + "\nRESULT_JSON: "
        + json.dumps({"uuid": proven_uuid, "metrics": {"tflops": 42.0}, "sealed": "cafe"})
    )


class FakeExecutor:
    """The executor's `/verify` as a fake: signature and replay checks as the real route, then a
    stub GPU whose scripts take `step_sleep` seconds each and run side by side when asked."""

    def __init__(
        self,
        keypair,
        *,
        advertise=True,
        step_sleep=0.0,
        matmul_uuid_from_intent=True,
        lib_sha="lib-sha",
        verifyx_ok=True,
    ):
        self.keypair = keypair
        self.advertise = advertise
        self.step_sleep = step_sleep
        self.matmul_uuid_from_intent = matmul_uuid_from_intent
        self.lib_sha = lib_sha
        self.verifyx_ok = verifyx_ok  # False: a full-length response libverifyx rejects
        self.seen_nonces: set[str] = set()
        self.intents: list[dict] = []
        self.answer_override = None  # callable(intent) -> dict | (status, body)
        self.version_override = None  # dict served by /version instead of the default
        self.app = web.Application()
        self.app.router.add_get("/version", self.version)
        self.app.router.add_post("/verify", self.verify)
        self.server = TestServer(self.app)

    async def __aenter__(self):
        await self.server.start_server()
        return self

    async def __aexit__(self, *exc):
        await self.server.close()

    @property
    def executor_info(self) -> ExecutorSSHInfo:
        return ExecutorSSHInfo(
            uuid=EXECUTOR_UUID,
            address=self.server.host,
            port=self.server.port,
            ssh_username="root",
            ssh_port=22,
            python_path="/usr/bin/python",
            root_dir="/root/app",
        )

    async def version(self, request):
        if self.version_override is not None:
            return web.json_response(self.version_override)
        return web.json_response(
            {"version": "4.1.0", "capabilities": [CAPABILITY] if self.advertise else []}
        )

    async def verify(self, request):
        raw = await request.json()
        if not self.keypair.verify(canonical_intent_message(raw), raw["signature"]):
            return web.json_response({"detail": "Invalid signature"}, status=401)
        if raw["nonce"] in self.seen_nonces:
            return web.json_response({"detail": "nonce already used"}, status=409)
        self.seen_nonces.add(raw["nonce"])
        self.intents.append(raw)
        if self.answer_override is not None:
            override = self.answer_override(raw)
            if isinstance(override, tuple):
                return web.json_response(override[1], status=override[0])
            return web.json_response(override)
        started = time.perf_counter()
        steps = raw["steps"]
        gpu = [n for n in ("matmul", "verifyx") if steps.get(n)]
        if raw["parallel_gpu"]:
            await asyncio.sleep(self.step_sleep)
        else:
            await asyncio.sleep(self.step_sleep * len(gpu))
        answer_steps = {}
        if steps.get("matmul"):
            answer_steps["matmul"] = {
                "status": "ok",
                "ms": int(self.step_sleep * 1000),
                "exit_status": 0,
                "stdout": matmul_stdout("GPU-1"),
                "stderr_tail": "",
            }
        if steps.get("verifyx"):
            answer_steps["verifyx"] = {
                "status": "ok",
                "ms": int(self.step_sleep * 1000),
                "exit_status": 0,
                "stdout": steps["verifyx"]["cipher_text"] + ("-ok" if self.verifyx_ok else "-no"),
                "stderr_tail": "",
                "data": {"lib_sha256": self.lib_sha},
            }
        for name in ("docker", "ports", "inspector"):
            if steps.get(name):
                answer_steps[name] = {"status": "ok", "ms": 3, "data": {"fact": name}}
        return web.json_response(
            {
                "schema": SCHEMA,
                "nonce": raw["nonce"],
                "executor_uuid": raw["executor_uuid"],
                "executor_version": "4.1.0",
                "started_at": int(time.time()),
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
                "deadline_hit": False,
                "steps": answer_steps,
                "signer": "none",
                "signature": None,
            }
        )


def context(
    keypair,
    executor_info,
    *,
    validation,
    verifyx,
    first_pass=True,
    verifyx_enabled=True,
    state=None,
):
    return make_context(
        executor=executor_info,
        services=build_services(
            validation=validation,
            verifyx=verifyx,
            redis=SimpleNamespace(renting_in_progress=AsyncMock(return_value=False)),
        ),
        config=build_context_config(
            validator_keypair=keypair, first_pass=first_pass, verifyx_enabled=verifyx_enabled
        ),
        state=state or build_state(specs=SPECS),
    )


def client_factory(keypair):
    return lambda ctx: LocalVerifyClient(
        keypair,
        timeout_s=settings.LOCAL_VERIFY_TIMEOUT_SECONDS,
        connect_timeout_s=settings.LOCAL_VERIFY_CONNECT_TIMEOUT_SECONDS,
    )


async def run_local_then_consumers(ctx, check: LocalVerifyCheck):
    """LocalVerifyCheck, then the two consumers on the state it left — the pipeline's order."""
    local = await check.run(ctx)
    state = local.updates.get("state", ctx.state)
    ctx2 = ctx.model_copy(update={"state": state})
    verifyx = await VerifyXCheck().run(ctx2)
    capability = await CapabilityCheck().run(ctx2)
    return local, verifyx, capability


# --- the client and the wire ---------------------------------------------------------------------


def test_intent_is_signed_over_the_canonical_document(keypair):
    intent = build_intent(
        executor_uuid="e",
        matmul={"dim_n": 1, "dim_k": 2, "seed": 3, "cipher_text": "c"},
        verifyx=None,
        parallel_gpu=True,
        deadline_s=60,
        now=1_000_000,
    )
    signed = sign_intent(intent, keypair)
    assert (
        signed["schema"] == SCHEMA
        and signed["issued_at"] == 1_000_000
        and signed["expires_at"] == 1_000_120
    )
    assert keypair.verify(canonical_intent_message(signed), signed["signature"])
    tampered = {**signed, "executor_uuid": "other"}
    assert not keypair.verify(canonical_intent_message(tampered), tampered["signature"])
    assert canonical_intent_message(signed) == canonical_intent_message(
        dict(reversed(list(signed.items())))
    )


def test_answer_must_echo_the_intent():
    intent = {"nonce": "n1", "executor_uuid": "e"}
    good = {
        "schema": SCHEMA,
        "nonce": "n1",
        "executor_uuid": "e",
        "steps": {"matmul": {"status": "ok", "stdout": "x"}},
    }
    answer = parse_answer(good, intent=intent, round_trip_ms=5)
    assert answer.step("matmul").stdout == "x" and answer.step("verifyx").status == "skipped"
    for bad, reason in (
        ({**good, "nonce": "n2"}, "nonce_mismatch"),
        ({**good, "executor_uuid": "z"}, "executor_mismatch"),
        ({**good, "schema": "lium.local_verify/0"}, "schema_mismatch"),
        ({**good, "steps": []}, "malformed"),
        ([], "malformed"),
    ):
        with pytest.raises(LocalVerifyUnavailable) as exc:
            parse_answer(bad, intent=intent, round_trip_ms=5)
        assert exc.value.reason == reason


def test_wire_contract_with_the_executor_side(keypair):
    """The intent this client builds is what the executor's own wire types accept, the signed bytes
    are the same function on both ends (datura), and the executor's result document parses here.
    The executor module (#1339, same repo) is loaded from its file: it needs only pydantic + datura."""
    import importlib.util
    import pathlib

    path = (
        pathlib.Path(__file__).resolve().parents[2] / "executor" / "src" / "payloads" / "verify.py"
    )
    spec = importlib.util.spec_from_file_location("executor_payloads_verify", path)
    exe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exe)
    from datura.requests.validator_requests import local_verify_signing_blob

    assert exe.SCHEMA == SCHEMA and exe.CAPABILITY == CAPABILITY
    assert canonical_intent_message is local_verify_signing_blob

    intent = build_intent(
        executor_uuid="e",
        matmul={"dim_n": 1900, "dim_k": 2_000_000, "seed": 3, "cipher_text": "c0ffee"},
        verifyx=None,
        parallel_gpu=True,
        deadline_s=executor_deadline_s(240),
    )
    signed = sign_intent(intent, keypair)
    # The executor validates the whole document, then rebuilds the message from the raw body.
    parsed = exe.VerifyIntent.model_validate(signed)
    assert parsed.steps.matmul.cipher_text == "c0ffee" and parsed.steps.verifyx is None
    assert parsed.deadline_s == 210 and parsed.schema_id == SCHEMA
    body = exe.VerifyIntentBody.model_validate(
        {k: v for k, v in signed.items() if k != "signature"}
    )
    assert keypair.verify(canonical_intent_message(signed), signed["signature"])
    assert body.nonce == intent["nonce"]

    result = exe.VerifyResult(
        nonce=intent["nonce"],
        executor_uuid="e",
        executor_version="4.1.0",
        started_at=1,
        elapsed_ms=2,
        steps={
            "matmul": exe.StepResult(status="ok", stdout="RESULT_JSON: {}", exit_status=0),
            "verifyx": exe.StepResult(status="timeout", error="timed out after 600s"),
            "docker": exe.StepResult(status="skipped"),
        },
    )
    answer = parse_answer(result.model_dump(by_alias=True), intent=intent, round_trip_ms=7)
    assert answer.step("matmul").stdout == "RESULT_JSON: {}"
    assert answer.step("verifyx").status == "timeout" and answer.step("ports").status == "skipped"

    # A number the executor got wrong reads as 0 and labels nothing else: the other steps' evidence
    # stays parseable instead of the whole answer becoming an internal error.
    doc = result.model_dump(by_alias=True)
    doc["elapsed_ms"] = "fast"
    doc["steps"]["matmul"]["ms"] = None
    answer = parse_answer(doc, intent=intent, round_trip_ms=7)
    assert answer.elapsed_ms == 0 and answer.step("matmul").ms == 0
    assert answer.step("matmul").stdout == "RESULT_JSON: {}"


@pytest.mark.asyncio
async def test_the_intent_the_check_sends_is_what_the_executor_accepts(
    keypair, monkeypatch, local_verify_on, verifyx_service
):
    """Not a hand-built intent: the document `LocalVerifyCheck._run` actually put on the wire (both
    GPU steps, the facts, the deadline) goes through the executor's own model."""
    import importlib.util
    import pathlib

    path = (
        pathlib.Path(__file__).resolve().parents[2] / "executor" / "src" / "payloads" / "verify.py"
    )
    spec = importlib.util.spec_from_file_location("executor_payloads_verify_2", path)
    exe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exe)

    validation = matmul_service(monkeypatch)
    async with FakeExecutor(keypair) as executor:
        ctx = context(
            keypair, executor.executor_info, validation=validation, verifyx=verifyx_service
        )
        await LocalVerifyCheck(client_factory=client_factory(keypair)).run(ctx)
    parsed = exe.VerifyIntent.model_validate(executor.intents[0])
    assert parsed.steps.matmul is not None and parsed.steps.verifyx is not None
    assert parsed.steps.docker and parsed.steps.ports and parsed.steps.inspector
    assert parsed.deadline_s == executor_deadline_s(settings.LOCAL_VERIFY_TIMEOUT_SECONDS)


@pytest.mark.asyncio
async def test_client_against_the_fake_executor(keypair, local_verify_on):
    async with FakeExecutor(keypair) as executor:
        client = client_factory(keypair)(None)
        assert await client.capabilities(executor.executor_info) == {CAPABILITY}
        intent = build_intent(
            executor_uuid=EXECUTOR_UUID,
            matmul={"dim_n": 1, "dim_k": 2, "seed": 3, "cipher_text": "c"},
            verifyx={"seed": 1, "cipher_text": "v"},
            parallel_gpu=True,
            deadline_s=5,
        )
        answer = await client.verify(executor.executor_info, intent)
        assert answer.nonce == intent["nonce"] and answer.step("matmul").status == "ok"
        # A replay of the same intent is refused by the executor (409, like a busy executor).
        with pytest.raises(LocalVerifyUnavailable) as exc:
            await client.verify(executor.executor_info, intent)
        assert exc.value.reason == "busy_or_replay"

        stranger = LocalVerifyClient(
            bittensor.Keypair.create_from_uri("//Stranger"), timeout_s=5, connect_timeout_s=2
        )
        with pytest.raises(LocalVerifyUnavailable) as exc:
            await stranger.verify(
                executor.executor_info,
                build_intent(
                    executor_uuid=EXECUTOR_UUID,
                    matmul=None,
                    verifyx=None,
                    parallel_gpu=False,
                    deadline_s=5,
                ),
            )
        assert exc.value.reason == "refused"


@pytest.mark.asyncio
async def test_client_reports_absent_route_and_dead_host_as_fallback_reasons(
    keypair, local_verify_on
):
    async with FakeExecutor(keypair) as executor:
        executor.answer_override = lambda raw: (404, {"detail": "Not Found"})
        client = client_factory(keypair)(None)
        with pytest.raises(LocalVerifyUnavailable) as exc:
            await client.verify(
                executor.executor_info,
                build_intent(
                    executor_uuid=EXECUTOR_UUID,
                    matmul=None,
                    verifyx=None,
                    parallel_gpu=False,
                    deadline_s=5,
                ),
            )
        assert exc.value.reason == "not_supported"
    dead = ExecutorSSHInfo(
        uuid=EXECUTOR_UUID,
        address="127.0.0.1",
        port=9,
        ssh_username="r",
        ssh_port=22,
        python_path="p",
        root_dir="/r",
    )
    assert await client.capabilities(dead) == set()
    with pytest.raises(LocalVerifyUnavailable) as exc:
        await client.verify(
            dead,
            build_intent(
                executor_uuid=EXECUTOR_UUID,
                matmul=None,
                verifyx=None,
                parallel_gpu=False,
                deadline_s=5,
            ),
        )
    assert exc.value.reason == "transport"


# --- the check and its consumers ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flag_off_leaves_everything_to_ssh(keypair, monkeypatch, verifyx_service):
    monkeypatch.setattr(settings, "VALIDATOR_LOCAL_VERIFY_ENABLED", False)
    validation = matmul_service(monkeypatch)
    validation.validate_gpu_model_and_process_job = AsyncMock(
        return_value=mvs.ValidationResult(success=True, metrics={"t": 1})
    )
    verifyx_service.validate_verifyx_and_process_job = AsyncMock(
        return_value=vvs.VerifyXResponse(
            data={"success": True, "network": {"download_speed": 900.0}}
        )
    )
    ctx = context(keypair, make_context().executor, validation=validation, verifyx=verifyx_service)
    factory = MagicMock(side_effect=AssertionError("no client when the flag is off"))

    local, verifyx, capability = await run_local_then_consumers(
        ctx, LocalVerifyCheck(client_factory=factory)
    )

    assert local.passed and local.event.reason_code == "LOCAL_VERIFY_DISABLED"
    assert "state" not in local.updates
    assert capability.passed and capability.event.what_we_saw["transport"] == "ssh"
    assert verifyx.passed and verifyx.event.what_we_saw["transport"] == "ssh"
    validation.validate_gpu_model_and_process_job.assert_awaited_once()
    verifyx_service.validate_verifyx_and_process_job.assert_awaited_once()


@pytest.mark.asyncio
async def test_advertised_executor_is_verified_in_one_call_and_ssh_is_not_used(
    keypair, monkeypatch, local_verify_on, verifyx_service
):
    validation = matmul_service(monkeypatch)
    ssh_matmul = AsyncMock(side_effect=AssertionError("SSH matmul must not run"))
    ssh_verifyx = AsyncMock(side_effect=AssertionError("SSH VerifyX must not run"))
    validation.validate_gpu_model_and_process_job = ssh_matmul
    verifyx_service.validate_verifyx_and_process_job = ssh_verifyx

    async with FakeExecutor(keypair) as executor:
        ctx = context(
            keypair, executor.executor_info, validation=validation, verifyx=verifyx_service
        )
        local, verifyx, capability = await run_local_then_consumers(
            ctx, LocalVerifyCheck(client_factory=client_factory(keypair))
        )

        assert local.passed and local.event.reason_code == "LOCAL_VERIFY_OK", local.event
        assert (
            local.event.what_we_saw["consumed"] == ["matmul", "verifyx"]
            and local.event.what_we_saw["fallbacks"] == {}
        )
        outcome: LocalVerifyOutcome = local.updates["state"].local_verify
        assert outcome.matmul.success and outcome.matmul.metrics == {"tflops": 42.0}
        assert outcome.verifyx.data["success"] and outcome.facts.keys() == {
            "docker",
            "ports",
            "inspector",
        }

        assert capability.passed and capability.event.what_we_saw["transport"] == "local_verify"
        assert capability.updates["state"].gpu_metrics == {"tflops": 42.0}
        assert verifyx.passed and verifyx.event.what_we_saw["transport"] == "local_verify"
        assert verifyx.updates["state"].specs["network"]["verifyx_download_speed"] == 900.0
        ssh_matmul.assert_not_awaited()
        ssh_verifyx.assert_not_awaited()

        # What went over the wire: the challenge as the SSH command would carry it, first-pass sized.
        sent = executor.intents[0]
        assert sent["parallel_gpu"] is True and sent["steps"]["matmul"]["cipher_text"] == "deadbeef"
        assert set(sent["steps"]["matmul"]) == {"dim_n", "dim_k", "seed", "cipher_text"}
        assert sent["steps"]["verifyx"]["cipher_text"].startswith("vx")
        assert sent["steps"]["docker"] and sent["steps"]["ports"] and sent["steps"]["inspector"]


@pytest.mark.asyncio
async def test_not_advertised_falls_back_before_any_challenge_is_built(
    keypair, monkeypatch, local_verify_on, verifyx_service
):
    validation = matmul_service(monkeypatch)
    validation.validate_gpu_model_and_process_job = AsyncMock(
        return_value=mvs.ValidationResult(success=True)
    )
    verifyx_service.validate_verifyx_and_process_job = AsyncMock(
        return_value=vvs.VerifyXResponse(data={"success": True, "network": {}})
    )
    async with FakeExecutor(keypair, advertise=False) as executor:
        ctx = context(
            keypair, executor.executor_info, validation=validation, verifyx=verifyx_service
        )
        local, verifyx, capability = await run_local_then_consumers(
            ctx, LocalVerifyCheck(client_factory=client_factory(keypair))
        )
    assert local.passed and local.event.reason_code == "LOCAL_VERIFY_NOT_ADVERTISED"
    assert executor.intents == [] and validation.wrapper.generateChallenge.call_count == 0
    assert (
        capability.event.what_we_saw["transport"] == "ssh"
        and verifyx.event.what_we_saw["transport"] == "ssh"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "override,reason",
    [
        (lambda raw: (404, {"detail": "Not Found"}), "not_supported"),
        (lambda raw: (409, {"detail": "busy"}), "busy_or_replay"),
        (lambda raw: (500, {"detail": "boom"}), "http_error"),
        (
            lambda raw: {
                "schema": SCHEMA,
                "nonce": "someone-elses",
                "executor_uuid": EXECUTOR_UUID,
                "steps": {},
            },
            "nonce_mismatch",
        ),
        (
            lambda raw: {
                "schema": SCHEMA,
                "nonce": raw["nonce"],
                "executor_uuid": "other",
                "steps": {},
            },
            "executor_mismatch",
        ),
    ],
)
async def test_every_refusal_or_mismatch_falls_back_to_ssh(
    keypair, monkeypatch, local_verify_on, verifyx_service, override, reason
):
    validation = matmul_service(monkeypatch)
    validation.validate_gpu_model_and_process_job = AsyncMock(
        return_value=mvs.ValidationResult(success=True)
    )
    verifyx_service.validate_verifyx_and_process_job = AsyncMock(
        return_value=vvs.VerifyXResponse(data={"success": True, "network": {}})
    )
    async with FakeExecutor(keypair) as executor:
        executor.answer_override = override
        ctx = context(
            keypair, executor.executor_info, validation=validation, verifyx=verifyx_service
        )
        local, verifyx, capability = await run_local_then_consumers(
            ctx, LocalVerifyCheck(client_factory=client_factory(keypair))
        )
    assert local.passed and local.event.reason_code == "LOCAL_VERIFY_FALLBACK"
    assert local.event.what_we_saw["reason"] == reason
    assert "state" not in local.updates
    assert capability.passed and capability.event.what_we_saw["transport"] == "ssh"
    assert verifyx.passed and verifyx.event.what_we_saw["transport"] == "ssh"
    # The challenge object is released even when the answer is unusable.
    validation.wrapper.free.assert_called_with("ptr")


@pytest.mark.asyncio
async def test_timeout_falls_back_to_ssh(keypair, monkeypatch, local_verify_on, verifyx_service):
    monkeypatch.setattr(settings, "LOCAL_VERIFY_TIMEOUT_SECONDS", 1)
    validation = matmul_service(monkeypatch)
    validation.validate_gpu_model_and_process_job = AsyncMock(
        return_value=mvs.ValidationResult(success=True)
    )
    verifyx_service.validate_verifyx_and_process_job = AsyncMock(
        return_value=vvs.VerifyXResponse(data={"success": True, "network": {}})
    )
    async with FakeExecutor(keypair, step_sleep=3.0) as executor:
        ctx = context(
            keypair, executor.executor_info, validation=validation, verifyx=verifyx_service
        )
        started = time.perf_counter()
        local, verifyx, capability = await run_local_then_consumers(
            ctx, LocalVerifyCheck(client_factory=client_factory(keypair))
        )
        assert time.perf_counter() - started < 2.5
    assert (
        local.event.reason_code == "LOCAL_VERIFY_FALLBACK"
        and local.event.what_we_saw["reason"] == "timeout"
    )
    assert (
        capability.event.what_we_saw["transport"] == "ssh"
        and verifyx.event.what_we_saw["transport"] == "ssh"
    )


@pytest.mark.asyncio
async def test_a_step_that_fails_the_judgement_is_left_to_ssh_per_step(
    keypair, monkeypatch, local_verify_on, verifyx_service
):
    """A judged FAILURE of the local run never decides: the SSH run of that step does. The other
    step is still consumed. The metric line names the step and reason."""
    validation = matmul_service(
        monkeypatch, sealed="not-the-challenge-uuid"
    )  # unseals to the wrong uuid
    ssh_matmul = AsyncMock(return_value=mvs.ValidationResult(success=True, metrics={"from": "ssh"}))
    validation.validate_gpu_model_and_process_job = ssh_matmul
    verifyx_service.validate_verifyx_and_process_job = AsyncMock(
        side_effect=AssertionError("VerifyX was consumed locally")
    )
    async with FakeExecutor(keypair) as executor:
        ctx = context(
            keypair, executor.executor_info, validation=validation, verifyx=verifyx_service
        )
        with patch("neurons.validators.src.services.task.checks.local_verify.logger") as log:
            local, verifyx, capability = await run_local_then_consumers(
                ctx, LocalVerifyCheck(client_factory=client_factory(keypair))
            )
    assert local.event.reason_code == "LOCAL_VERIFY_OK"
    assert local.event.what_we_saw["consumed"] == ["verifyx"]
    assert local.event.what_we_saw["fallbacks"] == {"matmul": "local_failed"}
    assert capability.passed and capability.event.what_we_saw["transport"] == "ssh"
    assert capability.updates["state"].gpu_metrics == {"from": "ssh"}
    assert verifyx.event.what_we_saw["transport"] == "local_verify"
    outcomes = [
        call.args[0].extra
        for call in log.info.call_args_list
        if str(call.args[0]) == "[local_verify] outcome"
    ]
    by_step = {o["step"]: o for o in outcomes}
    assert (
        by_step["matmul"]["outcome"] == "fallback" and by_step["matmul"]["reason"] == "local_failed"
    )
    assert by_step["matmul"]["detail"].startswith("UUID mismatch")
    assert by_step["verifyx"]["outcome"] == "consumed" and by_step["verifyx"]["reason"] == "ok"
    assert all(o["first_pass"] is True and "round_trip_ms" in o for o in outcomes)


@pytest.mark.asyncio
async def test_outdated_libverifyx_on_the_executor_falls_back_like_the_ssh_checksum_gate(
    keypair, monkeypatch, local_verify_on, verifyx_service
):
    validation = matmul_service(monkeypatch)
    validation.validate_gpu_model_and_process_job = AsyncMock(
        return_value=mvs.ValidationResult(success=True)
    )
    verifyx_service.validate_verifyx_and_process_job = AsyncMock(
        return_value=vvs.VerifyXResponse(data={"success": True, "network": {}})
    )
    async with FakeExecutor(keypair, lib_sha="stale") as executor:
        ctx = context(
            keypair, executor.executor_info, validation=validation, verifyx=verifyx_service
        )
        local, verifyx, _ = await run_local_then_consumers(
            ctx, LocalVerifyCheck(client_factory=client_factory(keypair))
        )
    assert local.event.what_we_saw["fallbacks"] == {"verifyx": "lib_mismatch"}
    assert local.event.what_we_saw["consumed"] == ["matmul"]
    assert verifyx.event.what_we_saw["transport"] == "ssh"


@pytest.mark.asyncio
async def test_a_verifyx_response_the_library_rejects_is_left_to_ssh(
    keypair, monkeypatch, local_verify_on, verifyx_service
):
    """The judged failure of the OTHER step: libverifyx refuses the response → VerifyXCheck runs
    over SSH; the matmul beside it is still consumed."""
    validation = matmul_service(monkeypatch)
    validation.validate_gpu_model_and_process_job = AsyncMock(
        side_effect=AssertionError("matmul was consumed locally")
    )
    ssh_verifyx = AsyncMock(
        return_value=vvs.VerifyXResponse(
            data={"success": True, "network": {"download_speed": 1.0, "upload_speed": 1.0}}
        )
    )
    verifyx_service.validate_verifyx_and_process_job = ssh_verifyx
    async with FakeExecutor(keypair, verifyx_ok=False) as executor:
        ctx = context(
            keypair, executor.executor_info, validation=validation, verifyx=verifyx_service
        )
        local, verifyx, capability = await run_local_then_consumers(
            ctx, LocalVerifyCheck(client_factory=client_factory(keypair))
        )
    assert local.event.what_we_saw["consumed"] == ["matmul"]
    assert local.event.what_we_saw["fallbacks"] == {"verifyx": "local_failed"}
    assert capability.event.what_we_saw["transport"] == "local_verify"
    assert verifyx.event.what_we_saw["transport"] == "ssh"
    ssh_verifyx.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "matmul_step,verifyx_step,expected_fallbacks,expected_consumed",
    [
        (
            {"status": "timeout", "error": "timed out after 120s"},
            None,
            {"matmul": "step_timeout"},
            ["verifyx"],
        ),
        (
            {"status": "failed", "exit_status": 1, "stderr_tail": "CUDA error"},
            None,
            {"matmul": "step_failed"},
            ["verifyx"],
        ),
        ({"status": "ok", "exit_status": 0}, None, {"matmul": "step_no_stdout"}, ["verifyx"]),
        (None, {"status": "unsupported"}, {"verifyx": "step_malformed"}, ["matmul"]),
        (None, {"status": "skipped"}, {"verifyx": "step_skipped"}, ["matmul"]),
        (None, "absent", {"verifyx": "step_skipped"}, ["matmul"]),
        (None, "not-an-object", {"verifyx": "step_malformed"}, ["matmul"]),
    ],
)
async def test_a_step_that_did_not_run_is_left_to_ssh_and_the_other_is_consumed(
    keypair,
    monkeypatch,
    local_verify_on,
    verifyx_service,
    matmul_step,
    verifyx_step,
    expected_fallbacks,
    expected_consumed,
):
    """Every non-`ok` status the executor can answer with — and an `ok` without stdout, a step it
    left out, a step that is not an object, a status outside the executor's closed set — falls back
    for THAT step only."""
    validation = matmul_service(monkeypatch)
    ssh_matmul = AsyncMock(return_value=mvs.ValidationResult(success=True, metrics={"from": "ssh"}))
    ssh_verifyx = AsyncMock(
        return_value=vvs.VerifyXResponse(
            data={"success": True, "network": {"download_speed": 1.0, "upload_speed": 1.0}}
        )
    )
    validation.validate_gpu_model_and_process_job = ssh_matmul
    verifyx_service.validate_verifyx_and_process_job = ssh_verifyx

    def cut(raw):
        # The executor's own answer with one step replaced — a real `deadline_hit` document.
        answer = {
            "schema": SCHEMA,
            "nonce": raw["nonce"],
            "executor_uuid": raw["executor_uuid"],
            "executor_version": "4.1.0",
            "started_at": int(time.time()),
            "elapsed_ms": 12,
            "deadline_hit": True,
            "steps": {
                "matmul": {
                    "status": "ok",
                    "ms": 1,
                    "exit_status": 0,
                    "stdout": matmul_stdout("GPU-1"),
                    "stderr_tail": "",
                },
                "verifyx": {
                    "status": "ok",
                    "ms": 1,
                    "exit_status": 0,
                    "stdout": raw["steps"]["verifyx"]["cipher_text"] + "-ok",
                    "stderr_tail": "",
                    "data": {"lib_sha256": "lib-sha"},
                },
            },
        }
        if matmul_step is not None:
            answer["steps"]["matmul"] = matmul_step
        if verifyx_step == "absent":
            del answer["steps"]["verifyx"]
        elif verifyx_step is not None:
            answer["steps"]["verifyx"] = verifyx_step
        return answer

    async with FakeExecutor(keypair) as executor:
        executor.answer_override = cut
        ctx = context(
            keypair, executor.executor_info, validation=validation, verifyx=verifyx_service
        )
        local, verifyx, capability = await run_local_then_consumers(
            ctx, LocalVerifyCheck(client_factory=client_factory(keypair))
        )
    assert local.passed and local.event.reason_code == "LOCAL_VERIFY_OK"
    assert local.event.what_we_saw["fallbacks"] == expected_fallbacks
    assert local.event.what_we_saw["consumed"] == expected_consumed
    assert local.event.what_we_saw["deadline_hit"] is True
    fell_back = set(expected_fallbacks)
    assert capability.event.what_we_saw["transport"] == (
        "ssh" if "matmul" in fell_back else "local_verify"
    )
    assert verifyx.event.what_we_saw["transport"] == (
        "ssh" if "verifyx" in fell_back else "local_verify"
    )
    assert ssh_matmul.await_count == (1 if "matmul" in fell_back else 0)
    assert ssh_verifyx.await_count == (1 if "verifyx" in fell_back else 0)


def test_executor_deadline_leaves_the_answer_room_inside_the_client_timeout():
    """The intent's `deadline_s` is the client's whole-call timeout minus a margin, never below the
    executor's minimum, so a cut answer can be consumed instead of timing out on the wire."""
    assert executor_deadline_s(240) == 240 - EXECUTOR_DEADLINE_MARGIN_SECONDS == 210
    assert executor_deadline_s(31) == 5 and executor_deadline_s(5) == 5
    # …and never above what the executor's `deadline_s` field accepts (le=3600 → a 422 otherwise).
    assert executor_deadline_s(10_000) == EXECUTOR_DEADLINE_MAX_SECONDS == 3600
    assert EXECUTOR_DEADLINE_MARGIN_SECONDS >= 10


@pytest.mark.asyncio
async def test_intent_carries_the_shortened_deadline(
    keypair, monkeypatch, local_verify_on, verifyx_service
):
    monkeypatch.setattr(settings, "LOCAL_VERIFY_TIMEOUT_SECONDS", 100)
    validation = matmul_service(monkeypatch)
    async with FakeExecutor(keypair) as executor:
        ctx = context(
            keypair, executor.executor_info, validation=validation, verifyx=verifyx_service
        )
        await LocalVerifyCheck(client_factory=client_factory(keypair)).run(ctx)
    assert executor.intents[0]["deadline_s"] == 100 - EXECUTOR_DEADLINE_MARGIN_SECONDS


@pytest.mark.asyncio
async def test_all_cards_check_keeps_the_matmul_on_ssh(
    keypair, monkeypatch, local_verify_on, verifyx_service
):
    """DAH-2671 item 3: `_probe_all_claimed_cards` runs inside the SSH matmul path only. With the
    check on, the intent carries no matmul, CapabilityCheck runs the SSH path (probe included) and
    VerifyX is still consumed locally."""
    monkeypatch.setattr(settings, "MATMUL_ALLCARDS_CHECK_ENABLED", True)
    validation = matmul_service(monkeypatch)
    ssh_matmul = AsyncMock(return_value=mvs.ValidationResult(success=True, metrics={"from": "ssh"}))
    validation.validate_gpu_model_and_process_job = ssh_matmul
    verifyx_service.validate_verifyx_and_process_job = AsyncMock(
        side_effect=AssertionError("VerifyX was consumed locally")
    )
    async with FakeExecutor(keypair) as executor:
        ctx = context(
            keypair, executor.executor_info, validation=validation, verifyx=verifyx_service
        )
        local, verifyx, capability = await run_local_then_consumers(
            ctx, LocalVerifyCheck(client_factory=client_factory(keypair))
        )
    assert executor.intents[0]["steps"]["matmul"] is None
    assert validation.wrapper.generateChallenge.call_count == 0  # no challenge built for nothing
    assert local.event.what_we_saw["consumed"] == ["verifyx"]
    assert local.event.what_we_saw["fallbacks"] == {"matmul": "allcards_ssh"}
    assert capability.event.what_we_saw["transport"] == "ssh"
    assert capability.updates["state"].gpu_metrics == {"from": "ssh"}
    ssh_matmul.assert_awaited_once()
    assert verifyx.event.what_we_saw["transport"] == "local_verify"


@pytest.mark.asyncio
async def test_all_cards_check_with_verifyx_off_makes_no_call(
    keypair, monkeypatch, local_verify_on, verifyx_service
):
    monkeypatch.setattr(settings, "MATMUL_ALLCARDS_CHECK_ENABLED", True)
    validation = matmul_service(monkeypatch)
    async with FakeExecutor(keypair) as executor:
        ctx = context(
            keypair,
            executor.executor_info,
            validation=validation,
            verifyx=verifyx_service,
            verifyx_enabled=False,
        )
        factory = MagicMock(side_effect=AssertionError("nothing to run: no client, no /version"))
        result = await LocalVerifyCheck(client_factory=factory).run(ctx)
    assert result.passed and result.event.reason_code == "LOCAL_VERIFY_FALLBACK"
    assert result.event.what_we_saw["reason"] == "allcards_ssh"
    assert executor.intents == [] and not factory.called


@pytest.mark.asyncio
async def test_a_bug_in_the_local_path_is_a_fallback_not_a_cycle_abort(
    keypair, monkeypatch, local_verify_on, verifyx_service
):
    """`Pipeline.run` has no exception guard: an unexpected error inside this check would end the
    node's whole cycle. It must surface as a fallback so both consumers still run over SSH."""
    validation = matmul_service(monkeypatch)
    validation.evaluate_matmul_output = MagicMock(side_effect=KeyError("metrics"))
    validation.validate_gpu_model_and_process_job = AsyncMock(
        return_value=mvs.ValidationResult(success=True, metrics={"from": "ssh"})
    )
    verifyx_service.validate_verifyx_and_process_job = AsyncMock(
        return_value=vvs.VerifyXResponse(data={"success": True, "network": {}})
    )
    async with FakeExecutor(keypair) as executor:
        ctx = context(
            keypair, executor.executor_info, validation=validation, verifyx=verifyx_service
        )
        local, verifyx, capability = await run_local_then_consumers(
            ctx, LocalVerifyCheck(client_factory=client_factory(keypair))
        )
    assert local.passed and local.event.reason_code == "LOCAL_VERIFY_FALLBACK"
    assert local.event.what_we_saw["reason"] == "internal_error"
    assert local.event.what_we_saw["detail"].startswith("KeyError")
    assert not local.updates
    assert verifyx.event.what_we_saw["transport"] == "ssh"
    assert capability.event.what_we_saw["transport"] == "ssh"


@pytest.mark.asyncio
async def test_no_specs_or_filler_only_skip_without_a_call(
    keypair, monkeypatch, local_verify_on, verifyx_service
):
    validation = matmul_service(monkeypatch)
    factory = MagicMock(side_effect=AssertionError("no client without specs"))
    ctx = context(
        keypair,
        make_context().executor,
        validation=validation,
        verifyx=verifyx_service,
        state=build_state(specs={}),
    )
    result = await LocalVerifyCheck(client_factory=factory).run(ctx)
    assert result.passed and result.event.reason_code == "LOCAL_VERIFY_SKIPPED"


@pytest.mark.asyncio
async def test_filler_only_node_is_skipped_without_a_call(
    keypair, monkeypatch, local_verify_on, verifyx_service
):
    """A node running only Lium's own filler (no customer pod) is not probed — the same rule as
    `CapabilityCheck` (`_get_filler_only_container`); a filler beside a customer pod is a rental
    and is handled by the rented branch, not here."""
    validation = matmul_service(monkeypatch)
    factory = MagicMock(side_effect=AssertionError("no client for a filler-only node"))
    async with FakeExecutor(keypair) as executor:
        ctx = context(
            keypair,
            executor.executor_info,
            validation=validation,
            verifyx=verifyx_service,
            state=build_state(
                specs=SPECS,
                rented_data=RentedExecutorsResponse(
                    executors={},
                    banned_guids=[],
                    filler_containers_by_executor={EXECUTOR_UUID: "filler_active"},
                ),
            ),
        )
        result = await LocalVerifyCheck(client_factory=factory).run(ctx)
        assert executor.intents == []
    assert result.passed and result.event.reason_code == "LOCAL_VERIFY_SKIPPED"
    assert result.event.what_we_saw["why"] == "filler only"
    factory.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("slow_step", ["matmul", "verifyx"])
async def test_a_pass_slower_than_the_ssh_cap_is_left_to_ssh(
    keypair, monkeypatch, local_verify_on, verifyx_service, slow_step
):
    """The SSH path fails a matmul past MATRIX_VERIFY_TIMEOUT_SECONDS and a VerifyX past
    VERIFYX_COMMAND_TIMEOUT_SECONDS; the executor's `ms` is its own word, so the validator's round
    trip is held to the same cap and a slower pass is not consumed — the SSH run decides."""
    validation = matmul_service(monkeypatch)
    ssh_matmul = AsyncMock(return_value=mvs.ValidationResult(success=True, metrics={"from": "ssh"}))
    ssh_verifyx = AsyncMock(
        return_value=vvs.VerifyXResponse(
            data={"success": True, "network": {"download_speed": 1.0, "upload_speed": 1.0}}
        )
    )
    validation.validate_gpu_model_and_process_job = ssh_matmul
    verifyx_service.validate_verifyx_and_process_job = ssh_verifyx
    # The fake answers in under a millisecond; only the slow step's cap is pulled under that.
    monkeypatch.setitem(STEP_WALL_CLOCK_CAP_MS, slow_step, -1)
    other = "verifyx" if slow_step == "matmul" else "matmul"

    async with FakeExecutor(keypair) as executor:
        ctx = context(
            keypair, executor.executor_info, validation=validation, verifyx=verifyx_service
        )
        local, verifyx, capability = await run_local_then_consumers(
            ctx, LocalVerifyCheck(client_factory=client_factory(keypair))
        )
    assert local.event.what_we_saw["fallbacks"] == {slow_step: "step_overtime"}
    assert local.event.what_we_saw["consumed"] == [other]
    assert local.event.what_we_saw["round_trip_ms"] > STEP_WALL_CLOCK_CAP_MS[slow_step]
    slow, fast = (capability, verifyx) if slow_step == "matmul" else (verifyx, capability)
    assert slow.event.what_we_saw["transport"] == "ssh"
    assert fast.event.what_we_saw["transport"] == "local_verify"
    assert (ssh_matmul.await_count, ssh_verifyx.await_count) == (
        (1, 0) if slow_step == "matmul" else (0, 1)
    )


def test_the_wall_clock_caps_are_the_ssh_paths_own():
    assert (
        STEP_WALL_CLOCK_CAP_MS
        == {
            "matmul": mvs.MATRIX_VERIFY_TIMEOUT_SECONDS * 1000,
            "verifyx": vvs.VERIFYX_COMMAND_TIMEOUT_SECONDS * 1000,
        }
        == {"matmul": 120_000, "verifyx": 600_000}
    )


def test_executor_controlled_strings_never_reach_a_label_uncapped():
    """PR_PROCESS §5: a status outside the executor's closed set is `malformed` (so every
    `step_<status>` label is from a closed set), unknown step keys are dropped, `executor_version`
    is capped, `exit_status` must be an int."""
    intent = build_intent(
        executor_uuid="e", matmul=None, verifyx=None, parallel_gpu=False, deadline_s=5
    )
    junk = "x" * 5000 + "\n"
    answer = parse_answer(
        {
            "schema": SCHEMA,
            "nonce": intent["nonce"],
            "executor_uuid": "e",
            "executor_version": "v" * 10_000,
            "elapsed_ms": 1,
            "steps": {
                "matmul": {"status": junk, "stdout": "RESULT_JSON: {}"},
                "verifyx": {"status": "ok", "exit_status": "0", "stdout": "x"},
                **{f"junk{i}": {"status": "ok"} for i in range(2000)},
            },
        },
        intent=intent,
        round_trip_ms=3,
    )
    assert set(answer.steps) == {"matmul", "verifyx"}
    assert answer.step("matmul").status == "malformed" and answer.step("matmul").stdout is None
    assert answer.step("verifyx").exit_status is None
    assert len(answer.executor_version) == EXECUTOR_VERSION_MAX_CHARS


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer, reason",
    [
        ({"schema": "s" * 5000, "nonce": "n"}, "schema_mismatch"),
        ((500, {"detail": "e" * 5000}), "http_error"),
    ],
)
async def test_the_fallback_event_detail_is_capped_like_the_metric(
    keypair, monkeypatch, local_verify_on, verifyx_service, answer, reason
):
    validation = matmul_service(monkeypatch)
    async with FakeExecutor(keypair) as executor:
        executor.answer_override = lambda raw: answer
        ctx = context(
            keypair, executor.executor_info, validation=validation, verifyx=verifyx_service
        )
        with patch("neurons.validators.src.services.task.checks.local_verify.logger") as log:
            result = await LocalVerifyCheck(client_factory=client_factory(keypair)).run(ctx)
    assert result.event.reason_code == "LOCAL_VERIFY_FALLBACK"
    assert result.event.what_we_saw["reason"] == reason
    assert len(result.event.what_we_saw["detail"]) <= 300
    lines = [
        c.args[0].extra for c in log.info.call_args_list if str(c.args[0]).startswith("[local")
    ]
    assert lines and all(len(o["detail"]) <= 300 for o in lines)


@pytest.mark.asyncio
async def test_an_oversized_capability_list_and_answer_are_bounded_before_use(
    keypair, monkeypatch, local_verify_on, verifyx_service
):
    """`/version` may list anything; only ≤ 32 strings of ≤ 64 chars are kept, so the
    NOT_ADVERTISED event is bounded. An answer longer than MAX_ANSWER_BYTES is `malformed`."""
    validation = matmul_service(monkeypatch)
    async with FakeExecutor(keypair) as executor:
        executor.version_override = {
            "version": "4.1.0",
            "capabilities": ["c" * 100_000, 7, *[f"cap{i}" for i in range(500)]],
        }
        ctx = context(
            keypair, executor.executor_info, validation=validation, verifyx=verifyx_service
        )
        result = await LocalVerifyCheck(client_factory=client_factory(keypair)).run(ctx)
        assert result.event.reason_code == "LOCAL_VERIFY_NOT_ADVERTISED"
        caps = result.event.what_we_saw["capabilities"]
        assert len(caps) <= MAX_CAPABILITIES and all(len(c) <= MAX_CAPABILITY_CHARS for c in caps)
        assert executor.intents == []

        executor.version_override = None
        executor.answer_override = lambda raw: {"pad": "x" * (MAX_ANSWER_BYTES + 10)}
        result = await LocalVerifyCheck(client_factory=client_factory(keypair)).run(ctx)
    assert result.event.what_we_saw["reason"] == "malformed"
    assert result.event.what_we_saw["detail"].startswith("answer longer than")


@pytest.mark.asyncio
async def test_a_scored_cycle_sends_full_size_challenges_serially(
    keypair, monkeypatch, local_verify_on, verifyx_service
):
    """Not the first pass: `parallel_gpu` is off (the two GPU steps run one after the other, the
    OOM-safety rule of the SSH path), the matmul has no VRAM budget and VerifyX no first-pass
    overrides, and every metric line says `first_pass: False`."""
    validation = matmul_service(monkeypatch)
    prepare_matmul = MagicMock(wraps=validation.prepare_matmul_challenge)
    prepare_verifyx = MagicMock(wraps=verifyx_service.prepare_verifyx_challenge)
    validation.prepare_matmul_challenge = prepare_matmul
    verifyx_service.prepare_verifyx_challenge = prepare_verifyx
    async with FakeExecutor(keypair) as executor:
        ctx = context(
            keypair,
            executor.executor_info,
            validation=validation,
            verifyx=verifyx_service,
            first_pass=False,
        )
        with patch("neurons.validators.src.services.task.checks.local_verify.logger") as log:
            local = await LocalVerifyCheck(client_factory=client_factory(keypair)).run(ctx)
    assert local.event.what_we_saw["consumed"] == ["matmul", "verifyx"]
    assert executor.intents[0]["parallel_gpu"] is False
    assert prepare_matmul.call_args.kwargs["vram_budget_mb"] is None
    assert prepare_verifyx.call_args.kwargs["challenge_config_overrides"] is None
    outcomes = [
        call.args[0].extra
        for call in log.info.call_args_list
        if str(call.args[0]) == "[local_verify] outcome"
    ]
    assert outcomes and all(o["first_pass"] is False for o in outcomes)


def test_pipeline_runs_local_verify_after_tenant_enforcement_and_before_both_consumers():
    names = [type(c).__name__ for c in PipelineFactory.build_checks()]
    assert names.index("TenantEnforcementCheck") < names.index("LocalVerifyCheck")
    assert (
        names.index("LocalVerifyCheck")
        < names.index("VerifyXCheck")
        < names.index("CapabilityCheck")
    )
    assert "LocalVerifyCheck" not in [
        type(c).__name__ for c in PipelineFactory.build_dry_run_checks()
    ]


# --- equivalence: one judge for both transports ------------------------------------------------


@pytest.mark.asyncio
async def test_ssh_and_local_transports_of_the_same_output_give_the_same_verdict(
    keypair, monkeypatch, verifyx_service
):
    """The stub's stdout, once through SSH (`validate_gpu_model_and_process_job`) and once as a
    `/verify` step (`evaluate_matmul_output` on a prepared challenge), yields equal ValidationResults;
    likewise VerifyX. That is what makes a local result comparable to an SSH one."""
    validation = matmul_service(monkeypatch)
    proven = matmul_stdout("GPU-1")
    stdout_holder = {}

    async def fake_ssh_run(command, timeout=None):
        stdout_holder["command"] = command
        return SimpleNamespace(stdout=proven, stderr="", exit_status=0)

    ssh_client = SimpleNamespace(run=fake_ssh_run)
    executor = SimpleNamespace(root_dir="/root/app", python_path="/usr/bin/python")
    ssh_result = await validation.validate_gpu_model_and_process_job(
        ssh_client, executor, {}, SPECS, vram_budget_mb=8192
    )

    challenge = validation.prepare_matmul_challenge(SPECS, {}, vram_budget_mb=8192)
    local_result = validation.evaluate_matmul_output(challenge, stdout=proven, stderr="")
    challenge.close()

    # Same uuid check, same sealed metrics, same stdout kept; only the per-call random uuid differs.
    for result in (ssh_result, local_result):
        assert (
            result.success
            and result.metrics == {"tflops": 42.0}
            and result.stdout == proven.strip()
        )
    assert (
        ssh_result.returned_uuid == ssh_result.expected_uuid
        and local_result.returned_uuid == local_result.expected_uuid
    )
    # The SSH command carried exactly the four arguments the intent carries.
    assert stdout_holder["command"].split("decrypt_challenge.py ")[1].split()[::2] == [
        "--dim_n",
        "--dim_k",
        "--seed",
        "--cipher_text",
    ]

    # A wrong answer is a wrong answer on both transports.
    wrong = (
        matmul_stdout("GPU-1").replace('"sealed": "cafe"', '"sealed": ""').replace("GPU-1", "GPU-9")
    )
    ssh_wrong = await validation.validate_gpu_model_and_process_job(
        SimpleNamespace(
            run=AsyncMock(return_value=SimpleNamespace(stdout=wrong, stderr="", exit_status=0))
        ),
        executor,
        {},
        SPECS,
    )
    challenge = validation.prepare_matmul_challenge(SPECS, {})
    local_wrong = validation.evaluate_matmul_output(challenge, stdout=wrong)
    challenge.close()
    assert not ssh_wrong.success and not local_wrong.success
    assert ssh_wrong.error_message.startswith(
        "UUID mismatch"
    ) and local_wrong.error_message.startswith("UUID mismatch")

    # VerifyX: the SSH path and the local path judge one response with one function.
    shell = MagicMock()
    shell.get_sha256_checksum_by_path = AsyncMock(return_value="lib-sha")
    captured = {}

    async def run(command, timeout=None):
        captured["cipher"] = command.split("--cipher_text ")[1]
        return SimpleNamespace(stdout=captured["cipher"] + "-ok", stderr="", exit_status=0)

    shell.ssh_client = SimpleNamespace(run=run)
    ssh_vx = await verifyx_service.validate_verifyx_and_process_job(shell, executor, {}, SPECS)
    vx_challenge = verifyx_service.prepare_verifyx_challenge(SPECS, {})
    local_vx = verifyx_service.evaluate_verifyx_capture(
        vx_challenge,
        vvs.SSHCapture(stdout=vx_challenge.cipher_text + "-ok", stderr="", exit_status=0),
        {},
    )
    assert ssh_vx.data == local_vx.data and ssh_vx.error is None and local_vx.error is None
    rejected = verifyx_service.evaluate_verifyx_capture(
        vx_challenge, vvs.SSHCapture(stdout="x" * 80, stderr="", exit_status=0), {}
    )
    assert (
        rejected.error.startswith("challenge verification failed")
        and rejected.diagnostics["failure_class"] == "CIPHER_REJECTED"
    )


# --- the e2e-stack measurement: SSH sequence vs one call on the stub ------------------------------


@pytest.mark.asyncio
async def test_measure_one_call_against_the_ssh_sequence_on_the_stub(
    keypair, monkeypatch, local_verify_on, verifyx_service, capsys
):
    """Stub node: matmul and VerifyX each take STEP seconds of node work; SSH costs RTT per command.
    SSH path today = checksum + VerifyX + matmul, three serial commands; local path = one HTTP call
    with the two probes side by side. Printed for the PR body; asserted loosely (CI boxes vary)."""
    STEP, RTT = 0.4, 0.15
    validation = matmul_service(monkeypatch)

    async def ssh_run(command, timeout=None):
        await asyncio.sleep(RTT)
        if "decrypt_challenge.py" in command:
            await asyncio.sleep(STEP)
            return SimpleNamespace(stdout=matmul_stdout("GPU-1"), stderr="", exit_status=0)
        await asyncio.sleep(STEP)
        return SimpleNamespace(
            stdout=command.split("--cipher_text ")[1] + "-ok", stderr="", exit_status=0
        )

    async def ssh_checksum(path):
        await asyncio.sleep(RTT)
        return "lib-sha"

    shell = SimpleNamespace(
        ssh_client=SimpleNamespace(run=ssh_run), get_sha256_checksum_by_path=ssh_checksum
    )
    executor_ssh = SimpleNamespace(root_dir="/root/app", python_path="/usr/bin/python")

    started = time.perf_counter()
    vx = await verifyx_service.validate_verifyx_and_process_job(shell, executor_ssh, {}, SPECS)
    mm = await validation.validate_gpu_model_and_process_job(
        shell.ssh_client, executor_ssh, {}, SPECS
    )
    ssh_s = time.perf_counter() - started
    assert vx.data["success"] and mm.success

    async with FakeExecutor(keypair, step_sleep=STEP) as executor:
        ctx = context(
            keypair, executor.executor_info, validation=validation, verifyx=verifyx_service
        )
        started = time.perf_counter()
        local, verifyx, capability = await run_local_then_consumers(
            ctx, LocalVerifyCheck(client_factory=client_factory(keypair))
        )
        local_s = time.perf_counter() - started
    assert local.event.what_we_saw["consumed"] == ["matmul", "verifyx"]

    print(
        f"\nMEASURE stub: ssh_sequence={ssh_s:.2f}s (3 commands, RTT {RTT}s, step {STEP}s serial) "
        f"one_call={local_s:.2f}s (1 GET /version + 1 POST /verify, steps parallel) saving={ssh_s - local_s:.2f}s"
    )
    assert local_s < ssh_s
