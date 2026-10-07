"""liumd phase 1 (DAH-2834): the matmul and VerifyX from one signed `POST /verify`, judged by the
same functions the SSH path uses, with the SSH path as the fallback for every other outcome.

Fake executor: an in-process aiohttp server that checks the intent signature the way the executor
does (canonical JSON, validator hotkey), refuses a replayed nonce, refuses 403 a `/verify` whose
peer did not come through the tunnel (the real route admits loopback peers only), and answers with
stub script output after a configurable per-step sleep. Fake tunnel (`FakeSSH`): the one asyncssh
call the client makes, `forward_local_port`, as a loopback listener piped to the destination the
way sshd's direct-tcpip channel is. Fake SSH command path: an `ssh_client.run` that sleeps one RTT
per command. The timing test at the end is the e2e-stack measurement quoted in the PR body.
"""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import bittensor
import pytest
import services.matrix_validation_service as mvs
import services.verifyx_validation_service as vvs
from aiohttp import web
from aiohttp.test_utils import TestServer
from datura.requests.miner_requests import ExecutorSSHInfo
from datura.requests.validator_requests import MatmulStep
from neurons.validators.src.services.task.checks.capability import CapabilityCheck
from neurons.validators.src.services.task.checks.local_verify import (
    LocalVerifyCheck,
)
from neurons.validators.src.services.task.checks.verifyx import VerifyXCheck
from services.local_verify_client import (
    CAPABILITY,
    SCHEMA,
    LocalVerifyClient,
    LocalVerifyUnavailable,
    build_intent,
    canonical_intent_message,
    parse_answer,
    sign_intent,
)

from core.config import settings
from tests.helpers import build_context_config, build_services, build_state, make_context

SPECS = {"gpu": {"count": 1, "details": [{"uuid": "GPU-1", "name": "H100", "capacity": 81559}]}}
EXECUTOR_UUID = "executor-123"
MINER_HOTKEY = "5MinerHotkeyOfTheExecutorUnderTest"


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


# The source ports of the connections FakeSSH tunnels are open right now. FakeExecutor's `/verify`
# admits a peer only from this set: on one host, sshd's direct-tcpip channel is the only way a
# request reaches the loopback-bound route, and the fake keeps that property.
_TUNNELLED_SOURCE_PORTS: set[int] = set()


class _FakeListener:
    """`asyncssh.SSHListener` as the client uses it: `get_port`, `close`, `wait_closed`."""

    def __init__(self, ssh: FakeSSH, server: asyncio.Server):
        self._ssh = ssh
        self._server = server

    def get_port(self) -> int:
        return self._server.sockets[0].getsockname()[1]

    def close(self) -> None:
        self._server.close()
        self._ssh.open_listeners -= 1

    async def wait_closed(self) -> None:
        await self._server.wait_closed()


class FakeSSH:
    """`asyncssh.SSHClientConnection.forward_local_port` as the pipeline's session provides it: a
    listener bound here, each accepted connection piped to `dest_host:dest_port` like the
    direct-tcpip channel sshd opens. A destination nobody listens on closes the local end without a
    byte — what asyncssh's `SSHLocalForwarder` does on `ChannelOpenError`."""

    def __init__(self):
        self.forwards: list[tuple[str, int]] = []  # every (dest_host, dest_port) asked for
        self.open_listeners = 0

    async def forward_local_port(self, listen_host, listen_port, dest_host, dest_port):
        self.forwards.append((dest_host, dest_port))

        async def pipe(src: asyncio.StreamReader, dst: asyncio.StreamWriter):
            try:
                while chunk := await src.read(65536):
                    dst.write(chunk)
                    await dst.drain()
            finally:
                dst.close()

        async def accepted(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
            try:
                up_reader, up_writer = await asyncio.open_connection(dest_host, dest_port)
            except OSError:
                writer.close()
                return
            source_port = up_writer.get_extra_info("sockname")[1]
            _TUNNELLED_SOURCE_PORTS.add(source_port)
            try:
                await asyncio.gather(
                    pipe(reader, up_writer), pipe(up_reader, writer), return_exceptions=True
                )
            finally:
                _TUNNELLED_SOURCE_PORTS.discard(source_port)

        server = await asyncio.start_server(accepted, listen_host, listen_port)
        self.open_listeners += 1
        return _FakeListener(self, server)


class FakeExecutor:
    """The executor's `/verify` as a fake: the loopback-peer, signature and replay checks as the
    real route, then a stub GPU whose scripts take `step_sleep` seconds each and run side by side
    when asked. `/version` names this server's own port as `local_verify_port` (the real executor
    names its INTERNAL_PORT), so a tunnel that targets it lands here."""

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
        self.refused_peers: list[str] = []  # `/verify` requests that did not come through a tunnel
        self.answer_override = None  # callable(intent) -> dict | (status, body)
        self.version_override = None  # dict served by /version instead of the default
        self.app = web.Application()
        self.app.router.add_get("/version", self.version)
        self.app.router.add_post("/verify", self.verify)
        self.server = TestServer(self.app)

    async def __aenter__(self):
        # A tunnel handler left pending when an earlier test's loop closed never ran its
        # `finally`; start every executor with no admitted peers, so a reused ephemeral port
        # cannot admit a direct POST.
        _TUNNELLED_SOURCE_PORTS.clear()
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
        version = {"version": "4.1.0", "capabilities": [CAPABILITY] if self.advertise else []}
        if self.advertise:
            version["local_verify_port"] = self.server.port
        return web.json_response(version)

    async def verify(self, request):
        peer = request.transport.get_extra_info("peername")
        if peer is None or peer[1] not in _TUNNELLED_SOURCE_PORTS:
            self.refused_peers.append(f"{peer[0]}:{peer[1]}" if peer else "?")
            return web.json_response(
                {"detail": "/verify is served on the loopback only (the validator's SSH tunnel)"},
                status=403,
            )
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
    ssh=None,
):
    return make_context(
        executor=executor_info,
        ssh=ssh or FakeSSH(),
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
        miner_hotkey=MINER_HOTKEY,
        matmul=MatmulStep(dim_n=1, dim_k=2, seed=3, cipher_text="c"),
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
    # the miner the intent is for is under the signature too: the executor refuses an intent
    # naming another miner (401), and a relay cannot re-address one it captured
    assert signed["miner_hotkey"] == MINER_HOTKEY
    readdressed = {**signed, "miner_hotkey": "5AnotherMiner"}
    assert not keypair.verify(canonical_intent_message(readdressed), readdressed["signature"])
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


# --- the check and its consumers ------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "override,reason",
    [
        (lambda raw: (403, {"detail": "not a loopback peer"}), "refused"),
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


# --- equivalence: one judge for both transports ------------------------------------------------


# --- the e2e-stack measurement: SSH sequence vs one call on the stub ------------------------------


