"""The liumd shadow comparison is logged, never scored, and runs nothing
with VALIDATOR_LIUMD_SHADOW off."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import bittensor
import pytest
import services.verifyx_validation_service as vvs
from datura.requests.miner_requests import ExecutorSSHInfo
from payload_models.payloads import MinerJobRequestPayload
from protocol.vc_protocol.compute_requests import NetworkEMA, RentedExecutorsResponse
from protocol.vc_protocol.validator_requests import ValidationEvent
from services.attestation_service import HostPolicyResult
from services.liumd_exec_client import LiumdRefusal
from services.local_verify_client import SCHEMA, LocalVerifyUnavailable, parse_answer
from services.task import liumd_shadow
from services.task import service as task_service_module
from services.task.checks import verifyx as verifyx_check
from services.task.checks.verifyx import VerifyXCheck
from services.task.liumd_shadow import (
    LIUMD_SHADOW_EVENT,
    TodayStep,
    run_liumd_shadow,
    today_verdicts,
)
from services.task.models import JobResult
from services.task.service import TaskService

from core.config import settings
from tests.helpers import build_context_config, build_services, build_state, make_context
from tests.test_local_verify import _FakeVerifyXValidator, matmul_service, matmul_stdout

SPECS = {"gpu": {"count": 1, "details": [{"uuid": "GPU-1", "name": "H100", "capacity": 81559}]}}
UUID = "exec-shadow"
EXECUTOR_INFO = ExecutorSSHInfo(
    uuid=UUID,
    address="10.0.0.5",
    port=8001,
    ssh_username="root",
    ssh_port=2200,
    python_path="/usr/bin/python",
    root_dir="/root/app",
)
RESERVED = f"5Miner/{UUID}/10.0.0.5/2200"
CAPABILITY = "gpu.validate.capability"
VERIFYX = "gpu.validate.verifyx"


def _event(check_id: str, reason: str, ms: int = 1500) -> ValidationEvent:
    return ValidationEvent(
        event="e",
        reason_code=reason,
        severity="info",
        impact="",
        check_id=check_id,
        when=datetime.now(UTC),
        context={"execution_time_ms": ms},
    )


PASSED = [_event(VERIFYX, "VERIFYX_OK", 80_000), _event(CAPABILITY, "GPU_VERIFY_OK", 26_000)]


@pytest.fixture(autouse=True)
def reserved_executor(monkeypatch):
    """The test executor is a reserved dev executor, the only kind the shadow runs on."""
    monkeypatch.setattr(
        settings, "LIUMD_SHADOW_RESERVED_EXECUTORS", f"5Miner/other-dev/10.0.0.6/22, {RESERVED}"
    )


@pytest.fixture
def keypair():
    return bittensor.Keypair.create_from_uri("//LiumdShadowTest")


JUDGED_VERIFYX = {"success": True, "ram": {}, "network": {"download_speed": 900.0}}


@pytest.fixture
def verifyx_service(monkeypatch):
    """The real service around a fake validator; `.judged` is what the checks return and
    `.record_tally` what each judgement was asked to do with the network-gate tally."""
    monkeypatch.setattr(vvs, "VerifyXValidator", _FakeVerifyXValidator)
    monkeypatch.setattr(vvs, "sha256_from_path", lambda _p: "lib-sha")
    service = vvs.VerifyXValidationService()
    service.judged = JUDGED_VERIFYX
    service.record_tally = []

    def checks(payload, *, record_tally=True):
        service.record_tally.append(record_tally)
        return service.judged

    monkeypatch.setattr(vvs, "_perform_verification_checks", checks)
    return service


class FakeLiumd:
    """`LiumdExecClient` answering as the binary does, per step as told."""

    def __init__(
        self, *, verifyx_ok=True, lib_sha="lib-sha", result=None, matmul=None, deadline_hit=False
    ):
        self.matmul = matmul  # a dict replacing the matmul step's reply
        self.deadline_hit = deadline_hit  # the document-wide flag: any step reached the deadline
        self.verifyx_ok = verifyx_ok
        self.lib_sha = lib_sha
        self.result = result
        self.intents: list[dict] = []
        self.timeouts: list[float] = []

    def factory(self, ctx, timeout_s):
        self.timeouts.append(timeout_s)
        return self

    async def run(self, ssh, intent):
        self.intents.append(intent)
        if callable(self.result):
            return await self.result(intent)
        if self.result is not None:
            if isinstance(self.result, BaseException):
                raise self.result
            return self.result
        steps = {}
        if intent["steps"]["matmul"]:
            steps["matmul"] = {
                "status": "ok",
                "ms": 20_000,
                "exit_status": 0,
                "stdout": matmul_stdout("GPU-1"),
                "stderr_tail": "",
            }
        if intent["steps"]["verifyx"]:
            cipher = intent["steps"]["verifyx"]["cipher_text"]
            steps["verifyx"] = {
                "status": "ok",
                "ms": 70_000,
                "exit_status": 0,
                "stdout": cipher + ("-ok" if self.verifyx_ok else "-no"),
                "data": {"lib_sha256": self.lib_sha},
            }
        if self.matmul is not None:
            steps["matmul"] = self.matmul
        for name in ("docker", "ports", "inspector"):
            steps[name] = {"status": "ok", "ms": 3}
        raw = {
            "schema": SCHEMA,
            "nonce": intent["nonce"],
            "executor_uuid": intent["executor_uuid"],
            "executor_version": "liumd/0.1.0",
            "elapsed_ms": 91_000,
            "deadline_hit": self.deadline_hit,
            "steps": steps,
        }
        return parse_answer(raw, intent=intent, round_trip_ms=92_000)


def _ctx(keypair, monkeypatch, verifyx_service, *, sealed="challenge", state=None, **config):
    return make_context(
        executor=EXECUTOR_INFO,
        miner_hotkey="5Miner",
        ssh=object(),
        services=build_services(
            validation=matmul_service(monkeypatch, sealed=sealed),
            verifyx=verifyx_service,
            redis=SimpleNamespace(renting_in_progress=AsyncMock(return_value=False)),
        ),
        config=build_context_config(
            validator_keypair=keypair,
            first_pass=config.get("first_pass", False),
        ),
        state=state or build_state(specs=SPECS),
    )


async def _shadow(ctx, fake, *, ok=True, events=PASSED, deadline_s=10_000.0):
    return await run_liumd_shadow(
        ctx,
        ok=ok,
        events=events,
        deadline_monotonic=time.monotonic() + deadline_s,
        client_factory=fake.factory,
    )


# --- today's verdicts -----------------------------------------------------------------------


def test_todays_verdicts_come_from_the_runs_events():
    assert today_verdicts(True, PASSED) == {
        "matmul": TodayStep("pass", "GPU_VERIFY_OK", 26_000),
        "verifyx": TodayStep("pass", "VERIFYX_OK", 80_000),
    }
    failed = [_event(VERIFYX, "VERIFYX_OK"), _event(CAPABILITY, "GPU_VERIFY_TIMEOUT")]
    assert today_verdicts(False, failed)["matmul"].verdict == "fail"
    assert today_verdicts(False, failed)["verifyx"].verdict == "pass"
    stopped_at_verifyx = [_event(VERIFYX, "VERIFYX_NETWORK_TOO_SLOW")]
    assert today_verdicts(False, stopped_at_verifyx) == {
        "matmul": TodayStep("not_reached"),
        "verifyx": TodayStep("fail", "VERIFYX_NETWORK_TOO_SLOW", 1500),
    }
    filler = [
        _event(VERIFYX, "VERIFYX_SKIPPED_ACTIVE_FILLER"),
        _event(CAPABILITY, "GPU_VERIFY_SKIPPED_ACTIVE_FILLER"),
    ]
    assert {s.verdict for s in today_verdicts(True, filler).values()} == {"skipped"}
    assert today_verdicts(True, [])["matmul"] == TodayStep("not_reached")


# --- the comparison -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agreeing_verdicts_are_logged_as_one_structured_line(
    keypair, monkeypatch, verifyx_service, caplog
):
    fake = FakeLiumd()
    ctx = _ctx(keypair, monkeypatch, verifyx_service)

    with caplog.at_level(logging.INFO, logger=liumd_shadow.__name__):
        record = await _shadow(ctx, fake)

    assert record["outcome"] == "compared" and record["agree"] is True
    assert record["steps"]["matmul"] == {
        "today": "pass",
        "today_reason_code": "GPU_VERIFY_OK",
        "today_ms": 26_000,
        "liumd_status": "ok",
        "liumd_verdict": "pass",
        "liumd_reason": "ok",
        "liumd_ms": 20_000,
        "agree": True,
    }
    assert record["steps"]["verifyx"]["agree"] is True
    # The shadow's judgement never counts in the live network-gate tally.
    assert verifyx_service.record_tally == [False]
    assert record["steps"]["docker"] == {"liumd_status": "ok", "liumd_ms": 3}
    assert (record["round_trip_ms"], record["executor_elapsed_ms"]) == (92_000, 91_000)
    [line] = [r for r in caplog.records if r.getMessage() == LIUMD_SHADOW_EVENT]
    assert line.msg.extra["transport"] == "liumd_shadow"
    assert line.msg.extra["agree"] is True and line.msg.extra["steps"] == record["steps"]


@pytest.mark.asyncio
async def test_the_intent_mirrors_todays_steps_and_sizing(keypair, monkeypatch, verifyx_service):
    fake = FakeLiumd()
    ctx = _ctx(keypair, monkeypatch, verifyx_service, first_pass=True)

    await _shadow(ctx, fake, deadline_s=200.0)

    [intent] = fake.intents
    assert intent["miner_hotkey"] == "5Miner" and intent["executor_uuid"] == UUID
    assert intent["parallel_gpu"] is True
    assert intent["steps"]["matmul"] and intent["steps"]["verifyx"]
    # The budget is what the task has left, and the host is told to stop 30 s before it.
    [timeout] = fake.timeouts
    assert 199.0 < timeout <= 200.0
    assert intent["deadline_s"] == int(timeout) - 30


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sealed,fake_args,ok,events,matmul_agree,verifyx_agree,agree",
    [
        ("GPU-SPOOFED", dict(verifyx_ok=False), True, PASSED, False, False, False),
        (
            "challenge",
            {},
            False,
            [_event(VERIFYX, "VERIFYX_OK"), _event(CAPABILITY, "GPU_VERIFY_FAILED")],
            False,
            True,
            False,
        ),
        ("challenge", dict(matmul={"status": "skipped", "ms": 0}), True, PASSED, None, True, None),
        (
            "challenge",
            dict(
                matmul={
                    "status": "failed",
                    "ms": 5,
                    "exit_status": 1,
                    "stdout": "",
                    "stderr_tail": "",
                }
            ),
            True,
            PASSED,
            False,
            True,
            False,
        ),
        # Matmul timed out at its own 120 s cap while another step reached the suite deadline:
        # the agent keeps the step's time, so it is a failure, not a deadline cut.
        (
            "challenge",
            dict(
                matmul={"status": "timeout", "ms": 120_005, "error": "timed out after 120s"},
                deadline_hit=True,
            ),
            True,
            PASSED,
            False,
            True,
            False,
        ),
        # Cut at the suite deadline: the agent resets the step's time to 0; it never ran to a verdict.
        (
            "challenge",
            dict(
                matmul={"status": "timeout", "ms": 0, "error": "suite deadline reached"},
                deadline_hit=True,
            ),
            True,
            PASSED,
            None,
            True,
            None,
        ),
    ],
    ids=[
        "liumd-fails",
        "today-fails",
        "step-skipped",
        "step-failed",
        "own-cap-timeout",
        "deadline-cut",
    ],
)
async def test_agreement_per_step_and_overall(
    keypair,
    monkeypatch,
    verifyx_service,
    sealed,
    fake_args,
    ok,
    events,
    matmul_agree,
    verifyx_agree,
    agree,
):
    ctx = _ctx(keypair, monkeypatch, verifyx_service, sealed=sealed)

    record = await _shadow(ctx, FakeLiumd(**fake_args), ok=ok, events=events)

    assert record["steps"]["matmul"].get("agree") is matmul_agree
    assert record["steps"]["verifyx"]["agree"] is verifyx_agree
    assert record["agree"] is agree


def test_only_a_step_cut_at_the_suite_deadline_is_not_run():
    own_cap = SimpleNamespace(status="timeout", ms=120005, error="timed out after 120s")
    cut = SimpleNamespace(status="timeout", ms=0, error="suite deadline reached")
    hit, not_hit = SimpleNamespace(deadline_hit=True), SimpleNamespace(deadline_hit=False)
    assert liumd_shadow._not_run_or_fail(own_cap, hit) == "fail"
    assert liumd_shadow._not_run_or_fail(own_cap, not_hit) == "fail"
    assert liumd_shadow._not_run_or_fail(cut, hit) == "not_run"
    assert liumd_shadow._not_run_or_fail(cut, not_hit) == "fail"


class _TodaysVerifyX:
    """Today's SSH transport, answering with the judged result the shadow gets too, then with
    each of `then` in turn (a cold-sample retry's samples)."""

    def __init__(self, judged, then=()):
        self.answers = [judged, *then]

    async def validate_verifyx_and_process_job(self, **_):
        data = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        return SimpleNamespace(data=data, error=None, diagnostics=None)


def _network_ema(download):
    return RentedExecutorsResponse(
        executors={}, network_ema={UUID: NetworkEMA(ema_verifyx_download_speed=download)}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prev_ema,network,first_pass,today,retry,prior_fallbacks",
    [
        # The reviewer's probe: raw success, but the EMA (100 → 75) falls under the floor.
        (100.0, {"download_speed": 50.0}, False, "fail", None, 0),
        (100.0, {"download_speed": 50.0}, True, "fail", None, 0),
        (None, {"download_speed": 50.0}, False, "fail", None, 0),
        # A first pass on a never-measured host defers the floor to the first scored cycle.
        (None, {"download_speed": 50.0}, True, "pass", None, 0),
        (300.0, {"download_speed": 900.0}, False, "pass", None, 0),
        # A Cloudflare fallback with no reading keeps the stored EMA, under and over the floor.
        (80.0, {"download_speed": None, "cloudflare_fallback": True}, False, "fail", None, 0),
        (150.0, {"download_speed": None, "cloudflare_fallback": True}, False, "pass", None, 0),
        # Nothing stored and nothing read: the EMA stays unseeded and the check passes.
        (None, {"download_speed": None, "cloudflare_fallback": True}, False, "pass", None, 0),
        # Cold retries on, never measured: today re-measures 50 then 150 Mbps and passes, where
        # the shadow's one 50 Mbps sample would fail. Not compared.
        (None, {"download_speed": 50.0}, False, "pass", {"download_speed": 150.0}, 0),
        # Cold retries on, but a stored EMA: no retry today, so compared as ever.
        (100.0, {"download_speed": 50.0}, False, "fail", {"download_speed": 150.0}, 0),
        # Past the package-fallback cap today's check scores the package reading as zero (EMA 150 →
        # 75, fail) while the shadow would read 1,000 Mbps and pass. Not compared.
        (
            150.0,
            {
                "download_speed": 1000.0,
                "capacity_download_speed": None,
                "cloudflare_fallback": True,
            },
            False,
            "fail",
            None,
            8,
        ),
    ],
    ids=[
        "ema-drops-under-floor",
        "first-pass-known-host",
        "never-measured-scored",
        "first-pass-deferred",
        "fast",
        "fallback-keeps-slow-ema",
        "fallback-keeps-fast-ema",
        "unseeded",
        "cold-retry-not-compared",
        "cold-retry-on-known-host",
        "package-fallback-cap-not-compared",
    ],
)
async def test_verifyx_verdict_is_the_checks_own_including_the_ema_gate(
    keypair,
    monkeypatch,
    verifyx_service,
    prev_ema,
    network,
    first_pass,
    today,
    retry,
    prior_fallbacks,
):
    monkeypatch.setattr(settings, "VERIFYX_COLD_SAMPLE_RETRY_ENABLED", retry is not None)
    verifyx_check._package_fallback_probes.clear()
    if prior_fallbacks:
        verifyx_check._package_fallback_probes[("5Miner", UUID)] = prior_fallbacks
    judged = {"success": True, "ram": {}, "network": network}
    retried = [{"success": True, "ram": {}, "network": retry}] if retry else []
    verifyx_service.judged = judged
    rented = _network_ema(prev_ema)
    ctx = _ctx(
        keypair,
        monkeypatch,
        verifyx_service,
        first_pass=first_pass,
        state=build_state(specs=SPECS, rented_data=rented),
    )
    # Today's verdict is the real check's, on the same judged result and the same stored EMA.
    todays_ctx = _ctx(
        keypair,
        monkeypatch,
        _TodaysVerifyX(judged, retried),
        first_pass=first_pass,
        state=build_state(specs=SPECS, rented_data=rented),
    )
    stored_before = rented.network_ema[UUID].model_dump()
    checked = await VerifyXCheck().run(todays_ctx)
    assert ("pass" if checked.passed else "fail") == today
    events = (
        [checked.event, _event(CAPABILITY, "GPU_VERIFY_OK")] if checked.passed else [checked.event]
    )
    tally_before = vvs.NETWORK_GATE_TALLY.probe_fallback

    fake = FakeLiumd()
    record = await _shadow(ctx, fake, ok=checked.passed, events=events)

    entry = record["steps"]["verifyx"]
    assert entry["today"] == today
    if retry and prev_ema is None:
        assert checked.passed is True
        assert entry.get("agree") is None
        assert entry["liumd_reason"] == "cold_sample_retry"
        assert "liumd_verdict" not in entry
        assert fake.intents[0]["steps"]["verifyx"] is None
    elif prior_fallbacks:
        assert entry.get("agree") is None
        assert entry["liumd_reason"] == "package_fallback_cap"
    else:
        assert entry["liumd_verdict"] == today
        assert entry["agree"] is True
        if today == "fail":
            assert entry["liumd_reason"] == "network_too_slow"
    # The stored EMA and the live tally are read, never written.
    assert rented.network_ema[UUID].model_dump() == stored_before
    assert vvs.NETWORK_GATE_TALLY.probe_fallback == tally_before


@pytest.mark.asyncio
async def test_an_outdated_verifyx_library_disagrees(keypair, monkeypatch, verifyx_service):
    fake = FakeLiumd(lib_sha="old")

    record = await _shadow(_ctx(keypair, monkeypatch, verifyx_service), fake, events=PASSED)

    expected = {"liumd_reason": "lib_mismatch", "agree": False}
    assert expected.items() <= record["steps"]["verifyx"].items()


@pytest.mark.asyncio
async def test_all_cards_mode_does_not_compare_the_matmul(keypair, monkeypatch, verifyx_service):
    monkeypatch.setattr(settings, "MATMUL_ALLCARDS_CHECK_ENABLED", True)
    fake = FakeLiumd()

    record = await _shadow(_ctx(keypair, monkeypatch, verifyx_service), fake)

    assert fake.intents[0]["steps"]["matmul"] is None
    assert record["steps"]["matmul"]["liumd_reason"] == "allcards_ssh"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "events,ok,reason,extra",
    [
        (
            [
                _event(VERIFYX, "VERIFYX_SKIPPED_ACTIVE_FILLER"),
                _event(CAPABILITY, "GPU_VERIFY_SKIPPED_ACTIVE_FILLER"),
            ],
            True,
            "workload",
            {},
        ),
        (
            [_event(VERIFYX, "VERIFYX_OK"), _event(CAPABILITY, "GPU_VERIFY_SKIPPED_RENTED")],
            True,
            "workload",
            {},
        ),
        ([], True, "no_gpu_step_ran", {}),
        ([_event("gpu.count", "GPU_COUNT_MISMATCH")], False, "no_gpu_step_ran", {}),
        (PASSED, True, "workload", {"rented": True}),
        (PASSED, True, "no_budget", {"deadline_s": 59.0}),
        # Only reserved dev executors run the shadow: a rentable executor never does, idle or not.
        (PASSED, True, "not_reserved", {"reserved": ""}),
        (PASSED, True, "not_reserved", {"reserved": "5Miner/other-dev/10.0.0.5/2200"}),
        (
            PASSED,
            True,
            "not_reserved",
            {"reserved": "5Miner/other-dev/10.0.0.5/2200", "rented": True},
        ),
        # UUIDs are miner-chosen: the reserved UUID under another miner or endpoint is not it.
        (PASSED, True, "not_reserved", {"reserved": f"5OtherMiner/{UUID}/10.0.0.5/2200"}),
        (PASSED, True, "not_reserved", {"reserved": f"5Miner/{UUID}/10.0.0.9/2200"}),
        (PASSED, True, "not_reserved", {"reserved": f"5Miner/{UUID}/10.0.0.5/2201"}),
    ],
    ids=[
        "filler",
        "rented-by-event",
        "no-events",
        "stopped-before-gpu",
        "rented-by-snapshot",
        "no-budget",
        "allowlist-empty",
        "rentable-idle",
        "rentable-rented",
        "same-uuid-other-miner",
        "same-uuid-other-address",
        "same-uuid-other-ssh-port",
    ],
)
async def test_no_gpu_work_where_today_ran_none(
    keypair, monkeypatch, verifyx_service, events, ok, reason, extra
):
    fake = FakeLiumd()
    extra = dict(extra)
    if "reserved" in extra:
        monkeypatch.setattr(settings, "LIUMD_SHADOW_RESERVED_EXECUTORS", extra.pop("reserved"))
    state = None
    if extra.pop("rented", False):
        pod = {"pod_id": "p1", "container_name": "c1"}
        node = {
            "miner_hotkey": "5Miner",
            "executor_ip_address": "10.0.0.5",
            "executor_ip_port": "8001",
            "pods": [pod],
        }
        rented = RentedExecutorsResponse.model_validate({"executors": {UUID: node}})
        state = build_state(specs=SPECS, rented_data=rented)
    ctx = _ctx(keypair, monkeypatch, verifyx_service, state=state)

    record = await _shadow(ctx, fake, ok=ok, events=events, **extra)

    assert (record["outcome"], record["reason"]) == ("skipped", reason)
    assert fake.intents == []


@pytest.mark.parametrize("entry", [UUID, f"5Miner/{UUID}/10.0.0.5", f"5Miner/{UUID}/10.0.0.5/ssh"])
@pytest.mark.asyncio
async def test_a_malformed_reserved_entry_runs_nothing(
    keypair, monkeypatch, verifyx_service, entry
):
    monkeypatch.setattr(settings, "LIUMD_SHADOW_RESERVED_EXECUTORS", f"{RESERVED},{entry}")
    fake = FakeLiumd()

    record = await _shadow(_ctx(keypair, monkeypatch, verifyx_service), fake)

    assert (record["outcome"], record["reason"]) == ("error", "internal_error")
    assert "LIUMD_SHADOW_RESERVED_EXECUTORS" in record["detail"]
    assert fake.intents == []


@pytest.mark.asyncio
async def test_a_host_that_outlives_the_task_budget_is_cut_off(
    keypair, monkeypatch, verifyx_service
):
    monkeypatch.setattr(liumd_shadow, "MIN_BUDGET_SECONDS", 0)

    async def never(intent):
        await asyncio.Event().wait()

    started = time.monotonic()
    record = await asyncio.wait_for(
        _shadow(
            _ctx(keypair, monkeypatch, verifyx_service), FakeLiumd(result=never), deadline_s=0.2
        ),
        5,
    )

    assert (record["outcome"], record["reason"]) == ("unavailable", "timeout")
    assert time.monotonic() - started < 2


@pytest.mark.asyncio
async def test_refusals_and_unavailability_are_logged_outcomes(
    keypair, monkeypatch, verifyx_service
):
    refused = LiumdRefusal(
        exit_status=5, error="busy", detail="another verify", echoed=True, round_trip_ms=4
    )
    ctx = _ctx(keypair, monkeypatch, verifyx_service)
    record = await _shadow(ctx, FakeLiumd(result=refused))
    assert (record["outcome"], record["reason"], record["exit_status"]) == ("refused", "busy", 5)
    # The native matmul generator is freed on every way out, not only after a judged answer.
    ctx.services.validation.wrapper.free.assert_called_once_with("ptr")

    gone = LocalVerifyUnavailable("not_supported", "exit 127: no liumd on this host")
    ctx = _ctx(keypair, monkeypatch, verifyx_service)
    record = await _shadow(ctx, FakeLiumd(result=gone))
    assert (record["outcome"], record["reason"]) == ("unavailable", "not_supported")
    ctx.services.validation.wrapper.free.assert_called_once_with("ptr")


# --- scoring: identical with the flag on and off ----------------------------------------------


class _Shell:
    ssh_client = object()

    def __init__(self, **_):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _result() -> JobResult:
    return JobResult(
        executor_info=EXECUTOR_INFO,
        score=3.25,
        job_score=1.5,
        job_batch_id="batch-1",
        log_status="info",
        log_text="ok",
        gpu_model="H100",
        gpu_count=1,
    )


async def _cycle(pipeline_ctx, *, ok=True, events=PASSED, first_pass=False) -> JobResult:
    service = TaskService.__new__(TaskService)
    service.ssh_service = MagicMock(decrypt_payload=MagicMock(return_value="key"))
    service.attestation_service = MagicMock(
        prepare_host_policy=AsyncMock(return_value=HostPolicyResult())
    )
    service.redis_service = MagicMock()
    service.pipeline_factory = MagicMock()
    service.pipeline_factory.build_context = AsyncMock(return_value=pipeline_ctx)
    service.pipeline_factory.build_pipeline.return_value = MagicMock(
        run=AsyncMock(return_value=(ok, events, pipeline_ctx))
    )
    handler = MagicMock()
    handler.return_value.handle_result = AsyncMock(side_effect=lambda **_: _result())
    with (
        patch.object(task_service_module, "InteractiveShellService", _Shell),
        patch.object(task_service_module, "ResultHandler", handler),
    ):
        return await service.create_task(
            miner_info=MinerJobRequestPayload(
                job_batch_id="batch-1",
                miner_hotkey="5Miner",
                miner_coldkey="5Cold",
                miner_address="10.0.0.5",
                miner_port=8080,
                executors=[],
            ),
            executor_info=pipeline_ctx.executor,
            keypair=MagicMock(ss58_address="5Val"),
            private_key="key",
            public_key="pub",
            encrypted_files=MagicMock(),
            rented_data=MagicMock(),
            default_docker_image_digests={},
            first_pass=first_pass,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "liumd",
    [
        pytest.param(dict(), id="liumd-agrees"),
        pytest.param(dict(verifyx_ok=False, lib_sha="old"), id="liumd-disagrees"),
        pytest.param(dict(result=RuntimeError("boom")), id="shadow-crashes"),
    ],
)
async def test_the_result_is_identical_with_the_shadow_on_and_off(
    keypair, monkeypatch, verifyx_service, liumd
):
    fake = FakeLiumd(**liumd)
    monkeypatch.setattr(liumd_shadow, "_default_client", fake.factory)
    ctx = _ctx(keypair, monkeypatch, verifyx_service)

    monkeypatch.setattr(settings, "VALIDATOR_LIUMD_SHADOW", False)
    off = await _cycle(ctx)
    assert fake.intents == []

    monkeypatch.setattr(settings, "VALIDATOR_LIUMD_SHADOW", True)
    on = await _cycle(ctx)
    assert len(fake.intents) == 1

    assert on.model_dump() == off.model_dump()
    assert (on.score, on.job_score) == (3.25, 1.5)


@pytest.mark.asyncio
async def test_flag_off_or_dry_run_never_calls_the_shadow(keypair, monkeypatch, verifyx_service):
    shadow = AsyncMock()
    monkeypatch.setattr(task_service_module, "run_liumd_shadow", shadow)
    ctx = _ctx(keypair, monkeypatch, verifyx_service)

    monkeypatch.setattr(settings, "VALIDATOR_LIUMD_SHADOW", False)
    await _cycle(ctx)
    monkeypatch.setattr(settings, "VALIDATOR_LIUMD_SHADOW", True)
    monkeypatch.setattr(settings, "DRY_RUN", True)
    await _cycle(ctx)
    assert shadow.await_count == 0

    monkeypatch.setattr(settings, "DRY_RUN", False)
    # An express-lane first pass publishes only after create_task returns: no shadow there.
    await _cycle(ctx, first_pass=True)
    assert shadow.await_count == 0

    await _cycle(ctx, ok=False, events=PASSED[:1])
    [call] = shadow.await_args_list
    assert call.args == (ctx,)
    assert call.kwargs["ok"] is False and call.kwargs["events"] == PASSED[:1]
