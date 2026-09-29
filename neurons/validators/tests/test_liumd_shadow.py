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
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse
from protocol.vc_protocol.validator_requests import ValidationEvent
from services.attestation_service import HostPolicyResult
from services.liumd_exec_client import LiumdRefusal
from services.local_verify_client import SCHEMA, LocalVerifyUnavailable, parse_answer
from services.task import liumd_shadow
from services.task import service as task_service_module
from services.task.liumd_shadow import (
    LIUMD_SHADOW_EVENT,
    TodayStep,
    run_liumd_shadow,
    shadow_deadline,
    today_verdicts,
)
from services.task.models import JobResult
from services.task.service import TaskService

from core.config import settings
from tests.helpers import build_context_config, build_services, build_state, make_context
from tests.test_local_verify import _FakeVerifyXValidator, matmul_service, matmul_stdout

SPECS = {"gpu": {"count": 1, "details": [{"uuid": "GPU-1", "name": "H100", "capacity": 81559}]}}
UUID = "exec-shadow"
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


@pytest.fixture
def keypair():
    return bittensor.Keypair.create_from_uri("//LiumdShadowTest")


@pytest.fixture
def verifyx_service(monkeypatch):
    monkeypatch.setattr(vvs, "VerifyXValidator", _FakeVerifyXValidator)
    monkeypatch.setattr(vvs, "sha256_from_path", lambda _p: "lib-sha")
    monkeypatch.setattr(
        vvs,
        "_perform_verification_checks",
        lambda payload: {"success": True, "ram": {}, "network": {"download_speed": 900.0}},
    )
    return vvs.VerifyXValidationService()


class FakeLiumd:
    """`LiumdExecClient` answering as the binary does, per step as told."""

    def __init__(self, *, matmul_uuid="challenge", verifyx_ok=True, lib_sha="lib-sha", result=None):
        self.matmul_uuid = matmul_uuid
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
        for name in ("docker", "ports", "inspector"):
            steps[name] = {"status": "ok", "ms": 3}
        raw = {
            "schema": SCHEMA,
            "nonce": intent["nonce"],
            "executor_uuid": intent["executor_uuid"],
            "executor_version": "liumd/0.1.0",
            "elapsed_ms": 91_000,
            "deadline_hit": False,
            "steps": steps,
        }
        return parse_answer(raw, intent=intent, round_trip_ms=92_000)


def _ctx(keypair, monkeypatch, verifyx_service, *, sealed="challenge", state=None, **config):
    return make_context(
        executor=ExecutorSSHInfo(
            uuid=UUID,
            address="10.0.0.5",
            port=8001,
            ssh_username="root",
            ssh_port=2200,
            python_path="/usr/bin/python",
            root_dir="/root/app",
        ),
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
            verifyx_enabled=config.get("verifyx_enabled", True),
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


def test_the_deadline_leaves_the_task_its_timeout_and_tail():
    assert shadow_deadline(1000.0) == 1000.0 + settings.JOB_TIME_OUT - 120 - 60


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
async def test_liumd_failing_where_today_passed_is_a_disagreement(
    keypair, monkeypatch, verifyx_service
):
    ctx = _ctx(keypair, monkeypatch, verifyx_service, sealed="GPU-SPOOFED")

    record = await _shadow(ctx, FakeLiumd(verifyx_ok=False))

    assert record["agree"] is False
    assert record["steps"]["matmul"]["liumd_verdict"] == "fail"
    assert record["steps"]["matmul"]["liumd_reason"] == "local_failed"
    assert record["steps"]["verifyx"]["agree"] is False


@pytest.mark.asyncio
async def test_today_failing_where_liumd_passed_is_a_disagreement(
    keypair, monkeypatch, verifyx_service
):
    events = [_event(VERIFYX, "VERIFYX_OK"), _event(CAPABILITY, "GPU_VERIFY_FAILED")]

    record = await _shadow(
        _ctx(keypair, monkeypatch, verifyx_service), FakeLiumd(), ok=False, events=events
    )

    assert record["steps"]["matmul"]["today"] == "fail"
    assert record["steps"]["matmul"]["agree"] is False
    assert record["steps"]["verifyx"]["agree"] is True
    assert record["agree"] is False


@pytest.mark.asyncio
async def test_an_outdated_verifyx_library_is_a_liumd_failure(
    keypair, monkeypatch, verifyx_service
):
    record = await _shadow(_ctx(keypair, monkeypatch, verifyx_service), FakeLiumd(lib_sha="old"))

    assert record["steps"]["verifyx"]["liumd_reason"] == "lib_mismatch"
    assert record["steps"]["verifyx"]["agree"] is False


@pytest.mark.asyncio
async def test_only_the_steps_today_ran_are_asked_for(keypair, monkeypatch, verifyx_service):
    fake = FakeLiumd()
    events = [_event(VERIFYX, "VERIFYX_DISABLED"), _event(CAPABILITY, "GPU_VERIFY_OK")]

    record = await _shadow(
        _ctx(keypair, monkeypatch, verifyx_service, verifyx_enabled=False), fake, events=events
    )

    [intent] = fake.intents
    assert intent["steps"]["verifyx"] is None and intent["steps"]["matmul"]
    assert record["steps"]["verifyx"] == {
        "today": "skipped",
        "today_reason_code": "VERIFYX_DISABLED",
        "today_ms": 1500,
    }
    assert record["agree"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "events,ok,reason",
    [
        (
            [
                _event(VERIFYX, "VERIFYX_SKIPPED_ACTIVE_FILLER"),
                _event(CAPABILITY, "GPU_VERIFY_SKIPPED_ACTIVE_FILLER"),
            ],
            True,
            "workload",
        ),
        (
            [_event(VERIFYX, "VERIFYX_OK"), _event(CAPABILITY, "GPU_VERIFY_SKIPPED_RENTED")],
            True,
            "workload",
        ),
        ([], True, "no_gpu_step_ran"),
        ([_event("gpu.count", "GPU_COUNT_MISMATCH")], False, "no_gpu_step_ran"),
    ],
)
async def test_no_gpu_work_where_today_ran_none(
    keypair, monkeypatch, verifyx_service, events, ok, reason
):
    fake = FakeLiumd()

    record = await _shadow(_ctx(keypair, monkeypatch, verifyx_service), fake, ok=ok, events=events)

    assert (record["outcome"], record["reason"]) == ("skipped", reason)
    assert fake.intents == []


@pytest.mark.asyncio
async def test_a_node_the_snapshot_shows_rented_is_skipped(keypair, monkeypatch, verifyx_service):
    rented = RentedExecutorsResponse.model_validate(
        {
            "executors": {
                UUID: {
                    "miner_hotkey": "5Miner",
                    "executor_ip_address": "10.0.0.5",
                    "executor_ip_port": "8001",
                    "pods": [{"pod_id": "p1", "container_name": "c1"}],
                }
            }
        }
    )
    fake = FakeLiumd()
    ctx = _ctx(
        keypair, monkeypatch, verifyx_service, state=build_state(specs=SPECS, rented_data=rented)
    )

    record = await _shadow(ctx, fake)

    assert record["reason"] == "workload" and fake.intents == []


@pytest.mark.asyncio
async def test_too_little_task_time_left_runs_nothing(keypair, monkeypatch, verifyx_service):
    fake = FakeLiumd()

    record = await _shadow(_ctx(keypair, monkeypatch, verifyx_service), fake, deadline_s=59.0)

    assert (record["outcome"], record["reason"]) == ("skipped", "no_budget")
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


@pytest.mark.asyncio
async def test_a_bug_in_the_shadow_never_raises(keypair, monkeypatch, verifyx_service):
    record = await _shadow(
        _ctx(keypair, monkeypatch, verifyx_service), FakeLiumd(result=RuntimeError("boom"))
    )

    assert (record["outcome"], record["reason"]) == ("error", "internal_error")
    assert "boom" in record["detail"]


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
        executor_info=ExecutorSSHInfo(
            uuid=UUID,
            address="10.0.0.5",
            port=8001,
            ssh_username="root",
            ssh_port=2200,
            python_path="/usr/bin/python",
            root_dir="/root/app",
        ),
        score=3.25,
        job_score=1.5,
        job_batch_id="batch-1",
        log_status="info",
        log_text="ok",
        gpu_model="H100",
        gpu_count=1,
    )


async def _cycle(pipeline_ctx, *, ok=True, events=PASSED) -> JobResult:
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
    await _cycle(ctx, ok=False, events=PASSED[:1])
    [call] = shadow.await_args_list
    assert call.args == (ctx,)
    assert call.kwargs["ok"] is False and call.kwargs["events"] == PASSED[:1]
