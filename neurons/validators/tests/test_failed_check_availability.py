"""DAH-5395: a run ended by a failure in REASONS_THAT_HIDE_AT_ONCE hides the node until a cycle passes."""

from unittest.mock import AsyncMock, MagicMock, patch

import asyncssh
import pytest
from datura.requests.miner_requests import ExecutorSSHInfo
from payload_models.payloads import MinerJobRequestPayload
from services.task.availability import (
    REASONS_THAT_HIDE_AT_ONCE,
    AvailabilityErrorCode,
    ReachSource,
    ReachTarget,
    availability_errors,
    build_availability_event,
    failed_check_event,
    silence_availability_errors_on_our_own_outage,
)
from services.task.models import JobResult, build_msg
from services.task.service import TaskService


def _event(check_id: str, reason: str = "SOME_REASON", severity: str = "error"):
    return build_msg(event="x", reason=reason, severity=severity, impact="x", check_id=check_id)


def _errors_the_service_sends(ok: bool, events: list) -> list[dict]:
    # the same composition as TaskService.create_task
    failed = failed_check_event(ok, events)
    return [
        e.model_dump(mode="json")
        for e in availability_errors([*events, failed] if failed else events)
    ]


HIDES_AT_ONCE = [
    ("gpu.validate.model", "GPU_DETAILS_MISMATCH"),
    ("gpu.validate.fingerprint", "GPU_UUID_CHANGED"),
    ("gpu.validate.nvlink_fault", "NVLINK_FAULT"),
]


def test_the_allowlist_is_exactly_these_reasons() -> None:
    assert REASONS_THAT_HIDE_AT_ONCE == {reason for _, reason in HIDES_AT_ONCE}


@pytest.mark.parametrize(("check_id", "reason"), HIDES_AT_ONCE)
def test_an_allowlisted_failure_hides_the_node(check_id, reason) -> None:
    events = [_event("gpu.scrape.machine_spec", "OK", "info"), _event(check_id, reason)]

    errors = _errors_the_service_sends(False, events)

    assert [e["reason_code"] for e in errors] == [AvailabilityErrorCode.VERIFICATION_FAILED]
    assert errors[0]["reach_source"] == "validator"
    assert errors[0]["reach_target"] == "node"
    assert errors[0]["what_we_saw"] == {"failed_check": check_id, "failed_reason": reason}


@pytest.mark.parametrize(
    ("check_id", "reason"),
    [
        ("gpu.validate.verifyx", "VERIFYX_FAILED"),
        ("executor.validate.ports", "INSUFFICIENT_PORTS"),
        ("gpu.validate.usage", "GPU_USAGE_HIGH"),
        ("gpu.scrape.machine_spec", "SCRAPE_TIMEOUT"),
        ("executor.validate.sysbox", "SYSBOX_REQUIRED_MISSING"),
        ("executor.validate.storage", "STORAGE_LIMIT_REQUIRED"),
        ("gpu.validate.fingerprint", "GPU_MISSING"),
        ("executor.validate.duplicate", "SOME_REASON"),
    ],
)
def test_any_other_failure_no_longer_hides_the_node(check_id, reason) -> None:
    events = [_event("gpu.scrape.machine_spec", "OK", "info"), _event(check_id, reason)]

    assert failed_check_event(False, events) is None
    assert _errors_the_service_sends(False, events) == []


def test_an_allowlisted_reason_earlier_in_the_run_does_not_count() -> None:
    # only the event that ended the run decides
    events = [
        _event("gpu.validate.model", "GPU_DETAILS_MISMATCH", "info"),
        _event("gpu.validate.verifyx", "VERIFYX_FAILED"),
    ]

    assert _errors_the_service_sends(False, events) == []


def test_a_policy_score_zero_keeps_the_node_listed() -> None:
    # a run that completes with score 0 is ok=True
    events = [
        _event("gpu.validate.verifyx", "OK", "info"),
        _event("score", "SCORING_EXCLUDED", "warning"),
    ]

    assert failed_check_event(True, events) is None
    assert _errors_the_service_sends(True, events) == []


def test_a_check_that_already_reported_an_availability_error_is_not_doubled() -> None:
    lost = build_availability_event(
        code=AvailabilityErrorCode.GPU_LOST,
        reach_source=ReachSource.VALIDATOR,
        reach_target=ReachTarget.GPU,
        event_text="lost",
        impact="x",
        remediation="x",
        what_we_saw={},
    )

    errors = _errors_the_service_sends(False, [lost])

    assert [e["reason_code"] for e in errors] == ["GPU_LOST"]


def test_the_next_pass_clears_it() -> None:
    assert _errors_the_service_sends(
        False, [_event("gpu.validate.fingerprint", "GPU_UUID_CHANGED")]
    )
    # an empty list (not None) is what makes the backend clear the stored entry
    assert _errors_the_service_sends(True, [_event("score", "OK", "info")]) == []


def _result(errors):
    return JobResult(
        executor_info=ExecutorSSHInfo(
            uuid="node",
            address="10.0.0.5",
            port=8080,
            ssh_username="root",
            ssh_port=2200,
            python_path="/usr/bin/python3",
            root_dir="/root/app",
        ),
        score=0,
        job_score=0,
        job_batch_id="b",
        log_status="info",
        log_text="",
        gpu_model=None,
        gpu_count=0,
        availability_errors=errors,
    )


@pytest.mark.parametrize("reason", sorted(REASONS_THAT_HIDE_AT_ONCE))
def test_most_of_a_cycle_failing_a_listed_reason_is_not_silenced(reason: str) -> None:
    failed = _errors_the_service_sends(False, [_event("gpu.validate.x", reason)])
    results = [_result(list(failed)) for _ in range(8)] + [_result([]) for _ in range(2)]

    assert silence_availability_errors_on_our_own_outage(results) == 0
    assert all(r.availability_errors == failed for r in results[:8])


def test_most_of_a_cycle_failing_the_connect_is_still_treated_as_our_own_outage() -> None:
    unreachable = build_availability_event(
        code=AvailabilityErrorCode.EXECUTOR_SSH_UNREACHABLE,
        reach_source=ReachSource.VALIDATOR,
        reach_target=ReachTarget.EXECUTOR_SSH,
        event_text="x",
        impact="x",
        remediation="x",
        what_we_saw={},
    )
    unreachable_errors = _errors_the_service_sends(False, [unreachable])
    results = [_result(list(unreachable_errors)) for _ in range(8)]
    results += [_result([]) for _ in range(2)]

    assert silence_availability_errors_on_our_own_outage(results) == 10
    assert all(r.availability_errors is None for r in results)


# --- through TaskService.create_task -------------------------------------------------------------


class _Shell:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _service(pipeline_run=None, build_context_error=None) -> TaskService:
    service = TaskService.__new__(TaskService)
    service.redis_service = MagicMock()
    service.ssh_service = MagicMock()
    service.ssh_service.decrypt_payload = MagicMock(return_value="key")
    service.attestation_service = MagicMock()
    service.attestation_service.prepare_host_policy = AsyncMock(
        return_value=MagicMock(
            known_hosts=None,
            attestation_digest=None,
            tee_type=None,
            gpu_attestation_passed=None,
            attestation_passed=False,
        )
    )
    service.pipeline_factory = MagicMock()
    service.pipeline_factory.build_context = AsyncMock(side_effect=build_context_error)
    service.pipeline_factory.build_pipeline.return_value.run = AsyncMock(return_value=pipeline_run)
    return service


async def _create_task(service: TaskService) -> JobResult:
    miner = MinerJobRequestPayload(
        job_batch_id="b",
        miner_hotkey="5Miner",
        miner_coldkey="5Cold",
        miner_address="10.0.0.5",
        miner_port=8080,
        executors=[],
    )
    handled = _result([])
    with (
        patch("services.task.service.InteractiveShellService", return_value=_Shell()),
        patch("services.task.service.PipelineFactory.takes_fast_path", return_value=False),
        patch("services.task.service.ResultHandler") as handler,
        patch("services.task.service.run_liumd_shadow", AsyncMock()),
    ):
        handler.return_value.handle_result = AsyncMock(return_value=handled)
        return await service.create_task(
            miner_info=miner,
            executor_info=handled.executor_info,
            keypair=MagicMock(ss58_address="5Val"),
            private_key="key",
            public_key="pub",
            encrypted_files=MagicMock(),
            rented_data=MagicMock(),
            default_docker_image_digests={},
        )


@pytest.mark.asyncio
async def test_create_task_hides_a_node_on_an_allowlisted_failure() -> None:
    run = (
        False,
        [_event("gpu.validate.fingerprint", "GPU_UUID_CHANGED")],
        MagicMock(success=False),
    )

    result = await _create_task(_service(pipeline_run=run))

    assert [e["reason_code"] for e in result.availability_errors] == ["VERIFICATION_FAILED"]
    assert (
        result.availability_errors[0]["what_we_saw"]["failed_check"] == "gpu.validate.fingerprint"
    )


@pytest.mark.asyncio
async def test_create_task_keeps_a_node_listed_on_any_other_failure() -> None:
    run = (False, [_event("gpu.validate.verifyx", "VERIFYX_FAILED")], MagicMock(success=False))

    result = await _create_task(_service(pipeline_run=run))

    assert result.availability_errors == []


@pytest.mark.asyncio
async def test_create_task_keeps_a_policy_score_zero_listed() -> None:
    run = (True, [_event("score", "SCORING_EXCLUDED", "warning")], MagicMock(success=True))

    result = await _create_task(_service(pipeline_run=run))

    assert result.availability_errors == []


@pytest.mark.asyncio
async def test_create_task_keeps_a_node_listed_when_its_shell_dies_under_the_run() -> None:
    result = await _create_task(_service(build_context_error=asyncssh.ConnectionLost("lost")))

    assert result.availability_errors == []
    assert result.failure_reason_code == "EXECUTOR_TRANSPORT_UNREACHABLE"


@pytest.mark.parametrize(
    "error", [RuntimeError("redis down"), ConnectionError("backend call"), TimeoutError()]
)
@pytest.mark.asyncio
async def test_create_task_says_nothing_against_the_node_for_our_own_errors(error) -> None:
    result = await _create_task(_service(build_context_error=error))

    assert result.availability_errors == []
