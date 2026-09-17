"""DAH-3593 — an expected outcome is one line at INFO/WARNING with a reason and no traceback.

Nine validator and central-miner templates were 96 % of the 247k ERROR/WARNING lines of two days
(15–17 Sep 2026), and none of them was a validator fault: a renter deleting the pod mid-create, a
miner that is offline, a provider with no collateral, port and DinD retries, a node failing the
inspector check, a delete finding nothing to delete. Each test here has two halves: the expected
outcome no longer logs at ERROR (fails on the old code), and a real failure on the same path still
does.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import aiohttp
import pytest
import services.rental_docker_sdk as sdk_module
from payload_models.payloads import (
    ContainerCreateRequest,
    MinerJobRequestPayload,
    PayloadPortMapping,
)
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse
from protocol.vc_protocol.validator_requests import ValidationEvent
from services.docker_service import DockerService, _CreateCancelledByDelete
from services.executor_connectivity.dind_probe import DindVerifier
from services.executor_connectivity.models import PortPair
from services.executor_connectivity.port_probe import PortProbe
from services.executor_connectivity.port_verifiers import BatchVerifier
from services.inspector_validation_service import InspectorValidationService
from services.miner_service import MinerService
from services.rental_docker_observability import run_logged_rental_docker_sdk_operation
from services.rental_docker_sdk import (
    ContainerExecSpec,
    RentalDockerContainerRestartingError,
    RentalDockerOperationError,
    RentalDockerSdkClient,
)
from services.task.messages import InspectorMessages
from services.task.pipeline import LoggerSink


def _records(caplog: pytest.LogCaptureFixture, message: str) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.getMessage() == message]


def _extra(record: logging.LogRecord) -> dict:
    return getattr(record.msg, "extra", {})


def _levels(caplog: pytest.LogCaptureFixture) -> set[int]:
    return {r.levelno for r in caplog.records}


# ---------------------------------------------------------------------------
# 1 + 2 — create_container: delete during create, workload container restarting
# ---------------------------------------------------------------------------


def _create_payload() -> ContainerCreateRequest:
    return ContainerCreateRequest(
        miner_hotkey="miner",
        executor_id=str(uuid4()),
        pod_id=str(uuid4()),
        docker_image="daturaai/pytorch:1.0.0",
        user_public_keys=["ssh-ed25519 test-key"],
        gpu_uuids=["GPU-test"],
        cpu_count=1,
        memory_gb=1,
        volume_limit_gb=2,
        storage_limit_gb=1,
        available_ports=[PayloadPortMapping(internal_port=20001, external_port=20001)],
        pod_mapping=[],
        active_container_names=[],
        active_volume_names=[],
    )


def _executor_info(payload: ContainerCreateRequest):
    return SimpleNamespace(
        uuid=payload.executor_id,
        address="203.0.113.10",
        port=8001,
        ssh_username="root",
        ssh_port=22,
        ssh_host_key="ssh-ed25519 AAAA host",
    )


async def _create_failing_with(exc: Exception, caplog: pytest.LogCaptureFixture):
    svc = DockerService(ssh_service=Mock(), redis_service=Mock(), attestation_service=Mock())
    svc.redis_service.remove_pending_pod = AsyncMock()
    svc.finish_stream_logs = AsyncMock()
    # the first awaited step inside create_container's try: raising here reaches the one handler
    # every create failure reaches, with the exception the real path would raise
    svc.generate_portMappings = AsyncMock(side_effect=exc)
    payload = _create_payload()
    caplog.set_level(logging.DEBUG, logger="services.docker_service")
    result = await svc.create_container(
        payload=payload,
        executor_info=_executor_info(payload),
        keypair=Mock(ss58_address="validator"),
        private_key="encrypted",
    )
    return payload, result


@pytest.mark.asyncio
async def test_create_cancelled_by_delete_logs_one_info_line_without_traceback(caplog):
    payload, result = await _create_failing_with(
        _CreateCancelledByDelete("delete for pod arrived while pod_x was being created"), caplog
    )

    lines = _records(caplog, "create cancelled by delete")
    assert len(lines) == 1
    assert lines[0].levelno == logging.INFO
    assert lines[0].exc_info is None
    assert _extra(lines[0])["reason"] == "cancelled_by_delete"
    assert _extra(lines[0])["pod_id"] == payload.pod_id
    assert _extra(lines[0])["workload_kind"] == payload.workload_kind.value
    assert logging.ERROR not in _levels(caplog)
    # the backend still receives the same failure it always did
    assert result.msg == "Failed create_container"
    assert result.failure_step == "cancelled_by_delete"


@pytest.mark.asyncio
async def test_create_on_restarting_workload_container_logs_warning_with_the_cause(caplog):
    _, result = await _create_failing_with(
        RentalDockerContainerRestartingError(
            "container restarting, exit_code=1, last log line: 'no CUDA device'"
        ),
        caplog,
    )

    lines = _records(caplog, "workload container keeps restarting; create failed")
    assert len(lines) == 1
    assert lines[0].levelno == logging.WARNING
    assert lines[0].exc_info is None
    assert _extra(lines[0])["reason"] == "workload_container_restarting"
    assert "exit_code=1" in _extra(lines[0])["error"]
    assert logging.ERROR not in _levels(caplog)
    assert result.msg == "Failed create_container"


@pytest.mark.asyncio
async def test_create_real_failure_still_logs_error_with_traceback(caplog):
    _, result = await _create_failing_with(RuntimeError("docker run exploded"), caplog)

    lines = _records(caplog, "Failed create_container")
    assert len(lines) == 1
    assert lines[0].levelno == logging.ERROR
    assert lines[0].exc_info is not None
    assert result.msg == "Failed create_container"


# ---------------------------------------------------------------------------
# 2 — the SDK names the container state instead of Docker's 409 text
# ---------------------------------------------------------------------------


class _RestartConflict(Exception):
    def __init__(self):
        super().__init__(
            "409 Client Error: Conflict (\"Container abc is restarting, wait until the container is running\")"
        )
        self.response = SimpleNamespace(status_code=409)


class _RestartingApiClient:
    def __init__(self):
        self.exec_create_calls = 0

    def inspect_container(self, name):
        return {"State": {"Status": "running", "Running": True, "Restarting": False, "ExitCode": 137}}

    def exec_create(self, *args, **kwargs):
        self.exec_create_calls += 1
        raise _RestartConflict()

    def logs(self, name, stdout=True, stderr=True, tail=1):
        return b"terminate called after throwing an instance of 'std::bad_alloc'\n"


@pytest.mark.asyncio
async def test_exec_on_restarting_container_names_exit_code_and_last_log_line(monkeypatch):
    monkeypatch.setattr(sdk_module, "_DOCKER_EXEC_TRANSIENT_RETRY_DELAYS_SECONDS", (0,))
    api = _RestartingApiClient()
    client = RentalDockerSdkClient(api)

    with pytest.raises(RentalDockerContainerRestartingError) as raised:
        await client.exec_in_container(
            ContainerExecSpec(container_name="pod_x", argv=("true",))
        )

    text = str(raised.value)
    assert text.startswith("container restarting, exit_code=137, last log line: ")
    assert "std::bad_alloc" in text
    assert "409" not in text
    assert api.exec_create_calls == 2  # the retry budget was spent before the cause was read
    assert isinstance(raised.value.__cause__, _RestartConflict)


# ---------------------------------------------------------------------------
# 8 — a remove that finds nothing is INFO; a run that fails is still ERROR
# ---------------------------------------------------------------------------


class _NotFound(Exception):
    def __init__(self, what: str):
        super().__init__(f"404 Client Error: Not Found (\"{what}\")")
        self.response = SimpleNamespace(status_code=404)


async def _raise_wrapped(label: str, cause: Exception):
    try:
        raise cause
    except Exception as exc:
        raise RentalDockerOperationError(f"Docker SDK {label} failed: {exc}") from exc


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["remove_container", "remove_volume"])
async def test_remove_of_an_already_gone_object_logs_info(caplog, operation):
    caplog.set_level(logging.DEBUG, logger="services.rental_docker_observability")

    with pytest.raises(RentalDockerOperationError):
        await run_logged_rental_docker_sdk_operation(
            operation=operation,
            log_extra={"pod_id": "p"},
            call=lambda: _raise_wrapped(operation, _NotFound("No such container: pod_p")),
        )

    failed = [r for r in caplog.records if _extra(r).get("operation_status") == "failed"]
    assert len(failed) == 1
    assert failed[0].levelno == logging.INFO
    assert _extra(failed[0])["reason"] == "already_gone"


@pytest.mark.asyncio
async def test_run_container_failure_still_logs_error(caplog):
    caplog.set_level(logging.DEBUG, logger="services.rental_docker_observability")
    bind_failed = RuntimeError("500 Server Error: Bind for 0.0.0.0:19100 failed: port is already allocated")

    with pytest.raises(RentalDockerOperationError):
        await run_logged_rental_docker_sdk_operation(
            operation="run_container",
            log_extra={"pod_id": "p"},
            call=lambda: _raise_wrapped("run container", bind_failed),
        )

    failed = [r for r in caplog.records if _extra(r).get("operation_status") == "failed"]
    assert len(failed) == 1
    assert failed[0].levelno == logging.ERROR
    assert "reason" not in _extra(failed[0])


@pytest.mark.asyncio
async def test_remove_that_fails_for_another_reason_still_logs_error(caplog):
    caplog.set_level(logging.DEBUG, logger="services.rental_docker_observability")

    with pytest.raises(RentalDockerOperationError):
        await run_logged_rental_docker_sdk_operation(
            operation="remove_container",
            log_extra={"pod_id": "p"},
            call=lambda: _raise_wrapped("remove container", TimeoutError("Read timed out")),
        )

    failed = [r for r in caplog.records if _extra(r).get("operation_status") == "failed"]
    assert failed[0].levelno == logging.ERROR


# ---------------------------------------------------------------------------
# 3 — an offline miner is one WARNING per miner per cycle
# ---------------------------------------------------------------------------


def _miner_service(mocker, rest_error: Exception) -> MinerService:
    from core.config import settings

    my_key = Mock(ss58_address="validator")
    my_key.sign.return_value = b"\x01\x02"
    mocker.patch(
        "core.config.Settings.get_bittensor_wallet",
        return_value=Mock(get_hotkey=Mock(return_value=my_key)),
    )
    mocker.patch.object(settings, "USE_REST_API", True)
    ssh_service = Mock()
    ssh_service.generate_ssh_key.return_value = (b"priv", b"ssh-ed25519 pub")
    service = MinerService(
        ssh_service=ssh_service,
        task_service=Mock(),
        redis_service=AsyncMock(),
        attestation_service=Mock(maybe_issue_nonce=AsyncMock(return_value=None)),
    )
    service._make_rest_request = AsyncMock(side_effect=rest_error)
    return service


def _rented(miner: str, other: str) -> RentedExecutorsResponse:
    def one(hotkey: str) -> RentedExecutor:
        return RentedExecutor(
            miner_hotkey=hotkey, executor_ip_address="203.0.113.5", executor_ip_port="8001", pods=[]
        )

    return RentedExecutorsResponse(
        executors={str(uuid4()): one(miner), str(uuid4()): one(miner), str(uuid4()): one(other)}
    )


def _job_payload() -> MinerJobRequestPayload:
    return MinerJobRequestPayload(
        job_batch_id="2026-09-17 12:00:00",
        miner_hotkey="miner-a",
        miner_coldkey="cold-a",
        miner_address="203.0.113.7",
        miner_port=8000,
    )


async def _request_job(service: MinerService):
    return await service.request_job_to_miner(
        payload=_job_payload(),
        encrypted_files=Mock(),
        rented_data=_rented("miner-a", "miner-b"),
        default_docker_image_digests={},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rest_error",
    [
        TimeoutError(),
        aiohttp.ClientConnectorError(
            SimpleNamespace(ssl=None, host="203.0.113.7", port=8000), OSError(111, "Connection refused")
        ),
    ],
    ids=["timeout", "connection refused"],
)
async def test_offline_miner_is_one_warning_with_the_executors_it_cost(mocker, caplog, rest_error):
    caplog.set_level(logging.DEBUG, logger="services.miner_service")
    service = _miner_service(mocker, rest_error)

    result = await _request_job(service)

    warnings = _records(
        caplog, "Miner did not answer the REST job request; its executors are skipped this cycle"
    )
    assert len(warnings) == 1
    assert warnings[0].levelno == logging.WARNING
    assert _extra(warnings[0])["reason"] == "miner_unreachable"
    assert _extra(warnings[0])["miner_hotkey"] == "miner-a"
    assert _extra(warnings[0])["rented_executors_skipped"] == 2
    assert logging.ERROR not in _levels(caplog)
    assert result["results"][0].log_status == "error"  # the cycle's own bookkeeping is unchanged


@pytest.mark.asyncio
async def test_miner_answering_garbage_is_still_an_error(mocker, caplog):
    caplog.set_level(logging.DEBUG, logger="services.miner_service")
    service = _miner_service(mocker, aiohttp.InvalidURL("http://[::1]:8000/x"))

    await _request_job(service)

    errors = _records(caplog, "Requesting job to miner via REST API resulted in an exception")
    assert len(errors) == 1 and errors[0].levelno == logging.ERROR
    assert not _records(
        caplog, "Miner did not answer the REST job request; its executors are skipped this cycle"
    )


@pytest.mark.asyncio
async def test_make_rest_request_no_longer_doubles_the_timeout_line(mocker, caplog):
    caplog.set_level(logging.DEBUG, logger="services.miner_service")
    service = MinerService(
        ssh_service=Mock(), task_service=Mock(), redis_service=AsyncMock(), attestation_service=Mock()
    )

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def request(self, **kwargs):
            raise TimeoutError()

    mocker.patch("services.miner_service.aiohttp.ClientSession", return_value=_Session())

    with pytest.raises(asyncio.TimeoutError):
        await service._make_rest_request(
            method="POST", url="http://203.0.113.7:8000/x", json_data={}, headers={}, timeout=1,
            log_extra={}, operation_name="SSH key submit",
        )

    timed_out = _records(caplog, "REST API SSH key submit timed out after 1s")
    assert len(timed_out) == 1 and timed_out[0].levelno == logging.DEBUG


@pytest.mark.asyncio
async def test_ssh_key_removal_from_an_offline_miner_is_info(caplog):
    caplog.set_level(logging.DEBUG, logger="services.miner_service")
    service = MinerService(
        ssh_service=Mock(), task_service=Mock(), redis_service=AsyncMock(), attestation_service=Mock()
    )
    service._make_rest_request = AsyncMock(side_effect=TimeoutError())
    my_key = Mock(ss58_address="validator")
    my_key.sign.return_value = b"\x01"

    ok = await service._remove_ssh_key_via_rest(
        "http://203.0.113.7:8000", my_key, b"ssh-ed25519 pub", "miner-a", None, {}
    )

    assert ok is False
    line = _records(caplog, "Failed to remove SSH key via REST API. Validator key may still be present on miner")
    assert len(line) == 1 and line[0].levelno == logging.INFO
    assert _extra(line[0])["reason"] == "miner_unreachable"


@pytest.mark.asyncio
async def test_ssh_key_removal_refused_by_the_miner_stays_a_warning(caplog):
    caplog.set_level(logging.DEBUG, logger="services.miner_service")
    service = MinerService(
        ssh_service=Mock(), task_service=Mock(), redis_service=AsyncMock(), attestation_service=Mock()
    )
    service._make_rest_request = AsyncMock(side_effect=ValueError("bad signature payload"))
    my_key = Mock(ss58_address="validator")
    my_key.sign.return_value = b"\x01"

    await service._remove_ssh_key_via_rest(
        "http://203.0.113.7:8000", my_key, b"ssh-ed25519 pub", "miner-a", None, {}
    )

    line = _records(caplog, "Failed to remove SSH key via REST API. Validator key may still be present on miner")
    assert line[0].levelno == logging.WARNING
    assert _extra(line[0])["reason"] == "remove_failed"


# ---------------------------------------------------------------------------
# 4 — provider-state verdicts log at INFO; other verdicts keep their severity
# ---------------------------------------------------------------------------


def _event(reason_code: str, severity: str) -> ValidationEvent:
    return ValidationEvent(
        event=f"event {reason_code}",
        reason_code=reason_code,
        severity=severity,
        impact="x",
        when=datetime(2026, 9, 17, tzinfo=UTC),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason_code",
    ["COLLATERAL_MISSING", "EXECUTOR_IMAGE_OUTDATED", "PROVIDER_BANNED", "PROVIDER_SIDE_LOAD_ABOVE_LIMIT"],
)
async def test_provider_state_verdict_logs_at_info_and_keeps_its_severity(caplog, reason_code):
    caplog.set_level(logging.DEBUG, logger="test.sink")
    event = _event(reason_code, "warning")

    await LoggerSink(logging.getLogger("test.sink")).emit(event)

    assert [r.levelno for r in caplog.records] == [logging.INFO]
    assert _extra(caplog.records[0])["reason"] == "provider_state"
    assert _extra(caplog.records[0])["severity"] == "warning"  # what the backend receives
    assert event.severity == "warning"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reason_code", "severity", "expected"),
    [
        ("GPU_FINGERPRINT_CHANGED", "warning", logging.WARNING),
        ("EXECUTOR_IMAGE_OUTDATED", "error", logging.ERROR),  # enforcement on: scored, stays loud
        ("PORT_VERIFICATION_FAILED", "error", logging.ERROR),
    ],
)
async def test_other_verdicts_keep_their_level(caplog, reason_code, severity, expected):
    caplog.set_level(logging.DEBUG, logger="test.sink")

    await LoggerSink(logging.getLogger("test.sink")).emit(_event(reason_code, severity))

    assert [r.levelno for r in caplog.records] == [expected]
    assert "reason" not in _extra(caplog.records[0])


# ---------------------------------------------------------------------------
# 5 — port verification: attempts and tiers are DEBUG, the exception path is still ERROR
# ---------------------------------------------------------------------------


class _Runner:
    def __init__(self, *, start_ok: bool = False, raise_on_run: Exception | None = None):
        self.start_ok = start_ok
        self.raise_on_run = raise_on_run
        self.cleanups = 0

    async def run(self, ssh_client, name, script, network_flag, timeout):
        if self.raise_on_run is not None:
            raise self.raise_on_run
        return SimpleNamespace(ok=self.start_ok, status="exited", logs="bind: address already in use")

    async def cleanup(self, ssh_client, name):
        self.cleanups += 1


def _ports(n: int) -> list[PortPair]:
    return [PortPair(internal=40000 + i, external=40000 + i) for i in range(n)]


@pytest.mark.asyncio
async def test_port_attempts_that_fail_to_start_are_debug_only(caplog, monkeypatch):
    caplog.set_level(logging.DEBUG, logger="services.executor_connectivity")
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    verifier = BatchVerifier(port_tester=Mock(), runner=_Runner(start_ok=False))

    successful, failed = await verifier.verify(_ports(3), ssh_client=Mock(), host="203.0.113.9")

    assert successful == [] and len(failed) == 3
    assert _records(caplog, "attempt 1 failed, retrying in 2s")[0].levelno == logging.DEBUG
    assert _records(caplog, "all 2 attempts failed")[0].levelno == logging.DEBUG
    assert not _levels(caplog) & {logging.WARNING, logging.ERROR}


@pytest.mark.asyncio
async def test_port_attempt_that_raises_is_still_an_error_with_traceback(caplog, monkeypatch):
    caplog.set_level(logging.DEBUG, logger="services.executor_connectivity")
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    verifier = BatchVerifier(
        port_tester=Mock(), runner=_Runner(raise_on_run=RuntimeError("netcat script template broke"))
    )

    await verifier.verify(_ports(2), ssh_client=Mock(), host="203.0.113.9")

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 2  # one per attempt
    assert all(r.exc_info is not None for r in errors)


@pytest.mark.asyncio
async def test_tier_fallthrough_is_debug(caplog):
    caplog.set_level(logging.DEBUG, logger="services.executor_connectivity")

    async def nothing(ports, **kwargs):
        return [], list(ports)

    probe = PortProbe(
        batch_verifier=Mock(verify=nothing),
        semi_batch_verifier=Mock(verify=nothing),
        fallback_verifier=Mock(verify=nothing),
    )

    result = await probe.probe(_ports(2), ssh_client=Mock(), host="203.0.113.9")

    assert result.successful == ()
    assert _records(caplog, "batch verification failed, trying semi-batch")[0].levelno == logging.DEBUG
    assert _records(caplog, "semi-batch verification failed, trying fallback")[0].levelno == logging.DEBUG
    assert not _levels(caplog) & {logging.WARNING, logging.ERROR}


# ---------------------------------------------------------------------------
# 6 — DinD: the node's refusal is one WARNING; our own exception is still ERROR
# ---------------------------------------------------------------------------


def _dind_verifier() -> DindVerifier:
    ssh_service = Mock()
    ssh_service.generate_keypair.return_value = ("priv", "ssh-ed25519 pub")
    return DindVerifier(ssh_service)


@pytest.mark.asyncio
async def test_dind_run_refused_by_the_host_is_a_warning(caplog):
    caplog.set_level(logging.DEBUG, logger="services.executor_connectivity")
    ssh_client = Mock()
    ssh_client.run = AsyncMock(
        return_value=SimpleNamespace(exit_status=125, stderr="Bind for 0.0.0.0:40032 failed: port is already allocated")
    )

    result = await _dind_verifier().verify(
        PortPair(internal=40032, external=40032),
        ssh_client=ssh_client,
        host="203.0.113.9",
        container_name_prefix="dind_test",
        sysbox=False,
    )

    assert result.success is False
    line = _records(caplog, "DinD creation failed")
    assert len(line) == 1 and line[0].levelno == logging.WARNING
    assert _extra(line[0])["reason"] == "dind_run_refused_by_host"
    assert logging.ERROR not in _levels(caplog)


@pytest.mark.asyncio
async def test_dind_sshd_never_answering_is_one_warning_without_traceback(caplog, mocker):
    caplog.set_level(logging.DEBUG, logger="services.executor_connectivity")
    ssh_client = Mock()
    ssh_client.run = AsyncMock(return_value=SimpleNamespace(exit_status=0, stderr=""))
    mocker.patch("services.executor_connectivity.dind_probe.asyncssh.import_private_key")
    verifier = _dind_verifier()
    verifier._connect_retrying_until_sshd_answers = AsyncMock(side_effect=TimeoutError())

    result = await verifier.verify(
        PortPair(internal=40032, external=40032),
        ssh_client=ssh_client,
        host="203.0.113.9",
        container_name_prefix="dind_test",
        sysbox=True,
    )

    assert result.success is False
    line = _records(caplog, "DinD check failed")
    assert len(line) == 1 and line[0].levelno == logging.WARNING
    assert line[0].exc_info is None
    assert _extra(line[0])["reason"] == "dind_node_unreachable"
    assert _extra(line[0])["error"] == "TimeoutError"


@pytest.mark.asyncio
async def test_dind_probe_bug_is_still_an_error_with_traceback(caplog, mocker):
    caplog.set_level(logging.DEBUG, logger="services.executor_connectivity")
    ssh_client = Mock()
    ssh_client.run = AsyncMock(return_value=SimpleNamespace(exit_status=0, stderr=""))
    mocker.patch(
        "services.executor_connectivity.dind_probe.asyncssh.import_private_key",
        side_effect=ValueError("not a private key"),
    )

    await _dind_verifier().verify(
        PortPair(internal=40032, external=40032),
        ssh_client=ssh_client,
        host="203.0.113.9",
        container_name_prefix="dind_test",
        sysbox=True,
    )

    line = _records(caplog, "DinD check failed")
    assert len(line) == 1 and line[0].levelno == logging.ERROR
    assert line[0].exc_info is not None


# ---------------------------------------------------------------------------
# 7 — inspector: the node's verdict is a WARNING; an unclassified exception is still ERROR
# ---------------------------------------------------------------------------


def _inspector() -> InspectorValidationService:
    return InspectorValidationService.__new__(InspectorValidationService)


def test_inspector_node_verdict_is_a_warning(caplog):
    caplog.set_level(logging.DEBUG, logger="services.inspector_validation_service")

    response = _inspector()._failure_response(
        error="inspector executor reported: collector not running",
        message=InspectorMessages.FAILED_INTERACTIVE,
        diagnostics={"executor_uuid": "exec-1"},
        default_extra={"miner_hotkey": "m"},
        error_type="InspectorInteractiveError",
    )

    line = _records(caplog, "Inspector validation failed")
    assert len(line) == 1 and line[0].levelno == logging.WARNING
    assert _extra(line[0])["reason"] == "INSPECTOR_FAILED_INTERACTIVE"
    assert _extra(line[0])["reason_class"] == "node_verdict"
    assert response.message is InspectorMessages.FAILED_INTERACTIVE


def test_inspector_unclassified_exception_is_still_an_error(caplog):
    caplog.set_level(logging.DEBUG, logger="services.inspector_validation_service")

    _inspector()._failure_response(
        error="Expecting value: line 1 column 1 (char 0)",
        message=InspectorMessages.VALIDATION_ERROR,
        diagnostics={"executor_uuid": "exec-1"},
        default_extra={"miner_hotkey": "m"},
        error_type="JSONDecodeError",
    )

    line = _records(caplog, "Inspector validation failed")
    assert len(line) == 1 and line[0].levelno == logging.ERROR
