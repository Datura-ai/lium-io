"""Characterization oracle group C-RETRY (C10-C17, DAH-2382 PR-1a).

Direct unit tests on the LIVE SDK create-retry loop
``DockerService._run_rental_docker_create_with_port_retry``
(src/services/docker_service.py:542, sole call site create_container:3499).

This suite is the port of the DEAD SSH twin's suite
(``_run_docker_create_with_port_retry``, tested at tests/test_docker_service.py:3335-4003
but never called from src/) onto the live function. The twin and its tests are
deleted in PR-3; these tests replace their intent on the production path.

Pinned live semantics (baseline 6be5649f):
- stale-mount match requires ALL THREE ``_VLOOPBACK_MOUNT_ERROR_PHRASES``
  ("VolumeDriver.Mount", "cannot create mount point dir", "file exists");
- vloopback repair runs at most once, only when ``local_volume`` is set, and
  retries immediately (no sleep, no rm) after event VLOOPBACK_STALE_MOUNTPOINT_RETRY;
- any single ``_PORT_ALLOCATED_PHRASES`` match retries on a 90s monotonic budget:
  event PORT_ALREADY_ALLOCATED_RETRY, sleep 5s, then
  ``remove_container(container_name=..., force=True, remove_volumes=True)``
  (rm failure warns PORT_RETRY_STALE_RM_FAILED and continues);
- unlike the dead twin, the final non-retryable error is logged as
  "Docker SDK run container failed" and streamed via ``stream_log`` before re-raise.
"""

import logging
from unittest.mock import AsyncMock, Mock

import pytest
from docker_oracle.harness import FakeRentalDockerClient, RecordedCall, make_docker_service
from services.rental_docker_sdk import ContainerRunSpec

# Same daemon texts the dead twin's suite used (tests/test_docker_service.py:3340/3392/3396),
# plus single-phrase variants so each _PORT_ALLOCATED_PHRASES entry is pinned in isolation.
_PORT_ALLOCATED_ERR = (
    "docker: Error response from daemon: failed to set up container "
    "networking: driver failed programming external connectivity on "
    "endpoint pod_599e3ed0-...: Bind for 0.0.0.0:9101 failed: port is "
    "already allocated"
)
_ADDRESS_IN_USE_ERR = (
    "docker: Error response from daemon: driver failed programming external "
    "connectivity on endpoint pod_test: Error starting userland proxy: "
    "listen tcp4 0.0.0.0:9030: bind: address already in use"
)
_BIND_HOST_PORT_ERR = (
    "docker: Error response from daemon: failed to bind host port 0.0.0.0:20001/tcp"
)
_VLOOPBACK_STALE_MOUNT_ERR = (
    "docker: Error response from daemon: failed to populate volume: "
    "error while mounting volume '/mnt/volume_test': "
    "VolumeDriver.Mount: error while mounting volume: "
    "cannot create mount point dir '/mnt/volume_test': "
    "mkdir /mnt/volume_test: file exists"
)
# Only 2 of the 3 required phrases ("file exists" absent).
_PARTIAL_VLOOPBACK_ERR = (
    "docker: Error response from daemon: "
    "VolumeDriver.Mount: error while mounting volume: "
    "cannot create mount point dir '/mnt/volume_test': permission denied"
)
_NO_SUCH_IMAGE_ERR = "docker: Error response from daemon: No such image: bogus:latest"


def _make_run_spec() -> ContainerRunSpec:
    return ContainerRunSpec(image="daturaai/pytorch:1.0.0", name="pod_test")


def _patch_sleep(
    monkeypatch: pytest.MonkeyPatch, journal: list[RecordedCall] | None = None
) -> list[float]:
    # record retry backoffs without waiting; same patch target as the dead twin's tests
    delays: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        delays.append(seconds)
        if journal is not None:
            journal.append(("sleep", {"seconds": seconds}))

    monkeypatch.setattr("services.docker_service.asyncio.sleep", _fake_sleep)
    return delays


class _SleepAdvancedClock:
    """Frozen fake monotonic clock that jumps forward only when the patched sleep fires.

    Deterministic regardless of how many callers read the clock (the retry loop,
    the observability wrapper, the event loop), unlike a finite scripted iterator.
    """

    def __init__(self, *, start: float, advance_per_sleep: float) -> None:
        self.now = start
        self.advance_per_sleep = advance_per_sleep

    def monotonic(self) -> float:
        return self.now

    def advance(self) -> None:
        self.now += self.advance_per_sleep


def _event_names(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records]


def _run_started_attempts(caplog: pytest.LogCaptureFixture) -> list[int]:
    # attempt field of the per-try observability "started" line for run_container
    return [
        record.msg.extra["attempt"]
        for record in caplog.records
        if record.getMessage() == "Rental Docker SDK operation started"
        and record.msg.extra.get("docker_operation") == "run_container"
    ]


@pytest.mark.asyncio
async def test_sdk_repair_once_then_retry_success(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Arrange
    caplog.set_level(logging.INFO)
    svc = make_docker_service()
    docker_client = FakeRentalDockerClient()
    docker_client.run_errors = [Exception(_VLOOPBACK_STALE_MOUNT_ERR)]
    repair = AsyncMock(return_value=True)
    monkeypatch.setattr(svc, "repair_stale_vloopback_mountpoint", repair)
    delays = _patch_sleep(monkeypatch)
    ssh_client = Mock()
    run_spec = _make_run_spec()

    # Act
    await svc._run_rental_docker_create_with_port_retry(
        docker_client=docker_client,
        ssh_client=ssh_client,
        run_spec=run_spec,
        container_name="pod_test",
        default_extra={},
        local_volume="volume_test",
        log_tag="t",
    )

    # Assert
    # WHY: a stale-mount failure (all 3 phrases) must repair exactly once and
    # immediately re-run the SAME spec -- no backoff sleep and no stale-rm on
    # the repair path, unlike the port-retry path.
    assert docker_client.run_specs == [run_spec, run_spec]
    assert docker_client.run_specs[0] is run_spec and docker_client.run_specs[1] is run_spec
    repair.assert_awaited_once_with(ssh_client, "volume_test", {})
    assert delays == []
    assert docker_client.removed_containers == []
    assert _event_names(caplog).count("VLOOPBACK_STALE_MOUNTPOINT_RETRY") == 1
    # WHY: the port-attempt counter is NOT bumped by a repair retry -- both run
    # tries are logged as attempt=1 (live-only observability field).
    assert _run_started_attempts(caplog) == [1, 1]


@pytest.mark.asyncio
async def test_sdk_repair_not_attempted_twice(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Arrange
    caplog.set_level(logging.INFO)
    svc = make_docker_service()
    docker_client = FakeRentalDockerClient()
    docker_client.run_error = Exception(_VLOOPBACK_STALE_MOUNT_ERR)
    repair = AsyncMock(return_value=True)
    monkeypatch.setattr(svc, "repair_stale_vloopback_mountpoint", repair)
    delays = _patch_sleep(monkeypatch)

    # Act
    with pytest.raises(Exception) as excinfo:
        await svc._run_rental_docker_create_with_port_retry(
            docker_client=docker_client,
            ssh_client=Mock(),
            run_spec=_make_run_spec(),
            container_name="pod_test",
            default_extra={},
            local_volume="volume_test",
            log_tag="t",
        )

    # Assert
    # WHY: the repair-once flag caps the loop at one repair + one re-run; a second
    # stale-mount failure propagates through the live error funnel (error log +
    # streamed error), which the dead twin did not have.
    assert _VLOOPBACK_STALE_MOUNT_ERR in str(excinfo.value)
    repair.assert_awaited_once()
    assert len(docker_client.run_specs) == 2
    assert delays == []
    assert _event_names(caplog).count("VLOOPBACK_STALE_MOUNTPOINT_RETRY") == 1
    assert "Docker SDK run container failed" in _event_names(caplog)
    assert svc.logs_queue == [
        {"log_text": _VLOOPBACK_STALE_MOUNT_ERR, "log_status": "error", "log_tag": "t"}
    ]


@pytest.mark.asyncio
async def test_sdk_repair_skipped_without_local_volume(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Arrange
    caplog.set_level(logging.INFO)
    svc = make_docker_service()
    docker_client = FakeRentalDockerClient()
    docker_client.run_error = Exception(_VLOOPBACK_STALE_MOUNT_ERR)
    repair = AsyncMock(return_value=True)
    monkeypatch.setattr(svc, "repair_stale_vloopback_mountpoint", repair)
    delays = _patch_sleep(monkeypatch)

    # Act
    with pytest.raises(Exception) as excinfo:
        await svc._run_rental_docker_create_with_port_retry(
            docker_client=docker_client,
            ssh_client=Mock(),
            run_spec=_make_run_spec(),
            container_name="pod_test",
            default_extra={},
            log_tag="t",
        )

    # Assert
    # WHY: the bool(local_volume) guard -- without a local volume a stale-mount
    # error must raise on the first try and never touch the repair helper.
    assert _VLOOPBACK_STALE_MOUNT_ERR in str(excinfo.value)
    repair.assert_not_awaited()
    assert len(docker_client.run_specs) == 1
    assert delays == []
    assert docker_client.removed_containers == []


@pytest.mark.asyncio
async def test_sdk_partial_mount_phrases_do_not_trigger_repair(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Arrange
    caplog.set_level(logging.INFO)
    svc = make_docker_service()
    docker_client = FakeRentalDockerClient()
    docker_client.run_error = Exception(_PARTIAL_VLOOPBACK_ERR)
    repair = AsyncMock(return_value=True)
    monkeypatch.setattr(svc, "repair_stale_vloopback_mountpoint", repair)
    delays = _patch_sleep(monkeypatch)

    # Act
    with pytest.raises(Exception) as excinfo:
        await svc._run_rental_docker_create_with_port_retry(
            docker_client=docker_client,
            ssh_client=Mock(),
            run_spec=_make_run_spec(),
            container_name="pod_test",
            default_extra={},
            local_volume="volume_test",
            log_tag="t",
        )

    # Assert
    # WHY: _is_stale_vloopback_mountpoint_error requires ALL THREE phrases
    # (all(), not any()); 2 of 3 must not trigger repair even with a local volume.
    assert _PARTIAL_VLOOPBACK_ERR in str(excinfo.value)
    repair.assert_not_awaited()
    assert len(docker_client.run_specs) == 1
    assert delays == []


@pytest.mark.asyncio
async def test_sdk_port_retry_then_success(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Arrange
    caplog.set_level(logging.INFO)
    svc = make_docker_service()
    docker_client = FakeRentalDockerClient()
    docker_client.run_errors = [Exception(_PORT_ALLOCATED_ERR)]
    delays = _patch_sleep(monkeypatch)
    run_spec = _make_run_spec()

    # Act
    await svc._run_rental_docker_create_with_port_retry(
        docker_client=docker_client,
        ssh_client=Mock(),
        run_spec=run_spec,
        container_name="pod_test",
        default_extra={},
        log_tag="t",
    )

    # Assert
    # WHY: DAH-1991/DAH-2018 -- one port-allocated failure means one 5s backoff,
    # one stale-container rm (force + anonymous volumes), then the SAME spec
    # re-run; the retry event carries attempt/sleep/phrase for the dashboards.
    assert docker_client.run_specs == [run_spec, run_spec]
    assert docker_client.run_specs[0] is run_spec and docker_client.run_specs[1] is run_spec
    assert delays == [5]
    assert docker_client.removed_containers == [
        {"container_name": "pod_test", "force": True, "remove_volumes": True}
    ]
    retry_records = [
        record for record in caplog.records
        if record.getMessage() == "PORT_ALREADY_ALLOCATED_RETRY"
    ]
    assert len(retry_records) == 1
    assert retry_records[0].msg.extra["attempt"] == 1
    assert retry_records[0].msg.extra["sleep_seconds"] == 5
    assert retry_records[0].msg.extra["port_allocation_phrase"] == "port is already allocated"
    assert _run_started_attempts(caplog) == [1, 2]
    # WHY: the success path streams nothing from inside the retry loop.
    assert svc.logs_queue == []


@pytest.mark.asyncio
async def test_sdk_port_retry_sleeps_then_removes_then_reruns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    journal: list[RecordedCall] = []
    svc = make_docker_service()
    docker_client = FakeRentalDockerClient(journal)
    docker_client.run_errors = [Exception(_PORT_ALLOCATED_ERR)]
    _patch_sleep(monkeypatch, journal)

    # Act
    await svc._run_rental_docker_create_with_port_retry(
        docker_client=docker_client,
        ssh_client=Mock(),
        run_spec=_make_run_spec(),
        container_name="pod_test",
        default_extra={},
        log_tag="t",
    )

    # Assert
    # WHY: DAH-2018 ordering -- backoff sleep FIRST, then rm, then re-run, so the
    # rm->run window stays as tight as possible (port of the dead twin's
    # events-order test onto the SDK journal).
    assert [name for name, _ in journal] == [
        "docker.run_container",
        "sleep",
        "docker.remove_container",
        "docker.run_container",
    ]


@pytest.mark.asyncio
async def test_sdk_port_retry_deadline_expired(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Arrange
    caplog.set_level(logging.INFO)
    svc = make_docker_service()
    docker_client = FakeRentalDockerClient()
    docker_client.run_error = Exception(_PORT_ALLOCATED_ERR)
    clock = _SleepAdvancedClock(start=1000.0, advance_per_sleep=50.0)
    monkeypatch.setattr("services.docker_service.time.monotonic", clock.monotonic)
    delays: list[float] = []

    async def _sleep_and_advance(seconds: float) -> None:
        delays.append(seconds)
        clock.advance()

    monkeypatch.setattr("services.docker_service.asyncio.sleep", _sleep_and_advance)

    # Act
    with pytest.raises(Exception) as excinfo:
        await svc._run_rental_docker_create_with_port_retry(
            docker_client=docker_client,
            ssh_client=Mock(),
            run_spec=_make_run_spec(),
            container_name="pod_test",
            default_extra={},
            log_tag="t",
        )

    # Assert
    # WHY: the 90s monotonic budget -- deadline checks at t=+0s and t=+50s allow
    # two retries, the check at t=+100s expires and the port error propagates
    # (so the create boundary later reports failure_step="docker_run").
    assert _PORT_ALLOCATED_ERR in str(excinfo.value)
    assert len(docker_client.run_specs) == 3
    assert delays == [5, 5]
    assert len(docker_client.removed_containers) == 2
    retry_attempts = [
        record.msg.extra["attempt"]
        for record in caplog.records
        if record.getMessage() == "PORT_ALREADY_ALLOCATED_RETRY"
    ]
    assert retry_attempts == [1, 2]
    assert "Docker SDK run container failed" in _event_names(caplog)
    assert svc.logs_queue == [
        {"log_text": _PORT_ALLOCATED_ERR, "log_status": "error", "log_tag": "t"}
    ]


@pytest.mark.asyncio
async def test_sdk_port_retry_second_phrase_address_in_use(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Arrange
    caplog.set_level(logging.INFO)
    svc = make_docker_service()
    docker_client = FakeRentalDockerClient()
    docker_client.run_errors = [Exception(_ADDRESS_IN_USE_ERR)]
    delays = _patch_sleep(monkeypatch)

    # Act
    await svc._run_rental_docker_create_with_port_retry(
        docker_client=docker_client,
        ssh_client=Mock(),
        run_spec=_make_run_spec(),
        container_name="pod_test",
        default_extra={},
        log_tag="t",
    )

    # Assert
    # WHY: DAH-2065 -- kernel-level bind(2) "address already in use" (2nd
    # _PORT_ALLOCATED_PHRASES entry, matched in isolation) takes the same retry path.
    assert len(docker_client.run_specs) == 2
    assert delays == [5]
    retry_records = [
        record for record in caplog.records
        if record.getMessage() == "PORT_ALREADY_ALLOCATED_RETRY"
    ]
    assert [r.msg.extra["port_allocation_phrase"] for r in retry_records] == [
        "address already in use"
    ]


@pytest.mark.asyncio
async def test_sdk_port_retry_third_phrase(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Arrange
    caplog.set_level(logging.INFO)
    svc = make_docker_service()
    docker_client = FakeRentalDockerClient()
    docker_client.run_errors = [Exception(_BIND_HOST_PORT_ERR)]
    delays = _patch_sleep(monkeypatch)

    # Act
    await svc._run_rental_docker_create_with_port_retry(
        docker_client=docker_client,
        ssh_client=Mock(),
        run_spec=_make_run_spec(),
        container_name="pod_test",
        default_extra={},
        log_tag="t",
    )

    # Assert
    # WHY: "failed to bind host port" (3rd _PORT_ALLOCATED_PHRASES entry, matched
    # in isolation) retries then succeeds -- untested even on the dead twin.
    assert len(docker_client.run_specs) == 2
    assert delays == [5]
    retry_records = [
        record for record in caplog.records
        if record.getMessage() == "PORT_ALREADY_ALLOCATED_RETRY"
    ]
    assert [r.msg.extra["port_allocation_phrase"] for r in retry_records] == [
        "failed to bind host port"
    ]


@pytest.mark.asyncio
async def test_sdk_port_rm_failure_continues(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Arrange
    caplog.set_level(logging.INFO)
    svc = make_docker_service()
    docker_client = FakeRentalDockerClient()
    docker_client.run_errors = [Exception(_PORT_ALLOCATED_ERR)]
    docker_client.remove_error = Exception("rm exploded")
    _patch_sleep(monkeypatch)

    # Act
    await svc._run_rental_docker_create_with_port_retry(
        docker_client=docker_client,
        ssh_client=Mock(),
        run_spec=_make_run_spec(),
        container_name="pod_test",
        default_extra={},
        log_tag="t",
    )

    # Assert
    # WHY: the stale-container rm is best-effort -- its failure warns
    # PORT_RETRY_STALE_RM_FAILED and the loop still re-runs to success.
    assert len(docker_client.run_specs) == 2
    assert len(docker_client.removed_containers) == 1
    rm_failed_records = [
        record for record in caplog.records
        if record.getMessage() == "PORT_RETRY_STALE_RM_FAILED"
    ]
    assert len(rm_failed_records) == 1
    assert rm_failed_records[0].levelno == logging.WARNING
    assert rm_failed_records[0].msg.extra["container_name"] == "pod_test"
    assert rm_failed_records[0].msg.extra["rm_error"] == "rm exploded"


@pytest.mark.asyncio
async def test_sdk_non_port_error_propagates(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Arrange
    caplog.set_level(logging.INFO)
    svc = make_docker_service()
    docker_client = FakeRentalDockerClient()
    docker_client.run_error = Exception(_NO_SUCH_IMAGE_ERR)
    repair = AsyncMock(return_value=True)
    monkeypatch.setattr(svc, "repair_stale_vloopback_mountpoint", repair)
    delays = _patch_sleep(monkeypatch)

    # Act (log_tag omitted on purpose: pins the signature default)
    with pytest.raises(Exception, match="No such image"):
        await svc._run_rental_docker_create_with_port_retry(
            docker_client=docker_client,
            ssh_client=Mock(),
            run_spec=_make_run_spec(),
            container_name="pod_test",
            default_extra={},
            local_volume="volume_test",
        )

    # Assert
    # WHY: an unrelated docker error must propagate on the first try -- no sleep,
    # no rm, no repair -- and (live-only, vs the bare-raise dead twin) be logged
    # with traceback and streamed under the default "container_creation" tag.
    assert len(docker_client.run_specs) == 1
    assert delays == []
    assert docker_client.removed_containers == []
    repair.assert_not_awaited()
    failed_records = [
        record for record in caplog.records
        if record.getMessage() == "Docker SDK run container failed"
    ]
    assert len(failed_records) == 1
    assert failed_records[0].levelno == logging.ERROR
    assert failed_records[0].exc_info is not None
    assert failed_records[0].msg.extra["container_name"] == "pod_test"
    assert failed_records[0].msg.extra["error"] == _NO_SUCH_IMAGE_ERR
    assert svc.logs_queue == [
        {"log_text": _NO_SUCH_IMAGE_ERR, "log_status": "error", "log_tag": "container_creation"}
    ]
