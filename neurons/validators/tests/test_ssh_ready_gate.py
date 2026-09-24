"""The SSH-ready gate: a pod is reported RUNNING only after its SSH port answers with an sshd banner."""

from __future__ import annotations

import asyncio
import functools
import socket
import time
from unittest.mock import AsyncMock, Mock

import pytest
import services.docker_service as ds_module
from payload_models.payloads import ContainerCreated, FailedContainerRequest, ProfilerStepName
from services.ssh_ready_gate import (
    SshNotReady,
    SshReadyMode,
    SshReadyOutcome,
    SshReadyResult,
    probe_ssh_banner,
    ssh_ready_gate_mode,
    wait_for_ssh_banner,
)
from test_deploy_optimizations import _executor_info, _patch_happy, _payload, _run, _ssh_client

BANNER = b"SSH-2.0-OpenSSH_9.6p1 Ubuntu-3ubuntu13\r\n"


async def _server(on_connect):
    async def _handle(reader, writer):
        try:
            await on_connect(reader, writer)
        finally:
            writer.close()

    server = await asyncio.start_server(_handle, "127.0.0.1", 0)
    return server, server.sockets[0].getsockname()[1]


def _closed_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds


def _probe_ready_after(
    clock: _FakeClock, seconds: float, before: SshReadyOutcome = SshReadyOutcome.REFUSED
):
    calls = []

    async def _probe(host, port, timeout):
        calls.append((host, port, timeout))
        return SshReadyOutcome.READY if clock.now >= seconds else before

    _probe.calls = calls
    return _probe


# ------------------------------------------------------------------
# One dial
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_probe_ready_when_sshd_answers_at_once():
    async def _banner(reader, writer):
        writer.write(BANNER)
        await writer.drain()

    server, port = await _server(_banner)
    async with server:
        assert await probe_ssh_banner("127.0.0.1", port, 1.0) is SshReadyOutcome.READY


@pytest.mark.asyncio
async def test_probe_skips_pre_banner_lines():
    async def _banner(reader, writer):
        writer.write(b"Authorized use only\r\n" + BANNER)
        await writer.drain()

    server, port = await _server(_banner)
    async with server:
        assert await probe_ssh_banner("127.0.0.1", port, 1.0) is SshReadyOutcome.READY


@pytest.mark.asyncio
async def test_probe_refused():
    assert await probe_ssh_banner("127.0.0.1", _closed_port(), 1.0) is SshReadyOutcome.REFUSED


@pytest.mark.asyncio
async def test_probe_accepts_but_sends_no_banner():
    async def _silent(reader, writer):
        await asyncio.sleep(2)

    server, port = await _server(_silent)
    async with server:
        assert await probe_ssh_banner("127.0.0.1", port, 0.3) is SshReadyOutcome.NO_BANNER


@pytest.mark.asyncio
async def test_probe_connect_and_banner_read_share_one_deadline(monkeypatch):
    async def _silent(reader, writer):
        await asyncio.sleep(2)

    server, port = await _server(_silent)
    real_open_connection = asyncio.open_connection

    async def _slow_connect(*args, **kwargs):
        await asyncio.sleep(0.3)
        return await real_open_connection(*args, **kwargs)

    monkeypatch.setattr(asyncio, "open_connection", _slow_connect)
    async with server:
        started = time.monotonic()
        assert await probe_ssh_banner("127.0.0.1", port, 0.4) is SshReadyOutcome.NO_BANNER
        assert time.monotonic() - started < 0.6


@pytest.mark.asyncio
async def test_probe_accept_then_close_is_no_banner():
    """docker-proxy accepts on the host port even when nothing listens in the container."""

    async def _close(reader, writer):
        return None

    server, port = await _server(_close)
    async with server:
        assert await probe_ssh_banner("127.0.0.1", port, 1.0) is SshReadyOutcome.NO_BANNER


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "line", [b"SSH-1.99-OpenSSH\r\n", b"SSH-2.0-\r\n", b"HTTP/1.1 400 Bad Request\r\n"]
)
async def test_probe_refuses_non_ssh2_lines(line):
    async def _send(reader, writer):
        writer.write(line)
        await writer.drain()

    server, port = await _server(_send)
    async with server:
        assert await probe_ssh_banner("127.0.0.1", port, 0.5) is SshReadyOutcome.NO_BANNER


# ------------------------------------------------------------------
# Grace period
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wait_passes_when_sshd_answers_after_10s():
    clock = _FakeClock()
    probe = _probe_ready_after(clock, 10)

    result = await wait_for_ssh_banner(
        "1.2.3.4",
        2222,
        grace_seconds=60,
        poll_seconds=2,
        probe=probe,
        clock=clock,
        sleep=clock.sleep,
    )

    assert result.ready
    assert result.attempts == 6
    assert result.elapsed_ms == 10_000
    assert {(host, port) for host, port, _ in probe.calls} == {("1.2.3.4", 2222)}


@pytest.mark.asyncio
async def test_wait_fails_with_last_outcome_when_refused_throughout():
    clock = _FakeClock()
    probe = _probe_ready_after(clock, float("inf"))

    result = await wait_for_ssh_banner(
        "1.2.3.4",
        2222,
        grace_seconds=60,
        poll_seconds=2,
        probe=probe,
        clock=clock,
        sleep=clock.sleep,
    )

    assert not result.ready
    assert result.outcome is SshReadyOutcome.REFUSED
    assert result.elapsed_ms == 60_000
    assert result.attempts == 30


@pytest.mark.asyncio
async def test_wait_never_runs_past_the_grace_even_when_every_dial_uses_its_whole_timeout():
    clock = _FakeClock()
    timeouts = []

    async def _slow(host, port, timeout):
        timeouts.append(timeout)
        clock.now += timeout
        return SshReadyOutcome.TIMED_OUT

    result = await wait_for_ssh_banner(
        "1.2.3.4",
        2222,
        grace_seconds=60,
        poll_seconds=2,
        probe=_slow,
        clock=clock,
        sleep=clock.sleep,
    )

    assert result.elapsed_ms == 60_000
    assert max(timeouts) == 5.0
    assert result.outcome is SshReadyOutcome.TIMED_OUT


@pytest.mark.asyncio
async def test_wait_stops_within_one_poll_of_a_cancel():
    clock = _FakeClock()
    cancelled_at = 9.0
    probe = _probe_ready_after(clock, float("inf"))

    result = await wait_for_ssh_banner(
        "1.2.3.4",
        2222,
        grace_seconds=60,
        poll_seconds=2,
        stop=lambda: clock.now >= cancelled_at,
        probe=probe,
        clock=clock,
        sleep=clock.sleep,
    )

    assert result.outcome is SshReadyOutcome.CANCELLED
    assert not result.ready
    assert cancelled_at <= clock.now <= cancelled_at + 2


@pytest.mark.asyncio
async def test_wait_cancelled_before_the_first_dial_dials_nothing():
    clock = _FakeClock()
    probe = _probe_ready_after(clock, 0)

    result = await wait_for_ssh_banner(
        "1.2.3.4",
        2222,
        grace_seconds=60,
        poll_seconds=2,
        stop=lambda: True,
        probe=probe,
        clock=clock,
    )

    assert result.outcome is SshReadyOutcome.CANCELLED and result.attempts == 0
    assert probe.calls == []


def test_not_ready_error_names_port_grace_and_outcome():
    result = SshReadyResult(outcome=SshReadyOutcome.REFUSED, attempts=31, elapsed_ms=60_000)
    assert str(SshNotReady(40022, 60.0, result)) == (
        "the pod's SSH port 40022 did not answer with an SSH banner within 60 s (connection refused)"
    )


@pytest.mark.parametrize(
    "raw, mode",
    [
        ("off", SshReadyMode.OFF),
        ("LOG", SshReadyMode.LOG),
        (" enforce ", SshReadyMode.ENFORCE),
        ("", SshReadyMode.OFF),
        (None, SshReadyMode.OFF),
        ("enforced", SshReadyMode.OFF),
    ],
)
def test_mode_parsing_defaults_to_off(raw, mode):
    assert ssh_ready_gate_mode(raw) is mode


# ------------------------------------------------------------------
# create_container wiring
# ------------------------------------------------------------------


@pytest.fixture
def svc():
    return ds_module.DockerService(
        ssh_service=Mock(), redis_service=Mock(), attestation_service=Mock()
    )


def _set_mode(monkeypatch, mode: str):
    monkeypatch.setattr(ds_module.settings, "SSH_READY_GATE_MODE", mode)
    monkeypatch.setattr(ds_module.settings, "SSH_READY_GATE_GRACE_SECONDS", 60.0)
    monkeypatch.setattr(ds_module.settings, "SSH_READY_GATE_POLL_SECONDS", 2.0)


def _patch_wait(monkeypatch, probe, clock):
    wait = AsyncMock(
        side_effect=functools.partial(
            wait_for_ssh_banner, probe=probe, clock=clock, sleep=clock.sleep
        )
    )
    monkeypatch.setattr(ds_module, "wait_for_ssh_banner", wait)
    return wait


def _gate_lines(mock_logger, level: str):
    return [
        c.args[0]
        for c in getattr(mock_logger, level).call_args_list
        if c.args and getattr(c.args[0], "message", None) == "SSH ready gate"
    ]


@pytest.mark.asyncio
async def test_mode_off_makes_no_dial(svc, monkeypatch):
    _patch_happy(svc, monkeypatch, _ssh_client())
    _set_mode(monkeypatch, "off")
    wait = AsyncMock()
    monkeypatch.setattr(ds_module, "wait_for_ssh_banner", wait)
    open_connection = AsyncMock()
    monkeypatch.setattr(asyncio, "open_connection", open_connection)

    result = await _run(svc, _payload(ships_sshd=True))

    assert isinstance(result, ContainerCreated)
    wait.assert_not_awaited()
    open_connection.assert_not_awaited()
    assert not ds_module._SSH_READY_LOG_TASKS
    assert ProfilerStepName.SSH_READY not in {p.name for p in result.profilers}


@pytest.mark.asyncio
async def test_enforce_passes_when_sshd_answers_at_once(svc, monkeypatch):
    _patch_happy(svc, monkeypatch, _ssh_client())
    _set_mode(monkeypatch, "enforce")
    clock = _FakeClock()
    probe = _probe_ready_after(clock, 0)
    _patch_wait(monkeypatch, probe, clock)

    result = await _run(svc, _payload(ships_sshd=True))

    assert isinstance(result, ContainerCreated)
    # the executor's public address and the external port mapped to the pod's port 22
    assert probe.calls[0][:2] == ("127.0.0.1", 20001)
    names = [p.name for p in result.profilers]
    assert names.index(ProfilerStepName.ADDING_PUBLIC_KEYS) < names.index(
        ProfilerStepName.SSH_READY
    )
    assert names.index(ProfilerStepName.SSH_READY) < names.index(
        ProfilerStepName.FINISHED_IN_SUBNET
    )


@pytest.mark.asyncio
async def test_enforce_passes_when_sshd_answers_after_10s(svc, monkeypatch):
    _patch_happy(svc, monkeypatch, _ssh_client())
    _set_mode(monkeypatch, "enforce")
    clock = _FakeClock()
    _patch_wait(monkeypatch, _probe_ready_after(clock, 10), clock)
    mock_logger = Mock()
    monkeypatch.setattr(ds_module, "logger", mock_logger)

    result = await _run(svc, _payload(ships_sshd=True))

    assert isinstance(result, ContainerCreated)
    (line,) = _gate_lines(mock_logger, "info")
    assert line.extra["ssh_ready"] is True
    assert line.extra["ssh_ready_duration_ms"] == 10_000


@pytest.mark.asyncio
async def test_enforce_fails_at_ssh_ready_when_refused_throughout(svc, monkeypatch):
    _patch_happy(svc, monkeypatch, _ssh_client())
    _set_mode(monkeypatch, "enforce")
    clock = _FakeClock()
    _patch_wait(monkeypatch, _probe_ready_after(clock, float("inf")), clock)
    cleanup = AsyncMock(return_value=False)
    monkeypatch.setattr(svc, "cleanup_failed_container_creation", cleanup)

    result = await _run(svc, _payload(ships_sshd=True))

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "ssh_ready"
    assert (
        "the pod's SSH port 20001 did not answer with an SSH banner within 60 s (connection refused)"
        in (result.detail)
    )
    cleanup.assert_awaited_once()
    svc.redis_service.remove_pending_pod.assert_awaited()


@pytest.mark.asyncio
async def test_enforce_fails_when_port_accepts_but_sends_no_banner(svc, monkeypatch):
    _patch_happy(svc, monkeypatch, _ssh_client())
    _set_mode(monkeypatch, "enforce")
    clock = _FakeClock()
    _patch_wait(
        monkeypatch, _probe_ready_after(clock, float("inf"), SshReadyOutcome.NO_BANNER), clock
    )

    result = await _run(svc, _payload(ships_sshd=True))

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "ssh_ready"
    assert "(no banner)" in result.detail


@pytest.mark.asyncio
async def test_a_delete_during_the_grace_ends_the_create_as_cancelled_by_delete(svc, monkeypatch):
    """Review (lium-io#1467): a renter's delete mid-grace must end the create as the renter's cancel, within
    one poll, not as an `ssh_ready` failure the platform would count against the node."""
    _patch_happy(svc, monkeypatch, _ssh_client())
    _set_mode(monkeypatch, "enforce")
    clock = _FakeClock()
    payload = _payload(ships_sshd=True)
    deleted_at = 9.0

    async def _probe(host, port, timeout):
        if clock.now >= deleted_at:
            ds_module.inflight_creates.cancel(payload.pod_id)
        return SshReadyOutcome.REFUSED

    _patch_wait(monkeypatch, _probe, clock)

    with ds_module.inflight_creates.track(payload.pod_id):
        result = await _run(svc, payload)

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "cancelled_by_delete"
    assert "did not answer with an SSH banner" not in (result.detail or "")
    assert clock.now <= deleted_at + 2


@pytest.mark.asyncio
async def test_enforce_gates_the_validator_bootstrap_path_too(svc, monkeypatch):
    """Every create is gated, not only images that start their own sshd."""
    _patch_happy(svc, monkeypatch, _ssh_client())
    _set_mode(monkeypatch, "enforce")
    clock = _FakeClock()
    _patch_wait(monkeypatch, _probe_ready_after(clock, float("inf")), clock)

    result = await _run(svc, _payload(ships_sshd=None))

    svc.install_open_ssh_server_and_start_ssh_service_with_rental_docker.assert_awaited_once()
    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "ssh_ready"


@pytest.mark.asyncio
async def test_log_mode_passes_and_logs_when_refused_throughout(svc, monkeypatch):
    _patch_happy(svc, monkeypatch, _ssh_client())
    _set_mode(monkeypatch, "log")
    clock = _FakeClock()
    _patch_wait(monkeypatch, _probe_ready_after(clock, float("inf")), clock)
    mock_logger = Mock()
    monkeypatch.setattr(ds_module, "logger", mock_logger)
    payload = _payload(ships_sshd=True)

    result = await _run(svc, payload)

    assert isinstance(result, ContainerCreated)
    assert ProfilerStepName.SSH_READY not in {p.name for p in result.profilers}
    await asyncio.gather(*list(ds_module._SSH_READY_LOG_TASKS))
    (line,) = _gate_lines(mock_logger, "warning")
    assert line.extra["pod_id"] == payload.pod_id
    assert line.extra["ssh_ready_mode"] == "log"
    assert line.extra["ssh_ready"] is False
    assert line.extra["ssh_ready_result"] == "connection refused"
    assert line.extra["ssh_ready_duration_ms"] == 60_000
    assert line.extra["image_manages_services"] is True


@pytest.mark.asyncio
async def test_log_mode_probe_error_never_escapes(svc, monkeypatch):
    _patch_happy(svc, monkeypatch, _ssh_client())
    _set_mode(monkeypatch, "log")
    monkeypatch.setattr(
        ds_module, "wait_for_ssh_banner", AsyncMock(side_effect=RuntimeError("boom"))
    )
    mock_logger = Mock()
    monkeypatch.setattr(ds_module, "logger", mock_logger)

    result = await _run(svc, _payload())

    assert isinstance(result, ContainerCreated)
    await asyncio.gather(*list(ds_module._SSH_READY_LOG_TASKS))
    assert any(
        getattr(c.args[0], "message", None) == "SSH ready gate probe errored"
        for c in mock_logger.warning.call_args_list
        if c.args
    )


@pytest.mark.asyncio
async def test_the_rental_probes_create_is_not_gated_in_enforce(svc, monkeypatch):
    _patch_happy(svc, monkeypatch, _ssh_client())
    _set_mode(monkeypatch, "enforce")
    wait = AsyncMock()
    monkeypatch.setattr(ds_module, "wait_for_ssh_banner", wait)
    mock_logger = Mock()
    monkeypatch.setattr(ds_module, "logger", mock_logger)
    payload = _payload(ships_sshd=True)

    result = await svc.create_container(
        payload=payload,
        executor_info=_executor_info(payload),
        keypair=Mock(ss58_address="validator-hotkey"),
        private_key="encrypted",
        ssh_ready_gate=False,
    )

    assert isinstance(result, ContainerCreated)
    wait.assert_not_awaited()
    assert not ds_module._SSH_READY_LOG_TASKS
    assert _gate_lines(mock_logger, "info") == [] and _gate_lines(mock_logger, "warning") == []


@pytest.mark.asyncio
async def test_the_rental_probes_create_writes_no_log_mode_line(svc, monkeypatch):
    _patch_happy(svc, monkeypatch, _ssh_client())
    _set_mode(monkeypatch, "log")
    wait = AsyncMock()
    monkeypatch.setattr(ds_module, "wait_for_ssh_banner", wait)
    payload = _payload(ships_sshd=True)

    result = await svc.create_container(
        payload=payload,
        executor_info=_executor_info(payload),
        keypair=Mock(ss58_address="validator-hotkey"),
        private_key="encrypted",
        ssh_ready_gate=False,
    )

    assert isinstance(result, ContainerCreated)
    assert not ds_module._SSH_READY_LOG_TASKS
    wait.assert_not_awaited()
