"""DAH-2870: a RUNNING rented pod that refuses its renter after a host reboot.

Regression under test: ticket-0326 (SSH port refuses after the host rebooted, pod RUNNING, score 1,
billing on) and ticket-0247 (port open, authorized_keys unreadable because the encrypted volume was
not remounted). Before this change TenantEnforcementCheck reported ALREADY_RENTED for both.
"""

from __future__ import annotations

import asyncio
import json
import socket
from unittest.mock import AsyncMock, patch

import pytest
from helpers import FakeRedis, build_context_config, build_services, build_state
from neurons.validators.src.services.task.checks import rented_pod_ssh
from neurons.validators.src.services.task.checks.rented_machine import TenantEnforcementCheck
from neurons.validators.src.services.task.checks.rented_pod_ssh import (
    FAULT_AUTHORIZED_KEYS_UNREADABLE,
    FAULT_TCP_REFUSED,
    FAULT_TCP_TIMEOUT,
    tcp_connect_fault,
)
from neurons.validators.src.services.task.messages import TenantEnforcementMessages as Msg
from test_rented_machine_check import DummyScoreCalculator, DummySSHClient, MockContainerCleanup

from protocol.vc_protocol.compute_requests import (
    PodSshUnreachableResponse,
    RentedExecutor,
    RentedExecutorsResponse,
    RentedPod,
)

EXECUTOR_UUID = "executor-123"
POD_ID = "pod-1"
SSH_PORT = 40199
SETTINGS_PATH = "neurons.validators.src.services.task.checks.rented_pod_ssh.settings"
TCP_PATH = "neurons.validators.src.services.task.checks.rented_pod_ssh.tcp_connect_fault"


def rented_data(ssh_port: int | None = SSH_PORT) -> RentedExecutorsResponse:
    return RentedExecutorsResponse(
        executors={
            EXECUTOR_UUID: RentedExecutor(
                miner_hotkey="test-miner",
                executor_ip_address="127.0.0.1",
                executor_ip_port="9001",
                pods=[RentedPod(pod_id=POD_ID, container_name="pod_1", ssh_port=ssh_port)],
            )
        }
    )


class Harness:
    """One validator with one Redis across cycles; each cycle builds a fresh context."""

    def __init__(self, context_factory, *, ssh_port: int | None = SSH_PORT):
        self.context_factory = context_factory
        self.redis = FakeRedis()
        self.backend = AsyncMock()
        self.backend.report_pod_ssh_unreachable.return_value = PodSshUnreachableResponse(recorded=True)
        self.ssh_port = ssh_port
        self.score_calculator = DummyScoreCalculator(actual_score=0.9, job_score=0.9)

    async def cycle(self, *, tcp_fault: str | None, ssh_keys: list[str], boot_id: str = "boot-a"):
        services = build_services(
            redis=self.redis,
            backend=self.backend,
            score_calculator=self.score_calculator,
            container_cleanup=MockContainerCleanup(),
        )
        ctx = self.context_factory(
            services=services,
            config=build_context_config(),
            state=build_state(rented_data=rented_data(self.ssh_port), specs={"boot_id": boot_id}),
            ssh=DummySSHClient(pod_running=True, ssh_keys=ssh_keys),
            collateral_deposited=True,
        )
        with patch(TCP_PATH, new=AsyncMock(return_value=tcp_fault)) as tcp:
            result = await TenantEnforcementCheck().run(ctx)
        self.tcp_calls = tcp.await_args_list
        return result

    def streak(self) -> dict | None:
        raw = self.redis.store.get(f"{rented_pod_ssh.RENTED_POD_SSH_FAIL_KEY_PREFIX}:{POD_ID}")
        return json.loads(raw) if raw else None


KEYS = ["ssh-ed25519 AAAA renter"]


@pytest.mark.asyncio
async def test_one_refused_cycle_after_a_healthy_one_counts_but_stays_quiet(context_factory):
    # A single refused connect is a blip (conntrack flush, sshd restart); it must not raise the event.
    h = Harness(context_factory)
    await h.cycle(tcp_fault=None, ssh_keys=KEYS)
    result = await h.cycle(tcp_fault=FAULT_TCP_REFUSED, ssh_keys=KEYS)

    assert result.event.reason_code == Msg.ALREADY_RENTED.reason
    assert h.streak()["count"] == 1
    h.backend.report_pod_ssh_unreachable.assert_not_awaited()


@pytest.mark.asyncio
async def test_second_consecutive_refused_cycle_raises_the_event_and_tells_the_backend_once(context_factory):
    # ticket-0326: host rebooted (boot_id changed), container back, port 40199 refuses, pod RUNNING.
    h = Harness(context_factory)
    await h.cycle(tcp_fault=None, ssh_keys=KEYS, boot_id="boot-a")
    await h.cycle(tcp_fault=FAULT_TCP_REFUSED, ssh_keys=KEYS, boot_id="boot-b")
    first_failed_at = h.streak()["first_failed_at"]
    result = await h.cycle(tcp_fault=FAULT_TCP_REFUSED, ssh_keys=KEYS, boot_id="boot-b")

    assert result.passed is True and result.halt is True
    assert result.event.reason_code == Msg.RENTED_POD_SSH_UNREACHABLE.reason
    assert result.event.severity == "error"
    # Score is the rented score: this verdict is reported, not scored (Rustam's call).
    assert result.updates["score"] == 0.9 and result.updates["job_score"] == 0.9
    assert "clear_verified_job_info" not in result.updates
    [pod] = result.event.what_we_saw["unreachable_pods"]
    assert pod["pod_id"] == POD_ID and pod["ssh_port"] == SSH_PORT
    assert pod["faults"] == [FAULT_TCP_REFUSED]
    assert pod["consecutive_cycles"] == 2 and pod["first_failed_at"] == first_failed_at
    assert pod["boot_id_changed"] is True
    assert pod["reported_to_backend"] is True and pod["backend_recorded"] is True
    h.backend.report_pod_ssh_unreachable.assert_awaited_once_with(
        POD_ID,
        ssh_port=SSH_PORT,
        faults=[FAULT_TCP_REFUSED],
        first_failed_at=first_failed_at,
        consecutive_cycles=2,
        boot_id_changed=True,
        boot_id_at_ok="boot-a",
        boot_id_now="boot-b",
    )
    # The TCP probe went to the executor's address on the pod's mapped port.
    assert h.tcp_calls[0].args[:2] == ("127.0.0.1", SSH_PORT)


@pytest.mark.asyncio
async def test_third_cycle_of_the_same_outage_keeps_the_event_but_does_not_post_again(context_factory):
    # One POST per outage: the backend dedupes too, but the validator must not spam it every 15 min.
    h = Harness(context_factory)
    await h.cycle(tcp_fault=None, ssh_keys=KEYS)
    await h.cycle(tcp_fault=FAULT_TCP_REFUSED, ssh_keys=KEYS)
    await h.cycle(tcp_fault=FAULT_TCP_REFUSED, ssh_keys=KEYS)
    result = await h.cycle(tcp_fault=FAULT_TCP_TIMEOUT, ssh_keys=KEYS)

    assert result.event.reason_code == Msg.RENTED_POD_SSH_UNREACHABLE.reason
    [pod] = result.event.what_we_saw["unreachable_pods"]
    assert pod["consecutive_cycles"] == 3 and pod["reported_to_backend"] is False
    assert h.backend.report_pod_ssh_unreachable.await_count == 1


@pytest.mark.asyncio
async def test_a_pod_never_seen_healthy_is_not_counted(context_factory):
    # A template without sshd, or a pod still coming up, refuses from the start: not this outage.
    h = Harness(context_factory)
    for _ in range(3):
        result = await h.cycle(tcp_fault=FAULT_TCP_REFUSED, ssh_keys=[])

    assert result.event.reason_code == Msg.ALREADY_RENTED.reason
    assert h.streak() is None
    h.backend.report_pod_ssh_unreachable.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_healthy_cycle_resets_the_streak(context_factory):
    # refused, healthy, refused is two blips, not one 30-minute outage.
    h = Harness(context_factory)
    await h.cycle(tcp_fault=None, ssh_keys=KEYS)
    await h.cycle(tcp_fault=FAULT_TCP_REFUSED, ssh_keys=KEYS)
    await h.cycle(tcp_fault=None, ssh_keys=KEYS)
    result = await h.cycle(tcp_fault=FAULT_TCP_REFUSED, ssh_keys=KEYS)

    assert result.event.reason_code == Msg.ALREADY_RENTED.reason
    assert h.streak()["count"] == 1
    h.backend.report_pod_ssh_unreachable.assert_not_awaited()


@pytest.mark.asyncio
async def test_open_port_with_unreadable_authorized_keys_is_the_ticket_0247_fault(context_factory):
    # sshd up, /root owned by nobody:nogroup, no authorized_keys: the renter gets a password prompt.
    h = Harness(context_factory)
    await h.cycle(tcp_fault=None, ssh_keys=KEYS)
    await h.cycle(tcp_fault=None, ssh_keys=[])
    result = await h.cycle(tcp_fault=None, ssh_keys=[])

    assert result.event.reason_code == Msg.RENTED_POD_SSH_UNREACHABLE.reason
    [pod] = result.event.what_we_saw["unreachable_pods"]
    assert pod["faults"] == [FAULT_AUTHORIZED_KEYS_UNREADABLE]
    assert pod["boot_id_changed"] is False
    assert h.backend.report_pod_ssh_unreachable.await_args.kwargs["faults"] == [FAULT_AUTHORIZED_KEYS_UNREADABLE]


@pytest.mark.asyncio
async def test_backend_without_the_field_sends_no_ssh_port_and_the_keys_alone_decide(context_factory):
    # An older backend omits ssh_port: no TCP connect is attempted, authorized_keys still judged.
    h = Harness(context_factory, ssh_port=None)
    result = await h.cycle(tcp_fault=FAULT_TCP_REFUSED, ssh_keys=KEYS)

    assert h.tcp_calls == []
    assert result.event.reason_code == Msg.ALREADY_RENTED.reason
    assert json.loads(h.redis.store[f"{rented_pod_ssh.RENTED_POD_SSH_OK_KEY_PREFIX}:{POD_ID}"])["boot_id"] == "boot-a"


@pytest.mark.asyncio
async def test_backend_error_on_the_report_does_not_fail_the_cycle(context_factory):
    # The event is the record of truth; a backend outage must not turn it into a validator failure.
    h = Harness(context_factory)
    h.backend.report_pod_ssh_unreachable.side_effect = RuntimeError("backend down")
    await h.cycle(tcp_fault=None, ssh_keys=KEYS)
    await h.cycle(tcp_fault=FAULT_TCP_REFUSED, ssh_keys=KEYS)
    result = await h.cycle(tcp_fault=FAULT_TCP_REFUSED, ssh_keys=KEYS)

    assert result.passed is True
    assert result.event.reason_code == Msg.RENTED_POD_SSH_UNREACHABLE.reason
    [pod] = result.event.what_we_saw["unreachable_pods"]
    assert pod["reported_to_backend"] is True and pod["backend_recorded"] is None


@pytest.mark.asyncio
async def test_probe_disabled_leaves_the_check_as_before(context_factory):
    h = Harness(context_factory)
    with patch(SETTINGS_PATH) as settings:
        settings.RENTED_POD_SSH_PROBE_ENABLED = False
        result = await h.cycle(tcp_fault=FAULT_TCP_REFUSED, ssh_keys=[])

    assert result.event.reason_code == Msg.ALREADY_RENTED.reason
    assert h.redis.store == {}
    assert h.tcp_calls == []


@pytest.mark.asyncio
async def test_tcp_connect_fault_tells_refused_from_timeout_from_open():
    # A bound-then-closed local port refuses; a listening one accepts; a hang is a timeout.
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    closed_port = probe.getsockname()[1]
    probe.close()
    assert await tcp_connect_fault("127.0.0.1", closed_port, timeout=2.0) == FAULT_TCP_REFUSED

    server = await asyncio.start_server(lambda r, w: w.close(), "127.0.0.1", 0)
    open_port = server.sockets[0].getsockname()[1]
    try:
        assert await tcp_connect_fault("127.0.0.1", open_port, timeout=2.0) is None
    finally:
        server.close()
        await server.wait_closed()

    async def hang(*_args, **_kwargs):
        await asyncio.sleep(10)

    with patch("asyncio.open_connection", new=hang):
        assert await tcp_connect_fault("127.0.0.1", open_port, timeout=0.05) == FAULT_TCP_TIMEOUT
