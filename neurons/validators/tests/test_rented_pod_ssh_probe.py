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
    FAULT_SSH_BANNER_MISSING,
    FAULT_TCP_REFUSED,
    FAULT_TCP_TIMEOUT,
    is_ssh2_identification,
    tcp_connect_fault,
)
from neurons.validators.src.services.task.messages import TenantEnforcementMessages as Msg
from protocol.vc_protocol.compute_requests import (
    PodSshUnreachableResponse,
    RentedExecutor,
    RentedExecutorsResponse,
    RentedPod,
)
from test_rented_machine_check import DummyScoreCalculator, DummySSHClient, MockContainerCleanup

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
        self.backend.report_pod_ssh_unreachable.return_value = PodSshUnreachableResponse(
            recorded=True
        )
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
async def test_second_consecutive_refused_cycle_raises_the_event_and_tells_the_backend_once(
    context_factory,
):
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
async def test_third_cycle_of_the_same_outage_keeps_the_event_but_does_not_post_again(
    context_factory,
):
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
    assert h.backend.report_pod_ssh_unreachable.await_args.kwargs["faults"] == [
        FAULT_AUTHORIZED_KEYS_UNREADABLE
    ]


@pytest.mark.asyncio
async def test_backend_without_the_field_sends_no_ssh_port_and_the_keys_alone_decide(
    context_factory,
):
    # An older backend omits ssh_port: no TCP connect is attempted, authorized_keys still judged.
    h = Harness(context_factory, ssh_port=None)
    result = await h.cycle(tcp_fault=FAULT_TCP_REFUSED, ssh_keys=KEYS)

    assert h.tcp_calls == []
    assert result.event.reason_code == Msg.ALREADY_RENTED.reason
    assert (
        json.loads(h.redis.store[f"{rented_pod_ssh.RENTED_POD_SSH_OK_KEY_PREFIX}:{POD_ID}"])[
            "boot_id"
        ]
        == "boot-a"
    )


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
    assert h.streak()["reported"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("no_answer", ["raises", "non_200"])
async def test_a_report_the_backend_did_not_answer_is_posted_again_until_it_does(
    context_factory, no_answer
):
    # Rustam's review (16 Sep): the POST went out on the threshold cycle only, so a backend that was
    # down (or a 404 from one too old: the client returns None on any non-200) for that one cycle
    # never heard of the outage. The backend's 200 is now kept in the streak (`reported`); no
    # answer means the next cycle posts again.
    h = Harness(context_factory)
    if no_answer == "raises":
        h.backend.report_pod_ssh_unreachable.side_effect = RuntimeError("backend down")
    else:
        h.backend.report_pod_ssh_unreachable.return_value = None
    await h.cycle(tcp_fault=None, ssh_keys=KEYS)
    await h.cycle(tcp_fault=FAULT_TCP_REFUSED, ssh_keys=KEYS)
    await h.cycle(tcp_fault=FAULT_TCP_REFUSED, ssh_keys=KEYS)  # threshold: POST, no answer
    assert h.streak()["reported"] is False
    h.backend.report_pod_ssh_unreachable.side_effect = None  # backend back, answers recorded=True
    h.backend.report_pod_ssh_unreachable.return_value = PodSshUnreachableResponse(recorded=True)
    recovered = await h.cycle(tcp_fault=FAULT_TCP_REFUSED, ssh_keys=KEYS)
    after = await h.cycle(tcp_fault=FAULT_TCP_REFUSED, ssh_keys=KEYS)

    [pod] = recovered.event.what_we_saw["unreachable_pods"]
    assert pod["consecutive_cycles"] == 3
    assert pod["reported_to_backend"] is True and pod["backend_recorded"] is True
    assert h.streak() == {
        "count": 4,
        "first_failed_at": h.streak()["first_failed_at"],
        "reported": True,
    }
    [pod] = after.event.what_we_saw["unreachable_pods"]
    assert pod["reported_to_backend"] is False
    assert h.backend.report_pod_ssh_unreachable.await_count == 2  # the threshold cycle and the next


@pytest.mark.asyncio
async def test_redis_down_skips_the_probe_and_leaves_the_rented_verdict_alone(context_factory):
    # Regression (Rustam, #1372): the first Redis call in a fatal check was unguarded, so a Redis
    # outage raised out of the check and failed validation on every rented node. Redis is an input
    # to the signal, not to the verdict: with Redis down the cycle is RENTED at the rented score,
    # nothing is counted, and the backend is never told.
    h = Harness(context_factory)
    h.redis.failing = True
    for _ in range(3):
        result = await h.cycle(tcp_fault=FAULT_TCP_REFUSED, ssh_keys=[])

    assert result.passed is True and result.halt is True
    assert result.event.reason_code == Msg.ALREADY_RENTED.reason
    assert result.updates["score"] == 0.9 and result.updates["job_score"] == 0.9
    assert h.redis.calls > 0 and h.redis.store == {}
    h.backend.report_pod_ssh_unreachable.assert_not_awaited()


@pytest.mark.asyncio
async def test_redis_down_mid_outage_does_not_report_and_the_streak_resumes_after(context_factory):
    # Redis down for the whole threshold cycle: no verdict, nothing written; the next cycle with
    # Redis back reads the streak the earlier cycles left and reports as usual.
    h = Harness(context_factory)
    await h.cycle(tcp_fault=None, ssh_keys=KEYS)
    await h.cycle(tcp_fault=FAULT_TCP_REFUSED, ssh_keys=KEYS)
    h.redis.failing = True
    blip = await h.cycle(tcp_fault=FAULT_TCP_REFUSED, ssh_keys=KEYS)
    h.redis.failing = False
    result = await h.cycle(tcp_fault=FAULT_TCP_REFUSED, ssh_keys=KEYS)

    assert blip.event.reason_code == Msg.ALREADY_RENTED.reason
    assert result.event.reason_code == Msg.RENTED_POD_SSH_UNREACHABLE.reason
    assert h.streak()["count"] == 2
    h.backend.report_pod_ssh_unreachable.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_write", ["ok", "fail"])
async def test_a_redis_error_on_either_write_of_the_threshold_cycle_still_posts_once(
    context_factory, failing_write
):
    # Regression (fresh review of #1372): the count was written before the ok mark was renewed, so
    # a Redis error on the renewal left `count == threshold` stored with no POST; the next cycle
    # read count 3, `consecutive != threshold`, and the backend was never told for that outage.
    # The count is now the last write before the report decision: whichever write fails, the
    # cycle after the blip is the one that reaches the threshold and POSTs.
    h = Harness(context_factory)
    await h.cycle(tcp_fault=None, ssh_keys=KEYS)
    await h.cycle(tcp_fault=FAULT_TCP_REFUSED, ssh_keys=KEYS)
    prefix = {
        "ok": rented_pod_ssh.RENTED_POD_SSH_OK_KEY_PREFIX,
        "fail": rented_pod_ssh.RENTED_POD_SSH_FAIL_KEY_PREFIX,
    }[failing_write]
    h.redis.fail_next_set_of.add(f"{prefix}:{POD_ID}")
    blip = await h.cycle(tcp_fault=FAULT_TCP_REFUSED, ssh_keys=KEYS)
    result = await h.cycle(tcp_fault=FAULT_TCP_REFUSED, ssh_keys=KEYS)

    assert blip.event.reason_code == Msg.ALREADY_RENTED.reason
    assert h.streak()["count"] == 2
    assert result.event.reason_code == Msg.RENTED_POD_SSH_UNREACHABLE.reason
    [pod] = result.event.what_we_saw["unreachable_pods"]
    assert pod["consecutive_cycles"] == 2 and pod["reported_to_backend"] is True
    h.backend.report_pod_ssh_unreachable.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_redis_blip_on_the_reported_mark_keeps_the_verdict_and_costs_one_more_post(
    context_factory,
):
    # Fresh review of round 3: the `reported` write comes after a 200; a Redis error there must not
    # drop the verdict the POST already went out for. It is caught on its own, the event still says
    # reported, `reported` stays False, and the next cycle posts once more (the backend dedupes).
    h = Harness(context_factory)
    await h.cycle(tcp_fault=None, ssh_keys=KEYS)
    await h.cycle(tcp_fault=FAULT_TCP_REFUSED, ssh_keys=KEYS)
    # the threshold cycle writes the fail key twice: the streak, then the reported mark
    h.redis.fail_set_of_after[f"{rented_pod_ssh.RENTED_POD_SSH_FAIL_KEY_PREFIX}:{POD_ID}"] = 1
    threshold = await h.cycle(tcp_fault=FAULT_TCP_REFUSED, ssh_keys=KEYS)
    after = await h.cycle(tcp_fault=FAULT_TCP_REFUSED, ssh_keys=KEYS)

    assert threshold.event.reason_code == Msg.RENTED_POD_SSH_UNREACHABLE.reason
    [pod] = threshold.event.what_we_saw["unreachable_pods"]
    assert pod["reported_to_backend"] is True and pod["backend_recorded"] is True
    [pod] = after.event.what_we_saw["unreachable_pods"]
    assert pod["reported_to_backend"] is True
    assert h.streak()["reported"] is True
    assert h.backend.report_pod_ssh_unreachable.await_count == 2


@pytest.mark.asyncio
async def test_both_marks_carry_the_ttl_and_every_probe_renews_it(context_factory):
    # Regression (Rustam, #1372): the marks were set without an expiry, so Redis kept one key pair
    # per pod for ever. Every write now carries RENTED_POD_SSH_PROBE_STATE_TTL_SECONDS, and an
    # unhealthy cycle renews the ok mark too, so a long outage keeps naming the pod.
    h = Harness(context_factory)
    ok_key = f"{rented_pod_ssh.RENTED_POD_SSH_OK_KEY_PREFIX}:{POD_ID}"
    fail_key = f"{rented_pod_ssh.RENTED_POD_SSH_FAIL_KEY_PREFIX}:{POD_ID}"
    with patch.object(rented_pod_ssh.settings, "RENTED_POD_SSH_PROBE_STATE_TTL_SECONDS", 3600):
        await h.cycle(tcp_fault=None, ssh_keys=KEYS)
        assert h.redis.ttl == {ok_key: 3600}
        h.redis.ttl.clear()
        await h.cycle(tcp_fault=FAULT_TCP_REFUSED, ssh_keys=KEYS)

    assert h.redis.ttl == {fail_key: 3600, ok_key: 3600}
    assert json.loads(h.redis.store[ok_key])["boot_id"] == "boot-a"


@pytest.mark.asyncio
async def test_a_closed_rental_deletes_both_marks(context_factory):
    # The pod's container is gone and the backend says the rental closed: the marks go with it
    # instead of waiting out the TTL.
    from test_rented_machine_check import DummyBackendClient

    redis = FakeRedis()
    ok_key = f"{rented_pod_ssh.RENTED_POD_SSH_OK_KEY_PREFIX}:{POD_ID}"
    fail_key = f"{rented_pod_ssh.RENTED_POD_SSH_FAIL_KEY_PREFIX}:{POD_ID}"
    redis.store[ok_key] = json.dumps({"at": "2026-09-14T11:00:00+00:00", "boot_id": "boot-a"})
    redis.store[fail_key] = json.dumps({"count": 1, "first_failed_at": "2026-09-14T11:15:00+00:00"})
    redis.store["rented_pod_ssh_ok:other-pod"] = json.dumps({"at": "x", "boot_id": "boot-a"})
    services = build_services(
        redis=redis,
        backend=DummyBackendClient(active=False),
        score_calculator=DummyScoreCalculator(actual_score=1.0, job_score=1.0),
        container_cleanup=MockContainerCleanup(),
    )
    ctx = context_factory(
        services=services,
        config=build_context_config(),
        state=build_state(rented_data=rented_data(), specs={"boot_id": "boot-b"}),
        ssh=DummySSHClient(pod_running=False, ssh_keys=[]),
        collateral_deposited=True,
    )
    result = await TenantEnforcementCheck().run(ctx)

    assert result.event.reason_code == Msg.STALE_POD_NOT_RUNNING.reason
    assert set(redis.store) == {"rented_pod_ssh_ok:other-pod"}


@pytest.mark.asyncio
async def test_a_closed_rental_with_redis_down_still_ends_as_stale_pod(context_factory):
    # The delete is a courtesy; the TTL does the same later. Redis down here changes nothing.
    from test_rented_machine_check import DummyBackendClient

    services = build_services(
        redis=FakeRedis(failing=True),
        backend=DummyBackendClient(active=False),
        score_calculator=DummyScoreCalculator(actual_score=1.0, job_score=1.0),
        container_cleanup=MockContainerCleanup(),
    )
    ctx = context_factory(
        services=services,
        config=build_context_config(),
        state=build_state(rented_data=rented_data(), specs={"boot_id": "boot-b"}),
        ssh=DummySSHClient(pod_running=False, ssh_keys=[]),
        collateral_deposited=True,
    )
    result = await TenantEnforcementCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.STALE_POD_NOT_RUNNING.reason


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
async def test_a_pod_the_recovery_path_just_restarted_is_not_probed_that_cycle(context_factory):
    # The container was down, _recover_downed_pod brought it back: judged from the renter's side
    # next cycle, not on the way up. A probe here would count a fail against a pod still starting.
    from test_rented_machine_check import DummyBackendClient

    ssh = DummySSHClient(pod_running=False, ssh_keys=[])
    docker = AsyncMock()

    async def bring_pod_back_up(**kwargs):
        ssh.pod_running = True
        return True

    docker.recover_pod_after_stale_vloopback_mount.side_effect = bring_pod_back_up
    redis = FakeRedis()
    redis.store[f"{rented_pod_ssh.RENTED_POD_SSH_OK_KEY_PREFIX}:{POD_ID}"] = json.dumps(
        {"at": "2026-09-14T11:00:00+00:00", "boot_id": "boot-a"}
    )
    services = build_services(
        redis=redis,
        backend=DummyBackendClient(active=True),
        score_calculator=DummyScoreCalculator(actual_score=1.0, job_score=1.0),
        container_cleanup=MockContainerCleanup(),
        pod_recovery=docker,
    )
    ctx = context_factory(
        services=services,
        config=build_context_config(),
        state=build_state(rented_data=rented_data(), specs={"boot_id": "boot-b"}),
        ssh=ssh,
        executor_ssh_private_key="ssh-key",
        collateral_deposited=True,
        is_rental_succeed=True,
    )
    with patch(TCP_PATH, new=AsyncMock(return_value=FAULT_TCP_REFUSED)) as tcp:
        result = await TenantEnforcementCheck().run(ctx)

    assert result.event.reason_code == Msg.ALREADY_RENTED.reason
    assert result.updates["default_extra"]["recovered_pods"] == ["pod_1"]
    assert tcp.await_args_list == []
    assert f"{rented_pod_ssh.RENTED_POD_SSH_FAIL_KEY_PREFIX}:{POD_ID}" not in redis.store


@pytest.mark.asyncio
async def test_dry_run_logs_the_event_but_does_not_tell_the_backend(context_factory):
    # DRY_RUN validates without publishing: a dry-run validator must not make the backend mail a renter.
    h = Harness(context_factory)
    await h.cycle(tcp_fault=None, ssh_keys=KEYS)
    await h.cycle(tcp_fault=FAULT_TCP_REFUSED, ssh_keys=KEYS)
    with patch.object(rented_pod_ssh.settings, "DRY_RUN", True):
        result = await h.cycle(tcp_fault=FAULT_TCP_REFUSED, ssh_keys=KEYS)

    assert result.event.reason_code == Msg.RENTED_POD_SSH_UNREACHABLE.reason
    h.backend.report_pod_ssh_unreachable.assert_not_awaited()

    # Rustam's review (16 Sep): the dry-run cycle counted on the same keys, so before the `reported`
    # flag a validator switched live mid-outage read count 3 != threshold and never posted.
    result = await h.cycle(tcp_fault=FAULT_TCP_REFUSED, ssh_keys=KEYS)
    [pod] = result.event.what_we_saw["unreachable_pods"]
    assert pod["consecutive_cycles"] == 3 and pod["reported_to_backend"] is True
    h.backend.report_pod_ssh_unreachable.assert_awaited_once()


@pytest.mark.asyncio
async def test_tcp_connect_fault_tells_refused_from_timeout_from_open():
    # A bound-then-closed local port refuses; a listening one accepts; a hang is a timeout.
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    closed_port = probe.getsockname()[1]
    probe.close()
    assert await tcp_connect_fault("127.0.0.1", closed_port, timeout=2.0) == FAULT_TCP_REFUSED

    # Rustam's review (16 Sep): an accept-then-close is what docker-proxy does on the host while
    # sshd inside the container is down (ticket-0326), so it is a fault, not health.
    server = await asyncio.start_server(lambda r, w: w.close(), "127.0.0.1", 0)
    open_port = server.sockets[0].getsockname()[1]
    try:
        assert (
            await tcp_connect_fault("127.0.0.1", open_port, timeout=2.0) == FAULT_SSH_BANNER_MISSING
        )
    finally:
        server.close()
        await server.wait_closed()

    async def greet_in_two_segments(_reader, writer):
        # Rustam's review (16 Sep): `read(255)` judged the first TCP segment. Split inside the
        # prefix, that read saw `SSH-2` and the `SSH-2.0-` rule called a healthy sshd unreachable.
        writer.write(b"SSH-2")
        await writer.drain()
        await asyncio.sleep(0.05)
        writer.write(b".0-OpenSSH_9.6p1 Ubuntu-3ubuntu13\r\n")
        await writer.drain()
        writer.close()

    sshd = await asyncio.start_server(greet_in_two_segments, "127.0.0.1", 0)
    sshd_port = sshd.sockets[0].getsockname()[1]
    try:
        assert await tcp_connect_fault("127.0.0.1", sshd_port, timeout=2.0) is None
    finally:
        sshd.close()
        await sshd.wait_closed()

    async def hang(*_args, **_kwargs):
        await asyncio.sleep(10)

    with patch("asyncio.open_connection", new=hang):
        assert await tcp_connect_fault("127.0.0.1", open_port, timeout=0.05) == FAULT_TCP_TIMEOUT


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "greeting",
    [
        pytest.param(b"SSH-1.99-OpenSSH_3.9p1\r\n", id="ssh_1.99"),
        pytest.param(b"SSH-1.5-Cisco-1.25\r\n", id="ssh_1.5"),
        pytest.param(b"HTTP/1.1 400 Bad Request\r\n\r\n", id="http"),
        pytest.param(b"\x00\xff\x16\x03\x01garbage\r\n", id="garbage"),
        pytest.param(b"SSH-2.0-\r\n", id="no_softwareversion"),
        pytest.param(b"SSH-2.0-OpenSSH_9.6p1", id="closed_before_the_line_ended"),
        pytest.param(b"SSH-2.0-" + b"x" * 246 + b"\r\n", id="256_bytes"),
        pytest.param(b"SSH-2.0-" + b"x" * 300 + b"\r\n", id="longer_than_255"),
        pytest.param(b"x" * 400, id="400_bytes_and_no_lf"),
    ],
)
async def test_tcp_connect_fault_rejects_a_non_2_0_or_malformed_identification(greeting):
    # Rustam's review (16 Sep): the `SSH-` prefix accepted an SSH 1.x server or a malformed line as
    # a healthy pod. Each greeting below is served on a real local socket and must be a fault.
    def serve(_reader, writer):
        writer.write(greeting)
        writer.close()

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        assert await tcp_connect_fault("127.0.0.1", port, timeout=2.0) == FAULT_SSH_BANNER_MISSING
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.parametrize(
    ("line", "ok"),
    [
        (b"SSH-2.0-OpenSSH_9.6p1 Ubuntu-3ubuntu13\r\n", True),
        (b"SSH-2.0-dropbear_2024.85\n", True),  # LF alone is accepted, as OpenSSH's client does
        (b"SSH-2.0-" + b"x" * 245 + b"\r\n", True),  # exactly 255 bytes
        (b"SSH-2.0-" + b"x" * 246 + b"\r\n", False),  # 256 bytes
        (b"SSH-1.99-OpenSSH_3.9p1\r\n", False),
        (b"SSH-1.5-Cisco-1.25\r\n", False),
        (b"SSH-2.0-\r\n", False),
        (b"SSH-2.0-OpenSSH_9.6p1", False),  # no LF: the line never completed
        (b"ssh-2.0-OpenSSH_9.6p1\r\n", False),  # the protocol name is case-sensitive
        (b"", False),
    ],
)
def test_is_ssh2_identification(line, ok):
    assert is_ssh2_identification(line) is ok


def test_streak_state_round_trips_and_a_corrupt_count_restarts_at_zero():
    # Regression for the bare-dict version: a `count` that was not a non-negative int fell back to 0
    # in one place and was read raw in another. The dataclass owns the rule and the wire format.
    now = "2026-09-15T15:00:00+00:00"
    stored = rented_pod_ssh.FailStreak(count=2, first_failed_at="2026-09-15T14:30:00+00:00")
    loaded = rented_pod_ssh.FailStreak.load(stored.dump().encode(), now_iso=now)
    assert loaded == stored
    assert loaded.next() == rented_pod_ssh.FailStreak(
        count=3, first_failed_at=stored.first_failed_at
    )
    assert json.loads(stored.dump()) == {
        "count": 2,
        "first_failed_at": stored.first_failed_at,
        "reported": False,
    }
    reported = rented_pod_ssh.FailStreak.load(
        b'{"count": 2, "first_failed_at": "x", "reported": true}', now_iso=now
    )
    assert reported.reported is True and reported.next().reported is True
    assert (
        rented_pod_ssh.FailStreak.load(b'{"count": 2, "reported": "yes"}', now_iso=now).reported
        is False
    )

    for raw in (None, b"not json", b"[1]", b'{"count": "2"}', b'{"count": -1}', b'{"count": true}'):
        assert rented_pod_ssh.FailStreak.load(raw, now_iso=now) == rented_pod_ssh.FailStreak(
            count=0, first_failed_at=now
        ), raw

    ok = rented_pod_ssh.OkMark(at=now, boot_id="boot-a")
    assert rented_pod_ssh.OkMark.load(ok.dump()) == ok
    assert rented_pod_ssh.OkMark.load(b'{"at": "x"}') == rented_pod_ssh.OkMark(at="x", boot_id=None)
    assert rented_pod_ssh.OkMark.load(None) is None
    assert rented_pod_ssh.OkMark.load(b"garbage") is None
