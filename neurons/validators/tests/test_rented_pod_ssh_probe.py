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
from helpers import FakeRedis, build_context_config, build_services, build_state, default_executor
from neurons.validators.src.core.utils import _m
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
from neurons.validators.src.services.task.models import JobResult
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

    async def cycle(
        self,
        *,
        tcp_fault: str | None,
        ssh_keys: list[str],
        boot_id: str = "boot-a",
        validator_outage: bool = False,
    ):
        """One cycle as the validator runs it: the rented check, then the cycle-end fleet gate."""
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
        self.gate = await rented_pod_ssh.flush_rented_pod_ssh_reports(
            self.redis, self.backend, "batch-1", validator_outage=validator_outage
        )
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
    assert pod["report_queued"] is True and h.streak()["reported"] is True
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
    assert pod["consecutive_cycles"] == 3 and pod["report_queued"] is False
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
    assert pod["report_queued"] is True
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
    assert pod["report_queued"] is True and h.streak()["reported"] is True
    assert h.streak() == {
        "count": 4,
        "first_failed_at": h.streak()["first_failed_at"],
        "reported": True,
    }
    [pod] = after.event.what_we_saw["unreachable_pods"]
    assert pod["report_queued"] is False
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
    assert pod["consecutive_cycles"] == 2 and pod["report_queued"] is True
    h.backend.report_pod_ssh_unreachable.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_healthy_cycle_whose_redis_write_is_lost_leaves_no_half_state(
    context_factory, caplog
):
    # Rustam's review (17 Sep): the healthy cycle wrote the ok mark, then deleted the streak. A
    # connection lost between the two left a fresh ok mark ("healthy now, on boot-b") next to the
    # old streak, so the stored state contradicted itself and the next report carried a
    # first_failed_at older than a recorded healthy sighting. Every transition is now one
    # MULTI/EXEC: the lost write applies nothing, the ok mark still says boot-a, and the cycle is a
    # skipped one (REDIS_UNAVAILABLE), exactly as when Redis is down for the whole cycle.
    h = Harness(context_factory)
    ok_key = f"{rented_pod_ssh.RENTED_POD_SSH_OK_KEY_PREFIX}:{POD_ID}"
    fail_key = f"{rented_pod_ssh.RENTED_POD_SSH_FAIL_KEY_PREFIX}:{POD_ID}"
    await h.cycle(tcp_fault=None, ssh_keys=KEYS, boot_id="boot-a")
    await h.cycle(tcp_fault=FAULT_TCP_REFUSED, ssh_keys=KEYS, boot_id="boot-a")
    before = dict(h.redis.store)

    h.redis.fail_delete_of.add(fail_key)
    with caplog.at_level("WARNING", logger=rented_pod_ssh.__name__):
        lost = await h.cycle(tcp_fault=None, ssh_keys=KEYS, boot_id="boot-b")

    assert lost.event.reason_code == Msg.ALREADY_RENTED.reason
    assert h.redis.store == before, "a lost transaction must apply none of its writes"
    assert json.loads(h.redis.store[ok_key])["boot_id"] == "boot-a"
    assert any("RENTED_POD_SSH_PROBE_REDIS_UNAVAILABLE" in r.getMessage() for r in caplog.records)

    # Redis back: the healthy cycle is recorded whole, and the streak starts over on the next fault.
    await h.cycle(tcp_fault=None, ssh_keys=KEYS, boot_id="boot-b")
    assert h.streak() is None and json.loads(h.redis.store[ok_key])["boot_id"] == "boot-b"
    result = await h.cycle(tcp_fault=FAULT_TCP_REFUSED, ssh_keys=KEYS, boot_id="boot-b")
    assert result.event.reason_code == Msg.ALREADY_RENTED.reason and h.streak()["count"] == 1
    h.backend.report_pod_ssh_unreachable.assert_not_awaited()


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
    assert pod["report_queued"] is True and h.streak()["reported"] is True
    [pod] = after.event.what_we_saw["unreachable_pods"]
    assert pod["report_queued"] is True
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
    assert pod["consecutive_cycles"] == 3 and pod["report_queued"] is True
    h.backend.report_pod_ssh_unreachable.assert_awaited_once()


class Fleet:
    """Several rented pods on one validator across cycles, each probed as its executor's task would.

    Rustam's review (16 Sep, three times): a validator whose own network fails sees every mapped
    port refuse at once, and per-pod reporting would tell every healthy renter their pod is down.
    The reports are queued per cycle and posted at the cycle's end only when the fleet reads clean.
    """

    def __init__(self, context_factory, pod_ids: list[str]):
        self.context_factory = context_factory
        self.pod_ids = pod_ids
        self.redis = FakeRedis()
        self.backend = AsyncMock()
        self.backend.report_pod_ssh_unreachable.return_value = PodSshUnreachableResponse(
            recorded=True
        )

    async def cycle(self, *, failing: set[str] = frozenset(), validator_outage: bool = False):
        services = build_services(redis=self.redis, backend=self.backend)
        verdicts = {}
        for index, pod_id in enumerate(self.pod_ids):
            ctx = self.context_factory(
                services=services,
                config=build_context_config(),
                state=build_state(specs={"boot_id": "boot-a"}),
                collateral_deposited=True,
            )
            pod = RentedPod(pod_id=pod_id, container_name=f"pod_{index}", ssh_port=40000 + index)
            fault = FAULT_TCP_TIMEOUT if pod_id in failing else None
            with patch(TCP_PATH, new=AsyncMock(return_value=fault)):
                verdicts[pod_id] = await rented_pod_ssh.probe_rented_pod_ssh(ctx, pod, KEYS)
        gate = await rented_pod_ssh.flush_rented_pod_ssh_reports(
            self.redis, self.backend, "batch-1", validator_outage=validator_outage
        )
        return verdicts, gate

    def posted_pods(self) -> list[str]:
        return [call.args[0] for call in self.backend.report_pod_ssh_unreachable.await_args_list]

    def reported(self, pod_id: str) -> bool:
        raw = self.redis.store.get(f"{rented_pod_ssh.RENTED_POD_SSH_FAIL_KEY_PREFIX}:{pod_id}")
        return json.loads(raw)["reported"] if raw else False


SIX_PODS = [f"pod-{n}" for n in range(6)]


@pytest.mark.asyncio
async def test_a_fleet_wide_port_outage_holds_every_report_back_until_the_fleet_reads_clean(
    context_factory, caplog
):
    # Four of six mapped ports time out for two cycles: that is the validator's network, not four
    # hosts rebooting in the same half hour. No renter is told. When the outage clears for three
    # of them and one pod is still down, that one is reported on the next cycle.
    fleet = Fleet(context_factory, SIX_PODS)
    await fleet.cycle()
    await fleet.cycle(failing={"pod-0", "pod-1", "pod-2", "pod-3"})
    with caplog.at_level("WARNING", logger=rented_pod_ssh.__name__):
        verdicts, gate = await fleet.cycle(failing={"pod-0", "pod-1", "pod-2", "pod-3"})

    assert [pod_id for pod_id, v in verdicts.items() if v.report] == SIX_PODS[:4]
    assert all(verdicts[pod_id].report_queued for pod_id in SIX_PODS[:4])
    assert gate.probed == 6 and gate.failed == 4 and gate.due == SIX_PODS[:4]
    assert gate.suppressed_by == "mapped_port_share" and gate.posted == []
    fleet.backend.report_pod_ssh_unreachable.assert_not_awaited()
    [record] = [
        r for r in caplog.records if "RENTED_POD_SSH_PROBE_SUPPRESSED_FLEET" in r.getMessage()
    ]
    logged = record.msg.extra
    assert logged["outcome"] == rented_pod_ssh.PROBE_SUPPRESSED_FLEET
    assert logged["due_pods"] == SIX_PODS[:4] and logged["fail_share"] == 0.667
    assert logged["suppressed_by"] == "mapped_port_share"
    assert not any(fleet.reported(pod_id) for pod_id in SIX_PODS)
    # the fleet keys are gone with the flush; nothing waits in Redis for a cycle that is over
    assert fleet.redis.hashes == {}

    _, gate = await fleet.cycle(failing={"pod-3"})
    assert gate.suppressed_by is None and gate.posted == ["pod-3"]
    assert fleet.posted_pods() == ["pod-3"]
    assert fleet.backend.report_pod_ssh_unreachable.await_args.kwargs["consecutive_cycles"] == 3
    assert fleet.reported("pod-3") and not fleet.reported("pod-0")


@pytest.mark.asyncio
async def test_pods_never_seen_healthy_are_not_in_the_fleet_share(context_factory):
    # Fresh review of round 5: three no-sshd templates refusing from the start read as 3 of 5
    # failing, and a real outage on one healthy pod was held back for ever. A pod this validator
    # never saw healthy is not counted (as before) and not in the share either.
    fleet = Fleet(context_factory, SIX_PODS)
    never_healthy = {"pod-0", "pod-1", "pod-2"}
    await fleet.cycle(failing=never_healthy)  # pods 3-5 healthy once; 0-2 never
    await fleet.cycle(failing=never_healthy | {"pod-5"})
    _, gate = await fleet.cycle(failing=never_healthy | {"pod-5"})

    assert gate.probed == 3 and gate.failed == 1 and gate.suppressed_by is None
    assert gate.due == ["pod-5"] and gate.posted == ["pod-5"]


@pytest.mark.asyncio
async def test_one_pod_down_in_a_healthy_fleet_is_reported_at_the_threshold(context_factory):
    # The fleet share is what makes one pod's outage believable: five ports answer, one refuses.
    fleet = Fleet(context_factory, SIX_PODS)
    await fleet.cycle()
    await fleet.cycle(failing={"pod-4"})
    verdicts, gate = await fleet.cycle(failing={"pod-4"})

    assert gate.probed == 6 and gate.failed == 1 and gate.suppressed_by is None
    assert gate.due == ["pod-4"] and gate.posted == ["pod-4"]
    assert verdicts["pod-4"].report_queued is True
    assert fleet.posted_pods() == ["pod-4"] and fleet.reported("pod-4")
    # a healthy fleet with nothing due logs the fleet line only; no report, no warning
    _, gate = await fleet.cycle()
    assert gate.due == [] and fleet.posted_pods() == ["pod-4"]


@pytest.mark.asyncio
async def test_the_cycles_executor_ssh_verdict_holds_the_reports_back_too(context_factory):
    # DAH-2748 already judges the validator's own egress each cycle (most executors refused its
    # SSH). That verdict gates these reports as well, on any fleet size: one pod on a validator
    # that could not reach its executors is not a report, it is the same outage seen twice.
    fleet = Fleet(context_factory, ["pod-a"])
    await fleet.cycle()
    await fleet.cycle(failing={"pod-a"})
    _, gate = await fleet.cycle(failing={"pod-a"}, validator_outage=True)

    assert gate.suppressed_by == "validator_outage" and gate.due == ["pod-a"]
    fleet.backend.report_pod_ssh_unreachable.assert_not_awaited()
    assert not fleet.reported("pod-a")

    _, gate = await fleet.cycle(failing={"pod-a"}, validator_outage=False)
    assert gate.posted == ["pod-a"] and fleet.reported("pod-a")


def _job_result(check_result, pod_ids: list[str] | None = None) -> JobResult:
    """The cycle's JobResult for one executor, as the task service builds it from the halt."""
    event = check_result.event.model_copy(deep=True)
    if pod_ids is not None:
        pods = event.what_we_saw["unreachable_pods"]
        event.what_we_saw["unreachable_pods"] = [
            {**pods[0], "pod_id": pod_id} for pod_id in pod_ids
        ]
    return JobResult(
        executor_info=default_executor(),
        score=0.9,
        job_score=0.9,
        job_batch_id="batch-1",
        log_status="info",
        log_text=_m(event.event, extra=event.model_dump()).to_full_string(),
        validation_event=event,
    )


@pytest.mark.asyncio
async def test_a_suppressed_cycles_events_publish_as_rented_with_the_gates_verdict(context_factory):
    # Rustam's review (17 Sep): the executor task rendered RENTED_POD_SSH_UNREACHABLE ("Reported to
    # the backend and the renter") before the gate ran; on a suppressed cycle nobody was told, and
    # the event published a claim that was false. The sync loop now rewrites those results to RENTED
    # before the publish, as DAH-2748 does for availability errors, with the gate's verdict kept.
    h = Harness(context_factory)
    await h.cycle(tcp_fault=None, ssh_keys=KEYS)
    await h.cycle(tcp_fault=FAULT_TCP_REFUSED, ssh_keys=KEYS)
    held = await h.cycle(tcp_fault=FAULT_TCP_REFUSED, ssh_keys=KEYS, validator_outage=True)
    assert held.event.reason_code == Msg.RENTED_POD_SSH_UNREACHABLE.reason
    assert h.gate.suppressed_by == "validator_outage" and h.gate.due == [POD_ID]
    trace_id = held.event.trace_id

    result = _job_result(held)
    other = _job_result(held, pod_ids=["pod-reported-last-cycle"])  # never due: reported before
    rewritten = rented_pod_ssh.silence_rented_pod_ssh_reports_on_our_own_outage(
        [result, other], h.gate
    )

    assert rewritten == 1
    event = result.validation_event
    assert event.reason_code == Msg.ALREADY_RENTED.reason and event.severity == "info"
    assert event.impact == "Reported rented score=0.9 (actual=0.9)"
    assert event.remediation == "No action needed."
    assert event.trace_id == trace_id and event.check_id == held.event.check_id
    assert "unreachable_pods" not in event.what_we_saw
    seen = event.what_we_saw[rented_pod_ssh.PROBE_SUPPRESSED_FLEET]
    assert (
        seen["suppressed_by"] == "validator_outage" and seen["probed"] == 1 and seen["failed"] == 1
    )
    assert [pod["pod_id"] for pod in seen["unreachable_pods"]] == [POD_ID]
    assert event.what_we_saw["job_score"] == 0.9 and event.what_we_saw["actual_score"] == 0.9
    assert result.log_text.startswith(f"{Msg.ALREADY_RENTED.event} >>> ")
    assert json.loads(result.log_text.split(" >>> ", 1)[1])["reason_code"] == "RENTED"
    assert result.score == 0.9  # the halt kept the rented score; the rewrite touches the event only
    # the executor whose pod was reported in an earlier cycle keeps its event: for it the impact is true
    assert other.validation_event.reason_code == Msg.RENTED_POD_SSH_UNREACHABLE.reason
    assert rented_pod_ssh.PROBE_SUPPRESSED_FLEET not in other.validation_event.what_we_saw

    # Rustam, round 7: a result naming both a held pod and a pod reported in an earlier cycle keeps
    # its reason for the told renter only; the held pod moves under probe_suppressed_fleet.
    mixed = _job_result(held, pod_ids=[POD_ID, "pod-reported-last-cycle"])
    assert rented_pod_ssh.silence_rented_pod_ssh_reports_on_our_own_outage([mixed], h.gate) == 1
    event = mixed.validation_event
    assert event.reason_code == Msg.RENTED_POD_SSH_UNREACHABLE.reason and event.trace_id == trace_id
    assert [pod["pod_id"] for pod in event.what_we_saw["unreachable_pods"]] == [
        "pod-reported-last-cycle"
    ]
    seen = event.what_we_saw[rented_pod_ssh.PROBE_SUPPRESSED_FLEET]
    assert [pod["pod_id"] for pod in seen["unreachable_pods"]] == [POD_ID]
    logged = json.loads(mixed.log_text.split(" >>> ", 1)[1])
    assert [pod["pod_id"] for pod in logged["what_we_saw"]["unreachable_pods"]] == [
        "pod-reported-last-cycle"
    ]

    # a posted cycle rewrites nothing; neither does a flush that never ran (Redis down: gate None)
    posted = await h.cycle(tcp_fault=FAULT_TCP_REFUSED, ssh_keys=KEYS)
    assert h.gate.posted == [POD_ID]
    assert (
        rented_pod_ssh.silence_rented_pod_ssh_reports_on_our_own_outage(
            [_job_result(posted)], h.gate
        )
        == 0
    )
    assert (
        rented_pod_ssh.silence_rented_pod_ssh_reports_on_our_own_outage([_job_result(posted)], None)
        == 0
    )


@pytest.mark.asyncio
async def test_a_fleet_too_small_for_a_share_reports_what_it_found(context_factory):
    # Two pods, both refusing, is a share of 1.0, and it says nothing: under
    # SMALLEST_FLEET_THAT_CAN_SHOW_AN_OUTAGE pods the share rule does not apply (the executor-SSH
    # verdict still does). A validator this small cannot notify "many" renters either way.
    fleet = Fleet(context_factory, ["pod-a", "pod-b"])
    await fleet.cycle()
    await fleet.cycle(failing={"pod-a", "pod-b"})
    _, gate = await fleet.cycle(failing={"pod-a", "pod-b"})

    assert gate.probed == 2 and gate.fail_share == 1.0 and gate.suppressed_by is None
    assert sorted(fleet.posted_pods()) == ["pod-a", "pod-b"]


@pytest.mark.asyncio
async def test_the_share_threshold_is_the_setting(context_factory):
    # Exactly half the fleet failing is not above the default 0.5 (the DAH-2748 rule); a lower
    # setting holds the same cycle back.
    fleet = Fleet(context_factory, SIX_PODS)
    await fleet.cycle()
    await fleet.cycle(failing=set(SIX_PODS[:3]))
    with patch.object(rented_pod_ssh.settings, "RENTED_POD_SSH_PROBE_FLEET_FAIL_MAX", 0.25):
        _, gate = await fleet.cycle(failing=set(SIX_PODS[:3]))
    assert gate.fail_share == 0.5 and gate.suppressed_by == "mapped_port_share"

    _, gate = await fleet.cycle(failing=set(SIX_PODS[:3]))
    assert gate.suppressed_by is None and sorted(gate.posted) == SIX_PODS[:3]


@pytest.mark.asyncio
async def test_a_redis_outage_at_the_flush_posts_nothing_and_the_streaks_ask_again(
    context_factory,
):
    # The flush reads the two hashes first; Redis down there means no gate and no POST this cycle.
    # The per-pod streaks are untouched, so the next flush with Redis back posts once.
    fleet = Fleet(context_factory, ["pod-a"])
    await fleet.cycle()
    await fleet.cycle(failing={"pod-a"})
    services = build_services(redis=fleet.redis, backend=fleet.backend)
    ctx = context_factory(
        services=services,
        config=build_context_config(),
        state=build_state(specs={"boot_id": "boot-a"}),
        collateral_deposited=True,
    )
    pod = RentedPod(pod_id="pod-a", container_name="pod_0", ssh_port=40000)
    with patch(TCP_PATH, new=AsyncMock(return_value=FAULT_TCP_TIMEOUT)):
        verdict = await rented_pod_ssh.probe_rented_pod_ssh(ctx, pod, KEYS)
    assert verdict.report_queued is True
    fleet.redis.failing = True
    gate = await rented_pod_ssh.flush_rented_pod_ssh_reports(fleet.redis, fleet.backend, "batch-1")
    fleet.redis.failing = False

    assert gate is None
    fleet.backend.report_pod_ssh_unreachable.assert_not_awaited()
    _, gate = await fleet.cycle(failing={"pod-a"})
    assert gate.posted == ["pod-a"] and fleet.reported("pod-a")


@pytest.mark.asyncio
async def test_flush_with_the_probe_off_touches_nothing():
    redis = FakeRedis()
    with patch(SETTINGS_PATH) as settings:
        settings.RENTED_POD_SSH_PROBE_ENABLED = False
        assert await rented_pod_ssh.flush_rented_pod_ssh_reports(redis, AsyncMock(), "b") is None
    assert redis.calls == 0


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
async def test_lines_before_the_identification_are_skipped_within_bounds():
    # Rustam's review (17 Sep): RFC 4253 §4.2 lets a server send other lines before `SSH-`, and a
    # client MUST skip them; the probe read the first line alone and called such a server unreachable.
    async def serve(lines: list[bytes]):
        def handler(_reader, writer):
            writer.write(b"".join(lines))
            writer.close()

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        return server, server.sockets[0].getsockname()[1]

    banner = [b"Welcome to the box\r\n", b"\r\n", b"No unauthorised access\r\n"]
    ssh2 = b"SSH-2.0-OpenSSH_9.6p1 Ubuntu-3ubuntu13\r\n"
    cases = [
        (banner + [ssh2], None),
        # up to SSH_PRE_BANNER_LINES_MAX lines are skipped; one more and the identification is not read
        ([b"x\r\n"] * rented_pod_ssh.SSH_PRE_BANNER_LINES_MAX + [ssh2], None),
        (
            [b"x\r\n"] * (rented_pod_ssh.SSH_PRE_BANNER_LINES_MAX + 1) + [ssh2],
            FAULT_SSH_BANNER_MISSING,
        ),
        # a pre-banner line is bounded like the identification: past 255 bytes the probe stops reading
        ([b"y" * 300 + b"\r\n", ssh2], FAULT_SSH_BANNER_MISSING),
        # the first `SSH-` line is the identification, and it is judged as before (1.99 is refused)
        (banner + [b"SSH-1.99-OpenSSH_3.9p1\r\n", ssh2], FAULT_SSH_BANNER_MISSING),
        # a peer that only ever sends other lines and closes has no identification
        (banner, FAULT_SSH_BANNER_MISSING),
    ]
    for lines, expected in cases:
        server, port = await serve(lines)
        try:
            assert await tcp_connect_fault("127.0.0.1", port, timeout=2.0) == expected, lines[:2]
        finally:
            server.close()
            await server.wait_closed()


@pytest.mark.asyncio
async def test_connect_and_banner_read_share_one_deadline():
    # Rustam's review (17 Sep): the timeout applied to the connect and again to the read, so one
    # probe could take twice the configured value. A connect that uses 0.2 s of a 0.3 s budget
    # leaves the read 0.1 s, and the probe ends at 0.3 s (not 0.5 s) as ssh_banner_missing.
    accepted: list[asyncio.StreamWriter] = []
    silent = await asyncio.start_server(lambda r, w: accepted.append(w), "127.0.0.1", 0)
    port = silent.sockets[0].getsockname()[1]
    real_open_connection = asyncio.open_connection

    async def slow_connect(*args, **kwargs):
        await asyncio.sleep(0.2)
        return await real_open_connection(*args, **kwargs)

    loop = asyncio.get_running_loop()
    try:
        with patch("asyncio.open_connection", new=slow_connect):
            started = loop.time()
            fault = await tcp_connect_fault("127.0.0.1", port, timeout=0.3)
            elapsed = loop.time() - started
    finally:
        for writer in accepted:
            writer.close()
        silent.close()
        await silent.wait_closed()

    assert fault == FAULT_SSH_BANNER_MISSING
    assert 0.25 <= elapsed < 0.45, elapsed

    # a connect that does not finish inside the budget is still tcp_timeout
    async def never_connects(*_args, **_kwargs):
        await asyncio.sleep(10)

    with patch("asyncio.open_connection", new=never_connects):
        assert await tcp_connect_fault("127.0.0.1", port, timeout=0.05) == FAULT_TCP_TIMEOUT


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


@pytest.mark.asyncio
async def test_redis_writes_go_through_the_client_as_one_multi_exec():
    # The real client path FakeRedis (tests/helpers) stands in for: a RedisWrites batch is one
    # MULTI/EXEC on redis-py's pipeline, every command lands, and a connection lost before EXEC
    # applies none of them (the healthy transition's SET and DELETE cannot come apart).
    import asyncio as _asyncio

    from fakeredis import FakeServer
    from fakeredis.aioredis import FakeRedis as FakeClient
    from neurons.validators.src.services.redis_service import RedisService, RedisWrites

    server = FakeServer()
    service = RedisService.__new__(RedisService)
    service.redis = FakeClient(server=server)
    service.lock = _asyncio.Lock()
    await service.redis.set("rented_pod_ssh_fail:pod-1", '{"count": 1}')

    healthy = RedisWrites()
    healthy.set("rented_pod_ssh_ok:pod-1", '{"at": "t", "boot_id": "b"}', ex=3600)
    healthy.delete("rented_pod_ssh_fail:pod-1").hset("rented_pod_ssh_fleet:c1", "pod-1", "ok")
    healthy.expire("rented_pod_ssh_fleet:c1", 60)
    await service.write_atomically(healthy)

    assert await service.redis.get("rented_pod_ssh_fail:pod-1") is None
    assert await service.redis.get("rented_pod_ssh_ok:pod-1") == b'{"at": "t", "boot_id": "b"}'
    assert 0 < await service.redis.ttl("rented_pod_ssh_ok:pod-1") <= 3600
    assert await service.redis.hgetall("rented_pod_ssh_fleet:c1") == {b"pod-1": b"ok"}
    assert 0 < await service.redis.ttl("rented_pod_ssh_fleet:c1") <= 60

    # the connection goes away before EXEC: the SET is not applied without its DELETE
    await service.redis.set("rented_pod_ssh_fail:pod-1", '{"count": 1}')
    server.connected = False
    again = (
        RedisWrites()
        .set("rented_pod_ssh_ok:pod-1", "new", ex=3600)
        .delete("rented_pod_ssh_fail:pod-1")
    )
    with pytest.raises(rented_pod_ssh.REDIS_ERRORS):
        await service.write_atomically(again)
    server.connected = True
    assert await service.redis.get("rented_pod_ssh_ok:pod-1") == b'{"at": "t", "boot_id": "b"}'
    assert await service.redis.get("rented_pod_ssh_fail:pod-1") == b'{"count": 1}'
    await service.write_atomically(RedisWrites())  # nothing to send, nothing sent


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
