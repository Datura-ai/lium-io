"""DAH-2255: a rented pod that refuses its renter fails the rented-state check, behind a flag.

Built on the DAH-2870 probe (lium-io#1372): the streak, the ok mark and the fleet gate are its.
With ``RENTED_POD_SSH_ENFORCEMENT_ENABLED`` off (the default) nothing here changes what #1372
does — record and report, rented score kept. On, a pod whose streak reaches
``RENTED_POD_SSH_ENFORCE_AFTER_CYCLES`` (default: the notify threshold) makes the check FAIL for
the cycle: score 0, verified job cleared, the way the rental probe fails an unreachable unrented
node. A healthy cycle brings the rented score back.
"""

from __future__ import annotations

import json
from dataclasses import replace
from unittest.mock import patch

import pytest
from helpers import default_executor
from neurons.validators.src.core.config import Settings
from neurons.validators.src.core.utils import _m
from neurons.validators.src.services.task.checks import rented_machine, rented_pod_ssh
from neurons.validators.src.services.task.checks.rented_pod_ssh import (
    FAULT_AUTHORIZED_KEYS_UNREADABLE,
    FAULT_TCP_REFUSED,
    RentedPodSshVerdict,
)
from neurons.validators.src.services.task.messages import TenantEnforcementMessages as Msg
from neurons.validators.src.services.task.models import JobResult
from neurons.validators.src.services.task.pipeline import (
    updates_with_clear_verified_job_evidence,
)
from pydantic import ValidationError
from test_rented_pod_ssh_probe import KEYS, POD_ID, SSH_PORT, Harness

ENFORCED_LOG = "RENTED_POD_SSH_UNREACHABLE_ENFORCED"
CHECK_ID = rented_machine.TenantEnforcementCheck.check_id
# The only key material the check ever holds: the renter's authorized_keys, read off the pod.
KEY_MATERIAL = KEYS[0].split()[1]


def enforcement(*, enabled: bool, after_cycles: int | None = None):
    return patch.multiple(
        rented_pod_ssh.settings,
        RENTED_POD_SSH_ENFORCEMENT_ENABLED=enabled,
        RENTED_POD_SSH_ENFORCE_AFTER_CYCLES=after_cycles,
    )


async def two_refused_cycles_after_a_healthy_one(h: Harness):
    await h.cycle(tcp_fault=None, ssh_keys=KEYS, boot_id="boot-a")
    await h.cycle(tcp_fault=FAULT_TCP_REFUSED, ssh_keys=KEYS, boot_id="boot-b")
    return await h.cycle(tcp_fault=FAULT_TCP_REFUSED, ssh_keys=KEYS, boot_id="boot-b")


async def accepted_then_one_more_refused_cycle(h: Harness):
    """Notify cycle (backend accepts) then the next refused cycle, which can enforce."""
    await two_refused_cycles_after_a_healthy_one(h)
    return await h.cycle(tcp_fault=FAULT_TCP_REFUSED, ssh_keys=KEYS, boot_id="boot-b")


@pytest.mark.asyncio
async def test_flag_off_keeps_the_rented_score_at_the_threshold(context_factory, caplog):
    # The default: #1372's behaviour, byte for byte — the event names the pod, the score stands,
    # the verified job is not touched, and nothing about enforcement is logged.
    h = Harness(context_factory)
    with enforcement(enabled=False), caplog.at_level("WARNING", logger=rented_machine.__name__):
        result = await two_refused_cycles_after_a_healthy_one(h)

    assert result.passed is True and result.halt is True
    assert result.event.reason_code == Msg.RENTED_POD_SSH_UNREACHABLE.reason
    assert result.event.impact == Msg.RENTED_POD_SSH_UNREACHABLE.impact
    assert result.updates["score"] == 0.9 and result.updates["job_score"] == 0.9
    assert "clear_verified_job_info" not in result.updates
    assert "enforced" not in result.event.what_we_saw
    assert not any(ENFORCED_LOG in record.getMessage() for record in caplog.records)
    h.backend.report_pod_ssh_unreachable.assert_awaited_once()


@pytest.mark.asyncio
async def test_flag_on_but_streak_below_the_threshold_keeps_the_rented_score(context_factory):
    # One refused cycle is a blip (conntrack flush, sshd restart) and never zeroes a node, flag or
    # no flag: the enforcement threshold is never below the notify threshold.
    h = Harness(context_factory)
    with enforcement(enabled=True):
        await h.cycle(tcp_fault=None, ssh_keys=KEYS)
        result = await h.cycle(tcp_fault=FAULT_TCP_REFUSED, ssh_keys=KEYS)

    assert result.passed is True and result.halt is True
    assert result.event.reason_code == Msg.ALREADY_RENTED.reason
    assert result.updates["score"] == 0.9 and result.updates["job_score"] == 0.9
    assert "clear_verified_job_info" not in result.updates
    assert h.streak()["count"] == 1
    h.backend.report_pod_ssh_unreachable.assert_not_awaited()


@pytest.mark.asyncio
async def test_flag_on_and_streak_at_the_threshold_fails_the_check(context_factory, caplog):
    # ticket-0326 with the flag on: the notify cycle posts once; the next refused cycle (backend
    # already accepted) fails the rented-state check the way the rental probe fails an unreachable
    # unrented node — score 0, verified job cleared with the outage as evidence.
    h = Harness(context_factory)
    with enforcement(enabled=True), caplog.at_level("WARNING", logger=rented_machine.__name__):
        told = await two_refused_cycles_after_a_healthy_one(h)
        assert told.passed is True and told.updates["score"] == 0.9
        assert told.event.what_we_saw.get("enforced") is not True
        h.backend.report_pod_ssh_unreachable.assert_awaited_once()
        result = await h.cycle(tcp_fault=FAULT_TCP_REFUSED, ssh_keys=KEYS, boot_id="boot-b")

    assert result.passed is False
    assert result.event.reason_code == Msg.RENTED_POD_SSH_UNREACHABLE.reason
    assert result.event.severity == "error"
    assert result.event.impact == Msg.RENTED_POD_SSH_UNREACHABLE_ENFORCED_IMPACT
    assert result.updates["score"] == 0.0 and result.updates["job_score"] == 0.0
    assert result.updates["clear_verified_job_info"] is True
    assert "clear_verified_job_reason" not in result.updates  # ResetVerifiedJobReason.DEFAULT
    assert result.updates["score_warning"]
    # What travels with the reset to the backend's penalty row (DAH-3386): the check, the pod, the
    # faults and the streak. The pipeline fills nothing else in because the check named itself.
    evidence = result.updates["clear_verified_job_evidence"]
    assert evidence == {
        "reason_code": Msg.RENTED_POD_SSH_UNREACHABLE.reason,
        "check_id": CHECK_ID,
        "pod_id": POD_ID,
        "ssh_port": SSH_PORT,
        "faults": [FAULT_TCP_REFUSED],
        "consecutive_cycles": 3,
        "first_failed_at": h.streak()["first_failed_at"],
        "boot_id_changed": True,
        "enforce_after_cycles": 2,
    }
    assert updates_with_clear_verified_job_evidence(result, CHECK_ID)[
        "clear_verified_job_evidence"
    ] == evidence
    what = result.event.what_we_saw
    assert what["enforced"] is True and what["enforce_after_cycles"] == 2
    [pod] = what["unreachable_pods"]
    assert pod["pod_id"] == POD_ID and pod["consecutive_cycles"] == 3
    # The renter-facing report is #1372's and still goes out once, at the notify threshold.
    assert pod["report_queued"] is False
    h.backend.report_pod_ssh_unreachable.assert_awaited_once()

    # One log line names the enforcement; neither it, the event nor the evidence carries key
    # material (the authorized_keys the check read off the pod).
    enforced = [record for record in caplog.records if ENFORCED_LOG in record.getMessage()]
    assert len(enforced) == 1
    # The structured message: the name, then the extra as JSON (what Loki and the DB row get).
    line = enforced[0].msg.to_full_string()
    assert POD_ID in line and str(SSH_PORT) in line and FAULT_TCP_REFUSED in line
    assert '"enforce_after_cycles": 2' in line
    for text in (line, json.dumps(result.event.model_dump(mode="json")), json.dumps(evidence)):
        assert KEY_MATERIAL not in text and "ssh-ed25519" not in text
    # The failing result carries no keys either: POD_NOT_RUNNING, its sibling in this check, does not.
    assert "ssh_pub_keys" not in result.updates


@pytest.mark.asyncio
async def test_recovery_returns_the_rented_score_on_the_next_healthy_cycle(context_factory):
    # The host is fixed (sshd back, volume mounted): the next healthy cycle is RENTED at the rented
    # score, the streak is gone, and a later blip starts a new streak at 1 — not enforced.
    h = Harness(context_factory)
    with enforcement(enabled=True):
        failed = await accepted_then_one_more_refused_cycle(h)
        assert failed.passed is False and failed.updates["score"] == 0.0

        recovered = await h.cycle(tcp_fault=None, ssh_keys=KEYS, boot_id="boot-b")
        assert recovered.passed is True and recovered.halt is True
        assert recovered.event.reason_code == Msg.ALREADY_RENTED.reason
        assert recovered.updates["score"] == 0.9 and recovered.updates["job_score"] == 0.9
        assert "clear_verified_job_info" not in recovered.updates
        assert h.streak() is None

        blip = await h.cycle(tcp_fault=FAULT_TCP_REFUSED, ssh_keys=KEYS, boot_id="boot-b")
        assert blip.passed is True and blip.updates["score"] == 0.9
        assert h.streak()["count"] == 1


@pytest.mark.asyncio
async def test_a_separate_enforce_threshold_waits_past_the_notify_threshold(context_factory):
    # RENTED_POD_SSH_ENFORCE_AFTER_CYCLES=3 with the notify threshold at 2: the renter is told on
    # cycle 2 (score kept), the provider is zeroed on cycle 3.
    h = Harness(context_factory)
    with enforcement(enabled=True, after_cycles=3):
        told = await two_refused_cycles_after_a_healthy_one(h)
        assert told.passed is True and told.updates["score"] == 0.9
        assert told.event.reason_code == Msg.RENTED_POD_SSH_UNREACHABLE.reason
        assert told.event.impact == Msg.RENTED_POD_SSH_UNREACHABLE.impact
        h.backend.report_pod_ssh_unreachable.assert_awaited_once()

        zeroed = await h.cycle(tcp_fault=FAULT_TCP_REFUSED, ssh_keys=KEYS, boot_id="boot-b")
        assert zeroed.passed is False and zeroed.updates["score"] == 0.0
        assert zeroed.updates["clear_verified_job_info"] is True
        assert zeroed.event.what_we_saw["enforce_after_cycles"] == 3
        assert zeroed.updates["clear_verified_job_evidence"]["consecutive_cycles"] == 3
        # Still one report per outage: enforcement adds no POST.
        h.backend.report_pod_ssh_unreachable.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_pod_never_seen_healthy_is_never_enforced(context_factory):
    # A template without sshd, or a pod still coming up, has no streak (#1372 counts nothing for
    # it), so the flag cannot zero a node for a pod this validator never saw reachable.
    h = Harness(context_factory)
    with enforcement(enabled=True):
        for _ in range(3):
            result = await h.cycle(tcp_fault=FAULT_TCP_REFUSED, ssh_keys=[])

    assert result.passed is True and result.event.reason_code == Msg.ALREADY_RENTED.reason
    assert result.updates["score"] == 0.9
    assert h.streak() is None


@pytest.mark.asyncio
async def test_redis_down_at_the_threshold_skips_enforcement_for_the_cycle(context_factory):
    # Redis is an input to the signal, never to the verdict (#1372): with no streak readable
    # there is nothing to enforce, and the check answers RENTED as it does with the probe off.
    h = Harness(context_factory)
    with enforcement(enabled=True):
        await h.cycle(tcp_fault=None, ssh_keys=KEYS)
        await h.cycle(tcp_fault=FAULT_TCP_REFUSED, ssh_keys=KEYS)
        h.redis.failing = True
        result = await h.cycle(tcp_fault=FAULT_TCP_REFUSED, ssh_keys=KEYS)
        h.redis.failing = False

    assert result.passed is True and result.event.reason_code == Msg.ALREADY_RENTED.reason
    assert result.updates["score"] == 0.9
    assert "clear_verified_job_info" not in result.updates


@pytest.mark.asyncio
async def test_the_ticket_0247_fault_is_enforced_the_same_way(context_factory):
    # Host rebooted, port open, authorized_keys unreadable (the volume never remounted): the renter
    # gets a password prompt. Backend accepts on the notify cycle; the next cycle fails the check.
    h = Harness(context_factory)
    with enforcement(enabled=True):
        await h.cycle(tcp_fault=None, ssh_keys=KEYS, boot_id="boot-a")
        await h.cycle(tcp_fault=None, ssh_keys=[], boot_id="boot-b")
        told = await h.cycle(tcp_fault=None, ssh_keys=[], boot_id="boot-b")
        assert told.passed is True and told.updates["score"] == 0.9
        result = await h.cycle(tcp_fault=None, ssh_keys=[], boot_id="boot-b")

    assert result.passed is False and result.updates["score"] == 0.0
    assert result.updates["clear_verified_job_evidence"]["faults"] == [
        FAULT_AUTHORIZED_KEYS_UNREADABLE
    ]
    assert result.updates["clear_verified_job_evidence"]["boot_id_changed"] is True


@pytest.mark.asyncio
async def test_authorized_keys_without_a_reboot_is_not_enforced(context_factory):
    # A renter who deletes authorized_keys leaves the host boot_id unchanged. The provider cannot
    # restore those keys, so the fault is reported and never zeroes the node.
    h = Harness(context_factory)
    with enforcement(enabled=True):
        await h.cycle(tcp_fault=None, ssh_keys=KEYS, boot_id="boot-a")
        await h.cycle(tcp_fault=None, ssh_keys=[], boot_id="boot-a")
        told = await h.cycle(tcp_fault=None, ssh_keys=[], boot_id="boot-a")
        assert told.passed is True
        h.backend.report_pod_ssh_unreachable.assert_awaited_once()
        result = await h.cycle(tcp_fault=None, ssh_keys=[], boot_id="boot-a")

    assert result.passed is True and result.updates["score"] == 0.9
    assert "clear_verified_job_info" not in result.updates
    assert result.event.what_we_saw.get("enforced") is not True


@pytest.mark.asyncio
async def test_a_held_fleet_cycle_does_not_enforce(context_factory):
    # The fleet gate holds the renter notice (validator-side outage). The backend never accepted,
    # so the check keeps the rented score — a port outage on our side must not zero every node.
    h = Harness(context_factory)
    with enforcement(enabled=True):
        await h.cycle(tcp_fault=None, ssh_keys=KEYS)
        await h.cycle(tcp_fault=FAULT_TCP_REFUSED, ssh_keys=KEYS)
        held = await h.cycle(tcp_fault=FAULT_TCP_REFUSED, ssh_keys=KEYS, validator_outage=True)

    assert held.passed is True and held.updates["score"] == 0.9
    assert held.event.what_we_saw.get("enforced") is not True
    assert h.gate.suppressed_by == "validator_outage" and h.gate.due == [POD_ID]
    h.backend.report_pod_ssh_unreachable.assert_not_awaited()

    result = JobResult(
        executor_info=default_executor(),
        score=0.9,
        job_score=0.9,
        job_batch_id="batch-1",
        log_status="info",
        log_text=_m(held.event.event, extra=held.event.model_dump()).to_full_string(),
        validation_event=held.event.model_copy(deep=True),
    )
    rewritten = rented_pod_ssh.silence_rented_pod_ssh_reports_on_our_own_outage([result], h.gate)
    assert rewritten == 1
    assert result.validation_event.reason_code == Msg.ALREADY_RENTED.reason


@pytest.mark.asyncio
async def test_an_already_accepted_outage_stays_enforced_on_a_later_held_cycle(
    context_factory,
):
    # The backend already accepted this pod's outage. A later validator-side outage does not
    # un-zero it: the report is on record, so is_enforced stays true. Nothing is queued this
    # cycle (already reported), so the silence rewrite has no held pod and leaves the event.
    h = Harness(context_factory)
    with enforcement(enabled=True):
        failed = await accepted_then_one_more_refused_cycle(h)
        assert failed.passed is False
        later = await h.cycle(
            tcp_fault=FAULT_TCP_REFUSED, ssh_keys=KEYS, boot_id="boot-b", validator_outage=True
        )

    assert later.passed is False and later.updates["score"] == 0.0
    assert later.event.what_we_saw["enforced"] is True
    assert h.gate.suppressed_by == "validator_outage"
    assert h.gate.due == []
    rewritten = rented_pod_ssh.silence_rented_pod_ssh_reports_on_our_own_outage(
        [
            JobResult(
                executor_info=default_executor(),
                score=0.0,
                job_score=0.0,
                job_batch_id="batch-1",
                log_status="warning",
                log_text=_m(later.event.event, extra=later.event.model_dump()).to_full_string(),
                validation_event=later.event.model_copy(deep=True),
            )
        ],
        h.gate,
    )
    assert rewritten == 0


def test_enforcement_settings_default_off_and_the_threshold_is_never_below_notify(
    monkeypatch: pytest.MonkeyPatch,
):
    for name in (
        "RENTED_POD_SSH_ENFORCEMENT_ENABLED",
        "RENTED_POD_SSH_ENFORCE_AFTER_CYCLES",
        "RENTED_POD_SSH_PROBE_CYCLES",
    ):
        monkeypatch.delenv(name, raising=False)
    shipped = Settings(_env_file=None)
    assert shipped.RENTED_POD_SSH_ENFORCEMENT_ENABLED is False
    assert shipped.RENTED_POD_SSH_ENFORCE_AFTER_CYCLES is None
    with patch.object(rented_pod_ssh, "settings", shipped):
        assert rented_pod_ssh.enforce_after_cycles() == shipped.RENTED_POD_SSH_PROBE_CYCLES == 2

    # A threshold under the notify threshold would zero a provider for an outage no renter was
    # told about, and ENFORCE_AFTER_CYCLES=1 under the default would let one blip cost a cycle.
    monkeypatch.setenv("RENTED_POD_SSH_ENFORCE_AFTER_CYCLES", "1")
    with pytest.raises(ValidationError, match="RENTED_POD_SSH_ENFORCE_AFTER_CYCLES"):
        Settings(_env_file=None)
    monkeypatch.setenv("RENTED_POD_SSH_ENFORCE_AFTER_CYCLES", "0")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
    monkeypatch.setenv("RENTED_POD_SSH_ENFORCE_AFTER_CYCLES", "3")
    explicit = Settings(_env_file=None)
    with patch.object(rented_pod_ssh, "settings", explicit):
        assert rented_pod_ssh.enforce_after_cycles() == 3
    # Equal to the notify threshold is the default made explicit.
    monkeypatch.setenv("RENTED_POD_SSH_ENFORCE_AFTER_CYCLES", "2")
    assert Settings(_env_file=None).RENTED_POD_SSH_ENFORCE_AFTER_CYCLES == 2
    # The check is against the configured notify threshold, not the shipped one.
    monkeypatch.setenv("RENTED_POD_SSH_PROBE_CYCLES", "4")
    monkeypatch.setenv("RENTED_POD_SSH_ENFORCE_AFTER_CYCLES", "3")
    with pytest.raises(ValidationError, match="RENTED_POD_SSH_ENFORCE_AFTER_CYCLES"):
        Settings(_env_file=None)


def test_is_enforced_reads_the_flag_the_streak_and_the_threshold():
    unhealthy = RentedPodSshVerdict(
        pod_id=POD_ID, container_name="pod_1", ssh_port=SSH_PORT, healthy=False,
        faults=[FAULT_TCP_REFUSED], consecutive_cycles=2, report=True,
    )
    with enforcement(enabled=False):
        assert rented_pod_ssh.is_enforced(unhealthy) is False
    with enforcement(enabled=True):
        assert rented_pod_ssh.is_enforced(unhealthy) is True
        assert rented_pod_ssh.is_enforced(replace(unhealthy, consecutive_cycles=1)) is False
        assert rented_pod_ssh.is_enforced(replace(unhealthy, healthy=True, consecutive_cycles=0)) is False
        assert rented_pod_ssh.is_enforced(replace(unhealthy, report_queued=True)) is False
        assert rented_pod_ssh.is_enforced(replace(unhealthy, report=False)) is False
        keys_only = replace(
            unhealthy,
            faults=[FAULT_AUTHORIZED_KEYS_UNREADABLE],
            boot_id_changed=False,
        )
        assert rented_pod_ssh.is_enforced(keys_only) is False
        assert rented_pod_ssh.is_enforced(replace(keys_only, boot_id_changed=True)) is True
    with enforcement(enabled=True, after_cycles=3):
        assert rented_pod_ssh.is_enforced(unhealthy) is False
