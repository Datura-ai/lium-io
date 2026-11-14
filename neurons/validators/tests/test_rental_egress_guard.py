"""Running rental containers get the link-local egress drop that builds already have."""

from __future__ import annotations

import ipaddress
import shlex
from types import SimpleNamespace

import pytest
from services.task import pipeline_factory
from services.task.checks import rental_egress_guard
from services.task.checks.rental_egress_guard import (
    LINK_LOCAL_CIDR,
    RentalEgressGuardCheck,
    link_local_drop_script,
    rental_subnets,
)
from services.task.pipeline_factory import (
    _RENTAL_EGRESS_GUARD_IDLE,
    _RENTAL_EGRESS_GUARD_RENTED,
    PipelineFactory,
)


def _ssh(subnets="192.0.2.0/24 ", helper_rc=0, inspect_rc=0):
    calls: list[str] = []

    async def run(cmd, *args, **kwargs):
        calls.append(cmd)
        if "network inspect" in cmd:
            return SimpleNamespace(exit_status=inspect_rc, stdout=subnets, stderr="")
        return SimpleNamespace(exit_status=helper_rc, stdout="", stderr="DOCKER-USER chain not found" if helper_rc else "")

    return SimpleNamespace(run=run, calls=calls)


def _helper_calls(ssh) -> list[str]:
    return [c for c in ssh.calls if "--cap-add=NET_ADMIN" in c]


def test_script_drops_link_local_from_the_rental_subnet_except_dns_idempotently():
    script = link_local_drop_script(["192.0.2.0/24"])
    for proto in ("tcp", "udp"):
        rule = f"DOCKER-USER -s 192.0.2.0/24 -d {LINK_LOCAL_CIDR} -p {proto} ! --dport 53 -j DROP"
        # -C first: a re-apply inserts nothing new
        assert f"$IPT -C {rule} 2>/dev/null || $IPT -I {rule}" in script
    assert "exit 3" in script  # no DOCKER-USER chain is reported, not ignored
    # only the two link-local DROP rules: private ranges stay reachable
    assert script.count(" -d ") == 4 and script.count(f"-d {LINK_LOCAL_CIDR} ") == 4


def test_only_ipv4_subnets_are_used():
    net = ipaddress.ip_network(LINK_LOCAL_CIDR)
    assert net.is_link_local and net.prefixlen == 16
    assert rental_subnets("192.0.2.0/24 fd00::/64 junk;rm 192.0.2.0/24 ") == ["192.0.2.0/24"]


@pytest.mark.asyncio
async def test_applies_the_drop_through_a_host_network_helper(context_factory):
    ssh = _ssh()
    result = await RentalEgressGuardCheck().run(context_factory(ssh=ssh))

    assert result.passed is True
    assert result.event.reason_code == "RENTAL_EGRESS_GUARD_APPLIED"
    [helper] = _helper_calls(ssh)
    assert "--network=host" in helper and helper.startswith("timeout ")
    assert shlex.quote(link_local_drop_script(["192.0.2.0/24"])) in helper


@pytest.mark.asyncio
async def test_runs_once_per_interval(context_factory):
    check = RentalEgressGuardCheck()
    ssh = _ssh()
    ctx = context_factory(ssh=ssh)
    await check.run(ctx)
    second = await check.run(ctx)

    assert len(_helper_calls(ssh)) == 1
    assert second.event.reason_code == "RENTAL_EGRESS_GUARD_SKIPPED"


@pytest.mark.asyncio
async def test_no_rental_network_runs_no_helper(context_factory):
    ssh = _ssh(subnets="", inspect_rc=1)
    result = await RentalEgressGuardCheck().run(context_factory(ssh=ssh))

    assert result.passed is True
    assert _helper_calls(ssh) == []


@pytest.mark.asyncio
async def test_helper_failure_is_fail_open(context_factory):
    ssh = _ssh(helper_rc=3)
    result = await RentalEgressGuardCheck().run(context_factory(ssh=ssh))

    assert result.passed is True
    assert result.event.reason_code == "RENTAL_EGRESS_GUARD_NOT_APPLIED"


@pytest.mark.asyncio
async def test_ssh_error_is_fail_open(context_factory):
    async def boom(*a, **k):
        raise ConnectionError("ssh dropped")

    result = await RentalEgressGuardCheck().run(context_factory(ssh=SimpleNamespace(run=boom)))

    assert result.passed is True
    assert result.event.reason_code == "RENTAL_EGRESS_GUARD_NOT_APPLIED"


@pytest.mark.parametrize("build", [PipelineFactory.build_checks, PipelineFactory.build_fast_path_checks])
def test_never_runs_before_a_scored_step(build):
    checks = build()
    names = [type(check).__name__ for check in checks]
    # rented: right before TenantEnforcementCheck, which halts a rented node; idle: after ScoreCheck
    assert names[checks.index(_RENTAL_EGRESS_GUARD_RENTED) + 1] == "TenantEnforcementCheck"
    assert names[checks.index(_RENTAL_EGRESS_GUARD_IDLE) - 1] == "ScoreCheck"
    assert names.count("RentalEgressGuardCheck") == 2


def test_not_in_the_dry_run():
    names = [type(check).__name__ for check in PipelineFactory.build_dry_run_checks()]
    assert "RentalEgressGuardCheck" not in names


def test_both_positions_share_one_cadence():
    assert _RENTAL_EGRESS_GUARD_RENTED._last_run is _RENTAL_EGRESS_GUARD_IDLE._last_run
    assert pipeline_factory._RENTAL_EGRESS_GUARD_LAST_RUN is _RENTAL_EGRESS_GUARD_IDLE._last_run


@pytest.mark.asyncio
@pytest.mark.parametrize("rented", [True, False])
async def test_each_position_guards_only_its_own_nodes(context_factory, monkeypatch, rented):
    rental = SimpleNamespace(pods=["pod"]) if rented else None
    monkeypatch.setattr(rental_egress_guard, "rental_of_this_node", lambda ctx: rental)
    ssh = _ssh()
    ctx = context_factory(ssh=ssh)

    other = await RentalEgressGuardCheck(rented_only=not rented).run(ctx)
    assert _helper_calls(ssh) == [] and other.event.reason_code == "RENTAL_EGRESS_GUARD_SKIPPED"
    own = await RentalEgressGuardCheck(rented_only=rented).run(ctx)
    assert len(_helper_calls(ssh)) == 1 and own.event.reason_code == "RENTAL_EGRESS_GUARD_APPLIED"


@pytest.mark.asyncio
async def test_skipped_without_cadence_stamp_when_task_time_is_short(context_factory, monkeypatch):
    check = RentalEgressGuardCheck()
    ssh = _ssh()
    left = rental_egress_guard.HELPER_TIMEOUT_SECONDS + rental_egress_guard.BUDGET_MARGIN_SECONDS - 1
    monkeypatch.setattr(rental_egress_guard, "task_seconds_left", lambda ctx: left)

    result = await check.run(context_factory(ssh=ssh))

    assert result.passed is True and result.event.reason_code == "RENTAL_EGRESS_GUARD_SKIPPED"
    assert ssh.calls == [] and check._last_run == {}
    # the next cycle with time left applies it
    monkeypatch.setattr(rental_egress_guard, "task_seconds_left", lambda ctx: 600)
    await check.run(context_factory(ssh=ssh))
    assert len(_helper_calls(ssh)) == 1


@pytest.mark.asyncio
async def test_a_stalled_inspect_is_cut_off_before_the_deadline_and_the_run_is_kept(context_factory, monkeypatch):
    import asyncio
    from unittest.mock import AsyncMock

    from services.task.pipeline import CheckResult, Pipeline

    margin = rental_egress_guard.BUDGET_MARGIN_SECONDS
    # enough for the pre-check, then 0.2 s of budget for the remote calls
    left = iter([rental_egress_guard.HELPER_TIMEOUT_SECONDS + margin, margin + 0.2])
    monkeypatch.setattr(rental_egress_guard, "task_seconds_left", lambda ctx: next(left))

    async def stalled(*a, **k):
        await asyncio.Event().wait()

    finalized = []

    class _Finalize:
        check_id = "finalize"
        fatal = True

        async def run(self, ctx):
            finalized.append(True)
            return CheckResult(passed=True, event=rental_egress_guard.render_message(
                rental_egress_guard.Msg.SKIPPED, ctx=ctx, check_id=self.check_id, what={}))

    guard = RentalEgressGuardCheck()
    ctx = context_factory(ssh=SimpleNamespace(run=stalled))
    started = asyncio.get_running_loop().time()
    ok, events, _ = await asyncio.wait_for(Pipeline([guard, _Finalize()], AsyncMock()).run(ctx), timeout=5)

    assert asyncio.get_running_loop().time() - started < 2
    assert ok is True and finalized == [True]
    assert events[0].reason_code == "RENTAL_EGRESS_GUARD_SKIPPED"
    assert guard._last_run == {}  # not a run: the next cycle tries again
