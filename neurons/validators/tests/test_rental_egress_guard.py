"""Running rental containers get the link-local egress drop that builds already have."""

from __future__ import annotations

import ipaddress
import shlex
from types import SimpleNamespace

import pytest
from services.task import pipeline_factory
from services.task.checks import rental_egress_guard
from services.task.checks.rental_egress_guard import (
    CHAINS,
    LINK_LOCAL_CIDR,
    RentalEgressGuardCheck,
    legacy_sources,
    link_local_cleanup_script,
    link_local_drop_script,
    rental_bridge,
)
from services.task.pipeline_factory import (
    _RENTAL_EGRESS_GUARD_IDLE,
    _RENTAL_EGRESS_GUARD_RENTED,
    PipelineFactory,
)


NET_ID = "0123456789abcdef" * 4


ABSENT_NETWORK = "Error response from daemon: network lium-rentals not found"


def _ssh(
    network=f"{NET_ID} <no value>",
    helper_rc=0,
    inspect_rc=0,
    legacy="",
    network_stderr=None,
    pods_rc=0,
    pods_stderr="",
):
    calls: list[str] = []
    if network_stderr is None:
        network_stderr = ABSENT_NETWORK if inspect_rc else ""

    async def run(cmd, *args, **kwargs):
        calls.append(cmd)
        if "network inspect" in cmd:
            return SimpleNamespace(exit_status=inspect_rc, stdout=network, stderr=network_stderr)
        if "docker inspect" in cmd:
            return SimpleNamespace(exit_status=pods_rc, stdout=legacy, stderr=pods_stderr)
        return SimpleNamespace(exit_status=helper_rc, stdout="", stderr="DOCKER-USER chain not found" if helper_rc else "")

    return SimpleNamespace(run=run, calls=calls)


def _helper_calls(ssh) -> list[str]:
    return [c for c in ssh.calls if "--cap-add=NET_ADMIN" in c]


def test_script_rebuilds_the_rules_in_a_fresh_chain_and_swaps_it_in():
    script = link_local_drop_script("br-0123456789ab", ["198.51.100.7/32"])
    for match in ("-i br-0123456789ab", "-s 198.51.100.7/32"):
        for proto in ("tcp", "udp"):
            assert f"$IPT -A $NEW {match} -d {LINK_LOCAL_CIDR} -p {proto} ! --dport 53 -j DROP" in script
    # the new chain is flushed and filled before its jump goes in; then every jump to the old one is removed
    order = ["$IPT -F $NEW", "$IPT -A $NEW", "$IPT -I DOCKER-USER -j $NEW", "$IPT -D DOCKER-USER -j $OLD", "$IPT -F $OLD"]
    assert [script.index(step) for step in order] == sorted(script.index(step) for step in order)
    assert all(chain in script for chain in CHAINS)
    assert "-s 192.0.2" not in script  # no subnet match: a subnet Docker reuses elsewhere is never caught
    assert "exit 3" in script  # no DOCKER-USER chain is reported, not ignored
    # only link-local DROP rules: private ranges stay reachable
    assert script.count(" -d ") == 4 and script.count(f"-d {LINK_LOCAL_CIDR} ") == 4


def test_an_ended_legacy_pod_or_a_recreated_network_leaves_no_rule():
    before = link_local_drop_script("br-aaaaaaaaaaaa", ["198.51.100.7/32"])
    after = link_local_drop_script("br-bbbbbbbbbbbb", [])
    assert "br-aaaaaaaaaaaa" not in after and "198.51.100.7" not in after
    assert "-i br-aaaaaaaaaaaa" in before


def test_bridge_name_and_legacy_sources_are_parsed_safely():
    net = ipaddress.ip_network(LINK_LOCAL_CIDR)
    assert net.is_link_local and net.prefixlen == 16
    assert rental_bridge(f"{NET_ID} <no value>") == "br-0123456789ab"
    assert rental_bridge(f"{NET_ID} lium0") == "lium0"  # com.docker.network.bridge.name is honoured
    assert rental_bridge(f"{NET_ID} bad;name") == "br-0123456789ab"
    assert rental_bridge("") is None and rental_bridge("Error: no such network") is None
    assert legacy_sources("198.51.100.7 fd00::2 junk;rm 198.51.100.7  ") == ["198.51.100.7/32"]


@pytest.mark.asyncio
async def test_applies_the_drop_through_a_host_network_helper(context_factory):
    ssh = _ssh()
    result = await RentalEgressGuardCheck().run(context_factory(ssh=ssh))

    assert result.passed is True
    assert result.event.reason_code == "RENTAL_EGRESS_GUARD_APPLIED"
    [helper] = _helper_calls(ssh)
    assert "--network=host" in helper and helper.startswith("timeout ")
    # the lock is taken inside the helper timeout and held around the whole docker run, so the apply is one unit
    lock = f"flock -w {rental_egress_guard.LOCK_WAIT_SECONDS} {rental_egress_guard.HOST_LOCK} /usr/bin/docker run"
    assert lock in helper and helper.index("timeout ") < helper.index("flock ")
    assert shlex.quote(link_local_drop_script("br-0123456789ab", [])) in helper


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
async def test_nothing_left_to_guard_removes_our_rules_pulling_a_pruned_image(context_factory):
    ssh = _ssh(network="", inspect_rc=1)
    result = await RentalEgressGuardCheck().run(context_factory(ssh=ssh))

    assert result.passed is True
    [helper] = _helper_calls(ssh)
    assert "--pull=never" not in helper and shlex.quote(link_local_cleanup_script()) in helper
    script = link_local_cleanup_script()
    for chain in CHAINS:
        assert f"$IPT -D DOCKER-USER -j {chain}" in script and f"$IPT -F {chain}" in script
    assert "-j DROP" not in script and script.endswith("exit 0")


@pytest.mark.asyncio
async def test_a_failed_cleanup_is_retried_next_cycle(context_factory):
    check = RentalEgressGuardCheck()
    ssh = _ssh(network="", inspect_rc=1, helper_rc=3)
    ctx = context_factory(ssh=ssh)
    first = await check.run(ctx)
    await check.run(ctx)

    assert first.passed is True and first.event.reason_code == "RENTAL_EGRESS_GUARD_SKIPPED"
    assert len(_helper_calls(ssh)) == 2


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
    # rented: inside TenantEnforcementCheck's rented halt; idle: after ScoreCheck
    [tenant] = [check for check in checks if type(check).__name__ == "TenantEnforcementCheck"]
    assert tenant.egress_guard is _RENTAL_EGRESS_GUARD_RENTED
    assert names[checks.index(_RENTAL_EGRESS_GUARD_IDLE) - 1] == "ScoreCheck"
    assert names.count("RentalEgressGuardCheck") == 1


def test_not_in_the_dry_run():
    checks = PipelineFactory.build_dry_run_checks()
    assert "RentalEgressGuardCheck" not in [type(check).__name__ for check in checks]
    assert all(getattr(check, "egress_guard", None) is None for check in checks)


def test_both_positions_share_one_cadence():
    assert _RENTAL_EGRESS_GUARD_RENTED._last_run is _RENTAL_EGRESS_GUARD_IDLE._last_run
    assert pipeline_factory._RENTAL_EGRESS_GUARD_LAST_RUN is _RENTAL_EGRESS_GUARD_IDLE._last_run


@pytest.mark.asyncio
@pytest.mark.parametrize("rented", [True, False])
async def test_each_position_guards_only_its_own_nodes(context_factory, monkeypatch, rented):
    rental = SimpleNamespace(pods=[SimpleNamespace(docker_container_name="pod")]) if rented else None
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


@pytest.mark.asyncio
async def test_a_running_legacy_rental_off_the_rental_network_is_guarded(context_factory, monkeypatch):
    rental = SimpleNamespace(pods=[SimpleNamespace(docker_container_name="pod-a"), SimpleNamespace(docker_container_name="pod b")])
    monkeypatch.setattr(rental_egress_guard, "rental_of_this_node", lambda ctx: rental)
    ssh = _ssh(network="", inspect_rc=1, legacy="198.51.100.7 ")

    result = await RentalEgressGuardCheck(rented_only=True).run(context_factory(ssh=ssh))

    [inspect] = [c for c in ssh.calls if c.startswith("/usr/bin/docker inspect")]
    assert "pod-a" in inspect and "'pod b'" in inspect and ".State.Running" in inspect
    [helper] = _helper_calls(ssh)
    assert shlex.quote(link_local_drop_script(None, ["198.51.100.7/32"])) in helper
    assert result.event.reason_code == "RENTAL_EGRESS_GUARD_APPLIED"


@pytest.mark.asyncio
async def test_cadence_is_per_miner_and_endpoint_not_per_uuid(context_factory):
    check = RentalEgressGuardCheck()
    ssh = _ssh()
    first = context_factory(ssh=ssh)
    other_miner = first.model_copy(update={"miner_hotkey": "other-miner"})
    other_host = first.model_copy(update={"executor": first.executor.model_copy(update={"address": "203.0.113.9"})})

    for ctx in (first, other_miner, other_host, first):
        await check.run(ctx)

    # same UUID, three hosts: each guarded once; the repeat of the first is within its window
    assert len(_helper_calls(ssh)) == 3


@pytest.mark.asyncio
async def test_apply_and_cleanup_take_the_same_host_lock(context_factory):
    """Two miners reporting one host serialize on one lock file, whichever of apply or cleanup each runs."""
    applied, cleaned = _ssh(), _ssh(network="", inspect_rc=1)
    await RentalEgressGuardCheck().run(context_factory(ssh=applied))
    await RentalEgressGuardCheck().run(context_factory(ssh=cleaned))

    [apply_cmd], [cleanup_cmd] = _helper_calls(applied), _helper_calls(cleaned)
    prefix = f"flock -w {rental_egress_guard.LOCK_WAIT_SECONDS} {rental_egress_guard.HOST_LOCK} "
    assert prefix in apply_cmd and prefix in cleanup_cmd
    assert rental_egress_guard.HOST_LOCK.startswith("/run/lock/")  # on the host, outside any container


def _legacy_rental(monkeypatch):
    rental = SimpleNamespace(pods=[SimpleNamespace(docker_container_name=name) for name in ("pod-a", "pod-b")])
    monkeypatch.setattr(rental_egress_guard, "rental_of_this_node", lambda ctx: rental)


@pytest.mark.parametrize(
    ("ssh_kwargs", "outcome"),
    [
        (dict(legacy="198.51.100.7 198.51.100.8 "), "applied"),
        (dict(network="", inspect_rc=1), "cleaned"),
        (dict(legacy="198.51.100.7 ", pods_rc=1, pods_stderr="Error: No such object: pod-b"), "applied"),
        (dict(legacy="", pods_rc=1, pods_stderr="Cannot connect to the Docker daemon at unix:///var/run/docker.sock"), "kept"),
        (dict(legacy="", pods_rc=1, pods_stderr=""), "kept"),
        (dict(network="", inspect_rc=1, network_stderr="Error: context deadline exceeded"), "kept"),
        (dict(network="", inspect_rc=1, network_stderr=""), "kept"),
        (dict(network="garbage", inspect_rc=0), "kept"),
    ],
    ids=[
        "inventory-read",
        "network-absent-nothing-left",
        "partial-one-pod-gone",
        "pods-inspect-failed",
        "pods-inspect-failed-silently",
        "network-inspect-failed",
        "network-inspect-failed-silently",
        "network-inspect-unreadable",
    ],
)
@pytest.mark.asyncio
async def test_an_unknown_inventory_keeps_the_rules_and_is_retried(context_factory, monkeypatch, ssh_kwargs, outcome):
    """Explicit absence may reconcile; a failed read must never be taken for an empty host."""
    _legacy_rental(monkeypatch)
    if "network" not in ssh_kwargs:
        ssh_kwargs = dict(ssh_kwargs, network="", inspect_rc=1)  # legacy pods only, so the pods read decides
    check = RentalEgressGuardCheck(rented_only=True)
    ssh = _ssh(**ssh_kwargs)

    result = await check.run(context_factory(ssh=ssh))

    helpers = _helper_calls(ssh)
    if outcome == "kept":
        assert helpers == [] and check._last_run == {}
        assert result.event.reason_code == "RENTAL_EGRESS_GUARD_SKIPPED"
    elif outcome == "cleaned":
        [helper] = helpers
        assert shlex.quote(link_local_cleanup_script()) in helper
    else:
        [helper] = helpers
        assert "-j DROP" in helper and result.event.reason_code == "RENTAL_EGRESS_GUARD_APPLIED"
