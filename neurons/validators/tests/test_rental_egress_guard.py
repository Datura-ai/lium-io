"""Running rental containers get the link-local egress drop that builds already have."""

from __future__ import annotations

import ipaddress
import os
import shlex
import subprocess
from types import SimpleNamespace

import pytest
from services.task import pipeline_factory
from services.task.checks import rental_egress_guard
from services.task.checks.rental_egress_guard import (
    CHAINS,
    LINK_LOCAL_CIDR,
    RentalEgressGuardCheck,
    legacy_bridge_inventory,
    legacy_network_ids,
    link_local_cleanup_script,
    helper_command,
    link_local_drop_script,
    locked_script,
    rental_bridge,
)
from services.task.pipeline_factory import (
    _RENTAL_EGRESS_GUARD_IDLE,
    _RENTAL_EGRESS_GUARD_RENTED,
    PipelineFactory,
)


NET_ID = "0123456789abcdef" * 4
LEGACY_NET_ID = "fedcba9876543210" * 4
LEGACY = f"{LEGACY_NET_ID} "  # the running legacy pod's network, as `docker inspect` lists it


ABSENT_NETWORK = "Error response from daemon: network lium-rentals not found"
POD_A, POD_B = "pod_00000000-0000-4000-8000-00000000000a", "pod_00000000-0000-4000-8000-00000000000b"


def _ssh(
    network=f"{NET_ID} <no value>",
    helper_rc=0,
    inspect_rc=0,
    legacy="",
    network_stderr=None,
    pods_rc=0,
    pods_stderr="",
    names="",
    ps_rc=0,
    legacy_nets=f"{LEGACY_NET_ID} bridge docker0",
    legacy_nets_rc=0,
    legacy_nets_stderr="",
):
    calls: list[str] = []
    if network_stderr is None:
        network_stderr = ABSENT_NETWORK if inspect_rc else ""

    async def run(cmd, *args, **kwargs):
        calls.append(cmd)
        if "docker ps" in cmd:
            return SimpleNamespace(exit_status=ps_rc, stdout=names, stderr="Cannot connect to the Docker daemon" if ps_rc else "")
        if "network inspect" in cmd and ".Driver" in cmd:
            return SimpleNamespace(exit_status=legacy_nets_rc, stdout=legacy_nets, stderr=legacy_nets_stderr)
        if "network inspect" in cmd:
            return SimpleNamespace(exit_status=inspect_rc, stdout=network, stderr=network_stderr)
        if "docker inspect" in cmd:
            return SimpleNamespace(exit_status=pods_rc, stdout=legacy, stderr=pods_stderr)
        return SimpleNamespace(exit_status=helper_rc, stdout="", stderr="DOCKER-USER chain not found" if helper_rc else "")

    return SimpleNamespace(run=run, calls=calls)


def _helper_calls(ssh) -> list[str]:
    return [c for c in ssh.calls if "nsenter -t 1 " in c]


def test_script_rebuilds_the_rules_in_a_fresh_chain_and_swaps_it_in():
    script = link_local_drop_script("br-0123456789ab", ["docker0"])
    for match in ("-i br-0123456789ab", "-i docker0"):
        for proto in ("tcp", "udp"):
            assert f"$IPT -A $NEW {match} -d {LINK_LOCAL_CIDR} -p {proto} ! --dport 53 -j DROP" in script
    # the new chain is flushed and filled before its jump goes in; then every jump to the old one is removed
    order = ["$IPT -F $NEW", "$IPT -A $NEW", "$IPT -I DOCKER-USER -j $NEW", "$IPT -D DOCKER-USER -j $OLD", "$IPT -F $OLD"]
    assert [script.index(step) for step in order] == sorted(script.index(step) for step in order)
    assert all(chain in script for chain in CHAINS)
    assert " -s " not in script  # never by source: a renter with NET_ADMIN can add another address
    assert "exit 3" in script  # no DOCKER-USER chain is reported, not ignored
    # only link-local DROP rules: private ranges stay reachable
    assert script.count(" -d ") == 4 and script.count(f"-d {LINK_LOCAL_CIDR} ") == 4


def test_an_ended_legacy_pod_or_a_recreated_network_leaves_no_rule():
    before = link_local_drop_script("br-aaaaaaaaaaaa", ["docker0"])
    after = link_local_drop_script("br-bbbbbbbbbbbb", [])
    assert "br-aaaaaaaaaaaa" not in after and "docker0" not in after
    assert "-i br-aaaaaaaaaaaa" in before


def test_bridge_name_and_legacy_sources_are_parsed_safely():
    net = ipaddress.ip_network(LINK_LOCAL_CIDR)
    assert net.is_link_local and net.prefixlen == 16
    assert rental_bridge(f"{NET_ID} <no value>") == "br-0123456789ab"
    assert rental_bridge(f"{NET_ID} lium0") == "lium0"  # com.docker.network.bridge.name is honoured
    assert rental_bridge(f"{NET_ID} bad;name") == "br-0123456789ab"
    assert rental_bridge("") is None and rental_bridge("Error: no such network") is None
    assert legacy_network_ids(f"{LEGACY_NET_ID} junk;rm {LEGACY_NET_ID} 198.51.100.7 ") == [LEGACY_NET_ID]
    nets = f"{LEGACY_NET_ID} bridge docker0\n{NET_ID} bridge <no value>\n{'a' * 64} host <no value>\n"
    assert legacy_bridge_inventory(SimpleNamespace(exit_status=0, stdout=nets, stderr="")) == ["docker0", "br-0123456789ab"]
    with pytest.raises(rental_egress_guard.InventoryUnknown):
        legacy_bridge_inventory(SimpleNamespace(exit_status=0, stdout="garbage bridge docker0\n", stderr=""))


@pytest.mark.asyncio
async def test_applies_the_drop_in_the_hosts_namespaces_without_any_image(context_factory):
    ssh = _ssh()
    result = await RentalEgressGuardCheck().run(context_factory(ssh=ssh))

    assert result.passed is True
    assert result.event.reason_code == "RENTAL_EGRESS_GUARD_APPLIED"
    [helper] = _helper_calls(ssh)
    assert helper.startswith(f"HOST_NS={shlex.quote(rental_egress_guard.HOST_NS)} exec timeout ")
    # no container: a registry refusing an anonymous pull (Docker Hub's rate limit) can never leave a node unguarded
    assert not any("docker run" in call or "docker pull" in call for call in ssh.calls)
    assert helper.endswith(helper_command(link_local_drop_script("br-0123456789ab", [])))


@pytest.mark.asyncio
async def test_the_lock_and_iptables_resolve_in_the_hosts_root_not_the_executors(context_factory):
    ssh = _ssh()
    await RentalEgressGuardCheck().run(context_factory(ssh=ssh))

    [helper] = _helper_calls(ssh)
    # without the host root two executors on one host take a lock in their own roots: two locks, interleaved swaps
    assert rental_egress_guard.HOST_LOCK.startswith("/proc/1/root/") and rental_egress_guard.HOST_LOCK in helper
    assert "-r" in shlex.split(rental_egress_guard.HOST_NS)


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
    assert helper_command(link_local_cleanup_script()) in helper
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
async def test_the_rented_position_guards_only_rented_nodes_and_completion_guards_every_node(
    context_factory, monkeypatch, rented
):
    rental = SimpleNamespace(pods=[SimpleNamespace(docker_container_name="pod")]) if rented else None
    monkeypatch.setattr(rental_egress_guard, "rental_of_this_node", lambda ctx: rental)
    ssh = _ssh()
    ctx = context_factory(ssh=ssh)

    last_run: dict = {}  # shared, as the factory's two instances share it
    at_tenant_enforcement = await RentalEgressGuardCheck(rented_only=True, last_run=last_run).run(ctx)
    assert len(_helper_calls(ssh)) == (1 if rented else 0)
    assert at_tenant_enforcement.event.reason_code == (
        "RENTAL_EGRESS_GUARD_APPLIED" if rented else "RENTAL_EGRESS_GUARD_SKIPPED"
    )
    at_completion = await RentalEgressGuardCheck(last_run=last_run).run(ctx)
    assert len(_helper_calls(ssh)) == 1
    assert at_completion.event.reason_code == (
        "RENTAL_EGRESS_GUARD_SKIPPED" if rented else "RENTAL_EGRESS_GUARD_APPLIED"
    )  # a rented node guarded at TenantEnforcementCheck is within the shared cadence at completion


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
    sink = AsyncMock()
    started = asyncio.get_running_loop().time()
    ok, events, _ = await asyncio.wait_for(Pipeline([guard, _Finalize()], sink).run(ctx), timeout=5)

    assert asyncio.get_running_loop().time() - started < 2
    assert ok is True and finalized == [True]
    assert sink.emit.await_args_list[-1].args[0].reason_code == "RENTAL_EGRESS_GUARD_SKIPPED"
    assert guard._last_run == {}  # not a run: the next cycle tries again


@pytest.mark.parametrize("rented", [True, False])
@pytest.mark.parametrize("exit_", ["passed", "fatal", "halt"])
@pytest.mark.asyncio
async def test_the_guard_runs_once_per_cycle_however_the_pipeline_ends(context_factory, monkeypatch, rented, exit_):
    """A fatal check that ends every cycle early (a node short of ports, a GPU count without its details) must not
    leave a rebooted rented node unguarded or keep the last legacy rental's rules: completion guards every node.
    A rented halt has guarded it inside TenantEnforcementCheck, so completion is within the shared cadence. The
    guard's event never becomes the run's, so the verdict is the run's own."""
    from unittest.mock import AsyncMock

    from services.task.pipeline import CheckResult, Pipeline

    rental = SimpleNamespace(pods=[SimpleNamespace(docker_container_name="pod")]) if rented else None
    monkeypatch.setattr(rental_egress_guard, "rental_of_this_node", lambda ctx: rental)
    ssh = _ssh()
    last_run: dict = {}
    rented_guard = RentalEgressGuardCheck(rented_only=True, last_run=last_run)
    ran = []

    class _Step:
        check_id = "step"
        fatal = True

        async def run(self, ctx):
            ran.append(self.check_id)
            if self.check_id == "step" and exit_ == "halt":
                await rented_guard.run(ctx)  # TenantEnforcementCheck's own guard call before its halt
            return CheckResult(
                passed=exit_ != "fatal" or self.check_id != "step",
                halt=exit_ == "halt" and self.check_id == "step",
                event=rental_egress_guard.render_message(
                    rental_egress_guard.Msg.SKIPPED, ctx=ctx, check_id=self.check_id, what={}
                ),
            )

    class _Scored(_Step):
        check_id = "scored"

    sink = AsyncMock()
    completion = RentalEgressGuardCheck(last_run=last_run)
    ok, events, _ = await Pipeline([_Step(), completion, _Scored()], sink).run(context_factory(ssh=ssh))

    assert ok is (exit_ != "fatal")
    assert ran == (["step", "scored"] if exit_ == "passed" else ["step"])
    assert [event.check_id for event in events] == ran
    assert sink.emit.await_args_list[-1].args[0].check_id == RentalEgressGuardCheck.check_id
    [helper] = _helper_calls(ssh)
    assert helper.endswith(helper_command(link_local_drop_script("br-0123456789ab", [])))
    assert len(last_run) == 1


@pytest.mark.parametrize("template", ["MISMATCH", "BINDING_UNVERIFIED"])
@pytest.mark.parametrize("rented", [True, False])
@pytest.mark.asyncio
async def test_no_completion_step_touches_a_host_refused_as_another_accounts(
    context_factory, monkeypatch, rented, template
):
    """A hostname pointed at another provider's machine is refused by NodeAccountCheck; that machine may still accept
    the validator's key, so the guard's NET_ADMIN helper must never run there. The same when the machine could not be
    confirmed as the account's (DAH-6539: the account read failed, which a relay can cause)."""
    from unittest.mock import AsyncMock

    from services.task.messages import NodeAccountMessages
    from services.task.pipeline import CheckResult, Pipeline

    rental = SimpleNamespace(pods=[SimpleNamespace(docker_container_name="pod")]) if rented else None
    monkeypatch.setattr(rental_egress_guard, "rental_of_this_node", lambda ctx: rental)
    ssh = _ssh()

    class _NodeAccount:
        check_id = "node.account"
        fatal = True

        async def run(self, ctx):
            return CheckResult(
                passed=False,
                event=rental_egress_guard.render_message(
                    getattr(NodeAccountMessages, template), ctx=ctx, check_id=self.check_id, what={}
                ),
            )

    sink = AsyncMock()
    ok, events, _ = await Pipeline([_NodeAccount(), RentalEgressGuardCheck()], sink).run(context_factory(ssh=ssh))

    assert ok is False and events[-1].reason_code == getattr(NodeAccountMessages, template).reason
    assert ssh.calls == []
    assert RentalEgressGuardCheck.check_id not in [call.args[0].check_id for call in sink.emit.await_args_list]


@pytest.mark.asyncio
async def test_a_running_legacy_rental_off_the_rental_network_is_guarded(context_factory, monkeypatch):
    _legacy_rental(monkeypatch)
    # pod and filler names, labelled or not, parked ones too; anything else on the host is not a rental
    names = f"{POD_A} filler_00000000-0000-4000-8000-00000000000c {POD_B}__prev executor-executor-1 pod_x my-app "
    ssh = _ssh(network="", inspect_rc=1, legacy=LEGACY, names=names)

    result = await RentalEgressGuardCheck(rented_only=True).run(context_factory(ssh=ssh))

    [inspect] = [c for c in ssh.calls if c.startswith("/usr/bin/docker inspect")]
    assert inspect.endswith(f"{POD_A} filler_00000000-0000-4000-8000-00000000000c {POD_B}__prev")
    assert ".State.Running" in inspect
    [helper] = _helper_calls(ssh)
    assert helper_command(link_local_drop_script(None, ["docker0"])) in helper
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
    assert apply_cmd.endswith(helper_command(link_local_drop_script("br-0123456789ab", [])))
    assert cleanup_cmd.endswith(helper_command(link_local_cleanup_script()))
    # the host file the previous release locked too, reached through PID 1's root
    assert rental_egress_guard.HOST_LOCK == "/proc/1/root/run/lock/lium-rental-egress/apply.lock"
    assert rental_egress_guard.HELPER_RUN_SECONDS < rental_egress_guard.HELPER_TIMEOUT_SECONDS


def _legacy_rental(monkeypatch):
    rental = SimpleNamespace(pods=[SimpleNamespace(docker_container_name=name) for name in (POD_A, POD_B)])
    monkeypatch.setattr(rental_egress_guard, "rental_of_this_node", lambda ctx: rental)


@pytest.mark.parametrize("network_present", [True, False])
@pytest.mark.parametrize("rented_first", [True, False])
@pytest.mark.asyncio
async def test_two_registrations_of_one_host_reconcile_the_same_host_wide_set(
    context_factory, monkeypatch, network_present, rented_first
):
    """A registration with no rental of its own must not drop another registration's legacy pod rules."""
    rented_ctx_key = "rented-miner"
    monkeypatch.setattr(
        rental_egress_guard,
        "rental_of_this_node",
        lambda ctx: SimpleNamespace(pods=[SimpleNamespace(docker_container_name=POD_A)])
        if ctx.miner_hotkey == rented_ctx_key
        else None,
    )
    network = dict(network=f"{NET_ID} <no value>") if network_present else dict(network="", inspect_rc=1)
    ssh = _ssh(**network, legacy=LEGACY, names=f"{POD_A} ")  # one host, both registrations see it
    rented = context_factory(ssh=ssh).model_copy(update={"miner_hotkey": rented_ctx_key})
    idle = context_factory(ssh=ssh)
    order = [(rented, True), (idle, False)] if rented_first else [(idle, False), (rented, True)]

    for ctx, rented_only in order:
        await RentalEgressGuardCheck(rented_only=rented_only).run(ctx)

    expected = helper_command(link_local_drop_script("br-0123456789ab" if network_present else None, ["docker0"]))
    assert [expected in helper for helper in _helper_calls(ssh)] == [True, True]


@pytest.mark.parametrize(
    ("ssh_kwargs", "outcome"),
    [
        (dict(legacy=f"{LEGACY_NET_ID} {LEGACY_NET_ID} "), "applied"),
        (dict(network="", inspect_rc=1), "cleaned"),
        (dict(legacy=LEGACY, pods_rc=1, pods_stderr="Error: No such object: pod-b"), "applied"),
        (dict(legacy="", pods_rc=1, pods_stderr="Cannot connect to the Docker daemon at unix:///var/run/docker.sock"), "kept"),
        (dict(legacy="", pods_rc=1, pods_stderr=""), "kept"),
        (dict(network="", inspect_rc=1, network_stderr="Error: context deadline exceeded"), "kept"),
        (dict(network="", inspect_rc=1, network_stderr=""), "kept"),
        (dict(network="garbage", inspect_rc=0), "kept"),
        (dict(ps_rc=1), "kept"),
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
        "container-list-failed",
    ],
)
@pytest.mark.asyncio
async def test_an_unknown_inventory_keeps_the_rules_and_is_retried(context_factory, monkeypatch, ssh_kwargs, outcome):
    """Explicit absence may reconcile; a failed read must never be taken for an empty host."""
    _legacy_rental(monkeypatch)
    if "network" not in ssh_kwargs:
        ssh_kwargs = dict(ssh_kwargs, network="", inspect_rc=1)  # legacy pods only, so the pods read decides
    ssh_kwargs = dict(ssh_kwargs, names=f"{POD_A} {POD_B} ")
    check = RentalEgressGuardCheck(rented_only=True)
    ssh = _ssh(**ssh_kwargs)

    result = await check.run(context_factory(ssh=ssh))

    helpers = _helper_calls(ssh)
    if outcome == "kept":
        assert helpers == [] and check._last_run == {}
        assert result.event.reason_code == "RENTAL_EGRESS_GUARD_SKIPPED"
    elif outcome == "cleaned":
        [helper] = helpers
        assert helper_command(link_local_cleanup_script()) in helper
    else:
        [helper] = helpers
        assert "-j DROP" in helper and result.event.reason_code == "RENTAL_EGRESS_GUARD_APPLIED"


# A stand-in for iptables: chains are files in $STATE, DOCKER-USER's jumps are lines of DOCKER-USER, and
# FAIL=<flag> makes that operation fail the way a busy or broken xtables would.
_IPTABLES_STUB = r"""#!/bin/sh
op=$1; [ "$op" = -n ] && { op=$2; shift; }
chain=$2
[ "$op" = "${FAIL:-}" ] && [ "$chain" != DOCKER-USER ] && { echo "iptables: Resource temporarily unavailable." >&2; exit 4; }
case "$op" in
  -N|-F|-A|-I|-D|-X) mutating=1 ;;
esac
case "$op" in
  -L) [ -f "$STATE/$chain" ] || { echo "iptables: No chain/target/match by that name." >&2; exit 1; } ;;
  -S) echo "-N DOCKER-USER"; grep -v RETURN "$STATE/DOCKER-USER" | sed 's/^/-A DOCKER-USER -j /' ;;
  -C) grep -qx "$4" "$STATE/DOCKER-USER" ;;
  -D) grep -qx "$4" "$STATE/DOCKER-USER" || exit 1
      awk -v c="$4" '$0 == c && !d { d = 1; next } 1' "$STATE/DOCKER-USER" > "$STATE/t"; mv "$STATE/t" "$STATE/DOCKER-USER" ;;
  -N) [ -f "$STATE/$chain" ] && exit 1; : > "$STATE/$chain" ;;
  -F) grep -qx "$chain" "$STATE/DOCKER-USER" && [ -s "$STATE/$chain" ] && touch "$STATE/FLUSHED_ACTIVE"
      [ -f "$STATE/$chain" ] && : > "$STATE/$chain" ;;
  -A) shift 2; echo "$*" >> "$STATE/$chain" ;;
  -I) { echo "$4"; cat "$STATE/DOCKER-USER"; } > "$STATE/t"; mv "$STATE/t" "$STATE/DOCKER-USER" ;;
  -X) [ -s "$STATE/$chain" ] && exit 1; rm "$STATE/$chain" ;;
  *) exit 2 ;;
esac
rc=$?
if [ -n "${mutating:-}" ] && [ -n "${DIE_AFTER:-}" ]; then
  n=$(( $(cat "$STATE/count" 2>/dev/null || echo 0) + 1 )); echo $n > "$STATE/count"
  [ "$n" -eq "$DIE_AFTER" ] && kill -9 "$(cat "$STATE/run_pid")"
fi
exit $rc
"""


def _run_cleanup(tmp_path, chains_present: bool, fail: str = "") -> tuple[int, list[str]]:
    bin_dir, state = tmp_path / "bin", tmp_path / "state"
    bin_dir.mkdir()
    state.mkdir()
    for name in ("iptables-nft", "iptables-legacy"):
        (bin_dir / name).write_text(_IPTABLES_STUB)
        (bin_dir / name).chmod(0o755)
    (state / "DOCKER-USER").write_text("".join(f"{chain}\n" for chain in CHAINS) if chains_present else "RETURN\n")
    if chains_present:
        for chain in CHAINS:
            (state / chain).write_text("-j DROP\n")
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "STATE": str(state), "FAIL": fail}
    done = subprocess.run(["sh", "-c", link_local_cleanup_script()], env=env, capture_output=True, text=True)
    return done.returncode, sorted(p.name for p in state.iterdir())


@pytest.mark.parametrize(
    ("chains_present", "fail", "rc", "left"),
    [
        (False, "", 0, ["DOCKER-USER"]),  # confirmed absent: nothing to do
        (True, "", 0, ["DOCKER-USER"]),  # removed: jumps, rules and chains gone
        (True, "-X", 4, ["DOCKER-USER", *CHAINS]),  # a chain that will not go is reported
        (True, "-F", 4, ["DOCKER-USER", *CHAINS]),
        (True, "-L", 4, ["DOCKER-USER", *CHAINS]),  # a failed read is not taken for absence
    ],
    ids=["absent", "removed", "delete-failed", "flush-failed", "read-failed"],
)
def test_cleanup_script_accepts_absence_and_reports_a_failed_removal(tmp_path, chains_present, fail, rc, left):
    returncode, state = _run_cleanup(tmp_path, chains_present, fail)
    assert returncode == rc and state == left


@pytest.mark.parametrize("script", ["apply", "cleanup"])
def test_a_failed_chain_probe_is_reported_not_taken_for_an_absent_chain(tmp_path, script):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("iptables-nft", "iptables-legacy", "iptables"):
        (bin_dir / name).write_text("#!/bin/sh\necho 'nsenter: reassociate to namespace: Operation not permitted' >&2\nexit 1\n")
        (bin_dir / name).chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    body = link_local_drop_script("br-0123456789ab", []) if script == "apply" else link_local_cleanup_script()

    done = subprocess.run(["sh", "-c", body], env=env, capture_output=True, text=True)

    assert done.returncode != 0
    assert "Operation not permitted" in done.stderr


_DENIED = "#!/bin/sh\necho 'nsenter: reassociate to namespace: Operation not permitted' >&2\nexit 1\n"
_NO_CHAIN = "#!/bin/sh\necho 'iptables: No chain/target/match by that name.' >&2\nexit 1\n"


@pytest.mark.parametrize(
    ("nft", "script", "rc"),
    [(_DENIED, "apply", 4), (_DENIED, "cleanup", 4), (_NO_CHAIN, "apply", 3), (_NO_CHAIN, "cleanup", 0)],
    ids=["denied-apply", "denied-cleanup", "absent-apply", "absent-cleanup"],
)
def test_a_later_backends_absence_never_hides_an_earlier_failed_probe(tmp_path, nft, script, rc):
    """The last legacy rental stops and the nft probe is refused: legacy and plain iptables saying "no chain" is not
    proof the host has none of our jumps, so cleanup reports it and is retried. Only absence on every backend is 0."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("iptables-nft", nft), ("iptables-legacy", _NO_CHAIN), ("iptables", _NO_CHAIN)):
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    body = link_local_drop_script("br-0123456789ab", []) if script == "apply" else link_local_cleanup_script()

    done = subprocess.run(["sh", "-c", body], env=env, capture_output=True, text=True)

    assert done.returncode == rc, done.stderr
    assert ("Operation not permitted" in done.stderr) == (nft is _DENIED)


@pytest.mark.parametrize("script", ["apply", "cleanup"])
def test_a_host_iptables_without_the_nft_or_legacy_names_is_used(tmp_path, script):
    """The rules run with the host's own binaries now: an older host ships plain `iptables` only."""
    bin_dir, state = tmp_path / "bin", tmp_path / "state"
    bin_dir.mkdir()
    state.mkdir()
    for name in ("iptables-nft", "iptables-legacy"):  # shadow any on this machine: absent on the modelled host
        (bin_dir / name).write_text("#!/bin/sh\nexit 127\n")
        (bin_dir / name).chmod(0o755)
    (bin_dir / "iptables").write_text(_IPTABLES_STUB)
    (bin_dir / "iptables").chmod(0o755)
    (state / "DOCKER-USER").write_text("RETURN\n")
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "STATE": str(state)}
    body = link_local_drop_script("br-0123456789ab", []) if script == "apply" else link_local_cleanup_script()

    done = subprocess.run(["sh", "-c", body], env=env, capture_output=True, text=True)

    assert done.returncode == 0, done.stderr
    if script == "apply":
        [(_, rules)] = _jumps_and_rules(state)
        assert rules == [f"-i br-0123456789ab -d {LINK_LOCAL_CIDR} -p {p} ! --dport 53 -j DROP" for p in ("tcp", "udp")]


def _jumps_and_rules(state) -> list[tuple[str, list[str]]]:
    jumps = [j for j in (state / "DOCKER-USER").read_text().split() if j in CHAINS]
    return [(j, (state / j).read_text().splitlines()) for j in jumps]


@pytest.mark.parametrize("die_after", range(1, 11))
def test_a_swap_killed_after_any_mutation_is_recovered_without_flushing_the_protection_in_force(tmp_path, die_after):
    """A is in force for the old bridge; a swap to the recreated bridge dies after its n-th mutation, then the next
    run completes. No run may flush a chain DOCKER-USER still jumps to, and the end state is one jump to a chain
    holding exactly the new bridge's rules."""
    bin_dir, state = tmp_path / "bin", tmp_path / "state"
    bin_dir.mkdir()
    state.mkdir()
    for name in ("iptables-nft", "iptables-legacy"):
        (bin_dir / name).write_text(_IPTABLES_STUB)
        (bin_dir / name).chmod(0o755)
    a, b = CHAINS
    old_rules = [f"-i br-aaaaaaaaaaaa -d {LINK_LOCAL_CIDR} -p {p} ! --dport 53 -j DROP" for p in ("tcp", "udp")]
    new_rules = [f"-i br-bbbbbbbbbbbb -d {LINK_LOCAL_CIDR} -p {p} ! --dport 53 -j DROP" for p in ("tcp", "udp")]
    (state / "DOCKER-USER").write_text(f"{a}\nRETURN\n")
    (state / a).write_text("".join(f"{r}\n" for r in old_rules))
    (state / b).write_text("")
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "STATE": str(state)}
    script = link_local_drop_script("br-bbbbbbbbbbbb", [])

    # the stub kills the pid the run started as, never its own $PPID: a shell that execs its last command (bash -c) makes
    # that command's parent the test runner itself
    killable = f'echo $$ > "$STATE/run_pid"; {script}'
    first = subprocess.run(["sh", "-c", killable], env={**env, "DIE_AFTER": str(die_after)}, capture_output=True)
    in_force = _jumps_and_rules(state)
    assert in_force and in_force[0][1] in (old_rules, new_rules)  # something still protects after the kill
    second = subprocess.run(["sh", "-c", script], env=env, capture_output=True, text=True)

    assert first.returncode in (0, -9) and second.returncode == 0, second.stderr
    assert not (state / "FLUSHED_ACTIVE").exists()
    [(_, rules)] = _jumps_and_rules(state)
    assert rules == new_rules


def test_a_failed_read_of_docker_user_changes_nothing(tmp_path):
    bin_dir, state = tmp_path / "bin", tmp_path / "state"
    bin_dir.mkdir()
    state.mkdir()
    stub = _IPTABLES_STUB.replace('  -S) echo', '  -S) [ -n "${FAIL_S:-}" ] && exit 4; echo')
    for name in ("iptables-nft", "iptables-legacy"):
        (bin_dir / name).write_text(stub)
        (bin_dir / name).chmod(0o755)
    a, _b = CHAINS
    (state / "DOCKER-USER").write_text(f"{a}\n")
    (state / a).write_text("-j DROP\n")
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "STATE": str(state), "FAIL_S": "1"}

    done = subprocess.run(["sh", "-c", link_local_drop_script("br-bbbbbbbbbbbb", [])], env=env, capture_output=True)

    assert done.returncode == 4 and _jumps_and_rules(state) == [(a, ["-j DROP"])]


# An iptables that records each call and whether the host lock was free at that moment; -C and -D find nothing.
# With SLOW=<op>, that call marks its start, runs for 3 s and marks its end.
_LOCK_PROBE_STUB = r"""#!/bin/sh
flock -n "$LOCK" true && echo UNLOCKED >> "$STATE/log"
echo "$*" >> "$STATE/log"
[ "$1" = "${SLOW:-}" ] && { echo $$ > "$STATE/started"; sleep 3; touch "$STATE/done"; }
case "$1" in -C|-D) exit 1 ;; esac
exit 0
"""


def _probe(tmp_path, monkeypatch, slow=""):
    bin_dir, state = tmp_path / "bin", tmp_path / "state"
    bin_dir.mkdir()
    state.mkdir()
    for name in ("iptables-nft", "iptables-legacy"):
        (bin_dir / name).write_text(_LOCK_PROBE_STUB)
        (bin_dir / name).chmod(0o755)
    lock = str(tmp_path / "egress.lock")
    monkeypatch.setattr(rental_egress_guard, "HOST_LOCK", lock)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "STATE": str(state), "LOCK": lock, "SLOW": slow}
    return state, lock, env


@pytest.mark.parametrize("script", ["apply", "cleanup"])
def test_every_iptables_call_of_the_helper_runs_under_the_host_lock(tmp_path, monkeypatch, script):
    state, _lock, env = _probe(tmp_path, monkeypatch)
    body = link_local_drop_script("br-0123456789ab", ["docker0"]) if script == "apply" else ""
    body = body or link_local_cleanup_script()

    done = subprocess.run(["sh", "-c", locked_script(body)], env=env, capture_output=True, text=True)

    log = (state / "log").read_text().splitlines()
    assert done.returncode == 0, done.stderr
    assert len(log) > 3 and "UNLOCKED" not in log


def test_a_held_lock_is_waited_for_and_never_skipped(tmp_path, monkeypatch):
    state, lock, env = _probe(tmp_path, monkeypatch)
    monkeypatch.setattr(rental_egress_guard, "LOCK_WAIT_SECONDS", 1)
    with open(lock, "w") as held:
        import fcntl

        fcntl.flock(held, fcntl.LOCK_EX)
        script = locked_script(link_local_cleanup_script())
        done = subprocess.run(["sh", "-c", script], env=env, capture_output=True, text=True)

    assert done.returncode == 5 and not (state / "log").exists()


@pytest.mark.parametrize("kills", ["group", "shell-only"])
def test_the_helper_bound_cannot_free_the_lock_while_a_mutation_runs(tmp_path, monkeypatch, kills):
    """Killed at its bound mid-swap: GNU timeout kills the whole group, busybox's (modelled by --foreground) only the
    shell, leaving the running iptables call; that call still holds the lock until it ends, and in the container the
    host it ends on its own. Either way the lock is never free while a mutation runs."""
    import time

    state, lock, env = _probe(tmp_path, monkeypatch, slow="-I")
    monkeypatch.setattr(rental_egress_guard, "HELPER_RUN_SECONDS", 1)
    command = helper_command(link_local_drop_script("br-0123456789ab", []))
    if kills == "shell-only":
        command = command.replace("timeout -s KILL", "timeout --foreground -s KILL", 1)

    # no pipes: an orphaned call holding them would make run() wait for it
    done = subprocess.run(["sh", "-c", command], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert done.returncode != 0 and (state / "started").exists()

    deadline = time.monotonic() + 6
    while subprocess.run(["flock", "-n", lock, "true"]).returncode != 0 and time.monotonic() < deadline:
        time.sleep(0.05)
    finished_when_freed = (state / "done").exists()
    calls_when_freed = (state / "log").read_text()
    time.sleep(3.5)  # past the slow call's own end

    # once the lock is free nothing touches iptables any more: the call had ended, or it was killed with the helper
    assert (state / "log").read_text() == calls_when_freed
    assert finished_when_freed or not (state / "done").exists()
    # busybox kills only the shell, and so does a plain uutils timeout (Ubuntu 26.04's): the call outlived it and kept
    # the lock until it ended. Only GNU timeout kills the whole group, so the call never finished
    version = subprocess.run(["timeout", "--version"], capture_output=True, text=True).stdout.splitlines()[:1]
    assert finished_when_freed == (kills == "shell-only" or "GNU coreutils" not in str(version)), version


# The host root as the executor reaches it through nsenter: only iptables is there, no shell or coreutils.
_HOST_NSENTER_STUB = r"""#!/bin/sh
while [ "$1" != -- ]; do shift; done; shift
echo "$1" >> "$STATE/host_exec"
case "$1" in
  iptables*) exec "$@" ;;
  *) echo "nsenter: failed to execute $1: No such file or directory" >&2; exit 127 ;;
esac
"""


@pytest.mark.parametrize("script", ["apply", "cleanup"])
def test_the_helper_runs_only_iptables_in_the_host_root(tmp_path, monkeypatch, script):
    """A host root without /bin/sh or timeout still gets the rules: the shell, lock and bound are the executor's."""
    bin_dir, state = tmp_path / "bin", tmp_path / "state"
    bin_dir.mkdir()
    state.mkdir()
    for name, body in (("iptables-nft", _IPTABLES_STUB), ("iptables-legacy", _IPTABLES_STUB), ("nsenter", _HOST_NSENTER_STUB)):
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    (state / "DOCKER-USER").write_text("RETURN\n")
    monkeypatch.setattr(rental_egress_guard, "HOST_LOCK", str(tmp_path / "lock" / "apply.lock"))
    body = link_local_drop_script("br-0123456789ab", []) if script == "apply" else link_local_cleanup_script()
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "STATE": str(state)}

    done = subprocess.run(["bash", "-c", rental_egress_guard._helper(body)], env=env, capture_output=True, text=True)

    assert done.returncode == 0, done.stderr
    assert set((state / "host_exec").read_text().split()) == {"iptables-nft"}
    if script == "apply":
        [(_, rules)] = _jumps_and_rules(state)
        assert rules == [f"-i br-0123456789ab -d {LINK_LOCAL_CIDR} -p {p} ! --dport 53 -j DROP" for p in ("tcp", "udp")]
