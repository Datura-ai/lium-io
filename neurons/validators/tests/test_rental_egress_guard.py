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
    helper_command,
    legacy_bridge_inventory,
    legacy_network_ids,
    link_local_cleanup_script,
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
    labelled="",
    labelled_rc=0,
    helper_stdout="",
):
    calls: list[str] = []
    if network_stderr is None:
        network_stderr = ABSENT_NETWORK if inspect_rc else ""

    async def run(cmd, *args, **kwargs):
        calls.append(cmd)
        if _is_generation_read(cmd):
            return SimpleNamespace(exit_status=0, stdout="0\n", stderr="")
        if "network ls" in cmd:
            return SimpleNamespace(exit_status=labelled_rc, stdout=labelled, stderr="daemon error" if labelled_rc else "")
        if "docker ps" in cmd:
            return SimpleNamespace(exit_status=ps_rc, stdout=names, stderr="Cannot connect to the Docker daemon" if ps_rc else "")
        if "network inspect" in cmd and ".Driver" in cmd:
            return SimpleNamespace(exit_status=legacy_nets_rc, stdout=legacy_nets, stderr=legacy_nets_stderr)
        if "network inspect" in cmd:
            return SimpleNamespace(exit_status=inspect_rc, stdout=network, stderr=network_stderr)
        if "docker inspect" in cmd:
            return SimpleNamespace(exit_status=pods_rc, stdout=legacy, stderr=pods_stderr)
        # a list is one exit per helper run; its last value repeats
        rc = (helper_rc.pop(0) if len(helper_rc) > 1 else helper_rc[0]) if isinstance(helper_rc, list) else helper_rc
        return SimpleNamespace(exit_status=rc, stdout=helper_stdout, stderr="DOCKER-USER chain not found" if rc else "")

    return SimpleNamespace(run=run, calls=calls)


def _is_generation_read(cmd: str) -> bool:
    return rental_egress_guard.HOST_GENERATION in cmd and "G=$(cat" not in cmd


def _helper_calls(ssh) -> list[str]:
    """The apply or cleanup helper runs, without the generation reads before each inventory."""
    return [c for c in ssh.calls if "nsenter -t 1 " in c and not _is_generation_read(c)]


def test_script_rebuilds_the_rules_in_a_fresh_chain_and_swaps_it_in():
    full = link_local_drop_script("br-0123456789ab", ["docker0"])
    script = full[: full.index("if [ ! -d ")]  # the rules; the boot copy after them is the same text again
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
    root = f"HOST_NS={shlex.quote(rental_egress_guard.HOST_NS)} " + helper_command(link_local_drop_script("br-0123456789ab", []), "0")
    assert helper.startswith('if ! { [ "$(id -u)" = 0 ] && { [ "$(readlink /proc/1/ns/mnt)" ') and helper.endswith(
        f"; fi; {root}"
    )
    assert "then /usr/bin/docker run --rm --privileged --pid=host " in helper
    # no container on a root executor: a registry refusing an anonymous pull (Docker Hub's rate limit) never matters
    assert not any("docker run" in call or "docker pull" in call for call in ssh.calls if call != helper and not _is_generation_read(call))


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
    assert helper_command(link_local_cleanup_script(), "0").removeprefix("exec ") in helper
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
    ok, events, _ = await asyncio.wait_for(Pipeline([_RootLogin(), guard, _Finalize()], sink).run(ctx), timeout=5)

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
    pipeline = Pipeline([_RootLogin(), _Step(), completion, _Scored()], sink)
    ok, events, _ = await pipeline.run(context_factory(ssh=ssh))

    assert ok is (exit_ != "fatal")
    assert ran == (["step", "scored"] if exit_ == "passed" else ["step"])
    assert [event.check_id for event in events] == ["login", *ran]
    assert sink.emit.await_args_list[-1].args[0].check_id == RentalEgressGuardCheck.check_id
    [helper] = _helper_calls(ssh)
    assert helper_command(link_local_drop_script("br-0123456789ab", []), "0").removeprefix("exec ") in helper
    assert len(last_run) == 1


class _RootLogin:
    """The run's login check, confirming root: completion steps run only after one."""

    check_id = "login"
    fatal = True

    async def run(self, ctx):
        from services.task.messages import LoginUserMessages
        from services.task.pipeline import CheckResult

        return CheckResult(
            passed=True,
            event=rental_egress_guard.render_message(LoginUserMessages.ROOT, ctx=ctx, check_id=self.check_id, what={}),
        )


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
    assert helper_command(link_local_drop_script(None, ["docker0"]), "0").removeprefix("exec ") in helper
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
    assert helper_command(link_local_drop_script("br-0123456789ab", []), "0").removeprefix("exec ") in apply_cmd
    assert helper_command(link_local_cleanup_script(), "0").removeprefix("exec ") in cleanup_cmd
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

    expected = helper_command(link_local_drop_script("br-0123456789ab" if network_present else None, ["docker0"]), "0").removeprefix("exec ")
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
        assert helper_command(link_local_cleanup_script(), "0").removeprefix("exec ") in helper
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
  -C) grep -qx "$4" "$STATE/$chain" 2>/dev/null || exit 1 ;;  # 1 is "no such rule", as iptables answers
  -D) grep -qx "$4" "$STATE/$chain" || exit 1
      awk -v c="$4" '$0 == c && !d { d = 1; next } 1' "$STATE/$chain" > "$STATE/t"; mv "$STATE/t" "$STATE/$chain" ;;
  -N) [ -f "$STATE/$chain" ] && exit 1; : > "$STATE/$chain" ;;
  -F) grep -qx "$chain" "$STATE/DOCKER-USER" && [ -s "$STATE/$chain" ] && touch "$STATE/FLUSHED_ACTIVE"
      [ -f "$STATE/$chain" ] && : > "$STATE/$chain" ;;
  -A) shift 2; echo "$*" >> "$STATE/$chain" ;;
  -I) { echo "$4"; cat "$STATE/$chain"; } > "$STATE/t"; mv "$STATE/t" "$STATE/$chain" ;;
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
    root_id = '#!/bin/sh\n[ "$1" = -u ] && echo 0 || echo root\n'  # the executor container runs as root
    for name, body in (
        ("iptables-nft", _IPTABLES_STUB),
        ("iptables-legacy", _IPTABLES_STUB),
        ("nsenter", _HOST_NSENTER_STUB),
        ("id", root_id),
    ):
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


@pytest.mark.asyncio
async def test_a_pod_bridge_whose_pod_is_stopped_or_not_started_yet_is_guarded():
    # no pod runs: the labelled bridge of a stopped or just-created pod is still in the rules (DAH-6737)
    ssh = _ssh(labelled=f"{LEGACY_NET_ID}\n", legacy_nets=f"{LEGACY_NET_ID} bridge <no value>")

    guarded, error = await rental_egress_guard.apply_rental_egress_guard(ssh)

    assert error is None and guarded["legacy_bridges"] == [f"br-{LEGACY_NET_ID[:12]}"]
    [helper] = _helper_calls(ssh)
    assert f"-i br-{LEGACY_NET_ID[:12]} " in helper


@pytest.mark.asyncio
async def test_an_unreadable_pod_bridge_list_leaves_the_rules_as_they_are():
    ssh = _ssh(labelled_rc=1)
    with pytest.raises(rental_egress_guard.InventoryUnknown):
        await rental_egress_guard.apply_rental_egress_guard(ssh)
    assert not _helper_calls(ssh)


# dockerd as a non-root executor reaches it: records the run, then runs the command after the image as the container would
_DOCKER_STUB = r"""#!/bin/sh
printf '%s\n' "$*" >> "$STATE/docker_run"
while [ "$1" != "docker.io/library/alpine:3.19" ]; do shift; done; shift
exec "$@"
"""


@pytest.mark.parametrize("script", ["apply", "cleanup"])
def test_an_executor_not_running_as_root_gets_the_guard_through_the_privileged_helper(tmp_path, monkeypatch, script):
    # DAH-6737: the host lock, nsenter and iptables need root; a non-root executor reaches dockerd, so the same script
    # runs in the validator's alpine helper, privileged with the host's PIDs, and still uses only the host's iptables
    bin_dir, state = tmp_path / "bin", tmp_path / "state"
    bin_dir.mkdir()
    state.mkdir()
    for name, body in (
        ("iptables-nft", _IPTABLES_STUB),
        ("iptables-legacy", _IPTABLES_STUB),
        ("nsenter", _HOST_NSENTER_STUB),
        ("id", '#!/bin/sh\n[ "$1" = -u ] && echo 1000 || echo provider\n'),
        ("docker", _DOCKER_STUB),
    ):
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    (state / "DOCKER-USER").write_text("RETURN\n")
    body = link_local_drop_script("br-0123456789ab", []) if script == "apply" else link_local_cleanup_script()
    monkeypatch.setattr(rental_egress_guard, "HOST_LOCK", str(tmp_path / "lock" / "apply.lock"))
    command = rental_egress_guard._helper(body).replace("/usr/bin/docker", str(bin_dir / "docker"))
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "STATE": str(state)}

    done = subprocess.run(["sh", "-c", command], env=env, capture_output=True, text=True)

    assert done.returncode == 0, done.stderr
    [run] = (state / "docker_run").read_text().splitlines()
    assert run.startswith("run --rm --privileged --pid=host --network host docker.io/library/alpine:3.19 /bin/sh -c ")
    assert set((state / "host_exec").read_text().split()) == {"iptables-nft"}
    if script == "apply":
        [(_, rules)] = _jumps_and_rules(state)
        assert rules == [f"-i br-0123456789ab -d {LINK_LOCAL_CIDR} -p {p} ! --dport 53 -j DROP" for p in ("tcp", "udp")]


@pytest.mark.parametrize("missing", ["host-pid1", "timeout"])
def test_a_root_executor_that_cannot_reach_the_host_itself_uses_the_helper(tmp_path, monkeypatch, missing):
    # prod 19 Nov: "/proc/1/root/run: Read-only file system" (PID 1 is the executor's own) and "timeout: not found"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "id").write_text('#!/bin/sh\n[ "$1" = -u ] && echo 0 || echo root\n')
    (bin_dir / "docker").write_text('#!/bin/sh\necho helper > "$STATE"\n')
    for name in ("id", "docker"):
        (bin_dir / name).chmod(0o755)
    host_run = tmp_path / "host" / "run"
    host_run.mkdir(parents=True)
    if missing == "host-pid1":
        if os.geteuid() == 0:
            pytest.skip("root writes a read-only directory")
        host_run.chmod(0o555)
    monkeypatch.setattr(rental_egress_guard, "HOST_LOCK", str(host_run / "lock" / "apply.lock"))
    command = rental_egress_guard._helper("touch ran").replace("/usr/bin/docker", str(bin_dir / "docker"))
    path = f"{bin_dir}:{os.environ['PATH']}" if missing == "host-pid1" else str(bin_dir)
    if missing == "host-pid1":
        (bin_dir / "nsenter").write_text("#!/bin/sh\n")
        (bin_dir / "nsenter").chmod(0o755)
    env = {**os.environ, "PATH": path, "STATE": str(tmp_path / "state")}

    done = subprocess.run(["/bin/sh", "-c", command], env=env, capture_output=True, text=True, cwd=tmp_path)
    host_run.chmod(0o755)

    assert done.returncode == 0, done.stderr
    assert (tmp_path / "state").read_text().strip() == "helper" and not (tmp_path / "ran").exists()


def test_a_root_executor_runs_no_container(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("id", "nsenter"):
        (bin_dir / name).write_text('#!/bin/sh\n[ "$1" = -u ] && echo 0 || echo root\n')
        (bin_dir / name).chmod(0o755)
    monkeypatch.setattr(rental_egress_guard, "HOST_LOCK", str(tmp_path / "host" / "run" / "lock" / "apply.lock"))
    (tmp_path / "host" / "run").mkdir(parents=True)
    command = rental_egress_guard._helper("touch ran").replace("/usr/bin/docker", "false")
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}

    done = subprocess.run(["sh", "-c", command], env=env, capture_output=True, text=True, cwd=tmp_path)

    assert done.returncode == 0, done.stderr
    assert (tmp_path / "ran").exists()


@pytest.mark.parametrize("script", ["apply", "cleanup"])
def test_chains_another_iptables_build_wrote_never_fail_the_guard(tmp_path, script):
    # DAH-6737: an older release's chains, written by a newer iptables-nft, are unreadable for the host's own; the
    # guard never reads, flushes or removes them, so it applies (or cleans up) its own chains regardless
    bin_dir, state = tmp_path / "bin", tmp_path / "state"
    bin_dir.mkdir()
    state.mkdir()
    old = ("LIUM-RENTAL-EGRESS-A", "LIUM-RENTAL-EGRESS-B")
    stub = _IPTABLES_STUB.replace(
        'chain=$2\n',
        'chain=$2\ncase "$chain $4" in *LIUM-RENTAL-EGRESS-*) '
        'echo "iptables v1.8.7 (nf_tables): chain is incompatible, use \'nft\' tool." >&2; exit 1;; esac\n',
    )
    for name in ("iptables-nft", "iptables-legacy"):
        (bin_dir / name).write_text(stub)
        (bin_dir / name).chmod(0o755)
    (state / "DOCKER-USER").write_text("".join(f"{c}\n" for c in old))
    for chain in old:
        (state / chain).write_text("-j DROP\n")
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "STATE": str(state)}
    body = link_local_drop_script("br-bbbbbbbbbbbb", []) if script == "apply" else link_local_cleanup_script()

    done = subprocess.run(["sh", "-c", body], env=env, capture_output=True, text=True)

    assert done.returncode == 0, done.stderr
    assert not set(old) & set(CHAINS)
    if script == "apply":
        assert _jumps_and_rules(state)[0][0] in CHAINS
    # a jump this iptables cannot remove is reported apart from our own rules (F-12)
    assert "old jump to LIUM-RENTAL-EGRESS-A not detached" in done.stdout


def _host(tmp_path, monkeypatch):
    """A host with the iptables stub as its iptables, the lock and the rule generation in tmp."""
    bin_dir, state = tmp_path / "bin", tmp_path / "state"
    bin_dir.mkdir()
    state.mkdir()
    for name in ("iptables-nft", "iptables-legacy"):
        (bin_dir / name).write_text(_IPTABLES_STUB)
        (bin_dir / name).chmod(0o755)
    (state / "DOCKER-USER").write_text("RETURN\n")
    lock_dir = tmp_path / "lock"
    monkeypatch.setattr(rental_egress_guard, "HOST_LOCK", str(lock_dir / "apply.lock"))
    monkeypatch.setattr(rental_egress_guard, "HOST_GENERATION", str(lock_dir / "generation"))
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "STATE": str(state)}
    return bin_dir, state, env


def test_a_periodic_run_whose_inventory_predates_a_pods_guard_never_removes_its_bridge(tmp_path, monkeypatch):
    """F-14: a periodic run reads the host before pod B's bridge exists; B's pre-start guard then applies B; the
    periodic run's older set must not replace B's rules, so it is refused as stale and reads the host again."""
    _bin, state, env = _host(tmp_path, monkeypatch)
    periodic_generation = "0"  # read before its inventory, as apply_rental_egress_guard does
    b_rules = [f"-i br-bbbbbbbbbbbb -d {LINK_LOCAL_CIDR} -p {p} ! --dport 53 -j DROP" for p in ("tcp", "udp")]

    pre_start = locked_script(link_local_drop_script("br-aaaaaaaaaaaa", ["br-bbbbbbbbbbbb"]), "0")
    assert subprocess.run(["sh", "-c", pre_start], env=env, capture_output=True).returncode == 0
    stale = locked_script(link_local_drop_script("br-aaaaaaaaaaaa", []), periodic_generation)
    late = subprocess.run(["sh", "-c", stale], env=env, capture_output=True, text=True)

    assert late.returncode == rental_egress_guard.STALE_INVENTORY_EXIT, late.stderr
    [(_, rules)] = _jumps_and_rules(state)
    assert set(b_rules) <= set(rules)  # still covered after the late periodic run


@pytest.mark.asyncio
async def test_a_stale_inventory_is_read_again_then_given_up_as_unknown(monkeypatch):
    ssh = _ssh(helper_rc=rental_egress_guard.STALE_INVENTORY_EXIT)
    with pytest.raises(rental_egress_guard.InventoryUnknown):
        await rental_egress_guard.apply_rental_egress_guard(ssh)
    assert len(_helper_calls(ssh)) == rental_egress_guard.INVENTORY_ATTEMPTS
    assert sum(_is_generation_read(c) for c in ssh.calls) == rental_egress_guard.INVENTORY_ATTEMPTS


@pytest.mark.asyncio
async def test_the_pre_start_guard_reads_again_until_concurrent_guards_settle():
    # four creates on one host: the last one loses three lock rounds and must still apply, not be refused
    stale = rental_egress_guard.STALE_INVENTORY_EXIT
    ssh = _ssh(helper_rc=[stale, stale, stale, stale, 0])
    _, error = await rental_egress_guard.apply_rental_egress_guard(ssh, attempts=None)
    assert error is None
    assert len(_helper_calls(ssh)) == 5


def test_the_lock_wait_works_with_busybox_flock_which_has_no_w(tmp_path, monkeypatch):
    # the alpine helper's busybox flock rejects -w: the wait is timeout around a plain flock
    _bin_dir, state, env = _host(tmp_path, monkeypatch)
    real = subprocess.run(["sh", "-c", "command -v flock"], capture_output=True, text=True).stdout.strip()
    (_bin_dir / "flock").write_text(
        f'#!/bin/sh\ncase "$1" in -w) echo "flock: unrecognized option: w" >&2; exit 1;; esac\nexec {real} "$@"\n'
    )
    (_bin_dir / "flock").chmod(0o755)

    done = subprocess.run(
        ["sh", "-c", locked_script(link_local_drop_script("br-0123456789ab", []), "0")], env=env, capture_output=True, text=True
    )

    assert done.returncode == 0, done.stderr
    assert _jumps_and_rules(state)[0][0] in CHAINS


@pytest.mark.parametrize("absent", ["nsenter", "flock", "timeout"])
def test_one_missing_tool_alone_sends_the_executor_to_the_helper(tmp_path, monkeypatch, absent):
    # `command -v a b c` succeeds when only some exist: each tool is checked on its own
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "id").write_text('#!/bin/sh\n[ "$1" = -u ] && echo 0 || echo root\n')
    (bin_dir / "docker").write_text('#!/bin/sh\necho helper > "$STATE"\n')
    for name in ("nsenter", "flock", "timeout"):
        if name != absent:
            (bin_dir / name).write_text("#!/bin/sh\n")
    for f in bin_dir.iterdir():
        f.chmod(0o755)
    (tmp_path / "host" / "run").mkdir(parents=True)
    monkeypatch.setattr(rental_egress_guard, "HOST_LOCK", str(tmp_path / "host" / "run" / "lock" / "apply.lock"))
    command = rental_egress_guard._helper("touch ran").replace("/usr/bin/docker", str(bin_dir / "docker"))
    env = {**os.environ, "PATH": str(bin_dir), "STATE": str(tmp_path / "state")}

    done = subprocess.run(["/bin/bash", "-c", command], env=env, capture_output=True, text=True, cwd=tmp_path)

    assert done.returncode == 0, done.stderr
    assert (tmp_path / "state").read_text().strip() == "helper" and not (tmp_path / "ran").exists()


@pytest.mark.parametrize("script", ["apply", "cleanup"])
def test_a_root_executor_whose_pid1_reaches_no_iptables_falls_back_to_the_helper(tmp_path, monkeypatch, script):
    # a writable private PID 1 root passes the directory check, but no iptables answers there (3): the helper decides
    bin_dir, state = tmp_path / "bin", tmp_path / "state"
    bin_dir.mkdir()
    state.mkdir()
    (bin_dir / "id").write_text('#!/bin/sh\n[ "$1" = -u ] && echo 0 || echo root\n')
    (bin_dir / "nsenter").write_text(_HOST_NSENTER_STUB.replace(
            'iptables*) exec "$@" ;;', 'iptables*) [ -n "$IN_HELPER" ] && exec "$@"; exit 127 ;;'  # not the host's root
        ))
    (bin_dir / "docker").write_text(
        '#!/bin/sh\necho run >> "$STATE/docker_run"\n'
        'while [ "$1" != "docker.io/library/alpine:3.19" ]; do shift; done; shift\nIN_HELPER=1 exec "$@"\n'
    )
    for name in ("iptables-nft", "iptables-legacy"):
        (bin_dir / name).write_text(_IPTABLES_STUB)
    for f in bin_dir.iterdir():
        f.chmod(0o755)
    (state / "DOCKER-USER").write_text("RETURN\n")
    monkeypatch.setattr(rental_egress_guard, "HOST_LOCK", str(tmp_path / "lock" / "apply.lock"))
    monkeypatch.setattr(rental_egress_guard, "HOST_GENERATION", str(tmp_path / "lock" / "generation"))
    body = link_local_drop_script("br-0123456789ab", []) if script == "apply" else link_local_cleanup_script()
    command = rental_egress_guard._helper(body, "0").replace("/usr/bin/docker", str(bin_dir / "docker"))
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "STATE": str(state)}

    done = subprocess.run(["sh", "-c", command], env=env, capture_output=True, text=True)

    assert done.returncode == 0, done.stderr
    assert (state / "docker_run").read_text().split() == ["run"]
    if script == "apply":
        [(_, rules)] = _jumps_and_rules(state)
        assert rules == [f"-i br-0123456789ab -d {LINK_LOCAL_CIDR} -p {p} ! --dport 53 -j DROP" for p in ("tcp", "udp")]


@pytest.mark.parametrize("script", ["apply", "cleanup"])
def test_an_old_releases_docker0_jump_is_detached(tmp_path, monkeypatch, script):
    """F-12: a host upgraded from a release with LIUM-RENTAL-EGRESS-A/B keeps no old jump, so an old docker0 drop does
    not outlive the last rental it guarded. Its chains are never read or flushed."""
    _bin, state, env = _host(tmp_path, monkeypatch)
    (state / "DOCKER-USER").write_text("LIUM-RENTAL-EGRESS-A\nRETURN\n")
    (state / "LIUM-RENTAL-EGRESS-A").write_text("-i docker0 -j DROP\n")
    body = link_local_drop_script("br-0123456789ab", []) if script == "apply" else link_local_cleanup_script()

    done = subprocess.run(["sh", "-c", body], env=env, capture_output=True, text=True)

    assert done.returncode == 0, done.stderr
    assert "LIUM-RENTAL-EGRESS-A" not in (state / "DOCKER-USER").read_text()
    assert (state / "LIUM-RENTAL-EGRESS-A").read_text() == "-i docker0 -j DROP\n"
    assert "old jump" not in done.stdout


def test_a_failed_generation_write_changes_no_rule_and_fails(tmp_path, monkeypatch):
    """F-8: a pre-start whose generation cannot be recorded (a full /run) must not apply rules an older inventory could
    later replace: it changes nothing and fails, so the pod is not started on an unrecorded rule set."""
    _bin, state, env = _host(tmp_path, monkeypatch)
    (tmp_path / "lock").mkdir()
    (tmp_path / "lock" / "generation.tmp").mkdir()  # the atomic write cannot happen
    before = (state / "DOCKER-USER").read_text()

    done = subprocess.run(
        ["sh", "-c", locked_script(link_local_drop_script("br-bbbbbbbbbbbb", []), "0")], env=env, capture_output=True, text=True
    )

    assert done.returncode == 4 and "rule generation not recorded" in done.stderr
    assert (state / "DOCKER-USER").read_text() == before
    # an older periodic run still reads generation 0 and may apply: nothing of the refused run's existed to erase
    assert not (tmp_path / "lock" / "generation").exists()


def test_a_root_executor_with_its_own_pid1_reads_and_writes_the_generation_through_the_helper(tmp_path, monkeypatch):
    """F-16: a root executor in a container without the host's PIDs passes every tool and directory check, and with the
    host network even reaches DOCKER-USER. Its PID 1 shares its mount namespace, so every call (the generation read and
    the apply) takes the helper and the host's one generation; a second guard applies, never stale."""
    bin_dir, state, env = _host(tmp_path, monkeypatch)
    (bin_dir / "id").write_text('#!/bin/sh\n[ "$1" = -u ] && echo 0 || echo root\n')
    (bin_dir / "readlink").write_text("#!/bin/sh\necho 'mnt:[4026531841]'\n")  # PID 1 in the executor's own namespace
    (bin_dir / "nsenter").write_text(_HOST_NSENTER_STUB)
    (bin_dir / "docker").write_text(
        '#!/bin/sh\necho run >> "$STATE/docker_run"\n'
        'while [ "$1" != "docker.io/library/alpine:3.19" ]; do shift; done; shift\nexec "$@"\n'
    )
    for f in bin_dir.iterdir():
        f.chmod(0o755)
    mark = tmp_path / "dockerenv"
    mark.touch()
    monkeypatch.setattr(rental_egress_guard, "CONTAINER_MARK", str(mark))

    def run(script, generation=None):
        command = rental_egress_guard._helper(script, generation).replace("/usr/bin/docker", str(bin_dir / "docker"))
        return subprocess.run(["sh", "-c", command], env=env, capture_output=True, text=True)

    read = f"cat {shlex.quote(rental_egress_guard.HOST_GENERATION)} 2>/dev/null; :"
    for expected in ("0", "1"):
        generation = run(read).stdout.strip() or "0"
        assert generation == expected
        done = run(link_local_drop_script("br-0123456789ab", []), generation)
        assert done.returncode == 0, done.stderr
    assert (state / "docker_run").read_text().split() == ["run"] * 4


def _modelled_host(tmp_path, *, docker_user: bool, systemd: bool = True):
    """A modelled host: the iptables stub, a host root for the boot copy and a systemctl that records daemon-reloads."""
    bin_dir, state, root = tmp_path / "bin", tmp_path / "state", tmp_path / "root"
    for d in (bin_dir, state, root):
        d.mkdir()
    for name in ("iptables-nft", "iptables-legacy"):  # absent on the modelled host: plain iptables only
        (bin_dir / name).write_text("#!/bin/sh\nexit 127\n")
        (bin_dir / name).chmod(0o755)
    (bin_dir / "iptables").write_text(_IPTABLES_STUB)
    (bin_dir / "iptables").chmod(0o755)
    (bin_dir / "systemctl").write_text(
        '#!/bin/sh\ncase "$1 $3" in "show NeedDaemonReload") echo no ;; show*) echo loaded ;; '
        'daemon-reload*) echo x >> "$STATE/reloads" ;; esac\n'
    )
    (bin_dir / "systemctl").chmod(0o755)
    (state / "FORWARD").write_text("")
    if docker_user:
        (state / "DOCKER-USER").write_text("RETURN\n")
    if systemd:
        (root / "run" / "systemd" / "system").mkdir(parents=True)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "STATE": str(state)}
    return state, root, env


@pytest.mark.parametrize("docker_user", [False, True])
def test_docker_user_always_has_a_forward_jump_whoever_made_the_chain(tmp_path, docker_user):
    """No chain: it is created and jumped to. A chain with no FORWARD jump (restored by a host firewall, or left by a
    dockerd that does not manage iptables) gets the jump too, so the rules never sit where no traffic passes."""
    state, root, env = _modelled_host(tmp_path, docker_user=docker_user)
    script = link_local_drop_script("br-0123456789ab", [], str(root))

    for _ in range(2):  # a second run keeps the one jump
        done = subprocess.run(["sh", "-c", script], env=env, capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
    assert (state / "FORWARD").read_text().split() == ["DOCKER-USER"]
    [(_, rules)] = _jumps_and_rules(state)
    assert rules == [f"-i br-0123456789ab -d {LINK_LOCAL_CIDR} -p {p} ! --dport 53 -j DROP" for p in ("tcp", "udp")]


def test_the_rules_are_back_before_dockerd_starts_a_container_after_a_reboot(tmp_path):
    state, root, env = _modelled_host(tmp_path, docker_user=True)
    script = link_local_drop_script("br-0123456789ab", ["docker0"], str(root))

    for _ in range(2):
        done = subprocess.run(["sh", "-c", script], env=env, capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
    dropin = (root / rental_egress_guard.BOOT_DROPIN.lstrip("/")).read_text()
    unit = (root / rental_egress_guard.BOOT_UNIT.lstrip("/")).read_text()
    name = rental_egress_guard.BOOT_UNIT.rsplit("/", 1)[1]
    # After= holds dockerd until the copy ends; Wants=, so a copy that gives up still lets Docker start
    assert f"ExecStart=/bin/sh {rental_egress_guard.BOOT_SCRIPT}" in unit.splitlines()
    assert {f"Wants={name}", f"After={name}"} <= set(dropin.splitlines())
    assert f"Requires={name}" not in dropin.splitlines()
    assert (state / "reloads").read_text().split() == ["x"]  # reloaded once, not on every apply

    # the reboot: no chain survives, dockerd has not run yet; the drop-in's script alone restores the rules
    for chain in [p for p in state.iterdir() if p.name not in ("FORWARD", "reloads")]:
        chain.unlink()
    (state / "FORWARD").write_text("")
    boot = root / rental_egress_guard.BOOT_SCRIPT.lstrip("/")
    restored = subprocess.run(["sh", str(boot)], env=env, capture_output=True, text=True)

    assert restored.returncode == 0, restored.stderr
    assert (state / "FORWARD").read_text().split() == ["DOCKER-USER"]
    [(_, rules)] = _jumps_and_rules(state)
    assert rules == [
        f"-i {i} -d {LINK_LOCAL_CIDR} -p {p} ! --dport 53 -j DROP" for i in ("br-0123456789ab", "docker0") for p in ("tcp", "udp")
    ]


def test_a_boot_copy_holds_the_docker_start_until_the_rules_are_back(tmp_path):
    """dockerd, which restarts rental containers on its own, does not start until the rules are back: a firewall write
    that fails for a while is tried again inside the one start, and Docker starts once the write succeeds."""
    state, root, env = _modelled_host(tmp_path, docker_user=True)
    applied = subprocess.run(
        ["sh", "-c", link_local_drop_script("br-0123456789ab", [], str(root))], env=env, capture_output=True, text=True
    )
    assert applied.returncode == 0, applied.stderr
    unit = (root / rental_egress_guard.BOOT_UNIT.lstrip("/")).read_text()
    # sergej-bot review at 90713e30: the start timeout ends a copy stuck in one call, after the script's own wait
    assert f"TimeoutStartSec={rental_egress_guard.BOOT_WAIT_SECONDS + 60}" in unit.splitlines()
    # a host firewall's start flushes the ruleset: the boot copy writes the rules after it, not in parallel
    assert f"After={rental_egress_guard.BOOT_AFTER}" in unit.splitlines()
    assert "firewalld.service" in rental_egress_guard.BOOT_AFTER.split()
    # sergej-bot review at 957d8e3: Docker's own start settings, the provider's included, are left as they are
    dropin = (root / rental_egress_guard.BOOT_DROPIN.lstrip("/")).read_text()
    assert dropin.splitlines()[0] == "[Unit]" and "[Service]" not in dropin and "Timeout" not in dropin

    # the reboot, with the firewall write failing for a while
    for chain in [p for p in state.iterdir() if p.name not in ("FORWARD", "reloads")]:
        chain.unlink()
    (state / "FORWARD").write_text("")
    iptables = tmp_path / "bin" / "iptables"
    working = iptables.read_text()
    iptables.write_text('#!/bin/sh\necho "write failed" >&2\nexit 4\n')
    boot = root / rental_egress_guard.BOOT_SCRIPT.lstrip("/")
    restoring = subprocess.Popen(["sh", str(boot)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            restoring.wait(timeout=2.5)  # still holding the start: neither failed nor passed without the rules
        iptables.write_text(working)
        assert restoring.wait(timeout=10) == 0
    finally:
        restoring.kill()
    [(_, rules)] = _jumps_and_rules(state)
    assert rules == [f"-i br-0123456789ab -d {LINK_LOCAL_CIDR} -p {p} ! --dport 53 -j DROP" for p in ("tcp", "udp")]
    assert not (root / rental_egress_guard.BOOT_LATE.lstrip("/")).exists()


def test_a_boot_copy_that_cannot_put_the_rules_back_gives_up_and_the_next_apply_reports_it(tmp_path):
    """sergej-bot review at 90713e30: with no iptables after an OS update the copy never succeeds. It gives up after
    its wait, so Docker (Wants=) starts and the validator can reach the host; the mark it leaves is reported by the
    next apply that puts the rules back, and cleared."""
    state, root, env = _modelled_host(tmp_path, docker_user=True)
    script = link_local_drop_script("br-0123456789ab", [], str(root))
    assert subprocess.run(["sh", "-c", script], env=env, capture_output=True, text=True).returncode == 0
    iptables = tmp_path / "bin" / "iptables"
    working = iptables.read_text()
    iptables.write_text("#!/bin/sh\nexit 127\n")
    late = root / rental_egress_guard.BOOT_LATE.lstrip("/")

    gave_up = subprocess.run(
        ["sh", str(root / rental_egress_guard.BOOT_SCRIPT.lstrip("/"))],
        env={**env, "LIUM_EGRESS_BOOT_WAIT": "2"},
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert gave_up.returncode != 0
    assert late.exists()
    iptables.write_text(working)
    reapplied = subprocess.run(["sh", "-c", script], env=env, capture_output=True, text=True)
    assert reapplied.returncode == 0, reapplied.stderr
    assert rental_egress_guard.BOOT_LATE_LINE in reapplied.stdout.splitlines()
    assert not late.exists()


def test_a_docker_start_after_the_last_rental_ended_restores_no_rules(tmp_path):
    """Cleanup leaves the boot copy a no-op under the same path the drop-in names, so a later Docker start puts back no
    rules of a rental that has ended."""
    state, root, env = _modelled_host(tmp_path, docker_user=True)
    applied = subprocess.run(
        ["sh", "-c", link_local_drop_script(None, ["docker0"], str(root))], env=env, capture_output=True, text=True
    )
    assert applied.returncode == 0, applied.stderr

    cleaned = subprocess.run(["sh", "-c", link_local_cleanup_script(str(root))], env=env, capture_output=True, text=True)
    assert cleaned.returncode == 0, cleaned.stderr
    replayed = subprocess.run(
        ["sh", str(root / rental_egress_guard.BOOT_SCRIPT.lstrip("/"))], env=env, capture_output=True, text=True
    )

    assert replayed.returncode == 0, replayed.stderr
    assert not [rules for _, rules in _jumps_and_rules(state) if rules]
    assert (root / rental_egress_guard.BOOT_DROPIN.lstrip("/")).exists()


@pytest.mark.parametrize("host", ["no systemd", "read-only /etc/systemd/system"])
def test_a_host_without_a_systemd_docker_service_is_reported_not_failed(tmp_path, host):
    state, root, env = _modelled_host(tmp_path, docker_user=True, systemd=host != "no systemd")
    if host != "no systemd":  # e.g. NixOS: the unit directory is in the read-only store
        (root / "etc" / "systemd").mkdir(parents=True)
        (root / "etc" / "systemd" / "system").write_text("")  # a file, so mkdir -p fails even as root

    done = subprocess.run(
        ["sh", "-c", link_local_drop_script("br-0123456789ab", [], str(root))], env=env, capture_output=True, text=True
    )

    assert done.returncode == 0, done.stderr
    assert "boot restore not installed" in done.stdout
    [(_, rules)] = _jumps_and_rules(state)
    assert rules  # the runtime rules stay in force
    if host == "no systemd":
        assert not (root / "etc").exists()


def test_a_docker_service_state_that_cannot_be_read_is_not_taken_for_no_docker_service(tmp_path):
    # sergej-bot review at 957d8e3: a failed systemctl query must not pass as "no systemd docker.service", which a pod
    # start accepts; it is reported as not installed, so the pod start refuses
    state, root, env = _modelled_host(tmp_path, docker_user=True)
    (tmp_path / "bin" / "systemctl").write_text("#!/bin/sh\nexit 1\n")

    done = subprocess.run(
        ["sh", "-c", link_local_drop_script("br-0123456789ab", [], str(root))], env=env, capture_output=True, text=True
    )

    assert done.returncode == 0, done.stderr
    assert "boot restore not installed: docker.service state not read" in done.stdout.splitlines()
    assert rental_egress_guard.NO_BOOT_UNIT not in done.stdout


@pytest.mark.asyncio
@pytest.mark.parametrize("on_host", [True, False])
async def test_the_pre_start_guard_requires_the_pods_own_bridge_in_the_applied_rules(on_host):
    # gateway review at 129a43b: a same-name bridge without the pod label is not in the labelled list, so an apply
    # covering only the template must not pass the start; the pod's bridge is inventoried by its ID, or refused
    nets = f"{LEGACY_NET_ID} bridge <no value>" if on_host else ""
    ssh = _ssh(labelled="", legacy_nets=nets, legacy_nets_rc=0 if on_host else 1, legacy_nets_stderr=ABSENT_NETWORK)

    guarded, error = await rental_egress_guard.apply_rental_egress_guard(
        ssh, attempts=None, required_network_id=LEGACY_NET_ID
    )

    if on_host:
        assert error is None and guarded["legacy_bridges"] == [f"br-{LEGACY_NET_ID[:12]}"]
        [helper] = _helper_calls(ssh)
        assert f"-i br-{LEGACY_NET_ID[:12]} " in helper
    else:
        assert error == f"the pod's bridge {LEGACY_NET_ID[:12]} is not in the applied rules"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boot_line, required, refused",
    [
        ("boot restore not installed: directories not created", True, True),
        ("boot restore not installed: docker.service drop-in not loaded", True, True),
        ("boot restore not installed: docker.service state not read", True, True),
        (rental_egress_guard.NO_BOOT_UNIT, True, False),
        ("boot restore not installed: directories not created", False, False),
    ],
)
async def test_a_pod_start_is_refused_where_the_boot_copy_is_not_installed(boot_line, required, refused):
    """Docker restarts a started pod at boot by itself: a pod start on a host whose systemd Docker could not take the
    boot copy is refused. The periodic run, which starts nothing, only reports it."""
    nets = f"{LEGACY_NET_ID} bridge <no value>"
    ssh = _ssh(labelled="", legacy_nets=nets, helper_stdout=boot_line + "\n")

    guarded, error = await rental_egress_guard.apply_rental_egress_guard(
        ssh, attempts=None, required_network_id=LEGACY_NET_ID if required else None
    )

    assert error == (boot_line if refused else None)


def test_a_failed_daemon_reload_is_retried_by_the_next_run_although_the_drop_in_is_unchanged(tmp_path):
    # gateway review at 129a43b: the first run moved the drop-in in place but its reload failed; the next run sees the
    # same bytes and must still reload while systemd says NeedDaemonReload=yes, or a Docker restart skips the rules
    state, root, env = _modelled_host(tmp_path, docker_user=True)
    dropin = root / rental_egress_guard.BOOT_DROPIN.lstrip("/")
    unit = root / rental_egress_guard.BOOT_UNIT.lstrip("/")
    (tmp_path / "bin" / "systemctl").write_text(
        "#!/bin/sh\n"
        'case "$1 $3" in\n'
        '  "show LoadState") echo loaded ;;\n'
        f'  "show NeedDaemonReload") cat {dropin} {unit} | cmp -s - "$STATE/loaded" && echo no || echo yes ;;\n'
        '  "daemon-reload ") [ -e "$STATE/reload_fails" ] && exit 1; '
        f'cat {dropin} {unit} > "$STATE/loaded"; echo x >> "$STATE/reloads" ;;\n'
        "esac\n"
    )
    (state / "reload_fails").touch()
    script = link_local_drop_script("br-0123456789ab", [], str(root))

    first = subprocess.run(["sh", "-c", script], env=env, capture_output=True, text=True)
    assert first.returncode == 0 and "docker.service drop-in not loaded" in first.stdout
    (state / "reload_fails").unlink()
    for _ in range(2):
        again = subprocess.run(["sh", "-c", script], env=env, capture_output=True, text=True)
        assert again.returncode == 0 and "boot restore not installed" not in again.stdout

    assert (state / "loaded").read_text() == dropin.read_text() + unit.read_text()
    assert (state / "reloads").read_text().split() == ["x"]  # reloaded once loaded, not on every apply


# the host's dockerd for the reservation tests: networks are files in $STATE/nets (name per ID), containers' networks
# lines in $STATE/ps
_RESERVE_DOCKER_STUB = r"""#!/bin/sh
case "$1 $2" in
  "network ls") ls "$STATE/nets" ;;
  "network inspect") cat "$STATE/nets/$5" ;;
  "network rm") rm "$STATE/nets/$3" && echo "$3" >> "$STATE/removed" ;;
  "ps -aq")
    shift 2
    for f in "$@"; do case "$f" in network=*) grep -qxF "${f#network=}" "$STATE/ps" 2>/dev/null && echo cid ;; esac; done
    exit 0 ;;
esac
"""


def test_a_bridge_another_process_reserved_is_never_reaped_until_its_holder_releases_it(tmp_path, monkeypatch):
    # gateway review at 129a43b: two executor registrations share one dockerd, each create in its own validator process.
    # Process A reserved its bridge and has no container on it yet (a long pull); process B's reaper, under the same
    # host lock, must keep it, so A's start neither fails on a lost bridge nor is blamed on the provider
    bin_dir, state, env = _host(tmp_path, monkeypatch)
    (bin_dir / "docker").write_text(_RESERVE_DOCKER_STUB)
    (bin_dir / "docker").chmod(0o755)
    nets = state / "nets"
    nets.mkdir()
    bridge = {name: f"{rental_egress_guard.RENTAL_NETWORK_NAME}-{name}" for name in ("pod_a", "pod_b", "pod_ended", "pod_stopped")}
    (nets / "ccc").write_text(bridge["pod_ended"] + "\n")
    (nets / "ddd").write_text(bridge["pod_stopped"] + "\n")
    (state / "ps").write_text(bridge["pod_stopped"] + "\n")  # a stopped pod's container still names its bridge

    def process(script):  # each in its own shell, as each validator process runs its own SSH command
        done = subprocess.run(["sh", "-c", locked_script(script)], env=env, capture_output=True, text=True)
        assert done.returncode == 0, done.stderr

    # A reserves, then ensures its bridge (the reservation comes first); B does the same while A has no container yet
    process(rental_egress_guard.reserve_pod_bridge_script(bridge["pod_a"], "tokA"))
    (nets / "aaa").write_text(bridge["pod_a"] + "\n")
    process(rental_egress_guard.reserve_pod_bridge_script(bridge["pod_b"], "tokB"))
    (nets / "bbb").write_text(bridge["pod_b"] + "\n")

    assert (state / "removed").read_text().split() == ["ccc"]  # only the ended rental's bridge
    process(rental_egress_guard.release_pod_bridge_script(bridge["pod_a"], "tokA"))
    process(rental_egress_guard.reserve_pod_bridge_script(bridge["pod_b"], "tokB2"))
    assert (state / "removed").read_text().split() == ["ccc", "aaa"]  # released and never used: reaped then
    assert sorted(p.name for p in nets.iterdir()) == ["bbb", "ddd"]


def test_a_reservation_its_holder_stopped_renewing_expires_and_its_unused_bridge_is_reaped(tmp_path, monkeypatch):
    # gateway review at 98b4318: a validator restart between reserve and release leaves the token behind; it must not
    # keep the bridge until a reboot. A renewed reservation keeps its bridge; an expired one is dropped, then reaped
    import os
    import time

    bin_dir, state, env = _host(tmp_path, monkeypatch)
    (bin_dir / "docker").write_text(_RESERVE_DOCKER_STUB)
    (bin_dir / "docker").chmod(0o755)
    nets = state / "nets"
    nets.mkdir()
    bridge = {name: f"{rental_egress_guard.RENTAL_NETWORK_NAME}-{name}" for name in ("pod_gone", "pod_live", "pod_new")}

    def process(script):
        done = subprocess.run(["sh", "-c", locked_script(script)], env=env, capture_output=True, text=True)
        assert done.returncode == 0, done.stderr

    process(rental_egress_guard.reserve_pod_bridge_script(bridge["pod_gone"], "tokG"))
    (nets / "ggg").write_text(bridge["pod_gone"] + "\n")
    process(rental_egress_guard.reserve_pod_bridge_script(bridge["pod_live"], "tokL"))
    (nets / "lll").write_text(bridge["pod_live"] + "\n")
    stale = time.time() - (rental_egress_guard.BRIDGE_RESERVATION_TTL_MINUTES + 1) * 60
    reservations = sorted(p for p in (tmp_path).rglob("*.tok*"))
    for path in reservations:
        os.utime(path, (stale, stale))
    process(rental_egress_guard.renew_pod_bridge_script(bridge["pod_live"], "tokL"))  # the live holder renews

    process(rental_egress_guard.reserve_pod_bridge_script(bridge["pod_new"], "tokN"))

    assert (state / "removed").read_text().split() == ["ggg"]
    assert sorted(p.name for p in nets.iterdir()) == ["lll"]


def test_a_release_that_cannot_remove_its_reservation_fails(tmp_path):
    script = rental_egress_guard.release_pod_bridge_script("net", "tok")
    assert subprocess.run(["sh", "-c", script.replace("rm -f", "false")]).returncode != 0


@pytest.mark.parametrize("failing", ["unit", "docker"])
def test_a_reload_query_that_fails_never_passes_for_loaded(tmp_path, failing):
    # gateway review at 98b4318: same bytes on disk, an earlier reload failed, and a NeedDaemonReload query now fails
    # with no output. That empty answer must not skip the reload: the run reloads, or reports the drop-in not loaded
    state, root, env = _modelled_host(tmp_path, docker_user=True)
    name = rental_egress_guard.BOOT_UNIT.rsplit("/", 1)[1]
    query = name if failing == "unit" else "docker.service"
    (tmp_path / "bin" / "systemctl").write_text(
        "#!/bin/sh\n"
        f'[ "$3" = NeedDaemonReload ] && [ "$5" = {query} ] && [ -e "$STATE/query_fails" ] && exit 1\n'
        'case "$1 $3" in\n'
        '  "show LoadState") echo loaded ;;\n'
        '  "show NeedDaemonReload") echo no ;;\n'
        '  "daemon-reload ") [ -e "$STATE/reload_fails" ] && exit 1; echo x >> "$STATE/reloads" ;;\n'
        "esac\n"
    )
    script = link_local_drop_script("br-0123456789ab", [], str(root))
    assert subprocess.run(["sh", "-c", script], env=env, capture_output=True, text=True).returncode == 0
    (state / "query_fails").touch()
    (state / "reload_fails").touch()

    again = subprocess.run(["sh", "-c", script], env=env, capture_output=True, text=True)

    assert again.returncode == 0 and "docker.service drop-in not loaded" in again.stdout


# An iptables that crashes on -I: "segv" the way a crashing binary does (the step's own status), "shell" by killing the
# shell that runs the steps, the way an "exit 139" with no stderr reaches the validator.
_CRASH_STUB = r"""#!/bin/sh
case "$1" in
  --version) echo "iptables v1.8.7 (nf_tables)" ;;
  -C|-D) exit 1 ;;
  -I) [ "$2" = FORWARD ] && exit 0; [ "$CRASH" = shell ] && kill -SEGV $PPID; kill -SEGV $$ ;;
esac
exit 0
"""


@pytest.mark.parametrize("crash", ["segv", "shell"])
def test_a_crash_names_the_failing_step_its_status_and_the_iptables_build(tmp_path, monkeypatch, crash):
    _state, _lock, env = _probe(tmp_path, monkeypatch)
    for name in ("iptables-nft", "iptables-legacy"):
        (tmp_path / "bin" / name).write_text(_CRASH_STUB)
    command = helper_command(link_local_drop_script("br-0123456789ab", []))

    done = subprocess.run(["sh", "-c", command], env={**env, "CRASH": crash}, capture_output=True, text=True)
    error = rental_egress_guard._error(SimpleNamespace(exit_status=done.returncode, stderr=done.stderr))

    if crash == "segv":
        assert done.returncode == 4
        assert error.startswith("step swap (exit 4): ") and "exit 139 (iptables v1.8.7 (nf_tables))" in error, error
    else:
        assert done.returncode == 139
        assert error.startswith("step swap (exit 139, signal 11)"), error
        assert "helper shell ended by signal 11" in error


def test_a_killed_helper_container_is_reported():
    command = rental_egress_guard._reporting_exit("sh -c 'kill -SEGV $$'", "helper container")

    done = subprocess.run(["sh", "-c", command], capture_output=True, text=True)

    assert done.returncode == 139 and "helper container ended by signal 11" in done.stderr
