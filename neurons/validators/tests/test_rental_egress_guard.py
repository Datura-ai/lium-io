"""Running rental containers get the link-local egress drop that builds already have."""

from __future__ import annotations

import ipaddress
import shlex
from types import SimpleNamespace

import pytest
from services.task.checks.rental_egress_guard import (
    LINK_LOCAL_CIDR,
    RentalEgressGuardCheck,
    link_local_drop_script,
    rental_subnets,
)
from services.task.pipeline_factory import _RENTAL_EGRESS_GUARD_SINGLETON


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


def test_runs_in_both_pipelines():
    import inspect

    from services.task import pipeline_factory

    source = inspect.getsource(pipeline_factory)
    assert source.count("_RENTAL_EGRESS_GUARD_SINGLETON,") == 2
    assert isinstance(_RENTAL_EGRESS_GUARD_SINGLETON, RentalEgressGuardCheck)
