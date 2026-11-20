"""DAH-6855: the executor applies the rental egress rules itself, from the one source the validator reads them by."""

import asyncio
import threading
from types import SimpleNamespace

import pytest
from datura import rental_egress as rules

from services import rental_egress_service as service

NET_ID = "0123456789abcdef" * 4
POD = "pod_00000000-0000-4000-8000-00000000000a"


class _Stop(BaseException):
    """Ends one of the service's forever loops in a test."""


def _host(calls, *, network=f"{NET_ID} <no value>", network_rc=0, names=""):
    """The commands the shared apply runs, answered as a host with the rental network and `names` running."""

    async def run(command):
        calls.append(command)
        if "docker ps" in command:
            return SimpleNamespace(exit_status=0, stdout=names, stderr="")
        if "network inspect" in command:
            stderr = "Error response from daemon: network lium-rentals not found" if network_rc else ""
            return SimpleNamespace(exit_status=network_rc, stdout=network, stderr=stderr)
        if "docker inspect" in command:
            return SimpleNamespace(exit_status=0, stdout="", stderr="")
        return SimpleNamespace(exit_status=0, stdout="", stderr="")

    return run


def test_the_executor_runs_the_same_rule_commands_the_validator_sends(monkeypatch):
    """Parity: the executor's apply is the shared module's, so its helper is byte for byte the one the validator runs
    over SSH for the same host (chains, swap, host lock and bound included)."""
    calls: list[str] = []
    monkeypatch.setattr(service, "run_local", _host(calls, names=f"{POD} "))

    assert service.apply_once() is True

    helper = calls[-1]
    assert helper == rules._helper(rules.link_local_drop_script("br-0123456789ab", []))
    assert rules.HOST_LOCK in helper and all(chain in helper for chain in rules.CHAINS)
    assert f"-d {rules.LINK_LOCAL_CIDR}" in helper and "! --dport 53 -j DROP" in helper


def test_nothing_to_guard_removes_stale_rules_like_the_validator(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(service, "run_local", _host(calls, network="", network_rc=1))

    assert service.apply_once() is True
    assert calls[-1] == rules._helper(rules.link_local_cleanup_script())


@pytest.mark.parametrize(
    ("stderr", "rc", "done"),
    [(rules.NO_DOCKER_USER, 3, True), ("host lock not taken", 5, False), ("iptables: Resource busy", 4, False)],
    ids=["no-docker-user-fail-open", "lock-busy-retried", "iptables-error-retried"],
)
def test_a_failed_apply_never_stops_the_executor_and_is_retried_unless_the_host_cannot_have_it(
    monkeypatch, stderr, rc, done
):
    async def run(command):
        if "nsenter" in command:
            return SimpleNamespace(exit_status=rc, stdout="", stderr=stderr)
        return await _host([])(command)

    monkeypatch.setattr(service, "run_local", run)
    assert service.apply_once() is done


def test_an_unread_docker_inventory_keeps_the_rules_and_is_retried(monkeypatch):
    calls: list[str] = []

    async def run(command):
        calls.append(command)
        return SimpleNamespace(exit_status=1, stdout="", stderr="Cannot connect to the Docker daemon")

    monkeypatch.setattr(service, "run_local", run)
    assert service.apply_once() is False
    assert not any("nsenter" in c for c in calls)


def test_at_start_the_rules_are_applied_at_once(monkeypatch):
    """A reboot brings the rentals back before any validator reaches the host: the first apply does not wait."""
    started = {}

    class _Thread:
        def __init__(self, target, args, name, daemon):
            started[name] = (target, args)

        def start(self):
            pass

    monkeypatch.setattr(service.threading, "Thread", _Thread)
    service.start_rental_egress_rules()
    target, (due,) = started["rental-egress-apply"]
    assert set(started) == {"rental-egress-apply", "rental-egress-watch"}

    applied = []

    def apply_once():
        applied.append(due.is_set())
        raise _Stop

    monkeypatch.setattr(service, "apply_once", apply_once)
    monkeypatch.setattr(due, "wait", lambda timeout: pytest.fail("waited before the first apply") if not due.is_set() else True)
    with pytest.raises(_Stop):
        target(due)
    assert applied == [False]  # cleared before the apply, so an event meanwhile triggers the next one


@pytest.mark.parametrize(
    ("event", "due"),
    [
        ({"Type": "network", "Action": "create", "Actor": {"Attributes": {"name": "lium-rentals"}}}, True),
        ({"Type": "container", "Action": "start", "Actor": {"Attributes": {"name": POD}}}, True),
        ({"Type": "container", "Action": "start", "Actor": {"Attributes": {"name": POD.replace("pod_", "filler_")}}}, True),
        ({"Type": "network", "Action": "create", "Actor": {"Attributes": {"name": "bridge2"}}}, False),
        ({"Type": "container", "Action": "start", "Actor": {"Attributes": {"name": "executor"}}}, False),
        ({"Type": "container", "Action": "create", "Actor": {"Attributes": {"name": POD}}}, False),
    ],
    ids=["rental-network-created", "pod-started", "filler-started", "other-network", "other-container", "pod-created"],
)
def test_the_rental_network_or_a_rental_container_coming_up_triggers_an_apply(monkeypatch, event, due):
    flag = threading.Event()
    seen = []

    class _Client:
        def events(self, decode, since, filters):
            seen.append(filters)
            yield {**event, "time": since + 1}
            raise _Stop

    monkeypatch.setattr(service.docker, "from_env", lambda: _Client())
    with pytest.raises(_Stop):
        service._watch_forever(flag)
    assert flag.is_set() is due
    assert seen == [{"type": ["container", "network"], "event": ["start", "create"]}]


def test_a_dropped_event_stream_reconciles_again(monkeypatch):
    flag = threading.Event()

    def from_env():
        raise ConnectionError("dockerd restarting")

    monkeypatch.setattr(service.docker, "from_env", from_env)
    monkeypatch.setattr(service.time, "sleep", lambda s: (_ for _ in ()).throw(_Stop()))
    with pytest.raises(_Stop):
        service._watch_forever(flag)
    assert flag.is_set()


def test_local_runner_reports_exit_status_and_output():
    result = asyncio.run(service.run_local("echo out; echo err >&2; exit 3"))
    assert (result.exit_status, result.stdout, result.stderr) == (3, "out\n", "err\n")


def test_version_advertises_that_the_executor_applies_the_rules():
    from routes import apis

    body = asyncio.run(apis.get_version())
    assert rules.RENTAL_EGRESS_CAPABILITY in body["capabilities"]
