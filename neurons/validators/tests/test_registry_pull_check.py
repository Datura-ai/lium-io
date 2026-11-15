"""REGISTRY_PULL_FAILED: an idle node whose Docker daemon cannot pull a Docker Hub image through its own registry path.

ticket-0361 root cause (reported 23 Sep 18:46Z): a node failed 16 rents in 24 h, every one of
them a template image the node did not have cached. Its dockerd pulls through the registry mirror
docker.m.daocloud.io, whose DNS lookup times out; cached templates started fine. No check pulled anything, so the
node passed every one.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.helpers import FakeRedis, build_services
from services.task.checks import registry_pull as module
from services.task.runner import SSHCommandResult

# Docker 29.1.3 (containerd image store) with that mirror configured and its DNS answers dropped: the 2.4 KB
# pull took 51 s on one run and over 60 s on another, and the 30 s bound ended it on both runs after; with the resolver refusing, the pull failed at once
# Docker 29.8.1, 23 Sep 2026, a REJECT rule in OUTPUT for every address of registry-1.docker.io (or the daemon's
# proxy), the committed hello-world digest pulled; the text is the daemon's own, verbatim


def test_defaults_bounded():
    """Regression: the check is on with no enforcement switch, pulls every 6 h and retries a failure after 30 min."""
    fields = type(module.settings).model_fields
    assert fields["REGISTRY_PULL_CHECK_ENABLED"].default is True
    assert "REGISTRY_PULL_ENFORCEMENT_ENABLED" not in fields
    assert fields["REGISTRY_PULL_PROBE_INTERVAL_HOURS"].default == 6.0
    assert fields["REGISTRY_PULL_PROBE_RETRY_MINUTES"].default == 30.0
    assert not [name for name in fields if name.startswith("REGISTRY_PULL_FLEET")]


# ------------------------------------------------------------------ the script under sh, docker stubbed


# ------------------------------------------------------------------ the check


# ------------------------------------------------------------------ the Docker Hub control


# ------------------------------------------------------------------ fleets over many cycles




@pytest.mark.asyncio
async def test_a_node_copying_the_uuid_keeps_its_own_pull_state(context_factory):
    redis = FakeRedis()
    owner = context_factory(services=build_services(redis=redis))
    copycat = context_factory(services=build_services(redis=redis), miner_hotkey="copycat-hotkey")
    check = module.RegistryPullCheck()

    await check._save(copycat, module._ProbeState(at=1.0, outcome=module._SCHEDULED))

    assert owner.executor.uuid == copycat.executor.uuid
    assert await check._load(owner) is None


@pytest.mark.asyncio
async def test_a_deploy_keeps_the_open_streak_saved_under_the_uuid_alone(context_factory):
    redis = FakeRedis()
    ctx = context_factory(services=build_services(redis=redis))
    legacy = module._ProbeState(at=1.0, outcome="pull_failed", failures_in_a_row=1)
    redis.store[f"{module._REDIS_PREFIX}:{ctx.executor.uuid}"] = legacy.as_json()

    loaded = await module.RegistryPullCheck()._load(ctx)

    assert loaded is not None and loaded.failures_in_a_row == 1
    assert redis.store[module._state_key(ctx)] == legacy.as_json()


# the classic store's wording of a Docker Hub connection the node's path resets mid-request (documentation addresses)
RESET_TEXT = (
    'Error response from daemon: Head "https://registry-1.docker.io/v2/library/hello-world/manifests/latest": '
    "read tcp [2001:db8::5]:51234->[2001:db8::10]:443: read: connection reset by peer"
)
RESET_PULL = (
    "lium_pull driver=overlay2 mirrors=[]\nlium_pull cached=no cache=removed\nlium_pull exit=1 seconds=1\n"
    f"{RESET_TEXT}\n"
)


def _ssh_result(stdout: str) -> SSHCommandResult:
    now = datetime.now(UTC)
    return SSHCommandResult(
        command="", command_id="c", exit_code=0, stdout=stdout, stderr="", duration_ms=10,
        started_at=now, finished_at=now, success=True,
    )


class _Runner:
    async def run(self, cmd, *, timeout=60, check=False, retryable=True, stdin_text=None):
        return _ssh_result(RESET_PULL)


@pytest.mark.parametrize(
    "text",
    [
        RESET_TEXT,
        "dial tcp 192.0.2.10:443: connect: connection reset by peer",
        "proxyconnect tcp: write tcp 192.0.2.5:40000->192.0.2.10:3128: write: connection reset by peer",
    ],
)
def test_a_connection_reset_on_the_way_to_docker_hub_is_unreachable_and_fails(text):
    """Regression: a node whose path to Docker Hub reset every pull read `other`, no verdict, and kept failing rents."""
    assert module.classify_pull_error(1, text) == "unreachable"
    assert "unreachable" in module.FAILING_OUTCOMES


def test_a_reset_on_the_daemon_socket_says_nothing_about_the_registry():
    assert module.classify_pull_error(1, "read unix @->/var/run/docker.sock: read: connection reset by peer") == "other"


@pytest.mark.asyncio
@pytest.mark.parametrize("hub_up", [True, False])
async def test_resets_fail_the_node_only_twice_in_a_row_and_only_while_the_validator_reaches_docker_hub(
    context_factory, hub_up
):
    clock = SimpleNamespace(now=1_790_000_000.0)
    clock.time = lambda: clock.now

    async def hub():
        return module.DockerHubAnswer(hub_up, "HTTP 401" if hub_up else "ClientConnectorError")

    ctx = context_factory(runner=_Runner(), services=build_services(redis=FakeRedis()))
    check = module.RegistryPullCheck()
    with (
        patch.object(module, "time", clock),
        patch.object(module, "probe_docker_hub", hub),
        patch.object(module, "_HUB_CONTROL_LOCK", asyncio.Lock()),
        patch.object(module, "scheduled_pull_offset_seconds", lambda uuid: clock.now % (6 * 3600)),
        patch.object(module, "_is_rented", lambda ctx: False),
    ):
        first = await check.run(ctx)
        clock.now += 7 * 3600  # past the retry and the interval: both runs pull
        second = await check.run(ctx)

    assert first.passed
    assert first.event.what_we_saw["pull"]["outcome"] == "unreachable"
    if hub_up:
        assert first.event.what_we_saw["failures_in_a_row"] == 1
        assert second.passed is False and second.event.what_we_saw["failures_in_a_row"] == 2
    else:
        assert second.passed and second.event.what_we_saw["no_verdict"] == "docker_hub_down"
        assert second.event.what_we_saw["failures_in_a_row"] == 0
