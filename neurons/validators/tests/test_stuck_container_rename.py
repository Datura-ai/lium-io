"""A stale container dockerd cannot kill no longer blocks the pod's name; the node is told.

16–21 Sep, 4 failed rents on 4 nodes: a previous `pod_*` / `filler_*` container was wedged in the
DAH-2991 way ("tried to kill container, but did not receive an exit event"), so the rent's
`container_cleanup` failed after its `docker rm -fv` attempts, or the new `containers/create`
answered `409 Conflict (name already in use)`. With STUCK_CONTAINER_RENAME_ENABLED the wedged
container is renamed `stuck_<name>_<unix time>` (a rename needs no exit event), the create goes on,
and one STUCK_CONTAINER event per renamed container names the node; the second on one node inside
24 h carries `repeat: true`.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, Mock

import pytest
from payload_models.payloads import ContainerCreated, FailedContainerRequest
from services.docker_service import (
    _PRERUN_HOST_PROBE_TIMEOUT_SECONDS,
    STUCK_CONTAINER_EVENT,
    STUCK_CONTAINER_PREFIX,
    STUCK_CONTAINER_REPEAT_WINDOW_SECONDS,
    DockerService,
)
from services.redis_service import RedisService
from test_deploy_optimizations import _executor_info, _patch_happy, _payload, _ssh_result

COULD_NOT_KILL = (
    "[clean_existing_containers] command: /usr/bin/docker rm -fv filler_old exit_code 1, stderr: "
    "Error response from daemon: Could not kill running container "
    "0f3a9c1e2b7d4a6f8e9c0b1d2a3f4e5c6d7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c, cannot remove - "
    "tried to kill container, but did not receive an exit event"
)
NO_SUCH_CONTAINER = (
    "[clean_existing_containers] command: /usr/bin/docker rm -fv filler_old exit_code 1, stderr: "
    "Error response from daemon: No such container: filler_old"
)


@pytest.fixture
def svc():
    return DockerService(ssh_service=Mock(), redis_service=Mock(), attestation_service=Mock())


@pytest.fixture
def rename_flag(monkeypatch):
    monkeypatch.setattr("services.docker_service.settings.STUCK_CONTAINER_RENAME_ENABLED", True)


class _Host:
    """An ssh mock answering `docker ps -a` and `docker rename` like a node holding ``stale``."""

    def __init__(self, stale: list[str], *, rename_exit: int = 0):
        self.names = list(stale)
        self.rename_exit = rename_exit
        self.commands: list[str] = []
        self.timeouts: list[float | None] = []
        self.client = AsyncMock()
        self.client.image_exists_result = True
        self.client.image_exists_error = None
        self.client.run = AsyncMock(side_effect=self._run)

    async def _run(self, cmd, *args, **kwargs):
        self.commands.append(cmd)
        self.timeouts.append(kwargs.get("timeout"))
        if cmd.startswith("/usr/bin/docker ps -a"):
            return _ssh_result(stdout="".join(f"{n}\n" for n in self.names))
        if cmd.startswith("/usr/bin/docker rename"):
            _, _, old, new = cmd.split()
            if self.rename_exit == 0 and old in self.names:
                self.names[self.names.index(old)] = new
            return _ssh_result(exit_status=self.rename_exit, stderr="rename failed" if self.rename_exit else "")
        return _ssh_result()

    @property
    def renames(self) -> list[str]:
        return [c for c in self.commands if c.startswith("/usr/bin/docker rename")]


def _rm_fails_with(monkeypatch, text: str | None):
    """`retry_ssh_command` raising ``text`` for the `docker rm -fv` (None: every command succeeds)."""
    calls: list[str] = []

    async def fake(ssh_client, command, tag, *args, **kwargs):
        calls.append(command)
        if text is not None and command.startswith("/usr/bin/docker rm -fv"):
            raise Exception(text)

    monkeypatch.setattr("services.docker_service.retry_ssh_command", fake)
    return calls


def _events(caplog) -> list[dict]:
    """The `extra` of every STUCK_CONTAINER record (typed fields; the message text is not the contract)."""
    return [
        record.msg.extra
        for record in caplog.records
        if getattr(getattr(record, "msg", None), "extra", {}).get("event") == STUCK_CONTAINER_EVENT
    ]


def _use_real_cleanup(svc, monkeypatch):
    # _patch_happy stubs the sweep away; this file is about the sweep, so put the real one back.
    monkeypatch.setattr(svc, "clean_existing_containers", DockerService.clean_existing_containers.__get__(svc))


async def _create(svc, host: _Host, monkeypatch):
    _patch_happy(svc, monkeypatch, host.client)
    _use_real_cleanup(svc, monkeypatch)
    svc.redis_service.count_stuck_container = AsyncMock(return_value=1)
    payload = _payload()
    return payload, await svc.create_container(
        payload=payload,
        executor_info=_executor_info(payload),
        keypair=Mock(ss58_address="validator-hotkey"),
        private_key="encrypted",
    )


@pytest.mark.asyncio
async def test_unkillable_stale_container_is_renamed_aside_and_the_create_succeeds(
    svc, monkeypatch, rename_flag, caplog
):
    host = _Host(["filler_old"])
    _rm_fails_with(monkeypatch, COULD_NOT_KILL)
    caplog.set_level(logging.WARNING)

    payload, result = await _create(svc, host, monkeypatch)

    assert isinstance(result, ContainerCreated), getattr(result, "detail", result)
    assert len(host.renames) == 1
    _, _, old, new = host.renames[0].split()
    assert old == "filler_old"
    assert new.startswith(f"{STUCK_CONTAINER_PREFIX}filler_old_") and new.removeprefix(
        f"{STUCK_CONTAINER_PREFIX}filler_old_"
    ).isdigit()
    assert "filler_old" not in host.names and new in host.names
    svc._run_rental_docker_create_with_port_retry.assert_awaited_once()
    # the re-listing and the rename are bounded: dockerd just failed a kill on this host
    bounded = [t for c, t in zip(host.commands, host.timeouts) if c.startswith(("/usr/bin/docker ps -a", "/usr/bin/docker rename"))]
    assert bounded[-2:] == [_PRERUN_HOST_PROBE_TIMEOUT_SECONDS] * 2

    (event,) = _events(caplog)
    assert event["container_name"] == "filler_old"
    assert event["stuck_name"] == new
    assert event["pod_name"] == f"pod_{payload.pod_id}"
    assert event["repeat"] is False and event["count_in_window"] == 1
    assert event["window_seconds"] == STUCK_CONTAINER_REPEAT_WINDOW_SECONDS
    svc.redis_service.count_stuck_container.assert_awaited_once_with(
        payload.miner_hotkey, payload.executor_id, STUCK_CONTAINER_REPEAT_WINDOW_SECONDS
    )


@pytest.mark.asyncio
async def test_a_kill_that_succeeds_renames_nothing(svc, monkeypatch, rename_flag, caplog):
    host = _Host(["filler_old"])
    rm_calls = _rm_fails_with(monkeypatch, None)
    caplog.set_level(logging.WARNING)

    _, result = await _create(svc, host, monkeypatch)

    assert isinstance(result, ContainerCreated)
    assert any(c.startswith("/usr/bin/docker rm -fv") and "filler_old" in c for c in rm_calls)
    assert host.renames == []
    assert _events(caplog) == []
    svc.redis_service.count_stuck_container.assert_not_awaited()


@pytest.mark.asyncio
async def test_flag_off_fails_the_create_at_container_cleanup_as_before(svc, monkeypatch, caplog):
    host = _Host(["filler_old"])
    _rm_fails_with(monkeypatch, COULD_NOT_KILL)
    caplog.set_level(logging.WARNING)

    _, result = await _create(svc, host, monkeypatch)

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "container_cleanup"
    assert "did not receive an exit event" in result.detail
    assert host.renames == []
    assert _events(caplog) == []


@pytest.mark.asyncio
async def test_an_rm_error_of_another_class_still_fails_the_create(svc, monkeypatch, rename_flag, caplog):
    host = _Host(["filler_old"])
    _rm_fails_with(monkeypatch, NO_SUCH_CONTAINER)
    caplog.set_level(logging.WARNING)

    _, result = await _create(svc, host, monkeypatch)

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "container_cleanup"
    assert host.renames == []
    assert _events(caplog) == []


@pytest.mark.asyncio
async def test_a_rename_that_fails_raises_the_original_rm_error(svc, monkeypatch, rename_flag, caplog):
    host = _Host(["filler_old"], rename_exit=1)
    _rm_fails_with(monkeypatch, COULD_NOT_KILL)
    caplog.set_level(logging.WARNING)

    _, result = await _create(svc, host, monkeypatch)

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "container_cleanup"
    assert "did not receive an exit event" in result.detail
    assert len(host.renames) == 1
    assert _events(caplog) == []


@pytest.mark.asyncio
async def test_a_listing_that_raises_leaves_the_original_rm_error_to_the_caller(svc, monkeypatch, rename_flag, caplog):
    host = _Host(["filler_old"])
    real_run = host._run

    async def run(cmd, *args, **kwargs):
        if cmd.startswith("/usr/bin/docker ps -a") and len(host.commands) >= 1:
            raise TimeoutError("docker ps hung")
        return await real_run(cmd, *args, **kwargs)

    host.client.run = AsyncMock(side_effect=run)
    _rm_fails_with(monkeypatch, COULD_NOT_KILL)
    caplog.set_level(logging.WARNING)

    _, result = await _create(svc, host, monkeypatch)

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "container_cleanup"
    assert "did not receive an exit event" in result.detail and "docker ps hung" not in result.detail
    assert host.renames == [] and _events(caplog) == []


@pytest.mark.asyncio
async def test_only_the_containers_still_on_the_host_are_renamed(svc, monkeypatch, rename_flag, caplog):
    """`docker rm -fv a b` removed `pod_gone` and wedged on `filler_old`: one rename, one event."""
    host = _Host(["filler_old"])  # pod_gone left with the rm; the sweep's first listing had both
    listings = iter(["pod_gone\nfiller_old\n"])
    real_run = host._run

    async def run(cmd, *args, **kwargs):
        if cmd.startswith("/usr/bin/docker ps -a"):
            first = next(listings, None)
            if first is not None:
                host.commands.append(cmd)
                return _ssh_result(stdout=first)
        return await real_run(cmd, *args, **kwargs)

    host.client.run = AsyncMock(side_effect=run)
    _rm_fails_with(monkeypatch, COULD_NOT_KILL)
    caplog.set_level(logging.WARNING)

    _, result = await _create(svc, host, monkeypatch)

    assert isinstance(result, ContainerCreated)
    assert len(host.renames) == 1 and "filler_old" in host.renames[0]
    assert [e["container_name"] for e in _events(caplog)] == ["filler_old"]


class _FakeRedis:
    def __init__(self):
        self.values: dict[str, int] = {}
        self.expires: list[tuple[str, int]] = []

    async def incr(self, key):
        self.values[key] = self.values.get(key, 0) + 1
        return self.values[key]

    async def expire(self, key, seconds):
        self.expires.append((key, seconds))
        return True


def _redis_service_with_fake_store() -> RedisService:
    service = RedisService()
    service.redis = _FakeRedis()
    return service


@pytest.mark.asyncio
async def test_second_stuck_container_on_the_node_inside_the_window_is_a_repeat(monkeypatch, rename_flag, caplog):
    """Two creates on one node each hit a wedged container: the first event is not a repeat, the
    second is; a wedged container on another node starts its own count."""
    svc = DockerService(ssh_service=Mock(), redis_service=_redis_service_with_fake_store(), attestation_service=Mock())
    caplog.set_level(logging.WARNING)

    async def stuck_create(executor: str, stale: str, pod_name: str) -> _Host:
        host = _Host([stale])
        assert await svc._rename_stuck_containers_aside(
            host.client,
            {"miner_hotkey": "miner", "executor_uuid": executor, "pod_id": pod_name.removeprefix("pod_")},
            [stale],
            pod_name=pod_name,
            cause=Exception(COULD_NOT_KILL),
        )
        return host

    first = await stuck_create("executor-1", "filler_a", "pod_1")
    second = await stuck_create("executor-1", "pod_1", "pod_2")
    other_node = await stuck_create("executor-2", "filler_b", "pod_3")

    assert first.names[0].startswith(f"{STUCK_CONTAINER_PREFIX}filler_a_")
    assert second.names[0].startswith(f"{STUCK_CONTAINER_PREFIX}pod_1_")
    events = _events(caplog)
    assert [(e["executor_uuid"], e["container_name"], e["count_in_window"], e["repeat"]) for e in events] == [
        ("executor-1", "filler_a", 1, False),
        ("executor-1", "pod_1", 2, True),
        ("executor-2", "filler_b", 1, False),
    ]
    assert all(e["window_seconds"] == STUCK_CONTAINER_REPEAT_WINDOW_SECONDS for e in events)
    # the window is set once per node, at the first stuck container
    assert [seconds for _, seconds in svc.redis_service.redis.expires] == [STUCK_CONTAINER_REPEAT_WINDOW_SECONDS] * 2
    assert other_node.names[0].startswith(f"{STUCK_CONTAINER_PREFIX}filler_b_")


@pytest.mark.asyncio
async def test_a_redis_error_leaves_the_event_without_a_count_and_the_create_goes_on(svc, monkeypatch, rename_flag, caplog):
    host = _Host(["pod_old"])
    svc.redis_service.count_stuck_container = AsyncMock(side_effect=ConnectionError("redis down"))
    caplog.set_level(logging.WARNING)

    handled = await svc._rename_stuck_containers_aside(
        host.client,
        {"miner_hotkey": "miner", "executor_uuid": "executor-1"},
        ["pod_old"],
        pod_name="pod_new",
        cause=Exception(COULD_NOT_KILL),
    )

    assert handled and len(host.renames) == 1
    (event,) = _events(caplog)
    assert event["count_in_window"] is None and event["repeat"] is False


@pytest.mark.asyncio
async def test_count_stuck_container_starts_the_window_at_the_first_one_only():
    service = _redis_service_with_fake_store()

    first = await service.count_stuck_container("miner", "executor-1", 100)
    second = await service.count_stuck_container("miner", "executor-1", 100)
    other_node = await service.count_stuck_container("miner", "executor-2", 100)

    assert (first, second, other_node) == (1, 2, 1)
    assert [seconds for _, seconds in service.redis.expires] == [100, 100]
    assert len({key for key, _ in service.redis.expires}) == 2
