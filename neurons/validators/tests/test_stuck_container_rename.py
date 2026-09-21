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
from fakeredis import FakeServer
from fakeredis.aioredis import FakeRedis
from payload_models.payloads import ContainerCreated, FailedContainerRequest
from services.docker_service import (
    _PRERUN_HOST_PROBE_TIMEOUT_SECONDS,
    STUCK_CONTAINER_EVENT,
    STUCK_CONTAINER_HOLDS_GPU_EVENT,
    STUCK_CONTAINER_HOLDS_GPU_STEP,
    STUCK_CONTAINER_PREFIX,
    STUCK_CONTAINER_REPEAT_WINDOW_SECONDS,
    DockerService,
    _gpu_rows_on_host,
    _gpu_uuids_with_compute_apps,
    _NvidiaSmiUnreadable,
    _wedged_gpu_uuids_per_card,
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

    def __init__(
        self,
        stale: list[str],
        *,
        rename_exit: int = 0,
        compute_apps: str = "",
        gpu_query: str = "GPU-test, 0, 0\n",
        nvidia_smi_exit: int = 0,
        gpu_query_exit: int = 0,
    ):
        self.names = list(stale)
        self.rename_exit = rename_exit
        # nvidia-smi as the node answers it after the rename: `--query-compute-apps=gpu_uuid,pid`
        # (a live CUDA process per card) and `--query-gpu=uuid,utilization.gpu,memory.used`
        self.compute_apps = compute_apps
        self.gpu_query = gpu_query
        self.nvidia_smi_exit = nvidia_smi_exit
        self.gpu_query_exit = gpu_query_exit
        self.commands: list[str] = []
        self.timeouts: list[float | None] = []
        self.client = AsyncMock()
        self.client.image_exists_result = True
        self.client.image_exists_error = None
        self.client.run = AsyncMock(side_effect=self._run)

    async def _run(self, cmd, *args, **kwargs):
        self.commands.append(cmd)
        self.timeouts.append(kwargs.get("timeout"))
        if cmd.startswith("nvidia-smi --query-compute-apps=gpu_uuid"):
            return _ssh_result(exit_status=self.nvidia_smi_exit, stdout=self.compute_apps, stderr="nvidia-smi failed" if self.nvidia_smi_exit else "")
        if cmd.startswith("nvidia-smi --query-gpu"):
            return _ssh_result(exit_status=self.gpu_query_exit, stdout=self.gpu_query, stderr="nvidia-smi failed" if self.gpu_query_exit else "")
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


def _hold_events(caplog) -> list[dict]:
    return [
        record.msg.extra
        for record in caplog.records
        if getattr(getattr(record, "msg", None), "extra", {}).get("event") == STUCK_CONTAINER_HOLDS_GPU_EVENT
    ]


def _use_real_cleanup(svc, monkeypatch):
    # _patch_happy stubs the sweep away; this file is about the sweep, so put the real one back.
    monkeypatch.setattr(svc, "clean_existing_containers", DockerService.clean_existing_containers.__get__(svc))


async def _create(svc, host: _Host, monkeypatch, **payload_over):
    _patch_happy(svc, monkeypatch, host.client)
    _use_real_cleanup(svc, monkeypatch)
    svc.redis_service.count_stuck_container = AsyncMock(return_value=1)
    payload = _payload(**payload_over)
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
async def test_a_renamed_container_that_still_holds_a_pod_gpu_fails_the_rent_before_docker_run(
    svc, monkeypatch, rename_flag, caplog
):
    # the wedged process keeps its CUDA context: nvidia-smi still lists it on the pod's card
    host = _Host(["filler_old"], compute_apps="GPU-test, 4242\n")
    _rm_fails_with(monkeypatch, COULD_NOT_KILL)
    caplog.set_level(logging.WARNING)

    payload, result = await _create(svc, host, monkeypatch)

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == STUCK_CONTAINER_HOLDS_GPU_STEP == "stuck_container_holds_gpu"
    assert "GPU-test" in result.detail and "stuck_filler_old_" in result.detail
    assert "the pod's GPUs are not free" in result.detail
    svc._run_rental_docker_create_with_port_retry.assert_not_awaited()
    # the name is still freed and the node still gets its event: the rename is what happened
    assert len(host.renames) == 1 and "filler_old" not in host.names
    (event,) = _events(caplog)  # one STUCK_CONTAINER event: the hold has its own name, no double count
    assert event["container_name"] == "filler_old"
    (hold,) = _hold_events(caplog)
    assert hold["held_gpu_uuids"] == ["GPU-test"] and hold["nvidia_smi_read"] is True and hold["whole_node"] is False
    assert hold["stuck_names"] == [event["stuck_name"]]


@pytest.mark.asyncio
async def test_the_wedge_signature_on_a_pod_gpu_fails_the_rent_too(svc, monkeypatch, rename_flag, caplog):
    # no process left, but the DAH-2427 latch: 100 % utilization, no memory
    host = _Host(["filler_old"], gpu_query="GPU-test, 100, 0\n")
    _rm_fails_with(monkeypatch, COULD_NOT_KILL)
    caplog.set_level(logging.WARNING)

    _, result = await _create(svc, host, monkeypatch)

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "stuck_container_holds_gpu"
    assert "GPU-test" in result.detail
    svc._run_rental_docker_create_with_port_retry.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_live_process_on_another_card_lets_the_rent_go_on(svc, monkeypatch, rename_flag, caplog):
    host = _Host(["filler_old"], compute_apps="GPU-other, 4242\n", gpu_query="GPU-test, 0, 0\nGPU-other, 100, 0\n")
    _rm_fails_with(monkeypatch, COULD_NOT_KILL)
    caplog.set_level(logging.WARNING)

    _, result = await _create(svc, host, monkeypatch)

    assert isinstance(result, ContainerCreated), getattr(result, "detail", result)
    svc._run_rental_docker_create_with_port_retry.assert_awaited_once()
    assert len(_events(caplog)) == 1 and _hold_events(caplog) == []  # the rename; no hold


@pytest.mark.asyncio
async def test_an_nvidia_smi_that_cannot_be_read_fails_the_rent_closed(svc, monkeypatch, rename_flag, caplog):
    host = _Host(["filler_old"], nvidia_smi_exit=1)
    _rm_fails_with(monkeypatch, COULD_NOT_KILL)
    caplog.set_level(logging.WARNING)

    _, result = await _create(svc, host, monkeypatch)

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "stuck_container_holds_gpu"
    assert "nvidia-smi could not be read" in result.detail and "compute-apps query exit 1" in result.detail
    svc._run_rental_docker_create_with_port_retry.assert_not_awaited()
    (hold,) = _hold_events(caplog)
    assert hold["held_gpu_uuids"] == [] and hold["nvidia_smi_read"] is False


@pytest.mark.asyncio
async def test_a_whole_node_rent_checks_every_card_and_fails_on_the_held_one(svc, monkeypatch, rename_flag, caplog):
    # gpu_uuids == [] is the whole-node rent (--gpus all): it lands on every card, the held one included
    host = _Host(
        ["filler_old"],
        compute_apps="GPU-b, 4242\n",
        gpu_query="GPU-a, 0, 0\nGPU-b, 0, 0\nGPU-c, 0, 0\n",
    )
    _rm_fails_with(monkeypatch, COULD_NOT_KILL)
    caplog.set_level(logging.WARNING)

    _, result = await _create(svc, host, monkeypatch, gpu_uuids=[])

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "stuck_container_holds_gpu"
    assert "GPU-b" in result.detail and "whole-node rent, every card checked" in result.detail
    svc._run_rental_docker_create_with_port_retry.assert_not_awaited()
    (hold,) = _hold_events(caplog)
    assert hold["whole_node"] is True and hold["pod_gpu_uuids"] == [] and hold["held_gpu_uuids"] == ["GPU-b"]


@pytest.mark.asyncio
async def test_a_whole_node_rent_with_no_card_held_goes_on(svc, monkeypatch, rename_flag, caplog):
    host = _Host(["filler_old"], gpu_query="GPU-a, 0, 0\nGPU-b, 3, 512\n")
    _rm_fails_with(monkeypatch, COULD_NOT_KILL)
    caplog.set_level(logging.WARNING)

    _, result = await _create(svc, host, monkeypatch, gpu_uuids=[])

    assert isinstance(result, ContainerCreated), getattr(result, "detail", result)
    svc._run_rental_docker_create_with_port_retry.assert_awaited_once()
    assert len(_events(caplog)) == 1 and _hold_events(caplog) == []
    # both reads ran (the check does not skip the whole-node rent)
    assert any(c.startswith("nvidia-smi --query-compute-apps=gpu_uuid") for c in host.commands)
    assert any(c.startswith("nvidia-smi --query-gpu") for c in host.commands)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("host_kwargs", "expected"),
    [
        ({"compute_apps": "Failed to initialize NVML: Driver/library version mismatch\n"}, "compute-apps line not"),
        ({"gpu_query": "No devices were found\n"}, "gpu line not"),
        ({"gpu_query": ""}, "listed no GPU"),
        ({"gpu_query_exit": 1}, "gpu query exit 1"),
    ],
    ids=["exit-0-garbage-compute-apps", "exit-0-garbage-gpu-query", "no-gpu-listed", "gpu-query-failed"],
)
async def test_an_unparsable_or_failed_nvidia_smi_read_fails_closed(svc, monkeypatch, rename_flag, caplog, host_kwargs, expected):
    host = _Host(["filler_old"], **host_kwargs)
    _rm_fails_with(monkeypatch, COULD_NOT_KILL)
    caplog.set_level(logging.WARNING)

    _, result = await _create(svc, host, monkeypatch)

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "stuck_container_holds_gpu"
    assert "nvidia-smi could not be read" in result.detail and expected in result.detail
    svc._run_rental_docker_create_with_port_retry.assert_not_awaited()
    (hold,) = _hold_events(caplog)
    assert hold["nvidia_smi_read"] is False and hold["held_gpu_uuids"] == []


def test_the_compute_apps_csv_is_parsed_strictly():
    assert _gpu_uuids_with_compute_apps("GPU-a, 11\nGPU-a, 12\nGPU-b, 13\n\n") == ({"GPU-a", "GPU-b"}, ["11", "12", "13"])
    assert _gpu_uuids_with_compute_apps("") == (set(), [])
    for garbage in ("[N/A], 14\n", "GPU-a\n", "GPU-a, x\n", "Failed to initialize NVML\n"):
        with pytest.raises(_NvidiaSmiUnreadable):
            _gpu_uuids_with_compute_apps(garbage)


def test_the_gpu_csv_is_parsed_strictly():
    assert _gpu_rows_on_host("GPU-a, 0, 0\nGPU-b, 100, 0\n") == [("GPU-a", 0.0, 0.0), ("GPU-b", 100.0, 0.0)]
    for garbage in ("", "No devices were found\n", "GPU-a, 0\n", "GPU-a, x, 0\n", "0, 0, 0\n"):
        with pytest.raises(_NvidiaSmiUnreadable):
            _gpu_rows_on_host(garbage)


def test_the_wedge_signature_is_judged_card_by_card():
    rows = [("GPU-pod", 100.0, 0.0), ("GPU-neighbour", 97.0, 40000.0), ("GPU-idle", 0.0, 0.0), ("GPU-hot", 100.0, 0.0)]
    # a neighbour's live job does not hide the wedge on another card
    assert _wedged_gpu_uuids_per_card(rows, busy={"GPU-neighbour"}) == {"GPU-pod", "GPU-hot"}
    # a card with its own live process is busy, not wedged, whatever its utilization
    assert _wedged_gpu_uuids_per_card(rows, busy={"GPU-pod", "GPU-neighbour"}) == {"GPU-hot"}
    # busy memory or idle utilization is not the signature
    assert _wedged_gpu_uuids_per_card([("GPU-a", 100.0, 512.0), ("GPU-b", 50.0, 0.0)], busy=set()) == set()


@pytest.mark.asyncio
async def test_on_a_shared_node_the_wedge_on_the_pods_own_card_is_not_hidden_by_a_busy_neighbour(
    svc, monkeypatch, rename_flag, caplog
):
    host = _Host(
        ["filler_old"],
        compute_apps="GPU-neighbour, 4242\n",
        gpu_query="GPU-test, 100, 0\nGPU-neighbour, 98, 40000\n",
    )
    _rm_fails_with(monkeypatch, COULD_NOT_KILL)
    caplog.set_level(logging.WARNING)

    _, result = await _create(svc, host, monkeypatch)

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "stuck_container_holds_gpu"
    assert "GPU-test" in result.detail and "wedge signature on ['GPU-test']" in result.detail
    svc._run_rental_docker_create_with_port_retry.assert_not_awaited()
    (hold,) = _hold_events(caplog)
    assert hold["held_gpu_uuids"] == ["GPU-test"]


@pytest.mark.asyncio
async def test_a_busy_neighbour_alone_does_not_hold_the_pods_card(svc, monkeypatch, rename_flag, caplog):
    host = _Host(
        ["filler_old"],
        compute_apps="GPU-neighbour, 4242\n",
        gpu_query="GPU-test, 0, 0\nGPU-neighbour, 98, 40000\n",
    )
    _rm_fails_with(monkeypatch, COULD_NOT_KILL)
    caplog.set_level(logging.WARNING)

    _, result = await _create(svc, host, monkeypatch)

    assert isinstance(result, ContainerCreated), getattr(result, "detail", result)
    svc._run_rental_docker_create_with_port_retry.assert_awaited_once()
    assert _hold_events(caplog) == []


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


def _redis_service_with_fake_store() -> RedisService:
    # a real Redis semantics stand-in: `SET NX EX` + `INCR` + `TTL` behave as on the server
    service = RedisService()
    service.redis = FakeRedis(server=FakeServer())
    return service


async def _ttls(redis: FakeRedis) -> dict[str, int]:
    return {key.decode(): await redis.ttl(key) for key in await redis.keys("*")}


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
    # one key per node, each carrying the window it was created with (SET NX EX, then INCR)
    ttls = await _ttls(svc.redis_service.redis)
    assert len(ttls) == 2 and all(0 < ttl <= STUCK_CONTAINER_REPEAT_WINDOW_SECONDS for ttl in ttls.values())
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
    ttls = await _ttls(service.redis)
    assert len(ttls) == 2 and all(0 < ttl <= 100 for ttl in ttls.values())
    # the key is created with its expiry in one command, so no INCR can leave it without a TTL
    assert all(ttl != -1 for ttl in ttls.values())
