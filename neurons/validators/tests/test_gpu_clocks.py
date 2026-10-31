"""DAH-4663: the renter GPU clock lock (services/gpu_clocks.py) against a fake host.

Every refusal, and the hard invariants: the GPUs touched are inside BOTH the container's own
DeviceRequests and the backend's GPUs, `-lgc`/`-rgc` never run without `-i`, `-pl` never runs, nothing
is read from inside the pod; the range is clamped to the supported clocks; the pod's delete and the
next create reset the lock; the thermal-throttle check spares a locked GPU only its low clock.
"""

import json
from unittest.mock import AsyncMock

import pytest
from payload_models.payloads import GpuResetErrorCode
from services.gpu_clocks import (
    build_clock_command,
    clamp_range,
    clock_locked_gpu_uuids,
    lock_pod_gpu_clocks,
    reset_leftover_clock_locks,
    reset_pod_clock_locks,
)

from tests.helpers import FakeRedis, build_state
from tests.test_machine_spec_scrape_check import make_command_result

NAME = "container_pod1"
U = [f"GPU-0000000{i}-0000-4000-8000-00000000000{i}" for i in range(4)]
BUS = ["00000000:1B:00.0", "00000000:0A:00.0", "00000000:2C:00.0", "00000000:3D:00.0"]
SUPPORTED = "1980\n1965\n1410\n345\n210\n"


class FakeHost:
    """Answers the commands by what they start with; records every command."""

    def __init__(self, *, device_ids=(U[0], U[1]), count=None, mig=(), runtime="runc", inspect_exit=0,
                 lgc_exit=0, rgc_exit=0, supported=SUPPORTED):
        self.commands: list[str] = []
        request = {"Count": count or 0, "DeviceIDs": list(device_ids) or None, "Capabilities": [["gpu"]]}
        self.inspect = json.dumps({"Id": "c" * 64, "HostConfig": {"Runtime": runtime, "DeviceRequests": [request]}})
        self.gpus = "\n".join(f"{i}, {U[i]}, {BUS[i]}, {'Enabled' if i in mig else 'Disabled'}" for i in range(4))
        self.inspect_exit, self.lgc_exit, self.rgc_exit, self.supported = inspect_exit, lgc_exit, rgc_exit, supported

    async def run(self, cmd: str, *, timeout: int = 60, retryable: bool = True, **_):
        self.commands.append(cmd)
        if cmd.startswith("/usr/bin/docker inspect"):
            return make_command_result(self.inspect_exit == 0, self.inspect, exit_code=self.inspect_exit)
        if cmd.startswith("nvidia-smi --query-gpu=index,uuid"):
            return make_command_result(True, self.gpus)
        if " -pm 1" in cmd:
            return make_command_result(True, "Enabled persistence mode")
        if "--query-supported-clocks=graphics" in cmd:
            return make_command_result(True, self.supported)
        if " -lgc " in cmd:
            return make_command_result(self.lgc_exit == 0, "not supported", exit_code=self.lgc_exit)
        if "--query-gpu=clocks.gr" in cmd:
            return make_command_result(True, "1410")
        if " -rgc " in cmd:  # the batched resets: one "<uuid> <exit>" line per GPU
            uuids = [part.split()[2] for part in cmd.split("; ") if part.startswith("nvidia-smi -i ")]
            return make_command_result(True, "\n".join(f"{u} {self.rgc_exit}" for u in uuids))
        raise AssertionError(f"unexpected command {cmd}")


def _assert_invariants(host: FakeHost, allowed) -> None:
    for cmd in host.commands:
        assert "docker exec" not in cmd and " -pl" not in cmd, "never inside the pod, never the power limit"
        for part in cmd.split("; "):
            if " -lgc" in part or " -rgc" in part:
                assert part.startswith("nvidia-smi -i GPU-"), "never -lgc/-rgc without -i"
                assert part.split()[2] in set(allowed)


def _reset_uuids(cmd: str) -> list[str]:
    return [part.split()[2] for part in cmd.split("; ") if part.startswith("nvidia-smi -i ") and " -rgc " in part]


async def _lock(host, redis=None, *, backend=(U[0], U[1]), indices=None, lo=1200, hi=1500, reset=False, **kw):
    redis = redis if redis is not None else FakeRedis()
    report = await lock_pod_gpu_clocks(
        host, redis, container_name=NAME, pod_id="pod1", executor_id="ex", pod_gpu_uuids=list(backend),
        gpu_indices=indices, min_mhz=lo, max_mhz=hi, reset=reset, **kw,
    )
    container = set(json.loads(host.inspect)["HostConfig"]["DeviceRequests"][0]["DeviceIDs"] or U)
    _assert_invariants(host, container & set(backend))
    return report, redis


@pytest.mark.asyncio
async def test_locks_the_pods_gpus_and_records_them():
    host = FakeHost()
    report, redis = await _lock(host)
    assert report.ok and report.error_code is None
    # pod index by host bus id: U[1] (0A) is pod GPU 0
    assert [(g.index, g.uuid, g.result, g.applied_min_mhz, g.applied_max_mhz) for g in report.gpus] == [
        (0, U[1], "locked", 1200, 1500), (1, U[0], "locked", 1200, 1500)
    ]
    assert f"nvidia-smi -i {U[0]} -lgc 1200,1500" in host.commands
    assert host.commands.index(f"nvidia-smi -i {U[0]} -pm 1") < host.commands.index(f"nvidia-smi -i {U[0]} -lgc 1200,1500")
    assert json.loads(redis.store[f"gpu_clock_lock:{U[0]}"])["pod_id"] == "pod1"
    assert set(json.loads(redis.store["gpu_clock_lock_pod:pod1"])) == {U[0], U[1]}


@pytest.mark.asyncio
async def test_the_range_is_clamped_to_the_supported_clocks():
    report, _ = await _lock(FakeHost(), lo=50, hi=9999, indices=[0])
    assert [(g.uuid, g.applied_min_mhz, g.applied_max_mhz) for g in report.gpus] == [(U[1], 210, 1980)]
    assert clamp_range(None, None, [210, 1980]) == (210, 1980)
    assert clamp_range(1800, 1500, [210, 1980]) == (1500, 1500)  # a min above the max locks at the max


@pytest.mark.parametrize(
    ("host", "kwargs", "code"),
    [
        (FakeHost(runtime="sysbox-runc"), {}, GpuResetErrorCode.ClockLockNotSupported),
        (FakeHost(), {"in_tee": True}, GpuResetErrorCode.ClockLockNotSupported),
        (FakeHost(mig=(0,)), {}, GpuResetErrorCode.MigEnabled),
        (FakeHost(inspect_exit=1), {}, GpuResetErrorCode.ContainerNotFound),
        (FakeHost(device_ids=(U[0],)), {}, GpuResetErrorCode.GpuNotInContainer),  # backend's U[1] not given
        (FakeHost(), {"indices": [2]}, GpuResetErrorCode.GpuNotInContainer),
        (FakeHost(), {"backend": ()}, GpuResetErrorCode.GpuNotInContainer),
    ],
)
@pytest.mark.asyncio
async def test_every_refusal_changes_nothing(host, kwargs, code):
    report, redis = await _lock(host, **kwargs)
    assert not report.ok and report.error_code == code and report.gpus == []
    assert not any(" -lgc" in c or " -rgc" in c or " -pm " in c for c in host.commands)
    assert redis.store == {}


@pytest.mark.asyncio
async def test_a_backend_subset_of_a_whole_host_container_touches_only_the_backends_gpus():
    host = FakeHost(device_ids=(), count=-1)
    report, _ = await _lock(host, backend=(U[2], U[3]))
    assert report.ok and {g.uuid for g in report.gpus} == {U[2], U[3]}
    # numbered as gpu_reset numbers them: the container's GPUs by bus id (U[1], U[0], U[2], U[3])
    assert [(g.index, g.uuid) for g in report.gpus] == [(2, U[2]), (3, U[3])]


@pytest.mark.asyncio
async def test_a_refused_lock_is_reported_and_its_record_kept_for_teardown():
    report, redis = await _lock(FakeHost(lgc_exit=1))
    assert not report.ok and report.error_code == GpuResetErrorCode.ClockLockNotSupported
    assert {g.result for g in report.gpus} == {"failed"}
    assert f"gpu_clock_lock:{U[0]}" in redis.store


@pytest.mark.asyncio
async def test_no_supported_clocks_fails_without_running_lgc():
    host = FakeHost(supported="")
    report, _ = await _lock(host)
    assert not report.ok and not any(" -lgc" in c for c in host.commands)


@pytest.mark.asyncio
async def test_reset_runs_rgc_per_gpu_and_clears_the_records():
    host = FakeHost()
    _, redis = await _lock(host)
    host.commands.clear()
    report, _ = await _lock(host, redis, reset=True, indices=[0])
    assert report.ok and [(g.uuid, g.result) for g in report.gpus] == [(U[1], "reset")]
    assert any(c.startswith(f"nvidia-smi -i {U[1]} -rgc ") for c in host.commands)
    assert f"gpu_clock_lock:{U[1]}" not in redis.store and f"gpu_clock_lock:{U[0]}" in redis.store
    assert json.loads(redis.store["gpu_clock_lock_pod:pod1"]) == [U[0]]


@pytest.mark.asyncio
async def test_the_pods_delete_resets_its_locks_and_a_failed_reset_is_caught_by_the_next_create():
    host = FakeHost()
    _, redis = await _lock(host)
    host.rgc_exit = 1
    assert await reset_pod_clock_locks(host, redis, "pod1") == 0
    assert "gpu_clock_lock_pod:pod1" not in redis.store and f"gpu_clock_lock:{U[0]}" in redis.store

    host.rgc_exit, host.commands = 0, []
    # U[2] has no record (a lost one looks the same): reset too
    assert await reset_leftover_clock_locks(host, redis, [U[0], U[1], U[2]]) == []
    assert len(host.commands) == 1 and _reset_uuids(host.commands[0]) == [U[0], U[1], U[2]], "one SSH call"
    assert not any(k.startswith("gpu_clock_lock:") for k in redis.store)


@pytest.mark.asyncio
async def test_teardown_without_a_lock_costs_no_ssh_and_a_whole_node_create_checks_every_host_gpu():
    host = FakeHost()
    redis = FakeRedis()
    assert await reset_pod_clock_locks(host, redis, "pod1") == 0 and host.commands == []
    redis.store[f"gpu_clock_lock:{U[3]}"] = json.dumps({"pod_id": "old"})
    assert await reset_leftover_clock_locks(host, redis, []) == []
    assert len(host.commands) == 2 and _reset_uuids(host.commands[1]) == U
    assert f"gpu_clock_lock:{U[3]}" not in redis.store


@pytest.mark.asyncio
async def test_a_create_reset_answers_only_the_recorded_gpus_it_could_not_reset_and_keeps_their_records():
    host, redis = FakeHost(rgc_exit=1), FakeRedis()
    redis.store[f"gpu_clock_lock:{U[0]}"] = json.dumps({"pod_id": "old"})

    # U[1] has no record: a GPU without -rgc support, never locked here, does not count
    assert await reset_leftover_clock_locks(host, redis, [U[0], U[1]]) == [U[0]]
    assert f"gpu_clock_lock:{U[0]}" in redis.store, "kept for the next try"


@pytest.mark.asyncio
async def test_a_create_reset_whose_ssh_call_fails_answers_every_recorded_gpu():
    host, redis = FakeHost(), FakeRedis()
    redis.store[f"gpu_clock_lock:{U[1]}"] = json.dumps({"pod_id": "old"})
    host.run = AsyncMock(side_effect=OSError("connection lost"))

    assert await reset_leftover_clock_locks(host, redis, [U[0], U[1]]) == [U[1]]


@pytest.mark.asyncio
@pytest.mark.parametrize(("still_locked", "refused"), [([], False), ([U[0]], True)])
async def test_a_create_is_refused_only_when_a_recorded_lock_could_not_be_reset(monkeypatch, still_locked, refused):
    import services.docker_service as ds

    service, _gates, _started = _service_with_gated_steps([])
    service.redis_service = None
    monkeypatch.setattr(ds, "reset_leftover_clock_locks", AsyncMock(return_value=still_locked))

    if refused:
        with pytest.raises(RuntimeError, match=U[0]):
            await service._reset_leftover_clock_locks_before_create(None, _create_payload(), {})
    else:
        await service._reset_leftover_clock_locks_before_create(None, _create_payload(), {})


def test_the_clock_command_always_names_one_pod_gpu():
    for bad in ("", "0", "all"):
        with pytest.raises(ValueError):
            build_clock_command("reset", bad)
    with pytest.raises(ValueError):
        build_clock_command("lock", U[2], pod_gpu_uuids={U[0]}, min_mhz=1, max_mhz=2)
    assert build_clock_command("lock", U[0], pod_gpu_uuids={U[0]}, min_mhz=1, max_mhz=2) == f"nvidia-smi -i {U[0]} -lgc 1,2"


# --- the checks ---------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_thermal_throttle_check_skips_a_locked_gpu(context_factory):
    from services.task.checks.gpu_thermal_throttle import GpuThermalThrottleCheck
    from services.task.messages import GpuThermalThrottleMessages as Msg

    from tests.helpers import build_services

    rows = [
        {"uuid": u, "gpu_utilization": 98, "graphics_speed": 600, "max_graphics_speed": 1980, "clocks_event_reasons": 0}
        for u in (U[0], U[1])
    ]
    redis = FakeRedis()
    ctx = context_factory(state=build_state(gpu_details=rows), services=build_services(redis=redis))
    flagged = await GpuThermalThrottleCheck().run(ctx)
    assert flagged.event.reason_code == Msg.THROTTLED.reason

    redis.store[f"gpu_clock_lock:{U[0]}"] = json.dumps({"pod_id": "pod1"})
    one = await GpuThermalThrottleCheck().run(ctx)
    assert [g["uuid"] for g in one.event.what_we_saw["gpus"]] == [U[1]]

    redis.store[f"gpu_clock_lock:{U[1]}"] = json.dumps({"pod_id": "pod1"})
    none = await GpuThermalThrottleCheck().run(ctx)
    assert none.passed and none.event.reason_code == Msg.OK.reason

    # a heat fault on a locked GPU is still judged: on an idle node it collapses, lock or not
    hot = [{**row, "clocks_event_reasons": 0x40} for row in rows]
    hot_ctx = context_factory(state=build_state(gpu_details=hot), services=build_services(redis=redis))
    collapsed = await GpuThermalThrottleCheck().run(hot_ctx)
    assert not collapsed.passed and collapsed.event.reason_code == Msg.COLLAPSED.reason


@pytest.mark.asyncio
async def test_a_redis_error_or_a_mock_never_skips_the_throttle_check():
    assert await clock_locked_gpu_uuids(FakeRedis(failing=True), U) == set()
    assert await clock_locked_gpu_uuids(AsyncMock(), U) == set()


@pytest.mark.asyncio
async def test_the_clock_lock_mark_declines_a_second_operation_but_skips_no_check():
    from services.gpu_reset import renter_gpu_reset_in_progress

    from tests.test_gpu_reset import PendingRedis

    redis = PendingRedis()
    await redis.add_pending_pod("hk", "ex", "pod1", kind="gpu_clocks")
    assert await redis.renting_in_progress("hk", "ex", "pod1")  # a second operation is declined
    assert not await renter_gpu_reset_in_progress(redis, "hk", "ex")  # the pod keeps running: checks run


def test_a_lock_that_never_reached_the_executor_answers_with_a_clocks_result():
    from payload_models.payloads import FailedContainerErrorCodes, LockPodGpuClocksRequest, PodGpuClocksResult
    from services.miner_service import MinerService

    payload = LockPodGpuClocksRequest(
        miner_hotkey="hk", executor_id="ex", pod_id="pod1", container_name=NAME, gpu_uuids=U[:2],
        min_mhz=1200, max_mhz=1500, request_id="r1",
    )
    service = MinerService.__new__(MinerService)
    busy = service._handle_container_error(payload, "busy", FailedContainerErrorCodes.RentingInProgress)
    assert isinstance(busy, PodGpuClocksResult) and busy.request_id == "r1" and not busy.ok
    assert busy.error_code == GpuResetErrorCode.OperationInProgress


def _service_with_gated_steps(order: list[str]):
    """A DockerService whose delete body and GPU operation body each wait on their own gate."""
    import asyncio

    from services.docker_service import DockerService

    gates = {"delete": asyncio.Event(), "gpu": asyncio.Event()}
    started = {"delete": asyncio.Event(), "gpu": asyncio.Event()}

    def step(name: str, answer):
        async def run(*_args, **_kwargs):
            started[name].set()
            await gates[name].wait()
            order.append(name)
            return answer

        return run

    service = DockerService.__new__(DockerService)
    service._delete_container = step("delete", "deleted")
    service._pod_gpu_operation_held = step("gpu", "locked")
    return service, gates, started


async def _gpu_operation(service):
    from payload_models.payloads import LockPodGpuClocksRequest
    from services.gpu_clocks import GpuClocksReport

    payload = LockPodGpuClocksRequest(
        miner_hotkey="hk", executor_id="ex", pod_id="pod1", container_name=NAME, gpu_uuids=U[:2],
        min_mhz=1200, max_mhz=1500, request_id="r1",
    )
    info = type("Info", (), {"address": "h", "port": 8001, "ssh_username": "u", "ssh_port": 22})()
    return await service._pod_gpu_operation(
        payload, info, None, "key", label="Lock pod GPU clocks", kind="gpu_clocks", operate=AsyncMock(),
        failed=lambda detail: GpuClocksReport(ok=False, error_code=GpuResetErrorCode.ExecutorUnreachable, detail=detail),
    )


def _delete_payload():
    from payload_models.payloads import ContainerDeleteRequest

    return ContainerDeleteRequest(miner_hotkey="hk", executor_id="ex", pod_id="pod1", container_name=NAME)


@pytest.mark.asyncio
async def test_a_delete_waits_for_a_clock_lock_in_flight_so_its_reset_runs_after_the_lgc():
    import asyncio

    from services.docker_service import pod_gpu_operation_locks

    order: list[str] = []
    service, gates, started = _service_with_gated_steps(order)
    info = type("Info", (), {"address": "h", "port": 8001, "ssh_username": "u", "ssh_port": 22})()

    gpu = asyncio.create_task(_gpu_operation(service))
    await started["gpu"].wait()
    delete = asyncio.create_task(service.delete_container(_delete_payload(), info, None, "key"))
    gates["delete"].set()
    await asyncio.sleep(0.05)
    assert not started["delete"].is_set()  # the delete waits for the -lgc
    gates["gpu"].set()

    assert await asyncio.gather(gpu, delete) == ["locked", "deleted"]
    assert order == ["gpu", "delete"]
    assert not pod_gpu_operation_locks._locks  # nothing kept per pod


@pytest.mark.asyncio
async def test_a_clock_lock_during_a_delete_is_declined_and_touches_nothing():
    import asyncio

    order: list[str] = []
    service, gates, started = _service_with_gated_steps(order)
    info = type("Info", (), {"address": "h", "port": 8001, "ssh_username": "u", "ssh_port": 22})()

    delete = asyncio.create_task(service.delete_container(_delete_payload(), info, None, "key"))
    await started["delete"].wait()
    report = await _gpu_operation(service)
    gates["delete"].set()
    await delete

    assert (report.ok, report.error_code) == (False, GpuResetErrorCode.OperationInProgress)
    assert order == ["delete"]  # the GPU operation never ran


@pytest.mark.asyncio
async def test_a_delete_still_waiting_when_its_wait_ends_is_retried_not_run(monkeypatch):
    import asyncio

    import services.docker_service as ds
    from payload_models.payloads import FailedContainerErrorCodes

    monkeypatch.setattr(ds, "POD_GPU_OPERATION_DELETE_WAIT_SECONDS", 0.01)
    order: list[str] = []
    service, gates, started = _service_with_gated_steps(order)
    info = type("Info", (), {"address": "h", "port": 8001, "ssh_username": "u", "ssh_port": 22})()

    gpu = asyncio.create_task(_gpu_operation(service))
    await started["gpu"].wait()
    gates["delete"].set()
    try:
        reply = await service.delete_container(_delete_payload(), info, None, "key")
        assert reply.error_code == FailedContainerErrorCodes.DeletionInProgress  # the backend re-asks
        assert not started["delete"].is_set()  # nothing was torn down under the running -lgc
    finally:
        gates["gpu"].set()
        await gpu


@pytest.mark.asyncio
async def test_a_gpu_operation_past_its_deadline_is_cancelled_and_frees_the_pod(monkeypatch):
    import services.docker_service as ds

    monkeypatch.setattr(ds, "POD_GPU_OPERATION_DEADLINE_SECONDS", 0.01)
    service, _gates, _started = _service_with_gated_steps([])  # the GPU gate never opens

    report = await _gpu_operation(service)

    assert (report.ok, report.error_code) == (False, GpuResetErrorCode.ExecutorUnreachable)
    assert not ds.pod_gpu_operation_locks._locks  # a delete gets the pod


def _create_payload():
    from payload_models.payloads import ContainerCreateRequest

    return ContainerCreateRequest.model_construct(pod_id="pod1", gpu_uuids=U[:2])


@pytest.mark.asyncio
async def test_a_reboot_waits_for_a_clock_lock_in_flight_so_its_clock_reset_runs_after_the_lgc(monkeypatch):
    import asyncio

    import services.docker_service as ds

    order: list[str] = []
    service, gates, started = _service_with_gated_steps(order)
    service.redis_service = None
    monkeypatch.setattr(ds, "reset_leftover_clock_locks", AsyncMock(side_effect=lambda *_a: order.append("rgc")))

    gpu = asyncio.create_task(_gpu_operation(service))
    await started["gpu"].wait()
    create = asyncio.create_task(service._reset_leftover_clock_locks_before_create(None, _create_payload(), {}))
    await asyncio.sleep(0.05)
    assert order == []  # the create's -rgc waits for the -lgc
    gates["gpu"].set()
    await asyncio.gather(gpu, create)

    assert order == ["gpu", "rgc"]
    assert not ds.pod_gpu_operation_locks._locks


@pytest.mark.asyncio
async def test_a_clock_lock_during_a_reboots_clock_reset_is_declined_and_touches_nothing(monkeypatch):
    import asyncio

    import services.docker_service as ds

    order: list[str] = []
    service, _gates, _started = _service_with_gated_steps(order)
    service.redis_service = None
    rgc_started, rgc_gate = asyncio.Event(), asyncio.Event()

    async def gated_rgc(*_args):
        rgc_started.set()
        await rgc_gate.wait()
        order.append("rgc")

    monkeypatch.setattr(ds, "reset_leftover_clock_locks", gated_rgc)

    create = asyncio.create_task(service._reset_leftover_clock_locks_before_create(None, _create_payload(), {}))
    await rgc_started.wait()
    report = await _gpu_operation(service)
    rgc_gate.set()
    await create

    assert (report.ok, report.error_code) == (False, GpuResetErrorCode.OperationInProgress)
    assert order == ["rgc"]  # the clock lock never ran


@pytest.mark.asyncio
async def test_a_delete_cancels_a_create_before_taking_the_pod_lock_its_clock_reset_needs(monkeypatch):
    import asyncio

    import services.docker_service as ds

    order: list[str] = []
    service, gates, _started = _service_with_gated_steps(order)
    service.redis_service = None
    monkeypatch.setattr(ds, "reset_leftover_clock_locks", AsyncMock(side_effect=lambda *_a: order.append("rgc")))
    info = type("Info", (), {"address": "h", "port": 8001, "ssh_username": "u", "ssh_port": 22})()
    create_go = asyncio.Event()

    async def create():
        with ds.inflight_creates.track("pod1"):
            await create_go.wait()
            await service._reset_leftover_clock_locks_before_create(None, _create_payload(), {})

    creating = asyncio.create_task(create())
    await asyncio.sleep(0)
    delete = asyncio.create_task(service.delete_container(_delete_payload(), info, None, "key"))
    await asyncio.sleep(0.05)  # the delete has cancelled the create and waits for it
    create_go.set()
    gates["delete"].set()

    # before the fix the delete held the lock the create's -rgc waited on: each waited for the other
    assert await asyncio.wait_for(asyncio.gather(creating, delete), 5) == [None, "deleted"]
    assert order == ["rgc", "delete"]
