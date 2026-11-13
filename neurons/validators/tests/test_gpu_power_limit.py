import asyncio
import json
from types import SimpleNamespace
import time
from unittest.mock import AsyncMock

import neurons.validators.src.services.gpu_power_limit as gpl
import pytest
from helpers import FakeRedis as SharedFakeRedis
from neurons.validators.src.services.gpu_power_limit import (
    POWER_STATE_CMD,
    GpuPowerCapResult,
    GpuPowerRestoreRecord,
    GpuPowerRestoreStateError,
    _pod_index_key,
    _restore_key,
    apply_filler_gpu_power_limits,
    restore_filler_pod_gpu_power_limits,
)
from payload_models.payloads import GpuPowerLimit


def _record(gpu_uuid: str, watts: int, executor_id: str = "executor-1") -> str:
    return GpuPowerRestoreRecord(
        gpu_uuid=gpu_uuid, watts=watts, pod_id=POD_ID, executor_id=executor_id, capped_at=time.time()
    ).model_dump_json()


# uuid, current, default, min, max
STATE_CSV = "GPU-a, 130, 400, 100, 400\nGPU-b, 250, 250, 100, 250\n"
POD_ID = "pod-1"


class FakeRun:
    def __init__(self, stdout: str = "", stderr: str = "", exit_status: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.exit_status = exit_status


def fake_ssh(*responses: FakeRun) -> AsyncMock:
    ssh = AsyncMock()
    ssh.run.side_effect = responses
    return ssh


def _set_ok(readback_watts: int, persistence: str = "Enabled") -> list[FakeRun]:
    """SSH responses for one successful verified set: -pm 1, -pl, readback confirming the target."""
    return [FakeRun(), FakeRun(), FakeRun(stdout=f"{readback_watts}.00, {persistence}\n")]


EXECUTOR_ID = "executor-1"


def _limits(**watts_by_uuid: int) -> list[GpuPowerLimit]:
    return [GpuPowerLimit(gpu_uuid=uuid.replace("_", "-"), watts=watts) for uuid, watts in watts_by_uuid.items()]


def _commands(ssh: AsyncMock) -> list[str]:
    return [call.args[0] for call in ssh.run.call_args_list]


def _set_commands(gpu_uuid: str, watts: int) -> list[str]:
    """The exact command triple one verified set issues."""
    return [
        f"nvidia-smi -i {gpu_uuid} -pm 1",
        f"nvidia-smi -i {gpu_uuid} -pl {watts}",
        f"nvidia-smi -i {gpu_uuid} --query-gpu=power.limit,persistence_mode --format=csv,noheader,nounits",
    ]


class FakeRedis:
    def __init__(self, initial: dict[str, str] | None = None):
        self.store: dict[str, str] = dict(initial or {})

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.store[key] = value

    async def get(self, key: str) -> str | None:
        return self.store.get(key)

    async def delete(self, key: str) -> None:
        self.store.pop(key, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "timeout", "nonzero", "partial", "redis", "record_redis", "record", "missing_record", "mismatched_record", "index", "delete", "absent"])
async def test_strict_pod_restore_requires_every_record_to_be_restored(failure):
    # Arrange: real restore records and setters; keep one GPU's failure distinct from a partial restore.
    redis = FakeRedis({_pod_index_key(POD_ID): '["GPU-a", "GPU-b"]',
                       _restore_key("GPU-a"): _record("GPU-a", 400),
                       _restore_key("GPU-b"): _record("GPU-b", 250)})
    if failure == "absent":
        redis.store.clear()
    if failure == "missing_record":
        del redis.store[_restore_key("GPU-a")]
    if failure == "mismatched_record":
        redis.store[_restore_key("GPU-a")] = _record("GPU-b", 250)
    if failure in ("record", "index"):
        redis.store[_restore_key("GPU-a") if failure == "record" else _pod_index_key(POD_ID)] = "broken"
    if failure == "redis":
        redis.get = AsyncMock(side_effect=ConnectionError("unreadable"))
    if failure == "record_redis":
        async def get(key):
            if key.startswith("gpu_power_restore:"):
                raise ConnectionError("unreadable record")
            return redis.store.get(key)
        redis.get = get
    if failure == "delete":
        redis.delete = AsyncMock(side_effect=ConnectionError("cannot clear record"))
    initial_records = redis.store.copy()
    async def run(command, **kwargs):
        if command == POWER_STATE_CMD:
            return FakeRun(stdout=STATE_CSV)
        if " -pl " in command and failure in ("timeout", "nonzero", "partial"):
            if failure == "timeout":
                raise TimeoutError("hung")
            if failure != "partial" or "GPU-b" in command:
                return FakeRun(exit_status=1)
        return FakeRun(stdout="400, Enabled\n" if "GPU-a" in command else "250, Enabled\n")
    ssh = AsyncMock(run=AsyncMock(side_effect=run))
    # Act: strict caller cannot succeed with an unknown or remaining cap.
    if failure not in (None, "absent"):
        error_type = RuntimeError if failure in ("timeout", "nonzero", "partial") else GpuPowerRestoreStateError
        with pytest.raises(error_type) as caught:
            await restore_filler_pod_gpu_power_limits(ssh, redis, POD_ID, require_complete=True)
        assert type(caught.value) is error_type
        # Assert: failed records and the index survive for a later retry.
        assert _pod_index_key(POD_ID) in redis.store
        assert _restore_key("GPU-b") in redis.store
        if failure == "mismatched_record":
            ssh.run.assert_not_awaited()
            assert redis.store == initial_records
    else:
        assert await restore_filler_pod_gpu_power_limits(ssh, redis, POD_ID, require_complete=True) == (0 if failure else 2)
        assert not redis.store


@pytest.mark.asyncio
async def test_strict_pod_restore_accepts_a_record_already_restored():
    # Arrange: the customer create's pre-run restore already restored GPU-a and cleared its record,
    # leaving the filler's index (prod 28 Oct 2026, failure_step gpu_power_restore_state).
    redis = FakeRedis({_pod_index_key(POD_ID): '["GPU-a", "GPU-b"]', _restore_key("GPU-b"): _record("GPU-b", 250)})
    ssh = fake_ssh(FakeRun(stdout="GPU-a, 400, 400, 100, 400\nGPU-b, 175, 250, 100, 250\n"), *_set_ok(250))
    # Act / Assert: GPU-a reads its default, so the create goes on; GPU-b is restored; nothing is left.
    assert await restore_filler_pod_gpu_power_limits(ssh, redis, POD_ID, require_complete=True) == 1
    assert not redis.store


@pytest.mark.asyncio
@pytest.mark.parametrize(("current_watts", "accepted"), [(700, False), (799, False), (800, True), (900, True)])
async def test_strict_pod_restore_without_record_holds_the_floor(current_watts, accepted):
    # Arrange: GPU-a's record is gone; only its live limit can show it was restored.
    redis = FakeRedis({_pod_index_key(POD_ID): '["GPU-a"]'})
    ssh = fake_ssh(FakeRun(stdout=f"GPU-a, {current_watts}, 1000, 100, 1000\n"))
    # Act / Assert: below the 80 % floor fails the create; a host's own 80-90 % setting does not.
    if accepted:
        assert await restore_filler_pod_gpu_power_limits(ssh, redis, POD_ID, require_complete=True) == 0
    else:
        with pytest.raises(GpuPowerRestoreStateError):
            await restore_filler_pod_gpu_power_limits(ssh, redis, POD_ID, require_complete=True)


@pytest.mark.asyncio
async def test_strict_pod_restore_with_an_unreadable_host_is_the_hosts_failure():
    # prod 11 Nov 2026 pod f43cdd12: the host's nvidia-smi query failed while confirming a record-less GPU
    redis = FakeRedis({_pod_index_key(POD_ID): '["GPU-a"]'})
    ssh = AsyncMock(run=AsyncMock(side_effect=TimeoutError()))
    with pytest.raises(RuntimeError) as caught:
        await restore_filler_pod_gpu_power_limits(ssh, redis, POD_ID, require_complete=True)
    assert type(caught.value) is RuntimeError  # not GpuPowerRestoreStateError (platform state)


@pytest.mark.asyncio
@pytest.mark.parametrize("require_complete", [True, False])
async def test_pod_index_delete_failure_after_successful_restore(require_complete):
    # Arrange: only the final index deletion fails; the GPU record can be cleared.
    redis = FakeRedis({_pod_index_key(POD_ID): '["GPU-a"]', _restore_key("GPU-a"): _record("GPU-a", 400)})
    delete_record = redis.delete

    async def delete(key):
        if key == _pod_index_key(POD_ID):
            raise ConnectionError("cannot clear index")
        await delete_record(key)

    redis.delete = delete
    ssh = fake_ssh(FakeRun(stdout=STATE_CSV), *_set_ok(400))
    # Act: strict customer checks fail, while existing best-effort cleanup still returns.
    if require_complete:
        with pytest.raises(GpuPowerRestoreStateError):
            await restore_filler_pod_gpu_power_limits(ssh, redis, POD_ID, require_complete=True)
    else:
        assert await restore_filler_pod_gpu_power_limits(ssh, redis, POD_ID) == 1
    # Assert: the record was deleted after restoration; the index the delete could not clear stays (the
    # strict path retries it, the best-effort one logs it: with no records left it restores nothing).
    assert redis.store == {_pod_index_key(POD_ID): '["GPU-a"]'}


def _power(*watts: int) -> dict[str, gpl.GpuPowerState]:
    return {
        f"GPU-{index}": gpl.GpuPowerState(current_watts=w, min_watts=100, max_watts=450, default_watts=350)
        for index, w in enumerate(watts)
    }


async def _rental_start(monkeypatch, redis, states, pod_id: str = "pod-1") -> int:
    monkeypatch.setattr(gpl, "_query_power_state", AsyncMock(return_value=states))
    return await gpl.raise_low_power_limits_to_default(
        AsyncMock(), "ex-1", list(states), redis=redis, pod_id=pod_id
    )


def _ratio(redis, gpu_uuid: str) -> dict:
    return json.loads(redis.store[gpl.rental_start_ratio_key("ex-1", gpu_uuid)])


@pytest.mark.asyncio
async def test_rental_start_raise_keeps_the_80_percent_tier_and_records_each_pod_gpu(monkeypatch) -> None:
    states = _power(298, 200, 298)
    set_side_by_side = AsyncMock(return_value=1)
    monkeypatch.setattr(gpl, "_set_side_by_side", set_side_by_side)
    monkeypatch.setattr(gpl, "_query_power_state", AsyncMock(return_value=states))
    redis = SharedFakeRedis()

    raised = await gpl.raise_low_power_limits_to_default(
        AsyncMock(), "ex-1", ["GPU-0", "GPU-1"], redis=redis, pod_id="pod-1"
    )

    assert raised == 1
    assert [gpu.gpu_uuid for gpu in set_side_by_side.await_args.args[1]] == ["GPU-1"]  # 85 % is left alone
    assert _ratio(redis, "GPU-0") == {"pod_id": "pod-1", "ratio": 0.8514}
    assert _ratio(redis, "GPU-1") == {"pod_id": "pod-1", "ratio": 1.0}
    assert gpl.rental_start_ratio_key("ex-1", "GPU-2") not in redis.store  # another pod's GPU


@pytest.mark.parametrize(
    ("first_pod", "second_pod", "expected"),
    [
        ("pod-1", "pod-1", {"pod_id": "pod-1", "ratio": 1.0}),  # reboot / edit: the start stands
        ("pod-0", "pod-1", {"pod_id": "pod-1", "ratio": 0.8514}),  # a new rental: its own start
    ],
)
@pytest.mark.asyncio
async def test_rental_start_record_survives_a_reboot_of_the_same_rental(
    monkeypatch, first_pod, second_pod, expected
) -> None:
    monkeypatch.setattr(gpl, "_set_side_by_side", AsyncMock(return_value=0))
    redis = SharedFakeRedis()
    await _rental_start(monkeypatch, redis, _power(350), first_pod)
    await _rental_start(monkeypatch, redis, _power(298), second_pod)  # host cut to 85 % meanwhile
    assert _ratio(redis, "GPU-0") == expected


@pytest.mark.asyncio
async def test_rental_start_of_a_rental_from_before_the_records_records_stock(monkeypatch) -> None:
    monkeypatch.setattr(gpl, "_set_side_by_side", AsyncMock(return_value=0))
    redis = SharedFakeRedis()
    redis.store[gpl.rental_legacy_pod_key("pod-1")] = "1"
    await _rental_start(monkeypatch, redis, _power(298))
    assert _ratio(redis, "GPU-0") == {"pod_id": "pod-1", "ratio": 1.0}


@pytest.mark.asyncio
async def test_rental_start_retry_after_a_partial_write_records_the_live_limit(monkeypatch) -> None:
    # the refused create wrote GPU-0 only; its retry must keep GPU-1 in the 80 % tier
    monkeypatch.setattr(gpl, "_set_side_by_side", AsyncMock(return_value=0))
    redis = SharedFakeRedis()
    redis.store[gpl.rental_start_ratio_key("ex-1", "GPU-0")] = json.dumps({"pod_id": "pod-1", "ratio": 1.0})
    await _rental_start(monkeypatch, redis, _power(350, 280))
    assert _ratio(redis, "GPU-1") == {"pod_id": "pod-1", "ratio": 0.8}


@pytest.mark.asyncio
async def test_rental_start_partial_write_failure_records_no_gpu(monkeypatch) -> None:
    # all of a pod's records in one transaction: a refused create leaves none behind to mix with a retry
    monkeypatch.setattr(gpl, "_set_side_by_side", AsyncMock(return_value=0))
    redis = SharedFakeRedis()
    redis.fail_next_set_of = {gpl.rental_start_ratio_key("ex-1", "GPU-1")}
    with pytest.raises(gpl.GpuPowerRestoreStateError):
        await _rental_start(monkeypatch, redis, _power(350, 350))
    assert not any(key.startswith("gpu_power_rental_start_ratio:") for key in redis.store)
    await _rental_start(monkeypatch, redis, _power(350, 280))  # host lowered GPU-1 before the retry
    assert [_ratio(redis, gpu)["ratio"] for gpu in ("GPU-0", "GPU-1")] == [1.0, 0.8]  # one snapshot


@pytest.mark.parametrize(("pod_on_host", "ratio"), [(True, 1.0), (False, 0.8514)])
@pytest.mark.asyncio
async def test_rental_start_of_a_pod_already_on_the_host_records_stock(monkeypatch, pod_on_host, ratio) -> None:
    # a rental from before the records, cut to 85 % and rebooted before its first check
    monkeypatch.setattr(gpl, "_set_side_by_side", AsyncMock(return_value=0))
    monkeypatch.setattr(gpl, "_query_power_state", AsyncMock(return_value=_power(298)))
    redis = SharedFakeRedis()
    await gpl.raise_low_power_limits_to_default(
        AsyncMock(), "ex-1", ["GPU-0"], redis=redis, pod_id="pod-1", pod_on_host=pod_on_host
    )
    assert _ratio(redis, "GPU-0") == {"pod_id": "pod-1", "ratio": ratio}


@pytest.mark.parametrize(
    ("watts", "failing_gpus"),
    [
        ((280,), {"GPU-0"}),  # a tier rental without its record would be held at 90 %
        ((350, 280), {"GPU-1"}),  # partial: the tier GPU's write failed
        ((350, 350), {"GPU-1"}),  # stock: an unrecorded GPU could be cut and re-recorded at a reboot
    ],
)
@pytest.mark.asyncio
async def test_rental_start_record_write_failure_fails_the_create_then_a_retry_records(
    monkeypatch, watts, failing_gpus
) -> None:
    monkeypatch.setattr(gpl, "_set_side_by_side", AsyncMock(return_value=0))
    redis = SharedFakeRedis()
    redis.fail_next_set_of = {gpl.rental_start_ratio_key("ex-1", u) for u in failing_gpus}
    with pytest.raises(gpl.GpuPowerRestoreStateError):
        await _rental_start(monkeypatch, redis, _power(*watts))
    await _rental_start(monkeypatch, redis, _power(*watts))  # the retried create
    assert all(_ratio(redis, f"GPU-{i}")["pod_id"] == "pod-1" for i in range(len(watts)))


@pytest.mark.asyncio
async def test_rental_start_record_read_outage_fails_the_create(monkeypatch) -> None:
    monkeypatch.setattr(gpl, "_set_side_by_side", AsyncMock(return_value=0))
    with pytest.raises(gpl.GpuPowerRestoreStateError):
        await _rental_start(monkeypatch, SharedFakeRedis(failing=True), _power(280))


@pytest.mark.asyncio
async def test_raise_without_redis_never_records_or_raises(monkeypatch) -> None:
    monkeypatch.setattr(gpl, "_query_power_state", AsyncMock(side_effect=RuntimeError("nvidia-smi hung")))
    assert await gpl.raise_low_power_limits_to_default(AsyncMock(), "ex-1", None) == 0


@pytest.mark.parametrize(
    ("container_names", "expected"),
    [
        (("pod_pod-1",), True),
        (("pod_pod-1__prev",), True),
        (("kept_unmounted_pod-1_1700000000",), True),
        (("kept_unmounted_pod-10_1700000000",), False),
        (("pod_pod-2",), False),
        (None, False),
    ],
)
@pytest.mark.asyncio
async def test_create_passes_pod_on_host_from_the_prerun_listing(monkeypatch, container_names, expected) -> None:
    import services.docker_service as ds

    raise_mock = AsyncMock(return_value=0)
    monkeypatch.setattr(ds, "restore_tracked_gpu_power_limits", AsyncMock())
    monkeypatch.setattr(ds, "raise_low_power_limits_to_default", raise_mock)
    payload = SimpleNamespace(gpu_uuids=["GPU-0"], executor_id="ex-1", pod_id="pod-1")
    probe = SimpleNamespace(container_names=container_names)
    await ds.DockerService._restore_gpu_power_for_uncapped_pod(
        SimpleNamespace(redis_service=object()), AsyncMock(), payload, probe, {}, record_rental_start=True
    )
    assert raise_mock.await_args.kwargs["pod_on_host"] is expected


@pytest.mark.asyncio
@pytest.mark.parametrize(("inspect_exit", "expected"), [(0, True), (1, False)])
async def test_create_without_a_probe_asks_docker_whether_the_pod_is_on_host(monkeypatch, inspect_exit, expected) -> None:
    # F-12 (lio#1924): a failed pre-run probe must not take a rebooted legacy rental for a new one.
    import services.docker_service as ds

    raise_mock = AsyncMock(return_value=0)
    monkeypatch.setattr(ds, "restore_tracked_gpu_power_limits", AsyncMock())
    monkeypatch.setattr(ds, "raise_low_power_limits_to_default", raise_mock)
    ssh = AsyncMock()
    ssh.run.return_value = SimpleNamespace(exit_status=inspect_exit, stdout="", stderr="")
    payload = SimpleNamespace(gpu_uuids=["GPU-0"], executor_id="ex-1", pod_id="pod-1")
    await ds.DockerService._restore_gpu_power_for_uncapped_pod(
        SimpleNamespace(redis_service=object()), ssh, payload, None, {}, record_rental_start=True
    )
    assert raise_mock.await_args.kwargs["pod_on_host"] is expected
    assert "docker container inspect pod_pod-1" in ssh.run.await_args.args[0]
    assert "docker container inspect pod_pod-1__prev" in ssh.run.await_args.args[0]
    assert "--filter name=kept_unmounted_pod-1_ " in ssh.run.await_args.args[0]
    assert "grep -q '^kept_unmounted_pod-1_'" in ssh.run.await_args.args[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("rented", "synced", "expected"),
    [
        ({"containers": [{"name": "pod_pod-1", "pod_id": "pod-1"}]}, False, True),
        ({"containers": [{"name": "pod_pod-2", "pod_id": "pod-2"}]}, True, False),
        (None, True, False),
        (None, False, "raises"),  # no backend list yet: absence proves nothing
    ],
)
async def test_the_backends_rented_list_marks_a_reboot_the_host_hides(monkeypatch, rented, synced, expected) -> None:
    # F-12 (lio#1924): the host removed both pod_pod-1 and pod_pod-1__prev; the backend's list still names the pod.
    import services.docker_service as ds

    raise_mock = AsyncMock(return_value=0)
    monkeypatch.setattr(ds, "restore_tracked_gpu_power_limits", AsyncMock())
    monkeypatch.setattr(ds, "raise_low_power_limits_to_default", raise_mock)
    redis_service = SimpleNamespace(
        get_rented_machine=AsyncMock(return_value=rented), rented_machines_synced=AsyncMock(return_value=synced)
    )
    payload = SimpleNamespace(gpu_uuids=["GPU-0"], executor_id="ex-1", pod_id="pod-1")
    restore = ds.DockerService._restore_gpu_power_for_uncapped_pod(
        SimpleNamespace(redis_service=redis_service), AsyncMock(), payload, SimpleNamespace(container_names=()), {},
        record_rental_start=True, executor_info=object(),
    )
    if expected == "raises":
        with pytest.raises(ds.GpuPowerRestoreStateError):
            await restore
        raise_mock.assert_not_awaited()
    else:
        await restore
        assert raise_mock.await_args.kwargs["pod_on_host"] is expected


@pytest.mark.asyncio
async def test_the_rented_list_is_rebuilt_in_one_transaction() -> None:
    # F-12: a reader between the old DELETE and the re-adds took a listed legacy rental for a new one
    from protocol.vc_protocol.compute_requests import RentedMachine
    from services.redis_service import RENTED_MACHINE_PREFIX, RENTED_MACHINES_SYNCED_KEY, RedisService

    service = SimpleNamespace(write_atomically=AsyncMock())
    machine = RentedMachine(
        miner_hotkey="hk", executor_id="ex-1", executor_ip_address="1.2.3.4", executor_ip_port="8001",
        containers=[{"name": "pod_pod-1", "pod_id": "pod-1"}],
    )
    await RedisService.replace_rented_machines(service, [machine])
    (writes,) = service.write_atomically.await_args.args
    assert [(op.command, op.args[0]) for op in writes.ops] == [
        ("delete", RENTED_MACHINE_PREFIX), ("hset", RENTED_MACHINE_PREFIX), ("set", RENTED_MACHINES_SYNCED_KEY),
    ]
    assert writes.ops[1].args[1] == "1.2.3.4:8001"


@pytest.mark.asyncio
async def test_an_unreadable_rented_list_fails_the_create_instead_of_guessing(monkeypatch) -> None:
    # a guess would record a new 80 % rental at 1.0 (or a reboot at its cut limit) for good
    import services.docker_service as ds

    monkeypatch.setattr(ds, "restore_tracked_gpu_power_limits", AsyncMock())
    raise_mock = AsyncMock(return_value=0)
    monkeypatch.setattr(ds, "raise_low_power_limits_to_default", raise_mock)
    redis_service = SimpleNamespace(get_rented_machine=AsyncMock(side_effect=RuntimeError("redis down")))
    payload = SimpleNamespace(gpu_uuids=["GPU-0"], executor_id="ex-1", pod_id="pod-1")
    with pytest.raises(ds.GpuPowerRestoreStateError):
        await ds.DockerService._restore_gpu_power_for_uncapped_pod(
            SimpleNamespace(redis_service=redis_service), AsyncMock(), payload, SimpleNamespace(container_names=()), {},
            record_rental_start=True, executor_info=object(),
        )
    raise_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_synced_marker_is_read_before_the_list(monkeypatch) -> None:
    # F-12: list read unsynced, then the first sync lands, then a set marker would vouch for a stale "absent"
    import services.docker_service as ds

    monkeypatch.setattr(ds, "restore_tracked_gpu_power_limits", AsyncMock())
    monkeypatch.setattr(ds, "raise_low_power_limits_to_default", AsyncMock(return_value=0))
    order: list[str] = []
    redis_service = SimpleNamespace(
        rented_machines_synced=AsyncMock(side_effect=lambda: order.append("marker") or False),
        get_rented_machine=AsyncMock(side_effect=lambda _: order.append("list")),
    )
    payload = SimpleNamespace(gpu_uuids=["GPU-0"], executor_id="ex-1", pod_id="pod-1")
    with pytest.raises(ds.GpuPowerRestoreStateError):
        await ds.DockerService._restore_gpu_power_for_uncapped_pod(
            SimpleNamespace(redis_service=redis_service), AsyncMock(), payload, SimpleNamespace(container_names=()), {},
            record_rental_start=True, executor_info=object(),
        )
    assert order == ["marker", "list"]


@pytest.mark.parametrize(
    ("first_query", "retry_watts", "scored", "after_first", "kept"),
    [
        # nvidia-smi timed out: nothing was read, nothing recorded; the retry records the 80 % tier
        (AsyncMock(side_effect=asyncio.TimeoutError()), (280, 280), None, None, [0.8, 0.8]),
        # partial output: GPU-0 read in the tier, GPU-1 missing and never scored in it: stock; the retry keeps both
        (AsyncMock(return_value=_power(280) | {"GPU-1": None}), (280, 280), None, [0.8, 1.0], [0.8, 1.0]),
        # F-36: GPU-1 missing but scored in the 80 % tier before the rental: it starts at what the renter saw
        (AsyncMock(return_value=_power(280) | {"GPU-1": None}), (280, 280), "0.8", [0.8, 0.8], [0.8, 0.8]),
        # F-43: GPU-1 missing, its tier cache a stale 0.8 (a lost clear); the retry reads it at stock: 1.0
        (AsyncMock(return_value=_power(350) | {"GPU-1": None}), (350, 350), "0.8", [1.0, 0.8], [1.0, 1.0]),
        # GPU-0 read at stock, GPU-1 missing; the host cuts GPU-0 to 80 % before the retry: it keeps 1.0
        (AsyncMock(return_value=_power(350) | {"GPU-1": None}), (280, 280), None, [1.0, 1.0], [1.0, 1.0]),
        # a scored value outside the tier proves nothing: stock
        (AsyncMock(return_value=_power(350) | {"GPU-1": None}), (280, 280), "0.5", [1.0, 1.0], [1.0, 1.0]),
    ],
)
@pytest.mark.asyncio
async def test_an_unreadable_rental_start_fails_the_create_and_a_retry_cannot_lower_it(
    monkeypatch, first_query, retry_watts, scored, after_first, kept
) -> None:
    monkeypatch.setattr(gpl, "_set_side_by_side", AsyncMock(return_value=0))
    monkeypatch.setattr(gpl, "_query_power_state", first_query)
    redis = SharedFakeRedis()
    if scored:
        await redis.set(gpl.scored_tier_ratio_key("ex-1", "GPU-1"), scored)
    with pytest.raises(RuntimeError):
        await gpl.raise_low_power_limits_to_default(AsyncMock(), "ex-1", ["GPU-0", "GPU-1"], redis=redis, pod_id="pod-1")
    recorded = any(k.startswith("gpu_power_rental_start_ratio:") for k in redis.store)
    assert ([_ratio(redis, g)["ratio"] for g in ("GPU-0", "GPU-1")] if recorded else None) == after_first
    await _rental_start(monkeypatch, redis, _power(*retry_watts))  # the retried create
    assert [_ratio(redis, g)["ratio"] for g in ("GPU-0", "GPU-1")] == kept
    await _rental_start(monkeypatch, redis, _power(*retry_watts))  # a reboot keeps the start
    assert [_ratio(redis, g)["ratio"] for g in ("GPU-0", "GPU-1")] == kept


@pytest.mark.parametrize(
    ("scored", "final_watts", "kept"),
    [
        (None, 350, 1.0),  # never scored in the tier: stock guess, a stock reading keeps it
        (None, 280, 1.0),  # stock guess: a later tier reading cannot lower it
        ("0.8", 280, 0.8),  # scored in the tier and read there: the tier
        ("0.8", 350, 1.0),  # stale tier cache, read at stock: stock (F-43)
    ],
)
@pytest.mark.asyncio
async def test_repeated_partial_reads_keep_failing_the_create_until_a_real_reading(
    monkeypatch, scored, final_watts, kept
) -> None:
    # lio#1924 r4477687020: a second miss must not let the rental start on the guess
    monkeypatch.setattr(gpl, "_set_side_by_side", AsyncMock(return_value=0))
    redis = SharedFakeRedis()
    if scored:
        await redis.set(gpl.scored_tier_ratio_key("ex-1", "GPU-1"), scored)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="GPU-1"):
            await _rental_start(monkeypatch, redis, _power(350) | {"GPU-1": None})
        assert _ratio(redis, "GPU-1")["guessed"] is True
    await _rental_start(monkeypatch, redis, _power(350, final_watts))
    assert _ratio(redis, "GPU-1") == {"pod_id": "pod-1", "ratio": kept}
    assert _ratio(redis, "GPU-0")["ratio"] == 1.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (None, GpuPowerCapResult.APPLIED),
        ("nonzero", GpuPowerCapResult.HOST_REFUSED),
        ("not_stuck", GpuPowerCapResult.HOST_REFUSED),
        ("timeout", GpuPowerCapResult.FAILED),
        ("redis", GpuPowerCapResult.FAILED),
    ],
)
async def test_apply_tells_a_host_refusal_from_our_own_error(failure, expected):
    # Arrange: one GPU to cap at 200 W; the host's -pl fails or does not stick, or our SSH or Redis fails.
    redis = FakeRedis()
    if failure == "redis":
        redis.set = AsyncMock(side_effect=ConnectionError("redis down"))

    async def run(command, **kwargs):
        if command == POWER_STATE_CMD:
            return FakeRun(stdout=STATE_CSV)
        if " -pl 200" in command:
            if failure == "timeout":
                raise TimeoutError("hung")
            return FakeRun(exit_status=1 if failure == "nonzero" else 0)
        return FakeRun(stdout=f"{400 if failure == 'not_stuck' else 200}, Enabled\n")

    ssh = AsyncMock(run=AsyncMock(side_effect=run))
    # Act / Assert: only the host's own refusal reads HOST_REFUSED.
    result = await apply_filler_gpu_power_limits(
        ssh, [GpuPowerLimit(gpu_uuid="GPU-a", watts=200)], redis, POD_ID, "executor-1"
    )
    assert result is expected


@pytest.mark.asyncio
async def test_pod_restore_never_raises_a_gpu_of_an_index_without_caps() -> None:
    # An index written before caps were stored cannot tell Lium's cap from the host's own limit, which
    # another path's restore may already have set: a GPU with no record is left as it is.
    ssh = fake_ssh(FakeRun(stdout=STATE_CSV))
    redis = FakeRedis({_pod_index_key(POD_ID): json.dumps(["GPU-a"])})

    restored = await restore_filler_pod_gpu_power_limits(ssh, redis, POD_ID, executor_id=EXECUTOR_ID)

    assert restored == 0
    assert not any(" -pl " in command for command in _commands(ssh))
    assert _pod_index_key(POD_ID) not in redis.store


@pytest.mark.asyncio
async def test_pod_restore_raises_a_gpu_capped_above_the_floor_without_record() -> None:
    # 368 W of 400 W is Lium's 0.92 cap: above the check's floor, still not the renter's default.
    ssh = fake_ssh(FakeRun(stdout="GPU-a, 368, 400, 100, 400\n"), *_set_ok(400))
    redis = FakeRedis({_pod_index_key(POD_ID): json.dumps({"GPU-a": 368})})

    await restore_filler_pod_gpu_power_limits(ssh, redis, POD_ID, executor_id=EXECUTOR_ID)

    assert _commands(ssh)[1:] == _set_commands("GPU-a", 400)


@pytest.mark.asyncio
async def test_pod_restore_leaves_a_recorded_gpu_to_its_record_and_raises_only_the_other() -> None:
    # GPU-a's record holds the host's own 380 W and is restored to it; GPU-b lost its record and goes
    # to its default. A GPU already at its default is not touched.
    state = "GPU-a, 368, 400, 100, 400\nGPU-b, 230, 250, 100, 250\nGPU-c, 250, 250, 100, 250\n"
    restored_state = state.replace("GPU-a, 368", "GPU-a, 380")
    ssh = fake_ssh(FakeRun(stdout=state), *_set_ok(380), FakeRun(stdout=restored_state), *_set_ok(250))
    redis = FakeRedis({
        _restore_key("GPU-a"): _record("GPU-a", 380),
        _pod_index_key(POD_ID): json.dumps({"GPU-a": 368, "GPU-b": 230, "GPU-c": None}),
    })

    restored = await restore_filler_pod_gpu_power_limits(ssh, redis, POD_ID, executor_id=EXECUTOR_ID)

    assert restored == 1
    commands = _commands(ssh)
    assert commands[1:4] == _set_commands("GPU-a", 380)
    assert commands[5:] == _set_commands("GPU-b", 250)
    assert _restore_key("GPU-a") not in redis.store


@pytest.mark.asyncio
async def test_a_second_restore_of_the_same_pod_finds_no_index_and_raises_nothing() -> None:
    # A customer create and the filler's own delete one after the other: the first restores GPU-a to the
    # host's 300 W and drops the index; the second must not raise GPU-a to 400 W.
    redis = FakeRedis({
        _restore_key("GPU-a"): _record("GPU-a", 300),
        _pod_index_key(POD_ID): json.dumps(["GPU-a"]),
    })
    first = fake_ssh(FakeRun(stdout="GPU-a, 368, 400, 100, 400\n"), *_set_ok(300))
    second = fake_ssh()

    assert await restore_filler_pod_gpu_power_limits(first, redis, POD_ID, executor_id=EXECUTOR_ID) == 1
    assert await restore_filler_pod_gpu_power_limits(second, redis, POD_ID, executor_id=EXECUTOR_ID) == 0

    second.run.assert_not_called()


@pytest.mark.asyncio
async def test_a_parallel_strict_restore_sees_the_index_until_the_restore_is_done() -> None:
    # The filler's own delete is restoring GPU-a when a customer create's strict restore of the same pod
    # starts: while the GPU is still capped the strict path must find the index, not pass on a missing one.
    redis = FakeRedis({
        _restore_key("GPU-a"): _record("GPU-a", 300),
        _pod_index_key(POD_ID): json.dumps({"GPU-a": 368}),
    })
    seen_during_restore: list[bool] = []
    runs = iter([FakeRun(stdout="GPU-a, 368, 400, 100, 400\n"), *_set_ok(300), FakeRun(stdout="GPU-a, 300, 400, 100, 400\n")])

    async def run(command, **kwargs):
        if " -pl " in command:
            seen_during_restore.append(_pod_index_key(POD_ID) in redis.store)
        return next(runs)

    ssh = AsyncMock(run=AsyncMock(side_effect=run))

    assert await restore_filler_pod_gpu_power_limits(ssh, redis, POD_ID, executor_id=EXECUTOR_ID) == 1

    assert seen_during_restore == [True]
    assert _pod_index_key(POD_ID) not in redis.store


@pytest.mark.asyncio
async def test_pod_restore_raises_a_gpu_still_at_the_recorded_cap_to_its_default() -> None:
    # The index says GPU-a was capped at 368 W; no record is left and it still reads 368 W.
    ssh = fake_ssh(FakeRun(stdout="GPU-a, 368, 400, 100, 400\n"), *_set_ok(400))
    redis = FakeRedis({_pod_index_key(POD_ID): json.dumps({"GPU-a": 368})})

    await restore_filler_pod_gpu_power_limits(ssh, redis, POD_ID, executor_id=EXECUTOR_ID)

    assert _commands(ssh)[1:] == _set_commands("GPU-a", 400)


@pytest.mark.asyncio
async def test_pod_restore_leaves_a_gpu_another_path_already_restored() -> None:
    # A create's restore net restored GPU-a to the host's 300 W and deleted its record before this
    # delete ran: it no longer reads the 368 W cap, so the host limit stays.
    ssh = fake_ssh(FakeRun(stdout="GPU-a, 300, 400, 100, 400\n"))
    redis = FakeRedis({_pod_index_key(POD_ID): json.dumps({"GPU-a": 368})})

    assert await restore_filler_pod_gpu_power_limits(ssh, redis, POD_ID, executor_id=EXECUTOR_ID) == 0

    assert not any(" -pl " in command for command in _commands(ssh))


@pytest.mark.asyncio
async def test_pod_restore_raises_a_gpu_whose_restore_failed_while_still_at_the_cap() -> None:
    # The record stays for the safety nets, but the renter must not start at the cap in the meantime.
    ssh = AsyncMock()
    ssh.run.side_effect = [
        FakeRun(stdout="GPU-a, 368, 400, 100, 400\n"),
        FakeRun(),
        TimeoutError("nvidia-smi hung"),
        FakeRun(stdout="GPU-a, 368, 400, 100, 400\n"),
        *_set_ok(400),
    ]
    redis = FakeRedis({
        _restore_key("GPU-a"): _record("GPU-a", 380),
        _pod_index_key(POD_ID): json.dumps({"GPU-a": 368}),
    })

    assert await restore_filler_pod_gpu_power_limits(ssh, redis, POD_ID, executor_id=EXECUTOR_ID) == 0

    assert _commands(ssh)[4:] == _set_commands("GPU-a", 400)
    assert _restore_key("GPU-a") in redis.store


@pytest.mark.asyncio
async def test_apply_keeps_the_cap_of_a_gpu_it_lowered() -> None:
    ssh = fake_ssh(FakeRun(stdout="GPU-a, 400, 400, 100, 400\n"), *_set_ok(368))
    redis = FakeRedis()

    assert await apply_filler_gpu_power_limits(ssh, _limits(GPU_a=368), redis, POD_ID, EXECUTOR_ID) is GpuPowerCapResult.APPLIED

    assert json.loads(redis.store[_pod_index_key(POD_ID)]) == {"GPU-a": 368}


@pytest.mark.asyncio
async def test_apply_keeps_no_cap_for_a_host_limit_within_a_watt_of_it() -> None:
    # 369 W against a 368 W cap would read as Lium's cap after the restore and be raised to default.
    ssh = fake_ssh(FakeRun(stdout="GPU-a, 369, 400, 100, 400\n"), *_set_ok(368))
    redis = FakeRedis()

    assert await apply_filler_gpu_power_limits(ssh, _limits(GPU_a=368), redis, POD_ID, EXECUTOR_ID) is GpuPowerCapResult.APPLIED

    assert json.loads(redis.store[_pod_index_key(POD_ID)]) == {"GPU-a": None}


@pytest.mark.asyncio
async def test_a_gpu_the_cap_raised_is_raised_to_default_when_its_record_is_lost() -> None:
    # A host under the floor (360 W of 400 W) was capped up to 368 W; its record is gone and it still
    # reads 368 W, so the renter must not start there.
    apply_ssh = fake_ssh(FakeRun(stdout="GPU-a, 360, 400, 100, 400\n"), *_set_ok(368))
    redis = FakeRedis()
    assert await apply_filler_gpu_power_limits(apply_ssh, _limits(GPU_a=368), redis, POD_ID, EXECUTOR_ID) is GpuPowerCapResult.APPLIED
    del redis.store[_restore_key("GPU-a")]
    delete_ssh = fake_ssh(FakeRun(stdout="GPU-a, 368, 400, 100, 400\n"), *_set_ok(400))

    await restore_filler_pod_gpu_power_limits(delete_ssh, redis, POD_ID, executor_id=EXECUTOR_ID)

    assert _commands(delete_ssh)[1:] == _set_commands("GPU-a", 400)


@pytest.mark.asyncio
async def test_a_leftover_record_near_the_cap_keeps_the_host_limit_after_a_reboot() -> None:
    # An earlier restore failed and kept the host's 369 W record; the host rebooted to its 400 W default.
    # The next filler caps at 368 W: what the restore writes back (369 W) is within a watt of the cap, so
    # no cap is kept and the delete leaves the host's 369 W in place.
    redis = FakeRedis({_restore_key("GPU-a"): _record("GPU-a", 369)})
    apply_ssh = fake_ssh(FakeRun(stdout="GPU-a, 400, 400, 100, 400\n"), *_set_ok(368))
    assert await apply_filler_gpu_power_limits(apply_ssh, _limits(GPU_a=368), redis, POD_ID, EXECUTOR_ID) is GpuPowerCapResult.APPLIED
    assert json.loads(redis.store[_pod_index_key(POD_ID)]) == {"GPU-a": None}
    delete_ssh = fake_ssh(FakeRun(stdout="GPU-a, 368, 400, 100, 400\n"), *_set_ok(369))

    assert await restore_filler_pod_gpu_power_limits(delete_ssh, redis, POD_ID, executor_id=EXECUTOR_ID) == 1

    assert _commands(delete_ssh)[1:] == _set_commands("GPU-a", 369)


@pytest.mark.asyncio
async def test_a_gpu_still_at_an_old_cap_keeps_the_new_cap_when_its_record_says_otherwise() -> None:
    # The GPU still reads an old 368 W cap, but its leftover record holds the host's real 400 W: the cap
    # is kept, so if that record is later lost the GPU is raised instead of reaching a renter at 368 W.
    redis = FakeRedis({_restore_key("GPU-a"): _record("GPU-a", 400)})
    ssh = fake_ssh(FakeRun(stdout="GPU-a, 368, 400, 100, 400\n"), *_set_ok(368))

    assert await apply_filler_gpu_power_limits(ssh, _limits(GPU_a=368), redis, POD_ID, EXECUTOR_ID) is GpuPowerCapResult.APPLIED

    assert json.loads(redis.store[_pod_index_key(POD_ID)]) == {"GPU-a": 368}


@pytest.mark.asyncio
async def test_pod_restore_never_raises_a_gpu_the_cap_did_not_lower() -> None:
    # The host runs at 365 W of 400 W and the backend asked for no more than that: the GPU read 365 W
    # before and after, so after the restore it is the host's limit, not Lium's cap.
    ssh = fake_ssh(FakeRun(stdout="GPU-a, 365, 400, 100, 400\n"), *_set_ok(365))
    redis = FakeRedis({
        _restore_key("GPU-a"): _record("GPU-a", 365),
        _pod_index_key(POD_ID): json.dumps({"GPU-a": None}),
    })

    assert await restore_filler_pod_gpu_power_limits(ssh, redis, POD_ID, executor_id=EXECUTOR_ID) == 1

    assert _commands(ssh)[1:] == _set_commands("GPU-a", 365)


class RecordReadFailsRedis(FakeRedis):
    async def get(self, key: str) -> str | None:
        if key.startswith("gpu_power_restore:"):
            raise ConnectionError("redis down")
        return await super().get(key)


@pytest.mark.asyncio
async def test_pod_restore_raises_nothing_when_the_record_read_fails() -> None:
    # A failed read says nothing about which records exist: raising could override the host's limit.
    ssh = fake_ssh()
    redis = RecordReadFailsRedis({_pod_index_key(POD_ID): json.dumps(["GPU-a"])})

    assert await restore_filler_pod_gpu_power_limits(ssh, redis, POD_ID, executor_id=EXECUTOR_ID) == 0

    ssh.run.assert_not_called()


