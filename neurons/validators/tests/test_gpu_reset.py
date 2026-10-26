"""DAH-4658: the renter GPU reset (services/gpu_reset.py) against a fake host.

Every refusal code, and the hard invariants: the reset set is inside BOTH the container's own
DeviceRequests and the backend's GPUs, `nvidia-smi -r` never runs without `-i`, nothing is read from
inside the pod, and a stopped container is always started again.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest
from payload_models.payloads import GpuResetErrorCode
from services.gpu_reset import (
    build_reset_command,
    parse_host_gpus,
    parse_nvlink_peers,
    reset_pod_gpus,
)
from services.redis_service import RedisService

from services import gpu_reset
from tests.helpers import FakeRedis, build_services, build_state
from tests.test_machine_spec_scrape_check import make_command_result

CID = "c" * 64
NAME = "container_pod1"
U = [f"GPU-0000000{i}-0000-4000-8000-00000000000{i}" for i in range(4)]
# host bus ids deliberately out of index order: pod index follows the bus id
BUS = ["00000000:1B:00.0", "00000000:0A:00.0", "00000000:2C:00.0", "00000000:3D:00.0"]
TOPO_PCIE = "\tGPU0\tGPU1\tGPU2\tGPU3\tCPU Affinity\nGPU0\t X \tSYS\tSYS\tSYS\t0-15\nGPU1\tSYS\t X \tSYS\tSYS\t0-15\nGPU2\tSYS\tSYS\t X \tSYS\t0-15\nGPU3\tSYS\tSYS\tSYS\t X \t0-15\n"
TOPO_PAIR01 = TOPO_PCIE.replace("GPU0\t X \tSYS", "GPU0\t X \tNV12").replace("GPU1\tSYS\t X ", "GPU1\tNV12\t X ")
TOPO_PAIR02 = TOPO_PCIE.replace("GPU0\t X \tSYS\tSYS", "GPU0\t X \tSYS\tNV12").replace(
    "GPU2\tSYS\tSYS\t X ", "GPU2\tNV12\tSYS\t X "
)


class FakeHost:
    """Answers the commands the reset runs, by what they start with; records every command."""

    def __init__(
        self,
        *,
        device_ids=(U[0], U[1]),
        count=None,
        mig=(),
        nvswitch=False,
        topo=TOPO_PCIE,
        apps=(),
        apps_after_stop=(),
        cgroups=None,
        reset_exit=0,
        inspect_exit=0,
        stop_exit=0,
        main_pid=4242,
        mounts=(),
        own_cgroup="0::/user.slice/user-0.slice/session-7.scope",
    ):
        self.commands: list[str] = []
        self.stopped = False
        request = {"Driver": "", "Count": count or 0, "DeviceIDs": list(device_ids) or None, "Capabilities": [["gpu"]]}
        self.inspect = json.dumps(
            {"Id": CID, "State": {"Pid": main_pid}, "HostConfig": {"DeviceRequests": [request]}, "Mounts": list(mounts)}
        )
        self.gpus = "\n".join(
            f"{i}, {U[i]}, {BUS[i]}, {'Enabled' if i in mig else 'Disabled'}" for i in range(4)
        )
        self.nvswitch, self.topo = nvswitch, topo
        self.apps, self.apps_after_stop = list(apps), list(apps_after_stop)
        self.cgroups = {main_pid: f"0::/../docker-{CID}.scope", **(cgroups or {})}
        self.reset_exit, self.inspect_exit, self.stop_exit = reset_exit, inspect_exit, stop_exit
        self.own_cgroup = own_cgroup

    async def run(self, cmd: str, *, timeout: int = 60, retryable: bool = True, **_):
        self.commands.append(cmd)
        if cmd.startswith("/usr/bin/docker inspect"):
            return make_command_result(self.inspect_exit == 0, self.inspect, exit_code=self.inspect_exit)
        if cmd.startswith("/usr/bin/docker stop"):
            self.stopped = self.stop_exit == 0
            return make_command_result(self.stop_exit == 0, stderr="boom", exit_code=self.stop_exit)
        if cmd.startswith("nvidia-smi --query-gpu="):
            return make_command_result(True, self.gpus)
        if cmd.startswith("nvidia-smi --query-compute-apps="):
            apps = self.apps_after_stop if self.stopped else self.apps
            return make_command_result(True, "\n".join(f"{pid}, {uuid}" for pid, uuid in apps))
        if cmd.startswith("for p in /dev/nvidia-nvswitch"):
            return make_command_result(True, "/dev/nvidia-nvswitch0\n" if self.nvswitch else "")
        if cmd == "nvidia-smi topo -m":
            return make_command_result(True, self.topo)
        if cmd.endswith("< /proc/self/cgroup"):
            return make_command_result(True, f"{self.own_cgroup} ")
        if cmd.startswith("for p in "):
            pids = [int(p) for p in cmd.split(";")[0].split()[3:]]
            out = "\n".join(f"{pid} {self.cgroups[pid]} " for pid in pids if pid in self.cgroups)
            return make_command_result(True, out)
        if cmd.startswith("nvidia-smi -r"):
            return make_command_result(self.reset_exit == 0, "refused", exit_code=self.reset_exit)
        if cmd.startswith("CUDA_VISIBLE_DEVICES="):
            return make_command_result(True, "ctx open/close OK")
        raise AssertionError(f"unexpected command {cmd}")


async def _reset(host: FakeHost, *, backend=(U[0], U[1]), indices=None, start=None, **kwargs):
    start = start or AsyncMock()
    report = await reset_pod_gpus(
        host,
        container_name=NAME,
        pod_gpu_uuids=list(backend),
        gpu_indices=indices,
        start_container=start,
        settle_seconds=0,
        **kwargs,
    )
    _assert_invariants(host, backend, report)
    return report, start


def _assert_invariants(host: FakeHost, backend, report) -> None:
    container = set(json.loads(host.inspect)["HostConfig"]["DeviceRequests"][0]["DeviceIDs"] or U)
    for cmd in host.commands:
        assert "docker exec" not in cmd, "never trust output from inside the pod"
        if cmd.startswith("nvidia-smi -r"):
            assert " -i " in cmd, "never nvidia-smi -r without -i"
            reset_set = set(cmd.split(" -i ")[1].strip("'").split(","))
            assert reset_set <= container and reset_set <= set(backend)
    assert {gpu.uuid for gpu in report.gpus} <= container & set(backend)


def _no_change(host: FakeHost, start) -> None:
    assert not any(cmd.startswith(("/usr/bin/docker stop", "nvidia-smi -r")) for cmd in host.commands)
    start.assert_not_awaited()


@pytest.mark.asyncio
async def test_resets_the_pods_gpus_and_starts_the_container_again():
    host = FakeHost(apps=[(4242, U[1])])
    report, start = await _reset(host)
    assert report.ok and report.container_restarted and report.error_code is None and report.pod_stopped
    # pod index by host bus id: U[1] (0A) is pod GPU 0, U[0] (1B) pod GPU 1
    assert [(g.index, g.uuid, g.result) for g in report.gpus] == [(0, U[1], "reset"), (1, U[0], "reset")]
    assert f"nvidia-smi -r -i {U[1]},{U[0]}" in host.commands
    assert host.commands.index(f"/usr/bin/docker stop -t 10 {NAME}") < host.commands.index(
        f"nvidia-smi -r -i {U[1]},{U[0]}"
    )
    start.assert_awaited_once()


@pytest.mark.asyncio
async def test_one_index_resets_only_that_gpu():
    host = FakeHost()
    report, _ = await _reset(host, indices=[1])
    assert report.ok and [g.uuid for g in report.gpus] == [U[0]]


@pytest.mark.asyncio
async def test_a_whole_host_rental_resolves_gpus_all_on_the_host():
    host = FakeHost(device_ids=(), count=-1)
    report, _ = await _reset(host, backend=U)
    assert report.ok and {g.uuid for g in report.gpus} == set(U)


@pytest.mark.parametrize(
    ("host", "kwargs", "code"),
    [
        (FakeHost(mig=(0,)), {}, GpuResetErrorCode.MigEnabled),
        (FakeHost(nvswitch=True), {}, GpuResetErrorCode.NvlinkPeersOutsidePod),
        (FakeHost(topo=TOPO_PAIR02), {}, GpuResetErrorCode.NvlinkPeersOutsidePod),
        (FakeHost(apps=[(999, U[0])], cgroups={999: "0::/../docker-other.scope"}), {}, GpuResetErrorCode.GpuInUseOutsidePod),
        (FakeHost(apps=[(998, U[0])]), {}, GpuResetErrorCode.GpuInUseOutsidePod),  # cgroup unreadable
        (FakeHost(), {"in_tee": True}, GpuResetErrorCode.ResetNotSupported),
        (FakeHost(cgroups={4242: "0::/"}), {}, GpuResetErrorCode.ResetNotSupported),  # no host PID view
        (FakeHost(main_pid=0), {}, GpuResetErrorCode.ResetNotSupported),
        (
            FakeHost(mounts=[{"Destination": "/lium-cipher"}]),
            {"encrypted_volume_mount": "/lium-cipher"},
            GpuResetErrorCode.ResetNotSupported,
        ),
        (FakeHost(inspect_exit=1), {}, GpuResetErrorCode.ContainerNotFound),
        (FakeHost(device_ids=(U[0],)), {}, GpuResetErrorCode.GpuNotInContainer),  # backend's U[1] not given
        (FakeHost(), {"indices": [2]}, GpuResetErrorCode.GpuNotInContainer),
    ],
)
@pytest.mark.asyncio
async def test_every_precheck_refusal_changes_nothing(host, kwargs, code):
    report, start = await _reset(host, **kwargs)
    assert not report.ok and report.error_code == code and report.gpus == [] and not report.pod_stopped
    _no_change(host, start)


@pytest.mark.asyncio
async def test_a_mig_nvlink_peer_refuses_the_reset():
    # pod index 1 is U[0] (bus 1B); its NVLink peer U[1] is MIG, so widening to it must refuse
    host = FakeHost(topo=TOPO_PAIR01, mig=(1,))
    report, start = await _reset(host, indices=[1])
    assert report.error_code == GpuResetErrorCode.MigEnabled
    _no_change(host, start)


@pytest.mark.asyncio
async def test_nvlink_peers_inside_the_pod_are_reset_together():
    host = FakeHost(topo=TOPO_PAIR01)
    report, _ = await _reset(host, indices=[1])
    assert report.ok and {g.uuid for g in report.gpus} == {U[0], U[1]}
    assert "NVLink" in report.detail


@pytest.mark.asyncio
async def test_an_nvswitch_node_the_pod_owns_whole_resets_every_gpu():
    host = FakeHost(device_ids=(), count=-1, nvswitch=True)
    report, _ = await _reset(host, backend=U, indices=[0])
    assert report.ok and {g.uuid for g in report.gpus} == set(U)


@pytest.mark.asyncio
async def test_a_process_left_after_the_stop_aborts_the_reset_but_restarts_the_pod():
    host = FakeHost(apps_after_stop=[(777, U[0])])
    report, start = await _reset(host)
    assert not report.ok and report.error_code == GpuResetErrorCode.GpuInUseOutsidePod
    assert not any(cmd.startswith("nvidia-smi -r") for cmd in host.commands)
    assert report.container_restarted
    start.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_refused_reset_falls_back_to_the_latch_cure_on_the_pods_gpus_only():
    host = FakeHost(reset_exit=1)
    report, start = await _reset(host)
    assert report.ok and report.error_code is None and "nvidia-smi -r refused" in report.detail
    assert {(g.uuid, g.result) for g in report.gpus} == {(U[0], "latch_cleared"), (U[1], "latch_cleared")}
    cured = [cmd for cmd in host.commands if cmd.startswith("CUDA_VISIBLE_DEVICES=")]
    assert len(cured) == 2 and all(U[2] not in c and U[3] not in c for c in cured)
    start.assert_awaited_once()


@pytest.mark.asyncio
async def test_the_container_is_started_even_when_the_reset_raises():
    host = FakeHost()
    with patch.object(gpu_reset, "build_reset_command", side_effect=RuntimeError("bug")):
        report, start = await _reset(host)
    assert not report.ok and report.container_restarted
    start.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_failed_start_is_reported():
    report, _ = await _reset(FakeHost(), start=AsyncMock(side_effect=RuntimeError("no start")))
    assert not report.ok and not report.container_restarted and "did not start" in report.detail


def test_reset_command_always_names_its_gpus_and_only_the_pods():
    with pytest.raises(ValueError):
        build_reset_command([], container_uuids=U, pod_gpu_uuids=U)
    with pytest.raises(ValueError):  # in the backend's list but not the container's
        build_reset_command([U[2]], container_uuids=U[:2], pod_gpu_uuids=U)
    with pytest.raises(ValueError):  # in the container but not the backend's list
        build_reset_command([U[2]], container_uuids=U, pod_gpu_uuids=U[:2])
    assert build_reset_command([U[0]], container_uuids=U, pod_gpu_uuids=U) == f"nvidia-smi -r -i {U[0]}"


def test_topology_parse_reads_nvlink_cells_only():
    hosts = parse_host_gpus("\n".join(f"{i}, {U[i]}, {BUS[i]}, Disabled" for i in range(4)))
    assert parse_nvlink_peers(TOPO_PCIE, hosts) == {}
    assert parse_nvlink_peers("\x1b[4m" + TOPO_PAIR01 + "\x1b[0m", hosts) == {U[0]: {U[1]}, U[1]: {U[0]}}


def test_topology_parse_reads_an_8x_h200_header_without_its_leading_tab():
    """`nvidia-smi topo -m` of an 8x H200 (NVSwitch, no /dev/nvidia-nvswitch*) as the runner returns it: stripped."""
    h200 = [f"GPU-8{i}000000-0000-4000-8000-00000000000{i}" for i in range(8)]
    hosts = parse_host_gpus("\n".join(f"{i}, {h200[i]}, 00000000:{i:02X}:00.0, Disabled" for i in range(8)))
    gpus = "\t".join(f"GPU{i}" for i in range(8))
    rows = [
        f"GPU{i}\t" + "\t".join(" X " if j == i else "NV18" for j in range(8)) + f"\tSYS\t{'0-47' if i < 4 else '48-95'}\t{i // 4}\t\tN/A"
        for i in range(8)
    ]
    topology = "\n".join([f"{gpus}\tNIC0\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID", *rows, "", "Legend:", "  X    = Self"])
    peers = parse_nvlink_peers(topology, hosts)
    assert peers[h200[0]] == set(h200[1:])  # a 2-of-8 pod has peers outside it: the reset is refused
    assert len(peers) == 8


# --- the pending-pod mark and the checks that skip on it ------------------------------------------


class PendingRedis(FakeRedis):
    get_pending_pods = RedisService.get_pending_pods
    add_pending_pod = RedisService.add_pending_pod
    remove_pending_pod = RedisService.remove_pending_pod
    gpu_reset_marks = RedisService.gpu_reset_marks
    renting_in_progress = RedisService.renting_in_progress


@pytest.mark.asyncio
async def test_the_gpu_reset_mark_is_its_own_kind_of_pending_pod():
    redis = PendingRedis()
    await redis.add_pending_pod("hk", "ex", "pod-rent")
    assert not await gpu_reset.renter_gpu_reset_in_progress(redis, "hk", "ex")
    await redis.add_pending_pod("hk", "ex", "pod1", kind="gpu_reset", gpu_uuids=U[:2])
    assert await gpu_reset.renter_gpu_reset_in_progress(redis, "hk", "ex")
    assert await gpu_reset.renter_gpu_reset_in_progress(redis, "hk", "ex", "pod1")
    assert not await gpu_reset.renter_gpu_reset_in_progress(redis, "hk", "ex", "pod-rent")
    assert (await redis.gpu_reset_marks("hk", "ex"))[0]["gpu_uuids"] == U[:2]
    assert await redis.renting_in_progress("hk", "ex", "pod1")  # a second reset or create is declined
    await redis.remove_pending_pod("hk", "ex", "pod1")
    assert not await gpu_reset.renter_gpu_reset_in_progress(redis, "hk", "ex")


@pytest.mark.asyncio
async def test_the_rented_gpu_checks_skip_a_node_whose_renter_is_resetting(context_factory):
    from services.task.checks.gpu_usage import GpuUsageCheck
    from services.task.checks.rented_gpu_drop import RentedGpuDropCheck
    from services.task.messages import GpuUsageMessages, RentedGpuDropMessages

    from tests.test_rented_gpu_drop_check import _ctx

    redis = PendingRedis()
    services = build_services(redis=redis)
    ctx = _ctx(context_factory, services, listed=[], count=0, scrape_error="NVMLError(999)")
    await redis.add_pending_pod(ctx.miner_hotkey, ctx.executor.uuid, "pod1", kind="gpu_reset")

    drop = await RentedGpuDropCheck().run(ctx)
    usage = await GpuUsageCheck().run(ctx)

    assert drop.passed and drop.event.reason_code == RentedGpuDropMessages.GPU_RESET_IN_PROGRESS.reason
    assert usage.passed and usage.event.reason_code == GpuUsageMessages.GPU_RESET_IN_PROGRESS.reason
    services.backend.report_rented_gpu_drop.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_drop_check_skips_only_when_every_missing_gpu_is_under_reset(context_factory):
    from services.task.checks.rented_gpu_drop import RentedGpuDropCheck
    from services.task.messages import RentedGpuDropMessages

    from tests.test_rented_gpu_drop_check import UUIDS, _ctx

    redis = PendingRedis()
    ctx = _ctx(context_factory, build_services(redis=redis), listed=UUIDS[1:], count=7)
    await redis.add_pending_pod(ctx.miner_hotkey, ctx.executor.uuid, "pod1", kind="gpu_reset", gpu_uuids=UUIDS[1:2])
    other_tenant = await RentedGpuDropCheck().run(ctx)  # UUIDS[0] is missing and not under reset
    await redis.add_pending_pod(ctx.miner_hotkey, ctx.executor.uuid, "pod2", kind="gpu_reset", gpu_uuids=UUIDS[:1])
    skipped = await RentedGpuDropCheck().run(ctx)

    assert other_tenant.event.reason_code != RentedGpuDropMessages.GPU_RESET_IN_PROGRESS.reason
    assert skipped.passed and skipped.event.reason_code == RentedGpuDropMessages.GPU_RESET_IN_PROGRESS.reason


@pytest.mark.asyncio
async def test_the_usage_check_skips_only_the_gpus_of_the_pod_under_reset(context_factory):
    from services.task.checks import gpu_usage
    from services.task.messages import GpuUsageMessages

    redis = PendingRedis()
    busy = [{"uuid": U[0], "gpu_utilization": 0, "memory_utilization": 0}, {"uuid": U[1], "gpu_utilization": 99}]
    ctx = context_factory(services=build_services(redis=redis), state=build_state(gpu_details=busy))
    await redis.add_pending_pod(ctx.miner_hotkey, ctx.executor.uuid, "pod1", kind="gpu_reset", gpu_uuids=[U[0]])

    with patch.object(gpu_usage.GpuUsageCheck, "_detect_and_cure_ghost_gpus", AsyncMock(return_value=None)), \
            patch.object(gpu_usage.GpuUsageCheck, "_judge_process_based_usage", AsyncMock()) as judge:
        await gpu_usage.GpuUsageCheck().run(ctx)
        assert judge.await_args.args[1] == busy[1:]  # the other tenant's GPU is still judged

        await redis.add_pending_pod(ctx.miner_hotkey, ctx.executor.uuid, "pod2", kind="gpu_reset", gpu_uuids=[U[1]])
        skipped = await gpu_usage.GpuUsageCheck().run(ctx)
    assert skipped.passed and skipped.event.reason_code == GpuUsageMessages.GPU_RESET_IN_PROGRESS.reason


@pytest.mark.asyncio
async def test_a_redis_error_never_skips_the_checks():
    assert await gpu_reset.renter_gpu_reset_in_progress(FakeRedis(failing=True), "hk", "ex") is False
    assert await gpu_reset.renter_gpu_reset_in_progress(AsyncMock(), "hk", "ex") is False  # not a real True


@pytest.mark.asyncio
async def test_no_verified_job_reset_is_sent_for_a_node_whose_renter_is_resetting():
    import asyncio
    from unittest.mock import MagicMock

    from clients.compute_client import ComputeClient
    from services.redis_service import RESET_VERIFIED_JOB_CHANNEL

    def message(executor_uuid: str, check_id: str = "executor.validate.rented_state", reason_code=None, evidence=None):
        data = {"miner_hotkey": "hk", "executor_uuid": executor_uuid, "reason": 0, "check_id": check_id,
                "reason_code": reason_code, "evidence": evidence}
        return {"channel": RESET_VERIFIED_JOB_CHANNEL.encode(), "data": json.dumps(data)}

    class PubSub:
        async def listen(self):
            yield message("resetting")
            yield message("resetting", "gpu.validate.duplicate")  # not something a reset causes: still sent
            yield message("resetting", "gpu.validate.fingerprint", "GPU_MISSING")
            yield message("resetting", "gpu.validate.fingerprint", "GPU_UUID_CHANGED")  # a swap: still sent
            yield message("resetting", evidence={"pod_id": "pod-other"})  # another pod on the node: still sent
            yield message("idle")
            raise asyncio.CancelledError  # ends the subscriber's reconnect loop

        async def aclose(self):
            pass

    redis = PendingRedis()
    await redis.add_pending_pod("hk", "resetting", "pod1", kind="gpu_reset")
    redis.subscribe = AsyncMock(return_value=PubSub())
    client = ComputeClient.__new__(ComputeClient)
    client.lock = asyncio.Lock()
    client.message_queue = []
    client.logging_extra = {}
    client.miner_service = MagicMock(redis_service=redis)
    client.my_hotkey = MagicMock(return_value="validator-hotkey")

    with pytest.raises(asyncio.CancelledError):
        await client.subscribe_mesages_from_redis()

    assert [(request.executor_uuid, request.check_id) for request in client.message_queue] == [
        ("resetting", "gpu.validate.duplicate"),
        ("resetting", "gpu.validate.fingerprint"),
        ("resetting", "executor.validate.rented_state"),
        ("idle", "executor.validate.rented_state"),
    ]


@pytest.mark.asyncio
async def test_a_reset_that_never_reached_the_executor_answers_with_a_reset_result():
    from payload_models.payloads import (
        FailedContainerErrorCodes,
        PodGpuResetResult,
        ResetPodGpuRequest,
    )
    from services.miner_service import MinerService

    payload = ResetPodGpuRequest(
        miner_hotkey="hk", executor_id="ex", pod_id="pod1", container_name=NAME, gpu_uuids=U[:2], request_id="r1"
    )
    service = MinerService.__new__(MinerService)
    busy = service._handle_container_error(payload, "busy", FailedContainerErrorCodes.RentingInProgress)
    gone = service._handle_container_error(payload, "gone", FailedContainerErrorCodes.UnknownError)

    assert isinstance(busy, PodGpuResetResult) and busy.request_id == "r1" and not busy.ok
    assert busy.error_code == GpuResetErrorCode.OperationInProgress
    assert gone.error_code == GpuResetErrorCode.ExecutorUnreachable


@pytest.mark.asyncio
async def test_the_validators_own_session_processes_do_not_block_the_reset():
    own = "0::/user.slice/user-0.slice/session-7.scope"
    host = FakeHost(apps=[(4242, U[0]), (900, U[1])], cgroups={900: own})
    report, _ = await _reset(host)
    assert report.ok and any(cmd.startswith("nvidia-smi -r") for cmd in host.commands)


@pytest.mark.asyncio
async def test_the_validators_own_probe_left_after_the_stop_does_not_refuse_the_reset():
    own = "0::/user.slice/user-0.slice/session-7.scope"
    host = FakeHost(apps_after_stop=[(900, U[0])], cgroups={900: own})
    report, _ = await _reset(host)
    assert report.ok and any(cmd.startswith("nvidia-smi -r") for cmd in host.commands)

    stranger = FakeHost(apps_after_stop=[(901, U[0])], cgroups={901: "0::/system.slice/other.service"})
    report, _ = await _reset(stranger)
    assert not report.ok and report.error_code == GpuResetErrorCode.GpuInUseOutsidePod


@pytest.mark.asyncio
async def test_a_root_own_cgroup_ignores_nothing():
    host = FakeHost(apps=[(900, U[0])], cgroups={900: "0::/"}, own_cgroup="0::/")
    report, start = await _reset(host)
    assert not report.ok and report.error_code == GpuResetErrorCode.GpuInUseOutsidePod
    _no_change(host, start)


@pytest.mark.asyncio
async def test_a_failed_docker_stop_does_not_report_the_pod_stopped():
    host = FakeHost(stop_exit=1)
    report, _ = await _reset(host)
    assert not report.ok and not report.pod_stopped
    assert not any(cmd.startswith("nvidia-smi -r") for cmd in host.commands)
