"""DAH-5301: a reboot resets a hung pod GPU (services/hung_gpu_reset.py) against a fake host.

A healthy reboot runs only the one query; a hung GPU is reset under Reset GPU's refusal rules, and
`nvidia-smi -r` never touches a GPU outside the pod.
"""

import contextlib

import pytest
from payload_models.payloads import ContainerWarningCode as W
from services import gpu_reset
from services.hung_gpu_reset import HUNG_GPU_QUERY_COMMAND, reset_hung_pod_gpus
from services.redis_service import GPU_RESET_PENDING_KIND
from tests.test_gpu_reset import TOPO_PAIR01, TOPO_PAIR02, U, FakeHost, PendingRedis
from tests.test_machine_spec_scrape_check import make_command_result

POD = [U[0], U[1]]


def _query(lines: dict[str, str] | None = None, apps: str = "") -> str:
    """The hung query's answer: every host GPU idle unless `lines` overrides one (None drops it)."""
    rows = {uuid: f"{uuid}, 0, 0" for uuid in U}
    rows.update(lines or {})
    return "\n".join(row for row in rows.values() if row is not None) + f"\n--compute-apps--\n{apps}"


class HungHost(FakeHost):
    def __init__(self, *, query=None, query_exit=0, cure_ok=True, **kwargs):
        super().__init__(**kwargs)
        self.query = _query() if query is None else query
        self.query_exit, self.cure_ok = query_exit, cure_ok

    async def run(self, cmd: str, *, timeout: int = 60, retryable: bool = True, **_):
        if cmd == HUNG_GPU_QUERY_COMMAND:
            self.commands.append(cmd)
            return make_command_result(self.query_exit == 0, self.query, exit_code=self.query_exit)
        if cmd.startswith("CUDA_VISIBLE_DEVICES=") and not self.cure_ok:
            self.commands.append(cmd)
            return make_command_result(False, "CUDA_ERROR_UNKNOWN", exit_code=1)
        return await super().run(cmd, timeout=timeout, retryable=retryable)


async def _reset(host: HungHost, pod=POD):
    result = await reset_hung_pod_gpus(host, list(pod))
    for cmd in host.commands:
        if cmd.startswith("nvidia-smi -r"):
            assert " -i " in cmd
            assert set(cmd.split(" -i ")[1].strip("'").split(",")) <= set(pod), "never reset a GPU outside the pod"
    return result


def _resets(host: HungHost) -> list[str]:
    return [cmd for cmd in host.commands if cmd.startswith("nvidia-smi -r")]


@pytest.mark.asyncio
async def test_a_healthy_reboot_runs_only_the_one_query():
    # a GPU outside the pod being broken is not the pod's business
    host = HungHost(query=_query({U[2]: f"{U[2]}, [Unknown Error], [Unknown Error]"}))
    result = await _reset(host)
    assert host.commands == [HUNG_GPU_QUERY_COMMAND]
    assert result.warning is None and result.hung_uuids == []


@pytest.mark.parametrize(
    "line",
    [
        f"{U[0]}, [Unknown Error], [Unknown Error]",
        f"{U[0]}, [GPU is lost], [GPU is lost]",
        f"{U[0]}, ERR!, ERR!",
        None,  # missing from the output
    ],
)
@pytest.mark.asyncio
async def test_a_hung_pod_gpu_is_reset_alone(line):
    host = HungHost(query=_query({U[0]: line}))
    result = await _reset(host)
    assert _resets(host) == [f"nvidia-smi -r -i {U[0]}"]
    assert result.warning == W.HungGpuReset and result.hung_uuids == [U[0]]


@pytest.mark.parametrize("cure_ok", [True, False])
@pytest.mark.asyncio
async def test_a_latched_gpu_gets_the_context_cycle_before_any_reset(cure_ok):
    # the DAH-2427 latch (busy, no memory, no process); a just-stopped healthy GPU can look the same
    host = HungHost(query=_query({U[0]: f"{U[0]}, 100, 0"}), cure_ok=cure_ok)
    result = await _reset(host)
    assert any(cmd.startswith("CUDA_VISIBLE_DEVICES=") for cmd in host.commands)
    assert _resets(host) == ([] if cure_ok else [f"nvidia-smi -r -i {U[0]}"])
    assert result.hung_uuids == [U[0]]
    # a latch clear that fixed the GPU tells the renter nothing, and is never reported as a reset
    assert result.warning == (None if cure_ok else W.HungGpuReset)


@pytest.mark.asyncio
async def test_a_busy_gpu_with_a_live_process_is_not_the_latch():
    host = HungHost(query=_query({U[0]: f"{U[0]}, 100, 0"}, apps="4242\n"))
    assert (await _reset(host)).warning is None
    assert host.commands == [HUNG_GPU_QUERY_COMMAND]


@pytest.mark.asyncio
async def test_a_query_that_times_out_resets_every_pod_gpu():
    host = HungHost(query="", query_exit=124)
    result = await _reset(host)
    assert _resets(host) == [f"nvidia-smi -r -i {U[0]},{U[1]}"]
    assert result.warning == W.HungGpuReset


@pytest.mark.asyncio
async def test_nvlink_peers_inside_the_pod_are_reset_together():
    host = HungHost(query=_query({U[0]: None}), topo=TOPO_PAIR01)
    await _reset(host)
    assert _resets(host) == [f"nvidia-smi -r -i {U[0]},{U[1]}"]


@pytest.mark.parametrize(
    ("kwargs", "warning"),
    [
        ({"mig": (0,)}, W.HungGpuNotResetMig),
        ({"topo": TOPO_PAIR02}, W.HungGpuNotResetNvlinkShared),  # U[0] <-> U[2], not the pod's
        ({"nvswitch": True}, W.HungGpuNotResetNvlinkShared),  # the pod does not own every host GPU
        ({"apps": [(999, U[0])], "cgroups": {999: "0::/system.slice/other.scope"}}, W.HungGpuNotResetInUse),
    ],
)
@pytest.mark.asyncio
async def test_a_refused_reset_changes_nothing_and_says_why(kwargs, warning):
    host = HungHost(query=_query({U[0]: None}), **kwargs)
    result = await _reset(host)
    assert _resets(host) == [] and not any(c.startswith("CUDA_VISIBLE_DEVICES=") for c in host.commands)
    assert result.warning == warning and result.hung_uuids == [U[0]]


@pytest.mark.asyncio
async def test_the_validators_own_probe_is_not_a_foreign_process():
    own = "0::/user.slice/user-0.slice/session-7.scope"
    host = HungHost(query=_query({U[0]: None}), apps=[(77, U[0])], cgroups={77: own}, own_cgroup=own)
    assert (await _reset(host)).warning == W.HungGpuReset


@pytest.mark.asyncio
async def test_an_nvswitch_pod_that_owns_every_gpu_resets_all_of_them():
    host = HungHost(query=_query({U[0]: None}), nvswitch=True)
    await _reset(host, pod=U)
    assert _resets(host) == [f"nvidia-smi -r -i {','.join(U)}"]


@pytest.mark.parametrize("cure_ok", [True, False])
@pytest.mark.asyncio
async def test_a_refused_reset_falls_back_to_the_context_cycle_on_the_pod_gpus(cure_ok):
    host = HungHost(query=_query({U[1]: None}), reset_exit=1, cure_ok=cure_ok)
    result = await _reset(host)
    assert [c.split()[0] for c in host.commands if c.startswith("CUDA_VISIBLE_DEVICES=")] == [
        f"CUDA_VISIBLE_DEVICES={U[1]}"
    ]
    # as reset_pod_gpus: a latch clear after a refused -r is no reset
    assert result.warning == W.HungGpuNotResetUnsupported and "were not reset" in result.detail
    assert ("only the latch clear ran" in result.detail) == cure_ok


@pytest.mark.asyncio
async def test_a_runner_error_changes_nothing():
    # the SSH session failed (no exit status): nothing is known about the GPUs; the shell timeout (124) still means hung
    host = HungHost(query="", query_exit=-1)
    result = await _reset(host)
    assert host.commands == [HUNG_GPU_QUERY_COMMAND]
    assert result.warning is None and result.hung_uuids == []


class LostGpuHost(HungHost):
    """U[0] fell off the bus: gone from the host listing and from `topo -m`."""

    async def run(self, cmd: str, *, timeout: int = 60, retryable: bool = True, **_):
        if cmd.startswith("nvidia-smi --query-gpu=index"):
            self.commands.append(cmd)
            return make_command_result(True, "\n".join(line for line in self.gpus.splitlines() if U[0] not in line))
        return await super().run(cmd, timeout=timeout, retryable=retryable)


@pytest.mark.parametrize(
    ("kwargs", "pod", "warning"),
    [
        ({"topo": TOPO_PAIR02}, POD, W.HungGpuNotResetNvlinkShared),  # NVLink host: its peers are unknown
        ({"nvswitch": True}, POD, W.HungGpuNotResetNvlinkShared),
        ({}, POD, W.HungGpuReset),  # PCIe only: no peers to worry about
        ({"topo": TOPO_PAIR02}, U, W.HungGpuReset),  # the pod owns every host GPU
    ],
)
@pytest.mark.asyncio
async def test_a_gpu_missing_from_the_listing_is_refused_on_an_nvlink_host_unless_the_pod_owns_it_all(
    kwargs, pod, warning
):
    host = LostGpuHost(query=_query({U[0]: None}), **kwargs)
    result = await _reset(host, pod=pod)
    assert result.warning == warning
    assert bool(_resets(host)) == (warning == W.HungGpuReset)


class FailedListingHost(HungHost):
    """The host listing fails part way: it names only the pod's GPUs and exits non-zero."""

    async def run(self, cmd: str, *, timeout: int = 60, retryable: bool = True, **_):
        if cmd.startswith("nvidia-smi --query-gpu=index"):
            self.commands.append(cmd)
            lines = [line for line in self.gpus.splitlines() if any(uuid in line for uuid in POD)]
            return make_command_result(False, "\n".join(lines), exit_code=1)
        return await super().run(cmd, timeout=timeout, retryable=retryable)


@pytest.mark.asyncio
async def test_a_failed_host_listing_is_refused_so_a_shared_nvswitch_node_never_looks_like_the_pods():
    # DAH-5301 (review), as reset_pod_gpus: the partial list would make the pod own "every" GPU and stop Fabric Manager
    host = FailedListingHost(query=_query({U[0]: None}), nvswitch=True, fabric_manager=True)
    result = await _reset(host)
    assert _resets(host) == []
    assert not any("fabricmanager" in cmd and "stop" in cmd for cmd in host.commands)
    assert result.warning == W.HungGpuNotResetUnsupported


def test_both_nvidia_smi_calls_are_killed_if_they_ignore_the_timeout():
    assert HUNG_GPU_QUERY_COMMAND.count("timeout -k 5 20 nvidia-smi") == 2


FM = "nvidia-fabricmanager"


@pytest.mark.asyncio
async def test_a_fabric_manager_pod_owning_every_gpu_stops_fm_around_the_reset():
    # HGX: no /dev/nvidia-nvswitch*, Fabric Manager running; the pod owns the whole node
    host = HungHost(query=_query({U[0]: None}), device_ids=(), count=-1, fabric_manager=True)
    result = await _reset(host, pod=U)
    assert result.warning == W.HungGpuReset
    assert _resets(host) == [f"nvidia-smi -r -i {','.join(U)}"]
    assert host.active_during_reset[FM] is False and host.active[FM] is True
    # stop, reset and start run as one host unit (as in reset_pod_gpus)
    assert ("stop", FM) in host.fm_log and ("start", FM) in host.fm_log
    assert host.fm_log.index(("stop", FM)) < host.fm_log.index(("start", FM))
    assert any(cmd.startswith("nsenter -t 1 -m -- systemd-run ") for cmd in host.commands)


@pytest.mark.asyncio
async def test_an_earlier_reset_unit_still_running_changes_nothing():
    host = HungHost(query=_query({U[0]: None}), device_ids=(), count=-1, fabric_manager=True)
    host.prior_unit_procs = 2
    result = await _reset(host, pod=U)
    assert result.warning == W.HungGpuNotResetUnsupported and "still running" in result.detail
    assert _resets(host) == [] and not any(c.startswith("CUDA_VISIBLE_DEVICES=") for c in host.commands)
    assert host.fm_log == []


@pytest.mark.asyncio
async def test_a_hung_fabric_reset_whose_unit_stays_is_not_a_reset_and_skips_the_fallback():
    host = HungHost(query=_query({U[0]: None}), device_ids=(), count=-1, fabric_manager=True)
    host.reset_hangs, host.reset_unit_stuck = True, True
    result = await _reset(host, pod=U)
    assert result.warning == W.HungGpuNotResetUnsupported and "still running" in result.detail
    assert not any(c.startswith("CUDA_VISIBLE_DEVICES=") for c in host.commands)


@pytest.mark.asyncio
async def test_fabric_manager_not_coming_back_is_not_a_reset():
    host = HungHost(query=_query({U[0]: None}), device_ids=(), count=-1, fabric_manager=True, fm_start_ok=False)
    result = await _reset(host, pod=U)
    assert result.warning == W.HungGpuNotResetUnsupported and "not active" in result.detail


@pytest.mark.asyncio
async def test_fabric_manager_is_started_again_when_the_reset_raises():
    host = HungHost(query=_query({U[0]: None}), device_ids=(), count=-1, fabric_manager=True)
    host.reset_raises = True
    with pytest.raises(RuntimeError):
        await _reset(host, pod=U)
    assert host.active[FM] is True


@pytest.mark.asyncio
async def test_a_fabric_manager_node_shared_with_other_pods_is_refused():
    host = HungHost(query=_query({U[0]: None}), fabric_manager=True)
    result = await _reset(host)
    assert result.warning == W.HungGpuNotResetNvlinkShared and _resets(host) == []
    assert not any(" stop " in cmd for cmd in host.commands)


class _Hold:
    """Records when the caller's gpu_reset pending mark is held, against the host's commands."""

    def __init__(self, host: HungHost):
        self.host, self.entered, self.exited = host, None, None

    @contextlib.asynccontextmanager
    async def __call__(self):
        self.entered = len(self.host.commands)
        try:
            yield
        finally:
            self.exited = len(self.host.commands)


@pytest.mark.asyncio
async def test_the_pending_mark_is_held_from_before_the_reset_until_it_is_done():
    host = HungHost(query=_query({U[0]: None}), reset_exit=1)  # -r refused: the context cycle runs too
    hold = _Hold(host)
    assert (await reset_hung_pod_gpus(host, POD, resetting=hold)).warning == W.HungGpuNotResetUnsupported
    reset_at = host.commands.index(f"nvidia-smi -r -i {U[0]}")
    assert hold.entered <= reset_at and hold.exited == len(host.commands)
    assert host.commands[-1].startswith("CUDA_VISIBLE_DEVICES=")


@pytest.mark.asyncio
async def test_a_healthy_reboot_never_takes_the_pending_mark():
    host = HungHost()
    hold = _Hold(host)
    await reset_hung_pod_gpus(host, POD, resetting=hold)
    assert hold.entered is None


@pytest.mark.asyncio
async def test_removing_the_gpu_reset_mark_keeps_the_creates_own_mark():
    redis = PendingRedis()
    await redis.add_pending_pod("hk", "ex", "pod1")
    await redis.add_pending_pod("hk", "ex", "pod1", kind=GPU_RESET_PENDING_KIND, gpu_uuids=POD)
    await redis.remove_pending_pod("hk", "ex", "pod1", kind=GPU_RESET_PENDING_KIND)
    assert [item.get("kind") for item in await redis.get_pending_pods("hk", "ex")] == [None]
    await redis.remove_pending_pod("hk", "ex", "pod1")
    assert await redis.get_pending_pods("hk", "ex") == []


@pytest.mark.parametrize("reset_exit", [0, 1])
@pytest.mark.asyncio
async def test_the_result_is_handed_to_the_hold_before_it_is_let_go(reset_exit):
    host = HungHost(query=_query({U[0]: None}), reset_exit=reset_exit, cure_ok=False)
    seen_on_exit = []

    @contextlib.asynccontextmanager
    async def hold():
        reset_result_holder = []
        yield reset_result_holder
        seen_on_exit.append(list(reset_result_holder))

    result = await reset_hung_pod_gpus(host, POD, resetting=hold)
    assert seen_on_exit == [[result]]
    assert result.warning == (W.HungGpuReset if reset_exit == 0 else W.HungGpuNotResetUnsupported)


DCGM, PERSISTENCED, NVLSM = "nvidia-dcgm", "nvidia-persistenced", "nvidia-nvlsm"
EXPORTER = "e" * 64


def _whole_host(**kwargs) -> HungHost:
    # as test_gpu_reset._b300, with the pod's container parked: only the dcgm-exporter container runs
    return HungHost(
        query=_query({U[0]: None}),
        fabric_manager=True,
        holders={DCGM: True, PERSISTENCED: True},
        ps=f"{EXPORTER} false nvcr.io/nvidia/k8s/dcgm-exporter:4.2.3\n",
        **kwargs,
    )


@pytest.mark.asyncio
async def test_a_whole_host_pod_stops_every_host_gpu_holder_around_the_reset():
    host = _whole_host()
    result = await reset_hung_pod_gpus(host, list(U), settle_seconds=0)
    assert result.warning == W.HungGpuReset
    assert host.active_during_reset == {FM: False, NVLSM: False, DCGM: False, PERSISTENCED: False, EXPORTER: False}
    stopped = [entry for entry in host.fm_log if entry[0] in ("stop", "docker stop")]
    assert stopped == [("docker stop", EXPORTER), ("stop", DCGM), ("stop", FM), ("stop", PERSISTENCED)]
    assert all(host.active[u] for u in (FM, DCGM, PERSISTENCED, EXPORTER))


@pytest.mark.asyncio
async def test_a_whole_host_pod_whose_exporter_does_not_come_back_is_not_a_reset():
    host = _whole_host(container_state="false false")
    result = await reset_hung_pod_gpus(host, list(U), settle_seconds=0)
    assert result.warning == W.HungGpuNotResetUnsupported and "is not running again" in result.detail


@pytest.mark.asyncio
async def test_a_refused_reset_on_a_whole_host_pod_cured_by_the_cycle_is_not_a_reset():
    host = _whole_host(reset_exit=1)
    result = await reset_hung_pod_gpus(host, list(U), settle_seconds=0)
    assert result.warning == W.HungGpuNotResetUnsupported and "only the latch clear ran" in result.detail
    assert all(host.active[u] for u in (FM, DCGM, PERSISTENCED, EXPORTER))
    # the cycle runs once every holder is back (CUDA on an NVSwitch node needs Fabric Manager)
    assert host.active_during_cure[FM] and host.active_during_cure[DCGM]


@pytest.mark.asyncio
async def test_a_pod_sharing_the_host_never_stops_the_host_gpu_holders():
    host = HungHost(query=_query({U[0]: None}), holders={DCGM: True, PERSISTENCED: True}, ps=f"{EXPORTER} false dcgm-exporter\n")
    assert (await _reset(host)).warning == W.HungGpuReset
    assert not any(entry[0] in ("stop", "docker stop") for entry in host.fm_log)
    assert gpu_reset.DOCKER_PS_COMMAND not in host.commands
