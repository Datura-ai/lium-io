"""DAH-5301: a reboot resets a hung pod GPU (services/hung_gpu_reset.py) against a fake host.

A healthy reboot runs only the one query; a hung GPU is reset under Reset GPU's refusal rules, and
`nvidia-smi -r` never touches a GPU outside the pod.
"""

import contextlib

import pytest
from payload_models.payloads import ContainerWarningCode as W
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


@pytest.mark.parametrize(("cure_ok", "warning"), [(True, W.HungGpuReset), (False, W.HungGpuNotResetUnsupported)])
@pytest.mark.asyncio
async def test_a_refused_reset_falls_back_to_the_context_cycle_on_the_pod_gpus(cure_ok, warning):
    host = HungHost(query=_query({U[1]: None}), reset_exit=1, cure_ok=cure_ok)
    result = await _reset(host)
    assert [c.split()[0] for c in host.commands if c.startswith("CUDA_VISIBLE_DEVICES=")] == [
        f"CUDA_VISIBLE_DEVICES={U[1]}"
    ]
    assert result.warning == warning


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
    stop = host.commands.index(f"nsenter -t 1 -m -- systemctl stop {FM}")
    start = host.commands.index(f"nsenter -t 1 -m -- systemctl start {FM}")
    assert stop < host.commands.index(_resets(host)[0]) < start


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
    assert (await reset_hung_pod_gpus(host, POD, resetting=hold)).warning == W.HungGpuReset
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
