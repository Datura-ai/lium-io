"""DAH-4658: the renter GPU reset (services/gpu_reset.py) against a fake host.

Every refusal code, and the hard invariants: the reset set is inside BOTH the container's own
DeviceRequests and the backend's GPUs, `nvidia-smi -r` never runs without `-i`, nothing is read from
inside the pod, and a stopped container is always started again.
"""

import json
import os
import re
import shlex
import subprocess
import time
from unittest.mock import AsyncMock, patch

import pytest
from payload_models.payloads import GpuResetErrorCode
from services.gpu_reset import (
    build_fabric_reset_command,
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
        fabric_manager=False,
        nvlsm=False,
        fm_start_ok=True,
        holders=None,
        ps="",
        systemd=True,
        docker_start_ok=True,
        container_state="true false",
        gpus_result=None,
        device_holders=(),
        transient_holders=(),
        reset_output="refused",
        holder_scan=None,
        host_holder_units="",
        start_fails=(),
        needs_fm=(),
        late_units=(),
    ):
        self.commands: list[str] = []
        # (pid, name) holding a target GPU's device file: every read, or the first read only
        self.device_holders, self.transient_holders = list(device_holders), list(transient_holders)
        self.holder_reads = 0
        self.holder_scan = holder_scan  # the scan's whole answer, when it is not the holders above
        # the host services the pre-stop scan finds holding a GPU (its stdout), or its whole answer
        self.host_holder_units = host_holder_units
        self.start_fails = set(start_fails)  # host services that do not start again
        self.needs_fm = set(needs_fm)  # host services whose start fails while Fabric Manager is down
        # host services the unit finds holding a GPU after its stops (opened one after the scan): it stops them and
        # starts them after the others. stopped_file: the unit's list of every service it stopped (never truncated)
        self.late_units, self.stopped_file = list(late_units), []
        self.reset_output = reset_output
        self.gpus_result = gpus_result
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
        self.active = {"nvidia-fabricmanager": fabric_manager, "nvidia-nvlsm": nvlsm, **(holders or {})}
        self.ps = ps
        self.systemd = systemd
        self.docker_start_ok, self.container_state = docker_start_ok, container_state
        self.fm_start_ok = fm_start_ok
        self.active_during_reset: dict[str, bool] | None = None
        self.reset_raises = False
        self.reset_hangs = False
        self.reset_unit_stuck = False
        self.reset_unit_state, self.reset_unit_procs = "inactive", 0
        # processes of an earlier reset's unit, still there when this reset begins
        self.prior_unit_procs = 0
        self.hung_units: list[str] = []
        self.active_during_cure: dict[str, bool] | None = None
        self.fm_log: list[tuple[str, str]] = []

    async def run(self, cmd: str, *, timeout: int = 60, retryable: bool = True, **_):
        self.commands.append(cmd)
        if cmd == gpu_reset.DOCKER_PS_COMMAND:
            return make_command_result(True, self.ps)
        if cmd.startswith(gpu_reset.CONTAINER_STATE_COMMAND):
            return make_command_result(True, f"{self.container_state}\n")
        if cmd.startswith("/usr/bin/docker inspect"):
            return make_command_result(self.inspect_exit == 0, self.inspect, exit_code=self.inspect_exit)
        if cmd.startswith("timeout 90 /usr/bin/docker wait"):
            return make_command_result(True, "137")
        if cmd.startswith("/usr/bin/docker start"):
            self.fm_log.append(("docker start", cmd.split()[-1]))
            ok = self.docker_start_ok
            return make_command_result(ok, stderr="" if ok else "no such container", exit_code=0 if ok else 1)
        if cmd.startswith("/usr/bin/docker stop"):
            self.stopped = self.stop_exit == 0
            return make_command_result(self.stop_exit == 0, stderr="boom", exit_code=self.stop_exit)
        if cmd.startswith("nvidia-smi --query-gpu="):
            return make_command_result(True, self.gpus) if self.gpus_result is None else self.gpus_result
        if cmd.startswith("nvidia-smi --query-compute-apps="):
            apps = self.apps_after_stop if self.stopped else self.apps
            return make_command_result(True, "\n".join(f"{pid}, {uuid}" for pid, uuid in apps))
        if cmd.startswith("for p in /dev/nvidia-nvswitch"):
            return make_command_result(True, "/dev/nvidia-nvswitch0\n" if self.nvswitch else "")
        if cmd == "nvidia-smi topo -m":
            return make_command_result(True, self.topo)
        if cmd.endswith("< /proc/self/cgroup"):
            return make_command_result(True, f"{self.own_cgroup} ")
        if cmd.startswith("for c in find stat"):
            self.holder_reads += 1
            if self.holder_scan is not None:
                return self.holder_scan
            holders = self.device_holders + (self.transient_holders if self.holder_reads == 1 else [])
            read = "".join(f"gpu {bus}\n" for bus in re.search(r"for b in ([^;]*);", cmd).group(1).split())
            return make_command_result(True, read + "".join(f"{pid} {name}\n" for pid, name in holders))
        if cmd.startswith("for p in "):
            pids = [int(p) for p in cmd.split(";")[0].split()[3:]]
            out = "\n".join(f"{pid} {self.cgroups[pid]} " for pid in pids if pid in self.cgroups)
            return make_command_result(True, out)
        if cmd.startswith("nsenter -t 1 -m -- systemd-run "):
            # the host unit: stop, reset, start again, whatever happens to the SSH session
            script = shlex.split(cmd)[-1]
            found = re.search(r"\|\| for u in ([^;$]*); do echo \"\$u\" >>\S+ && systemctl stop", script)
            units = found.group(1).split() if found else []
            containers = re.search(r"for u in ([^;]*); do timeout -k 2 \d+ docker stop", script)
            containers = containers.group(1).split() if containers else []
            for cid in containers:
                self.active[cid] = False
                self.fm_log.append(("docker stop", cid))
            for unit in units + self.late_units:
                self.active[unit] = False
                self.fm_log.append(("stop", unit))
            self.stopped_file += units + [gpu_reset.LATE_MARK + u for u in self.late_units]
            self.commands.append(re.search(r"\{ (nvidia-smi -r\b[^;]*);", script).group(1))
            self.active_during_reset = dict(self.active)
            if self.reset_hangs:
                # the SSH timeout fires while the host unit still holds the fabric down
                self.hung_units = units + self.late_units
                return make_command_result(False, stderr="timed out", exit_code=-1)
            for unit in [*reversed(units), *self.late_units]:
                self.active[unit] = (unit != "nvidia-fabricmanager" or self.fm_start_ok) and unit not in self.start_fails
                self.fm_log.append(("start", unit))
            for cid in reversed(containers):
                self.active[cid] = True
                self.fm_log.append(("docker start", cid))
            if self.reset_raises:
                raise RuntimeError("ssh dropped")
            return make_command_result(self.reset_exit == 0, "refused", exit_code=self.reset_exit)
        if cmd == gpu_reset.HOST_GPU_HOLDER_UNITS_COMMAND:
            found = self.host_holder_units
            return make_command_result(True, found) if isinstance(found, str) else found
        if cmd == gpu_reset.STOPPED_UNITS_READ_COMMAND:
            return make_command_result(True, "".join(f"{u}\n" for u in self.stopped_file))
        if cmd == gpu_reset.STOPPED_UNITS_CLEAR_COMMAND:
            self.stopped_file = []
            return make_command_result(True)
        if cmd == gpu_reset.HOST_SYSTEMD_PROBE:
            return make_command_result(True, "" if self.systemd else "no-systemd\n")
        if cmd == gpu_reset.FABRIC_RESET_UNIT_PROCS:
            if not self.hung_units and not self.reset_unit_stuck:
                if isinstance(self.prior_unit_procs, int):
                    return make_command_result(True, f"{self.prior_unit_procs}\n")
                if self.prior_unit_procs == "raise":
                    raise RuntimeError("ssh dropped")
                return make_command_result(False, exit_code=int(self.prior_unit_procs))
            if self.reset_unit_procs is None:
                return make_command_result(False, exit_code=1)
            return make_command_result(True, f"{self.reset_unit_procs}\n")
        if cmd.startswith("nsenter -t 1 -m -- systemctl "):
            verb, unit = shlex.split(cmd)[-2:]
            self.fm_log.append((verb, unit))
            if unit == gpu_reset.FABRIC_RESET_UNIT:
                if self.hung_units and not self.reset_unit_stuck and self.reset_unit_procs == 0:
                    # the unit finished late: its own loop started the fabric again before it was gone
                    for u in reversed(self.hung_units):
                        self.active[u] = u != "nvidia-fabricmanager" or self.fm_start_ok
                        self.fm_log.append(("unit start", u))
                    self.hung_units = []
                if verb == "is-active":
                    state = "deactivating" if self.reset_unit_stuck else self.reset_unit_state
                    return make_command_result(False, state, exit_code=0 if self.reset_unit_stuck else 3)
                return make_command_result(not self.reset_unit_stuck, exit_code=0 if not self.reset_unit_stuck else 1)
            if verb == "is-active":
                up = self.active.get(unit, False)
                return make_command_result(up, "active" if up else "inactive", exit_code=0 if up else 3)
            ok = verb == "stop" or (
                (unit != "nvidia-fabricmanager" or self.fm_start_ok)
                and unit not in self.start_fails
                and not (unit in self.needs_fm and not self.active.get("nvidia-fabricmanager"))
            )
            self.active[unit] = verb == "start" and ok
            return make_command_result(ok, exit_code=0 if ok else 1)
        if cmd.startswith("nvidia-smi -r"):
            self.active_during_reset = dict(self.active)
            if self.reset_raises:
                raise RuntimeError("ssh dropped")
            return make_command_result(self.reset_exit == 0, self.reset_output, exit_code=self.reset_exit)
        if cmd.startswith("CUDA_VISIBLE_DEVICES="):
            self.active_during_cure = self.active_during_cure or dict(self.active)
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
        if cmd == "nvidia-smi -r":
            # the whole fabric: only for a pod holding every host GPU while Fabric Manager is stopped
            assert host.active_during_reset.get(FM) is False, "never nvidia-smi -r without -i outside a fabric reset"
            assert container == set(backend) == set(U)
        elif cmd.startswith("nvidia-smi -r"):
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
        (
            FakeHost(mounts=[{"Destination": "/lium-cipher"}]),
            {"encrypted_volume_mount": "/lium-cipher"},
            GpuResetErrorCode.ResetNotSupported,
        ),
        (FakeHost(gpus_result=make_command_result(False, exit_code=255)), {}, GpuResetErrorCode.GpusLostOnHost),
        (
            FakeHost(gpus_result=make_command_result(False, exit_code=-1, error_type="timeout")),
            {},
            GpuResetErrorCode.GpusLostOnHost,
        ),
        (FakeHost(gpus_result=make_command_result(True, "")), {}, GpuResetErrorCode.GpusLostOnHost),
        *[
            (
                FakeHost(gpus_result=make_command_result(False, exit_code=-1, error_type=error_type)),
                {},
                GpuResetErrorCode.ExecutorUnreachable,
            )
            for error_type in ("no_exit_status", "ConnectionLost")
        ],
        (FakeHost(inspect_exit=1), {}, GpuResetErrorCode.ContainerNotFound),
        (FakeHost(device_ids=(U[0],)), {}, GpuResetErrorCode.GpuNotInContainer),  # backend's U[1] not given
        (FakeHost(), {"indices": [2]}, GpuResetErrorCode.GpuNotInContainer),
    ],
)
@pytest.mark.asyncio
async def test_every_precheck_refusal_changes_nothing(host, kwargs, code):
    report, start = await _reset(host, **kwargs)
    assert not report.ok and report.error_code == code and report.gpus == [] and not report.pod_stopped
    # seen running (False), or refused before its state was read (None: TEE, encrypted, no container), never True
    unread = "in_tee" in kwargs or "encrypted_volume_mount" in kwargs or code == GpuResetErrorCode.ContainerNotFound
    assert report.stopped_during_reset is (None if unread else False)
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


FM, NVLSM = "nvidia-fabricmanager", "nvidia-nvlsm"
CHECK_BOTH = [("is-active", FM), ("is-active", NVLSM)]


def _fm_commands(host: FakeHost) -> list[tuple[str, str]]:
    return [entry for entry in host.fm_log if entry[1] in (FM, NVLSM)]


@pytest.mark.asyncio
async def test_a_non_nvswitch_node_never_touches_fabric_manager():
    host = FakeHost()
    report, _ = await _reset(host)
    assert report.ok and _fm_commands(host) == CHECK_BOTH


@pytest.mark.asyncio
async def test_a_fabric_manager_node_the_pod_owns_whole_stops_fm_around_the_reset():
    # HGX B300: no /dev/nvidia-nvswitch*, Fabric Manager running
    host = FakeHost(device_ids=(), count=-1, fabric_manager=True)
    report, start = await _reset(host, backend=U, indices=[2])
    assert report.ok and {g.uuid for g in report.gpus} == set(U)
    assert host.active_during_reset == {FM: False, NVLSM: False} and host.active[FM]
    assert _fm_commands(host) == [*CHECK_BOTH, ("stop", FM), ("start", FM), ("is-active", FM)]
    assert sum("systemd-run" in cmd for cmd in host.commands) == 1
    start.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_fabric_manager_node_the_pod_owns_whole_resets_gpus_and_nvswitches_together():
    # HGX B300: with Fabric Manager stopped, `-r -i <every GPU>` was refused "in use" with no holder left
    host = FakeHost(device_ids=(), count=-1, fabric_manager=True)
    report, _ = await _reset(host, backend=U, indices=[2])
    assert report.ok and [c for c in host.commands if c.startswith("nvidia-smi -r")] == ["nvidia-smi -r"]


@pytest.mark.asyncio
async def test_an_nvswitch_node_without_fabric_manager_keeps_the_per_gpu_reset():
    host = FakeHost(device_ids=(), count=-1, nvswitch=True)
    await _reset(host, backend=U, indices=[0])
    resets = [c for c in host.commands if c.startswith("nvidia-smi -r")]
    assert len(resets) == 1 and set(resets[0].split(" -i ")[1].split(",")) == set(U)


def test_a_whole_fabric_reset_refuses_a_partial_gpu_set():
    with pytest.raises(ValueError):
        build_reset_command(U[:2], container_uuids=U, pod_gpu_uuids=U, whole_fabric=True)
    with pytest.raises(ValueError):
        build_reset_command(U[:2], container_uuids=U[:2], pod_gpu_uuids=U, whole_fabric=True)
    assert build_reset_command(U, container_uuids=U, pod_gpu_uuids=U, whole_fabric=True) == "nvidia-smi -r"


@pytest.mark.asyncio
async def test_nvlsm_is_stopped_after_fm_and_started_before_it():
    # HGX Blackwell: the NVLink subnet manager runs alongside Fabric Manager
    host = FakeHost(device_ids=(), count=-1, fabric_manager=True, nvlsm=True)
    report, _ = await _reset(host, backend=U)
    assert report.ok and host.active_during_reset == {FM: False, NVLSM: False}
    assert host.active == {FM: True, NVLSM: True}
    assert _fm_commands(host) == [
        *CHECK_BOTH,
        ("stop", FM),
        ("stop", NVLSM),
        ("start", NVLSM),
        ("start", FM),
        ("is-active", NVLSM),
        ("is-active", FM),
    ]


@pytest.mark.asyncio
async def test_a_fabric_manager_node_shared_with_other_pods_is_refused():
    host = FakeHost(fabric_manager=True)
    report, start = await _reset(host)
    assert not report.ok and report.error_code == GpuResetErrorCode.NvlinkPeersOutsidePod
    assert "provider must reset" in report.detail and host.active[FM]
    assert _fm_commands(host) == CHECK_BOTH
    _no_change(host, start)


@pytest.mark.asyncio
async def test_fabric_manager_is_started_again_when_the_reset_fails():
    host = FakeHost(device_ids=(), count=-1, fabric_manager=True, nvlsm=True)
    host.reset_raises = True
    report, start = await _reset(host, backend=U)
    assert not report.ok and host.active == {FM: True, NVLSM: True}
    start.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_hung_reset_waits_for_its_unit_to_restore_the_fabric_before_the_fallback_and_the_container():
    host = FakeHost(device_ids=(), count=-1, fabric_manager=True, nvlsm=True)
    host.reset_hangs = True
    active_at_start = {}
    start = AsyncMock(side_effect=lambda: active_at_start.update(host.active))
    report, _ = await _reset(host, backend=U, start=start)
    up = {FM: True, NVLSM: True}
    assert host.active_during_cure == up and active_at_start == up
    # never stopped: the unit started the fabric itself, so the validator only checks it
    i = host.fm_log.index(("is-active", gpu_reset.FABRIC_RESET_UNIT))
    assert host.fm_log[i:] == [
        ("is-active", gpu_reset.FABRIC_RESET_UNIT),
        ("unit start", NVLSM),
        ("unit start", FM),
        ("is-active", NVLSM),
        ("is-active", FM),
    ]
    assert report.container_restarted


@pytest.mark.asyncio
async def test_a_hung_reset_whose_fabric_stays_down_skips_the_fallback():
    host = FakeHost(device_ids=(), count=-1, fabric_manager=True, fm_start_ok=False)
    host.reset_hangs = True
    report, start = await _reset(host, backend=U)
    assert not report.ok and report.error_code == GpuResetErrorCode.ResetNotSupported
    assert "nvidia-fabricmanager is not active" in report.detail
    assert host.active_during_cure is None
    start.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_hung_reset_whose_unit_will_not_stop_skips_the_fallback_and_the_container_start():
    host = FakeHost(device_ids=(), count=-1, fabric_manager=True)
    host.reset_hangs = host.reset_unit_stuck = True
    report, start = await _reset(host, backend=U)
    assert not report.ok and report.error_code == GpuResetErrorCode.ResetNotSupported
    assert "still running on the host" in report.detail and not report.container_restarted
    # the unit is never stopped (its own loop starts the fabric when the reset ends) and no validator start
    # runs while it is active
    assert ("stop", gpu_reset.FABRIC_RESET_UNIT) not in host.fm_log and ("start", FM) not in host.fm_log
    assert host.active[FM] is False and host.active_during_cure is None
    start.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_retry_while_an_earlier_reset_unit_still_runs_changes_nothing():
    # the earlier reset outlived its SSH timeout: its container is stopped and Fabric Manager is down
    host = FakeHost(main_pid=0, device_ids=(), count=-1)
    host.prior_unit_procs = 2
    report, start = await _reset(host, backend=U)
    assert not report.ok and report.error_code == GpuResetErrorCode.OperationInProgress
    assert "still running on the host" in report.detail and not report.container_restarted
    assert report.stopped_during_reset is None  # DAH-6741 review: the earlier reset may have left it stopped
    assert host.commands == [gpu_reset.FABRIC_RESET_UNIT_PROCS]
    _no_change(host, start)


@pytest.mark.asyncio
@pytest.mark.parametrize("prior", ["1", "raise"])
async def test_a_retry_whose_earlier_reset_unit_count_is_unreadable_changes_nothing(prior):
    host = FakeHost(main_pid=0, device_ids=(), count=-1)
    host.prior_unit_procs = prior
    report, start = await _reset(host, backend=U)
    assert not report.ok and report.error_code == GpuResetErrorCode.OperationInProgress
    assert host.commands == [gpu_reset.FABRIC_RESET_UNIT_PROCS]
    _no_change(host, start)


@pytest.mark.asyncio
async def test_a_host_without_nsenter_never_ran_the_reset_unit():
    host = FakeHost(device_ids=(), count=-1)
    host.prior_unit_procs = "127"
    report, start = await _reset(host, backend=U)
    assert report.ok and report.container_restarted


@pytest.mark.asyncio
async def test_a_failed_reset_unit_with_surviving_processes_skips_the_fallback_and_the_container_start():
    # systemd marks the unit failed when the reset survives its final SIGKILL (D state on a wedged GPU)
    host = FakeHost(device_ids=(), count=-1, fabric_manager=True)
    host.reset_hangs = True
    host.reset_unit_state, host.reset_unit_procs = "failed", 1
    report, start = await _reset(host, backend=U)
    assert not report.ok and "still running on the host" in report.detail and not report.container_restarted
    assert ("start", FM) not in host.fm_log and host.active_during_cure is None
    start.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failed_reset_unit_with_an_empty_cgroup_counts_as_stopped():
    host = FakeHost(device_ids=(), count=-1, fabric_manager=True, nvlsm=True)
    host.reset_hangs = True
    host.reset_unit_state = "failed"
    report, start = await _reset(host, backend=U)
    assert report.container_restarted and host.active == {FM: True, NVLSM: True}
    start.assert_awaited_once()


def test_unit_process_count_reads_the_cgroup(tmp_path):
    """0 only for an unloaded unit or an existing empty cgroup.procs; a missing file is unknown (exit 1)."""
    root = tmp_path / "cg"
    script = shlex.split(gpu_reset.FABRIC_RESET_UNIT_PROCS)[-1].replace("/sys/fs/cgroup", str(root))
    (tmp_path / "systemctl").write_text('#!/bin/sh\necho "$CG"\n')
    (tmp_path / "systemctl").chmod(0o755)

    def count(cg):
        env = {"PATH": f"{tmp_path}:/usr/bin:/bin", "CG": cg}
        out = subprocess.run(["/bin/sh", "-c", script], env=env, capture_output=True, text=True)
        return out.returncode, out.stdout.strip()

    assert count("") == (0, "0")
    no_systemctl = tmp_path / "empty"
    no_systemctl.mkdir()
    out = subprocess.run(["/bin/sh", "-c", script], env={"PATH": str(no_systemctl)}, capture_output=True, text=True)
    assert (out.returncode, out.stdout.strip()) == (0, "0")
    assert count("/system.slice/x.service")[0] == 1
    (root / "systemd/system.slice/x.service").mkdir(parents=True)
    (root / "systemd/system.slice/x.service/cgroup.procs").write_text("")
    assert count("/system.slice/x.service") == (0, "0")
    (root / "system.slice/x.service").mkdir(parents=True)
    (root / "system.slice/x.service/cgroup.procs").write_text("4242\n")
    assert count("/system.slice/x.service") == (0, "1")


@pytest.mark.asyncio
async def test_an_unreadable_reset_unit_cgroup_skips_the_fallback_and_the_container_start():
    host = FakeHost(device_ids=(), count=-1, fabric_manager=True)
    host.reset_hangs = True
    host.reset_unit_state, host.reset_unit_procs = "failed", None
    report, start = await _reset(host, backend=U)
    assert not report.ok and report.error_code == GpuResetErrorCode.ResetNotSupported
    assert "still running on the host" in report.detail and not report.container_restarted
    start.assert_not_awaited()


# a fake stat: each fake fd links to ".../dev-<major>-<minor>" (tests cannot make device nodes); a link to
# "denied" fails the way an fd the scan may not read does. Prints "<major> <minor> <path>" for "%t %T %n" and
# "<major> <path>" for "%t %n", in hex like stat
FAKE_STAT = """#!/bin/sh
for f; do
  case "$f" in -L|-c) continue;; %*) fmt=$f; continue;; esac
  t=$(readlink -m "$f")
  case "$t" in
    */denied) echo "stat: cannot statx '$f': Permission denied" >&2; ok=1;;
    *dev-*) t=$(echo "$t" | sed 's/.*dev-//'); maj=$(printf '%x' "$(echo "$t" | cut -d- -f1)")
      if [ "$fmt" = "%t %n" ]; then echo "$maj $f"; else printf '%s %x %s\\n' "$maj" "$(echo "$t" | cut -d- -f2)" "$f"; fi;;
  esac
done
exit ${ok:-0}
"""
STALLING_STAT = (
    '#!/bin/sh\nfor a; do case "$(readlink "$a" 2>/dev/null)" in /mnt/*) case " $* " in *" -L "*) exec sleep 20;; esac;; '
    'esac; done\n' + FAKE_STAT.removeprefix("#!/bin/sh\n")
)
PROC_DEVICES = "Character devices:\n  1 mem\n195 nvidia-frontend\n226 drm\n510 nvidia-uvm\n\nBlock devices:\n  8 sd\n"


def _fake_fd(proc, pid: int, device: str) -> None:
    """pid holds "<major>:<minor>" open ("denied": an fd the scan cannot read)."""
    (proc / str(pid) / "fd").mkdir(parents=True, exist_ok=True)
    (proc / str(pid) / "fd" / "5").symlink_to("denied" if device == "denied" else "/dev/x/dev-" + device.replace(":", "-"))


def _unit_script(cmd: str, tmp_path) -> str:
    """The host unit's shell script, with its stopped-services file under tmp_path (stopped), not the host's /run."""
    script = shlex.split(cmd)[-1].replace(gpu_reset.STOPPED_UNITS_FILE, str(tmp_path / "stopped"))
    return script.replace(gpu_reset.LATE_SCAN_FILE, str(tmp_path / "late"))  # the scan unit's fresh files


def _run_fabric_script(
    tmp_path, *, stop_fails="", start_fails="", reset_exit=0, units=(FM, NVLSM), containers=(), procs=None, needs="",
    scan_busy=False, scan_launch_hangs=False,
):
    """Runs the host unit's script under /bin/sh with logging stand-ins for systemctl, docker and nvidia-smi. The
    host's /dev holds nvidia0 (195:0), gpu0 (another name for it, 195:0) and dri-card0 (226:0); procs is the host's
    /proc: {pid: (comm, cgroup)} holding /dev/nvidia0 open, or {pid: (comm, cgroup, node)} holding that /dev node
    (or that absolute path) open. Its stat answers for the /dev nodes but never returns when told to follow an open
    file (-L): a descriptor on a stalled filesystem. stop_fails and start_fails name the one unit whose systemctl stop
    or start exits 1; a start of any unit fails while the unit needs has not been started yet. The unit's list of
    stopped services is tmp_path/stopped."""
    log = tmp_path / "log"
    proc, dev = tmp_path / "proc", tmp_path / "dev"
    proc.mkdir()
    dev.mkdir()
    (proc / "devices").write_text(PROC_DEVICES)
    for node, numbers in (("nvidia0", "195-0"), ("gpu0", "195-0"), ("dri-card0", "226-0")):
        (dev / node).symlink_to(f"x/dev-{numbers}")  # a character device FAKE_STAT reads the numbers of
    for pid, (comm, cgroup, *node) in (procs or {}).items():
        (proc / str(pid) / "fd").mkdir(parents=True)
        target = (node or ["nvidia0"])[0]
        (proc / str(pid) / "fd" / "5").symlink_to(target if target.startswith("/") else f"{dev}/{target}")
        (proc / str(pid) / "comm").write_text(f"{comm}\n")
        (proc / str(pid) / "cgroup").write_text(f"{cgroup}\n")
    for name, body in (
        (
            "systemctl",
            f'echo "$1 $2" >> {log}; [ "$1" = stop ] && [ "$2" = "$STOP_FAILS" ] && exit 1; '
            '[ "$1" = start ] && [ "$2" = "$START_FAILS" ] && exit 1; '
            f'[ "$1" = start ] && [ -n "$NEEDS" ] && [ "$2" != "$NEEDS" ] && ! grep -qx "start $NEEDS" {log} && exit 1; '
            "exit 0",
        ),
        ("docker", f'echo "docker $*" >> {log}; [ "$1" = stop ] && [ "$4" = "$STOP_FAILS" ] && exit 1; exit 0'),
        # an open file on a stalled filesystem (/mnt): its stat -L never returns
        ("stat", STALLING_STAT),
        ("nvidia-smi", f'echo "reset $*" >> {log}; echo "In use by another client"; exit {reset_exit}'),
        # the holder scan's own host unit: started in the background, or refused while an earlier one is still there
        (
            "systemd-run",
            f'echo "systemd-run $*" | cut -d" " -f1-5 >> {tmp_path / "units"}; [ -n "$SCAN_BUSY" ] && exit 1; '
            '[ -n "$SCAN_LAUNCH_HANGS" ] && exec sleep 20; '
            'while [ $# -gt 0 ] && [ "$1" != /bin/sh ]; do shift; done; "$@" </dev/null >/dev/null 2>&1 & exit 0',
        ),
    ):
        (tmp_path / name).write_text(f"#!/bin/sh\n{body}\n")
        (tmp_path / name).chmod(0o755)
    cmd = build_fabric_reset_command(list(units), f"nvidia-smi -r -i {U[0]}", list(containers))
    assert cmd.startswith("nsenter -t 1 -m -- systemd-run ")
    script = _unit_script(cmd, tmp_path).replace("/proc/", f"{proc}/")
    script = script.replace("-lname '/dev/nvidia*'", f"-lname '{dev}/nvidia*'")
    # what systemd does to the command line before the shell runs: unknown ${...} becomes "", $$ becomes $
    script = re.sub(r"\$\{[^}]*\}", "", script).replace("$$", "$")
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "STOP_FAILS": stop_fails, "START_FAILS": start_fails, "NEEDS": needs,
           "SCAN_BUSY": "1" if scan_busy else "", "SCAN_LAUNCH_HANGS": "1" if scan_launch_hangs else ""}
    done = subprocess.run(["/bin/sh", "-c", script], env=env, capture_output=True, text=True, timeout=60)
    return done, log.read_text().splitlines()


def test_the_host_unit_starts_the_fabric_again_after_a_failed_reset(tmp_path):
    done, log = _run_fabric_script(tmp_path, reset_exit=3)
    assert done.returncode == 3
    assert log == [f"stop {FM}", f"stop {NVLSM}", f"reset -r -i {U[0]}", f"start {NVLSM}", f"start {FM}"]


def test_the_host_unit_stops_every_gpu_holder_and_starts_them_again_in_reverse(tmp_path):
    # B300: DCGM and nvidia-persistenced hold every GPU, a dcgm-exporter container too, and one more host
    # service the validator found holding a GPU before the stop (first in the list)
    procs = {
        4242: ("nv-hostengine", "0::/system.slice/some-gpu-agent.service"),
        4243: ("python3", "0::/system.slice/docker-abc.scope"),
    }
    units = ("some-gpu-agent.service", "nvidia-dcgm", FM, "nvidia-persistenced")
    done, log = _run_fabric_script(tmp_path, reset_exit=3, units=units, containers=("e1",), procs=procs)
    assert done.returncode == 3
    assert log == [
        "docker stop -t 10 e1",
        "stop some-gpu-agent.service",
        "stop nvidia-dcgm",
        f"stop {FM}",
        "stop nvidia-persistenced",
        f"reset -r -i {U[0]}",
        "start nvidia-persistenced",
        f"start {FM}",
        "start nvidia-dcgm",
        "start some-gpu-agent.service",
        "docker start e1",
    ]
    # a refused reset names the processes still holding a GPU
    assert "GPU holders left: nv-hostengine python3" in done.stdout


def test_a_late_look_stalled_on_an_open_file_skips_the_reset_and_still_starts_everything(tmp_path, monkeypatch):
    # DAH-6741 review: the second look follows open files (device numbers) in the background; a reader of a hung
    # mount keeps it from ending, so the holders are unknown: no reset, and the unit still reaches its starts
    monkeypatch.setattr(gpu_reset, "LATE_SCAN_SECONDS", 2)
    procs = {4242: ("agent", f"0::/system.slice/{AGENT}"), 4250: ("reader", "0::/system.slice/r.service", "/mnt/f")}
    began = time.monotonic()
    done, log = _run_fabric_script(tmp_path, procs=procs)
    assert time.monotonic() - began < 15  # the stalled scan (20 s here) is never waited on
    assert done.returncode == gpu_reset._FABRIC_STOP_FAILED_EXIT
    assert "could not read which host services hold a GPU after the stops" in done.stderr
    assert log == [f"stop {FM}", f"stop {NVLSM}", f"start {NVLSM}", f"start {FM}"]
    # the scan ran as its own host unit (outside the reset unit's cgroup, so it cannot keep the reset unit from
    # ending), one per host by its fixed name
    assert (tmp_path / "units").read_text().split("\n")[0] == (
        f"systemd-run --quiet --collect --no-block --unit={gpu_reset.LATE_SCAN_UNIT}"
    )


def test_a_scan_launch_that_does_not_return_skips_the_reset_and_starts_everything(tmp_path, monkeypatch):
    # DAH-6741 review: a busy systemd manager holding the launch call cannot keep the stopped services down
    monkeypatch.setattr(gpu_reset, "LATE_SCAN_LAUNCH_SECONDS", 1)
    began = time.monotonic()
    done, log = _run_fabric_script(tmp_path, procs={4242: ("agent", f"0::/system.slice/{AGENT}")}, scan_launch_hangs=True)
    assert time.monotonic() - began < 10  # the launch (20 s here) is cut at its deadline, the scan is not waited for
    assert done.returncode == gpu_reset._FABRIC_STOP_FAILED_EXIT
    assert log == [f"stop {FM}", f"stop {NVLSM}", f"start {NVLSM}", f"start {FM}"]


def test_a_holder_scan_unit_still_there_skips_the_reset_at_once_and_starts_everything(tmp_path):
    # DAH-6741 review: a stuck scan from an earlier reset holds the scan unit's name; no second scanner starts
    began = time.monotonic()
    done, log = _run_fabric_script(tmp_path, procs={4242: ("agent", f"0::/system.slice/{AGENT}")}, scan_busy=True)
    assert time.monotonic() - began < gpu_reset.LATE_SCAN_SECONDS  # no wait for a scan that never started
    assert done.returncode == gpu_reset._FABRIC_STOP_FAILED_EXIT
    assert log == [f"stop {FM}", f"stop {NVLSM}", f"start {NVLSM}", f"start {FM}"]


def test_the_late_look_finds_a_gpu_opened_through_a_bind_mounted_node(tmp_path):
    # DAH-6741 review: a service holding the GPU through a node outside the host's /dev (a bind mount such as /gpu0)
    # is found by the device number, stopped, and started again after the others
    (tmp_path / "ctr").mkdir()
    (tmp_path / "ctr" / "gpu0").symlink_to("x/dev-195-0")
    procs = {4242: ("monitor", "0::/system.slice/late-monitor.service", str(tmp_path / "ctr" / "gpu0"))}
    done, log = _run_fabric_script(tmp_path, units=(FM,), procs=procs)
    assert done.returncode == 0
    assert log == [f"stop {FM}", "stop late-monitor.service", f"reset -r -i {U[0]}", f"start {FM}", "start late-monitor.service"]


def test_the_host_units_list_of_stopped_services_is_never_truncated(tmp_path):
    # DAH-6741 review: a service an earlier unit stopped stays listed until a restore saw it up again
    (tmp_path / "stopped").write_text("old-monitor.service\n")
    done, _ = _run_fabric_script(tmp_path, units=(FM,))
    assert done.returncode == 0
    assert (tmp_path / "stopped").read_text().split() == ["old-monitor.service", FM]


def test_the_host_unit_stops_a_service_that_opened_a_gpu_after_the_scan_and_starts_it_last(tmp_path):
    # DAH-6741 review: a monitor started after the validator's scan, holding the GPU under another node name (same
    # device number); the given services and the unit are not stopped twice, a process outside system.slice
    # services (a container) is not a unit to stop, and another driver's device is no GPU holder
    procs = {
        4242: ("monitor", "0::/system.slice/late-monitor.service", "gpu0"),
        4246: ("compositor", "0::/system.slice/compositor.service", "dri-card0"),
        4243: ("dcgm", "0::/system.slice/nvidia-dcgm.service"),
        4244: ("sh", f"0::/system.slice/{gpu_reset.FABRIC_RESET_UNIT}.service"),
        4245: ("python3", "0::/system.slice/docker-abc.scope"),
    }
    done, log = _run_fabric_script(tmp_path, units=("nvidia-dcgm", FM), procs=procs)
    assert done.returncode == 0
    assert log == [
        "stop nvidia-dcgm",
        f"stop {FM}",
        "stop late-monitor.service",
        f"reset -r -i {U[0]}",
        f"start {FM}",
        "start nvidia-dcgm",
        "start late-monitor.service",
    ]
    # every stopped one listed for the validator's restore check, before its stop; the late one marked
    assert (tmp_path / "stopped").read_text().split() == ["nvidia-dcgm", FM, "+late-monitor.service"]


def test_the_host_unit_skips_the_reset_when_a_late_service_does_not_stop_and_starts_everything_again(tmp_path):
    procs = {4242: ("monitor", "0::/system.slice/late-monitor.service")}
    done, log = _run_fabric_script(tmp_path, stop_fails="late-monitor.service", units=(FM,), procs=procs)
    assert done.returncode == gpu_reset._FABRIC_STOP_FAILED_EXIT
    assert log == [f"stop {FM}", "stop late-monitor.service", f"start {FM}", "start late-monitor.service"]


def _run_holder_units_scan(tmp_path, procs: dict[int, tuple[str, str]]) -> subprocess.CompletedProcess:
    """GPU_HOLDER_UNITS_SCRIPT over a fake /proc: procs maps pid -> (cgroup, "<major>:<minor>" held open)."""
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "devices").write_text(PROC_DEVICES)
    for pid, (cgroup, device) in procs.items():
        _fake_fd(proc, pid, device)
        if cgroup is not None:  # None: a live process whose cgroup cannot be read
            (proc / str(pid) / "cgroup").write_text(f"{cgroup}\n")
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin/stat").write_text(FAKE_STAT)
    (tmp_path / "bin/stat").chmod(0o755)
    script = gpu_reset.GPU_HOLDER_UNITS_SCRIPT.replace("/proc/", f"{proc}/")
    env = {**os.environ, "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}"}
    return subprocess.run(["/bin/sh", "-c", script], env=env, capture_output=True, text=True, timeout=60)


def test_the_host_holder_scan_finds_services_by_device_number_not_by_path(tmp_path):
    # DAH-6741 review: any node with an NVIDIA driver's numbers opens the device, whatever it is called; another
    # driver's device (226, drm) is not a GPU holder, and the reset unit itself never counts
    done = _run_holder_units_scan(tmp_path, {
        4242: (f"0::/system.slice/{AGENT}", "195:7"),
        4243: ("0::/system.slice/uvm-user.service", "510:0"),
        4244: ("0::/system.slice/compositor.service", "226:0"),
        4245: (f"0::/system.slice/{gpu_reset.FABRIC_RESET_UNIT}.service", "195:0"),
    })
    assert done.returncode == 0 and done.stdout.split() == [AGENT, "uvm-user.service"]


def test_the_host_holder_scan_finds_the_service_enclosing_a_nested_cgroup(tmp_path):
    # DAH-6741 review: a service under a sub-slice, and a worker in a service's child cgroup, are that service
    done = _run_holder_units_scan(tmp_path, {
        4242: ("0::/system.slice/gpu.slice/monitor.service", "195:0"),
        4243: ("0::/system.slice/agent.service/worker", "195:0"),
        4244: ("0::/system.slice/docker-abc.scope", "195:0"),
    })
    assert done.returncode == 0 and done.stdout.split() == ["agent.service", "monitor.service"]


def test_the_host_holder_scan_exits_3_when_a_live_holders_cgroup_cannot_be_read(tmp_path):
    done = _run_holder_units_scan(tmp_path, {4242: (f"0::/system.slice/{AGENT}", "195:0"), 4243: (None, "195:0")})
    assert done.returncode == 3


def test_the_host_holder_scan_exits_3_when_an_open_file_cannot_be_read(tmp_path):
    done = _run_holder_units_scan(tmp_path, {4242: (f"0::/system.slice/{AGENT}", "195:0"), 4250: ("0::/", "denied")})
    assert done.returncode == 3


def test_the_host_unit_skips_the_reset_when_a_stop_fails_and_still_starts_the_fabric(tmp_path):
    done, log = _run_fabric_script(tmp_path, stop_fails=FM)
    assert done.returncode == gpu_reset._FABRIC_STOP_FAILED_EXIT and f"could not stop {FM}" in done.stderr
    assert log == [f"stop {FM}", f"start {NVLSM}", f"start {FM}"]


def test_the_host_unit_waits_for_an_exporter_whose_stop_failed_before_starting_it(tmp_path):
    # a stop that timed out before the exit event: the container is still dying, so a start right away is a no-op
    done, log = _run_fabric_script(tmp_path, units=(), containers=("e1",), stop_fails="e1")
    assert done.returncode == gpu_reset._FABRIC_STOP_FAILED_EXIT and "could not stop e1" in done.stderr
    assert log == ["docker stop -t 10 e1", "docker wait e1", "docker start e1"]


@pytest.mark.parametrize(
    "inspect_err, waits_to_exit, expected, handoff_rc",
    [
        ("", 2, "0 0", 0),
        # a transient daemon error is no proof the container is gone: keep waiting for its exit
        ("Cannot connect to the Docker daemon", 2, "0 0", 0),
        # a removed container ends the wait; its failed start is the validator's restoration failure
        ("Error: No such object: e1", 99, "1 1", 0),
        # an exporter that outlives its kill (stuck in the kernel) no longer holds the unit: it ends, hands the
        # start to a host unit that outlives it and names it (it still reads running, so the restore check cannot)
        ("", 99, "1 1", 0),
        # the hand-off itself failed: the unit says nothing will start it
        ("", 99, "1 1", 1),
    ],
)
def test_the_host_unit_keeps_waiting_for_an_exporter_whose_exit_outlasts_one_wait(
    tmp_path, inspect_err, waits_to_exit, expected, handoff_rc
):
    # the first wait times out with the exit still pending: a start then would be a no-op and nothing restarts it
    script = _unit_script(build_fabric_reset_command([], "true", ["e1"]), tmp_path)
    stubs = f"""
running=1; pending=0; waits=0; noops=0; inspect_err='{inspect_err}'; exit_at={waits_to_exit}
""" + """
docker() {
  case "$1" in
    stop) pending=1; return 1;;
    inspect) [ -z "$inspect_err" ] || { echo "$inspect_err" >&2; return 1; };;
    wait) waits=$((waits+1))
          [ "$waits" -ge "$exit_at" ] || return 124
          running=0; pending=0;;
    start) if [ "$running" = 1 ]; then noops=$((noops+1)); else running=1; fi;;
  esac
  return 0
}
timeout() { [ "$1" = -k ] && shift 2; shift; "$@"; }
sleep() { :; }
trap 'echo "$pending $noops"' EXIT
"""
    # `systemd-run` is no valid sh function name: a PATH stub records the hand-off
    (tmp_path / "systemd-run").write_text(f'#!/bin/sh\necho "$@" > "{tmp_path}/handoff"\nexit {handoff_rc}\n')
    (tmp_path / "systemd-run").chmod(0o755)
    (tmp_path / "systemctl").write_text("#!/bin/sh\necho not-found\n")
    (tmp_path / "systemctl").chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"}
    done = subprocess.run(["/bin/sh", "-c", stubs + script], capture_output=True, text=True, timeout=30, env=env)
    assert done.returncode == gpu_reset._FABRIC_STOP_FAILED_EXIT
    assert done.stdout.strip() == expected
    stuck = not inspect_err and waits_to_exit == 99
    handoff = tmp_path / "handoff"
    assert handoff.exists() == stuck
    if stuck:
        assert f"--unit={gpu_reset.EXPORTER_RESTART_UNIT}-e1 " in handoff.read_text()
        assert handoff.read_text().strip().endswith("e1")
        assert "-p Restart=on-failure" in handoff.read_text()
    assert ("e1 did not exit after a kill: restart pending" in done.stderr) == (stuck and handoff_rc == 0)
    assert ("could not start e1: it did not exit after a kill" in done.stderr) == (stuck and handoff_rc != 0)


@pytest.mark.parametrize(
    "wait_rc, wait_err, start_rc, state, rc, started",
    [
        (0, "", 0, "true false", 0, True),
        # a failed start (Docker restarting) fails the unit, so systemd runs it again
        (0, "", 1, "true false", 1, True),
        # a start that exits at once, or keeps restarting, fails the unit, so systemd runs it again
        (0, "", 0, "false false", 1, True),
        (0, "", 0, "true true", 1, True),
        # a failed wait (daemon restart) is no exit: fail without a no-op start, so systemd runs it again
        (1, "Cannot connect to the Docker daemon", 0, "true false", 1, False),
        # a removed container ends the unit: nothing is left to start
        (1, "Error: No such container: e1", 0, "true false", 0, False),
        # a wait that hit its deadline (a stalled dockerd) is no exit either
        (124, "", 0, "true false", 1, False),
    ],
)
def test_the_exporter_restart_unit_fails_until_it_has_started_the_exporter(
    wait_rc, wait_err, start_rc, state, rc, started
):
    stubs = f"""
sleep() {{ :; }}
timeout() {{ [ "$1" = -k ] && shift 2; shift; "$@"; }}
docker() {{
  case "$1" in
    wait) return {wait_rc};;
    inspect) [ "$2" = -f ] && {{ echo '{state}'; return 0; }}
      [ -z '{wait_err}' ] || {{ echo '{wait_err}' >&2; return 1; }};;
    start) echo started >&2; return {start_rc};;
  esac
}}
"""
    done = subprocess.run(
        ["/bin/sh", "-c", stubs + gpu_reset.EXPORTER_RESTART_SCRIPT, "e1"], capture_output=True, text=True, timeout=30
    )
    assert done.returncode == rc
    assert ("started" in done.stderr) == started


def _stuck_exporter_bin(tmp_path, docker_body):
    """PATH stand-ins for an exporter that outlives its kill: docker (the given case arms), and systemd-run plus
    systemctl keeping a registry of the transient units like systemd (a second unit of one name is rejected)."""
    units = tmp_path / "units"
    units.mkdir()
    for name, body in (
        ("docker", f'echo "docker $*" >> {tmp_path}/log; case "$1" in {docker_body} esac; exit 0'),
        (
            "systemd-run",
            'for a; do case "$a" in --unit=*) u="${a#--unit=}";; esac; done; '
            f'[ -e "{units}/$u" ] && exit 1; touch "{units}/$u"',
        ),
        ("systemctl", f'[ -e "{units}/$5" ] && echo loaded || echo not-found'),
    ):
        (tmp_path / name).write_text(f"#!/bin/sh\n{body}\n")
        (tmp_path / name).chmod(0o755)
    return {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"}, units


def test_two_stuck_exporters_each_get_their_own_restart_unit(tmp_path):
    env, units = _stuck_exporter_bin(tmp_path, "stop) exit 1;; wait) exit 124;;")
    # e1 still has its recovery unit when e2 (and later e1 again) outlives its kill
    for cid in ("e1", "e2", "e1"):
        script = _unit_script(build_fabric_reset_command([], "true", [cid]), tmp_path)
        done = subprocess.run(["/bin/sh", "-c", script], capture_output=True, text=True, timeout=30, env=env)
        assert done.returncode == gpu_reset._FABRIC_STOP_FAILED_EXIT
        assert f"{cid} did not exit after a kill: restart pending" in done.stderr, done.stderr
    assert sorted(p.name for p in units.iterdir()) == [f"{gpu_reset.EXPORTER_RESTART_UNIT}-{c}" for c in ("e1", "e2")]


def test_hanging_docker_calls_hit_their_deadline_and_still_hand_off(tmp_path, monkeypatch):
    # dockerd accepted the calls but never answers: the deadlines end them and the recovery path still runs
    monkeypatch.setattr(gpu_reset, "STOP_GRACE_SECONDS", 0)
    monkeypatch.setattr(gpu_reset, "_DOCKER_CALL_SECONDS", 1)
    env, units = _stuck_exporter_bin(tmp_path, "stop|kill|inspect|start) exec sleep 60;; wait) exit 124;;")
    script = _unit_script(build_fabric_reset_command([], "true", ["e1"]), tmp_path)
    done = subprocess.run(["/bin/sh", "-c", script], capture_output=True, text=True, timeout=30, env=env)
    assert done.returncode == gpu_reset._FABRIC_STOP_FAILED_EXIT and "could not stop e1" in done.stderr
    assert "e1 did not exit after a kill: restart pending" in done.stderr
    assert [p.name for p in units.iterdir()] == [f"{gpu_reset.EXPORTER_RESTART_UNIT}-e1"]
    assert (tmp_path / "log").read_text().splitlines()[-1] == "docker start e1"


def test_the_host_unit_starts_the_listed_prerequisites_before_a_found_holder(tmp_path):
    # a monitor that fails to start without Fabric Manager comes back, since Fabric Manager starts first
    done, log = _run_fabric_script(tmp_path, units=(AGENT, FM, PERSISTENCED), needs=FM)
    assert done.returncode == 0 and "could not start" not in done.stderr
    assert log[-2:] == [f"start {FM}", f"start {AGENT}"]


AGENT = "some-gpu-agent.service"


DCGM, PERSISTENCED = "nvidia-dcgm", "nvidia-persistenced"
EXPORTER = "e" * 64


def _b300(**kwargs) -> FakeHost:
    # HGX B300 the pod owns whole: Fabric Manager, DCGM, nvidia-persistenced and a dcgm-exporter container
    return FakeHost(
        device_ids=(),
        count=-1,
        fabric_manager=True,
        holders={DCGM: True, PERSISTENCED: True},
        **{"ps": f"{CID} false lium/pod\n{EXPORTER} false nvcr.io/nvidia/k8s/dcgm-exporter:4.2.3\n", **kwargs},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("auto_remove", [False, True])
async def test_a_whole_host_pod_stops_every_host_gpu_holder_around_the_reset(auto_remove):
    # an exporter run with --rm is deleted by its stop: it is left running (a reset it blocks is refused honestly)
    flag = str(auto_remove).lower()
    host = _b300(ps=f"{CID} false lium/pod\n{EXPORTER} {flag} nvcr.io/nvidia/k8s/dcgm-exporter:4.2.3\n")
    report, start = await _reset(host, backend=U)
    assert report.ok and {g.result for g in report.gpus} == {"reset"}
    exporter = {} if auto_remove else {EXPORTER: False}
    assert host.active_during_reset == {FM: False, NVLSM: False, DCGM: False, PERSISTENCED: False, **exporter}
    stopped = [entry for entry in host.fm_log if entry[0] in ("stop", "docker stop")]
    started = [entry for entry in host.fm_log if entry[0] in ("start", "docker start")]
    assert stopped == [*[("docker stop", c) for c in exporter], ("stop", DCGM), ("stop", FM), ("stop", PERSISTENCED)]
    # started again last stopped first; then the validator checks each one is up
    assert started[: 3 + len(exporter)] == [
        ("start", PERSISTENCED),
        ("start", FM),
        ("start", DCGM),
        *[("docker start", c) for c in exporter],
    ]
    assert (("docker start", EXPORTER) in started) == (not auto_remove)
    assert all(host.active[u] for u in (FM, DCGM, PERSISTENCED, *exporter))
    start.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "start_ok,state,ok",
    [
        (True, "true false", True),
        (False, "false false", False),  # the start failed
        (True, "false false", False),  # started, then exited at once
        (True, "true true", False),  # restarting in a loop
    ],
)
async def test_a_reset_is_ok_only_when_the_exporter_runs_again(start_ok, state, ok):
    host = _b300(docker_start_ok=start_ok, container_state=state)
    report, _ = await _reset(host, backend=U)
    assert report.ok == ok
    assert ("is not running again" in (report.detail or "")) == (not ok)


@pytest.mark.asyncio
async def test_a_refused_reset_on_a_whole_host_pod_is_not_ok_and_every_holder_is_up_again():
    host = _b300(reset_exit=1)
    report, start = await _reset(host, backend=U)
    assert not report.ok and report.error_code == GpuResetErrorCode.ResetNotSupported
    assert "the GPUs were not reset" in report.detail
    assert {g.result for g in report.gpus} == {"latch_cleared"}
    assert all(host.active[u] for u in (FM, DCGM, PERSISTENCED, EXPORTER))
    start.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_refused_reset_logs_the_whole_host_output_the_report_cuts(caplog):
    host = _b300(reset_exit=1)
    with caplog.at_level("WARNING", logger="services.gpu_reset"):
        report, _ = await _reset(host, backend=U)
    logged = [r.getMessage() for r in caplog.records if "nvidia-smi -r refused" in r.getMessage()]
    assert len(logged) == 1 and FM in logged[0]
    assert NAME in logged[0]
    assert "stdout" in logged[0] and "stderr" in logged[0]


@pytest.mark.asyncio
async def test_a_refused_reset_whose_holder_does_not_come_back_still_logs_the_host_output(caplog):
    host = _b300(reset_exit=1, fm_start_ok=False)
    with caplog.at_level("WARNING", logger="services.gpu_reset"):
        report, _ = await _reset(host, backend=U)
    assert not report.ok and "nvidia-fabricmanager is not active" in report.detail
    assert len([r for r in caplog.records if "nvidia-smi -r refused" in r.getMessage()]) == 1


@pytest.mark.asyncio
async def test_every_host_gpu_holder_comes_back_when_the_reset_raises():
    host = _b300()
    host.reset_raises = True
    report, start = await _reset(host, backend=U)
    assert not report.ok
    started = [entry for entry in host.fm_log if entry[0] in ("start", "docker start")]
    assert started[:4] == [("start", PERSISTENCED), ("start", FM), ("start", DCGM), ("docker start", EXPORTER)]
    assert all(host.active[u] for u in (FM, DCGM, PERSISTENCED, EXPORTER))
    start.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "backend,holders,uses_unit,systemd",
    [
        (U, {DCGM: True}, True, True),
        # only a holder the validator does not list: the host unit still finds and stops it
        (U, {"some-gpu-agent": True}, True, True),
        # a `--gpus all` container the backend gave only half of the host GPUs does not own the host
        (U[:2], {DCGM: True}, False, True),
        # a host proven without systemd, with nothing to stop: the direct reset, as before the host unit
        (U, {}, False, False),
    ],
)
async def test_only_a_pod_owning_the_host_by_container_and_backend_uses_the_host_unit(
    backend, holders, uses_unit, systemd
):
    host = FakeHost(device_ids=(), count=-1, holders=holders, systemd=systemd)
    report, _ = await _reset(host, backend=backend)
    assert report.ok and {g.result for g in report.gpus} == {"reset"}
    assert any(cmd.startswith("nsenter -t 1 -m -- systemd-run") for cmd in host.commands) == uses_unit
    assert any(cmd.startswith("nvidia-smi -r") for cmd in host.commands)
    if not uses_unit:
        assert not any(entry[0] in ("stop", "docker stop") for entry in host.fm_log)
        assert (gpu_reset.DOCKER_PS_COMMAND in host.commands) == (backend == U)


@pytest.mark.asyncio
async def test_a_pod_sharing_the_host_never_stops_the_host_gpu_holders():
    host = FakeHost(holders={DCGM: True, PERSISTENCED: True})
    report, _ = await _reset(host)
    assert report.ok and not any(entry[0] in ("stop", "docker stop") for entry in host.fm_log)
    assert not any("systemd-run" in cmd or cmd == gpu_reset.DOCKER_PS_COMMAND for cmd in host.commands)


@pytest.mark.asyncio
async def test_a_found_holder_that_does_not_start_again_fails_the_reset():
    host = _b300(host_holder_units=f"{AGENT}\n", start_fails=(AGENT,))
    report, start = await _reset(host, backend=U)
    assert not report.ok and report.error_code == GpuResetErrorCode.ResetNotSupported
    assert f"{AGENT} is not active again after the reset" in report.detail
    assert not any(cmd.startswith("CUDA_VISIBLE_DEVICES=") for cmd in host.commands)
    start.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_service_the_unit_stopped_on_its_own_that_does_not_start_again_fails_the_reset():
    # DAH-6741 review: the unit stopped a service that opened a GPU after the scan; its failed start is retried and
    # reported like any stopped holder, never a success with the service down
    late = "late-monitor.service"
    host = _b300(late_units=(late,), start_fails=(late,))
    report, start = await _reset(host, backend=U)
    assert not report.ok and report.error_code == GpuResetErrorCode.ResetNotSupported
    assert f"{late} is not active again after the reset" in report.detail
    assert ("start", late) in host.fm_log[host.fm_log.index(("start", late)) + 1:]  # the restore's retry
    start.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("starts", [True, False])
async def test_a_service_left_stopped_by_a_reset_whose_restore_did_not_run_is_restored_by_the_next(starts):
    # DAH-6741 review: a dropped session skipped the restore; the unit's list still names the service it stopped
    late = "late-monitor.service"
    host = _b300(start_fails=() if starts else (late,))
    host.active[late], host.stopped_file = False, [late]
    report, _ = await _reset(host, backend=U)
    assert report.ok == starts and host.active[late] == starts
    assert ("start", late) in host.fm_log
    # cleared only once every listed service is up again
    assert (host.stopped_file == []) == starts and (late in host.stopped_file) != starts
    if not starts:
        assert f"{late} is not active again after the reset" in report.detail


@pytest.mark.parametrize("starts", [True, False])
@pytest.mark.asyncio
async def test_a_service_left_stopped_is_restored_before_a_precheck_refusal_returns(starts):
    # DAH-6741 review: the earlier unit is gone and the host GPU listing fails (GpusLostOnHost), an early return
    late = "late-monitor.service"
    host = FakeHost(gpus_result=make_command_result(False, exit_code=255), start_fails=() if starts else (late,))
    host.active[late], host.stopped_file = False, [late]
    report, start = await _reset(host)
    assert host.active[late] == starts and ("start", late) in host.fm_log
    assert host.stopped_file == ([] if starts else [late])
    if starts:
        assert report.error_code == GpuResetErrorCode.GpusLostOnHost
    else:
        # nothing is stopped while it stays down; the list stays for the next restore
        assert report.error_code == GpuResetErrorCode.ResetNotSupported and f"{late} is not active" in report.detail
    _no_change(host, start)


@pytest.mark.asyncio
@pytest.mark.parametrize("listed", [[FM, "+late-monitor.service"], ["+late-monitor.service", FM]])
async def test_a_listed_late_service_starts_after_its_prerequisite_whatever_the_list_order(listed):
    # DAH-6741 review: an earlier whole-host reset timed out with Fabric Manager and a late monitor that needs it
    # both down; the precheck's restore starts Fabric Manager first, so the monitor comes back and the reset runs
    late = "late-monitor.service"
    host = _b300(needs_fm=(late,))
    host.active[FM], host.active[late], host.stopped_file = False, False, list(listed)
    report, _ = await _reset(host, backend=U)
    starts = [unit for verb, unit in host.fm_log if verb == "start" and unit in (FM, late)]
    assert starts[:2] == [FM, late]
    assert report.ok and host.active[late] and host.active[FM] and host.stopped_file == []


@pytest.mark.asyncio
async def test_an_empty_list_of_stopped_services_is_not_cleared():
    host = FakeHost()
    report, _ = await _reset(host)
    assert report.ok and gpu_reset.STOPPED_UNITS_CLEAR_COMMAND not in host.commands


@pytest.mark.asyncio
async def test_a_service_the_unit_stopped_on_its_own_is_checked_after_the_others():
    late = "late-monitor.service"
    host = _b300(late_units=(late,))
    report, _ = await _reset(host, backend=U)
    assert report.ok and host.active[late]
    checks = [unit for verb, unit in host.fm_log if verb == "is-active"]
    assert checks[-1] == late and host.stopped_file == []  # all up again: the list is cleared


@pytest.mark.asyncio
@pytest.mark.parametrize("left_by_earlier", [False, True])
async def test_an_escaped_service_name_is_checked_and_started_as_itself(left_by_earlier):
    # DAH-6741 review: systemd escapes "-" in an instance name as \x2d; unquoted, the shell drops the backslash and
    # is-active / start name another unit, so the real one stays down
    escaped = "gpu-agent@pci\\x2d0000.service"
    host = _b300(host_holder_units="" if left_by_earlier else f"{escaped}\n")
    if left_by_earlier:
        host.active[escaped], host.stopped_file = False, [escaped]
    report, _ = await _reset(host, backend=U)
    assert report.ok and host.active[escaped]
    assert ("start", escaped) in host.fm_log and ("is-active", escaped) in host.fm_log


@pytest.mark.asyncio
async def test_a_host_service_found_holding_a_gpu_stops_first_and_starts_last():
    # DAH-6741 review: found before anything stops; a listed one the scan also finds is not stopped twice
    host = _b300(host_holder_units=f"{AGENT}\n{DCGM}.service\n")
    report, _ = await _reset(host, backend=U)
    assert report.ok
    stopped = [unit for verb, unit in host.fm_log if verb == "stop"]
    started = [unit for verb, unit in host.fm_log if verb == "start"]
    assert stopped == [AGENT, DCGM, FM, PERSISTENCED] and started[:4] == [PERSISTENCED, FM, DCGM, AGENT]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer,code",
    [
        (make_command_result(False, exit_code=3), GpuResetErrorCode.ResetNotSupported),
        # a scan blocked on a stalled filesystem: the SSH timeout, with nothing stopped yet
        (make_command_result(False, exit_code=-1, error_type="timeout"), GpuResetErrorCode.ExecutorUnreachable),
    ],
)
async def test_a_host_holder_scan_that_is_not_whole_refuses_before_anything_stops(answer, code):
    host = _b300(host_holder_units=answer)
    report, start = await _reset(host, backend=U)
    assert not report.ok and report.error_code == code and "nothing was stopped" in report.detail
    assert report.stopped_during_reset is False
    _no_change(host, start)


@pytest.mark.asyncio
async def test_fabric_manager_not_coming_back_fails_the_reset():
    host = FakeHost(device_ids=(), count=-1, fabric_manager=True, fm_start_ok=False)
    report, _ = await _reset(host, backend=U)
    assert not report.ok and report.error_code == GpuResetErrorCode.ResetNotSupported
    assert "nvidia-fabricmanager is not active" in report.detail


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
    # a latch clear is no reset: never ok, and the detail says the GPUs were not reset
    assert not report.ok and report.error_code == GpuResetErrorCode.ResetNotSupported
    assert "nvidia-smi -r refused, the GPUs were not reset" in report.detail and "only the latch clear" in report.detail
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
    assert report.error_code is None and [g.result for g in report.gpus] == ["reset"] * 2


@pytest.mark.asyncio
async def test_a_failed_start_after_a_refused_reset_keeps_the_refusal_code():
    report, _ = await _reset(FakeHost(reset_exit=255), start=AsyncMock(side_effect=RuntimeError("no start")))
    assert report.error_code == GpuResetErrorCode.ResetNotSupported and not report.container_restarted


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
    remove_unacked_container_created = RedisService.remove_unacked_container_created
    _remove_unacked_container_created_unless = RedisService._remove_unacked_container_created_unless


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
async def test_the_forwarded_reset_carries_its_idle_only_mark_as_its_own_field():
    # DAH-6784: compute_client builds the wire request from the redis payload; drop the key and every idle-only
    # reset reaches the backend unmarked, ending a rental placed since the check read it
    import asyncio
    from unittest.mock import MagicMock

    from clients.compute_client import ComputeClient
    from services.redis_service import RESET_VERIFIED_JOB_CHANNEL

    def message(executor_uuid: str, **extra):
        data = {"miner_hotkey": "hk", "executor_uuid": executor_uuid, "reason": 0, "check_id": "c",
                "reason_code": None, "evidence": None, **extra}
        return {"channel": RESET_VERIFIED_JOB_CHANNEL.encode(), "data": json.dumps(data)}

    class PubSub:
        async def listen(self):
            yield message("marked", unrented_only=True)
            yield message("older-validator")
            yield message("garbled", unrented_only="yes")
            raise asyncio.CancelledError

        async def aclose(self):
            pass

    redis = PendingRedis()
    redis.subscribe = AsyncMock(return_value=PubSub())
    client = ComputeClient.__new__(ComputeClient)
    client.lock = asyncio.Lock()
    client.message_queue = []
    client.logging_extra = {}
    client.miner_service = MagicMock(redis_service=redis)
    client.my_hotkey = MagicMock(return_value="validator-hotkey")

    with pytest.raises(asyncio.CancelledError):
        await client.subscribe_mesages_from_redis()

    assert [(r.executor_uuid, r.unrented_only) for r in client.message_queue] == [
        ("marked", True),
        ("older-validator", False),
        ("garbled", False),
    ]
    assert "unrented_only" in client.message_queue[0].model_dump()


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
    # DAH-6741 review: the container was never read, so neither says it runs (an earlier reset may have stopped it)
    assert busy.pod_stopped is None and gone.pod_stopped is None


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
    assert not report.ok and not report.pod_stopped and report.stopped_during_reset
    assert not any(cmd.startswith("nvidia-smi -r") for cmd in host.commands)


def _waited(host: FakeHost) -> bool:
    return any(cmd.startswith("timeout 90 /usr/bin/docker wait") for cmd in host.commands)


@pytest.mark.asyncio
async def test_a_stop_without_an_exit_event_waits_for_the_exit_then_starts():
    # `docker stop` timed out waiting for the exit event; starting at once was a no-op on a dying container
    host = FakeHost(stop_exit=1)
    report, start = await _reset(host)
    assert not report.ok and report.container_restarted
    assert _waited(host)
    start.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_failed_start_waits_for_the_exit_and_starts_once_more():
    # the remount's exec into the dying container failed on setns: the second start after the exit brings it up
    start = AsyncMock(side_effect=[RuntimeError("OCI runtime exec failed: setns"), None])
    report, _ = await _reset(FakeHost(stop_exit=1), start=start)
    assert report.container_restarted and "did not start" not in (report.detail or "")
    assert start.await_count == 2


@pytest.mark.asyncio
async def test_a_good_stop_starts_without_waiting():
    host = FakeHost()
    report, start = await _reset(host)
    assert report.ok and report.container_restarted and not _waited(host)


@pytest.mark.asyncio
async def test_a_reset_of_a_stopped_container_starts_it_then_resets():
    host, running = FakeHost(main_pid=0), FakeHost()

    async def start_it():
        host.inspect, host.cgroups = running.inspect, running.cgroups

    start = AsyncMock(side_effect=start_it)
    report, _ = await _reset(host, start=start)
    assert report.ok and report.container_restarted and "not running" in report.detail
    assert any(cmd.startswith("nvidia-smi -r") for cmd in host.commands)
    assert start.await_count == 2  # once before the reset, once after its stop


@pytest.mark.asyncio
async def test_a_stopped_container_that_does_not_start_changes_nothing_else():
    host = FakeHost(main_pid=0)
    report, start = await _reset(host, start=AsyncMock(side_effect=RuntimeError("no start")))
    assert not report.ok and not report.container_restarted and "did not start" in report.detail
    assert report.stopped_during_reset is True  # DAH-6741 review: it is down, so the answer must not say it runs
    assert not any(cmd.startswith(("/usr/bin/docker stop", "nvidia-smi -r")) for cmd in host.commands)
    start.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_stopped_container_started_but_refused_by_the_precheck_reports_the_start():
    host = FakeHost(main_pid=0)
    report, start = await _reset(host)  # the start does not make the fake run: the precheck refuses again
    assert not report.ok and report.container_restarted and report.error_code == GpuResetErrorCode.ResetNotSupported
    assert report.stopped_during_reset is True  # it was found stopped
    assert not any(cmd.startswith("/usr/bin/docker stop") for cmd in host.commands)
    start.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_stopped_container_whose_final_start_fails_does_not_report_a_restart():
    # review 4447075986: the recovery start set the flag; the failed start after the reset must clear it
    host, running = FakeHost(main_pid=0), FakeHost()
    starts = 0

    async def start_it():
        nonlocal starts
        starts += 1
        if starts > 1:
            raise RuntimeError("no start")
        host.inspect, host.cgroups = running.inspect, running.cgroups

    report, _ = await _reset(host, start=AsyncMock(side_effect=start_it))
    assert not report.ok and not report.container_restarted and "did not start again" in report.detail


@pytest.mark.asyncio
async def test_a_stopped_encrypted_pod_without_its_plaintext_path_is_refused_before_any_start():
    # review 4447076002: the start would leave the pod running without its plaintext mount
    host = FakeHost(main_pid=0, mounts=[{"Destination": "/lium-cipher"}])
    report, start = await _reset(host, encrypted_volume_mount="/lium-cipher")
    assert not report.ok and report.error_code == GpuResetErrorCode.ResetNotSupported
    assert not report.container_restarted and report.stopped_during_reset is None  # refused before its state was read
    _no_change(host, start)


# DAH-6741: the driver refuses `nvidia-smi -r` while any process holds the GPU open, not only a compute app
XORG = (1500, "Xorg")
NVIDIA_SMI_REFUSAL = (
    "GPU 00000000:0A:00.0 is currently in use by another process.\n\n1 device is currently being used by one or more "
    "other processes (e.g., Fabric Manager, CUDA application, graphics application such as an X server, or a "
    "monitoring application such as another instance of nvidia-smi). Please first kill all processes using these "
    "devices and all compute applications running in the system."
)


@pytest.mark.asyncio
async def test_a_part_node_reset_held_open_by_a_host_process_is_refused_before_the_pod_stops():
    # Arrange
    host = FakeHost(device_holders=[XORG], cgroups={1500: "0::/system.slice/display-manager.service"})

    # Act
    report, start = await _reset(host)

    # Assert
    assert not report.ok and report.error_code == GpuResetErrorCode.GpuInUseOutsidePod
    assert "Xorg (pid 1500)" in report.detail and "the pod was not stopped" in report.detail
    assert not report.stopped_during_reset and not report.container_restarted
    _no_change(host, start)


@pytest.mark.asyncio
async def test_a_holder_whose_cgroup_cannot_be_read_is_not_called_outside_the_pod():
    # DAH-6741 review: an unreadable cgroup is not proof the holder is outside the pod; it may be the renter's own
    host = FakeHost(device_holders=[XORG])

    report, start = await _reset(host)

    assert report.error_code == GpuResetErrorCode.GpuInUseOutsidePod and "Xorg (pid 1500)" in report.detail
    assert "could not place in the pod" in report.detail
    assert "only if you can tell it is yours from inside the pod" in report.detail and "host PID" in report.detail
    assert "process outside the pod" not in report.detail and not report.stopped_during_reset
    _no_change(host, start)


@pytest.mark.asyncio
async def test_a_compute_app_whose_cgroup_cannot_be_read_is_not_called_outside_the_pod():
    host = FakeHost(apps=[(998, U[0])])

    report, start = await _reset(host)

    assert report.error_code == GpuResetErrorCode.GpuInUseOutsidePod
    assert "could not place in the pod" in report.detail
    assert "stop them only if you can tell they are yours" in report.detail
    _no_change(host, start)


@pytest.mark.asyncio
async def test_a_holder_seen_in_one_read_only_does_not_refuse_the_reset():
    # Arrange: a passing nvidia-smi of the host
    host = FakeHost(transient_holders=[(1600, "nvidia-smi")], cgroups={1600: "0::/system.slice/cron.service"})

    # Act
    report, _ = await _reset(host)

    # Assert
    assert report.ok and host.holder_reads == 2


@pytest.mark.asyncio
async def test_the_pods_own_processes_holding_its_gpus_do_not_refuse_the_reset():
    # Arrange
    host = FakeHost(device_holders=[(4243, "python")], cgroups={4243: f"0::/../docker-{CID}.scope"})

    # Act
    report, _ = await _reset(host)

    # Assert
    assert report.ok


@pytest.mark.asyncio
async def test_a_pod_that_owns_every_host_gpu_leaves_the_holders_to_the_host_unit():
    # Arrange: the unit stops the host holders itself
    host = FakeHost(device_ids=U, device_holders=[XORG], cgroups={1500: "0::/system.slice/display-manager.service"})

    # Act
    report, _ = await _reset(host, backend=U)

    # Assert
    assert report.ok and host.holder_reads == 0


@pytest.mark.asyncio
async def test_a_refused_reset_reports_the_start_of_the_nvidia_smi_text():
    # Arrange
    host = FakeHost(reset_exit=255, reset_output=NVIDIA_SMI_REFUSAL)

    # Act
    report, _ = await _reset(host)

    # Assert
    assert not report.ok
    assert "the GPUs were not reset: GPU 00000000:0A:00.0 is currently in use by another process." in report.detail


def _run_holder_scan(tmp_path, holders: dict[int, str], minors: dict[str, int], find: str | None = None) -> list[str]:
    """The holder command run by sh over a fake /proc: holders maps pid -> "<major>:<minor>" of the device it holds
    open ("denied": an fd the scan cannot read). find, when given, is the body of a fake find put first on PATH
    ($ROOT is the fake /proc)."""
    root = tmp_path / "proc"
    env = {**os.environ, "ROOT": str(root)}
    (tmp_path / "bin").mkdir()
    fakes = {"stat": FAKE_STAT, **({"find": f"#!/bin/sh\n{find}\n"} if find is not None else {})}
    for name, body in fakes.items():
        (tmp_path / "bin" / name).write_text(body)
        (tmp_path / "bin" / name).chmod(0o755)
    env["PATH"] = f"{tmp_path / 'bin'}:{env['PATH']}"
    for bus, minor in minors.items():
        (root / "driver/nvidia/gpus" / bus).mkdir(parents=True)
        (root / "driver/nvidia/gpus" / bus / "information").write_text(f"Model: x\nDevice Minor: \t {minor}\n")
    for pid, device in holders.items():
        _fake_fd(root, pid, device)
        (root / str(pid) / "comm").write_text(f"proc{pid}\n")
    command = gpu_reset.device_holders_command(["00000000:1b:00.0", "00000000:0a:00.0"])
    command = command.replace("/proc/driver", f"{root}/driver").replace("/proc/[0-9]*", f"{root}/[0-9]*")
    command = command.replace("/proc/$k/comm", f"{root}/$k/comm")
    done = subprocess.run(["sh", "-c", command], capture_output=True, text=True, env=env)
    assert done.returncode == 0, done.stderr
    return done.stdout.splitlines()


def test_a_find_error_other_than_an_exited_process_leaves_the_gpu_unread(tmp_path):
    # Arrange: find lists Xorg, then fails on a /proc entry it could not read
    find = """printf 'c3 3 %s\\n' "$ROOT/1500/fd/5"; echo "find: '/proc/9/fd': Permission denied" >&2; exit 1"""

    # Act
    lines = _run_holder_scan(tmp_path, {1500: "195:3"}, {"0000:1b:00.0": 3}, find=find)

    # Assert: the holder is named, and the GPU has no "gpu" line, so the reset is refused as unread
    assert lines == ["1500 proc1500"]


def test_a_find_killed_without_a_message_leaves_the_gpu_unread(tmp_path):
    # Act
    lines = _run_holder_scan(tmp_path, {}, {"0000:1b:00.0": 3}, find="exit 137")

    # Assert
    assert lines == []


def test_a_process_that_exits_mid_scan_keeps_the_gpu_read(tmp_path):
    # Arrange
    find = """printf 'c3 3 %s\\n' "$ROOT/1500/fd/5"; echo "find: '/proc/9/fd': No such file or directory" >&2; exit 1"""

    # Act
    lines = _run_holder_scan(tmp_path, {1500: "195:3"}, {"0000:1b:00.0": 3}, find=find)

    # Assert
    assert sorted(lines) == ["1500 proc1500", "gpu 0000:1b:00.0"]


def test_the_holder_scan_names_each_read_gpu_and_the_processes_holding_it(tmp_path):
    # Act: GPU 0a has no minor in the driver's file, so it is not read; 44 holds another device with minor 3
    holders = {41: "195:3", 42: "195:3", 43: "195:1", 44: "226:3"}
    lines = _run_holder_scan(tmp_path, holders, {"0000:1b:00.0": 3})

    # Assert
    assert sorted(lines) == ["41 proc41", "42 proc42", "gpu 0000:1b:00.0"]


def test_the_holder_scan_matches_the_gpus_device_number_whatever_the_node_is_called(tmp_path):
    # DAH-6741 review: any device node with the GPU's numbers opens it, so the path name proves nothing
    lines = _run_holder_scan(tmp_path, {1500: "195:10"}, {"0000:1b:00.0": 10})

    assert sorted(lines) == ["1500 proc1500", "gpu 0000:1b:00.0"]


def test_an_open_file_whose_device_cannot_be_read_leaves_the_gpu_unread(tmp_path):
    # DAH-6741 review: an fd the scan cannot stat may be the GPU, so the reading is not whole
    lines = _run_holder_scan(tmp_path, {1500: "denied", 41: "195:3"}, {"0000:1b:00.0": 3})

    assert lines == ["41 proc41"]


def test_the_holder_scan_never_names_its_own_session(tmp_path):
    # Arrange: the scanning shell's parent holds the device (a stand-in for the validator's SSH session)
    lines = _run_holder_scan(tmp_path, {os.getpid(): "195:3", 44: "195:3"}, {"0000:1b:00.0": 3})

    # Assert
    assert sorted(lines) == ["44 proc44", "gpu 0000:1b:00.0"]


@pytest.mark.asyncio
async def test_a_gpu_whose_holders_could_not_be_read_is_refused_before_the_pod_stops():
    # Arrange: the scan answered for GPU 1b (U[0]) only
    host = FakeHost(holder_scan=make_command_result(True, f"gpu {BUS[0][-12:].lower()}\n"))

    # Act
    report, start = await _reset(host)

    # Assert
    assert not report.ok and report.error_code == GpuResetErrorCode.ResetNotSupported
    assert "could not read which host processes hold GPU 0" in report.detail and "pod was not stopped" in report.detail
    _no_change(host, start)


@pytest.mark.asyncio
async def test_a_holder_scan_that_times_out_is_refused_before_the_pod_stops():
    # Arrange
    host = FakeHost(holder_scan=make_command_result(False, exit_code=-1, error_type="timeout"))

    # Act
    report, start = await _reset(host)

    # Assert
    assert not report.ok and report.error_code == GpuResetErrorCode.ExecutorUnreachable
    _no_change(host, start)


@pytest.mark.asyncio
async def test_a_failed_holder_scan_is_refused_before_the_pod_stops():
    # Arrange: no find on the host
    host = FakeHost(holder_scan=make_command_result(False, exit_code=3))

    # Act
    report, start = await _reset(host)

    # Assert
    assert not report.ok and report.error_code == GpuResetErrorCode.ResetNotSupported
    _no_change(host, start)


@pytest.mark.asyncio
async def test_an_executor_container_process_holding_the_gpu_refuses_the_reset():
    # Arrange: the executor's own metrics process, in the cgroup the validator's SSH session also runs in
    session = "0::/system.slice/docker-executor.scope"
    host = FakeHost(own_cgroup=session, device_holders=[(1700, "python")], cgroups={1700: session})

    # Act
    report, start = await _reset(host)

    # Assert
    assert report.error_code == GpuResetErrorCode.GpuInUseOutsidePod and "python (pid 1700)" in report.detail
    _no_change(host, start)