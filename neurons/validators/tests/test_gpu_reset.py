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
        if cmd.startswith("nsenter -t 1 -m -- systemd-run "):
            # the host unit: stop, reset, start again, whatever happens to the SSH session
            script = shlex.split(cmd)[-1]
            found = re.search(r"\|\| for u in ([^;$]*); do systemctl stop", script)
            units = found.group(1).split() if found else []
            containers = re.search(r"for u in ([^;]*); do docker stop", script)
            containers = containers.group(1).split() if containers else []
            for cid in containers:
                self.active[cid] = False
                self.fm_log.append(("docker stop", cid))
            for unit in units:
                self.active[unit] = False
                self.fm_log.append(("stop", unit))
            self.commands.append(re.search(r"\{ (nvidia-smi -r [^;]*);", script).group(1))
            self.active_during_reset = dict(self.active)
            if self.reset_hangs:
                # the SSH timeout fires while the host unit still holds the fabric down
                self.hung_units = units
                return make_command_result(False, stderr="timed out", exit_code=-1)
            for unit in reversed(units):
                self.active[unit] = unit != "nvidia-fabricmanager" or self.fm_start_ok
                self.fm_log.append(("start", unit))
            for cid in reversed(containers):
                self.active[cid] = True
                self.fm_log.append(("docker start", cid))
            if self.reset_raises:
                raise RuntimeError("ssh dropped")
            return make_command_result(self.reset_exit == 0, "refused", exit_code=self.reset_exit)
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
            verb, unit = cmd.split()[-2:]
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
            ok = verb == "stop" or unit != "nvidia-fabricmanager" or self.fm_start_ok
            self.active[unit] = verb == "start" and ok
            return make_command_result(ok, exit_code=0 if ok else 1)
        if cmd.startswith("nvidia-smi -r"):
            self.active_during_reset = dict(self.active)
            if self.reset_raises:
                raise RuntimeError("ssh dropped")
            return make_command_result(self.reset_exit == 0, "refused", exit_code=self.reset_exit)
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


def _run_fabric_script(
    tmp_path, *, stop_fails="", start_fails="", reset_exit=0, units=(FM, NVLSM), containers=(), procs=None
):
    """Runs the host unit's script under /bin/sh with logging stand-ins for systemctl, docker, fuser and
    nvidia-smi; procs is the host's /proc: {pid: (comm, cgroup)}, all of them holding a GPU device. stop_fails and
    start_fails name the one unit whose systemctl stop or start exits 1."""
    log = tmp_path / "log"
    proc = tmp_path / "proc"
    for pid, (comm, cgroup) in (procs or {}).items():
        (proc / str(pid)).mkdir(parents=True)
        (proc / str(pid) / "comm").write_text(f"{comm}\n")
        (proc / str(pid) / "cgroup").write_text(f"{cgroup}\n")
    fuser_out = " ".join(f"{pid}m" if i else str(pid) for i, pid in enumerate(procs or {}))
    for name, body in (
        (
            "systemctl",
            f'echo "$1 $2" >> {log}; [ "$1" = stop ] && [ "$2" = "$STOP_FAILS" ] && exit 1; '
            '[ "$1" = start ] && [ "$2" = "$START_FAILS" ] && exit 1; exit 0',
        ),
        ("docker", f'echo "docker $*" >> {log}; [ "$1" = stop ] && [ "$4" = "$STOP_FAILS" ] && exit 1; exit 0'),
        ("fuser", f'echo "$*" | grep -q "/dev/nvidia" && echo " {fuser_out}"; exit 0'),
        ("nvidia-smi", f'echo "reset $*" >> {log}; echo "In use by another client"; exit {reset_exit}'),
    ):
        (tmp_path / name).write_text(f"#!/bin/sh\n{body}\n")
        (tmp_path / name).chmod(0o755)
    cmd = build_fabric_reset_command(list(units), f"nvidia-smi -r -i {U[0]}", list(containers))
    assert cmd.startswith("nsenter -t 1 -m -- systemd-run ")
    script = shlex.split(cmd)[-1].replace("/proc/", f"{proc}/")
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "STOP_FAILS": stop_fails, "START_FAILS": start_fails}
    done = subprocess.run(["/bin/sh", "-c", script], env=env, capture_output=True, text=True)
    return done, log.read_text().splitlines()


def test_the_host_unit_starts_the_fabric_again_after_a_failed_reset(tmp_path):
    done, log = _run_fabric_script(tmp_path, reset_exit=3)
    assert done.returncode == 3
    assert log == [f"stop {FM}", f"stop {NVLSM}", f"reset -r -i {U[0]}", f"start {NVLSM}", f"start {FM}"]


def test_the_host_unit_stops_every_gpu_holder_and_starts_them_again_in_reverse(tmp_path):
    # B300: DCGM and nvidia-persistenced hold every GPU, a dcgm-exporter container too, and one more host
    # service the validator did not know of
    procs = {
        4242: ("nv-hostengine", "0::/system.slice/some-gpu-agent.service"),
        4243: ("python3", "0::/system.slice/docker-abc.scope"),
    }
    units = ("nvidia-dcgm", FM, "nvidia-persistenced")
    done, log = _run_fabric_script(tmp_path, reset_exit=3, units=units, containers=("e1",), procs=procs)
    assert done.returncode == 3
    assert log == [
        "docker stop -t 10 e1",
        "stop nvidia-dcgm",
        f"stop {FM}",
        "stop nvidia-persistenced",
        "stop some-gpu-agent.service",
        f"reset -r -i {U[0]}",
        "start some-gpu-agent.service",
        "is-active some-gpu-agent.service",
        "start nvidia-persistenced",
        f"start {FM}",
        "start nvidia-dcgm",
        "docker start e1",
    ]
    # a refused reset names the processes still holding a GPU
    assert "GPU holders left: nv-hostengine python3" in done.stdout


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
    "inspect_err, waits_to_exit, expected",
    [
        ("", 2, "0 0"),
        # a transient daemon error is no proof the container is gone: keep waiting for its exit
        ("Cannot connect to the Docker daemon", 2, "0 0"),
        # a removed container ends the wait; its failed start is the validator's restoration failure
        ("Error: No such object: e1", 99, "1 1"),
        # an exporter that outlives its kill (stuck in the kernel) no longer holds the unit: it ends and names it
        # (it still reads running, so the validator's restore check cannot)
        ("", 99, "1 1"),
    ],
)
def test_the_host_unit_keeps_waiting_for_an_exporter_whose_exit_outlasts_one_wait(
    tmp_path, inspect_err, waits_to_exit, expected
):
    # the first wait times out with the exit still pending: a start then would be a no-op and nothing restarts it
    script = shlex.split(build_fabric_reset_command([], "true", ["e1"]))[-1]
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
timeout() { shift; "$@"; }
sleep() { :; }
trap 'echo "$pending $noops"' EXIT
"""
    done = subprocess.run(["/bin/sh", "-c", stubs + script], capture_output=True, text=True, timeout=30)
    assert done.returncode == gpu_reset._FABRIC_STOP_FAILED_EXIT
    assert done.stdout.strip() == expected
    stuck = not inspect_err and waits_to_exit == 99
    assert ("could not start e1: it did not exit after a kill" in done.stderr) == stuck


AGENT = "some-gpu-agent.service"


@pytest.mark.parametrize(
    "stop_fails,start_fails,exit_code,log",
    [
        ("", "", 0, [f"stop {AGENT}", f"reset -r -i {U[0]}", f"start {AGENT}", f"is-active {AGENT}"]),
        # a stop that fails may still have stopped it: it is started again, and the reset is skipped
        (AGENT, "", gpu_reset._FABRIC_STOP_FAILED_EXIT, [f"stop {AGENT}", f"start {AGENT}", f"is-active {AGENT}"]),
        ("", AGENT, gpu_reset._FABRIC_START_FAILED_EXIT, [f"stop {AGENT}", f"reset -r -i {U[0]}", f"start {AGENT}"]),
    ],
)
def test_the_host_unit_starts_a_discovered_holder_again_and_fails_when_it_stays_down(
    tmp_path, stop_fails, start_fails, exit_code, log
):
    procs = {4242: ("agent", f"0::/system.slice/{AGENT}")}
    done, got = _run_fabric_script(
        tmp_path, units=(), procs=procs, stop_fails=stop_fails, start_fails=start_fails
    )
    assert done.returncode == exit_code and got == log
    assert (f"could not start {AGENT}" in done.stderr) == bool(start_fails)


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
async def test_a_holder_the_host_unit_could_not_start_again_fails_the_reset():
    host = _b300(reset_exit=gpu_reset._FABRIC_START_FAILED_EXIT)
    report, start = await _reset(host, backend=U)
    assert not report.ok and report.error_code == GpuResetErrorCode.ResetNotSupported
    assert not report.gpus and not any(cmd.startswith("CUDA_VISIBLE_DEVICES=") for cmd in host.commands)
    start.assert_awaited_once()


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
    assert not any(cmd.startswith(("/usr/bin/docker stop", "nvidia-smi -r")) for cmd in host.commands)
    start.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_stopped_container_started_but_refused_by_the_precheck_reports_the_start():
    host = FakeHost(main_pid=0)
    report, start = await _reset(host)  # the start does not make the fake run: the precheck refuses again
    assert not report.ok and report.container_restarted and report.error_code == GpuResetErrorCode.ResetNotSupported
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
    assert not report.container_restarted
    _no_change(host, start)
