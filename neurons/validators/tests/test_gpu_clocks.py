"""DAH-4663: the renter GPU clock lock (services/gpu_clocks.py) against a fake host.

Every refusal, and the hard invariants: the GPUs touched are inside BOTH the container's own
DeviceRequests and the backend's GPUs, `-lgc`/`-rgc` never run without `-i`, `-pl` never runs, nothing
is read from inside the pod; the range is clamped to the supported clocks; the pod's delete and the
next create reset the lock; the thermal-throttle check spares a locked GPU only its low clock.
"""

import asyncio
import json
import logging
import os
import re
import time
from unittest.mock import AsyncMock

import pytest
from payload_models.payloads import GpuResetErrorCode
from services.gpu_clocks import (
    PER_RESET_TIMEOUT_SECONDS,
    build_clock_command,
    clamp_range,
    clock_locked_gpu_uuids,
    lock_pod_gpu_clocks,
    reset_leftover_clock_locks,
    reset_pod_clock_locks,
)

from services import gpu_clocks
from tests.helpers import FakeRedis, build_state
from tests.test_machine_spec_scrape_check import make_command_result

NAME = "container_pod1"
U = [f"GPU-0000000{i}-0000-4000-8000-00000000000{i}" for i in range(4)]
BUS = ["00000000:1B:00.0", "00000000:0A:00.0", "00000000:2C:00.0", "00000000:3D:00.0"]
SUPPORTED = "1980\n1965\n1410\n345\n210\n"


class FakeHost:
    """Answers the commands by what they start with; records every command."""

    def __init__(self, *, device_ids=(U[0], U[1]), count=None, mig=(), runtime="runc", inspect_exit=0,
                 lgc_exit=0, rgc_exit=0, supported=SUPPORTED, app_mem_exit=0, answered=()):
        self.commands: list[str] = []
        self.answered = answered  # rgc_exit None: the GPUs whose lines reached the host's results file before the overrun
        self.files: dict[str, str] = {}
        self.timeouts: list[int] = []
        self.app_mem_exit = app_mem_exit  # None: the host's timeout kills the -rac/-rmc batch (no line of theirs)
        request = {"Count": count or 0, "DeviceIDs": list(device_ids) or None, "Capabilities": [["gpu"]]}
        self.inspect = json.dumps({"Id": "c" * 64, "HostConfig": {"Runtime": runtime, "DeviceRequests": [request]}})
        self.gpus = "\n".join(f"{i}, {U[i]}, {BUS[i]}, {'Enabled' if i in mig else 'Disabled'}" for i in range(4))
        self.inspect_exit, self.lgc_exit, self.rgc_exit, self.supported = inspect_exit, lgc_exit, rgc_exit, supported

    async def run(self, cmd: str, *, timeout: int = 60, retryable: bool = True, **_):
        self.commands.append(cmd)
        self.timeouts.append(timeout)
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
        if cmd.startswith("cat /tmp/lium-gpu-clock-reset-"):  # the read-back after a lost answer
            return make_command_result(True, self.files.pop(cmd.split()[1], ""))
        if " -rgc " in cmd:  # the batched resets: "<uuid> <exit>" per GPU, then "<uuid> <flag> <exit>" per -rac/-rmc
            if self.rgc_exit is None:  # the whole call times out: the runner answers -1, no output
                path = re.search(r"tee -a (\S+) ", cmd).group(1)
                self.files[path] = "".join(f"{u} 0\n{u} -rac 0\n{u} -rmc 0\n" for u in self.answered)
                return make_command_result(False, "", exit_code=-1)
            lines = [f"{u} {self.rgc_exit}" for u in _reset_uuids(cmd)]
            if self.app_mem_exit is not None:
                resets = re.findall(r"nvidia-smi -i (\S+) (-rac|-rmc) ", cmd)
                lines += [f"{u} {flag} {self.app_mem_exit}" for u, flag in resets]
            return make_command_result(True, "\n".join(lines))
        raise AssertionError(f"unexpected command {cmd}")


def _assert_invariants(host: FakeHost, allowed) -> None:
    for cmd in host.commands:
        assert "docker exec" not in cmd and " -pl" not in cmd, "never inside the pod, never the power limit"
        flags = re.findall(r"\S+ (?:-lgc|-rgc|-rac|-rmc)\b", cmd.replace("echo ", "echo_"))
        changes = re.findall(r"nvidia-smi -i (GPU-\S+) (?:-lgc|-rgc|-rac|-rmc)\b", cmd)
        assert len(changes) == len([f for f in flags if not f.startswith("echo_")]), "never a clock change without -i"
        assert set(changes) <= set(allowed)


def _reset_uuids(cmd: str) -> list[str]:
    return re.findall(r"nvidia-smi -i (\S+) -rgc ", cmd)


async def _lock(host, redis=None, *, backend=(U[0], U[1]), indices=None, lo=1200, hi=1500, reset=False, **kw):
    redis = redis if redis is not None else FakeRedis()
    report = await lock_pod_gpu_clocks(
        host, redis, container_name=NAME, pod_id="pod1", miner_hotkey="hk", executor_id="ex",
        pod_gpu_uuids=list(backend),
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
    assert set(json.loads(redis.store["gpu_clock_lock_pod:pod1:hk:ex"])) == {U[0], U[1]}
    # gateway review: no lock record ever expires, so a long rental keeps the next one fail-closed
    assert redis.ttl.get(f"gpu_clock_lock:{U[0]}") is None


@pytest.mark.asyncio
async def test_no_lock_record_expires_whatever_the_path():
    """An unanswered -lgc whose reset was refused, and a refused re-lock of a confirmed lock, keep a record with no
    expiry: the GPU may still be locked, so creates stay refused until an -rgc succeeds (gateway F-80/F-81)."""
    host, redis = FakeHost(device_ids=(U[0],), lgc_exit=-1, rgc_exit=4), FakeRedis()
    await _lock(host, redis, backend=[U[0]])
    assert f"gpu_clock_lock:{U[0]}" in redis.store and redis.ttl.get(f"gpu_clock_lock:{U[0]}") is None

    host, redis = FakeHost(device_ids=(U[0],)), FakeRedis()
    await _lock(host, redis, backend=[U[0]])
    host.lgc_exit = 4
    await _lock(host, redis, backend=[U[0]])
    assert f"gpu_clock_lock:{U[0]}" in redis.store and redis.ttl.get(f"gpu_clock_lock:{U[0]}") is None


@pytest.mark.asyncio
async def test_a_sysbox_pod_is_locked_like_any_other():
    # every renter pod runs under sysbox-runc; -lgc runs on the host with -i <uuid>, whatever the runtime
    host = FakeHost(runtime="sysbox-runc")
    report, _ = await _lock(host)
    assert report.ok and [g.result for g in report.gpus] == ["locked", "locked"]
    assert f"nvidia-smi -i {U[0]} -lgc 1200,1500" in host.commands


@pytest.mark.asyncio
async def test_a_redis_failure_on_the_second_gpu_still_answers_the_first_gpus_lock():
    """Gateway review: a record write that raises must not drop the outcomes of the GPUs already locked."""
    host, redis = FakeHost(), FakeRedis()
    redis.fail_next_set_of = {f"gpu_clock_lock:{U[0]}"}  # the second target (pod GPU 1)

    report, _ = await _lock(host, redis)

    assert not report.ok and report.error_code == GpuResetErrorCode.ExecutorUnreachable
    assert [(g.uuid, g.result) for g in report.gpus] == [(U[1], "locked"), (U[0], "failed")]
    assert f"nvidia-smi -i {U[0]} -lgc 1200,1500" not in host.commands, "never -lgc without its record"
    assert json.loads(redis.store[f"gpu_clock_lock:{U[1]}"])["pod_id"] == "pod1"  # the delete still resets it


@pytest.mark.asyncio
async def test_the_range_is_clamped_to_the_supported_clocks():
    report, _ = await _lock(FakeHost(), lo=50, hi=9999, indices=[0])
    assert [(g.uuid, g.applied_min_mhz, g.applied_max_mhz) for g in report.gpus] == [(U[1], 210, 1980)]
    assert clamp_range(None, None, [210, 1980]) == (210, 1980)
    assert clamp_range(1800, 1500, [210, 1980]) == (1500, 1500)  # a min above the max locks at the max


@pytest.mark.parametrize(
    ("host", "kwargs", "code"),
    [
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
async def test_an_unanswered_lock_is_reset_and_its_record_goes_only_once_that_reset_succeeded():
    report, redis = await _lock(FakeHost(lgc_exit=-1))  # the runner's timeout or lost answer

    assert not report.ok and report.error_code == GpuResetErrorCode.ClockLockNotSupported
    assert {g.result for g in report.gpus} == {"failed"}
    assert all(g.detail.endswith("clocks reset to default") for g in report.gpus)
    assert not any(k.startswith("gpu_clock_lock:") for k in redis.store), "reset: nothing for a create to refuse on"


@pytest.mark.asyncio
@pytest.mark.parametrize("lgc_exit", [3, 4, 1], ids=["not-supported", "insufficient-permissions", "other"])
async def test_a_completed_refusal_runs_no_reset_and_never_locks_the_gpu_out(lgc_exit):
    # review: a passthrough VM refuses -lgc and -rgc alike (exit 4); a kept record would refuse every create for good
    host = FakeHost(lgc_exit=lgc_exit, rgc_exit=lgc_exit)
    report, redis = await _lock(host)

    assert not report.ok and not any(" -rgc " in c for c in host.commands)
    assert not any(k.startswith("gpu_clock_lock:") for k in redis.store)
    # no clock control here (3, 4): rentals on it go on; any other failed reset may hide a lock, so it refuses
    expected = [] if lgc_exit in (3, 4) else [U[0], U[1]]
    assert await reset_leftover_clock_locks(host, redis, [U[0], U[1]]) == expected


@pytest.mark.asyncio
async def test_an_unsupported_lock_runs_no_reset_and_leaves_the_record_it_found():
    host = FakeHost()
    _, redis = await _lock(host, indices=[0])  # an earlier lock that holds: container index 0 is U[1] here
    earlier = redis.store[f"gpu_clock_lock:{U[1]}"]
    host.lgc_exit, host.rgc_exit, host.commands = 3, 3, []  # nvidia-smi: not supported, for both

    report, redis = await _lock(host, redis)

    assert not report.ok and not any(" -rgc " in c for c in host.commands)
    assert redis.store[f"gpu_clock_lock:{U[1]}"] == earlier, "its clocks are still locked"
    assert f"gpu_clock_lock:{U[0]}" not in redis.store, "never locked: rentals on it go on"
    assert await reset_leftover_clock_locks(host, redis, [U[0]]) == []


@pytest.mark.asyncio
async def test_a_lock_whose_answer_was_lost_and_whose_reset_failed_keeps_its_record_and_refuses_the_next_create():
    # -lgc may have applied on the host though its answer timed out; the cleanup -rgc fails
    host = FakeHost(lgc_exit=-1, rgc_exit=1)  # -1: what SSHCommandRunner answers for a timeout or a lost answer
    report, redis = await _lock(host)
    assert not report.ok

    assert await reset_pod_clock_locks(host, redis, "pod1", "hk", "ex") == 0  # teardown -rgc fails as well
    assert await reset_leftover_clock_locks(host, redis, [U[0], U[1]]) == [U[0], U[1]], "the next renter is refused"


@pytest.mark.asyncio
async def test_no_supported_clocks_fails_without_running_lgc_and_leaves_no_lock_record():
    host = FakeHost(supported="", rgc_exit=3)
    report, redis = await _lock(host)
    assert not report.ok and not any(" -lgc" in c for c in host.commands)
    # nothing was changed, so nothing for a create to refuse on, even where -rgc is unsupported too (exit 3)
    assert not any(k.startswith("gpu_clock_lock:") for k in redis.store)
    assert await reset_pod_clock_locks(host, redis, "pod1", "hk", "ex") == 0
    assert await reset_leftover_clock_locks(host, redis, [U[0], U[1]]) == []


@pytest.mark.asyncio
async def test_a_create_reset_counts_every_failed_reset_while_redis_cannot_say_which_gpus_are_recorded():
    host, redis = FakeHost(rgc_exit=1), FakeRedis()
    redis.get = AsyncMock(side_effect=ConnectionError("redis down"))

    assert await reset_leftover_clock_locks(host, redis, [U[0], U[1]]) == [U[0], U[1]]


@pytest.mark.asyncio
async def test_reset_runs_rgc_per_gpu_and_clears_the_records():
    host = FakeHost()
    _, redis = await _lock(host)
    host.commands.clear()
    report, _ = await _lock(host, redis, reset=True, indices=[0])
    assert report.ok and [(g.uuid, g.result) for g in report.gpus] == [(U[1], "reset")]
    assert any(_reset_uuids(c) == [U[1]] for c in host.commands)
    assert f"gpu_clock_lock:{U[1]}" not in redis.store and f"gpu_clock_lock:{U[0]}" in redis.store
    assert json.loads(redis.store["gpu_clock_lock_pod:pod1:hk:ex"]) == [U[0]]


@pytest.mark.asyncio
async def test_a_create_resets_graphics_application_and_memory_clocks_of_every_gpu_in_one_call():
    """A host's -ac/-lmc (or a pod's -ac once the host ran -acp UNRESTRICTED) never reaches the next renter: in one SSH
    call, every GPU in parallel runs -rgc, -rac, -rmc, each under its own host-side SIGKILL timeout, and echoes each
    exit."""
    host, redis = FakeHost(), FakeRedis()
    assert await reset_leftover_clock_locks(host, redis, [U[0], U[1]]) == []
    _assert_invariants(host, [U[0], U[1]])
    (cmd,) = host.commands
    t = f"timeout -s KILL {PER_RESET_TIMEOUT_SECONDS:g}"
    f = re.search(r"tee -a (/tmp/lium-gpu-clock-reset-[0-9a-f]{16}) ", cmd).group(1)
    jobs = " ".join(
        f"( {t} nvidia-smi -i {u} -rgc >/dev/null 2>&1; echo {u} $? | tee -a {f} 2>/dev/null; "
        + "; ".join(
            f"{t} nvidia-smi -i {u} {flag} >/dev/null 2>&1; echo {u} {flag} $? | tee -a {f} 2>/dev/null"
            for flag in ("-rac", "-rmc")
        )
        + " ) &"
        for u in (U[0], U[1])
    )
    # each line goes to stdout and the host file as its reset ends; the file goes once every reset is done
    assert cmd == f"sh -c '{jobs} wait; rm -f {f}'"
    # the runner waits past one GPU's three resets, so a stalled one is killed on the host before the call is given up
    assert host.timeouts[0] > 3 * PER_RESET_TIMEOUT_SECONDS


def _fake_nvidia_smi(tmp_path, body: str) -> dict:
    fake = tmp_path / "nvidia-smi"
    fake.write_text(f"#!/bin/sh\n{body}")
    fake.chmod(0o755)
    return {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"}


@pytest.mark.asyncio
async def test_the_resets_of_an_8_gpu_node_run_in_parallel_and_answer_whatever_the_gpu_count(tmp_path):
    """Prod 25 Nov (8x H200): the -rgc runs went one GPU after another with no bound of their own, so with the
    -rac/-rmc batch the call outlasted the runner's budget and gave no answer at all. Run for real with a 1 s
    nvidia-smi: 8 GPUs answer all 24 resets in about three runs' time (one after another: 8 s of -rgc alone)."""
    env = _fake_nvidia_smi(tmp_path, "sleep 1\n")
    uuids = [f"GPU-{i:08d}-0000-0000-0000-000000000000" for i in range(8)]
    t0 = time.monotonic()
    proc = await asyncio.create_subprocess_shell(
        gpu_clocks._reset_command(uuids, None, str(tmp_path / "results")), stdout=asyncio.subprocess.PIPE, env=env
    )
    out, _ = await proc.communicate()
    elapsed = time.monotonic() - t0
    lines = sorted(out.decode().split("\n")[:-1])
    assert lines == sorted([f"{u} 0" for u in uuids] + [f"{u} {flag} 0" for u in uuids for flag in ("-rac", "-rmc")])
    assert elapsed < 6, f"{elapsed:.1f} s: the resets ran one after another"


@pytest.mark.asyncio
@pytest.mark.parametrize("results", ["results", "missing-dir/results"])
async def test_every_line_reaches_stdout_whether_or_not_the_host_file_can_be_written(tmp_path, results):
    """Review (lio#2499): a host whose /tmp cannot be written still answers every reset; no file is left either way."""
    env = _fake_nvidia_smi(tmp_path, "exit 0\n")
    path = tmp_path / results
    proc = await asyncio.create_subprocess_shell(
        gpu_clocks._reset_command([U[0], U[1]], None, str(path)), stdout=asyncio.subprocess.PIPE, env=env
    )
    out, _ = await proc.communicate()
    expected = [f"{u} 0" for u in (U[0], U[1])] + [f"{u} {flag} 0" for u in (U[0], U[1]) for flag in ("-rac", "-rmc")]
    assert sorted(out.decode().split("\n")[:-1]) == sorted(expected)
    assert proc.returncode == 0
    assert not path.exists()


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


# uutils timeout (Ubuntu 26.04, the CI image) signals only its direct child, never the process group
_CHILD_ONLY_TIMEOUT = """#!/bin/sh
shift 2; t=$1; shift
"$@" & c=$!
( sleep "$t"; kill -KILL "$c" ) >/dev/null 2>&1 &
wait "$c"
"""


@pytest.mark.asyncio
@pytest.mark.parametrize("child_only_timeout", [False, True])
@pytest.mark.parametrize("ignores_term", [False, True])
@pytest.mark.parametrize("stalls", ["-rgc", "-rac"])
async def test_a_stalled_reset_is_killed_on_the_host_and_every_other_line_still_comes_back(
    tmp_path, monkeypatch, ignores_term, child_only_timeout, stalls
):
    """Review (lio#2309): a hung reset must not hold the other results. Run for real: an nvidia-smi that never
    returns is killed by its own host-side timeout (137), every other line comes back, and the call ends near that
    timeout. Review (lio#2362): one that ignores TERM is killed too, so it never changes clocks after the handoff (it
    would touch the marker 3 s in), and none is alive when the call returns. Prod 25 Nov: a hung -rgc is bounded too."""
    marker, pids = tmp_path / "still-running", tmp_path / "pids"
    trap = 'trap "" TERM\n' if ignores_term else ""
    env = _fake_nvidia_smi(tmp_path, f'[ "$3" = {stalls} ] || exit 0\necho $$ >> {pids}\n{trap}sleep 3\ntouch {marker}\n')
    if child_only_timeout:
        (tmp_path / "timeout").write_text(_CHILD_ONLY_TIMEOUT)
        (tmp_path / "timeout").chmod(0o755)
    monkeypatch.setattr(gpu_clocks, "PER_RESET_TIMEOUT_SECONDS", 1)
    t0 = time.monotonic()
    proc = await asyncio.create_subprocess_shell(
        gpu_clocks._reset_command([U[0], U[1]], None, str(tmp_path / "results")), stdout=asyncio.subprocess.PIPE, env=env
    )
    out, _ = await asyncio.wait_for(proc.communicate(), timeout=20)
    elapsed = time.monotonic() - t0
    expected = {f"{u} {'137' if stalls == '-rgc' else '0'}" for u in (U[0], U[1])} | {
        f"{u} {flag} {'137' if flag == stalls else '0'}" for u in (U[0], U[1]) for flag in ("-rac", "-rmc")
    }
    assert set(out.decode().split("\n")[:-1]) == expected
    assert elapsed < 2.5, f"{elapsed:.1f} s: the stalled reset was not killed on the host"
    started = [int(pid) for pid in pids.read_text().split()]
    assert len(started) == 2  # the stalled reset on both GPUs
    assert [pid for pid in started if _alive(pid)] == [], "an nvidia-smi was still running when the call returned"
    await asyncio.sleep(4 - elapsed)
    assert not marker.exists(), "an nvidia-smi outlived the call"


@pytest.mark.asyncio
@pytest.mark.parametrize("app_mem_exit", [1, 6])
async def test_an_answered_failed_or_not_found_application_or_memory_reset_keeps_the_rgc_results_and_the_create(
    app_mem_exit, caplog
):
    """Review: a -rac/-rmc that answered a failure, or 6 "not found" beside an answered -rgc, never erases the -rgc
    results: the GPUs' lock records from a previous rental go and the create is not refused; a failure is logged, and
    exit 3 (not supported) is not."""
    host, redis = FakeHost(app_mem_exit=app_mem_exit), FakeRedis()
    for u in (U[0], U[1]):
        redis.store[f"gpu_clock_lock:{u}"] = json.dumps({"pod_id": "old"})
    with caplog.at_level("WARNING", logger="services.gpu_clocks"):
        assert await reset_leftover_clock_locks(host, redis, [U[0], U[1]]) == []
    assert not any(k.startswith("gpu_clock_lock:") for k in redis.store)
    assert caplog.text.count("gpu_clock_action=rac_rmc_failed") == (4 if app_mem_exit == 1 else 0)

    caplog.clear()
    host.app_mem_exit = 3
    with caplog.at_level("WARNING", logger="services.gpu_clocks"):
        assert await reset_leftover_clock_locks(host, redis, [U[0]]) == []
    assert "rac_rmc_failed" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("app_mem_exit", [None, 137], ids=["no-line", "killed"])
@pytest.mark.parametrize("rgc_exit", [0, 3, 4])
@pytest.mark.parametrize("recorded", [False, True])
async def test_an_application_or_memory_reset_that_gave_no_answer_refuses_the_create_only_on_a_recorded_gpu(
    app_mem_exit, rgc_exit, recorded, caplog
):
    """Any clock reset that gave no answer (graphics, application or memory) leaves the GPU's clocks unknown: a GPU with
    a lock record refuses the create and keeps its record; one without is logged per GPU and the create goes on (review
    6076421144, staged until a week of 8-GPU creates is read)."""
    host, redis = FakeHost(rgc_exit=rgc_exit, app_mem_exit=app_mem_exit), FakeRedis()
    if recorded:
        redis.store[f"gpu_clock_lock:{U[0]}"] = json.dumps({"pod_id": "old"})
    with caplog.at_level("WARNING", logger="services.gpu_clocks"):
        refused = await reset_leftover_clock_locks(
            host, redis, [U[0], U[1]], keep_pod_id="pod9", purpose="create", node_id="node7"
        )
    assert refused == ([U[0]] if recorded else [])
    assert (f"gpu_clock_lock:{U[0]}" in redis.store) == recorded
    logged = [r.getMessage() for r in caplog.records if "reset_no_answer_unrecorded" in r.getMessage()]
    assert len(logged) == (1 if recorded else 2)
    assert logged[-1].startswith(f"gpu_clock_action=reset_no_answer_unrecorded purpose=create node=node7 pod=pod9 gpu={U[1]} ")
    assert f"gpu_clock_action=rgc_failed node=node7 pod=pod9 gpu={U[1]} recorded=false" in caplog.text


@pytest.mark.asyncio
async def test_a_lost_answers_read_back_is_bounded_to_10_seconds():
    """Review: the read-back after a lost answer keeps the create's wait under its old budget."""
    host = FakeHost(rgc_exit=None, answered=[U[0]])
    await reset_leftover_clock_locks(host, FakeRedis(), [U[0], U[1]])
    assert host.commands[-1].startswith("cat /tmp/lium-gpu-clock-reset-") and host.timeouts[-1] == 10


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("rgc_exit", "app_mem_exit", "counts", "warned"),
    [
        (0, 0, "rgc_ok=2 rgc_failed=0 rgc_no_answer=0 app_mem_ok=4 app_mem_unsupported=0 app_mem_not_found=0", 0),
        # prod 26 Nov (8x RTX 4090 48GB): -rac/-rmc answer 6 "not found" on a GPU whose -rgc answered: no such control
        (0, 6, "rgc_ok=2 rgc_failed=0 rgc_no_answer=0 app_mem_ok=0 app_mem_unsupported=0 app_mem_not_found=4", 0),
        # a 6 beside an -rgc that also answered 6 may be the GPU itself not found: a failure
        (6, 6, "rgc_ok=0 rgc_failed=2 rgc_no_answer=0 app_mem_ok=0 app_mem_unsupported=0 app_mem_not_found=0", 4),
        # killed by the host's timeout: no verdict
        (137, 137, "rgc_ok=0 rgc_failed=0 rgc_no_answer=2 app_mem_ok=0 app_mem_unsupported=0 app_mem_not_found=0", 4),
        # prod 25 Nov (8x H200): the whole call gave no answer
        (None, None, "rgc_ok=0 rgc_failed=0 rgc_no_answer=2 app_mem_ok=0 app_mem_unsupported=0 app_mem_not_found=0", 4),
    ],
)
async def test_every_create_reset_logs_one_outcome_line(rgc_exit, app_mem_exit, counts, warned, caplog):
    """A successful reset logs a line too, so the resets of creates can be counted, not only their failures."""
    host, redis = FakeHost(rgc_exit=rgc_exit, app_mem_exit=app_mem_exit), FakeRedis()
    with caplog.at_level("INFO", logger="services.gpu_clocks"):
        await reset_leftover_clock_locks(host, redis, [U[0], U[1]], keep_pod_id="pod9", purpose="create")
    (line,) = [r.getMessage() for r in caplog.records if "reset_outcome" in r.getMessage()]
    assert line.startswith(f"gpu_clock_action=reset_outcome purpose=create pod=pod9 gpus=2 {counts} ")
    assert caplog.text.count("gpu_clock_action=rac_rmc_failed") == warned


@pytest.mark.asyncio
async def test_a_gpu_whose_rgc_was_killed_has_no_verdict_and_refuses_the_create_only_with_a_record():
    """A -rgc killed on the host gave no verdict: a recorded GPU may still hold our lowered clocks, so it refuses the
    create and keeps its record; an unrecorded one is logged only (review 6076421144)."""
    host, redis = FakeHost(rgc_exit=137), FakeRedis()
    assert await reset_leftover_clock_locks(host, redis, [U[0], U[1]]) == []
    redis.store[f"gpu_clock_lock:{U[0]}"] = json.dumps({"pod_id": "old"})
    assert await reset_leftover_clock_locks(host, redis, [U[0], U[1]]) == [U[0]]
    assert f"gpu_clock_lock:{U[0]}" in redis.store


@pytest.mark.asyncio
async def test_the_pods_delete_resets_its_locks_and_a_failed_reset_is_caught_by_the_next_create():
    host = FakeHost()
    _, redis = await _lock(host)
    host.rgc_exit = 1
    assert await reset_pod_clock_locks(host, redis, "pod1", "hk", "ex") == 0
    assert "gpu_clock_lock_pod:pod1:hk:ex" not in redis.store and f"gpu_clock_lock:{U[0]}" in redis.store

    host.rgc_exit, host.commands = 0, []
    # U[2] has no record (a lost one looks the same): reset too
    assert await reset_leftover_clock_locks(host, redis, [U[0], U[1], U[2]]) == []
    assert len(host.commands) == 1 and _reset_uuids(host.commands[0]) == [U[0], U[1], U[2]], "one call"
    assert not any(k.startswith("gpu_clock_lock:") for k in redis.store)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("record", "reset"),
    [
        (None, False),
        ({"pod_id": "pod1", "executor_id": "ex"}, True),
        ({"pod_id": "pod2", "executor_id": "ex"}, False),
        ({"pod_id": "pod1", "executor_id": "new-node"}, False),
    ],
    ids=["missing", "own", "foreign", "same-pod-on-the-node-it-moved-to"],
)
async def test_teardown_resets_only_the_gpus_whose_record_still_names_the_pod(record, reset):
    host, redis = FakeHost(), FakeRedis()
    redis.store["gpu_clock_lock_pod:pod1:hk:ex"] = json.dumps([U[0]])
    if record:
        redis.store[f"gpu_clock_lock:{U[0]}"] = json.dumps(record)

    assert await reset_pod_clock_locks(host, redis, "pod1", "hk", "ex") == int(reset)
    assert bool(host.commands) == reset and "gpu_clock_lock_pod:pod1:hk:ex" not in redis.store
    assert (f"gpu_clock_lock:{U[0]}" in redis.store) == (record is not None and not reset), "another pod's record stays"


@pytest.mark.asyncio
async def test_a_delete_retried_after_its_redis_cleanup_failed_leaves_the_next_renters_lock_alone():
    host, redis = FakeHost(), FakeRedis()
    await lock_pod_gpu_clocks(host, redis, container_name=NAME, pod_id="podA", executor_id="ex", pod_gpu_uuids=[U[0]],
                              miner_hotkey="hk", gpu_indices=None, min_mhz=1200, max_mhz=1500, reset=False)
    redis.fail_delete_of = {"gpu_clock_lock_pod:podA:hk:ex"}
    with pytest.raises(Exception):
        await reset_pod_clock_locks(host, redis, "podA", "hk", "ex")  # -rgc ran and the record went; then Redis failed
    assert "gpu_clock_lock_pod:podA:hk:ex" in redis.store and f"gpu_clock_lock:{U[0]}" not in redis.store

    assert await reset_leftover_clock_locks(host, redis, [U[0]]) == []  # B's create
    await lock_pod_gpu_clocks(host, redis, container_name=NAME, pod_id="podB", executor_id="ex", pod_gpu_uuids=[U[0]],
                              miner_hotkey="hk", gpu_indices=None, min_mhz=1200, max_mhz=1500, reset=False)
    host.commands.clear()

    assert await reset_pod_clock_locks(host, redis, "podA", "hk", "ex") == 0  # A's delete, re-sent
    assert not any(" -rgc " in c for c in host.commands), "B's clocks stay locked"
    assert json.loads(redis.store[f"gpu_clock_lock:{U[0]}"])["pod_id"] == "podB"
    assert json.loads(redis.store["gpu_clock_lock_pod:podB:hk:ex"]) == [U[0]]


@pytest.mark.asyncio
async def test_another_pods_lock_during_a_late_teardowns_reset_waits_and_keeps_its_lock_and_record():
    """Gateway review: A's late delete has read A's record and is in its -rgc when B locks the same GPU."""
    import asyncio

    host, redis = FakeHost(device_ids=(U[0],)), FakeRedis()
    await lock_pod_gpu_clocks(host, redis, container_name=NAME, pod_id="podA", executor_id="ex", pod_gpu_uuids=[U[0]],
                              miner_hotkey="hk", gpu_indices=None, min_mhz=1200, max_mhz=1500, reset=False)
    in_reset, release = asyncio.Event(), asyncio.Event()
    answer = host.run

    async def slow_rgc(cmd, **kw):
        if " -rgc " in cmd and not release.is_set():
            in_reset.set()
            await release.wait()
        return await answer(cmd, **kw)

    host.run = slow_rgc
    teardown = asyncio.create_task(reset_pod_clock_locks(host, redis, "podA", "hk", "ex"))
    await in_reset.wait()
    lock_b = asyncio.create_task(lock_pod_gpu_clocks(
        host, redis, container_name=NAME, pod_id="podB", executor_id="ex", pod_gpu_uuids=[U[0]],
        miner_hotkey="hk", gpu_indices=None, min_mhz=1200, max_mhz=1500, reset=False,
    ))
    for _ in range(20):
        await asyncio.sleep(0)
    assert json.loads(redis.store[f"gpu_clock_lock:{U[0]}"])["pod_id"] == "podA", "B waits for A's reset"
    release.set()

    assert await teardown == 1
    assert (await lock_b).ok
    rgc, lgc = (max(i for i, c in enumerate(host.commands) if flag in c) for flag in (" -rgc ", " -lgc "))
    assert rgc < lgc, "A's -rgc never undoes B's -lgc"
    assert json.loads(redis.store[f"gpu_clock_lock:{U[0]}"])["pod_id"] == "podB"
    assert json.loads(redis.store["gpu_clock_lock_pod:podB:hk:ex"]) == [U[0]]


@pytest.mark.asyncio
async def test_a_moved_pods_old_node_delete_keeps_its_new_node_lock_and_index():
    """DAH-6540 (gateway review): the old node's delete is resetting while the renter locks the pod's GPU on the
    node it moved to; the new lock keeps its index, so the new node's delete still resets it."""
    import asyncio

    host, redis = FakeHost(device_ids=(U[0], U[1])), FakeRedis()
    await lock_pod_gpu_clocks(host, redis, container_name=NAME, pod_id="pod1", executor_id="old", pod_gpu_uuids=[U[0]],
                              miner_hotkey="hk", gpu_indices=None, min_mhz=1200, max_mhz=1500, reset=False)
    in_reset, release = asyncio.Event(), asyncio.Event()
    answer = host.run

    async def slow_rgc(cmd, **kw):
        if " -rgc " in cmd and not release.is_set():
            in_reset.set()
            await release.wait()
        return await answer(cmd, **kw)

    host.run = slow_rgc
    old_delete = asyncio.create_task(reset_pod_clock_locks(host, redis, "pod1", "hk", "old"))
    await in_reset.wait()
    assert (await lock_pod_gpu_clocks(
        host, redis, container_name=NAME, pod_id="pod1", executor_id="new", pod_gpu_uuids=[U[1]],
        miner_hotkey="hk", gpu_indices=None, min_mhz=1200, max_mhz=1500, reset=False,
    )).ok
    release.set()
    assert await old_delete == 1

    assert json.loads(redis.store["gpu_clock_lock_pod:pod1:hk:new"]) == [U[1]], "the new node's index survives"
    host.commands.clear()
    assert await reset_pod_clock_locks(host, redis, "pod1", "hk", "new") == 1
    assert any(" -rgc " in c and U[1] in c for c in host.commands), "the new node's delete resets the renter's lock"


@pytest.mark.asyncio
async def test_a_lock_indexed_before_dah6540_is_still_reset_by_its_delete():
    host, redis = FakeHost(), FakeRedis()
    redis.store["gpu_clock_lock_pod:pod1"] = json.dumps([U[0], U[1]])
    redis.store[f"gpu_clock_lock:{U[0]}"] = json.dumps({"pod_id": "pod1", "executor_id": "ex"})
    redis.store[f"gpu_clock_lock:{U[1]}"] = json.dumps({"pod_id": "pod1", "executor_id": "other"})

    assert await reset_pod_clock_locks(host, redis, "pod1", "hk", "ex") == 1
    assert json.loads(redis.store["gpu_clock_lock_pod:pod1"]) == [U[1]], "the other executor's entry stays"


@pytest.mark.asyncio
async def test_a_delete_on_one_miners_executor_never_resets_another_miners_executor_with_the_same_id():
    """Review: executor ids are unique per miner only, so the index and the owner check carry the miner hotkey."""
    host, redis = FakeHost(device_ids=(U[0], U[1])), FakeRedis()
    for hotkey, uuid in (("hk", U[0]), ("other-hk", U[1])):
        await lock_pod_gpu_clocks(host, redis, container_name=NAME, pod_id="pod1", executor_id="ex", pod_gpu_uuids=[uuid],
                                  miner_hotkey=hotkey, gpu_indices=None, min_mhz=1200, max_mhz=1500, reset=False)
    host.commands.clear()

    assert await reset_pod_clock_locks(host, redis, "pod1", "hk", "ex") == 1
    assert not any(U[1] in c for c in host.commands), "the other miner's GPU is never touched"
    assert json.loads(redis.store[f"gpu_clock_lock:{U[1]}"])["miner_hotkey"] == "other-hk"
    assert json.loads(redis.store["gpu_clock_lock_pod:pod1:other-hk:ex"]) == [U[1]]


@pytest.mark.asyncio
async def test_a_legacy_indexed_delete_does_not_wait_on_the_moved_pods_new_node_gpu():
    """Review: the pre-DAH-6540 index is per pod only; the old node's delete holds only the GPUs it owns."""
    import asyncio

    from services.gpu_clocks import _holding

    host, redis = FakeHost(), FakeRedis()
    redis.store["gpu_clock_lock_pod:pod1"] = json.dumps([U[0], U[1]])
    redis.store[f"gpu_clock_lock:{U[0]}"] = json.dumps({"pod_id": "pod1", "executor_id": "old"})
    redis.store[f"gpu_clock_lock:{U[1]}"] = json.dumps({"pod_id": "pod1", "miner_hotkey": "hk", "executor_id": "new"})

    async with _holding([U[1]]):  # the new node's GPU is busy
        assert await asyncio.wait_for(reset_pod_clock_locks(host, redis, "pod1", "hk", "old"), 1) == 1
    assert json.loads(redis.store["gpu_clock_lock_pod:pod1"]) == [U[1]]


@pytest.mark.asyncio
@pytest.mark.parametrize("host_read", ["lists_only_own_gpu", "fails"])
async def test_a_legacy_record_of_another_miners_same_executor_id_is_neither_held_nor_reset(host_read):
    """Review: a legacy record has no hotkey; only a GPU on the addressed host is taken, an unread host takes none."""
    import asyncio

    from services.gpu_clocks import _holding

    host, redis = FakeHost(), FakeRedis()
    host.gpus = "0, " + U[0] + ", " + BUS[0] + ", Disabled"
    if host_read == "fails":
        answer = host.run

        async def failing_host_read(cmd, **kw):
            if cmd.startswith("nvidia-smi --query-gpu=index,uuid"):
                return make_command_result(False, "", exit_code=1)
            return await answer(cmd, **kw)

        host.run = failing_host_read
    redis.store["gpu_clock_lock_pod:pod1"] = json.dumps([U[0], U[1]])
    for uuid in (U[0], U[1]):  # U[1] is miner B's GPU behind the same executor id
        redis.store[f"gpu_clock_lock:{uuid}"] = json.dumps({"pod_id": "pod1", "executor_id": "ex"})

    async with _holding([U[1]]):  # B's GPU is busy
        reset = await asyncio.wait_for(reset_pod_clock_locks(host, redis, "pod1", "hk", "ex"), 1)

    assert reset == (1 if host_read == "lists_only_own_gpu" else 0)
    assert not any(" -rgc " in c and U[1] in c for c in host.commands)
    assert f"gpu_clock_lock:{U[1]}" in redis.store


@pytest.mark.asyncio
async def test_teardown_without_a_lock_costs_no_ssh_and_a_whole_node_create_checks_every_host_gpu():
    host = FakeHost()
    redis = FakeRedis()
    assert await reset_pod_clock_locks(host, redis, "pod1", "hk", "ex") == 0 and host.commands == []
    redis.store[f"gpu_clock_lock:{U[3]}"] = json.dumps({"pod_id": "old"})
    assert await reset_leftover_clock_locks(host, redis, []) == []
    assert len(host.commands) == 2 and _reset_uuids(host.commands[1]) == U
    assert f"gpu_clock_lock:{U[3]}" not in redis.store


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rgc_exit, unrecorded_counts",
    [(3, False), (4, False), (1, True), ("unknown", True)],
    ids=["not-supported", "insufficient-permissions", "failed", "no-answer"],
)
async def test_a_create_reset_answers_every_gpu_it_could_not_reset_unless_it_has_no_clock_control(
    caplog, rgc_exit, unrecorded_counts
):
    host, redis = FakeHost(rgc_exit=rgc_exit), FakeRedis()
    redis.store[f"gpu_clock_lock:{U[0]}"] = json.dumps({"pod_id": "old"})

    # U[1] has no record: a lock set outside Lium may still hold its clocks, unless the GPU has no clock control here
    expected = [U[0], U[1]] if unrecorded_counts else [U[0]]
    with caplog.at_level(logging.WARNING):
        assert await reset_leftover_clock_locks(host, redis, [U[0], U[1]]) == expected
    assert f"gpu_clock_lock:{U[0]}" in redis.store, "kept for the next try"
    assert f"gpu_clock_action=rgc_failed node=- pod=- gpu={U[1]} recorded=false" in caplog.text, "every unrecorded failure is logged"


@pytest.mark.asyncio
async def test_the_pods_own_edit_or_reboot_keeps_its_lock_and_another_pods_create_resets_it():
    host = FakeHost()
    _, redis = await _lock(host)
    host.commands.clear()

    assert await reset_leftover_clock_locks(host, redis, [U[0], U[1], U[2]], keep_pod_id="pod1") == []
    assert _reset_uuids(host.commands[0]) == [U[2]], "only the GPU the pod has not locked"
    assert json.loads(redis.store[f"gpu_clock_lock:{U[0]}"])["pod_id"] == "pod1"

    host.commands.clear()
    assert await reset_leftover_clock_locks(host, redis, [U[0], U[1]], keep_pod_id="pod2") == []
    assert _reset_uuids(host.commands[0]) == [U[0], U[1]]
    assert not any(k.startswith("gpu_clock_lock:") for k in redis.store)


@pytest.mark.asyncio
@pytest.mark.parametrize("recorded", [[], [U[1]]], ids=["no-records", "one-record"])
async def test_a_create_reset_whose_ssh_call_fails_refuses_on_every_recorded_gpu(recorded):
    host, redis = FakeHost(), FakeRedis()
    for uuid in recorded:
        redis.store[f"gpu_clock_lock:{uuid}"] = json.dumps({"pod_id": "old"})
    host.run = AsyncMock(side_effect=OSError("connection lost"))

    assert await reset_leftover_clock_locks(host, redis, [U[0], U[1]]) == recorded


@pytest.mark.asyncio
@pytest.mark.parametrize("recorded", [[], [U[0], U[1]]], ids=["no-records", "both-recorded"])
async def test_a_create_reset_that_overruns_keeps_the_answered_gpus_and_refuses_on_the_unanswered(recorded):
    """The call's answer is lost past the runner's budget: the GPU that answered is read back from the host's results
    file (its record goes), and only the unanswered one refuses the create, when it has a lock record."""
    host, redis = FakeHost(rgc_exit=None, answered=[U[0]]), FakeRedis()
    for uuid in recorded:
        redis.store[f"gpu_clock_lock:{uuid}"] = json.dumps({"pod_id": "old"})

    assert await reset_leftover_clock_locks(host, redis, [U[0], U[1]]) == ([U[1]] if recorded else [])
    assert host.commands[-1].startswith("cat /tmp/lium-gpu-clock-reset-") and host.files == {}, "read back, removed"
    assert f"gpu_clock_lock:{U[0]}" not in redis.store
    assert (f"gpu_clock_lock:{U[1]}" in redis.store) == bool(recorded)


@pytest.mark.asyncio
@pytest.mark.parametrize("call_exit", [0, 1])
async def test_a_partial_answer_counts_its_lines_whatever_the_calls_exit_code(call_exit):
    """Review (lio#2499): GPU A answered all three resets, GPU B none for -rmc, and no results file is left. The lines
    that came back count with exit 1 as with exit 0: only B refuses the create, and A's lock record goes."""
    host, redis = FakeHost(), FakeRedis()
    for uuid in (U[0], U[1]):
        redis.store[f"gpu_clock_lock:{uuid}"] = json.dumps({"pod_id": "old"})
    stdout = f"{U[0]} 0\n{U[0]} -rac 0\n{U[0]} -rmc 0\n{U[1]} 0\n{U[1]} -rac 0"
    fake_run = host.run

    async def run(cmd, **kw):
        if " -rgc " in cmd and not cmd.startswith("cat "):
            host.commands.append(cmd)
            return make_command_result(call_exit == 0, stdout, exit_code=call_exit)
        return await fake_run(cmd, **kw)

    host.run = run
    assert await reset_leftover_clock_locks(host, redis, [U[0], U[1]]) == [U[1]]
    assert f"gpu_clock_lock:{U[0]}" not in redis.store
    assert f"gpu_clock_lock:{U[1]}" in redis.store


@pytest.mark.asyncio
async def test_a_create_reset_that_overruns_with_no_line_goes_on_for_gpus_without_a_lock_record():
    host, redis = FakeHost(rgc_exit=None), FakeRedis()
    assert await reset_leftover_clock_locks(host, redis, [U[0], U[1]]) == []


@pytest.mark.asyncio
async def test_an_unrecorded_gpus_answered_failed_reset_still_refuses_the_create():
    """Log-only covers no answer only: an answered failure (not 3/4) refuses, record or not."""
    assert await reset_leftover_clock_locks(FakeHost(rgc_exit=6), FakeRedis(), [U[0]]) == [U[0]]


@pytest.mark.asyncio
@pytest.mark.parametrize(("still_locked", "refused"), [([], False), ([U[0]], True)])
async def test_a_create_is_refused_when_a_gpu_could_not_be_confirmed_reset(monkeypatch, still_locked, refused):
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

    return ContainerCreateRequest.model_construct(pod_id="pod1", gpu_uuids=U[:2], miner_hotkey="hk", executor_id="ex")


@pytest.mark.asyncio
async def test_a_reboot_waits_for_a_clock_lock_in_flight_so_its_clock_reset_runs_after_the_lgc(monkeypatch):
    import asyncio

    import services.docker_service as ds

    order: list[str] = []
    service, gates, started = _service_with_gated_steps(order)
    service.redis_service = None
    monkeypatch.setattr(ds, "reset_leftover_clock_locks", AsyncMock(side_effect=lambda *_a, **_k: order.append("rgc")))

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

    async def gated_rgc(*_args, **_kwargs):
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
    monkeypatch.setattr(ds, "reset_leftover_clock_locks", AsyncMock(side_effect=lambda *_a, **_k: order.append("rgc")))
    info = type("Info", (), {"address": "h", "port": 8001, "ssh_username": "u", "ssh_port": 22})()
    create_go = asyncio.Event()

    async def create():
        with ds.inflight_creates.track("pod1", ("hk", "ex")):
            await create_go.wait()
            await service._reset_leftover_clock_locks_before_create(None, _create_payload(), {})

    creating = asyncio.create_task(create())
    await asyncio.sleep(0)
    delete = asyncio.create_task(service.delete_container(_delete_payload(), info, None, "key"))
    await asyncio.sleep(0.05)  # the delete has cancelled the create and waits for it
    create_go.set()
    gates["delete"].set()

    # before the fix the delete held the lock the create's clock reset waited on: each waited for the other.
    # The cancelled create now skips that step: no -rgc on GPUs that may be rented again.
    assert await asyncio.wait_for(asyncio.gather(creating, delete), 5) == [None, "deleted"]
    assert order == ["delete"]


@pytest.mark.asyncio
async def test_a_create_reset_after_every_gpu_was_reset_passes_even_when_redis_then_fails():
    host, redis = FakeHost(), FakeRedis()
    redis.get = AsyncMock(side_effect=ConnectionError("redis down"))

    # the records are unknown, but every -rgc exited 0: nothing is still locked
    assert await reset_leftover_clock_locks(host, redis, [U[0], U[1]]) == []


@pytest.mark.asyncio
async def test_a_reboots_hung_gpu_reset_waits_for_a_gpu_operation_in_flight(monkeypatch):
    # DAH-5301 (review): a renter's Reset GPU or clock lock never overlaps the reboot's hung GPU reset
    import asyncio

    import services.docker_service as ds
    from services.hung_gpu_reset import HungGpuResult

    order: list[str] = []
    service, gates, started = _service_with_gated_steps(order)
    service.redis_service = None
    monkeypatch.setattr(ds, "SSHCommandRunner", lambda *_a, **_k: None)
    monkeypatch.setattr(
        ds, "reset_hung_pod_gpus", AsyncMock(side_effect=lambda *_a, **_k: order.append("hung") or HungGpuResult([], None, ""))
    )

    gpu = asyncio.create_task(_gpu_operation(service))
    await started["gpu"].wait()
    reboot = asyncio.create_task(service._reset_hung_gpus_before_run(None, _create_payload(), {}))
    await asyncio.sleep(0.05)
    assert order == []
    gates["gpu"].set()
    await asyncio.gather(gpu, reboot)

    assert order == ["gpu", "hung"]
    assert not ds.pod_gpu_operation_locks._locks


@pytest.mark.asyncio
async def test_a_reboot_resets_a_hung_gpu_and_returns_its_warning(monkeypatch):
    # DAH-5301: the reset and its warning to the backend
    import services.docker_service as ds
    from services.hung_gpu_reset import HungGpuResult

    service, _gates, _started = _service_with_gated_steps([])
    service.redis_service = None
    monkeypatch.setattr(ds, "SSHCommandRunner", lambda *_a, **_k: None)
    warning = ds.ContainerWarningCode.HungGpuReset
    reset = AsyncMock(return_value=HungGpuResult(["gpu-0"], warning, "reset"))
    monkeypatch.setattr(ds, "reset_hung_pod_gpus", reset)

    assert await service._reset_hung_gpus_before_run(None, _create_payload(), {}) == warning


@pytest.mark.asyncio
async def test_a_reboot_whose_lock_wait_runs_out_runs_no_hung_gpu_reset(monkeypatch):
    import services.docker_service as ds

    service, _gates, _started = _service_with_gated_steps([])
    service.redis_service = None
    reset = AsyncMock()
    monkeypatch.setattr(ds, "reset_hung_pod_gpus", reset)
    monkeypatch.setattr(ds, "POD_GPU_OPERATION_DEADLINE_SECONDS", 0)

    async with ds.pod_gpu_operation_locks.hold("pod1", ("hk", "ex"), 0):  # another GPU operation holds the pod
        assert await service._reset_hung_gpus_before_run(None, _create_payload(), {}) is None

    reset.assert_not_awaited()  # never beside the other operation


@pytest.mark.asyncio
async def test_a_reboot_its_delete_cancelled_runs_no_hung_gpu_reset(monkeypatch):
    import services.docker_service as ds

    service, _gates, _started = _service_with_gated_steps([])
    service.redis_service = None
    reset = AsyncMock()
    monkeypatch.setattr(ds, "reset_hung_pod_gpus", reset)

    with ds.inflight_creates.track("pod1", ("hk", "ex")):
        ds.inflight_creates.cancel("pod1", ("hk", "ex"))
        assert await service._reset_hung_gpus_before_run(None, _create_payload(), {}) is None

    reset.assert_not_awaited()  # the GPUs may already be the next renter's
    assert not ds.pod_gpu_operation_locks._locks


@pytest.mark.asyncio
async def test_a_create_its_delete_cancelled_runs_no_clock_reset(monkeypatch):
    import services.docker_service as ds
    from payload_models.payloads import WorkloadKind

    service, _gates, _started = _service_with_gated_steps([])
    service.redis_service = None
    reset = AsyncMock(return_value=[])
    monkeypatch.setattr(ds, "reset_leftover_clock_locks", reset)
    payload = _create_payload()
    payload.workload_kind = WorkloadKind.CUSTOMER_RENTAL

    with ds.inflight_creates.track("pod1", ("hk", "ex")):
        ds.inflight_creates.cancel("pod1", ("hk", "ex"))
        # no raise: the create's next cancel check, inside its cleanup block, removes its volume and image
        await service._reset_leftover_clock_locks_before_create(None, payload, {})

    reset.assert_not_awaited()  # the GPUs may already be the next renter's
    assert not ds.pod_gpu_operation_locks._locks


@pytest.mark.asyncio
@pytest.mark.parametrize("rgc_exit, passed, reason", [(0, True, "GPU_CLOCK_LOCK_RESET"), (4, False, "GPU_CLOCK_LOCK_STUCK")])
async def test_an_idle_nodes_stuck_lock_is_reset_or_unlists_the_node(context_factory, caplog, rgc_exit, passed, reason):
    from protocol.vc_protocol.compute_requests import RentedExecutorsResponse
    from services.task.checks.gpu_clock_lock import STUCK_LOCK_LOG_MSG, GpuClockLockCheck

    from tests.helpers import build_services

    redis, host = FakeRedis(), FakeHost(rgc_exit=rgc_exit)
    old = {"pod_id": "gone", "locked_at": 0}
    redis.store[f"gpu_clock_lock:{U[0]}"] = json.dumps(old)
    redis.store[f"gpu_clock_lock:{U[1]}"] = json.dumps({**old, "locked_at": 9e12})  # too young: maybe a fresh rental
    rows = [{"uuid": u} for u in U[:3]]
    ctx = context_factory(state=build_state(gpu_details=rows), services=build_services(redis=redis), runner=host)

    result = await GpuClockLockCheck().run(ctx)

    assert result.passed is passed and result.event.reason_code == reason
    assert [u for c in host.commands for u in _reset_uuids(c)] == [U[0]]
    assert (f"gpu_clock_lock:{U[0]}" in redis.store) is not passed and f"gpu_clock_lock:{U[1]}" in redis.store
    assert (STUCK_LOCK_LOG_MSG in caplog.text) is not passed

    # a record naming a pod the snapshot still rents is that pod's lock, never reset
    ex = ctx.executor.uuid
    rented = RentedExecutorsResponse(executors={ex: {
        "miner_hotkey": "m", "executor_ip_address": "1", "executor_ip_port": "2",
        "pods": [{"pod_id": "gone", "container_name": "c"}],
    }})
    host.commands.clear()
    live_ctx = context_factory(
        state=build_state(gpu_details=rows, rented_data=rented), services=build_services(redis=redis), runner=host
    )
    assert (await GpuClockLockCheck().run(live_ctx)).passed and host.commands == []


@pytest.mark.asyncio
async def test_a_stale_lock_replaced_by_a_new_renters_lock_before_its_reset_is_left_alone():
    # F-83: the check sampled A's old record; B's create reset it and B locked the GPU before the check's reset ran
    host, redis = FakeHost(), FakeRedis()
    sampled = json.dumps({"pod_id": "A", "locked_at": 0})
    redis.store[f"gpu_clock_lock:{U[0]}"] = b_record = json.dumps({"pod_id": "B", "locked_at": 9e12})

    assert await reset_leftover_clock_locks(host, redis, [U[0]], expected={U[0]: sampled}) == []
    assert host.commands == [] and redis.store[f"gpu_clock_lock:{U[0]}"] == b_record

    redis.store[f"gpu_clock_lock:{U[0]}"] = sampled  # still A's: reset as before
    assert await reset_leftover_clock_locks(host, redis, [U[0]], expected={U[0]: sampled}) == []
    assert _reset_uuids(host.commands[0]) == [U[0]] and f"gpu_clock_lock:{U[0]}" not in redis.store
