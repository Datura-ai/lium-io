import ast
import json
from contextlib import contextmanager
from datetime import UTC, datetime
from unittest.mock import patch

import pytest
from neurons.validators.src.protocol.vc_protocol.compute_requests import RentedExecutorsResponse
from neurons.validators.src.services.const import FILLER_CONTAINER_PREFIX
from neurons.validators.src.services.task import pipeline_factory
from neurons.validators.src.services.task.checks import gpu_fault_probe as module
from neurons.validators.src.services.task.checks.gpu_fault_probe import (
    PROBE_JSON_MARKER,
    PROBE_SECONDS,
    PROBE_SOURCE,
    PROBE_TIMEOUT_SECONDS,
    GpuFaultProbeCheck,
)
from neurons.validators.src.services.task.messages import GpuFaultProbeMessages as Msg
from neurons.validators.src.services.task.runner import SSHCommandResult

from tests.helpers import build_context_config, build_services, build_state, default_executor


@contextmanager
def probe_gate(*, check_enabled: bool = True, enforce: bool = False):
    # shadow-first like every score-zeroing gate: the check runs and logs, enforcement fails the node
    with patch("neurons.validators.src.services.task.checks.gpu_fault_probe.settings") as s:
        s.GPU_FAULT_PROBE_CHECK_ENABLED = check_enabled
        s.GPU_FAULT_PROBE_ENFORCEMENT_ENABLED = enforce
        yield s


class FakeRunner:
    """Records the command the check sends and answers with a canned SSHCommandResult."""

    def __init__(
        self,
        stdout: str = "",
        *,
        exit_code: int = 0,
        error_type: str | None = None,
        stderr: str = "",
    ):
        self.stdout = stdout
        self.exit_code = exit_code
        self.error_type = error_type
        self.stderr = stderr
        self.calls: list[dict] = []

    async def run(self, cmd, *, timeout=60, check=False, retryable=True, stdin_text=None):
        self.calls.append(
            {"cmd": cmd, "timeout": timeout, "retryable": retryable, "stdin_text": stdin_text}
        )
        now = datetime.now(UTC)
        return SSHCommandResult(
            command=cmd,
            command_id="cid",
            exit_code=self.exit_code,
            stdout=self.stdout,
            stderr=self.stderr,
            duration_ms=5340,
            started_at=now,
            finished_at=now,
            success=self.exit_code == 0,
            error_type=self.error_type,
            error_message="timed out" if self.error_type == "timeout" else None,
        )


def probe_stdout(status: str, **extra) -> str:
    report = {
        "status": status,
        "elapsed_s": 5.34,
        "faults": [],
        "devices": [
            {
                "index": 0,
                "name": "NVIDIA GeForce RTX 4090",
                "status": "ok",
                "rounds": 7,
                "work_s": 4.55,
                "elapsed_s": 5.2,
                "working_set_mb": 2048,
                "jit_ms": 79,
                "elements": 134217728,
            }
        ],
        "nvml_before": {
            "available": True,
            "gpus": [{"index": 0, "ecc_uncorrected": None, "remapped_rows": [0, 0, 0, 0]}],
        },
        "nvml_after": {
            "available": True,
            "gpus": [{"index": 0, "ecc_uncorrected": None, "remapped_rows": [0, 0, 0, 0]}],
        },
        "xid": {"available": False},
    }
    report.update(extra)
    # the library the executor's interpreter loads prints debug lines around the verdict; only the marker counts
    return "some driver noise\n" + PROBE_JSON_MARKER + " " + json.dumps(report) + "\n"


def make_ctx(context_factory, runner, *, rented_data=None):
    state = build_state(specs={"gpu": {"count": 1}}, rented_data=rented_data)
    return context_factory(
        services=build_services(), config=build_context_config(), state=state, runner=runner
    )


@pytest.mark.asyncio
async def test_disabled_by_default_runs_nothing(context_factory):
    runner = FakeRunner()
    ctx = make_ctx(context_factory, runner)

    with probe_gate(check_enabled=False):
        result = await GpuFaultProbeCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.DISABLED.reason
    assert runner.calls == []


@pytest.mark.asyncio
async def test_probe_source_goes_over_stdin_to_the_executor_interpreter(context_factory):
    runner = FakeRunner(probe_stdout("ok"))
    ctx = make_ctx(context_factory, runner)

    with probe_gate():
        result = await GpuFaultProbeCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.PROBE_OK.reason
    (call,) = runner.calls
    assert call["cmd"] == f"{default_executor().python_path} -I - --seconds {PROBE_SECONDS}"
    assert call["stdin_text"] == PROBE_SOURCE
    assert call["timeout"] == PROBE_TIMEOUT_SECONDS
    assert call["retryable"] is False
    probe = result.event.what_we_saw["probe"]
    assert probe["status"] == "ok"
    assert probe["devices"][0]["rounds"] == 7
    assert result.event.what_we_saw["duration_ms"] == 5340


@pytest.mark.asyncio
async def test_fault_in_shadow_warns_and_passes(context_factory):
    stdout = probe_stdout(
        "fault",
        faults=["gpu 0: cuStreamSynchronize -> CUDA_ERROR_ILLEGAL_ADDRESS (700)"],
        devices=[
            {
                "index": 0,
                "status": "fault",
                "error": "cuStreamSynchronize -> CUDA_ERROR_ILLEGAL_ADDRESS (700)",
            }
        ],
    )
    ctx = make_ctx(context_factory, FakeRunner(stdout, exit_code=1))

    with probe_gate(enforce=False):
        result = await GpuFaultProbeCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.PROBE_FAILED.reason
    assert result.event.severity == "warning"
    assert "NOT changed" in result.event.impact
    assert "CUDA_ERROR_ILLEGAL_ADDRESS" in result.event.what_we_saw["error"]


@pytest.mark.asyncio
async def test_fault_under_enforcement_fails_the_fatal_check(context_factory):
    stdout = probe_stdout("fault", faults=["gpu 0: gather: 1 of 134217728 elements wrong"])
    ctx = make_ctx(context_factory, FakeRunner(stdout, exit_code=1))

    with probe_gate(enforce=True):
        result = await GpuFaultProbeCheck().run(ctx)

    assert GpuFaultProbeCheck.fatal is True
    assert result.passed is False
    assert result.event.reason_code == Msg.PROBE_FAILED.reason
    assert result.event.severity == "error"
    assert result.event.what_we_saw["error"] == "gpu 0: gather: 1 of 134217728 elements wrong"


@pytest.mark.asyncio
async def test_nvml_delta_is_a_fault_even_when_every_kernel_passed(context_factory):
    stdout = probe_stdout("fault", faults=["gpu 0: uncorrected ECC errors 0 -> 2"])
    ctx = make_ctx(context_factory, FakeRunner(stdout, exit_code=1))

    with probe_gate(enforce=True):
        result = await GpuFaultProbeCheck().run(ctx)

    assert result.passed is False
    assert result.event.what_we_saw["error"] == "gpu 0: uncorrected ECC errors 0 -> 2"


@pytest.mark.asyncio
async def test_ssh_timeout_is_unknown_not_a_fault(context_factory):
    # the runner sets "timeout" for any asyncio.TimeoutError around ssh.run, a stalled channel included; a card
    # that hangs is caught inside the probe (one budget, every worker drained until it) and comes back as a
    # verdict — error in setup, fault in the kernels — so no verdict means the probe could not be measured,
    # like the sibling no-report path
    ctx = make_ctx(context_factory, FakeRunner("", exit_code=-1, error_type="timeout"))

    with probe_gate(enforce=True):
        result = await GpuFaultProbeCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.UNKNOWN.reason
    assert f"{PROBE_TIMEOUT_SECONDS}s" in result.event.what_we_saw["error"]


def test_the_ssh_cap_sits_above_the_probes_own_largest_deadline():
    # otherwise the SSH timeout fires first and a hung 8th GPU is never reported as a fault by the probe.
    # The whole worst path counts, not the drain alone: eight workers hung once their kernels run, killed and
    # still held by the driver, then the reap grace, the NVML snapshot fork hanging on the wedged card (before
    # and after), and two dmesg reads at their timeout.
    namespace = _probe_namespace(
        "WORKER_GRACE_SECONDS",
        "WORKER_GRACE_PER_GPU_SECONDS",
        "WORKER_REAP_SECONDS",
        "NVML_GRACE_SECONDS",
        "DMESG_TIMEOUT_SECONDS",
    )
    drain = (
        PROBE_SECONDS
        + namespace["WORKER_GRACE_SECONDS"]
        + namespace["WORKER_GRACE_PER_GPU_SECONDS"] * 7
    )
    nvml_fork = namespace["NVML_GRACE_SECONDS"] + 1 + 5  # poll, join(1), kill + join(5)
    tail = namespace["WORKER_REAP_SECONDS"] + 2 * nvml_fork + 2 * namespace["DMESG_TIMEOUT_SECONDS"]
    assert PROBE_TIMEOUT_SECONDS > drain + tail + 5


@pytest.mark.asyncio
async def test_executor_controlled_report_fields_are_capped_in_the_event(context_factory):
    # faults, nvml_after and xid come from the executor: a hostile or noisy report must not blow up the event
    stdout = probe_stdout(
        "fault",
        faults=[f"gpu 0: {'x' * 5000}"] * 100 + [{"not": "a string"}, 7],
        devices=[
            {
                "index": 0,
                "name": "n" * 5000,
                "status": "fault",
                "error": "e" * 5000,
                "rounds": {"nested": "dict"},
                "elapsed_s": ["l"] * 5000,
            }
        ],
        nvml_after={
            "available": True,
            "gpus": [
                {
                    "index": i,
                    "uuid": "u" * 5000,
                    "junk": "y" * 5000,
                    "remapped_rows": list(range(5000)),
                    "recovery_action": {"deep": ["x" * 5000]},
                }
                for i in range(64)
            ],
        },
        xid={
            "available": True,
            "count": 3,
            "last": ["z" * 5000] * 50,
            "other_new": ["o" * 5000] * 50,
        },
    )
    ctx = make_ctx(context_factory, FakeRunner(stdout, exit_code=1))

    with probe_gate(enforce=True):
        result = await GpuFaultProbeCheck().run(ctx)

    seen = result.event.what_we_saw
    assert len(seen["probe"]["faults"]) == module.MAX_FAULT_LINES
    assert all(len(fault) <= module.MAX_FAULT_CHARS for fault in seen["probe"]["faults"])
    device = seen["probe"]["devices"][0]
    assert len(device["name"]) == module.MAX_FAULT_CHARS
    assert len(device["error"]) == module.MAX_FAULT_CHARS
    assert device["rounds"] is None  # a dict where a number belongs is dropped
    assert len(device["elapsed_s"]) == module.MAX_FAULT_LINES
    assert len(seen["probe"]["nvml_after"]["gpus"]) == module.MAX_NVML_GPUS
    gpu = seen["probe"]["nvml_after"]["gpus"][0]
    assert "junk" not in gpu
    assert len(gpu["uuid"]) == module.MAX_FAULT_CHARS
    assert len(gpu["remapped_rows"]) == module.MAX_FAULT_LINES
    assert gpu["recovery_action"] is None
    assert len(seen["xid"]["last"]) == module.MAX_XID_LINES
    assert all(len(line) <= module.MAX_XID_CHARS for line in seen["xid"]["last"])
    assert len(seen["xid"]["other_new"]) == module.MAX_XID_LINES
    assert len(seen["error"]) <= module.MAX_FAULT_LINES * module.MAX_FAULT_CHARS
    assert len(json.dumps(seen)) < 30_000


@pytest.mark.asyncio
async def test_a_fault_report_with_non_string_faults_still_fails_the_check(context_factory):
    # the fault path joins the faults list into the error line: an int, a None or a dict in it must not raise
    # (an exception here scores the executor 0 as a pipeline error, in shadow mode too)
    stdout = probe_stdout("fault", faults=["gpu 0: Xid 79", 7, None, {"x": 1}, True])
    ctx = make_ctx(context_factory, FakeRunner(stdout, exit_code=1))

    with probe_gate(enforce=True):
        result = await GpuFaultProbeCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.PROBE_FAILED.reason
    assert result.event.what_we_saw["error"] == "gpu 0: Xid 79; 7; True"


@pytest.mark.asyncio
async def test_a_hostile_report_on_the_unknown_path_is_capped_too(context_factory):
    # status=error carries report["error"] straight into the event; a non-string faults list must not raise
    stdout = probe_stdout("error", error="E" * 5000, devices=[], faults=[7, None, {"x": 1}])
    ctx = make_ctx(context_factory, FakeRunner(stdout, exit_code=2))

    with probe_gate(enforce=True):
        result = await GpuFaultProbeCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.UNKNOWN.reason
    assert len(result.event.what_we_saw["error"]) == module.MAX_FAULT_CHARS
    assert result.event.what_we_saw["probe"]["faults"] == [7, None, None]


@pytest.mark.asyncio
async def test_probe_that_could_not_start_passes_as_unknown(context_factory):
    stdout = probe_stdout(
        "error", error="gpu 0: PTX JIT failed: CUDA_ERROR_INVALID_PTX", devices=[]
    )
    ctx = make_ctx(context_factory, FakeRunner(stdout, exit_code=2))

    with probe_gate(enforce=True):
        result = await GpuFaultProbeCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.UNKNOWN.reason
    assert "PTX JIT failed" in result.event.what_we_saw["error"]


@pytest.mark.asyncio
async def test_no_report_in_output_passes_as_unknown_with_the_tails(context_factory):
    ctx = make_ctx(
        context_factory,
        FakeRunner("Traceback ...\n", exit_code=1, stderr="ModuleNotFoundError: ctypes"),
    )

    with probe_gate(enforce=True):
        result = await GpuFaultProbeCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.UNKNOWN.reason
    assert result.event.what_we_saw["stderr_tail"] == "ModuleNotFoundError: ctypes"


@pytest.mark.asyncio
async def test_malformed_marker_line_is_unknown_not_a_crash(context_factory):
    ctx = make_ctx(context_factory, FakeRunner(PROBE_JSON_MARKER + " [1, 2\n"))

    with probe_gate(enforce=True):
        result = await GpuFaultProbeCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.UNKNOWN.reason


@pytest.mark.asyncio
async def test_active_filler_skips_the_probe(context_factory):
    rented_data = RentedExecutorsResponse(
        executors={},
        banned_guids=[],
        filler_containers_by_executor={default_executor().uuid: f"{FILLER_CONTAINER_PREFIX}active"},
    )
    runner = FakeRunner(probe_stdout("ok"))
    ctx = make_ctx(context_factory, runner, rented_data=rented_data)

    with probe_gate():
        result = await GpuFaultProbeCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.FILLER_SKIPPED.reason
    assert runner.calls == []


def test_probe_follows_the_matmul_in_both_pipelines():
    for build in (
        pipeline_factory.PipelineFactory.build_checks,
        pipeline_factory.PipelineFactory.build_dry_run_checks,
    ):
        ids = [check.check_id for check in build()]
        assert ids.index("gpu.validate.fault_probe") == ids.index("gpu.validate.capability") + 1


def test_probe_source_is_standard_library_only_and_prints_the_marker():
    # the executor image is python:3.11-slim plus the executor's own dependencies; the probe must not
    # assume anything beyond the standard library (nvidia-ml-py is imported inside a try)
    tree = ast.parse(PROBE_SOURCE)
    imported = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module.split(".")[0])
    assert imported <= {
        "__future__",
        "argparse",
        "ctypes",
        "json",
        "multiprocessing",
        "os",
        "re",
        "subprocess",
        "sys",
        "time",
    }
    assert f'JSON_MARKER = "{PROBE_JSON_MARKER}"' in PROBE_SOURCE
    assert module.PROBE_SOURCE_PATH.name == "gpu_fault_probe.py"


def _probe_namespace(*names: str) -> dict:
    # the probe is stdin-shipped source, not an importable module: lift the named functions and constants
    namespace: dict = {}
    kept = [
        node
        for node in ast.parse(PROBE_SOURCE).body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (isinstance(node, ast.Assign) and node.targets[0].id in names)
    ]
    exec(compile(ast.Module(body=kept, type_ignores=[]), "probe", "exec"), namespace)
    return namespace


def test_permutation_is_a_bijection_for_every_seed():
    # the host replays this permutation to check the pointer chase; a collision would blame the GPU
    perm = _probe_namespace("perm", "PERM_MUL", "PERM_SHIFT")["perm"]
    for log2_n in (4, 12, 20):
        n = 1 << log2_n
        for seed in (0, 12345, 0x9E37 * 6 + 12345):
            assert len({perm(i, seed & (n - 1), n - 1) for i in range(n)}) == n


def test_a_row_remapped_during_the_run_is_a_fault_even_without_a_pending_or_failed_remap():
    # NVML remapped_rows = (corrected, uncorrected, isPending, failureOccurred): the counts moving is the event
    nvml_faults = _probe_namespace("nvml_faults")["nvml_faults"]
    before = {
        "gpus": [
            {"index": 0, "ecc_uncorrected": 0, "remapped_rows": [3, 0, 0, 0], "recovery_action": 0}
        ]
    }
    quiet = {
        "gpus": [
            {"index": 0, "ecc_uncorrected": 0, "remapped_rows": [3, 0, 0, 0], "recovery_action": 0}
        ]
    }
    corrected = {
        "gpus": [
            {"index": 0, "ecc_uncorrected": 0, "remapped_rows": [4, 0, 0, 0], "recovery_action": 0}
        ]
    }
    uncorrected = {
        "gpus": [
            {"index": 0, "ecc_uncorrected": 0, "remapped_rows": [3, 1, 0, 0], "recovery_action": 0}
        ]
    }
    pending = {
        "gpus": [
            {"index": 0, "ecc_uncorrected": 0, "remapped_rows": [3, 0, 1, 0], "recovery_action": 0}
        ]
    }

    assert nvml_faults(before, quiet) == []
    assert nvml_faults(before, corrected) == [
        "gpu 0: remapped rows corrected 3 -> 4, uncorrected 0 -> 0"
    ]
    assert nvml_faults(before, uncorrected) == [
        "gpu 0: remapped rows corrected 3 -> 3, uncorrected 0 -> 1"
    ]
    assert nvml_faults(before, pending) == ["gpu 0: remapped rows pending=0 -> 1, failure=0 -> 0"]


def test_a_remap_already_pending_before_the_run_is_not_a_fault_on_every_cycle():
    # isPending stays set until the GPU is reset: a card that was pending a remap before the probe must fault
    # once (the 0 -> 1 transition), not on every cycle after it
    nvml_faults = _probe_namespace("nvml_faults")["nvml_faults"]
    already_pending = {
        "gpus": [
            {"index": 0, "ecc_uncorrected": 0, "remapped_rows": [3, 0, 1, 0], "recovery_action": 0}
        ]
    }
    still_pending = {
        "gpus": [
            {"index": 0, "ecc_uncorrected": 0, "remapped_rows": [3, 0, 1, 0], "recovery_action": 0}
        ]
    }
    now_failed = {
        "gpus": [
            {"index": 0, "ecc_uncorrected": 0, "remapped_rows": [3, 0, 1, 1], "recovery_action": 0}
        ]
    }

    assert nvml_faults(already_pending, still_pending) == []
    assert nvml_faults(already_pending, now_failed) == [
        "gpu 0: remapped rows pending=1 -> 1, failure=0 -> 1"
    ]


def test_nvml_snapshots_are_paired_by_uuid_not_by_position():
    # a card that fell off the bus mid-run must not be read as the card that took its index
    nvml_faults = _probe_namespace("nvml_faults")["nvml_faults"]
    gpu_a = {"index": 0, "uuid": "GPU-a", "ecc_uncorrected": 0, "remapped_rows": [3, 0, 0, 0], "recovery_action": 0}
    gpu_b = {"index": 1, "uuid": "GPU-b", "ecc_uncorrected": 5, "remapped_rows": [0, 0, 0, 0], "recovery_action": 0}
    before = {"gpus": [gpu_a, gpu_b]}
    # GPU-a is gone; GPU-b now sits at index 0 with its own (unchanged) counters
    after = {"gpus": [{**gpu_b, "index": 0}]}

    assert nvml_faults(before, after) == ["gpu 0 (GPU-a): missing from the NVML snapshot after the run"]

    # without UUIDs (an old binding) the position still pairs them
    assert (
        nvml_faults(
            {"gpus": [{"index": 0, "ecc_uncorrected": 0}]}, {"gpus": [{"index": 0, "ecc_uncorrected": 1}]}
        )
        == ["gpu 0: uncorrected ECC errors 0 -> 1"]
    )


def test_a_dead_nvml_child_is_an_unavailable_snapshot_not_a_crash():
    # the child closes its pipe without sending (a driver call aborted the process): recv() raises EOFError
    # and the probe must still print its verdict
    import multiprocessing

    ns = _probe_namespace("nvml_snapshot_forked", "_nvml_worker", "NVML_GRACE_SECONDS", "_exit_code")

    class DeadProcess:
        exitcode = -6

        def __init__(self, target, args):
            self._conn = args[0]

        def start(self):
            self._conn.close()

        def join(self, timeout=None):
            pass

        def is_alive(self):
            return False

        def kill(self):
            raise AssertionError("a dead child is not killed again")

    class FakeMp:
        Pipe = staticmethod(multiprocessing.Pipe)
        Process = DeadProcess

    snapshot = ns["nvml_snapshot_forked"](FakeMp)

    assert snapshot == {"available": False, "error": "NVML snapshot worker died with exit code -6"}


def _fake_worker(index, behaviour, conn):
    # stands in for the probe's _worker: same pipe protocol ({"phase": ...} then one report), scripted
    import time as _time

    conn.send({"phase": "setup"})
    if behaviour == "hang-in-setup":
        _time.sleep(60)
    conn.send({"phase": "kernels"})
    if behaviour == "hang-in-kernels":
        _time.sleep(60)
    if behaviour == "die":
        import os as _os

        _os._exit(3)
    status = "fault" if behaviour == "fault" else "ok"
    conn.send({"index": index, "status": status, "error": "Xid 79" if status == "fault" else None})
    conn.close()


def test_a_hung_worker_does_not_swallow_the_verdicts_of_the_others():
    # taiberium on #1297: one shared deadline drained worker by worker meant a hang on GPU 0 left zero
    # iterations for GPU 1..n — their reports (a real fault included) were discarded as "died in setup".
    # Every pipe is polled together until the deadline, so each worker gets the whole budget and every
    # report already sent is read.
    import multiprocessing
    import time

    namespace = _probe_namespace("drain_workers", "_exit_code", "WORKER_REAP_SECONDS")
    namespace["multiprocessing"] = multiprocessing
    namespace["time"] = time
    drain_workers = namespace["drain_workers"]
    mp = multiprocessing.get_context("fork")
    workers = []
    for index, behaviour in enumerate(["hang-in-kernels", "fault", "ok", "die", "hang-in-setup"]):
        parent_conn, child_conn = mp.Pipe(duplex=False)
        process = mp.Process(target=_fake_worker, args=(index, behaviour, child_conn))
        process.start()
        child_conn.close()
        workers.append((index, process, parent_conn))

    started = time.perf_counter()
    reports = drain_workers(workers, started + 3.0, 3.0)
    elapsed = time.perf_counter() - started

    assert [r["index"] for r in reports] == [0, 1, 2, 3, 4]
    assert reports[0]["status"] == "fault" and "hung in kernels" in reports[0]["error"]
    assert reports[1] == {"index": 1, "status": "fault", "error": "Xid 79"}
    assert reports[2]["status"] == "ok"
    assert (
        reports[3]["status"] == "fault"
        and "died in kernels with exit code 3" in reports[3]["error"]
    )
    assert reports[4]["status"] == "error" and "hung in setup" in reports[4]["error"]
    assert 3.0 <= elapsed < 10.0  # the two hangs cost one budget, not one budget each
    assert not any(process.is_alive() for _, process, _ in workers)


def test_setup_calls_in_the_probe_are_errors_not_faults():
    # an out-of-memory card, a busy card or a memlock cap must not be written up as broken hardware: every CUDA
    # call before the first kernel launch is fault=False, and a budget below the smallest working set is a ProbeError
    tree = ast.parse(PROBE_SOURCE)
    probe_device = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "probe_device"
    )
    setup_calls = {
        "cuCtxCreate_v2",
        "cuMemGetInfo_v2",
        "cuStreamCreate",
        "cuMemAlloc_v2",
        "cuMemAllocHost_v2",
    }
    seen = set()
    for node in ast.walk(probe_device):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "call"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value in setup_calls
        ):
            seen.add(node.args[0].value)
            fault = [kw for kw in node.keywords if kw.arg == "fault"]
            assert fault and fault[0].value.value is False, (
                f"{node.args[0].value} would raise CudaFault"
            )
    assert seen == setup_calls
    assert "not enough free VRAM" in PROBE_SOURCE
    assert 'on_phase("kernels")' in PROBE_SOURCE


def test_new_xid_lines_count_only_for_probed_cards_and_hardware_types():
    # dmesg is host-wide: a rented pod's illegal address on another card (Xid 31) or a sibling executor's
    # fault must not be scored against this executor; an application Xid on our own card is not hardware either
    namespace = _probe_namespace("xid_faults", "pci_key", "SOFTWARE_XIDS")
    import re as _re

    namespace["re"] = _re
    xid_faults, pci_key = namespace["xid_faults"], namespace["pci_key"]

    assert pci_key("0000:81:00.0") == "0000:81:00"
    assert pci_key("00000000:81:00.0") == "0000:81:00"
    assert pci_key("0000:81:00") == "0000:81:00"
    assert pci_key("81:00.0") == "0000:81:00"
    assert pci_key(None) is None and pci_key("garbage") is None

    before = {"available": True, "lines": ["NVRM: Xid (PCI:0000:01:00): 31, pid=1, old line"]}
    after = {
        "available": True,
        "lines": before["lines"]
        + [
            "NVRM: Xid (PCI:0000:81:00): 79, pid=0, GPU has fallen off the bus.",  # ours, hardware
            "NVRM: Xid (PCI:0000:c1:00): 79, pid=0, GPU has fallen off the bus.",  # another card
            "NVRM: Xid (PCI:0000:81:00): 31, pid=4242, name=python, Ch 00000008",  # ours, application
            "NVRM: Xid 48 with no PCI id at all",
        ],
    }
    faults, other = xid_faults(before, after, {"0000:81:00"})

    assert faults == ["new NVRM Xid on a probed GPU: " + after["lines"][1]]
    assert other == after["lines"][2:]
    assert xid_faults({"available": False}, after, {"0000:81:00"}) == ([], [])
    assert xid_faults(before, before, {"0000:81:00"}) == ([], [])


def test_nvml_snapshot_runs_in_a_fork_with_a_deadline():
    # nvmlInit blocks on a card that wedged the driver: the snapshot must come back (as unavailable) in time,
    # and the parent must not be left with a live child
    import multiprocessing
    import time

    namespace = _probe_namespace(
        "nvml_snapshot_forked", "_nvml_worker", "_quiet_child", "NVML_GRACE_SECONDS"
    )
    namespace["multiprocessing"] = multiprocessing
    namespace["time"] = time
    import os as _os

    namespace["os"] = _os
    namespace["NVML_GRACE_SECONDS"] = 1
    namespace["nvml_snapshot"] = lambda: time.sleep(30)

    started = time.perf_counter()
    snapshot = namespace["nvml_snapshot_forked"](multiprocessing.get_context("fork"))

    assert snapshot["available"] is False and "hung" in snapshot["error"]
    assert time.perf_counter() - started < 10
    assert not multiprocessing.active_children()

    namespace["nvml_snapshot"] = lambda: {"available": True, "gpus": [{"index": 0}]}
    assert namespace["nvml_snapshot_forked"](multiprocessing.get_context("fork")) == {
        "available": True,
        "gpus": [{"index": 0}],
    }


def test_the_verdict_reaches_stdout_even_with_a_child_the_driver_still_holds(tmp_path):
    # the probe prints into a block-buffered pipe and multiprocessing joins live children at exit without a
    # timeout: a SIGKILLed worker the driver has not released would keep the verdict from the validator.
    # Run a stand-in with the probe's own exit sequence: a child that ignores SIGTERM and sleeps, then main()
    # prints and leaves via os._exit — the marker must arrive within seconds and the pipe must close.
    import subprocess
    import sys as _sys
    import textwrap

    tree = ast.parse(PROBE_SOURCE)
    main_guard = next(
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and getattr(getattr(node.test, "left", None), "id", None) == "__name__"
    )
    exit_sequence = ast.get_source_segment(PROBE_SOURCE, main_guard)
    assert "os._exit(code)" in exit_sequence and "sys.stdout.flush()" in exit_sequence

    quiet_child = ast.get_source_segment(
        PROBE_SOURCE,
        next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_quiet_child"),
    )
    script = textwrap.dedent(
        """
        import multiprocessing, os, signal, sys, time
        JSON_MARKER = "GPU_FAULT_PROBE_JSON:"
        %s

        def _stuck(conn):
            _quiet_child()
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            conn.send({"phase": "kernels"})
            time.sleep(20)

        def main(argv):
            mp = multiprocessing.get_context("fork")
            parent, child = mp.Pipe(duplex=False)
            mp.Process(target=_stuck, args=(child,)).start()
            child.close()
            parent.recv()
            print(JSON_MARKER, '{"status": "fault"}')
            return 1

        %s
        """
    ) % (quiet_child, exit_sequence)
    started = __import__("time").perf_counter()
    run = subprocess.run(
        [_sys.executable, "-I", "-"], input=script, capture_output=True, text=True, timeout=20
    )
    assert run.returncode == 1
    assert run.stdout.strip() == 'GPU_FAULT_PROBE_JSON: {"status": "fault"}'
    assert __import__("time").perf_counter() - started < 15
