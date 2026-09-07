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
    # that hangs is caught by the probe's own per-worker deadline and comes back as a verdict, so no verdict
    # means the probe could not be measured, like the sibling no-report path
    ctx = make_ctx(context_factory, FakeRunner("", exit_code=-1, error_type="timeout"))

    with probe_gate(enforce=True):
        result = await GpuFaultProbeCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.UNKNOWN.reason
    assert f"{PROBE_TIMEOUT_SECONDS}s" in result.event.what_we_saw["error"]


def test_the_ssh_cap_sits_above_the_probes_own_largest_deadline():
    # otherwise the SSH timeout fires first and a hung 8th GPU is never reported as a fault by the probe
    namespace = _probe_namespace("WORKER_GRACE_SECONDS", "WORKER_GRACE_PER_GPU_SECONDS")
    largest = (
        PROBE_SECONDS
        + namespace["WORKER_GRACE_SECONDS"]
        + namespace["WORKER_GRACE_PER_GPU_SECONDS"] * 7
    )
    assert PROBE_TIMEOUT_SECONDS > largest + 10


@pytest.mark.asyncio
async def test_executor_controlled_report_fields_are_capped_in_the_event(context_factory):
    # faults, nvml_after and xid come from the executor: a hostile or noisy report must not blow up the event
    stdout = probe_stdout(
        "fault",
        faults=[f"gpu 0: {'x' * 5000}"] * 100,
        nvml_after={
            "available": True,
            "gpus": [{"index": i, "uuid": "u" * 5000, "junk": "y" * 5000} for i in range(64)],
        },
        xid={"available": True, "count": 3, "last": ["z" * 5000] * 50},
    )
    ctx = make_ctx(context_factory, FakeRunner(stdout, exit_code=1))

    with probe_gate(enforce=True):
        result = await GpuFaultProbeCheck().run(ctx)

    seen = result.event.what_we_saw
    assert len(seen["probe"]["faults"]) == module.MAX_FAULT_LINES
    assert all(len(fault) <= module.MAX_FAULT_CHARS for fault in seen["probe"]["faults"])
    assert len(seen["probe"]["nvml_after"]["gpus"]) == module.MAX_NVML_GPUS
    assert "junk" not in seen["probe"]["nvml_after"]["gpus"][0]
    assert len(seen["probe"]["nvml_after"]["gpus"][0]["uuid"]) == module.MAX_FAULT_CHARS
    assert len(seen["xid"]["last"]) == module.MAX_XID_LINES
    assert all(len(line) <= module.MAX_XID_CHARS for line in seen["xid"]["last"])
    assert len(seen["error"]) <= module.MAX_FAULT_LINES * module.MAX_FAULT_CHARS
    assert len(json.dumps(seen)) < 30_000


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
    assert nvml_faults(before, pending) == ["gpu 0: remapped rows pending=1 failure=0"]


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
