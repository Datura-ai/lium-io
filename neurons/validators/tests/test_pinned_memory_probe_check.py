import ast
import json
import subprocess
import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from unittest.mock import patch

import pytest
from neurons.validators.src.core.config import settings
from neurons.validators.src.protocol.vc_protocol.compute_requests import RentedExecutorsResponse
from neurons.validators.src.services.const import FILLER_CONTAINER_PREFIX
from neurons.validators.src.services.task import pipeline_factory
from neurons.validators.src.services.task.checks import pinned_memory_probe as module
from neurons.validators.src.services.task.checks.pinned_memory_probe import (
    PROBE_JSON_MARKER,
    PROBE_SOURCE,
    PROBE_TIMEOUT_SECONDS,
    PinnedMemoryProbeCheck,
)
from neurons.validators.src.services.task.messages import PinnedMemoryProbeMessages as Msg
from neurons.validators.src.services.task.runner import SSHCommandResult

from tests.helpers import build_context_config, build_services, build_state, default_executor


@contextmanager
def probe_gate(*, check_enabled: bool = True):
    with patch("neurons.validators.src.services.task.checks.pinned_memory_probe.settings") as s:
        s.PINNED_MEMORY_PROBE_CHECK_ENABLED = check_enabled
        yield s


class FakeRunner:
    """Records the command the check sends and answers with a canned SSHCommandResult."""

    def __init__(self, stdout: str = "", *, exit_code: int = 0, error_type: str | None = None):
        self.stdout = stdout
        self.exit_code = exit_code
        self.error_type = error_type
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
            stderr="",
            duration_ms=812,
            started_at=now,
            finished_at=now,
            success=self.exit_code == 0,
            error_type=self.error_type,
            error_message="timed out" if self.error_type == "timeout" else None,
        )


H200_TDX_GUEST = {
    "dev_tdx_guest": True,
    "cpuinfo_tdx_guest": True,
    "gpu_cc_mode": "CC status: OFF",
}


def probe_stdout(status: str, **extra) -> str:
    report = {
        "status": status,
        "elapsed_s": 0.81,
        "faults": [],
        "devices": [
            {
                "index": 0,
                "name": "NVIDIA H200",
                "status": "ok",
                "size_mb": 64,
                "copy_ms": 9.4,
            }
        ],
        "tee": {
            "dev_tdx_guest": False,
            "cpuinfo_tdx_guest": False,
            "gpu_cc_mode": "CC status: OFF",
        },
    }
    report.update(extra)
    return "driver noise\n" + PROBE_JSON_MARKER + " " + json.dumps(report) + "\n"


def make_ctx(context_factory, runner):
    state = build_state(specs={"gpu": {"count": 1}})
    return context_factory(
        services=build_services(), config=build_context_config(), state=state, runner=runner
    )


def test_on_and_enforcing_by_default():
    assert type(settings).model_fields["PINNED_MEMORY_PROBE_CHECK_ENABLED"].default is True
    assert "PINNED_MEMORY_PROBE_ENFORCEMENT_ENABLED" not in type(settings).model_fields
    assert PinnedMemoryProbeCheck.fatal is True


@pytest.mark.asyncio
async def test_disabled_runs_nothing(context_factory):
    runner = FakeRunner()
    with probe_gate(check_enabled=False):
        result = await PinnedMemoryProbeCheck().run(make_ctx(context_factory, runner))

    assert result.passed is True
    assert result.event.reason_code == Msg.DISABLED.reason
    assert runner.calls == []


@pytest.mark.asyncio
async def test_probe_source_goes_over_stdin_and_a_good_copy_passes(context_factory):
    runner = FakeRunner(probe_stdout("ok"))
    with probe_gate():
        result = await PinnedMemoryProbeCheck().run(make_ctx(context_factory, runner))

    assert result.passed is True
    assert result.event.reason_code == "PINNED_MEMORY_PROBE_OK"
    (call,) = runner.calls
    assert call["cmd"] == f"{default_executor().python_path} -I -"
    assert call["stdin_text"] == PROBE_SOURCE
    assert call["timeout"] == PROBE_TIMEOUT_SECONDS
    assert call["retryable"] is False
    assert result.event.what_we_saw["probe"]["devices"][0]["copy_ms"] == 9.4


@pytest.mark.asyncio
async def test_a_tdx_guest_whose_gpu_cannot_see_pinned_memory_fails_the_node(context_factory):
    fault = (
        "gpu 3: pinned H2D/D2H round trip: device buffer still the sentinel (host-to-device copy read "
        "nothing); 64 of 64 MB chunks differ, first at byte offsets [0, 1048576, 2097152, 3145728]"
    )
    stdout = probe_stdout(
        "fault",
        faults=[fault],
        devices=[{"index": 3, "name": "NVIDIA H200", "status": "fault", "error": fault[7:]}],
        tee=H200_TDX_GUEST,
    )
    with probe_gate():
        result = await PinnedMemoryProbeCheck().run(
            make_ctx(context_factory, FakeRunner(stdout, exit_code=1))
        )

    assert result.passed is False
    assert result.event.reason_code == "PINNED_MEMORY_PROBE_FAILED"
    assert result.event.severity == "error"
    assert result.event.what_we_saw["error"] == fault
    # the TDX / CC evidence travels with the verdict, for the provider and for support
    assert result.event.what_we_saw["probe"]["tee"] == H200_TDX_GUEST
    assert "conf-compute" in result.event.remediation


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    [
        "gpu 0: cuStreamSynchronize -> CUDA_ERROR_ILLEGAL_ADDRESS (700)",
        "cuInit -> CUDA_ERROR_UNKNOWN (999)",  # a broken-GPU code is a fault too
    ],
    ids=["copy-fault", "cuda-broken"],
)
async def test_a_fault_fails_the_node(context_factory, fault):
    stdout = probe_stdout("fault", faults=[fault])
    with probe_gate():
        result = await PinnedMemoryProbeCheck().run(
            make_ctx(context_factory, FakeRunner(stdout, exit_code=1))
        )

    assert result.passed is False
    assert result.event.reason_code == "PINNED_MEMORY_PROBE_FAILED"
    assert result.event.severity == "error"
    assert result.event.what_we_saw["error"] == fault


@pytest.mark.parametrize(
    "runner",
    [
        FakeRunner("", error_type="timeout"),
        FakeRunner("python: command not found", exit_code=127),
        FakeRunner(PROBE_JSON_MARKER + " {not json"),
        FakeRunner(
            probe_stdout("error", error="gpu 0: cuMemAllocHost_v2 -> CUDA_ERROR_OUT_OF_MEMORY (2)"),
            exit_code=2,
        ),
    ],
    ids=["ssh-timeout", "no-report", "malformed", "could-not-start"],
)
@pytest.mark.asyncio
async def test_a_probe_that_could_not_measure_is_unknown_never_a_fault(context_factory, runner):
    with probe_gate():
        result = await PinnedMemoryProbeCheck().run(make_ctx(context_factory, runner))

    assert result.passed is True
    assert result.event.reason_code == "PINNED_MEMORY_PROBE_UNKNOWN"
    assert result.event.what_we_saw["error"]


@pytest.mark.asyncio
async def test_executor_controlled_report_fields_are_capped(context_factory):
    stdout = probe_stdout(
        "fault",
        faults=["x" * 5000] * 100,
        devices=[
            {"index": i, "status": "fault", "error": "y" * 5000, "junk": "z"} for i in range(40)
        ],
        tee={"gpu_cc_mode": "w" * 5000, "extra": {"nested": 1}},
    )
    with probe_gate():
        result = await PinnedMemoryProbeCheck().run(
            make_ctx(context_factory, FakeRunner(stdout, exit_code=1))
        )

    probe = result.event.what_we_saw["probe"]
    assert len(probe["devices"]) == module.MAX_GPUS
    assert all(
        "junk" not in d and len(d["error"]) == module.MAX_FAULT_CHARS for d in probe["devices"]
    )
    assert len(probe["faults"]) == module.MAX_FAULT_LINES
    assert probe["tee"] == {"gpu_cc_mode": "w" * module.MAX_FAULT_CHARS}


@pytest.mark.asyncio
async def test_active_filler_skips_the_probe(context_factory):
    filler = f"{FILLER_CONTAINER_PREFIX}ac0f616e-9d3e-4605-9c42-df2adc9f4b55"
    rented_data = RentedExecutorsResponse(
        executors={}, filler_containers_by_executor={default_executor().uuid: filler}
    )
    runner = FakeRunner(probe_stdout("fault", faults=["gpu 4: worker hung during copy"]))
    state = build_state(specs={"gpu": {"count": 8}}, rented_data=rented_data)
    ctx = context_factory(services=build_services(), state=state, runner=runner)

    with probe_gate():
        result = await PinnedMemoryProbeCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.FILLER_SKIPPED.reason
    assert result.event.what_we_saw["filler_container"] == filler
    assert runner.calls == []


def hung_copy_ctx(context_factory, *, faults: list[str], fresh: RentedExecutorsResponse | None):
    # the cycle snapshot knows no workload on this node: the filler started after it was taken
    services = build_services()
    services.backend.get_rented_executors_now.return_value = fresh
    state = build_state(specs={"gpu": {"count": 8}}, rented_data=RentedExecutorsResponse(executors={}))
    runner = FakeRunner(probe_stdout("fault", faults=faults), exit_code=1)
    return context_factory(services=services, state=state, runner=runner), services.backend


def backend_with_filler() -> RentedExecutorsResponse:
    return RentedExecutorsResponse(
        executors={},
        all_filler_containers_by_executor={default_executor().uuid: ["filler_run-1"]},
    )


@pytest.mark.asyncio
async def test_a_hung_copy_beside_a_filler_started_mid_cycle_is_not_a_fault(context_factory):
    fresh = backend_with_filler()
    ctx, backend = hung_copy_ctx(
        context_factory, faults=["gpu 4: worker hung during copy"], fresh=fresh
    )

    with probe_gate():
        result = await PinnedMemoryProbeCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.WORKLOAD_SKIPPED.reason
    assert result.event.what_we_saw["workload"] == "filler"
    assert result.event.what_we_saw["containers"] == ["filler_run-1"]
    backend.get_rented_executors_now.assert_awaited_once()
    assert result.updates["state"].rented_data is fresh


@pytest.mark.asyncio
async def test_a_hung_copy_with_no_lium_workload_still_fails_the_node(context_factory):
    ctx, backend = hung_copy_ctx(
        context_factory, faults=["gpu 4: worker hung during copy"], fresh=None
    )

    with probe_gate():
        result = await PinnedMemoryProbeCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.PROBE_FAILED.reason
    backend.get_rented_executors_now.assert_awaited_once()


@pytest.mark.asyncio
async def test_wrong_data_is_a_fault_even_beside_a_live_filler(context_factory):
    faults = ["gpu 4: worker hung during copy", "gpu 0: 1 of 64 MB chunks differ"]
    ctx, backend = hung_copy_ctx(context_factory, faults=faults, fresh=backend_with_filler())

    with probe_gate():
        result = await PinnedMemoryProbeCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.PROBE_FAILED.reason
    backend.get_rented_executors_now.assert_not_awaited()


def test_probe_runs_beside_the_kernel_fault_probe_in_every_pipeline():
    for build in (
        pipeline_factory.PipelineFactory.build_checks,
        pipeline_factory.PipelineFactory.build_dry_run_checks,
    ):
        ids = [check.check_id for check in build()]
        assert ids.index("gpu.validate.pinned_memory") == ids.index("gpu.validate.fault_probe") + 1


def test_probe_source_is_standard_library_only():
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


def test_the_probe_on_a_host_without_libcuda_reports_an_error_not_a_fault():
    # the real script, end to end, on the CI runner (no GPU): it must print one marker line and exit 2
    run = subprocess.run(
        [sys.executable, "-I", "-"], input=PROBE_SOURCE, capture_output=True, text=True, timeout=60
    )
    (line,) = [line for line in run.stdout.splitlines() if line.startswith(PROBE_JSON_MARKER)]
    report = json.loads(line[len(PROBE_JSON_MARKER) :])
    if report["status"] == "ok":
        pytest.skip("this host has working GPUs")
    assert run.returncode == 2
    assert report["status"] == "error"
    assert report["faults"] == []
    assert "dev_tdx_guest" in report["tee"]


def _load_probe_symbols(*names: str) -> dict[str, object]:
    namespace: dict = {"ctypes": __import__("ctypes")}
    kept = [
        node
        for node in ast.parse(PROBE_SOURCE).body
        if (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names)
        or (isinstance(node, ast.Assign) and node.targets[0].id in names)
    ]
    exec(compile(ast.Module(body=kept, type_ignores=[]), "probe", "exec"), namespace)
    return namespace


def test_a_driver_error_naming_a_broken_gpu_is_a_fault_even_in_setup():
    # a listed card whose cuInit answers CUDA_ERROR_UNKNOWN cannot run any renter's CUDA
    probe = _load_probe_symbols("Cuda", "ProbeError", "CopyFault", "BROKEN_GPU_CODES")
    cuda = object.__new__(probe["Cuda"])
    codes = iter([999, 2])
    cuda.lib = type("Lib", (), {"cuInit": staticmethod(lambda flags: next(codes))})()
    cuda.error_name = lambda code: f"CUDA_ERROR_{code}"

    with pytest.raises(probe["CopyFault"], match="cuInit -> CUDA_ERROR_999"):
        cuda.call("cuInit", 0, raise_as_fault=False, broken_gpu_is_fault=True)
    # out of memory in setup is still "could not measure"
    with pytest.raises(probe["ProbeError"]):
        cuda.call("cuInit", 0, raise_as_fault=False, broken_gpu_is_fault=True)
    # an allocation answering a broken-GPU code is not a driver-start fault
    codes = iter([999])
    with pytest.raises(probe["ProbeError"]):
        cuda.call("cuInit", 0, raise_as_fault=False)


def test_device_enumeration_that_reports_a_broken_gpu_raises_a_fault():
    probe = _load_probe_symbols(
        "_device_count", "_reap", "CopyFault", "ProbeError", "COUNT_BUDGET_SECONDS"
    )
    probe["_count_worker"] = None  # the fake process never runs its target

    class Conn:
        def poll(self, timeout):
            return True

        def recv(self):
            return {"fault": "cuInit -> CUDA_ERROR_UNKNOWN (999)"}

        def close(self):
            pass

    class Process:
        def __init__(self, target, args):
            pass

        def start(self):
            pass

        def join(self, timeout):
            pass

        def is_alive(self):
            return False

    mp = type("Mp", (), {"Pipe": staticmethod(lambda duplex: (Conn(), Conn())), "Process": Process})

    with pytest.raises(probe["CopyFault"], match="CUDA_ERROR_UNKNOWN"):
        probe["_device_count"](mp)


def test_mismatches_counts_chunks_and_finds_the_first_wrong_byte():
    namespace: dict = {}
    kept = [
        node
        for node in ast.parse(PROBE_SOURCE).body
        if (isinstance(node, ast.FunctionDef) and node.name == "_mismatches")
        or (isinstance(node, ast.Assign) and node.targets[0].id in {"MB", "MAX_MISMATCH_OFFSETS"})
    ]
    exec(compile(ast.Module(body=kept, type_ignores=[]), "probe", "exec"), namespace)
    mb = namespace["MB"]
    want = bytes(3 * mb)
    got = bytearray(want)
    got[mb + 7] = 1
    got[2 * mb] = 1
    assert namespace["_mismatches"](want, bytes(got)) == (2, [mb + 7, 2 * mb])
    assert namespace["_mismatches"](want, want) == (0, [])


def test_a_verdict_already_queued_is_read_after_the_deadline():
    # one card hung in setup used the whole budget; the next card's fault is in its pipe and must not become "error"
    namespace: dict = {"time": __import__("time")}
    kept = [
        node
        for node in ast.parse(PROBE_SOURCE).body
        if isinstance(node, ast.FunctionDef) and node.name == "drain_workers"
    ]
    exec(compile(ast.Module(body=kept, type_ignores=[]), "probe", "exec"), namespace)
    namespace["_reap"] = lambda process: None

    class Pipe:
        def __init__(self, messages):
            self.messages = list(messages)

        def poll(self, timeout):
            return bool(self.messages)

        def recv(self):
            return self.messages.pop(0)

    class Process:
        def is_alive(self):
            return True

    fault = {"index": 1, "status": "fault", "error": "wrong data"}
    workers = [(0, Process(), Pipe([])), (1, Process(), Pipe([{"phase": "copy"}, fault]))]

    reports = namespace["drain_workers"](workers, deadline=0.0)

    assert reports[0]["status"] == "error"
    assert reports[1] == fault
