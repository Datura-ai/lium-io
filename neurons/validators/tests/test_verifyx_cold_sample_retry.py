"""DAH-2959: a never-measured executor's first VerifyX download sample decides its first cycle alone
(the EMA bootstraps from it). The check takes one more sample inside the same task and gates on the
better one."""

import asyncio
import dataclasses
import os
import stat
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from redis.exceptions import TimeoutError as RedisTimeoutError

from neurons.validators.src.services.task.checks import verifyx as verifyx_check
from neurons.validators.src.services.task.checks.verifyx import VerifyXCheck, VerifyXColdRetryCheck
from neurons.validators.src.services.task.messages import VerifyXMessages as Msg
from neurons.validators.src.services.task.pipeline import ParallelStage
from neurons.validators.src.services.task.pipeline_factory import PipelineFactory
from neurons.validators.src.services.redis_service import (
    CREATE_STARTED_AT_TTL_SECONDS,
    PENDING_PODS_PREFIX,
    VERIFYX_STALE_RUNS_TTL_SECONDS,
    RedisService,
    create_started_at_key,
    verifyx_stale_runs_key,
)
from neurons.validators.src.services import verifyx_validation_service
from neurons.validators.src.services.verifyx_validation_service import (
    VERIFYX_COMMAND_TIMEOUT_SECONDS,
    VerifyXValidationService,
)
from protocol.vc_protocol.compute_requests import NetworkEMA, RentedExecutorsResponse

from tests.helpers import FakeRedis, build_context_config, build_services, build_state
from tests.test_verifyx_check import MockVerifyXResponse

EXECUTOR = "executor-123"  # tests.helpers.default_executor().uuid


def _probe(download: float | None, *, success: bool = True, upload: float | None = 40.0) -> MockVerifyXResponse:
    network = {} if download is None else {"download_speed": download, "upload_speed": upload}
    return MockVerifyXResponse(
        # a disk reading that clears the floor, as an adopted retry's is judged again (DiskSpaceCheck)
        data={
            "success": success,
            "ram": {"total": 64},
            "hard_disk": {"total": 1000, "utilization": 50.0, "free": 500 * 1024 * 1024},
            "network": network,
        },
    )


class _ProbeSequence:
    """Returns the queued responses in order and counts the calls."""

    def __init__(self, *responses: MockVerifyXResponse):
        self._responses = list(responses)
        self.calls = 0

    async def validate_verifyx_and_process_job(
        self, *, shell, executor_info, default_extra, machine_spec, run_dir=None
    ):
        self.calls += 1
        self.run_dir = run_dir
        return self._responses.pop(0)


def _never_measured() -> RentedExecutorsResponse:
    return RentedExecutorsResponse(executors={}, banned_guids=[], network_ema={})


def _known(download: float) -> RentedExecutorsResponse:
    return RentedExecutorsResponse(
        executors={},
        banned_guids=[],
        network_ema={EXECUTOR: NetworkEMA(ema_verifyx_download_speed=download, ema_verifyx_upload_speed=50.0)},
    )


def _ctx(context_factory, service, rented_data):
    return context_factory(
        services=build_services(verifyx=service),
        config=build_context_config(),
        state=build_state(specs={"gpu": {"count": 1}}, rented_data=rented_data),
    )


def _idle_now(services, fresh=None, creating=False):
    """The backend's rental list now (no pod, no filler unless given) and no rental being created."""
    services.backend.get_rented_executors_now = AsyncMock(return_value=fresh or _never_measured())
    services.redis.renting_in_progress = AsyncMock(return_value=creating)


async def _run(ctx, idle=True):
    """VerifyXCheck, then (as the pipeline does, last before scoring) VerifyXColdRetryCheck, on a
    node still idle then unless `idle` is False (the test set the rental state itself)."""
    if idle:
        _idle_now(ctx.services)
    first = await VerifyXCheck().run(ctx)
    if not first.passed:
        return first
    ctx = ctx.model_copy(update=first.updates)
    if ctx.state.verifyx_cold_retry is None:
        return first
    return await VerifyXColdRetryCheck().run(ctx)


@pytest.mark.asyncio
async def test_cold_first_sample_is_re_measured_and_the_better_one_seeds_the_ema(context_factory):
    # b3e4ca69, 4 Sep: 59.9 Mbps on the first sample, ~760 Mbps once the link was free.
    service = _ProbeSequence(_probe(59.9, upload=20.0), _probe(760.0, upload=110.0))

    result = await _run(_ctx(context_factory, service, _never_measured()))

    assert service.calls == 2
    assert result.passed is True
    assert result.event.reason_code == Msg.VERIFY_SUCCESS.reason
    network = result.updates["state"].specs["network"]
    assert network["verifyx_download_speed"] == pytest.approx(760.0)
    assert network["ema_verifyx_download_speed"] == pytest.approx(760.0)
    assert network["verifyx_upload_speed"] == pytest.approx(110.0)
    assert result.event.what_we_saw["cold_sample_retry"] == {
        "first_download_speed_mbps": 59.9,
        "retry_download_speed_mbps": 760.0,
        "used": "retry",
        "timed_out": False,
    }


@pytest.mark.asyncio
async def test_two_slow_samples_still_fail_the_unchanged_gate(context_factory):
    service = _ProbeSequence(_probe(40.0), _probe(88.0))

    result = await _run(_ctx(context_factory, service, _never_measured()))

    assert service.calls == 2
    assert result.passed is False
    assert result.event.reason_code == Msg.VERIFY_FAILED_NETWORK_SPEED_TOO_SLOW.reason
    # The better of the two is what the node is judged on and what seeds the EMA.
    assert result.updates["state"].specs["network"]["ema_verifyx_download_speed"] == pytest.approx(88.0)
    assert result.event.what_we_saw["cold_sample_retry"]["used"] == "retry"


@pytest.mark.asyncio
async def test_failed_network_probe_on_first_sample_is_retried(context_factory):
    # download_speed None (probe timed out) would bootstrap the EMA to 0.0.
    service = _ProbeSequence(_probe(None), _probe(300.0))

    result = await _run(_ctx(context_factory, service, _never_measured()))

    assert service.calls == 2
    assert result.passed is True
    assert result.updates["state"].specs["network"]["ema_verifyx_download_speed"] == pytest.approx(300.0)
    assert result.event.what_we_saw["cold_sample_retry"]["first_download_speed_mbps"] is None


@pytest.mark.asyncio
async def test_a_worse_retry_keeps_the_first_sample(context_factory):
    service = _ProbeSequence(_probe(70.0), _probe(30.0))

    result = await _run(_ctx(context_factory, service, _never_measured()))

    assert service.calls == 2
    assert result.passed is False
    assert result.updates["state"].specs["network"]["ema_verifyx_download_speed"] == pytest.approx(70.0)
    assert result.event.what_we_saw["cold_sample_retry"]["used"] == "first"


@pytest.mark.asyncio
async def test_a_retry_whose_probe_failed_outright_is_not_used(context_factory):
    service = _ProbeSequence(_probe(70.0), _probe(900.0, success=False))

    result = await _run(_ctx(context_factory, service, _never_measured()))

    assert service.calls == 2
    assert result.passed is False
    assert result.updates["state"].specs["network"]["ema_verifyx_download_speed"] == pytest.approx(70.0)


@pytest.mark.asyncio
async def test_known_host_below_the_gate_is_not_retried(context_factory):
    # prev EMA 110 + sample 20 -> EMA 65: a real slow host, judged as today.
    service = _ProbeSequence(_probe(20.0), _probe(900.0))

    result = await _run(_ctx(context_factory, service, _known(110.0)))

    assert service.calls == 1
    assert result.passed is False
    assert result.updates["state"].specs["network"]["ema_verifyx_download_speed"] == pytest.approx(65.0)


@pytest.mark.asyncio
async def test_fast_first_sample_is_not_retried(context_factory):
    service = _ProbeSequence(_probe(500.0), _probe(1.0))

    result = await _run(_ctx(context_factory, service, _never_measured()))

    assert service.calls == 1
    assert result.passed is True
    assert "cold_sample_retry" not in result.event.what_we_saw


@pytest.mark.asyncio
async def test_without_backend_data_nobody_looks_new_and_nothing_is_retried(context_factory):
    # rented_data None: every executor would look never-measured; a backend blip must not double VerifyX fleet-wide.
    service = _ProbeSequence(_probe(59.9), _probe(760.0))

    result = await _run(_ctx(context_factory, service, None))

    assert service.calls == 1
    assert result.passed is False


@pytest.mark.asyncio
async def test_ram_or_disk_failure_on_the_first_sample_is_not_retried(context_factory):
    # The retry is for the network sample only; a probe that failed on RAM/disk is judged as today.
    service = _ProbeSequence(_probe(59.9, success=False), _probe(760.0))

    result = await _run(_ctx(context_factory, service, _never_measured()))

    assert service.calls == 1
    assert result.passed is False
    assert result.event.reason_code == Msg.VERIFY_FAILED.reason


class _TimedProbeSequence(_ProbeSequence):
    """A _ProbeSequence whose retry takes `retry_s` and which counts remote kills."""

    def __init__(self, *responses, retry_s: float = 0.0):
        super().__init__(*responses)
        self.retry_s = retry_s
        self.kills = 0

    async def validate_verifyx_and_process_job(self, **kwargs):
        response = await super().validate_verifyx_and_process_job(**kwargs)
        if self.calls == 2:
            await asyncio.sleep(self.retry_s)
        return response

    async def kill_remote_verifyx(self, shell, run_dir, default_extra):
        assert run_dir == self.run_dir  # the retry's own scratch dir and PID, nothing else
        self.kills += 1
        return True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "seconds_left", "retry_s", "calls", "kills", "passed", "retry"),
    [
        # ad2382f9, 31 Oct: the first run ended with ~208 s left and the retry ran into the task
        # deadline, dropping the cycle. Too little time left before scoring: judged on the first sample.
        ("skipped", 200.0, 0.0, 1, 0, False, None),
        ("just_short", 299.0, 0.0, 1, 0, False, None),
        ("completed", 300.0, 0.0, 2, 0, True, {"used": "retry", "timed_out": False}),
        # the retry outlives its budget (time left less the margin): cut off, remote run killed,
        # judged on the first sample instead of losing the whole result
        ("over_budget", 60.05, 5.0, 2, 1, False, {"used": "first", "timed_out": True}),
        # plenty of task time left: the budget stops at the SSH command timeout (patched to 0.05 s)
        ("ssh_timeout_cap", 900.0, 5.0, 2, 1, False, {"used": "first", "timed_out": True}),
    ],
)
async def test_cold_retry_against_the_task_deadline(
    context_factory, monkeypatch, case, seconds_left, retry_s, calls, kills, passed, retry
):
    monkeypatch.setattr(verifyx_check, "task_seconds_left", lambda ctx: seconds_left)
    if case == "over_budget":
        monkeypatch.setattr(verifyx_check, "MIN_COLD_RETRY_SECONDS", 0)
    if case == "ssh_timeout_cap":
        monkeypatch.setattr(verifyx_check, "VERIFYX_COMMAND_TIMEOUT_SECONDS", 0.05)
    service = _TimedProbeSequence(_probe(88.0), _probe(760.0), retry_s=retry_s)

    result = await _run(_ctx(context_factory, service, _never_measured()))

    assert (service.calls, service.kills, result.passed) == (calls, kills, passed)
    saw = result.event.what_we_saw["cold_sample_retry"]
    if retry is None:
        assert saw == {"skipped_task_seconds_left": int(seconds_left)}
    else:
        assert {k: saw[k] for k in retry} == retry
    if not passed:
        assert result.event.reason_code == Msg.VERIFY_FAILED_NETWORK_SPEED_TOO_SLOW.reason
        assert result.updates["state"].specs["network"]["ema_verifyx_download_speed"] == pytest.approx(88.0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("transport_error", "timed_out"),
    [("TimeoutError: ", True), ("ConnectionLost: connection reset", False)],
)
async def test_a_retry_whose_ssh_call_failed_kills_the_remote_run(
    context_factory, monkeypatch, transport_error, timed_out
):
    # _run_ssh_command returns its own timeout (or a dropped connection) as an error response; the
    # remote run may still be going, so it is killed by its PID like a deadline cut-off.
    monkeypatch.setattr(verifyx_check, "task_seconds_left", lambda ctx: 900.0)
    failed = MockVerifyXResponse(error="ssh failed", diagnostics={"transport_error": transport_error})
    service = _TimedProbeSequence(_probe(88.0), failed)

    result = await _run(_ctx(context_factory, service, _never_measured()))

    assert (service.calls, service.kills, result.passed) == (2, 1, False)
    assert result.event.what_we_saw["cold_sample_retry"]["used"] == "first"
    assert result.event.what_we_saw["cold_sample_retry"]["timed_out"] is timed_out


@pytest.mark.asyncio
async def test_an_outer_cancel_during_the_retry_still_cancels_the_task(context_factory, monkeypatch):
    monkeypatch.setattr(verifyx_check, "task_seconds_left", lambda ctx: 900.0)
    service = _TimedProbeSequence(_probe(88.0), _probe(760.0), retry_s=5.0)
    task = asyncio.create_task(_run(_ctx(context_factory, service, _never_measured())))
    while service.calls < 2:
        await asyncio.sleep(0)

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert service.kills == 1  # the cut-off remote run is killed on the way out


class _LocalShell:
    """Runs the 'SSH' commands in a local bash, in `home` (the login dir)."""

    def __init__(self, home):
        self.ssh_client = self
        self.home = home

    async def run(self, command, timeout=None):
        proc = await asyncio.create_subprocess_shell(
            command, cwd=self.home, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout)
        return SimpleNamespace(stdout=out.decode(), stderr=err.decode(), exit_status=proc.returncode)


def _fake_python(tmp_path, body: str):
    path = tmp_path / "python"
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return SimpleNamespace(python_path=str(path), root_dir=str(tmp_path / "app"))


@pytest.mark.asyncio
async def test_a_retry_run_cleans_its_scratch_dir_and_keeps_its_exit_status(tmp_path):
    executor = _fake_python(tmp_path, "touch temp_1.dat; echo answer; exit 3")
    service = VerifyXValidationService()
    command = service._verifyx_command(executor, SimpleNamespace(seed=1, cipher_text="c"), "lib", ".vx-run")

    capture = await service._run_ssh_command(_LocalShell(tmp_path), command)

    assert (capture.stdout.strip(), capture.exit_status) == ("answer", 3)
    assert not (tmp_path / ".vx-run").exists()


def _timeout_without_k(tmp_path, monkeypatch):
    # busybox `timeout` has no -k
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "timeout"
    stub.write_text("#!/bin/sh\necho 'timeout: unrecognized option: k' >&2\nexit 1\n")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")


@pytest.mark.asyncio
async def test_a_retry_run_does_not_start_on_a_host_whose_timeout_lacks_k(tmp_path, monkeypatch):
    # without a remote deadline a run whose cleanup failed could outlive every reap
    _timeout_without_k(tmp_path, monkeypatch)
    executor = _fake_python(tmp_path, f"touch {tmp_path}/started")
    service = VerifyXValidationService()
    command = service._verifyx_command(executor, SimpleNamespace(seed=1, cipher_text="c"), "lib", ".vx-run")

    capture = await service._run_ssh_command(_LocalShell(tmp_path), command)

    assert capture.exit_status == 1
    assert not (tmp_path / "started").exists()
    assert not (tmp_path / ".vx-run").exists()


@pytest.mark.asyncio
async def test_killing_a_cut_off_retry_during_the_disk_test_spares_other_processes(tmp_path):
    # the disk test writes ./temp_<ts>.dat and removes it only when it finishes; SIGKILL skips that
    executor = _fake_python(tmp_path, "touch temp_1.dat; exec sleep 60")
    service = VerifyXValidationService()
    shell = _LocalShell(tmp_path)
    # the same command line in another container (the executor shares the host PID namespace)
    decoy = await asyncio.create_subprocess_exec(
        executor.python_path, f"{executor.root_dir}/src/verifyx_executor.py", "--seed", "1", cwd=tmp_path
    )
    command = service._verifyx_command(executor, SimpleNamespace(seed=1, cipher_text="c"), "lib", ".vx-run")
    run = asyncio.create_task(service._run_ssh_command(shell, command))
    while not (tmp_path / ".vx-run" / "temp_1.dat").exists() or not (tmp_path / ".vx-run" / "pid").exists():
        await asyncio.sleep(0.05)
    pid = int((tmp_path / ".vx-run" / "pid").read_text())

    await service.kill_remote_verifyx(shell, ".vx-run", {})

    capture = await asyncio.wait_for(run, 10)
    assert capture.exit_status == 137
    assert not (tmp_path / ".vx-run").exists()
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert decoy.returncode is None
    decoy.kill()
    await decoy.wait()


class _StalledChannelShell:
    """An SSH channel that never opens: run() ignores its own timeout, as AsyncSSH does before create_process returns."""

    def __init__(self):
        self.ssh_client = self

    async def run(self, command, timeout=None):
        await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_the_cleanup_of_a_cut_off_retry_is_bounded_when_the_channel_stalls(monkeypatch):
    monkeypatch.setattr(verifyx_validation_service, "VERIFYX_KILL_TIMEOUT_SECONDS", 0.05)

    assert (
        await asyncio.wait_for(
            VerifyXValidationService().kill_remote_verifyx(_StalledChannelShell(), ".vx-run", {}), 2
        )
        is False
    )


class _FlakyCleanupShell(_LocalShell):
    """A local shell on which the cleanup channel stalls while `stalled` is set."""

    stalled = True

    async def run(self, command, timeout=None):
        if self.stalled and command.startswith("p=$(cat"):
            await asyncio.Event().wait()
        return await super().run(command, timeout)


class _HungRetryService(VerifyXValidationService):
    """First samples come from `firsts`; the retry really starts the remote run (which hangs)."""

    def __init__(self, executor, *firsts):
        super().__init__()
        self.executor, self.firsts = executor, list(firsts)
        self.run_dirs = []

    async def validate_verifyx_and_process_job(
        self, *, shell, executor_info, default_extra, machine_spec, run_dir=None
    ):
        if run_dir is None:
            return self.firsts.pop(0)
        self.run_dirs.append(run_dir)
        challenge = SimpleNamespace(seed=1, cipher_text="c")
        capture = await self._run_ssh_command(shell, self._verifyx_command(self.executor, challenge, "lib", run_dir))
        return MockVerifyXResponse(error=f"exit {capture.exit_status}", diagnostics={})


@pytest.mark.asyncio
async def test_a_cut_off_retry_whose_cleanup_timed_out_is_reaped_on_the_next_run(
    context_factory, monkeypatch, tmp_path
):
    # PR review, 3 Nov: a cleanup channel that never opened left the cut-off run alive and forgotten
    monkeypatch.setattr(verifyx_check, "task_seconds_left", lambda ctx: 900.0)
    monkeypatch.setattr(verifyx_check, "VERIFYX_COMMAND_TIMEOUT_SECONDS", 1.0)
    monkeypatch.setattr(verifyx_validation_service, "VERIFYX_KILL_TIMEOUT_SECONDS", 0.2)
    executor = _fake_python(tmp_path, "touch temp_1.dat; exec sleep 60")
    service = _HungRetryService(executor, _probe(88.0), _probe(900.0))
    shell = _FlakyCleanupShell(tmp_path)
    ctx = context_factory(
        services=build_services(verifyx=service, shell=shell),
        config=build_context_config(),
        state=build_state(specs={"gpu": {"count": 1}}, rented_data=_never_measured()),
    )

    result = await _run(ctx)

    assert result.event.what_we_saw["cold_sample_retry"]["timed_out"] is True
    run_dir = tmp_path / service.run_dirs[0]
    pid = int((run_dir / "pid").read_text())
    os.kill(pid, 0)  # the cut-off run lives on: its cleanup never reached the node
    assert (run_dir / "temp_1.dat").exists()

    shell.stalled = False  # the node is reachable again on the next cycle
    await VerifyXCheck().run(ctx)

    for _ in range(100):  # the killed run's wrapper shell exits and is reaped by the local bash
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        await asyncio.sleep(0.05)
    else:
        pytest.fail("the old run is still alive")
    assert not run_dir.exists()
    assert await ctx.services.redis.hgetall(
        verifyx_stale_runs_key(ctx.miner_hotkey, ctx.executor.uuid)
    ) == {}


@pytest.mark.asyncio
async def test_no_timeout_k_with_failed_cleanup_and_pruning_leaves_no_forgotten_run(
    context_factory, monkeypatch, tmp_path
):
    # PR review, 3 Nov: a retry without a remote deadline, its cleanup stalled, could be pruned and forgotten
    monkeypatch.setattr(verifyx_check, "task_seconds_left", lambda ctx: 900.0)
    monkeypatch.setattr(verifyx_validation_service, "VERIFYX_KILL_TIMEOUT_SECONDS", 0.2)
    _timeout_without_k(tmp_path, monkeypatch)
    executor = _fake_python(tmp_path, f"touch {tmp_path}/started; exec sleep 60")
    service = _HungRetryService(executor, _probe(88.0))
    ctx = context_factory(
        services=build_services(verifyx=service, shell=_FlakyCleanupShell(tmp_path)),
        config=build_context_config(),
        state=build_state(specs={"gpu": {"count": 1}}, rented_data=_never_measured()),
    )
    key = verifyx_stale_runs_key(ctx.miner_hotkey, ctx.executor.uuid)
    now = time.time()
    for i in range(verifyx_check.STALE_RUNS_KEPT + 2):
        await ctx.services.redis.hset(key, f".verifyx-retry-{i}", str(now - i))

    result = await _run(ctx)

    assert result.passed is False
    assert result.event.what_we_saw["cold_sample_retry"]["used"] == "first"
    assert not (tmp_path / "started").exists()  # the retry never ran
    assert not (tmp_path / service.run_dirs[0]).exists()
    left = {k.decode() for k in await ctx.services.redis.hgetall(key)}
    assert left == {f".verifyx-retry-{i}" for i in range(verifyx_check.STALE_RUNS_KEPT)}  # still bounded


@pytest.mark.asyncio
async def test_a_stalled_stale_run_save_does_not_cost_the_first_result(context_factory, monkeypatch):
    # PR review, 3 Nov: HSET/EXPIRE behind a busy Redis lock ran past the cycle margin
    monkeypatch.setattr(verifyx_check, "task_seconds_left", lambda ctx: 60.05)
    monkeypatch.setattr(verifyx_check, "MIN_COLD_RETRY_SECONDS", 0)
    monkeypatch.setattr(verifyx_check, "STALE_RUN_SAVE_TIMEOUT_SECONDS", 0.01)
    service = _TimedProbeSequence(_probe(88.0), _probe(760.0), retry_s=5.0)
    service.kill_remote_verifyx = AsyncMock(return_value=False)
    ctx = _ctx(context_factory, service, _never_measured())

    async def blocked_write(*_args):
        await asyncio.Event().wait()

    ctx.services.redis.hset = AsyncMock(side_effect=blocked_write)

    result = await asyncio.wait_for(_run(ctx), 1.0)

    assert result.passed is False
    assert result.event.what_we_saw["cold_sample_retry"]["used"] == "first"


@pytest.mark.asyncio
async def test_reaping_stale_runs_is_bounded_per_cycle(context_factory, monkeypatch):
    # PR review, 3 Nov: a node stalling every cleanup must not grow the next cycle's reaping work
    service = _ProbeSequence(_probe(900.0))
    service.kill_remote_verifyx = AsyncMock(return_value=False)
    ctx = _ctx(context_factory, service, _known(900.0))
    key = verifyx_stale_runs_key(ctx.miner_hotkey, ctx.executor.uuid)
    now = time.time()
    for i in range(10):
        await ctx.services.redis.hset(key, f".verifyx-retry-{i}", str(now - i))
    await ctx.services.redis.hset(key, ".verifyx-retry-old", str(now - VERIFYX_STALE_RUNS_TTL_SECONDS - 1))

    await VerifyXCheck().run(ctx)

    tried = [c.args[1] for c in service.kill_remote_verifyx.await_args_list]
    assert tried == [".verifyx-retry-0", ".verifyx-retry-1"]  # newest first, capped
    left = {k.decode() for k in await ctx.services.redis.hgetall(key)}
    assert left == {f".verifyx-retry-{i}" for i in range(verifyx_check.STALE_RUNS_KEPT)}


@pytest.mark.parametrize("fast_path", [False, True])
def test_the_cold_retry_runs_after_every_other_check_right_before_scoring(fast_path):
    checks = PipelineFactory.build_checks(fast_path=fast_path)
    names = [type(c).__name__ for c in checks]
    assert names.index("VerifyXCheck") < names.index("VerifyXColdRetryCheck") == names.index("ScoreCheck") - 1
    # review 5771536147: the retry check judges the disk of the sample it keeps, so nothing but scoring follows it
    assert names[names.index("VerifyXColdRetryCheck") + 1 :] == ["ScoreCheck", "FinalizeCheck"]
    assert not any(isinstance(c, ParallelStage) for c in checks[names.index("VerifyXColdRetryCheck") :])


@pytest.mark.asyncio
@pytest.mark.parametrize(("left_before_score", "retried"), [(195.0, False), (400.0, True)])
async def test_the_retry_only_gets_the_time_the_other_checks_left(
    context_factory, monkeypatch, left_before_score, retried
):
    # PR review, 31 Oct: 680 s left after a 240 s first sample; a 240 s retry right after VerifyX
    # left 440 s for a 485 s tail and the deadline cancelled the cycle. The retry now runs after
    # that tail: with 195 s left it is skipped and the first sample decides; with 400 s left it may
    # use 340 s, the rest being the margin for scoring and publishing.
    monkeypatch.setattr(verifyx_check, "task_seconds_left", lambda ctx: left_before_score)
    timeouts = []
    wait_for = asyncio.wait_for

    async def spy(aw, timeout):
        timeouts.append(timeout)
        return await wait_for(aw, timeout)

    monkeypatch.setattr(verifyx_check.asyncio, "wait_for", spy)
    service = _TimedProbeSequence(_probe(88.0), _probe(760.0))

    result = await _run(_ctx(context_factory, service, _never_measured()))

    assert service.calls == (2 if retried else 1)
    assert result.passed is retried
    # with room to retry, the rental-state lookup (capped) and then the retry itself
    expected = [verifyx_check.RENTAL_STATE_TIMEOUT_SECONDS, left_before_score - verifyx_check.COLD_RETRY_MARGIN_SECONDS]
    assert timeouts == (expected if retried else [])


def _pod_on(uuid):
    pod = SimpleNamespace(container_name="container_1", rented_ports=[])
    return SimpleNamespace(pods=[pod])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("state", "retried"),
    [("idle", True), ("filler_only", True), ("newly_rented", False), ("seen_rented_by_a_check", False),
     ("create_in_progress", False), ("unavailable", False), ("lookup_failed", False)],
)
async def test_the_retry_runs_only_on_a_node_still_idle(context_factory, state, retried):
    # PR review, 31 Oct: a customer pod started after the first 88 Mbps sample must not get a second
    # RAM, disk and network test beside its workload. Busy or unknown: the first sample decides.
    service = _ProbeSequence(_probe(88.0), _probe(760.0))
    ctx = _ctx(context_factory, service, _never_measured())
    fresh = _never_measured()
    if state == "filler_only":
        fresh = RentedExecutorsResponse(
            executors={}, banned_guids=[], network_ema={}, all_filler_containers_by_executor={EXECUTOR: ["filler_1"]}
        )
    if state == "newly_rented":
        fresh = _never_measured().model_copy(update={"executors": {EXECUTOR: _pod_on(EXECUTOR)}})
    _idle_now(ctx.services, fresh=fresh, creating=state == "create_in_progress")
    if state == "unavailable":
        ctx.services.backend.get_rented_executors_now = AsyncMock(return_value=None)
    if state == "lookup_failed":
        ctx.services.redis.renting_in_progress = AsyncMock(side_effect=ConnectionError("redis down"))
    first = await VerifyXCheck().run(ctx)
    ctx = ctx.model_copy(update=first.updates)
    if state == "seen_rented_by_a_check":  # e.g. CapabilityCheck refreshed the snapshot
        rented = ctx.state.rented_data.model_copy(update={"executors": {EXECUTOR: _pod_on(EXECUTOR)}})
        ctx = ctx.model_copy(update={"state": dataclasses.replace(ctx.state, rented_data=rented)})

    result = await VerifyXColdRetryCheck().run(ctx)

    assert service.calls == (2 if retried else 1)
    assert result.passed is retried
    if not retried:
        assert result.event.reason_code == Msg.VERIFY_FAILED_NETWORK_SPEED_TOO_SLOW.reason
        assert result.event.what_we_saw["cold_sample_retry"]["skipped"] in {
            "rented", "rental_create_in_progress", "rental_state_unavailable"
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("seconds_left", [200.0, 900.0])
async def test_a_slow_rental_lookup_never_costs_the_first_sample(context_factory, monkeypatch, seconds_left):
    # PR review F-16, 31 Oct: the rental-state re-read ran before the time check and unbounded, so a
    # hung backend or Redis could carry the cycle past its deadline. Too little time: no lookup at
    # all. Otherwise the lookup is cut at its cap and the first sample decides.
    monkeypatch.setattr(verifyx_check, "task_seconds_left", lambda ctx: seconds_left)
    monkeypatch.setattr(verifyx_check, "RENTAL_STATE_TIMEOUT_SECONDS", 0.05)
    service = _ProbeSequence(_probe(88.0), _probe(760.0))
    ctx = _ctx(context_factory, service, _never_measured())
    _idle_now(ctx.services)

    async def hang(*_):
        await asyncio.sleep(3600)

    ctx.services.redis.renting_in_progress = AsyncMock(side_effect=hang)
    first = await VerifyXCheck().run(ctx)

    result = await asyncio.wait_for(VerifyXColdRetryCheck().run(ctx.model_copy(update=first.updates)), 5)

    assert service.calls == 1
    assert result.passed is False
    assert result.event.reason_code == Msg.VERIFY_FAILED_NETWORK_SPEED_TOO_SLOW.reason
    saw = result.event.what_we_saw["cold_sample_retry"]
    if seconds_left == 200.0:
        assert saw == {"skipped_task_seconds_left": 200}
        ctx.services.redis.renting_in_progress.assert_not_awaited()
    else:
        assert saw == {"skipped": "rental_state_unavailable"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("flip", "reason"),
    [("marker", "rental_create_in_progress"), ("unavailable", "rental_state_unavailable")],
)
async def test_a_rental_starting_during_the_retry_cuts_it_off(
    context_factory, monkeypatch, flip, reason
):
    # PR review, 31 Oct: the node was idle at the lookup, then a renter's create began (pending-pod
    # marker) or the marker could not be read while the retry ran. The retry is cancelled, its remote
    # run killed by its PID, and the first sample decides. During the retry only the per-executor
    # marker is watched: the backend's fleet-wide rental list is read once, before the retry.
    monkeypatch.setattr(verifyx_check, "task_seconds_left", lambda ctx: 900.0)
    monkeypatch.setattr(verifyx_check, "RENTAL_WATCH_INTERVAL_SECONDS", 0.01)
    service = _TimedProbeSequence(_probe(88.0), _probe(760.0), retry_s=5.0)
    ctx = _ctx(context_factory, service, _never_measured())
    _idle_now(ctx.services)
    first = await VerifyXCheck().run(ctx)
    later = {"marker": [False, False, True], "unavailable": [False, False, ConnectionError("redis down")]}[flip]
    ctx.services.redis.renting_in_progress = AsyncMock(side_effect=later + [True] * 10)
    ctx.services.backend.get_rented_executors_now = AsyncMock(return_value=_never_measured())

    result = await asyncio.wait_for(
        VerifyXColdRetryCheck().run(ctx.model_copy(update=first.updates)), 2
    )

    assert (service.calls, service.kills, result.passed) == (2, 1, False)
    assert result.event.reason_code == Msg.VERIFY_FAILED_NETWORK_SPEED_TOO_SLOW.reason
    assert result.event.what_we_saw["cold_sample_retry"] == {"cancelled": reason}
    assert ctx.services.backend.get_rented_executors_now.await_count == 1  # pre-retry check only
    assert result.updates["state"].specs["network"]["ema_verifyx_download_speed"] == pytest.approx(
        88.0
    )


class _PendingRedis(FakeRedis):
    get_pending_pods = RedisService.get_pending_pods
    add_pending_pod = RedisService.add_pending_pod
    remove_pending_pod = RedisService.remove_pending_pod
    renting_in_progress = RedisService.renting_in_progress


@pytest.mark.asyncio
async def test_a_failed_create_stamp_write_leaves_no_pending_pod_marker():
    # PR review, 3 Nov: the marker and its stamp were two writes; the stamp failing left a gpu_reset marker behind
    redis = _PendingRedis()
    redis.fail_next_set_of.add(create_started_at_key("miner", EXECUTOR))

    with pytest.raises(RedisTimeoutError):
        await redis.add_pending_pod("miner", EXECUTOR, "pod-1")

    assert redis.hashes.get(PENDING_PODS_PREFIX, {}) == {}
    assert await redis.get_pending_pods("miner", EXECUTOR) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["gpu_reset", "gpu_clocks"])
async def test_a_gpu_operation_marker_leaves_no_create_stamp(kind):
    # PR review, 5 Nov: a GPU reset or clock operation is no rental create
    redis = _PendingRedis()
    await redis.add_pending_pod("miner", EXECUTOR, "pod-1", kind=kind)

    assert [p["kind"] for p in await redis.get_pending_pods("miner", EXECUTOR)] == [kind]
    assert await redis.get(create_started_at_key("miner", EXECUTOR)) is None


class _CreateDuringRetry(_TimedProbeSequence):
    """The retry, during which a create's pending-pod marker is set and cleared inside one watcher interval."""

    def __init__(self, redis, hotkey, *responses, retry_s):
        super().__init__(*responses, retry_s=retry_s)
        self.redis, self.hotkey = redis, hotkey

    async def validate_verifyx_and_process_job(self, **kwargs):
        if self.calls == 1:
            await asyncio.sleep(0.01)  # just after a watcher read
            await self.redis.add_pending_pod(self.hotkey, EXECUTOR, "pod-fast")
            await self.redis.remove_pending_pod(self.hotkey, EXECUTOR, "pod-fast")
        return await super().validate_verifyx_and_process_job(**kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("create", ["during_retry", "before_retry"])
async def test_a_create_between_two_watcher_reads_still_cuts_the_retry_off(context_factory, monkeypatch, create):
    # PR review, 31 Oct: the watcher reads only the pending-pod marker, every interval, so a fast create
    # whose marker came and went between two reads was missed. Its create-started stamp is read too: a
    # create after the retry began cuts it off; an older one (seen by the pre-retry lookup) does not.
    assert CREATE_STARTED_AT_TTL_SECONDS > VERIFYX_COMMAND_TIMEOUT_SECONDS
    monkeypatch.setattr(verifyx_check, "task_seconds_left", lambda ctx: 900.0)
    monkeypatch.setattr(verifyx_check, "RENTAL_WATCH_INTERVAL_SECONDS", 0.05)
    redis = _PendingRedis()
    if create == "during_retry":
        service = _CreateDuringRetry(redis, None, _probe(88.0), _probe(760.0), retry_s=5.0)
    else:
        service = _TimedProbeSequence(_probe(88.0), _probe(760.0), retry_s=0.2)
    ctx = context_factory(
        services=build_services(redis=redis, verifyx=service),
        config=build_context_config(),
        state=build_state(specs={"gpu": {"count": 1}}, rented_data=_never_measured()),
    )
    service.hotkey = ctx.miner_hotkey
    if create == "before_retry":
        await redis.add_pending_pod(ctx.miner_hotkey, EXECUTOR, "pod-old")
        await redis.remove_pending_pod(ctx.miner_hotkey, EXECUTOR, "pod-old")
        await asyncio.sleep(0.01)
    ctx.services.backend.get_rented_executors_now = AsyncMock(return_value=_never_measured())
    first = await VerifyXCheck().run(ctx)

    result = await asyncio.wait_for(VerifyXColdRetryCheck().run(ctx.model_copy(update=first.updates)), 2)

    assert service.calls == 2
    if create == "during_retry":
        assert (service.kills, result.passed) == (1, False)
        assert result.event.what_we_saw["cold_sample_retry"] == {"cancelled": "rental_create_in_progress"}
        assert result.updates["state"].specs["network"]["ema_verifyx_download_speed"] == pytest.approx(88.0)
    else:
        assert (service.kills, result.passed) == (0, True)
        assert result.event.what_we_saw["cold_sample_retry"]["used"] == "retry"


@pytest.mark.asyncio
async def test_concurrent_cold_retries_share_one_backend_read(context_factory, monkeypatch):
    # PR review, 31 Oct: each cold executor read the fleet-wide rented list before its retry, so N cold
    # executors cost N backend reads. They now share one in-flight read; the snapshot still counts.
    monkeypatch.setattr(verifyx_check, "_rented_now", None)
    backend = AsyncMock()

    async def slow_read():
        await asyncio.sleep(0.05)
        return _never_measured()

    backend.get_rented_executors_now = AsyncMock(side_effect=slow_read)
    rented = _never_measured().model_copy(update={"executors": {EXECUTOR: _pod_on(EXECUTOR)}})
    ctxs = [
        context_factory(
            services=build_services(backend=backend),
            config=build_context_config(),
            state=build_state(specs={}, rented_data=rented if i == 0 else _never_measured()),
        )
        for i in range(8)
    ]
    for ctx in ctxs:
        ctx.services.redis.renting_in_progress = AsyncMock(return_value=False)

    busy = [b for b, _ in await asyncio.gather(*(verifyx_check._not_idle_now(ctx, 5) for ctx in ctxs))]

    assert busy == ["rented"] + [None] * 7
    assert backend.get_rented_executors_now.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("create", "busy"),
    [
        ("after_snapshot", "rental_create_in_progress"),
        ("before_snapshot", None),
        ("probe_done_after_snapshot", None),
        ("probe_running", "rental_create_in_progress"),
    ],
)
async def test_a_create_done_between_the_shared_read_and_the_retry_skips_it(
    context_factory, monkeypatch, create, busy
):
    # PR review F-20, 31 Oct: another executor's lookup filled the shared rented list at t0; this node's
    # customer create ran t1-t4 (marker set and cleared); its retry starts at t6 and reuses the t0 list.
    # A create stamped after the list was read counts as busy; one stamped before it is in the list.
    # PR review, 4 Nov: the validator's own rental probe leaves no stamp once done; while it runs its marker counts.
    monkeypatch.setattr(verifyx_check, "task_seconds_left", lambda ctx: 900.0)
    monkeypatch.setattr(verifyx_check, "_rented_now", None)
    redis = _PendingRedis()
    service = _TimedProbeSequence(_probe(88.0), _probe(760.0), retry_s=0.05)
    ctx = context_factory(
        services=build_services(redis=redis, verifyx=service),
        config=build_context_config(),
        state=build_state(specs={"gpu": {"count": 1}}, rented_data=_never_measured()),
    )
    ctx.services.backend.get_rented_executors_now = AsyncMock(return_value=_never_measured())
    if create == "before_snapshot":
        await redis.add_pending_pod(ctx.miner_hotkey, EXECUTOR, "pod-old")
        await redis.remove_pending_pod(ctx.miner_hotkey, EXECUTOR, "pod-old")
        await asyncio.sleep(0.01)
    await verifyx_check._rented_executors_now_shared(ctx.services.backend)  # t0: another executor's read
    if create != "before_snapshot":
        await asyncio.sleep(0.01)
        await redis.add_pending_pod(ctx.miner_hotkey, EXECUTOR, "pod-fast", synthetic_probe=create != "after_snapshot")
        if create != "probe_running":
            await redis.remove_pending_pod(ctx.miner_hotkey, EXECUTOR, "pod-fast")
    await asyncio.sleep(0.01)
    first = await VerifyXCheck().run(ctx)

    result = await asyncio.wait_for(VerifyXColdRetryCheck().run(ctx.model_copy(update=first.updates)), 2)

    assert ctx.services.backend.get_rented_executors_now.await_count == 1  # the t0 read was reused
    if busy:
        assert (service.calls, result.passed) == (1, False)
        assert result.event.what_we_saw["cold_sample_retry"] == {"skipped": busy}
    else:
        assert (service.calls, result.passed) == (2, True)
        assert result.event.what_we_saw["cold_sample_retry"]["used"] == "retry"


@pytest.mark.asyncio
async def test_a_used_retry_replaces_the_first_samples_judgement(context_factory, monkeypatch):
    # review 5765394179: one probe is judged once. A first sample below the gate, then a better
    # retry: the package fallback is counted once and the retry's disk reading is the one published.
    monkeypatch.setattr(verifyx_check, "_package_fallback_probes", {})
    tally = verifyx_validation_service.NetworkGateTally()
    monkeypatch.setattr(verifyx_check, "NETWORK_GATE_TALLY", tally)

    def probe(download, disk):
        response = _probe(download)
        response.data["network"]["cloudflare_fallback"] = True
        response.data["hard_disk"] = {**response.data["hard_disk"], "write_speed": disk}
        return response

    ctx = _ctx(context_factory, _ProbeSequence(probe(40.0, 100.0), probe(760.0, 900.0)), _never_measured())

    result = await _run(ctx)

    assert result.passed is True
    assert result.event.what_we_saw["cold_sample_retry"]["used"] == "retry"
    assert verifyx_check._package_fallback_probes == {(ctx.miner_hotkey, EXECUTOR): 1}
    assert tally.capacity_pass + tally.capacity_fail + tally.unmeasured == 1
    specs = result.updates["state"].specs
    assert specs["hard_disk"]["write_speed"] == 900.0
    assert specs["network"]["verifyx_download_speed"] == pytest.approx(760.0)


GIB_KB = 1024 * 1024


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "scraped", "first_download", "retry_download", "passed", "disk_judged"),
    [
        # no retry pending (the first sample cleared the gate): the disk floor is not judged again
        ("no_pending", {}, 760.0, 760.0, True, False),
        # no time left to re-measure (review 5766876557): the first sample stands, no second disk check
        ("skipped", {}, 88.0, 760.0, False, False),
        # the retry ran but was not better: the first sample stands, no second disk check
        ("first", {}, 88.0, 40.0, False, False),
        # adopted retry, the scrape omitted capacity (hard_disk_scrape_error): its 94%-full reading fails
        ("retry", {}, 88.0, 760.0, False, True),
        # adopted retry, the scrape's capacity fields win over either sample
        ("retry", {"utilization": 50.0, "free": 500 * GIB_KB}, 88.0, 760.0, True, True),
    ],
)
async def test_the_disk_floor_is_judged_again_only_on_an_adopted_retry(
    context_factory, monkeypatch, case, scraped, first_download, retry_download, passed, disk_judged
):
    from neurons.validators.src.services.task.checks import disk_space

    disk_runs = []
    real_run = disk_space.DiskSpaceCheck.run

    async def counted_run(self, ctx, **kwargs):
        disk_runs.append(ctx.state.specs["hard_disk"])
        return await real_run(self, ctx, **kwargs)

    monkeypatch.setattr(disk_space.DiskSpaceCheck, "run", counted_run)
    if case == "skipped":
        monkeypatch.setattr(verifyx_check, "task_seconds_left", lambda ctx: 200.0)

    def probe(download, utilization):
        response = _probe(download)
        response.data["hard_disk"] = {"total": 1000, "utilization": utilization, "free": 120 * GIB_KB}
        return response

    service = _ProbeSequence(probe(first_download, 80.0), probe(retry_download, 94.0))
    ctx = context_factory(
        services=build_services(verifyx=service),
        config=build_context_config(),
        state=build_state(specs={"gpu": {"count": 1}, "hard_disk": dict(scraped)}, rented_data=_never_measured()),
    )

    result = await _run(ctx)

    assert result.passed is passed
    assert len(disk_runs) == (1 if disk_judged else 0)
    if case == "skipped":
        assert result.event.what_we_saw["cold_sample_retry"] == {"skipped_task_seconds_left": 200}
    elif case in ("first", "retry") and not (disk_judged and not passed):
        assert result.event.what_we_saw["cold_sample_retry"]["used"] == case
    if disk_judged and not passed:
        # the disk failure is the result; the retry's own judgement is reported before it
        assert result.event.check_id == "executor.validate.disk_space"
        assert result.extra_events[-1].what_we_saw["cold_sample_retry"]["used"] == "retry"
        assert result.updates["state"].specs["network"]["verifyx_download_speed"] == pytest.approx(760.0)


@pytest.mark.asyncio
async def test_an_adopted_retry_that_fails_the_disk_floor_names_the_disk_check_as_failed(context_factory):
    # review 5767306334: the pipeline's steps_failed (the provider timeline's failed step) is the failed event's
    # check, executor.validate.disk_space, not the cold retry check that returned it
    from neurons.validators.src.services.task.pipeline import Pipeline

    class Sink:
        async def emit(self, event) -> None:
            pass

    def probe(download, utilization):
        response = _probe(download)
        response.data["hard_disk"] = {"total": 1000, "utilization": utilization, "free": 120 * GIB_KB}
        return response

    ctx = context_factory(
        services=build_services(verifyx=_ProbeSequence(probe(88.0, 80.0), probe(760.0, 94.0))),
        config=build_context_config(),
        state=build_state(specs={"gpu": {"count": 1}, "hard_disk": {}}, rented_data=_never_measured()),
    )
    _idle_now(ctx.services)

    ok, events, _ = await Pipeline([VerifyXCheck(), VerifyXColdRetryCheck()], Sink()).run(ctx)

    assert ok is False
    assert events[-1].check_id == "executor.validate.disk_space"
    assert events[-1].what_we_saw["steps_failed"] == "executor.validate.disk_space"


@pytest.mark.asyncio
@pytest.mark.parametrize(("retry_utilization", "ok"), [(50.0, True), (94.0, False)])
async def test_a_full_first_disk_reading_waits_for_the_sample_the_cold_retry_keeps(
    context_factory, retry_utilization, ok
):
    # review 5771536147: DiskSpaceCheck runs before the retry, so a 94%-full first reading must not fail the node
    # when the adopted retry reads 50%; the retry check judges the disk of the sample it keeps
    from neurons.validators.src.services.task.checks.disk_space import DiskSpaceCheck
    from neurons.validators.src.services.task.pipeline import Pipeline

    class Sink:
        async def emit(self, event) -> None:
            pass

    def probe(download, utilization):
        response = _probe(download)
        response.data["hard_disk"] = {"total": 1000, "utilization": utilization, "free": 120 * GIB_KB}
        return response

    service = _ProbeSequence(probe(88.0, 94.0), probe(760.0, retry_utilization))
    ctx = context_factory(
        services=build_services(verifyx=service),
        config=build_context_config(),
        state=build_state(specs={"gpu": {"count": 1}, "hard_disk": {}}, rented_data=_never_measured()),
    )
    _idle_now(ctx.services)

    passed, events, _ = await Pipeline([VerifyXCheck(), DiskSpaceCheck(), VerifyXColdRetryCheck()], Sink()).run(ctx)

    assert passed is ok
    assert service.calls == 2
    disk_events = [e for e in events if e.check_id == "executor.validate.disk_space"]
    assert disk_events[0].reason_code == "DISK_TOO_FULL" and disk_events[0].what_we_saw["judged_after_cold_retry"]
    # a full retry reading fails the node on the disk check; a passing one adds no second disk event
    assert [e.reason_code for e in disk_events[1:]] == ([] if ok else ["DISK_TOO_FULL"])
    if not ok:
        assert "judged_after_cold_retry" not in disk_events[-1].what_we_saw


@pytest.mark.asyncio
@pytest.mark.parametrize(("seconds_left", "cap"), [(65.0, 5.0), (40.0, 0.0), (900.0, 30.0)])
async def test_the_deferred_disk_rental_lookup_never_eats_the_scoring_margin(
    context_factory, monkeypatch, seconds_left, cap
):
    # review 5772289851: the disk floor judged after the retry caps its rental lookup at the time left
    # less the margin (and the usual rental-state cap), so it cannot run the task into its deadline
    from neurons.validators.src.services.task.checks import disk_space
    from neurons.validators.src.services.task.pipeline import CheckResult

    seen = []

    async def run(self, ctx, rental_lookup_timeout=None):
        seen.append(rental_lookup_timeout)
        return CheckResult.model_construct(passed=True, event=None)

    monkeypatch.setattr(disk_space.DiskSpaceCheck, "run", run)
    monkeypatch.setattr(verifyx_check, "task_seconds_left", lambda ctx: seconds_left)
    ctx = context_factory(state=build_state(specs={"gpu": {"count": 1}}))
    judged = CheckResult.model_construct(passed=True, event=None, updates={})

    assert await VerifyXColdRetryCheck()._with_disk(ctx, judged) is judged
    assert seen == [pytest.approx(cap)]
