"""DAH-2959: a never-measured executor's first VerifyX download sample decides its first cycle alone
(the EMA bootstraps from it). The check takes one more sample inside the same task and gates on the
better one."""

import asyncio
import os
import stat
from types import SimpleNamespace

import pytest

from neurons.validators.src.services.task.checks import verifyx as verifyx_check
from neurons.validators.src.services.task.checks.verifyx import VerifyXCheck
from neurons.validators.src.services.task.messages import VerifyXMessages as Msg
from neurons.validators.src.services.verifyx_validation_service import VerifyXValidationService
from protocol.vc_protocol.compute_requests import NetworkEMA, RentedExecutorsResponse

from tests.helpers import build_context_config, build_services, build_state
from tests.test_verifyx_check import MockVerifyXResponse

EXECUTOR = "executor-123"  # tests.helpers.default_executor().uuid


def _probe(download: float | None, *, success: bool = True, upload: float | None = 40.0) -> MockVerifyXResponse:
    network = {} if download is None else {"download_speed": download, "upload_speed": upload}
    return MockVerifyXResponse(
        data={"success": success, "ram": {"total": 64}, "hard_disk": {"total": 1000}, "network": network}
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


@pytest.mark.asyncio
async def test_cold_first_sample_is_re_measured_and_the_better_one_seeds_the_ema(context_factory):
    # b3e4ca69, 4 Sep: 59.9 Mbps on the first sample, ~760 Mbps once the link was free.
    service = _ProbeSequence(_probe(59.9, upload=20.0), _probe(760.0, upload=110.0))

    result = await VerifyXCheck().run(_ctx(context_factory, service, _never_measured()))

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

    result = await VerifyXCheck().run(_ctx(context_factory, service, _never_measured()))

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

    result = await VerifyXCheck().run(_ctx(context_factory, service, _never_measured()))

    assert service.calls == 2
    assert result.passed is True
    assert result.updates["state"].specs["network"]["ema_verifyx_download_speed"] == pytest.approx(300.0)
    assert result.event.what_we_saw["cold_sample_retry"]["first_download_speed_mbps"] is None


@pytest.mark.asyncio
async def test_a_worse_retry_keeps_the_first_sample(context_factory):
    service = _ProbeSequence(_probe(70.0), _probe(30.0))

    result = await VerifyXCheck().run(_ctx(context_factory, service, _never_measured()))

    assert service.calls == 2
    assert result.passed is False
    assert result.updates["state"].specs["network"]["ema_verifyx_download_speed"] == pytest.approx(70.0)
    assert result.event.what_we_saw["cold_sample_retry"]["used"] == "first"


@pytest.mark.asyncio
async def test_a_retry_whose_probe_failed_outright_is_not_used(context_factory):
    service = _ProbeSequence(_probe(70.0), _probe(900.0, success=False))

    result = await VerifyXCheck().run(_ctx(context_factory, service, _never_measured()))

    assert service.calls == 2
    assert result.passed is False
    assert result.updates["state"].specs["network"]["ema_verifyx_download_speed"] == pytest.approx(70.0)


@pytest.mark.asyncio
async def test_known_host_below_the_gate_is_not_retried(context_factory):
    # prev EMA 110 + sample 20 -> EMA 65: a real slow host, judged as today.
    service = _ProbeSequence(_probe(20.0), _probe(900.0))

    result = await VerifyXCheck().run(_ctx(context_factory, service, _known(110.0)))

    assert service.calls == 1
    assert result.passed is False
    assert result.updates["state"].specs["network"]["ema_verifyx_download_speed"] == pytest.approx(65.0)


@pytest.mark.asyncio
async def test_fast_first_sample_is_not_retried(context_factory):
    service = _ProbeSequence(_probe(500.0), _probe(1.0))

    result = await VerifyXCheck().run(_ctx(context_factory, service, _never_measured()))

    assert service.calls == 1
    assert result.passed is True
    assert "cold_sample_retry" not in result.event.what_we_saw


@pytest.mark.asyncio
async def test_without_backend_data_nobody_looks_new_and_nothing_is_retried(context_factory):
    # rented_data None: every executor would look never-measured; a backend blip must not double VerifyX fleet-wide.
    service = _ProbeSequence(_probe(59.9), _probe(760.0))

    result = await VerifyXCheck().run(_ctx(context_factory, service, None))

    assert service.calls == 1
    assert result.passed is False


@pytest.mark.asyncio
async def test_ram_or_disk_failure_on_the_first_sample_is_not_retried(context_factory):
    # The retry is for the network sample only; a probe that failed on RAM/disk is judged as today.
    service = _ProbeSequence(_probe(59.9, success=False), _probe(760.0))

    result = await VerifyXCheck().run(_ctx(context_factory, service, _never_measured()))

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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "seconds_left", "retry_s", "calls", "kills", "passed", "retry"),
    [
        # ad2382f9, 31 Oct: the first run ended with ~208 s left and the retry ran into the task
        # deadline, dropping the cycle. Too little time left: judged on the first sample.
        ("skipped", [200.0], 0.0, 1, 0, False, None),
        # a 265 s retry would leave 65 s, less than CapabilityCheck's matmul may take: not admitted
        ("no_room_for_capability", [330.0], 0.0, 1, 0, False, None),
        ("completed", [500.0, 500.0], 0.0, 2, 0, True, {"used": "retry", "timed_out": False}),
        # admitted by the estimate, but the retry outlives its budget (deadline less the reserve):
        # cut off, remote run killed, judged on the first sample instead of losing the whole result
        ("over_budget", [500.0, 180.05], 5.0, 2, 1, False, {"used": "first", "timed_out": True}),
        # plenty of task time left: the budget stops at the SSH command timeout (patched to 0.05 s)
        ("ssh_timeout_cap", [900.0, 900.0], 5.0, 2, 1, False, {"used": "first", "timed_out": True}),
    ],
)
async def test_cold_retry_against_the_task_deadline(
    context_factory, monkeypatch, case, seconds_left, retry_s, calls, kills, passed, retry
):
    first_seconds_left = int(seconds_left[0])
    monkeypatch.setattr(verifyx_check, "task_seconds_left", lambda ctx: seconds_left.pop(0))
    if case == "ssh_timeout_cap":
        monkeypatch.setattr(verifyx_check, "VERIFYX_COMMAND_TIMEOUT_SECONDS", 0.05)
    service = _TimedProbeSequence(_probe(88.0), _probe(760.0), retry_s=retry_s)

    result = await VerifyXCheck().run(_ctx(context_factory, service, _never_measured()))

    assert (service.calls, service.kills, result.passed) == (calls, kills, passed)
    saw = result.event.what_we_saw
    if retry is None:
        assert "cold_sample_retry" not in saw
        assert saw["cold_sample_retry_skipped_task_seconds_left"] == first_seconds_left
    else:
        assert {k: saw["cold_sample_retry"][k] for k in retry} == retry
        assert "cold_sample_retry_skipped_task_seconds_left" not in saw
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

    result = await VerifyXCheck().run(_ctx(context_factory, service, _never_measured()))

    assert (service.calls, service.kills, result.passed) == (2, 1, False)
    assert result.event.what_we_saw["cold_sample_retry"]["used"] == "first"
    assert result.event.what_we_saw["cold_sample_retry"]["timed_out"] is timed_out


@pytest.mark.asyncio
async def test_an_outer_cancel_during_the_retry_still_cancels_the_task(context_factory, monkeypatch):
    monkeypatch.setattr(verifyx_check, "task_seconds_left", lambda ctx: 500.0)
    service = _TimedProbeSequence(_probe(88.0), _probe(760.0), retry_s=5.0)
    task = asyncio.create_task(VerifyXCheck().run(_ctx(context_factory, service, _never_measured())))
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
