from __future__ import annotations

import pytest
from neurons.validators.src.services.executor_connectivity.dind_probe import (
    DOCKER_RUN_CAUSES,
    diagnose_docker_run_error,
)
from neurons.validators.src.services.task.checks.sysbox_required import (
    _NVIDIA_HOOK_TEMPLATES,
    SysboxRequiredCheck,
)
from neurons.validators.src.services.task.messages import SysboxRequiredMessages as Msg
from protocol.vc_protocol.compute_requests import (
    RentedExecutor,
    RentedExecutorsResponse,
    RentedPod,
)

from core.config import settings
from services.executor_connectivity.models import DindLogCause
from tests.helpers import build_state


def _rented_data() -> RentedExecutorsResponse:
    return RentedExecutorsResponse(
        executors={
            "executor-123": RentedExecutor(
                miner_hotkey="test-miner",
                executor_ip_address="127.0.0.1",
                executor_ip_port="22",
                pods=[RentedPod(pod_id="pod-1", container_name="container_test", rented_ports=[8080, 8081])],
            )
        },
    )


@pytest.mark.asyncio
async def test_no_sysbox_unrented_fails(context_factory):
    """Unrented executor without sysbox is rejected, but its verification is kept."""
    ctx = context_factory(state=build_state(sysbox_runtime=False))

    result = await SysboxRequiredCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.SYSBOX_MISSING.reason
    assert "clear_verified_job_info" not in result.updates


@pytest.mark.asyncio
async def test_no_sysbox_rented_passes(context_factory):
    """Rented executor without sysbox is left untouched so live rentals are not disrupted."""
    ctx = context_factory(state=build_state(sysbox_runtime=False, rented_data=_rented_data()))

    result = await SysboxRequiredCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.SYSBOX_OK.reason


@pytest.mark.asyncio
async def test_sysbox_present_passes(context_factory):
    """Executor with sysbox passes regardless of rental status."""
    ctx = context_factory(state=build_state(sysbox_runtime=True))

    result = await SysboxRequiredCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.SYSBOX_OK.reason


@pytest.mark.asyncio
async def test_disabled_flag_skips_enforcement(context_factory, monkeypatch):
    """With the kill-switch off, no-sysbox unrented executors are not rejected."""
    monkeypatch.setattr(settings, "REQUIRE_SYSBOX_FOR_UNRENTED", False)
    ctx = context_factory(state=build_state(sysbox_runtime=False))

    result = await SysboxRequiredCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.DISABLED.reason


@pytest.mark.asyncio
async def test_no_sysbox_with_dind_probe_error_names_the_cause(context_factory):
    """DAH-2856: when the probe's container never answered on sshd, the verdict says why instead of
    "install sysbox" (ticket-0309: three reinstalls on a host whose inner dockerd could not use
    legacy iptables). Scoring is unchanged: still a failed check."""
    cause = DindLogCause(
        "DIND_INNER_DOCKERD_IPTABLES",
        "the inner dockerd cannot use legacy iptables. dockerd said: can't initialize iptables table `nat'",
        dockerd_line="can't initialize iptables table `nat'",
    )
    ctx = context_factory(state=build_state(sysbox_runtime=False, dind_probe_error=cause))

    result = await SysboxRequiredCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.SYSBOX_MISSING.reason
    assert result.event.what_we_saw["dind_probe_error"] == cause.text
    assert cause.text in result.event.remediation
    assert "reinstalling sysbox does not change it" in result.event.remediation
    assert "Install the sysbox runtime" not in result.event.remediation


@pytest.mark.asyncio
async def test_inner_dockerd_down_with_dockerd_line_says_sysbox_is_not_the_fix(context_factory):
    """DIND_INNER_DOCKERD_DOWN with dockerd's own line read from the log: the fix is on the host."""
    cause = DindLogCause(
        "DIND_INNER_DOCKERD_DOWN",
        "the inner dockerd did not start. dockerd said: failed to start daemon: no space left on device",
        dockerd_line="failed to start daemon: no space left on device",
    )
    ctx = context_factory(state=build_state(sysbox_runtime=False, dind_probe_error=cause))

    result = await SysboxRequiredCheck().run(ctx)

    assert result.passed is False
    assert cause.text in result.event.remediation
    assert "reinstalling sysbox does not change it" in result.event.remediation


@pytest.mark.asyncio
async def test_inner_dockerd_down_without_dockerd_line_keeps_the_generic_guidance(context_factory):
    """DIND_INNER_DOCKERD_DOWN with no line read names the symptom only: the verdict must not claim
    sysbox is irrelevant, because an inner dockerd that never started can be a sysbox fault too."""
    cause = DindLogCause("DIND_INNER_DOCKERD_DOWN", "the inner dockerd did not start")
    ctx = context_factory(state=build_state(sysbox_runtime=False, dind_probe_error=cause))

    result = await SysboxRequiredCheck().run(ctx)

    assert result.passed is False
    assert result.event.what_we_saw["dind_probe_error"] == cause.text
    assert cause.text in result.event.remediation
    assert "reinstalling sysbox" not in result.event.remediation


@pytest.mark.asyncio
async def test_no_sysbox_with_unknown_dind_cause_does_not_claim_sysbox_is_irrelevant(context_factory):
    """DIND_SSHD_NOT_READY means the log showed nothing: the verdict quotes it and says no more."""
    cause = DindLogCause(
        "DIND_SSHD_NOT_READY",
        "sshd inside the DinD container did not answer within 30s and its log shows no dockerd error",
    )
    ctx = context_factory(state=build_state(sysbox_runtime=False, dind_probe_error=cause))

    result = await SysboxRequiredCheck().run(ctx)

    assert result.passed is False
    assert cause.text in result.event.remediation
    assert "reinstalling sysbox" not in result.event.remediation


@pytest.mark.asyncio
async def test_no_sysbox_without_dind_probe_error_keeps_the_install_advice(context_factory):
    ctx = context_factory(state=build_state(sysbox_runtime=False))

    result = await SysboxRequiredCheck().run(ctx)

    assert result.passed is False
    assert result.event.remediation == Msg.SYSBOX_MISSING.remediation
    assert "dind_probe_error" not in result.event.what_we_saw


# DAH-3634 — the probe's `docker run` was refused by the NVIDIA container hook.

# the hook's stderr verbatim from the validator log (`DinD creation failed`, 10 to 17 Sep 2026)
NVIDIA_MISMATCH_STDERR = (
    "docker: Error response from daemon: failed to create task for container: failed to create shim task: "
    "OCI runtime create failed: runc create failed: unable to start container process: error during container "
    "init: error running prestart hook #0: exit status 1, stdout: , stderr: Auto-detected mode as 'legacy'\n"
    "nvidia-container-cli: initialization error: nvml error: driver/library version mismatch\n\n"
    "Run 'docker run --help' for more information"
)
NVIDIA_GPU_RESET_STDERR = (
    "docker: Error response from daemon: failed to create task for container: failed to create shim task: "
    "OCI runtime create failed: container_linux.go:439: starting container process caused: process_linux.go:608: "
    "container init caused: Running hook #0:: error running hook: exit status 1, stdout: , stderr: Using requested "
    "mode 'legacy'\nnvidia-container-cli: detection error: nvml error: gpu requires reset\n\n"
    "Run 'docker run --help' for more information"
)


def _cause(stderr: str) -> DindLogCause:
    """The cause exactly as dind_probe writes it into DindProbeResult.error / ContextState.dind_probe_error."""
    cause = diagnose_docker_run_error(stderr)
    assert cause is not None
    return cause


def test_every_docker_run_cause_has_a_template_and_no_template_lacks_a_cause():
    """A rename on either side would route every hook refusal back to SYSBOX_REQUIRED_MISSING in silence."""
    assert {cause.code for _, cause in DOCKER_RUN_CAUSES} == set(_NVIDIA_HOOK_TEMPLATES)
    for code, template in _NVIDIA_HOOK_TEMPLATES.items():
        assert template.reason == code


@pytest.mark.asyncio
async def test_nvidia_driver_mismatch_gets_its_own_reason_code_not_sysbox_missing(context_factory):
    """The hook refused the probe: the node cannot start any GPU container and "install sysbox"
    cannot fix it. Scoring is unchanged: the check still fails."""
    cause = _cause(NVIDIA_MISMATCH_STDERR)
    ctx = context_factory(state=build_state(sysbox_runtime=False, dind_probe_error=cause))

    result = await SysboxRequiredCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == "NVIDIA_RUNTIME_MISMATCH"
    assert result.event.reason_code != Msg.SYSBOX_MISSING.reason
    assert result.event.what_we_saw["dind_probe_error"] == cause.text
    assert "Reboot the host after the NVIDIA driver update" in result.event.remediation
    assert "Sysbox was not measured" in result.event.remediation
    assert "reinstall the NVIDIA container toolkit" in result.event.remediation
    assert "nvml error: driver/library version mismatch" in result.event.remediation
    assert "Install the sysbox runtime" not in result.event.remediation
    assert "clear_verified_job_info" not in result.updates


@pytest.mark.asyncio
async def test_other_nvml_failure_gets_the_generic_hook_code(context_factory):
    ctx = context_factory(state=build_state(sysbox_runtime=False, dind_probe_error=_cause(NVIDIA_GPU_RESET_STDERR)))

    result = await SysboxRequiredCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == "NVIDIA_CONTAINER_HOOK_FAILED"
    assert "gpu requires reset" in result.event.remediation
    assert "sysbox check runs again" in result.event.remediation
    assert "Install the sysbox runtime" not in result.event.remediation


@pytest.mark.asyncio
async def test_load_library_hook_error_gets_the_hook_code_not_sysbox_missing(context_factory):
    stderr = (
        "docker: Error response from daemon: OCI runtime create failed: error running prestart hook #0: exit status 1, "
        "stdout: , stderr: nvidia-container-cli: initialization error: load library failed: "
        "libnvidia-ml.so.1: cannot open shared object file: no such file or directory: unknown"
    )
    ctx = context_factory(state=build_state(sysbox_runtime=False, dind_probe_error=_cause(stderr)))

    result = await SysboxRequiredCheck().run(ctx)

    assert result.event.reason_code == "NVIDIA_CONTAINER_HOOK_FAILED"
    assert "load library failed" in result.event.remediation
    assert "Install the sysbox runtime" not in result.event.remediation


@pytest.mark.asyncio
async def test_sysbox_mount_error_keeps_sysbox_required_missing(context_factory):
    """sysbox's shiftfs fallback breaks the hook's mount step: that is a sysbox problem."""
    stderr = (
        "docker: Error response from daemon: OCI runtime create failed: error running hook #0: exit status 1, "
        "stdout: , stderr: nvidia-container-cli: mount error: file lookup failed: "
        "/var/lib/sysbox/shiftfs/0a1b2c3d/merged/proc/driver/nvidia: no such file or directory"
    )
    cause = diagnose_docker_run_error(stderr)
    ctx = context_factory(state=build_state(sysbox_runtime=False, dind_probe_error=cause))

    result = await SysboxRequiredCheck().run(ctx)

    assert cause is None
    assert result.event.reason_code == Msg.SYSBOX_MISSING.reason
    assert result.event.remediation == Msg.SYSBOX_MISSING.remediation


@pytest.mark.asyncio
async def test_nvidia_hook_refusal_on_a_rented_executor_still_passes(context_factory):
    """Live rentals are never disrupted, whatever the probe's cause."""
    ctx = context_factory(
        state=build_state(
            sysbox_runtime=False, rented_data=_rented_data(), dind_probe_error=_cause(NVIDIA_MISMATCH_STDERR)
        )
    )

    result = await SysboxRequiredCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.SYSBOX_OK.reason


@pytest.mark.asyncio
async def test_measured_no_sysbox_keeps_sysbox_required_missing(context_factory):
    """The probe ran, `docker run --rm hello-world` failed under sysbox, no cause was carried: the
    real no-sysbox case keeps its code and its install advice."""
    ctx = context_factory(state=build_state(sysbox_runtime=False, dind_probe_error=None))

    result = await SysboxRequiredCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.SYSBOX_MISSING.reason
    assert result.event.remediation == Msg.SYSBOX_MISSING.remediation


@pytest.mark.asyncio
async def test_dind_cause_from_the_container_log_keeps_sysbox_required_missing(context_factory):
    """DAH-2856's causes are not ours: the code stays SYSBOX_REQUIRED_MISSING and the cause is quoted."""
    cause = DindLogCause("DIND_INNER_DOCKERD_IPTABLES", "the inner dockerd cannot use legacy iptables")
    ctx = context_factory(state=build_state(sysbox_runtime=False, dind_probe_error=cause))

    result = await SysboxRequiredCheck().run(ctx)

    assert result.event.reason_code == Msg.SYSBOX_MISSING.reason
    assert cause.text in result.event.remediation
