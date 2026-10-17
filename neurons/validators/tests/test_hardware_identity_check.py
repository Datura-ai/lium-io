"""DAH-4547 - HardwareIdentityCheck: DMI overlay, NVML vs kernel GPU disagreement, DMI drift; shadow vs enforced."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from neurons.validators.src.services.task.checks.hardware_identity import (
    DMI_ID_PATH,
    HW_ID_FS_SEPARATOR,
    HW_ID_VALUES_SEPARATOR,
    HardwareIdentityCheck,
    dmi_overlay_findings,
    gpu_sources_disagree,
)
from neurons.validators.src.services.task.messages import HardwareIdentityMessages as Msg, render_message
from neurons.validators.src.services.task.pipeline import CheckResult, Pipeline
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod

from tests.helpers import build_services, build_state, default_executor

SETTINGS = "neurons.validators.src.services.task.checks.hardware_identity.settings"
SYSFS_ROW = f"{DMI_ID_PATH}/product_uuid sysfs\n{DMI_ID_PATH}/product_uuid|regular file\n"
SYS_MOUNT = "25 20 0:22 / /sys ro,nosuid - sysfs sysfs ro\n"


def test_clean_sysfs_has_no_overlay():
    assert dmi_overlay_findings(SYS_MOUNT, SYSFS_ROW) == []


def test_bind_mounted_file_over_dmi_is_an_overlay():
    mount = f"40 25 259:2 /tmp/.uuid {DMI_ID_PATH}/product_uuid ro - ext4 /dev/nvme0n1p2 rw\n"
    assert dmi_overlay_findings(SYS_MOUNT + mount, SYSFS_ROW) == [f"{DMI_ID_PATH}/product_uuid ext4"]


def test_file_on_tmpfs_or_symlink_is_an_overlay():
    rows = f"{DMI_ID_PATH}/product_uuid tmpfs\n{DMI_ID_PATH}/board_serial|symbolic link\n"
    assert dmi_overlay_findings("", rows) == [
        f"{DMI_ID_PATH}/product_uuid tmpfs",
        f"{DMI_ID_PATH}/board_serial symbolic link",
    ]


def test_gpu_sources_disagree():
    assert gpu_sources_disagree(["GPU-a"], ["GPU-b"]) is True
    assert gpu_sources_disagree(["GPU-a"], ["GPU-a"]) is False
    assert gpu_sources_disagree(["GPU-a"], None) is False  # unreadable kernel list decides nothing


def _ctx(context_factory, *, stdout, kernel=("GPU-a",), redis=None, rented_data=None, executor=None):
    async def run(cmd, *args, **kwargs):
        return SimpleNamespace(exit_status=0, stdout=stdout, stderr="")

    state = build_state(
        gpu_uuids="GPU-a",
        kernel_gpu_uuids=list(kernel) if kernel is not None else None,
        kernel_gpu_uuids_read_attempted=True,
        rented_data=rented_data,
    )
    services = build_services(redis=redis) if redis is not None else build_services()
    return context_factory(
        state=state, ssh=SimpleNamespace(run=run), services=services, executor=executor or default_executor()
    )


def _stdout(mounts=SYS_MOUNT, rows=SYSFS_ROW, values="uuid-1|serial-1|"):
    return f"{mounts}{HW_ID_FS_SEPARATOR}\n{rows}{HW_ID_VALUES_SEPARATOR}\n{values}"


async def _run(ctx, enforce):
    with patch(SETTINGS) as s:
        s.HARDWARE_FINGERPRINT_ENFORCEMENT_ENABLED = enforce
        return await HardwareIdentityCheck().run(ctx)


@pytest.mark.asyncio
async def test_clean_host_passes_and_logs_fingerprint(context_factory):
    result = await _run(_ctx(context_factory, stdout=_stdout()), enforce=True)
    assert result.passed is True
    assert result.event.reason_code == Msg.OK.reason
    assert result.event.what_we_saw["gpu_fingerprint"]


@pytest.mark.asyncio
@pytest.mark.parametrize("enforce", [False, True])
async def test_overlay_fails_only_when_enforced(context_factory, enforce):
    stdout = _stdout(rows=f"{DMI_ID_PATH}/product_uuid tmpfs\n")
    result = await _run(_ctx(context_factory, stdout=stdout), enforce=enforce)
    assert result.passed is (not enforce)
    assert result.event.reason_code == Msg.SPOOFED.reason
    assert result.event.what_we_saw["findings"] == ["dmi_overlaid"]


@pytest.mark.asyncio
@pytest.mark.parametrize("enforce", [False, True])
async def test_nvml_kernel_mismatch_fails_only_when_enforced(context_factory, enforce):
    result = await _run(_ctx(context_factory, stdout=_stdout(), kernel=("GPU-b",)), enforce=enforce)
    assert result.passed is (not enforce)
    assert result.event.what_we_saw["findings"] == ["gpu_sources_disagree"]


@pytest.mark.asyncio
async def test_dmi_change_under_same_gpus_is_flagged(context_factory):
    ctx = _ctx(context_factory, stdout=_stdout(values="uuid-1|serial-1|"))
    redis = ctx.services.redis
    assert (await _run(ctx, enforce=True)).passed is True

    result = await _run(_ctx(context_factory, stdout=_stdout(values="uuid-2|serial-1|"), redis=redis), enforce=True)
    assert result.passed is False
    assert result.event.what_we_saw["findings"] == ["dmi_changed_under_same_gpus"]


@pytest.mark.asyncio
async def test_unreadable_host_passes(context_factory):
    async def run(cmd, *args, **kwargs):
        raise OSError("ssh closed")

    ctx = context_factory(
        state=build_state(gpu_uuids="GPU-a", kernel_gpu_uuids_read_attempted=True), ssh=SimpleNamespace(run=run)
    )
    result = await _run(ctx, enforce=True)
    assert result.passed is True


class _Next:
    check_id = "test.next"
    fatal = True

    def __init__(self):
        self.ran = False

    async def run(self, ctx):
        self.ran = True
        return CheckResult(passed=True, event=render_message(Msg.OK, ctx=ctx, check_id=self.check_id))


def _rented(executor):
    return RentedExecutorsResponse(
        executors={
            executor.uuid: RentedExecutor(
                miner_hotkey="miner-hotkey",
                executor_ip_address=executor.address,
                executor_ip_port=str(executor.port),
                pods=[RentedPod(pod_id="pod-1", container_name="pod_p1")],
            )
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("rented", [True, False])
async def test_enforced_finding_is_fatal_only_without_a_live_rental(context_factory, rented):
    executor = default_executor()
    ctx = _ctx(
        context_factory,
        stdout=_stdout(rows=f"{DMI_ID_PATH}/product_uuid tmpfs\n"),
        rented_data=_rented(executor) if rented else None,
        executor=executor,
    )
    result = await _run(ctx, enforce=True)
    assert result.passed is False
    assert result.fatal is (not rented)

    nxt = _Next()
    with patch(SETTINGS) as s:
        s.HARDWARE_FINGERPRINT_ENFORCEMENT_ENABLED = True
        await Pipeline(checks=[HardwareIdentityCheck(), nxt], sink=AsyncMock()).run(ctx)
    assert nxt.ran is rented  # with a live pod the pipeline goes on to TenantEnforcementCheck
