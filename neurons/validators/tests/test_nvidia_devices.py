from unittest.mock import AsyncMock

import pytest

from neurons.validators.src.services.nvidia_devices import (
    NvidiaDevicePlan,
    _gpus_flag,
    build_gpu_flags,
    discover_nvidia_devices,
)


class FakeRun:
    def __init__(self, stdout: str = "", stderr: str = "", exit_status: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.exit_status = exit_status


def fake_ssh(*responses: FakeRun) -> AsyncMock:
    ssh = AsyncMock()
    ssh.run.side_effect = responses
    return ssh


# ---------------------------- _gpus_flag (pure) ----------------------------


def test_gpus_flag_no_uuids_emits_all():
    assert _gpus_flag(None) == "--gpus all"
    assert _gpus_flag([]) == "--gpus all"


def test_gpus_flag_single_uuid():
    assert _gpus_flag(["GPU-aaa"]) == '--gpus \'"device=GPU-aaa"\''


def test_gpus_flag_multiple_uuids_joined_with_comma():
    assert _gpus_flag(["GPU-a", "GPU-b"]) == '--gpus \'"device=GPU-a,GPU-b"\''


# ---------------------------- NvidiaDevicePlan.as_device_flags ----------------------------


def test_plan_renders_per_gpu_then_shared():
    plan = NvidiaDevicePlan(
        per_gpu=("/dev/nvidia0", "/dev/nvidia1"),
        shared=("/dev/nvidiactl", "/dev/nvidia-uvm"),
    )
    assert plan.as_device_flags() == (
        "--device=/dev/nvidia0 --device=/dev/nvidia1 "
        "--device=/dev/nvidiactl --device=/dev/nvidia-uvm"
    )


def test_plan_with_no_nodes_is_empty_string():
    assert NvidiaDevicePlan(per_gpu=(), shared=()).as_device_flags() == ""


# ---------------------------- discover_nvidia_devices ----------------------------


@pytest.mark.asyncio
async def test_whole_host_rental_enumerates_all_minors():
    ssh = fake_ssh(
        FakeRun("/dev/nvidia0\n/dev/nvidia1\n"),
        FakeRun("/dev/nvidiactl\n/dev/nvidia-uvm\n"),
    )
    plan = await discover_nvidia_devices(ssh, gpu_uuids=None)

    assert plan.per_gpu == ("/dev/nvidia0", "/dev/nvidia1")
    assert plan.shared == ("/dev/nvidiactl", "/dev/nvidia-uvm")


@pytest.mark.asyncio
async def test_partial_rental_resolves_uuid_to_minor():
    ssh = fake_ssh(
        FakeRun("GPU-aaa, 0\nGPU-bbb, 1\nGPU-ccc, 2\n"),
        FakeRun(""),
    )
    plan = await discover_nvidia_devices(ssh, gpu_uuids=["GPU-bbb"])

    assert plan.per_gpu == ("/dev/nvidia1",)


@pytest.mark.asyncio
async def test_partial_rental_preserves_uuid_order_in_per_gpu():
    ssh = fake_ssh(
        FakeRun("GPU-aaa, 0\nGPU-bbb, 1\nGPU-ccc, 2\n"),
        FakeRun(""),
    )
    plan = await discover_nvidia_devices(ssh, gpu_uuids=["GPU-ccc", "GPU-aaa"])

    assert plan.per_gpu == ("/dev/nvidia2", "/dev/nvidia0")


@pytest.mark.asyncio
async def test_partial_rental_unknown_uuid_raises():
    ssh = fake_ssh(FakeRun("GPU-aaa, 0\n"))

    with pytest.raises(RuntimeError, match="not present on executor"):
        await discover_nvidia_devices(ssh, gpu_uuids=["GPU-bbb"])


@pytest.mark.asyncio
async def test_nvidia_smi_failure_raises():
    ssh = fake_ssh(FakeRun(stderr="nvidia-smi: not found", exit_status=127))

    with pytest.raises(RuntimeError, match="nvidia-smi query failed"):
        await discover_nvidia_devices(ssh, gpu_uuids=["GPU-aaa"])


@pytest.mark.asyncio
async def test_hgx_host_with_nvswitch_and_caps_nodes():
    ssh = fake_ssh(
        FakeRun("/dev/nvidia0\n/dev/nvidia1\n/dev/nvidia2\n/dev/nvidia3\n"),
        FakeRun(
            "/dev/nvidiactl\n/dev/nvidia-modeset\n/dev/nvidia-uvm\n/dev/nvidia-uvm-tools\n"
            "/dev/nvidia-nvswitch0\n/dev/nvidia-nvswitch1\n"
            "/dev/nvidia-caps/nvidia-cap1\n/dev/nvidia-caps/nvidia-cap2\n"
        ),
    )
    plan = await discover_nvidia_devices(ssh, gpu_uuids=None)

    assert plan.per_gpu == (
        "/dev/nvidia0",
        "/dev/nvidia1",
        "/dev/nvidia2",
        "/dev/nvidia3",
    )
    assert "/dev/nvidia-nvswitch0" in plan.shared
    assert "/dev/nvidia-caps/nvidia-cap1" in plan.shared


@pytest.mark.asyncio
async def test_host_without_optional_shared_nodes_has_empty_shared():
    ssh = fake_ssh(FakeRun("/dev/nvidia0\n"), FakeRun(""))
    plan = await discover_nvidia_devices(ssh, gpu_uuids=None)

    assert plan.per_gpu == ("/dev/nvidia0",)
    assert plan.shared == ()


# ---------------------------- build_gpu_flags ----------------------------


@pytest.mark.asyncio
async def test_build_gpu_flags_whole_host():
    ssh = fake_ssh(
        FakeRun("/dev/nvidia0\n"),
        FakeRun("/dev/nvidiactl\n"),
    )
    flags = await build_gpu_flags(ssh, gpu_uuids=None)

    assert flags == "--gpus all --device=/dev/nvidia0 --device=/dev/nvidiactl"


@pytest.mark.asyncio
async def test_build_gpu_flags_partial_rental():
    ssh = fake_ssh(
        FakeRun("GPU-aaa, 0\nGPU-bbb, 1\n"),
        FakeRun("/dev/nvidiactl\n"),
    )
    flags = await build_gpu_flags(ssh, gpu_uuids=["GPU-bbb"])

    assert flags == (
        '--gpus \'"device=GPU-bbb"\' --device=/dev/nvidia1 --device=/dev/nvidiactl'
    )


@pytest.mark.asyncio
async def test_build_gpu_flags_no_shared_nodes_only_per_gpu():
    ssh = fake_ssh(FakeRun("/dev/nvidia0\n"), FakeRun(""))
    flags = await build_gpu_flags(ssh, gpu_uuids=None)

    assert flags == "--gpus all --device=/dev/nvidia0"
