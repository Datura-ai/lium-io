from unittest.mock import AsyncMock

import pytest
from neurons.validators.src.services.nvidia_devices import (
    _query_gpu_nodes_for_uuids,
    _query_shared_nodes,
    build_gpu_flags,
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


# ---------------------------- _device_flags ----------------------------


# ---------------------------- remote queries ----------------------------


@pytest.mark.asyncio
async def test_partial_rental_unknown_uuid_raises():
    ssh = fake_ssh(FakeRun("GPU-aaa, 0\n"), FakeRun(""))

    with pytest.raises(RuntimeError, match="not present on executor"):
        await _query_gpu_nodes_for_uuids(ssh, ["GPU-bbb"])


@pytest.mark.asyncio
async def test_query_shared_nodes_skips_host_wide_nodes_on_a_partial_rental():
    ssh = fake_ssh(FakeRun(""))
    await _query_shared_nodes(ssh, is_whole_host_rental=False)

    cmd = ssh.run.call_args.args[0]
    assert "/dev/nvidia-caps" not in cmd
    assert "/dev/nvidia-caps-imex-channels" not in cmd
    # RDMA cards belong to the host too — see DAH-2571
    assert "/dev/infiniband" not in cmd


# ---------------------------- build_gpu_flags ----------------------------


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
    commands = "\n".join(call.args[0] for call in ssh.run.call_args_list)
    assert "nvidia-smi --query-gpu=uuid,minor_number" not in commands


# ---------------------------- build_gpu_flags fallback ----------------------------


