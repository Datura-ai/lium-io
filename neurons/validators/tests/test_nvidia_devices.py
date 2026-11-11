from unittest.mock import AsyncMock

import pytest
from neurons.validators.src.services.nvidia_devices import (
    MissingRentedGpuError,
    _query_gpu_nodes_for_uuids,
    _query_shared_nodes,
    build_gpu_docker_config_for_executor,
    build_gpu_flags,
)


NVIDIA_SMI_XML_OUTPUT = """\
<?xml version="1.0" ?>
<nvidia_smi_log>
    <gpu id="00000000:02:00.0">
        <uuid>GPU-aaa</uuid>
        <minor_number>0</minor_number>
    </gpu>
    <gpu id="00000000:03:00.0">
        <uuid>GPU-bbb</uuid>
        <minor_number>1</minor_number>
    </gpu>
</nvidia_smi_log>
"""


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
async def test_partial_rental_nvidia_smi_xml_cannot_replace_empty_proc():
    # the kernel (procfs) list is the truth; the spoofable XML may not supply a missing card
    ssh = fake_ssh(FakeRun(""), FakeRun(NVIDIA_SMI_XML_OUTPUT))
    with pytest.raises(MissingRentedGpuError):
        await _query_gpu_nodes_for_uuids(ssh, ["GPU-bbb"])


@pytest.mark.asyncio
async def test_partial_rental_nvidia_smi_xml_cannot_fill_incomplete_proc():
    ssh = fake_ssh(FakeRun("GPU-aaa, 0\n"), FakeRun(NVIDIA_SMI_XML_OUTPUT))
    with pytest.raises(MissingRentedGpuError):
        await _query_gpu_nodes_for_uuids(ssh, ["GPU-bbb"])


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


@pytest.mark.asyncio
@pytest.mark.parametrize(("whole_host", "gets_host_wide_nodes"), [(None, True), (True, True), (False, False)])
async def test_the_backend_marking_a_rental_partial_withholds_host_wide_nodes(whole_host, gets_host_wide_nodes):
    # every GPU the host shows is requested, but GPUs off the bus at this create are not shown (protocol 1.18.0)
    ssh = fake_ssh(FakeRun("GPU-aaa, 0\nGPU-bbb, 1\n"), FakeRun(""))

    await build_gpu_docker_config_for_executor(ssh, ["GPU-aaa", "GPU-bbb"], whole_host=whole_host)

    shared_nodes_cmd = ssh.run.call_args_list[-1].args[0]
    assert ("/dev/infiniband" in shared_nodes_cmd) is gets_host_wide_nodes


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




def _shared_nodes_on_fake_host(tmp_path, *, nvswitch_devices: list[str]) -> list[str]:
    # Runs the real shell command against a fake /dev and /proc under tmp_path.
    import subprocess

    from neurons.validators.src.services.nvidia_devices import shared_device_nodes_command

    for node in ("nvidiactl", "nvidia-uvm", "nvidia-nvswitchctl", "nvidia-nvswitch0"):
        (tmp_path / "dev").mkdir(exist_ok=True)
        (tmp_path / "dev" / node).touch()
    switches = tmp_path / "proc" / "driver" / "nvidia-nvswitch" / "devices"
    switches.mkdir(parents=True)
    for pci_address in nvswitch_devices:
        (switches / pci_address).mkdir()
    command = shared_device_nodes_command(is_whole_host_rental=False)
    command = command.replace("/dev/", f"{tmp_path}/dev/").replace("/proc/", f"{tmp_path}/proc/")
    output = subprocess.run(["sh", "-c", command], capture_output=True, text=True).stdout
    return [line.removeprefix(str(tmp_path)) for line in output.splitlines()]


def test_shared_nodes_skip_nvswitch_nodes_on_a_host_without_nvswitch(tmp_path):
    # DAH-6184: nvidia-modprobe creates /dev/nvidia-nvswitchctl inside the executor container on
    # every host; forwarding it where the host has no switch fails the create in dockerd.
    nodes = _shared_nodes_on_fake_host(tmp_path, nvswitch_devices=[])

    assert nodes == ["/dev/nvidiactl", "/dev/nvidia-uvm"]


def test_shared_nodes_keep_nvswitch_nodes_on_a_host_with_nvswitch(tmp_path):
    nodes = _shared_nodes_on_fake_host(tmp_path, nvswitch_devices=["0000:05:00.0"])

    assert nodes == ["/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-nvswitchctl", "/dev/nvidia-nvswitch0"]
