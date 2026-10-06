from unittest.mock import AsyncMock

import pytest
from neurons.validators.src.services.nvidia_devices import (
    _device_flags,
    _gpus_flag,
    _parse_nvidia_smi_xml_minor_map,
    _query_all_gpu_nodes,
    _query_gpu_nodes_for_uuids,
    _query_shared_nodes,
    build_gpu_flags,
)

PROC_GPU_INFO_OUTPUT = "GPU-aaa, 0\nGPU-bbb, 1\nGPU-ccc, 2\n"

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


def test_gpus_flag_no_uuids_emits_all():
    assert _gpus_flag(None) == "--gpus all"
    assert _gpus_flag([]) == "--gpus all"


def test_gpus_flag_single_uuid():
    assert _gpus_flag(["GPU-aaa"]) == '--gpus \'"device=GPU-aaa"\''


def test_gpus_flag_multiple_uuids_joined_with_comma():
    assert _gpus_flag(["GPU-a", "GPU-b"]) == '--gpus \'"device=GPU-a,GPU-b"\''


# ---------------------------- _device_flags ----------------------------


def test_device_flags_renders_nodes_in_order():
    assert _device_flags(
        ("/dev/nvidia0", "/dev/nvidia1", "/dev/nvidiactl", "/dev/nvidia-uvm")
    ) == (
        "--device=/dev/nvidia0 --device=/dev/nvidia1 "
        "--device=/dev/nvidiactl --device=/dev/nvidia-uvm"
    )


def test_device_flags_with_no_nodes_is_empty_string():
    assert _device_flags(()) == ""


# ---------------------------- remote queries ----------------------------


@pytest.mark.asyncio
async def test_whole_host_rental_enumerates_all_minors():
    ssh = fake_ssh(FakeRun("/dev/nvidia0\n/dev/nvidia1\n"))
    nodes = await _query_all_gpu_nodes(ssh)

    assert nodes == ("/dev/nvidia0", "/dev/nvidia1")


@pytest.mark.asyncio
async def test_partial_rental_resolves_uuid_to_minor():
    ssh = fake_ssh(FakeRun(PROC_GPU_INFO_OUTPUT))
    nodes, host_total = await _query_gpu_nodes_for_uuids(ssh, ["GPU-bbb"])

    assert nodes == ("/dev/nvidia1",)
    assert host_total == 3


@pytest.mark.asyncio
async def test_partial_rental_preserves_uuid_order_in_per_gpu():
    ssh = fake_ssh(FakeRun("GPU-aaa, 4\nGPU-bbb, 7\nGPU-ccc, 2\n"))
    nodes, _ = await _query_gpu_nodes_for_uuids(ssh, ["GPU-ccc", "GPU-aaa"])

    assert nodes == ("/dev/nvidia2", "/dev/nvidia4")


@pytest.mark.asyncio
async def test_partial_rental_falls_back_to_nvidia_smi_xml_when_proc_is_empty():
    ssh = fake_ssh(FakeRun(""), FakeRun(NVIDIA_SMI_XML_OUTPUT))
    nodes, host_total = await _query_gpu_nodes_for_uuids(ssh, ["GPU-bbb"])

    assert nodes == ("/dev/nvidia1",)
    assert host_total == 2


@pytest.mark.asyncio
async def test_partial_rental_falls_back_to_nvidia_smi_xml_when_proc_is_incomplete():
    ssh = fake_ssh(FakeRun("GPU-aaa, 0\n"), FakeRun(NVIDIA_SMI_XML_OUTPUT))
    nodes, host_total = await _query_gpu_nodes_for_uuids(ssh, ["GPU-bbb"])

    assert nodes == ("/dev/nvidia1",)
    assert host_total == 2


@pytest.mark.asyncio
async def test_partial_rental_unknown_uuid_raises():
    ssh = fake_ssh(FakeRun("GPU-aaa, 0\n"), FakeRun(""))

    with pytest.raises(RuntimeError, match="not present on executor"):
        await _query_gpu_nodes_for_uuids(ssh, ["GPU-bbb"])


@pytest.mark.asyncio
async def test_gpu_minor_discovery_failure_raises_with_stdout_and_stderr():
    ssh = fake_ssh(
        FakeRun(stdout="proc stdout", stderr="proc stderr", exit_status=1),
        FakeRun(
            stdout='Field "minor_number" is not a valid field to query.\n',
            stderr="",
            exit_status=2,
        ),
    )

    with pytest.raises(RuntimeError, match="minor_number"):
        await _query_gpu_nodes_for_uuids(ssh, ["GPU-aaa"])


def test_parse_nvidia_smi_xml_minor_map():
    assert _parse_nvidia_smi_xml_minor_map(NVIDIA_SMI_XML_OUTPUT) == {
        "GPU-aaa": 0,
        "GPU-bbb": 1,
    }


@pytest.mark.asyncio
async def test_hgx_host_shared_query_includes_nvswitch_and_caps_nodes():
    ssh = fake_ssh(
        FakeRun(
            "/dev/nvidiactl\n/dev/nvidia-modeset\n/dev/nvidia-uvm\n/dev/nvidia-uvm-tools\n"
            "/dev/nvidia-nvswitch0\n/dev/nvidia-nvswitch1\n"
            "/dev/nvidia-caps/nvidia-cap1\n/dev/nvidia-caps/nvidia-cap2\n"
        ),
    )
    nodes = await _query_shared_nodes(ssh)

    assert "/dev/nvidia-nvswitch0" in nodes
    assert "/dev/nvidia-caps/nvidia-cap1" in nodes


@pytest.mark.asyncio
async def test_host_without_optional_shared_nodes_has_empty_shared():
    ssh = fake_ssh(FakeRun(""))
    nodes = await _query_shared_nodes(ssh)

    assert nodes == ()


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
async def test_query_shared_nodes_includes_host_wide_nodes_by_default():
    ssh = fake_ssh(FakeRun(""))
    await _query_shared_nodes(ssh)

    cmd = ssh.run.call_args.args[0]
    assert "/dev/nvidia-caps" in cmd
    assert "/dev/nvidia-caps-imex-channels" in cmd
    # RDMA rides the same whole-host branch — see test_infiniband_passthrough.py
    assert "/dev/infiniband/uverbs[0-9]*" in cmd


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
    commands = "\n".join(call.args[0] for call in ssh.run.call_args_list)
    assert "nvidia-smi --query-gpu=uuid,minor_number" not in commands


@pytest.mark.asyncio
async def test_build_gpu_flags_no_shared_nodes_only_per_gpu():
    ssh = fake_ssh(FakeRun("/dev/nvidia0\n"), FakeRun(""))
    flags = await build_gpu_flags(ssh, gpu_uuids=None)

    assert flags == "--gpus all --device=/dev/nvidia0"


@pytest.mark.asyncio
async def test_build_gpu_flags_partial_rental_skips_caps_query():
    # 8-GPU host, tenant rents only 1 — caps must not be probed/forwarded.
    nvidia_smi_csv = "".join(f"GPU-{i:03d}, {i}\n" for i in range(8))
    ssh = fake_ssh(
        FakeRun(nvidia_smi_csv),
        FakeRun("/dev/nvidiactl\n/dev/nvidia-uvm\n"),
    )
    flags = await build_gpu_flags(ssh, gpu_uuids=["GPU-003"])

    # Inspect the shared-nodes probe — it must not look at /dev/nvidia-caps.
    shared_probe_cmd = ssh.run.call_args_list[1].args[0]
    assert "/dev/nvidia-caps" not in shared_probe_cmd

    # Per-GPU minor is correct, no cap-node leak.
    assert (
        flags == '--gpus \'"device=GPU-003"\' '
        "--device=/dev/nvidia3 --device=/dev/nvidiactl --device=/dev/nvidia-uvm"
    )


@pytest.mark.asyncio
async def test_build_gpu_flags_whole_host_via_explicit_uuids_keeps_caps():
    # Tenant rents ALL host GPUs explicitly — not partial, caps should be probed.
    ssh = fake_ssh(
        FakeRun("GPU-aaa, 0\nGPU-bbb, 1\n"),
        FakeRun("/dev/nvidiactl\n/dev/nvidia-caps/nvidia-cap1\n"),
    )
    flags = await build_gpu_flags(ssh, gpu_uuids=["GPU-aaa", "GPU-bbb"])

    shared_probe_cmd = ssh.run.call_args_list[1].args[0]
    assert "/dev/nvidia-caps" in shared_probe_cmd
    assert "/dev/nvidia-caps/nvidia-cap1" in flags


# ---------------------------- build_gpu_flags fallback ----------------------------


@pytest.mark.asyncio
async def test_build_gpu_flags_falls_back_when_nvidia_smi_fails(caplog):
    # Partial rental → both minor-discovery paths fail.
    ssh = fake_ssh(
        FakeRun(stderr="proc unavailable", exit_status=1),
        FakeRun(stderr="nvidia-smi: not found", exit_status=127),
    )

    with caplog.at_level("WARNING"):
        flags = await build_gpu_flags(ssh, gpu_uuids=["GPU-aaa"])

    # Falls back to legacy --gpus only — no --device flags.
    assert flags == '--gpus \'"device=GPU-aaa"\''
    assert any(
        "probe failed, falling back to legacy --gpus only" in rec.message
        for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_build_gpu_flags_falls_back_when_uuid_unknown(caplog):
    ssh = fake_ssh(FakeRun("GPU-aaa, 0\n"))  # GPU-bbb is not present

    with caplog.at_level("WARNING"):
        flags = await build_gpu_flags(ssh, gpu_uuids=["GPU-bbb"])

    assert flags == '--gpus \'"device=GPU-bbb"\''
    assert any("probe failed" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_build_gpu_flags_falls_back_when_ssh_raises(caplog):
    ssh = AsyncMock()
    ssh.run.side_effect = ConnectionError("ssh transport closed")

    with caplog.at_level("WARNING"):
        flags = await build_gpu_flags(ssh, gpu_uuids=None)

    # Whole-host rental falls back to plain `--gpus all`.
    assert flags == "--gpus all"
    assert any("probe failed" in rec.message for rec in caplog.records)
