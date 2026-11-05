"""DAH-2671 item 1 — surface the kernel (procfs) GPU verdict + refuse the spoofable XML fallback.

RED→GREEN:
  1a — before this change the missing-GPU verdict died as a bare WARNING; now it emits a stable,
       greppable structured event carrying requested/visible counts and both full UUID lists.
  1b — before this change the NVML-backed nvidia-smi XML always replaced the kernel map; now a
       kernel-vs-XML disagreement is emitted and the overwrite is refused
       (fail closed) so the kernel truth stands.
"""
from unittest.mock import AsyncMock

import pytest
from neurons.validators.src.services.nvidia_devices import (
    MissingRentedGpuError,
    _query_gpu_nodes_for_uuids,
    build_gpu_docker_config_for_executor,
)

import services.nvidia_devices as nd

_XML_BOTH = """\
<?xml version="1.0" ?>
<nvidia_smi_log>
    <gpu id="00000000:02:00.0"><uuid>GPU-aaa</uuid><minor_number>0</minor_number></gpu>
    <gpu id="00000000:03:00.0"><uuid>GPU-bbb</uuid><minor_number>1</minor_number></gpu>
</nvidia_smi_log>
"""


class FakeRun:
    def __init__(self, stdout="", stderr="", exit_status=0):
        self.stdout = stdout
        self.stderr = stderr
        self.exit_status = exit_status


def _fake_ssh(*responses):
    ssh = AsyncMock()
    ssh.run.side_effect = responses
    ssh.get_extra_info = lambda *a, **k: ("1.2.3.4", 22)
    return ssh


# ---------------------------- 1a: missing-GPU verdict emits ----------------------------


@pytest.mark.asyncio
async def test_missing_rented_gpu_emits_structured_event_and_refuses_the_container(caplog):
    # proc lists only GPU-aaa; XML empty → GPU-bbb absent everywhere → MissingRentedGpuError.
    # The verdict must reach the caller, not be swallowed into the legacy --gpus fallback.
    ssh = _fake_ssh(FakeRun("GPU-aaa, 0\n"), FakeRun(""))

    with caplog.at_level("WARNING"):
        with pytest.raises(MissingRentedGpuError):
            await build_gpu_docker_config_for_executor(
                ssh, ["GPU-bbb"], executor_id="exec-9", default_extra={"executor_uuid": "exec-9"}
            )

    rec = next(r for r in caplog.records if r.message == nd._KERNEL_GPU_VERDICT_MISSING_MSG)
    assert rec.requested_gpu_count == 1
    assert rec.visible_gpu_count == 1
    assert rec.requested_uuids == ["GPU-bbb"]
    assert rec.visible_uuids == ["GPU-aaa"]
    assert rec.missing_uuids == ["GPU-bbb"]
    assert rec.executor_id == "exec-9"


@pytest.mark.asyncio
async def test_disagreement_fails_closed_and_emits(caplog):
    # proc readable but missing GPU-bbb; XML supplies it. The XML may not overwrite the kernel
    # truth, so the missing verdict fires.
    ssh = _fake_ssh(FakeRun("GPU-aaa, 0\n"), FakeRun(_XML_BOTH))

    with caplog.at_level("WARNING"):
        with pytest.raises(MissingRentedGpuError):
            await _query_gpu_nodes_for_uuids(
                ssh, ["GPU-bbb"], executor_id="exec-9", default_extra={"miner_hotkey": "hk-1"}
            )

    rec = next(r for r in caplog.records if r.message == nd._KERNEL_GPU_VERDICT_DISAGREE_MSG)
    assert rec.proc_unreadable is False
    assert rec.enforced is True
    assert rec.xml_would_add_uuids == ["GPU-bbb"]
    # the event has to be correlatable by miner/batch, not just by host:port
    assert rec.executor_id == "exec-9"
    assert rec.miner_hotkey == "hk-1"


