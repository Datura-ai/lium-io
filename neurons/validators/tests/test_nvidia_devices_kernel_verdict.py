"""DAH-2671 item 1 — surface the kernel (procfs) GPU verdict + refuse the spoofable XML fallback.

RED→GREEN:
  1a — before this change the missing-GPU verdict died as a bare WARNING; now it emits a stable,
       greppable structured event carrying requested/visible counts and both full UUID lists.
  1b — before this change the NVML-backed nvidia-smi XML always replaced the kernel map; now a
       kernel-vs-XML disagreement is emitted, and under enforcement the overwrite is refused
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


@pytest.fixture(autouse=True)
def _shadow_flags(monkeypatch):
    # default posture for every test: master on, enforcement off; tests flip enforcement explicitly.
    monkeypatch.setattr(nd.settings, "KERNEL_GPU_VERDICT_CHECK_ENABLED", True, raising=False)
    monkeypatch.setattr(nd.settings, "KERNEL_GPU_VERDICT_ENFORCEMENT_ENABLED", False, raising=False)


# ---------------------------- 1a: missing-GPU verdict emits ----------------------------


@pytest.mark.asyncio
async def test_missing_rented_gpu_emits_structured_event(caplog):
    # proc lists only GPU-aaa; XML empty → GPU-bbb absent everywhere → MissingRentedGpuError.
    ssh = _fake_ssh(FakeRun("GPU-aaa, 0\n"), FakeRun(""))

    with caplog.at_level("WARNING"):
        config = await build_gpu_docker_config_for_executor(
            ssh, ["GPU-bbb"], executor_id="exec-9", default_extra={"executor_uuid": "exec-9"}
        )

    # Behavior unchanged: still falls back to the legacy --gpus config (no device mounts).
    assert config.device_mounts == ()

    rec = next(r for r in caplog.records if r.message == nd._KERNEL_GPU_VERDICT_MISSING_MSG)
    assert rec.requested_gpu_count == 1
    assert rec.visible_gpu_count == 1
    assert rec.requested_uuids == ["GPU-bbb"]
    assert rec.visible_uuids == ["GPU-aaa"]
    assert rec.missing_uuids == ["GPU-bbb"]
    assert rec.executor_id == "exec-9"


@pytest.mark.asyncio
async def test_missing_rented_gpu_under_enforcement_refuses_the_container(monkeypatch, caplog):
    # The verdict must reach the caller, not be swallowed into the legacy --gpus fallback: otherwise
    # enforcement only drops the --device mounts and the container is still created.
    monkeypatch.setattr(nd.settings, "KERNEL_GPU_VERDICT_ENFORCEMENT_ENABLED", True, raising=False)
    ssh = _fake_ssh(FakeRun("GPU-aaa, 0\n"), FakeRun(""))

    with caplog.at_level("WARNING"):
        with pytest.raises(MissingRentedGpuError):
            await build_gpu_docker_config_for_executor(ssh, ["GPU-bbb"], executor_id="exec-9")

    assert any(r.message == nd._KERNEL_GPU_VERDICT_MISSING_MSG for r in caplog.records)


# ---------------------------- 1b: kernel-vs-XML disagreement ----------------------------


@pytest.mark.asyncio
async def test_disagreement_shadow_preserves_behavior_and_emits(caplog):
    # proc readable but missing GPU-bbb; XML supplies it. Shadow keeps today's behavior: XML wins.
    ssh = _fake_ssh(FakeRun("GPU-aaa, 0\n"), FakeRun(_XML_BOTH))

    with caplog.at_level("WARNING"):
        nodes, host_total = await _query_gpu_nodes_for_uuids(ssh, ["GPU-bbb"])

    assert nodes == ("/dev/nvidia1",)  # XML overwrote the kernel map — placement unchanged
    assert host_total == 2

    rec = next(r for r in caplog.records if r.message == nd._KERNEL_GPU_VERDICT_DISAGREE_MSG)
    assert rec.proc_unreadable is False
    assert rec.enforced is False
    assert rec.xml_would_add_uuids == ["GPU-bbb"]


@pytest.mark.asyncio
async def test_disagreement_enforcement_fails_closed(monkeypatch, caplog):
    monkeypatch.setattr(nd.settings, "KERNEL_GPU_VERDICT_ENFORCEMENT_ENABLED", True, raising=False)
    ssh = _fake_ssh(FakeRun("GPU-aaa, 0\n"), FakeRun(_XML_BOTH))

    with caplog.at_level("WARNING"):
        # kernel truth stands (XML not allowed to overwrite) → the missing verdict fires.
        with pytest.raises(MissingRentedGpuError):
            await _query_gpu_nodes_for_uuids(ssh, ["GPU-bbb"])

    rec = next(r for r in caplog.records if r.message == nd._KERNEL_GPU_VERDICT_DISAGREE_MSG)
    assert rec.enforced is True


@pytest.mark.asyncio
async def test_unreadable_proc_recorded_distinctly(monkeypatch, caplog):
    # honest casualty: /proc map unreadable (non-zero exit) while XML answers. Enforcement refuses
    # the overwrite and records proc_unreadable=True so ops can tell it apart from a real spoof.
    monkeypatch.setattr(nd.settings, "KERNEL_GPU_VERDICT_ENFORCEMENT_ENABLED", True, raising=False)
    ssh = _fake_ssh(FakeRun("proc boom", "err", exit_status=1), FakeRun(_XML_BOTH))

    with caplog.at_level("WARNING"):
        with pytest.raises(RuntimeError):
            await _query_gpu_nodes_for_uuids(ssh, ["GPU-bbb"])

    rec = next(r for r in caplog.records if r.message == nd._KERNEL_GPU_VERDICT_DISAGREE_MSG)
    assert rec.proc_unreadable is True
    assert rec.enforced is True


@pytest.mark.asyncio
async def test_check_disabled_preserves_legacy_behavior(monkeypatch, caplog):
    # master switch off → no emit, XML overwrites exactly as before the change.
    monkeypatch.setattr(nd.settings, "KERNEL_GPU_VERDICT_CHECK_ENABLED", False, raising=False)
    ssh = _fake_ssh(FakeRun("GPU-aaa, 0\n"), FakeRun(_XML_BOTH))

    with caplog.at_level("WARNING"):
        nodes, _ = await _query_gpu_nodes_for_uuids(ssh, ["GPU-bbb"])

    assert nodes == ("/dev/nvidia1",)
    assert not any(r.message == nd._KERNEL_GPU_VERDICT_DISAGREE_MSG for r in caplog.records)
