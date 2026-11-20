import ctypes.util
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from services import rental_gpu_proof as proof
from services.rental_gpu_proof import PROOF_MARKER, RentalGpuProofFailed, judge_gpu_proof, prove_rented_gpus

A = "GPU-11111111-2222-3333-4444-555555555555"
B = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def _out(cards, init_rc=0):
    return "noise\n" + PROOF_MARKER + json.dumps({"init_rc": init_rc, "cards": cards}) + "\n"


class FakeSsh:
    def __init__(self, *results):
        self.results = list(results)
        self.commands = []

    async def run(self, command, timeout=None, check=False):
        self.commands.append(command)
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def test_every_rented_gpu_usable_passes():
    assert judge_gpu_proof([A, B.upper()], _out([{"uuid": A, "rc": 0}, {"uuid": B, "rc": 0}])) == ""


def test_busy_card_counts_as_present():
    assert judge_gpu_proof([A, B], _out([{"uuid": A, "rc": 2}, {"uuid": B, "rc": 46}])) == ""


def test_short_count_fails_with_count_only():
    # 8 listed, CUDA sees 1
    uuids = [A] + [f"GPU-{i:08d}-0000-0000-0000-000000000000" for i in range(7)]
    reason = judge_gpu_proof(uuids, _out([{"uuid": A, "rc": 0}]))
    assert reason.startswith("CUDA can use 1 of the 8 rented GPUs")
    assert "GPU-0000" not in reason


def test_card_that_fails_the_op_fails():
    assert "1 of the 2" in judge_gpu_proof([A, B], _out([{"uuid": A, "rc": 0}, {"uuid": B, "rc": 999}]))


def test_the_images_own_init_text_never_reaches_the_error():
    # the image's python3 writes the answer; the backend reads the error for GPU-fault text
    forged = judge_gpu_proof([A, B], _out([{"uuid": A, "rc": 0}], init_rc="GPU has fallen off the bus"))
    assert forged == "CUDA can use 1 of the 2 rented GPUs (cuInit=?, CUDA devices=1)"
    assert "cuInit=no_libcuda" in judge_gpu_proof([A], _out([], init_rc="no_libcuda"))
    assert "cuInit=100" in judge_gpu_proof([A], _out([], init_rc=100))


def test_no_answer_fails():
    assert judge_gpu_proof([A], "Segmentation fault") == "the CUDA probe gave no answer"
    assert judge_gpu_proof([A], PROOF_MARKER + "{bad") == "the CUDA probe gave an unreadable answer"


@pytest.mark.asyncio
async def test_container_pass_runs_once_and_unsets_cuda_visible_devices():
    ssh = FakeSsh(SimpleNamespace(exit_status=0, stdout=_out([{"uuid": A, "rc": 0}])))
    await prove_rented_gpus(ssh, container="c1", python_path="/py", gpu_uuids=[A], log_extra={})
    assert len(ssh.commands) == 1
    assert "docker exec -u 0 c1 env -u CUDA_VISIBLE_DEVICES python3 -c" in ssh.commands[0]


@pytest.mark.asyncio
async def test_image_without_python_falls_back_to_host_pinned_to_rented_uuids():
    ssh = FakeSsh(
        SimpleNamespace(exit_status=127, stdout=""),
        SimpleNamespace(exit_status=0, stdout=_out([{"uuid": A, "rc": 0}, {"uuid": B, "rc": 0}])),
    )
    await prove_rented_gpus(ssh, container="c1", python_path="/py", gpu_uuids=[A, B], log_extra={})
    assert ssh.commands[1].startswith(f"CUDA_VISIBLE_DEVICES={A},{B} /py -c ")


@pytest.mark.asyncio
async def test_short_count_raises():
    ssh = FakeSsh(SimpleNamespace(exit_status=0, stdout=_out([{"uuid": A, "rc": 0}])))
    with pytest.raises(RentalGpuProofFailed, match="CUDA can use 1 of the 2"):
        await prove_rented_gpus(ssh, container="c1", python_path="/py", gpu_uuids=[A, B], log_extra={})


@pytest.mark.asyncio
async def test_timeout_raises():
    ssh = FakeSsh(TimeoutError())
    with pytest.raises(RentalGpuProofFailed, match="did not finish"):
        await prove_rented_gpus(ssh, container="c1", python_path="/py", gpu_uuids=[A], log_extra={})


@pytest.mark.asyncio
async def test_whole_node_pod_without_uuids_is_not_probed():
    ssh = FakeSsh()
    await prove_rented_gpus(ssh, container="c1", python_path="/py", gpu_uuids=[], log_extra={})
    assert ssh.commands == []


def test_probe_code_runs_without_a_gpu():
    # the shipped probe script itself: compiles, and with no libcuda (or no device) prints a verdict, exit 0
    out = subprocess.run([sys.executable, "-c", proof._PROBE_CODE], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert PROOF_MARKER in out.stdout
    if not ctypes.util.find_library("cuda"):
        assert judge_gpu_proof([A], out.stdout).startswith("CUDA can use 0 of the 1")
