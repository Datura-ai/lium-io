"""nvidia_docker_sysbox_setup.sh: a boot oneshot makes the NVIDIA device nodes before Docker starts the pods."""

import os
import re
import subprocess

SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "nvidia_docker_sysbox_setup.sh"))


def _install(root) -> None:
    text = open(SCRIPT).read()
    match = re.search(r"^create_nvidia_device_nodes_before_docker\(\) \{\n.*?^\}\n", text, re.S | re.M)
    assert match, "create_nvidia_device_nodes_before_docker"
    script = 'host_path() { echo "${SYSBOX_SETUP_HOST_ROOT:-}$1"; }\n' + match.group(0) + "create_nvidia_device_nodes_before_docker\n"
    subprocess.run(["bash", "-c", script], check=True, env={**os.environ, "SYSBOX_SETUP_HOST_ROOT": str(root)})


def test_the_unit_runs_before_docker_and_docker_waits_for_it_without_requiring_it(tmp_path):
    _install(tmp_path)
    unit = (tmp_path / "etc/systemd/system/lium-nvidia-device-nodes.service").read_text()
    assert "Before=docker.service\n" in unit
    assert "ExecStart=/usr/local/sbin/lium-nvidia-device-nodes\n" in unit
    dropin = (tmp_path / "etc/systemd/system/docker.service.d/lium-nvidia-device-nodes.conf").read_text()
    assert "After=lium-nvidia-device-nodes.service\n" in dropin
    assert "Wants=lium-nvidia-device-nodes.service\n" in dropin
    assert "Requires=" not in dropin


def test_every_gpu_node_and_the_shared_nodes_are_made_by_minor(tmp_path):
    # the 19 Nov case: an 8 GPU host whose /dev/nvidia4 did not exist yet when Docker started the pod
    _install(tmp_path)
    gpus = tmp_path / "gpus"
    for minor in range(8):
        (gpus / f"0000:{minor:02x}:00.0").mkdir(parents=True)
        (gpus / f"0000:{minor:02x}:00.0" / "information").write_text(
            f"Model: \t\t NVIDIA H100 80GB HBM3\nIRQ:   \t\t 100\nDevice Minor: \t {minor}\n"
        )
    calls = tmp_path / "calls"
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "nvidia-modprobe").write_text(f'#!/bin/sh\necho "$*" >> {calls}\nexit 1\n')  # a failure stops nothing
    (bindir / "nvidia-modprobe").chmod(0o755)
    out = subprocess.run(
        [str(tmp_path / "usr/local/sbin/lium-nvidia-device-nodes")],
        env={**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}", "LIUM_NVIDIA_GPUS_DIR": str(gpus)},
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    assert calls.read_text().splitlines() == ["-c=255", *(f"-c={m}" for m in range(8)), "-u -c=0", "-m"]


def test_a_host_without_nvidia_modprobe_exits_cleanly(tmp_path):
    _install(tmp_path)
    out = subprocess.run(
        [str(tmp_path / "usr/local/sbin/lium-nvidia-device-nodes")], env={"PATH": "/nonexistent"}, capture_output=True
    )
    assert out.returncode == 0
