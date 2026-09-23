"""ticket-0331 - check_storage_limit_ability() in machine_scrape.py: `--storage-opt` alone passed on a host
where every rental volume failed to mount into its sysbox container ("setting up ID-mapped mount on path
206/fs"), so the check now also creates a vloopback volume, mounts it into a sysbox container, writes, reads
back and removes it. A failure is storage_limit_supported=false with "<REASON_CODE>: <detail>".

machine_scrape.py is a script, not a module - importing it runs the whole scrape - so the helpers are
extracted by ast and executed in their own namespace, with `subprocess` replaced by a model of the host's
docker CLI (same pattern as test_scrape_gpu_power_cap_probe.py).
"""

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from neurons.validators.tests.helpers import build_scrape_namespace

SRC = Path(__file__).resolve().parents[1] / "src"

STORAGE_HELPERS = {
    "COMMANDS",
    "VLOOPBACK_DRIVER_PREFIX",
    "VLOOPBACK_PLUGIN",
    "VLOOPBACK_PLUGIN_IMAGE",
    "VLOOPBACK_PROBE_IMAGE",
    "VLOOPBACK_PROBE_TOKEN",
    "VLOOPBACK_PLUGIN_INSTALL_TIMEOUT_SECONDS",
    "VLOOPBACK_COMMAND_TIMEOUT_SECONDS",
    "STORAGE_PROBE_DETAIL_CAP",
    "run_storage_probe",
    "check_vloopback_volume_ability",
    "check_storage_limit_ability",
}

TICKET_0331_RUNC_ERROR = (
    "docker: Error response from daemon: failed to create task for container: OCI runtime create failed: "
    "error during container init: error setting up ID-mapped mount on path 206/fs (likely means idmapped "
    "mounts are not supported on the filesystem at this path): lstat 206: no such file or directory: unknown."
)


class FakeDockerHost:
    """The docker CLI as the scrape sees it from inside the executor container: a vloopback plugin
    ("<enabled>", "<DATA_DIR>") or None, the volumes it holds, and one knob per way a step can fail."""

    def __init__(
        self,
        *,
        plugin: tuple[str, str] | None = ("true", "/var/lib/docker/loopback"),
        docker_root: str = "/var/lib/docker",
        storage_opt_ok: bool = True,
        mountpoint: str | None = None,
        failing: set[str] | frozenset[str] = frozenset(),
        timing_out: set[str] | frozenset[str] = frozenset(),
    ) -> None:
        self.plugin = plugin
        self.docker_root = docker_root
        self.storage_opt_ok = storage_opt_ok
        self.mountpoint = mountpoint
        self.failing = failing
        self.timing_out = timing_out
        self.volumes: set[str] = set()
        self.calls: list[list[str]] = []

    def run(self, command, stdout=None, stderr=None, text=None, timeout=None):
        self.calls.append(list(command))
        step = self._step(command)
        if step in self.timing_out:
            raise subprocess.TimeoutExpired(command, timeout)
        if step in self.failing:
            return SimpleNamespace(returncode=1, stdout="", stderr=self._error(step, command))
        return self._answer(step, command)

    @staticmethod
    def _step(command: list[str]) -> str:
        if command[:2] == ["docker", "run"]:
            return "run -v" if "-v" in command else "run --storage-opt"
        return " ".join(command[1:3])

    def _error(self, step: str, command: list[str]) -> str:
        return {
            "plugin install": "Error response from daemon: Head https://registry-1.docker.io/v2/ashald/docker-volume-loopback/manifests/latest: TLS handshake timeout",
            "volume create": f"Error response from daemon: create {command[-1]}: VolumeDriver.Create: no space left on device",
            "volume rm": f"Error response from daemon: remove {command[-1]}: VolumeDriver.Unmount: context deadline exceeded",
            "run -v": f"some earlier line\n{TICKET_0331_RUNC_ERROR}",
        }.get(step, "Error response from daemon: failed")

    def _answer(self, step: str, command: list[str]) -> SimpleNamespace:
        stdout = ""
        returncode = 0
        stderr = ""
        if step == "run --storage-opt" and not self.storage_opt_ok:
            returncode, stderr = 125, "docker: Error response from daemon: --storage-opt is supported only for overlay over xfs with 'pquota' mount option."
        elif step == "info --format":
            stdout = f"{self.docker_root}\n"
        elif step == "plugin inspect":
            if self.plugin is None:
                returncode, stdout, stderr = 1, "\n", f"Error: No such plugin: {command[-1]}"
            elif "{{.Enabled}}" in command:
                stdout = f"{self.plugin[0]}\n"
            else:
                stdout = f"DATA_DIR={self.plugin[1]}\nLOG_LEVEL=2\n"
        elif step == "plugin install":
            data_dir = next(arg for arg in command if arg.startswith("DATA_DIR="))
            self.plugin = ("true", data_dir.removeprefix("DATA_DIR="))
        elif step == "volume create":
            self.volumes.add(command[-1])
            stdout = command[-1]
        elif step == "volume inspect":
            stdout = self.mountpoint if self.mountpoint is not None else f"/mnt/{command[-1]}"
        elif step == "volume rm":
            self.volumes.discard(command[-1])
        elif step == "run -v":
            stdout = "lium-vloopback-ok\n"
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    def steps(self) -> list[str]:
        return [self._step(call) for call in self.calls]


def _scrape(host: FakeDockerHost) -> dict[str, Any]:
    fake_subprocess = SimpleNamespace(run=host.run, PIPE=subprocess.PIPE, TimeoutExpired=subprocess.TimeoutExpired)
    return build_scrape_namespace(
        SRC / "miner_jobs" / "machine_scrape.py", STORAGE_HELPERS, {"subprocess": fake_subprocess, "os": os}
    )


def _check(host: FakeDockerHost) -> tuple[bool, str]:
    return _scrape(host)["check_storage_limit_ability"]()


def test_a_host_that_mounts_a_vloopback_volume_in_sysbox_is_supported() -> None:
    # Arrange
    host = FakeDockerHost()

    # Act
    result = _check(host)

    # Assert
    assert result == (True, "Storage limit is supported.")
    assert host.steps() == [
        "run --storage-opt",
        "info --format",
        "plugin inspect",
        "plugin inspect",
        "volume create",
        "volume inspect",
        "run -v",
        "volume rm",
    ]
    create = next(call for call in host.calls if call[1:3] == ["volume", "create"])
    assert create[3:9] == ["-d", "vloopback", "-o", "size=1G", "-o", "sparse=true"]
    run = next(call for call in host.calls if "-v" in call)
    assert "--runtime=sysbox-runc" in run
    assert run[run.index("-v") + 1] == f"{create[-1]}:/lium-vol"
    assert run[-1] == "echo lium-vloopback-ok > /lium-vol/probe && cat /lium-vol/probe"
    assert host.volumes == set()


def test_a_missing_plugin_is_installed_the_way_the_rental_path_installs_it() -> None:
    # Arrange
    host = FakeDockerHost(plugin=None, docker_root="/mnt/lium-xfs/lium-docker")

    # Act
    result = _check(host)

    # Assert
    assert result == (True, "Storage limit is supported.")
    install = next(call for call in host.calls if call[1:3] == ["plugin", "install"])
    assert install == [
        "docker", "plugin", "install", "ashald/docker-volume-loopback",
        "--alias", "vloopback", "--grant-all-permissions", "DATA_DIR=/mnt/lium-xfs/lium-docker/loopback",
    ]


def test_the_ticket_0331_host_reports_the_mount_error_and_removes_the_test_volume() -> None:
    # Arrange: plugin enabled with an absolute DATA_DIR, `--storage-opt` fine, the sysbox mount fails
    host = FakeDockerHost(failing={"run -v"})

    # Act
    supported, reason = _check(host)

    # Assert
    assert supported is False
    assert reason == f"VLOOPBACK_SYSBOX_MOUNT_FAILED: {TICKET_0331_RUNC_ERROR[:300]}"
    assert host.steps()[-1] == "volume rm"
    assert host.volumes == set()


def test_a_relative_mountpoint_fails_before_any_container_starts() -> None:
    # Arrange: what `docker volume inspect t` showed on the ticket-0331 host
    host = FakeDockerHost(mountpoint="206/fs")

    # Act
    supported, reason = _check(host)

    # Assert
    assert supported is False
    assert reason == "VLOOPBACK_MOUNTPOINT_NOT_ABSOLUTE: docker volume inspect gave Mountpoint '206/fs'"
    assert "run -v" not in host.steps()
    assert host.volumes == set()


@pytest.mark.parametrize(
    ("host_kwargs", "expected_reason", "last_step"),
    [
        (
            {"storage_opt_ok": False},
            "STORAGE_OPT_UNSUPPORTED: Storage limit is not supported.",
            "run --storage-opt",
        ),
        (
            {"docker_root": ""},
            "VLOOPBACK_DOCKER_ROOT_UNREADABLE: data-root ''",
            "info --format",
        ),
        (
            {"plugin": ("false", "/var/lib/docker/loopback")},
            "VLOOPBACK_PLUGIN_DISABLED: docker plugin inspect says Enabled='false'",
            "plugin inspect",
        ),
        (
            {"plugin": ("true", "loopback")},
            "VLOOPBACK_DATA_DIR_NOT_ABSOLUTE: the vloopback plugin's DATA_DIR is 'loopback'",
            "plugin inspect",
        ),
        (
            {"plugin": None, "failing": {"plugin install"}},
            "VLOOPBACK_PLUGIN_INSTALL_FAILED: Error response from daemon: Head https://registry-1.docker.io/v2/ashald/docker-volume-loopback/manifests/latest: TLS handshake timeout",
            "plugin install",
        ),
        (
            {"failing": {"volume create"}},
            f"VLOOPBACK_VOLUME_CREATE_FAILED: Error response from daemon: create lium_storage_check_{os.getpid()}: VolumeDriver.Create: no space left on device",
            "volume create",
        ),
        (
            {"failing": {"volume rm"}},
            f"VLOOPBACK_VOLUME_REMOVE_FAILED: Error response from daemon: remove lium_storage_check_{os.getpid()}: VolumeDriver.Unmount: context deadline exceeded",
            "volume rm",
        ),
        (
            {"timing_out": {"run -v"}},
            "VLOOPBACK_SYSBOX_MOUNT_FAILED: docker run --rm timed out after 60s",
            "volume rm",
        ),
        (
            {"timing_out": {"plugin inspect"}},
            "VLOOPBACK_CHECK_TIMEOUT: docker plugin inspect timed out after 60s",
            "plugin inspect",
        ),
        (
            {"timing_out": {"run --storage-opt"}},
            "STORAGE_LIMIT_CHECK_TIMEOUT: Test command timed out.",
            "run --storage-opt",
        ),
    ],
)
def test_each_failing_step_is_reported_with_its_reason_code(
    host_kwargs: dict[str, Any], expected_reason: str, last_step: str
) -> None:
    # Arrange
    host = FakeDockerHost(**host_kwargs)

    # Act
    supported, reason = _check(host)

    # Assert
    assert supported is False
    assert reason == expected_reason
    assert host.steps()[-1] == last_step
    # the test volume never outlives the check, except when removing it is what failed
    assert host.volumes == (set() if "volume rm" not in host.failing else {f"lium_storage_check_{os.getpid()}"})


def test_no_docker_cli_is_an_error_with_its_own_code() -> None:
    # Arrange
    def missing_docker(command, **_):
        raise FileNotFoundError(2, "No such file or directory", "docker")

    fake_subprocess = SimpleNamespace(run=missing_docker, PIPE=subprocess.PIPE, TimeoutExpired=subprocess.TimeoutExpired)
    scrape = build_scrape_namespace(
        SRC / "miner_jobs" / "machine_scrape.py", STORAGE_HELPERS, {"subprocess": fake_subprocess, "os": os}
    )

    # Act
    supported, reason = scrape["check_storage_limit_ability"]()

    # Assert
    assert supported is False
    assert reason.startswith("STORAGE_LIMIT_CHECK_ERROR: An unexpected error occurred: ")
