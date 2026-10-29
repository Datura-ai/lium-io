"""ticket-0331 - the vloopback mount test in machine_scrape.py. `--storage-opt` alone passed on a host where every
rental volume failed to mount into its sysbox container ("setting up ID-mapped mount on path 206/fs"), so the scrape
also creates a vloopback volume, mounts it into a container under the runtime rentals get on the host, writes, reads
back and removes it. The verdict goes to the SCRAPE_OK event; storage_limit_supported stays the --storage-opt result.

machine_scrape.py is a script, not a module - importing it runs the whole scrape - so the helpers are
extracted by ast and executed in their own namespace, with `subprocess` replaced by a model of the host's
docker CLI (same pattern as test_scrape_gpu_power_cap_probe.py).
"""

import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from miner_jobs.obfuscator import obfuscate_code
from neurons.validators.src.services import file_encrypt_service
from neurons.validators.src.services.file_encrypt_service import (
    VLOOPBACK_CHECK_SWITCH_OFF,
    VLOOPBACK_CHECK_SWITCH_ON,
    FileEncryptService,
    with_vloopback_check_switch,
)
from neurons.validators.tests.helpers import build_scrape_namespace

SRC = Path(__file__).resolve().parents[1] / "src"
SCRAPE = SRC / "miner_jobs" / "machine_scrape.py"

STORAGE_HELPERS = {
    "VLOOPBACK_DRIVER_PREFIX",
    "VLOOPBACK_PLUGIN",
    "VLOOPBACK_PROBE_IMAGE",
    "VLOOPBACK_PROBE_TOKEN",
    "VLOOPBACK_CHECK_SWITCH",
    "VLOOPBACK_COMMAND_TIMEOUT_SECONDS",
    "VLOOPBACK_RUN_TIMEOUT_SECONDS",
    "VLOOPBACK_CLEANUP_TIMEOUT_SECONDS",
    "VLOOPBACK_CHECK_NAME_PREFIX",
    "STORAGE_PROBE_DETAIL_CAP",
    "run_storage_probe",
    "remove_storage_check",
    "vloopback_mount_test",
    "check_vloopback_volume_ability",
    "vloopback_check_payload",
}

TICKET_0331_RUNC_ERROR = (
    "docker: Error response from daemon: failed to create task for container: OCI runtime create failed: "
    "error during container init: error setting up ID-mapped mount on path 206/fs (likely means idmapped "
    "mounts are not supported on the filesystem at this path): lstat 206: no such file or directory: unknown."
)
NO_SYSBOX_ERROR = (
    "docker: Error response from daemon: unknown or invalid runtime name: sysbox-runc."
)
NAME = re.compile(rf"lium_storage_check_{os.getpid()}_[0-9a-f]{{6}}")


class FakeDockerHost:
    """The docker CLI as the scrape sees it from inside the executor container: a vloopback plugin
    (id, enabled, DATA_DIR) or None, the containers and volumes on the host, and one knob
    per way a step can fail. A container left running by a `docker run` timeout keeps
    its volume busy until `docker rm -f`."""

    def __init__(
        self,
        *,
        plugin: tuple[str, str, str] | None = ("5b1e0f", "true", "/var/lib/docker/loopback"),
        docker_root: str = "/var/lib/docker",
        sysbox: bool = True,
        mountpoint: str | None = None,
        failing: set[str] | frozenset[str] = frozenset(),
        timing_out: set[str] | frozenset[str] = frozenset(),
        containers: set[str] | None = None,
        volumes: set[str] | None = None,
        probe_output: str = "lium-vloopback-ok\n",
    ) -> None:
        self.plugin = plugin
        self.probe_output = probe_output
        self.docker_root = docker_root
        self.sysbox = sysbox
        self.mountpoint = mountpoint
        self.failing = failing
        self.timing_out = timing_out
        self.containers: set[str] = set(containers or ())
        self.volumes: set[str] = set(volumes or ())
        self.calls: list[list[str]] = []

    def run(self, command, stdout=None, stderr=None, text=None, timeout=None):
        self.calls.append(list(command))
        step = self._step(command)
        if step in self.timing_out:
            self._side_effect_before_timeout(step, command)
            raise subprocess.TimeoutExpired(command, timeout)
        if step in self.failing:
            return SimpleNamespace(returncode=1, stdout="", stderr=self._error(step, command))
        return self._answer(step, command)

    @staticmethod
    def _step(command: list[str]) -> str:
        if command[:2] == ["docker", "run"]:
            return "run"
        return " ".join(command[1:3])

    def _side_effect_before_timeout(self, step: str, command: list[str]) -> None:
        # `timeout` ends the CLI; the daemon still finished the create, and the container keeps running
        if step == "volume create":
            self.volumes.add(command[-1])
        elif step == "run":
            self.containers.add(command[command.index("--name") + 1])

    def _error(self, step: str, command: list[str]) -> str:
        return {
            "plugin inspect": "Error response from daemon: Get http://%2Frun%2Fdocker%2Fplugins: context deadline exceeded",
            "volume create": f"Error response from daemon: create {command[-1]}: VolumeDriver.Create: no space left on device",
            "volume rm": f"Error response from daemon: remove {command[-1]}: VolumeDriver.Unmount: context deadline exceeded",
            "run": f"some earlier line\n{TICKET_0331_RUNC_ERROR}",
        }.get(step, "Error response from daemon: failed")

    def _answer(self, step: str, command: list[str]) -> SimpleNamespace:
        stdout, returncode, stderr = "", 0, ""
        if step == "info --format":
            stdout = f"{self.docker_root}\n"
        elif step == "plugin inspect":
            if self.plugin is None:
                returncode, stdout, stderr = 1, "\n", f"Error: No such plugin: {command[-1]}"
            else:
                _, enabled, data_dir = self.plugin
                stdout = f"Enabled={enabled}\nDATA_DIR={data_dir}\nLOG_LEVEL=2\n"
        elif step == "volume create":
            self.volumes.add(command[-1])
            stdout = command[-1]
        elif step == "volume inspect":
            stdout = self.mountpoint if self.mountpoint is not None else f"/mnt/{command[-1]}"
        elif step == "rm -f":
            if command[-1] in self.containers:
                self.containers.discard(command[-1])
                stdout = command[-1]
            else:
                returncode, stderr = (
                    1,
                    f"Error response from daemon: No such container: {command[-1]}",
                )
        elif step == "volume rm":
            if command[-1] in self.containers:
                returncode = 1
                stderr = (
                    f"Error response from daemon: remove {command[-1]}: volume is in use - [4c1d]"
                )
            else:
                self.volumes.discard(command[-1])
        elif step == "run":
            if "--runtime=sysbox-runc" in command and not self.sysbox:
                returncode, stderr = 125, NO_SYSBOX_ERROR
            else:
                stdout = self.probe_output
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    def steps(self) -> list[str]:
        return [self._step(call) for call in self.calls]

    def call(self, step: str) -> list[str]:
        return next(call for call in self.calls if self._step(call) == step)


def _scrape(host: FakeDockerHost, tmp_path: Path) -> dict[str, Any]:
    fake_subprocess = SimpleNamespace(
        run=host.run, PIPE=subprocess.PIPE, TimeoutExpired=subprocess.TimeoutExpired
    )
    return build_scrape_namespace(
        SCRAPE,
        STORAGE_HELPERS,
        {
            "subprocess": fake_subprocess,
            "os": os,
            "re": re,
            "sys": sys,
        },
    )


def _check(
    host: FakeDockerHost, tmp_path: Path, *, storage_opt: bool = True, sysbox: bool = True
) -> dict[str, Any]:
    return _scrape(host, tmp_path)["vloopback_check_payload"](storage_opt, sysbox)


MOUNT_TEST_STEPS = [
    "info --format",
    "plugin inspect",
    "volume create",
    "volume inspect",
    "run",
    "rm -f",
    "volume rm",
]


def test_a_sysbox_host_that_mounts_a_vloopback_volume_passes(tmp_path: Path) -> None:
    # Arrange
    host = FakeDockerHost()

    # Act
    payload = _check(host, tmp_path)

    # Assert
    assert payload == {
        "vc_verdict": "pass",
        "vc_reason_code": "",
        "vc_detail": "",
        "vc_runtime": "sysbox-runc",
    }
    assert host.steps() == MOUNT_TEST_STEPS
    create = host.call("volume create")
    assert create[3:9] == ["-d", "vloopback", "-o", "size=1G", "-o", "sparse=true"]
    assert NAME.fullmatch(create[-1])
    run = host.call("run")
    assert run[:5] == ["docker", "run", "--rm", "--name", create[-1]]
    assert "--runtime=sysbox-runc" in run
    assert run[run.index("-v") + 1] == f"{create[-1]}:/lium-vol"
    assert run[-1] == "echo lium-vloopback-ok > /lium-vol/probe && cat /lium-vol/probe"
    assert host.volumes == set() and host.containers == set()


def test_a_host_without_sysbox_is_tested_under_the_default_runtime_and_passes(
    tmp_path: Path,
) -> None:
    # Arrange: rentals on this host run under runc (sysbox probe failed), their vloopback volumes still work
    host = FakeDockerHost(sysbox=False)

    # Act
    payload = _check(host, tmp_path, sysbox=False)

    # Assert
    assert payload["vc_verdict"] == "pass"
    assert payload["vc_runtime"] == "default"
    assert not [arg for arg in host.call("run") if arg.startswith("--runtime")]


@pytest.mark.parametrize(
    "probe_output", ["", "lium-vloopback-ok-not\n", "lium-vloopback-ok\nsomething else\n"]
)
def test_a_container_that_exits_zero_without_the_probe_token_fails_the_mount_test(
    tmp_path: Path, probe_output: str
) -> None:
    # Arrange: the run exits 0 but never reads back what it wrote into the volume
    host = FakeDockerHost(probe_output=probe_output)

    # Act
    payload = _check(host, tmp_path)

    # Assert
    assert payload["vc_verdict"] == "fail"
    assert payload["vc_reason_code"] == "VLOOPBACK_MOUNT_FAILED"
    assert payload["vc_detail"] == probe_output.strip()
    assert host.volumes == set() and host.containers == set()


@pytest.mark.parametrize(
    ("plugin", "reason_code"),
    [
        (None, "VLOOPBACK_PLUGIN_ABSENT"),
        (("5b1e0f", "false", "/var/lib/docker/loopback"), "VLOOPBACK_PLUGIN_DISABLED"),
    ],
)
def test_a_missing_or_disabled_plugin_is_skipped_and_never_changed(
    tmp_path: Path, plugin: tuple[str, str, str] | None, reason_code: str
) -> None:
    # Arrange: the first rental with a disk limit installs or enables it (create_local_volume), and setup does
    host = FakeDockerHost(plugin=plugin)

    # Act
    payload = _check(host, tmp_path)

    # Assert
    assert payload["vc_verdict"] == "skipped"
    assert payload["vc_reason_code"] == reason_code
    assert not [
        step for step in host.steps() if step.startswith("plugin ") and step != "plugin inspect"
    ]
    assert "volume create" not in host.steps()


@pytest.mark.parametrize(
    ("host_kwargs", "reason_code", "detail"),
    [
        ({"docker_root": ""}, "VLOOPBACK_DOCKER_ROOT_UNREADABLE", "data-root ''"),
        (
            {"plugin": ("5b1e0f", "", "/var/lib/docker/loopback")},
            "VLOOPBACK_PLUGIN_UNREADABLE",
            "docker plugin inspect says Enabled=''",
        ),
        (
            {"plugin": ("5b1e0f", "true", "loopback")},
            "VLOOPBACK_DATA_DIR_NOT_ABSOLUTE",
            "the vloopback plugin's DATA_DIR is 'loopback'",
        ),
        (
            {"failing": {"plugin inspect"}},
            "VLOOPBACK_PLUGIN_UNREADABLE",
            "Error response from daemon: Get http://%2Frun%2Fdocker%2Fplugins: context deadline exceeded",
        ),
        (
            {"timing_out": {"plugin inspect"}},
            "VLOOPBACK_CHECK_TIMEOUT",
            "docker plugin inspect timed out after 10s",
        ),
        (
            {"failing": {"volume create"}},
            "VLOOPBACK_VOLUME_CREATE_FAILED",
            "Error response from daemon: create {name}: VolumeDriver.Create: no space left on device",
        ),
        (
            {"failing": {"volume rm"}},
            "VLOOPBACK_VOLUME_REMOVE_FAILED",
            "Error response from daemon: remove {name}: VolumeDriver.Unmount: context deadline exceeded",
        ),
        # what `docker volume inspect t` showed on the ticket-0331 host: no container starts
        (
            {"mountpoint": "206/fs"},
            "VLOOPBACK_MOUNTPOINT_NOT_ABSOLUTE",
            "docker volume inspect gave Mountpoint '206/fs'",
        ),
        # the ticket-0331 host: plugin enabled with an absolute DATA_DIR, `--storage-opt` fine, the mount fails
        ({"failing": {"run"}}, "VLOOPBACK_MOUNT_FAILED", TICKET_0331_RUNC_ERROR[:300]),
        # the sysbox exit wedge: the CLI is killed at its timeout, the container keeps the volume busy
        ({"timing_out": {"run"}}, "VLOOPBACK_CHECK_TIMEOUT", "docker run --rm timed out after 30s"),
        # the daemon finished the create after the CLI timed out
        (
            {"timing_out": {"volume create"}},
            "VLOOPBACK_CHECK_TIMEOUT",
            "docker volume create timed out after 10s",
        ),
    ],
)
def test_each_failing_step_is_reported_with_its_reason_code(
    tmp_path: Path, host_kwargs: dict[str, Any], reason_code: str, detail: str
) -> None:
    # Arrange
    host = FakeDockerHost(**host_kwargs)

    # Act
    payload = _check(host, tmp_path)

    # Assert
    assert payload["vc_verdict"] == "fail"
    assert payload["vc_reason_code"] == reason_code
    name = host.call("volume create")[-1] if "volume create" in host.steps() else ""
    assert payload["vc_detail"] == detail.format(name=name)
    if "volume create" in host.steps():
        assert host.steps()[-2:] == ["rm -f", "volume rm"]
    # nothing left, but the volume that would not remove
    assert host.containers == set()
    assert host.volumes == ({name} if reason_code == "VLOOPBACK_VOLUME_REMOVE_FAILED" else set())


def test_without_storage_opt_the_mount_test_does_not_run(tmp_path: Path) -> None:
    # Arrange
    host = FakeDockerHost()

    # Act
    payload = _check(host, tmp_path, storage_opt=False)

    # Assert
    assert payload["vc_verdict"] == "skipped"
    assert payload["vc_reason_code"] == "STORAGE_OPT_UNSUPPORTED"
    assert host.calls == []


def test_the_switch_turned_off_runs_no_docker_command(tmp_path: Path) -> None:
    # Arrange
    host = FakeDockerHost()
    scrape = _scrape(host, tmp_path)
    scrape["VLOOPBACK_CHECK_SWITCH"] = VLOOPBACK_CHECK_SWITCH_OFF

    # Act
    payload = scrape["vloopback_check_payload"](True, True)

    # Assert
    assert payload["vc_verdict"] == "off"
    assert host.calls == []


def test_no_docker_cli_is_an_error_with_its_own_code(tmp_path: Path) -> None:
    # Arrange
    def missing_docker(command, **_):
        raise FileNotFoundError(2, "No such file or directory", "docker")

    host = FakeDockerHost()
    scrape = _scrape(host, tmp_path)
    scrape["subprocess"] = SimpleNamespace(
        run=missing_docker, PIPE=subprocess.PIPE, TimeoutExpired=subprocess.TimeoutExpired
    )

    # Act
    payload = scrape["vloopback_check_payload"](True, True)

    # Assert
    assert payload["vc_verdict"] == "fail"
    assert payload["vc_reason_code"] == "VLOOPBACK_CHECK_ERROR"


def test_the_kill_switch_survives_obfuscation_and_is_rewritten_when_disabled() -> None:
    # Arrange
    obfuscated = obfuscate_code(SCRAPE.read_text())

    # Act
    disabled = with_vloopback_check_switch(obfuscated, False)

    # Assert
    assert obfuscated.count(VLOOPBACK_CHECK_SWITCH_ON) == 1
    assert with_vloopback_check_switch(obfuscated, True) == obfuscated
    assert VLOOPBACK_CHECK_SWITCH_ON not in disabled
    assert disabled.count(VLOOPBACK_CHECK_SWITCH_OFF) == 1


@pytest.mark.parametrize("enabled", [True, False])
def test_the_packaged_scrape_follows_the_setting(monkeypatch, enabled: bool) -> None:
    # Arrange: the obfuscator and key substitution under test; freezing the result is PyInstaller's job
    monkeypatch.setattr(FileEncryptService, "make_binary_file", lambda self, tmp, path: "scrape")
    monkeypatch.setattr(file_encrypt_service.settings, "VLOOPBACK_SCRAPE_CHECK_ENABLED", enabled)

    # Act
    files = FileEncryptService(ssh_service=None).ecrypt_miner_job_files()

    # Assert
    expected = VLOOPBACK_CHECK_SWITCH_ON if enabled else VLOOPBACK_CHECK_SWITCH_OFF
    assert files.machine_scrape_source.count(expected) == 1
