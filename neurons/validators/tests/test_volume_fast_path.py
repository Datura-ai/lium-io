"""DAH-3240 — rental volume fast path (RENTAL_VOLUME_FAST_PATH_ENABLED).

The "Docker volume creation" stage of a rent ran five serial SSH commands before the volume
existed (`docker info`, the df helper container, `docker volume ls`, `docker info` again and an
unconditional `docker plugin install` that asks Docker Hub and then fails with "already exists").
With the flag on, one probe command returns the same facts and the plugin install is skipped when
the plugin is already enabled. Covered here:

- the probe command names every section and reuses the df helper command `df_available_bytes` runs;
- the parser reads root / df / vloopback names / plugin state and raises on a missing section or a
  failed `docker volume ls` (its `VOLS\t<exit status>` sentinel);
- the command and the parser together, through `sh -c` and a docker stub (volumes / empty list /
  failed `volume ls` / absent plugin);
- `probe_volume_host` is never fatal (SSH error or garbage → None → the per-command path);
- `resolve_volume_sizing` with a probe computes the SAME result as the per-command path for the
  same host facts, running only `docker volume inspect` (nothing at all without vloopback volumes);
- `create_local_volume` with an enabled plugin runs no SSH command and creates the identical volume;
  with the plugin absent it still installs (negative control); with the plugin installed but
  disabled it runs `docker plugin enable` and re-reads the state, failing fast when it stays off;
- `create_container` never probes with the flag off, probes once with it on and hands the probe to
  both the sizing and the create.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import stat
import subprocess
from unittest.mock import AsyncMock, Mock

import pytest
from core.config import settings
from core.docker_utils import ALPINE_HELPER_IMAGE, df_command
from payload_models.payloads import ContainerCreateRequest
from services import docker_service as docker_service_module
from services.docker_service import (
    DockerService,
    NoUsableLoopbackPluginError,
    SshSessionMayStillBeTakenError,
    VolumeHostProbe,
    _parse_volume_host_probe,
    _run_with_host_timeout,
    _volume_host_probe_command,
)
from test_deploy_optimizations import (
    _patch_happy,
    _payload as _deploy_payload,
    _run as _run_create_container,
    _ssh_client as _deploy_ssh_client,
)
from test_docker_service import (
    _SIZING_GB,
    _FakeRentalDockerFactory,
    _make_sizing_payload,
    _make_sizing_ssh_client,
    ssh_client_answering_through_run,
)

_DF_STDOUT = (
    "Filesystem           1-blocks       Used Available Capacity Mounted on\n"
    "/dev/vda1            1000 500 966367641600  80% /hostfs\n"
)


def _probe_stdout(
    *,
    root: str = "/var/lib/docker",
    df: str | None = _DF_STDOUT,
    volumes: str = "",
    volume_ls_status: str | None = "0",
    plugin: str = "true",
) -> str:
    lines = [f"ROOT\t{root}"]
    if df is not None:
        lines.append("DF\t" + df.replace("\n", "\r"))
    lines.extend(volumes.splitlines())
    if volume_ls_status is not None:
        lines.append(f"VOLS\t{volume_ls_status}")
    lines.append(f"PLUGIN\t{plugin}")
    return "\n".join(lines) + "\n"


def _bounded(command: str, seconds: int = 10) -> str:
    return f"timeout -k 5 {seconds} sh -c {shlex.quote(command)}"


async def _hang(command: str):
    await asyncio.Event().wait()


@pytest.fixture
def docker_service():
    return DockerService(
        ssh_service=Mock(),
        redis_service=Mock(),
        attestation_service=Mock(),
        rental_docker_client_factory=_FakeRentalDockerFactory(),
    )


# ---------------------------------------------------------------------------
# the probe command
# ---------------------------------------------------------------------------


def test_probe_command_is_one_line_with_every_section():
    command = _volume_host_probe_command(with_df=True)

    assert "\n" not in command
    assert "/usr/bin/docker info --format '{{.DockerRootDir}}'" in command
    assert (
        df_command('"$root"') in command
    ), "df must be the same helper-container command df_available_bytes runs"
    assert ALPINE_HELPER_IMAGE in command
    assert (
        "/usr/bin/docker volume ls --format 'VOL\\t{{.Name}}\\t{{.Driver}}'; printf 'VOLS\\t%s\\n' \"$?\"; "
        in command
    )
    assert "/usr/bin/docker plugin inspect --format '{{.Enabled}}' vloopback:v2 " in command
    assert "plugin install" not in command


def test_probe_command_without_df_skips_the_helper_container():
    command = _volume_host_probe_command(with_df=False)

    assert "docker run" not in command
    assert "DockerRootDir" in command and "plugin inspect" in command


# ---------------------------------------------------------------------------
# the parser
# ---------------------------------------------------------------------------


def test_parse_probe_reads_root_df_vloopback_names_and_plugin():
    stdout = _probe_stdout(
        volumes=(
            "VOL\tvolume_abc\tvloopback:latest\n"
            "VOL\tvolume_def\tvloopback\n"
            "VOL\tvolume_v2\tvloopback:v2\n"
            "VOL\tother_volume\tlocal\n"
            "VOL\tbad name;rm\tvloopback\n"
        ),
    )

    probe = _parse_volume_host_probe(stdout, with_df=True)

    assert probe.docker_root_dir == "/var/lib/docker"
    assert probe.df_avail_bytes == 966367641600
    assert probe.vloopback_volume_names == ["volume_abc", "volume_def", "volume_v2"]
    assert probe.loopback_plugin_enabled is True


@pytest.mark.parametrize("plugin_state", ["false", "absent", ""])
def test_parse_probe_plugin_not_true_is_not_enabled(plugin_state):
    probe = _parse_volume_host_probe(_probe_stdout(plugin=plugin_state), with_df=True)

    assert probe.loopback_plugin_enabled is False


@pytest.mark.parametrize(
    "plugin_state, installed", [("true", True), ("false", True), ("absent", False), ("", False)]
)
def test_parse_probe_plugin_installed_is_true_or_false(plugin_state, installed):
    probe = _parse_volume_host_probe(_probe_stdout(plugin=plugin_state), with_df=True)

    assert probe.loopback_plugin_installed is installed


def test_parse_probe_without_df_section_when_not_requested():
    probe = _parse_volume_host_probe(_probe_stdout(df=None), with_df=False)

    assert probe.df_avail_bytes is None
    assert probe.docker_root_dir == "/var/lib/docker"


@pytest.mark.parametrize(
    "stdout, match",
    [
        (_probe_stdout(root=""), "no DockerRootDir"),
        (_probe_stdout(df=None), "no df output"),
        (_probe_stdout(df="Filesystem\ngarbage line\n"), "unexpected df output"),
        (_probe_stdout(volume_ls_status=None), "docker volume ls exit status None"),
        (_probe_stdout(volume_ls_status="1"), "docker volume ls exit status '1'"),
        (
            "ROOT\t/var/lib/docker\n" + "DF\t" + _DF_STDOUT.replace("\n", "\r") + "\nVOLS\t0\n",
            "no plugin state",
        ),
    ],
)
def test_parse_probe_missing_section_raises(stdout, match):
    with pytest.raises(Exception, match=match):
        _parse_volume_host_probe(stdout, with_df=True)


@pytest.mark.parametrize(
    "stdout",
    [
        _probe_stdout(df="Filesystem\n" + "x" * 5000 + "\n"),  # garbage df record
        _probe_stdout(
            root="", volumes="VOL\t" + "v" * 5000 + "\tlocal\n"
        ),  # missing root, long list
    ],
)
def test_parse_probe_error_message_caps_the_echoed_host_output(stdout):
    with pytest.raises(Exception) as excinfo:
        _parse_volume_host_probe(stdout, with_df=True)

    assert len(str(excinfo.value)) < 512 + 200


# ---------------------------------------------------------------------------
# the command and the parser together, through a real shell and a docker stub
# ---------------------------------------------------------------------------

_DOCKER_STUB = """#!/bin/sh
# stands in for /usr/bin/docker: answers the four sub-commands the probe issues
mode="$(cat "$STUB_MODE_FILE")"
case "$1 $2" in
  "info --format") echo /var/lib/docker ;;
  "run --rm") printf 'Filesystem 1-blocks Used Available Capacity Mounted on\\n/dev/vda1 1000 500 42 80%% /hostfs\\n' ;;
  "volume ls")
    case "$mode" in
      volumes) printf 'VOL\\tvolume_abc\\tvloopback:latest\\nVOL\\tother\\tlocal\\n' ;;
      empty) ;;
      ls-fails) echo "Cannot connect to the Docker daemon" >&2; exit 1 ;;
    esac ;;
  "plugin inspect") [ "$mode" = plugin-absent ] && { echo; exit 1; }  # real docker: blank stdout line, then exit 1
    [ "$mode" = plugin-disabled ] && { echo false; exit 0; }; echo true ;;
  *) echo "unexpected: $*" >&2; exit 2 ;;
esac
"""


def _run_probe_command_with_stub(tmp_path, mode: str) -> str:
    stub = tmp_path / "docker"
    stub.write_text(_DOCKER_STUB)
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    mode_file = tmp_path / "mode"
    mode_file.write_text(mode)
    command = _volume_host_probe_command(with_df=True).replace("/usr/bin/docker", str(stub))
    # bytes, not text: text mode would translate the \r the DF record relies on
    completed = subprocess.run(
        ["sh", "-c", command],
        capture_output=True,
        env={**os.environ, "STUB_MODE_FILE": str(mode_file)},
        check=False,
    )
    return completed.stdout.decode()


def test_probe_command_through_a_shell_parses_volumes(tmp_path):
    stdout = _run_probe_command_with_stub(tmp_path, "volumes")

    probe = _parse_volume_host_probe(stdout, with_df=True)

    assert probe.docker_root_dir == "/var/lib/docker"
    assert probe.df_avail_bytes == 42
    assert probe.vloopback_volume_names == ["volume_abc"]
    assert probe.loopback_plugin_enabled is True


def test_probe_command_through_a_shell_empty_volume_list_is_zero_volumes(tmp_path):
    probe = _parse_volume_host_probe(_run_probe_command_with_stub(tmp_path, "empty"), with_df=True)

    assert probe.vloopback_volume_names == []
    assert probe.loopback_plugin_enabled is True


def test_probe_command_through_a_shell_failed_volume_ls_raises(tmp_path):
    stdout = _run_probe_command_with_stub(tmp_path, "ls-fails")

    assert "VOLS\t1" in stdout
    with pytest.raises(Exception, match="docker volume ls exit status '1'"):
        _parse_volume_host_probe(stdout, with_df=True)


def test_probe_command_through_a_shell_absent_plugin_is_not_enabled(tmp_path):
    stdout = _run_probe_command_with_stub(tmp_path, "plugin-absent")
    probe = _parse_volume_host_probe(stdout, with_df=True)

    assert probe.loopback_plugin_enabled is False
    # the blank line docker prints before failing must not be what the field carries
    assert "PLUGIN\tabsent\n" in stdout
    assert probe.loopback_plugin_installed is False


def test_probe_command_through_a_shell_disabled_plugin_is_installed_not_enabled(tmp_path):
    stdout = _run_probe_command_with_stub(tmp_path, "plugin-disabled")
    probe = _parse_volume_host_probe(stdout, with_df=True)

    assert "PLUGIN\tfalse\n" in stdout
    assert probe.loopback_plugin_enabled is False
    assert probe.loopback_plugin_installed is True


# ---------------------------------------------------------------------------
# probe_volume_host is never fatal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_probe_volume_host_returns_probe_from_one_command(docker_service):
    ssh_client = Mock()
    ssh_client.run = AsyncMock(return_value=Mock(stdout=_probe_stdout(), exit_status=0))
    ssh_client = ssh_client_answering_through_run(ssh_client.run)

    probe = await docker_service.probe_volume_host(ssh_client, with_df=True, log_extra={})

    assert isinstance(probe, VolumeHostProbe)
    assert ssh_client.run.await_count == 1
    assert ssh_client.run.await_args.args[0] == _bounded(_volume_host_probe_command(with_df=True), 30)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [Exception("ssh boom"), _hang])
async def test_probe_volume_host_ssh_error_returns_none(docker_service, monkeypatch, failure):
    monkeypatch.setattr(docker_service_module, "_VOLUME_HOST_PROBE_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(docker_service_module, "_HOST_TIMEOUT_VALIDATOR_MARGIN_SECONDS", 0.05)
    ssh_client = ssh_client_answering_through_run(AsyncMock(side_effect=failure))

    probe = docker_service.probe_volume_host(ssh_client, with_df=True, log_extra={})

    assert await asyncio.wait_for(probe, timeout=5) is None


@pytest.mark.asyncio
async def test_probe_volume_host_lets_a_session_that_may_stay_taken_stop_the_create(docker_service, monkeypatch):
    monkeypatch.setattr(
        docker_service_module, "_run_with_host_timeout", AsyncMock(side_effect=SshSessionMayStillBeTakenError())
    )

    with pytest.raises(SshSessionMayStillBeTakenError):
        await docker_service.probe_volume_host(Mock(), with_df=True, log_extra={})


@pytest.mark.asyncio
async def test_probe_volume_host_garbage_output_returns_none(docker_service):
    ssh_client = Mock()
    ssh_client.run = AsyncMock(return_value=Mock(stdout="nothing useful\n", exit_status=0))
    ssh_client = ssh_client_answering_through_run(ssh_client.run)

    assert await docker_service.probe_volume_host(ssh_client, with_df=True, log_extra={}) is None


# ---------------------------------------------------------------------------
# resolve_volume_sizing: same numbers, fewer commands
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides, expected_path",
    [
        (dict(disk_share=0.5, storage_limit_gb=1), "fresh"),
        (dict(disk_share=0.5, storage_limit_gb=None), "storage_opt_unsupported"),
        (dict(disk_share=None, storage_limit_gb=1), "legacy"),
    ],
)
async def test_measures_host_for_volume_sizing_is_the_predicate_resolve_volume_sizing_uses(
    docker_service, overrides, expected_path
):
    # The probe asks for df exactly when the sizing will measure: the same predicate decides both.
    payload = _make_sizing_payload(**overrides)
    ssh_client = _make_sizing_ssh_client(df_avail_bytes=900 * _SIZING_GB)

    result = await docker_service.resolve_volume_sizing(ssh_client, payload, "tag", {})

    assert result.path == expected_path
    assert DockerService.measures_host_for_volume_sizing(payload) is (result.path == "fresh")
    assert (ssh_client.run.await_count > 0) is (result.path == "fresh")


@pytest.mark.asyncio
async def test_resolve_volume_sizing_with_probe_matches_per_command_result(docker_service):
    # Arrange: the same host facts through both paths (the per-command fixture from
    # test_docker_service, and a probe carrying what its `docker info` / df / `volume ls` say).
    payload = _make_sizing_payload(disk_share=0.5, storage_limit_gb=1)
    volume_inspect_stdout = f"{300 * _SIZING_GB}|<no value>\n{200 * _SIZING_GB}|<no value>\n"
    per_command_ssh = _make_sizing_ssh_client(
        df_avail_bytes=900 * _SIZING_GB,
        volume_ls_stdout="volume_abc vloopback:latest\nvolume_v2 vloopback:v2\nother_volume local\n",
        volume_inspect_stdout=volume_inspect_stdout,
    )
    probe = VolumeHostProbe(
        docker_root_dir="/var/lib/docker",
        df_avail_bytes=900 * _SIZING_GB,
        vloopback_volume_names=["volume_abc", "volume_v2"],
        loopback_plugin_enabled=True,
    )
    probe_ssh = Mock()
    probe_ssh.run = AsyncMock(return_value=Mock(stdout=volume_inspect_stdout, exit_status=0))

    # Act
    per_command = await docker_service.resolve_volume_sizing(per_command_ssh, payload, "tag", {})
    with_probe = await docker_service.resolve_volume_sizing(
        probe_ssh, payload, "tag", {}, host_probe=probe
    )

    # Assert: identical sizing, and the probe path ran exactly the inspect command.
    assert with_probe == per_command
    assert with_probe.path == "fresh" and with_probe.volume_limit_gb == 460
    assert with_probe.existing_volumes_bytes == 500 * _SIZING_GB  # both plugin generations count
    assert per_command_ssh.run.await_count == 4  # info, df, volume ls, volume inspect
    assert probe_ssh.run.await_count == 1
    assert probe_ssh.run.await_args.args[0].startswith("/usr/bin/docker volume inspect volume_abc ")


@pytest.mark.asyncio
async def test_resolve_volume_sizing_with_probe_and_no_vloopback_volumes_runs_no_command(
    docker_service,
):
    payload = _make_sizing_payload(disk_share=0.5, storage_limit_gb=1)
    probe = VolumeHostProbe(
        docker_root_dir="/var/lib/docker",
        df_avail_bytes=900 * _SIZING_GB,
        vloopback_volume_names=[],
        loopback_plugin_enabled=True,
    )
    ssh_client = Mock()
    ssh_client.run = AsyncMock()

    result = await docker_service.resolve_volume_sizing(
        ssh_client, payload, "tag", {}, host_probe=probe
    )

    assert result.path == "fresh"
    assert result.existing_volumes_bytes == 0
    ssh_client.run.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_volume_sizing_probe_without_df_falls_back_to_per_command_measurement(
    docker_service,
):
    # A probe taken with with_df=False carries no df figure; the fresh path must still measure.
    payload = _make_sizing_payload(disk_share=0.5, storage_limit_gb=1)
    probe = VolumeHostProbe(
        docker_root_dir="/var/lib/docker",
        df_avail_bytes=None,
        vloopback_volume_names=[],
        loopback_plugin_enabled=True,
    )
    ssh_client = _make_sizing_ssh_client(df_avail_bytes=900 * _SIZING_GB)

    result = await docker_service.resolve_volume_sizing(
        ssh_client, payload, "tag", {}, host_probe=probe
    )

    assert result.path == "fresh"
    assert (
        ssh_client.run.await_count == 3
    )  # info, df, volume ls (no vloopback volumes → no inspect)


# ---------------------------------------------------------------------------
# create_local_volume: skip the Docker Hub round trip when the plugin is enabled
# ---------------------------------------------------------------------------


async def _create_volume(docker_service, ssh_client, probe):
    docker_service.stream_log = AsyncMock()
    docker_client = docker_service.rental_docker_client_factory.client
    await docker_service.create_local_volume(
        ssh_client=ssh_client,
        docker_client=docker_client,
        local_volume="volume_test",
        log_tag="tag",
        log_text="Creating docker volume volume_test",
        log_extra={},
        limit=40,
        timeout=10,
        host_probe=probe,
    )
    return docker_client.created_volumes


@pytest.mark.asyncio
async def test_create_local_volume_with_enabled_plugin_runs_no_ssh_command(docker_service):
    ssh_client = Mock()
    ssh_client.run = AsyncMock()
    probe = VolumeHostProbe(
        docker_root_dir="/var/lib/docker",
        df_avail_bytes=None,
        vloopback_volume_names=[],
        loopback_plugin_enabled=True,
    )

    created = await _create_volume(docker_service, ssh_client, probe)

    ssh_client.run.assert_not_called()
    assert created == [
        {
            "volume_name": "volume_test",
            "driver": "vloopback:v2",
            "driver_opts": {"size": "40g"},
            "timeout": 10,
        }
    ]


@pytest.mark.asyncio
async def test_create_local_volume_with_plugin_absent_still_installs_it(docker_service):
    # Negative control: the probe saw no enabled v2 → the install command runs as before, with
    # DATA_DIR on the host under the probe's root dir (no second `docker info`), outside the plugin rootfs.
    ssh_client = Mock()
    ssh_client.run = AsyncMock(return_value=Mock(stdout="", exit_status=0))
    ssh_client = ssh_client_answering_through_run(ssh_client.run)
    probe = VolumeHostProbe(
        docker_root_dir="/data/docker",
        df_avail_bytes=None,
        vloopback_volume_names=[],
        loopback_plugin_enabled=False,
    )

    created = await _create_volume(docker_service, ssh_client, probe)

    assert ssh_client.run.await_count == 1
    assert ssh_client.run.await_args.args[0] == _bounded(
        "/usr/bin/docker plugin install daturaai/docker-volume-loopback:1.0.0-lium1 "
        "--alias vloopback:v2 --grant-all-permissions "
        "DATA_DIR=/srv/data/docker/vloopback-v2 STATE_DIR=/srv/run/docker-volume-loopback-v2",
        60,
    )
    assert created[0]["driver"] == "vloopback:v2"


def _disabled_plugin_probe() -> VolumeHostProbe:
    return VolumeHostProbe(
        docker_root_dir="/var/lib/docker",
        df_avail_bytes=None,
        vloopback_volume_names=[],
        loopback_plugin_enabled=False,
        loopback_plugin_installed=True,
    )


@pytest.mark.asyncio
async def test_create_local_volume_with_disabled_plugin_enables_it_instead_of_installing(
    docker_service,
):
    # 7fcd02af (20-27 Sep): installed but disabled → `plugin install` failed "already exists"
    # and every create failed "plugin vloopback found but disabled". Enable, re-read, create.
    ssh_client = Mock()
    ssh_client.run = AsyncMock(
        side_effect=[
            Mock(stdout="vloopback:v2\n", stderr="", exit_status=0),
            Mock(stdout="true\n", stderr="", exit_status=0),
        ]
    )
    ssh_client = ssh_client_answering_through_run(ssh_client.run)

    created = await _create_volume(docker_service, ssh_client, _disabled_plugin_probe())

    calls = ssh_client.run.await_args_list
    assert [c.args[0] for c in calls] == [
        _bounded("/usr/bin/docker plugin enable vloopback:v2"),
        _bounded(
            "( /usr/bin/docker plugin inspect --format '{{.Enabled}}' vloopback:v2 2>/dev/null "
            "|| echo absent) | tail -n 1"
        ),
    ]
    assert not any("plugin install" in c.args[0] for c in calls)
    assert created == [
        {
            "volume_name": "volume_test",
            "driver": "vloopback:v2",
            "driver_opts": {"size": "40g"},
            "timeout": 10,
        }
    ]


@pytest.mark.asyncio
async def test_create_local_volume_disabled_plugin_that_will_not_enable_fails_fast(docker_service):
    ssh_client = Mock()
    ssh_client.run = AsyncMock(
        side_effect=[
            Mock(stdout="", stderr="Error response from daemon: dial unix plugin.sock: connect: no such file", exit_status=1),
            Mock(stdout="false\n", stderr="", exit_status=0),
            Mock(stdout="false\n", stderr="", exit_status=0),
        ]
    )
    ssh_client = ssh_client_answering_through_run(ssh_client.run)

    with pytest.raises(NoUsableLoopbackPluginError) as exc_info:
        await _create_volume(docker_service, ssh_client, _disabled_plugin_probe())

    message = str(exc_info.value)
    assert message.startswith("no usable vloopback plugin on host (v2: vloopback plugin disabled on host")
    assert "enable exit 1, state false" in message and "plugin.sock" in message
    assert message.endswith("; old plugin vloopback state false)")
    # lium-platform's classifier files it as volume.plugin_disabled on these two substrings
    assert "plugin vloopback" in message.lower() and "disabled" in message.lower()
    assert docker_service.rental_docker_client_factory.client.created_volumes == []


@pytest.mark.asyncio
async def test_create_local_volume_disabled_plugin_enable_timeout_fails_fast(docker_service):
    ssh_client = Mock()
    ssh_client.run = AsyncMock(side_effect=asyncio.TimeoutError())
    ssh_client = ssh_client_answering_through_run(ssh_client.run)

    with pytest.raises(NoUsableLoopbackPluginError, match="could not be enabled.*old plugin vloopback state unknown"):
        await _create_volume(docker_service, ssh_client, _disabled_plugin_probe())

    assert ssh_client.run.await_count == 2
    assert docker_service.rental_docker_client_factory.client.created_volumes == []


@pytest.mark.asyncio
async def test_create_local_volume_disabled_plugin_enable_error_logs_only_its_type(
    docker_service, monkeypatch
):
    secret_text = "ssh ubuntu@10.1.2.3: token=ghp_leakme at /home/provider/.ssh/id_rsa"
    ssh_client = Mock()
    ssh_client.run = AsyncMock(side_effect=[OSError(secret_text), TimeoutError()])
    ssh_client = ssh_client_answering_through_run(ssh_client.run)
    warning = Mock()
    monkeypatch.setattr(docker_service_module.logger, "warning", warning)

    with pytest.raises(NoUsableLoopbackPluginError) as exc_info:
        await _create_volume(docker_service, ssh_client, _disabled_plugin_probe())

    assert "enable error: OSError" in str(exc_info.value)
    assert secret_text not in str(exc_info.value)
    (logged,), _ = warning.call_args
    assert str(logged).startswith("Loopback plugin enable failed")
    assert logged.extra["error_type"] == "OSError"
    assert logged.extra["loopback_plugin"] == "vloopback:v2"
    assert "error" not in logged.extra
    full = logged.to_full_string()
    assert "10.1.2.3" not in full and "ghp_leakme" not in full and "id_rsa" not in full


@pytest.mark.asyncio
async def test_create_local_volume_without_probe_keeps_the_per_command_path(docker_service):
    ssh_client = Mock()
    ssh_client.run = AsyncMock(return_value=Mock(stdout="/var/lib/docker\n", exit_status=0))
    ssh_client = ssh_client_answering_through_run(ssh_client.run)

    await _create_volume(docker_service, ssh_client, None)

    commands = [c.args[0] for c in ssh_client.run.await_args_list]
    assert commands[0] == "/usr/bin/docker info --format '{{.DockerRootDir}}'"
    assert commands[1].startswith("timeout -k 5 60 sh -c '/usr/bin/docker plugin install daturaai/")
    assert len(commands) == 2


# v2 beside the old plugin: a stalled command frees its SSH channel (MaxSessions=1) or stops the create;
# a v2 that cannot be used falls back to the old plugin only when it is enabled


def _channel_ssh(*, open_s: float, run_s: float, closes: bool = True) -> tuple[Mock, Mock]:
    async def wait():
        await asyncio.sleep(run_s)
        return "answer"

    async def wait_closed():
        if not closes:
            await asyncio.Event().wait()

    async def create_process(command: str):
        await asyncio.sleep(open_s)
        return channel

    channel = Mock(wait=wait, wait_closed=wait_closed)
    return Mock(create_process=create_process), channel


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "open_s, run_s, closes, outcome",
    [
        (0, 0, True, None),
        (0, 1, True, TimeoutError),  # the command outlives the deadline: its channel is closed
        (0, 1, False, SshSessionMayStillBeTakenError),  # ... and the close is not confirmed
        (0.075, 0, True, TimeoutError),  # the open answers after the deadline, within the second wait
        (0.3, 0, True, SshSessionMayStillBeTakenError),  # never in time: closed whenever it does answer
    ],
)
async def test_run_with_host_timeout_frees_the_channel_or_stops_the_create(
    monkeypatch, open_s, run_s, closes, outcome
):
    monkeypatch.setattr(docker_service_module, "_HOST_TIMEOUT_VALIDATOR_MARGIN_SECONDS", 0)
    ssh_client, channel = _channel_ssh(open_s=open_s, run_s=run_s, closes=closes)

    ran = asyncio.wait_for(_run_with_host_timeout(ssh_client, "cmd", 0.05), timeout=5)
    if outcome is None:
        assert await ran == "answer"
    else:
        with pytest.raises(outcome):
            await ran
    await asyncio.sleep(open_s)

    assert channel.close.called is (outcome is not None)


def _plugin_host(*, v2_state: str = "absent", old_state: str = "true", install: str = "fails") -> Mock:
    # the host's docker through ssh: a v2 install or enable that fails (or never answers), and the plugin states
    async def answer(command: str):
        command = shlex.split(command)[-1] if command.startswith("timeout -k 5 ") else command
        if "docker info" in command:
            return Mock(stdout="/var/lib/docker\n", stderr="", exit_status=0)
        if "plugin inspect" in command:
            state = v2_state if "vloopback:v2" in command else old_state
            if state == "unreadable":
                raise TimeoutError
            return Mock(stdout=f"{state}\n", stderr="", exit_status=0)
        if install == "hangs":
            await asyncio.Event().wait()
        return Mock(stdout="", stderr="Error response from daemon: registry unreachable", exit_status=1)

    return ssh_client_answering_through_run(AsyncMock(side_effect=answer))


@pytest.mark.asyncio
async def test_run_with_host_timeout_waits_the_margin_past_the_host_timeout(monkeypatch):
    monkeypatch.setattr(docker_service_module, "_HOST_TIMEOUT_VALIDATOR_MARGIN_SECONDS", 0.2)
    ssh_client, channel = _channel_ssh(open_s=0, run_s=0.1)

    assert await _run_with_host_timeout(ssh_client, "cmd", 0.05) == "answer"
    assert not channel.close.called


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "probe, install, v2_state, reason",
    [
        (None, "fails", "absent", "install exit 1, state absent: Error response"),
        (None, "hangs", "absent", "install timed out after 0.1 s"),
        (_disabled_plugin_probe(), "fails", "false", "vloopback plugin disabled on host"),
        (None, "fails", "unreadable", "install exit 1, state unknown: Error response"),
    ],
)
async def test_create_local_volume_falls_back_to_the_enabled_old_plugin_when_v2_cannot_be_used(
    docker_service, monkeypatch, probe, install, v2_state, reason
):
    warning = Mock()
    monkeypatch.setattr(docker_service_module.logger, "warning", warning)
    monkeypatch.setattr(docker_service_module, "_LOOPBACK_PLUGIN_INSTALL_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(docker_service_module, "_HOST_TIMEOUT_VALIDATOR_MARGIN_SECONDS", 0.05)
    ssh_client = _plugin_host(v2_state=v2_state, install=install)

    created = await asyncio.wait_for(_create_volume(docker_service, ssh_client, probe), timeout=5)

    assert [volume["driver"] for volume in created] == ["vloopback"]
    commands = [call.args[0] for call in ssh_client.run.await_args_list]
    assert all(command.startswith("timeout -k 5 ") for command in commands if "docker info" not in command)
    (logged,), _ = warning.call_args
    assert str(logged).startswith("Loopback plugin v2 unusable; creating this volume on the old plugin")
    assert logged.extra["reason"].startswith(reason)


@pytest.mark.asyncio
@pytest.mark.parametrize("old_state", ["absent", "false"])
async def test_create_local_volume_fails_naming_both_and_never_installs_or_enables_the_old_plugin(
    docker_service, old_state
):
    ssh_client = _plugin_host(old_state=old_state)

    with pytest.raises(NoUsableLoopbackPluginError) as exc_info:
        await _create_volume(docker_service, ssh_client, None)

    message = str(exc_info.value)
    assert message.startswith("no usable vloopback plugin on host (v2: install exit 1, state absent: Error response")
    assert message.endswith(f"; old plugin vloopback state {old_state})")
    commands = [call.args[0] for call in ssh_client.run.await_args_list]
    assert all("vloopback:v2" in command for command in commands if "plugin install" in command)
    assert not any("plugin enable" in command for command in commands)
    assert docker_service.rental_docker_client_factory.client.created_volumes == []


@pytest.mark.asyncio
async def test_create_local_volume_without_probe_keeps_v2_when_its_install_says_already_exists(docker_service):
    created = await _create_volume(docker_service, _plugin_host(v2_state="true"), None)

    assert [volume["driver"] for volume in created] == ["vloopback:v2"]


@pytest.mark.asyncio
@pytest.mark.parametrize("probe", [None, _disabled_plugin_probe()])
async def test_create_local_volume_sends_no_fallback_command_when_a_channel_may_still_hold_the_session(
    docker_service, monkeypatch, probe
):
    run_with_host_timeout = AsyncMock(side_effect=SshSessionMayStillBeTakenError())
    monkeypatch.setattr(docker_service_module, "_run_with_host_timeout", run_with_host_timeout)

    with pytest.raises(SshSessionMayStillBeTakenError):
        await _create_volume(docker_service, _plugin_host(), probe)

    assert run_with_host_timeout.await_count == 1


# ---------------------------------------------------------------------------
# create_container: the flag decides whether the probe runs at all
# ---------------------------------------------------------------------------


@pytest.fixture
def svc():
    return DockerService(ssh_service=Mock(), redis_service=Mock(), attestation_service=Mock())


def _create_payload() -> ContainerCreateRequest:
    return _deploy_payload(disk_share=0.5, storage_limit_gb=1, volume_limit_gb=2)


@pytest.mark.asyncio
async def test_create_container_flag_off_never_probes(svc, monkeypatch):
    monkeypatch.setattr(settings, "RENTAL_VOLUME_FAST_PATH_ENABLED", False)
    ssh_client = _deploy_ssh_client()
    _patch_happy(svc, monkeypatch, ssh_client)
    monkeypatch.setattr(svc, "probe_volume_host", AsyncMock())

    await _run_create_container(svc, _create_payload())

    svc.probe_volume_host.assert_not_awaited()
    assert svc.resolve_volume_sizing.await_args.kwargs["host_probe"] is None
    assert svc.create_local_volume.await_args.kwargs["host_probe"] is None


@pytest.mark.asyncio
async def test_create_container_flag_on_probes_once_and_passes_it_on(svc, monkeypatch):
    monkeypatch.setattr(settings, "RENTAL_VOLUME_FAST_PATH_ENABLED", True)
    ssh_client = _deploy_ssh_client()
    _patch_happy(svc, monkeypatch, ssh_client)
    probe = VolumeHostProbe(
        docker_root_dir="/var/lib/docker",
        df_avail_bytes=900 * _SIZING_GB,
        vloopback_volume_names=[],
        loopback_plugin_enabled=True,
    )
    monkeypatch.setattr(svc, "probe_volume_host", AsyncMock(return_value=probe))

    await _run_create_container(svc, _create_payload())

    svc.probe_volume_host.assert_awaited_once()
    assert svc.probe_volume_host.await_args.kwargs["with_df"] is True
    assert svc.resolve_volume_sizing.await_args.kwargs["host_probe"] is probe
    assert svc.create_local_volume.await_args.kwargs["host_probe"] is probe


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides",
    [
        dict(disk_share=None, storage_limit_gb=1, volume_limit_gb=2),  # legacy passthrough
        dict(storage_limit_gb=None, volume_limit_gb=2),  # storage_opt_unsupported passthrough
    ],
)
async def test_create_container_flag_on_passthrough_sizing_with_a_limit_probes_without_df(
    svc, monkeypatch, overrides
):
    # Sizing is a passthrough (no df to measure) but the volume has a size, so the create still
    # needs the root dir and the plugin state: probe, without the helper container.
    monkeypatch.setattr(settings, "RENTAL_VOLUME_FAST_PATH_ENABLED", True)
    ssh_client = _deploy_ssh_client()
    _patch_happy(svc, monkeypatch, ssh_client)
    monkeypatch.setattr(svc, "probe_volume_host", AsyncMock(return_value=None))

    await _run_create_container(svc, _deploy_payload(**overrides))

    assert svc.probe_volume_host.await_args.kwargs["with_df"] is False
    assert svc.create_local_volume.await_args.kwargs["host_probe"] is None


@pytest.mark.asyncio
async def test_create_container_flag_on_unlimited_volume_on_passthrough_never_probes(
    svc, monkeypatch
):
    # No df to measure (passthrough) and no `-o size=` volume (no plugin, no root dir needed):
    # nothing would read the probe, so it is not run.
    monkeypatch.setattr(settings, "RENTAL_VOLUME_FAST_PATH_ENABLED", True)
    ssh_client = _deploy_ssh_client()
    _patch_happy(svc, monkeypatch, ssh_client)
    monkeypatch.setattr(svc, "probe_volume_host", AsyncMock())

    await _run_create_container(svc, _deploy_payload(storage_limit_gb=None, volume_limit_gb=None))

    svc.probe_volume_host.assert_not_awaited()
    assert svc.create_local_volume.await_args.kwargs["host_probe"] is None


@pytest.mark.asyncio
async def test_create_container_flag_on_probe_failure_takes_the_per_command_path(svc, monkeypatch):
    monkeypatch.setattr(settings, "RENTAL_VOLUME_FAST_PATH_ENABLED", True)
    ssh_client = _deploy_ssh_client()
    _patch_happy(svc, monkeypatch, ssh_client)
    monkeypatch.setattr(svc, "probe_volume_host", AsyncMock(return_value=None))

    result = await _run_create_container(svc, _create_payload())

    assert result.__class__.__name__ == "ContainerCreated"
    assert svc.resolve_volume_sizing.await_args.kwargs["host_probe"] is None
    assert svc.create_local_volume.await_args.kwargs["host_probe"] is None
