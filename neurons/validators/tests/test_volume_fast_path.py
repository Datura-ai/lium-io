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
import re
import stat
import subprocess
from unittest.mock import AsyncMock, Mock

import asyncssh
import pytest
from core.config import settings
from core.docker_utils import ALPINE_HELPER_IMAGE, df_command
from payload_models.payloads import ContainerCreateRequest
from services.docker_service import (
    DockerService,
    LoopbackPluginDisabledError,
    VolumeHostProbe,
    _parse_volume_host_probe,
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
            "VOL\tother_volume\tlocal\n"
            "VOL\tbad name;rm\tvloopback\n"
        ),
    )

    probe = _parse_volume_host_probe(stdout, with_df=True)

    assert probe.docker_root_dir == "/var/lib/docker"
    assert probe.df_avail_bytes == 966367641600
    assert probe.vloopback_volume_names == ["volume_abc", "volume_def"]
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

    probe = await docker_service.probe_volume_host(ssh_client, with_df=True, log_extra={})

    assert isinstance(probe, VolumeHostProbe)
    assert ssh_client.run.await_count == 1
    assert ssh_client.run.await_args.args[0] == _volume_host_probe_command(with_df=True)


@pytest.mark.asyncio
async def test_probe_volume_host_ssh_error_returns_none(docker_service):
    ssh_client = Mock()
    ssh_client.run = AsyncMock(side_effect=Exception("ssh boom"))

    assert await docker_service.probe_volume_host(ssh_client, with_df=True, log_extra={}) is None


@pytest.mark.asyncio
async def test_probe_volume_host_garbage_output_returns_none(docker_service):
    ssh_client = Mock()
    ssh_client.run = AsyncMock(return_value=Mock(stdout="nothing useful\n", exit_status=0))

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
    volume_inspect_stdout = f"{300 * _SIZING_GB}|<no value>\n"
    per_command_ssh = _make_sizing_ssh_client(
        df_avail_bytes=900 * _SIZING_GB,
        volume_ls_stdout="volume_abc vloopback:latest\nother_volume local\n",
        volume_inspect_stdout=volume_inspect_stdout,
    )
    probe = VolumeHostProbe(
        docker_root_dir="/var/lib/docker",
        df_avail_bytes=900 * _SIZING_GB,
        vloopback_volume_names=["volume_abc"],
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
    assert with_probe.path == "fresh" and with_probe.volume_limit_gb == 393
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
async def test_create_local_volume_installs_v2_with_host_data_dir_and_own_state_dir_when_v2_is_absent(
    docker_service,
):
    # The probe saw no v2 plugin → the v2 install runs, with DATA_DIR on the host under the
    # probe's root dir (no second `docker info`), outside the plugin rootfs.
    ssh_client = ssh_client_answering_through_run(
        AsyncMock(return_value=Mock(stdout="", exit_status=0))
    )
    probe = VolumeHostProbe(
        docker_root_dir="/data/docker",
        df_avail_bytes=None,
        vloopback_volume_names=[],
        loopback_plugin_enabled=False,
    )

    created = await _create_volume(docker_service, ssh_client, probe)

    assert ssh_client.run.await_count == 1
    assert ssh_client.run.await_args.args[0] == (
        "timeout -k 5 60 /usr/bin/docker plugin install daturaai/docker-volume-loopback:1.0.0-lium1 "
        "--alias vloopback:v2 --grant-all-permissions "
        "DATA_DIR=/srv/data/docker/vloopback-v2 STATE_DIR=/srv/run/docker-volume-loopback-v2"
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
async def test_create_local_volume_enables_v2_when_v2_is_installed_but_disabled(docker_service):
    # 7fcd02af (20-27 Sep): installed but disabled → `plugin install` failed "already exists"
    # and every create failed "plugin vloopback found but disabled". Enable, re-read, create.
    ssh_client = Mock()
    ssh_client.run = AsyncMock(
        side_effect=[
            Mock(stdout="vloopback:v2\n", stderr="", exit_status=0),
            Mock(stdout="true\n", stderr="", exit_status=0),
        ]
    )

    created = await _create_volume(docker_service, ssh_client, _disabled_plugin_probe())

    calls = ssh_client.run.await_args_list
    assert [c.args[0] for c in calls] == [
        "/usr/bin/docker plugin enable vloopback:v2",
        "( /usr/bin/docker plugin inspect --format '{{.Enabled}}' vloopback:v2 2>/dev/null "
        "|| echo absent) | tail -n 1",
    ]
    assert all(c.kwargs == {"timeout": 10} for c in calls)
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
async def test_create_local_volume_fails_fast_when_neither_v2_nor_the_old_plugin_will_enable(docker_service):
    enable_failed = Mock(
        stdout="", stderr="Error response from daemon: dial unix plugin.sock: connect: no such file", exit_status=1
    )
    still_disabled = Mock(stdout="false\n", stderr="", exit_status=0)
    ssh_client = Mock()
    ssh_client.run = AsyncMock(
        side_effect=[enable_failed, still_disabled, still_disabled, enable_failed, still_disabled]
    )

    with pytest.raises(LoopbackPluginDisabledError) as exc_info:
        await _create_volume(docker_service, ssh_client, _disabled_plugin_probe())

    message = str(exc_info.value)
    assert message.startswith("vloopback plugin disabled on host and could not be enabled")
    assert "(plugin vloopback: enable exit 1, state false" in message and "plugin.sock" in message
    # lium-platform's classifier files it as volume.plugin_disabled on these two substrings
    assert "plugin vloopback" in message.lower() and "disabled" in message.lower()
    assert docker_service.rental_docker_client_factory.client.created_volumes == []


@pytest.mark.asyncio
async def test_create_local_volume_falls_back_to_the_old_plugin_when_the_v2_enable_times_out(docker_service):
    ssh_client = Mock()
    ssh_client.run = AsyncMock(side_effect=[asyncio.TimeoutError(), Mock(stdout="true\n", stderr="", exit_status=0)])

    created = await _create_volume(docker_service, ssh_client, _disabled_plugin_probe())

    assert ssh_client.run.await_count == 2
    assert [volume["driver"] for volume in created] == ["vloopback"]


@pytest.mark.asyncio
async def test_create_local_volume_disabled_plugin_enable_error_logs_only_its_type(
    docker_service, monkeypatch
):
    from services import docker_service as docker_service_module

    secret_text = "ssh ubuntu@10.1.2.3: token=ghp_leakme at /home/provider/.ssh/id_rsa"
    ssh_client = Mock()
    ssh_client.run = AsyncMock(side_effect=[OSError(secret_text), Mock(stdout="true\n", stderr="", exit_status=0)])
    warning = Mock()
    monkeypatch.setattr(docker_service_module.logger, "warning", warning)

    await _create_volume(docker_service, ssh_client, _disabled_plugin_probe())

    (enable_failed,), _ = warning.call_args_list[0]
    (fell_back,), _ = warning.call_args_list[1]
    assert str(enable_failed).startswith("Loopback plugin enable failed")
    assert enable_failed.extra["error_type"] == "OSError"
    assert enable_failed.extra["loopback_plugin"] == "vloopback:v2"
    assert "error" not in enable_failed.extra
    assert "enable error: OSError" in fell_back.extra["reason"]
    for logged in (enable_failed, fell_back):
        full = logged.to_full_string()
        assert "10.1.2.3" not in full and "ghp_leakme" not in full and "id_rsa" not in full


@pytest.mark.asyncio
async def test_create_local_volume_without_probe_keeps_the_per_command_path(docker_service):
    ssh_client = ssh_client_answering_through_run(
        AsyncMock(return_value=Mock(stdout="/var/lib/docker\n", exit_status=0))
    )

    await _create_volume(docker_service, ssh_client, None)

    commands = [c.args[0] for c in ssh_client.run.await_args_list]
    assert commands[0] == "/usr/bin/docker info --format '{{.DockerRootDir}}'"
    assert commands[1].startswith(
        "timeout -k 5 60 /usr/bin/docker plugin install daturaai/docker-volume-loopback:1.0.0-lium1 "
    )
    assert len(commands) == 2


@pytest.mark.asyncio
async def test_create_local_volume_without_probe_keeps_v2_when_its_install_says_already_exists(docker_service):
    ssh_client = ssh_client_answering_through_run(
        AsyncMock(
            side_effect=[
                Mock(stdout="/var/lib/docker\n", stderr="", exit_status=0),
                Mock(
                    stdout="",
                    stderr="Error response from daemon: plugin vloopback:v2 already exists",
                    exit_status=1,
                ),
                Mock(stdout="true\n", stderr="", exit_status=0),
            ]
        )
    )

    created = await _create_volume(docker_service, ssh_client, None)

    assert ssh_client.run.await_count == 3
    assert [volume["driver"] for volume in created] == ["vloopback:v2"]


# ---------------------------------------------------------------------------
# vloopback v2 beside the old plugin: new volumes on `vloopback:v2`, the old one never named
# ---------------------------------------------------------------------------

_OLD = "vloopback:latest"
_V2 = "vloopback:v2"
_STATE_INSPECT_RE = re.compile(r"plugin inspect --format '\{\{\.Enabled\}\}' (\S+) ")
# a plugin named in a command (`--alias vloopback:v2`, `enable vloopback`), not the v2 data dir path
_LOOPBACK_ALIAS_RE = re.compile(r"(?<![\w/-])vloopback[\w:.-]*")


def _docker_plugin_name(name: str) -> str:
    return name if ":" in name else f"{name}:latest"


class _TwoLoopbackPluginHost:
    """Both the SSH connection and the Docker SDK client of a host, answering like dockerd with the
    old `vloopback:latest` and the new `vloopback:v2` plugins: an untagged name is `:latest`, a
    volume goes to the plugin its driver names, a missing or disabled plugin refuses the create."""

    def __init__(
        self,
        *,
        plugins: dict[str, bool],
        volumes: dict[str, tuple[str, int]] | None = None,
        v2_install_fails: bool = False,
        v2_enable_fails: bool = False,
        v2_install_hangs: bool = False,
        v2_install_answer_lost: bool = False,
        v2_install_channel_open_seconds: float = 0,
        v2_state_channel_open_hangs: bool = False,
        max_sessions: int | None = None,
    ):
        self.plugins = dict(plugins)  # plugin name -> enabled
        self.volumes = dict(volumes or {})  # volume name -> (driver, declared bytes)
        self.v2_install_fails = v2_install_fails
        self.v2_enable_fails = v2_enable_fails
        self.v2_install_hangs = v2_install_hangs
        self.v2_install_answer_lost = v2_install_answer_lost
        self.v2_install_channel_open_seconds = v2_install_channel_open_seconds
        self.v2_state_channel_open_hangs = v2_state_channel_open_hangs
        self.max_sessions = max_sessions
        self.open_channels = 0
        self.commands: list[str] = []

    def _state(self, command: str) -> str:
        name = _docker_plugin_name(_STATE_INSPECT_RE.search(command).group(1))
        if name not in self.plugins:
            return "absent"
        return "true" if self.plugins[name] else "false"

    async def create_process(self, command: str):
        if self.max_sessions is not None and self.open_channels >= self.max_sessions:
            raise asyncssh.ChannelOpenError(
                asyncssh.OPEN_ADMINISTRATIVELY_PROHIBITED, "open failed"
            )
        if (
            self.v2_state_channel_open_hangs
            and command.startswith("( /usr/bin/docker plugin inspect")
            and _V2 in command
        ):
            await asyncio.Event().wait()
        if "plugin install" in command and f"--alias {_V2} " in command:
            await asyncio.sleep(self.v2_install_channel_open_seconds)
        self.open_channels += 1
        return _HostProcess(self, command)

    async def run(self, command: str, **kwargs):
        process = await self.create_process(command)
        return await process.wait()

    async def answer(self, command: str, process: _HostProcess):
        self.commands.append(command)
        host_timeout = None
        bounded = re.match(r"timeout -k 5 (\S+) (.*)", command)
        if bounded:
            host_timeout, command = float(bounded.group(1)), bounded.group(2)
        if command.startswith("root="):  # the volume host probe
            lines = ["ROOT\t/var/lib/docker"]
            lines += [f"VOL\t{name}\t{driver}" for name, (driver, _) in self.volumes.items()]
            lines += ["VOLS\t0", f"PLUGIN\t{self._state(command)}"]
            return Mock(stdout="\n".join(lines) + "\n", stderr="", exit_status=0)
        if command.startswith("( /usr/bin/docker plugin inspect"):
            return Mock(stdout=self._state(command) + "\n", stderr="", exit_status=0)
        if command.startswith("/usr/bin/docker plugin install "):
            name = _docker_plugin_name(command.split("--alias ")[1].split()[0])
            if self.v2_install_hangs and name == _V2:
                # a pull from a registry that stopped answering, until coreutils' timeout ends it
                if host_timeout is None:
                    await asyncio.Event().wait()
                await asyncio.sleep(host_timeout)
                return Mock(stdout="", stderr="", exit_status=124)
            if self.v2_install_answer_lost and name == _V2:
                process.child_alive = False  # the host is done; its answer never arrives
                await asyncio.Event().wait()
            if self.v2_install_fails and name == _V2:
                return Mock(stdout="", stderr="Error response from daemon: Get https://registry-1.docker.io/v2/: net/http: request canceled", exit_status=1)
            self.plugins[name] = True
            return Mock(stdout="Installed plugin\n", stderr="", exit_status=0)
        if command.startswith("/usr/bin/docker plugin enable "):
            name = _docker_plugin_name(command.split()[-1])
            if self.v2_enable_fails and name == _V2:
                return Mock(stdout="", stderr="Error response from daemon: dial unix plugin.sock: connect: no such file", exit_status=1)
            if name in self.plugins:
                self.plugins[name] = True
            return Mock(stdout=command.split()[-1] + "\n", stderr="", exit_status=0)
        if command.startswith("/usr/bin/docker volume inspect "):
            names = command.split("--format")[0].split()[3:]
            return Mock(
                stdout="".join(f"{self.volumes[name][1]}|<no value>\n" for name in names),
                stderr="",
                exit_status=0,
            )
        raise AssertionError(f"unexpected command: {command}")

    async def create_volume(self, *, volume_name, driver=None, driver_opts=None, timeout=None):
        name = _docker_plugin_name(driver)
        if name not in self.plugins:
            raise RuntimeError(
                f"create {volume_name}: error looking up volume plugin {driver}: plugin \"{driver}\" not found"
            )
        if not self.plugins[name]:
            raise RuntimeError(
                f"create {volume_name}: error looking up volume plugin {driver}: plugin {name} found but disabled"
            )
        self.volumes[volume_name] = (name, 0)

    def plugin_commands(self) -> list[str]:
        return [command for command in self.commands if "docker plugin" in command]


class _HostProcess:
    # one SSH channel: closed by the host once the command exits, or by close(); like OpenSSH,
    # the session stays taken while its command still runs
    def __init__(self, host: _TwoLoopbackPluginHost, command: str):
        self.host = host
        self.command = command
        self.is_open = True
        self.child_alive = False

    async def wait(self):
        self.child_alive = True
        answer = await self.host.answer(self.command, self)
        self.child_alive = False
        self.close()
        return answer

    def close(self) -> None:
        if self.is_open and not self.child_alive:
            self.is_open = False
            self.host.open_channels -= 1


async def _rent_a_new_volume(
    docker_service, host: _TwoLoopbackPluginHost, timeout: float = 10
) -> None:
    docker_service.stream_log = AsyncMock()
    probe = await docker_service.probe_volume_host(host, with_df=False, log_extra={})
    await docker_service.create_local_volume(
        ssh_client=host,
        docker_client=host,
        local_volume="volume_new",
        log_tag="tag",
        log_text="Creating docker volume volume_new",
        log_extra={},
        limit=40,
        timeout=timeout,
        host_probe=probe,
    )


@pytest.mark.asyncio
async def test_create_local_volume_creates_the_new_volume_on_the_v2_driver(docker_service):
    host = _TwoLoopbackPluginHost(plugins={_OLD: True, _V2: True})

    await _rent_a_new_volume(docker_service, host)

    assert host.volumes["volume_new"][0] == _V2


@pytest.mark.asyncio
async def test_create_local_volume_installs_v2_beside_an_enabled_old_plugin(docker_service):
    host = _TwoLoopbackPluginHost(plugins={_OLD: True}, volumes={"volume_renter": (_OLD, 10**9)})

    await _rent_a_new_volume(docker_service, host)

    installs = [command for command in host.commands if "plugin install" in command]
    assert len(installs) == 1 and "--alias vloopback:v2 " in installs[0]
    assert host.plugins == {_OLD: True, _V2: True}
    assert host.volumes == {"volume_renter": (_OLD, 10**9), "volume_new": (_V2, 0)}


@pytest.mark.asyncio
async def test_create_local_volume_skips_install_when_v2_is_enabled(docker_service):
    host = _TwoLoopbackPluginHost(plugins={_V2: True})

    await _rent_a_new_volume(docker_service, host)

    assert not any("plugin install" in command for command in host.commands)
    assert host.volumes["volume_new"][0] == _V2


@pytest.mark.asyncio
async def test_create_local_volume_falls_back_to_the_old_plugin_when_v2_install_fails(
    docker_service, monkeypatch
):
    from services import docker_service as docker_service_module

    host = _TwoLoopbackPluginHost(plugins={_OLD: True}, v2_install_fails=True)
    warning = Mock()
    monkeypatch.setattr(docker_service_module.logger, "warning", warning)

    await _rent_a_new_volume(docker_service, host)

    assert host.volumes["volume_new"][0] == _OLD
    assert host.plugins == {_OLD: True}
    assert not any("ashald" in command for command in host.commands)
    (logged,), _ = warning.call_args
    assert str(logged).startswith("Loopback plugin v2 unusable; creating this volume on the old plugin")
    assert logged.extra["reason"].startswith("install exit 1, state absent: Error response from daemon")


@pytest.mark.asyncio
async def test_create_local_volume_falls_back_to_the_old_plugin_when_the_v2_install_hangs(
    docker_service, monkeypatch
):
    from services import docker_service as docker_service_module

    host = _TwoLoopbackPluginHost(plugins={_OLD: True}, v2_install_hangs=True, max_sessions=1)
    warning = Mock()
    monkeypatch.setattr(docker_service_module.logger, "warning", warning)
    monkeypatch.setattr(docker_service_module, "_LOOPBACK_PLUGIN_INSTALL_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(docker_service_module, "_LOOPBACK_PLUGIN_INSTALL_DEADLINE_SECONDS", 2)

    await asyncio.wait_for(_rent_a_new_volume(docker_service, host), timeout=5)

    assert host.volumes["volume_new"][0] == _OLD
    assert host.plugins == {_OLD: True}
    (logged,), _ = warning.call_args
    assert logged.extra["reason"].startswith("install exit 124, state absent")


@pytest.mark.asyncio
async def test_create_local_volume_waits_for_the_host_side_install_timeout_after_a_slow_channel_open(
    docker_service, monkeypatch
):
    # the channel opens late, so the host's timer ends the install after the validator's deadline
    # would if it counted from before the open: the session would still be taken (MaxSessions=1)
    from services import docker_service as docker_service_module

    host = _TwoLoopbackPluginHost(
        plugins={_OLD: True},
        v2_install_hangs=True,
        v2_install_channel_open_seconds=0.3,
        max_sessions=1,
    )
    monkeypatch.setattr(docker_service_module, "_LOOPBACK_PLUGIN_INSTALL_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(docker_service_module, "_LOOPBACK_PLUGIN_INSTALL_DEADLINE_SECONDS", 0.4)

    await asyncio.wait_for(_rent_a_new_volume(docker_service, host), timeout=5)

    assert host.volumes["volume_new"][0] == _OLD


@pytest.mark.asyncio
async def test_create_local_volume_falls_back_and_frees_the_channel_when_the_v2_install_answer_is_lost(
    docker_service, monkeypatch
):
    from services import docker_service as docker_service_module

    host = _TwoLoopbackPluginHost(plugins={_OLD: True}, v2_install_answer_lost=True, max_sessions=1)
    warning = Mock()
    monkeypatch.setattr(docker_service_module.logger, "warning", warning)
    monkeypatch.setattr(docker_service_module, "_LOOPBACK_PLUGIN_INSTALL_DEADLINE_SECONDS", 0.05)

    await asyncio.wait_for(_rent_a_new_volume(docker_service, host), timeout=5)

    assert host.volumes["volume_new"][0] == _OLD
    (logged,), _ = warning.call_args
    assert logged.extra["reason"].startswith("install timed out after 0.05 s, state absent")


@pytest.mark.asyncio
async def test_create_local_volume_falls_back_when_the_v2_state_read_after_a_hung_install_hangs_too(
    docker_service, monkeypatch
):
    from services import docker_service as docker_service_module

    host = _TwoLoopbackPluginHost(
        plugins={_OLD: True}, v2_install_hangs=True, v2_state_channel_open_hangs=True
    )
    monkeypatch.setattr(docker_service_module, "_LOOPBACK_PLUGIN_INSTALL_TIMEOUT_SECONDS", 0.05)

    await asyncio.wait_for(_rent_a_new_volume(docker_service, host, timeout=0.05), timeout=5)

    assert host.volumes["volume_new"][0] == _OLD


@pytest.mark.asyncio
async def test_create_local_volume_falls_back_to_the_old_plugin_when_v2_cannot_be_enabled(docker_service):
    host = _TwoLoopbackPluginHost(plugins={_OLD: True, _V2: False}, v2_enable_fails=True)

    await _rent_a_new_volume(docker_service, host)

    assert host.volumes["volume_new"][0] == _OLD
    assert host.plugins == {_OLD: True, _V2: False}


@pytest.mark.asyncio
async def test_create_local_volume_installs_the_old_plugin_as_main_does_when_v2_fails_and_old_is_absent(
    docker_service,
):
    host = _TwoLoopbackPluginHost(plugins={}, v2_install_fails=True)

    await _rent_a_new_volume(docker_service, host)

    installs = [command for command in host.commands if "plugin install" in command]
    assert installs[1] == (
        "/usr/bin/docker plugin install ashald/docker-volume-loopback "
        "--alias vloopback --grant-all-permissions DATA_DIR=/var/lib/docker/loopback"
    )
    assert len(installs) == 2
    assert host.volumes["volume_new"][0] == _OLD


@pytest.mark.asyncio
async def test_create_local_volume_enables_a_disabled_old_plugin_as_main_does_when_v2_fails(docker_service):
    host = _TwoLoopbackPluginHost(plugins={_OLD: False}, v2_install_fails=True)

    await _rent_a_new_volume(docker_service, host)

    assert "/usr/bin/docker plugin enable vloopback" in host.commands
    assert sum("plugin install" in command for command in host.commands) == 1
    assert host.volumes["volume_new"][0] == _OLD


@pytest.mark.asyncio
async def test_create_local_volume_with_v2_enabled_never_names_a_disabled_old_plugin(docker_service):
    host = _TwoLoopbackPluginHost(plugins={_OLD: False, _V2: True})

    await _rent_a_new_volume(docker_service, host)

    named = {alias for command in host.plugin_commands() for alias in _LOOPBACK_ALIAS_RE.findall(command)}
    assert named == {_V2}
    assert host.plugins == {_OLD: False, _V2: True}
    assert host.volumes["volume_new"][0] == _V2


@pytest.mark.asyncio
async def test_volume_host_probe_keeps_old_and_v2_volume_names(docker_service):
    host = _TwoLoopbackPluginHost(
        plugins={_OLD: True},
        volumes={
            "volume_old": (_OLD, 10**9),
            "volume_v2": (_V2, 10**9),
            "volume_local": ("local", 0),
        },
    )

    probe = await docker_service.probe_volume_host(host, with_df=False, log_extra={})

    assert probe.vloopback_volume_names == ["volume_old", "volume_v2"]
    # the plugin state is v2's: an enabled old plugin must not read as "nothing to install"
    assert probe.loopback_plugin_enabled is False
    assert probe.loopback_plugin_installed is False


@pytest.mark.asyncio
async def test_resolve_volume_sizing_sums_declared_sizes_of_old_and_v2_volumes(docker_service):
    # both generations' backing files sit on the DockerRootDir disk, so both declared sizes count
    host = _TwoLoopbackPluginHost(
        plugins={_OLD: True, _V2: True},
        volumes={"volume_old": (_OLD, 300 * _SIZING_GB), "volume_v2": (_V2, 200 * _SIZING_GB)},
    )
    probe = await docker_service.probe_volume_host(host, with_df=False, log_extra={})
    probe.df_avail_bytes = 900 * _SIZING_GB
    payload = _make_sizing_payload(disk_share=0.5, storage_limit_gb=1)

    sizing = await docker_service.resolve_volume_sizing(host, payload, "tag", {}, host_probe=probe)

    assert sizing.existing_volumes_bytes == 500 * _SIZING_GB


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
