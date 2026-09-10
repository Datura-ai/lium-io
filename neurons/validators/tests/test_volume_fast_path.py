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
  with the plugin absent it still installs (negative control);
- `create_container` never probes with the flag off, probes once with it on and hands the probe to
  both the sizing and the create.
"""

from __future__ import annotations

import os
import stat
import subprocess
from unittest.mock import AsyncMock, Mock

import pytest
from core.config import settings
from core.docker_utils import ALPINE_HELPER_IMAGE, df_command
from payload_models.payloads import ContainerCreateRequest
from services.docker_service import (
    DockerService,
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
    assert "/usr/bin/docker plugin inspect --format '{{.Enabled}}' vloopback" in command
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
  "plugin inspect") [ "$mode" = plugin-absent ] && { echo; exit 1; }; echo true ;;  # real docker: blank stdout line, then exit 1
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
            "driver": "vloopback",
            "driver_opts": {"size": "40g"},
            "timeout": 10,
        }
    ]


@pytest.mark.asyncio
async def test_create_local_volume_with_plugin_absent_still_installs_it(docker_service):
    # Negative control: the probe saw no enabled plugin → the install command runs as before,
    # with DATA_DIR taken from the probe's root dir (no second `docker info`).
    ssh_client = Mock()
    ssh_client.run = AsyncMock(return_value=Mock(stdout="", exit_status=0))
    probe = VolumeHostProbe(
        docker_root_dir="/data/docker",
        df_avail_bytes=None,
        vloopback_volume_names=[],
        loopback_plugin_enabled=False,
    )

    created = await _create_volume(docker_service, ssh_client, probe)

    assert ssh_client.run.await_count == 1
    assert ssh_client.run.await_args.args[0] == (
        "/usr/bin/docker plugin install ashald/docker-volume-loopback "
        "--alias vloopback --grant-all-permissions DATA_DIR=/data/docker/loopback"
    )
    assert created[0]["driver"] == "vloopback"


@pytest.mark.asyncio
async def test_create_local_volume_without_probe_keeps_the_per_command_path(docker_service):
    ssh_client = Mock()
    ssh_client.run = AsyncMock(return_value=Mock(stdout="/var/lib/docker\n", exit_status=0))

    await _create_volume(docker_service, ssh_client, None)

    commands = [c.args[0] for c in ssh_client.run.await_args_list]
    assert commands[0] == "/usr/bin/docker info --format '{{.DockerRootDir}}'"
    assert commands[1].startswith("/usr/bin/docker plugin install ashald/docker-volume-loopback ")
    assert len(commands) == 2


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
