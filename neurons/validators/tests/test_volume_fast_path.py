"""DAH-3240 rental volume fast path: a failed loopback plugin enable never leaks host text into errors or logs."""

from __future__ import annotations

import asyncio
import shlex
from unittest.mock import AsyncMock, Mock

import asyncssh
import pytest
from core.config import settings
from payload_models.payloads import ContainerCreateRequest
from services import docker_service as docker_service_module
from services.loopback_plugin_setup import loopback_plugin_setup_command
from services.docker_service import (
    DockerService,
    NoUsableLoopbackPluginError,
    SshSessionMayStillBeTakenError,
    VolumeHostProbe,
    _run_with_host_timeout,
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
    ssh_client_answering_through_run,
)


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


# ---------------------------------------------------------------------------
# the parser
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# the command and the parser together, through a real shell and a docker stub
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# probe_volume_host is never fatal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_probe_volume_host_lets_a_session_that_may_stay_taken_stop_the_create(docker_service, monkeypatch):
    monkeypatch.setattr(
        docker_service_module, "_run_with_host_timeout", AsyncMock(side_effect=SshSessionMayStillBeTakenError())
    )

    with pytest.raises(SshSessionMayStillBeTakenError):
        await docker_service.probe_volume_host(Mock(), with_df=True, log_extra={})


# ---------------------------------------------------------------------------
# resolve_volume_sizing: same numbers, fewer commands
# ---------------------------------------------------------------------------


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
async def test_v2_switch_uses_enabled_v1_even_when_the_probe_saw_enabled_v2(docker_service, monkeypatch):
    # Arrange
    monkeypatch.setattr(settings, "LOOPBACK_PLUGIN_V2_ENABLED", False)
    ssh_client = _plugin_host(old_state="true")
    probe = VolumeHostProbe("/var/lib/docker", None, [], loopback_plugin_enabled=True)

    # Act
    created = await _create_volume(docker_service, ssh_client, probe)

    # Assert: the switch prevents v2 installation, verification, enable and volume creation.
    assert created[0]["driver"] == "vloopback"
    commands = [call.args[0] for call in ssh_client.run.await_args_list]
    assert len(commands) == 1
    assert "vloopback:v2" not in commands[0]
    assert "python3" not in commands[0]


@pytest.mark.asyncio
async def test_v2_setup_transport_loss_stops_without_fallback_or_volume_creation(docker_service, monkeypatch):
    # Arrange
    run = AsyncMock(side_effect=asyncssh.ConnectionLost("connection closed"))
    monkeypatch.setattr(docker_service_module, "_run_with_host_timeout", run)

    # Act
    with pytest.raises(asyncssh.ConnectionLost):
        await _create_volume(docker_service, Mock(), _disabled_plugin_probe())

    # Assert: a lost SSH transport cannot carry another command or a Docker SDK operation.
    assert run.await_count == 1
    assert docker_service.rental_docker_client_factory.client.created_volumes == []


def _disabled_plugin_probe() -> VolumeHostProbe:
    return VolumeHostProbe(
        docker_root_dir="/var/lib/docker",
        df_avail_bytes=None,
        vloopback_volume_names=[],
        loopback_plugin_enabled=False,
        loopback_plugin_installed=True,
    )


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

    assert "v2 setup error: OSError" in str(exc_info.value)
    assert secret_text not in str(exc_info.value)
    assert all(secret_text not in str(call) for call in warning.call_args_list)



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


def _plugin_host(
    *, v2_state: str = "absent", old_state: str = "true", install: str = "fails", docker_root: str = "/var/lib/docker", enable_succeeds: bool = False
) -> Mock:
    # the host's docker through ssh: a v2 install or enable that fails (or never answers), and the plugin states
    async def answer(command: str):
        nonlocal v2_state
        command = shlex.split(command)[-1] if command.startswith("timeout -k 5 ") else command
        if command.startswith("python3 "):
            if v2_state == "true" or (v2_state == "false" and enable_succeeds):
                return Mock(stdout="verified v2 ready\n", stderr="", exit_status=0)
            if install == "hangs":
                await asyncio.Event().wait()
            return Mock(stdout="", stderr="v2 installation did not produce a plugin", exit_status=1)
        if "docker info" in command:
            return Mock(stdout=f"{docker_root}\n", stderr="", exit_status=0)
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
        (None, "fails", "absent", "v2 setup exit 1:"),
        (None, "hangs", "absent", "v2 setup timed out"),
        (_disabled_plugin_probe(), "fails", "false", "v2 setup exit 1:"),
        (None, "fails", "false", "v2 setup exit 1:"),
        (None, "fails", "unreadable", "v2 setup exit 1:"),
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
    assert message.startswith("no usable vloopback plugin on host (v2: v2 setup exit 1:")
    assert message.endswith(f"; old plugin vloopback state {old_state})")
    commands = [call.args[0] for call in ssh_client.run.await_args_list]
    assert all("vloopback:v2" in command for command in commands if "plugin install" in command)
    assert not any("plugin enable" in command for command in commands)
    assert docker_service.rental_docker_client_factory.client.created_volumes == []


@pytest.mark.asyncio
@pytest.mark.parametrize("v2_state, old_state, driver", [("true", "absent", "vloopback:v2"), ("absent", "true", "vloopback")])
async def test_create_local_volume_without_a_docker_data_root_never_installs_v2(
    docker_service, v2_state, old_state, driver
):
    ssh_client = _plugin_host(v2_state=v2_state, old_state=old_state, docker_root="")

    created = await _create_volume(docker_service, ssh_client, None)

    assert [volume["driver"] for volume in created] == [driver]
    assert not any("plugin install" in call.args[0] for call in ssh_client.run.await_args_list)


@pytest.mark.asyncio
@pytest.mark.parametrize("v2_state", ["true", "false"])
@pytest.mark.parametrize("old_state", ["absent", "false", "true"])
async def test_create_local_volume_without_probe_keeps_v2_when_its_install_says_already_exists(
    docker_service, v2_state, old_state
):
    # Arrange: a failed probe leaves the per-command path with an installed v2.
    ssh_client = _plugin_host(v2_state=v2_state, old_state=old_state, enable_succeeds=True)
    # Act
    created = await _create_volume(docker_service, ssh_client, None)
    # Assert: recover disabled v2 before considering any legacy plugin, whatever its state.
    assert [volume["driver"] for volume in created] == ["vloopback:v2"]
    commands = [call.args[0] for call in ssh_client.run.await_args_list]
    assert commands[-1] == _bounded(loopback_plugin_setup_command("/var/lib/docker", "vloopback:v2"), 60)
    assert all("vloopback:v2" in command for command in commands if "plugin" in command)


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
async def test_create_container_probes_once_and_passes_it_on(svc, monkeypatch):
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
async def test_create_container_passthrough_sizing_with_a_limit_probes_without_df(
    svc, monkeypatch, overrides
):
    # Sizing is a passthrough (no df to measure) but the volume has a size, so the create still
    # needs the root dir and the plugin state: probe, without the helper container.
    ssh_client = _deploy_ssh_client()
    _patch_happy(svc, monkeypatch, ssh_client)
    monkeypatch.setattr(svc, "probe_volume_host", AsyncMock(return_value=None))

    await _run_create_container(svc, _deploy_payload(**overrides))

    assert svc.probe_volume_host.await_args.kwargs["with_df"] is False
    assert svc.create_local_volume.await_args.kwargs["host_probe"] is None


@pytest.mark.asyncio
async def test_create_container_unlimited_volume_on_passthrough_never_probes(
    svc, monkeypatch
):
    # No df to measure (passthrough) and no `-o size=` volume (no plugin, no root dir needed):
    # nothing would read the probe, so it is not run.
    ssh_client = _deploy_ssh_client()
    _patch_happy(svc, monkeypatch, ssh_client)
    monkeypatch.setattr(svc, "probe_volume_host", AsyncMock())

    await _run_create_container(svc, _deploy_payload(storage_limit_gb=None, volume_limit_gb=None))

    svc.probe_volume_host.assert_not_awaited()
    assert svc.create_local_volume.await_args.kwargs["host_probe"] is None


@pytest.mark.asyncio
async def test_create_container_probe_failure_takes_the_per_command_path(svc, monkeypatch):
    ssh_client = _deploy_ssh_client()
    _patch_happy(svc, monkeypatch, ssh_client)
    monkeypatch.setattr(svc, "probe_volume_host", AsyncMock(return_value=None))

    result = await _run_create_container(svc, _create_payload())

    assert result.__class__.__name__ == "ContainerCreated"
    assert svc.resolve_volume_sizing.await_args.kwargs["host_probe"] is None
    assert svc.create_local_volume.await_args.kwargs["host_probe"] is None
