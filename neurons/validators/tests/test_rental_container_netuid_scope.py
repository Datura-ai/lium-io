"""The sweeps remove only their own network's rental containers and volumes.

A testnet executor can share a Docker host with a mainnet one, and the validator's sweeps see the
whole host. Each container and volume the validator creates carries `io.lium.netuid`; a sweep
removes a labeled resource only when the label is its own netuid, and an unlabeled (legacy) one
only when it runs on mainnet (51).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from core.config import settings
from services.container_cleanup import ContainerCleanup
from services.docker_service import DockerService
from services.prerun_host_probe import (
    DOCKER_PS_ALL_NAMES_NETUID_CMD,
    DOCKER_VOLUME_LS_NAME_DRIVER_NETUID_CMD,
    PrerunHostProbe,
    ProbedVolume,
    parse_prerun_host_probe,
)
from services.rental_container_labels import (
    KIND_LABEL,
    NETUID_LABEL,
    VALIDATOR_LABEL,
    LabeledName,
    container_in_scope,
    foreign_rental_containers,
    netuid_owns,
    parse_names_with_netuid,
    ps_filter_names_netuid_command,
    rental_labels,
)

MAINNET = 51
STAGING = 37


def _result(stdout: str = "", exit_status: int = 0):
    return Mock(stdout=stdout, stderr="", exit_status=exit_status)


@pytest.fixture
def docker_service() -> DockerService:
    return DockerService(ssh_service=Mock(), redis_service=Mock(), attestation_service=Mock())


@pytest.fixture
def removals(monkeypatch) -> list[str]:
    ran: list[str] = []

    async def _retry(ssh, command, _tag):
        ran.append(command)

    monkeypatch.setattr("services.docker_service.retry_ssh_command", _retry)
    return ran


# a mainnet pod from before the label, a mainnet pod and filler with it, a staging pod and filler
HOST_LISTING = (
    "pod_legacy \npod_prod 51\nfiller_fprod 51\npod_stage 37\nfiller_fstage 37\nwatchtower \n"
)


# -------------------------------------------------------------------------------------------------
# the rule
# -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "caller", "owned"),
    [
        ("51", MAINNET, True),
        (None, MAINNET, True),
        ("37", MAINNET, False),
        ("37", STAGING, True),
        ("51", STAGING, False),
        (None, STAGING, False),
    ],
)
def test_netuid_owns(label, caller, owned):
    assert netuid_owns(label, caller) is owned


def test_unscoped_prefixes_keep_the_name_only_rule():
    assert container_in_scope("health_check_123", None, STAGING) is True
    assert container_in_scope("container_legacy", None, STAGING) is False
    assert container_in_scope("container_legacy", None, MAINNET) is True


def test_rental_labels_name_network_validator_and_kind():
    assert rental_labels(netuid=STAGING, validator_hotkey="5Validator", kind="probe") == {
        NETUID_LABEL: "37",
        VALIDATOR_LABEL: "5Validator",
        KIND_LABEL: "probe",
    }
    assert VALIDATOR_LABEL not in rental_labels(netuid=MAINNET, validator_hotkey=None, kind="pod")


def test_listings_print_the_netuid_label():
    label_field = '{{.Label "io.lium.netuid"}}'
    assert (
        DOCKER_PS_ALL_NAMES_NETUID_CMD
        == f"/usr/bin/docker ps -a --format '{{{{.Names}}}} {label_field}'"
    )
    assert label_field in DOCKER_VOLUME_LS_NAME_DRIVER_NETUID_CMD
    assert label_field in ps_filter_names_netuid_command("pod_*")


def test_parse_names_with_netuid_reads_empty_labels_as_none():
    assert parse_names_with_netuid("pod_a 51\npod_b \npod_c\n\n") == [
        LabeledName("pod_a", "51"),
        LabeledName("pod_b", None),
        LabeledName("pod_c", None),
    ]


def test_prerun_probe_keeps_container_and_volume_labels():
    stdout = "\n".join(
        [
            "PS\tpod_prod 51",
            "PS\tpod_legacy ",
            "PS_RC\t0",
            "VOL\tvolume_prod vloopback 51",
            "VOL\tvolume_legacy vloopback:latest ",
            "VOL_RC\t0",
            "MNT_RC\t0",
            "GPUMINORMAP_RC\t0",
            "GPUDEV_RC\t0",
            "SHARED_RC\t0",
            "SHAREDW_RC\t0",
            "LABEL_RC\t0",
        ]
    )
    probe = parse_prerun_host_probe(stdout, with_power=False)
    assert probe.container_names == ("pod_prod", "pod_legacy")
    assert probe.container_netuid_labels == (("pod_prod", "51"),)
    assert probe.volumes == (
        ProbedVolume("volume_prod", "vloopback", "51"),
        ProbedVolume("volume_legacy", "vloopback:latest", None),
    )


# -------------------------------------------------------------------------------------------------
# the create-time sweep (DockerService.clean_existing_containers)
# -------------------------------------------------------------------------------------------------


async def _sweep(docker_service, *, netuid, monkeypatch, host_probe=None):
    monkeypatch.setattr(settings, "BITTENSOR_NETUID", netuid)
    ssh = AsyncMock()
    ssh.run = AsyncMock(return_value=_result(HOST_LISTING))
    removed = await docker_service.clean_existing_containers(
        ssh_client=ssh,
        default_extra={},
        pod_name="pod_new",
        active_container_names=[],
        active_volume_names=[],
        host_probe=host_probe,
    )
    return removed, ssh


@pytest.mark.asyncio
async def test_create_sweep_on_mainnet_removes_legacy_and_own_and_keeps_staging(
    docker_service, removals, monkeypatch
):
    removed, ssh = await _sweep(docker_service, netuid=MAINNET, monkeypatch=monkeypatch)
    assert [call.args[0] for call in ssh.run.await_args_list] == [DOCKER_PS_ALL_NAMES_NETUID_CMD]
    assert removed == ["pod_legacy", "pod_prod", "filler_fprod"]
    assert removals == [
        "/usr/bin/docker rm -fv pod_legacy pod_prod filler_fprod",
        "/usr/bin/docker volume rm volume_legacy volume_prod volume_fprod 2>/dev/null || true",
    ]


@pytest.mark.asyncio
async def test_create_sweep_on_staging_keeps_prod_labeled_and_unlabeled(
    docker_service, removals, monkeypatch
):
    removed, _ = await _sweep(docker_service, netuid=STAGING, monkeypatch=monkeypatch)
    assert removed == ["pod_stage", "filler_fstage"]
    assert removals == [
        "/usr/bin/docker rm -fv pod_stage filler_fstage",
        "/usr/bin/docker volume rm volume_stage volume_fstage 2>/dev/null || true",
    ]


@pytest.mark.asyncio
async def test_create_sweep_on_staging_with_only_prod_and_legacy_removes_nothing(
    docker_service, removals, monkeypatch
):
    monkeypatch.setattr(settings, "BITTENSOR_NETUID", STAGING)
    ssh = AsyncMock()
    ssh.run = AsyncMock(return_value=_result("pod_legacy \npod_prod 51\nfiller_prod 51\n"))
    removed = await docker_service.clean_existing_containers(
        ssh_client=ssh, default_extra={}, pod_name="pod_new", active_container_names=[]
    )
    assert removed == []
    assert removals == []


@pytest.mark.asyncio
async def test_create_sweep_reads_labels_from_the_prerun_probe(
    docker_service, removals, monkeypatch
):
    probe = PrerunHostProbe(
        container_names=("pod_legacy", "pod_prod", "pod_stage"),
        volumes=(),
        mounted_volume_names=(),
        gpu_minor_map_stdout="",
        gpu_device_nodes=(),
        shared_nodes=(),
        shared_nodes_whole_host_only=(),
        power_state_stdout=None,
        image_label_value="",
        container_netuid_labels=(("pod_prod", "51"), ("pod_stage", "37")),
    )
    removed, ssh = await _sweep(
        docker_service, netuid=STAGING, monkeypatch=monkeypatch, host_probe=probe
    )
    assert ssh.run.await_args_list == []
    assert removed == ["pod_stage"]


@pytest.mark.asyncio
async def test_create_sweep_still_clears_its_own_pod_name(docker_service, removals, monkeypatch):
    monkeypatch.setattr(settings, "BITTENSOR_NETUID", STAGING)
    ssh = AsyncMock()
    ssh.run = AsyncMock(return_value=_result("pod_new \n"))
    removed = await docker_service.clean_existing_containers(
        ssh_client=ssh, default_extra={}, pod_name="pod_new", active_container_names=[]
    )
    assert removed == ["pod_new"]


# -------------------------------------------------------------------------------------------------
# the create-time vloopback volume sweep (DockerService.clean_stale_vloopback_volumes)
# -------------------------------------------------------------------------------------------------

VOLUME_LISTING = (
    "volume_legacy vloopback \nvolume_prod vloopback:latest 51\nvolume_stage vloopback 37\n"
)


async def _volume_sweep(docker_service, netuid, monkeypatch):
    monkeypatch.setattr(settings, "BITTENSOR_NETUID", netuid)
    ssh = AsyncMock()
    ssh.run = AsyncMock(side_effect=[_result(VOLUME_LISTING), _result("")])
    return await docker_service.clean_stale_vloopback_volumes(ssh_client=ssh, default_extra={})


@pytest.mark.asyncio
async def test_vloopback_sweep_on_mainnet_removes_legacy_and_own(
    docker_service, removals, monkeypatch
):
    assert await _volume_sweep(docker_service, MAINNET, monkeypatch) == [
        "volume_legacy",
        "volume_prod",
    ]
    assert removals == ["/usr/bin/docker volume rm volume_legacy volume_prod 2>/dev/null || true"]


@pytest.mark.asyncio
async def test_vloopback_sweep_on_staging_removes_only_its_own(
    docker_service, removals, monkeypatch
):
    assert await _volume_sweep(docker_service, STAGING, monkeypatch) == ["volume_stage"]
    assert removals == ["/usr/bin/docker volume rm volume_stage 2>/dev/null || true"]


# -------------------------------------------------------------------------------------------------
# the stale sweep every validation pass runs (ContainerCleanup.cleanup)
# -------------------------------------------------------------------------------------------------

STALE_LISTING = "pod_legacy \npod_prod 51\npod_stage 37\ncontainer_old \nhealth_check_1 \n"


def _stale_ssh() -> tuple[AsyncMock, list[str]]:
    commands: list[str] = []

    async def run(command, *args, **kwargs):
        commands.append(command)
        if command.startswith("/usr/bin/docker ps -a"):
            return _result(STALE_LISTING)
        if "json .Created" in command:
            return _result("1000")
        if command == "date +%s":
            return _result(str(1000 + 60 * 60))
        return _result("")

    ssh = AsyncMock()
    ssh.run = AsyncMock(side_effect=run)
    return ssh, commands


def _removed(commands: list[str]) -> list[str]:
    return [
        command.split()[-1] for command in commands if command.startswith("/usr/bin/docker rm -fv")
    ]


@pytest.mark.asyncio
async def test_stale_sweep_on_mainnet_removes_legacy_and_own_and_keeps_staging():
    ssh, commands = _stale_ssh()
    removed_count, removed, _ = await ContainerCleanup(netuid=MAINNET).cleanup(ssh, None, "ex")
    assert removed == ["pod_legacy", "pod_prod", "container_old", "health_check_1"]
    assert removed_count == 4
    assert "pod_stage" not in _removed(commands)
    assert commands[0] == ps_filter_names_netuid_command(
        "pod_*", "filler_*", "container_*", "health_check_*"
    )


@pytest.mark.asyncio
async def test_stale_sweep_on_staging_keeps_prod_labeled_and_unlabeled_pods():
    ssh, commands = _stale_ssh()
    _, removed, _ = await ContainerCleanup(netuid=STAGING).cleanup(ssh, None, "ex")
    # health_check_* is the backend's short-lived port probe and keeps the name-only rule
    assert removed == ["pod_stage", "health_check_1"]
    assert _removed(commands) == ["pod_stage", "health_check_1"]
    assert "/usr/bin/docker volume rm volume_stage 2>/dev/null || true" in commands
    assert not any("volume_prod" in command or "volume_legacy" in command for command in commands)


def test_container_cleanup_defaults_to_the_validators_netuid(monkeypatch):
    monkeypatch.setattr(settings, "BITTENSOR_NETUID", STAGING)
    assert ContainerCleanup().netuid == STAGING


# -------------------------------------------------------------------------------------------------
# the rental probe's guard
# -------------------------------------------------------------------------------------------------


def test_foreign_rental_containers_counts_unlabeled_even_on_mainnet():
    listed = parse_names_with_netuid("pod_legacy \npod_prod 51\nfiller_stage 37\nwatchtower \n")
    assert foreign_rental_containers(listed, MAINNET) == ["pod_legacy", "filler_stage"]
    assert foreign_rental_containers(listed, STAGING) == ["pod_legacy", "pod_prod"]
    assert foreign_rental_containers(parse_names_with_netuid("pod_x 37\n"), STAGING) == []
