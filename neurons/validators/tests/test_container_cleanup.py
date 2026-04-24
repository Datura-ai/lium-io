"""Tests for services.container_cleanup.ContainerCleanup.

DAH-1991: cleanup must pick up `health_check_*` in addition to `pod_*` and
`container_*` so a stale backend probe cannot hold the rental port range.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from services.container_cleanup import ContainerCleanup


def _ssh_mock_from_calls(call_handler):
    """Return an AsyncMock ssh_client whose .run(cmd) routes to call_handler(cmd)."""
    ssh = AsyncMock()
    ssh.run = AsyncMock(side_effect=call_handler)
    return ssh


def _make_ssh_mock(containers: list[str], ages_by_name: dict[str, int], current_ts: int = 1_000_000_000):
    """Build an ssh mock that simulates:
      - docker ps -a filter -> newline list of containers
      - docker inspect ... Created -> age seconds (current_ts - age_seconds*60)
      - date +%s -> current_ts
      - docker rm -f / volume rm -> success
    """
    rm_calls: list[str] = []

    async def handler(cmd, *args, **kwargs):
        # docker ps -a ... --filter name=... returns the container list
        if "docker ps -a" in cmd:
            return MagicMock(exit_status=0, stdout="\n".join(containers), stderr="")
        # current time on host
        if cmd.strip() == "date +%s":
            return MagicMock(exit_status=0, stdout=str(current_ts), stderr="")
        # docker inspect --format '{{json .Created}}' | xargs -I {} date -d {} +%s
        if "docker inspect" in cmd and "Created" in cmd:
            # match which container
            for name in ages_by_name:
                if name in cmd:
                    age_min = ages_by_name[name]
                    created_ts = current_ts - int(age_min * 60)
                    return MagicMock(exit_status=0, stdout=str(created_ts), stderr="")
            return MagicMock(exit_status=1, stdout="", stderr="not found")
        # docker rm -f
        if "docker rm -f" in cmd:
            rm_calls.append(cmd)
            return MagicMock(exit_status=0, stdout="", stderr="")
        # docker volume rm
        if "docker volume rm" in cmd:
            rm_calls.append(cmd)
            return MagicMock(exit_status=0, stdout="", stderr="")
        return MagicMock(exit_status=0, stdout="", stderr="")

    return _ssh_mock_from_calls(handler), rm_calls


def _rented_data(executor_uuid: str, pods: list[str]):
    """Build a RentedExecutorsResponse-like mock."""
    pod_mocks = [MagicMock(container_name=name) for name in pods]
    executor_mock = MagicMock(pods=pod_mocks)
    resp = MagicMock()
    resp.executors = {executor_uuid: executor_mock}
    return resp


EXECUTOR_UUID = "00000000-0000-0000-0000-000000000000"


@pytest.mark.asyncio
async def test_cleanup_removes_stale_running_health_check():
    """A running health_check_* older than threshold gets removed (port-freeing case)."""
    name = "health_check_1777040756"
    ssh, rm_calls = _make_ssh_mock(
        containers=[name],
        ages_by_name={name: 30},  # 30 min — older than default 15 min threshold
    )
    cleanup = ContainerCleanup(stale_threshold_minutes=15)

    removed_count, removed_names = await cleanup.cleanup(
        ssh_client=ssh,
        rented_data=None,
        executor_uuid=EXECUTOR_UUID,
    )

    assert removed_count == 1
    assert removed_names == [name]
    assert any("docker rm -f" in c and name in c for c in rm_calls)


@pytest.mark.asyncio
async def test_cleanup_removes_stale_exited_health_check():
    """An exited health_check_* older than threshold is still cleaned up (cosmetic).

    docker ps -a returns both running and exited containers; the cleanup path
    does not branch on status, so exited containers follow the same path.
    Port release already happened on exit — this is cosmetic cleanup only.
    """
    name = "health_check_1600000000"
    ssh, rm_calls = _make_ssh_mock(
        containers=[name],
        ages_by_name={name: 120},
    )
    cleanup = ContainerCleanup(stale_threshold_minutes=15)

    removed_count, removed_names = await cleanup.cleanup(
        ssh_client=ssh,
        rented_data=None,
        executor_uuid=EXECUTOR_UUID,
    )

    assert removed_count == 1
    assert removed_names == [name]


@pytest.mark.asyncio
async def test_cleanup_preserves_young_health_check():
    """A health_check_* younger than threshold is left alone (normal ~30s probe lifecycle)."""
    name = "health_check_1777040800"
    ssh, rm_calls = _make_ssh_mock(
        containers=[name],
        ages_by_name={name: 1},  # 1 min — well under 15-min threshold
    )
    cleanup = ContainerCleanup(stale_threshold_minutes=15)

    removed_count, removed_names = await cleanup.cleanup(
        ssh_client=ssh,
        rented_data=None,
        executor_uuid=EXECUTOR_UUID,
    )

    assert removed_count == 0
    assert removed_names == []
    assert not any("docker rm -f" in c for c in rm_calls)


@pytest.mark.asyncio
async def test_cleanup_preserves_rented_health_check():
    """A container appearing in rented_data is preserved even if old (filter precondition)."""
    name = "health_check_ACTIVE"
    ssh, rm_calls = _make_ssh_mock(
        containers=[name],
        ages_by_name={name: 60},
    )
    cleanup = ContainerCleanup(stale_threshold_minutes=15)

    removed_count, removed_names = await cleanup.cleanup(
        ssh_client=ssh,
        rented_data=_rented_data(EXECUTOR_UUID, [name]),
        executor_uuid=EXECUTOR_UUID,
    )

    assert removed_count == 0
    assert removed_names == []
    assert not any("docker rm -f" in c for c in rm_calls)


@pytest.mark.asyncio
async def test_cleanup_filter_includes_all_rental_prefixes():
    """_get_all_rental_containers issues a docker ps with pod_*, container_*, health_check_*.

    Regression guard for DAH-1991: the prefix list is unified via
    RENTAL_CONTAINER_PREFIXES in services/const.py. If a new prefix is added
    there, both callsites (this one and wait_for_port_check_containers) inherit.
    """
    seen_cmds: list[str] = []

    async def handler(cmd, *args, **kwargs):
        seen_cmds.append(cmd)
        if "docker ps -a" in cmd:
            return MagicMock(exit_status=0, stdout="", stderr="")
        return MagicMock(exit_status=0, stdout="", stderr="")

    ssh = AsyncMock()
    ssh.run = AsyncMock(side_effect=handler)
    cleanup = ContainerCleanup(stale_threshold_minutes=15)

    await cleanup.cleanup(ssh_client=ssh, rented_data=None, executor_uuid=EXECUTOR_UUID)

    ps_cmd = next((c for c in seen_cmds if "docker ps -a" in c), "")
    assert "pod_*" in ps_cmd
    assert "container_*" in ps_cmd
    assert "health_check_*" in ps_cmd
