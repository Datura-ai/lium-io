"""Tests for services.container_cleanup.ContainerCleanup.

DAH-1991: cleanup must pick up `health_check_*` in addition to `pod_*` and
`container_*` so a stale backend probe cannot hold the rental port range.

DAH-2375: prune_dangling_anonymous_volumes must reap orphaned anonymous
volumes left behind by historical `docker rm` without `-v`.
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
      - docker rm -fv / volume rm -> success
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
        # docker rm -f / docker rm -fv
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
    resp.filler_containers_by_executor = {}
    resp.get_filler_container.side_effect = lambda key: resp.filler_containers_by_executor.get(str(key))
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
    assert "filler_*" in ps_cmd
    assert "container_*" in ps_cmd
    assert "health_check_*" in ps_cmd


@pytest.mark.asyncio
async def test_cleanup_preserves_active_filler_container():
    name = "filler_active"
    ssh, rm_calls = _make_ssh_mock(
        containers=[name],
        ages_by_name={name: 60},
    )
    cleanup = ContainerCleanup(stale_threshold_minutes=15)
    rented_data = _rented_data(EXECUTOR_UUID, [])
    rented_data.filler_containers_by_executor = {EXECUTOR_UUID: name}

    removed_count, removed_names = await cleanup.cleanup(
        ssh_client=ssh,
        rented_data=rented_data,
        executor_uuid=EXECUTOR_UUID,
    )

    assert removed_count == 0
    assert removed_names == []
    assert not any("docker rm -f" in c for c in rm_calls)


@pytest.mark.asyncio
async def test_cleanup_preserves_young_unknown_filler_container():
    name = "filler_young"
    ssh, rm_calls = _make_ssh_mock(
        containers=[name],
        ages_by_name={name: 5},
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
async def test_cleanup_removes_stale_unknown_filler_container():
    name = "filler_stale"
    ssh, rm_calls = _make_ssh_mock(
        containers=[name],
        ages_by_name={name: 30},
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
    assert any("docker rm -fv" in c and name in c for c in rm_calls)


# ---------------------------------------------------------------------------
# DAH-1991: force_remove_health_checks — eager cleanup at the spawn site
# (RentalVerificationCheck), no age threshold.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_force_remove_health_checks_issues_filtered_rm_command():
    """The cleanup must issue a single docker ps | xargs docker rm -f scoped to health_check_."""
    seen_cmds: list[str] = []

    async def handler(cmd, *args, **kwargs):
        seen_cmds.append(cmd)
        # Two container IDs echoed (one per line) on docker rm -f stdout.
        return MagicMock(exit_status=0, stdout="abc123\ndef456\n", stderr="")

    ssh = AsyncMock()
    ssh.run = AsyncMock(side_effect=handler)
    cleanup = ContainerCleanup()

    count = await cleanup.force_remove_health_checks(ssh_client=ssh, executor_uuid=EXECUTOR_UUID)

    assert count == 2
    assert len(seen_cmds) == 1
    cmd = seen_cmds[0]
    assert "docker ps -q" in cmd
    assert "name=^health_check_" in cmd
    assert "docker rm -f" in cmd
    # Must NOT mention pod_ or container_ — this method is health_check_-only.
    assert "pod_" not in cmd
    assert "name=^container_" not in cmd


@pytest.mark.asyncio
async def test_force_remove_health_checks_returns_zero_when_nothing_found():
    """Empty stdout from `docker rm -f` means no containers matched."""
    ssh = AsyncMock()
    ssh.run = AsyncMock(return_value=MagicMock(exit_status=0, stdout="", stderr=""))
    cleanup = ContainerCleanup()

    count = await cleanup.force_remove_health_checks(ssh_client=ssh, executor_uuid=EXECUTOR_UUID)

    assert count == 0


@pytest.mark.asyncio
async def test_force_remove_health_checks_ignores_age_threshold():
    """Even with stale_threshold_minutes=15, fresh probes are still removed.

    The whole point of this method is to bypass the age check, since the caller
    has just produced the probe and wants it gone now.
    """
    ssh = AsyncMock()
    ssh.run = AsyncMock(return_value=MagicMock(exit_status=0, stdout="abc123\n", stderr=""))
    cleanup = ContainerCleanup(stale_threshold_minutes=15)

    count = await cleanup.force_remove_health_checks(ssh_client=ssh, executor_uuid=EXECUTOR_UUID)

    # No `docker inspect ... Created` calls were issued — age was never consulted.
    issued = [call.args[0] for call in ssh.run.await_args_list]
    assert not any("docker inspect" in cmd for cmd in issued)
    assert count == 1


@pytest.mark.asyncio
async def test_force_remove_health_checks_swallows_exceptions():
    """SSH failure during cleanup must not propagate — it's best-effort."""
    ssh = AsyncMock()
    ssh.run = AsyncMock(side_effect=Exception("ssh broken"))
    cleanup = ContainerCleanup()

    count = await cleanup.force_remove_health_checks(ssh_client=ssh, executor_uuid=EXECUTOR_UUID)

    assert count == 0


# ---------------------------------------------------------------------------
# DAH-2375: prune_dangling_anonymous_volumes — reap anonymous volumes orphaned
# by container removals without -v.
# ---------------------------------------------------------------------------

ANON_VOLUME_A = "a" * 64
ANON_VOLUME_B = "b" * 64


def _volume_ssh_mock(dangling: list[str]):
    """Build an ssh mock simulating docker volume ls / rm."""
    rm_calls: list[str] = []

    async def handler(cmd, *args, **kwargs):
        if "docker volume ls" in cmd:
            return MagicMock(exit_status=0, stdout="\n".join(dangling), stderr="")
        if "docker volume rm" in cmd:
            rm_calls.append(cmd)
            return MagicMock(exit_status=0, stdout="", stderr="")
        return MagicMock(exit_status=0, stdout="", stderr="")

    return _ssh_mock_from_calls(handler), rm_calls


@pytest.mark.asyncio
async def test_prune_keeps_named_volumes_even_if_dangling():
    ssh, rm_calls = _volume_ssh_mock(
        dangling=["volume_abc", "hc_x", "executor_db_data", "pod_123"],
    )
    cleanup = ContainerCleanup()

    removed = await cleanup.prune_dangling_anonymous_volumes(ssh, EXECUTOR_UUID)

    assert removed == 0
    assert not rm_calls


@pytest.mark.asyncio
async def test_prune_removes_anonymous_keeps_named_mixed():
    ssh, rm_calls = _volume_ssh_mock(
        dangling=[ANON_VOLUME_A, ANON_VOLUME_B, "volume_pod1"],
    )
    cleanup = ContainerCleanup()

    removed = await cleanup.prune_dangling_anonymous_volumes(ssh, EXECUTOR_UUID)

    assert removed == 2
    assert len(rm_calls) == 1
    assert ANON_VOLUME_A in rm_calls[0]
    assert ANON_VOLUME_B in rm_calls[0]
    assert "volume_pod1" not in rm_calls[0]


@pytest.mark.asyncio
async def test_prune_never_raises_on_ssh_failure():
    ssh = AsyncMock()
    ssh.run = AsyncMock(side_effect=Exception("ssh broken"))
    cleanup = ContainerCleanup()

    removed = await cleanup.prune_dangling_anonymous_volumes(ssh, EXECUTOR_UUID)

    assert removed == 0


@pytest.mark.asyncio
async def test_prune_dry_run_skips_removal():
    ssh, rm_calls = _volume_ssh_mock(dangling=[ANON_VOLUME_A])
    cleanup = ContainerCleanup(dry_run=True)

    removed = await cleanup.prune_dangling_anonymous_volumes(ssh, EXECUTOR_UUID)

    assert removed == 0
    assert not rm_calls


@pytest.mark.asyncio
async def test_cleanup_invokes_volume_prune():
    """cleanup() must trigger the dangling-volume sweep at the end."""
    ssh, rm_calls = _volume_ssh_mock(dangling=[ANON_VOLUME_A])
    cleanup = ContainerCleanup(stale_threshold_minutes=15)

    removed_count, removed_names = await cleanup.cleanup(
        ssh_client=ssh,
        rented_data=None,
        executor_uuid=EXECUTOR_UUID,
    )

    # Container return contract unchanged; the volume was still pruned.
    assert removed_count == 0
    assert removed_names == []
    assert any("docker volume rm" in c and ANON_VOLUME_A in c for c in rm_calls)
