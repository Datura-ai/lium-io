"""Tests for services.container_cleanup.ContainerCleanup.

DAH-1991: cleanup must pick up `health_check_*` in addition to `pod_*` and
`container_*` so a stale backend probe cannot hold the rental port range.

DAH-2375: prune_dangling_anonymous_volumes must reap orphaned anonymous
volumes left behind by historical `docker rm` without `-v`.
"""
import json
import logging
import random
from unittest.mock import AsyncMock, MagicMock

import pytest

from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod
from services.container_cleanup import (
    RENTED_LIST_EMPTY,
    RENTED_LIST_UNAVAILABLE,
    VOLUME_RM_MAX_PER_PASS,
    ContainerCleanup,
    listed_container_names,
    rented_list_unknown_reason,
)


def _ssh_mock_from_calls(call_handler):
    """Return an AsyncMock ssh_client whose .run(cmd) routes to call_handler(cmd)."""
    ssh = AsyncMock()
    ssh.run = AsyncMock(side_effect=call_handler)
    return ssh


def _ps_line(name: str, label: str) -> str:
    """One `docker ps` line as the listing's format prints it: a JSON [name, pod id label]."""
    return json.dumps([name, label])


def _label_of(name: str) -> str:
    """The LIUM_POD_ID_LABEL a Lium create gives; the backend's health_check_ probe has none."""
    if name.startswith("health_check_"):
        return ""
    if name.startswith("container_"):
        return name
    return name.removeprefix("pod_").removeprefix("filler_").removesuffix("__prev")


def _make_ssh_mock(
    containers: list[str],
    ages_by_name: dict[str, int],
    current_ts: int = 1_000_000_000,
    unlabelled: tuple[str, ...] = (),
):
    """Build an ssh mock that simulates:
      - docker ps -a filter -> newline list of `name|pod id label` (none for `unlabelled`)
      - docker inspect ... Created -> age seconds (current_ts - age_seconds*60)
      - date +%s -> current_ts
      - docker rm -fv / volume rm -> success
    """
    rm_calls: list[str] = []

    async def handler(cmd, *args, **kwargs):
        # docker ps -a ... --filter name=... returns the container list
        if "docker ps -a" in cmd:
            lines = [
                _ps_line(name, "" if name in unlabelled else _label_of(name)) for name in containers
            ]
            return MagicMock(exit_status=0, stdout="\n".join(lines), stderr="")
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
    pod_mocks = [RentedPod(pod_id=f"pod-{i}", container_name=name) for i, name in enumerate(pods)]
    executor_mock = MagicMock(pods=pod_mocks)
    resp = MagicMock()
    resp.executors = {executor_uuid: executor_mock}
    resp.filler_containers_by_executor = {}
    resp.all_filler_containers_by_executor = {}
    return resp


EXECUTOR_UUID = "00000000-0000-0000-0000-000000000000"

# A rented pod the backend lists but that is not on the host: removal is only allowed against a
# fetched, non-empty rented list, so tests that exercise a removal carry one.
LISTED_LIVE_POD = "pod_listed_live"


def _listed():
    return _rented_data(EXECUTOR_UUID, [LISTED_LIVE_POD])


@pytest.mark.asyncio
async def test_cleanup_removes_stale_running_health_check():
    """A running health_check_* older than threshold gets removed (port-freeing case)."""
    name = "health_check_1777040756"
    ssh, rm_calls = _make_ssh_mock(
        containers=[name],
        ages_by_name={name: 30},  # 30 min — older than default 15 min threshold
    )
    cleanup = ContainerCleanup(stale_threshold_minutes=15)

    removed_count, removed_names, _ = await cleanup.cleanup(
        ssh_client=ssh,
        rented_data=_listed(),
        executor_uuid=EXECUTOR_UUID,
    )

    assert removed_count == 1
    assert removed_names == [name]
    assert any("docker rm -f" in c and name in c for c in rm_calls)


@pytest.mark.asyncio
async def test_cleanup_awaits_on_before_remove_before_each_removal():
    """DAH-3338: the caller queues the reap for the backend BEFORE the container goes, so a crash
    between the removal and the report cannot lose it. Fails on a cleanup that removes first."""
    name = "pod_11655dc5-53ba-4a8d-a341-fe6c9d12bda7"
    ssh, rm_calls = _make_ssh_mock(containers=[name], ages_by_name={name: 30})
    order: list[str] = []

    async def on_before_remove(container_name: str) -> None:
        container_removals = [c for c in rm_calls if "docker rm -f" in c]
        order.append(f"queued {container_name} with {len(container_removals)} removal(s) so far")

    removed_count, removed_names, _ = await cleanup_with_hook(ssh, on_before_remove)

    assert removed_names == [name]
    assert order == [f"queued {name} with 0 removal(s) so far"]
    assert any("docker rm -f" in c and name in c for c in rm_calls)


@pytest.mark.asyncio
async def test_a_volume_rm_error_after_a_successful_rm_still_counts_the_container_as_removed():
    """Regression (review of DAH-3338): the volume removal ran inside the same try as `docker rm`,
    so an SSH error on it returned False for a container that was already gone. The caller then
    named it unremovable and dropped its `reaped` report. Fails on the old shape."""
    name = "pod_11655dc5-53ba-4a8d-a341-fe6c9d12bda7"
    ssh, rm_calls = _make_ssh_mock(containers=[name], ages_by_name={name: 30})
    plain_handler = ssh.run.side_effect

    async def handler(cmd, *args, **kwargs):
        if "docker volume rm" in cmd:
            raise OSError("ssh transport closed")
        return await plain_handler(cmd, *args, **kwargs)

    ssh.run.side_effect = handler

    removed_count, removed_names, unremovable = await ContainerCleanup(stale_threshold_minutes=15).cleanup(
        ssh_client=ssh, rented_data=_listed(), executor_uuid=EXECUTOR_UUID
    )

    assert (removed_count, removed_names, unremovable) == (1, [name], [])
    assert any("docker rm -f" in c and name in c for c in rm_calls)


@pytest.mark.asyncio
async def test_a_docker_rm_error_still_reports_the_container_as_unremovable():
    """Control for the test above: an SSH error on the `docker rm` itself is a failed removal."""
    name = "pod_11655dc5-53ba-4a8d-a341-fe6c9d12bda7"
    ssh, _ = _make_ssh_mock(containers=[name], ages_by_name={name: 30})
    plain_handler = ssh.run.side_effect

    async def handler(cmd, *args, **kwargs):
        if "docker rm -f" in cmd:
            raise OSError("ssh transport closed")
        return await plain_handler(cmd, *args, **kwargs)

    ssh.run.side_effect = handler

    removed_count, removed_names, unremovable = await ContainerCleanup(stale_threshold_minutes=15).cleanup(
        ssh_client=ssh, rented_data=_listed(), executor_uuid=EXECUTOR_UUID
    )

    assert (removed_count, removed_names, unremovable) == (0, [], [name])


def _removed_volume_names(rm_calls: list[str]) -> list[str]:
    return [c.split("docker volume rm ")[1].split()[0] for c in rm_calls if "docker volume rm" in c]


@pytest.mark.asyncio
async def test_reaper_removes_a_stale_filler_and_only_its_own_volume():
    """A filler create sweeps only filler-labelled volumes, so the reaper must take the `volume_<run id>` of a
    filler it removes, or the preallocated volume holds the disk."""
    filler = "filler_00000000-0000-4000-8000-0000000000a1"
    customer_pod = "pod_00000000-0000-4000-8000-0000000000c1"
    ssh, rm_calls = _make_ssh_mock(containers=[filler, customer_pod], ages_by_name={filler: 30, customer_pod: 30})

    _, removed_names, _ = await ContainerCleanup(stale_threshold_minutes=15).cleanup(
        ssh_client=ssh, rented_data=_rented_data(EXECUTOR_UUID, [customer_pod]), executor_uuid=EXECUTOR_UUID
    )

    assert removed_names == [filler]
    assert _removed_volume_names(rm_calls) == ["volume_00000000-0000-4000-8000-0000000000a1"]


@pytest.mark.parametrize(
    ("stderr", "warned"),
    [
        ("Error response from daemon: volume is in use", True),
        ("", True),
        ("Error response from daemon: get volume_x: no such volume", False),
    ],
    ids=["named_error", "failure_without_output", "volume_already_gone"],
)
@pytest.mark.asyncio
async def test_a_failed_filler_volume_rm_is_logged_unless_the_volume_is_gone(caplog, stderr: str, warned: bool):
    filler = "filler_00000000-0000-4000-8000-0000000000a1"
    ssh, _ = _make_ssh_mock(containers=[filler], ages_by_name={filler: 30})
    plain_handler = ssh.run.side_effect

    async def handler(cmd, *args, **kwargs):
        if "docker volume rm" in cmd:
            # a command ending in `|| true` hides the failure from the caller, as the shell would
            return MagicMock(exit_status=0 if cmd.rstrip().endswith("|| true") else 1, stdout="", stderr=stderr)
        return await plain_handler(cmd, *args, **kwargs)

    ssh.run.side_effect = handler

    with caplog.at_level(logging.WARNING, logger="services.container_cleanup"):
        await ContainerCleanup(stale_threshold_minutes=15).cleanup(
            ssh_client=ssh, rented_data=_listed(), executor_uuid=EXECUTOR_UUID
        )

    logged = any(f"Removed container {filler} but not its volume" in r.getMessage() for r in caplog.records)
    assert logged is warned


async def cleanup_with_hook(ssh, on_before_remove):
    return await ContainerCleanup(stale_threshold_minutes=15).cleanup(
        ssh_client=ssh,
        rented_data=_listed(),
        executor_uuid=EXECUTOR_UUID,
        on_before_remove=on_before_remove,
    )


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

    removed_count, removed_names, _ = await cleanup.cleanup(
        ssh_client=ssh,
        rented_data=_listed(),
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

    removed_count, removed_names, _ = await cleanup.cleanup(
        ssh_client=ssh,
        rented_data=_listed(),
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

    removed_count, removed_names, _ = await cleanup.cleanup(
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

    await cleanup.cleanup(ssh_client=ssh, rented_data=_listed(), executor_uuid=EXECUTOR_UUID)

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

    removed_count, removed_names, _ = await cleanup.cleanup(
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

    removed_count, removed_names, _ = await cleanup.cleanup(
        ssh_client=ssh,
        rented_data=_listed(),
        executor_uuid=EXECUTOR_UUID,
    )

    assert removed_count == 0
    assert removed_names == []
    assert not any("docker rm -f" in c for c in rm_calls)


@pytest.mark.asyncio
async def test_cleanup_removes_stale_unknown_filler_container():
    name = "filler_00000000-0000-4000-8000-0000000000a2"
    ssh, rm_calls = _make_ssh_mock(
        containers=[name],
        ages_by_name={name: 30},
    )
    cleanup = ContainerCleanup(stale_threshold_minutes=15)

    removed_count, removed_names, _ = await cleanup.cleanup(
        ssh_client=ssh,
        rented_data=_listed(),
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
    assert "docker ps --format" in cmd
    assert "name=^health_check_" in cmd
    assert "grep -xE 'health_check_([0-9a-f]{32}|[0-9]{10})'" in cmd
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
async def test_prune_caps_removals_per_pass():
    """A backlog larger than the cap only removes the first VOLUME_RM_MAX_PER_PASS."""
    backlog = [f"{i:064x}" for i in range(VOLUME_RM_MAX_PER_PASS + 50)]
    ssh, rm_calls = _volume_ssh_mock(dangling=backlog)
    cleanup = ContainerCleanup()

    removed = await cleanup.prune_dangling_anonymous_volumes(ssh, EXECUTOR_UUID)

    assert removed == VOLUME_RM_MAX_PER_PASS
    assert len(rm_calls) == 1


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

    removed_count, removed_names, _ = await cleanup.cleanup(
        ssh_client=ssh,
        rented_data=_listed(),
        executor_uuid=EXECUTOR_UUID,
    )

    # Container return contract unchanged; the volume was still pruned.
    assert removed_count == 0
    assert removed_names == []
    assert any("docker volume rm" in c and ANON_VOLUME_A in c for c in rm_calls)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rented_data",
    [pytest.param(None, id="fetch-failed-none"), pytest.param(RentedExecutorsResponse(executors={}), id="empty")],
)
async def test_an_empty_or_unknown_rented_list_still_prunes_dangling_volumes(rented_data):
    ssh, rm_calls = _volume_ssh_mock(dangling=[ANON_VOLUME_A])

    result = await ContainerCleanup(stale_threshold_minutes=15).cleanup(
        ssh_client=ssh, rented_data=rented_data, executor_uuid=EXECUTOR_UUID
    )

    assert result == (0, [], [])
    assert any("docker volume rm" in c and ANON_VOLUME_A in c for c in rm_calls)


@pytest.mark.asyncio
async def test_cleanup_preserves_every_filler_bundle_on_split_node():
    # DAH-2465: both bundle fillers must survive the stale-reaper — protecting only one let the
    # validator kill the sibling mid-cycle.
    bundle_a, bundle_b = "filler_bundle_a", "filler_bundle_b"
    ssh, rm_calls = _make_ssh_mock(
        containers=[bundle_a, bundle_b],
        ages_by_name={bundle_a: 60, bundle_b: 60},
    )
    cleanup = ContainerCleanup(stale_threshold_minutes=15)
    rented_data = _rented_data(EXECUTOR_UUID, [])
    rented_data.all_filler_containers_by_executor = {EXECUTOR_UUID: [bundle_a, bundle_b]}

    removed_count, removed_names, _ = await cleanup.cleanup(
        ssh_client=ssh, rented_data=rented_data, executor_uuid=EXECUTOR_UUID
    )

    assert removed_count == 0
    assert removed_names == []
    assert not any("docker rm -f" in c for c in rm_calls)


# ---------------------------------------------------------------------------
# DAH-2991: an orphan dockerd cannot kill (ticket-0287)
# ---------------------------------------------------------------------------

COULD_NOT_KILL = (
    'Error response from daemon: cannot remove container "/pod_x": could not kill: '
    "tried to kill container, but did not receive an exit event"
)


def _make_unkillable_ssh_mock(name: str, rm_failures: int, current_ts: int = 1_000_000_000):
    """docker rm -fv fails `rm_failures` times with dockerd's could-not-kill error, then succeeds."""
    calls: list[str] = []
    state = {"rm_left": rm_failures}

    async def handler(cmd, *args, **kwargs):
        calls.append(cmd)
        if "docker ps -a" in cmd:
            return MagicMock(exit_status=0, stdout=_ps_line(name, _label_of(name)), stderr="")
        if cmd.strip() == "date +%s":
            return MagicMock(exit_status=0, stdout=str(current_ts), stderr="")
        if "docker inspect" in cmd and "Created" in cmd:
            return MagicMock(exit_status=0, stdout=str(current_ts - 120 * 60), stderr="")
        if "docker rm -f" in cmd:
            if state["rm_left"] > 0:
                state["rm_left"] -= 1
                return MagicMock(exit_status=1, stdout="", stderr=COULD_NOT_KILL)
            return MagicMock(exit_status=0, stdout="", stderr="")
        if ".State.Pid" in cmd:
            return MagicMock(exit_status=0, stdout="killed pid=4242 shim=4141", stderr="")
        return MagicMock(exit_status=0, stdout="", stderr="")

    return _ssh_mock_from_calls(handler), calls


@pytest.mark.asyncio
async def test_cleanup_kills_processes_directly_when_docker_rm_cannot_kill():
    """rm -f fails with 'did not receive an exit event' -> kill init+shim over ssh -> rm -f again succeeds."""
    name = "pod_11655dc5-53ba-4a8d-a341-fe6c9d12bda7"
    ssh, calls = _make_unkillable_ssh_mock(name, rm_failures=1)

    removed_count, removed_names, unremovable = await ContainerCleanup(stale_threshold_minutes=15).cleanup(
        ssh_client=ssh, rented_data=_listed(), executor_uuid=EXECUTOR_UUID
    )

    assert removed_names == [name] and unremovable == []
    rm_idx = [i for i, c in enumerate(calls) if "docker rm -f" in c and name in c]
    kill_idx = [i for i, c in enumerate(calls) if ".State.Pid" in c and "kill -9" in c and name in c]
    assert len(rm_idx) == 2 and len(kill_idx) == 1
    assert rm_idx[0] < kill_idx[0] < rm_idx[1]


@pytest.mark.asyncio
async def test_cleanup_reports_container_that_survives_the_direct_kill():
    """Still not removable after the escalation -> reported as unremovable, not silently dropped."""
    name = "pod_11655dc5-53ba-4a8d-a341-fe6c9d12bda7"
    ssh, calls = _make_unkillable_ssh_mock(name, rm_failures=2)

    removed_count, removed_names, unremovable = await ContainerCleanup(stale_threshold_minutes=15).cleanup(
        ssh_client=ssh, rented_data=_listed(), executor_uuid=EXECUTOR_UUID
    )

    assert removed_count == 0 and removed_names == []
    assert unremovable == [name]
    assert sum(1 for c in calls if "docker rm -f" in c and name in c) == 2  # one retry, no loop


# ---------------------------------------------------------------------------
# Only an empty or missing fleet snapshot means "don't know"; any listed name is protected on every host
# ---------------------------------------------------------------------------

LIVE_POD = "pod_00000000-0000-4000-8000-000000000001"
LIVE_FILLER = "filler_00000000-0000-4000-8000-0000000000f1"
STRAY_POD = "pod_00000000-0000-4000-8000-000000000002"
ELSEWHERE_POD = "pod_00000000-0000-4000-8000-000000000003"


def _real_rented_data(executor_uuid: str, pods: list[str], fillers: list[str] | None = None, key: str | None = None):
    return RentedExecutorsResponse(
        executors={
            (key or executor_uuid): RentedExecutor(
                miner_hotkey="miner-hotkey",
                executor_ip_address="127.0.0.1",
                executor_ip_port="8080",
                pods=[RentedPod(pod_id=name.removeprefix("pod_"), container_name=name) for name in pods],
            )
        },
        all_filler_containers_by_executor={executor_uuid: fillers} if fillers else {},
    )


def _host_with_live_and_stray():
    return _make_ssh_mock(
        containers=[LIVE_POD, LIVE_FILLER, STRAY_POD],
        ages_by_name={LIVE_POD: 28, LIVE_FILLER: 29, STRAY_POD: 60},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rented_data, reason",
    [
        pytest.param(None, RENTED_LIST_UNAVAILABLE, id="fetch-failed-none"),
        pytest.param(RentedExecutorsResponse(executors={}), RENTED_LIST_EMPTY, id="fleet-snapshot-empty"),
        pytest.param(_real_rented_data(EXECUTOR_UUID, []), RENTED_LIST_EMPTY, id="only-executor-listed-with-no-pods"),
    ],
)
async def test_an_empty_or_unknown_fleet_snapshot_removes_nothing_and_warns(rented_data, reason, caplog):
    ssh, rm_calls = _host_with_live_and_stray()

    with caplog.at_level(logging.WARNING, logger="services.container_cleanup"):
        result = await ContainerCleanup(stale_threshold_minutes=15).cleanup(
            ssh_client=ssh, rented_data=rented_data, executor_uuid=EXECUTOR_UUID
        )

    assert result == (0, [], [])
    assert not any("docker rm -f" in c for c in rm_calls)
    assert not any("docker ps -a" in c for c in (call.args[0] for call in ssh.run.call_args_list))
    assert rented_list_unknown_reason(rented_data) == reason
    assert any("snapshot is empty or unknown" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rented_data, expected_removed",
    [
        pytest.param(_real_rented_data(EXECUTOR_UUID, [LIVE_POD], fillers=[LIVE_FILLER]), {STRAY_POD}, id="own-pod-and-filler"),
        pytest.param(_real_rented_data(EXECUTOR_UUID, [], fillers=[LIVE_FILLER]), {LIVE_POD, STRAY_POD}, id="own-filler-only"),
        pytest.param(
            _real_rented_data(EXECUTOR_UUID, [LIVE_POD], fillers=[LIVE_FILLER], key="twin-executor-id"),
            {STRAY_POD},
            id="pod-listed-under-twin-id-filler-under-this-id",
        ),
        pytest.param(
            _real_rented_data("another-node", [ELSEWHERE_POD]),
            {LIVE_POD, LIVE_FILLER, STRAY_POD},
            id="idle-node-orphans-still-reaped",
        ),
    ],
)
async def test_a_non_empty_snapshot_removes_every_unlisted_container_and_no_listed_one(rented_data, expected_removed):
    ssh, rm_calls = _host_with_live_and_stray()

    removed_count, removed_names, unremovable = await ContainerCleanup(stale_threshold_minutes=15).cleanup(
        ssh_client=ssh, rented_data=rented_data, executor_uuid=EXECUTOR_UUID
    )

    assert rented_list_unknown_reason(rented_data) is None
    assert (removed_count, set(removed_names), unremovable) == (len(expected_removed), expected_removed, [])


@pytest.mark.asyncio
async def test_cleanup_logs_the_rented_count_not_the_fleet_set(caplog):
    fleet_pods = [f"pod_00000000-0000-4000-8000-{i:012d}" for i in range(500)]
    stale = [f"pod_00000000-0000-4000-9000-{i:012d}" for i in range(5)]
    ssh, _ = _make_ssh_mock(containers=stale, ages_by_name={name: 60 for name in stale})
    rented_data = _real_rented_data("another-node", fleet_pods)

    with caplog.at_level(logging.INFO, logger="services.container_cleanup"):
        removed_count, _, _ = await ContainerCleanup(stale_threshold_minutes=15).cleanup(
            ssh_client=ssh, rented_data=rented_data, executor_uuid=EXECUTOR_UUID
        )

    cleanup_logs = [r.msg for r in caplog.records if hasattr(r.msg, "to_full_string")]
    assert removed_count == len(stale) and len(cleanup_logs) == len(stale) + 1
    for msg in cleanup_logs:
        assert msg.extra["rented_container_count"] == len(listed_container_names(rented_data))
        assert "4000-8000" not in msg.to_full_string()


@pytest.mark.parametrize("rented_data", [None, _listed(), _rented_data("twin-executor-id", ["pod_on_twin"])])
def test_get_rented_containers_answers_the_dind_volume_sweep_like_listed_container_names(rented_data):
    # #1448's prune_orphaned_dind_volumes calls it with this signature
    names = ContainerCleanup()._get_rented_containers(rented_data, EXECUTOR_UUID)

    assert names == listed_container_names(rented_data)


@pytest.mark.parametrize("validator_dry_run", [False, True])
def test_pipeline_factory_cleanup_dry_run_follows_validator_dry_run(monkeypatch, validator_dry_run):
    """DAH-4330: no separate cleanup toggle; cleanup only reports under the validator's DRY_RUN."""
    from core.config import settings
    from services.task import pipeline_factory as pipeline_factory_module

    monkeypatch.setattr(pipeline_factory_module, "InspectorValidationService", MagicMock)
    monkeypatch.setattr(settings, "DRY_RUN", validator_dry_run)
    factory = pipeline_factory_module.PipelineFactory(*(MagicMock() for _ in range(7)))

    assert factory.container_cleanup.dry_run is validator_dry_run
    assert not hasattr(settings, "CONTAINER_CLEANUP_DRY_RUN")


def test_listed_container_names_protects_the_pod_container_of_an_unnamed_pod():
    # DAH-4315: the backend lists a new pod before it records the container name
    names = listed_container_names(_rented_data(EXECUTOR_UUID, [""]))

    assert "pod_pod-0" in names and "pod_pod-0__prev" in names
    assert "" not in names and "__prev" not in names


PROVIDER_NAMES = [
    "podman",
    "mypod_x",
    "pod_customer",
    "filler_1",
    "container_dindtest",
    "health_check_customer",
    "health_check_",
    "my_pod_00000000-0000-4000-8000-000000000001",
]
# a name Lium gives, on a container without the label, and no backend record to confirm it by
LEGACY_UNPROVABLE_NAMES = ["container_5F7X5UpKSr26KU3jKfpLmT8kuKtBNyHhEnfS8xtxPCqCb13p_40016"]
LEGACY_POD = "pod_00000000-0000-4000-8000-0000000000e1"
LEGACY_FILLER = "filler_00000000-0000-4000-8000-0000000000e2"


@pytest.mark.asyncio
async def test_an_unlabelled_container_is_never_removed_whatever_its_name():
    """DAH-5117: prod removed a provider's "podman" (25 Oct); only Lium's label authorises it."""
    legacy = [LEGACY_POD, f"{LEGACY_POD}__prev", LEGACY_FILLER]
    names = PROVIDER_NAMES + LEGACY_UNPROVABLE_NAMES + legacy
    ssh, rm_calls = _make_ssh_mock(names, {name: 60 for name in names}, unlabelled=tuple(names))
    ended = AsyncMock(return_value=False)

    result = await ContainerCleanup().cleanup(
        ssh, _listed(), EXECUTOR_UUID, lium_workload_ended=ended
    )

    assert result == (0, [], [])
    assert not any("docker rm -f" in c for c in rm_calls)
    # only Lium's exact pod/filler names are worth asking the backend about
    assert sorted(call.args[0] for call in ended.await_args_list) == sorted(legacy)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name, removed",
    [
        ("health_check_0123456789abcdef0123456789abcdef", True),  # health_check_{uuid4().hex}
        ("health_check_1777040800", True),  # the backend's older int(time.time()) suffix
        ("health_check_customer", False),
        ("health_check_", False),
        ("health_check_0123456789ABCDEF0123456789ABCDEF", False),
        ("health_check_17770408001", False),
    ],
)
async def test_an_unlabelled_health_check_is_removed_only_under_the_backends_exact_name(
    name, removed
):
    ssh, commands = _make_ssh_mock([name], {name: 30})

    result = await ContainerCleanup().cleanup(ssh, _listed(), EXECUTOR_UUID)

    assert result == ((1, [name], []) if removed else (0, [], []))
    assert any("docker rm -f" in c for c in commands) is removed


@pytest.mark.asyncio
@pytest.mark.parametrize("name", [LEGACY_POD, f"{LEGACY_POD}__prev", LEGACY_FILLER])
@pytest.mark.parametrize("ended", [None, False, True])
async def test_a_legacy_container_is_removed_only_when_the_backends_record_shows_it_ended(
    name, ended
):
    ssh, rm_calls = _make_ssh_mock([name], {name: 60}, unlabelled=(name,))
    callback = None if ended is None else AsyncMock(return_value=ended)

    result = await ContainerCleanup().cleanup(
        ssh, _listed(), EXECUTOR_UUID, lium_workload_ended=callback
    )

    assert result == ((1, [name], []) if ended else (0, [], []))
    if callback is not None:
        callback.assert_awaited_once_with(name)


@pytest.mark.asyncio
async def test_a_labelled_container_whose_pod_is_live_stays_under_any_name():
    """The label names the pod: a live pod's container stays under a name the snapshot does not list."""
    renamed = "pod_00000000-0000-4000-8000-0000000000d1"
    ssh, rm_calls = _make_ssh_mock([renamed], {renamed: 60})
    rented_data = _real_rented_data(EXECUTOR_UUID, [])
    rented_data.executors[EXECUTOR_UUID].pods = [
        RentedPod(pod_id=renamed.removeprefix("pod_"), container_name="pod_named_otherwise")
    ]

    result = await ContainerCleanup().cleanup(ssh, rented_data, EXECUTOR_UUID)

    assert result == (0, [], [])
    assert not any("docker rm -f" in c for c in rm_calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("name", [STRAY_POD, f"{STRAY_POD}__prev", LEGACY_FILLER])
async def test_an_inherited_label_naming_another_live_pod_does_not_keep_an_ended_container(name):
    """A renter's image can carry io.lium.pod_id=<another live pod>; only a label bound to the name counts."""
    ssh, rm_calls = _make_ssh_mock([name], {name: 60})
    handler = ssh.run.side_effect

    async def inherited_label(cmd, *args, **kwargs):
        if "docker ps -a" in cmd:
            return MagicMock(exit_status=0, stdout=_ps_line(name, LIVE_POD.removeprefix("pod_")), stderr="")
        return await handler(cmd, *args, **kwargs)

    ssh.run.side_effect = inherited_label
    ended = AsyncMock(return_value=True)

    result = await ContainerCleanup().cleanup(
        ssh, _real_rented_data(EXECUTOR_UUID, [LIVE_POD]), EXECUTOR_UUID, lium_workload_ended=ended
    )

    assert result == (1, [name], [])
    ended.assert_awaited_once_with(name)  # treated as unlabelled: the legacy rule decides


@pytest.mark.parametrize(
    "name, label, bound",
    [
        ("pod_a", "a", True),
        ("pod_a__prev", "a", True),
        ("filler_r", "r", True),
        ("pod_a", "b", False),
        ("pod_a", "pod_a", False),
        ("container_x_1", "container_x_1", True),
        ("container_x_1", "x_1", False),
        ("health_check_1777040800", "other", False),
        ("pod_a", "", False),
    ],
)
def test_a_pod_id_label_counts_only_when_bound_to_the_name(name, label, bound):
    from services.container_cleanup import _bound_pod_id_label

    assert _bound_pod_id_label(name, label) == (label if bound else "")


@pytest.mark.asyncio
async def test_the_listing_reads_each_containers_pod_id_label():
    seen: list[str] = []

    async def handler(cmd, *args, **kwargs):
        seen.append(cmd)
        stdout = f'{_ps_line("pod_a", "a")}\n{_ps_line("podman", "")}\n'
        return MagicMock(exit_status=0, stdout=stdout, stderr="")

    ssh = AsyncMock()
    ssh.run = AsyncMock(side_effect=handler)

    listed = await ContainerCleanup()._get_all_rental_containers(ssh)

    assert listed == [("pod_a", "a"), ("podman", "")]
    assert """--format '[{{json .Names}},{{json (.Label "io.lium.pod_id")}}]'""" in seen[0]


@pytest.mark.asyncio
async def test_a_multiline_pod_id_label_cannot_forge_a_second_container_record():
    forged = "ignored\nprovider_db|provider_db\n[\"provider_db\",\"provider_db\"]"

    async def handler(cmd, *args, **kwargs):
        return MagicMock(exit_status=0, stdout=_ps_line("pod_a", forged) + "\n", stderr="")

    ssh = AsyncMock()
    ssh.run = AsyncMock(side_effect=handler)

    listed = await ContainerCleanup()._get_all_rental_containers(ssh)

    assert listed == [("pod_a", forged)]  # one container, one record; the label stays inside it


@pytest.mark.asyncio
async def test_an_unparseable_listing_line_is_skipped():
    async def handler(cmd, *args, **kwargs):
        stdout = 'provider_db|provider_db\n["x"]\n' + _ps_line("pod_a", "a")
        return MagicMock(exit_status=0, stdout=stdout, stderr="")

    ssh = AsyncMock()
    ssh.run = AsyncMock(side_effect=handler)

    assert await ContainerCleanup()._get_all_rental_containers(ssh) == [("pod_a", "a")]


def _cleanup_check_ctx(removed: list[str], gpu_processes: list[dict], backend=None):
    from tests.helpers import build_services, build_state, make_context

    cleanup = AsyncMock()
    cleanup.cleanup.return_value = (len(removed), removed, [])
    cleanup.sweep_abandoned_download_temporaries.return_value = 0
    cleanup.reclaim_dphn_cache_when_disk_is_tight.return_value = 0
    redis = AsyncMock()
    redis.sadd.return_value = 0  # seen before: no first-sight grace
    redis.hgetall.return_value = {}
    overrides = {"backend": backend} if backend else {}
    return make_context(
        services=build_services(container_cleanup=cleanup, redis=redis, **overrides),
        state=build_state(
            rented_data=_real_rented_data("another-node", [LIVE_POD]), gpu_processes=gpu_processes
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "removed, owner, dropped",
    [
        ("filler_00000000-0000-4000-8000-0000000000a1", "THIS", True),
        ("pod_00000000-0000-4000-8000-0000000000a1", "THIS", True),
        # review 5644590326: removed by label alone, no backend record of it ending here
        ("filler_00000000-0000-4000-8000-0000000000a1", None, False),
        ("pod_00000000-0000-4000-8000-0000000000a1", "another-node", False),
        ("health_check_1777040800", None, False),
    ],
)
async def test_the_check_drops_the_gpu_processes_only_of_backend_confirmed_removals(
    removed, owner, dropped
):
    """The processes were read before the cleanup; a reaped container's count in no later check,
    but only when the backend confirms it was Lium's: a forged label recreated each cycle must not
    exempt its GPU load from the gate every cycle."""
    from protocol.vc_protocol.compute_requests import (
        FillerRunActiveResponse,
        PodRentalActiveResponse,
    )
    from services.task.checks.stale_container_cleanup import StaleContainerCleanupCheck

    backend = AsyncMock()
    reaped = {"pid": 1, "container_name": removed}
    foreign = {"pid": 2, "container_name": "my_miner"}
    ctx = _cleanup_check_ctx([removed], [reaped, foreign], backend=backend)
    executor_id = str(ctx.executor.uuid) if owner == "THIS" else owner
    backend.get_filler_run_active.return_value = FillerRunActiveResponse(
        active=False, status="STOPPED", executor_id=executor_id
    )
    backend.get_pod_rental_active.return_value = PodRentalActiveResponse(
        active=False, rental_closed_at=CLOSED, executor_id=executor_id
    )

    result = await StaleContainerCleanupCheck().run(ctx)

    assert result.passed is True
    assert result.updates["state"].gpu_processes == ([foreign] if dropped else [reaped, foreign])


def _ended_here_backend(ctx_holder: dict):
    """A backend whose records show every pod and filler ended on the context's node."""
    from protocol.vc_protocol.compute_requests import (
        FillerRunActiveResponse,
        PodRentalActiveResponse,
    )

    backend = AsyncMock()

    async def pod(_id):
        return PodRentalActiveResponse(
            active=False, rental_closed_at=CLOSED, executor_id=str(ctx_holder["ctx"].executor.uuid)
        )

    async def filler(_id):
        return FillerRunActiveResponse(
            active=False, status="STOPPED", executor_id=str(ctx_holder["ctx"].executor.uuid)
        )

    backend.get_pod_rental_active.side_effect = pod
    backend.get_filler_run_active.side_effect = filler
    return backend


@pytest.mark.asyncio
async def test_a_confirmed_container_that_survives_docker_rm_keeps_its_gpu_processes():
    """Review 4371963872: confirmed STOPPED and owned here, but docker rm failed: still running."""
    from services.task.checks.stale_container_cleanup import StaleContainerCleanupCheck

    name = "filler_00000000-0000-4000-8000-0000000000a1"
    processes = [{"pid": 1, "container_name": name}]
    holder: dict = {}
    ctx = holder["ctx"] = _cleanup_check_ctx([], processes, backend=_ended_here_backend(holder))

    async def cleanup(**kwargs):
        assert await kwargs["lium_workload_ended"](name) is True
        return 0, [], [name]  # the rm failed

    ctx.services.container_cleanup.cleanup.side_effect = cleanup

    result = await StaleContainerCleanupCheck().run(ctx)

    assert result.updates["state"].gpu_processes == processes
    assert result.updates["state"].orphaned_containers == [name]


@pytest.mark.asyncio
async def test_backend_lookups_are_capped_per_node_per_cycle(caplog):
    """Review 4371963877: thousands of unlabelled pod_/filler_ names must not each cost a backend
    request; the excess is left alone (not removed, its GPU load still counts) and the cap is logged."""
    from services.const import LIUM_WORKLOAD_LOOKUPS_PER_CYCLE as CAP
    from services.task.checks.stale_container_cleanup import StaleContainerCleanupCheck

    names = [f"pod_00000000-0000-4000-8000-{i:012x}" for i in range(CAP + 5)]
    processes = [{"pid": i, "container_name": name} for i, name in enumerate(names)]
    holder: dict = {}
    backend = _ended_here_backend(holder)
    ctx = holder["ctx"] = _cleanup_check_ctx([], processes, backend=backend)

    async def cleanup(**kwargs):
        removed = [name for name in names if await kwargs["lium_workload_ended"](name)]
        return len(removed), removed, []

    ctx.services.container_cleanup.cleanup.side_effect = cleanup

    with caplog.at_level(logging.WARNING):
        result = await StaleContainerCleanupCheck().run(ctx)

    assert backend.get_pod_rental_active.await_count == CAP
    assert result.updates["state"].gpu_processes == processes[CAP:]
    assert "Backend lookup cap reached" in caplog.text


@pytest.mark.asyncio
async def test_the_lookup_order_rotates_so_forged_names_cannot_starve_a_real_leftover():
    """With more unlabelled candidates than the lookup cap, each cycle asks about a fresh random
    order, so a real leftover behind 10+ forged names is checked within a few cycles."""
    from services.const import LIUM_WORKLOAD_LOOKUPS_PER_CYCLE as CAP

    forged = [f"pod_00000000-0000-4000-8000-{i:012x}" for i in range(1, CAP + 6)]
    real = "filler_00000000-0000-4000-8000-0000000000ff"
    names = [*forged, real]  # docker lists the real one last
    ssh, _ = _make_ssh_mock(names, {name: 60 for name in names}, unlabelled=tuple(names))

    async def cycle(seed: int) -> list[str]:
        asked: list[str] = []

        async def ended(name: str) -> bool:
            if len(asked) >= CAP:
                return False  # the check's cap: no lookup made
            asked.append(name)
            return False

        random.seed(seed)
        await ContainerCleanup().cleanup(ssh, _listed(), EXECUTOR_UUID, lium_workload_ended=ended)
        return asked

    first, second = await cycle(1), await cycle(2)
    assert len(first) == len(second) == CAP
    assert first != second
    assert any([real in await cycle(seed) for seed in range(20)])


@pytest.mark.asyncio
async def test_a_failed_backend_read_keeps_the_removed_containers_gpu_processes():
    from services.task.checks.stale_container_cleanup import StaleContainerCleanupCheck

    backend = AsyncMock()
    backend.get_pod_rental_active.side_effect = RuntimeError("backend down")
    removed = "pod_00000000-0000-4000-8000-0000000000a1"
    processes = [{"pid": 1, "container_name": removed}]
    ctx = _cleanup_check_ctx([removed], processes, backend=backend)

    result = await StaleContainerCleanupCheck().run(ctx)

    assert result.updates["state"].gpu_processes == processes


CLOSED = "2026-10-25T00:00:00Z"
ENDED = {"active": False, "rental_closed_at": CLOSED}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name, pod_answer, filler_answer, ended",
    [
        (LEGACY_POD, {**ENDED, "executor_id": "THIS"}, None, True),  # owner matches
        (f"{LEGACY_POD}__prev", {**ENDED, "executor_id": "THIS"}, None, True),
        (LEGACY_POD, {**ENDED, "executor_id": "another-node"}, None, False),  # owner mismatches
        (LEGACY_POD, ENDED, None, False),  # owner missing: never matches
        # no close time: no proof Lium's rental ended
        (LEGACY_POD, {"active": False, "executor_id": "THIS"}, None, False),
        (LEGACY_POD, {"active": True}, None, False),
        (LEGACY_POD, None, None, False),  # no record (404) or the backend did not answer
        (LEGACY_FILLER, None, {"active": False, "status": "STOPPED", "executor_id": "THIS"}, True),
        (LEGACY_FILLER, None, {"active": False, "status": "FAILED", "executor_id": "x"}, False),
        (LEGACY_FILLER, None, {"active": False, "status": "STOPPED"}, False),  # owner missing
        (LEGACY_FILLER, None, {"active": False, "status": "STARTING"}, False),
        (LEGACY_FILLER, None, {"active": False}, False),  # no status: proves nothing
        (LEGACY_FILLER, None, None, False),
    ],
)
async def test_the_check_confirms_a_legacy_container_ended_from_the_backends_record(
    name, pod_answer, filler_answer, ended
):
    from protocol.vc_protocol.compute_requests import (
        FillerRunActiveResponse,
        PodRentalActiveResponse,
    )
    from services.task.checks.stale_container_cleanup import _lium_workload_ended_here

    backend = AsyncMock()
    ctx = _cleanup_check_ctx([], [], backend=backend)

    def answer(fields, model):
        if fields is None:
            return None
        if fields.get("executor_id") == "THIS":
            fields = {**fields, "executor_id": str(ctx.executor.uuid)}
        return model(**fields)

    backend.get_pod_rental_active.return_value = answer(pod_answer, PodRentalActiveResponse)
    backend.get_filler_run_active.return_value = answer(filler_answer, FillerRunActiveResponse)

    assert await _lium_workload_ended_here(ctx, name) is ended
    record_id = name.removesuffix("__prev").partition("_")[2]
    if name.startswith("filler_"):
        backend.get_filler_run_active.assert_awaited_once_with(record_id)
    else:
        backend.get_pod_rental_active.assert_awaited_once_with(record_id)
