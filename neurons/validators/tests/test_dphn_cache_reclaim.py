"""DAH-2475: the periodic backstop that gives the DPHN filler cache back under disk pressure.

The create-time sweep only removes STALE cache versions and never reclaims — a node's free disk is
low precisely BECAUSE the cache is downloaded, so reclaiming there would make the backend grant it
again next cycle and the node would re-download ~37 GB forever. This backstop is the one place that
reclaims, and it decides on real free space: below the rental listing floor the node earns nothing.
"""

from types import SimpleNamespace

import pytest

from services.container_cleanup import DPHN_CACHE_RECLAIM_FREE_FLOOR_GB, ContainerCleanup

GB = 1024**3


class FakeSshClient:
    def __init__(self, containers: list[str], volumes: list[str], free_gb: float):
        self._containers = "\n".join(containers)
        self._volumes = "\n".join(volumes)
        self._free_bytes = str(int(free_gb * GB))
        self.commands: list[str] = []

    async def run(self, command: str, *args, **kwargs):
        self.commands.append(command)
        if "df -B1" in command:
            stdout = self._free_bytes
        elif "volume ls" in command:
            stdout = self._volumes
        elif "docker ps" in command:
            # Only `docker ps` without -a lists RUNNING containers; the reclaim guard uses that one.
            stdout = "" if "-a" in command else self._containers
        else:
            stdout = ""
        return SimpleNamespace(exit_status=0, stdout=stdout, stderr="")


def _removed(ssh_client: FakeSshClient) -> list[str]:
    removals = [c for c in ssh_client.commands if "volume rm" in c]
    if not removals:
        return []
    return sorted(removals[0].split("volume rm ", 1)[1].split(" 2>/dev/null")[0].split())


@pytest.fixture
def cleanup() -> ContainerCleanup:
    return ContainerCleanup()


@pytest.mark.asyncio
async def test_reclaims_the_cache_when_free_disk_is_below_the_listing_floor(cleanup):
    ssh_client = FakeSshClient(
        containers=[], volumes=["dphn_cache_hf_x", "dphn_cache_dp_x", "volume_pod"], free_gb=40
    )

    removed = await cleanup.reclaim_dphn_cache_when_disk_is_tight(ssh_client, "exec-1")

    assert removed == 2
    assert _removed(ssh_client) == ["dphn_cache_dp_x", "dphn_cache_hf_x"]


@pytest.mark.asyncio
async def test_keeps_the_cache_when_the_node_is_comfortably_above_the_floor(cleanup):
    ssh_client = FakeSshClient(
        containers=[], volumes=["dphn_cache_hf_x"], free_gb=DPHN_CACHE_RECLAIM_FREE_FLOOR_GB + 500
    )

    assert await cleanup.reclaim_dphn_cache_when_disk_is_tight(ssh_client, "exec-1") == 0
    assert _removed(ssh_client) == []


@pytest.mark.asyncio
async def test_never_reclaims_while_a_filler_container_is_live(cleanup):
    # A cache a sibling still mounts is in use, and a node still running fillers is not the stranded
    # node this backstop exists to rescue.
    ssh_client = FakeSshClient(containers=["filler_abc"], volumes=["dphn_cache_hf_x"], free_gb=10)

    assert await cleanup.reclaim_dphn_cache_when_disk_is_tight(ssh_client, "exec-1") == 0
    assert _removed(ssh_client) == []


@pytest.mark.asyncio
async def test_no_cache_on_the_host_is_a_no_op(cleanup):
    ssh_client = FakeSshClient(containers=[], volumes=["volume_pod"], free_gb=10)

    assert await cleanup.reclaim_dphn_cache_when_disk_is_tight(ssh_client, "exec-1") == 0
    assert _removed(ssh_client) == []


@pytest.mark.asyncio
async def test_dry_run_reports_without_removing(cleanup):
    dry = ContainerCleanup(dry_run=True)
    ssh_client = FakeSshClient(containers=[], volumes=["dphn_cache_hf_x"], free_gb=10)

    assert await dry.reclaim_dphn_cache_when_disk_is_tight(ssh_client, "exec-1") == 0
    assert _removed(ssh_client) == []


@pytest.mark.asyncio
async def test_a_stopped_filler_does_not_block_the_reclaim(cleanup):
    # `docker ps -a` would list a long-exited filler_* forever and wedge the guard; the check must
    # look at RUNNING containers only.
    ssh_client = FakeSshClient(containers=[], volumes=["dphn_cache_hf_x"], free_gb=10)

    assert await cleanup.reclaim_dphn_cache_when_disk_is_tight(ssh_client, "exec-1") == 1
    assert _removed(ssh_client) == ["dphn_cache_hf_x"]
    assert not any("ps -a" in c for c in ssh_client.commands), "guard must not use `docker ps -a`"
