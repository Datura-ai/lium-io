"""DAH-2475: the periodic backstop that gives the DPHN filler cache back under disk pressure.

The create-time sweep only removes STALE cache versions and never reclaims — a node's free disk is
low precisely BECAUSE the cache is downloaded, so reclaiming there would make the backend grant it
again next cycle and the node would re-download ~37 GB forever. This backstop is the one place that
reclaims, and it decides on real free space: below the rental listing floor the node earns nothing.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from services.container_cleanup import ContainerCleanup, never_started_check_containers

GB = 1024**3


class FakeSshClient:
    """`docker volume rm` ends in `|| true`, so exit status says nothing about what was removed —
    the fake must model the host's actual volume list, not just echo success."""

    def __init__(
        self,
        containers: list[str],
        volumes: list[str],
        free_gb: float,
        refuses: set[str] | None = None,
        docker_root: str = "/var/lib/docker",
    ):
        self._containers = "\n".join(containers)
        self._volumes = list(volumes)
        self._refuses = refuses or set()
        self._free_bytes = str(int(free_gb * GB))
        self.docker_root = docker_root
        self.commands: list[str] = []

    async def run(self, command: str, *args, **kwargs):
        self.commands.append(command)
        if "docker info" in command:
            stdout = self.docker_root
        elif "df -P -B1" in command:
            # Only answer for the real data root: a node with docker on a dedicated disk would
            # otherwise be judged by the root filesystem, which says nothing about docker's space.
            avail = self._free_bytes if self.docker_root.strip() in command else "999999999999"
            stdout = f"Filesystem 1B-blocks Used Available Capacity\n/dev/sda1 1 1 {avail} 1% /hostfs"
        elif "volume rm" in command:
            # dockerd refuses a volume referenced by any container, in any state.
            for name in _removal_targets(command):
                if name not in self._refuses and name in self._volumes:
                    self._volumes.remove(name)
            stdout = ""
        elif "volume ls" in command:
            stdout = "\n".join(self._volumes)
        elif "docker ps" in command:
            # Only `docker ps` without -a lists RUNNING containers; the reclaim guard uses that one.
            stdout = "" if "-a" in command else self._containers
        else:
            stdout = ""
        return SimpleNamespace(exit_status=0, stdout=stdout, stderr="")


def _removal_targets(command: str) -> list[str]:
    return command.split("volume rm ", 1)[1].split(" 2>/dev/null")[0].split()


def _removed(ssh_client: FakeSshClient) -> list[str]:
    return sorted(name for c in ssh_client.commands if "volume rm" in c for name in _removal_targets(c))


@pytest.fixture
def cleanup() -> ContainerCleanup:
    return ContainerCleanup()


# Orphan named volumes: volume_<uuid> whose container is gone, judged by the backend (is_orphan).
POD = "11111111-2222-3333-4444-555555555555"
LIVE = "66666666-7777-8888-9999-000000000000"
NOW = 1_800_000_000


class VolumeHostSsh(FakeSshClient):
    def __init__(self, dangling: list[str], created: dict[str, str], **kwargs):
        super().__init__(containers=[], free_gb=500, **kwargs)
        self._dangling = dangling
        self._created = created

    async def run(self, command: str, *args, **kwargs):
        if "dangling=true" in command:
            self.commands.append(command)
            return SimpleNamespace(exit_status=0, stdout="\n".join(self._dangling), stderr="")
        if "volume inspect" in command:
            self.commands.append(command)
            lines = [str(NOW)] + [f"{name} {at}" for name, at in self._created.items() if name in command]
            return SimpleNamespace(exit_status=0, stdout="\n".join(lines), stderr="")
        return await super().run(command, *args, **kwargs)


def _snapshot(pod_ids: list[str]):
    from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod

    pods = [RentedPod(pod_id=pod_id, container_name=f"pod_{pod_id}") for pod_id in pod_ids]
    executor = RentedExecutor(miner_hotkey="m", executor_ip_address="1.2.3.4", executor_ip_port="1", pods=pods)
    return RentedExecutorsResponse(executors={"exec-1": executor})


@pytest.mark.asyncio
async def test_removes_only_unlisted_unmounted_volume_uuids_the_backend_calls_orphans(cleanup):
    volumes = [f"volume_{POD}", f"volume_{LIVE}", "volume_custom", "dphn_cache_x"]
    created = {name: "2026-09-10T08:00:00Z" for name in volumes}
    ssh_client = VolumeHostSsh(dangling=volumes, created=created, volumes=volumes)
    seen = []

    async def is_orphan(volume_id, age):
        seen.append((volume_id, age.days))
        return True

    assert await cleanup.remove_orphan_rental_volumes(ssh_client, _snapshot([LIVE]), "exec-1", is_orphan) == 1
    assert _removed(ssh_client) == [f"volume_{POD}"]
    assert [volume_id for volume_id, _ in seen] == [POD] and seen[0][1] > 7  # age from CreatedAt


@pytest.mark.asyncio
@pytest.mark.parametrize("dangling, listed", [([], []), ([f"volume_{POD}"], [POD])])
async def test_a_mounted_volume_or_a_listed_pod_is_never_a_candidate(cleanup, dangling, listed):
    ssh_client = VolumeHostSsh(dangling=dangling, created={f"volume_{POD}": "2026-09-10T08:00:00Z"}, volumes=[f"volume_{POD}"])
    is_orphan = AsyncMock(return_value=True)
    assert await cleanup.remove_orphan_rental_volumes(ssh_client, _snapshot([LIVE, *listed]), "exec-1", is_orphan) == 0
    is_orphan.assert_not_called()
    assert _removed(ssh_client) == []


@pytest.mark.asyncio
async def test_an_empty_snapshot_or_a_kept_verdict_removes_no_volume(cleanup):
    created = {f"volume_{POD}": "2026-09-10T08:00:00Z"}
    ssh_client = VolumeHostSsh(dangling=[f"volume_{POD}"], created=created, volumes=[f"volume_{POD}"])
    assert await cleanup.remove_orphan_rental_volumes(ssh_client, None, "exec-1", AsyncMock(return_value=True)) == 0
    assert await cleanup.remove_orphan_rental_volumes(ssh_client, _snapshot([LIVE]), "exec-1", AsyncMock(return_value=False)) == 0
    assert _removed(ssh_client) == []


@pytest.mark.asyncio
async def test_kept_candidates_never_starve_the_volumes_behind_the_per_pass_cap(cleanup):
    ids = [str(UUID(int=i)) for i in range(1, 52)]
    names = [f"volume_{uid}" for uid in ids]
    ssh_client = VolumeHostSsh(dangling=names, created={name: "2026-09-10T08:00:00Z" for name in names}, volumes=names)
    seen = []

    async def is_orphan(volume_id, age):
        seen.append(volume_id)
        return volume_id == ids[-1]  # the first 50 are kept (e.g. failed pods), only the 51st is over

    passes = [await cleanup.remove_orphan_rental_volumes(ssh_client, _snapshot([LIVE]), "exec-1", is_orphan) for _ in range(2)]
    assert passes == [0, 1]
    assert seen[:50] == ids[:50] and ids[-1] in seen[50:]
    assert _removed(ssh_client) == [names[-1]]


@pytest.mark.asyncio
async def test_a_slow_backend_ends_the_pass_on_budget_and_later_passes_resume(cleanup, monkeypatch):
    monkeypatch.setattr("services.container_cleanup.ORPHAN_VOLUME_LOOKUP_BUDGET_SECONDS", 0.05)
    ids = [str(UUID(int=i)) for i in range(1, 4)]
    names = [f"volume_{uid}" for uid in ids]
    ssh_client = VolumeHostSsh(dangling=names, created={name: "2026-09-10T08:00:00Z" for name in names}, volumes=names)
    slow = {ids[1]}
    seen = []

    async def is_orphan(volume_id, age):
        seen.append(volume_id)
        if volume_id in slow:
            await asyncio.sleep(1)  # far past the budget: the pass must return, not hang
        return True

    assert await cleanup.remove_orphan_rental_volumes(ssh_client, _snapshot([LIVE]), "exec-1", is_orphan) == 1
    assert _removed(ssh_client) == [names[0]]
    slow.clear()  # the backend recovers: the next pass picks up after the volume it stopped on
    ssh_client._dangling = names[1:]
    ssh_client.commands.clear()
    seen.clear()
    assert await cleanup.remove_orphan_rental_volumes(ssh_client, _snapshot([LIVE]), "exec-1", is_orphan) == 2
    assert seen == [ids[1], ids[2]] and _removed(ssh_client) == names[1:]


@pytest.mark.asyncio
async def test_a_budget_cut_resumes_at_the_cut_volume_so_it_is_not_skipped_forever(cleanup, monkeypatch):
    # kept A then eligible B, each lookup over half the budget: every pass used to finish A and cut B
    monkeypatch.setattr("services.container_cleanup.ORPHAN_VOLUME_LOOKUP_BUDGET_SECONDS", 0.3)
    ids = [str(UUID(int=i)) for i in range(1, 3)]
    names = [f"volume_{uid}" for uid in ids]
    ssh_client = VolumeHostSsh(dangling=names, created={name: "2026-09-10T08:00:00Z" for name in names}, volumes=names)

    async def is_orphan(volume_id, age):
        await asyncio.sleep(0.2)
        return volume_id == ids[1]

    passes = [await cleanup.remove_orphan_rental_volumes(ssh_client, _snapshot([LIVE]), "exec-1", is_orphan) for _ in range(2)]
    assert passes == [0, 1] and _removed(ssh_client) == [names[1]]


@pytest.mark.asyncio
async def test_a_first_volume_that_eats_the_whole_budget_does_not_pin_the_pass(cleanup, monkeypatch):
    monkeypatch.setattr("services.container_cleanup.ORPHAN_VOLUME_LOOKUP_BUDGET_SECONDS", 0.05)
    ids = [str(UUID(int=i)) for i in range(1, 3)]
    names = [f"volume_{uid}" for uid in ids]
    ssh_client = VolumeHostSsh(dangling=names, created={name: "2026-09-10T08:00:00Z" for name in names}, volumes=names)
    seen = []

    async def is_orphan(volume_id, age):
        seen.append(volume_id)
        if volume_id == ids[0]:
            await asyncio.sleep(1)
        return True

    for _ in range(2):
        await cleanup.remove_orphan_rental_volumes(ssh_client, _snapshot([LIVE]), "exec-1", is_orphan)
    assert seen[:2] == ids and _removed(ssh_client) == [names[1]]


class EvictingSshClient(FakeSshClient):
    """DAH-5014: each removed volume gives its size back, and docker reports when each was created."""

    def __init__(self, volumes: dict[str, tuple[str, float]], free_gb: float):
        super().__init__(containers=[], volumes=list(volumes), free_gb=free_gb)
        self._sizes = volumes
        self._free_gb = free_gb

    async def run(self, command: str, *args, **kwargs):
        if "volume inspect" in command:
            self.commands.append(command)
            body = "\n".join(f"{name} {created}" for name, (created, _) in self._sizes.items())
            return SimpleNamespace(exit_status=0, stdout=body, stderr="")
        result = await super().run(command, *args, **kwargs)
        if "volume rm" in command:
            for name in _removal_targets(command):
                self._free_gb += self._sizes[name][1]
            self._free_bytes = str(int(self._free_gb * GB))
        return result


@pytest.mark.asyncio
async def test_the_engy_checkpoint_is_reclaimed_oldest_first_and_only_as_far_as_the_floor(cleanup):
    # An old Dolphin cache goes first; that alone gets the node back over the floor, so the newer
    # 1.5 TB Engy checkpoint stays for the next fast restart.
    ssh_client = EvictingSshClient(
        {
            "engy_cache_hf_kimi-k3": ("2026-10-22T10:00:00Z", 1500),
            "dphn_cache_hf_x": ("2026-10-01T10:00:00Z", 80),
        },
        free_gb=40,
    )

    assert await cleanup.reclaim_dphn_cache_when_disk_is_tight(ssh_client, "exec-1") == 1
    assert _removed(ssh_client) == ["dphn_cache_hf_x"]


@pytest.mark.asyncio
async def test_a_live_customer_pod_keeps_the_cache_even_under_the_floor(cleanup):
    # Fish 26 Oct: the cache is kept across rentals and listed as used, so a renter who took the
    # listed free disk leaves the node under the floor by design
    ssh_client = FakeSshClient(["pod_abc"], ["engy_cache_hf_kimi-k3"], free_gb=10)

    assert await cleanup.reclaim_dphn_cache_when_disk_is_tight(ssh_client, "exec-1") == 0
    assert _removed(ssh_client) == []
    (guard,) = [c for c in ssh_client.commands if "docker ps" in c]
    assert 'name=filler_' in guard and 'name=pod_' in guard


class CreatedSweepSshClient(FakeSshClient):
    """A sweep `docker run` killed between create and start: `lium_cache_sweep` sits in Created, never
    auto-removed, and dockerd refuses `volume rm` on the cache volume it mounts until it is gone."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sweep_container_exists = True

    async def run(self, command: str, *args, **kwargs):
        if "volume rm" in command:
            # the shell runs the command's parts left to right
            if "docker rm -f lium_cache_sweep" in command.split("volume rm", 1)[0]:
                self.sweep_container_exists = False
            self._refuses = set(self._volumes) if self.sweep_container_exists else set()
        return await super().run(command, *args, **kwargs)


@pytest.mark.asyncio
async def test_a_never_started_sweep_container_cannot_pin_the_cache_the_reclaim_removes(cleanup):
    ssh_client = CreatedSweepSshClient(containers=[], volumes=["engy_cache_hf_glm"], free_gb=40)

    assert await cleanup.reclaim_dphn_cache_when_disk_is_tight(ssh_client, "exec-1") == 1
    assert _removed(ssh_client) == ["engy_cache_hf_glm"] and ssh_client._volumes == []


@pytest.mark.asyncio
async def test_the_download_sweep_removes_its_helper_container_after_the_run_too(cleanup):
    ssh_client = FakeSshClient(containers=[], volumes=["engy_cache_hf_glm"], free_gb=40)

    await cleanup.sweep_abandoned_download_temporaries(ssh_client, "exec-1")

    (sweep,) = [c for c in ssh_client.commands if "docker run" in c]
    before, _, after = sweep.partition("docker run")
    # `--rm` never fires for a container the killed client created but never started
    assert "docker rm -f lium_cache_sweep" in before and "docker rm -f lium_cache_sweep" in after


def test_only_never_started_capability_checks_are_picked():
    listing = "\n".join(
        [
            "a1 lium_capability_check_12_ab daturaai/compute-subnet-executor:latest \"nvidia-smi\"",
            "b2 eager_kalam daturaai/compute-subnet-executor:latest \"nvidia-smi\"",
            "c3 executor daturaai/compute-subnet-executor:latest \"/start.sh\"",
            "d4 his_job nvidia/cuda:12.4 \"nvidia-smi\"",
        ]
    )

    assert never_started_check_containers(listing) == ["a1", "b2"]


@pytest.mark.asyncio
async def test_the_stale_cleanup_removes_never_started_check_containers(cleanup):
    class CreatedChecksSshClient(FakeSshClient):
        async def run(self, command: str, *args, **kwargs):
            if "status=created" in command:
                self.commands.append(command)
                body = "a1 lium_capability_check_12_ab daturaai/compute-subnet-executor:latest nvidia-smi"
                return SimpleNamespace(exit_status=0, stdout=body, stderr="")
            return await super().run(command, *args, **kwargs)

    ssh_client = CreatedChecksSshClient(containers=[], volumes=[], free_gb=900)

    assert await cleanup.remove_never_started_check_containers(ssh_client, "exec-1") == 1
    assert any(c.startswith("/usr/bin/docker rm -f a1 ") for c in ssh_client.commands)
