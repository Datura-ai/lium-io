"""DAH-2475: the periodic backstop that gives the DPHN filler cache back under disk pressure.

The create-time sweep only removes STALE cache versions and never reclaims — a node's free disk is
low precisely BECAUSE the cache is downloaded, so reclaiming there would make the backend grant it
again next cycle and the node would re-download ~37 GB forever. This backstop is the one place that
reclaims, and it decides on real free space: below the rental listing floor the node earns nothing.
"""

from types import SimpleNamespace

import pytest

from services.container_cleanup import ContainerCleanup

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
