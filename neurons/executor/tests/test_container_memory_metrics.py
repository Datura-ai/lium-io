"""The pod panel's RAM reading leaves out the reclaimable page cache, like `docker stats`."""

import sys
from unittest.mock import MagicMock, patch

try:
    import pynvml  # noqa: F401
except ImportError:  # not needed here: the tests pass no GPU UUIDs
    sys.modules["pynvml"] = MagicMock()

from services import hardware_service

GIB = 1024**3


def _metrics(memory_stats: dict) -> dict:
    container = MagicMock()
    container.attrs = {"HostConfig": {}, "Mounts": []}
    container.stats.return_value = {
        "cpu_stats": {"cpu_usage": {"total_usage": 2}, "system_cpu_usage": 2, "online_cpus": 4},
        "precpu_stats": {"cpu_usage": {"total_usage": 1}, "system_cpu_usage": 1},
        "memory_stats": memory_stats,
    }
    client = MagicMock()
    client.containers.get.return_value = container
    with (
        patch.object(hardware_service, "get_docker_client", return_value=client),
        patch.object(hardware_service, "_get_filesystem_usage", return_value={}),
    ):
        return hardware_service.get_container_metrics("pod_x", [])["memory"]


def test_cgroup_v2_page_cache_is_not_used_memory():
    # 126 GiB of processes plus a page cache that fills the rest of a 1000 GiB limit
    memory = _metrics({"usage": 998 * GIB, "limit": 1000 * GIB, "stats": {"inactive_file": 872 * GIB}})
    assert memory["used"] == 126 * GIB
    assert memory["utilization"] == 12.6


def test_cgroup_v1_uses_total_inactive_file():
    memory = _metrics({"usage": 10 * GIB, "limit": 20 * GIB, "stats": {"total_inactive_file": 6 * GIB}})
    assert memory["used"] == 4 * GIB
    assert memory["utilization"] == 20.0


def test_no_cache_figures_keep_docker_usage():
    assert _metrics({"usage": 5 * GIB, "limit": 10 * GIB})["used"] == 5 * GIB
    # a cache figure above usage is not trusted (docker CLI does the same)
    assert _metrics({"usage": 5 * GIB, "limit": 10 * GIB, "stats": {"inactive_file": 6 * GIB}})["used"] == 5 * GIB
