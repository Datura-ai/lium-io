"""DAH-6379 — get_host_memory() in machine_scrape.py: host memory outside the workload containers, swap, hugepages,
mappings outside the containers and the last OOM. machine_scrape.py is a script (importing it runs the scrape), so the
helpers are extracted by ast (as in test_scrape_interconnect.py)."""

import ast
import os
from pathlib import Path

from neurons.validators.tests.helpers import build_scrape_namespace, dict_literal_keys

SRC = Path(__file__).resolve().parents[1] / "src"
HELPERS = {
    "HOST_MEMORY_ROOT",
    "WORKLOAD_CONTAINER_PREFIXES",
    "CONTAINER_MEMORY_CGROUPS",
    "KMSG_PATH",
    "KMSG_MAX_RECORDS",
    "parse_meminfo_kb",
    "parse_memory_stat_kb",
    "last_oom_from_kmsg",
    "read_kmsg_records",
    "container_memory_kb",
    "get_host_memory",
}
NS = build_scrape_namespace(SRC / "miner_jobs" / "machine_scrape.py", HELPERS, {"os": os})

MEMINFO = """MemTotal:       100000 kB
MemFree:         20000 kB
MemAvailable:    50000 kB
Buffers:          1000 kB
Cached:          30000 kB
SwapCached:          0 kB
SwapTotal:        8000 kB
SwapFree:         6000 kB
Shmem:            4000 kB
Mapped:           9000 kB
HugePages_Total:     16
HugePages_Free:      10
Hugepagesize:     2048 kB
"""
POD = "a" * 64
FILLER = "b" * 64
OTHER = "c" * 64


def _host(tmp_path: Path, *, pod_cgroup: bool = True) -> dict:
    (tmp_path / "meminfo").write_text(MEMINFO)
    (tmp_path / "vmstat").write_text("nr_free_pages 1\noom_kill 3\n")
    (tmp_path / "stat").write_text("cpu 1 2 3\nbtime 1700000000\n")
    root = tmp_path / "host"
    cgroups = [(f"sys/fs/cgroup/docker/{FILLER}", 10000, 4000, 1000, 500)]
    if pod_cgroup:
        cgroups.append((f"sys/fs/cgroup/system.slice/docker-{POD}.scope", 40000, 10000, 2000, 3000))
    for path, current, file_, shmem, mapped in cgroups:
        d = root / path
        d.mkdir(parents=True)
        (d / "memory.current").write_text(f"{current * 1024}\n")
        (d / "memory.stat").write_text(
            f"anon {(current - file_) * 1024}\nfile {file_ * 1024}\nshmem {shmem * 1024}\nfile_mapped {mapped * 1024}\n"
        )
    return {
        "host_root": str(root),
        "meminfo_path": str(tmp_path / "meminfo"),
        "vmstat_path": str(tmp_path / "vmstat"),
        "stat_path": str(tmp_path / "stat"),
    }


CONTAINERS = [
    {"each_container_id": POD, "each_name": "pod_1"},
    {"each_container_id": FILLER, "each_name": "filler_2"},
    {"each_container_id": OTHER, "each_name": "executor-executor-1"},
]


def test_host_memory_outside_the_workloads(tmp_path: Path) -> None:
    kmsg = [
        "3,10,5000000,-;Out of memory: Killed process 12 (python)",
        "3,11,7000000,-;Memory cgroup out of memory: Killed process 99 (train)",
        "6,12,8000000,-;eth0: link up",
    ]
    facts = NS["get_host_memory"](CONTAINERS, read_kmsg=lambda: kmsg, **_host(tmp_path))

    assert facts["hm_errors"] == {}
    assert facts["hm_swap_total_kb"] == 8000 and facts["hm_swap_used_kb"] == 2000
    assert (facts["hm_hugepages_total"], facts["hm_hugepages_free"], facts["hm_hugepage_size_kb"]) == (16, 10, 2048)
    assert facts["hm_workload_containers"] == 2  # the executor's own container is the host's
    # used 80000 - workloads 50000
    assert facts["hm_outside_workloads_kb"] == 30000
    # (80000 - 1000 - 30000 + 4000) - (50000 - 14000 + 3000)
    assert facts["hm_outside_workloads_no_cache_kb"] == 14000
    assert facts["hm_mapped_outside_workloads_kb"] == 9000 - 3500
    assert facts["hm_oom_kills_since_boot"] == 3
    assert (facts["hm_last_oom_kind"], facts["hm_last_oom_at"]) == ("memcg", 1700000007.0)


def test_an_unreadable_container_cgroup_nulls_only_the_outside_fields(tmp_path: Path) -> None:
    facts = NS["get_host_memory"](CONTAINERS, read_kmsg=lambda: [], **_host(tmp_path, pod_cgroup=False))

    assert facts["hm_outside_workloads_kb"] is None
    assert "no memory cgroup for container aaaaaaaaaaaa" in facts["hm_errors"]["hm_outside_workloads_kb"]
    assert facts["hm_swap_used_kb"] == 2000
    # kills counted, but the kernel log no longer has them
    assert facts["hm_last_oom_at"] is None
    assert facts["hm_errors"]["hm_last_oom_at"] == "OOM kills since boot, none left in the kernel log"


def test_an_unreadable_kernel_log_is_a_reason_not_a_failure(tmp_path: Path) -> None:
    def denied():
        raise PermissionError("kmsg")

    facts = NS["get_host_memory"](CONTAINERS, read_kmsg=denied, **_host(tmp_path))

    assert facts["hm_last_oom_kind"] is None
    assert "PermissionError" in facts["hm_errors"]["hm_last_oom_at"]
    assert facts["hm_outside_workloads_kb"] == 30000


def test_host_memory_keys_are_wired_through_both_obfuscation_tables() -> None:
    service = ast.parse((SRC / "services" / "file_encrypt_service.py").read_text())
    scrape = (SRC / "miner_jobs" / "machine_scrape.py").read_text()
    original_keys = dict_literal_keys(service, "ORIGINAL_KEYS")
    all_keys = dict_literal_keys(service, "all_keys")
    used = {k for k in original_keys if k.startswith(("hm_", "data_host_memory"))}

    assert len(used) == 20
    for key in used:
        assert f'"{key}"' in scrape
        assert key in all_keys
