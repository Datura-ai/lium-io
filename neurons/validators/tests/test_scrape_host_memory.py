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
    "list_workload_container_ids",
    "get_host_memory",
}
NS = build_scrape_namespace(SRC / "miner_jobs" / "machine_scrape.py", HELPERS, {"os": os, "docker_api_get": None})

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


DOCKER_CONTAINERS = [
    {"Id": POD, "Names": ["/pod_1"]},
    {"Id": FILLER, "Names": ["/filler_2"]},
    {"Id": OTHER, "Names": ["/executor-executor-1"]},
]


def workloads():
    return sorted([POD, FILLER])


def test_the_workloads_are_the_running_pod_and_filler_containers(monkeypatch) -> None:
    monkeypatch.setitem(NS, "docker_api_get", lambda path: DOCKER_CONTAINERS if path == "/containers/json" else [])

    # the executor's own container is the host's
    assert NS["list_workload_container_ids"]() == [POD, FILLER]


def test_host_memory_outside_the_workloads(tmp_path: Path) -> None:
    kmsg = [
        "3,10,5000000,-;Out of memory: Killed process 12 (python)",
        "3,11,7000000,-;Memory cgroup out of memory: Killed process 99 (train)",
        "6,12,8000000,-;eth0: link up",
    ]
    facts = NS["get_host_memory"](list_workloads=workloads, read_kmsg=lambda: kmsg, **_host(tmp_path))

    assert facts["hm_errors"] == {}
    assert facts["hm_swap_total_kb"] == 8000 and facts["hm_swap_used_kb"] == 2000
    assert (facts["hm_hugepages_total"], facts["hm_hugepages_free"], facts["hm_hugepage_size_kb"]) == (16, 10, 2048)
    assert facts["hm_workload_containers"] == 2
    # used 80000 - workloads 50000
    assert facts["hm_outside_workloads_kb"] == 30000
    # (80000 - 1000 - 30000 + 4000) - (50000 - 14000 + 3000)
    assert facts["hm_outside_workloads_no_cache_kb"] == 14000
    assert facts["hm_mapped_outside_workloads_kb"] == 9000 - 3500
    assert facts["hm_oom_kills_since_boot"] == 3
    assert (facts["hm_last_oom_kind"], facts["hm_last_oom_at"]) == ("memcg", 1700000007.0)


def test_an_unreadable_container_cgroup_nulls_only_the_outside_fields(tmp_path: Path) -> None:
    facts = NS["get_host_memory"](list_workloads=workloads, read_kmsg=lambda: [], **_host(tmp_path, pod_cgroup=False))

    assert facts["hm_outside_workloads_kb"] is None
    assert "no memory cgroup for container aaaaaaaaaaaa" in facts["hm_errors"]["hm_outside_workloads_kb"]
    assert facts["hm_swap_used_kb"] == 2000
    # kills counted, but the kernel log no longer has them
    assert facts["hm_last_oom_at"] is None
    assert facts["hm_errors"]["hm_last_oom_at"] == "OOM kills since boot, none left in the kernel log"


def test_an_unreadable_kernel_log_is_a_reason_not_a_failure(tmp_path: Path) -> None:
    def denied():
        raise PermissionError("kmsg")

    facts = NS["get_host_memory"](list_workloads=workloads, read_kmsg=denied, **_host(tmp_path))

    assert facts["hm_last_oom_kind"] is None
    assert "PermissionError" in facts["hm_errors"]["hm_last_oom_at"]
    assert facts["hm_outside_workloads_kb"] == 30000


def test_a_pod_started_during_the_reads_nulls_the_subtraction(tmp_path: Path) -> None:
    # review on lium-io#2361: a listing that changed between before and after the reads leaves a container uncounted
    listings = iter([[FILLER], sorted([POD, FILLER])])
    facts = NS["get_host_memory"](lambda: next(listings), read_kmsg=lambda: [], **_host(tmp_path))

    assert facts["hm_workload_containers"] is None
    for key in ("hm_outside_workloads_kb", "hm_outside_workloads_no_cache_kb", "hm_mapped_outside_workloads_kb"):
        assert facts[key] is None
        assert facts["hm_errors"][key] == "the workload containers changed during the memory reads"
    assert facts["hm_errors"]["hm_workload_containers"] == "the workload containers changed during the memory reads"
    assert facts["hm_swap_used_kb"] == 2000


def test_a_failed_docker_listing_is_no_count_not_zero_workloads(tmp_path: Path) -> None:
    def refused():
        raise OSError("docker.sock")

    facts = NS["get_host_memory"](refused, read_kmsg=lambda: [], **_host(tmp_path))

    assert facts["hm_workload_containers"] is None
    assert facts["hm_outside_workloads_no_cache_kb"] is None
    assert facts["hm_errors"]["hm_outside_workloads_no_cache_kb"].startswith("docker inventory failed: OSError")
    assert facts["hm_mem_total_kb"] == 100000


def test_a_kernel_oom_warning_is_not_a_kill() -> None:
    records = [
        "3,10,7000000,-;Memory cgroup out of memory: Killed process 200 (python)",
        "4,11,9000000,-;Out of memory and no killable processes...",
        "6,12,9500000,-;app: Out of memory: Killed process in a log line",
    ]
    assert NS["last_oom_from_kmsg"](records, 1700000000) == ("memcg", 1700000007.0)
    assert NS["last_oom_from_kmsg"](["3,1,2000000,-;Out of memory (oom_kill_allocating_task): Killed process 5 (x)"], 10) == (
        "global",
        12.0,
    )


def test_the_obfuscated_scrape_reads_host_memory(tmp_path: Path) -> None:
    # review on lium-io#2361: run get_host_memory as it ships, out of obfuscator.py's output (a nested helper's
    # parameter kept its name while its body was renamed, so the scrape only reported an error)
    from miner_jobs.obfuscator import obfuscate_code

    tree = ast.parse(obfuscate_code((SRC / "miner_jobs" / "machine_scrape.py").read_text()))
    # the script up to its run: the scrape itself starts at `machine_specs = get_machine_specs()`
    run = next(
        i
        for i, node in enumerate(tree.body)
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "machine_specs"
    )
    definitions = tree.body[:run]
    namespace: dict = {}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "obfuscated", "exec"), namespace)
    function = next(
        namespace[node.name]
        for node in definitions
        if isinstance(node, ast.FunctionDef)
        and any(isinstance(child, ast.Constant) and child.value == "hm_swap_used_kb" for child in ast.walk(node))
    )
    paths = _host(tmp_path)
    kmsg = ["3,11,7000000,-;Memory cgroup out of memory: Killed process 99 (train)"]

    facts = function(
        workloads, paths["host_root"], paths["meminfo_path"], paths["vmstat_path"], paths["stat_path"], lambda: kmsg
    )

    assert facts["hm_errors"] == {}
    assert facts["hm_swap_used_kb"] == 2000
    assert facts["hm_outside_workloads_no_cache_kb"] == 14000
    assert facts["hm_last_oom_kind"] == "memcg"


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
