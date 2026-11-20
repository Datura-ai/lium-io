"""DAH-6379 — get_host_memory() in machine_scrape.py: host memory outside the workload containers, swap, hugepages,
mappings outside the containers and the last OOM. machine_scrape.py is a script (importing it runs the scrape), so the
helpers are extracted by ast (as in test_scrape_interconnect.py)."""

import ast
import errno
import os
import time
import types
from pathlib import Path

import pytest

from neurons.validators.tests.helpers import build_scrape_namespace, dict_literal_keys

SRC = Path(__file__).resolve().parents[1] / "src"
HELPERS = {
    "HOST_MEMORY_ROOT",
    "WORKLOAD_CONTAINER_PREFIXES",
    "CONTAINER_MEMORY_CGROUPS",
    "KMSG_PATH",
    "KMSG_MAX_READS",
    "HOST_MEMORY_INVENTORY_SECONDS",
    "parse_meminfo_kb",
    "parse_memory_stat_kb",
    "last_oom_from_kmsg",
    "read_kmsg_records",
    "container_memory_kb",
    "list_workload_container_ids",
    "get_host_memory",
}
NS = build_scrape_namespace(SRC / "miner_jobs" / "machine_scrape.py", HELPERS, {"os": os, "time": time, "docker_api_get": None})

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
            "hugetlb 0\n"  # hugetlb accounting on: the pool the fixture reserves is the host's
        )
        (d / "hugetlb.2MB.current").write_text("0\n")
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


STARTED = "2026-11-19T10:00:00.000000000Z"


def workloads(deadline=None):
    return sorted([(POD, STARTED), (FILLER, STARTED)])


def _docker(path, timeout=None, deadline=None):
    if path == "/containers/json":
        return DOCKER_CONTAINERS
    return {"Id": path.split("/")[2], "State": {"StartedAt": STARTED}}


def test_the_workloads_are_the_running_pod_and_filler_containers(monkeypatch) -> None:
    monkeypatch.setitem(NS, "docker_api_get", _docker)

    # the executor's own container is the host's
    assert NS["list_workload_container_ids"](time.monotonic() + 5) == [(POD, STARTED), (FILLER, STARTED)]


def test_both_listings_share_one_short_time_budget(monkeypatch, tmp_path: Path) -> None:
    # review on lium-io#2361: each docker read waited up to 20 s, so two listings could outlast the 300 s scrape
    clock = [1000.0]
    timeouts = []

    def slow_docker(path, timeout=None, deadline=None):
        timeouts.append(timeout)
        clock[0] += 2  # every docker read takes 2 s
        return _docker(path)

    monkeypatch.setitem(NS, "time", types.SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setitem(NS, "docker_api_get", slow_docker)
    facts = NS["get_host_memory"](read_kmsg=lambda: [], **_host(tmp_path))

    assert clock[0] - 1000.0 <= NS["HOST_MEMORY_INVENTORY_SECONDS"] + 2
    assert all(t <= NS["HOST_MEMORY_INVENTORY_SECONDS"] for t in timeouts)
    assert facts["hm_workload_containers"] is None
    assert "time budget" in facts["hm_errors"]["hm_outside_workloads_kb"]
    assert facts["hm_mem_total_kb"] == 100000


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
    listings = iter([[(FILLER, STARTED)], workloads()])
    facts = NS["get_host_memory"](lambda deadline: next(listings), read_kmsg=lambda: [], **_host(tmp_path))

    assert facts["hm_workload_containers"] is None
    for key in ("hm_outside_workloads_kb", "hm_outside_workloads_no_cache_kb", "hm_mapped_outside_workloads_kb"):
        assert facts[key] is None
        assert facts["hm_errors"][key] == "the workload containers changed during the memory reads"
    assert facts["hm_errors"]["hm_workload_containers"] == "the workload containers changed during the memory reads"
    assert facts["hm_swap_used_kb"] == 2000


def test_a_failed_docker_listing_is_no_count_not_zero_workloads(tmp_path: Path) -> None:
    def refused(deadline):
        raise OSError("docker.sock")

    facts = NS["get_host_memory"](refused, read_kmsg=lambda: [], **_host(tmp_path))

    assert facts["hm_workload_containers"] is None
    assert facts["hm_outside_workloads_no_cache_kb"] is None
    assert facts["hm_errors"]["hm_outside_workloads_no_cache_kb"].startswith("docker inventory failed: OSError")
    assert facts["hm_mem_total_kb"] == 100000


def test_a_cgroup_v1_pod_counts_its_child_cgroups_shmem(tmp_path: Path) -> None:
    # review on lium-io#2361: v1 usage includes child cgroups; its plain shmem is local, total_shmem hierarchical
    paths = _host(tmp_path)
    # no huge page pool: a v1 memory cgroup never accounts huge pages (see test_the_huge_page_pool_counts_where_its_owner_is_known)
    Path(paths["meminfo_path"]).write_text(MEMINFO.replace("HugePages_Total:     16", "HugePages_Total:      0"))
    pod = Path(paths["host_root"]) / f"sys/fs/cgroup/system.slice/docker-{POD}.scope"
    for f in pod.iterdir():
        f.unlink()
    pod.rmdir()
    v1 = Path(paths["host_root"]) / f"sys/fs/cgroup/memory/docker/{POD}"
    v1.mkdir(parents=True)
    (v1 / "memory.usage_in_bytes").write_text(f"{40000 * 1024}\n")
    (v1 / "memory.stat").write_text(
        "\n".join(
            f"{k} {v * 1024}"
            for k, v in (("cache", 0), ("shmem", 0), ("mapped_file", 0), ("total_cache", 10000), ("total_shmem", 2000), ("total_mapped_file", 3000))
        )
        + "\n"
    )
    facts = NS["get_host_memory"](list_workloads=workloads, read_kmsg=lambda: [], **paths)

    # the same figures as the v2 pod in test_host_memory_outside_the_workloads
    assert facts["hm_outside_workloads_no_cache_kb"] == 14000
    assert facts["hm_mapped_outside_workloads_kb"] == 9000 - 3500


def test_a_pod_restarted_during_the_reads_nulls_the_subtraction(tmp_path: Path) -> None:
    # review on lium-io#2361: a restart keeps the id, so only the start time shows the cgroup is a new run
    restarted = sorted([(POD, "2026-11-19T10:05:00.000000000Z"), (FILLER, STARTED)])
    listings = iter([workloads(), restarted])
    facts = NS["get_host_memory"](lambda deadline: next(listings), read_kmsg=lambda: [], **_host(tmp_path))

    assert facts["hm_outside_workloads_no_cache_kb"] is None
    assert facts["hm_errors"]["hm_workload_containers"] == "the workload containers changed during the memory reads"


def test_a_workload_that_grew_between_the_reads_is_no_negative_figure(tmp_path: Path) -> None:
    # review on lium-io#2361: 5,000,000 kB host use, then 15,000,000 kB workload use, published -10,000,000 kB
    paths = _host(tmp_path)
    Path(paths["meminfo_path"]).write_text(MEMINFO.replace("132768 kB", "5052768 kB"))
    pod = Path(paths["host_root"]) / f"sys/fs/cgroup/system.slice/docker-{POD}.scope"
    (pod / "memory.current").write_text(f"{15000000 * 1024}\n")
    facts = NS["get_host_memory"](list_workloads=workloads, read_kmsg=lambda: [], **paths)

    for key in ("hm_outside_workloads_kb", "hm_outside_workloads_no_cache_kb", "hm_mapped_outside_workloads_kb"):
        assert facts[key] is None
        assert "negative residual" in facts["hm_errors"][key]


@pytest.mark.parametrize(
    "workload_set, pod_current_gib, pod_stat, expected_gib, pod_hugetlb_gib, pod_rsvd_gib",
    [
        # review on lium-io#2361: 64 GiB total, 46 GiB free, a 16 GiB huge page pool; used = MemTotal - MemFree = 18 GiB
        ("none", 0, None, 18, None, None),  # a host-only pool is host use
        ("pod", 0, "anon 0\nhugetlb 0\n", 18, 0, None),  # an ordinary pod beside a host-only pool
        ("pod", 16, f"anon 0\nhugetlb {16 * 1024 * 1024 * 1024}\n", 2, 16, None),  # the pod owns the pool, accounting on
        # review at ceb64cb: hugetlb.<size>.rsvd.current counts reservations, not usage, and is never summed
        ("pod", 16, f"anon 0\nhugetlb {16 * 1024 * 1024 * 1024}\n", 2, 16, 16),
        ("pod", 0, "anon 0\n", None, None, None),  # the pod may own the pool, accounting unavailable: unknown
        # review on lium-io#2361 at 34bf9ed: pages faulted before the remount turned accounting on are in
        # hugetlb.*.current but not in memory.stat, so the charge cannot be trusted: unknown
        ("pod", 0, "anon 0\nhugetlb 0\n", None, 16, None),
    ],
)
def test_the_huge_page_pool_counts_where_its_owner_is_known(
    tmp_path: Path,
    workload_set: str,
    pod_current_gib: int,
    pod_stat: str | None,
    expected_gib: int | None,
    pod_hugetlb_gib: int | None,
    pod_rsvd_gib: int | None,
) -> None:
    gib = 1024 * 1024
    paths = _host(tmp_path)
    Path(paths["meminfo_path"]).write_text(
        f"MemTotal: {64 * gib} kB\nMemFree: {46 * gib} kB\nMemAvailable: {46 * gib} kB\nBuffers: 0 kB\nCached: 0 kB\n"
        f"SwapTotal: 0 kB\nSwapFree: 0 kB\nShmem: 0 kB\nMapped: 0 kB\nHugePages_Total: 8\nHugePages_Free: 0\n"
        f"Hugepagesize: {2 * gib} kB\nHugetlb: {16 * gib} kB\n"
    )
    filler = Path(paths["host_root"]) / f"sys/fs/cgroup/docker/{FILLER}"
    pod = Path(paths["host_root"]) / f"sys/fs/cgroup/system.slice/docker-{POD}.scope"
    (filler / "memory.current").write_text("0\n")
    (filler / "memory.stat").write_text("anon 0\nhugetlb 0\n")
    if pod_stat is not None:
        (pod / "memory.current").write_text(f"{pod_current_gib * gib * 1024}\n")
        (pod / "memory.stat").write_text(pod_stat)
        hugetlb_file = pod / "hugetlb.2MB.current"
        if pod_hugetlb_gib is None:
            hugetlb_file.unlink()
        else:
            hugetlb_file.write_text(f"{pod_hugetlb_gib * gib * 1024}\n")
        if pod_rsvd_gib is not None:
            (pod / "hugetlb.2MB.rsvd.current").write_text(f"{pod_rsvd_gib * gib * 1024}\n")
    listing = (lambda deadline=None: []) if workload_set == "none" else workloads

    facts = NS["get_host_memory"](list_workloads=listing, read_kmsg=lambda: [], **paths)
    expected_kb = None if expected_gib is None else expected_gib * gib
    assert facts["hm_outside_workloads_kb"] == expected_kb
    assert facts["hm_outside_workloads_no_cache_kb"] == expected_kb
    if expected_kb is None:
        assert "hugetlb" in facts["hm_errors"]["hm_outside_workloads_kb"]
        assert facts["hm_mapped_outside_workloads_kb"] == 0


@pytest.mark.parametrize(
    "free_gib, hugetlb_gib, expected_gib",
    [
        # review on lium-platform#3255 at fe9b5e7: an ordinary pod using 8 GiB, no hugetlb line, beside a pool whose
        # eight 2 GiB pages are all free (64 total, 38 free): no workload holds a huge page, so 26 - 8 = 18 GiB
        (38, 16, 18),
        # a second, non-default pool (Hugetlb above the free default pages) may be a workload's: unknown
        (36, 18, None),
    ],
    ids=["all_pages_free", "non_default_pool"],
)
def test_a_pool_with_every_page_free_holds_no_workload_share(
    tmp_path: Path, free_gib: int, hugetlb_gib: int, expected_gib: int | None
) -> None:
    gib = 1024 * 1024
    paths = _host(tmp_path)
    Path(paths["meminfo_path"]).write_text(
        f"MemTotal: {64 * gib} kB\nMemFree: {free_gib * gib} kB\nMemAvailable: 0 kB\nBuffers: 0 kB\nCached: 0 kB\n"
        f"SwapTotal: 0 kB\nSwapFree: 0 kB\nShmem: 0 kB\nMapped: 0 kB\nHugePages_Total: 8\nHugePages_Free: 8\n"
        f"Hugepagesize: {2 * gib} kB\nHugetlb: {hugetlb_gib * gib} kB\n"
    )
    filler = Path(paths["host_root"]) / f"sys/fs/cgroup/docker/{FILLER}"
    pod = Path(paths["host_root"]) / f"sys/fs/cgroup/system.slice/docker-{POD}.scope"
    (filler / "memory.current").write_text("0\n")
    (filler / "memory.stat").write_text("anon 0\nhugetlb 0\n")
    (pod / "memory.current").write_text(f"{8 * gib * 1024}\n")
    (pod / "memory.stat").write_text("anon 0\n")
    (pod / "hugetlb.2MB.current").unlink()

    facts = NS["get_host_memory"](list_workloads=workloads, read_kmsg=lambda: [], **paths)
    expected_kb = None if expected_gib is None else expected_gib * gib
    assert facts["hm_outside_workloads_kb"] == expected_kb
    assert facts["hm_outside_workloads_no_cache_kb"] == expected_kb


def test_a_docker_reply_trickled_in_parts_still_ends_at_the_deadline(tmp_path: Path) -> None:
    # review on lium-io#2361: each part of the body arrived inside the per-read timeout, so the reply overran the budget
    import socket
    import threading

    ns = build_scrape_namespace(
        SRC / "miner_jobs" / "machine_scrape.py",
        {"DeadlineSocket", "UnixSocketHTTPConnection", "docker_api_get"},
        {"socket": socket, "time": time, "json": __import__("json"), "http": __import__("http.client"),
         "DOCKER_API_TIMEOUT_SECONDS": 20, "DOCKER_SOCKET_PATH": str(tmp_path / "docker.sock")},
    )
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(tmp_path / "docker.sock"))
    server.listen(1)

    def trickle():
        conn, _ = server.accept()
        conn.recv(4096)
        conn.sendall(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n")
        try:
            for _ in range(10):
                time.sleep(0.1)  # each part well inside the 1 s per-read timeout
                conn.sendall(b"1\r\n \r\n")
        except OSError:
            pass
        conn.close()

    threading.Thread(target=trickle, daemon=True).start()
    start = time.monotonic()
    try:
        ns["docker_api_get"]("/containers/json", 1, start + 0.25)
        raise AssertionError("the trickled reply was accepted past its deadline")
    except TimeoutError:
        pass
    finally:
        server.close()
    assert time.monotonic() - start < 0.6


class _Kmsg:
    O_RDONLY = O_NONBLOCK = 0

    def __init__(self, reads):
        self.reads = iter(reads)

    def open(self, path, flags):
        return 3

    def close(self, fd):
        pass

    def read(self, fd, size):
        item = next(self.reads)
        if isinstance(item, Exception):
            raise item
        return item


def test_an_overwritten_kernel_log_record_is_a_reason_not_a_stale_kill(monkeypatch) -> None:
    # review on lium-io#2361: an EPIPE skipped records, so an older kill could read as the newest
    epipe = OSError(errno.EPIPE, "overwritten")
    monkeypatch.setitem(NS, "os", _Kmsg([b"3,1,1,-;Memory cgroup out of memory: Killed process 1 (a)", epipe, b""]))
    try:
        NS["read_kmsg_records"]()
    except RuntimeError as exc:
        assert "overwritten" in str(exc)
    else:
        raise AssertionError("an overwritten record must not pass as a whole scan")


def test_a_kernel_log_past_the_read_budget_is_a_reason(monkeypatch) -> None:
    # review on lium-io#2361: reaching the cap returned the partial, oldest-first scan as success; EPIPE bypassed it
    monkeypatch.setitem(NS, "KMSG_MAX_READS", 3)
    monkeypatch.setitem(NS, "os", _Kmsg([b"6,1,1,-;a", b"6,2,2,-;b", b"6,3,3,-;c", b"6,4,4,-;d", b""]))
    try:
        NS["read_kmsg_records"]()
    except RuntimeError as exc:
        assert "3 reads" in str(exc)
    else:
        raise AssertionError("a scan cut at the budget must not pass as a whole scan")


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


def test_a_user_written_oom_lookalike_is_not_a_kill() -> None:
    # LOG_USER (facility 1): pri 14 = 1 << 3 | info, written by a process into /dev/kmsg
    records = [
        "3,10,7000000,-;Memory cgroup out of memory: Killed process 200 (python)",
        "14,11,9000000,-;Out of memory: Killed process 1 (fake)",
    ]
    assert NS["last_oom_from_kmsg"](records, 1700000000) == ("memcg", 1700000007.0)


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


@pytest.mark.parametrize(
    "pod_stat_extra, host_swapcached_kb, expected_no_cache_kb",
    [
        # review on lium-io#2361 at ceb64cb: the kernel counts swap cache in the node's file pages only (the cgroup's
        # in swapcached), so neither the pod's file nor the host's Cached holds it: nothing is taken out
        (f"swapcached {2 * 1024 * 1024 * 1024}\n", 2 * 1024 * 1024, 14000),
        # the host holds swap cache and the pod has no swapcached line: the figure still stands
        ("", 2 * 1024 * 1024, 14000),
        # no swap cache anywhere: the figure stands without the counter
        ("", 0, 14000),
    ],
)
def test_a_pods_swap_cache_counts_as_used_on_both_sides(
    tmp_path: Path, pod_stat_extra: str, host_swapcached_kb: int, expected_no_cache_kb: int
) -> None:
    paths = _host(tmp_path)
    Path(paths["meminfo_path"]).write_text(MEMINFO.replace("SwapCached:          0 kB", f"SwapCached: {host_swapcached_kb} kB"))
    for cgroup, file_kb in ((f"sys/fs/cgroup/docker/{FILLER}", 4000), (f"sys/fs/cgroup/system.slice/docker-{POD}.scope", 10000)):
        stat = Path(paths["host_root"]) / cgroup / "memory.stat"
        extra = pod_stat_extra if POD in cgroup else ("swapcached 0\n" if pod_stat_extra else "")
        lines = [l for l in stat.read_text().splitlines() if not l.startswith("file ")]
        stat.write_text("\n".join(lines + [f"file {file_kb * 1024}"]) + "\n" + extra)

    facts = NS["get_host_memory"](list_workloads=workloads, read_kmsg=lambda: [], **paths)
    assert facts["hm_outside_workloads_no_cache_kb"] == expected_no_cache_kb
    assert "hm_outside_workloads_no_cache_kb" not in facts["hm_errors"]
