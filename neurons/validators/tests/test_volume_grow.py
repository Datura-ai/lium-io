import asyncio
import fcntl
import os
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from services.volume_grow import (
    GrowPass,
    pod_ids_at_endpoint,
    GROW_HEADROOM_BYTES,
    GROW_SCRIPT,
    ORDER_LABEL,
    grow_short_volumes,
    credit_volumes_on_host,
    host_free_bytes,
    order_labels,
)

GB = 1024**3


@pytest.fixture(autouse=True)
def _host_lock(tmp_path, monkeypatch):
    # the host script's lock lives in /run on a node; here a writable path
    monkeypatch.setenv("VOLUME_GROW_LOCK", str(tmp_path / "grow.lock"))


def test_only_a_volume_granted_short_of_its_order_is_labelled():
    assert order_labels(466, 110) == {ORDER_LABEL: "466"}
    assert order_labels(466, 466) == {}
    assert order_labels(None, 110) == {}
    assert order_labels(466, None) == {}


def _ran(stdout="", exit_status=0):
    return SimpleNamespace(stdout=stdout, stderr="", exit_status=exit_status)


@pytest.mark.asyncio
async def test_only_a_live_pods_short_volume_is_grown_to_its_order():
    ssh = SimpleNamespace(
        run=AsyncMock(
            side_effect=[
                _ran("volume_live\nvolume_retained\n"),
                _ran("volume_live|466|vloopback:v3|\n"),
                _ran("/var/lib/docker\n"),
                _ran("abc\n"),
                _ran(f"grown {466 * GB}\n"),
            ]
        )
    )

    grown = await grow_short_volumes(ssh, {"live"})

    inspect, grow = ssh.run.await_args_list[1].args[0], ssh.run.await_args_list[4].args[0]
    assert (grown.sizes_gb, "volume_retained" in inspect) == ({"live": 466}, False)
    assert grow.startswith("timeout -k 5 120 nsenter -t 1 -m -- sh -c ")
    # the host path of the mount, not the plugin-internal /mnt/<volume> docker reports
    assert grow.endswith(
        f" grow /var/lib/docker/plugins/abc/propagated-mount/volume_live {466 * GB} {GROW_HEADROOM_BYTES} false"
    )


@pytest.mark.asyncio
async def test_a_failed_listing_grows_nothing_and_does_not_raise():
    ssh = SimpleNamespace(run=AsyncMock(side_effect=OSError("channel closed")))

    assert await grow_short_volumes(ssh, {"live"}) == GrowPass()


def _stub(bin_dir, name, body):
    path = bin_dir / name
    path.write_text("#!/bin/sh\n" + body + "\n")
    path.chmod(0o755)


def _xfs(bin_dir, log, before, after):
    """xfs_info answers `before` until xfs_growfs ran, then `after`."""
    grew = bin_dir / "grew"
    _stub(bin_dir, "xfs_growfs", f'echo "xfs_growfs $*" >> {log}; touch {grew}')
    _stub(
        bin_dir,
        "xfs_info",
        f"b={before // 4096}; [ -e {grew} ] && b={after // 4096}; "
        'echo "meta-data=/dev/loop7 isize=512 agcount=4"; echo "data     =   bsize=4096   blocks=$b, imaxpct=25"',
    )


@pytest.mark.parametrize(
    ("cur_gb", "alloc_gb", "avail_gb", "sparse", "outcome", "grow_call"),
    [
        (
            110,
            110,
            500,
            "false",
            f"grown {466 * GB}",
            f"fallocate -o {110 * GB} -l {356 * GB} /data/volume_live",
        ),
        (
            110,
            110,
            50,
            "false",
            f"grown {150 * GB}",
            f"fallocate -o {110 * GB} -l {40 * GB} /data/volume_live",
        ),
        (110, 5, 500, "true", f"grown {466 * GB}", f"truncate -s >{466 * GB} /data/volume_live"),
        # sparse, 5 GB written: its 105 GB unwritten is already in avail, so 115 free is no room, not 215
        (110, 5, 115, "true", f"no-room {110 * GB}", None),
        (110, 110, 5, "false", f"no-room {110 * GB}", None),
        (466, 466, 500, "false", f"full {466 * GB}", None),
    ],
    ids=[
        "to-order",
        "to-headroom",
        "sparse",
        "sparse-unwritten-not-counted-twice",
        "no-room",
        "already-full",
    ],
)
def test_the_host_script_grows_up_to_the_order_and_never_into_the_headroom(
    tmp_path, cur_gb, alloc_gb, avail_gb, sparse, outcome, grow_call
):
    log = tmp_path / "calls"
    _stub(tmp_path, "findmnt", "echo /dev/loop7")
    _stub(
        tmp_path,
        "losetup",
        f'[ "$1" = -c ] && echo "losetup $*" >> {log} || echo /data/volume_live',
    )
    # %s apparent size, %b blocks allocated, %B bytes per block
    _stub(
        tmp_path,
        "stat",
        f'case "$2" in %s) echo {cur_gb * GB};; %b) echo {alloc_gb * GB // 512};; %B) echo 512;; esac',
    )
    _stub(
        tmp_path,
        "df",
        f'echo "Filesystem 1-blocks Used Available Use% Mounted"; echo "/dev/x 0 0 {avail_gb * GB} 0% /"',
    )
    for tool in ("fallocate", "truncate"):
        _stub(tmp_path, tool, f'echo "{tool} $*" >> {log}')
    _xfs(tmp_path, log, before=cur_gb * GB, after=int(outcome.split()[-1]))

    result = subprocess.run(
        ["sh", "-c", GROW_SCRIPT, "grow", "/mnt/volume_live", str(466 * GB), str(10 * GB), sparse],
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"},
    )

    calls = log.read_text().splitlines() if log.exists() else []
    expected = (
        [grow_call, "losetup -c /dev/loop7", "xfs_growfs /mnt/volume_live"] if grow_call else []
    )
    # a preallocated grow first prints the disk it took
    took = (
        f"allocated {(int(outcome.split()[-1]) - cur_gb * GB)}"
        if grow_call and sparse == "false"
        else None
    )
    assert (result.stdout.strip().splitlines(), calls) == (
        [took, outcome] if took else [outcome],
        expected,
    )


def test_a_pods_volume_size_from_the_cleanup_pass_rides_on_its_pod_state():
    from services.task.checks.rented_machine import _with_pod_states
    from services.task.pipeline import CheckResult, ContextState

    from protocol.vc_protocol.compute_requests import RentedPod
    from protocol.vc_protocol.validator_requests import ContainerState

    ctx = SimpleNamespace(state=ContextState(volume_sizes_gb={"grown": 466}))
    result = CheckResult.model_construct(passed=True, event=None, updates={})
    pods = [
        RentedPod(pod_id="grown", container_name="c1"),
        RentedPod(pod_id="other", container_name="c2"),
    ]

    states = (
        _with_pod_states(result, ctx, pods, {"grown": ContainerState.RUNNING})
        .updates["state"]
        .pod_states
    )

    assert [(state.pod_id, state.volume_limit_gb) for state in states] == [
        ("grown", 466),
        ("other", None),
    ]


def test_a_create_reply_says_the_volume_grows_only_when_it_was_labelled_short():
    from payload_models.payloads import ContainerCreated

    def reply(grows: bool) -> dict:
        return ContainerCreated(
            miner_hotkey="hk",
            executor_id="ex",
            pod_id="pod",
            container_name="c",
            volume_name="v",
            port_maps=[],
            volume_grows_to_order=bool(order_labels(466, 110 if grows else 466)),
        ).model_dump()

    assert reply(True)["volume_grows_to_order"] is True
    assert "volume_grows_to_order" not in reply(False)


def test_the_host_script_finds_the_backing_file_outside_the_plugins_srv_view(tmp_path):
    host_file = tmp_path / "vloopback-v3" / "volume_live"
    host_file.parent.mkdir()
    host_file.write_text("")
    log = tmp_path / "calls"
    _stub(tmp_path, "findmnt", "echo /dev/loop7")
    _stub(tmp_path, "losetup", f'[ "$1" = -c ] || echo /srv{host_file}')
    _stub(tmp_path, "stat", f'echo "$3" >> {log}; echo {466 * GB}')

    result = subprocess.run(
        ["sh", "-c", GROW_SCRIPT, "grow", "/mnt/x", str(466 * GB), str(10 * GB), "false"],
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"},
    )

    assert (result.stdout.strip(), log.read_text().split()) == (
        f"full {466 * GB}",
        [str(host_file)],
    )


@pytest.mark.parametrize(
    ("alloc_reads_gb", "avail_gb"),
    [((5, 105), 15), ((105, 5), 115)],
    ids=["renter-write-between-reads", "fstrim-between-reads"],
)
def test_bytes_moving_between_the_reads_are_never_counted_twice(tmp_path, alloc_reads_gb, avail_gb):
    # sparse, 110 GB apparent: 100 GB is written (or trimmed) while free disk is read, so there is no room either way
    log = tmp_path / "calls"
    reads = tmp_path / "reads"
    reads.write_text("".join(f"{gb * GB // 512}\n" for gb in alloc_reads_gb))
    _stub(tmp_path, "findmnt", "echo /dev/loop7")
    _stub(
        tmp_path,
        "losetup",
        f'[ "$1" = -c ] && echo "losetup $*" >> {log} || echo /data/volume_live',
    )
    _stub(
        tmp_path,
        "stat",
        f'case "$2" in %s) echo {110 * GB};; %B) echo 512;; '
        f"%b) head -n1 {reads}; tail -n +2 {reads} > {reads}.n; mv {reads}.n {reads};; esac",
    )
    _stub(tmp_path, "df", f'echo h; echo "/dev/x 0 0 {avail_gb * GB} 0% /"')
    _stub(tmp_path, "truncate", f'echo "truncate $*" >> {log}')
    _xfs(tmp_path, log, before=110 * GB, after=110 * GB)

    result = subprocess.run(
        ["sh", "-c", GROW_SCRIPT, "grow", "/mnt/volume_live", str(466 * GB), str(10 * GB), "true"],
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"},
    )

    assert (result.stdout.strip(), log.exists()) == (f"no-room {110 * GB}", False)


@pytest.mark.parametrize(("cur_gb", "outcome"), [(466, "full"), (300, "no-room")])
def test_a_file_extended_by_a_pass_whose_xfs_growfs_failed_is_grown_on_the_next(
    tmp_path, cur_gb, outcome
):
    # the file is already at its size, so the size check alone would stop before the filesystem ever caught up
    log = tmp_path / "calls"
    _stub(tmp_path, "findmnt", "echo /dev/loop7")
    _stub(
        tmp_path,
        "losetup",
        f'[ "$1" = -c ] && echo "losetup $*" >> {log} || echo /data/volume_live',
    )
    _stub(
        tmp_path,
        "stat",
        f'case "$2" in %s) echo {cur_gb * GB};; %b) echo {cur_gb * GB // 512};; %B) echo 512;; esac',
    )
    _stub(tmp_path, "df", 'echo h; echo "/dev/x 0 0 0 0% /"')
    _xfs(tmp_path, log, before=110 * GB, after=cur_gb * GB)

    result = subprocess.run(
        ["sh", "-c", GROW_SCRIPT, "grow", "/mnt/volume_live", str(466 * GB), str(10 * GB), "false"],
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"},
    )

    assert (result.stdout.strip(), log.read_text().splitlines()) == (
        f"{outcome} {cur_gb * GB}",
        ["losetup -c /dev/loop7", "xfs_growfs /mnt/volume_live"],
    )


def test_the_reported_size_is_the_filesystems_not_the_files(tmp_path):
    log = tmp_path / "calls"
    _stub(tmp_path, "findmnt", "echo /dev/loop7")
    _stub(
        tmp_path,
        "losetup",
        f'[ "$1" = -c ] && echo "losetup $*" >> {log} || echo /data/volume_live',
    )
    _stub(
        tmp_path,
        "stat",
        f'case "$2" in %s) echo {110 * GB};; %b) echo {110 * GB // 512};; %B) echo 512;; esac',
    )
    _stub(tmp_path, "df", f'echo h; echo "/dev/x 0 0 {500 * GB} 0% /"')
    _stub(tmp_path, "fallocate", "true")
    # the grow of the file worked, xfs_growfs ran but the filesystem stayed at 110 GB
    _xfs(tmp_path, log, before=110 * GB, after=110 * GB)

    result = subprocess.run(
        ["sh", "-c", GROW_SCRIPT, "grow", "/mnt/volume_live", str(466 * GB), str(10 * GB), "false"],
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"},
    )

    assert result.stdout.strip().splitlines()[-1] == f"grown {110 * GB}"


def test_a_pass_waits_for_another_on_the_same_host_and_never_shrinks_a_file(tmp_path):
    # pass A holds the host lock (it targets 200 while pass B already grew to 400): B gives up, A's truncate
    # can only extend
    lock = tmp_path / "lock"
    log = tmp_path / "calls"
    _stub(tmp_path, "findmnt", "echo /dev/loop7")
    _stub(tmp_path, "losetup", "echo /data/volume_live")
    _stub(tmp_path, "stat", f"echo {110 * GB}")
    _stub(tmp_path, "truncate", f'echo "truncate $*" >> {log}')
    with open(lock, "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        result = subprocess.run(
            ["sh", "-c", GROW_SCRIPT, "grow", "/mnt/x", str(466 * GB), str(10 * GB), "true"],
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "PATH": f"{tmp_path}:{os.environ['PATH']}",
                "VOLUME_GROW_LOCK": str(lock),
                "VOLUME_GROW_LOCK_WAIT": "0",
            },
        )

    assert (result.stdout.strip(), log.exists()) == ("locked", False)
    assert 'truncate -s ">$target"' in GROW_SCRIPT


@pytest.mark.asyncio
async def test_a_stalled_resize_ends_the_pass_at_its_budget_with_the_sizes_it_has():
    async def run(command, **kwargs):
        if command.startswith("timeout "):
            if "volume_stuck" in command:
                await asyncio.Event().wait()
            return _ran(f"grown {466 * GB}\n")
        if "volume ls" in command:
            return _ran("volume_done\nvolume_stuck\n")
        if "volume inspect" in command:
            return _ran("volume_done|466|vloopback:v3|\nvolume_stuck|466|vloopback:v3|\n")
        if "docker info" in command:
            return _ran("/var/lib/docker\n")
        return _ran("abc\n")

    ssh = SimpleNamespace(run=AsyncMock(side_effect=run))

    grown = await asyncio.wait_for(
        grow_short_volumes(ssh, {"done", "stuck"}, budget_seconds=0.2), 5
    )

    assert grown.sizes_gb == {"done": 466}
    # the stalled grow may still allocate on the host: its order stays counted as taken
    assert grown.unreported_bytes == 466 * GB
    assert all("timeout" in call.kwargs for call in ssh.run.await_args_list)


def test_an_unreadable_filesystem_size_reports_no_size_and_grows_nothing(tmp_path):
    # the file reached 466 GB, its xfs_growfs failed, and now xfs_info fails too: never "full 466"
    log = tmp_path / "calls"
    _stub(tmp_path, "findmnt", "echo /dev/loop7")
    _stub(
        tmp_path,
        "losetup",
        f'[ "$1" = -c ] && echo "losetup $*" >> {log} || echo /data/volume_live',
    )
    _stub(tmp_path, "stat", f"echo {466 * GB}")
    _stub(tmp_path, "xfs_info", "exit 1")
    _stub(tmp_path, "xfs_growfs", f'echo "xfs_growfs $*" >> {log}; exit 1')

    result = subprocess.run(
        ["sh", "-c", GROW_SCRIPT, "grow", "/mnt/volume_live", str(466 * GB), str(10 * GB), "false"],
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"},
    )

    assert (result.stdout.strip(), log.exists()) == ("fs-size-unread", False)


@pytest.mark.asyncio
async def test_a_grow_whose_xfs_growfs_failed_reports_the_disk_it_took_and_no_size():
    ssh = SimpleNamespace(
        run=AsyncMock(
            side_effect=[
                _ran("volume_live\n"),
                _ran("volume_live|466|vloopback:v3|\n"),
                _ran("/var/lib/docker\n"),
                _ran("abc\n"),
                _ran(f"allocated {60 * GB}\n", exit_status=1),
            ]
        )
    )

    assert await grow_short_volumes(ssh, {"live"}) == GrowPass(sizes_gb={}, allocated_bytes=60 * GB, attempted=True)


def test_only_pods_listed_at_this_nodes_endpoint_may_grow():
    from protocol.vc_protocol.compute_requests import (
        RentedExecutor,
        RentedExecutorsResponse,
        RentedPod,
    )

    def at(address, port, *pod_ids):
        return RentedExecutor(
            miner_hotkey="h",
            executor_ip_address=address,
            executor_ip_port=port,
            pods=[RentedPod(pod_id=p, container_name=p, rented_ports=[]) for p in pod_ids],
        )

    rented = RentedExecutorsResponse(
        executors={
            "this": at("192.0.2.1", "8080", "here"),
            # the same endpoint under a second executor id
            "alias": at("192.0.2.1", "8080", "alias-pod"),
            # the renter moved on: its pod is listed on another node, the copy here is retained
            "other": at("192.0.2.2", "8080", "moved"),
            "other-port": at("192.0.2.1", "9090", "next-door"),
        }
    )

    assert pod_ids_at_endpoint(rented, "192.0.2.1", 8080) == {"here", "alias-pod"}
    assert pod_ids_at_endpoint(None, "192.0.2.1", 8080) == set()


@pytest.mark.parametrize(
    ("outcome", "taken_gb", "free_after_gb", "listable_gpus"),
    [
        ("no-room", 0, None, None),
        ("partial", 15, 55, 1),
        ("full", 60, 10, 0),
        # the file was extended, xfs_growfs failed: the disk is taken all the same
        ("failed-growfs", 60, 10, 0),
    ],
)
def test_the_disk_a_grow_took_is_what_the_listing_and_paid_count_judge(
    context_factory, outcome, taken_gb, free_after_gb, listable_gpus
):
    from services.task.checks.disk_space import KB_PER_GB, specs_after_disk_taken

    from tests.test_disk_space_check import _split_state

    # 2 GPUs split 1 each, 1 rented, 70 GB free against the 50 GB floor
    state = _split_state(70, gpu_count=2, gpu_splitting_min_count=1, pods=(1,))
    state.specs["hard_disk"]["used"] = 30 * KB_PER_GB
    ctx = context_factory(state=state, rented=True)

    specs = specs_after_disk_taken(ctx, taken_gb * GB)

    if free_after_gb is None:
        assert specs is None
        return
    assert specs["hard_disk"]["free"] == free_after_gb * KB_PER_GB
    assert specs["hard_disk"]["utilization"] == pytest.approx((30 + taken_gb) / 100 * 100)
    assert specs["split_disk"]["listable_gpu_count"] == listable_gpus


def test_a_pass_that_released_more_than_it_took_lists_on_the_disk_now_free(context_factory):
    from services.task.checks.disk_space import KB_PER_GB, specs_after_disk_taken

    from tests.test_disk_space_check import _split_state

    # 40 GB free is under the 50 GB floor; the pass's removals freed 30 GB more than its grow took
    state = _split_state(40, gpu_count=2, gpu_splitting_min_count=1, pods=(1,))
    state.specs["hard_disk"]["used"] = 60 * KB_PER_GB
    ctx = context_factory(state=state, rented=True)

    specs = specs_after_disk_taken(ctx, -30 * GB)

    assert specs["hard_disk"]["free"] == 70 * KB_PER_GB
    assert specs["hard_disk"]["utilization"] == pytest.approx(30)
    assert specs["split_disk"]["listable_gpu_count"] == 1


def test_a_read_after_the_pass_caps_what_the_listing_judges(context_factory):
    from services.task.checks.disk_space import KB_PER_GB, specs_after_disk_taken

    from tests.test_disk_space_check import _split_state

    # scrape: 100 GB free; renters then wrote 30 GB and the grow took 21: 49 GB is left, not 100 - 21 = 79
    state = _split_state(100, gpu_count=2, gpu_splitting_min_count=1, pods=(1,))
    state.specs["hard_disk"]["used"] = 0
    ctx = context_factory(state=state, rented=True)

    specs = specs_after_disk_taken(ctx, 21 * GB, free_now_bytes=49 * GB)

    assert specs["hard_disk"]["free"] == 49 * KB_PER_GB
    assert specs["split_disk"]["listable_gpu_count"] == 0  # under the 50 GB floor


@pytest.mark.parametrize(
    ("taken_gb", "free_now_gb", "credit_gb", "free_published_gb", "listable_gpus"),
    [
        # the grow's take was offset by a release, but the read after the pass is under the floor
        (0, 40, 0, 40, 0),
        # no read after the pass: nothing is known free beyond the filler credit
        (0, 0, 0, 0, 0),
        # the read after is raw; the filler cache the scrape counts free still is
        (21, 49, 30, 79, 1),
        # the read after is higher than the scrape's figure: the cap never raises it
        (0, 120, 30, None, None),
    ],
)
def test_the_read_after_the_pass_caps_on_the_scrapes_footing(
    context_factory, taken_gb, free_now_gb, credit_gb, free_published_gb, listable_gpus
):
    from services.task.checks.disk_space import KB_PER_GB, specs_after_disk_taken

    from tests.test_disk_space_check import _split_state

    state = _split_state(100, gpu_count=2, gpu_splitting_min_count=1, pods=(1,))
    state.specs["hard_disk"]["used"] = 0
    state.specs["hard_disk"]["filler_released"] = credit_gb * KB_PER_GB
    ctx = context_factory(state=state, rented=True)

    specs = specs_after_disk_taken(
        ctx,
        taken_gb * GB,
        free_now_bytes=free_now_gb * GB,
        credit_volumes={"engy_cache_c": credit_gb * KB_PER_GB},
    )

    if free_published_gb is None:
        assert specs is None
        return
    assert specs["hard_disk"]["free"] == free_published_gb * KB_PER_GB
    assert specs["split_disk"]["listable_gpu_count"] == listable_gpus


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("free_reads_gb", "reported_gb", "unreported_gb", "free_after_gb"),
    [
        # the grow reported 60 GB but the disk around it went down 30 (the Dolphin reclaim gave some back): measured
        ((70, 40), 60, 0, 40),
        # the grow's report was lost after it allocated, the read after it saw the disk go
        ((70, 10), 0, 0, 10),
        # no read after the grow: what was written since the scrape is unknown, so no free disk is published
        ((None, None), 60, 0, 0),
        ((None, None), 0, 60, 0),
        ((70, None), 60, 0, 0),
        # measured, but a grow with no report may still be allocating after the read: its order counts too
        ((70, 50), 0, 40, 10),
    ],
)
async def test_the_cleanup_pass_judges_the_disk_it_measured_around_the_grow(
    monkeypatch, free_reads_gb, reported_gb, unreported_gb, free_after_gb
):
    from dataclasses import replace

    from services.task.checks import stale_container_cleanup as module
    from services.task.checks.disk_space import KB_PER_GB

    from tests.test_container_cleanup import _cleanup_check_ctx

    reads = [None if gb is None else gb * GB for gb in free_reads_gb]
    monkeypatch.setattr(module, "pod_ids_at_endpoint", lambda *_: {"live"})
    monkeypatch.setattr(module, "host_free_bytes", AsyncMock(side_effect=reads))
    monkeypatch.setattr(
        module,
        "grow_short_volumes",
        AsyncMock(
            return_value=GrowPass(
                allocated_bytes=reported_gb * GB, attempted=True, unreported_bytes=unreported_gb * GB
            )
        ),
    )
    ctx = _cleanup_check_ctx([], [])
    disk = {"hard_disk": {"free": 70 * KB_PER_GB, "used": 30 * KB_PER_GB, "utilization": 30}}
    ctx = ctx.model_copy(update={"state": replace(ctx.state, specs=disk)})

    result = await module.StaleContainerCleanupCheck().run(ctx)

    assert result.updates["state"].specs["hard_disk"]["free"] == free_after_gb * KB_PER_GB


@pytest.mark.asyncio
async def test_with_no_task_time_left_the_pass_neither_measures_nor_grows(monkeypatch):
    from services.task.checks import stale_container_cleanup as module

    from tests.test_container_cleanup import _cleanup_check_ctx

    ssh = SimpleNamespace(run=AsyncMock())
    grow = AsyncMock(return_value=GrowPass())
    monkeypatch.setattr(module, "pod_ids_at_endpoint", lambda *_: {"live"})
    monkeypatch.setattr(module, "task_seconds_left", lambda _ctx: 0)
    monkeypatch.setattr(module, "grow_short_volumes", grow)

    await module.StaleContainerCleanupCheck().run(_cleanup_check_ctx([], []))

    grow.assert_not_called()
    assert await host_free_bytes(ssh, 0) is None
    ssh.run.assert_not_called()


@pytest.mark.asyncio
async def test_the_disk_is_read_after_the_removals_so_a_freed_filler_volume_is_not_counted_twice(monkeypatch):
    # a removal can free a filler run volume the scrape already lists as free: it must fall outside the window
    from services.task.checks import stale_container_cleanup as module

    from tests.test_container_cleanup import _cleanup_check_ctx

    order = []
    ctx = _cleanup_check_ctx([], [])
    cleanup = ctx.services.container_cleanup
    removal = cleanup.cleanup.side_effect

    async def remove(**kwargs):
        order.append("remove")
        return await removal(**kwargs)

    async def orphans(**_kwargs):
        order.append("orphans")
        return 0

    async def read(*_args):
        order.append("read")
        return 70 * GB

    async def grow(*_args):
        order.append("grow")
        return GrowPass(attempted=True)

    cleanup.cleanup.side_effect = remove
    cleanup.remove_orphan_rental_volumes.side_effect = orphans
    monkeypatch.setattr(module, "pod_ids_at_endpoint", lambda *_: {"live"})
    monkeypatch.setattr(module, "host_free_bytes", read)
    monkeypatch.setattr(module, "grow_short_volumes", grow)

    await module.StaleContainerCleanupCheck().run(ctx)

    assert order == ["remove", "orphans", "read", "grow", "read"]


@pytest.mark.asyncio
async def test_host_free_bytes_reads_the_hosts_docker_root_through_pid_1():
    # the session's own `/` is the executor container's overlay; the scrape reads /proc/1/root<DockerRootDir>
    ssh = SimpleNamespace(run=AsyncMock(return_value=SimpleNamespace(exit_status=0, stdout="123\n", stderr="")))

    assert await host_free_bytes(ssh) == 123
    command = ssh.run.await_args.args[0]
    assert "h=/proc/1/root$r" in command and 'df -P -B1 "$h"' in command


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("credit_volumes", "removed_orphans", "free_published_gb"),
    [
        # a customer create evicted the Engy cache since the scrape: its space is in the raw read already
        ({}, 0, 10),
        # the sweep removed an orphan, but the cache the credit stands for is still on the host, as large
        ({"engy_cache_x": 200 * GB // 1024}, 1, 150),
        # the cache is still there but emptied since the scrape: its space is in the raw read already
        ({"engy_cache_x": 0}, 0, 10),
        # half emptied: only what it holds now is credited
        ({"engy_cache_x": 100 * GB // 1024}, 0, 110),
        # a preallocated filler run volume keeps its size while it exists: the scrape's figure stands
        ({"engy_cache_x": None}, 0, 150),
        # the inventory could not be read: no credit, whatever the pass removed
        (None, 0, 10),
    ],
    ids=["evicted", "kept", "emptied", "half-emptied", "preallocated", "unread"],
)
async def test_the_cap_credits_only_the_cache_volumes_still_on_the_host(
    monkeypatch, credit_volumes, removed_orphans, free_published_gb
):
    from dataclasses import replace

    from services.task.checks import stale_container_cleanup as module
    from services.task.checks.disk_space import KB_PER_GB

    from tests.test_container_cleanup import _cleanup_check_ctx

    monkeypatch.setattr(module, "pod_ids_at_endpoint", lambda *_: {"live"})
    monkeypatch.setattr(module, "host_free_bytes", AsyncMock(side_effect=[160 * GB, 10 * GB]))
    monkeypatch.setattr(module, "credit_volumes_on_host", AsyncMock(return_value=credit_volumes))
    monkeypatch.setattr(
        module, "grow_short_volumes", AsyncMock(return_value=GrowPass(allocated_bytes=150 * GB, attempted=True))
    )
    ctx = _cleanup_check_ctx([], [])
    ctx.services.container_cleanup.remove_orphan_rental_volumes.return_value = removed_orphans
    disk = {
        "hard_disk": {
            "free": 300 * KB_PER_GB,
            "used": 0,
            "utilization": 0,
            "filler_released": 200 * KB_PER_GB,
            "by_volume": {"engy_cache_x": 200 * KB_PER_GB},
        }
    }
    ctx = ctx.model_copy(update={"state": replace(ctx.state, specs=disk)})

    result = await module.StaleContainerCleanupCheck().run(ctx)

    assert result.updates["state"].specs["hard_disk"]["free"] == free_published_gb * KB_PER_GB


@pytest.mark.asyncio
async def test_credit_volumes_on_host_sizes_each_engy_cache_now():
    # filler run volumes are preallocated (their scraped size stands); an Engy cache is sized as it is now
    ssh = MagicMock()
    stdout = "F filler_vol\nE 0 engy_cache_a\nE 42 engy_cache_b\n"
    ssh.run = AsyncMock(return_value=MagicMock(exit_status=0, stdout=stdout))

    expected = {"filler_vol": None, "engy_cache_a": 0, "engy_cache_b": 42}
    assert await credit_volumes_on_host(ssh) == expected
    command = ssh.run.await_args.args[0]
    assert "du -sk" in command and "/proc/1/root" in command


@pytest.mark.asyncio
async def test_credit_volumes_on_host_credits_only_names_the_scrape_counts():
    # docker's name filter matches anywhere in the name; the scrape credits only names that start with the prefix
    ssh = MagicMock()
    stdout = "E 7 engy_cache_a\nE 9 backup_engy_cache_x\n"
    ssh.run = AsyncMock(return_value=MagicMock(exit_status=0, stdout=stdout))

    assert await credit_volumes_on_host(ssh) == {"engy_cache_a": 7}


@pytest.mark.asyncio
@pytest.mark.parametrize("stdout", ["E x engy_cache_a\n", "E 5\n", "? engy_cache_a\n"])
async def test_credit_volumes_on_host_is_unread_on_a_malformed_line(stdout):
    ssh = MagicMock()
    ssh.run = AsyncMock(return_value=MagicMock(exit_status=0, stdout=stdout))

    assert await credit_volumes_on_host(ssh) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(("removed_orphans", "free_published_gb"), [(1, 10), (0, 10)])
async def test_a_pass_that_removed_volumes_caps_on_the_raw_read_without_the_scrapes_credit(
    monkeypatch, removed_orphans, free_published_gb
):
    # scrape: 100 GB raw + a 200 GB orphan filler volume = 300 free; renters wrote 140, the sweep removed the orphan
    # (free_before 160), the grow took 150 (free_after 10). The removed volume's credit is already in the raw read.
    from dataclasses import replace

    from services.task.checks import stale_container_cleanup as module
    from services.task.checks.disk_space import KB_PER_GB

    from tests.test_container_cleanup import _cleanup_check_ctx

    monkeypatch.setattr(module, "pod_ids_at_endpoint", lambda *_: {"live"})
    monkeypatch.setattr(module, "host_free_bytes", AsyncMock(side_effect=[160 * GB, 10 * GB]))
    # the inventory of credit volumes is unread: no credit either way
    monkeypatch.setattr(module, "credit_volumes_on_host", AsyncMock(return_value=None))
    monkeypatch.setattr(
        module, "grow_short_volumes", AsyncMock(return_value=GrowPass(allocated_bytes=150 * GB, attempted=True))
    )
    ctx = _cleanup_check_ctx([], [])
    ctx.services.container_cleanup.remove_orphan_rental_volumes.return_value = removed_orphans
    disk = {"hard_disk": {"free": 300 * KB_PER_GB, "used": 0, "utilization": 0, "filler_released": 200 * KB_PER_GB}}
    ctx = ctx.model_copy(update={"state": replace(ctx.state, specs=disk)})

    result = await module.StaleContainerCleanupCheck().run(ctx)

    assert result.updates["state"].specs["hard_disk"]["free"] == free_published_gb * KB_PER_GB
