import asyncio
import fcntl
import os
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from services.volume_grow import (
    GrowPass,
    pod_ids_at_endpoint,
    GROW_HEADROOM_BYTES,
    GROW_SCRIPT,
    ORDER_LABEL,
    grow_short_volumes,
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

    assert await grow_short_volumes(ssh, {"live"}) == GrowPass(sizes_gb={}, allocated_bytes=60 * GB)


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
