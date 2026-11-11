import os
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from services.volume_grow import (
    GROW_HEADROOM_BYTES,
    GROW_SCRIPT,
    ORDER_LABEL,
    grow_short_volumes,
    order_labels,
)

GB = 1024**3


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
    assert (grown, "volume_retained" in inspect) == ({"live": 466}, False)
    assert grow.startswith("nsenter -t 1 -m -- sh -c ")
    # the host path of the mount, not the plugin-internal /mnt/<volume> docker reports
    assert grow.endswith(
        f" grow /var/lib/docker/plugins/abc/propagated-mount/volume_live {466 * GB} {GROW_HEADROOM_BYTES} false"
    )


@pytest.mark.asyncio
async def test_a_failed_listing_grows_nothing_and_does_not_raise():
    ssh = SimpleNamespace(run=AsyncMock(side_effect=OSError("channel closed")))

    assert await grow_short_volumes(ssh, {"live"}) == {}


def _stub(bin_dir, name, body):
    path = bin_dir / name
    path.write_text("#!/bin/sh\n" + body + "\n")
    path.chmod(0o755)


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
        (110, 5, 500, "true", f"grown {466 * GB}", f"truncate -s {466 * GB} /data/volume_live"),
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
    for tool in ("fallocate", "truncate", "xfs_growfs"):
        _stub(tmp_path, tool, f'echo "{tool} $*" >> {log}')

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
    assert (result.stdout.strip(), calls) == (outcome, expected)


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


def test_the_host_script_reads_the_allocated_size_before_free_disk(tmp_path):
    # a renter write between the reads must lower free disk only, never count the same bytes twice
    log = tmp_path / "calls"
    _stub(tmp_path, "findmnt", "echo /dev/loop7")
    _stub(tmp_path, "losetup", "echo /data/volume_live")
    _stub(
        tmp_path, "stat", f'echo "stat $2" >> {log}; case "$2" in %b) echo 1;; *) echo 512;; esac'
    )
    _stub(tmp_path, "df", f'echo df >> {log}; echo "h"; echo "/dev/x 0 0 0 0% /"')
    _stub(tmp_path, "xfs_growfs", "true")

    subprocess.run(
        ["sh", "-c", GROW_SCRIPT, "grow", "/mnt/x", str(466 * GB), str(10 * GB), "true"],
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"},
    )

    assert log.read_text().split("\n").index("df") > log.read_text().split("\n").index("stat %B")
