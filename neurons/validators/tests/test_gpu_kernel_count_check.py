"""GpuKernelCountCheck: the reported GPU list against the kernel's list.

The reported count and UUIDs come from the scrape. /proc/driver/nvidia/gpus is the kernel's own
inventory. A kernel list that is present and shorter than the reported GPU count is a mismatch:
logged every cycle and failing the validation.
A missing kernel list is never a failure.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from core.config import settings
from services import nvidia_devices
from services.nvidia_devices import (
    HOST_KERNEL_GPU_VIEW_CMD,
    HOST_PID1_ROOT,
    KERNEL_GPU_FS_SEPARATOR,
    KERNEL_GPU_VIEW_CMD,
    KERNEL_GPU_VIEW_SEPARATOR,
)

from neurons.validators.src.services.task.checks.gpu_kernel_count import (
    GpuKernelCountCheck,
    evaluate_kernel_view,
)
from neurons.validators.src.services.task.messages import GpuKernelCountMessages as Msg
from neurons.validators.src.services.task.pipeline import Context, Pipeline
from protocol.vc_protocol.compute_requests import (
    RentedExecutor,
    RentedExecutorsResponse,
    RentedPod,
)

from tests.helpers import build_state

REAL = [f"GPU-0000000{i}-5281-aabc-1a90-c91764f90d1{i}" for i in range(8)]
# Further reported cards; most tests leave them off the kernel list.
OTHER_CARDS = [f"GPU-1111111{i}-d04f-4862-{v}c7d-95b6f13e8c4{i}" for i, v in enumerate("89ab89a")]


class _Sink:
    async def emit(self, event) -> None:
        pass


def _host_stdout(uuids, *, mounts=(), fstype="proc", kind="regular empty file"):
    """What HOST_KERNEL_GPU_VIEW_CMD prints: PID 1's mounts, the stat rows, then the UUID rows."""
    files = [
        f"{HOST_PID1_ROOT}/proc/driver/nvidia/gpus/0000:00:{i + 7:02x}.0/information"
        for i in range(len(uuids))
    ]
    return "\n".join(
        [
            *mounts,
            KERNEL_GPU_FS_SEPARATOR,
            *(f"{path} {fstype}" for path in files),
            *(f"{path}|{kind}" for path in files),
            KERNEL_GPU_VIEW_SEPARATOR,
            *(f"{gpu_uuid}, {i}" for i, gpu_uuid in enumerate(uuids)),
        ]
    )


NO_HOST_VIEW = _host_stdout([])


def _ctx(
    context_factory,
    *,
    reported,
    kernel,
    read_attempted=True,
    count=None,
    host=NO_HOST_VIEW,
):
    """`host` is the stdout of the host-view read, or an exception that read raises."""
    specs = {"gpu": {"count": len(reported) if count is None else count}}

    async def run(command, *args, **kwargs):
        if command != HOST_KERNEL_GPU_VIEW_CMD:
            raise AssertionError("the kernel list is read once per cycle")
        if isinstance(host, BaseException):
            raise host
        return SimpleNamespace(exit_status=0, stdout=host, stderr="")

    ssh = MagicMock()
    ssh.run = AsyncMock(side_effect=run)
    return context_factory(
        state=build_state(
            specs=specs,
            gpu_count=len(reported) if count is None else count,
            gpu_uuids=",".join(reported),
            kernel_gpu_uuids=kernel,
            kernel_gpu_uuids_read_attempted=read_attempted,
        ),
        ssh=ssh,
    )


def _with_live_rental(ctx: Context) -> Context:
    rented = RentedExecutorsResponse(
        executors={
            ctx.executor.uuid: RentedExecutor(
                miner_hotkey="miner-hotkey",
                executor_ip_address=ctx.executor.address,
                executor_ip_port=str(ctx.executor.port),
                pods=[RentedPod(pod_id="pod-1", container_name="pod_p1")],
            )
        }
    )
    return ctx.model_copy(update={"state": replace(ctx.state, rented_data=rented)})


@pytest.fixture
def check_enabled(monkeypatch):
    monkeypatch.setattr(settings, "GPU_KERNEL_COUNT_CHECK_ENABLED", True)


def test_the_check_is_on_by_default():
    assert type(settings).model_fields["GPU_KERNEL_COUNT_CHECK_ENABLED"].default is True


@pytest.mark.asyncio
async def test_kernel_one_reported_eight_fails_the_validation(
    context_factory, check_enabled
):
    ctx = _ctx(context_factory, reported=[REAL[0], *OTHER_CARDS], kernel=[REAL[0]])

    result = await GpuKernelCountCheck().run(ctx)
    ok, events, _ = await Pipeline([GpuKernelCountCheck()], _Sink()).run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.COUNT_MISMATCH.reason
    assert result.event.severity == "error"
    assert result.event.what_we_saw["would_fail"] is True
    assert ok is False
    assert events[-1].what_we_saw["steps_failed"] == GpuKernelCountCheck.check_id


@pytest.mark.asyncio
async def test_mismatch_on_a_rented_node_fails_but_does_not_stop_the_run(
    context_factory, check_enabled
):
    # same rule as the ban checks: a node with a renter's pod on it keeps being verified
    ctx = _with_live_rental(
        _ctx(context_factory, reported=[REAL[0], *OTHER_CARDS], kernel=[REAL[0]])
    )

    result = await GpuKernelCountCheck().run(ctx)
    ok, _, _ = await Pipeline([GpuKernelCountCheck()], _Sink()).run(ctx)

    assert result.passed is False
    assert result.fatal is False
    assert result.updates == {"gpu_kernel_count_passed": False}
    assert result.event.reason_code == Msg.COUNT_MISMATCH.reason
    assert ok is True


@pytest.mark.asyncio
async def test_match_on_a_rented_node_passes_and_keeps_the_score_flag(
    context_factory, check_enabled
):
    ctx = _with_live_rental(_ctx(context_factory, reported=REAL, kernel=REAL))

    result = await GpuKernelCountCheck().run(ctx)

    assert result.passed is True
    assert "gpu_kernel_count_passed" not in result.updates
    assert result.event.reason_code == Msg.COUNT_OK.reason


@pytest.mark.asyncio
async def test_not_evaluated_on_a_rented_node_passes_and_keeps_the_score_flag(
    context_factory, check_enabled
):
    ctx = _with_live_rental(_ctx(context_factory, reported=REAL, kernel=None))

    result = await GpuKernelCountCheck().run(ctx)

    assert result.passed is True
    assert "gpu_kernel_count_passed" not in result.updates
    assert result.event.reason_code == Msg.NOT_EVALUATED.reason


@pytest.mark.asyncio
async def test_mismatch_on_an_idle_node_is_fatal(context_factory, check_enabled):
    ctx = _ctx(context_factory, reported=[REAL[0], *OTHER_CARDS], kernel=[REAL[0]])

    result = await GpuKernelCountCheck().run(ctx)

    assert result.passed is False
    assert result.fatal is True


@pytest.mark.asyncio
async def test_reported_count_above_the_listed_cards_is_judged_on_the_count(
    context_factory, check_enabled
):
    # the node claims 8 and lists one card: the claim is what is compared
    ctx = _ctx(context_factory, reported=[REAL[0]], kernel=[REAL[0]], count=8)

    result = await GpuKernelCountCheck().run(ctx)

    assert result.event.reason_code == Msg.COUNT_MISMATCH.reason
    assert result.event.what_we_saw["reported_gpu_count"] == 8


# ---------------------------- container given a subset of the host's cards ----------------------------

CONTAINER_MOUNTS_OF_ONE_CARD = (
    "504 494 0:63 / /proc/driver/nvidia rw,nosuid,nodev,noexec,relatime - tmpfs tmpfs rw,mode=555",
    "546 504 0:23 /driver/nvidia/gpus/0000:00:07.0 /proc/driver/nvidia/gpus/0000:00:07.0 "
    "ro,nosuid,nodev,noexec,relatime master:12 - proc proc rw",
)


def test_container_read_is_the_same_command_and_the_host_read_goes_through_pid_1():
    assert "/proc/1/" not in KERNEL_GPU_VIEW_CMD
    assert "/proc/self/mountinfo" in KERNEL_GPU_VIEW_CMD
    assert HOST_KERNEL_GPU_VIEW_CMD == KERNEL_GPU_VIEW_CMD.replace(
        "/proc/self/mountinfo", "/proc/1/mountinfo"
    ).replace("/proc/driver/nvidia/gpus/*", "/proc/1/root/proc/driver/nvidia/gpus/*")


@pytest.mark.asyncio
async def test_container_one_host_four_reported_four_matches_on_the_host_view(
    context_factory, check_enabled
):
    ctx = _ctx(context_factory, reported=REAL[:4], kernel=[REAL[0]], host=_host_stdout(REAL[:4]))

    result = await GpuKernelCountCheck().run(ctx)
    ok, events, _ = await Pipeline([GpuKernelCountCheck()], _Sink()).run(ctx)

    assert result.passed is True
    assert ok is True
    assert result.event.reason_code == Msg.COUNT_OK.reason
    assert result.event.severity == "info"
    what = result.event.what_we_saw
    assert what["kernel_verdict"] == "match_host_view"
    assert what["kernel_view_source"] == "host_pid1_root"
    assert what["container_kernel_gpu_count"] == 1
    assert what["host_kernel_gpu_count"] == 4
    assert what["reported_uuids_not_in_host_kernel"] == []
    assert what["would_fail"] is False
    assert [call.args[0] for call in ctx.ssh.run.await_args_list] == [HOST_KERNEL_GPU_VIEW_CMD] * 2


@pytest.mark.asyncio
async def test_host_listing_more_cards_than_reported_matches_on_the_host_view(
    context_factory, check_enabled
):
    ctx = _ctx(context_factory, reported=REAL[:4], kernel=[REAL[0]], host=_host_stdout(REAL))

    result = await GpuKernelCountCheck().run(ctx)

    assert result.passed is True
    assert result.event.what_we_saw["kernel_verdict"] == "match_host_view"
    assert result.event.what_we_saw["host_kernel_gpu_count"] == 8


@pytest.mark.asyncio
async def test_container_one_host_one_reported_eight_still_fails(
    context_factory, check_enabled
):
    ctx = _ctx(
        context_factory,
        reported=[REAL[0], *OTHER_CARDS],
        kernel=[REAL[0]],
        host=_host_stdout([REAL[0]]),
    )

    result = await GpuKernelCountCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.COUNT_MISMATCH.reason
    what = result.event.what_we_saw
    assert what["kernel_verdict"] == "count_lower"
    assert what["kernel_view_source"] == "container"
    assert what["container_kernel_gpu_count"] == 1
    assert what["host_kernel_gpu_count"] == 1
    assert ctx.ssh.run.await_count == 1


@pytest.mark.asyncio
async def test_host_with_fewer_cards_than_reported_still_fails(
    context_factory, check_enabled
):
    ctx = _ctx(context_factory, reported=REAL, kernel=[REAL[0]], host=_host_stdout(REAL[:4]))

    result = await GpuKernelCountCheck().run(ctx)

    assert result.passed is False
    assert result.event.what_we_saw["kernel_verdict"] == "count_lower"
    assert result.event.what_we_saw["host_kernel_gpu_count"] == 4


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "host",
    [
        # PID 1 is the container's own init: its mount table is the container's again
        _host_stdout(REAL[:4], mounts=CONTAINER_MOUNTS_OF_ONE_CARD),
        # a tmpfs laid over the gpus path on the host
        _host_stdout(
            REAL[:4], mounts=("90 25 0:70 / /proc/driver/nvidia/gpus rw - tmpfs tmpfs rw",)
        ),
        # an overlay one level up, with nothing but the stat rows to show it
        _host_stdout(REAL[:4], mounts=("90 25 0:70 / /proc/driver/nvidia rw - overlay o rw",)),
        _host_stdout(REAL[:4], fstype="tmpfs"),
        _host_stdout(REAL[:4], kind="symbolic link"),
    ],
)
async def test_host_view_behind_a_foreign_mount_is_ignored(context_factory, check_enabled, host):
    ctx = _ctx(context_factory, reported=REAL[:4], kernel=[REAL[0]], host=host)

    result = await GpuKernelCountCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.COUNT_MISMATCH.reason
    what = result.event.what_we_saw
    assert what["kernel_verdict"] == "count_lower"
    assert what["kernel_view_source"] == "container"
    assert what["host_kernel_gpu_count"] is None
    assert what["host_kernel_gpu_foreign_mounts"]


@pytest.mark.asyncio
async def test_host_uuid_rows_without_procfs_evidence_are_ignored(context_factory, check_enabled):
    uuid_rows_only = "\n".join(
        [
            KERNEL_GPU_FS_SEPARATOR,
            KERNEL_GPU_VIEW_SEPARATOR,
            *(f"{u}, {i}" for i, u in enumerate(REAL[:4])),
        ]
    )
    ctx = _ctx(context_factory, reported=REAL[:4], kernel=[REAL[0]], host=uuid_rows_only)

    result = await GpuKernelCountCheck().run(ctx)

    assert result.passed is False
    assert result.event.what_we_saw["kernel_verdict"] == "count_lower"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "host", [OSError("connection lost"), NO_HOST_VIEW, "", "not the expected output"]
)
async def test_host_read_error_or_empty_host_view_is_a_mismatch_and_never_raises(
    context_factory, check_enabled, host
):
    ctx = _ctx(context_factory, reported=REAL[:4], kernel=[REAL[0]], host=host)

    result = await GpuKernelCountCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.COUNT_MISMATCH.reason
    assert result.event.what_we_saw["kernel_verdict"] == "count_lower"
    assert result.event.what_we_saw["host_kernel_gpu_count"] is None


@pytest.mark.asyncio
async def test_host_read_non_zero_exit_is_a_mismatch(context_factory, check_enabled):
    ctx = _ctx(context_factory, reported=REAL[:4], kernel=[REAL[0]])
    ctx.ssh.run = AsyncMock(
        return_value=SimpleNamespace(exit_status=1, stdout=_host_stdout(REAL[:4]), stderr="")
    )

    result = await GpuKernelCountCheck().run(ctx)

    assert result.passed is False
    assert result.event.what_we_saw["kernel_verdict"] == "count_lower"


@pytest.mark.asyncio
async def test_host_read_timeout_is_a_mismatch_and_never_raises(
    context_factory, check_enabled, monkeypatch
):
    monkeypatch.setattr(nvidia_devices, "KERNEL_GPU_UUID_READ_TIMEOUT_SECONDS", 0.01)
    ctx = _ctx(context_factory, reported=REAL[:4], kernel=[REAL[0]])

    async def hang(command, *args, **kwargs):
        await asyncio.sleep(30)

    ctx.ssh.run = AsyncMock(side_effect=hang)

    result = await GpuKernelCountCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.COUNT_MISMATCH.reason
    assert result.event.what_we_saw["kernel_verdict"] == "count_lower"


@pytest.mark.asyncio
async def test_no_ssh_connection_leaves_the_mismatch_as_it_was(context_factory, check_enabled):
    with_ssh = _ctx(context_factory, reported=REAL[:4], kernel=[REAL[0]])
    ctx = context_factory(state=with_ssh.state, ssh=None)
    assert ctx.ssh is None

    result = await GpuKernelCountCheck().run(ctx)

    assert result.passed is False
    assert result.event.what_we_saw["kernel_verdict"] == "count_lower"


@pytest.mark.asyncio
async def test_host_has_the_count_but_not_a_reported_uuid_is_not_a_match(
    context_factory, check_enabled
):
    reported = [*REAL[:3], OTHER_CARDS[0]]
    ctx = _ctx(context_factory, reported=reported, kernel=[REAL[0]], host=_host_stdout(REAL[:4]))

    result = await GpuKernelCountCheck().run(ctx)

    # as for a container that has every card: the count holds, the UUID difference is logged
    assert result.passed is True
    assert result.event.reason_code == Msg.UUID_MISMATCH.reason
    what = result.event.what_we_saw
    assert what["kernel_verdict"] == "uuid_mismatch"
    assert what["kernel_view_source"] == "host_pid1_root"
    assert what["reported_uuids_not_in_host_kernel"] == [OTHER_CARDS[0].lower()]
    assert what["would_fail"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reported", "kernel"),
    [(REAL[:4], REAL[:4]), (REAL[:2], REAL[:4]), ([*REAL[:3], OTHER_CARDS[0]], REAL[:4])],
)
async def test_host_view_is_not_read_when_the_container_list_is_long_enough(
    context_factory, check_enabled, reported, kernel
):
    ctx = _ctx(context_factory, reported=reported, kernel=kernel, host=_host_stdout(REAL))

    result = await GpuKernelCountCheck().run(ctx)

    assert result.passed is True
    assert result.event.what_we_saw["kernel_view_source"] == "container"
    assert "host_kernel_gpu_count" not in result.event.what_we_saw
    assert ctx.ssh.run.await_count == 0


@pytest.mark.asyncio
async def test_host_view_is_not_read_when_the_container_list_is_missing(context_factory, check_enabled):
    ctx = _ctx(context_factory, reported=REAL[:4], kernel=None, host=_host_stdout(REAL))

    result = await GpuKernelCountCheck().run(ctx)

    assert result.event.reason_code == Msg.NOT_EVALUATED.reason
    assert ctx.ssh.run.await_count == 0


# ---------------------------- kernel 8 vs reported 8 ----------------------------


@pytest.mark.asyncio
async def test_kernel_eight_reported_eight_passes(context_factory, check_enabled):
    # the kernel prints the same UUIDs; letter case is not a difference
    ctx = _ctx(context_factory, reported=REAL, kernel=[u.upper() for u in reversed(REAL)])

    result = await GpuKernelCountCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.COUNT_OK.reason
    what = result.event.what_we_saw
    assert what["kernel_verdict"] == "match"
    assert what["would_fail"] is False
    assert what["reported_uuids_not_in_kernel"] == []


@pytest.mark.asyncio
async def test_kernel_listing_more_cards_than_reported_passes(context_factory, check_enabled):
    ctx = _ctx(context_factory, reported=REAL[:2], kernel=REAL[:4])

    result = await GpuKernelCountCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.COUNT_OK.reason
    assert result.event.what_we_saw["kernel_uuids_not_reported"] == [u.lower() for u in REAL[2:4]]


# ---------------------------- kernel list missing ----------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("kernel", [None, []])
async def test_missing_kernel_list_is_not_evaluated_and_never_fails(
    context_factory, check_enabled, kernel
):
    ctx = _ctx(context_factory, reported=REAL, kernel=kernel)

    result = await GpuKernelCountCheck().run(ctx)
    ok, _, _ = await Pipeline([GpuKernelCountCheck()], _Sink()).run(ctx)

    assert result.passed is True
    assert ok is True
    assert result.event.reason_code == Msg.NOT_EVALUATED.reason
    assert result.event.severity == "info"
    what = result.event.what_we_saw
    assert what["kernel_verdict"] == "not_evaluated"
    assert what["not_evaluated_reason"] == "kernel_list_unavailable"
    assert what["kernel_gpu_count"] is None
    assert what["would_fail"] is False


@pytest.mark.asyncio
async def test_kernel_list_withheld_behind_a_foreign_mount_is_not_evaluated(
    context_factory, check_enabled
):
    ctx = _ctx(context_factory, reported=REAL, kernel=None)
    mounts = ["/proc/driver/nvidia/gpus tmpfs"]
    ctx = ctx.model_copy(update={"state": replace(ctx.state, kernel_gpu_foreign_mounts=mounts)})

    result = await GpuKernelCountCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.NOT_EVALUATED.reason
    assert result.event.what_we_saw["not_evaluated_reason"] == "kernel_list_behind_foreign_mount"
    assert result.event.what_we_saw["kernel_gpu_foreign_mounts"] == mounts


@pytest.mark.asyncio
async def test_failed_kernel_read_is_not_evaluated(context_factory, check_enabled):
    # BannedProviderCheck did not run: the check reads the list itself, and the read fails
    ctx = _ctx(context_factory, reported=REAL, kernel=None, read_attempted=False)
    ctx.ssh.run = AsyncMock(side_effect=OSError("connection lost"))

    result = await GpuKernelCountCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.NOT_EVALUATED.reason


@pytest.mark.asyncio
async def test_malformed_specs_never_fail_the_check(context_factory, check_enabled):
    ctx = _ctx(context_factory, reported=REAL, kernel=REAL)
    specs = {"gpu": "x"}
    state = replace(ctx.state, specs=specs)

    result = await GpuKernelCountCheck().run(ctx.model_copy(update={"state": state}))

    assert result.passed is True
    assert result.event.reason_code == Msg.COUNT_OK.reason


@pytest.mark.asyncio
async def test_disabled_check_reads_nothing(context_factory, monkeypatch):
    monkeypatch.setattr(settings, "GPU_KERNEL_COUNT_CHECK_ENABLED", False)
    ctx = _ctx(context_factory, reported=[REAL[0], *OTHER_CARDS], kernel=[REAL[0]])

    result = await GpuKernelCountCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.SKIPPED.reason


# ---------------------------- reported UUID absent from the kernel list ----------------------------


@pytest.mark.asyncio
async def test_reported_uuid_absent_from_kernel_list_is_flagged_not_failed(
    context_factory, check_enabled
):
    reported = [*REAL[:7], "GPU-99999999-5281-aabc-1a90-c91764f90d19"]
    ctx = _ctx(context_factory, reported=reported, kernel=REAL)

    result = await GpuKernelCountCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.UUID_MISMATCH.reason
    assert result.event.severity == "warning"
    what = result.event.what_we_saw
    assert what["kernel_verdict"] == "uuid_mismatch"
    assert what["reported_uuids_not_in_kernel"] == ["gpu-99999999-5281-aabc-1a90-c91764f90d19"]
    assert what["kernel_uuids_not_reported"] == [REAL[7].lower()]
    assert what["would_fail"] is False


def test_evaluate_kernel_view_verdicts():
    assert (
        evaluate_kernel_view(
            reported_count=8, reported_uuids=REAL, kernel_uuids=REAL[:1]
        ).kernel_verdict
        == "count_lower"
    )
    assert (
        evaluate_kernel_view(
            reported_count=8, reported_uuids=REAL, kernel_uuids=REAL
        ).kernel_verdict
        == "match"
    )
    assert (
        evaluate_kernel_view(
            reported_count=8, reported_uuids=REAL, kernel_uuids=None
        ).kernel_verdict
        == "not_evaluated"
    )
    assert (
        evaluate_kernel_view(
            reported_count=0, reported_uuids=[], kernel_uuids=REAL[:1]
        ).kernel_verdict
        == "match"
    )


# ---------------------------- wiring ----------------------------


def test_check_is_in_every_pipeline_after_the_kernel_read():
    from neurons.validators.src.services.task.pipeline_factory import PipelineFactory

    def flat(checks):
        out = []
        for chk in checks:
            lanes = getattr(chk, "lanes", None)
            out.extend(c.check_id for lane in lanes for c in lane) if lanes else out.append(
                chk.check_id
            )
        return out

    for ids in (
        flat(PipelineFactory.build_checks()),
        flat(PipelineFactory.build_fast_path_checks()),
        flat(PipelineFactory.build_dry_run_checks()),
    ):
        assert ids.count(GpuKernelCountCheck.check_id) == 1
        assert ids.index("gpu.validate.banned_provider") < ids.index(GpuKernelCountCheck.check_id)
