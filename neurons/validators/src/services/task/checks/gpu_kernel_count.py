from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from core.config import settings
from services.nvidia_devices import read_kernel_gpu_view

from ..messages import GpuKernelCountMessages as Msg
from ..messages import MessageTemplate, render_message
from ..models import ValidationEvent
from ..pipeline import CheckResult, Context
from .banned_provider import has_live_rental, kernel_gpu_view, reported_gpu_uuids

KERNEL_VIEW_SOURCE_CONTAINER = "container"
KERNEL_VIEW_SOURCE_HOST = "host_pid1_root"


def _normalized(uuids: list[str] | None) -> list[str]:
    seen: dict[str, None] = {}
    for gpu_uuid in uuids or []:
        key = str(gpu_uuid).strip().lower()
        if key:
            seen.setdefault(key)
    return list(seen)


def _as_count(value: Any) -> int | None:
    try:
        count = int(value)
    except (TypeError, ValueError):
        return None
    return count if count >= 0 else None


def reported_gpu_count(ctx: Context, reported_uuids: list[str]) -> int:
    """The GPU count the node claims: the scraped count, or the number of cards it listed if higher."""
    count = _as_count(ctx.state.gpu_count)
    if count is None:
        count = _as_count(((ctx.state.specs or {}).get("gpu") or {}).get("count")) or 0
    return max(count, len(reported_uuids))


@dataclass(frozen=True)
class KernelViewResult:
    """The reported cards compared with one kernel list. `kernel_verdict` is one of
    not_evaluated, match, count_lower, uuid_mismatch."""

    reported_gpu_count: int
    reported_uuid_count: int
    kernel_gpu_count: int | None
    reported_uuids_not_in_kernel: list[str]
    kernel_uuids_not_reported: list[str]
    kernel_verdict: str

    def as_fields(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_kernel_view(
    *, reported_count: int, reported_uuids: list[str], kernel_uuids: list[str] | None
) -> KernelViewResult:
    reported = _normalized(reported_uuids)
    kernel = _normalized(kernel_uuids)
    if not kernel:
        return KernelViewResult(
            reported_gpu_count=reported_count,
            reported_uuid_count=len(reported),
            kernel_gpu_count=None,
            reported_uuids_not_in_kernel=[],
            kernel_uuids_not_reported=[],
            kernel_verdict="not_evaluated",
        )
    kernel_set, reported_set = set(kernel), set(reported)
    not_in_kernel = [u for u in reported if u not in kernel_set]
    if len(kernel) < reported_count:
        verdict = "count_lower"
    elif not_in_kernel:
        verdict = "uuid_mismatch"
    else:
        verdict = "match"
    return KernelViewResult(
        reported_gpu_count=reported_count,
        reported_uuid_count=len(reported),
        kernel_gpu_count=len(kernel),
        reported_uuids_not_in_kernel=not_in_kernel,
        kernel_uuids_not_reported=[u for u in kernel if u not in reported_set],
        kernel_verdict=verdict,
    )


@dataclass(frozen=True)
class HostViewResult:
    """What the host-view reading saw. `kernel_verdict` is set only when the host's list
    replaces the container's verdict (match_host_view or uuid_mismatch); otherwise the first
    verdict stands."""

    host_kernel_gpu_count: int | None = None
    host_kernel_gpu_foreign_mounts: list[str] = field(default_factory=list)
    host_kernel_view_error: str | None = None
    kernel_verdict: str | None = None
    host_kernel_gpu_uuids: list[str] | None = None
    reported_uuids_not_in_host_kernel: list[str] | None = None

    def as_fields(self) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "host_kernel_gpu_count": self.host_kernel_gpu_count,
            "host_kernel_gpu_foreign_mounts": self.host_kernel_gpu_foreign_mounts,
        }
        if self.host_kernel_view_error is not None:
            fields["host_kernel_view_error"] = self.host_kernel_view_error
        if self.kernel_verdict is not None:
            fields.update(
                {
                    "kernel_verdict": self.kernel_verdict,
                    "kernel_view_source": KERNEL_VIEW_SOURCE_HOST,
                    "host_kernel_gpu_uuids": self.host_kernel_gpu_uuids,
                    "reported_uuids_not_in_host_kernel": self.reported_uuids_not_in_host_kernel,
                }
            )
        return fields


async def host_view_verdict(
    ctx: Context, *, reported_count: int, reported_uuids: list[str]
) -> HostViewResult:
    """Second reading, taken only when the container's list is shorter than the reported count.

    A container that was handed a subset of the host's cards has only those under its own /proc,
    while the scrape can still list every card. The host's list, read through PID 1's root, is
    the full one. It is used only when it is the host's real procfs (read_kernel_gpu_view's
    rules) and lists at least the reported number of cards. A host that itself has fewer cards
    than reported is short there too, and the first verdict stands. Never raises.
    """
    if ctx.ssh is None:
        return HostViewResult()
    try:
        host_uuids, host_foreign_mounts = await read_kernel_gpu_view(ctx.ssh, host_view=True)
    except Exception as exc:
        return HostViewResult(host_kernel_view_error=repr(exc)[:300])
    if not host_uuids:
        return HostViewResult(host_kernel_gpu_foreign_mounts=host_foreign_mounts)
    judged = evaluate_kernel_view(
        reported_count=reported_count, reported_uuids=reported_uuids, kernel_uuids=host_uuids
    )
    if judged.kernel_verdict not in ("match", "uuid_mismatch"):
        return HostViewResult(
            host_kernel_gpu_count=judged.kernel_gpu_count,
            host_kernel_gpu_foreign_mounts=host_foreign_mounts,
        )
    return HostViewResult(
        host_kernel_gpu_count=judged.kernel_gpu_count,
        host_kernel_gpu_foreign_mounts=host_foreign_mounts,
        kernel_verdict="match_host_view" if judged.kernel_verdict == "match" else "uuid_mismatch",
        host_kernel_gpu_uuids=host_uuids,
        reported_uuids_not_in_host_kernel=judged.reported_uuids_not_in_kernel,
    )


class GpuKernelCountCheck:
    """Cross-check the reported GPU list against the kernel's view of the cards.

    The reported count and UUIDs come from the scrape (NVML); the kernel list is
    /proc/driver/nvidia/gpus as BannedProviderCheck reads it once per cycle. A node whose kernel
    list is present and shorter than its reported GPU count gets GPU_KERNEL_COUNT_MISMATCH. That
    verdict fails the validation and zeroes the score. As with the ban checks, a failure stops
    the run only when the node has no live rental.

    The container the check reads in may have been handed only some of the host's cards. So a
    short list is checked once more against the host's own list (host_view_verdict): if that one
    covers the reported cards the verdict is match_host_view and nothing fails.

    A kernel list that is missing (unreadable, empty, read failed, withheld behind a foreign
    mount) is never a failure: the verdict is GPU_KERNEL_COUNT_NOT_EVALUATED.
    """

    check_id = "gpu.validate.kernel_count"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        if not settings.GPU_KERNEL_COUNT_CHECK_ENABLED:
            event = render_message(Msg.SKIPPED, ctx=ctx, check_id=self.check_id)
            return CheckResult(passed=True, event=event)
        try:
            return await self._evaluate(ctx)
        except Exception as exc:  # an evaluation error must not fail the node
            event = render_message(
                Msg.NOT_EVALUATED,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "kernel_verdict": "not_evaluated",
                    "not_evaluated_reason": "evaluation_error",
                    "error": repr(exc)[:300],
                },
            )
            return CheckResult(passed=True, event=event)

    async def _evaluate(self, ctx: Context) -> CheckResult:
        reported_uuids = reported_gpu_uuids(ctx)
        reported_count = reported_gpu_count(ctx, reported_uuids)
        kernel_uuids, foreign_mounts = await kernel_gpu_view(ctx)

        kernel_check_fields = evaluate_kernel_view(
            reported_count=reported_count,
            reported_uuids=reported_uuids,
            kernel_uuids=kernel_uuids,
        ).as_fields()
        kernel_check_fields["kernel_view_source"] = KERNEL_VIEW_SOURCE_CONTAINER
        if kernel_check_fields["kernel_verdict"] == "count_lower":
            kernel_check_fields["container_kernel_gpu_count"] = kernel_check_fields[
                "kernel_gpu_count"
            ]
            host_view = await host_view_verdict(
                ctx, reported_count=reported_count, reported_uuids=reported_uuids
            )
            kernel_check_fields.update(host_view.as_fields())
        verdict = kernel_check_fields["kernel_verdict"]
        if verdict == "not_evaluated":
            kernel_check_fields["not_evaluated_reason"] = (
                "kernel_list_behind_foreign_mount" if foreign_mounts else "kernel_list_unavailable"
            )
        would_fail = verdict == "count_lower"
        kernel_check_fields.update(
            {
                "reported_uuids": reported_uuids,
                "kernel_gpu_uuids": kernel_uuids,
                "kernel_gpu_foreign_mounts": foreign_mounts,
                "would_fail": would_fail,
            }
        )

        event = self._render_event(ctx, kernel_check_fields)
        if would_fail:
            # A node with a renter's pod on it keeps being verified, so the failure stops the
            # run only when there is no live rental; the flag zeroes the score in either case.
            return CheckResult(
                passed=False,
                event=event,
                fatal=not has_live_rental(ctx),
                updates={"gpu_kernel_count_passed": False},
            )
        return CheckResult(passed=True, event=event)

    def _render_event(self, ctx: Context, kernel_check_fields: dict[str, Any]) -> ValidationEvent:
        """The event for a finished evaluation."""
        template = (
            Msg.COUNT_MISMATCH
            if kernel_check_fields["would_fail"]
            else self._template_for(kernel_check_fields)
        )
        return render_message(template, ctx=ctx, check_id=self.check_id, what=kernel_check_fields)

    @staticmethod
    def _template_for(kernel_check_fields: dict[str, Any]) -> MessageTemplate:
        verdict = kernel_check_fields["kernel_verdict"]
        if verdict == "uuid_mismatch":
            return Msg.UUID_MISMATCH
        if verdict == "not_evaluated":
            return Msg.NOT_EVALUATED
        return Msg.COUNT_OK
