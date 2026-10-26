from __future__ import annotations

import json
import logging
import shlex
from enum import Enum
from pathlib import Path
from typing import Any

from services.const import DEFAULT_JOB_OWNER_LIUM
from services.redis_service import NVLINK_CHECKED_PREFIX, NVLINK_FAULT_PREFIX

from core.config import settings
from core.utils import _m, get_extra_info

from ..messages import NvlinkFaultMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context
from .capability import get_filler_only_container
from .sshd_bootstrap import remove_lium_fillers

logger = logging.getLogger(__name__)

# Stdlib-only, shipped over stdin like the pinned-memory probe: nothing to upload or update.
PROBE_SOURCE_PATH = Path(__file__).resolve().parents[3] / "miner_jobs" / "nvlink_multicast_probe.py"
PROBE_SOURCE = PROBE_SOURCE_PATH.read_text()
PROBE_JSON_MARKER = "NVLINK_MULTICAST_PROBE_JSON:"
# above the probe's own 45 s budget; only catches a stalled SSH channel (scored UNKNOWN)
PROBE_TIMEOUT_SECONDS = 75
MAX_CHARS = 300
# a node that runs our filler loses it at most once a day to a probe
CHECKED_TTL_SECONDS = 24 * 3600
FAULT_TTL_SECONDS = 3 * 24 * 3600


class RecordedVerdict(Enum):
    FAULT = "fault"
    CHECKED = "checked"
    NONE = "none"
    UNREADABLE = "unreadable"


def fault_key(miner_hotkey: str, executor_uuid: str) -> str:
    return f"{NVLINK_FAULT_PREFIX}:{miner_hotkey}:{executor_uuid}"


def checked_key(miner_hotkey: str, executor_uuid: str) -> str:
    return f"{NVLINK_CHECKED_PREFIX}:{miner_hotkey}:{executor_uuid}"


class NvlinkFaultCheck:
    """Fail an all-NVLink node whose GPUs cannot bind NVLink multicast memory (sticky NVLink fault).

    An 8x B300 (23 Oct) took an uncorrectable NVLink error; afterwards every link still read active and
    every single-GPU probe passed, but NCCL could not bind NVLS multicast memory on any GPU, so it was
    listed and unusable for multi-GPU work. The probe (miner_jobs/nvlink_multicast_probe.py) does that
    bind. Only boards the scrape reports as NVLink between every GPU pair are probed; a GPU without
    multicast support, a driver without the API, or any error other than the NVLink-fault codes passes
    with NVLINK_UNKNOWN. Runs after the rented halt, so a rented node is never probed or failed here.

    A filler does not hide a fault: with no probe in the last CHECKED_TTL_SECONDS on record, the check
    removes the node's Lium fillers and probes. A fault on record fails a node whose filler it cannot remove.
    """

    check_id = "gpu.validate.nvlink_fault"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        interconnect = (ctx.state.specs or {}).get("interconnect")
        if not (isinstance(interconnect, dict) and interconnect.get("nvlink") is True):
            return CheckResult(
                passed=True, event=render_message(Msg.SKIPPED, ctx=ctx, check_id=self.check_id)
            )

        what: dict[str, Any] = {}
        filler_container = get_filler_only_container(ctx)
        if filler_container:
            what["filler_container"] = filler_container
            recorded = await _read_verdict(ctx)
            # Redis down: keep the filler; the next readable cycle decides
            if recorded in (RecordedVerdict.CHECKED, RecordedVerdict.UNREADABLE):
                return self._filler_skipped(ctx, what)
            what["removed_fillers"] = await _remove_fillers(ctx)
            if not what["removed_fillers"]:
                if recorded is RecordedVerdict.FAULT:
                    what["remembered_fault"] = True
                    return CheckResult(
                        passed=False,
                        event=render_message(
                            Msg.PROBE_FAILED, ctx=ctx, check_id=self.check_id, what=what
                        ),
                    )
                return self._filler_skipped(ctx, what)

        return await self._probe(ctx, what)

    def _filler_skipped(self, ctx: Context, what: dict[str, Any]) -> CheckResult:
        return CheckResult(
            passed=True,
            event=render_message(Msg.FILLER_SKIPPED, ctx=ctx, check_id=self.check_id, what=what),
        )

    async def _probe(self, ctx: Context, what: dict[str, Any]) -> CheckResult:
        run = await ctx.runner.run(
            f"{shlex.quote(ctx.executor.python_path)} -I -",
            timeout=PROBE_TIMEOUT_SECONDS,
            retryable=False,
            stdin_text=PROBE_SOURCE,
        )
        report = _parse_report(run.stdout)
        what.update(duration_ms=run.duration_ms, exit_code=run.exit_code)
        if report is None:
            what["error"] = (run.error_message or "no probe report in output")[:MAX_CHARS]
            what["stderr_tail"] = run.stderr[-MAX_CHARS:]
            await _record_verdict(ctx, None)
            return CheckResult(
                passed=True,
                event=render_message(Msg.UNKNOWN, ctx=ctx, check_id=self.check_id, what=what),
            )
        # the report is executor-controlled: copy only bounded scalars
        what["probe"] = {
            key: (value[:MAX_CHARS] if isinstance(value, str) else value)
            for key, value in report.items()
            if key in ("status", "error", "gpus", "size_mb")
            and isinstance(value, (str, int, float))
        }
        status = report.get("status")
        await _record_verdict(ctx, status)
        if status == "fault":
            return CheckResult(
                passed=False,
                event=render_message(Msg.PROBE_FAILED, ctx=ctx, check_id=self.check_id, what=what),
            )
        template = Msg.PROBE_OK if status == "ok" else Msg.UNKNOWN
        return CheckResult(
            passed=True, event=render_message(template, ctx=ctx, check_id=self.check_id, what=what)
        )


async def _read_verdict(ctx: Context) -> RecordedVerdict:
    keys = [
        fault_key(ctx.miner_hotkey, ctx.executor.uuid),
        checked_key(ctx.miner_hotkey, ctx.executor.uuid),
    ]
    try:
        fault, checked = await ctx.services.redis.mget(keys)
    except Exception:  # noqa: BLE001
        return RecordedVerdict.UNREADABLE
    if fault is not None:
        return RecordedVerdict.FAULT
    return RecordedVerdict.NONE if checked is None else RecordedVerdict.CHECKED


async def _record_verdict(ctx: Context, status: str | None) -> None:
    """Only a pass clears a fault: a host can make the probe report nothing at will. Never raises."""
    if settings.DRY_RUN:
        return
    faulted = fault_key(ctx.miner_hotkey, ctx.executor.uuid)
    checked = checked_key(ctx.miner_hotkey, ctx.executor.uuid)
    try:
        if status == "fault":
            await ctx.services.redis.set(faulted, "1", ex=FAULT_TTL_SECONDS)
            await ctx.services.redis.delete(checked)
            return
        await ctx.services.redis.set(checked, "1", ex=CHECKED_TTL_SECONDS)
        if status == "ok":
            await ctx.services.redis.delete(faulted)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            _m(
                "NVLINK_VERDICT_WRITE_FAILED",
                extra=get_extra_info({**ctx.default_extra, "error": str(exc)}),
            )
        )


async def _remove_fillers(ctx: Context) -> list[str]:
    rented_data = ctx.state.rented_data
    if (
        settings.DRY_RUN
        or rented_data.get_default_job_owner(ctx.executor.uuid) != DEFAULT_JOB_OWNER_LIUM
    ):
        return []
    return await remove_lium_fillers(ctx, rented_data.get_filler_containers(ctx.executor.uuid))


def _parse_report(stdout: str) -> dict[str, Any] | None:
    # the last marker line wins; stdout is executor-controlled
    for line in reversed(stdout.splitlines()):
        if line.startswith(PROBE_JSON_MARKER):
            try:
                parsed = json.loads(line[len(PROBE_JSON_MARKER) :])
            except (ValueError, TypeError):
                return None
            return parsed if isinstance(parsed, dict) else None
    return None
