from __future__ import annotations

import json
import logging
import shlex
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from core.config import settings
from core.utils import _m, get_extra_info

from ...const import DEFAULT_JOB_OWNER_LIUM
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
# a fault stays on record while a filler blocks the probe; the next probe that runs replaces it
FAULT_KEY_PREFIX = "nvlink_fault"
FAULT_KEY_TTL_SECONDS = 7 * 24 * 3600


@dataclass(frozen=True)
class FaultRecord:
    """The bounded probe fields kept in Redis while a filler blocks the probe."""

    status: str | None = None
    error: str | None = None
    gpus: int | None = None
    size_mb: int | float | None = None


class NvlinkFaultCheck:
    """Fail an all-NVLink node whose GPUs cannot bind NVLink multicast memory (sticky NVLink fault).

    An 8x B300 (23 Oct) took an uncorrectable NVLink error; afterwards every link still read active and
    every single-GPU probe passed, but NCCL could not bind NVLS multicast memory on any GPU, so it was
    listed and unusable for multi-GPU work. The probe (miner_jobs/nvlink_multicast_probe.py) does that
    bind. Only boards the scrape reports as NVLink between every GPU pair are probed; a GPU without
    multicast support, a driver without the API, or any error other than the NVLink-fault codes passes
    with NVLINK_UNKNOWN. Runs after the rented halt, so a rented node is never probed or failed here.
    A fault fails the node only with NVLINK_FAULT_ENFORCEMENT_ENABLED; off, it is logged and the node passes.

    A filler holds every GPU, so the probe is skipped while one runs. A fault seen before the filler started
    is kept in Redis and still reported on those cycles: on 24 Oct an 8x board faulted at 18:52Z, a filler
    started, and every later cycle skipped the probe and passed the faulted node. When enforced, our own
    (Lium) fillers on a recorded-faulty node are removed and the probe runs, so a repaired node recovers.
    """

    check_id = "gpu.validate.nvlink_fault"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        interconnect = (ctx.state.specs or {}).get("interconnect")
        if not (isinstance(interconnect, dict) and interconnect.get("nvlink") is True):
            return CheckResult(
                passed=True, event=render_message(Msg.SKIPPED, ctx=ctx, check_id=self.check_id)
            )
        # the probe opens a context on every GPU beside the filler's own NCCL, like the fault probe it skips
        filler_container = get_filler_only_container(ctx)
        if filler_container:
            last_fault = await self._last_fault(ctx)
            if last_fault is not None and not await self._removed_our_fillers(ctx):
                return self._fault(
                    ctx, {"filler_container": filler_container, "last_probe": asdict(last_fault)}
                )
            if last_fault is None:
                return CheckResult(
                    passed=True,
                    event=render_message(
                        Msg.FILLER_SKIPPED,
                        ctx=ctx,
                        check_id=self.check_id,
                        what={"filler_container": filler_container},
                    ),
                )

        run = await ctx.runner.run(
            f"{shlex.quote(ctx.executor.python_path)} -I -",
            timeout=PROBE_TIMEOUT_SECONDS,
            retryable=False,
            stdin_text=PROBE_SOURCE,
        )
        report = _parse_report(run.stdout)
        what: dict[str, Any] = {"duration_ms": run.duration_ms, "exit_code": run.exit_code}
        if report is None:
            what["error"] = (run.error_message or "no probe report in output")[:MAX_CHARS]
            what["stderr_tail"] = run.stderr[-MAX_CHARS:]
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
        if status == "fault":
            await self._save_fault_record(ctx, FaultRecord(**what["probe"]))
            return self._fault(ctx, what)
        if status == "ok":
            await self._clear_fault_record(ctx)
        template = Msg.PROBE_OK if status == "ok" else Msg.UNKNOWN
        return CheckResult(
            passed=True, event=render_message(template, ctx=ctx, check_id=self.check_id, what=what)
        )

    def _fault(self, ctx: Context, what: dict[str, Any]) -> CheckResult:
        enforce = settings.NVLINK_FAULT_ENFORCEMENT_ENABLED
        event = render_message(
            Msg.PROBE_FAILED,
            ctx=ctx,
            check_id=self.check_id,
            severity=None if enforce else "warning",
            impact=None if enforce else "Shadow observation only: score was NOT changed",
            what=what,
        )
        return CheckResult(passed=not enforce, event=event)

    @staticmethod
    async def _removed_our_fillers(ctx: Context) -> bool:
        """Remove our own fillers so the probe can re-run; True only if every filler is gone."""
        rented_data = ctx.state.rented_data
        if (
            settings.DRY_RUN
            or not settings.NVLINK_FAULT_ENFORCEMENT_ENABLED
            or rented_data.get_default_job_owner(ctx.executor.uuid) != DEFAULT_JOB_OWNER_LIUM
        ):
            return False
        fillers = rented_data.get_filler_containers(ctx.executor.uuid)
        return set(await remove_lium_fillers(ctx, fillers)) == set(fillers)

    async def _last_fault(self, ctx: Context) -> FaultRecord | None:
        try:
            raw = await ctx.services.redis.get(f"{FAULT_KEY_PREFIX}:{ctx.executor.uuid}")
            if raw is None:
                return None
            data = json.loads(raw)
            return FaultRecord(**{f.name: data.get(f.name) for f in fields(FaultRecord)})
        except Exception:
            # an unreadable record keeps the plain filler skip, as before the record existed
            logger.warning(_m("NVLink fault record unreadable", extra=get_extra_info(ctx.default_extra)))
            return None

    async def _save_fault_record(self, ctx: Context, record: FaultRecord) -> None:
        try:
            await ctx.services.redis.set(
                f"{FAULT_KEY_PREFIX}:{ctx.executor.uuid}",
                json.dumps(asdict(record)),
                ex=FAULT_KEY_TTL_SECONDS,
            )
        except Exception:
            logger.warning(_m("NVLink fault record not written", extra=get_extra_info(ctx.default_extra)))

    async def _clear_fault_record(self, ctx: Context) -> None:
        try:
            await ctx.services.redis.delete(f"{FAULT_KEY_PREFIX}:{ctx.executor.uuid}")
        except Exception:
            logger.warning(_m("NVLink fault record not cleared", extra=get_extra_info(ctx.default_extra)))


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
