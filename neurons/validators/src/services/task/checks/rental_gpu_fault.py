from __future__ import annotations

import json
import logging
import shlex
from datetime import UTC, datetime
from typing import Any

from services.gpu_xid_attribution import (
    ATTRIBUTION_HARDWARE,
    ATTRIBUTION_WORKLOAD,
    CONTAINER_PIDS_COMMAND,
    CONTAINER_STARTED_AT_COMMAND,
    ECC_QUERY_COMMAND,
    HOST_COMMAND_TIMEOUT_SECONDS,
    XID_LOG_COMMAND,
    XidAttribution,
    attribute,
    dmesg_unavailable,
    parse_container_pids,
    parse_docker_started_at,
    parse_ecc_uncorrected,
    parse_xid_lines,
)

from core.config import settings
from core.utils import _m, get_extra_info

from ..messages import RentalGpuFaultMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context

logger = logging.getLogger(__name__)

PHASE_MID_RENTAL = "mid_rental"
# redis hash: pod_id -> how many workload Xid lines the backend was told about, so a rental that logged its
# fault once is not reported again every cycle; a new line re-reports.
REPORTED_WORKLOAD_LINES_KEY = "rental_gpu_fault_reported"


class RentalGpuFaultCheck:
    """DAH-3490: read the host's kernel log for every rented pod and say who broke the card.

    Runs on rented nodes only, right after the spec scrape and before every fatal GPU check, so a card that stopped
    being listed mid-rental is attributed in the same cycle that halts on it; never fatal, never changes the score.
    The Xid lines inside [container start, now] are split by `services.gpu_xid_attribution` into the renter's
    application errors (Xid 13/31/43/45 whose PID is one of the container's) and the provider's hardware
    faults (every other Xid, or uncorrected ECC). A workload verdict is posted to the backend as its own
    request (`POST /internal/executors/{uuid}/gpu-fault-probe`, phase mid_rental), which tells the renter
    and the provider. A hardware verdict is logged and left to the existing checks.
    """

    check_id = "gpu.validate.rental_fault"
    fatal = False

    async def run(self, ctx: Context) -> CheckResult:
        if not settings.RENTAL_GPU_FAULT_PROBE_ENABLED:
            return CheckResult(passed=True, event=render_message(Msg.DISABLED, ctx=ctx, check_id=self.check_id))

        rented_data = ctx.state.rented_data
        rented_executor = rented_data.executors.get(ctx.executor.uuid) if rented_data else None
        pods = [pod for pod in (rented_executor.pods if rented_executor else []) if pod.container_name]
        if not pods:
            return CheckResult(passed=True, event=render_message(Msg.NOT_RENTED, ctx=ctx, check_id=self.check_id))
        try:
            return await self._attribute(ctx, pods)
        except Exception as exc:
            # host text is never trusted to parse, and this check must never cost a rented node its cycle
            logger.warning(
                _m("Rental GPU-fault attribution failed", extra=get_extra_info({**ctx.default_extra, "error": repr(exc)})),
                exc_info=True,
            )
            what = {"executor_uuid": ctx.executor.uuid, "error": repr(exc)[:300]}
            return CheckResult(passed=True, event=render_message(Msg.PROBE_ERROR, ctx=ctx, check_id=self.check_id, what=what))

    async def _attribute(self, ctx: Context, pods) -> CheckResult:
        xid_log = await ctx.runner.run(XID_LOG_COMMAND, timeout=HOST_COMMAND_TIMEOUT_SECONDS, retryable=False)
        if (xid_log.exit_code != 0 and not xid_log.stdout) or dmesg_unavailable(xid_log.stdout):
            what = {
                "executor_uuid": ctx.executor.uuid,
                "error": xid_log.error_message or xid_log.stderr[-300:] or "the host's kernel log is not readable (dmesg)",
            }
            return CheckResult(
                passed=True, event=render_message(Msg.PROBE_ERROR, ctx=ctx, check_id=self.check_id, what=what)
            )
        lines = parse_xid_lines(xid_log.stdout)
        ecc_query = await ctx.runner.run(ECC_QUERY_COMMAND, timeout=HOST_COMMAND_TIMEOUT_SECONDS, retryable=False)
        ecc_uncorrected = parse_ecc_uncorrected(ecc_query.stdout) if ecc_query.exit_code == 0 else {}

        now = datetime.now(UTC)
        verdicts: dict[str, XidAttribution] = {}
        reported: list[str] = []
        for pod in pods:
            # two host reads per pod, one after the other (the runner is one SSH session)
            started_at_result = await ctx.runner.run(
                CONTAINER_STARTED_AT_COMMAND.format(name=shlex.quote(pod.container_name)),
                timeout=HOST_COMMAND_TIMEOUT_SECONDS,
                retryable=False,
            )
            pids_result = await ctx.runner.run(
                CONTAINER_PIDS_COMMAND.format(name=shlex.quote(pod.container_name)),
                timeout=HOST_COMMAND_TIMEOUT_SECONDS,
                retryable=False,
            )
            started_at = parse_docker_started_at(started_at_result.stdout) if started_at_result.exit_code == 0 else None
            container_pids = parse_container_pids(pids_result.stdout) if pids_result.exit_code == 0 else None
            verdict = attribute(
                lines,
                window_start=started_at,
                window_end=now,
                container_pids=container_pids,
                ecc_uncorrected=ecc_uncorrected,
            )
            verdicts[pod.pod_id] = verdict
            if verdict.attribution == ATTRIBUTION_WORKLOAD and await self._is_new(ctx, pod.pod_id, verdict):
                report = {
                    "pod_id": pod.pod_id,
                    "phase": PHASE_MID_RENTAL,
                    "probed_at": now.isoformat(),
                    "container_started_at": started_at.isoformat() if started_at else None,
                    **verdict.as_report(),
                }
                await ctx.services.backend.report_gpu_fault_probe(ctx.executor.uuid, report)
                await self._remember(ctx, pod.pod_id, verdict)
                reported.append(pod.pod_id)

        what: dict[str, Any] = {
            "executor_uuid": ctx.executor.uuid,
            "pods": {pod_id: verdict.as_report() for pod_id, verdict in verdicts.items()},
            "reported_pod_ids": reported,
        }
        attributions = {verdict.attribution for verdict in verdicts.values()}
        if ATTRIBUTION_HARDWARE in attributions:
            template = Msg.HARDWARE_FAULT
        elif ATTRIBUTION_WORKLOAD in attributions:
            template = Msg.WORKLOAD_FAULT
        else:
            template = Msg.NO_FAULT
        return CheckResult(passed=True, event=render_message(template, ctx=ctx, check_id=self.check_id, what=what))

    @staticmethod
    async def _is_new(ctx: Context, pod_id: str, verdict: XidAttribution) -> bool:
        redis = ctx.services.redis
        if redis is None:
            return True
        try:
            known = await redis.hget(REPORTED_WORKLOAD_LINES_KEY, pod_id)
        except Exception:
            logger.warning(
                _m("Could not read the reported GPU-fault lines", extra=get_extra_info({**ctx.default_extra, "pod_id": pod_id})),
                exc_info=True,
            )
            return True
        if not known:
            return True
        try:
            return len(verdict.workload) > int(json.loads(known))
        except (ValueError, TypeError):
            return True

    @staticmethod
    async def _remember(ctx: Context, pod_id: str, verdict: XidAttribution) -> None:
        redis = ctx.services.redis
        if redis is None:
            return
        try:
            await redis.hset(REPORTED_WORKLOAD_LINES_KEY, pod_id, json.dumps(len(verdict.workload)))
        except Exception:
            logger.warning(
                _m("Could not record the reported GPU-fault lines", extra=get_extra_info({**ctx.default_extra, "pod_id": pod_id})),
                exc_info=True,
            )
