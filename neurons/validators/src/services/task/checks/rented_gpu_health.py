"""RENTED_GPU_HEALTH (shadow): read per-card GPU health on a rented node without GPU work.

TenantEnforcementCheck halts a rented node's cycle before every GPU test, so a card that degrades during a
rental is only seen once it drops out of NVML. This check runs the fault probe in `--health` mode (NVML
reads and the kernel log only: no CUDA context, no kernels, nothing inside the renter's container) and
logs one `rented_gpu_health_shadow` line with a verdict per card. It never changes the score and never
fails the cycle.
"""

from __future__ import annotations

import logging
import shlex
from typing import Any

from services.redis_service import GPU_ANCHOR_KEY

from core.config import settings
from core.utils import _m, get_extra_info

from ..messages import RentedGpuHealthMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context
from .gpu_fault_probe import MAX_NVML_GPUS, PROBE_SOURCE, _cap, _parse_report
from .gpu_fingerprint import split_uuids
from .rented_gpu_drop import POD_STATUS_RUNNING, _rented_gpu_count

logger = logging.getLogger(__name__)

SHADOW_LOG_KEY = "rented_gpu_health_shadow"
# NVML init, the per-card reads (10 s fork deadline in the probe) and one dmesg read (5 s)
HEALTH_TIMEOUT_SECONDS = 30
MAX_ERROR_CHARS = 200

CARD_OK = "ok"
CARD_MISSING = "missing"
REASON_NVML_ERROR = "nvml_error"
REASON_READ_ERROR = "read_error"
REASON_ECC_UNCORRECTED = "ecc_uncorrected"
REASON_RETIRED_PAGES_PENDING = "retired_pages_pending"
REASON_REMAP_PENDING = "remapped_rows_pending"
REASON_REMAP_FAILED = "remapped_rows_failed"
REASON_RECOVERY_ACTION = "recovery_action"


def judge_card(gpu: dict[str, Any]) -> list[str]:
    """The fault reasons one card's health read shows; empty when the card is healthy."""
    if gpu.get("error"):
        return [REASON_NVML_ERROR]
    reasons: list[str] = []
    if gpu.get("read_errors"):
        reasons.append(REASON_READ_ERROR)
    ecc = gpu.get("ecc_uncorrected")
    if isinstance(ecc, int) and not isinstance(ecc, bool) and ecc > 0:
        reasons.append(REASON_ECC_UNCORRECTED)
    if gpu.get("retired_pages_pending") is True:
        reasons.append(REASON_RETIRED_PAGES_PENDING)
    rows = gpu.get("remapped_rows")
    # (corrected, uncorrected, isPending, failureOccurred)
    if isinstance(rows, list) and len(rows) >= 4:
        if rows[2]:
            reasons.append(REASON_REMAP_PENDING)
        if rows[3]:
            reasons.append(REASON_REMAP_FAILED)
    if gpu.get("recovery_action"):
        reasons.append(REASON_RECOVERY_ACTION)
    xids = gpu.get("hardware_xids")
    if isinstance(xids, list):
        reasons.extend(f"xid_{xid}" for xid in xids if isinstance(xid, int))
    return reasons


def judge_health(nvml: dict[str, Any], expected: int) -> dict[str, Any]:
    """Per-card verdicts and the node verdict from the probe's `nvml` block."""
    raw = nvml.get("gpus") if isinstance(nvml.get("gpus"), list) else []
    cards: list[dict[str, Any]] = []
    for gpu in raw[:MAX_NVML_GPUS]:
        if not isinstance(gpu, dict):
            continue
        reasons = judge_card(gpu)
        card: dict[str, Any] = {
            "index": _cap(gpu.get("index")),
            "uuid": _cap(gpu.get("uuid")),
            "verdict": "fault" if reasons else CARD_OK,
        }
        if reasons:
            card["reasons"] = reasons
            if gpu.get("error"):
                card["error"] = str(gpu["error"])[:MAX_ERROR_CHARS]
            if gpu.get("read_errors"):
                card["read_errors"] = _cap(gpu["read_errors"])
        cards.append(card)
    count = nvml.get("count")
    visible = count if isinstance(count, int) and not isinstance(count, bool) else len(cards)
    missing = max(expected - visible, 0)
    cards.extend({"index": None, "verdict": CARD_MISSING} for _ in range(min(missing, MAX_NVML_GPUS)))
    faulty = [card for card in cards if card["verdict"] != CARD_OK]
    return {
        "verdict": "fault" if faulty else CARD_OK,
        "expected_gpu_count": expected,
        "nvml_gpu_count": visible,
        "faulty_card_count": len(faulty),
        "cards": cards,
    }


class RentedGpuHealthShadowCheck:
    """Shadow per-card health read on a rented node; non-fatal, never touches the score."""

    check_id = "gpu.validate.rented_gpu_health"
    fatal = False

    async def run(self, ctx: Context) -> CheckResult:
        if not settings.RENTED_GPU_HEALTH_SHADOW_ENABLED:
            return CheckResult(
                passed=True, event=render_message(Msg.DISABLED, ctx=ctx, check_id=self.check_id)
            )

        rented_data = ctx.state.rented_data
        rented_executor = rented_data.executors.get(ctx.executor.uuid) if rented_data else None
        # the executor UUID is miner-reported: a rental held under another hotkey is not this node's
        if rented_executor is not None and rented_executor.miner_hotkey != ctx.miner_hotkey:
            rented_executor = None
        all_pods = list(rented_executor.pods) if rented_executor else []
        if not any(pod.status in (None, POD_STATUS_RUNNING) for pod in all_pods):
            return CheckResult(
                passed=True, event=render_message(Msg.NOT_RENTED, ctx=ctx, check_id=self.check_id)
            )

        anchor = split_uuids((ctx.verified or {}).get(GPU_ANCHOR_KEY) or "")
        expected = max(_rented_gpu_count(all_pods) or 1, len(anchor))
        what: dict[str, Any] = {"executor_uuid": ctx.executor.uuid, "shadow": True}

        command = f"{shlex.quote(ctx.executor.python_path)} -I - --health"
        run = await ctx.runner.run(
            command, timeout=HEALTH_TIMEOUT_SECONDS, retryable=False, stdin_text=PROBE_SOURCE
        )
        report = _parse_report(run.stdout)
        nvml = report.get("nvml") if report else None
        if not isinstance(nvml, dict) or not nvml.get("available"):
            what.update(
                verdict="unknown",
                expected_gpu_count=expected,
                error=(
                    str(nvml.get("error"))[:MAX_ERROR_CHARS]
                    if isinstance(nvml, dict) and nvml.get("error")
                    else run.error_message or "no health report in output"
                ),
            )
            self._log(ctx, what, logging.INFO)
            return CheckResult(
                passed=True,
                event=render_message(Msg.SHADOW_UNKNOWN, ctx=ctx, check_id=self.check_id, what=what),
            )

        what.update(judge_health(nvml, expected))
        xid = report.get("xid") if isinstance(report.get("xid"), dict) else {}
        what["xid_readable"] = bool(xid.get("available"))
        if what["verdict"] == CARD_OK:
            self._log(ctx, what, logging.INFO)
            return CheckResult(
                passed=True,
                event=render_message(Msg.SHADOW_OK, ctx=ctx, check_id=self.check_id, what=what),
            )
        self._log(ctx, what, logging.WARNING)
        return CheckResult(
            passed=True,
            event=render_message(Msg.SHADOW_FAULT, ctx=ctx, check_id=self.check_id, what=what),
        )

    @staticmethod
    def _log(ctx: Context, what: dict[str, Any], level: int) -> None:
        logger.log(level, _m(SHADOW_LOG_KEY, extra=get_extra_info({**ctx.default_extra, **what})))
