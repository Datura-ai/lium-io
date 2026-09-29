"""liumd M2 step 1 (DAH-2834): the shadow comparison of `liumd run` with today's verify.

`TaskService.create_task` calls `run_liumd_shadow` once per node, after the pipeline has run and
the node's `JobResult` is built, on the SSH session the pipeline used. With
`VALIDATOR_LIUMD_SHADOW` off nothing here runs. With it on, the GPU steps today's run executed
(the capability matmul, VerifyX) are prepared again the way their checks prepare them, sent in one
signed intent through `liumd run` (`LiumdExecClient`), judged with the same functions
(`evaluate_matmul_output`, `evaluate_verifyx_capture`), and compared with today's verdicts read from
the run's events. The outcome is one `[liumd_shadow] comparison` log line. Nothing is returned to
the caller, stored, or scored.

Today's verdict per step comes from the pipeline's events: both checks are fatal, so a check that
failed is the run's last event on a run that did not pass; one that reported a skip reason did not
run its probe; one with no event was never reached. The shadow runs no GPU work where today's run
ran none: a node with a filler or a pod (by today's events or the run's rented snapshot), a step
whose check skipped, or a run that stopped before either check. It is bounded by the time the
executor task has left, so it cannot push the node's result past the task's timeout.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

from datura.requests.validator_requests import MatmulStep, VerifyXStep
from services.liumd_exec_client import LiumdExecClient, LiumdRefusal
from services.local_verify_client import (
    DETAIL_MAX_CHARS,
    STEP_NAMES,
    LocalVerifyAnswer,
    LocalVerifyUnavailable,
    build_intent,
    executor_deadline_s,
)
from services.verifyx_validation_service import SSHCapture

from core.config import settings
from core.utils import _m, get_extra_info

from .checks.capability import CapabilityCheck
from .checks.local_verify import _step_reason
from .checks.verifyx import VerifyXCheck, _first_pass_challenge_config
from .pipeline import Context

logger = logging.getLogger(__name__)

LIUMD_SHADOW_EVENT = "[liumd_shadow] comparison"
# `MinerService` runs each executor task under `wait_for(..., JOB_TIME_OUT - 120)`; the express
# lane allows the whole JOB_TIME_OUT, so the tighter one bounds both.
TASK_TIMEOUT_MARGIN_SECONDS = 120
# Left after the shadow for the rest of the task (removing the validator's key, returning).
TAIL_MARGIN_SECONDS = 60
# Below this, not even a first-pass run fits; the shadow is skipped rather than cut off.
MIN_BUDGET_SECONDS = 60

TODAY_CHECK_IDS = {"matmul": CapabilityCheck.check_id, "verifyx": VerifyXCheck.check_id}
# Reason codes of a check that did not run its probe because a workload holds the cards.
WORKLOAD_REASONS = frozenset(
    {
        "GPU_VERIFY_SKIPPED_ACTIVE_FILLER",
        "GPU_VERIFY_SKIPPED_RENTED",
        "VERIFYX_SKIPPED_ACTIVE_FILLER",
    }
)
# Reason codes of a check that did not run its probe for any other reason.
SKIP_REASONS = WORKLOAD_REASONS | {"VERIFYX_DISABLED"}
RAN = ("pass", "fail")


@dataclass(frozen=True)
class TodayStep:
    """What today's run decided for one step: pass | fail | skipped | not_reached."""

    verdict: str
    reason_code: str | None = None
    ms: int | None = None


def today_verdicts(ok: bool, events: list) -> dict[str, TodayStep]:
    stopped_at = events[-1].check_id if events and not ok else None
    verdicts: dict[str, TodayStep] = {}
    for step, check_id in TODAY_CHECK_IDS.items():
        event = next((e for e in reversed(events) if e.check_id == check_id), None)
        if event is None:
            verdicts[step] = TodayStep("not_reached")
            continue
        reason = event.reason_code
        ms = (event.context or {}).get("execution_time_ms")
        if check_id == stopped_at:
            verdict = "fail"
        elif reason in SKIP_REASONS:
            verdict = "skipped"
        else:
            verdict = "pass"
        verdicts[step] = TodayStep(verdict, reason, ms if isinstance(ms, int) else None)
    return verdicts


def shadow_deadline(task_started_at: float) -> float:
    """The monotonic time by which the shadow must be done, for a task started at
    `task_started_at` (`time.monotonic()`)."""
    return (
        task_started_at + settings.JOB_TIME_OUT - TASK_TIMEOUT_MARGIN_SECONDS - TAIL_MARGIN_SECONDS
    )


def _workload_on_node(ctx: Context, today: dict[str, TodayStep]) -> bool:
    if any(step.reason_code in WORKLOAD_REASONS for step in today.values()):
        return True
    rented = ctx.state.rented_data
    if rented is None:
        return False
    if rented.get_filler_containers(ctx.executor.uuid):
        return True
    executor = rented.executors.get(ctx.executor.uuid)
    return bool(executor and executor.pods)


def _default_client(ctx: Context, timeout_s: float) -> LiumdExecClient:
    return LiumdExecClient(ctx.config.validator_keypair, timeout_s=timeout_s)


async def run_liumd_shadow(
    ctx: Context,
    *,
    ok: bool,
    events: list,
    deadline_monotonic: float,
    client_factory: Callable[[Context, float], Any] | None = None,
) -> dict[str, Any]:
    """Compare, log, and return the logged record (for tests). Never raises."""
    record: dict[str, Any]
    try:
        remaining = deadline_monotonic - time.monotonic()
        record = await asyncio.wait_for(
            _compare(ctx, ok, events, remaining, client_factory or _default_client),
            max(remaining, 0.001),
        )
    except TimeoutError:
        record = {"outcome": "unavailable", "reason": "timeout", "detail": "task budget spent"}
    except Exception as exc:  # noqa: BLE001 — a shadow bug must never reach the node's cycle
        record = {
            "outcome": "error",
            "reason": "internal_error",
            "detail": f"{type(exc).__name__}: {exc}"[:DETAIL_MAX_CHARS],
        }
    try:
        logger.info(
            _m(
                LIUMD_SHADOW_EVENT,
                extra=get_extra_info({**ctx.default_extra, "transport": "liumd_shadow", **record}),
            )
        )
    except Exception:  # noqa: BLE001
        pass
    return record


def _skipped(reason: str, **fields: Any) -> dict[str, Any]:
    return {"outcome": "skipped", "reason": reason, **fields}


def _today_fields(today: TodayStep) -> dict[str, Any]:
    return {"today": today.verdict, "today_reason_code": today.reason_code, "today_ms": today.ms}


async def _compare(
    ctx: Context,
    ok: bool,
    events: list,
    remaining: float,
    client_factory: Callable[[Context, float], Any],
) -> dict[str, Any]:
    today = today_verdicts(ok, events)
    steps = {name: _today_fields(step) for name, step in today.items()}
    if _workload_on_node(ctx, today):
        return _skipped("workload", steps=steps)
    ask_matmul = today["matmul"].verdict in RAN
    ask_verifyx = today["verifyx"].verdict in RAN and ctx.config.verifyx_enabled
    if not (ask_matmul or ask_verifyx):
        return _skipped("no_gpu_step_ran", steps=steps)
    specs = ctx.state.specs
    if not specs:
        return _skipped("no_specs", steps=steps)
    if ctx.config.validator_keypair is None:
        return _skipped("no_keypair", steps=steps)
    budget = min(float(settings.LIUMD_SHADOW_TIMEOUT_SECONDS), remaining)
    if budget < MIN_BUDGET_SECONDS:
        return _skipped("no_budget", steps=steps, budget_s=int(budget))

    first_pass = ctx.config.first_pass
    # The judging functions log with this extra; the transport field keeps the shadow's lines
    # apart from today's.
    extra = {**ctx.default_extra, "transport": "liumd_shadow"}
    matmul_challenge = None
    try:
        if ask_matmul:
            matmul_challenge = ctx.services.validation.prepare_matmul_challenge(
                specs,
                extra,
                vram_budget_mb=settings.FIRST_PASS_MATMUL_VRAM_MB if first_pass else None,
            )
            if not matmul_challenge.params.cipher_text:
                return _skipped("cipher_generation_failed", steps=steps)
        verifyx_challenge = (
            ctx.services.verifyx.prepare_verifyx_challenge(
                specs,
                extra,
                challenge_config_overrides=_first_pass_challenge_config() if first_pass else None,
            )
            if ask_verifyx
            else None
        )
        intent = build_intent(
            executor_uuid=ctx.executor.uuid,
            miner_hotkey=ctx.miner_hotkey,
            matmul=(
                MatmulStep(
                    dim_n=matmul_challenge.params.dim_n,
                    dim_k=matmul_challenge.params.dim_k,
                    seed=matmul_challenge.params.seed,
                    cipher_text=matmul_challenge.params.cipher_text,
                )
                if matmul_challenge is not None
                else None
            ),
            verifyx=(
                VerifyXStep(seed=verifyx_challenge.seed, cipher_text=verifyx_challenge.cipher_text)
                if verifyx_challenge is not None
                else None
            ),
            parallel_gpu=first_pass,
            deadline_s=executor_deadline_s(int(budget)),
        )
        client = client_factory(ctx, budget)
        try:
            answer = await client.run(ctx.ssh, intent)
        except LocalVerifyUnavailable as exc:
            return {
                "outcome": "unavailable",
                "reason": exc.reason,
                "detail": exc.detail[:DETAIL_MAX_CHARS],
                "steps": steps,
                "first_pass": first_pass,
            }
        if isinstance(answer, LiumdRefusal):
            return {
                "outcome": "refused",
                "reason": answer.error,
                "detail": answer.detail,
                "exit_status": answer.exit_status,
                "echoed": answer.echoed,
                "round_trip_ms": answer.round_trip_ms,
                "steps": steps,
                "first_pass": first_pass,
            }
        return _judged(ctx, answer, today, steps, matmul_challenge, verifyx_challenge, extra)
    finally:
        if matmul_challenge is not None:
            matmul_challenge.close()


def _liumd_fields(step, verdict: str, reason: str) -> dict[str, Any]:
    return {
        "liumd_status": step.status,
        "liumd_verdict": verdict,
        "liumd_reason": reason,
        "liumd_ms": step.ms,
    }


def _judged(
    ctx: Context,
    answer: LocalVerifyAnswer,
    today: dict[str, TodayStep],
    steps: dict[str, dict[str, Any]],
    matmul_challenge,
    verifyx_challenge,
    extra: dict,
) -> dict[str, Any]:
    if matmul_challenge is not None:
        step = answer.step("matmul")
        if step.status != "ok" or step.stdout is None:
            verdict, reason = (
                ("not_run" if step.status == "skipped" else "fail"),
                _step_reason(step),
            )
        else:
            result = ctx.services.validation.evaluate_matmul_output(
                matmul_challenge, stdout=step.stdout, stderr=step.stderr_tail or ""
            )
            verdict, reason = ("pass", "ok") if result.success else ("fail", "local_failed")
        steps["matmul"].update(_liumd_fields(step, verdict, reason))
    if verifyx_challenge is not None:
        step = answer.step("verifyx")
        if step.status != "ok" or step.stdout is None:
            verdict, reason = (
                ("not_run" if step.status == "skipped" else "fail"),
                _step_reason(step),
            )
        elif step.data.get("lib_sha256") != verifyx_challenge.expected_lib_sha256:
            verdict, reason = "fail", "lib_mismatch"
        else:
            response = ctx.services.verifyx.evaluate_verifyx_capture(
                verifyx_challenge,
                SSHCapture(
                    stdout=step.stdout, stderr=step.stderr_tail, exit_status=step.exit_status
                ),
                extra,
            )
            passed = bool(response.data and response.data.get("success"))
            verdict, reason = ("pass", "ok") if passed else ("fail", "local_failed")
        steps["verifyx"].update(_liumd_fields(step, verdict, reason))
    for name in STEP_NAMES:
        if name not in steps and name in answer.steps:
            steps[name] = {
                "liumd_status": answer.steps[name].status,
                "liumd_ms": answer.steps[name].ms,
            }

    compared = []
    for name in TODAY_CHECK_IDS:
        entry = steps[name]
        if today[name].verdict in RAN and entry.get("liumd_verdict") in RAN:
            entry["agree"] = entry["liumd_verdict"] == today[name].verdict
            compared.append(entry["agree"])
    return {
        "outcome": "compared",
        "reason": "ok",
        "agree": all(compared) if compared else None,
        "round_trip_ms": answer.round_trip_ms,
        "executor_elapsed_ms": answer.elapsed_ms,
        "executor_version": answer.executor_version,
        "deadline_hit": answer.deadline_hit,
        "first_pass": ctx.config.first_pass,
        "steps": steps,
    }
