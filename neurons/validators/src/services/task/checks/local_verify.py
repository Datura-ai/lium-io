"""liumd phase 1 (DAH-2834): the matmul and VerifyX from one signed call instead of the SSH sequence.

For an executor whose `/version` advertises `local_verify/1`, this check prepares the same two
challenges the SSH-driven checks would (`ValidationService.prepare_matmul_challenge`,
`VerifyXValidationService.prepare_verifyx_challenge`), sends them in one validator-signed intent to
`POST /verify`, and judges the answer with the same two functions those checks use on SSH output
(`evaluate_matmul_output`, `evaluate_verifyx_capture`). A judged, passing step is left in
`ctx.state.local_verify`; `CapabilityCheck` and `VerifyXCheck` consume it and skip their SSH run.

Everything else — flag off, capability absent, refusal, timeout, a mismatched answer, a step that
did not run, or a step that ran and FAILED the judgement — leaves that step to the SSH path, so the
new transport can only save time, never change a verdict on its own. The matmul is not asked for
at all while `MATMUL_ALLCARDS_CHECK_ENABLED` is on: the all-cards work-proof runs inside the SSH
matmul path and a consumed local pass must not skip it. Every outcome is one `[local_verify]
outcome` log line with `outcome`, `step` and `reason` (the per-outcome metric). Off by default
(VALIDATOR_LOCAL_VERIFY_ENABLED).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any

from services.local_verify_client import (
    CAPABILITY,
    LocalVerifyAnswer,
    LocalVerifyClient,
    LocalVerifyUnavailable,
    build_intent,
    executor_deadline_s,
)
from services.verifyx_validation_service import SSHCapture

from core.config import settings
from core.utils import _m, get_extra_info

from ..messages import LocalVerifyMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context
from .capability import _get_filler_only_container
from .verifyx import _first_pass_challenge_config

logger = logging.getLogger(__name__)

LOCAL_VERIFY_OUTCOME_EVENT = "[local_verify] outcome"


def _step_reason(step) -> str:
    """The metric label for a step that cannot be judged: `step_<status>` for the executor's own
    statuses (failed | timeout | skipped | unsupported, `malformed` from the parser), and
    `step_no_stdout` for an `ok` step that carries nothing to judge."""
    return "step_no_stdout" if step.status == "ok" else f"step_{step.status}"


@dataclass
class LocalVerifyOutcome:
    """What the consuming checks read. A field is set only when the local step ran AND passed the
    validator's judgement; None means "run it over SSH"."""

    matmul: Any | None = None  # matrix_validation_service.ValidationResult
    verifyx: Any | None = None  # verifyx_validation_service.VerifyXResponse
    facts: dict[str, Any] = field(default_factory=dict)  # docker / ports / inspector evidence
    round_trip_ms: int = 0
    executor_elapsed_ms: int = 0
    executor_version: str = ""
    fallbacks: dict[str, str] = field(default_factory=dict)  # step -> reason


class LocalVerifyCheck:
    check_id = "executor.local_verify"
    fatal = False

    def __init__(self, client_factory=None):
        # Injectable for tests; production builds one client per run from the pipeline's keypair.
        self._client_factory = client_factory or self._default_client

    @staticmethod
    def _default_client(ctx: Context) -> LocalVerifyClient:
        return LocalVerifyClient(
            ctx.config.validator_keypair,
            timeout_s=settings.LOCAL_VERIFY_TIMEOUT_SECONDS,
            connect_timeout_s=settings.LOCAL_VERIFY_CONNECT_TIMEOUT_SECONDS,
        )

    async def run(self, ctx: Context) -> CheckResult:
        if not settings.VALIDATOR_LOCAL_VERIFY_ENABLED:
            return CheckResult(
                passed=True, event=render_message(Msg.DISABLED, ctx=ctx, check_id=self.check_id)
            )
        try:
            return await self._run(ctx)
        except Exception as exc:  # noqa: BLE001 — the pipeline has no guard; a bug here must not end the node's cycle
            return self._fallback(ctx, "call", "internal_error", f"{type(exc).__name__}: {exc}")

    async def _run(self, ctx: Context) -> CheckResult:
        specs = ctx.state.specs
        if not specs:
            return self._skipped(ctx, "no specs")
        if _get_filler_only_container(ctx):
            # Both consuming checks skip on an idle filler; there is nothing to run locally.
            return self._skipped(ctx, "filler only")
        if ctx.config.validator_keypair is None:
            return self._fallback(
                ctx, "call", "no_keypair", "pipeline has no validator keypair to sign with"
            )

        client = self._client_factory(ctx)
        capabilities = await client.capabilities(ctx.executor)
        if CAPABILITY not in capabilities:
            self._metric(ctx, "fallback", "call", "not_advertised")
            return CheckResult(
                passed=True,
                event=render_message(
                    Msg.NOT_ADVERTISED,
                    ctx=ctx,
                    check_id=self.check_id,
                    what={"capabilities": sorted(capabilities)},
                ),
            )

        # The same challenges the SSH checks would build, sized the same way (DAH-3011 first pass).
        first_pass = ctx.config.first_pass
        matmul_challenge = None
        verifyx_challenge = None
        # DAH-2671 item 3: the all-cards work-proof (`_probe_all_claimed_cards`, one pinned run per
        # card) lives inside the SSH matmul path only. While that check is on, the matmul stays on
        # SSH so a consumed local pass can never skip the probe or its enforcement; phase 2 carries
        # `devices` in the intent and judges the per-card output here.
        matmul_on_ssh = settings.MATMUL_ALLCARDS_CHECK_ENABLED
        if matmul_on_ssh and not ctx.config.verifyx_enabled:
            return self._fallback(
                ctx, "call", "allcards_ssh", "all-cards check on and VerifyX off: nothing to run"
            )
        try:
            try:
                if not matmul_on_ssh:
                    matmul_challenge = ctx.services.validation.prepare_matmul_challenge(
                        specs,
                        ctx.default_extra,
                        vram_budget_mb=settings.FIRST_PASS_MATMUL_VRAM_MB if first_pass else None,
                    )
                if ctx.config.verifyx_enabled:
                    verifyx_challenge = ctx.services.verifyx.prepare_verifyx_challenge(
                        specs,
                        ctx.default_extra,
                        challenge_config_overrides=_first_pass_challenge_config()
                        if first_pass
                        else None,
                    )
            except Exception as exc:  # a native library error is ours, not the node's
                return self._fallback(ctx, "call", "prepare_failed", f"{type(exc).__name__}: {exc}")

            if matmul_challenge is not None and not matmul_challenge.params.cipher_text:
                # Our own native error (the SSH path reports it the same way); nothing to send.
                return self._fallback(
                    ctx, "call", "cipher_generation_failed", "matmul cipher text is empty"
                )

            matmul_step = None
            if matmul_challenge is not None:
                params = matmul_challenge.params
                matmul_step = {
                    "dim_n": params.dim_n,
                    "dim_k": params.dim_k,
                    "seed": params.seed,
                    "cipher_text": params.cipher_text,
                }
            intent = build_intent(
                executor_uuid=ctx.executor.uuid,
                matmul=matmul_step,
                verifyx=(
                    {"seed": verifyx_challenge.seed, "cipher_text": verifyx_challenge.cipher_text}
                    if verifyx_challenge is not None
                    else None
                ),
                # Side by side only at first-pass sizes: a full-size VerifyX beside the matmul OOMs
                # 64–128 GB hosts (SWEEP_provider_verify §3 #6).
                parallel_gpu=first_pass,
                # Shorter than the client's whole-call timeout by a margin, so an answer the
                # executor cut at its deadline (`deadline_hit`, finished steps inside) still arrives
                # before the client gives up and is consumed step by step.
                deadline_s=executor_deadline_s(settings.LOCAL_VERIFY_TIMEOUT_SECONDS),
            )

            try:
                answer = await client.verify(ctx.executor, intent)
            except LocalVerifyUnavailable as exc:
                return self._fallback(ctx, "call", exc.reason, exc.detail)

            outcome = self._judge(ctx, answer, matmul_challenge, verifyx_challenge)
        finally:
            if matmul_challenge is not None:
                matmul_challenge.close()

        what = {
            "round_trip_ms": outcome.round_trip_ms,
            "executor_elapsed_ms": outcome.executor_elapsed_ms,
            "executor_version": outcome.executor_version,
            "steps": {name: step.status for name, step in answer.steps.items()},
            "consumed": [
                name for name in ("matmul", "verifyx") if getattr(outcome, name) is not None
            ],
            "fallbacks": outcome.fallbacks,
            "deadline_hit": answer.deadline_hit,
        }
        template = Msg.CONSUMED if what["consumed"] else Msg.FALLBACK
        return CheckResult(
            passed=True,
            event=render_message(template, ctx=ctx, check_id=self.check_id, what=what),
            updates={"state": replace(ctx.state, local_verify=outcome)},
        )

    def _judge(
        self, ctx: Context, answer: LocalVerifyAnswer, matmul_challenge, verifyx_challenge
    ) -> LocalVerifyOutcome:
        outcome = LocalVerifyOutcome(
            round_trip_ms=answer.round_trip_ms,
            executor_elapsed_ms=answer.elapsed_ms,
            executor_version=answer.executor_version,
        )
        common = {"round_trip_ms": answer.round_trip_ms, "executor_elapsed_ms": answer.elapsed_ms}

        step = answer.step("matmul")
        if matmul_challenge is None:
            # Not asked for: the all-cards work-proof keeps the matmul on SSH (see _run).
            outcome.fallbacks["matmul"] = "allcards_ssh"
            self._metric(ctx, "fallback", "matmul", "allcards_ssh", **common)
        elif step.status != "ok" or step.stdout is None:
            reason = _step_reason(step)
            outcome.fallbacks["matmul"] = reason
            self._metric(ctx, "fallback", "matmul", reason, detail=step.error or "", **common)
        else:
            result = ctx.services.validation.evaluate_matmul_output(
                matmul_challenge, stdout=step.stdout, stderr=step.stderr_tail or ""
            )
            if result.success:
                outcome.matmul = result
                self._metric(ctx, "consumed", "matmul", "ok", step_ms=step.ms, **common)
            else:
                # Judged and failed: the SSH run decides, so a false negative of the new transport
                # can never zero a score on its own. The reason is kept for the log.
                outcome.fallbacks["matmul"] = "local_failed"
                self._metric(
                    ctx, "fallback", "matmul", "local_failed", detail=result.error_message, **common
                )

        if verifyx_challenge is None:
            return outcome
        step = answer.step("verifyx")
        if step.status != "ok" or step.stdout is None:
            reason = _step_reason(step)
            outcome.fallbacks["verifyx"] = reason
            self._metric(ctx, "fallback", "verifyx", reason, detail=step.error or "", **common)
        elif step.data.get("lib_sha256") != verifyx_challenge.expected_lib_sha256:
            # The SSH path refuses an outdated libverifyx before running; here the digest rides
            # along with the answer and the same refusal applies.
            outcome.fallbacks["verifyx"] = "lib_mismatch"
            self._metric(ctx, "fallback", "verifyx", "lib_mismatch", **common)
        else:
            capture = SSHCapture(
                stdout=step.stdout, stderr=step.stderr_tail, exit_status=step.exit_status
            )
            response = ctx.services.verifyx.evaluate_verifyx_capture(
                verifyx_challenge, capture, ctx.default_extra
            )
            if response.data and response.data.get("success"):
                outcome.verifyx = response
                self._metric(ctx, "consumed", "verifyx", "ok", step_ms=step.ms, **common)
            else:
                outcome.fallbacks["verifyx"] = "local_failed"
                self._metric(
                    ctx, "fallback", "verifyx", "local_failed", detail=str(response.error), **common
                )

        for name in ("docker", "ports", "inspector"):
            fact = answer.step(name)
            if fact.status == "ok":
                outcome.facts[name] = fact.data
        return outcome

    def _skipped(self, ctx: Context, why: str) -> CheckResult:
        return CheckResult(
            passed=True,
            event=render_message(Msg.SKIPPED, ctx=ctx, check_id=self.check_id, what={"why": why}),
        )

    def _fallback(self, ctx: Context, step: str, reason: str, detail: str) -> CheckResult:
        self._metric(ctx, "fallback", step, reason, detail=detail)
        return CheckResult(
            passed=True,
            event=render_message(
                Msg.FALLBACK,
                ctx=ctx,
                check_id=self.check_id,
                what={"step": step, "reason": reason, "detail": detail},
            ),
        )

    @staticmethod
    def _metric(
        ctx: Context, outcome: str, step: str, reason: str, detail: str = "", **fields: Any
    ) -> None:
        # One line per outcome; Loki counts them by (outcome, step, reason).
        logger.info(
            _m(
                LOCAL_VERIFY_OUTCOME_EVENT,
                extra=get_extra_info(
                    {
                        **ctx.default_extra,
                        "outcome": outcome,
                        "step": step,
                        "reason": reason,
                        "detail": detail[:300],
                        "first_pass": ctx.config.first_pass,
                        **fields,
                    }
                ),
            )
        )
