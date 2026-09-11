"""liumd phase 2 (DAH-2834): one early, facts-only `POST /verify` in place of the read-only SSH
listings the checks after it would run.

The GPU call (`LocalVerifyCheck`) already returns the executor's `docker` / `ports` / `inspector`
facts, but it runs after the checks that could use them (stale cleanup, port selection, inspector).
This check asks for the facts alone — no matmul, no VerifyX, ≈ 1 s on the executor — right before
the first consumer, and leaves the parsed, bounded result in `ctx.state.local_facts`
(`services.local_verify_facts.LocalFacts`). Its `/version` answer is kept there too, so
`LocalVerifyCheck` need not fetch it again.

Every outcome other than a well-formed answer inside `LOCAL_VERIFY_FACTS_TIMEOUT_SECONDS` leaves
the state empty and the consumers on their SSH commands; nothing here can fail a node. Off by
default (`LOCAL_VERIFY_FACTS_ENABLED`, under `VALIDATOR_LOCAL_VERIFY_ENABLED`).
"""

from __future__ import annotations

import logging
from dataclasses import replace

from services.local_verify_client import (
    CAPABILITY,
    DETAIL_MAX_CHARS,
    LocalVerifyClient,
    LocalVerifyUnavailable,
    build_intent,
)
from services.local_verify_facts import LocalFacts, parse_facts

from core.config import settings
from core.utils import _m, get_extra_info

from ..messages import LocalFactsMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context

logger = logging.getLogger(__name__)

LOCAL_VERIFY_OUTCOME_EVENT = "[local_verify] outcome"
# The executor caps each fact collector at 20 s (FAST_STEP_TIMEOUT_SECONDS); ask for no more.
FACTS_DEADLINE_S = 20


class LocalFactsCheck:
    check_id = "executor.local_facts"
    fatal = False

    def __init__(self, client_factory=None):
        self._client_factory = client_factory or self._default_client

    @staticmethod
    def _default_client(ctx: Context) -> LocalVerifyClient:
        return LocalVerifyClient(
            ctx.config.validator_keypair,
            timeout_s=settings.LOCAL_VERIFY_FACTS_TIMEOUT_SECONDS,
            connect_timeout_s=settings.LOCAL_VERIFY_CONNECT_TIMEOUT_SECONDS,
        )

    async def run(self, ctx: Context) -> CheckResult:
        if not (settings.VALIDATOR_LOCAL_VERIFY_ENABLED and settings.LOCAL_VERIFY_FACTS_ENABLED):
            return CheckResult(
                passed=True, event=render_message(Msg.DISABLED, ctx=ctx, check_id=self.check_id)
            )
        try:
            return await self._run(ctx)
        except Exception as exc:  # noqa: BLE001 — the pipeline has no guard; a bug here must not end the node's cycle
            # A validator-side bug, not an executor answer: logged as a WARNING with the traceback and
            # `outcome=error`, so it never hides in the fallback rate (11 Sep: a missing build_intent
            # kwarg made every facts call a quiet `internal_error` fallback; only the tests noticed).
            logger.warning(
                _m(
                    LOCAL_VERIFY_OUTCOME_EVENT,
                    extra=get_extra_info(
                        {
                            **ctx.default_extra,
                            "outcome": "error",
                            "step": "facts",
                            "reason": "internal_error",
                            "detail": f"{type(exc).__name__}: {exc}"[:DETAIL_MAX_CHARS],
                            "first_pass": ctx.config.first_pass,
                        }
                    ),
                ),
                exc_info=exc,
            )
            return self._unavailable(ctx, "internal_error", f"{type(exc).__name__}: {exc}")

    async def _run(self, ctx: Context) -> CheckResult:
        if ctx.config.validator_keypair is None:
            return self._unavailable(ctx, "no_keypair", "pipeline has no validator keypair to sign with")

        client = self._client_factory(ctx)
        capabilities = await client.capabilities(ctx.executor)
        if CAPABILITY not in capabilities:
            self._metric(ctx, "fallback", "not_advertised")
            return CheckResult(
                passed=True,
                event=render_message(
                    Msg.SKIPPED,
                    ctx=ctx,
                    check_id=self.check_id,
                    what={"why": "not advertised", "capabilities": sorted(capabilities)},
                ),
                # The capability answer is worth keeping either way: LocalVerifyCheck reads it.
                updates={
                    "state": replace(
                        ctx.state, local_facts=LocalFacts(capabilities=frozenset(capabilities))
                    )
                },
            )

        intent = build_intent(
            executor_uuid=ctx.executor.uuid,
            miner_hotkey=ctx.miner_hotkey,
            matmul=None,
            verifyx=None,
            parallel_gpu=False,
            deadline_s=FACTS_DEADLINE_S,
        )
        try:
            answer = await client.verify(ctx.executor, intent)
        except LocalVerifyUnavailable as exc:
            return self._unavailable(ctx, exc.reason, exc.detail, capabilities=capabilities)

        facts = parse_facts(
            answer.steps,
            capabilities=capabilities,
            round_trip_ms=answer.round_trip_ms,
            executor_elapsed_ms=answer.elapsed_ms,
        )
        return self._report(ctx, facts)

    def _report(self, ctx: Context, facts: LocalFacts) -> CheckResult:
        """The event and the `[local_verify] outcome` line for a parsed fact table: which facts a
        later check can use (`usable`) decides OK vs UNAVAILABLE and consumed vs fallback."""
        what = {
            "round_trip_ms": facts.round_trip_ms,
            "executor_elapsed_ms": facts.executor_elapsed_ms,
            "steps": facts.step_statuses,
            "containers": None if facts.containers is None else len(facts.containers),
            "host_now": facts.host_now,
            "published_ports": None
            if facts.published_ports is None
            else len(facts.published_ports),
            "inspector_digest": facts.inspector_lib_sha256 is not None,
        }
        usable = [
            name
            for name, present in (
                ("containers", facts.can_age_containers()),
                ("ports", facts.published_ports is not None),
                ("inspector", facts.inspector_lib_sha256 is not None),
            )
            if present
        ]
        self._metric(
            ctx,
            "consumed" if usable else "fallback",
            "ok" if usable else "no_usable_fact",
            round_trip_ms=facts.round_trip_ms,
            executor_elapsed_ms=facts.executor_elapsed_ms,
            usable=",".join(usable),
        )
        return CheckResult(
            passed=True,
            event=render_message(
                Msg.OK if usable else Msg.UNAVAILABLE,
                ctx=ctx,
                check_id=self.check_id,
                what={**what, "usable": usable},
            ),
            updates={"state": replace(ctx.state, local_facts=facts)},
        )

    def _unavailable(
        self, ctx: Context, reason: str, detail: str, capabilities: set[str] | None = None
    ) -> CheckResult:
        detail = detail[:DETAIL_MAX_CHARS]
        self._metric(ctx, "fallback", reason, detail=detail)
        updates = (
            {"state": replace(ctx.state, local_facts=LocalFacts(capabilities=frozenset(capabilities)))}
            if capabilities is not None
            else {}
        )
        return CheckResult(
            passed=True,
            event=render_message(
                Msg.UNAVAILABLE,
                ctx=ctx,
                check_id=self.check_id,
                what={"reason": reason, "detail": detail},
            ),
            updates=updates,
        )

    @staticmethod
    def _metric(ctx: Context, outcome: str, reason: str, detail: str = "", **fields) -> None:
        logger.info(
            _m(
                LOCAL_VERIFY_OUTCOME_EVENT,
                extra=get_extra_info(
                    {
                        **ctx.default_extra,
                        "outcome": outcome,
                        "step": "facts",
                        "reason": reason,
                        "detail": detail[:DETAIL_MAX_CHARS],
                        "first_pass": ctx.config.first_pass,
                        **fields,
                    }
                ),
            )
        )
