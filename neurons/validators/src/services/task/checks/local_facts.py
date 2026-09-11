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
import re
from dataclasses import replace

from datura.requests.validator_requests import (
    LOCAL_VERIFY_DIND_NAME_MAX,
    LOCAL_VERIFY_DIND_NAME_PATTERN,
    LOCAL_VERIFY_DIND_PUBLIC_KEY_MAX,
    LOCAL_VERIFY_DIND_PUBLIC_KEY_PATTERN,
    DindStep,
)
from services.executor_connectivity.models import PortPair
from services.local_verify_client import (
    CAPABILITY,
    DETAIL_MAX_CHARS,
    DIND_CAPABILITY,
    EXECUTOR_DEADLINE_MIN_SECONDS,
    LocalVerifyClient,
    LocalVerifyUnavailable,
    build_intent,
)
from services.local_verify_facts import LocalFacts, PreparedDind, judge_dind_step, parse_facts
from services.port_utils import get_all_ports

from core.config import settings
from core.utils import _m, get_extra_info

from ..messages import LocalFactsMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context

logger = logging.getLogger(__name__)

LOCAL_VERIFY_OUTCOME_EVENT = "[local_verify] outcome"
# The executor caps each fact collector at 20 s (FAST_STEP_TIMEOUT_SECONDS); ask for no more.
FACTS_STEP_CAP_S = 20
# The answer has to travel back inside the client's whole-call timeout; the GPU call's 30 s margin
# (`executor_deadline_s`) would leave a 25 s facts budget with the 5 s floor, so the facts call has
# its own, sized for a ≈ 1 s call on a far node.
FACTS_DEADLINE_MARGIN_S = 5


def facts_deadline_s(timeout_s: int) -> int:
    """The `deadline_s` for the facts-only intent given the client's whole-call timeout
    (`LOCAL_VERIFY_FACTS_TIMEOUT_SECONDS`): the timeout less the margin, never above the executor's
    per-collector cap (asking for more buys nothing) and never below the floor the executor accepts
    (`EXECUTOR_DEADLINE_MIN_SECONDS`, the intent's `ge=5`).
    Default 25 → 20; an operator who tightens the timeout tightens the executor's stop with it, so a
    cut answer still arrives before the client gives up."""
    wanted = int(timeout_s) - FACTS_DEADLINE_MARGIN_S
    return max(EXECUTOR_DEADLINE_MIN_SECONDS, min(FACTS_STEP_CAP_S, wanted))


_DIND_NAME_RE = re.compile(LOCAL_VERIFY_DIND_NAME_PATTERN)
_DIND_KEY_RE = re.compile(LOCAL_VERIFY_DIND_PUBLIC_KEY_PATTERN)


def dind_step_fits_the_wire_bounds(name: str, public_key: str) -> bool:
    """The executor's `DindStep` bounds (datura), applied here first so the two ends cannot drift
    apart silently: a step this returns False for would be a 422 on the whole intent."""
    return (
        len(name) <= LOCAL_VERIFY_DIND_NAME_MAX
        and _DIND_NAME_RE.fullmatch(name) is not None
        and len(public_key) <= LOCAL_VERIFY_DIND_PUBLIC_KEY_MAX
        and _DIND_KEY_RE.fullmatch(public_key) is not None
    )


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

        prepared = self._choose_dind_identity(ctx, capabilities)
        intent = build_intent(
            executor_uuid=ctx.executor.uuid,
            miner_hotkey=ctx.miner_hotkey,
            matmul=None,
            verifyx=None,
            parallel_gpu=False,
            deadline_s=facts_deadline_s(settings.LOCAL_VERIFY_FACTS_TIMEOUT_SECONDS),
            dind=None
            if prepared is None
            else DindStep(
                name=prepared.name,
                port=prepared.port.external,
                public_key=prepared.public_key,
                sysbox=prepared.sysbox,
            ),
        )
        try:
            answer = await client.verify(ctx.executor, intent)
        except LocalVerifyUnavailable as exc:
            # The executor may have started the container before the answer was lost, so the name
            # the validator asked for stays in the state (`started` False, reason = the loss):
            # ProviderSideLoadCheck excuses that name instead of billing its cores to the provider,
            # and the settle step removes the container if it exists. PortConnectivityCheck runs
            # as today (it takes only a `started` container).
            if prepared is not None:
                prepared.reason = exc.reason
            return self._unavailable(
                ctx, exc.reason, exc.detail, capabilities=capabilities, dind=prepared
            )

        facts = parse_facts(
            answer.steps,
            capabilities=capabilities,
            round_trip_ms=answer.round_trip_ms,
            executor_elapsed_ms=answer.elapsed_ms,
        )
        if prepared is not None:
            facts = replace(facts, dind=judge_dind_step(prepared, answer.steps))
            self._metric(
                ctx,
                "consumed" if facts.dind.started else "fallback",
                "ok" if facts.dind.started else facts.dind.reason,
                step_name="dind",
                port=prepared.port.external,
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
                ("dind", facts.dind is not None and facts.dind.started),
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

    def _choose_dind_identity(self, ctx: Context, capabilities: set[str]) -> PreparedDind | None:
        """Phase 2c: the name, port and key pair the DinD probe would create over SSH, chosen now so
        the executor can start the container beside the other steps. The port is the first one the
        selector would offer (configured range minus rented and filler ports — the validator's own
        set, no fact involved); the name is the probe's own `container_<hotkey>_<port>`."""
        if not settings.LOCAL_VERIFY_DIND_IN_INTENT or DIND_CAPABILITY not in capabilities:
            return None
        if not ctx.config.job_batch_id or ctx.services.ssh is None:
            return None  # PortConnectivityCheck would not probe, or nothing can mint a key
        rented_data = ctx.state.rented_data
        rented = rented_data.executors.get(ctx.executor.uuid) if rented_data else None
        taken = set(rented.get_rented_ports() if rented else [])
        taken |= set(rented_data.get_filler_ports(ctx.executor.uuid) if rented_data else [])
        try:
            pairs = get_all_ports(
                ctx.executor.port_range, ctx.executor.port_mappings, ctx.executor.ssh_port
            )
        except (ValueError, TypeError):
            return None
        pair = next(((i, e) for i, e in pairs if e not in taken), None)
        if pair is None:
            return None
        private_key, public_key = ctx.services.ssh.generate_keypair()
        port = PortPair(*pair)
        name, public_key = f"container_{ctx.miner_hotkey}_{port.external}", public_key.strip()
        if not dind_step_fits_the_wire_bounds(name, public_key):
            # The executor's DindStep would reject it with a 422 on the WHOLE intent — every fact
            # lost for a step that is an optimisation. Leave the step out; the probe runs as today.
            logger.warning(
                _m(
                    "[local_facts] dind step outside the shared bounds; not asking",
                    extra=get_extra_info(
                        {**ctx.default_extra, "name_len": len(name), "key_len": len(public_key)}
                    ),
                )
            )
            return None
        return PreparedDind(
            name=name,
            port=port,
            private_key=private_key,
            public_key=public_key,
            sysbox=ctx.state.sysbox_runtime,
        )

    def _unavailable(
        self,
        ctx: Context,
        reason: str,
        detail: str,
        capabilities: set[str] | None = None,
        dind: PreparedDind | None = None,
    ) -> CheckResult:
        detail = detail[:DETAIL_MAX_CHARS]
        self._metric(ctx, "fallback", reason, detail=detail)
        updates = (
            {
                "state": replace(
                    ctx.state,
                    local_facts=LocalFacts(capabilities=frozenset(capabilities), dind=dind),
                )
            }
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
    def _metric(
        ctx: Context, outcome: str, reason: str, detail: str = "", step_name: str = "facts", **fields
    ) -> None:
        logger.info(
            _m(
                LOCAL_VERIFY_OUTCOME_EVENT,
                extra=get_extra_info(
                    {
                        **ctx.default_extra,
                        "outcome": outcome,
                        "step": step_name,
                        "reason": reason,
                        "detail": detail[:DETAIL_MAX_CHARS],
                        "first_pass": ctx.config.first_pass,
                        **fields,
                    }
                ),
            )
        )
