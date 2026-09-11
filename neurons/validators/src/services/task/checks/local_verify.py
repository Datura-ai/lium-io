"""liumd phase 1 (DAH-2834): the matmul and VerifyX from one signed call instead of the SSH sequence.

For an executor whose `/version` advertises `local_verify/1`, this check prepares the same two
challenges the SSH-driven checks would (`ValidationService.prepare_matmul_challenge`,
`VerifyXValidationService.prepare_verifyx_challenge`), sends them in one validator-signed intent to
`POST /verify`, and judges the answer with the same two functions those checks use on SSH output
(`evaluate_matmul_output`, `evaluate_verifyx_capture`). A judged, passing step is left in
`ctx.state.local_verify`; `CapabilityCheck` and `VerifyXCheck` consume it and skip their SSH run.

Everything else — flag off, capability absent, refusal, timeout, a mismatched answer, a step that
did not run, a step that ran and FAILED the judgement, or a pass that arrived later than the SSH
path's own cap for that step (`ROUND_TRIP_CAP_MS_BY_STEP`, measured on the validator's clock) — leaves
that step to the SSH path, so the new transport can only save time, never change a verdict on its
own. The matmul is not asked for
at all while `MATMUL_ALLCARDS_CHECK_ENABLED` is on: the all-cards work-proof runs inside the SSH
matmul path and a consumed local pass must not skip it. Every outcome is one `[local_verify]
outcome` log line with `outcome`, `step` and `reason` (the per-outcome metric). Off by default
(VALIDATOR_LOCAL_VERIFY_ENABLED). Phase 2: the call is made on the first pass only while
`LOCAL_VERIFY_FIRST_PASS_ONLY` is on (default) — the saving is first-pass only, and a scored
cycle's serial full-size run can push a passing matmul past its cap and re-run it over SSH.
Phase 3 (`LOCAL_VERIFY_GPU_EARLY_START`): `LocalVerifyStartCheck`, placed right after the facts
call, sends the very same intent as a background task, so the executor's GPU steps run while the
port check, the sysbox proof and the image check still take their SSH round trips; this check
then awaits and judges that answer instead of making the call — same intent, same judge, same
caps, one call per cycle either way.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from datura.requests.validator_requests import MatmulStep, VerifyXStep
from services.local_verify_client import (
    CAPABILITY,
    DETAIL_MAX_CHARS,
    LocalVerifyAnswer,
    LocalVerifyClient,
    LocalVerifyOutcome,
    LocalVerifyUnavailable,
    build_intent,
    executor_deadline_s,
)
from services.matrix_validation_service import MATRIX_VERIFY_TIMEOUT_SECONDS
from services.verifyx_validation_service import VERIFYX_COMMAND_TIMEOUT_SECONDS, SSHCapture

from core.config import settings
from core.utils import _m, get_extra_info

from ..messages import LocalVerifyMessages as Msg
from ..messages import render_message
from ..pipeline import RENTAL_PROBE_OUTCOME_EVENT, CheckResult, Context
from .capability import _get_filler_only_container
from .gpu_usage import lium_workload_containers, reads_the_live_card
from .rental_verification import RentalProbe, rental_probe_request
from .verifyx import _first_pass_challenge_config

logger = logging.getLogger(__name__)

LOCAL_VERIFY_OUTCOME_EVENT = RENTAL_PROBE_OUTCOME_EVENT  # one Loki event name for every probe line


def _step_reason(step) -> str:
    """The metric label for a step that cannot be judged: `step_<status>` for the executor's own
    statuses (failed | timeout | skipped; `malformed` from the parser for anything else), and
    `step_no_stdout` for an `ok` step that carries nothing to judge."""
    return "step_no_stdout" if step.status == "ok" else f"step_{step.status}"


# The SSH path fails a matmul that runs past MATRIX_VERIFY_TIMEOUT_SECONDS and a VerifyX run past
# VERIFYX_COMMAND_TIMEOUT_SECONDS. The executor's own per-step caps and `step.ms` are its word, so
# the same bound is applied to the one clock the validator holds: the whole call's round trip,
# which is an upper bound on any step's wall-clock (serial or side by side).
ROUND_TRIP_CAP_MS_BY_STEP = {
    "matmul": MATRIX_VERIFY_TIMEOUT_SECONDS * 1000,
    "verifyx": VERIFYX_COMMAND_TIMEOUT_SECONDS * 1000,
}


def _over_time(name: str, answer: LocalVerifyAnswer) -> bool:
    return answer.round_trip_ms > ROUND_TRIP_CAP_MS_BY_STEP[name]


@dataclass
class PendingVerify:
    """Phase 3: the GPU `/verify` in flight, started by `LocalVerifyStartCheck` right after the facts
    call so the executor's GPU steps overlap the SSH-driven checks in between. The challenges are
    the ones the intent was built from — `LocalVerifyCheck` judges the answer against them exactly
    as it judges its own call's. `consumed` tells `Pipeline.run`'s settle step that nothing is left
    to cancel; `close()` releases the matmul challenge's native handle on every path."""

    task: asyncio.Task | None  # None only while `LocalVerifyStartCheck._start` is creating it
    matmul_challenge: Any | None
    verifyx_challenge: Any | None
    started_at: float = field(default_factory=time.perf_counter)
    consumed: bool = False
    _closed: bool = field(default=False, repr=False)

    def close(self) -> None:
        if not self._closed and self.matmul_challenge is not None:
            self.matmul_challenge.close()
        self._closed = True

    async def cancel_and_await(self) -> str:
        """Cancel if still running and retrieve the outcome so nothing is left un-awaited. Returns
        the metric reason: `cancelled` (was still running), `done` (finished, never consumed)."""
        was_running = not self.task.done()
        if was_running:
            self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)
        self.close()
        return "cancelled" if was_running else "done"


@dataclass(frozen=True)
class _Prepared:
    """The challenges and the signed intent one `/verify` call is made of — built the same way
    whether the call is made now (`LocalVerifyCheck`) or started early (`LocalVerifyStartCheck`)."""

    matmul_challenge: Any | None
    verifyx_challenge: Any | None
    intent: dict[str, Any]


@dataclass(frozen=True)
class _PrepareFailed:
    """Our own native error, nothing to send; each caller renders it under its own check id and
    step label (`call` for the check, `gpu_early` for the start check)."""

    reason: str
    detail: str


def _pending_of(ctx: Context) -> PendingVerify | None:
    pending = getattr(ctx.state.local_verify, "pending", None)
    return pending if pending is not None and not pending.consumed else None


def _gate_reason(ctx: Context) -> tuple[str, str, str] | None:
    """The reasons not to make the call at all, decided before any round trip: `(kind, reason,
    detail)` with kind `skipped` (nothing to run) or `fallback` (the SSH path decides). ONE list
    for `LocalVerifyCheck` and `LocalVerifyStartCheck`, so the early call is made under exactly the
    conditions the check would make its own. None = go on."""
    if not ctx.state.specs:
        return "skipped", "no_specs", "no specs"
    if _get_filler_only_container(ctx):
        # Both consuming checks skip on an idle filler; there is nothing to run locally.
        return "skipped", "filler_only", "filler only"
    if settings.LOCAL_VERIFY_FIRST_PASS_ONLY and not ctx.config.unscored:
        # Scored cycles keep the SSH path, decided before the `/version` round trip; why is
        # written once, on LOCAL_VERIFY_FIRST_PASS_ONLY in core/config.py.
        return "fallback", "not_first_pass", "not the first pass: the one-call path is first-pass only"
    if ctx.config.validator_keypair is None:
        return "fallback", "no_keypair", "pipeline has no validator keypair to sign with"
    # DAH-2671 item 3: the all-cards work-proof (`_probe_all_claimed_cards`, one pinned run per
    # card) lives inside the SSH matmul path only. While that check is on, the matmul stays on
    # SSH so a consumed local pass can never skip the probe or its enforcement; phase 2 carries
    # `devices` in the intent and judges the per-card output here.
    if settings.MATMUL_ALLCARDS_CHECK_ENABLED and not ctx.config.verifyx_enabled:
        return "fallback", "allcards_ssh", "all-cards check on and VerifyX off: nothing to run"
    return None


def _has_customer_rental(ctx: Context) -> bool:
    rented_data = ctx.state.rented_data
    rented = rented_data.executors.get(ctx.executor.uuid) if rented_data else None
    return bool(rented and rented.pods)


def _gpu_usage_reads_the_card(ctx: Context) -> bool:
    """Whether `GpuUsageCheck` (fatal, between the start and the judge) would take a live
    `nvidia-smi` read this cycle — with the early call running, our own matmul would be the compute
    app it sees. Decided on the same snapshot that check reads."""
    workload = lium_workload_containers(ctx) if ctx.state.rented_data is not None else set()
    return reads_the_live_card(ctx.state.gpu_details or [], ctx.state.gpu_processes or [], workload)


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
        started: list[RentalProbe] = []  # the probe, if _run started one before it raised
        pending = _pending_of(ctx)
        try:
            result = await self._run(ctx, started)
        except Exception as exc:  # noqa: BLE001 — the pipeline has no guard; a bug here must not end the node's cycle
            result = self._fallback(
                ctx,
                "call",
                "internal_error",
                f"{type(exc).__name__}: {exc}",
                probe=started[0] if started else None,
            )
        except BaseException:
            # A cancellation (the per-executor wait_for in miner_service) or an exit while
            # `client.verify` is awaited: nothing returns, so the pipeline's `finally` never sees
            # the probe. Cancel it here — a probe pod nobody will read is not rented on.
            for probe in started:
                if not probe.task.done():
                    probe.task.cancel()
                elif not probe.task.cancelled():
                    # nothing to cancel; read the exception so asyncio does not report it unretrieved
                    probe.task.exception()
            if pending is not None and not pending.consumed:
                pending.consumed = True
                pending.task.cancel()
                pending.close()
            raise
        if pending is not None and not pending.consumed:
            # `_run` returned before it got to the early call (a gate that reads state the start
            # check could not — a filler that appeared, a bug): the answer is not judged by anyone,
            # so it is settled here rather than left to the pipeline's finally.
            pending.consumed = True
            reason = await pending.cancel_and_await()
            self._metric(ctx, "fallback", "gpu_early", f"unconsumed_{reason}")
        return result

    def _start_rental_probe(self, ctx: Context) -> RentalProbe | None:
        """Phase 2 (LOCAL_VERIFY_RENTAL_PROBE_PARALLEL): start the backend rental probe now, so
        its ≈ 25 s (rent a probe pod, wait for sshd + nvidia-smi inside) overlap the executor's
        GPU steps instead of following them. First pass only — the probe pod takes the GPUs by
        UUID while the first-pass matmul (8 GB VRAM) and VerifyX run; full-size cycles keep the
        serial order (the `parallel_gpu` rule). The request is the one `RentalVerificationCheck`
        would build (`rental_probe_request`), and that check consumes the task only if its own
        request is identical."""
        if not settings.LOCAL_VERIFY_RENTAL_PROBE_PARALLEL or not ctx.config.first_pass:
            return None
        request = rental_probe_request(ctx)
        if request is None:
            return None
        task = asyncio.create_task(
            ctx.services.backend.check_executor_health(**asdict(request)), name="local_verify.rental_probe"
        )
        return RentalProbe(task=task, request=request)

    def _gate(self, ctx: Context) -> CheckResult | None:
        """The reasons not to make the call at all, decided before any round trip, rendered as
        this check's own skip / fallback. None = go on."""
        gate = _gate_reason(ctx)
        if gate is None:
            return None
        kind, reason, detail = gate
        if kind == "skipped":
            return self._skipped(ctx, detail)
        return self._fallback(ctx, "call", reason, detail)

    def _prepare(self, ctx: Context) -> _Prepared | _PrepareFailed:
        """The same challenges the SSH checks would build, sized the same way (DAH-3011 first
        pass), and the signed intent that carries them. `_PrepareFailed` is our own native error
        (nothing to send); on that path nothing is left open."""
        specs = ctx.state.specs
        first_pass = ctx.config.first_pass
        matmul_challenge = None
        verifyx_challenge = None
        try:
            if not settings.MATMUL_ALLCARDS_CHECK_ENABLED:
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
            if matmul_challenge is not None:
                matmul_challenge.close()
            return _PrepareFailed("prepare_failed", f"{type(exc).__name__}: {exc}")

        if matmul_challenge is not None and not matmul_challenge.params.cipher_text:
            # Our own native error (the SSH path reports it the same way); nothing to send.
            matmul_challenge.close()
            return _PrepareFailed("cipher_generation_failed", "matmul cipher text is empty")

        matmul_step = None
        if matmul_challenge is not None:
            params = matmul_challenge.params
            matmul_step = MatmulStep(
                dim_n=params.dim_n,
                dim_k=params.dim_k,
                seed=params.seed,
                cipher_text=params.cipher_text,
            )
        intent = build_intent(
            executor_uuid=ctx.executor.uuid,
            miner_hotkey=ctx.miner_hotkey,
            matmul=matmul_step,
            verifyx=(
                VerifyXStep(seed=verifyx_challenge.seed, cipher_text=verifyx_challenge.cipher_text)
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
        return _Prepared(matmul_challenge, verifyx_challenge, intent)

    async def _run(self, ctx: Context, started: list[RentalProbe]) -> CheckResult:
        gate = self._gate(ctx)
        if gate is not None:
            return gate

        pending = _pending_of(ctx)
        if pending is None:
            client = self._client_factory(ctx)
            # Phase 2: the early facts call (checks/local_facts) already read /version this cycle —
            # also when that read came back empty (nothing advertised, or `capabilities()` folded a
            # refusal/timeout into `set()`): this cycle takes the SSH path rather than paying a second call.
            facts = ctx.state.local_facts
            if facts is not None:
                capabilities = set(facts.capabilities)
            else:
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
            prepared = self._prepare(ctx)
            if isinstance(prepared, _PrepareFailed):
                return self._fallback(ctx, "call", prepared.reason, prepared.detail)
            matmul_challenge = prepared.matmul_challenge
            verifyx_challenge = prepared.verifyx_challenge
        else:
            # Phase 3: the call is already in flight since LocalVerifyStartCheck.
            matmul_challenge = pending.matmul_challenge
            verifyx_challenge = pending.verifyx_challenge

        early_lead_ms = (
            int((time.perf_counter() - pending.started_at) * 1000) if pending is not None else None
        )
        waited = time.perf_counter()
        try:
            # Started right before the call (or, phase 3, right before this check waits on it) so
            # it overlaps the executor's GPU steps; every return from here on carries it in
            # `ctx.state.local_verify` for RentalVerificationCheck.
            probe = self._start_rental_probe(ctx)
            if probe is not None:
                started.append(probe)

            try:
                if pending is None:
                    answer = await client.verify(ctx.executor, prepared.intent)
                else:
                    # From here on this check owns the call (the settle step has nothing left to
                    # cancel); marked right before the await so anything that raises before it —
                    # the rental probe's request — still leaves the task to `run`'s settle.
                    pending.consumed = True
                    answer = await pending.task
            except LocalVerifyUnavailable as exc:
                return self._fallback(ctx, "call", exc.reason, exc.detail, probe=probe)

            outcome = self._judge(ctx, answer, matmul_challenge, verifyx_challenge)
            outcome.rental_probe = probe
        finally:
            if pending is not None:
                pending.close()
            elif matmul_challenge is not None:
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
            "rental_probe_started": probe is not None,
        }
        if pending is not None:
            # How much of the call ran under the checks before this one (the overlap won) and how
            # long this check still had to wait — the two numbers that check §5's projection.
            what["early_lead_ms"] = early_lead_ms
            what["early_wait_ms"] = int((time.perf_counter() - waited) * 1000)
            self._metric(
                ctx,
                "consumed",
                "gpu_early",
                "ok",
                early_lead_ms=early_lead_ms,
                early_wait_ms=what["early_wait_ms"],
                round_trip_ms=outcome.round_trip_ms,
            )
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
        elif _over_time("matmul", answer):
            # Past the cap the SSH run would have failed it as timed out; a pass that slow is
            # not consumed — the SSH run decides.
            outcome.fallbacks["matmul"] = "step_overtime"
            self._metric(ctx, "fallback", "matmul", "step_overtime", **common)
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
        elif _over_time("verifyx", answer):
            outcome.fallbacks["verifyx"] = "step_overtime"
            self._metric(ctx, "fallback", "verifyx", "step_overtime", **common)
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
        return outcome

    def _skipped(self, ctx: Context, why: str) -> CheckResult:
        return CheckResult(
            passed=True,
            event=render_message(Msg.SKIPPED, ctx=ctx, check_id=self.check_id, what={"why": why}),
        )

    def _fallback(
        self,
        ctx: Context,
        step: str,
        reason: str,
        detail: str,
        probe: RentalProbe | None = None,
    ) -> CheckResult:
        detail = detail[:DETAIL_MAX_CHARS]  # executor-derived text: same cap as the metric line
        self._metric(ctx, "fallback", step, reason, detail=detail)
        # A started rental probe rides along even when the call itself fell back: the state
        # carries no consumed step (both consumers take SSH), only the task for the rental check.
        updates = (
            {"state": replace(ctx.state, local_verify=LocalVerifyOutcome(rental_probe=probe))}
            if probe is not None
            else {}
        )
        return CheckResult(
            passed=True,
            event=render_message(
                Msg.FALLBACK,
                ctx=ctx,
                check_id=self.check_id,
                what={"step": step, "reason": reason, "detail": detail},
            ),
            updates=updates,
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
                        "detail": detail[:DETAIL_MAX_CHARS],
                        "first_pass": ctx.config.first_pass,
                        "unscored": ctx.config.unscored,
                        **fields,
                    }
                ),
            )
        )


class LocalVerifyStartCheck(LocalVerifyCheck):
    """Phase 3 (`LOCAL_VERIFY_GPU_EARLY_START`): make `LocalVerifyCheck`'s call now, as a background
    task, and leave it in `ctx.state.local_verify.pending` for that check to await and judge.

    Placed right after `LocalFactsCheck`, whose `/version` answer says whether the executor takes
    the call at all (no second GET here: without facts nothing starts and the later check decides
    as today). Between here and the judge run the port check (the connect-back batch, the sysbox
    proof, the removal), the image check and the tenant check — ≈ 10–12 s of SSH round trips on the
    first pass that the executor's ≈ 45 s of GPU work now overlaps. The intent, the gate and the
    judge are the later check's own (`_gate_reason`, `_prepare`, `_judge`), so what is asked and
    what is accepted does not change; only when the call leaves. The GPU is asked for only on an
    unscored cycle whose `rented_data` shows no customer pod on the node (`TenantEnforcementCheck`
    would halt a rented node before the later check, and its GPUs are a renter's) and whose scrape
    gives `GpuUsageCheck` no reason to read the live card (a wedge candidate, ownerless VRAM: our
    matmul must not be the compute app those reads see). A fatal check or a halt in between leaves
    the task to `Pipeline.run`'s settle step; the executor stops at its own deadline.
    """

    check_id = "executor.local_verify_start"
    fatal = False

    async def run(self, ctx: Context) -> CheckResult:
        if not (settings.VALIDATOR_LOCAL_VERIFY_ENABLED and settings.LOCAL_VERIFY_GPU_EARLY_START):
            return CheckResult(
                passed=True, event=render_message(Msg.DISABLED, ctx=ctx, check_id=self.check_id)
            )
        try:
            return self._start(ctx)
        except Exception as exc:  # noqa: BLE001 — the pipeline has no guard; a bug here must not end the node's cycle
            return self._not_started(ctx, "internal_error", f"{type(exc).__name__}: {exc}")

    def _start(self, ctx: Context) -> CheckResult:
        if _pending_of(ctx) is not None:
            return self._not_started(ctx, "already_pending", "a call is already in flight")
        facts = ctx.state.local_facts
        capabilities = set(facts.capabilities) if facts is not None else None
        if capabilities is None:
            return self._not_started(ctx, "no_facts", "no facts call this cycle; the check decides later")
        if CAPABILITY not in capabilities:
            return self._not_started(ctx, "not_advertised", "executor does not advertise local verification")
        if not facts.step_statuses:
            # `/version` answered but the facts POST did not (busy, timeout, refused): the executor
            # is not taking intents right now; the later check asks again where it does today.
            return self._not_started(ctx, "facts_unanswered", "the facts call did not come back; the check decides later")
        if not ctx.config.unscored:
            # The early start is for the first pass only, whatever LOCAL_VERIFY_FIRST_PASS_ONLY
            # says: the overlap pays where the owner waits, and a scored cycle's full-size serial
            # call has more fatal checks ahead of the judge to be cancelled by.
            return self._not_started(ctx, "not_first_pass", "not the first pass")
        if _has_customer_rental(ctx):
            return self._not_started(ctx, "rented", "the node has a customer pod; its GPUs are not ours to load")
        if _gpu_usage_reads_the_card(ctx):
            # GpuUsageCheck's ghost-GPU cure / re-sample or its ownerless-VRAM confirmation read
            # the live card between here and the judge; our matmul must not be what they see.
            return self._not_started(ctx, "gpu_reread_pending", "the GPU usage check reads the live card this cycle")
        gate = _gate_reason(ctx)
        if gate is not None:
            return self._not_started(ctx, gate[1], gate[2])

        prepared = self._prepare(ctx)
        if isinstance(prepared, _PrepareFailed):
            # Nothing to send and nothing open; the later check builds its own challenges and
            # reports its own outcome (this line is `step=gpu_early`, never its `step=call`).
            return self._not_started(ctx, prepared.reason, prepared.detail)
        pending = PendingVerify(
            task=None,
            matmul_challenge=prepared.matmul_challenge,
            verifyx_challenge=prepared.verifyx_challenge,
        )
        try:
            client = self._client_factory(ctx)
            pending.task = asyncio.create_task(
                client.verify(ctx.executor, prepared.intent), name="local_verify.gpu_early"
            )
            gpu_steps = [name for name in ("matmul", "verifyx") if prepared.intent["steps"].get(name)]
            self._metric(ctx, "started", "gpu_early", "ok", steps=",".join(gpu_steps))
            return CheckResult(
                passed=True,
                event=render_message(
                    Msg.EARLY_STARTED, ctx=ctx, check_id=self.check_id, what={"steps": gpu_steps}
                ),
                updates={"state": replace(ctx.state, local_verify=LocalVerifyOutcome(pending=pending))},
            )
        except BaseException:
            # Nothing has been handed to the state yet: whatever was started is ours to stop.
            if pending.task is not None:
                pending.task.cancel()
            pending.close()
            raise

    def _not_started(self, ctx: Context, reason: str, detail: str) -> CheckResult:
        detail = detail[:DETAIL_MAX_CHARS]
        self._metric(ctx, "fallback", "gpu_early", reason, detail=detail)
        return CheckResult(
            passed=True,
            event=render_message(
                Msg.EARLY_SKIPPED,
                ctx=ctx,
                check_id=self.check_id,
                what={"reason": reason, "detail": detail},
            ),
        )
