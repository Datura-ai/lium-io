from __future__ import annotations

import asyncio
import secrets
import shlex
import time

from services.gpu_signature import (
    evaluate_card,
    kernel_uuid_mismatch,
    parse_result_line,
    seal_within_bounds,
    summarize,
)
from services.gpusig_validator import GpuSigVerifier

from core.config import settings

from ..messages import GpuSignatureMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context

# The verifier .so is published from celium-gpu-verifier and moved to /usr/lib by
# the validator image build (same place as libverifyx.so / libdmcompverify.so).
GPU_SIGNATURE_LIB = "/usr/lib/libgpusig.so"


class GpuSignatureCheck:
    """DAH-3137 — nonce-bound, sealed per-GPU hardware-signature challenge (observe-only).

    Runs the pre-placed executor-image binary (``bin/gpu_sig``, shipped in the
    image — NOT uploaded per check) once per claimed card, pinned via
    ``CUDA_VISIBLE_DEVICES``, with a fresh per-call nonce. Each result is
    authenticated by ``libgpusig.so`` (the verifier from celium-gpu-verifier),
    which recomputes the seal, checks the scheme tag and nonce freshness, and
    returns the AUTHENTICATED per-card numbers. The check then applies the
    per-class envelope and aggregates for count (device-selection failure,
    duplicate kernel UUID, or a blown aggregate wall-clock all flag a spoof).

    Additive and observe-only: it logs a verdict and never changes the score.
    ``GPU_SIGNATURE_ENFORCEMENT_ENABLED`` currently only raises the failing event
    from warning to error; wiring a score gate is a follow-up once the envelope
    is calibrated on real hardware (DAH-3137 design doc §9).
    """

    check_id = "gpu.validate.signature"
    fatal = False

    async def run(self, ctx: Context) -> CheckResult:
        if not settings.ENABLE_GPU_SIGNATURE_CHECK:
            return self._skip(ctx, "disabled")

        gpu_details = ctx.state.gpu_details or []
        gpu_count = ctx.state.gpu_count or len(gpu_details)
        if gpu_count <= 0 or not gpu_details:
            return self._skip(ctx, "no_gpus")

        # Do not run heavy GPU work on a card a filler is using (mirrors CapabilityCheck).
        filler = self._filler_only_container(ctx)
        if filler:
            return self._skip(ctx, "filler", extra_what={"filler_container": filler})

        binary_path = (
            f"{ctx.executor.root_dir.rstrip('/')}/"
            f"{settings.GPU_SIGNATURE_BINARY_RELATIVE.lstrip('/')}"
        )
        if not await self._binary_present(ctx, binary_path):
            return self._skip(ctx, "binary_absent", extra_what={"binary_path": binary_path})

        try:
            verifier = GpuSigVerifier(GPU_SIGNATURE_LIB)
        except Exception as exc:  # missing .so, missing symbol, load error — never fatal
            return self._skip(ctx, "verifier_unavailable", extra_what={"error": str(exc)[:200]})

        master_key = settings.GPU_SIGNATURE_KEY
        claimed_model = ctx.state.gpu_model
        claimed_uuids = [detail.get("uuid", "") for detail in gpu_details]

        gate = asyncio.Semaphore(max(1, settings.GPU_SIGNATURE_MAX_CONCURRENT))

        async def run_slot(slot: int):
            async with gate:
                # A FRESH nonce per card, so one sealed answer cannot be replayed across
                # slots. It is hex and the slot is an int, so the command is injection-safe;
                # the binary path is the only variable and is shlex-quoted.
                slot_nonce = secrets.token_hex(32)
                cmd = (
                    f"CUDA_VISIBLE_DEVICES={slot} {shlex.quote(binary_path)} "
                    f"--nonce {slot_nonce} --device 0"
                )
                try:
                    result = await ctx.ssh.run(cmd, timeout=settings.GPU_SIGNATURE_TIMEOUT_SECONDS)
                    stdout = getattr(result, "stdout", "") or ""
                except Exception as exc:  # timeout / channel error / device selection
                    return evaluate_card(
                        {"sealed": False, "reason": f"ssh:{type(exc).__name__}"},
                        claimed_model,
                        slot,
                    )
                parsed = parse_result_line(stdout)
                return self._verify_one(
                    verifier, master_key, slot_nonce, claimed_model, slot, parsed
                )

        start = time.perf_counter()
        verdicts = await asyncio.gather(*(run_slot(slot) for slot in range(gpu_count)))
        elapsed = time.perf_counter() - start

        verdict = summarize(verdicts, elapsed, gpu_count)

        # Cross-check the AUTHENTICATED /proc-sourced kernel UUIDs against the NVML-claimed
        # set: a kernel UUID that NVML never advertised is a userspace NVML shim (DAH-2662).
        # A card that merely failed (empty UUID) is not treated as a mismatch — that is the
        # per-card verdict's job, not this signal's.
        kernel_uuids = sorted(v.kernel_uuid for v in verdicts if v.kernel_uuid)
        uuid_mismatch = kernel_uuid_mismatch(kernel_uuids, claimed_uuids)

        node_ok = verdict.passed and not uuid_mismatch
        what = {
            "passed": node_ok,
            "reasons": verdict.reasons
            + (["kernel_vs_nvml_uuid_mismatch"] if uuid_mismatch else []),
            "claimed_count": verdict.claimed_count,
            "verified_count": verdict.verified_count,
            "elapsed_seconds": verdict.elapsed_seconds,
            "over_wall_clock": verdict.over_wall_clock,
            "gpu_model": claimed_model,
            "per_card": verdict.per_card,
            "kernel_uuids": kernel_uuids,
            "kernel_vs_nvml_uuid_mismatch": uuid_mismatch,
            "would_enforce": settings.GPU_SIGNATURE_ENFORCEMENT_ENABLED,
        }

        template = Msg.OK if node_ok else Msg.FAILED
        # Enforcement (not yet wired to scoring) raises the failing event from warning to error
        # so it is visible in alerting during the warn phase; the score is still untouched.
        severity = "error" if (settings.GPU_SIGNATURE_ENFORCEMENT_ENABLED and not node_ok) else None
        event = render_message(
            template, ctx=ctx, check_id=self.check_id, what=what, severity=severity
        )
        # Observe-only: never fail the pipeline or the score yet.
        return CheckResult(passed=True, event=event)

    def _verify_one(
        self,
        verifier: GpuSigVerifier,
        master_key: str,
        nonce: str,
        claimed_model: str | None,
        slot: int,
        parsed: dict | None,
    ):
        """Authenticate one raw response through libgpusig.so, then score it.

        Always returns a CardVerdict — a bad/absent response, a verifier exception or a
        non-dict verifier body all become an unsealed (failed) verdict, never an exception
        that could escape into the pipeline.
        """
        if parsed is None:
            return evaluate_card({"sealed": False, "reason": "no_result"}, claimed_model, slot)
        if parsed.get("ok") is not True:
            err = str(parsed.get("error", "unknown"))[:64]
            return evaluate_card(
                {"sealed": False, "reason": f"prober_error:{err}"},
                claimed_model,
                slot,
            )
        msg = parsed.get("msg")
        sig = parsed.get("sig")
        if not isinstance(msg, str) or not isinstance(sig, str):
            return evaluate_card({"sealed": False, "reason": "missing_seal"}, claimed_model, slot)
        if not seal_within_bounds(msg, sig):
            return evaluate_card({"sealed": False, "reason": "oversized_seal"}, claimed_model, slot)
        try:
            seal_verdict = verifier.verify_seal(master_key, nonce, msg, sig)
        except Exception as exc:
            return evaluate_card(
                {"sealed": False, "reason": f"verify:{type(exc).__name__}"}, claimed_model, slot
            )
        if not isinstance(seal_verdict, dict):
            seal_verdict = {"sealed": False, "reason": "verifier_bad_output"}
        return evaluate_card(
            seal_verdict, claimed_model, slot, have_kernel=bool(parsed.get("have_kernel"))
        )

    async def _binary_present(self, ctx: Context, binary_path: str) -> bool:
        try:
            result = await ctx.ssh.run(
                f"test -x {shlex.quote(binary_path)} && echo GPUSIG_PRESENT", timeout=15
            )
            return "GPUSIG_PRESENT" in (getattr(result, "stdout", "") or "")
        except Exception:
            return False

    def _filler_only_container(self, ctx: Context) -> str | None:
        rented_data = ctx.state.rented_data
        if not rented_data:
            return None
        filler_container = rented_data.get_filler_container(ctx.executor.uuid)
        rented_executor = rented_data.executors.get(ctx.executor.uuid)
        has_customer_rental = bool(rented_executor and rented_executor.pods)
        return filler_container if filler_container and not has_customer_rental else None

    def _skip(self, ctx: Context, why: str, extra_what: dict | None = None) -> CheckResult:
        what: dict = {"skipped": why}
        if extra_what:
            what.update(extra_what)
        event = render_message(Msg.SKIPPED, ctx=ctx, check_id=self.check_id, what=what)
        return CheckResult(passed=True, event=event)
