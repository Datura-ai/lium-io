from __future__ import annotations

import asyncio
import secrets
import shlex
import time

from core.config import settings
from services.gpu_signature import evaluate_card, parse_result_line, summarize

from ..messages import GpuSignatureMessages as Msg, render_message
from ..pipeline import CheckResult, Context


class GpuSignatureCheck:
    """DAH-3137 — nonce-bound, sealed per-GPU hardware-signature challenge (observe-only).

    Runs the pre-placed executor-image binary (``bin/gpu_sig``, shipped in the
    image — NOT uploaded per check) once per claimed card, pinned via
    ``CUDA_VISIBLE_DEVICES``, with a fresh per-call nonce. Each result is
    HMAC-sealed with a key derived from the card's kernel-reported UUID
    (``/proc/driver/nvidia``, outside NVML), which the validator derives
    independently. The check verifies every card is present, sealed, fresh, and
    within its class's ground-truth envelope, then aggregates for count
    (device-selection failure, duplicate kernel UUID, or a blown aggregate
    wall-clock all flag a count/type spoof).

    Additive and observe-only: it logs a verdict and never changes the score.
    ``GPU_SIGNATURE_ENFORCEMENT_ENABLED`` currently only raises the event
    severity; wiring a score gate is a follow-up once the envelope is calibrated
    on real hardware (see ~/lium-ads/verification-binary/DESIGN.md §9).
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

        nonce = secrets.token_hex(32)
        master_key = settings.GPU_SIGNATURE_HMAC_KEY.encode("utf-8")
        claimed_model = ctx.state.gpu_model
        claimed_uuids = [detail.get("uuid", "") for detail in gpu_details]

        gate = asyncio.Semaphore(max(1, settings.GPU_SIGNATURE_MAX_CONCURRENT))

        async def run_slot(slot: int):
            async with gate:
                # nonce is hex and slot is int, so the command is injection-safe; the
                # binary path is the only variable and is shlex-quoted.
                cmd = (
                    f"CUDA_VISIBLE_DEVICES={slot} {shlex.quote(binary_path)} "
                    f"--nonce {nonce} --device 0"
                )
                try:
                    result = await ctx.ssh.run(
                        cmd, timeout=settings.GPU_SIGNATURE_TIMEOUT_SECONDS
                    )
                    stdout = getattr(result, "stdout", "") or ""
                except Exception as exc:  # timeout / channel error / device selection
                    return evaluate_card(
                        master_key, nonce, claimed_model, slot,
                        {"ok": False, "error": f"ssh:{type(exc).__name__}"},
                    )
                return evaluate_card(
                    master_key, nonce, claimed_model, slot, parse_result_line(stdout)
                )

        start = time.perf_counter()
        verdicts = await asyncio.gather(*(run_slot(slot) for slot in range(gpu_count)))
        elapsed = time.perf_counter() - start

        verdict = summarize(verdicts, elapsed, gpu_count)

        # Cross-check the /proc-sourced kernel UUIDs against the NVML-claimed set:
        # a userspace NVML shim (DAH-2662) shows up as a mismatch here.
        kernel_uuids = sorted(v.kernel_uuid for v in verdicts if v.kernel_uuid)
        claimed_sorted = sorted(uuid for uuid in claimed_uuids if uuid)
        uuid_mismatch = bool(kernel_uuids) and kernel_uuids != claimed_sorted

        node_ok = verdict.passed and not uuid_mismatch
        what = {
            "passed": node_ok,
            "reasons": verdict.reasons + (["kernel_vs_nvml_uuid_mismatch"] if uuid_mismatch else []),
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
        event = render_message(template, ctx=ctx, check_id=self.check_id, what=what)
        # Observe-only: never fail the pipeline or the score yet.
        return CheckResult(passed=True, event=event)

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
