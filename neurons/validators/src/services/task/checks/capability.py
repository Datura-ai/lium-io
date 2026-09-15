from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

from core.config import settings

from ..messages import CapabilityMessages as Msg, render_message
from ..pipeline import CheckResult, Context

if TYPE_CHECKING:
    from protocol.vc_protocol.compute_requests import RentedExecutorsResponse
    from services.matrix_validation_service import ValidationResult

logger = logging.getLogger(__name__)


class CapabilityCheck:
    """Run the containerised GPU capability probe (nvidia-smi in Docker).

    This executes the same `validate_gpu_model_and_process_job` command used before, which
    verifies that containers can see the GPUs. Failing it previously zeroed the score, so
    keeping it prevents miners from hiding driver issues behind a good scrape.
    """

    check_id = "gpu.validate.capability"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        specs = ctx.state.specs
        if not specs:
            event = render_message(
                Msg.NO_SPECS,
                ctx=ctx,
                check_id=self.check_id,
            )
            return CheckResult(passed=False, event=event)

        filler_container = _get_filler_only_container(ctx)
        if filler_container:
            event = render_message(
                Msg.FILLER_SKIPPED,
                ctx=ctx,
                check_id=self.check_id,
                what={"filler_container": filler_container},
            )
            return CheckResult(passed=True, event=event)

        validation_service = ctx.services.validation

        # DAH-3011: a first, unscored verification sizes the matmul from a VRAM budget instead of
        # the whole card (same challenge/seal/UUID check). The keyword is only passed on that path
        # so the scored call is byte-for-byte today's.
        sizing = {"vram_budget_mb": settings.FIRST_PASS_MATMUL_VRAM_MB} if ctx.config.first_pass else {}

        result = None
        failure_reason = None
        # liumd phase 1: LocalVerifyCheck already ran this challenge through `POST /verify` and
        # judged it with evaluate_matmul_output — the same function the SSH path ends in. Only a
        # PASSING local result is consumed; anything else runs over SSH exactly as before.
        local = getattr(ctx.state, "local_verify", None)
        transport = "ssh"
        if local is not None and local.matmul is not None:
            result = local.matmul
            transport = "local_verify"
        else:
            try:
                result = await validation_service.validate_gpu_model_and_process_job(
                    ssh_client=ctx.ssh,
                    executor_info=ctx.executor,
                    default_extra=ctx.default_extra,
                    machine_spec=specs,
                    **sizing,
                )
            except Exception as exc:
                failure_reason = str(exc)

        if result and result.success:
            what: dict = {"metrics": result.metrics, "transport": transport}
            if sizing:
                what["first_pass_vram_budget_mb"] = sizing["vram_budget_mb"]
            event = render_message(
                Msg.VERIFY_OK,
                ctx=ctx,
                check_id=self.check_id,
                what=what,
            )
            # Carry FP32 TFLOPS metrics into pipeline state so ResultHandler can nest them
            # in the published specs. Only on success and only when present (fail-safe:
            # None leaves state untouched and no gpu_metrics key is ever published).
            updates: dict = {}
            if result.metrics is not None:
                updates["state"] = replace(ctx.state, gpu_metrics=result.metrics)
            return CheckResult(passed=True, event=event, updates=updates)

        # Build detailed failure information
        failure_details = {}
        if result is not None:
            failure_details = {
                "error": result.error_message,
                "expected_uuid": result.expected_uuid,
                "returned_uuid": result.returned_uuid,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "metrics": result.metrics,
            }
        elif failure_reason:
            failure_details = {"error": failure_reason}

        if _probe_gave_no_answer(result):
            lium_workload = await _lium_workload_live_now(ctx)
            if lium_workload is not None:
                event = render_message(
                    Msg.RENTED_SKIPPED,
                    ctx=ctx,
                    check_id=self.check_id,
                    what={
                        "workload": lium_workload.kind,
                        "containers": list(lium_workload.container_names),
                        "probe": failure_details,
                    },
                )
                # The fresh snapshot replaces the stale one, so RentalVerificationCheck and
                # GpuFaultProbeCheck later in the cycle see the same workload this waiver saw.
                return CheckResult(
                    passed=True,
                    event=event,
                    updates={"state": replace(ctx.state, rented_data=lium_workload.snapshot)},
                )

        template = Msg.VERIFY_TIMEOUT if result is not None and result.timed_out else Msg.VERIFY_FAILED
        event = render_message(
            template,
            ctx=ctx,
            check_id=self.check_id,
            what=failure_details,
        )
        return CheckResult(passed=False, event=event)


# What the probe answers when it never reached the UUID step: the wrapper prints "UUID:  None"
# after a failed cudaMalloc, and the service reports that as a mismatch against 'None'.
_NO_UUID = frozenset({"", "none", "null"})
_UUID_MISMATCH_ERROR_PREFIX = "UUID mismatch"


def _probe_gave_no_answer(result: ValidationResult | None) -> bool:
    """True when the probe timed out or came back without any UUID.

    A returned UUID that does not match is the anti-spoof case and is never waived: the expected
    UUID is a per-call nonce, so a genuine executor never answers a wrong one. An exception on
    the validator's side (result None) is not the card's doing either.
    """
    if result is None:
        return False
    if result.timed_out:
        return True
    if not (result.error_message or "").startswith(_UUID_MISMATCH_ERROR_PREFIX):
        return False
    return (result.returned_uuid or "").strip().lower() in _NO_UUID


@dataclass(frozen=True)
class _LiumWorkload:
    """A workload the backend says holds this node's cards right now: which kind (`filler` or
    `pod`), the container names it listed, sorted, and the snapshot it came from."""

    kind: Literal["filler", "pod"]
    container_names: tuple[str, ...]
    snapshot: RentedExecutorsResponse


async def _lium_workload_live_now(ctx: Context) -> _LiumWorkload | None:
    """Ask the backend whether a workload it started holds this node's cards right now.

    `rented_data` is read once, at cycle start. A filler the backend starts right after a rental
    closes, or a pod created during the cycle, is absent from that snapshot, so the probe runs
    against a busy card and cannot allocate (DAH-3480: 60 of 267 `GPU_VERIFY_FAILED` "Failed to
    allocate" rows in 48 h had a filler created after the snapshot and live at probe time; 0 fell
    inside a rental the snapshot knew about). DAH-2757 closed the same race for the GPU usage
    gate. Asked only after a failed probe, never on the healthy path; one call, 30 s timeout, no
    retry, and a backend that does not answer keeps the failure (fail closed).

    What counts is the backend's list, never the node's: a filler container it runs on the node
    (a default job, Lium's or the miner's own, both already exempt from the probe when the
    snapshot knows them), or a pod it lists (BROKEN and DELETING pods are not listed). The same
    two lists are what `TenantEnforcementCheck` and the filler skip above trust at cycle start,
    so this grants at most the one cycle the snapshot missed.
    """
    try:
        fresh = await ctx.services.backend.get_rented_executors_now()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Rented-executors re-read failed; the probe verdict stands: %s", exc)
        return None
    if fresh is None:
        return None
    executor_uuid = ctx.executor.uuid
    filler_containers = fresh.get_filler_containers(executor_uuid)
    if filler_containers:
        return _LiumWorkload(
            kind="filler", container_names=tuple(sorted(filler_containers)), snapshot=fresh
        )
    rented_executor = fresh.executors.get(executor_uuid)
    if rented_executor and rented_executor.pods:
        return _LiumWorkload(
            kind="pod",
            container_names=tuple(sorted(pod.container_name for pod in rented_executor.pods)),
            snapshot=fresh,
        )
    return None


def _get_filler_only_container(ctx: Context) -> str | None:
    rented_data = ctx.state.rented_data
    if not rented_data:
        return None

    filler_container = rented_data.get_filler_container(ctx.executor.uuid)
    rented_executor = rented_data.executors.get(ctx.executor.uuid)
    has_customer_rental = bool(rented_executor and rented_executor.pods)
    return filler_container if filler_container and not has_customer_rental else None
