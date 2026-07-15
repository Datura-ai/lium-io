from datetime import datetime, timezone

from core.docker_utils import collect_container_death_diagnostics
from core.utils import _m, get_extra_info, get_logger

from ..messages import FillerMiningHealthMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context

logger = get_logger(__name__)

# A wedge persists for hours and collect_container_death_diagnostics SSHes into the miner host, so we
# snap the worker log at most once per this window per executor rather than every validation cycle.
_CAPTURE_COOLDOWN_SECONDS = 3600
_CAPTURE_KEY = "filler_mining_health:last_capture:{executor_id}"


class FillerMiningHealthCheck:
    """Diagnostic (non-fatal): when a Lium filler is RUNNING but the GPU shows it isn't actually
    mining (dead worker, or a firmware-wedged card the worker no longer holds), snap the worker's
    `docker logs` so the in-container failure — which lives on the miner host, not in our logs — is
    captured in Loki. NEVER fails the miner: `passed` is always True and `fatal` is False.
    """

    check_id = "filler.mining.health"
    fatal = False

    async def run(self, ctx: Context) -> CheckResult:
        rented_data = ctx.state.rented_data
        filler_container = rented_data.get_filler_container(ctx.executor.uuid) if rented_data else None
        if not filler_container:
            return CheckResult(passed=True, event=render_message(Msg.NO_FILLER, ctx=ctx, check_id=self.check_id))

        if _filler_holds_gpu(filler_container, ctx.state.gpu_processes):
            return CheckResult(passed=True, event=render_message(Msg.MINING_OK, ctx=ctx, check_id=self.check_id))

        captured = await self._capture_worker_log(ctx, filler_container)
        return CheckResult(
            passed=True,
            event=render_message(
                Msg.NOT_MINING,
                ctx=ctx,
                check_id=self.check_id,
                what={"filler_container": filler_container, "log_captured": captured},
            ),
        )

    async def _capture_worker_log(self, ctx: Context, container: str) -> bool:
        if not await self._claim_capture_slot(ctx):
            return False
        try:
            diagnostics = await collect_container_death_diagnostics(ctx.ssh, container)
        except Exception as exc:  # a diagnostics failure must never fail the pipeline
            logger.warning(
                _m(
                    "Filler mining-health: worker log capture failed",
                    extra=get_extra_info({**ctx.default_extra, "container": container, "error": str(exc)}),
                )
            )
            return False
        logger.warning(
            _m(
                "Filler running but not mining — worker diagnostics",
                extra=get_extra_info({**ctx.default_extra, "container": container} | diagnostics.to_log_fields()),
            )
        )
        return True

    async def _claim_capture_slot(self, ctx: Context) -> bool:
        # Rate-limit per executor via a stored timestamp (the RedisService wrapper has no TTL set). A
        # redis hiccup must never block or fail the check, so on any error we skip the capture.
        key = _CAPTURE_KEY.format(executor_id=ctx.executor.uuid)
        now = datetime.now(timezone.utc).timestamp()
        try:
            last = await ctx.services.redis.get(key)
            if last is not None and now - float(last) < _CAPTURE_COOLDOWN_SECONDS:
                return False
            await ctx.services.redis.set(key, str(now))
        except Exception:
            return False
        return True


def _filler_holds_gpu(filler_container: str, gpu_processes: list[dict]) -> bool:
    # Mining => the filler container is the process actually holding the GPU. If no GPU process belongs
    # to it (dead worker, or a firmware-wedged card whose process is already gone), it isn't mining.
    return any(process.get("container_name") == filler_container for process in gpu_processes)
