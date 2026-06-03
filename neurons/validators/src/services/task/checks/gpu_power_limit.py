from __future__ import annotations

from ..messages import GpuPowerLimitMessages as Msg, render_message
from ..pipeline import CheckResult, Context

MIN_POWER_LIMIT_RATIO = 0.8


def _to_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class GpuPowerLimitCheck:
    """Gate validation on the current GPU power cap relative to the default cap."""

    check_id = "gpu.validate.power_limit"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        measurements = []
        incomplete = []
        rejected = []

        for index, detail in enumerate(ctx.state.gpu_details or []):
            current_limit = _to_float(detail.get("power_limit"))
            default_limit = _to_float(detail.get("power_default_limit"))
            max_limit = _to_float(detail.get("power_max_limit"))
            measurement = {
                "index": index,
                "name": detail.get("name"),
                "uuid": detail.get("uuid"),
                "power_limit": current_limit,
                "power_default_limit": default_limit,
                "power_max_limit": max_limit,
            }

            if current_limit is None or default_limit is None or default_limit <= 0:
                incomplete.append(measurement)
                continue

            ratio = current_limit / default_limit
            measurement["power_limit_ratio"] = round(ratio, 4)
            measurements.append(measurement)

            if ratio < MIN_POWER_LIMIT_RATIO:
                rejected.append(measurement)

        if rejected:
            event = render_message(
                Msg.LIMIT_BELOW_DEFAULT,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "threshold": MIN_POWER_LIMIT_RATIO,
                    "rejected_gpus": rejected,
                    "measurements": measurements,
                    "incomplete_gpus": incomplete,
                },
            )
            return CheckResult(
                passed=False,
                event=event,
                updates={
                    "score": 0.0,
                    "job_score": 0.0,
                    "score_warning": (
                        " WARNING: GPU power limit is below 80% of the default power limit"
                    ),
                },
            )

        if incomplete:
            event = render_message(
                Msg.DATA_INCOMPLETE,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "threshold": MIN_POWER_LIMIT_RATIO,
                    "measurements": measurements,
                    "incomplete_gpus": incomplete,
                },
            )
            return CheckResult(passed=True, event=event)

        event = render_message(
            Msg.LIMIT_OK,
            ctx=ctx,
            check_id=self.check_id,
            what={"threshold": MIN_POWER_LIMIT_RATIO, "measurements": measurements},
        )
        return CheckResult(passed=True, event=event)
