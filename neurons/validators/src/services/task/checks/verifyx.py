from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from typing import Any, Literal

from core.config import settings
from core.utils import _m, get_extra_info
from services.verifyx_validation_service import NETWORK_GATE_TALLY, _is_speed_reading

from ..messages import VerifyXMessages as Msg, render_message
from ..pipeline import CheckResult, Context
from .network_ema import compute_ema

logger = logging.getLogger(__name__)

MIN_VERIFYX_EMA_DOWNLOAD_SPEED_MBPS = 100.0


@dataclass(frozen=True)
class ColdSampleRetry:
    """DAH-2959: a never-measured node's two download samples and the one the check kept.

    Attached to the event with `asdict`, so it serializes like the rest of `what_we_saw`.
    """

    first_download_speed_mbps: float | None
    retry_download_speed_mbps: float | None
    used: Literal["first", "retry"]


class VerifyXCheck:
    """Run the optional VerifyX hardware probe and update specs with its findings.

    This preserves the legacy feature flag behaviour: when VerifyX is enabled we block on
    its success, otherwise the check becomes a no-op. Documenting it here lets us debate
    the value of the external probe independently from the rest of the pipeline.
    """

    check_id = "gpu.validate.verifyx"
    fatal = True

    def _extract_errors(self, result) -> str | list | None:
        """Extract errors from result with clear priority: data.errors > result.error."""
        if result.data and result.data.get("errors"):
            return result.data["errors"]
        return result.error

    async def run(self, ctx: Context) -> CheckResult:
        if not ctx.config.verifyx_enabled:
            event = render_message(
                Msg.DISABLED,
                ctx=ctx,
                check_id=self.check_id,
            )
            return CheckResult(passed=True, event=event)
        verifyx_service = ctx.services.verifyx
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
            updated_specs = specs_with_last_known_verifyx_ema(ctx)
            event = render_message(
                Msg.FILLER_SKIPPED,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "filler_container": filler_container,
                    "ema_verifyx_download_speed": (
                        (updated_specs.get("network") or {}).get("ema_verifyx_download_speed")
                    ),
                },
            )
            return CheckResult(
                passed=True,
                event=event,
                updates={"state": replace(ctx.state, specs=updated_specs)},
            )

        # DAH-3011: a first, unscored verification writes less RAM/disk (the measurements are still
        # taken and published). The keyword is only passed on that path so the scored call is
        # byte-for-byte today's.
        sizing = (
            {"challenge_config_overrides": _first_pass_challenge_config()}
            if ctx.config.first_pass
            else {}
        )
        # liumd phase 1: LocalVerifyCheck already ran this challenge through `POST /verify` and
        # judged it with evaluate_verifyx_capture — the same function the SSH path ends in. Only a
        # PASSING local result is consumed; anything else (and the cold-sample retry below) runs
        # over SSH exactly as before.
        local = getattr(ctx.state, "local_verify", None)
        transport = "ssh"
        if local is not None and local.verifyx is not None:
            result = local.verifyx
            transport = "local_verify"
        else:
            result = await verifyx_service.validate_verifyx_and_process_job(
                shell=ctx.services.shell,
                executor_info=ctx.executor,
                default_extra=ctx.default_extra,
                machine_spec=specs,
                **sizing,
            )

        prev_ema = (
            ctx.state.rented_data.network_ema.get(ctx.executor.uuid)
            if ctx.state.rented_data
            else None
        )

        cold_sample_retry: ColdSampleRetry | None = None
        if _is_cold_sample_below_gate(ctx, result, prev_ema):
            # DAH-2959: a node's first sample decides its first cycle on its own (the EMA bootstraps
            # from it). 11 of 80 fresh nodes (2–5 Sep) measured 32–88 Mbps on that one sample — one
            # single-stream 260–500 MB CDN object, cold (the scrape's own speedtest read 0.7–1.7 Gbps
            # on two of the traced hosts, 104 Mbps on the third whose link the executor's on-boot
            # template pre-pull was sharing) — and passed the next cycle at 205–760 Mbps. One more
            # sample inside the same task costs at most one VerifyX run; the failure costs the
            # provider a whole cycle. Known hosts (any prior EMA) are not retried, and the gate is
            # unchanged: one of the two samples must clear it.
            retry = await verifyx_service.validate_verifyx_and_process_job(
                shell=ctx.services.shell,
                executor_info=ctx.executor,
                default_extra=ctx.default_extra,
                machine_spec=specs,
                **sizing,
            )
            first_speed = _download_speed(result)
            retry_speed = _download_speed(retry)
            use_retry = (
                retry.data is not None
                and bool(retry.data.get("success"))
                and (retry_speed or 0.0) > (first_speed or 0.0)
            )
            cold_sample_retry = ColdSampleRetry(
                first_download_speed_mbps=first_speed,
                retry_download_speed_mbps=retry_speed,
                used="retry" if use_retry else "first",
            )
            logger.info(
                _m(
                    "VerifyX cold first sample below the gate, re-measured once",
                    extra=get_extra_info({**ctx.default_extra, **asdict(cold_sample_retry)}),
                )
            )
            if use_retry:
                # The retry always runs over SSH, so a consumed local answer that loses to it is
                # no longer what the event describes.
                result = retry
                transport = "ssh"

        # Extract errors with clear priority: data.errors > result.error
        errors = self._extract_errors(result)

        if result.data and result.data.get("success"):
            base_specs = ctx.state.specs
            sanitized = _to_iso(result.data)
            verifyx_network = sanitized.get("network", {})
            speedtest_network = base_specs.get("network", {}) or {}
            updated_specs = dict(base_specs)
            updated_specs.update(
                {
                    "ram": sanitized.get("ram", updated_specs.get("ram")),
                }
            )

            # Always compute verifyx network EMA. A Cloudflare probe failure that fell back to
            # the package download feeds that number, never 0. A malformed reading never reaches
            # compute_ema: the previous EMA stands.
            if "network" not in updated_specs:
                updated_specs["network"] = {}
            download_speed = verifyx_network.get("download_speed")
            unavailable_readings: list[str] = []
            ema_download = _feed_ema(
                ctx,
                updated_specs["network"],
                "download",
                download_speed,
                prev_ema.ema_verifyx_download_speed if prev_ema else None,
                unavailable_readings,
                keep_previous_on_none=bool(verifyx_network.get("cloudflare_fallback")),
            )
            _feed_ema(
                ctx,
                updated_specs["network"],
                "upload",
                verifyx_network.get("upload_speed"),
                prev_ema.ema_verifyx_upload_speed if prev_ema else None,
                unavailable_readings,
            )

            # Update storage specs if storage is present. Merged rather than replaced: VerifyX
            # measures only its own df-derived fields, so overwriting the dict wholesale would drop
            # the scrape's docker usage breakdown on every node where VerifyX runs.
            if "hard_disk" in sanitized:
                verifyx_hard_disk = sanitized.get("hard_disk")
                scraped_hard_disk = updated_specs.get("hard_disk")
                if isinstance(verifyx_hard_disk, dict) and isinstance(scraped_hard_disk, dict):
                    updated_specs["hard_disk"] = {**scraped_hard_disk, **verifyx_hard_disk}
                else:
                    updated_specs["hard_disk"] = verifyx_hard_disk

            event = render_message(
                Msg.VERIFY_SUCCESS,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "verifyx_success": True,
                    "verifyx_network_success": verifyx_network.get("success"),
                    "network": speedtest_network,
                    "transport": transport,
                },
            )
            if errors:
                event.what_we_saw["errors"] = errors
            if sizing:
                event.what_we_saw["first_pass_challenge_config"] = sizing[
                    "challenge_config_overrides"
                ]
            if cold_sample_retry is not None:
                event.what_we_saw["cold_sample_retry"] = asdict(cold_sample_retry)
            if unavailable_readings:
                event.what_we_saw["unavailable_speed_readings"] = unavailable_readings
            network_gate = _network_gate(verifyx_network, prev_ema, ema_download)
            if network_gate is not None:
                event.what_we_saw["network_gate"] = network_gate
                NETWORK_GATE_TALLY.record(
                    ema_download,
                    MIN_VERIFYX_EMA_DOWNLOAD_SPEED_MBPS,
                    fallback=bool(verifyx_network.get("cloudflare_fallback")),
                )

            updated_state = replace(ctx.state, specs=updated_specs)

            never_measured = prev_ema is None or prev_ema.ema_verifyx_download_speed is None
            if ema_download is None:
                # A malformed download reading on a never-measured host: there is no EMA to keep
                # and none to gate on. The event says why; the next sample seeds it.
                return CheckResult(passed=True, event=event, updates={"state": updated_state})
            if (
                ema_download < MIN_VERIFYX_EMA_DOWNLOAD_SPEED_MBPS
                and ctx.config.first_pass
                and never_measured
            ):
                # DAH-3011: the first pass is never scored, and a fresh node's single cold sample fails
                # this gate 14 % of the time (DAH-2959). Publish the raw sample, leave the EMA unseeded
                # so the first SCORED cycle bootstraps from a warm sample and enforces the gate, and say
                # so on the event. A known host, or any scored cycle, never takes this branch.
                deferred_network = {
                    key: value
                    for key, value in updated_specs["network"].items()
                    if key not in ("ema_verifyx_download_speed", "ema_verifyx_upload_speed")
                }
                updated_state = replace(
                    ctx.state, specs={**updated_specs, "network": deferred_network}
                )
                event.what_we_saw["bandwidth_gate"] = "deferred_to_first_scored_cycle"
                event.what_we_saw["first_sample_download_speed_mbps"] = download_speed
                event.what_we_saw["min_download_speed_mbps"] = MIN_VERIFYX_EMA_DOWNLOAD_SPEED_MBPS
                return CheckResult(passed=True, event=event, updates={"state": updated_state})

            if ema_download < MIN_VERIFYX_EMA_DOWNLOAD_SPEED_MBPS:
                slow_event = render_message(
                    Msg.VERIFY_FAILED_NETWORK_SPEED_TOO_SLOW,
                    ctx=ctx,
                    check_id=self.check_id,
                    what={
                        "ema_verifyx_download_speed": ema_download,
                        "min_download_speed_mbps": MIN_VERIFYX_EMA_DOWNLOAD_SPEED_MBPS,
                    },
                )
                if cold_sample_retry is not None:
                    slow_event.what_we_saw["cold_sample_retry"] = asdict(cold_sample_retry)
                if network_gate is not None:
                    slow_event.what_we_saw["network_gate"] = network_gate
                return CheckResult(passed=False, event=slow_event, updates={"state": updated_state})

            return CheckResult(
                passed=True,
                event=event,
                updates={
                    "state": updated_state,
                },
            )

        NETWORK_GATE_TALLY.record_probe_failed()

        # Ensure we have an error message for the failure case
        error_message = errors or "Unknown errors"

        diagnostics = getattr(result, "diagnostics", None) or {}
        template = _FAILURE_TEMPLATE_BY_CLASS.get(
            diagnostics.get("failure_class"), Msg.VERIFY_FAILED
        )
        what: dict[str, Any] = {"errors": error_message, **diagnostics}

        event = render_message(
            template,
            ctx=ctx,
            check_id=self.check_id,
            what=what,
        )

        return CheckResult(passed=False, event=event)


_FAILURE_TEMPLATE_BY_CLASS = {
    "SSH_TRANSPORT": Msg.VERIFY_FAILED_SSH_TRANSPORT,
    "EXECUTOR_CRASH": Msg.VERIFY_FAILED_EXECUTOR_CRASH,
    "EMPTY_RESPONSE": Msg.VERIFY_FAILED_EMPTY_RESPONSE,
    "CIPHER_REJECTED": Msg.VERIFY_FAILED_CIPHER_REJECTED,
}


def _first_pass_challenge_config() -> dict[str, int]:
    return {
        "memory_max_test_gb": settings.FIRST_PASS_VERIFYX_MEMORY_MAX_TEST_GB,
        "storage_throughput_test_gb": settings.FIRST_PASS_VERIFYX_STORAGE_TEST_GB,
    }


def _download_speed(result) -> float | None:
    if not result.data:
        return None
    speed = (result.data.get("network") or {}).get("download_speed")
    return speed if _is_speed_reading(speed) else None


def _feed_ema(
    ctx: Context,
    network: dict,
    direction: Literal["download", "upload"],
    reading: object,
    prev: float | None,
    unavailable: list[str],
    keep_previous_on_none: bool = False,
) -> float | None:
    """Publish one VerifyX speed reading and its EMA into `network`; return the EMA.

    None is a failed measurement: the EMA takes 0.0 so repeated failures decay it toward
    exclusion, except keep_previous_on_none (a Cloudflare fallback with no package reading).
    A reading `_is_speed_reading` rejects is malformed: it never reaches `compute_ema`.
    """
    if reading is None and keep_previous_on_none:
        unavailable.append(direction)
        if prev is not None:
            network[f"ema_verifyx_{direction}_speed"] = prev
        return prev
    if reading is not None and not _is_speed_reading(reading):
        unavailable.append(direction)
        logger.warning(
            _m(
                f"VerifyX {direction} speed reading unavailable, previous EMA kept",
                extra=get_extra_info(
                    {
                        **ctx.default_extra,
                        "direction": direction,
                        "reading_type": type(reading).__name__,
                        f"ema_verifyx_{direction}_speed": prev,
                    }
                ),
            )
        )
        if prev is not None:
            network[f"ema_verifyx_{direction}_speed"] = prev
        return prev
    if reading is not None:
        network[f"verifyx_{direction}_speed"] = reading
    ema = compute_ema(prev, reading if reading is not None else 0.0)
    network[f"ema_verifyx_{direction}_speed"] = ema
    return ema


def _ema_if_gated(prev: float | None, reading: object) -> float | None:
    """The download EMA `_feed_ema` would store if `reading` were the gated one."""
    if reading is not None and not _is_speed_reading(reading):
        return prev
    return compute_ema(prev, reading if reading is not None else 0.0)


def _network_gate(verifyx_network: dict, prev_ema, ema_gated: float | None) -> dict:
    """DAH-2774 record: the floor read against the package and the capacity reading."""
    prev = prev_ema.ema_verifyx_download_speed if prev_ema else None
    package = verifyx_network.get("package_download_speed")
    capacity = verifyx_network.get("capacity_download_speed")
    return {
        "floor_mbps": MIN_VERIFYX_EMA_DOWNLOAD_SPEED_MBPS,
        "package_download_speed": package,
        "capacity_download_speed": capacity,
        "ema_package": _ema_if_gated(prev, package),
        "ema_capacity": ema_gated,
        "cloudflare_fallback": bool(verifyx_network.get("cloudflare_fallback")),
    }


def _is_cold_sample_below_gate(ctx: Context, result, prev_ema) -> bool:
    """The one case DAH-2959 re-measures: a never-measured executor whose probe otherwise passed
    but whose single download sample would fail the EMA gate on its own.

    `rented_data` present is required — without the backend's answer every executor looks
    never-measured, and a backend blip must not double VerifyX for the whole fleet.
    """
    if not settings.VERIFYX_COLD_SAMPLE_RETRY_ENABLED:
        return False
    if ctx.state.rented_data is None:
        return False
    if prev_ema is not None and prev_ema.ema_verifyx_download_speed is not None:
        return False
    if not result.data or not result.data.get("success"):
        return False
    speed = _download_speed(result)
    return speed is None or speed < MIN_VERIFYX_EMA_DOWNLOAD_SPEED_MBPS


def _get_filler_only_container(ctx: Context) -> str | None:
    rented_data = ctx.state.rented_data
    if not rented_data:
        return None

    filler_container = rented_data.get_filler_container(ctx.executor.uuid)
    rented_executor = rented_data.executors.get(ctx.executor.uuid)
    has_customer_rental = bool(rented_executor and rented_executor.pods)
    return filler_container if filler_container and not has_customer_rental else None


def specs_with_last_known_verifyx_ema(ctx: Context) -> dict:
    specs = dict(ctx.state.specs or {})
    rented_data = ctx.state.rented_data
    network_ema = rented_data.network_ema.get(ctx.executor.uuid) if rented_data else None
    if not network_ema or network_ema.ema_verifyx_download_speed is None:
        return specs

    network = dict(specs.get("network") or {})
    network.setdefault("ema_verifyx_download_speed", network_ema.ema_verifyx_download_speed)
    if network_ema.ema_verifyx_upload_speed is not None:
        network.setdefault("ema_verifyx_upload_speed", network_ema.ema_verifyx_upload_speed)
    specs["network"] = network
    return specs


def _to_iso(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _to_iso(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_iso(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_iso(v) for v in value)
    return value
