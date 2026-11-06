from __future__ import annotations

import asyncio
import copy
import logging
import secrets
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from typing import Any, Literal

from protocol.vc_protocol.compute_requests import NetworkEMA

from core.config import settings
from core.utils import _m, get_extra_info
from services.redis_service import (
    VERIFYX_STALE_RUNS_TTL_SECONDS,
    create_started_at_key,
    verifyx_stale_runs_key,
)
from services.verifyx_validation_service import (
    NETWORK_GATE_TALLY,
    VERIFYX_COMMAND_TIMEOUT_SECONDS,
    NetworkGateTally,
    _is_positive_number,
    _is_speed_reading,
)

from ..messages import VerifyXMessages as Msg, render_message
from ..pipeline import CheckResult, Context
from .disk_space import DiskSpaceCheck
from .network_ema import compute_ema
from .task_time import task_seconds_left

logger = logging.getLogger(__name__)

MIN_VERIFYX_EMA_DOWNLOAD_SPEED_MBPS = 100.0
MIN_VERIFYX_EMA_UPLOAD_SPEED_MBPS = 10.0
_EMA_KEYS = ("ema_verifyx_download_speed", "ema_verifyx_upload_speed")

# A host can bring on a real Cloudflare 429 by flooding speed.cloudflare.com from its own IP just
# before each probe. So a 429 keeps the upload EMA for at most this many probes in a row; after
# that the probe's 0.0 is fed. One probe per job cycle (BLOCKS_FOR_JOB, ~15 min): 8 is ~2 h,
# well past Cloudflare's Retry-After. Per process, so a validator restart starts the count again.
# Keyed by (miner hotkey, executor uuid): two miners can list the same executor uuid.
MAX_KEPT_UPLOAD_PROBES = 8
_kept_upload_probes: dict[tuple[str, str], int] = {}
# The same host can make the Cloudflare download fail too (block speed.cloudflare.com), and then
# the gated number is the package download, a fixed file a LAN cache can serve. Probes in a row
# gated on the package reading are counted the same way; past this many each one is logged, and
# the missing capacity reading is fed as the 0 it is.
MAX_PACKAGE_FALLBACK_PROBES = 8
_package_fallback_probes: dict[tuple[str, str], int] = {}
# The cold-sample retry runs last, before scoring, with the time the other checks left less this
# margin for scoring, publishing and cleanup; with less than MIN_COLD_RETRY_SECONDS of that it is skipped.
MIN_COLD_RETRY_SECONDS = 240
COLD_RETRY_MARGIN_SECONDS = 60
# cap on the rental-state re-read before the retry; a slower answer counts as unknown (busy)
RENTAL_STATE_TIMEOUT_SECONDS = 30
# during the retry only the per-executor pending-pod marker is re-read, every interval; the backend's
# fleet-wide rental list is read once, before the retry, so concurrent retries do not multiply its load
RENTAL_WATCH_INTERVAL_SECONDS = 5
# the pre-retry rental lookup shares one backend read among all executors (single flight), reused
# this long: N cold executors in a cycle cost one fleet-wide read, not N
RENTED_NOW_TTL_SECONDS = 30
# cut-off runs whose cleanup failed: saving one is capped so the first result still returns in time,
# and reaping them before the next run is bounded in count and time
STALE_RUN_SAVE_TIMEOUT_SECONDS = 10
STALE_RUNS_KEPT = 4
STALE_RUNS_REAPED_PER_CYCLE = 2
STALE_RUNS_REAP_BUDGET_SECONDS = 25
_rented_now: tuple[Any, float, float, asyncio.Future] | None = None  # (backend, monotonic, epoch, read)


async def _rented_executors_now_shared(backend) -> tuple[Any, float]:
    """(`backend.get_rented_executors_now()`, epoch time the read began), one call in flight and
    reused for RENTED_NOW_TTL_SECONDS by every caller. A failed or empty answer is not reused."""
    global _rented_now
    now = time.monotonic()
    cached = _rented_now
    stale = (
        cached is None
        or cached[0] is not backend
        or now - cached[1] > RENTED_NOW_TTL_SECONDS
        or (cached[3].done() and (cached[3].cancelled() or cached[3].exception() or cached[3].result() is None))
    )
    if stale:
        cached = _rented_now = (
            backend, now, time.time(), asyncio.ensure_future(backend.get_rented_executors_now())
        )
    # shield: one caller's timeout must not cancel the read the others wait on
    return await asyncio.shield(cached[3]), cached[2]
# DAH-5742: an idle filler-only node keeps its stored EMA for at most this many skips in a row (~24 h of
# job cycles); then it is probed once, so a slower link than the stored figure shows up.
# shortcut: per process, so a validator restart starts the count again; a backend-held measured-at would survive it.
MAX_FILLER_SKIPS = 96
# (skips, the stored EMA they reused): the count restarts only once the backend stores a new reading, so a failed
# day-probe (stored only on a passing cycle) probes again next cycle instead of skipping another day on the old speed
_filler_skips: dict[tuple[str, str], tuple[int, tuple]] = {}


@dataclass(frozen=True)
class ColdSampleRetry:
    """DAH-2959: a never-measured node's two download samples and the one the check kept.

    Attached to the event with `asdict`, so it serializes like the rest of `what_we_saw`.
    """

    first_download_speed_mbps: float | None
    retry_download_speed_mbps: float | None
    used: Literal["first", "retry"]
    timed_out: bool = False


@dataclass(frozen=True)
class ColdRetryPending:
    """What VerifyXCheck leaves for VerifyXColdRetryCheck: how to re-run the probe, the first
    sample, and the specs it was judged on. Exactly one of the two samples is judged for good
    (counters, tally, specs), there."""

    sizing: dict
    prev_ema: NetworkEMA | None
    first_download_speed_mbps: float | None
    first_result: Any
    transport: str
    base_specs: dict


class VerifyXCheck:
    """Run the VerifyX hardware probe and update specs with its findings.

    The check blocks on the probe's success. It always runs (DAH-4328 removed the
    ENABLE_VERIFYX setting, on everywhere since Sep 2025); only an idle filler-only node
    with a stored download EMA and an upload EMA that is absent or at the floor skips it,
    keeping that EMA.
    """

    check_id = "gpu.validate.verifyx"
    # a replay would advance this cycle's counters a second time
    replays_after_session_drop = False
    fatal = True

    def _extract_errors(self, result) -> str | list | None:
        """Extract errors from result with clear priority: data.errors > result.error."""
        if result.data and result.data.get("errors"):
            return result.data["errors"]
        return result.error

    async def run(self, ctx: Context) -> CheckResult:
        verifyx_service = ctx.services.verifyx
        specs = ctx.state.specs
        if not specs:
            event = render_message(
                Msg.NO_SPECS,
                ctx=ctx,
                check_id=self.check_id,
            )
            return CheckResult(passed=False, event=event)

        await _reap_stale_runs(ctx)
        filler_container = _get_filler_only_container(ctx)
        stored_ema = _stored_network_ema(ctx)
        # A never-measured node is probed even under a filler: with no EMA to reuse it would
        # score 0 every cycle and stay hidden for as long as the filler runs.
        # A stored upload EMA below the floor is probed too: skipping would pass it while the
        # filler runs, and a fresh sample is what lets it recover.
        skip_key = (ctx.miner_hotkey, ctx.executor.uuid)
        stored = (
            (stored_ema.ema_verifyx_download_speed, stored_ema.ema_verifyx_upload_speed) if stored_ema else None
        )
        skips, skipped_on = _filler_skips.get(skip_key, (0, stored))
        if skipped_on != stored:
            skips = 0
        reusable = (
            filler_container
            and stored_ema
            and stored_ema.ema_verifyx_download_speed is not None
            and (
                stored_ema.ema_verifyx_upload_speed is None
                or stored_ema.ema_verifyx_upload_speed >= MIN_VERIFYX_EMA_UPLOAD_SPEED_MBPS
            )
        )
        # a day of reuse is up: this probe refreshes the network reading only (r4472607210)
        filler_refresh = bool(reusable) and skips >= MAX_FILLER_SKIPS
        if reusable and not filler_refresh:
            _filler_skips[skip_key] = (skips + 1, stored)
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
            else {"challenge_config_overrides": FILLER_REFRESH_CHALLENGE_CONFIG}
            if filler_refresh
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

        if not _is_cold_sample_below_gate(ctx, result, prev_ema):
            return self._judge(ctx, result, transport=transport, sizing=sizing, prev_ema=prev_ema)
        # DAH-2959: a node's first sample decides its first cycle on its own (the EMA bootstraps
        # from it). 11 of 80 fresh nodes (2–5 Sep) measured 32–88 Mbps on that one sample — one
        # single-stream 260–500 MB CDN object, cold (the scrape's own speedtest read 0.7–1.7 Gbps
        # on two of the traced hosts, 104 Mbps on the third whose link the executor's on-boot
        # template pre-pull was sharing) — and passed the next cycle at 205–760 Mbps. One more
        # sample inside the same task costs at most one VerifyX run; the failure costs the
        # provider a whole cycle. Known hosts (any prior EMA) are not retried, and the gate is
        # unchanged: one of the two samples must clear it.
        # DAH-5462: the second sample is taken by VerifyXColdRetryCheck, last before scoring, so it
        # only ever uses time the other checks left. Until then the first sample's specs stand,
        # uncounted; the one sample used is judged there, and a gate it fails fails the node there.
        base_specs = copy.deepcopy(_judged_specs(ctx.state.specs))  # _judge edits network in place
        judged = self._judge(
            ctx, result, transport=transport, sizing=sizing, prev_ema=prev_ema, counted=False
        )
        pending = ColdRetryPending(
            sizing=sizing,
            prev_ema=prev_ema,
            first_download_speed_mbps=_download_speed(result),
            first_result=result,
            transport=transport,
            base_specs=base_specs,
        )
        state = (judged.updates or {}).get("state", ctx.state)
        judged.event.what_we_saw["cold_sample_retry"] = "pending"
        return CheckResult(
            passed=True,
            event=judged.event,
            updates={**(judged.updates or {}), "state": replace(state, verifyx_cold_retry=pending)},
        )

    def _judge(
        self,
        ctx: Context,
        result,
        *,
        transport: str,
        sizing: dict,
        prev_ema: NetworkEMA | None,
        cold_sample_retry: ColdSampleRetry | None = None,
        counted: bool = True,
        tally: NetworkGateTally | None = None,
    ) -> CheckResult:
        """Judge one VerifyX result: the specs it updates, the EMA it feeds and the download gate.
        `counted=False` leaves the per-host probe counters and the gate tally as they were."""
        if not counted:
            key = (ctx.miner_hotkey, ctx.executor.uuid)
            saved = [(c, c.get(key)) for c in (_package_fallback_probes, _kept_upload_probes)]
            try:
                return self._judge(
                    ctx,
                    result,
                    transport=transport,
                    sizing=sizing,
                    prev_ema=prev_ema,
                    cold_sample_retry=cold_sample_retry,
                    tally=NetworkGateTally(),
                )
            finally:
                for counter, value in saved:
                    if value is None:
                        counter.pop(key, None)
                    else:
                        counter[key] = value
        tally = NETWORK_GATE_TALLY if tally is None else tally
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

            # Each direction keeps its previous EMA when Cloudflare, not the host, failed it (see
            # _fallback_upload_reading); a download fallback feeds the package reading, never 0.
            if "network" not in updated_specs:
                updated_specs["network"] = {}
            download_speed = verifyx_network.get("download_speed")
            cloudflare_fallback = bool(verifyx_network.get("cloudflare_fallback"))
            package_gated = _is_package_reading_gated(ctx, cloudflare_fallback)
            if cloudflare_fallback and not package_gated:
                capacity = verifyx_network.get("capacity_download_speed")
                download_speed = capacity if _is_positive_number(capacity) else 0.0
            upload_blocked = _upload_kept_within_cap(
                ctx, bool(verifyx_network.get("cloudflare_upload_fallback"))
            )
            unavailable_readings: list[str] = []
            ema_download = _feed_ema(
                ctx,
                updated_specs["network"],
                "download",
                download_speed,
                prev_ema.ema_verifyx_download_speed if prev_ema else None,
                unavailable_readings,
                keep_previous_on_none=package_gated,
            )
            upload_reading = _fallback_upload_reading(
                verifyx_network.get("upload_speed"), upload_blocked
            )
            prev_upload = prev_ema.ema_verifyx_upload_speed if prev_ema else None
            # A missing reading on a never-measured upload leaves it unseeded, not a synthetic 0.0.
            ema_upload = _feed_ema(
                ctx,
                updated_specs["network"],
                "upload",
                upload_reading,
                _upload_ema_base(prev_upload, upload_reading),
                unavailable_readings,
                keep_previous_on_none=upload_blocked or prev_upload is None,
            )

            # Update storage specs if storage is present. Merged rather than replaced: VerifyX
            # measures only its own df-derived fields, so overwriting the dict wholesale would drop
            # the scrape's docker usage breakdown on every node where VerifyX runs.
            # The scrape's own total/used/free win: they count the filler's preallocated run volume
            # as free (a rental removes it), and VerifyX's raw df put it back as used, so an idle
            # node listed terabytes less free disk until the next cycle without VerifyX.
            if "hard_disk" in sanitized:
                verifyx_hard_disk = sanitized.get("hard_disk")
                scraped_hard_disk = updated_specs.get("hard_disk")
                if isinstance(verifyx_hard_disk, dict) and isinstance(scraped_hard_disk, dict):
                    updated_specs["hard_disk"] = {**verifyx_hard_disk, **scraped_hard_disk}
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
            network_gate = _network_gate(verifyx_network, prev_ema, ema_download, ema_upload)
            if network_gate is not None:
                event.what_we_saw["network_gate"] = network_gate
                tally.record(
                    ema_download,
                    MIN_VERIFYX_EMA_DOWNLOAD_SPEED_MBPS,
                    fallback=cloudflare_fallback,
                )

            updated_state = replace(ctx.state, specs=updated_specs)

            gate = download_gate(ema_download, prev_ema, first_pass=ctx.config.first_pass)
            if gate == "unseeded":
                # A malformed download reading on a never-measured host: there is no EMA to keep
                # and none to gate on. The event says why; the next sample seeds it.
                return CheckResult(passed=True, event=event, updates={"state": updated_state})
            if gate == "deferred":
                # DAH-3011: the first pass is never scored, and a fresh node's single cold sample fails
                # this gate 14 % of the time (DAH-2959). Publish the raw sample, leave the EMA unseeded
                # so the first SCORED cycle bootstraps from a warm sample and enforces the gate, and say
                # so on the event. A known host, or any scored cycle, never takes this branch.
                deferred_network = {
                    key: value
                    for key, value in updated_specs["network"].items()
                    if key not in _EMA_KEYS
                }
                updated_state = replace(
                    ctx.state, specs={**updated_specs, "network": deferred_network}
                )
                event.what_we_saw["bandwidth_gate"] = "deferred_to_first_scored_cycle"
                event.what_we_saw["first_sample_download_speed_mbps"] = download_speed
                event.what_we_saw["min_download_speed_mbps"] = MIN_VERIFYX_EMA_DOWNLOAD_SPEED_MBPS
                return CheckResult(passed=True, event=event, updates={"state": updated_state})

            if gate == "too_slow":
                slow_event = render_message(
                    Msg.VERIFY_FAILED_NETWORK_SPEED_TOO_SLOW,
                    ctx=ctx,
                    check_id=self.check_id,
                    what={
                        "direction": "download",
                        "ema_verifyx_download_speed": ema_download,
                        "min_download_speed_mbps": MIN_VERIFYX_EMA_DOWNLOAD_SPEED_MBPS,
                    },
                )
                if cold_sample_retry is not None:
                    slow_event.what_we_saw["cold_sample_retry"] = asdict(cold_sample_retry)
                if network_gate is not None:
                    slow_event.what_we_saw["network_gate"] = network_gate
                return CheckResult(passed=False, event=slow_event, updates={"state": updated_state})

            if upload_too_slow(
                ema_upload, upload_reading, prev_ema, first_pass=ctx.config.first_pass
            ):
                slow_event = render_message(
                    Msg.VERIFY_FAILED_NETWORK_SPEED_TOO_SLOW,
                    ctx=ctx,
                    check_id=self.check_id,
                    what={
                        "direction": "upload",
                        "ema_verifyx_upload_speed": ema_upload,
                        "min_upload_speed_mbps": MIN_VERIFYX_EMA_UPLOAD_SPEED_MBPS,
                    },
                )
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

        tally.record_probe_failed()

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


# DAH-5742 (r4472607210): the RAM and disk a filler holds are ours (lium-platform keeps a 50 GiB disk and 4 GiB RAM
# reserve beside it), so the day's refresh tests them at 1 GB and keeps the network legs and the EMA gates as they are
FILLER_REFRESH_CHALLENGE_CONFIG = {
    "memory_min_test_gb": 1,
    "memory_max_test_gb": 1,
    "storage_min_available_gb": 1,
    "storage_throughput_test_gb": 1,
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


def gated_download_ema(verifyx_network: dict, prev_ema: NetworkEMA | None) -> float | None:
    """The download EMA the gate reads for one judged probe: what `_feed_ema` stores for the
    download direction, computed from the stored EMA without writing anything."""
    prev = prev_ema.ema_verifyx_download_speed if prev_ema else None
    reading = verifyx_network.get("download_speed")
    if reading is None and verifyx_network.get("cloudflare_fallback"):
        return prev
    return _ema_if_gated(prev, reading)


def download_gate(
    ema_download: float | None, prev_ema: NetworkEMA | None, *, first_pass: bool
) -> Literal["unseeded", "deferred", "too_slow", "pass"]:
    """The download-EMA floor's decision. `unseeded`, `deferred` and `pass` pass the check.

    `deferred` (DAH-3011): a first pass on a never-measured host below the floor; the EMA is left
    unseeded so the first scored cycle enforces it."""
    if ema_download is None:
        return "unseeded"
    if ema_download >= MIN_VERIFYX_EMA_DOWNLOAD_SPEED_MBPS:
        return "pass"
    never_measured = prev_ema is None or prev_ema.ema_verifyx_download_speed is None
    if first_pass and never_measured:
        return "deferred"
    return "too_slow"


def upload_too_slow(
    ema_upload: float | None,
    upload_reading: object,
    prev_ema: NetworkEMA | None,
    *,
    first_pass: bool,
) -> bool:
    """The upload-EMA floor. A 0.0 is a judged sample (a host can block its own upload), and a
    missing, malformed or Cloudflare-429 reading judges the EMA it leaves, so an established
    below-floor upload keeps failing. Only a never-measured upload with no usable sample, or on a
    first pass, is not judged."""
    if ema_upload is None or ema_upload >= MIN_VERIFYX_EMA_UPLOAD_SPEED_MBPS:
        return False
    never_measured = prev_ema is None or prev_ema.ema_verifyx_upload_speed is None
    return not (never_measured and (first_pass or not _is_speed_reading(upload_reading)))


def verifyx_verdict(
    result_data: dict | None, prev_ema: NetworkEMA | None, *, first_pass: bool
) -> bool:
    """Whether `VerifyXCheck.run` passes one judged probe result (`VerifyXResponse.data`) against
    the stored EMA `prev_ema`, with no state, EMA or tally written. The cold-sample retry, which
    takes a second sample, is not part of it."""
    if not (result_data and result_data.get("success")):
        return False
    network = result_data.get("network") or {}
    ema_download = gated_download_ema(network, prev_ema)
    if download_gate(ema_download, prev_ema, first_pass=first_pass) == "too_slow":
        return False
    # The verdict holds no count: a 429 is read as within MAX_KEPT_UPLOAD_PROBES, and the shadow
    # skips the probes past it (liumd_shadow "upload_fallback_cap").
    upload_fallback = bool(network.get("cloudflare_upload_fallback"))
    upload = _fallback_upload_reading(network.get("upload_speed"), upload_fallback)
    prev_upload = prev_ema.ema_verifyx_upload_speed if prev_ema else None
    kept = upload is None and (upload_fallback or prev_upload is None)
    ema_upload = prev_upload if kept else _ema_if_gated(_upload_ema_base(prev_upload, upload), upload)
    return not upload_too_slow(ema_upload, upload, prev_ema, first_pass=first_pass)


def _upload_ema_base(prev_upload: float | None, reading: object) -> float | None:
    """The stored upload EMA a measured reading averages with. The backend stores a node's specs only
    on a passing cycle, so a stored upload EMA below the floor predates it (or can never be raised by
    a failing cycle: 2 Mbps stays 2, and a steady 12 Mbps computes 7 every cycle, review 5704509865).
    Such an EMA is dropped and the reading judged alone; any stored EMA at the floor is kept."""
    if prev_upload is not None and prev_upload < MIN_VERIFYX_EMA_UPLOAD_SPEED_MBPS and _is_speed_reading(reading):
        return None
    return prev_upload


def _upload_kept_within_cap(ctx: Context, upload_blocked: bool) -> bool:
    """`upload_blocked`, until this host's upload EMA has been kept MAX_KEPT_UPLOAD_PROBES probes
    in a row; a probe that does not keep it starts the count again."""
    key = (ctx.miner_hotkey, ctx.executor.uuid)
    if not upload_blocked:
        _kept_upload_probes.pop(key, None)
        return False
    kept = _kept_upload_probes.get(key, 0) + 1
    _kept_upload_probes[key] = kept
    if kept <= MAX_KEPT_UPLOAD_PROBES:
        return True
    logger.warning(
        _m(
            "VerifyX upload EMA kept too many probes in a row on Cloudflare's 429, feeding the 0",
            extra=get_extra_info({**ctx.default_extra, "kept_upload_probes": kept}),
        )
    )
    return False


def _is_package_reading_gated(ctx: Context, cloudflare_fallback: bool) -> bool:
    """Whether this probe's package reading is the gated one: `cloudflare_fallback`, except past
    MAX_PACKAGE_FALLBACK_PROBES in a row. A probe Cloudflare answered starts the count again."""
    key = (ctx.miner_hotkey, ctx.executor.uuid)
    if not cloudflare_fallback:
        _package_fallback_probes.pop(key, None)
        return False
    probes = _package_fallback_probes.get(key, 0) + 1
    _package_fallback_probes[key] = probes
    if probes <= MAX_PACKAGE_FALLBACK_PROBES:
        return True
    logger.warning(
        _m(
            "VERIFYX_PACKAGE_FALLBACK_STREAK",
            extra=get_extra_info({**ctx.default_extra, "package_fallback_probes": probes}),
        )
    )
    return False


def _fallback_upload_reading(reading: object, upload_blocked: bool) -> object:
    """The upload reading `_feed_ema` gets. When Cloudflare failed the upload, or failed the
    download so the upload never ran (`cloudflare_upload_fallback`: Cloudflare's 429),
    the 0.0 (or missing value) the probe reports is no measurement: None, and the previous upload
    EMA stands. Otherwise a 0.0, a direction timeout included, still lowers the EMA; a positive
    upload is always a measurement."""
    if upload_blocked and not _is_positive_number(reading):
        return None
    return reading


def _ema_if_gated(prev: float | None, reading: object) -> float | None:
    """The download EMA `_feed_ema` would store if `reading` were the gated one."""
    if reading is not None and not _is_speed_reading(reading):
        return prev
    return compute_ema(prev, reading if reading is not None else 0.0)


def _network_gate(
    verifyx_network: dict, prev_ema, ema_gated: float | None, ema_upload: float | None
) -> dict:
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
        "upload_floor_mbps": MIN_VERIFYX_EMA_UPLOAD_SPEED_MBPS,
        "upload_speed": verifyx_network.get("upload_speed"),
        "ema_upload": ema_upload,
    }


# the specs keys VerifyXCheck._judge writes
_JUDGED_SPEC_KEYS = ("ram", "network", "hard_disk")


def _judged_specs(specs: dict) -> dict:
    return {k: specs[k] for k in _JUDGED_SPEC_KEYS if k in specs}


def _is_cold_sample_below_gate(ctx: Context, result, prev_ema) -> bool:
    """The one case DAH-2959 re-measures: a never-measured executor whose probe otherwise passed
    but whose single download sample would fail the EMA gate on its own.

    `rented_data` present is required — without the backend's answer every executor looks
    never-measured, and a backend blip must not double VerifyX for the whole fleet.
    """
    if ctx.state.rented_data is None:
        return False
    if prev_ema is not None and prev_ema.ema_verifyx_download_speed is not None:
        return False
    if not result.data or not result.data.get("success"):
        return False
    speed = _download_speed(result)
    return speed is None or speed < MIN_VERIFYX_EMA_DOWNLOAD_SPEED_MBPS


async def _not_idle_now(
    ctx: Context, timeout: float, *, backend: bool = True, created_since: float | None = None
) -> tuple[str | None, float | None]:
    """(Why the cold retry must not run now or None, the time the stamp was compared against).

    Busy: a customer pod on the node (in the cycle-start snapshot, as refreshed by earlier checks, or
    in the backend's shared recent answer), a rental being created, a create stamped after the rental
    data in use was read (its pod may be in neither source and its marker already gone), or a rental
    state that cannot be read within `timeout`. A filler alone keeps the node eligible, as for the
    first sample. `backend=False` re-reads only the pending-pod marker and the stamp. The stamp is
    compared against the earlier of `created_since` (epoch s) and the backend read's start."""
    uuid = str(ctx.executor.uuid)

    async def read() -> tuple[bool, Any, float | None]:
        if await ctx.services.redis.renting_in_progress(ctx.miner_hotkey, uuid):
            return True, None, created_since
        fresh, since = ctx.state.rented_data, created_since
        if backend:
            fresh, read_at = await _rented_executors_now_shared(ctx.services.backend)
            since = read_at if since is None else min(since, read_at)
        if since is not None:
            started = await ctx.services.redis.get(create_started_at_key(ctx.miner_hotkey, uuid))
            if started is not None and float(started) > since:
                return True, None, since
        return False, fresh, since

    try:
        creating, fresh, since = await asyncio.wait_for(read(), timeout=timeout)
    except Exception:  # noqa: BLE001 — unknown (incl. too slow) counts as busy
        return "rental_state_unavailable", created_since
    if creating:
        return "rental_create_in_progress", since
    if fresh is None:
        return "rental_state_unavailable", since
    for rented_data in (ctx.state.rented_data, fresh):
        rented = rented_data.executors.get(uuid) if rented_data else None
        if rented is not None and rented.pods:
            return "rented", since
    return None, since


async def _until_not_idle(ctx: Context, since: float) -> str:
    """Polls the pending-pod marker and the node's create-started stamp while the retry runs; returns
    why the node stopped being idle. The stamp catches a create (after `since`) whose marker came and
    went between two polls."""
    while True:
        await asyncio.sleep(RENTAL_WATCH_INTERVAL_SECONDS)
        busy, _ = await _not_idle_now(ctx, RENTAL_STATE_TIMEOUT_SECONDS, backend=False, created_since=since)
        if busy is not None:
            return busy


def verifyx_ema_hold_reason(ctx: Context, failed_check_id: str | None) -> str | None:
    """Why this cycle's VerifyX sample must not move the published EMA, or None.

    Held: a cycle another check failed, and a passing cycle without the image cached on a
    never-measured node, since the image pre-pull shares the link. Not held: a cycle VerifyX itself
    failed (the gate working), and anything without the backend's answer, as for the cold retry.
    """
    if ctx.state.rented_data is None:
        return None
    if failed_check_id:
        if failed_check_id not in (VerifyXCheck.check_id, VerifyXColdRetryCheck.check_id):
            return f"cycle failed {failed_check_id}"
        return None
    if ctx.state.recommended_image_cached is False:
        prev_ema = _stored_network_ema(ctx)
        if prev_ema is None or prev_ema.ema_verifyx_download_speed is None:
            return "never-measured node, image not cached yet"
    return None


def _stored_network_ema(ctx: Context) -> NetworkEMA | None:
    rented_data = ctx.state.rented_data
    return rented_data.network_ema.get(ctx.executor.uuid) if rented_data else None


def hold_verifyx_ema(ctx: Context, specs: dict[str, Any]) -> dict[str, Any]:
    """``specs`` with the VerifyX EMA put back to what the backend held before this cycle.

    Only keys this cycle wrote are touched. A never-measured node publishes none, which the backend
    already reads as unseeded (the backend's first-pass deferral relies on it). The raw samples stay.
    """
    network = specs.get("network")
    if not isinstance(network, dict) or not any(key in network for key in _EMA_KEYS):
        return specs
    prev_ema = _stored_network_ema(ctx)
    held = dict(network)
    for key in _EMA_KEYS:
        if key not in held:
            continue
        previous = getattr(prev_ema, key, None)
        if previous is None:
            del held[key]
        else:
            held[key] = previous
    return {**specs, "network": held}


async def _remember_stale_run(ctx: Context, run_dir: str) -> None:
    """Keep a cut-off run whose cleanup failed, so the next VerifyX run on the node reaps it."""
    key = verifyx_stale_runs_key(ctx.miner_hotkey, ctx.executor.uuid)
    try:
        # a busy Redis lock must not eat the margin the first result still needs
        async with asyncio.timeout(STALE_RUN_SAVE_TIMEOUT_SECONDS):
            await ctx.services.redis.hset(key, run_dir, str(time.time()))
            await ctx.services.redis.expire(key, VERIFYX_STALE_RUNS_TTL_SECONDS)
    except Exception as e:
        logger.warning(
            _m(
                "Failed to record a stale VerifyX run",
                extra=get_extra_info({**ctx.default_extra, "error": repr(e)}),
            )
        )


async def _reap_stale_runs(ctx: Context) -> None:
    """Kill and remove the runs earlier cycles cut off but could not clean up.

    Bounded per cycle: entries past the TTL or beyond STALE_RUNS_KEPT (oldest first) are dropped
    unreaped (the run's remote deadline ends it), at most STALE_RUNS_REAPED_PER_CYCLE are tried,
    within STALE_RUNS_REAP_BUDGET_SECONDS, so a node that stalls every cleanup cannot grow the work."""
    key = verifyx_stale_runs_key(ctx.miner_hotkey, ctx.executor.uuid)
    try:
        async with asyncio.timeout(STALE_RUNS_REAP_BUDGET_SECONDS):
            runs = {}
            for raw_dir, raw_at in (await ctx.services.redis.hgetall(key) or {}).items():
                run_dir = raw_dir.decode() if isinstance(raw_dir, bytes) else raw_dir
                try:
                    runs[run_dir] = float(raw_at)
                except (TypeError, ValueError):
                    runs[run_dir] = 0.0
            newest = sorted(runs, key=runs.get, reverse=True)
            cutoff = time.time() - VERIFYX_STALE_RUNS_TTL_SECONDS
            keep = [d for d in newest[:STALE_RUNS_KEPT] if runs[d] >= cutoff]
            dropped = [d for d in newest if d not in keep]
            if dropped:
                await ctx.services.redis.hdel(key, *dropped)
            for run_dir in keep[:STALE_RUNS_REAPED_PER_CYCLE]:
                if await ctx.services.verifyx.kill_remote_verifyx(
                    ctx.services.shell, run_dir, ctx.default_extra
                ):
                    await ctx.services.redis.hdel(key, run_dir)
    except Exception as e:
        logger.warning(
            _m(
                "Failed to reap stale VerifyX runs",
                extra=get_extra_info({**ctx.default_extra, "error": repr(e)}),
            )
        )


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


class VerifyXColdRetryCheck:
    """DAH-5462: the second VerifyX sample of a never-measured node whose first one was below the
    gate (VerifyXCheck leaves `verifyx_cold_retry`). Runs last before scoring, with the time the
    other checks left less COLD_RETRY_MARGIN_SECONDS, so it can never push them past the task
    deadline. A better, successful retry replaces the first sample's network reading; otherwise the
    first sample stands, and a gate it failed fails the node here."""

    check_id = "gpu.validate.verifyx_cold_retry"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        pending: ColdRetryPending | None = ctx.state.verifyx_cold_retry
        if pending is None:
            return CheckResult(
                passed=True,
                event=render_message(
                    Msg.COLD_RETRY, ctx=ctx, check_id=self.check_id, what={"pending": False}
                ),
            )
        busy = None
        # creates started after the rental data the lookup relies on was read (or after now, if
        # earlier) cut the retry off; the lookup's data shows the earlier ones
        idle_since = time.time()
        if task_seconds_left(ctx) - COLD_RETRY_MARGIN_SECONDS >= MIN_COLD_RETRY_SECONDS:
            # never let the lookup itself run into the margin and cost the first sample
            busy, idle_since = await _not_idle_now(
                ctx,
                min(
                    RENTAL_STATE_TIMEOUT_SECONDS, task_seconds_left(ctx) - COLD_RETRY_MARGIN_SECONDS
                ),
                created_since=idle_since,
            )
        if busy is not None:
            # a renter may be on the node now: never load it with a second RAM, disk and network test
            return await self._first_sample_stands(ctx, pending, {"skipped": busy})
        budget = task_seconds_left(ctx) - COLD_RETRY_MARGIN_SECONDS
        if budget < MIN_COLD_RETRY_SECONDS:
            logger.info(
                _m(
                    "VerifyX cold first sample below the gate, no time left to re-measure",
                    extra=get_extra_info(
                        {
                            **ctx.default_extra,
                            "task_seconds_left": int(budget + COLD_RETRY_MARGIN_SECONDS),
                        }
                    ),
                )
            )
            return await self._first_sample_stands(
                ctx,
                pending,
                {"skipped_task_seconds_left": int(budget + COLD_RETRY_MARGIN_SECONDS)},
            )

        # Cut at the deadline less the margin (never past its own SSH timeout) and judge on the
        # first sample. Raced against a rental-state watcher: a create (customer or filler) started
        # after the rental data the pre-retry lookup used was read cuts it off at the next poll, within RENTAL_WATCH_INTERVAL_SECONDS
        # plus one Redis read, even if its pending-pod marker is already gone. A pod can share the node
        # for that long at most. An outer cancel still propagates.
        verifyx_service = ctx.services.verifyx
        run_dir = f".verifyx-retry-{secrets.token_hex(8)}"
        retry = None
        transport_error = None
        probe = asyncio.ensure_future(
            asyncio.wait_for(
                verifyx_service.validate_verifyx_and_process_job(
                    shell=ctx.services.shell,
                    executor_info=ctx.executor,
                    default_extra=ctx.default_extra,
                    machine_spec=ctx.state.specs,
                    run_dir=run_dir,
                    **pending.sizing,
                ),
                timeout=min(budget, VERIFYX_COMMAND_TIMEOUT_SECONDS),
            )
        )
        watch = asyncio.ensure_future(_until_not_idle(ctx, idle_since))
        try:
            await asyncio.wait({probe, watch}, return_when=asyncio.FIRST_COMPLETED)
            if probe.done():
                try:
                    retry = probe.result()
                    transport_error = (retry.diagnostics or {}).get("transport_error")
                except TimeoutError:
                    pass
            else:
                busy = watch.result()
        finally:
            probe.cancel()
            watch.cancel()
            await asyncio.gather(probe, watch, return_exceptions=True)
            # Cut off (deadline, SSH timeout, lost connection, rental, outer cancel): the remote run may live on.
            if (retry is None or transport_error is not None) and not (
                await verifyx_service.kill_remote_verifyx(
                    ctx.services.shell, run_dir, ctx.default_extra
                )
            ):
                await _remember_stale_run(ctx, run_dir)
        if busy is not None:
            logger.info(
                _m(
                    "VerifyX cold retry cut off: the node stopped being idle",
                    extra=get_extra_info({**ctx.default_extra, "reason": busy}),
                )
            )
            return await self._first_sample_stands(ctx, pending, {"cancelled": busy})
        retry_speed = _download_speed(retry) if retry is not None else None
        use_retry = (
            retry is not None
            and retry.data is not None
            and bool(retry.data.get("success"))
            and (retry_speed or 0.0) > (pending.first_download_speed_mbps or 0.0)
        )
        cold_sample_retry = ColdSampleRetry(
            first_download_speed_mbps=pending.first_download_speed_mbps,
            retry_download_speed_mbps=retry_speed,
            used="retry" if use_retry else "first",
            timed_out=retry is None or "TimeoutError" in (transport_error or ""),
        )
        logger.info(
            _m(
                "VerifyX cold first sample below the gate, re-measured once",
                extra=get_extra_info({**ctx.default_extra, **asdict(cold_sample_retry)}),
            )
        )
        if not use_retry:
            return await self._first_sample_stands(ctx, pending, asdict(cold_sample_retry))
        return await self._with_disk(
            ctx, self._judge_once(ctx, pending, retry, "ssh", asdict(cold_sample_retry))
        )

    async def _first_sample_stands(
        self, ctx: Context, pending: ColdRetryPending, what: dict
    ) -> CheckResult:
        return await self._with_disk(
            ctx, self._judge_once(ctx, pending, pending.first_result, pending.transport, what)
        )

    async def _with_disk(self, ctx: Context, judged: CheckResult) -> CheckResult:
        """DiskSpaceCheck left a pending cold retry's disk to this check: judge the disk readings of
        the sample kept (first or retry), the ones the node is scored on."""
        if not judged.passed:
            return judged
        # the deferred rental lookup must not eat the margin left for scoring (review 5772289851)
        disk = await DiskSpaceCheck().run(
            ctx.model_copy(update=judged.updates),
            rental_lookup_timeout=max(
                0.0, min(RENTAL_STATE_TIMEOUT_SECONDS, task_seconds_left(ctx) - COLD_RETRY_MARGIN_SECONDS)
            ),
        )
        if disk.passed:
            return judged
        return disk.model_copy(
            update={"updates": judged.updates, "extra_events": [*judged.extra_events, judged.event]}
        )

    def _judge_once(
        self, ctx: Context, pending: ColdRetryPending, result, transport: str, what: dict
    ) -> CheckResult:
        """The probe's one counted judgement, of the sample used, on the specs the first sample was
        judged on (VerifyXCheck applied the first sample's readings uncounted)."""
        specs = {k: v for k, v in ctx.state.specs.items() if k not in _JUDGED_SPEC_KEYS}
        specs.update(copy.deepcopy(pending.base_specs))
        state = replace(ctx.state, specs=specs, verifyx_cold_retry=None)
        judged = VerifyXCheck()._judge(
            ctx.model_copy(update={"state": state}),
            result,
            transport=transport,
            sizing=pending.sizing,
            prev_ema=pending.prev_ema,
        )
        judged.event.what_we_saw["cold_sample_retry"] = what
        if judged.updates is None or "state" not in judged.updates:
            judged.updates = {**(judged.updates or {}), "state": state}
        return judged
