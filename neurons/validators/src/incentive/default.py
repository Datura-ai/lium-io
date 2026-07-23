"""Default incentive algorithm implementation.

This implementation extracts the original score calculation and weight distribution
logic to maintain backward compatibility with the existing system.
"""

from datetime import UTC, datetime

import bittensor

from core.config import get_total_burn_emission, settings
from core.utils import _m, get_extra_info, get_logger
from incentive.miner_incentive_log import MinerLogLine
from incentive.base import BaseIncentive
from services.task_service import JobResult

logger = get_logger(__name__)


def _referral_coldkey_verified(funding: bittensor.NeuronInfo) -> bool:
    """Fail-closed coldkey check for the dedicated referral funding wallet (DAH-2481).

    Emission is routed to ``REFERRAL_EMISSION_UID`` only when that UID's on-chain coldkey
    equals the explicitly configured ``settings.REFERRAL_EMISSION_COLDKEY``. An empty
    (unset) expected coldkey, or any mismatch, is treated as UNVERIFIED — so the referral
    wallet is verified against its own key (independent of the burn ``BURNER_COLDKEYS``),
    and emission is never sent to whatever hotkey happens to occupy the UID slot.
    """
    expected_coldkey = settings.REFERRAL_EMISSION_COLDKEY
    return bool(expected_coldkey) and funding.coldkey == expected_coldkey


def _parse_driver_version(value: str) -> tuple[int, ...] | None:
    """Parse a dotted NVIDIA driver version (e.g. "580.95.05") into an int tuple for
    ordering. Returns None when the value is missing or non-numeric."""
    if not value:
        return None
    try:
        return tuple(int(part) for part in str(value).strip().split("."))
    except (TypeError, ValueError):
        return None


def get_min_driver_multiplier(
    driver_version: str,
    is_rented: bool = False,
    reference_time: datetime | None = None,
) -> float:
    """Minimum NVIDIA driver multiplier with a grace period.

    An executor whose reported driver is at least ``settings.MIN_NVIDIA_DRIVER_VERSION``
    (compared as a dotted version tuple) is unaffected (multiplier 1.0). A non-compliant
    unrented executor gets a grace period until ``settings.MIN_DRIVER_CUTOFF`` (multiplier
    1.0) and is fully gated on/after it (multiplier 0.0).

    Currently-rented executors are always exempt — an active customer must not be
    penalised because their miner has not yet upgraded the host driver.

    A missing or unparseable ``driver_version`` means the value was not reported (a real
    GPU always reports a driver string), so the gate fails open rather than penalising an
    unknown reading.

    ``reference_time`` is normalised to naive-UTC so it can be compared with the naive
    cutoff (same convention as ``SYSBOX_RENTED_CUTOFF``); tests may inject it.
    """
    if is_rented:
        return 1.0
    reported = _parse_driver_version(driver_version)
    required = _parse_driver_version(settings.MIN_NVIDIA_DRIVER_VERSION)
    if reported is None or required is None or reported >= required:
        return 1.0
    now = reference_time or datetime.now(UTC)
    if now.tzinfo is not None:
        now = now.astimezone(UTC).replace(tzinfo=None)
    if now < settings.MIN_DRIVER_CUTOFF:
        return 1.0
    return 0.0


class DefaultIncentive(BaseIncentive):
    """Default incentive algorithm.

    Implements the original scoring and weight distribution logic from the validator.
    This maintains backward compatibility with the existing incentive system.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.burn_share = get_total_burn_emission()

        # Metrics
        self.total_executors = 0
        self.successful_executors = 0
        self.failed_executors = 0
        
        # Incentive states
        self.total_mining_score = 0
        self.miner_incentives = {}
        self.mining_share = 1 - self.burn_share

    async def _pre_process_job_result(self, hotkey: str, result: JobResult) -> JobResult:
        """Process a job result.

        Args:
            result: Job execution result to process
        """
        self.total_executors += 1
        result = await self.calculate_executor_score(result)
        self.total_mining_score += result.mining_score
        if result.job_score == 1.0:
            self.successful_executors += 1
        else:
            self.failed_executors += 1
        return result

    async def _post_process_job_result(self, hotkey: str, result: JobResult) -> JobResult:
        """Process a job result.

        Args:
            result: Job execution result to process
        """
        result.mining_share = self.mining_share
        result.total_mining_score = self.total_mining_score
        if result.mining_score is None:
            error_report: MinerLogLine = MinerLogLine.mining_score_missing(hotkey, result)
            result.incentive_logs.append(error_report.to_log_line())
            return result

        result.incentive = (self.mining_share * result.mining_score / self.total_mining_score) if self.total_mining_score > 0 else 0.0
        self.miner_incentives[hotkey] = self.miner_incentives.get(hotkey, 0.0) + result.incentive
        report: MinerLogLine = MinerLogLine.mining_incentive_calculated(
            hotkey, result, self.total_mining_score, self.mining_share
        )
        result.incentive_logs.append(report.to_log_line())
        return result

    async def calculate_executor_score(
        self,
        job_result: JobResult,
    ) -> JobResult:
        """Calculate mining score for a single executor/job result.

        This method implements the original calc_job_score() logic from validator.py
        lines 146-199, including GPU count calculation, base scoring, and multipliers
        for sysbox runtime and uptime.

        Args:
            job_result: Job execution result to score

        Returns:
            JobResult with calculated mining score
        """
        # Early exit check - if job score is 0, return immediately
        if not job_result.score:
            job_result.mining_score = 0
            return job_result

        # Fallback for deployments using the legacy/default incentive algorithm
        # directly. The active rental_price algorithm handles this before it
        # calls into DefaultIncentive.
        if job_result.is_new_rentals_paused and not job_result.is_rented:
            logger.info(
                _m(
                    "Executor excluded from mining pool - paused for new rentals",
                    extra=get_extra_info(
                        {
                            "executor_id": str(job_result.executor_info.uuid),
                            "gpu_model": job_result.gpu_model,
                            "gpu_count": job_result.gpu_count,
                            "reason": "new_rentals_paused",
                            "score": 0,
                            "pool": "none",
                        }
                    ),
                )
            )
            job_result.mining_score = 0
            return job_result

        # GPU count calculation
        job_result.total_gpu_count = self.total_gpu_model_count_map.get(job_result.gpu_model, 0)
        if not job_result.total_gpu_count:
            job_result.mining_score = 0
            return job_result

        # Base score calculation
        job_result.gpu_portion = await self.redis_service.get_portion_per_gpu_type(job_result.gpu_model)
        job_result.mining_score = job_result.score * job_result.gpu_portion * job_result.gpu_count / job_result.total_gpu_count

        # Sysbox runtime multiplier
        is_rented_after_cutoff = (
            job_result.is_rented
            and job_result.rental_created_at
            and job_result.rental_created_at >= settings.SYSBOX_RENTED_CUTOFF
        )
        if job_result.sysbox_runtime:
            job_result.sysbox_multiplier = 1
        else:
            portion = settings.PORTION_FOR_SYSBOX_RENTED if is_rented_after_cutoff else settings.PORTION_FOR_SYSBOX

            job_result.sysbox_multiplier = 1 - portion

        # Minimum NVIDIA driver multiplier (phased gate); rented executors are exempt.
        job_result.driver_multiplier = get_min_driver_multiplier(
            job_result.nvidia_driver_version, is_rented=job_result.is_rented
        )

        # Uptime multiplier
        if settings.SKIP_COLLATERAL_PENALTY or job_result.collateral_deposited:
            job_result.uptime_multiplier = 1
        else:
            uptime_in_minutes = await self.redis_service.get_executor_uptime(job_result.executor_info)
            job_result.uptime_multiplier = (
                1
                - settings.PORTION_FOR_UPTIME
                + settings.PORTION_FOR_UPTIME
                * min(1, uptime_in_minutes / settings.UPTIME_REQUIRED_MINUTES)
            )

        # Apply multiplier
        job_result.mining_score *= (
            job_result.sysbox_multiplier * job_result.uptime_multiplier * job_result.driver_multiplier
        )
        line: MinerLogLine = MinerLogLine.mining_score_calculated(job_result, is_rented_after_cutoff)
        job_result.incentive_logs.append(line.to_log_line())
        logger.info(line.as_internal_log())
        return job_result

    async def calculate_final_weights(
        self,
        miners: list[bittensor.NeuronInfo],
        last_mechanism_step_block: int | None,
    ) -> dict[str, float]:
        """Calculate scores with burning logic for this cycle.

        This method applies burning logic and returns scores to be accumulated.
        It does NOT return final weights for blockchain - those are calculated
        later in set_weights by normalizing accumulated scores.

        Args:
            miners: List of miner neuron information
            last_mechanism_step_block: Last mechanism step block number

        Returns:
            dict[str, float]: Scores with burning applied for each miner
        """
        # Calculate burn scores using BurnService
        burn_scores = self.burn_service.calculate_burn_scores(
            miners=miners,
            burn_share=self.burn_share,  # burn-emission share sourced from shared config (DAH-2274)
            last_mechanism_step_block=last_mechanism_step_block,
        )
        cycle_scores = dict(burn_scores)
        for miner in miners:
            # Check if miner is a burner
            if miner.hotkey in cycle_scores:
                continue
            cycle_scores[miner.hotkey] = self.miner_incentives.get(miner.hotkey, 0.0)

        # DAH-2481: redirect a fixed fraction of the miner (non-burn) emission to the
        # funding wallet so it can fund referral incentives (see _apply_referral_carveout).
        self._apply_referral_carveout(cycle_scores, burn_scores, miners)

        return cycle_scores

    def _apply_referral_carveout(
        self,
        cycle_scores: dict[str, float],
        burn_scores: dict[str, float],
        miners: list[bittensor.NeuronInfo],
    ) -> None:
        """DAH-2481 — grant a fixed fraction of miner emission to the referral funding wallet.

        A configurable share (``settings.REFERRAL_EMISSION_SHARE``) of the miner
        (non-burn) emission is carved off every miner proportionally and granted to the
        dedicated referral funding wallet (``settings.REFERRAL_EMISSION_UID`` — a hotkey
        registered on SN51 under its own coldkey, distinct from the burn UID) so it can
        fund referral incentives.

        Because the accumulated scores are normalized before they reach chain (see
        ``subtensor_client.set_weights``), the carve-out only has to preserve proportions.
        Totals still sum to 1 — each miner keeps ``1 - share`` of its emission and the
        funding wallet gains the carved sum (its emission is separate from, and additive to,
        the burn wallet's, so it is distinguishable on-chain).

        No-op when the share is 0 (default). Fails closed via ``_referral_coldkey_verified``:
        if the funding wallet is absent this cycle, or ``REFERRAL_EMISSION_COLDKEY`` is
        unset, or that coldkey does not match on-chain, the miner emission is left
        untouched — emission is never redirected to an unverified key. The share is clamped
        to [0, 1] (and a non-positive/NaN share is a no-op) so a misconfigured value can
        never drive miner weights negative.
        """
        share = min(max(settings.REFERRAL_EMISSION_SHARE, 0.0), 1.0)
        if not (share > 0):  # also rejects NaN, which the clamp preserves
            return

        funding = next((m for m in miners if m.uid == settings.REFERRAL_EMISSION_UID), None)
        if funding is None:
            return
        if not _referral_coldkey_verified(funding):
            logger.warning(
                _m(
                    "Referral carve-out skipped — funding wallet coldkey unverified",
                    extra={
                        "referral_uid": settings.REFERRAL_EMISSION_UID,
                        "funding_hotkey": funding.hotkey,
                        "funding_coldkey": funding.coldkey,
                        "expected_coldkey": settings.REFERRAL_EMISSION_COLDKEY,
                    },
                )
            )
            return

        referral_total = 0.0
        for hotkey, score in cycle_scores.items():
            # Only carve from the miner pool: burn slots and the funding wallet's own
            # entry are left untouched.
            if hotkey in burn_scores or hotkey == funding.hotkey:
                continue
            cut = score * share
            cycle_scores[hotkey] = score - cut
            referral_total += cut

        if referral_total <= 0:
            return

        cycle_scores[funding.hotkey] = cycle_scores.get(funding.hotkey, 0.0) + referral_total
        logger.info(
            _m(
                "Referral emission carve-out applied",
                extra={
                    "referral_uid": settings.REFERRAL_EMISSION_UID,
                    "referral_hotkey": funding.hotkey,
                    "referral_share": share,
                    "referral_total": referral_total,
                    "pool": "referral",
                },
            )
        )
