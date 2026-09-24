"""Every reason a node earns 0 is recorded, not only the first one found.

A node blocked by one rule used to hide every later rule: an 8x flagship with no Discord
learned about the flagship gate only after connecting Discord. And a node whose validation
failed earned 0 with an empty reason list. Both cases now reach the backend as data.
"""
import logging
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from core.config import settings, shared_client
from datura.requests.miner_requests import ExecutorSSHInfo
from incentive.config import IncentiveConfig
from incentive.default import DefaultIncentive
from incentive.miner_incentive_log import UNCLASSIFIED_VALIDATION_FAILURE, ZeroIncentiveReason
from incentive.rental_price import RentalPriceIncentive
from services.miner_service import MinerService
from services.task.messages import ExecutorImageMessages, FinalizeMessages, TenantEnforcementMessages
from services.task.models import build_msg
from services.task_service import JobResult

H200 = "NVIDIA H200"
# a scrape without the NCU observation: the flagship gate reads it as "not unrestricted"
FLAGSHIP_SPEC: dict = {"hard_disk": {"total": 200 * 1024**2}, "gpu": {"details": [{"capacity": 141 * 1024}] * 8}}


def _incentive() -> RentalPriceIncentive:
    return RentalPriceIncentive(IncentiveConfig(), AsyncMock(), {}, {})


def _idle_flagship(**overrides) -> JobResult:
    fields: dict = dict(
        executor_info=ExecutorSSHInfo(
            uuid="exec-1",
            address="10.0.0.1",
            port=8080,
            ssh_username="root",
            ssh_port=22,
            python_path="/usr/bin/python3",
            root_dir="/tmp",
            price_per_gpu=1.0,
        ),
        spec=dict(FLAGSHIP_SPEC),
        score=1.0,
        job_score=1.0,
        job_batch_id="2026-09-24 00:00:00",
        log_status="success",
        log_text="ok",
        gpu_model=H200,
        gpu_count=8,
        collateral_deposited=True,
        sysbox_runtime=True,
    )
    fields.update(overrides)
    return JobResult(**fields)


def _failed(**overrides) -> JobResult:
    return _idle_flagship(
        spec=None, score=0, job_score=0, log_status="error", gpu_model=None, gpu_count=0, **overrides
    )


def _codes(result: JobResult) -> list[str]:
    return [reason.reason for reason in result.zero_incentive_reasons]


@pytest.fixture
def discord_cutoff_passed(monkeypatch):
    monkeypatch.setattr(settings, "DISCORD_INCENTIVE_CUTOFF", datetime(2020, 1, 1))


@pytest.mark.asyncio
async def test_flagship_gate_is_recorded_even_when_discord_blocks(monkeypatch, discord_cutoff_passed):
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_FLAGSHIP_CAPABILITY_LIMIT", True)

    result = await _incentive().calculate_executor_score(_idle_flagship(provider_discord_connected=False))

    # the first entry is the one the old first-match evaluation reported
    assert _codes(result) == ["provider_discord_not_connected", "flagship_without_ncu_or_split"]
    assert result.mining_score == 0
    assert result.eligible_for_rental_share is False


@pytest.mark.asyncio
async def test_every_hard_exclusion_and_every_enforced_gate_is_recorded_in_order(
    monkeypatch, discord_cutoff_passed
):
    for flag in (
        "ENABLE_UNRENTED_VRAM_OVER_DISK_LIMIT",
        "ENABLE_UNRENTED_FLAGSHIP_CAPABILITY_LIMIT",
        "ENABLE_UNRENTED_SOFT_PRICE_LIMIT",
    ):
        monkeypatch.setattr(settings, flag, True)
    new_cfg = shared_client.config.model_copy(update={"machine_prices_p90": {H200: 0.5}})
    monkeypatch.setattr(shared_client, "_config", new_cfg)

    result = await _incentive().calculate_executor_score(
        _idle_flagship(provider_discord_connected=False, is_new_rentals_paused=True, is_spot=True)
    )

    assert _codes(result) == [
        "spot_tier",
        "provider_discord_not_connected",
        "new_rentals_paused",
        "price_above_market_p90_soft_limit",
        "insufficient_disk_for_vram",
        "flagship_without_ncu_or_split",
    ]
    assert result.mining_score == 0


@pytest.mark.asyncio
async def test_two_idle_pool_gates_are_both_recorded(monkeypatch):
    # price used to stop the chain: a node over the price limit never heard about its disk
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_SOFT_PRICE_LIMIT", True)
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_VRAM_OVER_DISK_LIMIT", True)
    new_cfg = shared_client.config.model_copy(update={"machine_prices_p90": {H200: 0.5}})
    monkeypatch.setattr(shared_client, "_config", new_cfg)

    result = await _incentive().calculate_executor_score(_idle_flagship(gpu_count=4))

    assert _codes(result) == ["price_above_market_p90_soft_limit", "insufficient_disk_for_vram"]
    assert result.eligible_for_rental_share is False


def _logged(caplog, reason: str) -> int:
    return sum(1 for record in caplog.records if getattr(record.msg, "extra", {}).get("reason") == reason)


# (the gate's flag, the node, the reason it records when enforced, the reason its log line carries).
# The last two are the probes' "cannot measure ...; unrented incentive kept" lines: no reason, and
# the sentence is false for a node that is already excluded.
GATE_LOG_CASES = [
    ("ENABLE_UNRENTED_FLAGSHIP_CAPABILITY_LIMIT", {}, "flagship_without_ncu_or_split", "flagship_without_ncu_or_split"),
    ("ENABLE_UNRENTED_VRAM_OVER_DISK_LIMIT", {"gpu_count": 4}, "insufficient_disk_for_vram", "insufficient_disk_for_vram"),
    (
        "ENABLE_UNRENTED_VRAM_OVER_DISK_LIMIT",
        {"gpu_count": 4, "spec": {"gpu": {"details": []}}},
        None,
        "insufficient_disk_unmeasured",
    ),
    ("ENABLE_UNRENTED_POWER_CAP_LIMIT", {"gpu_count": 4}, None, "power_cap_capability_unmeasured"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("enforced", [False, True], ids=["shadow", "enforced"])
@pytest.mark.parametrize(("flag", "node", "reason", "log_reason"), GATE_LOG_CASES, ids=lambda v: v if isinstance(v, str) else None)
async def test_a_gate_logs_only_while_the_node_is_still_eligible(
    monkeypatch, caplog, discord_cutoff_passed, enforced, flag, node, reason, log_reason
):
    # the gate lines count eligible nodes only, as before every reason was recorded; an enforced
    # gate still records its reason on the blocked node
    monkeypatch.setattr(settings, flag, enforced)

    with caplog.at_level(logging.INFO):
        blocked = await _incentive().calculate_executor_score(
            _idle_flagship(provider_discord_connected=False, **node)
        )

    assert _logged(caplog, log_reason) == 0
    assert _codes(blocked) == ["provider_discord_not_connected"] + ([reason] if enforced and reason else [])

    # the same node, not blocked: the line is there, so the zero above is not a blind caplog
    caplog.clear()
    with caplog.at_level(logging.INFO):
        await _incentive().calculate_executor_score(_idle_flagship(**node))
    assert _logged(caplog, log_reason) == 1


@pytest.mark.asyncio
async def test_driver_and_sysbox_are_recorded_on_a_node_blocked_before_pricing(
    monkeypatch, discord_cutoff_passed
):
    monkeypatch.setattr(settings, "MIN_DRIVER_CUTOFF", datetime(2020, 1, 1))
    monkeypatch.setattr(settings, "PORTION_FOR_SYSBOX_UNRENTED", 1)

    result = await _incentive().calculate_executor_score(
        _idle_flagship(
            gpu_count=4,
            provider_discord_connected=False,
            nvidia_driver_version="470.0.1",
            sysbox_runtime=False,
        )
    )

    assert _codes(result) == [
        "provider_discord_not_connected",
        "nvidia_driver_below_minimum",
        "sysbox_not_enabled",
    ]
    assert result.zero_incentive_reasons[1].context["driver_multiplier"] == 0.0


@pytest.mark.asyncio
async def test_an_eligible_node_whose_rate_collapses_on_two_factors_gets_both(monkeypatch):
    incentive = _incentive()
    job = _idle_flagship(gpu_count=1)
    job.eligible_for_rental_share = True
    job.hourly_rate = 5.0
    job.sysbox_multiplier = 0.0
    job.sysbox_runtime = False
    job.driver_multiplier = 0.0
    job.count_bucket = 1
    job.max_cap = 0

    await incentive._post_process_job_result("hk", job)

    assert _codes(job) == [
        "no_unrented_capacity_for_gpu_count",
        "nvidia_driver_below_minimum",
        "sysbox_not_enabled",
    ]


@pytest.mark.asyncio
async def test_a_healthy_idle_node_has_no_reason():
    result = await _incentive().calculate_executor_score(_idle_flagship(gpu_count=4, spec=None))

    assert result.zero_incentive_reasons == []
    assert result.eligible_for_rental_share is True


# ── a zero caused by a failed validation carries the failing check's code ─────


def _failed_check_event(reason: str, check_id: str) -> object:
    return build_msg(
        event="Recommended image is not cached on the executor",
        reason=reason,
        severity="error",
        impact="Validation failed",
        remediation="Pull the recommended image on the host.",
        check_id=check_id,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", [RentalPriceIncentive, DefaultIncentive])
async def test_failed_validation_zero_carries_the_check_reason_code(engine):
    event = _failed_check_event("RECOMMENDED_IMAGE_NOT_CACHED", "recommended_image")
    result = _failed(validation_event=event, failure_reason_code="RECOMMENDED_IMAGE_NOT_CACHED")
    incentive = engine(IncentiveConfig(), AsyncMock(), {"hk": [result]}, {})

    await incentive._pre_process_job_result("hk", result)

    assert _codes(result) == [ZeroIncentiveReason.VALIDATION_FAILED]
    context = result.zero_incentive_reasons[0].context
    assert context["reason_code"] == "RECOMMENDED_IMAGE_NOT_CACHED"
    assert context["check_id"] == "recommended_image"
    assert context["remediation"] == "Pull the recommended image on the host."
    assert "RECOMMENDED_IMAGE_NOT_CACHED" in result.zero_incentive_reasons[0].message_for_miner


@pytest.mark.asyncio
async def test_failed_validation_code_comes_from_the_run_not_a_later_event():
    # the check that ended the run wins; an unrelated event's check_id is not borrowed
    event = _failed_check_event("SOMETHING_ELSE", "other_check")
    result = _failed(validation_event=event, failure_reason_code="VERIFYX_FAILED_GPU_MISMATCH")

    await _incentive()._pre_process_job_result("hk", result)

    context = result.zero_incentive_reasons[0].context
    assert context["reason_code"] == "VERIFYX_FAILED_GPU_MISMATCH"
    assert context["check_id"] is None


@pytest.mark.asyncio
async def test_failed_validation_without_any_code_is_never_empty():
    # an exception in the pipeline produces no event and no code
    result = _failed()

    await _incentive()._pre_process_job_result("hk", result)

    assert _codes(result) == [ZeroIncentiveReason.VALIDATION_FAILED]
    assert result.zero_incentive_reasons[0].context["reason_code"] == UNCLASSIFIED_VALIDATION_FAILURE


@pytest.mark.asyncio
async def test_a_full_cycle_publishes_the_failed_node_reason_once(monkeypatch):
    failed = _failed(failure_reason_code="RECOMMENDED_IMAGE_NOT_CACHED")
    healthy = _idle_flagship(
        gpu_count=4, spec=None, executor_info=failed.executor_info.model_copy(update={"uuid": "exec-2"})
    )
    redis = AsyncMock()
    redis.get_portion_per_gpu_type = AsyncMock(return_value=0.3)
    incentive = RentalPriceIncentive(IncentiveConfig(), redis, {"hk": [failed, healthy]}, {H200: 4})
    incentive.price_provider = AsyncMock(
        get_tao_price=AsyncMock(return_value=300.0), get_alpha_rate=AsyncMock(return_value=0.001)
    )

    await incentive.calculate_mining_scores()
    failed.scored_at = datetime.now(UTC)
    redis_service = MagicMock(publish=AsyncMock())
    service = MinerService(
        ssh_service=MagicMock(), task_service=MagicMock(), redis_service=redis_service, attestation_service=MagicMock()
    )
    await service.publish_machine_specs([failed], miner_hotkey="hk", miner_coldkey="ck")

    _, payload = redis_service.publish.await_args.args
    assert [reason["reason"] for reason in payload["incentive_reasons"]] == ["validation_failed"]
    assert payload["incentive_reasons"][0]["context"]["reason_code"] == "RECOMMENDED_IMAGE_NOT_CACHED"
    assert payload["incentive"] == 0.0


# ── a run that passed every check but scored 0 is not a failed check ─────────


def _passed_but_zero(event_reason: str, remediation: str, **overrides) -> JobResult:
    # the score gate zeroed a run whose checks all passed: failure_reason_code is the code of the
    # event that ended it, the finalize event or the rented halt (services/task/service.py)
    event = build_msg(
        event="Validation task completed",
        reason=event_reason,
        severity="warning",
        impact="Job score=0, actual score=0",
        remediation=remediation,
        check_id="pipeline.finalize",
    )
    return _idle_flagship(
        score=0, job_score=0, log_status="warning", validation_event=event, failure_reason_code=event_reason, **overrides
    )


PASSED_BUT_ZERO = [
    (FinalizeMessages.COMPLETED.reason, "Address issues: WARNING: Collateral required but not deposited", {}),
    (
        TenantEnforcementMessages.ALREADY_RENTED.reason,
        "No action needed. WARNING: Collateral required but not deposited",
        {"is_rented": True},
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", [RentalPriceIncentive, DefaultIncentive])
@pytest.mark.parametrize(("event_reason", "remediation", "overrides"), PASSED_BUT_ZERO, ids=["finalize", "rented"])
async def test_a_run_that_passed_every_check_gets_no_validation_failed(engine, event_reason, remediation, overrides):
    result = _passed_but_zero(event_reason, remediation, **overrides)
    incentive = engine(IncentiveConfig(), AsyncMock(), {"hk": [result]}, {})

    await incentive._pre_process_job_result("hk", result)

    # no "fix the failed check" for a node that failed none; nothing new is recorded for it
    assert _codes(result) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", [RentalPriceIncentive, DefaultIncentive])
@pytest.mark.parametrize(
    ("failure_reason_code", "overrides"),
    [
        (TenantEnforcementMessages.ALREADY_RENTED.reason, {"is_rented": True}),
        (ExecutorImageMessages.OUTDATED.reason, {}),
    ],
    ids=["rented-halt", "image-check-failed"],
)
async def test_an_outdated_image_is_one_reason_for_one_cause(monkeypatch, engine, failure_reason_code, overrides):
    monkeypatch.setattr(settings, "EXECUTOR_IMAGE_CHECK_ENFORCE", True)
    result = _idle_flagship(
        score=0,
        job_score=0,
        log_status="error",
        failure_reason_code=failure_reason_code,
        executor_image_report={"status": "OUTDATED", "expected_ref": "daturaai/compute-subnet-executor:latest"},
        **overrides,
    )
    incentive = engine(IncentiveConfig(), AsyncMock(), {"hk": [result]}, {})

    await incentive._pre_process_job_result("hk", result)

    assert _codes(result) == ["outdated_executor_image"]

