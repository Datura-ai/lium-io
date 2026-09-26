"""Every failing idle-pay requirement is reported, and reporting them never moves a number.

A node that fails several requirements gets one zero-incentive reason per requirement in
`JobResult.zero_incentive_reasons`, in `ZERO_INCENTIVE_REPORT_ORDER`, so the provider can fix
them all in one go.

`test_scoring_is_unchanged` pins the scoring outputs of a table of executors to the values the
same table produced when only the first reason was reported (the fixture file). They are
compared as JSON text, so the rate and multipliers must be bit-identical.
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo

from core.config import settings, shared_client
from incentive.config import IncentiveConfig
from incentive.miner_incentive_log import (
    INFORMATIONAL_ZERO_INCENTIVE_REASONS,
    ZERO_INCENTIVE_REPORT_ORDER,
    MinerLogLine,
    ZeroIncentiveReason,
)
from incentive.rental_price import RentalPriceIncentive
from services.const import DEFAULT_JOB_OWNER_MINER
from services.task_service import JobResult

H200 = "NVIDIA H200"
H100 = "NVIDIA H100 80GB HBM3"
RTX_5080 = "NVIDIA GeForce RTX 5080"  # valid GPU, not in the unrented incentive program
OLD_DRIVER = "570.211.01"
GOOD_DRIVER = "580.95.05"
CASE_EXECUTOR = "case-exec"

ENFORCEMENT_FLAGS = (
    "ENABLE_UNRENTED_SOFT_PRICE_LIMIT",
    "ENABLE_UNRENTED_VRAM_OVER_DISK_LIMIT",
    "ENABLE_UNRENTED_FLAGSHIP_CAPABILITY_LIMIT",
    "ENABLE_UNRENTED_POWER_CAP_LIMIT",
    "ENABLE_UNRENTED_PORT_FLOOR_FOR_SPLIT_REMAINDER",
    "EXECUTOR_IMAGE_CHECK_ENFORCE",
)


def _spec(gpu_count: int, **overrides: Any) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "gpu": {"details": [{"capacity": 141 * 1024} for _ in range(gpu_count)]},
        "hard_disk": {"total": 4000 * 1024**2 * gpu_count},
        "container_cap_eff": "000001ffffffffff",
        "nvidiactl_owner_uid": 0,
        "ncu_profiling_access": "unrestricted",
    }
    spec.update(overrides)
    return spec


SMALL_DISK: dict[str, Any] = {"hard_disk": {"total": 10 * 1024**2}}
NO_POWER_CAP: dict[str, Any] = {
    "container_cap_eff": "00000000a80425fb",
    "nvidiactl_owner_uid": 65534,
}
NCU_RESTRICTED: dict[str, Any] = {"ncu_profiling_access": "restricted"}
PARTIAL_SPLIT: dict[str, Any] = {
    "gpu_count": 8,
    "is_rented": True,
    "supports_gpu_splitting": True,
    "gpu_splitting_min_count": 1,
    "rented_gpu_count": 4,
}

# name -> (JobResult overrides, spec overrides). Singles, then multi-failure combinations.
CASES: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
    "clean_idle": ({}, {}),
    "rented": ({"is_rented": True}, {}),
    "rented_old_driver_no_sysbox": (
        {"is_rented": True, "nvidia_driver_version": OLD_DRIVER, "sysbox_runtime": False},
        {},
    ),
    "not_in_program": ({"gpu_model": RTX_5080}, {}),
    "banned": ({"is_provider_banned": True}, {}),
    "spot": ({"is_spot": True}, {}),
    "discord": ({"provider_discord_connected": False}, {}),
    "paused": ({"is_new_rentals_paused": True}, {}),
    "own_default_job": ({"default_job_owner": DEFAULT_JOB_OWNER_MINER}, {}),
    "outdated_image": ({"executor_image_report": {"status": "OUTDATED"}}, {}),
    "price": ({"price_per_gpu": 50.0}, {}),
    "disk": ({}, SMALL_DISK),
    "power_cap": ({}, NO_POWER_CAP),
    "flagship": ({"gpu_count": 8}, NCU_RESTRICTED),
    "old_driver": ({"nvidia_driver_version": OLD_DRIVER}, {}),
    "no_sysbox": ({"sysbox_runtime": False}, {}),
    "no_capacity": ({"gpu_count": 2}, {}),
    "capacity_driver_sysbox": (
        {"gpu_count": 2, "nvidia_driver_version": OLD_DRIVER, "sysbox_runtime": False},
        {},
    ),
    "driver_sysbox": ({"nvidia_driver_version": OLD_DRIVER, "sysbox_runtime": False}, {}),
    "price_disk_power_cap": ({"price_per_gpu": 50.0}, {**SMALL_DISK, **NO_POWER_CAP}),
    "price_driver_sysbox": (
        {"price_per_gpu": 50.0, "nvidia_driver_version": OLD_DRIVER, "sysbox_runtime": False},
        {},
    ),
    "discord_price_driver": (
        {
            "provider_discord_connected": False,
            "price_per_gpu": 50.0,
            "nvidia_driver_version": OLD_DRIVER,
        },
        {},
    ),
    "spot_discord_paused_default_job": (
        {
            "is_spot": True,
            "provider_discord_connected": False,
            "is_new_rentals_paused": True,
            "default_job_owner": DEFAULT_JOB_OWNER_MINER,
        },
        {},
    ),
    "outdated_image_discord_disk": (
        {"executor_image_report": {"status": "OUTDATED"}, "provider_discord_connected": False},
        SMALL_DISK,
    ),
    "not_in_program_discord": ({"gpu_model": RTX_5080, "provider_discord_connected": False}, {}),
    "flagship_everything": (
        {
            "gpu_count": 8,
            "price_per_gpu": 50.0,
            "nvidia_driver_version": OLD_DRIVER,
            "sysbox_runtime": False,
            "provider_discord_connected": False,
            "executor_image_report": {"status": "OUTDATED"},
        },
        {**SMALL_DISK, **NO_POWER_CAP, **NCU_RESTRICTED},
    ),
    "partial_split_port_limited": (PARTIAL_SPLIT, {"available_port_count": 1}),
    "partial_split_port_limited_old_driver": (
        {**PARTIAL_SPLIT, "nvidia_driver_version": OLD_DRIVER},
        {"available_port_count": 1},
    ),
}


def _job(
    executor_id: str, gpu_model: str, gpu_count: int, spec: dict[str, Any], **overrides: Any
) -> JobResult:
    price_per_gpu: float | None = overrides.pop("price_per_gpu", None)
    fields: dict[str, Any] = {
        "executor_info": ExecutorSSHInfo(
            uuid=executor_id,
            address="10.0.0.1",
            port=8080,
            ssh_username="root",
            ssh_port=22,
            python_path="/usr/bin/python3",
            root_dir="/tmp",
            price_per_gpu=price_per_gpu,
        ),
        "score": 1.0,
        "job_score": 1.0,
        "job_batch_id": "batch",
        "log_status": "success",
        "log_text": "ok",
        "gpu_model": gpu_model,
        "gpu_count": gpu_count,
        "collateral_deposited": True,
        "sysbox_runtime": True,
        "nvidia_driver_version": GOOD_DRIVER,
        "spec": spec,
    }
    fields.update(overrides)
    return JobResult(**fields)


def _case_job(name: str) -> JobResult:
    overrides, spec_overrides = CASES[name]
    overrides = dict(overrides)
    gpu_model: str = overrides.pop("gpu_model", H200)
    gpu_count: int = overrides.pop("gpu_count", 1)
    return _job(
        CASE_EXECUTOR, gpu_model, gpu_count, _spec(gpu_count, **spec_overrides), **overrides
    )


def _fleet(name: str) -> dict[str, list[JobResult]]:
    # neighbours keep both pools non-empty, so a zero on the case executor is its own
    return {
        "hk-case": [_case_job(name)],
        "hk-rented": [_job("rented-h100", H100, 1, _spec(1), is_rented=True)],
        "hk-idle": [_job("idle-h100", H100, 1, _spec(1))],
    }


def _set_market(monkeypatch: pytest.MonkeyPatch, enforce: bool) -> None:
    for flag in ENFORCEMENT_FLAGS:
        monkeypatch.setattr(settings, flag, enforce)
    monkeypatch.setattr(settings, "ENABLE_SPLIT_PARTIAL_RENTAL_SCORING", True)
    config = shared_client.config.model_copy(
        update={"machine_prices_p90": {H200: 3.0}, "soft_limit_price_rate": 1.1}
    )
    monkeypatch.setattr(shared_client, "_config", config)


async def _score_cycle(name: str) -> tuple[RentalPriceIncentive, JobResult]:
    job_results: dict[str, list[JobResult]] = _fleet(name)
    gpu_totals: dict[str, int] = {}
    for results in job_results.values():
        for result in results:
            gpu_totals[result.gpu_model] = gpu_totals.get(result.gpu_model, 0) + result.gpu_count
    redis_service = AsyncMock()
    redis_service.get_portion_per_gpu_type = AsyncMock(
        side_effect=lambda gpu_model: {H100: 0.3, H200: 0.25}.get(gpu_model, 0.1)
    )
    redis_service.get_executor_uptime = AsyncMock(return_value=100)
    incentive = RentalPriceIncentive(IncentiveConfig(), redis_service, job_results, gpu_totals)
    incentive.price_provider = AsyncMock()
    incentive.price_provider.get_tao_price.return_value = 400.0
    incentive.price_provider.get_alpha_rate.return_value = 0.001
    await incentive.calculate_mining_scores()
    return incentive, job_results["hk-case"][0]


SCORED_FIELDS = (
    "eligible_for_rental_share",
    "mining_score",
    "hourly_rate",
    "sysbox_multiplier",
    "driver_multiplier",
    "unrented_cap_multiplier",
    "effective_rate",
    "count_bucket",
    "max_cap",
    "incentive",
    "incentive_rented",
    "incentive_idle",
)


def _scoring_outputs(incentive: RentalPriceIncentive, case: JobResult) -> dict[str, Any]:
    outputs: dict[str, Any] = {field: getattr(case, field) for field in SCORED_FIELDS}
    outputs["cycle_rental_share"] = incentive.rental_share
    outputs["cycle_total_rental_cost"] = incentive.total_rental_cost
    outputs["cycle_miner_incentives"] = dict(sorted(incentive.miner_incentives.items()))
    return outputs


def _codes(result: JobResult) -> list[str]:
    return [reason.reason for reason in result.zero_incentive_reasons]


# _scoring_outputs over CASES, captured from the scoring code that reported only the first reason.
GOLDEN: dict[str, dict[str, Any]] = json.loads(
    (Path(__file__).parent / "fixtures" / "zero_incentive_scoring_first_reason_only.json").read_text()
)


@pytest.mark.asyncio
@pytest.mark.parametrize("enforce", [True, False], ids=["flags_on", "flags_off"])
@pytest.mark.parametrize("name", list(CASES))
async def test_scoring_is_unchanged(
    monkeypatch: pytest.MonkeyPatch, name: str, enforce: bool
) -> None:
    _set_market(monkeypatch, enforce)

    incentive, case = await _score_cycle(name)

    golden: dict[str, Any] = GOLDEN[f"{name}|{'flags_on' if enforce else 'flags_off'}"]
    # compared as JSON text so an int 1 turning into a float 1.0 also fails
    assert json.dumps(_scoring_outputs(incentive, case), sort_keys=True) == json.dumps(
        golden, sort_keys=True
    )


R = ZeroIncentiveReason
# Every case whose reason list differs from main's first-reason-only list.
EXPECTED_REASONS: dict[tuple[str, bool], list[ZeroIncentiveReason]] = {
    ("capacity_driver_sysbox", True): [
        R.NVIDIA_DRIVER_BELOW_MINIMUM,
        R.SYSBOX_NOT_ENABLED,
        R.NO_UNRENTED_CAPACITY_FOR_GPU_COUNT,
    ],
    ("capacity_driver_sysbox", False): [
        R.NVIDIA_DRIVER_BELOW_MINIMUM,
        R.SYSBOX_NOT_ENABLED,
        R.NO_UNRENTED_CAPACITY_FOR_GPU_COUNT,
    ],
    ("driver_sysbox", True): [R.NVIDIA_DRIVER_BELOW_MINIMUM, R.SYSBOX_NOT_ENABLED],
    ("driver_sysbox", False): [R.NVIDIA_DRIVER_BELOW_MINIMUM, R.SYSBOX_NOT_ENABLED],
    ("price_disk_power_cap", True): [
        R.PRICE_ABOVE_MARKET_P90_SOFT_LIMIT,
        R.INSUFFICIENT_DISK_FOR_VRAM,
        R.CANNOT_APPLY_GPU_POWER_CAP,
    ],
    ("price_driver_sysbox", True): [
        R.PRICE_ABOVE_MARKET_P90_SOFT_LIMIT,
        R.NVIDIA_DRIVER_BELOW_MINIMUM,
        R.SYSBOX_NOT_ENABLED,
    ],
    ("price_driver_sysbox", False): [R.NVIDIA_DRIVER_BELOW_MINIMUM, R.SYSBOX_NOT_ENABLED],
    ("discord_price_driver", True): [
        R.PROVIDER_DISCORD_NOT_CONNECTED,
        R.PRICE_ABOVE_MARKET_P90_SOFT_LIMIT,
        R.NVIDIA_DRIVER_BELOW_MINIMUM,
    ],
    ("discord_price_driver", False): [
        R.PROVIDER_DISCORD_NOT_CONNECTED,
        R.NVIDIA_DRIVER_BELOW_MINIMUM,
    ],
    ("spot_discord_paused_default_job", True): [
        R.SPOT_TIER,
        R.PROVIDER_DISCORD_NOT_CONNECTED,
        R.NEW_RENTALS_PAUSED,
        R.MINER_DEFAULT_JOB,
    ],
    ("outdated_image_discord_disk", True): [
        R.PROVIDER_DISCORD_NOT_CONNECTED,
        R.INSUFFICIENT_DISK_FOR_VRAM,
        R.OUTDATED_EXECUTOR_IMAGE,
    ],
    ("not_in_program_discord", True): [
        R.PROVIDER_DISCORD_NOT_CONNECTED,
        R.GPU_MODEL_NOT_ELIGIBLE_FOR_UNRENTED_INCENTIVE,
    ],
    ("flagship_everything", True): [
        R.PROVIDER_DISCORD_NOT_CONNECTED,
        R.PRICE_ABOVE_MARKET_P90_SOFT_LIMIT,
        R.NVIDIA_DRIVER_BELOW_MINIMUM,
        R.INSUFFICIENT_DISK_FOR_VRAM,
        R.SYSBOX_NOT_ENABLED,
        R.FLAGSHIP_WITHOUT_NCU_OR_SPLIT,
        R.CANNOT_APPLY_GPU_POWER_CAP,
        R.OUTDATED_EXECUTOR_IMAGE,
    ],
    ("flagship_everything", False): [
        R.PROVIDER_DISCORD_NOT_CONNECTED,
        R.NVIDIA_DRIVER_BELOW_MINIMUM,
        R.SYSBOX_NOT_ENABLED,
    ],
    ("partial_split_port_limited_old_driver", True): [
        R.NVIDIA_DRIVER_BELOW_MINIMUM,
        R.PORT_LIMITED_REMAINDER,
    ],
}


@pytest.mark.asyncio
async def test_capacity_driver_and_sysbox_are_all_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    # no capacity must not hide the two requirements the provider can fix
    _set_market(monkeypatch, True)

    _, case = await _score_cycle("capacity_driver_sysbox")

    assert case.incentive == 0
    assert _codes(case) == [
        "nvidia_driver_below_minimum",
        "sysbox_not_enabled",
        "no_unrented_capacity_for_gpu_count",
    ]
    log = "\n".join(case.incentive_logs)
    assert OLD_DRIVER in log and "sysbox" in log and "no unrented-incentive capacity" in log


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "enforce"),
    list(EXPECTED_REASONS),
    ids=[f"{name}-{'flags_on' if enforce else 'flags_off'}" for name, enforce in EXPECTED_REASONS],
)
async def test_every_failing_requirement_is_reported(
    monkeypatch: pytest.MonkeyPatch, name: str, enforce: bool
) -> None:
    _set_market(monkeypatch, enforce)

    _, case = await _score_cycle(name)

    assert _codes(case) == [code.value for code in EXPECTED_REASONS[(name, enforce)]]


@pytest.mark.asyncio
async def test_driver_reason_of_an_excluded_node_carries_the_gate_multiplier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # excluded before the rental-share formula, so no multiplier was scored onto the result
    _set_market(monkeypatch, True)

    _, case = await _score_cycle("discord_price_driver")

    driver = next(
        r for r in case.zero_incentive_reasons if r.reason == "nvidia_driver_below_minimum"
    )
    assert driver.context["driver_multiplier"] == 0.0
    assert driver.context["nvidia_driver_version"] == OLD_DRIVER
    assert case.driver_multiplier is None


def test_report_order_is_fixable_first_then_informational() -> None:
    assert sorted(ZERO_INCENTIVE_REPORT_ORDER) == sorted(ZeroIncentiveReason)
    assert len(set(ZERO_INCENTIVE_REPORT_ORDER)) == len(ZeroIncentiveReason)
    assert ZERO_INCENTIVE_REPORT_ORDER[-2:] == (
        R.GPU_MODEL_NOT_ELIGIBLE_FOR_UNRENTED_INCENTIVE,
        R.NO_UNRENTED_CAPACITY_FOR_GPU_COUNT,
    )
    assert set(INFORMATIONAL_ZERO_INCENTIVE_REASONS) == {
        R.GPU_MODEL_NOT_ELIGIBLE_FOR_UNRENTED_INCENTIVE,
        R.NO_UNRENTED_CAPACITY_FOR_GPU_COUNT,
    }


def test_reasons_are_kept_in_report_order_whatever_order_they_are_recorded_in() -> None:
    job = _case_job("capacity_driver_sysbox")

    job.record_incentive_log(
        MinerLogLine.no_payout_because_no_unrented_capacity_for_gpu_count(job, 2)
    )
    job.record_incentive_log(MinerLogLine.no_payout_because_sysbox_not_enabled(job))
    job.record_incentive_log(MinerLogLine.no_payout_because_spot_tier(job))

    assert _codes(job) == ["spot_tier", "sysbox_not_enabled", "no_unrented_capacity_for_gpu_count"]


@pytest.mark.asyncio
async def test_an_already_excluded_node_writes_no_shadow_or_unmeasured_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # guard, passes on main too: the shadow numbers are read against the flags that were on,
    # so a node excluded by an enforced gate must not start appearing in them
    _set_market(monkeypatch, True)
    incentive = RentalPriceIncentive(IncentiveConfig(), AsyncMock(), {}, {})
    job = _job(CASE_EXECUTOR, H200, 1, {"container_cap_eff": 7}, price_per_gpu=50.0)
    shadow_logs: list[str] = []
    for method in (
        "_log_insufficient_disk",
        "_log_insufficient_disk_unmeasured",
        "_log_power_cap_limit",
        "_log_power_cap_unmeasured",
        "_log_flagship_capability_limit",
        "_log_port_limited_remainder",
    ):
        monkeypatch.setattr(incentive, method, lambda *args, _m=method: shadow_logs.append(_m))

    await incentive.calculate_executor_score(job)

    assert job.eligible_for_rental_share is False
    assert _codes(job) == ["price_above_market_p90_soft_limit"]
    assert shadow_logs == []
