from datetime import UTC, datetime

from protocol.vc_protocol.validator_requests import ExecutorSpecRequest
from services.task.models import JobResult


def test_executor_spec_accepts_structured_incentive_cycle_fields() -> None:
    scored_at = datetime(2026, 7, 17, 0, 2, tzinfo=UTC)

    request = ExecutorSpecRequest(
        miner_hotkey="miner",
        miner_coldkey="coldkey",
        validator_hotkey="validator",
        executor_uuid="dc95d60e-b0c9-4016-a669-360d2bc08904",
        executor_ip="127.0.0.1",
        executor_port=8000,
        specs={},
        score=1.0,
        synthetic_job_score=1.0,
        log_text="ok",
        log_status="success",
        job_batch_id="2026-07-16 23:55:00",
        netuid=51,
        scored_at=scored_at,
        incentive=0.125,
        incentive_source="rented_emission",
        node_state_at_cycle="rented",
        incentive_formula_version="mining_v1",
        incentive_formula_inputs={
            "mining_share": 0.13,
            "mining_score": 0.25,
            "total_mining_score": 0.26,
        },
        collateral_deposited=True,
    )

    assert request.scored_at == scored_at
    assert request.netuid == 51
    assert request.incentive == 0.125
    assert request.incentive_source == "rented_emission"
    assert request.node_state_at_cycle == "rented"
    assert request.incentive_formula_version == "mining_v1"
    assert request.incentive_formula_inputs == {
        "mining_share": 0.13,
        "mining_score": 0.25,
        "total_mining_score": 0.26,
    }


def test_job_result_snapshots_idle_formula_inputs() -> None:
    result = JobResult.model_construct(
        eligible_for_rental_share=True,
        gpu_model="NVIDIA H200",
        gpu_count=8,
        rental_share=0.08,
        rental_share_raw=0.09,
        total_burn_emission=0.87,
        burn_share=0.79,
        hourly_rate=4.0,
        unrented_cap_multiplier=0.5,
        sysbox_multiplier=1.0,
        driver_multiplier=1.0,
        effective_rate=2.0,
        total_rental_cost=16.0,
        count_bucket=8,
        max_cap=64,
        total_unrented_by_gpu_type=128,
        cap_dilution_applied=True,
        validator_tao_price_usd=300.0,
        validator_alpha_rate_tao_per_block=0.001,
        estimated_epoch_emission_usd=108.0,
        tempo_blocks=360,
        seconds_per_block=12,
        fixed_ratio=0.5,
    )

    assert result.incentive_formula_version == "rental_price_v2"
    assert result.incentive_formula_inputs["max_cap"] == 64
    assert result.incentive_formula_inputs["validator_tao_price_usd"] == 300.0
    assert result.incentive_formula_inputs["validator_alpha_rate_tao_per_block"] == 0.001
    assert result.incentive_formula_inputs["tempo_blocks"] == 360
