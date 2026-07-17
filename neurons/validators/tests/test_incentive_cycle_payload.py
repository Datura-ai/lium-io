from datetime import UTC, datetime

from protocol.vc_protocol.validator_requests import ExecutorSpecRequest


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
        collateral_deposited=True,
    )

    assert request.scored_at == scored_at
    assert request.netuid == 51
    assert request.incentive == 0.125
    assert request.incentive_source == "rented_emission"
    assert request.node_state_at_cycle == "rented"
