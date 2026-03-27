import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from clients.compute_client import ComputeClient
from payload_models.payloads import GetEstimateRequest


@pytest.mark.asyncio
async def test_handle_get_estimate_queues_protocol_safe_response_and_logs(monkeypatch):
    client = ComputeClient.__new__(ComputeClient)
    client.lock = asyncio.Lock()
    client.message_queue = []
    client.logging_extra = {"validator_hotkey": "validator-hotkey"}
    client.miner_service = MagicMock()
    client.miner_service.redis_service = AsyncMock()
    client.miner_service.redis_service.get_incentive_snapshot = AsyncMock(
        return_value={
            "epoch_subnet_emission": 1.0,
            "rental_share": 0.2,
            "burn_share": 0.1,
            "mining": {
                "total_gpu_count": 1,
                "total_mining_score": 1.0,
                "total_gpu_model_count_map": {"H100": 1},
            },
            "rental": {
                "total_rental_cost": 10.0,
                "by_gpu_type": {
                    "H100": {
                        "unrented_count": 1,
                        "max_cap": 16,
                        "cap_multiplier": 1.0,
                        "weighted_rate_sum": 3.5,
                    }
                },
            },
        }
    )

    estimate_result = MagicMock()
    estimate_result.model_dump.return_value = {"gpu_model": "H100", "usd_per_epoch": 12.5}
    logger_info = MagicMock()
    estimate_executor_mock = AsyncMock(return_value=estimate_result)

    monkeypatch.setattr("clients.compute_client.estimate_executor", estimate_executor_mock)
    monkeypatch.setattr("clients.compute_client.logger.info", logger_info)
    monkeypatch.setattr("clients.compute_client.time.perf_counter", MagicMock(side_effect=[1.0, 1.05, 1.12, 1.25]))

    await client._handle_get_estimate(
        GetEstimateRequest(
            request_id="req-1",
            gpu_model="H100",
            gpu_count=2,
            is_rented=False,
            gpu_splitting=True,
            gpu_splitting_min_count=1,
            sysbox_runtime=True,
            collateral_deposited=True,
        )
    )

    assert len(client.message_queue) == 1
    response = client.message_queue[0]
    payload = response.model_dump()
    assert payload["request_id"] == "req-1"
    assert payload["estimate"] == {"gpu_model": "H100", "usd_per_epoch": 12.5}
    assert "snapshot" not in payload

    assert logger_info.call_count == 1
    message = logger_info.call_args.args[0]
    assert str(message) == "Processed estimate request"
    assert message.extra["gpu_model"] == "H100"
    assert message.extra["request_id"] == "req-1"
    assert message.extra["snapshot_extraction_ms"] == 70.0
    assert message.extra["total_processing_ms"] == 250.0

    params = estimate_executor_mock.await_args.kwargs["params"]
    assert params.gpu_splitting is True
    assert params.gpu_splitting_min_count == 1
