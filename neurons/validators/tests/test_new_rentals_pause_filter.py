import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from datura.requests.miner_requests import (
    AcceptSSHKeyRequest,
    ExecutorSSHInfo,
    RequestType,
)

from core.config import settings
from payload_models.payloads import MinerJobRequestPayload
from protocol.vc_protocol.compute_requests import (
    RentedExecutor,
    RentedExecutorsResponse,
    RentedPod,
)
from services.miner_service import MinerService
from services.task.result_handler import ResultHandler


def _executor(executor_id: str) -> ExecutorSSHInfo:
    return ExecutorSSHInfo(
        uuid=executor_id,
        address="127.0.0.1",
        port=22,
        ssh_username="root",
        ssh_port=22,
        python_path="/usr/bin/python3",
        root_dir="/tmp",
    )


def _rented_data(
    *,
    rented_ids: list[str] | None = None,
    paused_ids: list[str] | None = None,
) -> RentedExecutorsResponse:
    executors = {
        executor_id: RentedExecutor(
            miner_hotkey="miner-hotkey",
            executor_ip_address="127.0.0.1",
            executor_ip_port="22",
            pods=[
                RentedPod(
                    pod_id=f"pod-{executor_id}",
                    container_name=f"pod_{executor_id}",
                )
            ],
        )
        for executor_id in rented_ids or []
    }
    return RentedExecutorsResponse(
        executors=executors,
        banned_guids=[],
        new_rentals_paused_executor_ids=paused_ids or [],
    )


def _payload() -> MinerJobRequestPayload:
    return MinerJobRequestPayload(
        job_batch_id="2026-05-28 00:00:00",
        miner_hotkey="miner-hotkey",
        miner_coldkey="miner-coldkey",
        miner_address="127.0.0.1",
        miner_port=8091,
    )


def _accepted_response(*executor_ids: str) -> AcceptSSHKeyRequest:
    return AcceptSSHKeyRequest(
        message_type=RequestType.AcceptSSHKeyRequest,
        executors=[_executor(executor_id) for executor_id in executor_ids],
    )


def _keypair() -> MagicMock:
    keypair = MagicMock()
    keypair.ss58_address = "validator-hotkey"
    keypair.sign.return_value = b"\x01" * 64
    return keypair


def _wallet(keypair: MagicMock) -> MagicMock:
    wallet = MagicMock()
    wallet.get_hotkey.return_value = keypair
    return wallet


def _miner_service() -> MinerService:
    service = MinerService.__new__(MinerService)
    service.ssh_service = MagicMock()
    service.ssh_service.generate_ssh_key.return_value = (
        b"private-key",
        b"public-key",
    )
    service.task_service = MagicMock()
    service.redis_service = MagicMock()
    service.attestation_service = MagicMock()
    # request_job_to_miner awaits this before submitting the SSH key (G3);
    # None = no attestation event, the legacy path.
    service.attestation_service.maybe_issue_nonce = AsyncMock(return_value=None)
    return service


async def _create_task_result(**kwargs):
    return SimpleNamespace(executor_info=kwargs["executor_info"], score=1)


class _FakeMinerClient:
    def __init__(self, accepted_message: AcceptSSHKeyRequest):
        future = asyncio.get_running_loop().create_future()
        future.set_result(accepted_message)
        self.job_state = SimpleNamespace(
            miner_accepted_ssh_key_or_failed_future=future
        )
        self.send_model = AsyncMock()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None


def test_rented_executors_response_defaults_new_rentals_paused_executor_ids():
    """Older backend responses without paused IDs remain compatible."""
    result = RentedExecutorsResponse.model_validate({"executors": {}})

    assert result.new_rentals_paused_executor_ids == []


def test_rented_executors_response_parses_new_rentals_paused_executor_ids():
    """Backend paused executor IDs are available to incentive handling."""
    result = RentedExecutorsResponse.model_validate(
        {
            "executors": {},
            "new_rentals_paused_executor_ids": ["paused-executor"],
        }
    )

    assert result.new_rentals_paused_executor_ids == ["paused-executor"]


def test_rented_executors_response_defaults_provider_discord_connected_executor_ids_to_unknown():
    """Older backend responses without Discord executor IDs remain non-penalizing."""
    result = RentedExecutorsResponse.model_validate({"executors": {}})

    assert result.provider_discord_connected_executor_ids is None


def test_rented_executors_response_parses_provider_discord_connected_executor_ids():
    """Backend Discord executor IDs are available to incentive handling."""
    result = RentedExecutorsResponse.model_validate(
        {
            "executors": {},
            "provider_discord_connected_executor_ids": ["discord-executor"],
        }
    )

    assert result.provider_discord_connected_executor_ids == ["discord-executor"]


def test_rented_executors_response_defaults_default_job_owner_by_executor():
    """Older backend responses without default-job owners remain compatible."""
    result = RentedExecutorsResponse.model_validate({"executors": {}})

    assert result.default_job_owner_by_executor == {}


def test_rented_executors_response_parses_default_job_owner_by_executor():
    """Default-job owners are parsed leniently so an unknown future value cannot break the whole response."""
    result = RentedExecutorsResponse.model_validate(
        {
            "executors": {},
            "default_job_owner_by_executor": {
                "miner-executor": "miner",
                "lium-executor": "lium",
                "future-executor": "partner",
            },
        }
    )

    assert result.default_job_owner_by_executor == {
        "miner-executor": "miner",
        "lium-executor": "lium",
        "future-executor": "partner",
    }


def test_get_default_job_owner_returns_owner_or_none():
    """The accessor used by ResultHandler returns the owner for a known executor, None otherwise."""
    result = RentedExecutorsResponse(
        executors={},
        default_job_owner_by_executor={"miner-executor": "miner"},
    )

    assert result.get_default_job_owner("miner-executor") == "miner"
    assert result.get_default_job_owner("unknown-executor") is None


def test_result_handler_treats_missing_provider_discord_list_as_connected():
    """Missing backend field means unknown, so do not penalize during rollout."""
    context = SimpleNamespace(
        state=SimpleNamespace(rented_data=RentedExecutorsResponse.model_validate({"executors": {}})),
        executor=SimpleNamespace(uuid="executor-with-unknown-discord"),
    )

    assert ResultHandler._get_provider_discord_connected(context) is True


def test_result_handler_reads_provider_discord_connected_executor_ids():
    """A present backend list determines whether an executor provider has Discord."""
    connected_context = SimpleNamespace(
        state=SimpleNamespace(
            rented_data=RentedExecutorsResponse(
                executors={},
                provider_discord_connected_executor_ids=["connected-executor"],
            )
        ),
        executor=SimpleNamespace(uuid="connected-executor"),
    )
    disconnected_context = SimpleNamespace(
        state=SimpleNamespace(
            rented_data=RentedExecutorsResponse(
                executors={},
                provider_discord_connected_executor_ids=["connected-executor"],
            )
        ),
        executor=SimpleNamespace(uuid="disconnected-executor"),
    )

    assert ResultHandler._get_provider_discord_connected(connected_context) is True
    assert ResultHandler._get_provider_discord_connected(disconnected_context) is False


@pytest.mark.asyncio
async def test_websocket_request_job_validates_paused_unrented_executors(monkeypatch):
    """Paused executors are still scheduled for validation and spec updates."""
    service = _miner_service()
    service.task_service.create_task = AsyncMock(side_effect=_create_task_result)
    keypair = _keypair()
    fake_client = _FakeMinerClient(
        _accepted_response("paused-executor", "available-executor")
    )
    rented_data = _rented_data(paused_ids=["paused-executor"])

    monkeypatch.setattr(settings, "USE_REST_API", False)
    monkeypatch.setattr(settings, "JOB_TIME_OUT", 300)
    monkeypatch.setattr(
        type(settings),
        "get_bittensor_wallet",
        lambda self: _wallet(keypair),
    )

    with patch("services.miner_service.MinerClient", return_value=fake_client):
        result = await service.request_job_to_miner(
            _payload(),
            MagicMock(),
            rented_data,
        )

    assert [job.executor_info.uuid for job in result["results"]] == ["paused-executor", "available-executor"]
    created_executor_ids = [
        call.kwargs["executor_info"].uuid
        for call in service.task_service.create_task.call_args_list
    ]
    assert created_executor_ids == ["paused-executor", "available-executor"]


@pytest.mark.asyncio
async def test_rest_request_job_validates_paused_unrented_executors(monkeypatch):
    """The REST miner path also validates paused executors."""
    service = _miner_service()
    service.task_service.create_task = AsyncMock(side_effect=_create_task_result)
    service._generate_auth_headers = MagicMock(return_value={})
    service._make_rest_request = AsyncMock(
        return_value=(
            200,
            _accepted_response(
                "paused-executor",
                "available-executor",
            ).model_dump(mode="json"),
        )
    )
    service._remove_ssh_key_via_rest = AsyncMock()
    keypair = _keypair()
    rented_data = _rented_data(paused_ids=["paused-executor"])

    monkeypatch.setattr(settings, "JOB_TIME_OUT", 300)
    monkeypatch.setattr(
        type(settings),
        "get_bittensor_wallet",
        lambda self: _wallet(keypair),
    )

    result = await service._request_job_to_miner(
        _payload(),
        MagicMock(),
        rented_data,
    )

    assert [job.executor_info.uuid for job in result["results"]] == ["paused-executor", "available-executor"]
    created_executor_ids = [
        call.kwargs["executor_info"].uuid
        for call in service.task_service.create_task.call_args_list
    ]
    assert created_executor_ids == ["paused-executor", "available-executor"]
