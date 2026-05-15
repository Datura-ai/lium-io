from unittest.mock import AsyncMock, MagicMock

import pytest
from datura.requests.miner_requests import AcceptSSHKeyRequest, ExecutorSSHInfo
from payload_models.payloads import MinerJobEnryptedFiles, MinerJobRequestPayload

import services.miner_service as miner_service_module
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse
from services.forced_validation import (
    ForceValidationConflict,
    ForceValidationNotFound,
    ForceValidationRequestStore,
)
from services.miner_service import MinerService
from services.task_service import JobResult


class FakeRedis:
    def __init__(self):
        self.data = {}

    async def set(self, key, value, nx=False, ex=None):
        if nx and key in self.data:
            return None
        self.data[key] = value
        return True

    async def get(self, key):
        return self.data.get(key)

    async def delete(self, key):
        self.data.pop(key, None)


class FakeRedisService:
    def __init__(self):
        self.redis = FakeRedis()


def _executor(executor_id: str = "exec-1") -> ExecutorSSHInfo:
    return ExecutorSSHInfo(
        uuid=executor_id,
        address="10.0.0.1",
        port=8000,
        ssh_username="root",
        ssh_port=22,
        python_path="/usr/bin/python3",
        root_dir="/root",
    )


def _payload() -> MinerJobRequestPayload:
    return MinerJobRequestPayload(
        job_batch_id="forced-req",
        miner_hotkey="miner-hotkey",
        miner_coldkey="miner-coldkey",
        miner_address="127.0.0.1",
        miner_port=8000,
    )


def _encrypted_files() -> MinerJobEnryptedFiles:
    return MinerJobEnryptedFiles(
        encrypt_key="key",
        all_keys={},
        tmp_directory="/tmp",
        machine_scrape_file_name="machine.py",
    )


def _job_result(executor: ExecutorSSHInfo) -> JobResult:
    return JobResult(
        spec={},
        executor_info=executor,
        score=1,
        job_score=1,
        job_batch_id="forced-req",
        log_status="info",
        log_text="ok",
        gpu_model="H100",
        gpu_count=1,
    )


@pytest.mark.asyncio
async def test_store_accepts_request_and_returns_status():
    store = ForceValidationRequestStore(
        FakeRedisService(), request_ttl_seconds=60, active_ttl_seconds=60
    )

    record = await store.create_request(
        executor_id="exec-1",
        miner_hotkey="miner-hotkey",
    )
    loaded = await store.get_request(record.request_id)

    assert loaded.request_id == record.request_id
    assert loaded.executor_id == "exec-1"
    assert loaded.status == "queued"


@pytest.mark.asyncio
async def test_store_returns_latest_request_for_executor():
    store = ForceValidationRequestStore(
        FakeRedisService(), request_ttl_seconds=60, active_ttl_seconds=60
    )
    first_record = await store.create_request(
        executor_id="exec-1",
        miner_hotkey="miner-hotkey",
    )
    await store.update(first_record.request_id, status="failed", stage="completed")
    await store.release_active_executor("exec-1", first_record.request_id)
    second_record = await store.create_request(
        executor_id="exec-1",
        miner_hotkey="miner-hotkey",
    )

    loaded = await store.get_latest_request("exec-1")

    assert loaded.request_id == second_record.request_id


@pytest.mark.asyncio
async def test_store_rejects_duplicate_active_executor():
    store = ForceValidationRequestStore(
        FakeRedisService(), request_ttl_seconds=60, active_ttl_seconds=60
    )
    await store.create_request(executor_id="exec-1", miner_hotkey="miner-hotkey")

    with pytest.raises(ForceValidationConflict):
        await store.create_request(executor_id="exec-1", miner_hotkey="miner-hotkey")


@pytest.mark.asyncio
async def test_store_releases_active_marker_after_terminal_state():
    store = ForceValidationRequestStore(
        FakeRedisService(), request_ttl_seconds=60, active_ttl_seconds=60
    )
    record = await store.create_request(
        executor_id="exec-1",
        miner_hotkey="miner-hotkey",
    )
    await store.update(record.request_id, status="failed", stage="completed")
    await store.release_active_executor("exec-1", record.request_id)

    next_record = await store.create_request(
        executor_id="exec-1",
        miner_hotkey="miner-hotkey",
    )
    assert next_record.request_id != record.request_id


@pytest.mark.asyncio
async def test_store_removes_latest_pointer_after_terminal_state():
    store = ForceValidationRequestStore(
        FakeRedisService(), request_ttl_seconds=60, active_ttl_seconds=60
    )
    record = await store.create_request(
        executor_id="exec-1",
        miner_hotkey="miner-hotkey",
    )

    await store.update(record.request_id, status="failed", stage="completed")

    loaded = await store.get_request(record.request_id)
    assert loaded.status == "failed"
    with pytest.raises(ForceValidationNotFound):
        await store.get_latest_request("exec-1")


@pytest.mark.asyncio
async def test_store_keeps_newer_latest_pointer_when_old_request_finishes_late():
    store = ForceValidationRequestStore(
        FakeRedisService(), request_ttl_seconds=60, active_ttl_seconds=60
    )
    old_record = await store.create_request(
        executor_id="exec-1",
        miner_hotkey="miner-hotkey",
    )
    await store.release_active_executor("exec-1", old_record.request_id)
    new_record = await store.create_request(
        executor_id="exec-1",
        miner_hotkey="miner-hotkey",
    )

    await store.update(old_record.request_id, status="failed", stage="completed")

    loaded = await store.get_latest_request("exec-1")
    assert loaded.request_id == new_record.request_id


@pytest.mark.asyncio
async def test_store_hides_existing_terminal_latest_request():
    store = ForceValidationRequestStore(
        FakeRedisService(), request_ttl_seconds=60, active_ttl_seconds=60
    )
    record = await store.create_request(
        executor_id="exec-1",
        miner_hotkey="miner-hotkey",
    )
    await store.update(record.request_id, status="failed", stage="completed")
    await store.redis_service.redis.set(store._latest_key("exec-1"), record.request_id)

    with pytest.raises(ForceValidationNotFound):
        await store.get_latest_request("exec-1")
    assert await store.redis_service.redis.get(store._latest_key("exec-1")) is None


@pytest.mark.asyncio
async def test_single_executor_validation_runs_matching_executor_and_cleans_up(monkeypatch):
    executor = _executor("exec-1")
    service = MinerService.__new__(MinerService)
    service.ssh_service = MagicMock()
    service.ssh_service.generate_ssh_key.return_value = (b"private", b"public")
    service.task_service = MagicMock()
    service.task_service.create_task = AsyncMock(return_value=_job_result(executor))
    service._make_rest_request = AsyncMock(
        return_value=(200, AcceptSSHKeyRequest(executors=[executor]).model_dump(mode="json"))
    )
    service._remove_ssh_key_via_rest = AsyncMock(return_value=True)

    keypair = MagicMock(ss58_address="validator-hotkey")
    keypair.sign.return_value = b"sig"
    wallet = MagicMock()
    wallet.get_hotkey.return_value = keypair
    monkeypatch.setattr("services.miner_service.settings.USE_REST_API", True)
    monkeypatch.setattr(
        type(miner_service_module.settings),
        "get_bittensor_wallet",
        lambda self: wallet,
    )

    result = await service.request_single_executor_validation(
        payload=_payload(),
        encrypted_files=_encrypted_files(),
        rented_data=RentedExecutorsResponse(executors={}),
        executor_id="exec-1",
    )

    assert result.executor_info.uuid == "exec-1"
    service.task_service.create_task.assert_awaited_once()
    service._remove_ssh_key_via_rest.assert_awaited_once()


@pytest.mark.asyncio
async def test_single_executor_validation_rejects_missing_executor_and_cleans_up(monkeypatch):
    service = MinerService.__new__(MinerService)
    service.ssh_service = MagicMock()
    service.ssh_service.generate_ssh_key.return_value = (b"private", b"public")
    service.task_service = MagicMock()
    service.task_service.create_task = AsyncMock()
    service._make_rest_request = AsyncMock(
        return_value=(200, AcceptSSHKeyRequest(executors=[_executor("other")]).model_dump(mode="json"))
    )
    service._remove_ssh_key_via_rest = AsyncMock(return_value=True)

    keypair = MagicMock(ss58_address="validator-hotkey")
    keypair.sign.return_value = b"sig"
    wallet = MagicMock()
    wallet.get_hotkey.return_value = keypair
    monkeypatch.setattr("services.miner_service.settings.USE_REST_API", True)
    monkeypatch.setattr(
        type(miner_service_module.settings),
        "get_bittensor_wallet",
        lambda self: wallet,
    )

    with pytest.raises(ValueError):
        await service.request_single_executor_validation(
            payload=_payload(),
            encrypted_files=_encrypted_files(),
            rented_data=RentedExecutorsResponse(executors={}),
            executor_id="exec-1",
        )

    service.task_service.create_task.assert_not_called()
    service._remove_ssh_key_via_rest.assert_awaited_once()
