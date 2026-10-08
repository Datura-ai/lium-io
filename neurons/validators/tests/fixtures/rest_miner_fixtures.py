"""A MinerService whose REST boundary is mocked. A module that loads this plugin defines
`miner_task_result`: executor uuid -> the passing JobResult every executor task resolves to."""

from unittest.mock import AsyncMock, Mock

import pytest
from datura.requests.miner_requests import AcceptSSHKeyRequest, ExecutorSSHInfo
from services.miner_service import MinerService

VALIDATOR_HOTKEY = "validator-hotkey"


def executor_info(executor_id: str) -> ExecutorSSHInfo:
    return ExecutorSSHInfo(
        uuid=executor_id,
        address="198.51.100.7",
        port=8001,
        ssh_username="root",
        ssh_port=2200,
        python_path="/usr/bin/python3",
        root_dir="/root/app",
    )


@pytest.fixture
def wallet(mocker):
    my_key = Mock(ss58_address=VALIDATOR_HOTKEY)
    my_key.sign.return_value = b"\x01\x02\x03"
    mocker.patch(
        "core.config.Settings.get_bittensor_wallet",
        return_value=Mock(get_hotkey=Mock(return_value=my_key)),
    )
    return my_key


@pytest.fixture
def rest_miner_service(mocker, wallet, monkeypatch, miner_task_result):
    """The miner accepts the key and returns the executors the test hands it."""
    from core.config import settings

    monkeypatch.setattr(settings, "USE_REST_API", True)
    ssh_service = mocker.Mock()
    ssh_service.generate_ssh_key.return_value = (b"---PRIV---", b"ssh-ed25519 pub")
    ssh_service.decrypt_payload.return_value = "---DECRYPTED-PRIV---"
    task_service = mocker.Mock()

    async def create_task(miner_info, executor_info, **_):
        return miner_task_result(executor_info.uuid)

    task_service.create_task = AsyncMock(side_effect=create_task)
    service = MinerService(
        ssh_service=ssh_service,
        task_service=task_service,
        redis_service=mocker.AsyncMock(),
        attestation_service=Mock(maybe_issue_nonce=AsyncMock(return_value=None)),
    )
    mocker.patch("services.miner_service.measure_and_attach", AsyncMock())

    def miner_returns(*executor_ids: str):
        service.rest_calls = []

        async def _make_rest_request(method, url, json_data, headers, timeout, log_extra, operation_name):
            service.rest_calls.append((url.rsplit("/", 1)[-1], json_data))
            if url.endswith("ssh-pubkey-submit"):
                return 200, AcceptSSHKeyRequest(
                    executors=[executor_info(e) for e in executor_ids]
                ).model_dump(mode="json")
            return 200, {"message_type": "SSHKeyRemoved"}

        service._make_rest_request = _make_rest_request

    service.miner_returns = miner_returns
    return service
