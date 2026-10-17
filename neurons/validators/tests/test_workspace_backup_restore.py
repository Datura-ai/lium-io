import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

MINER_JOBS_PATH = Path(__file__).resolve().parents[1] / "src" / "miner_jobs"
sys.path.insert(0, str(MINER_JOBS_PATH))

from datura.requests.miner_requests import ExecutorSSHInfo
from services.docker_service import DockerService
from workspace_mount import (
    normalize_workspace_path,
)

from services import docker_service as docker_service_module


def test_workspace_path_normalization_rejects_escape():
    assert normalize_workspace_path("/root/test", "/root") == "/workspace/test"
    assert normalize_workspace_path("test", "/root") == "/workspace/test"
    assert normalize_workspace_path("/workspace/test", "/root") == "/workspace/test"
    with pytest.raises(ValueError, match="escapes"):
        normalize_workspace_path("/root/../etc", "/root")


def _executor_info() -> ExecutorSSHInfo:
    return ExecutorSSHInfo(
        uuid="executor",
        address="127.0.0.1",
        port=8080,
        ssh_username="root",
        ssh_port=22,
        port_mappings="[]",
        port_range="40000-40100",
        python_path="/usr/bin/python3",
        root_dir="/root",
    )


def _walk_json(value, path=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _walk_json(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_json(item, f"{path}[{index}]")
    else:
        yield path, value


@pytest.mark.asyncio
async def test_encrypted_bootstrap_restore_spec_never_carries_the_passphrase(monkeypatch):
    # DAH-3274: the operation spec is SFTP'd into the executor container = a file on the
    # provider's disk. It must reach the data through the running pod, never unlock it.
    import json

    from services.volume_keys import derive_volume_passphrase

    master_secret = "test-master-secret-32-chars-long!!"
    monkeypatch.setattr(docker_service_module.settings, "VOLUME_MASTER_SECRET", master_secret)
    monkeypatch.setattr(docker_service_module, "supports_storage_operation", AsyncMock(return_value=True))
    start = AsyncMock(return_value={"spec": "/tmp/spec.json"})
    monkeypatch.setattr(docker_service_module, "start_storage_operation", start)
    monkeypatch.setattr(docker_service_module, "wait_for_storage_operation", AsyncMock())
    service = DockerService.__new__(DockerService)
    restore = SimpleNamespace(
        backup_engine="restic",
        restore_log_id="33333333-3333-4333-8333-333333333333",
        restore_path="",
        backup_volume_info=SimpleNamespace(
            name="bucket",
            iam_user_access_key="ak",
            iam_user_secret_key="sk",
            session_token=None,
        ),
        repository_password="repo-password",
        repository_pod_id="pod-1",
        snapshot_id="a" * 64,
        legacy_object_key=None,
        legacy_object_size_bytes=None,
        auth_token="token",
        failure_timeout_seconds=600,
    )

    await service._run_bootstrap_restore(
        ssh_client=AsyncMock(),
        executor_info=_executor_info(),
        payload=SimpleNamespace(pod_id="pod-1"),
        restore=restore,
        local_volume="volume_pod-1",
        local_volume_path="/root",
        encrypted=True,
        container_name="pod_pod-1",
    )

    spec = start.await_args.args[3]
    passphrase = derive_volume_passphrase(master_secret, "pod-1")
    assert passphrase not in json.dumps(spec)
    assert not [path for path, _ in _walk_json(spec) if "passphrase" in path.lower()]
    assert spec["workspace"]["mode"] == "encrypted_running"
    assert spec["workspace"]["container_name"] == "pod_pod-1"
    assert spec["workspace"]["requested_path"] == "/root"
    # create-time: the entrypoint may have touched the fresh mount already; the backup wins
    assert spec["workspace"]["bootstrap"] is True


