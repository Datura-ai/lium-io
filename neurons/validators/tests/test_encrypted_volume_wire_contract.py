import json

from payload_models.payloads import (
    BackupContainerRequest,
    BaseServerRequest,
    ContainerDeleteRequest,
    RestoreContainerRequest,
)


def _base_payload(message_type: str) -> dict:
    return {
        "message_type": message_type,
        "miner_hotkey": "miner-hotkey",
        "executor_id": "executor-id",
        "pod_id": "pod-id",
    }


def _backup_volume_info() -> dict:
    return {
        "name": "backup-bucket",
        "plugin": "s3fs",
        "iam_user_access_key": "access-key",
        "iam_user_secret_key": "secret-key",
    }


def test_backend_delete_json_parses():
    payload = {
        **_base_payload("ContainerDeleteRequest"),
        "container_name": "pod_abc",
        "local_volume": "volume_abc",
    }

    parsed = BaseServerRequest.parse(json.dumps(payload))

    assert isinstance(parsed, ContainerDeleteRequest)
    assert parsed.local_volume == "volume_abc"


def test_backend_backup_json_parses_encryption_fields():
    payload = {
        **_base_payload("BackupContainerRequest"),
        "source_volume": "volume_abc",
        "backup_volume_info": _backup_volume_info(),
        "backup_path": "/workspace/data",
        "source_volume_path": "/workspace",
        "backup_target_path": "pod/backup.tgz",
        "auth_token": "token",
        "backup_log_id": "backup-log-id",
        "volume_encrypted": True,
        "container_name": "pod_abc",
        "s3_connections": 64,
    }

    parsed = BaseServerRequest.parse(json.dumps(payload))

    assert isinstance(parsed, BackupContainerRequest)
    assert parsed.volume_encrypted is True
    assert parsed.container_name == "pod_abc"
    assert not hasattr(parsed, "s3_connections")


def test_backend_restore_json_parses_encryption_fields():
    payload = {
        **_base_payload("RestoreContainerRequest"),
        "target_volume": "volume_abc",
        "backup_volume_info": _backup_volume_info(),
        "backup_source_path": "pod/backup.tgz",
        "target_volume_path": "/workspace",
        "auth_token": "token",
        "restore_log_id": "restore-log-id",
        "restore_path": "/workspace/data",
        "volume_encrypted": True,
        "container_name": "pod_abc",
        "s3_connections": 64,
    }

    parsed = BaseServerRequest.parse(json.dumps(payload))

    assert isinstance(parsed, RestoreContainerRequest)
    assert parsed.volume_encrypted is True
    assert parsed.container_name == "pod_abc"
    assert not hasattr(parsed, "s3_connections")
