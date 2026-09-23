"""DAH-1482: renter secrets reach the pod as 0400 files on a tmpfs at /run/lium/secrets, behind
POD_SECRETS_TMPFS_ENABLED — never in the container env, /etc/environment, exec argv or the logs; with
the flag off a rent is exactly today's."""

import dataclasses
import logging
import subprocess
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from core.config import settings
from datura.requests.miner_requests import ExecutorSSHInfo
from payload_models.payloads import ContainerCreateRequest
from services import rental_docker_sdk
from services.docker_service import DockerService
from services.rental_docker_sdk import (
    POD_SECRETS_DIR,
    POD_SECRETS_TMPFS_OPTIONS,
    _build_host_config_kwargs,
    build_pod_secrets_tmpfs,
    build_secret_file_exec_specs,
    valid_pod_secrets,
)
from test_docker_service_rental_security import (
    RecordingRentalDockerFactory,
    RecordingSSHClient,
    _base_create_payload,
    _patch_create_harness,
)

SECRETS = {
    "HF_TOKEN": "hf_SECRET_VALUE_MARKER",
    "WANDB_API_KEY": "wandb'; echo SECRET_SHELL_MARKER; $(id)",
}
SECRET_VALUES = tuple(SECRETS.values())


@pytest.fixture
def docker_service():
    lock = AsyncMock()
    lock.__aenter__ = AsyncMock(return_value=lock)
    lock.__aexit__ = AsyncMock(return_value=None)
    redis_service = Mock()
    redis_service.acquire_executor_lock = Mock(return_value=lock)
    return DockerService(
        ssh_service=Mock(),
        redis_service=redis_service,
        attestation_service=Mock(),
        rental_docker_client_factory=RecordingRentalDockerFactory(),
    )


@pytest.fixture
def executor_info():
    return ExecutorSSHInfo(
        uuid=str(uuid4()),
        address="127.0.0.1",
        port=8080,
        ssh_username="root",
        ssh_port=2200,
        python_path="/usr/bin/python3",
        root_dir="/root/app",
        ssh_host_key="ssh-ed25519 AAAATESTKEY",
    )


@pytest.fixture
def keypair():
    return Mock(ss58_address="validator-hotkey")


def _secret_specs(docker_client):
    return [spec for spec in docker_client.exec_specs if POD_SECRETS_DIR in " ".join(spec.argv)]


async def _create(docker_service, executor_info, keypair, monkeypatch, *, flag: bool, secrets):
    monkeypatch.setattr(settings, "POD_SECRETS_TMPFS_ENABLED", flag)
    docker_service.rental_docker_client_factory.client.__init__()
    _patch_create_harness(monkeypatch, docker_service, RecordingSSHClient())
    payload = _base_create_payload(secrets=secrets)
    await docker_service.create_container(
        payload=payload,
        executor_info=executor_info,
        keypair=keypair,
        private_key="encrypted-private-key",
    )
    return payload, docker_service.rental_docker_client_factory.client


def test_mount_spec_is_a_private_noexec_tmpfs_at_run_lium_secrets():
    assert POD_SECRETS_DIR == "/run/lium/secrets"
    assert build_pod_secrets_tmpfs(SECRETS) == {POD_SECRETS_DIR: POD_SECRETS_TMPFS_OPTIONS}
    options = POD_SECRETS_TMPFS_OPTIONS.split(",")
    assert {"noexec", "nosuid", "nodev", "mode=0700"} <= set(options)
    assert build_pod_secrets_tmpfs({}) == {}
    assert build_pod_secrets_tmpfs(None) == {}


def test_each_secret_is_one_exec_with_the_value_on_stdin_only():
    specs = build_secret_file_exec_specs(container_name="pod", secrets=SECRETS)

    assert [spec.stdin for spec in specs] == list(SECRET_VALUES)
    for spec, name in zip(specs, SECRETS):
        script = spec.argv[2]
        assert spec.argv[:2] == ("sh", "-c")
        assert f"{POD_SECRETS_DIR}/{name}" in script
        assert "chmod 0400" in script and "umask 077" in script
        assert "/etc/environment" not in script
        assert spec.environment == {}
        for value in SECRET_VALUES:
            assert value not in " ".join(spec.argv)


@pytest.mark.parametrize("name", ["", "1ABC", "A-B", "../etc/passwd", "A B", "A;id", "x" * 129])
def test_a_name_that_is_not_a_plain_identifier_is_refused(name):
    with pytest.raises(ValueError):
        valid_pod_secrets({name: "value"})


def test_an_empty_value_is_refused_without_echoing_other_values():
    with pytest.raises(ValueError) as excinfo:
        valid_pod_secrets({"HF_TOKEN": "hf_SECRET_VALUE_MARKER", "EMPTY": ""})
    assert "hf_SECRET_VALUE_MARKER" not in str(excinfo.value)


def test_the_write_refuses_a_directory_that_is_not_a_tmpfs(tmp_path, monkeypatch):
    """Negative control on a real shell: an ordinary directory stands in for a missing mount."""
    plain_dir = tmp_path / "secrets"
    plain_dir.mkdir()
    monkeypatch.setattr(rental_docker_sdk, "POD_SECRETS_DIR", str(plain_dir))
    spec = build_secret_file_exec_specs(
        container_name="pod", secrets={"HF_TOKEN": "hf_SECRET_VALUE_MARKER"}
    )[0]

    run = subprocess.run(
        list(spec.argv), input=spec.stdin, capture_output=True, text=True, timeout=30
    )

    assert run.returncode == 1
    assert "is not a tmpfs mount" in run.stderr
    assert list(plain_dir.iterdir()) == []


def test_the_request_is_parsed_but_never_shown_or_dumped_with_a_value():
    payload = ContainerCreateRequest.model_validate(
        {**_base_create_payload().model_dump(), "secrets": SECRETS}
    )
    assert payload.secrets == SECRETS
    for value in SECRET_VALUES:
        assert value not in str(payload)
        assert value not in repr(payload)
        assert value not in payload.model_dump_json()
    assert "secrets" not in payload.model_dump()


@pytest.mark.asyncio
async def test_flag_on_mounts_the_tmpfs_and_writes_files_with_no_env_leak(
    docker_service, executor_info, keypair, monkeypatch, caplog
):
    caplog.set_level(logging.DEBUG)
    payload, docker_client = await _create(
        docker_service, executor_info, keypair, monkeypatch, flag=True, secrets=SECRETS
    )

    run_spec = docker_client.run_specs[0]
    assert run_spec.tmpfs == {POD_SECRETS_DIR: POD_SECRETS_TMPFS_OPTIONS}
    assert _build_host_config_kwargs(run_spec)["tmpfs"] == {
        POD_SECRETS_DIR: POD_SECRETS_TMPFS_OPTIONS
    }
    # `docker inspect` Env is exactly create_container's environment
    assert not set(SECRETS) & set(run_spec.environment)
    assert not set(SECRET_VALUES) & set(run_spec.environment.values())

    env_specs = [
        spec for spec in docker_client.exec_specs if "/etc/environment" in " ".join(spec.argv)
    ]
    assert len(env_specs) == 1
    for value in SECRET_VALUES:
        assert value not in env_specs[0].stdin

    secret_specs = _secret_specs(docker_client)
    assert [spec.stdin for spec in secret_specs] == list(SECRET_VALUES)
    for spec in docker_client.exec_specs:
        for value in SECRET_VALUES:
            assert value not in " ".join(spec.argv)

    logged = "\n".join(
        f"{record.getMessage()} {getattr(record.msg, 'extra', '')}" for record in caplog.records
    )
    assert "exec_write_pod_secret" in logged
    for value in SECRET_VALUES:
        assert value not in logged


@pytest.mark.asyncio
async def test_flag_off_is_todays_rent_even_when_the_backend_sends_secrets(
    docker_service, executor_info, keypair, monkeypatch
):
    _, with_secrets = await _create(
        docker_service, executor_info, keypair, monkeypatch, flag=False, secrets=SECRETS
    )
    with_secrets_run = dataclasses.asdict(with_secrets.run_specs[0])
    with_secrets_host = _build_host_config_kwargs(with_secrets.run_specs[0])
    with_secrets_execs = [dataclasses.asdict(spec) for spec in with_secrets.exec_specs]

    _, today = await _create(
        docker_service, executor_info, keypair, monkeypatch, flag=False, secrets=None
    )

    assert with_secrets_run == dataclasses.asdict(today.run_specs[0])
    assert with_secrets_host == _build_host_config_kwargs(today.run_specs[0])
    assert "tmpfs" not in with_secrets_host
    assert with_secrets_execs == [dataclasses.asdict(spec) for spec in today.exec_specs]
    assert _secret_specs(with_secrets) == []


@pytest.mark.asyncio
async def test_flag_on_without_secrets_adds_nothing(
    docker_service, executor_info, keypair, monkeypatch
):
    _, flag_on = await _create(
        docker_service, executor_info, keypair, monkeypatch, flag=True, secrets=None
    )
    flag_on_host = _build_host_config_kwargs(flag_on.run_specs[0])
    flag_on_execs = [dataclasses.asdict(spec) for spec in flag_on.exec_specs]

    _, flag_off = await _create(
        docker_service, executor_info, keypair, monkeypatch, flag=False, secrets=None
    )

    assert flag_on_host == _build_host_config_kwargs(flag_off.run_specs[0])
    assert flag_on_execs == [dataclasses.asdict(spec) for spec in flag_off.exec_specs]


@pytest.mark.asyncio
async def test_a_failed_secret_write_fails_the_rent_naming_only_the_secret(
    docker_service, executor_info, keypair, monkeypatch
):
    monkeypatch.setattr(settings, "POD_SECRETS_TMPFS_ENABLED", True)
    docker_client = docker_service.rental_docker_client_factory.client
    docker_client.__init__()
    _patch_create_harness(monkeypatch, docker_service, RecordingSSHClient())
    original_exec = docker_client.exec_in_container

    async def failing_secret_exec(spec):
        result = await original_exec(spec)
        if POD_SECRETS_DIR in " ".join(spec.argv):
            return rental_docker_sdk.ContainerExecResult(
                exit_status=1, stderr=f"{POD_SECRETS_DIR} is not a tmpfs mount"
            )
        return result

    monkeypatch.setattr(docker_client, "exec_in_container", failing_secret_exec)
    stream_messages = []

    async def record_stream_log(message, *args, **kwargs):
        stream_messages.append(message)

    monkeypatch.setattr(docker_service, "stream_log", record_stream_log)
    await docker_service.create_container(
        payload=_base_create_payload(secrets=SECRETS),
        executor_info=executor_info,
        keypair=keypair,
        private_key="encrypted-private-key",
    )

    assert "Failed to write secret HF_TOKEN" in stream_messages
    assert len(_secret_specs(docker_client)) == 1
    for message in stream_messages:
        for value in SECRET_VALUES:
            assert value not in str(message)
