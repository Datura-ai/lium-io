"""DAH-1482: renter secrets reach the pod as 0400 files on a tmpfs at /run/lium/secrets, behind
POD_SECRETS_TMPFS_ENABLED — never in the container env, /etc/environment, exec argv or the logs; with
the flag off a rent is exactly today's."""

import dataclasses
import logging
import os
import stat
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
    CONTAINER_DEFAULT_USER,
    POD_SECRETS_DIR,
    POD_SECRETS_TMPFS_OPTIONS,
    POD_SECRETS_TMPFS_SIZE_BYTES,
    ContainerExecResult,
    ContainerExecSpec,
    RentalDockerSdkClient,
    _build_host_config_kwargs,
    build_pod_secrets_owner_probe_spec,
    build_pod_secrets_tmpfs,
    build_secret_file_exec_specs,
    parse_pod_secrets_owner,
    valid_pod_secrets,
)
from payload_models.payloads import FailedContainerRequest
from test_rental_docker_sdk import FakeApiClient
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
    return [
        spec
        for spec in docker_client.exec_specs
        if spec.stdin is not None and POD_SECRETS_DIR in " ".join(spec.argv)
    ]


def _handover_specs(docker_client):
    return [
        spec
        for spec in docker_client.exec_specs
        if spec.stdin is None and "chown -h" in " ".join(spec.argv)
    ]


# What `id -u && id -g` prints when Docker runs it as the image's Config.User; an unknown name fails
# the exec the way Docker does ("unable to find user").
IMAGE_USERS = {
    "": "0\n0\n",
    "root": "0\n0\n",
    "1000": "1000\n0\n",
    "1000:1000": "1000\n1000\n",
    "app": "1001\n1002\n",
    "app:staff": "1001\n50\n",
}


def _answer_owner_probe(monkeypatch, docker_client, image_user: str):
    original_exec = docker_client.exec_in_container

    async def exec_as(spec):
        result = await original_exec(spec)
        if spec.user is CONTAINER_DEFAULT_USER:
            if image_user not in IMAGE_USERS:
                return ContainerExecResult(exit_status=126, stderr=f"unable to find user {image_user}")
            return ContainerExecResult(exit_status=0, stdout=IMAGE_USERS[image_user])
        return result

    monkeypatch.setattr(docker_client, "exec_in_container", exec_as)


async def _create(
    docker_service, executor_info, keypair, monkeypatch, *, flag: bool, secrets, image_user: str = ""
):
    monkeypatch.setattr(settings, "POD_SECRETS_TMPFS_ENABLED", flag)
    docker_client = docker_service.rental_docker_client_factory.client
    docker_client.__init__()
    _answer_owner_probe(monkeypatch, docker_client, image_user)
    _patch_create_harness(monkeypatch, docker_service, RecordingSSHClient())
    payload = _base_create_payload(secrets=secrets)
    result = await docker_service.create_container(
        payload=payload,
        executor_info=executor_info,
        keypair=keypair,
        private_key="encrypted-private-key",
    )
    docker_client.last_result = result
    return payload, docker_client


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


NAME_AS_VALUE_MARKER = "hfPASTEDASNAMEMARKER"
# 64 base64-ish characters that also match the name pattern once any '=' padding is stripped
RANDOM_TOKEN = "Zq3xK9vLmP2wR7tYbN4cJ8hG1sD6fA0eUoIiQyTrWxVz5B" + "k" * 18

# (rejected name, the pasted value that must never be shown)
PASTED_NAMES = [
    (f"HF_TOKEN={NAME_AS_VALUE_MARKER}", NAME_AS_VALUE_MARKER),
    (f"HF TOKEN={NAME_AS_VALUE_MARKER}", NAME_AS_VALUE_MARKER),
    (f"={NAME_AS_VALUE_MARKER}", NAME_AS_VALUE_MARKER),
    (f"{NAME_AS_VALUE_MARKER}==", NAME_AS_VALUE_MARKER),
    (f"{NAME_AS_VALUE_MARKER}=", NAME_AS_VALUE_MARKER),
    (f"{RANDOM_TOKEN}==", RANDOM_TOKEN),
    (f"{RANDOM_TOKEN}=", RANDOM_TOKEN),
    (f"{RANDOM_TOKEN}-x", RANDOM_TOKEN),
    (f"1{NAME_AS_VALUE_MARKER}", NAME_AS_VALUE_MARKER),
    ("HF_TOKEN=", "HF_TOKEN"),
    ("==", "=="),
]


@pytest.mark.parametrize("name, pasted", PASTED_NAMES)
def test_no_part_of_a_refused_name_is_ever_echoed(name, pasted):
    with pytest.raises(ValueError) as excinfo:
        valid_pod_secrets({name: "value"})
    message = str(excinfo.value)
    assert pasted not in message
    assert name not in message
    assert message == rental_docker_sdk._invalid_secret_name_message(None)


@pytest.mark.asyncio
@pytest.mark.parametrize("name, pasted", PASTED_NAMES)
async def test_a_value_typed_as_a_name_never_reaches_the_rent_result_or_logs(
    docker_service, executor_info, keypair, monkeypatch, caplog, name, pasted
):
    caplog.set_level(logging.DEBUG)
    _, docker_client = await _create(
        docker_service,
        executor_info,
        keypair,
        monkeypatch,
        flag=True,
        secrets={name: "value"},
    )

    result = docker_client.last_result
    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "validate_request"
    logged = "\n".join(
        f"{record.getMessage()} {getattr(record.msg, 'extra', '')}" for record in caplog.records
    )
    assert "Invalid pod secrets" in logged
    for text in [result.msg, result.model_dump_json(), logged]:
        assert pasted not in text
        assert name not in text


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
    _answer_owner_probe(monkeypatch, docker_client, "")
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


def test_the_owner_probe_runs_as_the_image_user_and_parses_uid_gid():
    spec = build_pod_secrets_owner_probe_spec(container_name="pod")
    assert spec.user is CONTAINER_DEFAULT_USER
    assert spec.stdin is None
    assert parse_pod_secrets_owner("1000\n1000\n") == (1000, 1000)
    assert parse_pod_secrets_owner("0\n0") == (0, 0)


@pytest.mark.parametrize("stdout", ["", "1000\n", "app\n1000\n", "1000\n1000\nextra\n", "-1\n0\n"])
def test_an_owner_probe_that_is_not_two_numbers_is_refused(stdout):
    with pytest.raises(ValueError):
        parse_pod_secrets_owner(stdout)


@pytest.mark.asyncio
@pytest.mark.parametrize("user, docker_user", [(CONTAINER_DEFAULT_USER, ""), ("0", "0")])
async def test_exec_create_runs_the_probe_as_the_image_user_and_everything_else_as_root(
    user, docker_user
):
    api_client = FakeApiClient()
    await RentalDockerSdkClient(api_client).exec_in_container(
        ContainerExecSpec(container_name="pod", argv=("id", "-u"), user=user)
    )
    assert api_client.exec_created[0]["user"] == docker_user
    assert ContainerExecSpec(container_name="pod", argv=("true",)).user == "0"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "image_user, owner",
    [
        ("1000", "1000:0"),
        ("1000:1000", "1000:1000"),
        ("app", "1001:1002"),
        ("app:staff", "1001:50"),
        ("", "0:0"),
        ("root", "0:0"),
    ],
)
async def test_secrets_are_owned_by_the_container_user_and_stay_0700_0400(
    docker_service, executor_info, keypair, monkeypatch, image_user, owner
):
    _, docker_client = await _create(
        docker_service,
        executor_info,
        keypair,
        monkeypatch,
        flag=True,
        secrets=SECRETS,
        image_user=image_user,
    )

    assert not isinstance(docker_client.last_result, FailedContainerRequest)
    probes = [spec for spec in docker_client.exec_specs if spec.user is CONTAINER_DEFAULT_USER]
    assert len(probes) == 1
    secret_specs = _secret_specs(docker_client)
    assert len(secret_specs) == len(SECRETS)
    for spec, name in zip(secret_specs, SECRETS):
        script = spec.argv[2]
        assert spec.user == "0"
        assert "chown" not in script
        assert "set -euC" in script
        assert f"chmod 0400 {POD_SECRETS_DIR}/.{name}.partial;" in script
    handovers = _handover_specs(docker_client)
    assert len(handovers) == 1 and handovers[0].user == "0"
    assert docker_client.exec_specs.index(handovers[0]) > max(
        docker_client.exec_specs.index(spec) for spec in secret_specs
    )
    script = handovers[0].argv[2]
    assert f'chown -h {owner} "$path"; chmod 0400 "$path"' in script
    assert script.endswith(f"chown -h {owner} {POD_SECRETS_DIR}; chmod 0700 {POD_SECRETS_DIR}")
    for name in SECRETS:
        assert f"{POD_SECRETS_DIR}/{name}" in script
    assert "mode=0700" in POD_SECRETS_TMPFS_OPTIONS.split(",")


@pytest.mark.asyncio
@pytest.mark.parametrize("image_user", ["nosuchuser", "ghost:nogroup"])
async def test_an_unresolvable_container_user_fails_the_rent_closed(
    docker_service, executor_info, keypair, monkeypatch, caplog, image_user
):
    caplog.set_level(logging.DEBUG)
    _, docker_client = await _create(
        docker_service,
        executor_info,
        keypair,
        monkeypatch,
        flag=True,
        secrets=SECRETS,
        image_user=image_user,
    )

    result = docker_client.last_result
    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "write_pod_secrets"
    logged = "\n".join(
        f"{record.getMessage()} {getattr(record.msg, 'extra', '')}" for record in caplog.records
    )
    assert "Failed to resolve pod secrets owner" in logged
    assert _secret_specs(docker_client) == []
    for text in [result.msg, logged]:
        for value in SECRET_VALUES:
            assert value not in text


def test_a_secret_bigger_than_the_tmpfs_is_refused_naming_only_the_secret():
    huge = "BIG_SECRET_VALUE_MARKER" + "x" * POD_SECRETS_TMPFS_SIZE_BYTES
    with pytest.raises(ValueError) as excinfo:
        valid_pod_secrets({"HF_TOKEN": "hf_SECRET_VALUE_MARKER", "BIG": huge})
    assert "BIG" in str(excinfo.value)
    assert "BIG_SECRET_VALUE_MARKER" not in str(excinfo.value)
    assert "hf_SECRET_VALUE_MARKER" not in str(excinfo.value)


def test_secrets_that_together_overflow_the_tmpfs_are_refused():
    half = "HALF_VALUE_MARKER" + "x" * (POD_SECRETS_TMPFS_SIZE_BYTES // 2)
    with pytest.raises(ValueError) as excinfo:
        valid_pod_secrets({"A": half, "B": half})
    assert "secret B " in str(excinfo.value)
    assert "A" not in str(excinfo.value).replace("secrets tmpfs", "")
    assert "HALF_VALUE_MARKER" not in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        valid_pod_secrets({"SMALL": "s", "A": half, "OTHER": "o", "B": half})
    message = str(excinfo.value)
    assert "secret B " in message
    for innocent in ("SMALL", "OTHER"):
        assert innocent not in message
    exactly_full = {"A": "x" * (POD_SECRETS_TMPFS_SIZE_BYTES // 2), "B": "y" * (POD_SECRETS_TMPFS_SIZE_BYTES // 2)}
    assert valid_pod_secrets(exactly_full) == exactly_full


@pytest.mark.asyncio
async def test_an_oversize_secret_fails_the_rent_before_anything_is_created(
    docker_service, executor_info, keypair, monkeypatch
):
    huge = "BIG_SECRET_VALUE_MARKER" + "x" * POD_SECRETS_TMPFS_SIZE_BYTES
    _, docker_client = await _create(
        docker_service,
        executor_info,
        keypair,
        monkeypatch,
        flag=True,
        secrets={"BIG": huge},
    )

    result = docker_client.last_result
    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "validate_request"
    assert "BIG" in result.msg and "BIG_SECRET_VALUE_MARKER" not in result.msg
    assert docker_service.rental_docker_client_factory.connect_calls == []
    assert docker_client.run_specs == [] and docker_client.exec_specs == []


def _run_script(spec):
    return subprocess.run(list(spec.argv), input=spec.stdin, capture_output=True, text=True, timeout=30)


@pytest.fixture
def plain_secrets_dir(tmp_path, monkeypatch):
    secrets_dir = tmp_path / "secrets"
    secrets_dir.mkdir(mode=0o700)
    victim = tmp_path / "victim"
    victim.write_text("original\n")
    victim.chmod(0o644)
    monkeypatch.setattr(rental_docker_sdk, "POD_SECRETS_DIR", str(secrets_dir))
    # a plain directory stands in for the mount; only the tmpfs/root-only guard is skipped
    monkeypatch.setattr(rental_docker_sdk, "_pod_secrets_dir_guard", lambda secrets_dir: "")
    return secrets_dir, victim


def test_the_write_script_needs_a_root_only_tmpfs_dir_that_is_not_a_link():
    script = build_secret_file_exec_specs(container_name="pod", secrets={"A": "v"})[0].argv[2]
    handover = rental_docker_sdk.build_pod_secrets_handover_spec(
        container_name="pod", secrets={"A": "v"}, owner=(1000, 1000)
    ).argv[2]
    for text in (script, handover):
        assert f"[ ! -L {POD_SECRETS_DIR} ]" in text
        assert f"grep -qs ' '{POD_SECRETS_DIR}' tmpfs ' /proc/mounts" in text
        assert f'[ "$(stat -c %u:%a {POD_SECRETS_DIR})" = 0:700 ]' in text


def test_a_clean_write_then_handover_gives_the_owner_0400_files_in_a_0700_dir(plain_secrets_dir, monkeypatch):
    secrets_dir, _ = plain_secrets_dir
    owner = (os.getuid(), os.getgid())
    for spec in build_secret_file_exec_specs(container_name="pod", secrets=SECRETS):
        assert _run_script(spec).returncode == 0
    handover = rental_docker_sdk.build_pod_secrets_handover_spec(container_name="pod", secrets=SECRETS, owner=owner)
    assert _run_script(handover).returncode == 0
    assert sorted(os.listdir(secrets_dir)) == sorted(SECRETS)
    for name, value in SECRETS.items():
        info = (secrets_dir / name).lstat()
        assert stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o400
        assert (info.st_uid, info.st_gid) == owner
        assert (secrets_dir / name).read_text() == value
    assert stat.S_IMODE(secrets_dir.stat().st_mode) == 0o700


@pytest.mark.parametrize("planted", [".HF_TOKEN.partial", "HF_TOKEN"])
@pytest.mark.parametrize("kind", ["symlink", "dangling symlink", "file"])
def test_a_planted_path_cannot_redirect_a_write_and_the_write_fails_closed(
    plain_secrets_dir, monkeypatch, planted, kind
):
    secrets_dir, victim = plain_secrets_dir
    path = secrets_dir / planted
    if kind == "symlink":
        path.symlink_to(victim)
    elif kind == "dangling symlink":
        path.symlink_to(secrets_dir.parent / "does-not-exist")
    else:
        path.write_text("planted")
    spec = build_secret_file_exec_specs(container_name="pod", secrets={"HF_TOKEN": SECRETS["HF_TOKEN"]})[0]

    run = _run_script(spec)

    assert run.returncode != 0
    assert "already exists" in run.stderr
    assert victim.read_text() == "original\n"
    assert stat.S_IMODE(victim.stat().st_mode) == 0o644
    assert not (secrets_dir.parent / "does-not-exist").exists()
    for value in SECRET_VALUES:
        assert value not in run.stderr


def test_a_link_swapped_in_between_writes_fails_the_handover_without_touching_its_target(
    plain_secrets_dir, monkeypatch
):
    secrets_dir, victim = plain_secrets_dir
    specs = build_secret_file_exec_specs(container_name="pod", secrets=SECRETS)
    assert _run_script(specs[0]).returncode == 0
    (secrets_dir / "HF_TOKEN").unlink()
    (secrets_dir / "HF_TOKEN").symlink_to(victim)
    assert _run_script(specs[1]).returncode == 0
    handover = rental_docker_sdk.build_pod_secrets_handover_spec(
        container_name="pod", secrets=SECRETS, owner=(os.getuid(), os.getgid())
    )

    run = _run_script(handover)

    assert run.returncode != 0
    assert "is not a regular file" in run.stderr
    assert stat.S_IMODE(victim.stat().st_mode) == 0o644
    assert victim.read_text() == "original\n"
    assert stat.S_IMODE(secrets_dir.stat().st_mode) == 0o700


@pytest.mark.asyncio
async def test_a_failed_handover_fails_the_rent(docker_service, executor_info, keypair, monkeypatch):
    monkeypatch.setattr(settings, "POD_SECRETS_TMPFS_ENABLED", True)
    docker_client = docker_service.rental_docker_client_factory.client
    docker_client.__init__()
    _patch_create_harness(monkeypatch, docker_service, RecordingSSHClient())
    _answer_owner_probe(monkeypatch, docker_client, "1000")
    original_exec = docker_client.exec_in_container

    async def failing_handover(spec):
        result = await original_exec(spec)
        if "chown -h" in " ".join(spec.argv):
            return ContainerExecResult(exit_status=1, stderr="HF_TOKEN is not a regular file")
        return result

    monkeypatch.setattr(docker_client, "exec_in_container", failing_handover)
    result = await docker_service.create_container(
        payload=_base_create_payload(secrets=SECRETS),
        executor_info=executor_info,
        keypair=keypair,
        private_key="encrypted-private-key",
    )

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "write_pod_secrets"
    assert len(_handover_specs(docker_client)) == 1
