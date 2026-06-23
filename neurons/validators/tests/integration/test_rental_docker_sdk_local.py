import os
import socket
import time
import uuid
from dataclasses import dataclass
from io import StringIO

import pytest

from datura.requests.miner_requests import ExecutorSSHInfo
from services.rental_docker_sdk import (
    ContainerExecSpec,
    ContainerRunSpec,
    RentalDockerSdkClient,
    RentalDockerSdkClientFactory,
)


TARGET_IMAGE = "alpine:3.20"
HOSTILE_ENV_VALUE = "value'; echo SHOULD_NOT_RUN; $(echo nope)"
HOSTILE_PUBLIC_KEY = 'ssh-ed25519 AAAA user"; echo KEY_MARKER; $(echo nope)\n'


requires_local_docker = pytest.mark.skipif(
    os.environ.get("RUN_RENTAL_DOCKER_SDK_INTEGRATION") != "1",
    reason="set RUN_RENTAL_DOCKER_SDK_INTEGRATION=1 to run local Docker smoke",
)

requires_ssh_local_docker = pytest.mark.skipif(
    os.environ.get("RUN_RENTAL_DOCKER_SDK_SSH_INTEGRATION") != "1",
    reason=(
        "set RUN_RENTAL_DOCKER_SDK_SSH_INTEGRATION=1 to run local "
        "Docker-over-SSH smoke"
    ),
)


@dataclass(frozen=True)
class LocalSshDockerEndpoint:
    executor_info: ExecutorSSHInfo
    private_key: str


@dataclass(frozen=True)
class RentalLifecycleResult:
    env_value_preserved: bool
    stdin_marker_echoed: bool
    public_key_write_succeeded: bool
    stopped: bool
    restarted: bool
    removed: bool


@requires_local_docker
@pytest.mark.asyncio
async def test_local_docker_sdk_rental_smoke(local_docker_api_client):
    client = RentalDockerSdkClient(local_docker_api_client)
    container_name = _container_name("lium-rental-sdk-smoke")

    try:
        result = await _run_rental_lifecycle(
            client,
            inspector_api_client=local_docker_api_client,
            container_name=container_name,
            stdin_marker="LOCAL_SDK_STDIN_MARKER",
        )
        assert result.env_value_preserved
        assert result.stdin_marker_echoed
        assert result.public_key_write_succeeded
        assert result.stopped
        assert result.restarted
        assert result.removed
    finally:
        _remove_container_if_present(local_docker_api_client, container_name)
        await client.aclose()


@requires_ssh_local_docker
@pytest.mark.asyncio
async def test_local_docker_sdk_factory_over_ssh(
    local_docker_api_client,
    local_ssh_docker_endpoint,
):
    factory = RentalDockerSdkClientFactory()
    container_name = _container_name("lium-rental-sdk-ssh-target")

    # This is the validator-facing SDK path: the factory builds a docker-py
    # ssh:// client from executor SSH info, then the client manages a rental
    # target container through the remote Docker API.
    try:
        async with factory.connect(
            executor_info=local_ssh_docker_endpoint.executor_info,
            private_key=local_ssh_docker_endpoint.private_key,
        ) as rental_client:
            result = await _run_rental_lifecycle(
                rental_client,
                inspector_api_client=local_docker_api_client,
                container_name=container_name,
                stdin_marker="LOCAL_SSH_SDK_STDIN_MARKER",
            )
            assert result.env_value_preserved
            assert result.stdin_marker_echoed
            assert result.public_key_write_succeeded
            assert result.stopped
            assert result.restarted
            assert result.removed
    finally:
        _remove_container_if_present(local_docker_api_client, container_name)


@pytest.fixture
def local_docker_api_client():
    import docker
    from docker.utils import kwargs_from_env

    api_client = docker.APIClient(timeout=60, **kwargs_from_env())
    try:
        api_client.ping()
    except Exception as exc:
        api_client.close()
        pytest.skip(f"local Docker daemon unavailable: {exc}")

    try:
        yield api_client
    finally:
        api_client.close()


@pytest.fixture
def local_docker_client(local_docker_api_client):
    import docker

    client = docker.from_env()
    try:
        yield client
    finally:
        client.close()


@pytest.fixture
def local_ssh_docker_endpoint(tmp_path, local_docker_client):
    import paramiko

    image_tag = f"lium-rental-sdk-ssh-executor:{uuid.uuid4().hex[:12]}"
    executor_name = _container_name("lium-rental-sdk-ssh-executor")
    private_key, authorized_key = _generate_rsa_key_pair(paramiko)
    executor_container = None

    try:
        _build_ssh_executor_image(
            docker_client=local_docker_client,
            build_dir=tmp_path,
            image_tag=image_tag,
        )
        executor_container = _start_ssh_executor_container(
            docker_client=local_docker_client,
            image_tag=image_tag,
            container_name=executor_name,
            authorized_key=authorized_key,
        )

        host, ssh_port = _wait_for_ssh_port(executor_container)
        ssh_host_key = _wait_for_ssh_host_key(
            host,
            ssh_port,
            paramiko,
            executor_container,
        )

        yield LocalSshDockerEndpoint(
            executor_info=_executor_info(
                host=host,
                ssh_port=ssh_port,
                ssh_host_key=ssh_host_key,
            ),
            private_key=private_key,
        )
    finally:
        if executor_container is not None:
            try:
                executor_container.remove(force=True, v=True)
            except Exception:
                pass
        try:
            local_docker_client.images.remove(image_tag, force=True)
        except Exception:
            pass


async def _run_rental_lifecycle(
    rental_client: RentalDockerSdkClient,
    *,
    inspector_api_client,
    container_name: str,
    stdin_marker: str,
) -> RentalLifecycleResult:
    import docker

    await rental_client.pull(image=TARGET_IMAGE)
    await rental_client.run_container(
        ContainerRunSpec(
            image=TARGET_IMAGE,
            name=container_name,
            command=("sh", "-c", "trap : TERM INT; sleep infinity & wait"),
            environment={"HOSTILE_ENV": HOSTILE_ENV_VALUE},
            restart_policy=None,
        )
    )

    inspect = inspector_api_client.inspect_container(container_name)
    env_value_preserved = f"HOSTILE_ENV={HOSTILE_ENV_VALUE}" in (
        inspect["Config"].get("Env") or []
    )

    result = await rental_client.exec_in_container(
        ContainerExecSpec(
            container_name=container_name,
            argv=("sh", "-c", "cat > /tmp/sdk-smoke && cat /tmp/sdk-smoke"),
            stdin=f"{stdin_marker}\nsecond-line\n",
        )
    )
    stdin_marker_echoed = result.exit_status == 0 and stdin_marker in result.stdout

    key_result = await rental_client.exec_in_container(
        ContainerExecSpec(
            container_name=container_name,
            argv=(
                "sh",
                "-c",
                "mkdir -p /root/.ssh && cat >> /root/.ssh/authorized_keys",
            ),
            stdin=HOSTILE_PUBLIC_KEY,
        )
    )
    public_key_write_succeeded = key_result.exit_status == 0

    await rental_client.stop(container_name=container_name)
    stopped = (
        inspector_api_client.inspect_container(container_name)["State"]["Running"]
        is False
    )

    await rental_client.start(container_name=container_name)
    restarted = (
        inspector_api_client.inspect_container(container_name)["State"]["Running"]
        is True
    )

    await rental_client.remove_container(
        container_name=container_name,
        force=True,
        remove_volumes=True,
    )
    try:
        inspector_api_client.inspect_container(container_name)
        removed = False
    except docker.errors.NotFound:
        removed = True

    return RentalLifecycleResult(
        env_value_preserved=env_value_preserved,
        stdin_marker_echoed=stdin_marker_echoed,
        public_key_write_succeeded=public_key_write_succeeded,
        stopped=stopped,
        restarted=restarted,
        removed=removed,
    )


def _build_ssh_executor_image(*, docker_client, build_dir, image_tag: str) -> None:
    dockerfile = build_dir / "Dockerfile"
    dockerfile.write_text(
        "\n".join(
            [
                "FROM alpine:3.20",
                "RUN apk add --no-cache docker-cli openssh-server",
                "RUN mkdir -p /run/sshd /root/.ssh && chmod 700 /root/.ssh",
                "EXPOSE 22",
                "",
            ]
        ),
        encoding="utf-8",
    )
    docker_client.images.build(path=str(build_dir), tag=image_tag, rm=True)


def _start_ssh_executor_container(
    *,
    docker_client,
    image_tag: str,
    container_name: str,
    authorized_key: str,
):
    # Local stand-in for a rented executor host. It exposes sshd and mounts the
    # host Docker socket, so docker-py still uses its real ssh:// transport.
    return docker_client.containers.run(
        image_tag,
        name=container_name,
        detach=True,
        ports={"22/tcp": None},
        volumes={
            "/var/run/docker.sock": {
                "bind": "/var/run/docker.sock",
                "mode": "rw",
            }
        },
        environment={"AUTHORIZED_KEY": authorized_key},
        command=[
            "sh",
            "-c",
            (
                "set -eu; "
                "mkdir -p /root/.ssh /run/sshd; "
                "printf '%s\n' \"$AUTHORIZED_KEY\" > /root/.ssh/authorized_keys; "
                "chmod 700 /root/.ssh; "
                "chmod 600 /root/.ssh/authorized_keys; "
                "printf '%s\n' "
                "'PermitRootLogin yes' "
                "'PasswordAuthentication no' "
                "'PubkeyAuthentication yes' "
                ">> /etc/ssh/sshd_config; "
                "ssh-keygen -A; "
                "exec /usr/sbin/sshd -D -e"
            ),
        ],
    )


def _executor_info(
    *,
    host: str,
    ssh_port: int,
    ssh_host_key: str,
) -> ExecutorSSHInfo:
    return ExecutorSSHInfo(
        uuid=str(uuid.uuid4()),
        address=host,
        port=8080,
        ssh_username="root",
        ssh_port=ssh_port,
        python_path="/usr/bin/python3",
        root_dir="/root",
        ssh_host_key=ssh_host_key,
    )


def _generate_rsa_key_pair(paramiko_module):
    key = paramiko_module.RSAKey.generate(2048)
    private_key = StringIO()
    key.write_private_key(private_key)
    authorized_key = f"{key.get_name()} {key.get_base64()} local-test"
    return private_key.getvalue(), authorized_key


def _container_name(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _remove_container_if_present(api_client, container_name: str) -> None:
    try:
        api_client.remove_container(container_name, force=True, v=True)
    except Exception:
        pass


def _wait_for_ssh_port(container):
    container.reload()
    bindings = container.attrs["NetworkSettings"]["Ports"]["22/tcp"]
    assert bindings, "SSH executor container did not publish port 22"
    host = bindings[0].get("HostIp") or "127.0.0.1"
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    port = int(bindings[0]["HostPort"])

    deadline = time.monotonic() + 30
    last_error = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return host, port
        except OSError as exc:
            last_error = exc
            time.sleep(0.25)

    logs = container.logs(tail=100).decode(errors="replace")
    raise AssertionError(
        f"SSH executor did not accept connections on {host}:{port}: "
        f"{last_error}; logs={logs}"
    )


def _wait_for_ssh_host_key(host: str, port: int, paramiko_module, container) -> str:
    deadline = time.monotonic() + 30
    last_error = None
    while time.monotonic() < deadline:
        try:
            return _read_ssh_host_key(host, port, paramiko_module)
        except Exception as exc:
            last_error = exc
            time.sleep(0.25)

    logs = container.logs(tail=100).decode(errors="replace")
    raise AssertionError(
        f"SSH executor did not expose a usable host key on {host}:{port}: "
        f"{last_error}; logs={logs}"
    )


def _read_ssh_host_key(host: str, port: int, paramiko_module) -> str:
    sock = socket.create_connection((host, port), timeout=5)
    try:
        transport = paramiko_module.Transport(sock)
    except Exception:
        sock.close()
        raise
    try:
        transport.start_client(timeout=5)
        key = transport.get_remote_server_key()
        return f"{key.get_name()} {key.get_base64()}"
    finally:
        transport.close()
