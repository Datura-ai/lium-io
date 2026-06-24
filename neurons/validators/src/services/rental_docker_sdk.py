from __future__ import annotations

import asyncio
import os
import socket as socket_module
import tempfile
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path

from datura.requests.miner_requests import ExecutorSSHInfo


DEFAULT_DOCKER_PULL_TIMEOUT_SECONDS = 3 * 60 * 60
_DOCKER_SDK_SSH_HOME_LOCK = threading.Lock()


class RentalDockerConnectionError(RuntimeError):
    """Raised when Docker SDK over SSH cannot be constructed safely."""


class RentalDockerOperationError(RuntimeError):
    """Raised when Docker SDK reports a rental Docker operation failure."""


@dataclass(slots=True)
class PortBinding:
    container_port: int
    host_port: int
    protocol: str = "tcp"


@dataclass(slots=True)
class VolumeMount:
    source: str
    target: str
    read_only: bool = False


@dataclass(slots=True)
class DeviceMount:
    path_on_host: str
    path_in_container: str | None = None
    permissions: str = "rwm"


@dataclass(slots=True)
class GpuDeviceRequest:
    count: int | None = None
    device_ids: tuple[str, ...] = ()
    capabilities: tuple[tuple[str, ...], ...] = (("gpu",),)


@dataclass(slots=True)
class GpuDockerConfig:
    device_requests: tuple[GpuDeviceRequest, ...] = ()
    device_mounts: tuple[DeviceMount, ...] = ()


@dataclass(slots=True)
class ContainerRunSpec:
    image: str
    name: str
    command: tuple[str, ...] = ()
    environment: dict[str, str] = field(default_factory=dict)
    ports: tuple[PortBinding, ...] = ()
    volumes: tuple[VolumeMount, ...] = ()
    restart_policy: str | None = "unless-stopped"
    runtime: str | None = None
    cap_add: tuple[str, ...] = ()
    sysctls: dict[str, str] = field(default_factory=dict)
    devices: tuple[DeviceMount, ...] = ()
    device_requests: tuple[GpuDeviceRequest, ...] = ()
    cpu_count: int | None = None
    memory_gb: int | None = None
    storage_limit_gb: int | None = None
    shm_size: str | None = None
    entrypoint: str | None = None


@dataclass(slots=True)
class ContainerExecSpec:
    container_name: str
    argv: tuple[str, ...]
    stdin: str | bytes | None = None
    environment: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ContainerExecResult:
    exit_status: int
    stdout: str = ""
    stderr: str = ""


class RentalDockerSdkClient:
    def __init__(
        self,
        api_client,
        *,
        pull_timeout_seconds: int | float | None = DEFAULT_DOCKER_PULL_TIMEOUT_SECONDS,
    ):
        self._api_client = api_client
        self._pull_timeout_seconds = pull_timeout_seconds

    async def login(self, *, username: str, password: str) -> None:
        await self._call_api(
            operation_label="login",
            api_method=self._api_client.login,
            username=username,
            password=password,
            reauth=True,
        )

    async def pull(self, *, image: str) -> None:
        timeout_seconds = self._normalized_pull_timeout_seconds()
        try:
            pull_call = asyncio.to_thread(self._pull_sync, image)
            if timeout_seconds is not None:
                await asyncio.wait_for(
                    pull_call,
                    timeout=timeout_seconds,
                )
            else:
                await pull_call
        except asyncio.TimeoutError as exc:
            with suppress(Exception):
                await self.aclose()
            raise RentalDockerOperationError(
                f"Docker SDK pull timed out after {timeout_seconds} seconds"
            ) from exc
        except Exception as exc:
            raise RentalDockerOperationError(
                _wrap_error_message("Docker SDK pull failed", exc)
            ) from exc

    async def image_exists(self, *, image: str) -> bool:
        try:
            await asyncio.to_thread(self._api_client.inspect_image, image)
        except Exception as exc:
            if _is_docker_not_found_error(exc):
                return False
            raise RentalDockerOperationError(
                _wrap_error_message("Docker SDK inspect image failed", exc)
            ) from exc
        return True

    async def run_container(self, spec: ContainerRunSpec) -> None:
        try:
            await asyncio.to_thread(self._run_container_sync, spec)
        except Exception as exc:
            raise RentalDockerOperationError(
                _wrap_error_message("Docker SDK run container failed", exc)
            ) from exc

    async def exec_in_container(self, spec: ContainerExecSpec) -> ContainerExecResult:
        try:
            return await asyncio.to_thread(self._exec_in_container_sync, spec)
        except Exception as exc:
            raise RentalDockerOperationError(
                _wrap_error_message("Docker SDK exec failed", exc)
            ) from exc

    async def start(self, *, container_name: str) -> None:
        await self._call_api(
            container_name,
            operation_label="start",
            api_method=self._api_client.start,
        )

    async def stop(self, *, container_name: str) -> None:
        await self._call_api(
            container_name,
            operation_label="stop",
            api_method=self._api_client.stop,
        )

    async def remove_container(
        self,
        *,
        container_name: str,
        force: bool = True,
        remove_volumes: bool = True,
    ) -> None:
        await self._call_api(
            container_name,
            operation_label="remove container",
            api_method=self._api_client.remove_container,
            force=force,
            v=remove_volumes,
        )

    async def create_volume(
        self,
        *,
        volume_name: str,
        driver: str | None = None,
        driver_opts: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> None:
        try:
            await asyncio.to_thread(
                self._create_volume_sync,
                volume_name=volume_name,
                driver=driver,
                driver_opts=driver_opts,
                timeout=timeout,
            )
        except Exception as exc:
            raise RentalDockerOperationError(
                _wrap_error_message("Docker SDK create volume failed", exc)
            ) from exc

    async def remove_volume(self, *, volume_name: str, force: bool = False) -> None:
        await self._call_api(
            volume_name,
            operation_label="remove volume",
            api_method=self._api_client.remove_volume,
            force=force,
        )

    async def prune_images(self) -> None:
        await self._call_api(
            operation_label="prune images",
            api_method=self._api_client.prune_images,
        )

    async def aclose(self) -> None:
        close = getattr(self._api_client, "close", None)
        if close is not None:
            await asyncio.to_thread(close)

    async def _call_api(self, *args, operation_label: str, api_method, **kwargs) -> None:
        try:
            await asyncio.to_thread(api_method, *args, **kwargs)
        except Exception as exc:
            raise RentalDockerOperationError(
                _wrap_error_message(f"Docker SDK {operation_label} failed", exc)
            ) from exc

    def _run_container_sync(self, spec: ContainerRunSpec) -> None:
        host_config = self._api_client.create_host_config(
            **_build_host_config_kwargs(spec)
        )
        self._api_client.create_container(
            image=spec.image,
            command=list(spec.command) or None,
            detach=True,
            ports=_container_ports(spec.ports) or None,
            environment=spec.environment or None,
            volumes=_container_volumes(spec.volumes) or None,
            name=spec.name,
            entrypoint=spec.entrypoint or None,
            host_config=host_config,
        )
        self._api_client.start(spec.name)

    def _create_volume_sync(
        self,
        *,
        volume_name: str,
        driver: str | None,
        driver_opts: dict[str, str] | None,
        timeout: int | None,
    ) -> None:
        original_timeout = getattr(self._api_client, "timeout", None)
        should_override_timeout = timeout is not None and hasattr(
            self._api_client,
            "timeout",
        )
        if should_override_timeout:
            self._api_client.timeout = None if timeout == 0 else timeout
        try:
            self._api_client.create_volume(
                name=volume_name,
                driver=driver,
                driver_opts=driver_opts,
            )
        finally:
            if should_override_timeout:
                self._api_client.timeout = original_timeout

    def _pull_sync(self, image: str) -> None:
        # docker.APIClient.pull() hardcodes timeout=None for /images/create.
        # Call the same endpoint directly so rental image pulls keep the
        # create_container timeout cap instead of waiting forever.
        from docker import auth, utils

        repository, image_tag = utils.parse_repository_tag(image)
        tag = image_tag or "latest"
        registry, _ = auth.resolve_repository_name(repository)
        response = self._api_client._post(
            self._api_client._url("/images/create"),
            params={"tag": tag, "fromImage": repository},
            headers=_build_pull_headers(self._api_client, registry),
            stream=True,
            timeout=self._normalized_pull_timeout_seconds(),
        )
        self._api_client._raise_for_status(response)

        for event in self._api_client._stream_helper(response, decode=True) or ():
            _raise_pull_event_error(event)

    def _normalized_pull_timeout_seconds(self) -> int | float | None:
        if self._pull_timeout_seconds is None or self._pull_timeout_seconds <= 0:
            return None
        return self._pull_timeout_seconds

    def _exec_in_container_sync(self, spec: ContainerExecSpec) -> ContainerExecResult:
        stdin_data = _encode_exec_stdin(spec.stdin)
        exec_create_result = self._api_client.exec_create(
            container=spec.container_name,
            cmd=list(spec.argv),
            stdin=stdin_data is not None,
            environment=spec.environment or None,
        )
        exec_id = exec_create_result["Id"]

        if stdin_data is None:
            output = self._api_client.exec_start(exec_id, demux=True)
        else:
            exec_socket = self._api_client.exec_start(exec_id, socket=True)
            output = _write_stdin_and_read_exec_output(exec_socket, stdin_data)

        inspect_result = self._api_client.exec_inspect(exec_id)
        stdout, stderr = _decode_exec_output(output)
        return ContainerExecResult(
            exit_status=int(inspect_result.get("ExitCode") or 0),
            stdout=stdout,
            stderr=stderr,
        )


@dataclass(slots=True)
class RentalDockerSdkClientFactory:
    api_client_factory: Callable[..., object] | None = None
    timeout: int = 60
    pull_timeout_seconds: int = DEFAULT_DOCKER_PULL_TIMEOUT_SECONDS

    @asynccontextmanager
    async def connect(
        self,
        *,
        executor_info: ExecutorSSHInfo,
        private_key: str,
    ) -> AsyncIterator[RentalDockerSdkClient]:
        if not executor_info.ssh_host_key or not executor_info.ssh_host_key.strip():
            raise RentalDockerConnectionError(
                "Executor SSH host key is required for Docker SDK SSH access"
            )

        with tempfile.TemporaryDirectory(prefix="lium-rental-docker-ssh-") as temp_dir:
            ssh_home = Path(temp_dir)
            ssh_dir = ssh_home / ".ssh"
            ssh_dir.mkdir(mode=0o700)

            key_path = ssh_dir / "id_executor"
            key_path.write_text(private_key)
            key_path.chmod(0o600)

            known_hosts_path = ssh_dir / "known_hosts"
            known_hosts_path.write_text(
                _build_known_hosts_text(
                    host=executor_info.address,
                    port=executor_info.ssh_port,
                    host_key=executor_info.ssh_host_key,
                )
            )
            known_hosts_path.chmod(0o600)
            _validate_paramiko_known_hosts(known_hosts_path)

            config_path = ssh_dir / "config"
            config_path.write_text(
                "\n".join(
                    [
                        f"Host {executor_info.address}",
                        f"    HostName {executor_info.address}",
                        f"    Port {executor_info.ssh_port}",
                        f"    User {executor_info.ssh_username}",
                        f"    IdentityFile {key_path}",
                        "    IdentitiesOnly yes",
                        "",
                    ]
                )
            )
            config_path.chmod(0o600)

            try:
                api_client = await asyncio.to_thread(
                    self._create_api_client,
                    _build_docker_ssh_base_url(executor_info),
                    ssh_home,
                )
            except Exception as exc:
                raise RentalDockerConnectionError(
                    _wrap_error_message("Docker SDK client construction failed", exc)
                ) from exc

            client = RentalDockerSdkClient(
                api_client,
                pull_timeout_seconds=self.pull_timeout_seconds,
            )
            try:
                yield client
            finally:
                await client.aclose()

    def _create_api_client(self, base_url: str, ssh_home: Path):
        with _DOCKER_SDK_SSH_HOME_LOCK:
            original_home = os.environ.get("HOME")
            os.environ["HOME"] = str(ssh_home)
            try:
                factory = self.api_client_factory or _default_docker_api_client_factory
                return factory(
                    base_url=base_url,
                    timeout=self.timeout,
                    use_ssh_client=False,
                )
            finally:
                if original_home is None:
                    os.environ.pop("HOME", None)
                else:
                    os.environ["HOME"] = original_home


def build_container_command_argv(startup_commands: str | None) -> tuple[str, ...]:
    if not startup_commands or not startup_commands.strip():
        return ()
    import shlex

    try:
        return tuple(shlex.split(startup_commands))
    except ValueError:
        return ()


def build_gpu_docker_config(
    gpu_uuids: tuple[str, ...] | list[str] | None,
    *,
    device_nodes: tuple[str, ...] | list[str] = (),
) -> GpuDockerConfig:
    requested_uuids = tuple(gpu_uuids or ())
    device_request = (
        GpuDeviceRequest(device_ids=requested_uuids)
        if requested_uuids
        else GpuDeviceRequest(count=-1)
    )
    return GpuDockerConfig(
        device_requests=(device_request,),
        device_mounts=tuple(
            DeviceMount(path_on_host=node, path_in_container=node)
            for node in device_nodes
        ),
    )


def build_authorized_keys_exec_spec(
    *,
    container_name: str,
    public_keys: list[str] | tuple[str, ...],
    target_path: str = "/root/.ssh/authorized_keys",
) -> ContainerExecSpec:
    import shlex

    key_data = "".join(f"{public_key}\n" for public_key in public_keys)
    return ContainerExecSpec(
        container_name=container_name,
        argv=(
            "sh",
            "-c",
            f"mkdir -p {shlex.quote(target_path.rsplit('/', 1)[0])} "
            f"&& cat >> {shlex.quote(target_path)}",
        ),
        stdin=key_data,
    )


def build_remove_authorized_keys_exec_spec(
    *,
    container_name: str,
    public_keys: list[str] | tuple[str, ...],
    target_path: str = "/root/.ssh/authorized_keys",
) -> ContainerExecSpec:
    import shlex

    key_data = "".join(f"{public_key}\n" for public_key in public_keys)
    quoted_dir = shlex.quote(target_path.rsplit("/", 1)[0])
    quoted_path = shlex.quote(target_path)
    # This shell runs inside the target container. Public keys are supplied via
    # stdin and matched from a temp file, so key contents are never interpolated
    # into host-side Docker shell text or into this argv.
    script = (
        "set -e; "
        f"mkdir -p {quoted_dir} && "
        f"touch {quoted_path} && "
        "keys=$(mktemp) && filtered=$(mktemp) && "
        "trap 'rm -f \"$keys\" \"$filtered\"' EXIT && "
        "cat > \"$keys\" && "
        f"if grep -vxF -f \"$keys\" {quoted_path} > \"$filtered\"; then "
        ":; else status=$?; "
        "[ \"$status\" -eq 1 ] || exit \"$status\"; fi; "
        f"cat \"$filtered\" > {quoted_path}"
    )
    return ContainerExecSpec(
        container_name=container_name,
        argv=("sh", "-c", script),
        stdin=key_data,
    )


def build_environment_exec_spec(
    *,
    container_name: str,
    environment: dict[str, str] | None,
) -> ContainerExecSpec | None:
    env_lines = [
        f"{key}={value}"
        for key, value in (environment or {}).items()
        if key and value and key.strip() and str(value).strip()
    ]
    if not env_lines:
        return None
    return ContainerExecSpec(
        container_name=container_name,
        argv=("sh", "-c", "cat >> /etc/environment"),
        stdin="".join(f"{line}\n" for line in env_lines),
    )


def _default_docker_api_client_factory(**kwargs):
    import docker

    return docker.APIClient(**kwargs)


def _build_host_config_kwargs(spec: ContainerRunSpec) -> dict:
    kwargs = {
        "port_bindings": _port_bindings(spec.ports),
        "binds": _binds(spec.volumes),
        "restart_policy": _restart_policy(spec.restart_policy),
        "runtime": spec.runtime,
        "cap_add": list(spec.cap_add) or None,
        "sysctls": spec.sysctls or None,
        "devices": _devices(spec.devices),
        "device_requests": _device_requests(spec.device_requests),
        "nano_cpus": spec.cpu_count * 1_000_000_000 if spec.cpu_count else None,
        "mem_limit": f"{spec.memory_gb}g" if spec.memory_gb else None,
        "storage_opt": (
            {"size": f"{spec.storage_limit_gb}g"}
            if spec.storage_limit_gb
            else None
        ),
        "shm_size": spec.shm_size,
    }
    return {key: value for key, value in kwargs.items() if value is not None}


def _container_ports(ports: tuple[PortBinding, ...]) -> list[str]:
    return [_port_key(port) for port in ports]


def _port_bindings(ports: tuple[PortBinding, ...]) -> dict[str, int]:
    return {_port_key(port): port.host_port for port in ports}


def _port_key(port: PortBinding) -> str:
    return f"{port.container_port}/{port.protocol}"


def _container_volumes(volumes: tuple[VolumeMount, ...]) -> list[str]:
    return [volume.target for volume in volumes]


def _binds(volumes: tuple[VolumeMount, ...]) -> list[str]:
    return [
        f"{volume.source}:{volume.target}:{'ro' if volume.read_only else 'rw'}"
        for volume in volumes
    ]


def _restart_policy(policy: str | None) -> dict[str, str] | None:
    if not policy:
        return None
    return {"Name": policy}


def _devices(devices: tuple[DeviceMount, ...]) -> list[str]:
    return [_device_arg(device) for device in devices]


def _device_arg(device: DeviceMount) -> str:
    target = device.path_in_container or device.path_on_host
    return f"{device.path_on_host}:{target}:{device.permissions}"


def _device_requests(device_requests: tuple[GpuDeviceRequest, ...]) -> list:
    if not device_requests:
        return []

    from docker.types import DeviceRequest

    return [
        DeviceRequest(
            count=device_request.count,
            device_ids=list(device_request.device_ids) or None,
            capabilities=[list(capability) for capability in device_request.capabilities],
        )
        for device_request in device_requests
    ]


def _encode_exec_stdin(value: str | bytes | None) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value
    return value.encode()


def _write_stdin_and_read_exec_output(exec_socket, stdin_data: bytes):
    try:
        _write_socket_data(exec_socket, stdin_data)
        _shutdown_socket_write(exec_socket)
        return _read_exec_socket_output(exec_socket)
    finally:
        close = getattr(exec_socket, "close", None)
        if close is not None:
            close()


def _write_socket_data(exec_socket, data: bytes) -> None:
    if not data:
        return

    write_socket = _socket_write_target(exec_socket)
    sendall = getattr(write_socket, "sendall", None)
    if sendall is not None:
        sendall(data)
        return

    write = getattr(write_socket, "write", None)
    if write is not None:
        write(data)
        flush = getattr(write_socket, "flush", None)
        if flush is not None:
            flush()
        return

    send = getattr(write_socket, "send", None)
    if send is None:
        raise TypeError("Docker SDK exec socket does not support stdin writes")

    sent = 0
    while sent < len(data):
        bytes_sent = send(data[sent:])
        if bytes_sent == 0:
            raise RuntimeError("Docker SDK exec socket write failed")
        sent += bytes_sent


def _shutdown_socket_write(exec_socket) -> None:
    write_socket = _socket_write_target(exec_socket)

    shutdown_write = getattr(write_socket, "shutdown_write", None)
    if shutdown_write is not None:
        shutdown_write()
        return

    shutdown = getattr(write_socket, "shutdown", None)
    if shutdown is not None:
        shutdown(socket_module.SHUT_WR)


def _socket_write_target(exec_socket):
    return getattr(exec_socket, "_sock", exec_socket)


def _read_exec_socket_output(exec_socket):
    from docker.utils.socket import consume_socket_output, demux_adaptor, frames_iter

    demuxed_frames = (
        demux_adaptor(stream, frame)
        for stream, frame in frames_iter(exec_socket, tty=False)
    )
    return consume_socket_output(demuxed_frames, demux=True)


def _decode_exec_output(output) -> tuple[str, str]:
    if isinstance(output, tuple):
        stdout, stderr = output
    else:
        stdout, stderr = output, b""
    return _decode_output_part(stdout), _decode_output_part(stderr)


def _decode_output_part(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def _build_pull_headers(api_client, registry: str) -> dict[str, str]:
    auth_configs = getattr(api_client, "_auth_configs", None)
    if not auth_configs or getattr(auth_configs, "is_empty", True):
        return {}

    from docker import auth

    header = auth.get_config_header(api_client, registry)
    return {"X-Registry-Auth": header} if header else {}


def _raise_pull_event_error(event) -> None:
    if not isinstance(event, dict):
        return

    message = event.get("error")
    error_detail = event.get("errorDetail")
    if not message and isinstance(error_detail, dict):
        message = error_detail.get("message")
    if not message:
        return

    from docker.errors import APIError

    raise APIError("Docker image pull failed", explanation=str(message))


def _is_docker_not_found_error(exc: Exception) -> bool:
    if exc.__class__.__name__ in {"ImageNotFound", "NotFound"}:
        return True
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) == 404


def _build_docker_ssh_base_url(executor_info: ExecutorSSHInfo) -> str:
    return f"ssh://{executor_info.ssh_username}@{executor_info.address}:{executor_info.ssh_port}"


def _build_known_hosts_text(*, host: str, port: int, host_key: str) -> str:
    key = host_key.strip()
    entries = [f"{host} {key}"]
    if port != 22:
        entries.append(f"[{host}]:{port} {key}")
    return "\n".join(entries) + "\n"


def _validate_paramiko_known_hosts(known_hosts_path: Path) -> None:
    try:
        import paramiko

        host_keys = paramiko.HostKeys(str(known_hosts_path))
        if not host_keys:
            raise ValueError("no usable host keys loaded")
    except Exception as exc:
        raise RentalDockerConnectionError(
            _wrap_error_message(
                "Executor SSH host key is not usable by Docker SDK SSH",
                exc,
            )
        ) from exc


def _wrap_error_message(message: str, exc: Exception) -> str:
    detail = str(exc) or exc.__class__.__name__
    return f"{message}: {detail}"
