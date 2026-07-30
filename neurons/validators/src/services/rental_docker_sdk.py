from __future__ import annotations

import asyncio
import logging
import socket as socket_module
import tempfile
import threading
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

from core.utils import _m, get_extra_info
from datura.requests.miner_requests import ExecutorSSHInfo


DEFAULT_DOCKER_PULL_TIMEOUT_SECONDS = 3 * 60 * 60
_DOCKER_EXEC_READY_TIMEOUT_SECONDS = 15
_DOCKER_EXEC_READY_POLL_INTERVAL_SECONDS = 0.5
_DOCKER_EXEC_TRANSIENT_RETRY_DELAYS_SECONDS = (1, 2, 4, 8)
_DOCKER_SDK_SSH_ADAPTER_LOCK = threading.Lock()
# DAH-2475: the Docker SDK is synchronous, so every call below has to run in a thread. It must not be
# the event loop's default executor: asyncio resolves DNS there too (loop.getaddrinfo runs in it), and
# a wave of filler creates fills that pool with minutes-long pulls. Name resolution then queues behind
# them and every new Redis/WebSocket connection times out before it ever reaches the socket. A thread
# pool is FIFO, so a bigger default pool does not help — the queue has to be a separate one.
_DOCKER_EXECUTOR_MAX_WORKERS = 32
_DOCKER_EXECUTOR = ThreadPoolExecutor(
    max_workers=_DOCKER_EXECUTOR_MAX_WORKERS, thread_name_prefix="docker-sdk"
)
logger = logging.getLogger(__name__)


async def _in_docker_thread(func: Callable, /, *args, **kwargs):
    # runs a blocking Docker SDK call in the dedicated pool instead of the loop's default executor
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_DOCKER_EXECUTOR, partial(func, *args, **kwargs))


class RentalDockerConnectionError(RuntimeError):
    """Raised when Docker SDK over SSH cannot be constructed safely."""


class RentalDockerOperationError(RuntimeError):
    """Raised when Docker SDK reports a rental Docker operation failure."""


def require_rental_docker_ssh_host_key(executor_info: ExecutorSSHInfo) -> str:
    """Return the host key required by rental Docker SDK SSH connections."""
    # ExecutorSSHInfo keeps this optional for non-connectable placeholder/error
    # records; the rental Docker SDK path requires it to build known_hosts.
    host_key = executor_info.ssh_host_key.strip() if executor_info.ssh_host_key else ""
    if not host_key:
        raise RentalDockerConnectionError(
            "Executor is missing ssh_host_key; rental Docker operations require the executor SSH host public key"
        )
    return host_key


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


@dataclass(slots=True)
class _ContainerExecReadiness:
    ready: bool
    terminal: bool
    detail: str


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
            pull_call = _in_docker_thread(self._pull_sync, image)
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
            await _in_docker_thread(self._api_client.inspect_image, image)
        except Exception as exc:
            if _is_docker_not_found_error(exc):
                return False
            raise RentalDockerOperationError(
                _wrap_error_message("Docker SDK inspect image failed", exc)
            ) from exc
        return True

    async def run_container(self, spec: ContainerRunSpec) -> None:
        try:
            await _in_docker_thread(self._run_container_sync, spec)
        except Exception as exc:
            raise RentalDockerOperationError(
                _wrap_error_message("Docker SDK run container failed", exc)
            ) from exc

    async def exec_in_container(self, spec: ContainerExecSpec) -> ContainerExecResult:
        last_restart_error: Exception | None = None
        for attempt in range(len(_DOCKER_EXEC_TRANSIENT_RETRY_DELAYS_SECONDS) + 1):
            retry_result: ContainerExecResult | None = None
            # A container can pass start/running checks while still moving through
            # transient restart/runtime states where exec is rejected or runc cannot
            # enter its namespaces/cgroups. Inspect first, then keep a narrow retry
            # for the remaining inspect-vs-exec race.
            await self._wait_for_container_exec_ready(spec.container_name)
            try:
                result = await _in_docker_thread(self._exec_in_container_sync, spec)
            except Exception as exc:
                if not _is_docker_container_restarting_error(exc):
                    raise RentalDockerOperationError(
                        _wrap_error_message("Docker SDK exec failed", exc)
                    ) from exc
                last_restart_error = exc
                retry_reason = "container_restarting"
                retry_error = str(exc)
            else:
                if not _is_transient_oci_exec_result(result):
                    return result
                retry_result = result
                retry_reason = "oci_runtime_exec_transient"
                retry_error = _format_exec_result_failure(result)

            delay_seconds = _retry_delay_for_exec_attempt(attempt)
            if delay_seconds is None:
                # Nonzero exec results keep the existing caller behavior after
                # retry budget is exhausted. Exception-based failures have no
                # result, so preserve the original Docker daemon detail below.
                if retry_result is not None:
                    return retry_result
                break

            _log_transient_exec_retry(
                spec=spec,
                retry_reason=retry_reason,
                retry_error=retry_error,
                attempt=attempt + 1,
                delay_seconds=delay_seconds,
            )
            if delay_seconds > 0:
                await asyncio.sleep(delay_seconds)

        assert last_restart_error is not None
        raise RentalDockerOperationError(
            _wrap_error_message("Docker SDK exec failed", last_restart_error)
        ) from last_restart_error

    async def start(self, *, container_name: str) -> None:
        await self._call_api(
            container_name,
            operation_label="start",
            api_method=self._api_client.start,
        )

    async def stop(self, *, container_name: str, stop_grace_seconds: int | None = None) -> None:
        # stop_grace_seconds is the SIGTERM grace window: dockerd waits this many seconds for
        # the container to exit before escalating to SIGKILL (docker stop -t semantics). Named
        # distinctly because `timeout` on this class means the HTTP client timeout (create_volume).
        await self._call_api(
            container_name,
            operation_label="stop",
            api_method=self._api_client.stop,
            timeout=stop_grace_seconds,
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

    async def mount_source_for_destination(
        self,
        *,
        container_name: str,
        destination: str,
    ) -> str | None:
        try:
            return await _in_docker_thread(
                self._mount_source_for_destination_sync,
                container_name,
                destination,
            )
        except Exception as exc:
            raise RentalDockerOperationError(
                _wrap_error_message("Docker SDK inspect container failed", exc)
            ) from exc

    async def create_volume(
        self,
        *,
        volume_name: str,
        driver: str | None = None,
        driver_opts: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> None:
        try:
            await _in_docker_thread(
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
            await _in_docker_thread(close)

    async def _call_api(self, *args, operation_label: str, api_method, **kwargs) -> None:
        try:
            await _in_docker_thread(api_method, *args, **kwargs)
        except Exception as exc:
            raise RentalDockerOperationError(
                _wrap_error_message(f"Docker SDK {operation_label} failed", exc)
            ) from exc

    async def _wait_for_container_exec_ready(
        self,
        container_name: str,
        *,
        timeout_seconds: int | float | None = None,
    ) -> None:
        if timeout_seconds is None:
            timeout_seconds = _DOCKER_EXEC_READY_TIMEOUT_SECONDS

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds
        while True:
            readiness = await _in_docker_thread(
                self._inspect_container_exec_readiness,
                container_name,
            )
            if readiness.ready:
                return
            if readiness.terminal:
                raise RentalDockerOperationError(
                    f"Docker container is not ready for exec: {readiness.detail}"
                )

            remaining_seconds = deadline - loop.time()
            if remaining_seconds <= 0:
                raise RentalDockerOperationError(
                    "Docker container did not become ready for exec "
                    f"after {timeout_seconds} seconds: {readiness.detail}"
                )

            await asyncio.sleep(
                min(_DOCKER_EXEC_READY_POLL_INTERVAL_SECONDS, remaining_seconds)
            )

    def _inspect_container_exec_readiness(
        self,
        container_name: str,
    ) -> _ContainerExecReadiness:
        inspect_container = getattr(self._api_client, "inspect_container", None)
        if inspect_container is None:
            raise RentalDockerOperationError(
                "Docker SDK inspect container is unavailable"
            )

        try:
            inspect_result = inspect_container(container_name)
        except Exception as exc:
            raise RentalDockerOperationError(
                _wrap_error_message("Docker SDK inspect container failed", exc)
            ) from exc

        state = inspect_result.get("State") if isinstance(inspect_result, dict) else None
        if not isinstance(state, dict):
            return _ContainerExecReadiness(
                ready=False,
                terminal=False,
                detail="Docker inspect did not include container State",
            )

        running = bool(state.get("Running"))
        restarting = bool(state.get("Restarting"))
        paused = bool(state.get("Paused"))
        dead = bool(state.get("Dead"))
        ready = running and not restarting and not paused and not dead
        terminal = paused or dead or (
            not running
            and not restarting
            and str(state.get("Status") or "").lower() != "created"
        )
        return _ContainerExecReadiness(
            ready=ready,
            terminal=terminal,
            detail=_format_container_state_detail(state),
        )

    def _mount_source_for_destination_sync(
        self, container_name: str, destination: str
    ) -> str | None:
        info = self._api_client.inspect_container(container_name)
        for mount in info.get("Mounts", []) or ():
            if mount.get("Destination") == destination:
                return mount.get("Name") or mount.get("Source") or None
        return None

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
        # Every spec routed here is rental bootstrap writing to /root or /etc, so
        # it must not inherit a non-root image USER (DAH-2534). The renter's own
        # workload still runs as the image's USER — only these execs are pinned.
        exec_create_result = self._api_client.exec_create(
            container=spec.container_name,
            cmd=list(spec.argv),
            stdin=stdin_data is not None,
            environment=spec.environment or None,
            user="root",
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
        host_key = require_rental_docker_ssh_host_key(executor_info)

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
                    host_key=host_key,
                )
            )
            known_hosts_path.chmod(0o600)
            _validate_paramiko_known_hosts(known_hosts_path)

            try:
                api_client = await _in_docker_thread(
                    self._create_api_client,
                    _build_docker_ssh_base_url(executor_info),
                    key_path,
                    known_hosts_path,
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

    def _create_api_client(
        self,
        base_url: str,
        key_path: Path,
        known_hosts_path: Path,
    ):
        with _DOCKER_SDK_SSH_ADAPTER_LOCK:
            if self.api_client_factory is not None:
                return self.api_client_factory(
                    base_url=base_url,
                    timeout=self.timeout,
                    use_ssh_client=False,
                )

            return _default_docker_api_client_factory(
                base_url=base_url,
                timeout=self.timeout,
                use_ssh_client=False,
                key_path=key_path,
                known_hosts_path=known_hosts_path,
            )


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
    # chmod up front: key injection now runs before the sshd bootstrap's own
    # `chmod 700` (DAH-2341), and sshd may come up in between.
    quoted_dir = shlex.quote(target_path.rsplit("/", 1)[0])
    return ContainerExecSpec(
        container_name=container_name,
        argv=(
            "sh",
            "-c",
            f"mkdir -p {quoted_dir} && chmod 700 {quoted_dir} "
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

    key_path = kwargs.pop("key_path", None)
    known_hosts_path = kwargs.pop("known_hosts_path", None)
    if key_path is not None and known_hosts_path is not None:
        return _create_docker_api_client_with_rental_ssh_adapter(
            docker_module=docker,
            key_path=key_path,
            known_hosts_path=known_hosts_path,
            **kwargs,
        )

    return docker.APIClient(**kwargs)


def _create_docker_api_client_with_rental_ssh_adapter(
    *,
    docker_module,
    key_path: Path,
    known_hosts_path: Path,
    **kwargs,
):
    import docker.api.client as docker_api_client

    # docker-py normally discovers SSH identity/known_hosts via ~/.ssh. Patch
    # its adapter only during construction so this rental client uses our
    # operation-scoped files without mutating process-global HOME.
    original_adapter = docker_api_client.SSHHTTPAdapter
    docker_api_client.SSHHTTPAdapter = _build_rental_ssh_http_adapter_class(
        key_path=key_path,
        known_hosts_path=known_hosts_path,
    )
    try:
        return docker_module.APIClient(**kwargs)
    finally:
        docker_api_client.SSHHTTPAdapter = original_adapter


def _build_rental_ssh_http_adapter_class(
    *,
    key_path: Path,
    known_hosts_path: Path,
):
    from docker.transport.sshconn import SSHHTTPAdapter

    class RentalSSHHTTPAdapter(SSHHTTPAdapter):
        def _create_paramiko_client(self, base_url):
            import logging
            import urllib.parse

            import paramiko

            logging.getLogger("paramiko").setLevel(logging.WARNING)
            parsed_base_url = urllib.parse.urlparse(base_url)
            self.ssh_client = paramiko.SSHClient()
            self.ssh_params = {
                "hostname": parsed_base_url.hostname,
                "port": parsed_base_url.port,
                "username": parsed_base_url.username,
                "key_filename": str(key_path),
                "look_for_keys": False,
                "allow_agent": False,
            }
            self.ssh_client.load_host_keys(str(known_hosts_path))
            self.ssh_client.set_missing_host_key_policy(paramiko.RejectPolicy())

    RentalSSHHTTPAdapter.__name__ = "RentalSSHHTTPAdapter"
    return RentalSSHHTTPAdapter


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


def _container_ports(ports: tuple[PortBinding, ...]) -> list[tuple[int, str]]:
    return [(port.container_port, port.protocol) for port in ports]


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


def _is_docker_container_restarting_error(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    message = str(exc).lower()
    is_restart_conflict = (
        "is restarting" in message
        and "wait until the container is running" in message
    )
    if status_code is not None:
        return status_code == 409 and is_restart_conflict
    return "409" in message and "conflict" in message and is_restart_conflict


def _is_transient_oci_exec_result(result: ContainerExecResult) -> bool:
    if result.exit_status == 0:
        return False

    message = f"{result.stdout}\n{result.stderr}".lower()
    if "oci runtime exec failed" not in message:
        return False

    return any(
        transient_marker in message
        for transient_marker in (
            "executing setns process caused",
            "cgroup.procs",
            "init.scope (deleted)",
        )
    )


def _format_exec_result_failure(result: ContainerExecResult) -> str:
    return (
        f"exit_status={result.exit_status}; "
        f"stderr={result.stderr}; stdout={result.stdout}"
    )


def _format_container_state_detail(state: dict) -> str:
    fields = {
        "status": state.get("Status"),
        "running": state.get("Running"),
        "restarting": state.get("Restarting"),
        "paused": state.get("Paused"),
        "dead": state.get("Dead"),
        "exit_code": state.get("ExitCode"),
        "error": state.get("Error"),
    }
    return " ".join(f"{key}={value!r}" for key, value in fields.items())


def _retry_delay_for_exec_attempt(attempt: int) -> int | float | None:
    if attempt >= len(_DOCKER_EXEC_TRANSIENT_RETRY_DELAYS_SECONDS):
        return None
    return _DOCKER_EXEC_TRANSIENT_RETRY_DELAYS_SECONDS[attempt]


def _log_transient_exec_retry(
    *,
    spec: ContainerExecSpec,
    retry_reason: str,
    retry_error: str,
    attempt: int,
    delay_seconds: int | float,
) -> None:
    logger.warning(
        _m(
            "Docker SDK exec hit transient container state; retrying",
            extra=get_extra_info(
                {
                    "container_name": spec.container_name,
                    "attempt": attempt,
                    "retry_delay_seconds": delay_seconds,
                    "retry_reason": retry_reason,
                    "error": retry_error,
                }
            ),
        )
    )


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
