"""DAH-1524 — deploy-time optimizations for cached-template pods.

Covers (from the plan's Test Plan):
- Pull-skip (#1): skip when image present, pull when absent, fail-open on probe
  error, and `check=False` on the inspect probe.
- SSH bootstrap (#2): run the idempotent bootstrap for the default rental path
  (`ships_sshd` None/False), and SKIP it for templates that set `ships_sshd`
  (DAH-2265) — the image ships+starts sshd itself, so the validator only
  re-ensures ~/.ssh for key injection. Includes the DEFAULT-PATH REGRESSION
  GUARD. When `ships_sshd` is set with `enable_jupyter`, the validator forwards
  JUPYTER_PASSWORD as a docker-run env var instead of running run_jupyter — that
  is the only Jupyter variable the image's start.sh reads — and falls back to
  run_jupyter if the mapped docker port is not the 8888 the image hardcodes.
- Profile summary log (#3): emitted once on success, survives the mixed-shape
  profilers list, and NOT emitted on the failure path.
- Backward compat: payloads omitting `ships_sshd` deserialize with None.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo
from payload_models.payloads import (
    ContainerCreated,
    ContainerCreateRequest,
    PayloadPortMapping,
)
from services.docker_service import DockerService
from services.rental_docker_sdk import ContainerExecResult, ContainerStateSnapshot, build_gpu_docker_config

# ------------------------------------------------------------------
# Fixtures / helpers
# ------------------------------------------------------------------


@pytest.fixture
def svc():
    return DockerService(
        ssh_service=Mock(),
        redis_service=Mock(),
        attestation_service=Mock(),
    )


class _ConnCtx:
    def __init__(self, ssh):
        self.ssh = ssh

    async def __aenter__(self):
        return self.ssh

    async def __aexit__(self, *exc):
        return None


def _ssh_result(exit_status: int = 0, stdout: str = "", stderr: str = ""):
    r = Mock()
    r.exit_status = exit_status
    r.stdout = stdout
    r.stderr = stderr
    return r


def _ssh_client(*, inspect_exit: int = 0, inspect_raises: bool = False):
    """AsyncMock ssh connection plus SDK image-probe controls.

    The image presence probe moved from host-side `docker image inspect` to
    Docker SDK data calls; keep the old test inputs so the test intent stays
    readable while `_patch_happy` maps them onto the fake SDK client.
    """
    client = AsyncMock()
    client.image_exists_result = inspect_exit == 0
    client.image_exists_error = RuntimeError("probe boom") if inspect_raises else None

    def _side(cmd, *args, **kwargs):
        if cmd == "cat /proc/sys/kernel/pid_max":
            return _ssh_result(exit_status=0, stdout="4194304\n")
        return _ssh_result(exit_status=0)

    client.run = AsyncMock(side_effect=_side)

    # a bounded command opens its channel with create_process; route it through run so it is recorded
    async def _create_process(command: str):
        return Mock(wait=lambda: client.run(command), wait_closed=AsyncMock())

    client.create_process = _create_process
    return client


class _FakeRentalDockerClient:
    def __init__(self, *, image_exists_result: bool, image_exists_error=None):
        self.image_exists_result = image_exists_result
        self.image_exists_error = image_exists_error
        self.image_exists_calls = []
        self.repo_digests = ()
        self.local_image_current = True
        self.local_image_current_error = None
        self.freshness_calls = []
        self.pulled_images = []
        self.run_specs = []
        self.exec_specs = []
        self.login_calls = []
        self.login_error = None
        self.pull_error = None

    async def login(self, *, username: str, password: str, image: str) -> None:
        self.login_calls.append({"username": username, "password": password, "image": image})
        if self.login_error is not None:
            raise self.login_error

    async def image_exists(self, *, image: str) -> bool:
        self.image_exists_calls.append(image)
        if self.image_exists_error is not None:
            raise self.image_exists_error
        return self.image_exists_result

    async def local_image_repo_digests(self, *, image: str) -> tuple[str, ...] | None:
        return self.repo_digests if await self.image_exists(image=image) else None

    async def local_image_is_current(self, *, image: str, auth_config: dict[str, str] | None = None) -> bool:
        self.freshness_calls.append({"image": image, "auth_config": auth_config})
        if self.local_image_current_error is not None:
            raise self.local_image_current_error
        return self.local_image_current

    async def pull(self, *, image: str) -> None:
        self.pulled_images.append(image)
        if self.pull_error is not None:
            raise self.pull_error

    async def run_container(self, spec) -> None:
        self.run_specs.append(spec)

    async def exec_in_container(self, spec) -> ContainerExecResult:
        self.exec_specs.append(spec)
        return ContainerExecResult(exit_status=0)

    async def inspect_container_state(self, *, container_name: str) -> ContainerStateSnapshot:
        return ContainerStateSnapshot(
            status="running", running=True, restarting=False, exit_code=0, restart_count=0, error=None,
            oom_killed=False,
        )


class _FakeRentalDockerFactory:
    def __init__(self, client):
        self.client = client

    def connect(self, *, executor_info: ExecutorSSHInfo, private_key: str):
        return self

    async def __aenter__(self):
        return self.client

    async def __aexit__(self, *exc):
        return None


def _payload(**over) -> ContainerCreateRequest:
    base = dict(
        miner_hotkey="miner",
        executor_id=str(uuid4()),
        pod_id=str(uuid4()),
        docker_image="daturaai/pytorch:1.0.0",
        user_public_keys=["ssh-ed25519 test-key"],
        gpu_uuids=["GPU-test"],
        cpu_count=1,
        memory_gb=1,
        volume_limit_gb=2,
        storage_limit_gb=1,
        available_ports=[PayloadPortMapping(internal_port=20001, external_port=20001)],
        pod_mapping=[],
        active_container_names=[],
        active_volume_names=[],
    )
    base.update(over)
    return ContainerCreateRequest(**base)


def _executor_info(payload: ContainerCreateRequest) -> ExecutorSSHInfo:
    return ExecutorSSHInfo(
        uuid=payload.executor_id,
        address="127.0.0.1",
        port=8080,
        ssh_username="root",
        ssh_port=2200,
        python_path="/usr/bin/python",
        root_dir="/root/app",
        ssh_host_key="ssh-ed25519 AAAATESTKEY",
    )


def _patch_happy(svc, monkeypatch, ssh_client):
    """Stub the whole deploy flow so only the new branch logic is exercised and
    create_container reaches a real ContainerCreated."""
    docker_client = _FakeRentalDockerClient(
        image_exists_result=ssh_client.image_exists_result,
        image_exists_error=ssh_client.image_exists_error,
    )
    svc.rental_docker_client_factory = _FakeRentalDockerFactory(docker_client)
    monkeypatch.setattr(
        "services.docker_service.asyncssh.connect",
        Mock(return_value=_ConnCtx(ssh_client)),
    )
    monkeypatch.setattr("services.docker_service.asyncssh.import_private_key", Mock())
    monkeypatch.setattr(
        "services.docker_service.build_gpu_docker_config_for_executor",
        AsyncMock(return_value=build_gpu_docker_config(["GPU-test"])),
    )
    svc.ssh_service.decrypt_payload = Mock(return_value="private-key")
    svc.redis_service.add_pending_pod = AsyncMock()
    svc.redis_service.remove_pending_pod = AsyncMock()
    svc.redis_service.add_rented_pod = AsyncMock()
    lock = AsyncMock()
    lock.__aenter__ = AsyncMock(return_value=lock)
    lock.__aexit__ = AsyncMock(return_value=None)
    svc.redis_service.acquire_executor_lock = Mock(return_value=lock)
    monkeypatch.setattr(svc, "_prepare_known_hosts_policy", AsyncMock(return_value=None))
    monkeypatch.setattr(
        svc, "generate_portMappings",
        AsyncMock(return_value=([(22, 20001, 20001)], None)),
    )
    monkeypatch.setattr(svc, "clean_existing_containers", AsyncMock(return_value=[]))
    monkeypatch.setattr(svc, "clean_stale_vloopback_volumes", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        svc, "resolve_volume_sizing",
        AsyncMock(return_value=Mock(volume_limit_gb=10, storage_limit_gb=20)),
    )
    monkeypatch.setattr(svc, "create_local_volume", AsyncMock())
    monkeypatch.setattr(
        svc, "wait_for_port_check_containers",
        AsyncMock(return_value=(True, "ok")),
    )
    monkeypatch.setattr(svc, "_run_rental_docker_create_with_port_retry", AsyncMock())
    monkeypatch.setattr(svc, "check_container_running", AsyncMock(return_value=True))
    monkeypatch.setattr(
        svc,
        "install_open_ssh_server_and_start_ssh_service_with_rental_docker",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(svc, "run_jupyter", AsyncMock())
    monkeypatch.setattr(svc, "execute_and_stream_logs", AsyncMock(return_value=(True, "")))
    monkeypatch.setattr(svc, "stream_log", AsyncMock())
    monkeypatch.setattr(svc, "finish_stream_logs", AsyncMock())
    monkeypatch.setattr(svc, "handle_stream_logs", AsyncMock())


async def _run(svc, payload):
    return await svc.create_container(
        payload=payload,
        executor_info=_executor_info(payload),
        keypair=Mock(ss58_address="validator-hotkey"),
        private_key="encrypted",
    )


def _docker_client(svc):
    return svc.rental_docker_client_factory.client


def _created_run_spec(svc):
    """The ContainerRunSpec handed to _run_rental_docker_create_with_port_retry
    (carries the create-time `docker run` environment)."""
    return svc._run_rental_docker_create_with_port_retry.await_args.kwargs["run_spec"]


# ------------------------------------------------------------------
# #1 — pull-skip
# ------------------------------------------------------------------


def test_a_rental_sent_without_a_memory_limit_is_capped_below_host_ram():
    """A legacy pod row (ram_total 0) arrives as memory_gb 0; the limit becomes host RAM less the 4 GiB host
    reserve, so the renter cannot starve the executor of memory. 64 GiB host - 4 GiB = 60 GiB."""
    assert DockerService._rental_memory_gb(0, 64 * 1024 * 1024) == 60


def test_a_rental_with_a_memory_limit_keeps_the_backend_value():
    assert DockerService._rental_memory_gb(8, 64 * 1024 * 1024) == 8


def test_an_unreadable_host_ram_leaves_a_limitless_rental_as_it_was():
    # host RAM unreadable -> the backend's falsy value is kept, so the rental stays limitless as before
    assert not DockerService._rental_memory_gb(0, None)
    assert DockerService._rental_memory_gb(None, None) is None


def test_a_split_host_legacy_rental_gets_its_gpu_share_of_host_ram():
    """Review finding #1534: on a split host the fallback follows the backend's (host - reserve) * share
    (models.executor.pod_ram_total_kib). A 4-of-8 rental on a 64 GiB host gets (64 - 4) * 0.5 = 30 GiB, so
    two such legacy pods sum to 60 GiB — the whole host less the reserve — and cannot together exceed it
    and OOM the executor, as they would if each were capped near the full host RAM."""
    half = DockerService._rental_memory_gb(0, 64 * 1024 * 1024, 0.5)
    assert half == 30
    assert half * 2 <= DockerService._rental_memory_gb(0, 64 * 1024 * 1024, 1.0)


def test_the_gpu_share_is_clamped_into_range():
    # a share over 1 (never expected) cannot exceed the whole-host cap; a non-positive share floors at 1 GiB
    assert DockerService._rental_memory_gb(0, 64 * 1024 * 1024, 2.0) == 60
    assert DockerService._rental_memory_gb(0, 64 * 1024 * 1024, 0.0) == 1


@pytest.mark.asyncio
async def test_read_host_gpu_count_counts_device_nodes(svc):
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(return_value=_ssh_result(stdout="8\n"))
    assert await svc._read_host_gpu_count(ssh_client) == 8


@pytest.mark.asyncio
async def test_read_host_gpu_count_returns_none_when_zero_or_unreadable(svc):
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(return_value=_ssh_result(stdout="0\n"))
    assert await svc._read_host_gpu_count(ssh_client) is None
    ssh_client.run = AsyncMock(return_value=_ssh_result(exit_status=1))
    assert await svc._read_host_gpu_count(ssh_client) is None


@pytest.mark.asyncio
async def test_read_host_ram_kib_parses_memtotal(svc):
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(return_value=_ssh_result(stdout="MemTotal:       67108864 kB\nMemFree: 1 kB\n"))
    assert await svc._read_host_ram_kib(ssh_client) == 67108864


@pytest.mark.asyncio
async def test_read_host_ram_kib_returns_none_when_the_read_fails(svc):
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(return_value=_ssh_result(exit_status=1))
    assert await svc._read_host_ram_kib(ssh_client) is None


@pytest.mark.asyncio
async def test_rental_is_refused_when_the_host_pid_max_cannot_be_read(svc, monkeypatch):
    """An unreadable kernel.pid_max fails closed, and before any host side-effect.

    Review finding on #1531: the refusal must sit ahead of local-volume creation and the filler
    GPU power cap, or a refusal leaks a volume / leaves GPUs capped with no container. Only the
    pid_max read fails here (every other command succeeds) so the flow reaches the read, and we
    assert create_local_volume was never awaited — it runs before the volume (and the later
    power cap), so neither is reached.
    """
    ssh_client = _ssh_client(inspect_exit=0)
    _patch_happy(svc, monkeypatch, ssh_client)
    monkeypatch.setattr("services.docker_service.settings.RENTAL_PIDS_LIMIT_PER_CPU", 4096)

    def _only_pid_max_fails(cmd, *a, **k):
        if cmd == "cat /proc/sys/kernel/pid_max":
            return _ssh_result(exit_status=1)
        return _ssh_result(exit_status=0)
    ssh_client.run = AsyncMock(side_effect=_only_pid_max_fails)

    result = await _run(svc, _payload())

    assert not isinstance(result, ContainerCreated)
    assert _docker_client(svc).run_specs == []
    svc.create_local_volume.assert_not_awaited()


@pytest.mark.asyncio
async def test_rental_is_refused_when_the_host_pid_max_is_too_low_for_the_margin(svc, monkeypatch):
    """A readable but tiny kernel.pid_max also fails closed before any host side-effect.

    Review finding on #1531: a host whose pid_max cannot reserve RENTAL_PIDS_LIMIT_HOST_MARGIN
    (here 512 with the 4096 default margin) would otherwise hand the container every PID.
    """
    ssh_client = _ssh_client(inspect_exit=0)
    _patch_happy(svc, monkeypatch, ssh_client)
    monkeypatch.setattr("services.docker_service.settings.RENTAL_PIDS_LIMIT_PER_CPU", 4096)

    def _tiny_pid_max(cmd, *a, **k):
        if cmd == "cat /proc/sys/kernel/pid_max":
            return _ssh_result(exit_status=0, stdout="512\n")
        return _ssh_result(exit_status=0)
    ssh_client.run = AsyncMock(side_effect=_tiny_pid_max)

    result = await _run(svc, _payload())

    assert not isinstance(result, ContainerCreated)
    assert _docker_client(svc).run_specs == []
    svc.create_local_volume.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_legacy_rental_is_refused_when_the_host_ram_cannot_be_read(svc, monkeypatch):
    """Review finding #1534: a legacy ram_total 0 pod (memory_gb 0) whose host MemTotal is unreadable
    fails closed BEFORE any host side-effect, like host_pid_max. Falling through to no mem_limit is the
    exact fail-open the limit exists to close, and the host contention that delays this SSH probe is when
    a limitless renter can OOM the executor. Only the meminfo read fails here, so the flow reaches it and
    no volume or container is created."""
    ssh_client = _ssh_client(inspect_exit=0)
    _patch_happy(svc, monkeypatch, ssh_client)

    def _only_meminfo_fails(cmd, *a, **k):
        if cmd == "cat /proc/sys/kernel/pid_max":
            return _ssh_result(exit_status=0, stdout="4194304\n")
        if cmd == "cat /proc/meminfo":
            return _ssh_result(exit_status=1)
        return _ssh_result(exit_status=0)
    ssh_client.run = AsyncMock(side_effect=_only_meminfo_fails)

    result = await _run(svc, _payload(memory_gb=0))

    assert not isinstance(result, ContainerCreated)
    assert _docker_client(svc).run_specs == []
    svc.create_local_volume.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_legacy_rental_probe_failure_leaves_the_filler_in_place(svc, monkeypatch):
    """Review finding #1534 (round 2): the host-limit probes must run BEFORE the stale-container
    cleanup, not after. clean_existing_containers force-removes a co-tenant filler (an irreversible
    `docker rm`); if the probe failed after that, the host would be left with neither the filler nor
    the new rental. Only the meminfo read fails here, so the refusal must happen with
    clean_existing_containers never awaited — the filler is still running."""
    ssh_client = _ssh_client(inspect_exit=0)
    _patch_happy(svc, monkeypatch, ssh_client)

    def _only_meminfo_fails(cmd, *a, **k):
        if cmd == "cat /proc/sys/kernel/pid_max":
            return _ssh_result(exit_status=0, stdout="4194304\n")
        if cmd == "cat /proc/meminfo":
            return _ssh_result(exit_status=1)
        return _ssh_result(exit_status=0)
    ssh_client.run = AsyncMock(side_effect=_only_meminfo_fails)

    result = await _run(svc, _payload(memory_gb=0))

    assert not isinstance(result, ContainerCreated)
    svc.clean_existing_containers.assert_not_awaited()
    svc.create_local_volume.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_split_legacy_rental_is_refused_when_the_host_gpu_count_cannot_be_read(svc, monkeypatch):
    """Review finding #1534: a legacy split rental whose host GPU count is unreadable fails closed too —
    without it the GPU share is unknown, so sizing to the whole host (less reserve) would hand each of two
    co-tenant legacy pods near all of host RAM. Only the GPU-count read fails here."""
    ssh_client = _ssh_client(inspect_exit=0)
    _patch_happy(svc, monkeypatch, ssh_client)

    def _only_gpu_count_fails(cmd, *a, **k):
        if cmd == "cat /proc/sys/kernel/pid_max":
            return _ssh_result(exit_status=0, stdout="4194304\n")
        if cmd == "cat /proc/meminfo":
            return _ssh_result(exit_status=0, stdout="MemTotal:       67108864 kB\n")
        if cmd == "ls -1d /dev/nvidia[0-9]* 2>/dev/null | wc -l":
            return _ssh_result(exit_status=1)
        return _ssh_result(exit_status=0)
    ssh_client.run = AsyncMock(side_effect=_only_gpu_count_fails)

    result = await _run(svc, _payload(memory_gb=0, gpu_uuids=["GPU-a", "GPU-b"]))

    assert not isinstance(result, ContainerCreated)
    svc.create_local_volume.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_split_legacy_rental_is_sized_to_its_gpu_share(svc, monkeypatch):
    """The create flow reads host RAM and host GPU count for a legacy ram_total 0 split rental and sizes
    the mem_limit to the rented GPUs' share: 4 of 8 GPUs on a 64 GiB host -> (64 - 4) * 4/8 = 30 GiB."""
    ssh_client = _ssh_client(inspect_exit=0)
    _patch_happy(svc, monkeypatch, ssh_client)

    def _host_reads(cmd, *a, **k):
        if cmd == "cat /proc/sys/kernel/pid_max":
            return _ssh_result(exit_status=0, stdout="4194304\n")
        if cmd == "cat /proc/meminfo":
            return _ssh_result(exit_status=0, stdout="MemTotal:       67108864 kB\n")
        if cmd == "ls -1d /dev/nvidia[0-9]* 2>/dev/null | wc -l":
            return _ssh_result(exit_status=0, stdout="8\n")
        return _ssh_result(exit_status=0)
    ssh_client.run = AsyncMock(side_effect=_host_reads)

    result = await _run(svc, _payload(memory_gb=0, gpu_uuids=["GPU-a", "GPU-b", "GPU-c", "GPU-d"]))

    assert isinstance(result, ContainerCreated)
    assert _created_run_spec(svc).memory_gb == 30


# ------------------------------------------------------------------
# #2 — SSH bootstrap
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# #3 — profile summary log
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# Backward compatibility
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# DAH-1524 — cleanup step no longer blocks the critical path on a 10s sleep
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# DAH-1524 — ProfilerStep schema: typed `name` enum + wire-shape preservation
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# #4 — docker login only when a pull follows (DAH-3246)
# ------------------------------------------------------------------


