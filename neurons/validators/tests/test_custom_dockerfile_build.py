"""DAH-2211 — validator-owned subset of §3.A and §3.B tests for
custom-dockerfile pod deployment.

Covers (from the plan + DAH-2211 isolated-build flow):
- A.1 Golden-snapshot regression — image-pull JSON byte-identical
- A.2 Build success path
- A.4 Build failure (bad RUN)
- A.5 Unreachable base image (network ON → generic docker_build)
- A.6 Hard timeout
- A.9 Build runs in a sysbox DinD container WITH network (no --network=none)
- A.11 Empty dockerfile_content guard (validator-level, no SSH issued)
- A.12 sysbox-runc unavailable → build_sysbox_unavailable (never builds)
- A.13 egress firewall failure → build_egress_setup (never builds)
- A.14 DinD container always torn down (finally)
- A.15 image export (save|load) failure → build_export
- A.20–A.26 (DAH-3521) the egress block lets the DinD's own DNS through;
  every setup command before the build is bounded, the firewall helpers
  included (named, bounded on the executor, force-removed before the DinD)
- B.1 SSE latency p95 ≤ 2000 ms (stubbed redis consumer)
"""

from __future__ import annotations

import asyncio
import logging
import re
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
import pytest_asyncio
from datura.requests.miner_requests import ExecutorSSHInfo
from payload_models.payloads import (
    ContainerCreateRequest,
    FailedContainerRequest,
    PayloadPortMapping,
)
from services.docker_service import DockerService
from services.rental_docker_sdk import (
    ContainerExecResult,
    ContainerStateSnapshot,
    build_gpu_docker_config,
)

# ------------------------------------------------------------------
# Shared fixtures / helpers
# ------------------------------------------------------------------


@pytest.fixture
def deps():
    ssh_service = Mock()
    redis_service = Mock()
    attestation_service = Mock()
    lock = AsyncMock()
    lock.__aenter__ = AsyncMock(return_value=lock)
    lock.__aexit__ = AsyncMock(return_value=None)
    redis_service.acquire_executor_lock = Mock(return_value=lock)
    return ssh_service, redis_service, attestation_service


@pytest_asyncio.fixture
async def svc(deps):
    ssh_service, redis_service, attestation_service = deps
    service = DockerService(
        ssh_service=ssh_service,
        redis_service=redis_service,
        attestation_service=attestation_service,
        rental_docker_client_factory=_FakeRentalDockerFactory(),
    )
    # the pre-run kernel.pid_max read is mandatory and the SSH doubles here do not model /proc
    service._read_host_pid_max = AsyncMock(return_value=4_194_304)
    return service


class _FakeRentalDockerClient:
    def __init__(self):
        self.login_calls = []
        self.pulled_images = []
        self.run_specs = []
        self.exec_specs = []
        self.created_volumes = []
        self.removed_volumes = []
        self.pruned_images = 0
        self.pull_error = None
        self.run_error = None

    async def login(self, *, username: str, password: str, image: str) -> None:
        self.login_calls.append({"username": username, "password": password, "image": image})

    async def pull(self, *, image: str) -> None:
        self.pulled_images.append(image)
        if self.pull_error is not None:
            raise self.pull_error

    async def run_container(self, spec) -> None:
        self.run_specs.append(spec)
        if self.run_error is not None:
            raise self.run_error

    async def exec_in_container(self, spec) -> ContainerExecResult:
        self.exec_specs.append(spec)
        return ContainerExecResult(exit_status=0)

    async def inspect_container_state(self, *, container_name: str) -> ContainerStateSnapshot:
        return ContainerStateSnapshot(
            status="running", running=True, restarting=False, exit_code=0, restart_count=0, error=None,
            oom_killed=False,
        )

    async def create_volume(
        self,
        *,
        volume_name: str,
        driver: str | None = None,
        driver_opts: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> None:
        self.created_volumes.append(
            {
                "volume_name": volume_name,
                "driver": driver,
                "driver_opts": driver_opts,
                "timeout": timeout,
            }
        )

    async def remove_volume(self, *, volume_name: str, force: bool = False) -> None:
        self.removed_volumes.append(
            {"volume_name": volume_name, "force": force}
        )

    async def prune_images(self) -> None:
        self.pruned_images += 1


class _FakeRentalDockerFactory:
    def __init__(self):
        self.client = _FakeRentalDockerClient()
        self.connect_calls = []

    def connect(self, *, executor_info: ExecutorSSHInfo, private_key: str):
        self.connect_calls.append(
            {"executor_info": executor_info, "private_key": private_key}
        )
        return self

    async def __aenter__(self):
        return self.client

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


class _ConnCtx:
    def __init__(self, ssh):
        self.ssh = ssh

    async def __aenter__(self):
        return self.ssh

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


def _ssh_result(exit_status: int = 0, stdout: str = "", stderr: str = ""):
    r = Mock()
    r.exit_status = exit_status
    r.stdout = stdout
    r.stderr = stderr
    return r


def _is_probe(command: str) -> bool:
    return "docker exec" in command and command.rstrip().endswith("docker info")


class _FakeProbeProcess:
    """What `ssh_client.create_process` hands the readiness loop: an async
    context manager whose `wait(check, timeout)` does what asyncssh's does.
    It returns the completed process, or raises `TimeoutError` when the wait
    outlives `timeout` and leaves the channel OPEN (asyncssh's `run()` drops
    the process right there). `closed` records the `async with` exit (`close`
    then `wait_closed`), which is what frees the sshd session slot.

    `hangs`: a hung dockerd. With the executor-side `timeout -k 2 N` prefix
    the probe returns exit 124 after N s (the remote `docker exec` is killed);
    without it the wait runs to asyncssh's `timeout=` and raises, and with no
    bound at all it blocks for the hour. `backstop`: a host where even
    `timeout(1)` did not return, so only asyncssh's bound ends the wait; the
    fake raises at once and records the bound it was given (`wait_timeout`),
    the timing is asyncssh's.
    """

    def __init__(self, command: str, *, exit_status: int = 0, hangs: bool = False,
                 backstop: bool = False, on_wait=None, on_close=None):
        self.command = command
        self._exit_status = exit_status
        self._hangs, self._backstop = hangs, backstop
        self._on_wait, self._on_close = on_wait, on_close
        self.wait_timeout = None
        self.closed = False
        self.close_awaited = False

    async def wait(self, check: bool = False, timeout=None):
        self.wait_timeout = timeout
        if self._on_wait:
            self._on_wait(self.command, {"timeout": timeout})
        if self._backstop:
            if timeout is None:
                await asyncio.sleep(3600)
            raise asyncio.TimeoutError()
        if self._hangs:
            bound = re.match(r"timeout -k \d+ (\d+) ", self.command)
            if bound:
                await asyncio.sleep(int(bound.group(1)))
                return _ssh_result(exit_status=124)
            await asyncio.sleep(timeout if timeout else 3600)
            if timeout:
                raise asyncio.TimeoutError()
        return _ssh_result(exit_status=self._exit_status)

    def close(self):
        self.closed = True

    async def wait_closed(self):
        self.close_awaited = True
        if self._on_close:
            self._on_close(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.close()
        await self.wait_closed()
        return False


def _make_dind_ssh(
    *,
    sysbox: bool = True,
    dind_start_exit: int = 0,
    dind_start_hangs: bool = False,
    ready_exit: int = 0,
    ready_hangs: bool = False,
    ready_hangs_first: int = 0,
    ready_backstop_first: int = 0,
    dind_ip: str = "172.20.0.2",
    resolv_conf: str = "nameserver 8.8.8.8\n",
    resolv_exit: int = 0,
    remove_helper_exit: int = 0,
):
    """An ssh client emulating the DAH-2211 DinD build control commands.

    `run` routes by command substring: sysbox preflight, DinD `run -d`, IP
    inspect, the DinD `cat /etc/resolv.conf` read, and everything else
    (Dockerfile write, teardown) → exit 0. `create_process` serves the
    readiness `docker exec ... docker info` probe with a `_FakeProbeProcess`
    (each one on `ssh.probe_processes`; its open/close order on
    `ssh.probe_events`); the probe's command and `wait` bound are recorded on
    `ssh.calls` / `ssh.call_kwargs` with the `run` commands. The build / egress
    / export steps go through `execute_and_stream_logs`, stubbed by
    `_make_esl`. `dind_start_hangs` makes the `run -d` raise
    `asyncio.TimeoutError`, what asyncssh raises when the command outlives its
    `timeout=`; `ready_hangs` makes every readiness probe hang (a hung host
    dockerd); `ready_hangs_first=n` makes only the first n probes hang (a
    dockerd still starting), the rest answer `ready_exit`;
    `ready_backstop_first=n` makes the first n probes outlive even the
    executor-side bound, so asyncssh's own `timeout=` ends them.
    `remove_helper_exit` is the teardown's firewall remove helper's exit status
    (124 = the executor's `timeout(1)` killed it).
    """
    calls: list[str] = []
    call_kwargs: list[dict] = []
    probe_processes: list[_FakeProbeProcess] = []
    probe_events: list[tuple[str, int]] = []

    async def _run(cmd, **kw):
        calls.append(cmd)
        call_kwargs.append(kw)
        if "info --format" in cmd and "Runtimes" in cmd:
            runtimes = '{"runc":{"path":"runc"}'
            if sysbox:
                runtimes += ',"sysbox-runc":{"path":"/usr/bin/sysbox-runc"}'
            runtimes += "}"
            return _ssh_result(stdout=runtimes)
        if "run -d --runtime=sysbox-runc" in cmd:
            if dind_start_hangs:
                raise asyncio.TimeoutError()
            return _ssh_result(exit_status=dind_start_exit, stdout="dind-cid")
        if "docker inspect -f" in cmd:
            return _ssh_result(stdout=dind_ip)
        if "docker exec" in cmd and cmd.rstrip().endswith("cat /etc/resolv.conf"):
            return _ssh_result(exit_status=resolv_exit, stdout=resolv_conf)
        if "--network=host" in cmd and "-D DOCKER-USER" in cmd:  # the firewall remove helper
            return _ssh_result(exit_status=remove_helper_exit)
        return _ssh_result(exit_status=0)

    def _record(cmd, kw):
        calls.append(cmd)
        call_kwargs.append(kw)

    def _probe_process(command: str) -> _FakeProbeProcess:
        n = len(probe_processes) + 1
        proc = _FakeProbeProcess(
            command,
            exit_status=ready_exit,
            hangs=ready_hangs or n <= ready_hangs_first,
            backstop=n <= ready_backstop_first,
            on_wait=_record,
            on_close=lambda p: probe_events.append(("closed", n)),
        )
        probe_processes.append(proc)
        probe_events.append(("open", n))
        return proc

    def _create_process(command: str):
        if _is_probe(command):
            return _probe_process(command)
        raise AssertionError(
            f"unexpected create_process (the streamed steps are stubbed here): {command}"
        )

    ssh = AsyncMock()
    ssh.run = _run
    ssh.create_process = _create_process
    ssh.probe_process = _probe_process
    ssh.probe_processes = probe_processes
    ssh.probe_events = probe_events
    ssh.calls = calls
    ssh.call_kwargs = call_kwargs
    return ssh


def _make_esl(*, egress=(True, ""), build=(True, ""), export=(True, "")):
    """Stub `execute_and_stream_logs`, routing by the step it serves."""
    seen: list[str] = []

    async def _esl(**kwargs):
        cmd = kwargs.get("command", "")
        seen.append(cmd)
        if "--network=host" in cmd:  # egress firewall helper
            return egress
        if "docker save" in cmd:  # export (save | load)
            return export
        if "docker build" in cmd:  # the build itself
            return build
        return (True, "")

    _esl.seen = seen
    return _esl


def _base_payload(*, dockerfile_content: str | None = None, docker_image: str = "daturaai/pytorch:test") -> ContainerCreateRequest:
    return ContainerCreateRequest(
        miner_hotkey="miner",
        executor_id=str(uuid4()),
        pod_id=str(uuid4()),
        docker_image=docker_image,
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
        dockerfile_content=dockerfile_content,
    )


def _executor_info_for(payload: ContainerCreateRequest) -> ExecutorSSHInfo:
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


def _patch_create_container_happy(svc, monkeypatch, ssh_client):
    """Stub everything around the pull/build site so the test only exercises
    the new branch logic."""
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
    monkeypatch.setattr(svc, "_prepare_known_hosts_policy", AsyncMock(return_value=None))
    monkeypatch.setattr(
        svc, "generate_portMappings",
        AsyncMock(return_value=([(22, 20001, 20001)], None)),
    )
    monkeypatch.setattr(svc, "clean_existing_containers", AsyncMock())
    monkeypatch.setattr(svc, "clean_stale_vloopback_volumes", AsyncMock())
    monkeypatch.setattr(svc, "create_local_volume", AsyncMock())
    monkeypatch.setattr(
        svc, "wait_for_port_check_containers",
        AsyncMock(return_value=(True, "ok")),
    )
    monkeypatch.setattr(svc, "check_container_running", AsyncMock(return_value=True))
    monkeypatch.setattr(
        svc,
        "install_open_ssh_server_and_start_ssh_service_with_rental_docker",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(svc, "stream_log", AsyncMock())
    monkeypatch.setattr(svc, "finish_stream_logs", AsyncMock())
    monkeypatch.setattr(svc, "handle_stream_logs", AsyncMock())


# ------------------------------------------------------------------
# A.1 — Golden snapshot regression
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# A.11 — Empty Dockerfile guard (validator-level, no SSH command)
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# A.2 — Build success path
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# A.4 — Build failure (bad RUN) routes through CCF UnknownError
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# A.5 — Unreachable base image. Network is ON now, so a resolve failure is a
#       genuine build error (no dedicated network-blocked step).
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# A.6 — Hard timeout → CCF with failure_step="build_timeout"
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# A.9 — Build runs inside a sysbox DinD container WITH network (no
#       --network=none), then the image is exported to the host.
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# A.12 — sysbox-runc unavailable → build_sysbox_unavailable, NO build attempted
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_A12_sysbox_unavailable_aborts_before_build(svc, monkeypatch):
    ssh_client = _make_dind_ssh(sysbox=False)
    esl = _make_esl()
    monkeypatch.setattr(svc, "execute_and_stream_logs", esl)
    monkeypatch.setattr(svc, "stream_log", AsyncMock())

    payload = _base_payload(dockerfile_content="FROM alpine\nRUN echo hi\n")
    ok, step, _tail = await svc._custom_build_image(
        ssh_client=ssh_client,
        payload=payload,
        log_tag="t",
        default_extra={"pod_id": payload.pod_id},
    )
    assert ok is False
    assert step == "build_sysbox_unavailable"
    # Never started a DinD container, never built (no runc fallback).
    assert not any("run -d --runtime=sysbox-runc" in c for c in ssh_client.calls)
    assert not any("docker build" in c for c in esl.seen)


# ------------------------------------------------------------------
# A.13 — egress firewall failure → build_egress_setup, NO build attempted
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# A.14 — DinD container is always torn down (finally), even on success
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# A.15 — image export (save | load) failure → build_export
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# A.16–A.19 — a failed build says WHY (its last output lines), not only WHERE.
# ------------------------------------------------------------------


def _log_record_texts(caplog) -> list[str]:
    """Everything a log handler could format from each record: the message, the structured extra
    (`_StructuredMessage.to_full_string`) and the exception text — what Loki would receive."""
    texts = []
    for record in caplog.records:
        parts = [record.getMessage()]
        if hasattr(record.msg, "to_full_string"):
            parts.append(record.msg.to_full_string())
        if record.exc_info:
            parts.append(repr(record.exc_info[1]))
        texts.append("\n".join(parts))
    return texts


@pytest.mark.asyncio
async def test_A16d_a_credential_in_the_build_tail_never_reaches_a_log_record(svc, monkeypatch, caplog):
    """BuildKit's `--progress=plain` echoes `ARG HF_TOKEN=…` and `user:TOKEN@host` lines as the renter
    wrote them. The tail goes to the renter (the secret's owner) in `build_log_tail`; the validator's
    own log line (`Custom build failed`, Loki) carries the step and the tail's size, never its text."""
    ssh_client = _make_dind_ssh()
    _patch_create_container_happy(svc, monkeypatch, ssh_client)
    monkeypatch.setattr(svc, "_cleanup_custom_build_artifacts", AsyncMock())
    # the values are resolved at build time (`--build-arg`, an `.env` in the context): they are in
    # the build output, not in the Dockerfile text the payload carries
    leaked = (
        "#3 [2/4] ARG HF_TOKEN=abc\n"
        "#4 [3/4] RUN pip install --extra-index-url https://user:abc@pypi.example.invalid/simple torch\n"
        "#4 1.201 ERROR: Could not find a version that satisfies the requirement torch\n"
        '#4 ERROR: process "/bin/sh -c pip install …" did not complete successfully: exit code: 1\n'
        "BUILD_FAILED_RC=1\n"
    )
    monkeypatch.setattr(svc, "execute_and_stream_logs", _make_esl(build=(False, leaked)))

    payload = _base_payload(
        dockerfile_content=(
            "FROM python:3.11\nARG HF_TOKEN\nARG PYPI_TOKEN\n"
            "RUN pip install --extra-index-url https://user:${PYPI_TOKEN}@pypi.example.invalid/simple torch\n"
        )
    )
    with caplog.at_level(logging.DEBUG):
        result = await svc.create_container(
            payload=payload,
            executor_info=_executor_info_for(payload),
            keypair=Mock(ss58_address="validator-hotkey"),
            private_key="encrypted",
        )

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "docker_build"
    # the renter gets their own output back
    assert "HF_TOKEN=abc" in result.build_log_tail and "user:abc@" in result.build_log_tail
    # no log record anywhere carries a byte of the tail
    texts = _log_record_texts(caplog)
    for text in texts:
        assert "HF_TOKEN=abc" not in text
        assert "user:abc@" not in text
        assert "Could not find a version" not in text
    # the ops log line exists and says how much was sent, not what
    failed_lines = [t for t in texts if t.startswith("Custom build failed")]
    assert len(failed_lines) == 1
    assert '"failure_step": "docker_build"' in failed_lines[0]
    assert '"build_log_tail_lines": 4' in failed_lines[0]
    assert f'"build_log_tail_chars": {len(result.build_log_tail)}' in failed_lines[0]


# ------------------------------------------------------------------
# A.20–A.24 — the egress block must not eat the DinD's own DNS (DAH-2211
# docker_build class: 18 of 23 prod failures were `FROM` failing on a DNS
# timeout because the host resolver sits in a blocked range), and every
# setup command before the build is bounded.
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# B.1 — SSE latency: stub the redis publish path and measure p95
# ------------------------------------------------------------------


