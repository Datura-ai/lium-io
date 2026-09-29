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
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
import pytest_asyncio
from datura.requests.miner_requests import ExecutorSSHInfo
from payload_models.payloads import (
    ContainerCreateRequest,
    FailedContainerErrorCodes,
    FailedContainerRequest,
    PayloadPortMapping,
)
from services.docker_service import DIND_DNS_RULE_TAG, DockerService
from services.rental_docker_sdk import ContainerExecResult, build_gpu_docker_config

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
    return DockerService(
        ssh_service=ssh_service,
        redis_service=redis_service,
        attestation_service=attestation_service,
        rental_docker_client_factory=_FakeRentalDockerFactory(),
    )


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


_GOLDEN_PATH = Path(__file__).parent / "fixtures" / "container_create_request_pull.json"


def _make_pull_payload_for_golden() -> ContainerCreateRequest:
    """Fixed-shape payload for the golden snapshot. NB: `dockerfile_content` is
    omitted entirely so this proves the image-pull wire stays byte-identical
    when the new optional field is at its default `None` value."""
    return ContainerCreateRequest(
        miner_hotkey="miner-hotkey-A.1",
        executor_id="00000000-0000-0000-0000-000000000001",
        pod_id="00000000-0000-0000-0000-0000000000aa",
        docker_image="daturaai/pytorch:1.0.0",
        user_public_keys=["ssh-ed25519 AAAA test-key user@host"],
        gpu_uuids=["GPU-aaaaaaaaaaaa"],
        cpu_count=2,
        memory_gb=8,
        volume_limit_gb=10,
        storage_limit_gb=20,
        available_ports=[PayloadPortMapping(internal_port=20001, external_port=30001)],
        pod_mapping=[],
        active_container_names=[],
        active_volume_names=[],
    )


def test_A1_golden_snapshot_image_pull_wire(update_snapshot):
    """AC-7: serialized image-pull `ContainerCreateRequest` is byte-identical."""
    payload = _make_pull_payload_for_golden()
    serialized = payload.model_dump_json()

    _GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)

    if update_snapshot or not _GOLDEN_PATH.exists():
        _GOLDEN_PATH.write_text(serialized + "\n", encoding="utf-8")
        if not update_snapshot:
            pytest.fail(
                f"Golden snapshot did not exist; wrote it. Re-run to assert. Path={_GOLDEN_PATH}"
            )
        return

    expected = _GOLDEN_PATH.read_text(encoding="utf-8").rstrip("\n")
    assert serialized == expected, (
        "Image-pull ContainerCreateRequest JSON drifted! AC-7 violation.\n"
        f"expected={expected!r}\nactual={serialized!r}"
    )


def test_A1_dockerfile_field_omitted_when_none():
    """Defense-in-depth: `dockerfile_content` must not appear in the wire when
    None and a model_dump excludes None — but pydantic default model_dump
    INCLUDES the field as `null`. Either is fine for byte-identity provided
    the field is present in both pre- and post-change snapshots. The actual
    AC-7 guard is the golden file above; this test pins the convention."""
    payload = _make_pull_payload_for_golden()
    data = json.loads(payload.model_dump_json())
    # The field must exist with the documented default. The wire shape is
    # `str | None` — null when not set, which is what backend serializers can
    # safely drop or include.
    assert data["dockerfile_content"] is None
    # DAH-1524: the new optional flag follows the same convention (null by default).
    assert data["ships_sshd"] is None
    assert data["enable_volume_encryption"] is None


# ------------------------------------------------------------------
# A.11 — Empty Dockerfile guard (validator-level, no SSH command)
# ------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["", "   \n  ", "\t \n\n  "])
async def test_A11_empty_dockerfile_content_emits_ccf_without_ssh(svc, monkeypatch, content):
    """`dockerfile_content` empty / whitespace-only → CCF before any SSH connect."""
    # If create_container reaches asyncssh.connect, fail loudly.
    def _fail(*a, **kw):
        raise AssertionError("asyncssh.connect must NOT be called for empty dockerfile_content")

    monkeypatch.setattr("services.docker_service.asyncssh.connect", _fail)
    svc.redis_service.remove_pending_pod = AsyncMock()

    payload = _base_payload(dockerfile_content=content)
    result = await svc.create_container(
        payload=payload,
        executor_info=_executor_info_for(payload),
        keypair=Mock(ss58_address="validator-hotkey"),
        private_key="encrypted",
    )

    assert isinstance(result, FailedContainerRequest)
    assert result.error_code == FailedContainerErrorCodes.UnknownError
    assert result.failure_step == "build_input_empty"


# ------------------------------------------------------------------
# A.2 — Build success path
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_A2_build_success_overrides_docker_image_tag(svc, monkeypatch):
    """`dockerfile_content` present → build runs, image tag becomes `lium-build-{pod_id}`."""
    ssh_client = AsyncMock()
    # df reports plenty of free space (KiB)
    ssh_client.run = AsyncMock(return_value=_ssh_result(stdout=str(1024 * 1024 * 100)))
    _patch_create_container_happy(svc, monkeypatch, ssh_client)

    # Stub the build helper to succeed and assert it's called.
    build_mock = AsyncMock(return_value=(True, None, None))
    monkeypatch.setattr(svc, "_custom_build_image", build_mock)
    # Mock execute_and_stream_logs so the test does not try to run the real pull either.
    monkeypatch.setattr(svc, "execute_and_stream_logs", AsyncMock(return_value=(True, "")))

    payload = _base_payload(
        dockerfile_content="FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04\n",
        docker_image="ignored-when-building",
    )
    await svc.create_container(
        payload=payload,
        executor_info=_executor_info_for(payload),
        keypair=Mock(ss58_address="validator-hotkey"),
        private_key="encrypted",
    )

    build_mock.assert_awaited_once()
    # docker_image was replaced with lium-build-{pod_id}
    assert payload.docker_image == f"lium-build-{payload.pod_id}"

    run_spec = svc.rental_docker_client_factory.client.run_specs[-1]
    assert run_spec.image == f"lium-build-{payload.pod_id}"


@pytest.mark.asyncio
async def test_A2_image_pull_path_unchanged_when_dockerfile_none(svc, monkeypatch):
    """Default branch still pulls the requested image and skips custom build."""
    ssh_client = AsyncMock()

    # DAH-1524: the pull is now guarded by a `docker image inspect` probe. Make
    # the probe report the image as ABSENT (exit !=0) so the pull still runs,
    # which is what this test asserts. All other ssh commands succeed (exit 0).
    def _ssh_run_side(cmd, *args, **kwargs):
        if "image inspect" in cmd:
            return _ssh_result(exit_status=1)
        return _ssh_result()

    ssh_client.run = AsyncMock(side_effect=_ssh_run_side)
    _patch_create_container_happy(svc, monkeypatch, ssh_client)

    build_mock = AsyncMock(return_value=(True, None, None))
    monkeypatch.setattr(svc, "_custom_build_image", build_mock)

    monkeypatch.setattr(svc, "execute_and_stream_logs", AsyncMock(return_value=(True, "")))

    payload = _base_payload(dockerfile_content=None, docker_image="daturaai/pytorch:1.2.3")
    await svc.create_container(
        payload=payload,
        executor_info=_executor_info_for(payload),
        keypair=Mock(ss58_address="validator-hotkey"),
        private_key="encrypted",
    )

    build_mock.assert_not_awaited()
    assert svc.rental_docker_client_factory.client.pulled_images == [
        "daturaai/pytorch:1.2.3"
    ]


# ------------------------------------------------------------------
# A.4 — Build failure (bad RUN) routes through CCF UnknownError
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_A4_build_failure_returns_ccf_unknown_error(svc, monkeypatch):
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(return_value=_ssh_result())
    _patch_create_container_happy(svc, monkeypatch, ssh_client)

    monkeypatch.setattr(svc, "_custom_build_image", AsyncMock(return_value=(False, "docker_build", "ERROR: failed to solve: process \"/bin/sh -c false\" did not complete successfully: exit code: 1")))
    monkeypatch.setattr(svc, "_cleanup_custom_build_artifacts", AsyncMock())
    monkeypatch.setattr(svc, "execute_and_stream_logs", AsyncMock(return_value=(True, "")))

    payload = _base_payload(dockerfile_content="FROM scratch\nRUN false\n")
    result = await svc.create_container(
        payload=payload,
        executor_info=_executor_info_for(payload),
        keypair=Mock(ss58_address="validator-hotkey"),
        private_key="encrypted",
    )

    assert isinstance(result, FailedContainerRequest)
    assert result.error_code == FailedContainerErrorCodes.UnknownError
    assert result.failure_step == "docker_build"
    svc._cleanup_custom_build_artifacts.assert_awaited()


# ------------------------------------------------------------------
# A.5 — Unreachable base image. Network is ON now, so a resolve failure is a
#       genuine build error (no dedicated network-blocked step).
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_A5_unreachable_base_image_ccf_classification(svc, monkeypatch):
    ssh_client = _make_dind_ssh()
    _patch_create_container_happy(svc, monkeypatch, ssh_client)
    monkeypatch.setattr(svc, "_cleanup_custom_build_artifacts", AsyncMock())

    # Egress applies fine; the build step fails resolving an unreachable base.
    monkeypatch.setattr(
        svc, "execute_and_stream_logs",
        _make_esl(build=(False, "ERROR: failed to solve: could not resolve host gcr.io")),
    )

    payload = _base_payload(dockerfile_content="FROM gcr.io/this-does-not-exist:latest\n")
    result = await svc.create_container(
        payload=payload,
        executor_info=_executor_info_for(payload),
        keypair=Mock(ss58_address="validator-hotkey"),
        private_key="encrypted",
    )

    assert isinstance(result, FailedContainerRequest)
    assert result.error_code == FailedContainerErrorCodes.UnknownError
    # With network enabled there is no special network-blocked step anymore.
    assert result.failure_step == "docker_build"


# ------------------------------------------------------------------
# A.6 — Hard timeout → CCF with failure_step="build_timeout"
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_A6_build_hard_timeout(svc, monkeypatch):
    ssh_client = _make_dind_ssh()
    _patch_create_container_happy(svc, monkeypatch, ssh_client)

    # Egress applies fine; the build step times out.
    monkeypatch.setattr(
        svc, "execute_and_stream_logs",
        _make_esl(build=(False, "Process timed out")),
    )
    monkeypatch.setattr(svc, "_cleanup_custom_build_artifacts", AsyncMock())

    payload = _base_payload(dockerfile_content="FROM scratch\nRUN sleep 99999\n")
    result = await svc.create_container(
        payload=payload,
        executor_info=_executor_info_for(payload),
        keypair=Mock(ss58_address="validator-hotkey"),
        private_key="encrypted",
    )

    assert isinstance(result, FailedContainerRequest)
    assert result.error_code == FailedContainerErrorCodes.UnknownError
    assert result.failure_step == "build_timeout"


# ------------------------------------------------------------------
# A.9 — Build runs inside a sysbox DinD container WITH network (no
#       --network=none), then the image is exported to the host.
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_A9_build_runs_in_sysbox_dind_with_network(svc, monkeypatch):
    """`_custom_build_image` launches a sysbox DinD container, builds inside it
    WITHOUT `--network=none`, firewalls egress, and exports the image."""
    ssh_client = _make_dind_ssh()
    esl = _make_esl()
    monkeypatch.setattr(svc, "execute_and_stream_logs", esl)
    monkeypatch.setattr(svc, "stream_log", AsyncMock())

    payload = _base_payload(dockerfile_content="FROM alpine\nRUN curl -m 2 http://example.com\n")
    ok, step, _tail = await svc._custom_build_image(
        ssh_client=ssh_client,
        payload=payload,
        log_tag="t",
        default_extra={"pod_id": payload.pod_id},
    )
    assert ok is True and step is None

    all_run = "\n".join(ssh_client.calls)
    all_esl = "\n".join(esl.seen)
    # DinD launched under sysbox-runc with the per-pod name.
    assert "run -d --runtime=sysbox-runc" in all_run
    assert f"lium-dind-build-{payload.pod_id}" in all_run
    # Build happens inside the DinD container and NEVER with --network=none.
    assert any("docker build" in c for c in esl.seen)
    assert "--network=none" not in all_esl
    # Egress firewall applied + image exported (save | load) onto the host.
    assert any("--network=host" in c and "DOCKER-USER" in c for c in esl.seen)
    assert any("docker save" in c and "docker load" in c for c in esl.seen)
    assert any(f"lium-build-{payload.pod_id}" in c for c in esl.seen)


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


@pytest.mark.asyncio
async def test_A13_egress_failure_aborts_before_build(svc, monkeypatch):
    ssh_client = _make_dind_ssh()
    esl = _make_esl(egress=(False, "DOCKER-USER chain not found"))
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
    assert step == "build_egress_setup"
    # The build must NOT run when egress filtering can't be guaranteed.
    assert not any("docker build" in c for c in esl.seen)
    # But the DinD container is still torn down.
    assert any(f"docker rm -fv" in c and f"lium-dind-build-{payload.pod_id}" in c
               for c in ssh_client.calls)
    # And so are the egress rules: an apply that timed out may have inserted
    # some of them before the deadline. The remove script is idempotent, so it runs whenever the
    # DinD IP is known, not only after a successful apply.
    remove_runs = [c for c in ssh_client.calls if "--network=host" in c and "-D DOCKER-USER" in c]
    assert len(remove_runs) == 1 and "172.20.0.2" in remove_runs[0]


# ------------------------------------------------------------------
# A.14 — DinD container is always torn down (finally), even on success
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_A14_dind_container_always_torn_down(svc, monkeypatch):
    ssh_client = _make_dind_ssh()
    monkeypatch.setattr(svc, "execute_and_stream_logs", _make_esl())
    monkeypatch.setattr(svc, "stream_log", AsyncMock())

    payload = _base_payload(dockerfile_content="FROM alpine\nRUN echo hi\n")
    ok, step, _tail = await svc._custom_build_image(
        ssh_client=ssh_client,
        payload=payload,
        log_tag="t",
        default_extra={"pod_id": payload.pod_id},
    )
    assert ok is True and step is None
    assert any(
        "docker rm -fv" in c and f"lium-dind-build-{payload.pod_id}" in c
        for c in ssh_client.calls
    ), ssh_client.calls


# ------------------------------------------------------------------
# A.15 — image export (save | load) failure → build_export
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_A15_export_failure_classifies_build_export(svc, monkeypatch):
    ssh_client = _make_dind_ssh()
    esl = _make_esl(export=(False, "Error: no space left on device"))
    monkeypatch.setattr(svc, "execute_and_stream_logs", esl)
    monkeypatch.setattr(svc, "stream_log", AsyncMock())

    payload = _base_payload(dockerfile_content="FROM alpine\nRUN echo hi\n")
    ok, step, tail = await svc._custom_build_image(
        ssh_client=ssh_client,
        payload=payload,
        log_tag="t",
        default_extra={"pod_id": payload.pod_id},
    )
    assert ok is False
    assert step == "build_export"
    # the export's stderr is the HOST daemon's `docker load`: never the renter's tail
    assert tail == "built image could not be loaded onto the executor"
    assert "no space left" not in tail


# ------------------------------------------------------------------
# A.16–A.19 — a failed build says WHY (its last output lines), not only WHERE.
# ------------------------------------------------------------------

_BUILD_ERROR_OUTPUT = (
    "#5 [2/2] RUN apt-get install -y nonexistent-package\n"
    "#5 0.412 E: Unable to locate package nonexistent-package\n"
    '#5 ERROR: process "/bin/sh -c apt-get install -y nonexistent-package" did not complete successfully: exit code: 100\n'
    "BUILD_FAILED_RC=1\n"
)


@pytest.mark.asyncio
async def test_A16_build_failure_carries_the_build_output_tail(svc, monkeypatch):
    """The CCF for a failed `docker build` carries the last lines the build printed on the
    wire (`build_log_tail`) — not only the step name. `detail` (ops; the backend classifies it
    for GPU quarantine) keeps the step and never the renter's output."""
    ssh_client = _make_dind_ssh()
    _patch_create_container_happy(svc, monkeypatch, ssh_client)
    monkeypatch.setattr(svc, "_cleanup_custom_build_artifacts", AsyncMock())
    monkeypatch.setattr(
        svc, "execute_and_stream_logs", _make_esl(build=(False, _BUILD_ERROR_OUTPUT))
    )

    payload = _base_payload(
        dockerfile_content="FROM ubuntu\nRUN apt-get install -y nonexistent-package\n"
    )
    result = await svc.create_container(
        payload=payload,
        executor_info=_executor_info_for(payload),
        keypair=Mock(ss58_address="validator-hotkey"),
        private_key="encrypted",
    )

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "docker_build"
    assert result.build_log_tail is not None
    assert "Unable to locate package nonexistent-package" in result.build_log_tail
    assert "exit code: 100" in result.build_log_tail
    # the marker is the streamer's signal, not a build line
    assert "BUILD_FAILED_RC" not in result.build_log_tail
    # ops path (detail -> backend logs / filler_run.failure_reason) names the step, not the output:
    # the backend runs classify_gpu_runtime_error over `detail or msg`
    assert "(failure_step=docker_build)" in result.detail
    assert "Unable to locate package nonexistent-package" not in result.detail
    # the headline stays the renter-safe constant the backend trims to
    assert result.msg.startswith("Failed create_container")
    # the wire field is the renter's build output only — never the executor host
    assert "127.0.0.1" not in result.build_log_tail and "2200" not in result.build_log_tail


@pytest.mark.asyncio
async def test_A16c_renter_output_never_reaches_detail(svc, monkeypatch):
    """A Dockerfile can print anything. The backend runs its GPU-quarantine classifier over
    `detail or msg` (lium-platform validator_consumer.quarantine_on_gpu_runtime_error), so a
    `RUN echo` of an NVML fault signature must stay in `build_log_tail`, never in `detail`."""
    ssh_client = _make_dind_ssh()
    _patch_create_container_happy(svc, monkeypatch, ssh_client)
    monkeypatch.setattr(svc, "_cleanup_custom_build_artifacts", AsyncMock())
    forged = (
        "#4 [1/1] RUN echo nvidia-container-cli: device error: GPU-0: unknown device\n"
        "#4 0.101 nvidia-container-cli: device error: GPU-0: unknown device\n"
        '#4 ERROR: process "/bin/sh -c false" did not complete successfully: exit code: 1\n'
        "BUILD_FAILED_RC=1\n"
    )
    monkeypatch.setattr(svc, "execute_and_stream_logs", _make_esl(build=(False, forged)))

    payload = _base_payload(
        dockerfile_content="FROM alpine\nRUN echo nvidia-container-cli: device error: GPU-0: unknown device; false\n"
    )
    result = await svc.create_container(
        payload=payload,
        executor_info=_executor_info_for(payload),
        keypair=Mock(ss58_address="validator-hotkey"),
        private_key="encrypted",
    )

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "docker_build"
    assert "nvidia-container-cli" in result.build_log_tail
    assert "nvidia-container-cli" not in (result.detail or "")
    assert "nvidia-container-cli" not in result.msg


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


@pytest.mark.asyncio
async def test_A16b_non_build_failures_have_no_tail(svc, monkeypatch):
    """`build_log_tail` is None for a template pod whose creation fails: the field belongs to
    custom builds only, so the backend can trust it as renter-safe build output."""
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(return_value=_ssh_result())
    _patch_create_container_happy(svc, monkeypatch, ssh_client)
    monkeypatch.setattr(svc, "execute_and_stream_logs", AsyncMock(return_value=(True, "")))
    monkeypatch.setattr(
        svc, "create_local_volume", AsyncMock(side_effect=RuntimeError("no space left on device"))
    )

    payload = _base_payload(dockerfile_content=None)
    result = await svc.create_container(
        payload=payload,
        executor_info=_executor_info_for(payload),
        keypair=Mock(ss58_address="validator-hotkey"),
        private_key="encrypted",
    )

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "volume_creation"
    assert result.build_log_tail is None


@pytest.mark.asyncio
async def test_A20_volume_step_failure_carries_the_daemon_reason(svc, monkeypatch):
    """A `volume_creation` failure carries the Docker daemon's reason in `step_detail`, with the
    dead-session hint in front of docker-py's dead-transport text."""
    from services.docker_service import DEAD_DOCKER_SSH_SESSION_HINT
    from services.rental_docker_sdk import RentalDockerOperationError

    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(return_value=_ssh_result())
    _patch_create_container_happy(svc, monkeypatch, ssh_client)
    monkeypatch.setattr(svc, "execute_and_stream_logs", AsyncMock(return_value=(True, "")))
    monkeypatch.setattr(
        svc,
        "create_local_volume",
        AsyncMock(
            side_effect=RentalDockerOperationError(
                "Docker SDK create volume failed: 'NoneType' object has no attribute 'settimeout'"
            )
        ),
    )

    payload = _base_payload(dockerfile_content=None)
    result = await svc.create_container(
        payload=payload,
        executor_info=_executor_info_for(payload),
        keypair=Mock(ss58_address="validator-hotkey"),
        private_key="encrypted",
    )

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "volume_creation"
    assert result.step_detail is not None
    assert result.step_detail.startswith(DEAD_DOCKER_SSH_SESSION_HINT)
    assert "Docker SDK create volume failed" in result.step_detail
    assert "127.0.0.1" not in result.step_detail and "2200" not in result.step_detail
    assert result.build_log_tail is None


@pytest.mark.asyncio
async def test_A20c_volume_sizing_failure_carries_the_min_size_reason(svc, monkeypatch):
    """`volume_sizing` is the other step a pod dies at before its volume exists; its
    VolumeMinSizeError text is the renter's answer and travels as `step_detail` unchanged."""
    from services.docker_service import VolumeMinSizeError

    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(return_value=_ssh_result())
    _patch_create_container_happy(svc, monkeypatch, ssh_client)
    monkeypatch.setattr(svc, "execute_and_stream_logs", AsyncMock(return_value=(True, "")))
    monkeypatch.setattr(
        svc,
        "resolve_volume_sizing",
        AsyncMock(
            side_effect=VolumeMinSizeError(
                "Fresh vloopback sizing produced 3GB volume, below required minimum 20GB"
            )
        ),
    )

    payload = _base_payload(dockerfile_content=None)
    result = await svc.create_container(
        payload=payload,
        executor_info=_executor_info_for(payload),
        keypair=Mock(ss58_address="validator-hotkey"),
        private_key="encrypted",
    )

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "volume_sizing"
    assert result.step_detail == "Fresh vloopback sizing produced 3GB volume, below required minimum 20GB"


@pytest.mark.asyncio
async def test_A20d_template_switch_dead_transport_at_docker_run_carries_the_hint_alone(svc, monkeypatch):
    """A template switch keeps the pod's volume, so the dead session surfaces at `docker_run`:
    `step_detail` is the fixed hint alone, never that step's raw text."""
    from services.docker_service import DEAD_DOCKER_SSH_SESSION_HINT
    from services.rental_docker_sdk import RentalDockerOperationError

    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(return_value=_ssh_result())
    _patch_create_container_happy(svc, monkeypatch, ssh_client)
    monkeypatch.setattr(svc, "execute_and_stream_logs", AsyncMock(return_value=(True, "")))
    monkeypatch.setattr(svc, "_custom_build_image", AsyncMock(return_value=(True, None, None)))
    create_volume = AsyncMock()
    monkeypatch.setattr(svc, "create_local_volume", create_volume)
    monkeypatch.setattr(
        svc,
        "_run_rental_docker_create_with_port_retry",
        AsyncMock(
            side_effect=RentalDockerOperationError(
                "Docker SDK create container failed: 'NoneType' object has no attribute 'settimeout' "
                "(executor 127.0.0.1:2200)"
            )
        ),
    )

    payload = _base_payload(dockerfile_content="FROM alpine\n")
    payload = payload.model_copy(update={"local_volume": f"volume_{payload.pod_id}"})
    result = await svc.create_container(
        payload=payload,
        executor_info=_executor_info_for(payload),
        keypair=Mock(ss58_address="validator-hotkey"),
        private_key="encrypted",
    )

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "docker_run"
    create_volume.assert_not_awaited()
    assert result.step_detail == DEAD_DOCKER_SSH_SESSION_HINT
    assert "settimeout" not in result.step_detail
    assert "127.0.0.1" not in result.step_detail and "2200" not in result.step_detail
    assert result.build_log_tail is None


@pytest.mark.asyncio
async def test_A20b_non_volume_failures_have_no_step_detail(svc, monkeypatch):
    """`step_detail` belongs to the volume step, plus the dead-transport hint: a failure elsewhere
    with any other text (here the custom build) leaves it None so the backend can trust it."""
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(return_value=_ssh_result())
    _patch_create_container_happy(svc, monkeypatch, ssh_client)
    monkeypatch.setattr(svc, "_custom_build_image", AsyncMock(return_value=(False, "docker_build", "boom")))

    payload = _base_payload(dockerfile_content="FROM alpine\nRUN false\n")
    result = await svc.create_container(
        payload=payload,
        executor_info=_executor_info_for(payload),
        keypair=Mock(ss58_address="validator-hotkey"),
        private_key="encrypted",
    )

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "docker_build"
    assert result.step_detail is None


def test_A21_volume_step_detail_is_bounded_and_plain():
    from services.docker_service import (
        DEAD_DOCKER_SSH_SESSION_HINT,
        VOLUME_STEP_DETAIL_MAX_CHARS,
        volume_step_detail,
    )

    assert volume_step_detail(RuntimeError("")) is None
    assert volume_step_detail(RuntimeError("  \n ")) is None
    plain = volume_step_detail(RuntimeError("Docker SDK create volume failed:\n  no space left on device"))
    assert plain == "Docker SDK create volume failed: no space left on device"
    assert not plain.startswith(DEAD_DOCKER_SSH_SESSION_HINT)
    long = volume_step_detail(RuntimeError("x" * 1000))
    assert len(long) == VOLUME_STEP_DETAIL_MAX_CHARS


def test_A21b_failure_step_detail_routes_by_step():
    from services.docker_service import DEAD_DOCKER_SSH_SESSION_HINT, CustomBuildFailed, failure_step_detail

    stale = RuntimeError("Docker SDK create container failed: SSH session not active (host 10.0.0.9)")
    # volume steps: the raw daemon text, hinted
    assert failure_step_detail(stale, "volume_creation").startswith(DEAD_DOCKER_SSH_SESSION_HINT)
    assert "SSH session not active" in failure_step_detail(stale, "volume_creation")
    assert failure_step_detail(RuntimeError("no space left on device"), "volume_sizing") == "no space left on device"
    # any other step: the hint alone for a dead transport, nothing of the raw text
    assert failure_step_detail(stale, "docker_run") == DEAD_DOCKER_SSH_SESSION_HINT
    assert failure_step_detail(stale, "container_health_check") == DEAD_DOCKER_SSH_SESSION_HINT
    assert failure_step_detail(stale, None) == DEAD_DOCKER_SSH_SESSION_HINT
    # any other step with any other text: None
    assert failure_step_detail(RuntimeError("image not found"), "docker_run") is None
    assert failure_step_detail(RuntimeError(""), "docker_run") is None
    # a failed custom build: its text is the renter's build output, which may say anything
    assert failure_step_detail(CustomBuildFailed("docker_build", "Socket is closed"), "docker_build") is None


@pytest.mark.asyncio
async def test_A17_build_command_puts_the_output_tail_on_stderr(svc, monkeypatch):
    """The build runs through `execute_and_stream_logs`, which returns stderr only. So the
    command run inside DinD must tee the build output to a file outside the build context and,
    on a non-zero exit, print its tail to stderr before the BUILD_FAILED_RC marker."""
    ssh_client = _make_dind_ssh()
    esl = _make_esl(build=(False, _BUILD_ERROR_OUTPUT))
    monkeypatch.setattr(svc, "execute_and_stream_logs", esl)
    monkeypatch.setattr(svc, "stream_log", AsyncMock())

    payload = _base_payload(dockerfile_content="FROM alpine\nRUN false\n")
    ok, step, tail = await svc._custom_build_image(
        ssh_client=ssh_client,
        payload=payload,
        log_tag="t",
        default_extra={"pod_id": payload.pod_id},
    )
    assert (ok, step) == (False, "docker_build")
    assert tail is not None and "exit code: 100" in tail

    build_cmd = next(c for c in esl.seen if "docker build" in c)
    assert "tee /tmp/lium-build.log" in build_cmd
    # blank lines are dropped before the cap, so all 25 slots carry build output
    assert 'grep -v "^[[:space:]]*$" /tmp/lium-build.log | tail -n 25 >&2' in build_cmd
    assert "echo BUILD_FAILED_RC=$rc >&2" in build_cmd
    # the tail goes out before the marker, and the exit code is preserved
    assert build_cmd.index("tail -n 25") < build_cmd.index("BUILD_FAILED_RC")
    assert build_cmd.rstrip("'").endswith("exit $rc")
    # the log never lands inside the build context (a `COPY .` must not pick it up)
    assert "/build/" not in build_cmd.split("| tee", 1)[1]


def test_A17b_rendered_build_command_runs_under_sh(tmp_path, monkeypatch):
    """The inner command really does what A17 asserts by substring: run it under `sh` with a
    `docker` stub that prints 30 lines (3 of them blank) and exits 7 — stderr is the last 25
    NON-blank lines then `BUILD_FAILED_RC=7`, the exit code is 7 without pipefail."""
    import subprocess

    from services import docker_service
    from services.docker_service import custom_build_inner_command

    bindir = tmp_path / "bin"
    bindir.mkdir()
    stub = bindir / "docker"
    stub.write_text(
        "#!/bin/sh\n"
        'i=1; while [ $i -le 30 ]; do echo "line $i"; '
        "if [ $((i % 10)) -eq 0 ]; then echo; fi; i=$((i + 1)); done\n"
        "exit 7\n"
    )
    stub.chmod(0o755)
    monkeypatch.setattr(docker_service, "CUSTOM_BUILD_LOG_FILE", str(tmp_path / "lium-build.log"))
    monkeypatch.setattr(docker_service, "CUSTOM_BUILD_RC_FILE", str(tmp_path / "lium-build.rc"))
    inner = custom_build_inner_command("lium-custom-test:latest", "/build")
    run = subprocess.run(
        ["sh", "-c", inner],
        capture_output=True,
        text=True,
        env={"PATH": f"{bindir}:/usr/bin:/bin"},
    )
    assert run.returncode == 7
    stderr_lines = run.stderr.splitlines()
    assert stderr_lines[-1] == "BUILD_FAILED_RC=7"
    assert stderr_lines[:-1] == [f"line {i}" for i in range(6, 31)]
    assert "" not in stderr_lines


def test_A17c_an_unwritable_rc_file_is_a_failure_with_a_reason(tmp_path, monkeypatch):
    """The exit code round-trips through `CUSTOM_BUILD_RC_FILE` in the DinD container's /tmp. When
    that file cannot be written (the overlay is full or read-only) the build is not reported as a
    success by a bare `exit`: rc is 1, the tail is printed and ends with the fixed reason, then
    the marker."""
    import subprocess

    from services import docker_service
    from services.docker_service import (
        CUSTOM_BUILD_RC_UNREADABLE_REASON,
        custom_build_inner_command,
        custom_build_log_tail,
    )

    bindir = tmp_path / "bin"
    bindir.mkdir()
    stub = bindir / "docker"
    stub.write_text("#!/bin/sh\necho step one\necho step two\nexit 0\n")
    stub.chmod(0o755)
    monkeypatch.setattr(docker_service, "CUSTOM_BUILD_LOG_FILE", str(tmp_path / "lium-build.log"))
    monkeypatch.setattr(docker_service, "CUSTOM_BUILD_RC_FILE", str(tmp_path / "no-such-dir" / "lium-build.rc"))
    inner = custom_build_inner_command("lium-custom-test:latest", "/build")
    run = subprocess.run(
        ["sh", "-c", inner],
        capture_output=True,
        text=True,
        env={"PATH": f"{bindir}:/usr/bin:/bin"},
    )
    assert run.returncode == 1
    stderr_lines = run.stderr.splitlines()
    assert stderr_lines[-1] == "BUILD_FAILED_RC=1"
    assert stderr_lines[-2] == CUSTOM_BUILD_RC_UNREADABLE_REASON
    assert stderr_lines[-4:-2] == ["step one", "step two"]
    assert "Illegal number" not in run.stderr
    # what the renter reads: the build lines, then why it counts as failed
    tail = custom_build_log_tail(run.stderr)
    assert tail.endswith(CUSTOM_BUILD_RC_UNREADABLE_REASON)
    assert "BUILD_FAILED_RC" not in tail


@pytest.mark.asyncio
async def test_A18_build_timeout_tail_names_the_limit(svc, monkeypatch):
    ssh_client = _make_dind_ssh()
    monkeypatch.setattr(
        svc, "execute_and_stream_logs", _make_esl(build=(False, "Process timed out"))
    )
    monkeypatch.setattr(svc, "stream_log", AsyncMock())

    payload = _base_payload(dockerfile_content="FROM alpine\nRUN sleep 99999\n")
    ok, step, tail = await svc._custom_build_image(
        ssh_client=ssh_client,
        payload=payload,
        log_tag="t",
        default_extra={"pod_id": payload.pod_id},
    )
    assert (ok, step) == (False, "build_timeout")
    assert tail == "docker build exceeded 1200 s"


@pytest.mark.asyncio
async def test_A18b_a_build_that_prints_process_timed_out_is_not_a_timeout(svc, monkeypatch):
    """The streamer's timeout puts exactly `Process timed out` on stderr and nothing else. A build
    whose own output says "process timed out" and then fails comes back with its tail and the
    BUILD_FAILED_RC= marker: that is a docker_build failure carrying that text, not build_timeout."""
    ssh_client = _make_dind_ssh()
    printed = (
        "#6 [3/3] RUN ./fetch-weights.sh\n"
        "#6 12.40 ERROR: download process timed out after 10s\n"
        '#6 ERROR: process "/bin/sh -c ./fetch-weights.sh" did not complete successfully: exit code: 1\n'
        "BUILD_FAILED_RC=1\n"
    )
    monkeypatch.setattr(svc, "execute_and_stream_logs", _make_esl(build=(False, printed)))
    monkeypatch.setattr(svc, "stream_log", AsyncMock())

    payload = _base_payload(dockerfile_content="FROM alpine\nRUN ./fetch-weights.sh\n")
    ok, step, tail = await svc._custom_build_image(
        ssh_client=ssh_client,
        payload=payload,
        log_tag="t",
        default_extra={"pod_id": payload.pod_id},
    )
    assert (ok, step) == (False, "docker_build")
    assert "download process timed out after 10s" in tail
    assert "BUILD_FAILED_RC" not in tail


@pytest.mark.asyncio
async def test_A18c_stderr_without_the_marker_is_host_text_not_the_build_tail(svc, monkeypatch):
    """No BUILD_FAILED_RC= marker means the build script never finished: the stderr is the host
    daemon's `docker exec` error, so the renter gets a fixed reason instead of that text."""
    from services.docker_service import CUSTOM_BUILD_INTERRUPTED_REASON

    ssh_client = _make_dind_ssh()
    daemon_error = "Error response from daemon: container 3f2a9c is not running\n"
    monkeypatch.setattr(svc, "execute_and_stream_logs", _make_esl(build=(False, daemon_error)))
    monkeypatch.setattr(svc, "stream_log", AsyncMock())

    payload = _base_payload(dockerfile_content="FROM alpine\nRUN echo hi\n")
    ok, step, tail = await svc._custom_build_image(
        ssh_client=ssh_client,
        payload=payload,
        log_tag="t",
        default_extra={"pod_id": payload.pod_id},
    )
    assert (ok, step) == (False, "docker_build")
    assert tail == CUSTOM_BUILD_INTERRUPTED_REASON


def test_A19_log_tail_is_bounded_and_keeps_the_end():
    from services.docker_service import (
        CUSTOM_BUILD_LOG_TAIL_LINES,
        CUSTOM_BUILD_LOG_TAIL_MAX_CHARS,
        custom_build_log_tail,
    )

    assert custom_build_log_tail(None) is None
    assert custom_build_log_tail("\n  \n") is None
    # blank lines are dropped, trailing spaces trimmed, order kept
    assert custom_build_log_tail("a  \n\nb\n") == "a\nb"
    # the exit-code marker is the streamer's signal, not a build line: dropped before the cap
    assert custom_build_log_tail("a\nBUILD_FAILED_RC=7\n") == "a"
    assert custom_build_log_tail("BUILD_FAILED_RC=7\n") is None
    with_marker = "\n".join(f"line {i}" for i in range(30)) + "\nBUILD_FAILED_RC=1\n"
    assert custom_build_log_tail(with_marker).splitlines() == [
        f"line {i}" for i in range(30 - CUSTOM_BUILD_LOG_TAIL_LINES, 30)
    ]
    # many short lines: the last CUSTOM_BUILD_LOG_TAIL_LINES survive
    many = "\n".join(f"line {i}" for i in range(200))
    tail = custom_build_log_tail(many)
    assert tail.splitlines() == [f"line {i}" for i in range(200 - CUSTOM_BUILD_LOG_TAIL_LINES, 200)]
    # few long lines: cut from the front so the error at the end stays
    long = "x" * 5000 + "\nERROR: the reason"
    tail = custom_build_log_tail(long)
    assert len(tail) == CUSTOM_BUILD_LOG_TAIL_MAX_CHARS
    assert tail.endswith("ERROR: the reason")


# ------------------------------------------------------------------
# A.20–A.24 — the egress block must not eat the DinD's own DNS (DAH-2211
# docker_build class: 18 of 23 prod failures were `FROM` failing on a DNS
# timeout because the host resolver sits in a blocked range), and every
# setup command before the build is bounded.
# ------------------------------------------------------------------

_BLOCK = ["169.254.0.0/16", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"]


def test_A20_dind_nameservers_inside_the_block_are_the_only_ones_kept():
    resolv = (
        "# Generated by Docker Engine.\n"
        "nameserver 172.31.0.2 # AWS VPC resolver\n"
        "nameserver 10.0.0.2\n"
        "nameserver 8.8.8.8\n"
        "nameserver fd00::53\n"
        "nameserver not-an-ip\n"
        "nameserver 10.0.0.2\n"
        "search ec2.internal\n"
        "options ndots:0\n"
    )
    assert DockerService._dind_nameservers_inside_blocked_cidrs(resolv, _BLOCK) == ["172.31.0.2", "10.0.0.2"]
    # A public resolver is reachable already: nothing to allow, no rule.
    assert DockerService._dind_nameservers_inside_blocked_cidrs("nameserver 1.1.1.1\n", _BLOCK) == []
    assert DockerService._dind_nameservers_inside_blocked_cidrs("", _BLOCK) == []


def test_A21_egress_script_allows_port_53_to_the_resolver_above_the_drops():
    apply = DockerService._egress_filter_script("172.20.0.2", _BLOCK, apply=True, dns_servers=["10.0.0.2"])
    steps = apply.split("; ")
    tag = f"-m comment --comment {DIND_DNS_RULE_TAG}"
    accept_udp = f"$IPT -I DOCKER-USER -s 172.20.0.2 -d 10.0.0.2 -p udp --dport 53 {tag} -j ACCEPT"
    accept_tcp = f"$IPT -I DOCKER-USER -s 172.20.0.2 -d 10.0.0.2 -p tcp --dport 53 {tag} -j ACCEPT"
    drop_10 = "$IPT -I DOCKER-USER -s 172.20.0.2 -d 10.0.0.0/8 -j DROP"
    assert any(accept_udp in s for s in steps) and any(accept_tcp in s for s in steps)
    # `-I` inserts at the top, so the ACCEPT lines must run AFTER the DROP lines
    # to land above them. Only port 53 is opened; port 80 to the same address
    # (a metadata service) still hits the DROP.
    last_drop = max(i for i, s in enumerate(steps) if "-j DROP" in s)
    first_accept = min(i for i, s in enumerate(steps) if "-j ACCEPT" in s)
    assert last_drop < first_accept
    assert any(drop_10 in s for s in steps)
    assert "-p udp --dport 53" in apply and "--dport 80" not in apply
    # A stale ACCEPT from a build whose teardown failed sits BELOW the DROP just
    # inserted; `-C` finds it, the insert is skipped and DNS stays dropped. So
    # the ACCEPT is never `-C`-guarded: every tagged rule for this DinD IP is
    # purged from the chain first, then one copy is inserted at the top.
    assert not any("-C DOCKER-USER" in s and "-j ACCEPT" in s for s in steps)
    assert "-D DOCKER-USER -s 172.20.0.2 -d 10.0.0.2" not in apply
    assert apply.index(_PURGE_TAGGED) < apply.index(accept_udp)

    remove = DockerService._egress_filter_script("172.20.0.2", _BLOCK, apply=False)
    assert _PURGE_TAGGED_TEARDOWN in remove and "exit 4" not in remove
    assert "$IPT -D DOCKER-USER -s 172.20.0.2 -d 10.0.0.0/8 -j DROP" in remove
    assert "-I DOCKER-USER" not in remove

    # No resolver inside the block: the script is exactly today's plus the purge.
    plain = DockerService._egress_filter_script("172.20.0.2", _BLOCK, apply=True)
    assert plain == DockerService._egress_filter_script("172.20.0.2", _BLOCK, apply=True, dns_servers=[])
    assert "ACCEPT" not in plain


# The purge `_egress_filter_script` emits for DinD IP 172.20.0.2: every rule in the
# chain that carries this source and the DNS tag, deleted as `iptables -S` prints it.
def _purge_loop(delete: str) -> str:
    return (
        "printf '%s\\n' \"$rules\" | while read -r _a spec; do "
        f'case "$spec" in *"-s 172.20.0.2/32 "*"--comment {DIND_DNS_RULE_TAG} "*) '
        f"{delete};; esac; done"
    )


# apply: a failed listing aborts (a pipe would make the purge a silent no-op) and a
# failed delete aborts too (the old resolver's ACCEPT would stay in the chain)
_PURGE_TAGGED = (
    'rules=$($IPT -S DOCKER-USER 2>/dev/null) || { echo "DOCKER-USER listing failed" >&2; exit 4; }; '
    + _purge_loop("$IPT -D $spec || exit 5")
    + ' || { echo "tagged DNS rule delete failed" >&2; exit 5; }'
)
# teardown: a failed listing or delete is tolerated so the DROP deletes still run
_PURGE_TAGGED_TEARDOWN = 'rules=$($IPT -S DOCKER-USER 2>/dev/null) || rules=""; ' + _purge_loop("$IPT -D $spec")


def test_A21b_purge_removes_an_old_resolver_the_current_list_does_not_name():
    """Regression: teardown failed on a build whose resolver
    was 10.0.0.2; the DinD IP came back for a build whose resolver is 10.0.0.9.
    Deleting the current resolvers' rules, or a `-C` check, leaves the 10.0.0.2
    ACCEPT in place. The purge matches the tag and the source IP only, so the
    old rule goes whatever address it names; a rule for another DinD IP stays."""
    apply = DockerService._egress_filter_script("172.20.0.2", _BLOCK, apply=True, dns_servers=["10.0.0.9"])
    assert "10.0.0.2" not in apply
    assert apply.index(_PURGE_TAGGED) < apply.index("-d 10.0.0.9 -p udp --dport 53")
    # Run the purge line against a fake chain listing and record what `-D` sees.
    chain = "\n".join(
        [
            "-N DOCKER-USER",
            f"-A DOCKER-USER -s 172.20.0.2/32 -d 10.0.0.2/32 -p udp -m udp --dport 53 -m comment --comment {DIND_DNS_RULE_TAG} -j ACCEPT",
            f"-A DOCKER-USER -s 172.20.0.2/32 -d 10.0.0.2/32 -p tcp -m tcp --dport 53 -m comment --comment {DIND_DNS_RULE_TAG} -j ACCEPT",
            f"-A DOCKER-USER -s 172.20.0.7/32 -d 10.0.0.2/32 -p udp -m udp --dport 53 -m comment --comment {DIND_DNS_RULE_TAG} -j ACCEPT",
            "-A DOCKER-USER -s 172.20.0.2/32 -d 10.0.0.0/8 -j DROP",
            "-A DOCKER-USER -j RETURN",
        ]
    )
    with tempfile.TemporaryDirectory() as tmp:
        ipt = os.path.join(tmp, "ipt.sh")
        with open(ipt, "w") as fh:
            fh.write(
                "#!/bin/sh\n"
                f'[ "$1" = -S ] && {{ printf \'%b\\n\' \'{chain}\'; exit 0; }}\n'
                f'echo "$@" >> {shlex.quote(os.path.join(tmp, "deleted"))}\n'
            )
        os.chmod(ipt, 0o755)
        script = _PURGE_TAGGED.replace("$IPT", shlex.quote(ipt))
        subprocess.run(["sh", "-c", script], check=True, timeout=10)
        with open(os.path.join(tmp, "deleted")) as fh:
            deleted = fh.read().splitlines()
    assert deleted == [
        f"-D DOCKER-USER -s 172.20.0.2/32 -d 10.0.0.2/32 -p udp -m udp --dport 53 -m comment --comment {DIND_DNS_RULE_TAG} -j ACCEPT",
        f"-D DOCKER-USER -s 172.20.0.2/32 -d 10.0.0.2/32 -p tcp -m tcp --dport 53 -m comment --comment {DIND_DNS_RULE_TAG} -j ACCEPT",
    ]
    # A listing that fails (xtables lock) must not pass as "nothing to purge":
    # the apply exits 4 and deletes nothing; the teardown carries on.
    with tempfile.TemporaryDirectory() as tmp:
        ipt = os.path.join(tmp, "ipt.sh")
        with open(ipt, "w") as fh:
            fh.write("#!/bin/sh\n" '[ "$1" = -S ] && exit 4\n' f'echo "$@" >> {shlex.quote(os.path.join(tmp, "deleted"))}\n')
        os.chmod(ipt, 0o755)
        rc = subprocess.run(["sh", "-c", _PURGE_TAGGED.replace("$IPT", shlex.quote(ipt))], timeout=10).returncode
        assert rc == 4 and not os.path.exists(os.path.join(tmp, "deleted"))
        rc = subprocess.run(["sh", "-c", _PURGE_TAGGED_TEARDOWN.replace("$IPT", shlex.quote(ipt))], timeout=10).returncode
        assert rc == 0 and not os.path.exists(os.path.join(tmp, "deleted"))


def test_A21c_a_tagged_rule_the_apply_cannot_delete_aborts_the_apply():
    """Regression: the chain holds an old resolver's ACCEPT
    for this DinD IP and `iptables -D` on it fails (xtables lock, a rule the
    backend cannot parse back). The previous head exited 0 and went on to
    insert the new ACCEPTs next to the old one, so the chain kept 10.0.0.2
    allowed for whatever build reused the IP. Now the apply exits 5 before
    any insert, DROPs included, so an abort leaves nothing half-applied; the
    teardown still deletes what it can and exits 0."""
    apply = DockerService._egress_filter_script("172.20.0.2", _BLOCK, apply=True, dns_servers=["10.0.0.9"])
    assert apply.index("tagged DNS rule delete failed") < apply.index("-d 10.0.0.9 -p udp --dport 53")
    assert apply.index("tagged DNS rule delete failed") < apply.index("-j DROP")
    chain = "\n".join(
        [
            f"-A DOCKER-USER -s 172.20.0.2/32 -d 10.0.0.2/32 -p udp -m udp --dport 53 -m comment --comment {DIND_DNS_RULE_TAG} -j ACCEPT",
            f"-A DOCKER-USER -s 172.20.0.2/32 -d 10.0.0.2/32 -p tcp -m tcp --dport 53 -m comment --comment {DIND_DNS_RULE_TAG} -j ACCEPT",
        ]
    )
    # The fake iptables lists the chain, records every call and fails each `-D`.
    with tempfile.TemporaryDirectory() as tmp:
        ipt = os.path.join(tmp, "ipt.sh")
        calls = os.path.join(tmp, "calls")
        with open(ipt, "w") as fh:
            fh.write(
                "#!/bin/sh\n"
                f'[ "$1" = -S ] && {{ printf \'%b\\n\' \'{chain}\'; exit 0; }}\n'
                f'echo "$@" >> {shlex.quote(calls)}\n'
                '[ "$1" = -D ] && exit 1\n'
                "exit 0\n"
            )
        os.chmod(ipt, 0o755)
        # The whole apply script (chain check, DROPs, purge, ACCEPTs): it stops at the purge.
        proc = subprocess.run(
            ["sh", "-c", apply.replace("$IPT", shlex.quote(ipt))], capture_output=True, text=True, timeout=10
        )
        with open(calls) as fh:
            seen = fh.read().splitlines()
        assert proc.returncode == 5 and "tagged DNS rule delete failed" in proc.stderr
        assert seen[-1].startswith("-D DOCKER-USER -s 172.20.0.2/32 -d 10.0.0.2/32 -p udp")
        assert not any("-j ACCEPT" in c and c.startswith("-I") for c in seen), seen
        assert not any(c.startswith(("-I", "-C")) for c in seen), seen  # no DROP inserted either
    # Teardown: the first `-D` fails, the second tagged rule and the DROPs are still tried.
    remove = DockerService._egress_filter_script("172.20.0.2", _BLOCK, apply=False)
    with tempfile.TemporaryDirectory() as tmp:
        ipt = os.path.join(tmp, "ipt.sh")
        calls = os.path.join(tmp, "calls")
        with open(ipt, "w") as fh:
            fh.write(
                "#!/bin/sh\n"
                f'[ "$1" = -S ] && {{ printf \'%b\\n\' \'{chain}\'; exit 0; }}\n'
                f'echo "$@" >> {shlex.quote(calls)}\n'
                '[ "$1" = -D ] && exit 1\n'
                "exit 0\n"
            )
        os.chmod(ipt, 0o755)
        rc = subprocess.run(["sh", "-c", remove.replace("$IPT", shlex.quote(ipt))], timeout=10).returncode
        with open(calls) as fh:
            seen = fh.read().splitlines()
        assert rc == 0
        assert sum(1 for c in seen if f"--comment {DIND_DNS_RULE_TAG}" in c) == 2
        assert sum(1 for c in seen if "-j DROP" in c) == len(_BLOCK)


@pytest.mark.asyncio
async def test_A22_build_on_a_host_with_a_private_resolver_lets_dns_through(svc, monkeypatch):
    """The regression: a DinD whose resolv.conf names 10.0.0.2 built under the
    four DROP rules alone, so `FROM` failed on `lookup registry-1.docker.io on
    10.0.0.2:53: i/o timeout`. Now the applied firewall carries an ACCEPT for
    port 53 to that resolver, and the teardown removes it."""
    ssh_client = _make_dind_ssh(resolv_conf="nameserver 10.0.0.2\nsearch example.internal\n")
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
    assert ok is True and step is None

    # The resolver list is read from the DinD container itself.
    assert any(
        f"docker exec lium-dind-build-{payload.pod_id} cat /etc/resolv.conf" in c
        for c in ssh_client.calls
    ), ssh_client.calls
    applied = [c for c in esl.seen if "--network=host" in c and "DOCKER-USER" in c]
    assert len(applied) == 1
    tag = f"-m comment --comment {DIND_DNS_RULE_TAG}"
    assert f"-s 172.20.0.2 -d 10.0.0.2 -p udp --dport 53 {tag} -j ACCEPT" in applied[0]
    assert f"-s 172.20.0.2 -d 10.0.0.2 -p tcp --dport 53 {tag} -j ACCEPT" in applied[0]
    assert "-s 172.20.0.2 -d 10.0.0.0/8 -j DROP" in applied[0]
    # Teardown (ssh.run, not the streamer) purges the tagged ACCEPTs with the DROPs.
    removed = [c for c in ssh_client.calls if "--network=host" in c and "-D DOCKER-USER" in c]
    assert len(removed) == 1
    assert f"--comment {DIND_DNS_RULE_TAG} " in removed[0] and "-D $spec" in removed[0]
    assert "-d 10.0.0.0/8 -j DROP" in removed[0]


@pytest.mark.asyncio
async def test_A22b_public_resolver_or_unreadable_resolv_keeps_todays_rules(svc, monkeypatch):
    for ssh_client in (
        _make_dind_ssh(resolv_conf="nameserver 8.8.8.8\nnameserver 8.8.4.4\n"),
        _make_dind_ssh(resolv_exit=1, resolv_conf=""),
    ):
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
        assert ok is True and step is None
        applied = [c for c in esl.seen if "--network=host" in c and "DOCKER-USER" in c]
        assert len(applied) == 1 and "ACCEPT" not in applied[0]
        assert "-d 169.254.0.0/16 -j DROP" in applied[0]


@pytest.mark.asyncio
async def test_A23_setup_commands_are_bounded_and_a_hung_dind_start_fails_the_step(svc, monkeypatch):
    """Before the bound, `docker run -d <dind image>` (which pulls the image on
    a node's first custom build) had no timeout: a hang left the pod PENDING
    until the backend's 1 h stale sweep. Now every setup command carries
    `timeout=CUSTOM_DOCKERFILE_SETUP_STEP_TIMEOUT_SECONDS` and a hang fails
    the build at `build_dind_start` with the DinD torn down."""
    from core.config import settings

    monkeypatch.setattr(settings, "CUSTOM_DOCKERFILE_SETUP_STEP_TIMEOUT_SECONDS", 7)

    ssh_client = _make_dind_ssh()
    esl = _make_esl()
    monkeypatch.setattr(svc, "execute_and_stream_logs", esl)
    monkeypatch.setattr(svc, "stream_log", AsyncMock())
    payload = _base_payload(dockerfile_content="FROM alpine\nRUN echo hi\n")
    ok, step, _tail = await svc._custom_build_image(
        ssh_client=ssh_client, payload=payload, log_tag="t", default_extra={"pod_id": payload.pod_id},
    )
    assert ok is True and step is None
    # Every ssh.run, teardown and readiness probes included, carries a bound.
    # The DinD start also runs under the executor's own `timeout(1)`, which
    # kills a hung pull.
    unbounded = [
        c for c, kw in zip(ssh_client.calls, ssh_client.call_kwargs)
        if kw.get("timeout") is None
    ]
    assert unbounded == [], unbounded
    start = next(c for c in ssh_client.calls if "run -d --runtime=sysbox-runc" in c)
    assert start.startswith("timeout -k 5 7 /usr/bin/docker run -d"), start
    for marker in ("info --format", "docker inspect -f", "cat /etc/resolv.conf", "cat > /build/Dockerfile"):
        assert any(marker in c and kw.get("timeout") == 7
                   for c, kw in zip(ssh_client.calls, ssh_client.call_kwargs)), marker

    hung = _make_dind_ssh(dind_start_hangs=True)
    esl2 = _make_esl()
    monkeypatch.setattr(svc, "execute_and_stream_logs", esl2)
    payload2 = _base_payload(dockerfile_content="FROM alpine\nRUN echo hi\n")
    ok, step, _tail = await svc._custom_build_image(
        ssh_client=hung, payload=payload2, log_tag="t", default_extra={"pod_id": payload2.pod_id},
    )
    assert ok is False and step == "build_dind_start"
    assert not any("docker build" in c for c in esl2.seen)
    assert any("docker rm -fv" in c and f"lium-dind-build-{payload2.pod_id}" in c for c in hung.calls)


@pytest.mark.asyncio
async def test_A23b_a_hung_readiness_probe_fails_at_build_dind_unready_within_the_ready_bound(svc, monkeypatch):
    from core.config import settings

    monkeypatch.setattr(settings, "CUSTOM_DOCKERFILE_DIND_READY_TIMEOUT_SECONDS", 1)
    ssh_client = _make_dind_ssh(ready_hangs=True)
    esl = _make_esl()
    monkeypatch.setattr(svc, "execute_and_stream_logs", esl)
    monkeypatch.setattr(svc, "stream_log", AsyncMock())
    payload = _base_payload(dockerfile_content="FROM alpine\nRUN echo hi\n")
    # Without the per-probe bound the loop waits on the first probe for an
    # hour; the executor-side `timeout -k 2 1` on the probe turns that into one
    # not-ready probe (exit 124) after 1 s, and with a budget of one probe the
    # build fails at `build_dind_unready` instead of hanging.
    ok, step, _tail = await asyncio.wait_for(
        svc._custom_build_image(
            ssh_client=ssh_client, payload=payload, log_tag="t", default_extra={"pod_id": payload.pod_id},
        ),
        timeout=5,
    )
    assert ok is False and step == "build_dind_unready"
    probes = [(c, k) for c, k in zip(ssh_client.calls, ssh_client.call_kwargs) if c.rstrip().endswith("docker info")]
    assert len(probes) == 1
    # the bound is on the executor (`timeout(1)`), asyncssh's own timeout is only the backstop
    assert probes[0][0].startswith("timeout -k 2 1 /usr/bin/docker exec") and probes[0][1].get("timeout") == 6
    assert not any("docker build" in c for c in esl.seen)
    assert any("docker rm -fv" in c and f"lium-dind-build-{payload.pod_id}" in c for c in ssh_client.calls)


@pytest.mark.asyncio
async def test_A23c_readiness_keeps_its_probe_budget_when_probes_fail_fast(svc, monkeypatch):
    """An outer `wait_for(ready_timeout_s)` counted
    probe time against the sleep budget, so a slow host got fewer probes than
    the setting says. Now the loop runs `ready_timeout_s` probes a second apart,
    each bounded on its own."""
    from core.config import settings

    monkeypatch.setattr(settings, "CUSTOM_DOCKERFILE_DIND_READY_TIMEOUT_SECONDS", 2)
    ssh_client = _make_dind_ssh(ready_exit=1)
    esl = _make_esl()
    monkeypatch.setattr(svc, "execute_and_stream_logs", esl)
    monkeypatch.setattr(svc, "stream_log", AsyncMock())
    payload = _base_payload(dockerfile_content="FROM alpine\nRUN echo hi\n")
    ok, step, _tail = await svc._custom_build_image(
        ssh_client=ssh_client, payload=payload, log_tag="t", default_extra={"pod_id": payload.pod_id},
    )
    assert ok is False and step == "build_dind_unready"
    probes = [c for c in ssh_client.calls if c.rstrip().endswith("docker info")]
    # two probes, one second apart, each bounded on the executor by min(N, 10) = 2 s
    assert len(probes) == 2 and all(c.startswith("timeout -k 2 2 ") for c in probes)


@pytest.mark.asyncio
async def test_A23d_a_probe_that_hits_its_bound_is_not_ready_and_the_next_probe_runs(svc, monkeypatch):
    """Regression: a probe that timed out ended the
    loop, so a DinD whose first `docker info` was slow while dockerd started was
    rejected at `build_dind_unready` although the next probe would have passed.
    A timed-out probe is now "not ready" like an exit 1 (exit 124 from the
    executor-side `timeout(1)`, which also kills the remote `docker exec` so no
    session channel is left open per slow probe), and the remaining probes run."""
    from core.config import settings

    monkeypatch.setattr(settings, "CUSTOM_DOCKERFILE_DIND_READY_TIMEOUT_SECONDS", 3)
    ssh_client = _make_dind_ssh(ready_hangs_first=1)
    esl = _make_esl()
    monkeypatch.setattr(svc, "execute_and_stream_logs", esl)
    monkeypatch.setattr(svc, "stream_log", AsyncMock())
    payload = _base_payload(dockerfile_content="FROM alpine\nRUN echo hi\n")
    ok, step, _tail = await asyncio.wait_for(
        svc._custom_build_image(
            ssh_client=ssh_client, payload=payload, log_tag="t", default_extra={"pod_id": payload.pod_id},
        ),
        timeout=20,
    )
    assert ok is True and step is None
    probes = [c for c in ssh_client.calls if c.rstrip().endswith("docker info")]
    # the first probe hit its 3 s bound, the second answered ready, no third
    assert len(probes) == 2
    assert any("docker build" in c for c in esl.seen)


@pytest.mark.asyncio
async def test_A23e_a_probe_that_outlives_the_backstop_is_closed_before_the_next_one_opens(svc, monkeypatch):
    """Regression: the probe ran through `ssh_client.run(timeout=)`, which drops
    its process when the wait times out and leaves the session channel open
    until the remote command exits. On a host where even `timeout(1)` did not
    return, every probe that hit the backstop kept a channel, and sshd's
    MaxSessions (10) refused the eleventh. The probe is now `create_process` +
    `wait` inside `async with`, so a timed-out probe is closed (`close`, then
    `wait_closed`) before the next one opens, the timeout still reads as "not
    ready", and the next probe runs."""
    from core.config import settings

    monkeypatch.setattr(settings, "CUSTOM_DOCKERFILE_DIND_READY_TIMEOUT_SECONDS", 3)
    ssh_client = _make_dind_ssh(ready_backstop_first=1)
    esl = _make_esl()
    monkeypatch.setattr(svc, "execute_and_stream_logs", esl)
    monkeypatch.setattr(svc, "stream_log", AsyncMock())
    payload = _base_payload(dockerfile_content="FROM alpine\nRUN echo hi\n")
    ok, step, _tail = await asyncio.wait_for(
        svc._custom_build_image(
            ssh_client=ssh_client, payload=payload, log_tag="t", default_extra={"pod_id": payload.pod_id},
        ),
        timeout=20,
    )
    assert ok is True and step is None
    first, second = ssh_client.probe_processes
    # the backstop is asyncssh's `timeout=` on the wait: the executor bound (3) + 5
    assert first.wait_timeout == 3 + 5 and second.wait_timeout == 3 + 5
    assert first.command.startswith("timeout -k 2 3 /usr/bin/docker exec")
    # the timed-out probe's channel was closed, and closed before the second probe opened;
    # never more than one probe channel open at a time
    assert first.closed and first.close_awaited
    assert ssh_client.probe_events == [("open", 1), ("closed", 1), ("open", 2), ("closed", 2)]
    assert any("docker build" in c for c in esl.seen)


@pytest.mark.asyncio
async def test_A24_egress_helper_runs_under_the_setup_bound(svc, monkeypatch):
    from core.config import settings

    monkeypatch.setattr(settings, "CUSTOM_DOCKERFILE_SETUP_STEP_TIMEOUT_SECONDS", 11)
    seen_kwargs: list[dict] = []

    async def _esl(**kwargs):
        seen_kwargs.append(kwargs)
        return (True, "")

    monkeypatch.setattr(svc, "execute_and_stream_logs", _esl)
    monkeypatch.setattr(svc, "stream_log", AsyncMock())
    payload = _base_payload(dockerfile_content="FROM alpine\nRUN echo hi\n")
    ok, step, _tail = await svc._custom_build_image(
        ssh_client=_make_dind_ssh(), payload=payload, log_tag="t", default_extra={"pod_id": payload.pod_id},
    )
    assert ok is True and step is None
    egress = [k for k in seen_kwargs if "--network=host" in k.get("command", "")]
    assert len(egress) == 1
    # The bound is the executor's `timeout(1)` on the helper, named after the
    # DinD so the teardown can force-remove it; the streamer's own timeout is
    # the backstop past that bound (it only stops reading, the remote helper
    # would run on).
    cmd = egress[0]["command"]
    assert cmd.startswith("timeout -k 5 11 /usr/bin/docker run --rm --network=host --cap-add=NET_ADMIN "
                          f"--name lium-dind-build-{payload.pod_id}-fw-apply "), cmd
    # the wrapper's tail: exit 124 gets the stderr line the streamer fails on; a bare
    # `timeout` prefix would leave a killed helper reading as an applied firewall
    assert cmd.endswith("; exit $rc") and "'Build egress firewall helper timed out after 11s' >&2" in cmd, cmd
    assert egress[0].get("timeout") == 11 + 15
    # the streamer reads the helper's exit status for this step: a silent kill
    # (137) has no stderr line to fail on (A24c)
    assert egress[0].get("check_exit_status") is True
    build = [k for k in seen_kwargs if "docker build" in k.get("command", "")]
    assert build and build[0].get("timeout") == int(settings.CUSTOM_DOCKERFILE_BUILD_TIMEOUT_SECONDS)


@pytest.mark.skipif(shutil.which("timeout") is None, reason="needs coreutils timeout(1), as the executor has")
def test_A24b_a_firewall_helper_that_hits_its_bound_exits_124_with_a_stderr_line():
    """Regression: the helper ran under the streamer's `timeout=` only, which
    stops reading and leaves the remote `docker run` inserting rules, and
    `execute_and_stream_logs` fails a step on stderr alone, so a helper killed
    by `timeout(1)` (exit 124, silent) would have read as an applied firewall.
    The wrapper echoes one stderr line on 124 and keeps the exit status."""
    slow = DockerService._bounded_helper_command("sleep 5", 1, "Build egress firewall helper")
    proc = subprocess.run(["sh", "-c", slow], capture_output=True, text=True, timeout=10)
    assert proc.returncode == 124
    assert proc.stderr.strip() == "Build egress firewall helper timed out after 1s"
    # a helper that finishes in time: its own status, nothing on stderr
    quick = DockerService._bounded_helper_command("true", 5, "Build egress firewall helper")
    proc = subprocess.run(["sh", "-c", quick], capture_output=True, text=True, timeout=10)
    assert proc.returncode == 0 and proc.stderr == ""
    failing = DockerService._bounded_helper_command("sh -c 'echo boom >&2; exit 5'", 5, "x")
    proc = subprocess.run(["sh", "-c", failing], capture_output=True, text=True, timeout=10)
    assert proc.returncode == 5 and proc.stderr.strip() == "boom"


class _FakeStreamProcess:
    """What `ssh_client.create_process` hands `execute_and_stream_logs`: an
    async context manager with `stdout` / `stderr` line iterators, whose
    `exit_status` is known only once the channel is closed (`wait_closed`),
    as asyncssh's is. `exit_status=None` after the close is a process that
    ended on a signal and never reported a status."""

    def __init__(self, exit_status: int | None, *, stdout=(), stderr=()):
        self._final_exit_status = exit_status
        self._stdout, self._stderr = list(stdout), list(stderr)
        self.exit_status = None
        self.exit_signal = None
        self.closed = False

    @property
    def stdout(self):
        return self._lines(self._stdout)

    @property
    def stderr(self):
        return self._lines(self._stderr)

    @staticmethod
    async def _lines(lines):
        for line in lines:
            yield line

    async def wait_closed(self):
        self.closed = True
        self.exit_status = self._final_exit_status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None


def _make_process_ssh(*, egress_exit: int | None, egress_stderr=()):
    """An `ssh_client` for the REAL `execute_and_stream_logs`: `run` is the
    `_make_dind_ssh` router (DinD start, teardown), the readiness probes are
    its `_FakeProbeProcess`es; `create_process` serves the streamed steps, the
    egress helper with `egress_exit` and no stdout, every other command
    (build, export) with exit 0."""
    ssh = _make_dind_ssh()
    processes: list[tuple[str, _FakeStreamProcess]] = []

    def _create_process(command: str):
        if _is_probe(command):
            return ssh.probe_process(command)
        if "--network=host" in command:
            proc = _FakeStreamProcess(egress_exit, stderr=egress_stderr)
        else:
            proc = _FakeStreamProcess(0)
        processes.append((command, proc))
        return proc

    ssh.create_process = _create_process
    ssh.processes = processes
    return ssh


@pytest.mark.asyncio
@pytest.mark.parametrize("helper_exit", [137, 1])
async def test_A24c_egress_helper_silent_non_zero_exit_fails_the_apply_and_never_builds(
    svc, monkeypatch, helper_exit
):
    """Regression (taiberium, round 2): `execute_and_stream_logs` failed a
    step on stderr only, so a helper that died without a word read as an
    applied firewall and the build started open. 137 is `timeout -k`'s
    SIGKILL after the grace period (the wrapper's 124 line never prints); 1
    is `docker run` failing before iptables ran. Both, and any other
    non-zero, fail the apply at `build_egress_setup` through the REAL
    streamer, on the exit status alone."""
    ssh_client = _make_process_ssh(egress_exit=helper_exit)
    stream_log = AsyncMock()
    monkeypatch.setattr(svc, "stream_log", stream_log)
    errors: list = []
    monkeypatch.setattr("services.docker_service.logger.error", lambda msg, *a, **k: errors.append(msg))
    payload = _base_payload(dockerfile_content="FROM alpine\nRUN echo hi\n")
    ok, step, _tail = await svc._custom_build_image(
        ssh_client=ssh_client, payload=payload, log_tag="t", default_extra={"pod_id": payload.pod_id},
    )
    assert (ok, step) == (False, "build_egress_setup")
    egress = [(c, p) for c, p in ssh_client.processes if "--network=host" in c]
    assert len(egress) == 1 and egress[0][1].closed, "the exit status is read after the channel closed"
    # the exit status was the only signal: the one error line in the pod log is the status line
    error_lines = [c.args[0] for c in stream_log.call_args_list if c.args[1] == "error"]
    assert error_lines == [f"Process exited with status {helper_exit}"], error_lines
    assert not any("docker build" in c for c, _ in ssh_client.processes)
    fw_errors = [e for e in errors if "egress firewall failed" in str(e)]
    assert len(fw_errors) == 1 and f"Process exited with status {helper_exit}" in fw_errors[0].extra["error"]
    # the DinD is still torn down and the (idempotent) remove helper still runs
    assert any("docker rm -fv" in c and f"lium-dind-build-{payload.pod_id}" in c for c in ssh_client.calls)
    assert any("--network=host" in c and "-D DOCKER-USER" in c for c in ssh_client.calls)


@pytest.mark.asyncio
async def test_A24d_egress_helper_exit_zero_proceeds_to_the_build(svc, monkeypatch):
    """The other half of A24c: a helper that exits 0 with nothing on stderr
    is a confirmed firewall, and the build runs through the same real
    streamer."""
    ssh_client = _make_process_ssh(egress_exit=0)
    monkeypatch.setattr(svc, "stream_log", AsyncMock())
    payload = _base_payload(dockerfile_content="FROM alpine\nRUN echo hi\n")
    ok, step, _tail = await svc._custom_build_image(
        ssh_client=ssh_client, payload=payload, log_tag="t", default_extra={"pod_id": payload.pod_id},
    )
    assert (ok, step) == (True, None)
    commands = [c for c, _ in ssh_client.processes]
    egress_at = next(i for i, c in enumerate(commands) if "--network=host" in c)
    build_at = next(i for i, c in enumerate(commands) if "docker build" in c)
    assert egress_at < build_at


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_status", [137, 1, None])
async def test_A24e_execute_and_stream_logs_fails_on_the_exit_status_only_when_asked(
    svc, monkeypatch, exit_status
):
    """`check_exit_status=True` fails on any non-zero exit status, or on none
    (a signal death that reported no status), with the status in the error
    and one error line in the pod log. Without the flag the streamer keeps its
    stderr-only verdict, so the other callers' commands (whose exit status is
    not a verdict) are unchanged."""
    ssh_client = AsyncMock()
    ssh_client.create_process = lambda command: _FakeStreamProcess(exit_status)
    stream_log = AsyncMock()
    monkeypatch.setattr(svc, "stream_log", stream_log)

    ok, err = await svc.execute_and_stream_logs(
        ssh_client=ssh_client, command="helper", log_tag="t", log_text="Applying",
        raise_exception=False, check_exit_status=True,
    )
    assert ok is False
    expected = (
        f"Process exited with status {exit_status}" if exit_status is not None
        else "Process ended without an exit status"
    )
    assert expected in err
    assert any(c.args[1] == "error" and expected in c.args[0] for c in stream_log.call_args_list)

    ok, err = await svc.execute_and_stream_logs(
        ssh_client=ssh_client, command="helper", log_tag="t", log_text="Applying", raise_exception=False,
    )
    assert (ok, err) == (True, "")

    # `raise_exception=True` (the default) raises on the exit status as it does on stderr
    with pytest.raises(Exception, match=re.escape(expected)):
        await svc.execute_and_stream_logs(
            ssh_client=ssh_client, command="helper", log_tag="t", log_text="Applying", check_exit_status=True,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("remove_helper_exit", [0, 124])
async def test_A26_teardown_bounds_the_remove_helper_and_removes_both_helpers_before_the_dind(
    svc, monkeypatch, remove_helper_exit
):
    """The teardown's remove helper runs under the executor's `timeout -k 5`,
    exit 124 is logged as a failed teardown (not silence), and both firewall
    helpers are `docker rm -f`'d by name before `docker rm -fv` releases the
    DinD IP: a helper that outlived its bound is still editing rules for that
    IP, and the next build to get the IP would inherit them."""
    from core.config import settings

    monkeypatch.setattr(settings, "CUSTOM_DOCKERFILE_SETUP_STEP_TIMEOUT_SECONDS", 9)
    ssh_client = _make_dind_ssh(remove_helper_exit=remove_helper_exit)
    monkeypatch.setattr(svc, "execute_and_stream_logs", _make_esl())
    monkeypatch.setattr(svc, "stream_log", AsyncMock())
    warnings: list = []  # the _StructuredMessage objects: message + extra
    monkeypatch.setattr("services.docker_service.logger.warning", lambda msg, *a, **k: warnings.append(msg))
    payload = _base_payload(dockerfile_content="FROM alpine\nRUN echo hi\n")
    ok, step, _tail = await svc._custom_build_image(
        ssh_client=ssh_client, payload=payload, log_tag="t", default_extra={"pod_id": payload.pod_id},
    )
    assert ok is True and step is None
    dind = f"lium-dind-build-{payload.pod_id}"
    remove_helper = next(
        (i, c) for i, c in enumerate(ssh_client.calls) if "--network=host" in c and "-D DOCKER-USER" in c
    )
    helpers_rm = next(
        (i, c) for i, c in enumerate(ssh_client.calls)
        if c.startswith("/usr/bin/docker rm -f ") and f"{dind}-fw-apply" in c
    )
    dind_rm = next((i, c) for i, c in enumerate(ssh_client.calls) if "docker rm -fv" in c and dind in c)
    assert remove_helper[1].startswith(
        f"timeout -k 5 9 /usr/bin/docker run --rm --network=host --cap-add=NET_ADMIN --name {dind}-fw-remove "
    ), remove_helper[1]
    assert remove_helper[1].endswith("; exit $rc") and "remove helper timed out after 9s' >&2" in remove_helper[1]
    assert ssh_client.call_kwargs[remove_helper[0]].get("timeout") == 9 + 15
    assert f"{dind}-fw-remove" in helpers_rm[1]
    assert remove_helper[0] < helpers_rm[0] < dind_rm[0], ssh_client.calls
    teardown_warnings = [w for w in warnings if "egress rule teardown failed" in str(w)]
    if remove_helper_exit == 124:
        assert len(teardown_warnings) == 1, warnings
        assert teardown_warnings[0].extra.get("timed_out") is True
        assert teardown_warnings[0].extra.get("exit_status") == 124
    else:
        assert teardown_warnings == []


@pytest.mark.parametrize("bad", [0, -1])
def test_A25b_dind_ready_timeout_rejects_zero_and_negative(bad):
    """`range(0)` runs no probe at all, so zero or a negative
    value would fail every build at `build_dind_unready` without asking dockerd
    once. The setting refuses both at load time."""
    from pydantic import ValidationError

    from core.config import Settings

    with pytest.raises(ValidationError):
        Settings(_env_file=None, CUSTOM_DOCKERFILE_DIND_READY_TIMEOUT_SECONDS=bad)


@pytest.mark.parametrize("bad", [0, -1])
def test_A25_setup_step_timeout_rejects_zero_and_negative(bad):
    """asyncssh's `run(timeout=0)` and `timeout=-1` time out at once, while
    `execute_and_stream_logs(timeout=0)` disables its bound: a zero or negative
    value would fail every setup command on the spot or unbound the helper.
    The setting refuses both at load time."""
    from pydantic import ValidationError

    from core.config import Settings

    with pytest.raises(ValidationError):
        Settings(_env_file=None, CUSTOM_DOCKERFILE_SETUP_STEP_TIMEOUT_SECONDS=bad)
    assert Settings(_env_file=None, CUSTOM_DOCKERFILE_SETUP_STEP_TIMEOUT_SECONDS=1).CUSTOM_DOCKERFILE_SETUP_STEP_TIMEOUT_SECONDS == 1


# ------------------------------------------------------------------
# B.1 — SSE latency: stub the redis publish path and measure p95
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_B1_log_emit_to_publish_p95_under_2s(svc):
    """Each `stream_log` -> `redis.publish` chunk must reach the SSE consumer
    within p95 ≤ 2000 ms over 200 emitted lines. This stubs the redis
    publish so the test runs in CI without infrastructure."""
    received: list[tuple[str, float]] = []

    class _Redis:
        async def publish(self, channel, payload):
            # Record the receive timestamp per line entry.
            now = time.perf_counter()
            for entry in payload.get("logs", []):
                received.append((entry["log_text"], now))

    svc.redis_service = _Redis()

    # Start the log shipper coroutine.
    shipper = asyncio.create_task(svc.handle_stream_logs(
        miner_hotkey="m", executor_id="e", pod_id="p",
    ))

    # Emit 200 lines, recording emit timestamps.
    emit_times: dict[str, float] = {}
    for i in range(200):
        msg = f"build-line-{i:03d}"
        emit_times[msg] = time.perf_counter()
        await svc.stream_log(msg, "success", "build")
        await asyncio.sleep(0.001)  # 1ms cadence

    # Give the shipper a short grace period to flush, then stop it.
    deadline = time.perf_counter() + 6.0  # well under the 2s p95 requirement budget
    while len(received) < 200 and time.perf_counter() < deadline:
        await asyncio.sleep(0.05)

    await svc.finish_stream_logs()
    shipper.cancel()
    try:
        await shipper
    except asyncio.CancelledError:
        pass

    assert len(received) >= 200, f"only {len(received)} of 200 lines received"

    latencies_ms = sorted(
        (recv_t - emit_times[msg]) * 1000 for msg, recv_t in received[:200] if msg in emit_times
    )
    p95 = latencies_ms[int(0.95 * len(latencies_ms))]
    assert p95 <= 2000, f"p95 emit→publish latency={p95:.1f}ms exceeds 2000ms budget"


def test_A16e_str_of_a_failed_request_leaves_out_the_build_tail():
    """compute_client logs `str(response)` at INFO; the tail must stay out of it and stay on the wire."""
    request = FailedContainerRequest(
        miner_hotkey="miner",
        executor_id=str(uuid4()),
        pod_id=str(uuid4()),
        msg="Custom build failed",
        build_log_tail="#3 ARG HF_TOKEN=abc",
    )

    assert "HF_TOKEN=abc" not in str(request)
    assert request.model_dump()["build_log_tail"] == "#3 ARG HF_TOKEN=abc"
