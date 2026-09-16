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
- A.20–A.24 (DAH-3521) the egress block lets the DinD's own DNS through;
  every setup command before the build is bounded
- B.1 SSE latency p95 ≤ 2000 ms (stubbed redis consumer)
"""

from __future__ import annotations

import asyncio
import json
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
from services.docker_service import DockerService
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


def _make_dind_ssh(
    *,
    sysbox: bool = True,
    dind_start_exit: int = 0,
    dind_start_hangs: bool = False,
    ready_exit: int = 0,
    ready_hangs: bool = False,
    dind_ip: str = "172.20.0.2",
    resolv_conf: str = "nameserver 8.8.8.8\n",
    resolv_exit: int = 0,
):
    """An `ssh.run` router emulating the DAH-2211 DinD build control commands.

    Routes by command substring: sysbox preflight, DinD `run -d`, the
    readiness `docker exec ... docker info` probe, IP inspect, the DinD
    `cat /etc/resolv.conf` read, and everything else (Dockerfile write,
    teardown) → exit 0. The build / egress / export steps go through
    `execute_and_stream_logs`, not `ssh.run`. `dind_start_hangs` makes the
    `run -d` raise `asyncio.TimeoutError`, what asyncssh raises when the
    command outlives its `timeout=`; `ready_hangs` makes every readiness
    probe block for an hour (a hung host dockerd).
    """
    calls: list[str] = []
    call_kwargs: list[dict] = []

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
        if "docker exec" in cmd and cmd.rstrip().endswith("docker info"):
            if ready_hangs:
                await asyncio.sleep(3600)
            return _ssh_result(exit_status=ready_exit)
        if "docker inspect -f" in cmd:
            return _ssh_result(stdout=dind_ip)
        if "docker exec" in cmd and cmd.rstrip().endswith("cat /etc/resolv.conf"):
            return _ssh_result(exit_status=resolv_exit, stdout=resolv_conf)
        return _ssh_result(exit_status=0)

    ssh = AsyncMock()
    ssh.run = _run
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
    build_mock = AsyncMock(return_value=(True, None))
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

    build_mock = AsyncMock(return_value=(True, None))
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

    monkeypatch.setattr(svc, "_custom_build_image", AsyncMock(return_value=(False, "docker_build")))
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
    ok, step = await svc._custom_build_image(
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
    ok, step = await svc._custom_build_image(
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
    ok, step = await svc._custom_build_image(
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


# ------------------------------------------------------------------
# A.14 — DinD container is always torn down (finally), even on success
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_A14_dind_container_always_torn_down(svc, monkeypatch):
    ssh_client = _make_dind_ssh()
    monkeypatch.setattr(svc, "execute_and_stream_logs", _make_esl())
    monkeypatch.setattr(svc, "stream_log", AsyncMock())

    payload = _base_payload(dockerfile_content="FROM alpine\nRUN echo hi\n")
    ok, step = await svc._custom_build_image(
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
    ok, step = await svc._custom_build_image(
        ssh_client=ssh_client,
        payload=payload,
        log_tag="t",
        default_extra={"pod_id": payload.pod_id},
    )
    assert ok is False
    assert step == "build_export"


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
    assert DockerService._parse_dind_nameservers(resolv, _BLOCK) == ["172.31.0.2", "10.0.0.2"]
    # A public resolver is reachable already: nothing to allow, no rule.
    assert DockerService._parse_dind_nameservers("nameserver 1.1.1.1\n", _BLOCK) == []
    assert DockerService._parse_dind_nameservers("", _BLOCK) == []


def test_A21_egress_script_allows_port_53_to_the_resolver_above_the_drops():
    apply = DockerService._egress_filter_script("172.20.0.2", _BLOCK, apply=True, dns_servers=["10.0.0.2"])
    steps = apply.split("; ")
    accept_udp = "$IPT -I DOCKER-USER -s 172.20.0.2 -d 10.0.0.2 -p udp --dport 53 -j ACCEPT"
    accept_tcp = "$IPT -I DOCKER-USER -s 172.20.0.2 -d 10.0.0.2 -p tcp --dport 53 -j ACCEPT"
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
    # the ACCEPT is never `-C`-guarded: every copy is deleted, then one is
    # inserted at the top.
    assert not any("-C DOCKER-USER" in s and "-j ACCEPT" in s for s in steps)
    for proto, insert in (("udp", accept_udp), ("tcp", accept_tcp)):
        rule = f"-s 172.20.0.2 -d 10.0.0.2 -p {proto} --dport 53 -j ACCEPT"
        assert f"while $IPT -D DOCKER-USER {rule} 2>/dev/null; do :; done" in apply
        assert apply.index(f"-D DOCKER-USER {rule}") < apply.index(insert)

    remove = DockerService._egress_filter_script("172.20.0.2", _BLOCK, apply=False, dns_servers=["10.0.0.2"])
    assert "while $IPT -D DOCKER-USER -s 172.20.0.2 -d 10.0.0.2 -p udp --dport 53 -j ACCEPT 2>/dev/null; do :; done" in remove
    assert "while $IPT -D DOCKER-USER -s 172.20.0.2 -d 10.0.0.2 -p tcp --dport 53 -j ACCEPT 2>/dev/null; do :; done" in remove
    assert "$IPT -D DOCKER-USER -s 172.20.0.2 -d 10.0.0.0/8 -j DROP" in remove
    assert "-I DOCKER-USER" not in remove

    # No resolver inside the block: the script is exactly today's.
    plain = DockerService._egress_filter_script("172.20.0.2", _BLOCK, apply=True)
    assert plain == DockerService._egress_filter_script("172.20.0.2", _BLOCK, apply=True, dns_servers=[])
    assert "ACCEPT" not in plain


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
    ok, step = await svc._custom_build_image(
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
    assert "-s 172.20.0.2 -d 10.0.0.2 -p udp --dport 53 -j ACCEPT" in applied[0]
    assert "-s 172.20.0.2 -d 10.0.0.2 -p tcp --dport 53 -j ACCEPT" in applied[0]
    assert "-s 172.20.0.2 -d 10.0.0.0/8 -j DROP" in applied[0]
    # Teardown (ssh.run, not the streamer) deletes the ACCEPT with the DROPs.
    removed = [c for c in ssh_client.calls if "--network=host" in c and "-D DOCKER-USER" in c]
    assert len(removed) == 1
    assert "-d 10.0.0.2 -p udp --dport 53 -j ACCEPT" in removed[0]
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
        ok, step = await svc._custom_build_image(
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
    ok, step = await svc._custom_build_image(
        ssh_client=ssh_client, payload=payload, log_tag="t", default_extra={"pod_id": payload.pod_id},
    )
    assert ok is True and step is None
    # Every ssh.run, teardown included, carries the bound, except the readiness
    # probes (their loop is bounded as a whole, below). The DinD start also
    # runs under the executor's own `timeout(1)`, which kills a hung pull.
    unbounded = [
        c for c, kw in zip(ssh_client.calls, ssh_client.call_kwargs)
        if kw.get("timeout") is None and not c.rstrip().endswith("docker info")
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
    ok, step = await svc._custom_build_image(
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
    # Without the bound the loop waits on the first probe for an hour; the
    # outer wait_for turns that into a failure here instead of a hung run.
    ok, step = await asyncio.wait_for(
        svc._custom_build_image(
            ssh_client=ssh_client, payload=payload, log_tag="t", default_extra={"pod_id": payload.pod_id},
        ),
        timeout=5,
    )
    assert ok is False and step == "build_dind_unready"
    assert not any("docker build" in c for c in esl.seen)
    assert any("docker rm -fv" in c and f"lium-dind-build-{payload.pod_id}" in c for c in ssh_client.calls)


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
    ok, step = await svc._custom_build_image(
        ssh_client=_make_dind_ssh(), payload=payload, log_tag="t", default_extra={"pod_id": payload.pod_id},
    )
    assert ok is True and step is None
    egress = [k for k in seen_kwargs if "--network=host" in k.get("command", "")]
    assert len(egress) == 1 and egress[0].get("timeout") == 11
    build = [k for k in seen_kwargs if "docker build" in k.get("command", "")]
    assert build and build[0].get("timeout") == int(settings.CUSTOM_DOCKERFILE_BUILD_TIMEOUT_SECONDS)


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
