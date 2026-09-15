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
    ready_exit: int = 0,
    dind_ip: str = "172.20.0.2",
):
    """An `ssh.run` router emulating the DAH-2211 DinD build control commands.

    Routes by command substring: sysbox preflight, DinD `run -d`, the
    readiness `docker exec ... docker info` probe, IP inspect, and everything
    else (Dockerfile write, teardown) → exit 0. The build / egress / export
    steps go through `execute_and_stream_logs`, not `ssh.run`.
    """
    calls: list[str] = []

    async def _run(cmd, **kw):
        calls.append(cmd)
        if "info --format" in cmd and "Runtimes" in cmd:
            runtimes = '{"runc":{"path":"runc"}'
            if sysbox:
                runtimes += ',"sysbox-runc":{"path":"/usr/bin/sysbox-runc"}'
            runtimes += "}"
            return _ssh_result(stdout=runtimes)
        if "run -d --runtime=sysbox-runc" in cmd:
            return _ssh_result(exit_status=dind_start_exit, stdout="dind-cid")
        if "docker exec" in cmd and cmd.rstrip().endswith("docker info"):
            return _ssh_result(exit_status=ready_exit)
        if "docker inspect -f" in cmd:
            return _ssh_result(stdout=dind_ip)
        return _ssh_result(exit_status=0)

    ssh = AsyncMock()
    ssh.run = _run
    ssh.calls = calls
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
    ok, step, _tail = await svc._custom_build_image(
        ssh_client=ssh_client,
        payload=payload,
        log_tag="t",
        default_extra={"pod_id": payload.pod_id},
    )
    assert ok is False
    assert step == "build_export"


# ------------------------------------------------------------------
# A.16–A.19 — the failure says WHY, not only WHERE. Regression: in the 14 d to
# 15 Sep 2026 every one of the 22 docker_build failures reached the backend as
# "Custom dockerfile build failed (failure_step=docker_build)" and nothing else;
# the build output only ever went to the live log stream, which is deleted with
# the pod 3 min later.
# ------------------------------------------------------------------

_BUILD_ERROR_OUTPUT = (
    "#5 [2/2] RUN apt-get install -y nonexistent-package\n"
    "#5 0.412 E: Unable to locate package nonexistent-package\n"
    "#5 ERROR: process \"/bin/sh -c apt-get install -y nonexistent-package\" did not complete successfully: exit code: 100\n"
    "BUILD_FAILED_RC=1\n"
)


@pytest.mark.asyncio
async def test_A16_build_failure_carries_the_build_output_tail(svc, monkeypatch):
    """The CCF for a failed `docker build` carries the last lines the build printed, on the
    wire (`build_log_tail`), in `detail` and in the log text — not only the step name."""
    ssh_client = _make_dind_ssh()
    _patch_create_container_happy(svc, monkeypatch, ssh_client)
    monkeypatch.setattr(svc, "_cleanup_custom_build_artifacts", AsyncMock())
    monkeypatch.setattr(svc, "execute_and_stream_logs", _make_esl(build=(False, _BUILD_ERROR_OUTPUT)))

    payload = _base_payload(dockerfile_content="FROM ubuntu\nRUN apt-get install -y nonexistent-package\n")
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
    # ops path (detail -> backend logs / filler_run.failure_reason) says the same thing
    assert "Unable to locate package nonexistent-package" in result.detail
    # the headline stays the renter-safe constant the backend trims to
    assert result.msg.startswith("Failed create_container")
    # the wire field is the renter's build output only — never the executor host
    assert "127.0.0.1" not in result.build_log_tail and "2200" not in result.build_log_tail


@pytest.mark.asyncio
async def test_A16b_non_build_failures_have_no_tail(svc, monkeypatch):
    """`build_log_tail` is None for a template pod whose creation fails: the field belongs to
    custom builds only, so the backend can trust it as renter-safe build output."""
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(return_value=_ssh_result())
    _patch_create_container_happy(svc, monkeypatch, ssh_client)
    monkeypatch.setattr(svc, "execute_and_stream_logs", AsyncMock(return_value=(True, "")))
    monkeypatch.setattr(svc, "create_local_volume", AsyncMock(side_effect=RuntimeError("no space left on device")))

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
    """Regression (DAH-3504): a `volume_creation` failure reached the backend as the bare step
    name; `step_detail` now carries the Docker daemon's reason. The dead-transport text docker-py
    raises after a long build gets a plain-language hint in front of it."""
    from services.docker_service import STALE_SDK_TRANSPORT_HINT
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
    assert result.step_detail.startswith(STALE_SDK_TRANSPORT_HINT)
    assert "Docker SDK create volume failed" in result.step_detail
    assert "127.0.0.1" not in result.step_detail and "2200" not in result.step_detail
    assert result.build_log_tail is None


@pytest.mark.asyncio
async def test_A20c_volume_sizing_failure_carries_the_min_size_reason(svc, monkeypatch):
    """`volume_sizing` is the other step a pod dies at before its volume exists; its
    VolumeMinSizeError text is the renter's answer and travels as `step_detail`, without the hint."""
    from services.docker_service import STALE_SDK_TRANSPORT_HINT, VolumeMinSizeError

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
    assert not result.step_detail.startswith(STALE_SDK_TRANSPORT_HINT)

@pytest.mark.asyncio
async def test_A20b_non_volume_failures_have_no_step_detail(svc, monkeypatch):
    """`step_detail` belongs to the volume step only: a failure elsewhere (here the custom build)
    leaves it None so the backend can trust it as the daemon's volume reason."""
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
        STALE_SDK_TRANSPORT_HINT,
        VOLUME_STEP_DETAIL_MAX_CHARS,
        volume_step_detail,
    )

    assert volume_step_detail(RuntimeError("")) is None
    assert volume_step_detail(RuntimeError("  \n ")) is None
    plain = volume_step_detail(RuntimeError("Docker SDK create volume failed:\n  no space left on device"))
    assert plain == "Docker SDK create volume failed: no space left on device"
    assert not plain.startswith(STALE_SDK_TRANSPORT_HINT)
    long = volume_step_detail(RuntimeError("x" * 1000))
    assert len(long) == VOLUME_STEP_DETAIL_MAX_CHARS


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
    assert "tail -n 25 /tmp/lium-build.log >&2" in build_cmd
    assert "echo BUILD_FAILED_RC=$rc >&2" in build_cmd
    # the log never lands inside the build context (a `COPY .` must not pick it up)
    assert "/build/build.log" not in build_cmd


@pytest.mark.asyncio
async def test_A18_build_timeout_tail_names_the_limit(svc, monkeypatch):
    ssh_client = _make_dind_ssh()
    monkeypatch.setattr(svc, "execute_and_stream_logs", _make_esl(build=(False, "Process timed out")))
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
