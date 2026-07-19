"""Characterization goldens for the LIVE startup-commands argv sink and the
rental known-hosts chokepoint (DAH-2382 PR-1a, DRAFT sketches D21-D24).

The live builder is `build_container_command_argv` (rental_docker_sdk.py) —
NOT the dead host-shell-quoting `build_startup_command_args` — and its live
sink is `run_spec.command` via `_build_rental_container_run_spec`, so these
tests pin the spec boundary. D24 characterizes the documented fail-open hole
AS-IS per decision D-A (backlog, not fix-first). Baseline 6be5649f.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock

import pytest
from docker_oracle.harness import (
    make_create_request,
    make_docker_service,
    make_executor_info,
)
from payload_models.payloads import CustomOptions
from services.attestation_service import AttestationError
from services.rental_docker_sdk import ContainerRunSpec, build_gpu_docker_config

from core.config import settings


def _build_run_spec(startup_commands: str) -> ContainerRunSpec:
    # drive the LIVE argv builder through its real sink: run_spec.command
    svc = make_docker_service()
    return svc._build_rental_container_run_spec(
        payload=make_create_request(),
        container_name="pod_argv",
        custom_options=CustomOptions(startup_commands=startup_commands),
        port_maps=[(22, 20001, 20001)],
        local_volume="volume_argv",
        local_volume_path="/root",
        external_volume_name=None,
        gpu_devices=build_gpu_docker_config(["GPU-test"]),
        effective_storage_limit_gb=1,
        cpu_count=1,
    )


# --- D21-D23: build_container_command_argv -> run_spec.command ----------------


def test_startup_commands_argv_neutralizes_injection() -> None:
    # Arrange / Act
    spec = _build_run_spec("echo hi; rm -rf /")

    # Assert
    # WHY (D21 case 2): the live builder shlex-splits into an SDK argv tuple
    # executed with NO shell — ';' stays glued inside the "hi;" token instead
    # of chaining a second command (case 1, "\n\nsh ...", is pinned
    # full-boundary in test_docker_service_rental_security.py).
    assert spec.command == ("echo", "hi;", "rm", "-rf", "/")


def test_startup_commands_argv_preserves_quoted_metacharacters() -> None:
    # Arrange / Act
    spec = _build_run_spec('/bin/bash -c "a && b"')

    # Assert
    # WHY (D22): legitimate in-container shell usage survives — the quoted
    # '&&' payload reaches the container as ONE argv element, so neutralizing
    # injection does not break `bash -c` workloads.
    assert spec.command == ("/bin/bash", "-c", "a && b")


def test_startup_commands_unbalanced_quotes_fall_back() -> None:
    # Arrange / Act
    spec = _build_run_spec('bash -c "unterminated')

    # Assert
    # WHY (D23): shlex.split raises ValueError on the unbalanced quote and the
    # builder falls back to an EMPTY argv (image default CMD) rather than
    # propagating or guessing a tokenization.
    assert spec.command == ()


# --- D24: _prepare_known_hosts_policy fail-open / fail-closed -----------------


@pytest.mark.asyncio
async def test_prepare_known_hosts_policy_fail_open_returns_none(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    svc = make_docker_service()
    svc.attestation_service.prepare_host_policy = AsyncMock(
        side_effect=RuntimeError("host key store unavailable")
    )
    monkeypatch.setattr(settings, "ENABLE_RENTAL_HOSTKEY_FAIL_CLOSED", False)
    payload = make_create_request()
    executor_info = make_executor_info(payload)

    # Act
    with caplog.at_level(logging.WARNING):
        policy = await svc._prepare_known_hosts_policy(executor_info, "miner", {})

    # Assert
    # WHY (D24, decision D-A): the documented fail-open hole is characterized
    # AS-IS and NOT fixed here — a non-attestation error with the flag off
    # returns None (=> unpinned rental SSH downstream) and the warning line is
    # the only trace the branch fired.
    assert policy is None
    assert "Unable to prepare known_hosts policy" in [
        str(record.msg) for record in caplog.records
    ]


@pytest.mark.asyncio
async def test_prepare_known_hosts_policy_fail_closed_raises_attestation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    svc = make_docker_service()
    svc.attestation_service.prepare_host_policy = AsyncMock(
        side_effect=RuntimeError("host key store unavailable")
    )
    monkeypatch.setattr(settings, "ENABLE_RENTAL_HOSTKEY_FAIL_CLOSED", True)
    payload = make_create_request()
    executor_info = make_executor_info(payload)

    # Act / Assert
    # WHY (D24 counterpart): with the flag on, the SAME non-attestation error
    # refuses unpinned SSH via AttestationError; the byte-frozen message
    # interpolates executor.address:port (the executor API port 8080, NOT
    # ssh_port 2200).
    with pytest.raises(AttestationError) as excinfo:
        await svc._prepare_known_hosts_policy(executor_info, "miner", {})
    assert str(excinfo.value) == (
        "Unable to prepare host-key policy for executor 127.0.0.1:8080; "
        "refusing unpinned rental SSH"
    )
