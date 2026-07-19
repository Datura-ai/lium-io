"""Characterization goldens for add_ssh_key / remove_ssh_keys / install_jupyter_server
(DAH-2382 PR-1a, DRAFT sketches D14-D20 + the D11-style attestation variants).

Every `msg` assert is byte-for-byte against the template literal in
docker_service.py: `str(_m(...))` returns only the bare message (DRAFT §0
discovery 1), so the serialized `msg` IS the template. Baseline 6be5649f.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from docker_oracle.harness import (
    FakeRentalDockerClient,
    FakeRentalDockerFactory,
    make_docker_service,
    make_executor_info,
    patch_rental_ssh_connect,
)
from payload_models.payloads import (
    AddSshPublicKeyRequest,
    FailedContainerErrorCodes,
    FailedContainerErrorTypes,
    FailedContainerRequest,
    InstallJupyterServerRequest,
    JupyterInstallationFailed,
    JupyterServerInstalled,
    RemoveSshPublicKeysRequest,
    WorkloadKind,
)
from services.attestation_service import AttestationError
from services.docker_service import DockerService

_HEX_DIGITS = set("0123456789abcdef")


def _make_service() -> DockerService:
    # honest attestation stub so the REAL _prepare_known_hosts_policy chokepoint runs
    svc = make_docker_service()
    svc.ssh_service.decrypt_payload = Mock(return_value="private-key")
    svc.attestation_service.prepare_host_policy = AsyncMock(
        return_value=SimpleNamespace(known_hosts=None)
    )
    svc.rental_docker_client_factory = FakeRentalDockerFactory(FakeRentalDockerClient())
    return svc


def _keypair() -> Mock:
    return Mock(ss58_address="validator-hotkey")


def _add_key_payload(**overrides: Any) -> AddSshPublicKeyRequest:
    base: dict[str, Any] = dict(
        miner_hotkey="miner",
        executor_id=str(uuid4()),
        pod_id="pod-id",
        workload_kind=WorkloadKind.CUSTOMER_RENTAL,
        container_name="pod_test",
        user_public_keys=["ssh-ed25519 test-key"],
    )
    base.update(overrides)
    return AddSshPublicKeyRequest(**base)


def _remove_keys_payload(**overrides: Any) -> RemoveSshPublicKeysRequest:
    base: dict[str, Any] = dict(
        miner_hotkey="miner",
        executor_id=str(uuid4()),
        pod_id="pod-id",
        workload_kind=WorkloadKind.CUSTOMER_RENTAL,
        container_name="pod_test",
        user_public_keys=["ssh-ed25519 test-key"],
    )
    base.update(overrides)
    return RemoveSshPublicKeysRequest(**base)


def _jupyter_payload(**overrides: Any) -> InstallJupyterServerRequest:
    base: dict[str, Any] = dict(
        miner_hotkey="miner",
        executor_id=str(uuid4()),
        pod_id="pod-id",
        workload_kind=WorkloadKind.CUSTOMER_RENTAL,
        container_name="pod_test",
        jupyter_port_map=(8888, 30888),
    )
    base.update(overrides)
    return InstallJupyterServerRequest(**base)


# --- D14 / D15: empty-keys early returns -------------------------------------


@pytest.mark.asyncio
async def test_add_ssh_key_no_keys() -> None:
    # Arrange
    svc = _make_service()
    payload = _add_key_payload(user_public_keys=[])
    executor_info = make_executor_info(payload)

    # Act
    result = await svc.add_ssh_key(payload, executor_info, _keypair(), "encrypted")

    # Assert
    # WHY (D14): the only NoSshKeys site on the add path — byte-frozen msg
    # template, no failure_step, and the early return opens no docker connection.
    assert isinstance(result, FailedContainerRequest)
    assert result.error_type is FailedContainerErrorTypes.AddSSkeyFailed
    assert result.error_code is FailedContainerErrorCodes.NoSshKeys
    assert result.msg == "ssh key Add error: no public key"
    assert result.failure_step is None
    assert svc.rental_docker_client_factory.connect_calls == []


@pytest.mark.asyncio
async def test_remove_ssh_keys_no_keys() -> None:
    # Arrange
    svc = _make_service()
    payload = _remove_keys_payload(user_public_keys=[])
    executor_info = make_executor_info(payload)

    # Act
    result = await svc.remove_ssh_keys(payload, executor_info, _keypair(), "encrypted")

    # Assert
    # WHY (D15): the remove path REUSES the AddSSkeyFailed error type with its
    # own byte-frozen "Remove" msg template; no docker connection is opened.
    assert isinstance(result, FailedContainerRequest)
    assert result.error_type is FailedContainerErrorTypes.AddSSkeyFailed
    assert result.error_code is FailedContainerErrorCodes.NoSshKeys
    assert result.msg == "ssh key Remove error: no public key"
    assert result.failure_step is None
    assert svc.rental_docker_client_factory.connect_calls == []


# --- D16 / D17: generic-exception funnels ------------------------------------


@pytest.mark.asyncio
async def test_add_ssh_key_generic_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    svc = _make_service()
    monkeypatch.setattr(
        svc,
        "add_ssh_public_keys_with_rental_docker",
        AsyncMock(side_effect=RuntimeError("exec exploded")),
    )
    payload = _add_key_payload()
    executor_info = make_executor_info(payload)

    # Act
    result = await svc.add_ssh_key(payload, executor_info, _keypair(), "encrypted")

    # Assert
    # WHY (D16): any exception past the guards funnels to one byte-frozen shape
    # ("Failed add_ssh_key" + UnknownError) with the no-failure_step asymmetry
    # that distinguishes the ssh-key methods from create_container.
    assert isinstance(result, FailedContainerRequest)
    assert result.error_type is FailedContainerErrorTypes.AddSSkeyFailed
    assert result.error_code is FailedContainerErrorCodes.UnknownError
    assert result.msg == "Failed add_ssh_key"
    assert result.failure_step is None


@pytest.mark.asyncio
async def test_remove_ssh_keys_generic_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    svc = _make_service()
    monkeypatch.setattr(
        svc,
        "remove_ssh_public_keys_with_rental_docker",
        AsyncMock(side_effect=RuntimeError("exec exploded")),
    )
    payload = _remove_keys_payload()
    executor_info = make_executor_info(payload)

    # Act
    result = await svc.remove_ssh_keys(payload, executor_info, _keypair(), "encrypted")

    # Assert
    # WHY (D17): the remove funnel uses a DIFFERENT template family than add
    # ("Unknown Error remove_ssh_keys", delete_container-style) — freeze the
    # asymmetry so extraction cannot silently unify the two.
    assert isinstance(result, FailedContainerRequest)
    assert result.error_type is FailedContainerErrorTypes.AddSSkeyFailed
    assert result.error_code is FailedContainerErrorCodes.UnknownError
    assert result.msg == "Unknown Error remove_ssh_keys"
    assert result.failure_step is None


# --- attestation variants (D11-style, per DRAFT §D.2 parenthetical) -----------


@pytest.mark.parametrize(
    ("method_name", "payload_factory"),
    [
        ("add_ssh_key", _add_key_payload),
        ("remove_ssh_keys", _remove_keys_payload),
    ],
)
@pytest.mark.asyncio
async def test_ssh_key_methods_attestation_error_returns_addsskeyfailed(
    method_name: str,
    payload_factory: Callable[..., AddSshPublicKeyRequest | RemoveSshPublicKeysRequest],
) -> None:
    # Arrange
    svc = _make_service()
    svc.attestation_service.prepare_host_policy = AsyncMock(
        side_effect=AttestationError("attestation rejected")
    )
    payload = payload_factory()
    executor_info = make_executor_info(payload)

    # Act
    result = await getattr(svc, method_name)(payload, executor_info, _keypair(), "encrypted")

    # Assert
    # WHY: attestation gates BOTH ssh-key methods with one frozen failure shape —
    # error_code is UnknownError (NOT the AttestationError enum member) and the
    # byte-frozen "Attestation failed" msg — before any docker connection opens.
    assert isinstance(result, FailedContainerRequest)
    assert result.error_type is FailedContainerErrorTypes.AddSSkeyFailed
    assert result.error_code is FailedContainerErrorCodes.UnknownError
    assert result.msg == "Attestation failed"
    assert result.failure_step is None
    assert svc.rental_docker_client_factory.connect_calls == []


# --- D18-D20: install_jupyter_server (zero tests before this file) ------------


@pytest.mark.asyncio
async def test_install_jupyter_success_url_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    svc = _make_service()
    patch_rental_ssh_connect(monkeypatch)
    run_jupyter_mock = AsyncMock()
    monkeypatch.setattr(svc, "run_jupyter", run_jupyter_mock)
    payload = _jupyter_payload(jupyter_port_map=(8888, 30888))
    executor_info = make_executor_info(payload)

    # Act
    result = await svc.install_jupyter_server(payload, executor_info, _keypair(), "encrypted")

    # Assert
    # WHY (D18): the URL is built from the executor ADDRESS + the EXTERNAL port
    # (jupyter_port_map[1]) while run_jupyter receives the INTERNAL one
    # (jupyter_port_map[0]); the token is one fresh secrets.token_hex(16) —
    # 32 lowercase hex chars — shared between the install and the URL.
    assert isinstance(result, JupyterServerInstalled)
    url_prefix = "http://127.0.0.1:30888/lab?token="
    assert result.jupyter_url.startswith(url_prefix)
    token = result.jupyter_url.removeprefix(url_prefix)
    assert len(token) == 32
    assert set(token) <= _HEX_DIGITS
    assert run_jupyter_mock.await_count == 1
    assert run_jupyter_mock.await_args.kwargs["jupyter_token"] == token
    assert run_jupyter_mock.await_args.kwargs["jupyter_port"] == 8888


@pytest.mark.asyncio
async def test_install_jupyter_attestation_returns_failed_container_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    svc = _make_service()
    svc.attestation_service.prepare_host_policy = AsyncMock(
        side_effect=AttestationError("attestation rejected")
    )
    _, connect_mock = patch_rental_ssh_connect(monkeypatch)
    payload = _jupyter_payload()
    executor_info = make_executor_info(payload)

    # Act
    result = await svc.install_jupyter_server(payload, executor_info, _keypair(), "encrypted")

    # Assert
    # WHY (D19): the SURPRISING contract — the jupyter attestation branch reports
    # error_type=ContainerCreationFailed (not a jupyter-specific type), with
    # UnknownError and the byte-frozen "Attestation failed" msg, and never
    # opens the SSH connection.
    assert isinstance(result, FailedContainerRequest)
    assert result.error_type is FailedContainerErrorTypes.ContainerCreationFailed
    assert result.error_code is FailedContainerErrorCodes.UnknownError
    assert result.msg == "Attestation failed"
    assert result.failure_step is None
    assert connect_mock.call_count == 0


@pytest.mark.asyncio
async def test_install_jupyter_runtime_exception_returns_jupyter_installation_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    svc = _make_service()
    patch_rental_ssh_connect(monkeypatch)
    monkeypatch.setattr(
        svc, "run_jupyter", AsyncMock(side_effect=RuntimeError("jupyter exploded"))
    )
    payload = _jupyter_payload()
    executor_info = make_executor_info(payload)

    # Act
    result = await svc.install_jupyter_server(payload, executor_info, _keypair(), "encrypted")

    # Assert
    # WHY (D20): the two-shape failure asymmetry — runtime failures return
    # JupyterInstallationFailed with ONLY a byte-frozen msg; the model carries
    # no error_type/error_code/failure_step fields at all (unlike D19's
    # FailedContainerRequest on the attestation branch).
    assert isinstance(result, JupyterInstallationFailed)
    assert result.msg == "Failed install jupyter server"
    assert not hasattr(result, "error_type")
    assert not hasattr(result, "error_code")
    assert not hasattr(result, "failure_step")
