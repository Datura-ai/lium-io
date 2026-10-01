"""DAH-3980: the validator's create of a pre-started ("warm") pod.

A warm pod is the default-template container started while the machine is idle, with no renter
yet; an eligible rent adopts it later. Its create is the normal one with these differences: no
keys, fillers keep running, a 1 GB volume, the label `lium.warm_pod=<pod_id>`, no GPU power
restore (a PEARL filler may run beside it), no rented-pod cache entry, its pending mark cleared on
success, and a refusal when a renter's `pod_*` is on the host. The env marker `LIUM_WARM_POD=1` on a
keyless CUSTOMER_RENTAL stands for WorkloadKind.WARM_POD while the backend cannot send it.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest
from payload_models.payloads import (
    ContainerCreated,
    CustomOptions,
    FailedContainerErrorCodes,
    FailedContainerRequest,
    WorkloadKind,
)
from test_deploy_optimizations import (
    _created_run_spec,
    _docker_client,
    _patch_happy,
    _payload,
    _run,
    _ssh_client,
    _ssh_result,
)

import services.docker_service as ds_module
from services.docker_service import (
    WARM_POD_ENV_MARKER,
    WARM_POD_LABEL,
    WARM_POD_VOLUME_GB,
    DockerService,
)


@pytest.fixture
def svc():
    return DockerService(ssh_service=Mock(), redis_service=Mock(), attestation_service=Mock())


def _warm_payload(**over):
    return _payload(workload_kind=WorkloadKind.WARM_POD, user_public_keys=[], **over)


def _marker_payload(**over):
    return _payload(
        user_public_keys=[],
        custom_options=CustomOptions(environment={WARM_POD_ENV_MARKER: "1", "HF_HOME": "/root/hf"}),
        **over,
    )


def _ssh_client_listing_containers(names_and_warm_labels: str):
    ssh_client = _ssh_client(inspect_exit=0)

    def _side(cmd, *args, **kwargs):
        if WARM_POD_LABEL in cmd:
            return _ssh_result(stdout=names_and_warm_labels)
        return _ssh_result()

    ssh_client.run = AsyncMock(side_effect=_side)
    return ssh_client


def _enable_encrypted_volume(svc, monkeypatch) -> AsyncMock:
    monkeypatch.setattr(ds_module.settings, "ENABLE_VOLUME_ENCRYPTION", True)
    monkeypatch.setattr(svc, "_image_has_encrypted_volume_label", AsyncMock(return_value=True))
    setup = AsyncMock()
    monkeypatch.setattr(svc, "setup_encrypted_local_volume", setup)
    return setup


@pytest.mark.asyncio
async def test_warm_create_run_spec_carries_the_warm_pod_label(svc, monkeypatch):
    _patch_happy(svc, monkeypatch, _ssh_client(inspect_exit=0))
    payload = _warm_payload()

    result = await _run(svc, payload)

    assert isinstance(result, ContainerCreated), result
    assert _created_run_spec(svc).labels == {WARM_POD_LABEL: payload.pod_id}
    assert result.workload_kind == WorkloadKind.WARM_POD
    assert result.container_name == f"pod_{payload.pod_id}"
    assert result.volume_name == f"volume_{payload.pod_id}"


@pytest.mark.asyncio
async def test_customer_create_run_spec_has_no_warm_pod_label(svc, monkeypatch):
    _patch_happy(svc, monkeypatch, _ssh_client(inspect_exit=0))

    result = await _run(svc, _payload())

    assert isinstance(result, ContainerCreated), result
    assert _created_run_spec(svc).labels == {}


@pytest.mark.asyncio
async def test_warm_create_sets_up_the_encrypted_volume_with_no_keys(svc, monkeypatch):
    _patch_happy(svc, monkeypatch, _ssh_client(inspect_exit=0))
    setup = _enable_encrypted_volume(svc, monkeypatch)

    result = await _run(svc, _warm_payload(is_sysbox=True, enable_volume_encryption=True))

    assert isinstance(result, ContainerCreated), result
    assert not setup.await_args.kwargs["authorized_keys"]
    assert not any("authorized_keys" in " ".join(spec.argv) for spec in _docker_client(svc).exec_specs)


@pytest.mark.asyncio
async def test_warm_create_on_a_plain_volume_writes_no_keys(svc, monkeypatch):
    _patch_happy(svc, monkeypatch, _ssh_client(inspect_exit=0))

    result = await _run(svc, _warm_payload())

    assert isinstance(result, ContainerCreated), result
    assert not any("authorized_keys" in " ".join(spec.argv) for spec in _docker_client(svc).exec_specs)


@pytest.mark.asyncio
async def test_warm_create_volume_is_one_gb_without_sizing(svc, monkeypatch):
    _patch_happy(svc, monkeypatch, _ssh_client(inspect_exit=0))

    result = await _run(svc, _warm_payload())

    assert isinstance(result, ContainerCreated), result
    assert svc.create_local_volume.await_args.kwargs["limit"] == WARM_POD_VOLUME_GB == 1
    assert result.volume_limit_gb == 1
    svc.resolve_volume_sizing.assert_not_awaited()


@pytest.mark.asyncio
async def test_warm_create_cleanup_keeps_fillers(svc, monkeypatch):
    _patch_happy(svc, monkeypatch, _ssh_client(inspect_exit=0))

    result = await _run(svc, _warm_payload())

    assert isinstance(result, ContainerCreated), result
    assert svc.clean_existing_containers.await_args.kwargs["remove_every_filler"] is False


@pytest.mark.asyncio
async def test_marker_form_is_a_warm_create_and_the_marker_stays_out_of_the_env(svc, monkeypatch):
    _patch_happy(svc, monkeypatch, _ssh_client(inspect_exit=0))
    payload = _marker_payload()

    result = await _run(svc, payload)

    assert isinstance(result, ContainerCreated), result
    run_spec = _created_run_spec(svc)
    assert run_spec.labels == {WARM_POD_LABEL: payload.pod_id}
    assert WARM_POD_ENV_MARKER not in run_spec.environment
    assert run_spec.environment["HF_HOME"] == "/root/hf"
    assert not any(WARM_POD_ENV_MARKER in str(spec) for spec in _docker_client(svc).exec_specs)
    assert svc.clean_existing_containers.await_args.kwargs["remove_every_filler"] is False
    assert result.workload_kind == WorkloadKind.CUSTOMER_RENTAL


@pytest.mark.asyncio
async def test_marker_on_a_renters_create_with_keys_is_stripped_and_ignored(svc, monkeypatch):
    _patch_happy(svc, monkeypatch, _ssh_client(inspect_exit=0))

    result = await _run(
        svc,
        _payload(custom_options=CustomOptions(environment={WARM_POD_ENV_MARKER: "1"})),
    )

    assert isinstance(result, ContainerCreated), result
    run_spec = _created_run_spec(svc)
    assert run_spec.labels == {}
    assert WARM_POD_ENV_MARKER not in run_spec.environment
    assert svc.clean_existing_containers.await_args.kwargs["remove_every_filler"] is True


@pytest.mark.asyncio
async def test_customer_create_with_no_keys_is_still_refused(svc, monkeypatch):
    _patch_happy(svc, monkeypatch, _ssh_client(inspect_exit=0))

    result = await _run(svc, _payload(user_public_keys=[]))

    assert isinstance(result, FailedContainerRequest)
    assert result.error_code == FailedContainerErrorCodes.NoSshKeys
    svc._run_rental_docker_create_with_port_retry.assert_not_awaited()


@pytest.mark.asyncio
async def test_warm_create_clears_its_pending_mark_on_success(svc, monkeypatch):
    # a mark left behind would decline the renter's create of the same id as RentingInProgress
    _patch_happy(svc, monkeypatch, _ssh_client(inspect_exit=0))
    payload = _warm_payload()

    await _run(svc, payload)

    svc.redis_service.remove_pending_pod.assert_awaited_once_with(
        payload.miner_hotkey, payload.executor_id, payload.pod_id
    )


@pytest.mark.asyncio
async def test_warm_create_leaves_gpu_power_and_the_rented_cache_alone(svc, monkeypatch):
    _patch_happy(svc, monkeypatch, _ssh_client(inspect_exit=0))
    power_restore = AsyncMock()
    monkeypatch.setattr(svc, "_restore_gpu_power_for_uncapped_pod", power_restore)

    result = await _run(svc, _warm_payload())

    assert isinstance(result, ContainerCreated), result
    power_restore.assert_not_called()
    svc.redis_service.add_rented_pod.assert_not_awaited()


@pytest.mark.asyncio
async def test_warm_create_is_refused_beside_a_renters_pod(svc, monkeypatch):
    _patch_happy(svc, monkeypatch, _ssh_client_listing_containers("pod_renter \nfiller_x \n"))

    result = await _run(svc, _warm_payload())

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "warm_pod_fence"
    svc.clean_existing_containers.assert_not_awaited()
    svc._run_rental_docker_create_with_port_retry.assert_not_awaited()


@pytest.mark.asyncio
async def test_warm_create_goes_on_beside_an_old_warm_pod_and_fillers(svc, monkeypatch):
    # the old warm pod is not listed by the backend, so the normal sweep removes it
    _patch_happy(svc, monkeypatch, _ssh_client_listing_containers("pod_old old\nfiller_x \n"))

    result = await _run(svc, _warm_payload())

    assert isinstance(result, ContainerCreated), result
