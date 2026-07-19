"""Characterization oracle — delete_container flow (DAH-2382 PR-1a, group D-DELETE).

Pins the observable delete boundary on baseline 6be5649f: the exact
ContainerDeleted / FailedContainerRequest wire shapes, the FILLER-only
wedge-sweep call-site wiring (DAH-2427), and the best-effort post-teardown
log contract. The sweep INTERNALS are covered by
tests/test_gpu_wedge_teardown_sweep.py — here the module-level
``_sweep_wedged_gpus_after_teardown`` seam is always patched so only the
delete_container wiring is under test.

The delete scaffold (make_delete_request / DeleteContainerHarness /
patch_delete_happy_path and the redis delete-path surface) lives in the shared
docker_oracle/harness.py.
"""

import logging
from unittest.mock import Mock

import pytest
from docker_oracle.harness import (
    make_delete_request,
    make_docker_service,
    make_executor_info,
    patch_delete_happy_path,
)
from payload_models.payloads import (
    ContainerDeleted,
    ContainerResponseType,
    FailedContainerErrorCodes,
    FailedContainerErrorTypes,
    FailedContainerRequest,
    WorkloadKind,
)
from services.attestation_service import AttestationError

_POST_TEARDOWN_FAILED_MSG = "delete_container post-teardown step failed (non-fatal)"
_FUNNEL_MSG_PREFIX = "Unknown Error delete_container: "


@pytest.mark.asyncio
async def test_delete_success_returns_container_deleted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    svc = make_docker_service()
    patch_delete_happy_path(svc, monkeypatch)
    payload = make_delete_request()
    executor_info = make_executor_info(payload)

    # Act
    result = await svc.delete_container(
        payload=payload,
        executor_info=executor_info,
        keypair=Mock(ss58_address="validator-hotkey"),
        private_key="encrypted",
    )

    # Assert
    # WHY: the success ack is the exact backend wire object — pin the full
    # field set so extraction can neither add, drop, nor rename a field
    # (ContainerDeleted carries NO container_name / failure_step / msg at all).
    assert isinstance(result, ContainerDeleted)
    assert result.model_dump() == {
        "message_type": ContainerResponseType.ContainerDeleted,
        "miner_hotkey": "miner",
        "executor_id": payload.executor_id,
        "pod_id": payload.pod_id,
        "workload_kind": WorkloadKind.CUSTOMER_RENTAL,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("step", "workload_kind", "enable_inspector"),
    [
        ("restore_filler_gpu_power", WorkloadKind.FILLER, False),
        ("sweep_wedged_gpus", WorkloadKind.FILLER, False),
        ("inspector_stop", WorkloadKind.CUSTOMER_RENTAL, True),
    ],
)
async def test_delete_teardown_step_failure_nonfatal(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    step: str,
    workload_kind: WorkloadKind,
    enable_inspector: bool,
) -> None:
    # Arrange: fail exactly one post-teardown step after the forced removal succeeded
    # (these three step failure sites are the ones no legacy delete test exercises
    # with a log assert; prune/volumes/remove_rented_machine live in
    # tests/test_docker_service.py:1286-1553)
    svc = make_docker_service()
    harness = patch_delete_happy_path(svc, monkeypatch, enable_inspector=enable_inspector)
    boom = Exception(f"{step} boom")
    if step == "restore_filler_gpu_power":
        harness.mocks["restore_filler_pod_gpu_power_limits"].side_effect = boom
    elif step == "sweep_wedged_gpus":
        harness.mocks["_sweep_wedged_gpus_after_teardown"].side_effect = boom
    else:
        harness.redis.get_rented_machine_error = boom
    payload = make_delete_request(workload_kind=workload_kind)
    executor_info = make_executor_info(payload)

    # Act
    with caplog.at_level(logging.INFO, logger="services.docker_service"):
        result = await svc.delete_container(
            payload=payload,
            executor_info=executor_info,
            keypair=Mock(ss58_address="validator-hotkey"),
            private_key="encrypted",
        )

    # Assert
    # WHY: after the forced removal the container is gone — a failing teardown
    # step must never fail the undeploy, or the backend retries a doomed
    # request and penalizes the miner.
    assert isinstance(result, ContainerDeleted)
    # WHY: the non-fatal funnel is one ERROR line with a step= key — the only
    # observable trace of which best-effort step broke (Loki triage surface).
    records = [r for r in caplog.records if str(r.msg) == _POST_TEARDOWN_FAILED_MSG]
    assert [r.levelno for r in records] == [logging.ERROR]
    assert records[0].msg.extra["step"] == step
    assert records[0].msg.extra["error"] == f"{step} boom"


@pytest.mark.asyncio
async def test_delete_filler_sweeps_wedged_gpus_on_success_nonfatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange: FILLER success path; the patched sweep itself blows up
    svc = make_docker_service()
    harness = patch_delete_happy_path(svc, monkeypatch)
    sweep = harness.mocks["_sweep_wedged_gpus_after_teardown"]
    sweep.side_effect = Exception("sweep boom")
    payload = make_delete_request(workload_kind=WorkloadKind.FILLER)
    executor_info = make_executor_info(payload)

    # Act
    result = await svc.delete_container(
        payload=payload,
        executor_info=executor_info,
        keypair=Mock(ss58_address="validator-hotkey"),
        private_key="encrypted",
    )

    # Assert
    # WHY: DAH-2427 — the success-path ghost-GPU sweep runs exactly once per
    # FILLER teardown, over the delete's own SSH session, and is best-effort:
    # a failing cure must not fail an undeploy whose container is already gone.
    assert isinstance(result, ContainerDeleted)
    assert sweep.await_count == 1
    assert sweep.await_args.args[0] is harness.ssh_client


@pytest.mark.asyncio
async def test_delete_filler_sweeps_before_propagating_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange: the fatal force-remove fails on a FILLER teardown
    svc = make_docker_service()
    harness = patch_delete_happy_path(svc, monkeypatch)
    harness.docker_client.remove_error = Exception(
        "Docker SDK remove container failed: 500 Server Error: daemon exploded"
    )
    payload = make_delete_request(workload_kind=WorkloadKind.FILLER)
    executor_info = make_executor_info(payload)

    # Act
    result = await svc.delete_container(
        payload=payload,
        executor_info=executor_info,
        keypair=Mock(ss58_address="validator-hotkey"),
        private_key="encrypted",
    )

    # Assert
    # WHY: DAH-2427 — a failed force-remove is the classic wedge path; the
    # sweep must run exactly once, AFTER the failed remove and BEFORE the
    # failure result, so a wedged card cannot outlive the failed delete.
    assert harness.mocks["_sweep_wedged_gpus_after_teardown"].await_count == 1
    journal_names = [name for name, _ in harness.journal]
    assert journal_names.index("docker.remove_container") < journal_names.index(
        "sweep.teardown_sweep"
    )
    # WHY: the remove failure stays fatal — the sweep is squeezed in but the
    # undeploy still reports the frozen funnel shape and no later teardown
    # step (prune / redis cleanup) runs.
    assert isinstance(result, FailedContainerRequest)
    assert result.error_type == FailedContainerErrorTypes.ContainerDeletionFailed
    assert result.error_code == FailedContainerErrorCodes.UnknownError
    assert result.msg.startswith(_FUNNEL_MSG_PREFIX)
    assert result.workload_kind == WorkloadKind.FILLER
    assert harness.docker_client.pruned_images == 0
    assert "remove_rented_machine" not in [name for name, _ in harness.redis.calls]


@pytest.mark.asyncio
async def test_delete_customer_rental_does_not_sweep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange: successful CUSTOMER_RENTAL teardown
    svc = make_docker_service()
    harness = patch_delete_happy_path(svc, monkeypatch)
    payload = make_delete_request(workload_kind=WorkloadKind.CUSTOMER_RENTAL)
    executor_info = make_executor_info(payload)

    # Act
    result = await svc.delete_container(
        payload=payload,
        executor_info=executor_info,
        keypair=Mock(ss58_address="validator-hotkey"),
        private_key="encrypted",
    )

    # Assert
    # WHY: the DAH-2427 sweep and the DAH-2356 power restore share ONE
    # workload gate — a customer teardown must never pay the FILLER-only steps.
    assert isinstance(result, ContainerDeleted)
    assert harness.mocks["_sweep_wedged_gpus_after_teardown"].await_count == 0
    assert harness.mocks["restore_filler_pod_gpu_power_limits"].await_count == 0


@pytest.mark.asyncio
async def test_delete_unsafe_volume_name_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange: a local_volume with shell metacharacters must die at the guard
    svc = make_docker_service()
    harness = patch_delete_happy_path(svc, monkeypatch)
    payload = make_delete_request(local_volume="vol;rm -rf")
    executor_info = make_executor_info(payload)

    # Act
    result = await svc.delete_container(
        payload=payload,
        executor_info=executor_info,
        keypair=Mock(ss58_address="validator-hotkey"),
        private_key="encrypted",
    )

    # Assert
    # WHY: the volume-name guard is the delete path's injection boundary — its
    # typed failure is byte-frozen and fires before any credential decrypt,
    # SSH connect, or docker SDK call.
    assert isinstance(result, FailedContainerRequest)
    assert result.msg == "Invalid Docker volume name"
    assert result.error_type == FailedContainerErrorTypes.ContainerDeletionFailed
    assert result.error_code == FailedContainerErrorCodes.UnknownError
    assert result.failure_step is None
    assert harness.docker_client.calls == []
    harness.mocks["asyncssh.connect"].assert_not_called()
    svc.ssh_service.decrypt_payload.assert_not_called()


@pytest.mark.asyncio
async def test_delete_attestation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange: the attestation service rejects the executor; the REAL
    # _prepare_known_hosts_policy chokepoint re-raises AttestationError as-is
    svc = make_docker_service()
    harness = patch_delete_happy_path(svc, monkeypatch)
    harness.mocks["prepare_host_policy"].side_effect = AttestationError(
        "host key mismatch for executor"
    )
    payload = make_delete_request()
    executor_info = make_executor_info(payload)

    # Act
    result = await svc.delete_container(
        payload=payload,
        executor_info=executor_info,
        keypair=Mock(ss58_address="validator-hotkey"),
        private_key="encrypted",
    )

    # Assert
    # WHY: attestation failure maps to the generic delete-failure shape — the
    # frozen "Attestation failed" msg with error_code UnknownError (NOT the
    # AttestationError enum member) and no failure_step — and no SSH session
    # or docker SDK call is ever opened.
    assert isinstance(result, FailedContainerRequest)
    assert result.msg == "Attestation failed"
    assert result.error_type == FailedContainerErrorTypes.ContainerDeletionFailed
    assert result.error_code == FailedContainerErrorCodes.UnknownError
    assert result.failure_step is None
    assert harness.docker_client.calls == []
    harness.mocks["asyncssh.connect"].assert_not_called()
