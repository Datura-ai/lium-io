"""DAH-3480: a GPU probe that cannot allocate because a pod or filler took the cards after the
cycle's rental snapshot is not scored as GPU_VERIFY_FAILED.

Every failure result below is the prod transcript of B-175 (ticket-0324, 14 Sep 2026 12:33Z):
the wrapper printed `UUID:  None`, stderr carried `Failed to allocate d_A: out of memory`, and the
service reported a UUID mismatch against 'None'.
"""

import pytest

from neurons.validators.src.services.matrix_validation_service import ValidationResult
from neurons.validators.src.services.task.checks.capability import CapabilityCheck
from neurons.validators.src.services.task.messages import CapabilityMessages as Msg
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod

from tests.helpers import build_services, build_state

EXECUTOR = "executor-123"  # tests.helpers.default_executor().uuid
EXPECTED_UUID = "ab50d08a-bdc9-4b58-b977-e3ecda9edada"
OOM_STDERR = "Failed to allocate d_A: out of memory"


class ProbeReturning:
    def __init__(self, result: ValidationResult | None, *, raises: Exception | None = None):
        self.result = result
        self.raises = raises
        self.calls = 0

    async def validate_gpu_model_and_process_job(self, **_kwargs):
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        return self.result


def allocation_failure() -> ValidationResult:
    return ValidationResult(
        success=False,
        expected_uuid=EXPECTED_UUID,
        returned_uuid="None",
        stdout='UUID:  None\nRESULT_JSON: {"uuid": null, "metrics": {}, "sealed": null}',
        stderr=OOM_STDERR,
        error_message=f"UUID mismatch: expected '{EXPECTED_UUID}', got 'None'",
    )


def wrong_uuid_answer() -> ValidationResult:
    return ValidationResult(
        success=False,
        expected_uuid=EXPECTED_UUID,
        returned_uuid="11111111-2222-3333-4444-555555555555",
        stdout="UUID:  11111111-2222-3333-4444-555555555555",
        stderr=OOM_STDERR,
        error_message=f"UUID mismatch: expected '{EXPECTED_UUID}', got '11111111-2222-3333-4444-555555555555'",
    )


def timeout() -> ValidationResult:
    return ValidationResult(
        success=False,
        expected_uuid=EXPECTED_UUID,
        error_message="Matrix multiplication timed out after 120s",
        timed_out=True,
    )


def backend_view(*, filler: str | None = None, pod: str | None = None) -> RentedExecutorsResponse:
    """What the backend reports for EXECUTOR when asked again after the probe failed."""
    executors = {}
    if pod:
        executors[EXECUTOR] = RentedExecutor(
            miner_hotkey="miner-hotkey",
            executor_ip_address="203.0.113.10",
            executor_ip_port="22",
            pods=[RentedPod(pod_id="pod-1", container_name=pod, rented_ports=[8080])],
        )
    return RentedExecutorsResponse(
        executors=executors,
        all_filler_containers_by_executor={EXECUTOR: [filler]} if filler else {},
    )


def run_check(context_factory, *, probe: ProbeReturning, fresh: RentedExecutorsResponse | None):
    services = build_services(validation=probe)
    services.backend.get_rented_executors_now.return_value = fresh
    # The cycle snapshot knows nothing about this node: the workload arrived after it was taken.
    state = build_state(specs={"gpu": {"count": 8}}, rented_data=RentedExecutorsResponse(executors={}))
    ctx = context_factory(services=services, state=state)
    return CapabilityCheck().run(ctx), services.backend


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fresh,workload,containers",
    [
        (backend_view(filler="filler_run-1"), "filler", ["filler_run-1"]),
        (backend_view(pod="pod_50c3d835"), "pod", ["pod_50c3d835"]),
    ],
)
async def test_allocation_failure_is_not_scored_when_the_backend_reports_a_workload(
    context_factory, fresh, workload, containers
):
    probe = ProbeReturning(allocation_failure())
    pending, backend = run_check(context_factory, probe=probe, fresh=fresh)
    result = await pending

    assert result.passed is True
    assert result.event.reason_code == Msg.RENTED_SKIPPED.reason
    assert result.event.what_we_saw["workload"] == workload
    assert result.event.what_we_saw["containers"] == containers
    # The probe did run; its answer travels with the event so the row stays explainable.
    assert probe.calls == 1
    assert result.event.what_we_saw["probe"]["stderr"] == OOM_STDERR
    backend.get_rented_executors_now.assert_awaited_once()
    # The fresh read replaces the cycle snapshot: the checks after this one (rental verification,
    # the GPU fault probe) read `ctx.state.rented_data` and would otherwise keep the stale one.
    assert result.updates["state"].rented_data is fresh


@pytest.mark.asyncio
async def test_timeout_is_not_scored_when_the_backend_reports_a_workload(context_factory):
    pending, _ = run_check(
        context_factory, probe=ProbeReturning(timeout()), fresh=backend_view(pod="pod_50c3d835")
    )
    result = await pending

    assert result.passed is True
    assert result.event.reason_code == Msg.RENTED_SKIPPED.reason
    assert result.event.what_we_saw["probe"]["error"] == "Matrix multiplication timed out after 120s"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fresh",
    [
        pytest.param(backend_view(), id="idle: the backend reports nothing on the node"),
        pytest.param(None, id="backend did not answer"),
    ],
)
async def test_allocation_failure_stays_failed_without_a_backend_workload(context_factory, fresh):
    pending, backend = run_check(context_factory, probe=ProbeReturning(allocation_failure()), fresh=fresh)
    result = await pending

    assert result.passed is False
    assert result.event.reason_code == Msg.VERIFY_FAILED.reason
    assert result.event.what_we_saw["stderr"] == OOM_STDERR
    backend.get_rented_executors_now.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_backend_read_that_raises_keeps_the_failure(context_factory):
    probe = ProbeReturning(allocation_failure())
    services = build_services(validation=probe)
    services.backend.get_rented_executors_now.side_effect = RuntimeError("backend down")
    state = build_state(specs={"gpu": {"count": 8}}, rented_data=RentedExecutorsResponse(executors={}))

    result = await CapabilityCheck().run(context_factory(services=services, state=state))

    assert result.passed is False
    assert result.event.reason_code == Msg.VERIFY_FAILED.reason


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "probe",
    [
        pytest.param(ProbeReturning(wrong_uuid_answer()), id="a returned UUID that does not match"),
        pytest.param(ProbeReturning(None, raises=RuntimeError("ssh dropped")), id="the probe raised"),
    ],
)
async def test_only_a_probe_with_no_answer_is_waived(context_factory, probe):
    pending, backend = run_check(context_factory, probe=probe, fresh=backend_view(pod="pod_50c3d835"))
    result = await pending

    assert result.passed is False
    assert result.event.reason_code == Msg.VERIFY_FAILED.reason
    backend.get_rented_executors_now.assert_not_awaited()
