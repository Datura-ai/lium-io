"""DAH-3480 follow-up: the mid-cycle GPU-verify waiver is not granted for the provider's own rental.

`_lium_workload_live_now` (lium-io#1368) waives a failed probe when the backend lists a pod on the
node. lium-platform#432 marks a self-rented pod with `RentedExecutor.owner_flag=True`. Such a pod
pays no incentive and must not skip the probe either. Without this change the check reads the
pod and waives regardless of the flag.

The payload tests go through `RentedExecutorsResponse.model_validate` on the JSON shape
`GET /internal/executors/rented` returns, so the key the backend writes is the key the check reads.
"""

import pytest

from neurons.validators.src.services.task.messages import CapabilityMessages as Msg
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse

from tests.test_capability_check_mid_cycle_workload import (
    EXECUTOR,
    ProbeReturning,
    allocation_failure,
    backend_view,
    run_check,
    timeout,
)


def rented_payload(*, owner_flag: bool | None) -> dict:
    """The JSON of `/internal/executors/rented` for one pod on EXECUTOR.

    `owner_flag=None` leaves the key out, as a backend without lium-platform#432 does.
    """
    executor = {
        "miner_hotkey": "miner-hotkey",
        "executor_ip_address": "203.0.113.10",
        "executor_ip_port": "22",
        "pods": [{"pod_id": "pod-1", "container_name": "pod_50c3d835", "rented_ports": [8080]}],
    }
    if owner_flag is not None:
        executor["owner_flag"] = owner_flag
    return {"executors": {EXECUTOR: executor}}


def owner_view() -> RentedExecutorsResponse:
    view = backend_view(pod="pod_50c3d835")
    view.executors[EXECUTOR].owner_flag = True
    return view


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "probe,expected_reason",
    [
        pytest.param(allocation_failure(), Msg.VERIFY_FAILED.reason, id="allocation failure"),
        pytest.param(timeout(), Msg.VERIFY_TIMEOUT.reason, id="timeout"),
    ],
)
async def test_the_providers_own_rental_does_not_waive_the_probe(context_factory, probe, expected_reason):
    pending, backend = run_check(context_factory, probe=ProbeReturning(probe), fresh=owner_view())
    result = await pending

    # The probe's own verdict stands, with the probe's own code and output.
    assert result.passed is False
    assert result.event.reason_code == expected_reason
    assert result.event.what_we_saw["error"] == probe.error_message
    # The backend was asked and answered; the answer was "your own pod", which is not a waiver.
    backend.get_rented_executors_now.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "owner_flag",
    [
        pytest.param(False, id="backend says customer"),
        pytest.param(None, id="backend predates the field"),
    ],
)
async def test_a_customer_pod_from_the_backend_payload_waives_as_before(context_factory, owner_flag):
    fresh = RentedExecutorsResponse.model_validate(rented_payload(owner_flag=owner_flag))
    assert fresh.executors[EXECUTOR].owner_flag is False

    pending, _ = run_check(context_factory, probe=ProbeReturning(allocation_failure()), fresh=fresh)
    result = await pending

    assert result.passed is True
    assert result.event.reason_code == Msg.RENTED_SKIPPED.reason
    assert result.event.what_we_saw["workload"] == "pod"


@pytest.mark.asyncio
async def test_owner_flag_true_in_the_backend_payload_reaches_the_check(context_factory):
    fresh = RentedExecutorsResponse.model_validate(rented_payload(owner_flag=True))
    assert fresh.executors[EXECUTOR].owner_flag is True

    pending, _ = run_check(context_factory, probe=ProbeReturning(allocation_failure()), fresh=fresh)
    result = await pending

    assert result.passed is False
    assert result.event.reason_code == Msg.VERIFY_FAILED.reason
