"""DAH-2734: provider-side CPU/disk gate — the twin of the DAH-2735 foreign-GPU gate."""
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neurons.validators.src.protocol.vc_protocol.compute_requests import (
    RentedExecutor,
    RentedExecutorsResponse,
    RentedPod,
)
from neurons.validators.src.services.task.checks.provider_side_load import (
    ProviderSideLoadCheck,
)
from neurons.validators.src.services.task.pipeline import ContextState
from neurons.validators.src.services.task.messages import ProviderSideLoadMessages as Msg

from tests.helpers import build_state

GB_KB = 1024 * 1024

# The SN13 case: 32 threads, host at 33%, the renter's pod at 158% — ~9 cores are the miner's.
SN13_SPECS = {
    "cpu": {"count": 32},
    "docker": {
        "container_id": "executor-container-id",
        "host_cpu_percent": 33.0,
        "containers": [{"name": "pod_renter", "cpu_percent": 158.0}],
    },
    "hard_disk": {
        "used": 1700 * GB_KB,
        "images": 100 * GB_KB,
        "containers": 12 * GB_KB,
        "volumes": 60 * GB_KB,
    },
}


def confirming_runner(
    host_busy_cores: float,
    core_count: int,
    rows: list[tuple[str, str, float]],
    dockerd_cores: float = 0.0,
):
    """A runner whose second reading repeats what the scrape saw — a real load, not a spike."""
    total_jiffies = 100_000
    busy = int(total_jiffies * host_busy_cores / core_count)
    window_seconds = total_jiffies / (100 * core_count)
    dockerd_jiffies = int(dockerd_cores * window_seconds * 100)
    body = "\n".join(f"{container_id}|{name}|{percent:.2f}%" for container_id, name, percent in rows)
    stdout = (
        f"0 0 {core_count} 0\n@@@\n{body}\n@@@\n"
        f"{total_jiffies} {total_jiffies - busy} {core_count} {dockerd_jiffies}"
    )
    runner = AsyncMock()
    runner.run = AsyncMock(return_value=MagicMock(success=True, stdout=stdout))
    return runner


def _no_rentals() -> RentedExecutorsResponse:
    """The backend answered, and this node holds nothing."""
    return RentedExecutorsResponse(executors={}, banned_guids=[])


def _rented_state(specs: dict[str, object]) -> ContextState:
    return build_state(
        specs=specs,
        rented_data=RentedExecutorsResponse(
            executors={
                "executor-123": RentedExecutor(
                    miner_hotkey="miner-hotkey",
                    executor_ip_address="1.2.3.4",
                    executor_ip_port="40022",
                    pods=[RentedPod(pod_id="pod-1", container_name="pod_renter")],
                )
            },
            banned_guids=[],
        ),
    )


@contextmanager
def provider_load_gate(*, enforce: bool = True, check_enabled: bool = True):
    """DAH-2734 ships shadow-first, like every other money-withholding gate."""
    with patch("neurons.validators.src.services.task.checks.provider_side_load.settings") as s:
        s.PROVIDER_SIDE_LOAD_CHECK_ENABLED = check_enabled
        s.PROVIDER_SIDE_LOAD_ENFORCEMENT_ENABLED = enforce
        yield s


@pytest.mark.asyncio
async def test_enforcement_zeroes_the_score_on_a_rented_machine(context_factory):
    ctx = context_factory(
        state=_rented_state(SN13_SPECS),
        runner=confirming_runner(10.56, 32, [("c1", "pod_renter", 158.0)]),
    )

    with provider_load_gate(enforce=True):
        result = await ProviderSideLoadCheck().run(ctx)

    assert result.passed is False
    assert result.updates["provider_side_load_passed"] is False
    assert result.event.reason_code == Msg.LOAD_ABOVE_LIMIT.reason
    assert result.event.what_we_saw["provider_cpu_cores"] == 8.9
    assert result.updates["state"].specs["provider_side_load"] == {
        "cpu_cores": 8.9,
        "disk_kb": 1528 * GB_KB,
    }


@pytest.mark.asyncio
async def test_unmeasurable_signal_never_withholds_money(context_factory):
    ctx = context_factory(state=build_state(specs={"gpu": {"count": 1}}))

    with provider_load_gate(enforce=True):
        result = await ProviderSideLoadCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.NOT_MEASURABLE.reason
    assert result.updates == {}


@pytest.mark.asyncio
async def test_a_forged_rental_name_is_counted_and_reported(context_factory):
    # A name the backend disowns: `pod_fresh` carries no id it ever issued. It counts against
    # the provider, and the shadow week sees it separately.
    specs = {
        "cpu": {"count": 32},
        "docker": {
            "host_cpu_percent": 20.0,
            "containers": [{"name": "pod_fresh", "cpu_percent": 300.0}],
        },
    }
    ctx = context_factory(
        state=build_state(specs=specs, rented_data=_no_rentals()),
        runner=confirming_runner(6.4, 32, [("c1", "pod_fresh", 300.0)]),
    )

    with provider_load_gate(enforce=True):
        result = await ProviderSideLoadCheck().run(ctx)

    assert result.event.what_we_saw["provider_cpu_cores"] == 6.4
    assert result.event.what_we_saw["lium_named_outside_snapshot_cores"] == 3.0


