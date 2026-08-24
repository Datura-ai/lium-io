"""DAH-2734: provider-side CPU/disk gate — the twin of the DAH-2735 foreign-GPU gate."""
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from neurons.validators.src.protocol.vc_protocol.compute_requests import (
    RentedExecutor,
    RentedExecutorsResponse,
    RentedPod,
)
from neurons.validators.src.services.task.checks.provider_side_load import (
    ProviderSideLoad,
    ProviderSideLoadCheck,
    compute_provider_side_load,
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


def test_provider_side_load_attributes_cpu_and_disk():
    load = compute_provider_side_load(SN13_SPECS, {"pod_renter"})
    assert load.cpu_cores == 9.0  # 33% * 32 cores - 1.58 cores
    assert load.disk_kb == 1528 * GB_KB  # 1700 - (100 + 12 + 60)


def test_provider_side_load_clamps_negative_to_zero():
    load = compute_provider_side_load(
        {
            "cpu": {"count": 4},
            # 0.4 host cores over the stats window; container jitter above host clamps to zero
            "docker": {
                "host_cpu_percent": 10.0,
                "containers": [{"name": "pod_a", "cpu_percent": 90.0}],
            },
            "hard_disk": {"used": 10 * GB_KB, "images": 20 * GB_KB, "containers": 0, "volumes": 0},
        },
        {"pod_a"},
    )
    assert load == ProviderSideLoad(cpu_cores=0.0, disk_kb=0)


def test_provider_side_load_skips_parts_with_missing_or_bad_inputs():
    # no docker stats, no disk breakdown, and a non-numeric core count -> no signal, never a guess
    assert compute_provider_side_load(
        {
            "cpu": {"count": "thirty-two"},
            "docker": {"host_cpu_percent": 33.0, "containers": [{"name": "pod_x"}]},
            "hard_disk": {"used": 1700 * GB_KB},
        },
        {"pod_x"},
    ).is_measured is False


def test_provider_side_load_requires_cpu_percent_on_every_container():
    # a partial `docker stats` result would attribute the uncovered container's CPU to the
    # provider — one container without cpu_percent must void the CPU part, not skew it
    load = compute_provider_side_load(
        {
            "cpu": {"count": 32},
            "docker": {
                "host_cpu_percent": 33.0,
                "containers": [{"name": "pod_renter", "cpu_percent": 158.0}, {"name": "no_stats_row"}],
            },
        },
        {"pod_renter", "no_stats_row"},
    )
    assert load.cpu_cores is None


def test_a_foreign_container_counts_against_the_provider():
    # The evasion the GPU twin already blocks: the provider runs the other subnet with
    # `docker run`, so its CPU would be subtracted as if Lium put it there.
    load = compute_provider_side_load(
        {
            "cpu": {"count": 32},
            "docker": {
                "container_id": "executor-container-id",
                "host_cpu_percent": 33.0,
                "containers": [
                    {"name": "pod_renter", "cpu_percent": 158.0},
                    {"name": "sn13_miner", "cpu_percent": 700.0},
                ],
            },
        },
        {"pod_renter"},
    )
    assert load.cpu_cores == 9.0  # the miner's 7 cores stay on the provider's side


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
    ctx = context_factory(state=_rented_state(SN13_SPECS))

    with provider_load_gate(enforce=True):
        result = await ProviderSideLoadCheck().run(ctx)

    assert result.passed is False
    assert result.updates["provider_side_load_passed"] is False
    assert result.event.reason_code == Msg.LOAD_ABOVE_LIMIT.reason
    assert result.event.what_we_saw["provider_cpu_cores"] == 9.0
    assert result.updates["state"].specs["provider_side_load"] == {
        "cpu_cores": 9.0,
        "disk_kb": 1528 * GB_KB,
    }


@pytest.mark.asyncio
async def test_idle_machine_is_judged_the_same_as_a_rented_one(context_factory):
    # The provider cheats the renter when rented and the marketplace when idle — same verdict.
    ctx = context_factory(state=build_state(specs=SN13_SPECS))

    with provider_load_gate(enforce=True):
        result = await ProviderSideLoadCheck().run(ctx)

    assert result.passed is False
    assert result.event.what_we_saw["is_rented"] is False


@pytest.mark.asyncio
async def test_shadow_logs_the_verdict_and_keeps_the_score(context_factory):
    ctx = context_factory(state=_rented_state(SN13_SPECS))

    with provider_load_gate(enforce=False):
        result = await ProviderSideLoadCheck().run(ctx)

    assert result.passed is True
    assert "provider_side_load_passed" not in result.updates
    assert result.event.reason_code == Msg.LOAD_ABOVE_LIMIT.reason
    assert result.event.severity == "warning"


@pytest.mark.asyncio
async def test_load_below_the_limits_passes(context_factory):
    # A provider's own nginx: 0.4 cores and an OS baseline of 30 GB is not cheating.
    specs = {
        "cpu": {"count": 32},
        "docker": {"host_cpu_percent": 2.0, "containers": [{"cpu_percent": 24.0}]},
        "hard_disk": {"used": 40 * GB_KB, "images": 8 * GB_KB, "containers": 1 * GB_KB, "volumes": 1 * GB_KB},
    }
    ctx = context_factory(state=_rented_state(specs))

    with provider_load_gate(enforce=True):
        result = await ProviderSideLoadCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.LOAD_OK.reason


@pytest.mark.asyncio
async def test_unmeasurable_signal_never_withholds_money(context_factory):
    ctx = context_factory(state=build_state(specs={"gpu": {"count": 1}}))

    with provider_load_gate(enforce=True):
        result = await ProviderSideLoadCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.NOT_MEASURABLE.reason
    assert result.updates == {}


@pytest.mark.asyncio
async def test_disabled_check_skips_entirely(context_factory):
    ctx = context_factory(state=_rented_state(SN13_SPECS))

    with provider_load_gate(enforce=True, check_enabled=False):
        result = await ProviderSideLoadCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.SKIPPED.reason


@pytest.mark.asyncio
async def test_the_executor_container_is_subtracted_without_being_rented(context_factory):
    # Lium's own agent is on no rental list, but Lium put it there: the check has to find it
    # through docker.container_id, or every idle node reads as a cheating one.
    specs = {
        "cpu": {"count": 32},
        "docker": {
            "container_id": "executor-container-id",
            "host_cpu_percent": 10.0,
            "containers": [{"container_id": "executor-container-id", "cpu_percent": 300.0}],
        },
    }
    ctx = context_factory(state=build_state(specs=specs))

    with provider_load_gate(enforce=True):
        result = await ProviderSideLoadCheck().run(ctx)

    assert result.passed is True
    assert result.event.what_we_saw["provider_cpu_cores"] == 0.2  # 3.2 host cores - 3.0 ours


@pytest.mark.asyncio
async def test_a_foreign_container_zeroes_the_score_of_a_rented_machine(context_factory):
    # The whole point of the ticket, end to end: the provider mines another subnet with
    # `docker run` while a renter holds the pod.
    specs = {
        **SN13_SPECS,
        "docker": {
            **SN13_SPECS["docker"],
            "containers": [
                {"name": "pod_renter", "cpu_percent": 158.0},
                {"name": "sn13_miner", "cpu_percent": 700.0},
            ],
        },
    }
    ctx = context_factory(state=_rented_state(specs))

    with provider_load_gate(enforce=True):
        result = await ProviderSideLoadCheck().run(ctx)

    assert result.passed is False
    assert result.updates["provider_side_load_passed"] is False
    assert result.event.what_we_saw["provider_cpu_cores"] == 9.0
