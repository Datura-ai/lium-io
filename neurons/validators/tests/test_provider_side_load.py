"""DAH-2734: provider-side CPU/disk on a rented machine surfaces as a triage signal."""
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
    load = compute_provider_side_load(SN13_SPECS)
    assert load.cpu_cores == 9.0  # 33% * 32 cores - 1.58 cores
    assert load.disk_kb == 1528 * GB_KB  # 1700 - (100 + 12 + 60)


def test_provider_side_load_clamps_negative_to_zero():
    load = compute_provider_side_load(
        {
            "cpu": {"count": 4},
            # 0.4 host cores over the stats window; container jitter above host clamps to zero
            "docker": {"host_cpu_percent": 10.0, "containers": [{"cpu_percent": 90.0}]},
            "hard_disk": {"used": 10 * GB_KB, "images": 20 * GB_KB, "containers": 0, "volumes": 0},
        }
    )
    assert load == ProviderSideLoad(cpu_cores=0.0, disk_kb=0)


def test_provider_side_load_skips_parts_with_missing_or_bad_inputs():
    # no docker stats, no disk breakdown, and a non-numeric core count -> no signal, never a guess
    assert compute_provider_side_load(
        {
            "cpu": {"count": "thirty-two"},
            "docker": {"host_cpu_percent": 33.0, "containers": [{"name": "pod_x"}]},
            "hard_disk": {"used": 1700 * GB_KB},
        }
    ).is_measured is False


def test_provider_side_load_requires_cpu_percent_on_every_container():
    # a partial `docker stats` result would attribute the uncovered container's CPU to the
    # provider — one container without cpu_percent must void the CPU part, not skew it
    load = compute_provider_side_load(
        {
            "cpu": {"count": 32},
            "docker": {
                "host_cpu_percent": 33.0,
                "containers": [{"cpu_percent": 158.0}, {"name": "no_stats_row"}],
            },
        }
    )
    assert load.cpu_cores is None


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


@pytest.mark.asyncio
async def test_check_warns_and_stores_signal_while_rented(context_factory):
    ctx = context_factory(state=_rented_state(SN13_SPECS))

    result = await ProviderSideLoadCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.LOAD_HIGH.reason
    assert result.event.what_we_saw["provider_cpu_cores"] == 9.0
    assert result.updates["state"].specs["provider_side_load"] == {
        "cpu_cores": 9.0,
        "disk_kb": 1528 * GB_KB,
    }


@pytest.mark.asyncio
async def test_check_records_without_warning_when_not_rented(context_factory):
    ctx = context_factory(state=build_state(specs=SN13_SPECS))

    result = await ProviderSideLoadCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.LOAD_RECORDED.reason
    assert result.updates["state"].specs["provider_side_load"]["cpu_cores"] == 9.0


@pytest.mark.asyncio
async def test_check_passes_quietly_when_signal_not_measurable(context_factory):
    ctx = context_factory(state=build_state(specs={"gpu": {"count": 1}}))

    result = await ProviderSideLoadCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.NOT_MEASURABLE.reason
    assert result.updates == {}


def test_provider_side_load_survives_malformed_containers():
    # a null container entry must void the signal, never abort the validation pipeline
    assert compute_provider_side_load(
        {
            "cpu": {"count": 32},
            "docker": {"host_cpu_percent": 33.0, "containers": [None]},
            "hard_disk": None,
        }
    ).is_measured is False


def test_provider_side_load_survives_non_dict_docker_section():
    assert compute_provider_side_load(
        {"cpu": {"count": 32}, "docker": ["bad"], "hard_disk": {"used": 1}}
    ).cpu_cores is None


def test_provider_side_load_voids_cpu_on_non_finite_host_sample():
    load = compute_provider_side_load(
        {
            "cpu": {"count": 32},
            "docker": {"host_cpu_percent": "nan", "containers": [{"cpu_percent": 10.0}]},
        }
    )
    assert load.cpu_cores is None


def test_provider_side_load_voids_cpu_on_negative_container_percent():
    # a negative percent would ADD to the provider share and fabricate a warning
    load = compute_provider_side_load(
        {
            "cpu": {"count": 32},
            "docker": {"host_cpu_percent": 33.0, "containers": [{"cpu_percent": -500.0}]},
        }
    )
    assert load.cpu_cores is None


def test_provider_side_load_survives_infinite_inputs():
    load = compute_provider_side_load(
        {
            "cpu": {"count": float("inf")},
            "docker": {"host_cpu_percent": 33.0, "containers": [{"cpu_percent": 1.0}]},
            "hard_disk": {"used": float("inf"), "images": 0, "containers": 0, "volumes": 0},
        }
    )
    assert load.is_measured is False


def test_provider_side_load_voids_signal_on_negative_inputs():
    # negative disk components would inflate the provider share and fabricate a warning
    load = compute_provider_side_load(
        {
            "cpu": {"count": 32},
            "docker": {"host_cpu_percent": -33.0, "containers": [{"cpu_percent": 1.0}]},
            "hard_disk": {"used": 500 * GB_KB, "images": -400 * GB_KB, "containers": 0, "volumes": 0},
        }
    )
    assert load.is_measured is False
