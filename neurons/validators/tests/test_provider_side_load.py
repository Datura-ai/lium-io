"""DAH-2734: provider-side CPU/disk gate — the twin of the DAH-2735 foreign-GPU gate."""
import asyncio
import time
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from neurons.validators.src.clients.backend_client import PodRentalActiveResponse
from neurons.validators.src.protocol.vc_protocol.compute_requests import (
    RentedExecutor,
    RentedExecutorsResponse,
    RentedPod,
    StorageOperationActiveResponse,
)
from neurons.validators.src.services.task.checks.provider_side_load import (
    ContainerRow,
    ProviderSideLoad,
    ProviderSideLoadCheck,
    compute_provider_side_load,
    provider_cores_outside_lium,
)
from neurons.validators.src.services.task.pipeline import ContextState
from neurons.validators.src.services.task.messages import ProviderSideLoadMessages as Msg

from neurons.validators.src.services import storage_operations
from neurons.validators.src.services.storage_operations import (
    StorageHelperRecord,
    forget_storage_helper,
    note_started_storage_helper,
    settle_excused_storage_helper_keys,
    storage_helper_key,
    storage_helper_name,
)
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


# the same machine with the disk part removed, to judge the CPU verdict on its own
CPU_ONLY_SPECS = {key: value for key, value in SN13_SPECS.items() if key != "hard_disk"}


def test_provider_side_load_attributes_cpu_and_disk():
    load = compute_provider_side_load(SN13_SPECS, 32, {"pod_renter"})
    assert load.cpu_cores == 8.9  # 33% * 32 cores - 1.58 cores, floored
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
        4,
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
        None,
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
        32,
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
        32,
        {"pod_renter"},
    )
    assert load.cpu_cores == 8.9  # the miner's 7 cores stay on the provider's side


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
def provider_load_gate(*, enforce: bool = True, stack_cap: bool = False):
    """DAH-2734 ships shadow-first, like every other money-withholding gate."""
    with patch("neurons.validators.src.services.task.checks.provider_side_load.settings") as s:
        s.PROVIDER_SIDE_LOAD_ENFORCEMENT_ENABLED = enforce
        s.PROVIDER_SIDE_LOAD_STACK_EXCUSE_CORES = 8.0
        s.PROVIDER_SIDE_LOAD_STACK_EXCUSE_CAP_ENABLED = stack_cap
        yield s


@pytest.mark.asyncio
async def test_enforcement_zeroes_the_score_on_an_idle_machine(context_factory):
    ctx = context_factory(
        state=build_state(specs=SN13_SPECS, rented_data=_no_rentals()),
        runner=confirming_runner(10.56, 32, [("c1", "pod_renter", 158.0)]),
    )

    with provider_load_gate(enforce=True):
        result = await ProviderSideLoadCheck().run(ctx)

    assert result.passed is False
    assert result.updates["provider_side_load_passed"] is False
    assert result.event.reason_code == Msg.LOAD_ABOVE_LIMIT.reason
    assert result.event.what_we_saw["is_rented"] is False
    assert result.event.what_we_saw["provider_cpu_cores"] == 10.5  # nothing of Lium's to subtract
    assert result.updates["state"].specs["provider_side_load"] == {
        "cpu_cores": 10.5,
        "disk_kb": 1528 * GB_KB,
    }


@pytest.mark.asyncio
async def test_a_rented_machine_is_observed_only(context_factory):
    # beside a renter's pod the host reading carries the renter's own load
    ctx = context_factory(
        state=_rented_state(SN13_SPECS),
        runner=confirming_runner(10.56, 32, [("c1", "pod_renter", 158.0)]),
    )

    with provider_load_gate(enforce=True):
        result = await ProviderSideLoadCheck().run(ctx)

    assert result.passed is True
    assert "provider_side_load_passed" not in result.updates
    assert result.event.reason_code == Msg.LOAD_ABOVE_LIMIT.reason
    assert result.event.impact == "Observed only beside a Lium workload: score was NOT changed"
    assert result.event.what_we_saw["provider_cpu_cores"] == 8.9


@pytest.mark.asyncio
async def test_an_idle_node_with_only_a_filler_is_observed_only(context_factory):
    fillers = {"executor-123": ["filler_run"]}  # the default executor's uuid
    ctx = context_factory(
        state=build_state(
            specs=CPU_ONLY_SPECS,
            rented_data=RentedExecutorsResponse(
                executors={}, banned_guids=[], all_filler_containers_by_executor=fillers
            ),
        ),
        runner=confirming_runner(10.56, 32, [("c1", "filler_run", 158.0)]),
    )

    with provider_load_gate(enforce=True):
        result = await ProviderSideLoadCheck().run(ctx)

    assert result.passed is True
    assert "provider_side_load_passed" not in result.updates
    assert result.event.impact == "Observed only beside a Lium workload: score was NOT changed"


@pytest.mark.asyncio
async def test_a_pod_started_after_the_snapshot_still_voids_the_cpu_verdict(context_factory):
    # the backend confirms the late pod, so the node is not judged even while the rest stays high
    pod_id = "0e6f3a2c-1f4d-4a9b-9a51-1f2c3d4e5f60"
    specs = {
        "cpu": {"count": 32},
        "docker": {
            "host_cpu_percent": 40.0,
            "containers": [{"name": f"pod_{pod_id}", "cpu_percent": 158.0}],
        },
    }
    ctx = context_factory(
        state=build_state(specs=specs, rented_data=_no_rentals()),
        runner=confirming_runner(12.8, 32, [("c1", f"pod_{pod_id}", 158.0)]),
    )
    ctx.services.backend.get_pod_rental_active.return_value = PodRentalActiveResponse(
        active=True, executor_id=ctx.executor.uuid
    )

    with provider_load_gate(enforce=True):
        result = await ProviderSideLoadCheck().run(ctx)

    assert result.passed is True
    assert "provider_side_load_passed" not in result.updates
    assert result.event.impact == "Observed only beside a Lium workload: score was NOT changed"


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
    ctx = context_factory(state=build_state(specs=specs, rented_data=_no_rentals()))

    with provider_load_gate(enforce=True):
        result = await ProviderSideLoadCheck().run(ctx)

    assert result.passed is True
    assert result.event.what_we_saw["provider_cpu_cores"] == 0.2  # 3.2 host cores - 3.0 ours


@pytest.mark.asyncio
async def test_a_foreign_container_zeroes_the_score_of_an_idle_machine(context_factory):
    # The whole point of the ticket, end to end: the provider mines another subnet with
    # `docker run` on a listed machine.
    specs = {
        **SN13_SPECS,
        "docker": {
            **SN13_SPECS["docker"],
            "containers": [
                {"name": "sn13_miner", "cpu_percent": 700.0},
            ],
        },
    }
    ctx = context_factory(
        state=build_state(specs=specs, rented_data=_no_rentals()),
        runner=confirming_runner(10.56, 32, [("c2", "sn13_miner", 700.0)]),
    )

    with provider_load_gate(enforce=True):
        result = await ProviderSideLoadCheck().run(ctx)

    assert result.passed is False
    assert result.updates["provider_side_load_passed"] is False
    assert result.event.what_we_saw["provider_cpu_cores"] == 10.5


@pytest.mark.asyncio
async def test_no_backend_truth_voids_the_cpu_verdict(context_factory):
    # Without the backend's container list every filler reads as a foreign workload. The
    # foreign-GPU twin gives no verdict in that case and neither does this gate.
    ctx = context_factory(state=build_state(specs=CPU_ONLY_SPECS))  # rented_data defaults to None

    with provider_load_gate(enforce=True):
        result = await ProviderSideLoadCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.NOT_MEASURABLE.reason


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


@pytest.mark.asyncio
async def test_a_custom_image_build_is_not_the_providers_load(context_factory):
    # `lium-dind-build-<pod_id>` is Lium building a renter's image. It is on no rental list,
    # and a build takes cores, so without this the gate zeroes an honest machine.
    specs = {
        "cpu": {"count": 32},
        "docker": {
            "host_cpu_percent": 20.0,
            "containers": [
                {"name": "pod_renter", "cpu_percent": 40.0},
                {"name": "lium-dind-build-pod-1", "cpu_percent": 600.0},
            ],
        },
    }
    ctx = context_factory(state=_rented_state(specs))

    with provider_load_gate(enforce=True):
        result = await ProviderSideLoadCheck().run(ctx)

    assert result.passed is True
    assert result.event.what_we_saw["provider_cpu_cores"] == 0.0  # 6.4 host - 0.4 pod - 6.0 build


@pytest.mark.asyncio
async def test_the_monitor_container_shares_the_executor_image_and_is_ours(context_factory):
    # The executor stack runs the same image twice (executor + monitor). docker.container_id
    # names only one of them, so the other must be found by digest.
    specs = {
        "cpu": {"count": 32},
        "docker": {
            "container_id": "executor-id",
            "host_cpu_percent": 20.0,
            "containers": [
                {"container_id": "executor-id", "digest": "sha256:exec", "cpu_percent": 300.0},
                {"container_id": "monitor-id", "digest": "sha256:exec", "cpu_percent": 300.0},
            ],
        },
    }
    ctx = context_factory(state=build_state(specs=specs, rented_data=_no_rentals()))

    with provider_load_gate(enforce=True):
        result = await ProviderSideLoadCheck().run(ctx)

    assert result.passed is True
    assert result.event.what_we_saw["provider_cpu_cores"] == 0.4  # 6.4 host - 3.0 - 3.0


@pytest.mark.asyncio
async def test_a_spike_that_is_gone_on_the_second_look_costs_nothing(context_factory):
    # Lium's own watchtower pulling an image can hold two cores for a second. It is gone by the
    # second reading, and an honest machine keeps its score.
    ctx = context_factory(
        state=_rented_state(CPU_ONLY_SPECS),
        runner=confirming_runner(1.9, 32, [("c1", "pod_renter", 30.0)]),
    )

    with provider_load_gate(enforce=True):
        result = await ProviderSideLoadCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.LOAD_OK.reason
    assert result.event.what_we_saw["provider_cpu_cores"] == 1.6


@pytest.mark.asyncio
async def test_an_unreadable_second_look_withholds_nothing(context_factory):
    # ssh is down, so the load cannot be confirmed. The CPU verdict voids.
    ctx = context_factory(state=_rented_state(CPU_ONLY_SPECS))  # runner defaults to None

    with provider_load_gate(enforce=True):
        result = await ProviderSideLoadCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.NOT_MEASURABLE.reason


@pytest.mark.asyncio
async def test_a_docker_that_does_not_answer_the_second_look_withholds_nothing(context_factory):
    # An empty container section means docker did not answer. Reading it as "Lium runs nothing
    # here" would charge the machine's whole load to the provider.
    runner = AsyncMock()
    runner.run = AsyncMock(
        return_value=MagicMock(success=True, stdout="0 0 32 0\n@@@\n\n@@@\n1000 0 32 0")
    )
    ctx = context_factory(state=_rented_state(CPU_ONLY_SPECS), runner=runner)

    with provider_load_gate(enforce=True):
        result = await ProviderSideLoadCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.NOT_MEASURABLE.reason


@pytest.mark.asyncio
@pytest.mark.parametrize("probe_name", ["lium_roce_probe", "lium_egress_probe_12_ab12cd"])
async def test_the_roce_link_probe_is_not_the_providers_load(context_factory, probe_name):
    # DAH-2667 runs `ib_write_bw` in `lium_roce_probe` on the host; DAH-6004's egress check runs
    # in `lium_egress_probe_*`. Both are ours, burn cores while they run, and are on no rental list.
    specs = {
        "cpu": {"count": 32},
        "docker": {
            "host_cpu_percent": 4.4,
            # `ib_write_bw` busy-polls one core
            "containers": [{"name": probe_name, "cpu_percent": 100.0}],
        },
    }
    ctx = context_factory(state=build_state(specs=specs, rented_data=_no_rentals()))

    with provider_load_gate(enforce=True):
        result = await ProviderSideLoadCheck().run(ctx)

    assert result.passed is True
    assert result.event.what_we_saw["provider_cpu_cores"] == 0.4  # 1.4 host - 1.0 the probe


@pytest.mark.asyncio
async def test_a_pod_started_after_the_snapshot_is_not_the_providers_load(context_factory):
    # DAH-2757: rented_data is read once at cycle start, so a pod started later in the same
    # cycle is ours and missing from it. The backend confirms the id inside the name.
    pod_id = "0e6f3a2c-1f4d-4a9b-9a51-1f2c3d4e5f60"
    specs = {
        "cpu": {"count": 32},
        "docker": {
            "host_cpu_percent": 20.0,
            "containers": [{"name": f"pod_{pod_id}", "cpu_percent": 500.0}],
        },
    }
    ctx = context_factory(state=build_state(specs=specs, rented_data=_no_rentals()))
    ctx.services.backend.get_pod_rental_active.return_value = PodRentalActiveResponse(
        active=True, executor_id=ctx.executor.uuid
    )

    with provider_load_gate(enforce=True):
        result = await ProviderSideLoadCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.LOAD_OK.reason
    assert result.event.what_we_saw["provider_cpu_cores"] == 1.4  # 6.4 host - 5.0 the fresh pod
    assert result.event.what_we_saw["lium_named_outside_snapshot_cores"] == 0.0


@pytest.mark.asyncio
async def test_a_miner_wearing_an_infra_name_is_not_excused(context_factory):
    # The infra tier matches by name, and a name is forgeable. A port probe holds a fraction of
    # a core; 7 cores under the same name is a workload, not a probe.
    specs = {
        "cpu": {"count": 32},
        "docker": {
            "host_cpu_percent": 25.0,
            "containers": [{"name": "container_sn13", "cpu_percent": 700.0}],
        },
    }
    ctx = context_factory(
        state=build_state(specs=specs, rented_data=_no_rentals()),
        runner=confirming_runner(8.0, 32, [("c1", "container_sn13", 700.0)]),
    )

    with provider_load_gate(enforce=True):
        result = await ProviderSideLoadCheck().run(ctx)

    assert result.passed is False
    assert result.event.what_we_saw["provider_cpu_cores"] == 8.0


@pytest.mark.asyncio
async def test_a_quiet_port_probe_is_still_excused(context_factory):
    specs = {
        "cpu": {"count": 32},
        "docker": {
            "host_cpu_percent": 5.0,
            "containers": [{"name": "container_port_check", "cpu_percent": 20.0}],
        },
    }
    ctx = context_factory(state=build_state(specs=specs, rented_data=_no_rentals()))

    with provider_load_gate(enforce=True):
        result = await ProviderSideLoadCheck().run(ctx)

    assert result.passed is True
    assert result.event.what_we_saw["provider_cpu_cores"] == 1.4  # 1.6 host - 0.2 the probe


@pytest.mark.asyncio
async def test_a_lium_image_pull_is_not_the_providers_load(context_factory):
    # Pulling a pod's image is work Lium asked for. It runs in dockerd, belongs to no container
    # row, and a big pull outlives both samples — unsubtracted it zeroes an honest machine.
    ctx = context_factory(
        state=_rented_state(CPU_ONLY_SPECS),
        runner=confirming_runner(3.5, 32, [("c1", "pod_renter", 158.0)], dockerd_cores=1.9),
    )

    with provider_load_gate(enforce=True):
        result = await ProviderSideLoadCheck().run(ctx)

    assert result.passed is True
    assert result.event.what_we_saw["provider_cpu_cores"] == 0.0  # 3.5 - 1.58 - 1.9


@pytest.mark.asyncio
async def test_many_forged_infra_names_do_not_stack_the_excuse(context_factory):
    # The cap is on the SUM: six forged `container_*` names at 1.5 cores each are a 9-core
    # miner, not six probes.
    specs = {
        "cpu": {"count": 32},
        "docker": {
            "host_cpu_percent": 30.0,
            "containers": [
                {"name": f"container_sn13_{worker}", "cpu_percent": 150.0} for worker in range(6)
            ],
        },
    }
    ctx = context_factory(
        state=build_state(specs=specs, rented_data=_no_rentals()),
        runner=confirming_runner(9.6, 32, [(f"c{worker}", f"container_sn13_{worker}", 150.0) for worker in range(6)]),
    )

    with provider_load_gate(enforce=True):
        result = await ProviderSideLoadCheck().run(ctx)

    assert result.passed is False
    assert result.event.what_we_saw["provider_cpu_cores"] == 9.6


@pytest.mark.asyncio
async def test_a_disk_only_verdict_never_withholds_money(context_factory):
    # Rustam's call, after the review: the disk figure is read once with no second look, and
    # every docker category nobody enumerated lands in it. It is reported, never scored.
    disk_only_specs = {"hard_disk": SN13_SPECS["hard_disk"]}
    ctx = context_factory(state=_rented_state(disk_only_specs))

    with provider_load_gate(enforce=True):
        result = await ProviderSideLoadCheck().run(ctx)

    assert result.passed is True
    assert "provider_side_load_passed" not in result.updates
    assert result.event.reason_code == Msg.LOAD_ABOVE_LIMIT.reason
    assert result.event.severity == "warning"
    assert result.event.what_we_saw["provider_disk_gb"] == 1528.0
    assert result.event.impact == "Disk is observed only: score was NOT changed"


@pytest.mark.asyncio
async def test_the_rest_of_the_executor_stack_is_not_the_providers_load(context_factory):
    # watchtower, autoheal, the runner and postgres run beside the executor under the same
    # compose project. They are Lium's, and at two cores they would zero an honest node.
    specs = {
        "cpu": {"count": 32},
        "docker": {
            "host_cpu_percent": 8.0,
            "containers": [
                {"name": "executor-watchtower-1", "cpu_percent": 100.0},
                {"name": "executor-autoheal-1", "cpu_percent": 50.0},
            ],
        },
    }
    ctx = context_factory(state=build_state(specs=specs, rented_data=_no_rentals()))

    with provider_load_gate(enforce=True):
        result = await ProviderSideLoadCheck().run(ctx)

    assert result.passed is True
    assert result.event.what_we_saw["provider_cpu_cores"] == 1.0  # 2.56 host - 1.5 the stack


@pytest.mark.asyncio
async def test_a_half_finished_stats_run_withholds_nothing(context_factory):
    # `docker stats` can die after printing some rows. The chain's exit status belongs to the
    # last command, so the run reports its own failure and the reading is dropped.
    runner = AsyncMock()
    runner.run = AsyncMock(
        return_value=MagicMock(
            success=True,
            stdout="0 0 32 0\n@@@\nc1|pod_renter|10.00%\nSTATS_FAILED\n@@@\n1000 0 32 0",
        )
    )
    ctx = context_factory(state=_rented_state(CPU_ONLY_SPECS), runner=runner)

    with provider_load_gate(enforce=True):
        result = await ProviderSideLoadCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.NOT_MEASURABLE.reason


@pytest.mark.asyncio
async def test_a_miner_renamed_dockerd_is_capped_like_every_other_forged_name(context_factory):
    # `pgrep -x dockerd` matches a process NAME, and a provider owns the host: copying the miner
    # to `dockerd` would otherwise hide nine cores on the reading that decides money.
    ctx = context_factory(
        state=build_state(specs=CPU_ONLY_SPECS, rented_data=_no_rentals()),
        runner=confirming_runner(10.56, 32, [("c1", "sn13_miner", 158.0)], dockerd_cores=9.0),
    )

    with provider_load_gate(enforce=True):
        result = await ProviderSideLoadCheck().run(ctx)

    assert result.passed is False
    assert result.event.what_we_saw["provider_cpu_cores"] == 8.5  # 10.56 - 2.0 capped


def _miner_on_the_executor_image() -> dict[str, object]:
    """A third container started from the executor's image, holding nine cores."""
    return {
        "cpu": {"count": 32},
        "docker": {
            "container_id": "executor-id",
            "host_cpu_percent": 35.0,
            "containers": [
                {"container_id": "executor-id", "digest": "sha256:exec", "cpu_percent": 100.0},
                {"container_id": "monitor-id", "digest": "sha256:exec", "cpu_percent": 50.0},
                {"container_id": "miner-id", "digest": "sha256:exec", "cpu_percent": 900.0},
            ],
        },
    }


@pytest.mark.asyncio
async def test_a_workload_on_the_executor_image_is_excused_and_recorded_by_default(context_factory):
    ctx = context_factory(state=build_state(specs=_miner_on_the_executor_image(), rented_data=_no_rentals()))

    with provider_load_gate(enforce=True):
        result = await ProviderSideLoadCheck().run(ctx)

    assert result.passed is True
    assert result.event.what_we_saw["executor_stack_containers"] == 3
    assert result.event.what_we_saw["executor_stack_cores"] == 10.5


@pytest.mark.asyncio
async def test_under_the_cap_a_workload_on_the_executor_image_is_the_providers_load(context_factory):
    specs = _miner_on_the_executor_image()
    ctx = context_factory(
        state=build_state(specs=specs, rented_data=_no_rentals()),
        runner=confirming_runner(
            11.2,
            32,
            [("executor-id", "executor", 100.0), ("monitor-id", "monitor", 50.0), ("miner-id", "miner", 900.0)],
        ),
    )

    with provider_load_gate(enforce=True, stack_cap=True):
        result = await ProviderSideLoadCheck().run(ctx)

    assert result.passed is False
    assert result.event.what_we_saw["provider_cpu_cores"] >= 2.0


@pytest.mark.asyncio
async def test_under_the_cap_the_honest_stack_is_still_excused(context_factory):
    specs = _miner_on_the_executor_image()
    specs["docker"]["containers"].pop()
    specs["docker"]["host_cpu_percent"] = 6.0
    ctx = context_factory(state=build_state(specs=specs, rented_data=_no_rentals()))

    with provider_load_gate(enforce=True, stack_cap=True):
        result = await ProviderSideLoadCheck().run(ctx)

    assert result.passed is True
    assert result.event.what_we_saw["executor_stack_cores"] == 1.5


OPERATION = UUID("01234567-89ab-4def-8123-456789abcdef")
BACKUP_HELPER = storage_helper_name(OPERATION)  # "lium-storage-01234567-89a"
POD = ("c1", "pod_renter", 158.0)


def _on_node(runner, states: dict[UUID, str]):
    """The node also serves runner files claiming `states`; the check must never ask them."""
    second_look = runner.run.return_value

    def run(command, **_):
        if "storage-operations" not in command:
            return second_look
        return MagicMock(success=True, stdout="".join(f"{op} {st}\n" for op, st in states.items()))

    runner.run = AsyncMock(side_effect=run)
    return runner


def _helper_runner(name: str, helper_percent: float = 700.0, host_cores: float = 10.56):
    runner = confirming_runner(host_cores, 32, [POD, ("c2", name, helper_percent)])
    return _on_node(runner, {OPERATION: "RUNNING"})


def _backend_says(ctx, answer, report_age: int | None = 30):
    """The backend's answer on the operation: open (True), ended (False), none (None) or raise."""
    if isinstance(answer, Exception):
        mock = AsyncMock(side_effect=answer)
    elif answer is None:
        mock = AsyncMock(return_value=None)
    else:
        mock = AsyncMock(
            return_value=StorageOperationActiveResponse(
                active=answer,
                status="RUNNING" if answer else "COMPLETED",
                last_report_at=(
                    None
                    if report_age is None
                    else datetime.now(timezone.utc) - timedelta(seconds=report_age)
                ),
            )
        )
    ctx.services.backend.get_storage_operation_active = mock
    return ctx


def _with_recorded_helpers(
    ctx,
    *operations: UUID,
    records: set[str] | None = None,
    age: int = 0,
    backend: object = True,
    report_age: int | None = 30,
):
    """The operations this validator recorded when it started them here (DAH-6148).

    `records` shares one store across cycles; it is returned so a test can read it back."""
    store = records if records is not None else set()
    store.update(f"{op}|{int(time.time()) - age}" for op in operations)
    redis = ctx.services.redis
    redis.smembers = AsyncMock(side_effect=lambda key: {record.encode() for record in store})
    redis.srem = AsyncMock(side_effect=lambda key, record: store.discard(record))
    redis.sadd = AsyncMock(side_effect=lambda key, record: store.add(record))
    return _backend_says(ctx, backend, report_age)


async def _provider_cores(ctx) -> float:
    with provider_load_gate(enforce=True):
        result = await ProviderSideLoadCheck().run(ctx)
    return result.event.what_we_saw["provider_cpu_cores"]


@pytest.mark.asyncio
async def test_a_backup_helper_this_validator_started_is_lium_load(context_factory):
    # DAH-6148: the hourly backup's restic helper starts after the scrape; its cores are Lium's
    ctx = _with_recorded_helpers(
        context_factory(state=_rented_state(SN13_SPECS), runner=_helper_runner(BACKUP_HELPER)),
        OPERATION,
    )

    assert await _provider_cores(ctx) == 1.9  # below the floor; was 8.9


@pytest.mark.parametrize(
    "name", ["lium-storage-ffffffff-fff", "lium-storage-"], ids=["foreign_helper", "bare_prefix"]
)
@pytest.mark.asyncio
async def test_a_lium_storage_name_the_validator_did_not_start_is_the_providers(
    context_factory, name
):
    # a name is forgeable: only the exact helpers this validator recorded are excused
    ctx = _with_recorded_helpers(
        context_factory(state=_rented_state(SN13_SPECS), runner=_helper_runner(name)),
        OPERATION,
    )

    assert await _provider_cores(ctx) == 8.9


@pytest.mark.asyncio
async def test_node_files_saying_running_do_not_excuse_an_operation_the_backend_ended(
    context_factory,
):
    # the runner files are the provider's to write; only the backend's status counts, and an
    # ended operation's record goes
    store: set[str] = set()
    ctx = _with_recorded_helpers(
        context_factory(state=_rented_state(SN13_SPECS), runner=_helper_runner(BACKUP_HELPER)),
        OPERATION,
        records=store,
        backend=False,
    )

    assert await _provider_cores(ctx) == 8.9
    assert store == set()
    ctx.services.backend.get_storage_operation_active.assert_awaited_once_with(str(OPERATION))
    assert not any("storage-operations" in c.args[0] for c in ctx.runner.run.await_args_list)


@pytest.mark.parametrize(
    "answer", [None, RuntimeError("backend down")], ids=["no_answer", "raised"]
)
@pytest.mark.asyncio
async def test_no_backend_answer_excuses_nothing_and_keeps_the_record(context_factory, answer):
    store: set[str] = set()
    ctx = _with_recorded_helpers(
        context_factory(state=_rented_state(SN13_SPECS), runner=_helper_runner(BACKUP_HELPER)),
        OPERATION,
        records=store,
        backend=answer,
    )

    assert await _provider_cores(ctx) == 8.9
    assert len(store) == 1


async def _cycle(context_factory, store: set[str], rows) -> float:
    """One validation cycle whose scrape and second look both list `rows` (id, name, percent)."""
    containers = [
        {"container_id": cid, "name": name, "cpu_percent": pct} for cid, name, pct in rows
    ]
    specs = {**SN13_SPECS, "docker": {**SN13_SPECS["docker"], "containers": containers}}
    runner = _on_node(confirming_runner(10.56, 32, rows), {OPERATION: "RUNNING"})
    ctx = _with_recorded_helpers(
        context_factory(state=_rented_state(specs), runner=runner), records=store
    )
    return await _provider_cores(ctx)


@pytest.mark.parametrize(
    "report_age",
    [11 * 60, -2 * 60, None],
    ids=["went_quiet", "report_from_the_future", "backend_without_report_time"],
)
@pytest.mark.asyncio
async def test_an_open_operation_whose_runner_went_quiet_excuses_nothing(
    context_factory, report_age
):
    # Rustam (DAH-6148): the backup's progress token is on the node, so open is not enough; a
    # last report over 10 min old excuses nothing, and the record stays for when it reports again.
    # So does a report time past the clock skew, or none (a backend older than the field)
    store: set[str] = set()
    ctx = _with_recorded_helpers(
        context_factory(state=_rented_state(SN13_SPECS), runner=_helper_runner(BACKUP_HELPER)),
        OPERATION,
        records=store,
        report_age=report_age,
    )

    assert await _provider_cores(ctx) == 8.9
    assert len(store) == 1


@pytest.mark.asyncio
async def test_a_backup_that_starts_before_the_second_look_is_lium_load(context_factory):
    # the records are read again just before the second look, so a helper recorded after the
    # first read is excused in the same cycle
    store: set[str] = set()
    ctx = _with_recorded_helpers(
        context_factory(state=_rented_state(SN13_SPECS), runner=_helper_runner(BACKUP_HELPER)),
        records=store,
    )
    first_read = ctx.services.redis.smembers.side_effect

    def started_after_the_first_read(key):
        store.add(f"{OPERATION}|{int(time.time())}")
        ctx.services.redis.smembers.side_effect = first_read
        return set()

    ctx.services.redis.smembers.side_effect = started_after_the_first_read

    assert await _provider_cores(ctx) == 1.9


@pytest.mark.asyncio
async def test_a_backup_that_ends_before_the_second_look_is_not_excused_on_it(context_factory):
    # the second look trusts only the fresh read: an operation the backend ended since the first
    # read leaves no excuse behind for a container under its helper name
    ctx = _with_recorded_helpers(
        context_factory(state=_rented_state(SN13_SPECS), runner=_helper_runner(BACKUP_HELPER)),
        OPERATION,
    )
    still_open = ctx.services.backend.get_storage_operation_active.return_value
    ended = StorageOperationActiveResponse(active=False, status="COMPLETED", last_report_at=None)
    ctx.services.backend.get_storage_operation_active.side_effect = [still_open, ended]

    assert await _provider_cores(ctx) == 8.9


@pytest.mark.asyncio
async def test_a_rerun_helper_of_an_open_operation_stays_lium_load(context_factory):
    # restic renews its credentials or retries an upload in a new container under the same name
    store = {f"{OPERATION}|{int(time.time()) - 3700}"}

    first = await _cycle(context_factory, store, [POD, ("c2", BACKUP_HELPER, 700.0)])
    await _cycle(context_factory, store, [POD])  # between the two helpers
    rerun = await _cycle(context_factory, store, [POD, ("c3", BACKUP_HELPER, 700.0)])

    assert (first, rerun) == (1.9, 1.9)
    assert len(store) == 1


@pytest.mark.asyncio
async def test_a_helper_renamed_away_from_its_name_is_the_providers(context_factory):
    # the excuse is the exact helper name: a rename keeps the id but loses it, uncapped or not
    store = {f"{OPERATION}|{int(time.time())}"}

    assert await _cycle(context_factory, store, [POD, ("c2", BACKUP_HELPER, 700.0)]) == 1.9
    assert await _cycle(context_factory, store, [POD, ("c2", "miner", 700.0)]) == 8.9


@pytest.mark.asyncio
async def test_a_record_older_than_any_backup_is_dropped(context_factory):
    # the backend fails a backup still open after 4 h, so no helper outlives that
    store: set[str] = set()
    ctx = _with_recorded_helpers(
        context_factory(state=_rented_state(SN13_SPECS), runner=_helper_runner(BACKUP_HELPER)),
        OPERATION,
        records=store,
        age=5 * 3600,
    )

    assert await _provider_cores(ctx) == 8.9
    assert store == set()


@pytest.mark.asyncio
async def test_with_redis_unreadable_no_helper_is_excused(context_factory):
    ctx = context_factory(state=_rented_state(SN13_SPECS), runner=_helper_runner(BACKUP_HELPER))
    ctx.services.redis.smembers = AsyncMock(side_effect=ConnectionError("Redis down"))

    assert await _provider_cores(ctx) == 8.9


@pytest.mark.asyncio
async def test_backup_helpers_excuse_no_more_than_restics_own_cores(context_factory):
    ctx = _with_recorded_helpers(
        context_factory(
            state=_rented_state(SN13_SPECS),
            runner=_helper_runner(BACKUP_HELPER, helper_percent=1000.0, host_cores=14.0),
        ),
        OPERATION,
    )

    assert await _provider_cores(ctx) == 4.4


def test_a_helper_key_never_matches_a_container_by_id():
    # the helper tier is always capped: its key is no container's name or id
    rows = [ContainerRow(container_id="c2", name="renamed", cpu_percent=700.0)]

    assert provider_cores_outside_lium(10.0, rows, {storage_helper_key(OPERATION)}) == 10.0


def _set_redis(store: set[str]) -> MagicMock:
    return MagicMock(
        sadd=AsyncMock(side_effect=lambda key, record: store.add(record)),
        srem=AsyncMock(side_effect=lambda key, record: store.discard(record)),
        expire=AsyncMock(),
        smembers=AsyncMock(side_effect=lambda key: set(store)),
    )


@pytest.mark.asyncio
async def test_a_started_operation_records_its_helper_until_it_is_forgotten():
    store: set[str] = set()
    redis = _set_redis(store)

    async def running(op):
        return True

    await note_started_storage_helper(redis, "executor-1", OPERATION)

    assert await settle_excused_storage_helper_keys(redis, "executor-1", running) == {
        storage_helper_key(OPERATION)
    }
    redis.expire.assert_awaited_once()
    await forget_storage_helper(redis, "executor-1", OPERATION)
    assert await settle_excused_storage_helper_keys(redis, "executor-1", running) == set()


@pytest.mark.asyncio
async def test_a_backend_answer_slower_than_the_redis_timeout_still_excuses(monkeypatch):
    # only the Redis calls are cut at their short timeout; the backend call has its own
    monkeypatch.setattr(storage_operations, "_STARTED_STORAGE_HELPERS_REDIS_TIMEOUT_SEC", 0.05)
    store = {f"{OPERATION}|{int(time.time())}"}

    async def slow_running(op):
        await asyncio.sleep(0.2)
        return True

    keys = await settle_excused_storage_helper_keys(_set_redis(store), "executor-1", slow_running)

    assert keys == {storage_helper_key(OPERATION)}


def test_the_helper_record_timeout_outlasts_the_redis_pool_wait():
    from neurons.validators.src.services.redis_service import REDIS_POOL_WAIT_TIMEOUT_SECONDS

    # a wave queues on the pool by design; a shorter cut fails every node's read in the wave
    timeout = storage_operations._STARTED_STORAGE_HELPERS_REDIS_TIMEOUT_SEC
    assert timeout >= REDIS_POOL_WAIT_TIMEOUT_SECONDS


def test_a_helper_record_round_trips():
    assert StorageHelperRecord.parse(StorageHelperRecord(str(OPERATION), 5).format()) == (
        StorageHelperRecord(str(OPERATION), 5)
    )
    assert StorageHelperRecord.parse("lium-storage-01234567-89a|5|c2") is None
