"""DAH-2734: provider-side CPU/disk gate — the twin of the DAH-2735 foreign-GPU gate."""
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neurons.validators.src.clients.backend_client import PodRentalActiveResponse
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
    dockerd_usec = int(dockerd_cores * window_seconds * 1_000_000)
    body = "\n".join(f"{container_id}|{name}|{percent:.2f}%" for container_id, name, percent in rows)
    stdout = (
        f"0 0 0\n@@@\n{body}\n@@@\n"
        f"{total_jiffies} {total_jiffies - busy} {dockerd_usec}"
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
async def test_idle_machine_is_judged_the_same_as_a_rented_one(context_factory):
    # The provider cheats the renter when rented and the marketplace when idle — same verdict.
    ctx = context_factory(
        state=build_state(specs=SN13_SPECS, rented_data=_no_rentals()),
        runner=confirming_runner(10.56, 32, [("c1", "pod_renter", 158.0)]),
    )

    with provider_load_gate(enforce=True):
        result = await ProviderSideLoadCheck().run(ctx)

    assert result.passed is False
    assert result.event.what_we_saw["is_rented"] is False
    assert result.event.what_we_saw["provider_cpu_cores"] == 10.5  # nothing of Lium's to subtract


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
    ctx = context_factory(state=build_state(specs=specs, rented_data=_no_rentals()))

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
    ctx = context_factory(
        state=_rented_state(specs),
        runner=confirming_runner(10.56, 32, [("c1", "pod_renter", 158.0), ("c2", "sn13_miner", 700.0)]),
    )

    with provider_load_gate(enforce=True):
        result = await ProviderSideLoadCheck().run(ctx)

    assert result.passed is False
    assert result.updates["provider_side_load_passed"] is False
    assert result.event.what_we_saw["provider_cpu_cores"] == 8.9


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
        return_value=MagicMock(success=True, stdout="0 0 0\n@@@\n\n@@@\n1000 0 0")
    )
    ctx = context_factory(state=_rented_state(CPU_ONLY_SPECS), runner=runner)

    with provider_load_gate(enforce=True):
        result = await ProviderSideLoadCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.NOT_MEASURABLE.reason


@pytest.mark.asyncio
async def test_the_roce_link_probe_is_not_the_providers_load(context_factory):
    # DAH-2667 runs `ib_write_bw` in `lium_roce_probe` on the host. It is ours, it burns cores
    # while it runs, and it appears on no rental list.
    specs = {
        "cpu": {"count": 32},
        "docker": {
            "host_cpu_percent": 4.4,
            # `ib_write_bw` busy-polls one core
            "containers": [{"name": "lium_roce_probe", "cpu_percent": 100.0}],
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
        runner=confirming_runner(10.56, 32, [("c1", "pod_renter", 158.0)], dockerd_cores=9.0),
    )

    with provider_load_gate(enforce=True):
        result = await ProviderSideLoadCheck().run(ctx)

    assert result.passed is True
    assert result.event.what_we_saw["provider_cpu_cores"] == 0.0  # 10.56 - 1.58 - 9.0


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
