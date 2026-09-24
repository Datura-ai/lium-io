"""The port check samples the whole declared range instead of its lowest BATCH_PORT_VERIFICATION_SIZE ports.

End-to-end cases run the real selector, probe cascade (batch, semi-batch, sequential) and connectivity
service against a fake host whose reachable ports are a predicate; only containers and HTTP are faked.
"""

import bisect
from unittest.mock import AsyncMock

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo

from neurons.validators.src.services.task.checks.port_connectivity import PortConnectivityCheck
from neurons.validators.src.services.task.checks.port_count import PortCountCheck
from neurons.validators.src.services.task.messages import PortCountMessages
from services.const import (
    BATCH_PORT_VERIFICATION_SIZE,
    MIN_PORT_COUNT,
    SAMPLED_PORTS_LOWEST_PASS_BELOW,
)
from services.executor_connectivity.models import ContainerStartResult, DindProbeResult, PortPair
from services.executor_connectivity.orchestrator import ConnectivityOrchestrator
from services.executor_connectivity.port_probe import PortProbe
from services.executor_connectivity.port_selector import (
    SELECTION_ALL,
    SELECTION_STRATIFIED,
    SELECTION_STRATIFIED_LOWEST,
    PortSelector,
    estimate_usable_ports,
)
from services.executor_connectivity.port_verifiers import (
    BatchVerifier,
    FallbackVerifier,
    SemiBatchVerifier,
)
from services.executor_connectivity.service import ExecutorConnectivityService
from services.port_utils import get_all_ports
from tests.helpers import build_context_config, build_services, build_state, make_context

WIDE_RANGE = "40000-65535"


def _executor(
    port_range: str | None = WIDE_RANGE, *, port_mappings: str | None = None
) -> ExecutorSSHInfo:
    return ExecutorSSHInfo(
        uuid="executor-sampling",
        address="203.0.113.10",
        port=8080,
        ssh_username="root",
        ssh_port=22,
        port_range=port_range,
        port_mappings=port_mappings,
        python_path="/usr/bin/python3",
        root_dir="/tmp",
    )


class FakeHost:
    """Answers a probe on a port iff `is_open(port)`, for every tier and for the DinD container."""

    def __init__(self, is_open):
        self.is_open = is_open

    async def test_one(self, session, host, port: PortPair, token) -> bool:
        return self.is_open(port)

    async def test_many(self, session, host, ports, token, log_ctx=None):
        return [p for p in ports if self.is_open(p)], [p for p in ports if not self.is_open(p)]


class FakeDind:
    def __init__(self, host: FakeHost):
        self.host = host
        self.ports: list[PortPair] = []

    async def verify(self, port, **_kwargs) -> DindProbeResult:
        self.ports.append(port)
        ok = self.host.is_open(port)
        return DindProbeResult(success=ok, sysbox_runtime=ok, port=port)


def _service(is_open, mocker) -> tuple[ExecutorConnectivityService, FakeHost]:
    mocker.patch("services.executor_connectivity.port_verifiers.asyncio.sleep", AsyncMock())
    host = FakeHost(is_open)
    runner = mocker.Mock()
    runner.run = AsyncMock(
        return_value=ContainerStartResult(ok=True, container_id="c", status="Up")
    )
    runner.cleanup = AsyncMock()
    probe = PortProbe(
        BatchVerifier(host, runner), SemiBatchVerifier(host, runner), FallbackVerifier(host, runner)
    )
    orchestrator = ConnectivityOrchestrator(PortSelector(), probe, FakeDind(host))
    return ExecutorConnectivityService(orchestrator), host


def _lowest_budget(executor: ExecutorSSHInfo) -> list[PortPair]:
    """What the probe set was before sampling: the lowest BATCH_PORT_VERIFICATION_SIZE free ports."""
    pairs = get_all_ports(executor.port_range, executor.port_mappings, executor.ssh_port)
    return [PortPair(i, e) for i, e in pairs[:BATCH_PORT_VERIFICATION_SIZE]]


def _redis() -> AsyncMock:
    redis = AsyncMock()
    redis.renting_in_progress.return_value = False
    redis.record_dind_probe_miss.return_value = True
    return redis


async def _run_checks(
    service: ExecutorConnectivityService, executor: ExecutorSSHInfo, job_batch_id: str = "cycle-1"
):
    services = build_services(connectivity=service, redis=_redis())
    services.backend.get_all_rented_executors.return_value = None
    ctx = make_context(
        executor=executor,
        services=services,
        config=build_context_config(job_batch_id=job_batch_id),
        state=build_state(sysbox_runtime=False),
    )
    connectivity = await PortConnectivityCheck().run(ctx)
    count_ctx = make_context(
        executor=executor,
        services=ctx.services,
        config=ctx.config,
        state=connectivity.updates["state"],
    )
    return connectivity, await PortCountCheck().run(count_ctx)


# --- selection ---------------------------------------------------------------------------------


def test_a_small_range_is_probed_whole_in_order():
    info = _executor("9000-9249")

    sample = PortSelector().select(info, BATCH_PORT_VERIFICATION_SIZE, set(), seed="s")

    assert [p.internal for p in sample.ports] == list(range(9000, 9250))
    assert sample.selection == SELECTION_ALL
    assert sample.declared_count == sample.free_count == 250
    assert sample.lowest_ports == []


def test_a_range_of_exactly_the_budget_is_probed_whole():
    info = _executor(f"9000-{9000 + BATCH_PORT_VERIFICATION_SIZE - 1}")

    sample = PortSelector().select(info, BATCH_PORT_VERIFICATION_SIZE, set(), seed="s")

    assert len(sample.ports) == BATCH_PORT_VERIFICATION_SIZE
    assert sample.selection == SELECTION_ALL


def test_a_small_range_minus_in_use_ports_is_probed_whole():
    # 350 declared, 60 held by pods and fillers: the 290 left fit the budget and are all probed
    info = _executor("9000-9349")
    in_use = set(range(9000, 9060))

    sample = PortSelector().select(info, BATCH_PORT_VERIFICATION_SIZE, in_use, seed="s")

    assert [p.internal for p in sample.ports] == list(range(9060, 9350))
    assert sample.selection == SELECTION_ALL
    assert (sample.declared_count, sample.free_count) == (350, 290)


def test_a_wide_range_is_sampled_one_port_per_stratum():
    info = _executor()
    free = get_all_ports(info.port_range, None, info.ssh_port)

    sample = PortSelector().select(info, BATCH_PORT_VERIFICATION_SIZE, set(), seed="executor:cycle")

    assert sample.selection == SELECTION_STRATIFIED
    assert len(sample.ports) == len(set(sample.ports)) == BATCH_PORT_VERIFICATION_SIZE
    assert sample.declared_count == sample.free_count == 25536
    n, size = len(free), BATCH_PORT_VERIFICATION_SIZE
    starts = [k * n // size for k in range(size)]
    index = {pair[0]: i for i, pair in enumerate(free)}
    strata = sorted(bisect.bisect_right(starts, index[p.internal]) - 1 for p in sample.ports)
    assert strata == list(range(size))
    assert sample.lowest_ports == _lowest_budget(info)


def test_a_wide_sample_spreads_over_every_tenth_of_the_range():
    info = _executor()

    ports = (
        PortSelector()
        .select(info, BATCH_PORT_VERIFICATION_SIZE, set(), seed="executor:cycle")
        .ports
    )

    per_tenth = [0] * 10
    for p in ports:
        per_tenth[(p.internal - 40000) * 10 // 25536] += 1
    assert all(29 <= c <= 31 for c in per_tenth), per_tenth
    assert sum(1 for p in ports if p.internal < 40000 + BATCH_PORT_VERIFICATION_SIZE) <= 5


@pytest.mark.parametrize("seed", [f"executor:cycle-{i}" for i in range(20)])
def test_the_published_port_tiers_prefix_is_spread_too(seed):
    # SemiBatchVerifier takes the first 50 and FallbackVerifier the first 10 of the probe list
    ports = PortSelector().select(_executor(), BATCH_PORT_VERIFICATION_SIZE, set(), seed=seed).ports

    tenths = {(p.internal - 40000) * 10 // 25536 for p in ports[:50]}
    assert len(tenths) >= 7


def test_the_sample_is_deterministic_per_seed_and_moves_between_cycles():
    info = _executor()
    selector = PortSelector()

    first = selector.select(
        info, BATCH_PORT_VERIFICATION_SIZE, set(), seed="executor:cycle-1"
    ).ports
    again = selector.select(
        info, BATCH_PORT_VERIFICATION_SIZE, set(), seed="executor:cycle-1"
    ).ports
    next_cycle = selector.select(
        info, BATCH_PORT_VERIFICATION_SIZE, set(), seed="executor:cycle-2"
    ).ports

    assert first == again
    assert set(first) != set(next_cycle)
    assert len(set(first) & set(next_cycle)) < 30


def test_without_a_seed_the_sample_is_seeded_by_the_executor():
    info = _executor()
    selector = PortSelector()

    assert (
        selector.select(info, BATCH_PORT_VERIFICATION_SIZE, set()).ports
        == selector.select(info, BATCH_PORT_VERIFICATION_SIZE, set(), seed=info.uuid).ports
    )


def test_in_use_ports_are_never_sampled():
    info = _executor()
    # every third port across the range is held by the node's own pods or fillers
    in_use = set(range(40000, 65536, 3))

    sample = PortSelector().select(
        info, BATCH_PORT_VERIFICATION_SIZE, in_use, seed="executor:cycle"
    )

    assert len(sample.ports) == BATCH_PORT_VERIFICATION_SIZE
    assert not {p.external for p in sample.ports} & in_use
    assert sample.free_count == 25536 - len(in_use)
    assert sample.declared_count == 25536


def test_in_use_ports_are_matched_on_the_external_side_of_mappings():
    mappings = str([[20000 + i, 50000 + i] for i in range(1000)])
    info = _executor(None, port_mappings=mappings)
    in_use = {50000 + i for i in range(0, 1000, 2)}

    ports = PortSelector().select(info, BATCH_PORT_VERIFICATION_SIZE, in_use, seed="s").ports

    assert len(ports) == BATCH_PORT_VERIFICATION_SIZE
    assert all(p.external % 2 == 1 and p.internal == p.external - 30000 for p in ports)


@pytest.mark.parametrize(
    "verified,probed,free,expected",
    [
        (250, 250, 250, 250),  # probed whole: the estimate is the verified count
        (150, 300, 25536, 12768),
        (296, 300, 25536, 25195),
        (0, 300, 25536, 0),
        (5, 0, 25536, 0),
        (10, 10, 25536, 25536),  # capped at the free ports
    ],
)
def test_estimate_usable_ports(verified, probed, free, expected):
    assert estimate_usable_ports(verified, probed, free) == expected


# --- end to end through the probe cascade and both checks -------------------------------------


@pytest.mark.asyncio
async def test_a_host_whose_lowest_ports_are_closed_now_verifies_most_of_its_range(mocker):
    # an 8x B300 host declared 40000-65535; its lowest 300 ports were busy or unforwarded but for two
    info = _executor()
    low_open = {40010, 40020}

    def is_open(p: PortPair) -> bool:
        return p.internal >= 40000 + BATCH_PORT_VERIFICATION_SIZE or p.internal in low_open

    assert sum(is_open(p) for p in _lowest_budget(info)) == 2  # what it published before
    service, _ = _service(is_open, mocker)

    connectivity, count = await _run_checks(service, info)

    state = connectivity.updates["state"]
    assert connectivity.passed and count.passed
    assert count.event.reason_code == PortCountMessages.PORT_COUNT_RECORDED.reason
    assert state.verified_port_count >= 290
    assert count.updates["state"].specs["available_port_count"] == state.verified_port_count
    assert state.probed_port_count == BATCH_PORT_VERIFICATION_SIZE
    assert state.declared_port_count == 25536
    assert state.estimated_usable_port_count >= 24000
    extra = connectivity.event.context
    assert extra["port_selection"] == SELECTION_STRATIFIED
    assert extra["probed_port_count"] == BATCH_PORT_VERIFICATION_SIZE
    assert extra["declared_port_count"] == 25536
    assert extra["estimated_usable_port_count"] == state.estimated_usable_port_count
    assert (
        count.event.what_we_saw["estimated_usable_port_count"] == state.estimated_usable_port_count
    )


@pytest.mark.asyncio
async def test_a_truly_firewalled_range_stays_below_the_floor(mocker):
    info = _executor()
    only_open = {41234, 60001}
    service, _ = _service(lambda p: p.internal in only_open, mocker)

    connectivity, count = await _run_checks(service, info)

    state = connectivity.updates["state"]
    assert state.verified_port_count <= len(only_open)
    assert state.verified_port_count < MIN_PORT_COUNT
    assert count.updates["state"].specs["available_port_count"] == state.verified_port_count
    assert state.estimated_usable_port_count < 25536 // 10
    assert count.passed is False
    assert count.event.reason_code == PortCountMessages.INSUFFICIENT_PORTS.reason
    assert count.event.what_we_saw["declared_port_count"] == 25536
    assert connectivity.event.context["port_selection"] == SELECTION_STRATIFIED_LOWEST


@pytest.mark.asyncio
async def test_a_host_forwarding_only_its_lowest_ports_is_still_listed(mocker):
    # declares 40000-65535 but forwards only 40000-40149: the sample lands at most two ports there
    info = _executor()
    service, _ = _service(lambda p: p.internal < 40150, mocker)

    connectivity, count = await _run_checks(service, info)

    state = connectivity.updates["state"]
    assert connectivity.event.context["port_selection"] == SELECTION_STRATIFIED_LOWEST
    assert state.verified_port_count == 150
    assert count.passed
    assert state.estimated_usable_port_count >= state.verified_port_count


@pytest.mark.asyncio
async def test_three_working_ports_among_the_lowest_are_still_found(mocker):
    info = _executor()
    open_ports = {40250, 40260, 40270}
    service, _ = _service(lambda p: p.internal in open_ports, mocker)

    connectivity, count = await _run_checks(service, info)

    assert {p for p, _ in connectivity.updates["state"].verified_port_pairs} == open_ports
    assert count.passed


@pytest.mark.asyncio
async def test_a_host_forwarding_a_low_slice_keeps_its_count(mocker):
    # 1000 working ports at the low end of 25536 declared: the sample finds about 12 of them, so the
    # lowest 300 are probed too and the host publishes what it did before, not 12
    info = _executor()
    service, _ = _service(lambda p: p.internal < 41000, mocker)

    connectivity, count = await _run_checks(service, info)

    state = connectivity.updates["state"]
    assert connectivity.event.context["port_selection"] == SELECTION_STRATIFIED_LOWEST
    assert state.verified_port_count >= BATCH_PORT_VERIFICATION_SIZE
    assert 500 <= state.estimated_usable_port_count <= 1500
    assert count.passed


PREDICATES = {
    "lowest 150 open": lambda p: p.internal < 40150,
    "three low ports": lambda p: p.internal in {40250, 40260, 40270},
    "lowest 300 closed": lambda p: p.internal >= 40300,
    "upper half only": lambda p: p.internal >= 52768,
    "every other port": lambda p: p.internal % 2 == 0,
    "two anywhere": lambda p: p.internal in {41234, 60001},
    "lowest 1000 open": lambda p: p.internal < 41000,
    "lowest 5000 open": lambda p: p.internal < 45000,
    "nothing": lambda p: False,
}


@pytest.mark.asyncio
@pytest.mark.parametrize("name", PREDICATES)
@pytest.mark.parametrize("cycle", range(4))
async def test_sampling_never_publishes_less_than_the_lowest_ports_would(name, cycle, mocker):
    # the count is at least min(what the lowest 300 verify, SAMPLED_PORTS_LOWEST_PASS_BELOW); a host
    # they listed stays listed
    info = _executor()
    is_open = PREDICATES[name]
    listed_before = sum(is_open(p) for p in _lowest_budget(info)) >= MIN_PORT_COUNT
    service, _ = _service(is_open, mocker)

    _, count = await _run_checks(service, info, job_batch_id=f"cycle-{cycle}")

    if listed_before:
        assert count.passed
    assert count.updates["state"].specs["available_port_count"] >= min(
        SAMPLED_PORTS_LOWEST_PASS_BELOW, sum(is_open(p) for p in _lowest_budget(info))
    )


@pytest.mark.asyncio
async def test_a_small_range_is_probed_whole_end_to_end(mocker):
    info = _executor("9000-9099")
    service, _ = _service(lambda p: p.internal % 2 == 0, mocker)

    connectivity, count = await _run_checks(service, info)

    state = connectivity.updates["state"]
    assert state.probed_port_count == state.declared_port_count == 100
    assert state.verified_port_count == state.estimated_usable_port_count == 50
    assert count.updates["state"].specs["available_port_count"] == 50
    assert connectivity.event.context["port_selection"] == SELECTION_ALL
    assert count.passed


@pytest.mark.asyncio
async def test_a_rented_nodes_pod_and_filler_ports_are_never_probed(mocker):
    info = _executor()
    rented = list(range(40000, 65536, 7))
    fillers = list(range(40001, 65536, 11))
    probed: list[PortPair] = []

    def is_open(p: PortPair) -> bool:
        probed.append(p)
        return True

    service, _ = _service(is_open, mocker)

    result = await service.verify_ports(
        AsyncMock(),
        "miner",
        info,
        sysbox_runtime=True,
        rented_ports=rented,
        filler_ports=fillers,
        probe_seed="executor:cycle",
    )

    assert result.status == "ok"
    assert len(result.successful_ports) == BATCH_PORT_VERIFICATION_SIZE
    assert not {p.external for p in probed} & (set(rented) | set(fillers))
    assert result.declared_port_count == 25536
    assert result.probed_port_count == BATCH_PORT_VERIFICATION_SIZE


@pytest.mark.asyncio
async def test_the_cycle_seed_is_the_executor_and_the_job_batch(mocker):
    info = _executor()
    service, _ = _service(lambda p: True, mocker)
    verify = mocker.spy(service, "verify_ports")

    await _run_checks(service, info, job_batch_id="batch-42")

    assert verify.call_args.kwargs["probe_seed"] == f"{info.uuid}:batch-42"
