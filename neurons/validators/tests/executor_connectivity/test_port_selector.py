import pytest
from datura.requests.miner_requests import ExecutorSSHInfo

from services.const import BATCH_PORT_VERIFICATION_SIZE, MIN_PORT_COUNT, PORT_RANGE_MAX_ENTRIES
from services.executor_connectivity import port_selector as port_selector_module
from services.executor_connectivity.dind_probe import DindProbe
from services.executor_connectivity.models import (
    DindProbeResult,
    PortPair,
    PortProbeResult,
    PortRangeResult,
)
from services.executor_connectivity.orchestrator import ConnectivityOrchestrator
from services.executor_connectivity.port_probe import PortProbe
from services.executor_connectivity.port_selector import PortSelector, sample_ports, tally_port_ranges
from services.port_utils import get_all_ports


def _executor_info(*, port_mappings=None, port_range=None, ssh_port=22):
    return ExecutorSSHInfo(
        uuid="executor-1",
        address="127.0.0.1",
        port=8080,
        ssh_username="root",
        ssh_port=ssh_port,
        port_mappings=port_mappings,
        port_range=port_range,
        python_path="/usr/bin/python3",
        root_dir="/tmp",
    )


def test_port_selector_from_mappings_excludes_ssh_and_rented():
    mappings = str([[22, 2200], [9000, 9000], [9001, 9001], [9002, 9002]])
    info = _executor_info(port_mappings=mappings)

    selector = PortSelector()
    result = selector.select(info, size=5, unavailable_ports={9001})

    ports = {(p.internal, p.external) for p in result}
    assert (22, 2200) not in ports
    assert (9001, 9001) not in ports
    assert (9000, 9000) in ports
    assert (9002, 9002) in ports


def test_port_selector_from_range_uses_range_list():
    info = _executor_info(port_range="9000,9001,9002")

    selector = PortSelector()
    result = selector.select(info, size=2, unavailable_ports=set())

    assert len(result) == 2
    for port in result:
        assert port.internal in {9000, 9001, 9002}
        assert port.external == port.internal


def test_port_selector_from_range_excludes_ssh_port():
    info = _executor_info(port_range="22,9000,9001")

    selector = PortSelector()
    result = selector.select(info, size=2, unavailable_ports=set())

    assert all(p.internal != 22 for p in result)


def test_port_selector_default_range_when_missing():
    info = _executor_info(port_range=None)

    selector = PortSelector()
    result = selector.select(info, size=3, unavailable_ports=set())

    assert len(result) == 3
    for port in result:
        assert 20000 <= port.internal <= 65535
        assert port.external == port.internal


def test_port_selector_empty_when_all_rented():
    info = _executor_info(port_range="9000-9001")

    selector = PortSelector()
    result = selector.select(info, size=2, unavailable_ports={9000, 9001})

    assert result == []


def _available(info, unavailable=frozenset()):
    """Every declared pair the selector may pick from, in the order it sees them."""
    return [
        PortPair(i, e)
        for i, e in get_all_ports(info.port_range, info.port_mappings, info.ssh_port)
        if e not in unavailable
    ]


def test_wide_range_open_only_high_regression():
    """Declared 40000-65535, only ports above 60000 forwarded: the lowest-300 cut verified none of
    them and the host was hidden from the listing though it passed everything else."""
    info = _executor_info(port_range="40000-65535")
    open_ports = set(range(60001, 65536))

    old = _available(info)[:BATCH_PORT_VERIFICATION_SIZE]
    new = PortSelector().select(info, BATCH_PORT_VERIFICATION_SIZE, set())

    assert sum(p.external in open_ports for p in old) < MIN_PORT_COUNT
    assert sum(p.external in open_ports for p in new) >= MIN_PORT_COUNT
    assert len(new) == len(old) == BATCH_PORT_VERIFICATION_SIZE
    assert new[-1] == PortPair(65535, 65535)


@pytest.mark.asyncio
async def test_wide_range_open_only_high_orchestrator_verifies_high_ports(mocker):
    info = _executor_info(port_range="40000-65535")
    open_ports = set(range(60001, 65536))

    async def probe(ports, **_):
        return PortProbeResult(
            successful=tuple(p for p in ports if p.external in open_ports),
            failed=tuple(p for p in ports if p.external not in open_ports),
        )

    port_probe = mocker.Mock(spec=PortProbe)
    port_probe.probe = mocker.AsyncMock(side_effect=probe)
    dind_probe = mocker.Mock(spec=DindProbe)
    dind_probe.verify = mocker.AsyncMock(
        return_value=DindProbeResult(success=True, sysbox_runtime=False, port=None)
    )

    result = await ConnectivityOrchestrator(PortSelector(), port_probe, dind_probe).verify(
        executor_info=info,
        miner_hotkey="miner",
        sysbox_runtime=False,
        unavailable_ports=[],
        ssh_client=mocker.Mock(),
    )

    assert result.status == "ok"
    assert len(result.successful_ports) >= MIN_PORT_COUNT
    assert all(p.external in open_ports for p in result.successful_ports)

    ranges = {r.first: r for r in result.port_ranges}
    assert sum(r.declared for r in result.port_ranges) == 65535 - 40000 + 1
    assert sum(r.probed for r in result.port_ranges) == BATCH_PORT_VERIFICATION_SIZE
    assert sum(r.answered for r in result.port_ranges) == len(result.successful_ports)
    assert all(ranges[first].answered == 0 for first in (40000, 45000, 50000, 55000))
    assert ranges[60000].answered > 0 and ranges[65000].answered > 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"port_range": "9000-9299"},
        {"port_range": "9000-9100"},
        {"port_range": "9000-9300", "ssh_port": 9150},
        {"port_range": "9005,9001,9003,9002"},
        {"port_range": "9000"},
        {"port_mappings": str([[22, 2200]] + [[8000 + i, 30000 + i] for i in range(299)])},
    ],
)
def test_small_declaration_selects_exactly_todays_ports(kwargs):
    """A declaration of at most BATCH_PORT_VERIFICATION_SIZE ports is probed in full, unchanged."""
    info = _executor_info(**kwargs)
    available = _available(info)
    assert len(available) <= BATCH_PORT_VERIFICATION_SIZE

    new = PortSelector().select(info, BATCH_PORT_VERIFICATION_SIZE, set())

    assert new == available[:BATCH_PORT_VERIFICATION_SIZE]


def test_small_declaration_with_rented_ports_selects_todays_ports():
    info = _executor_info(port_range="9000-9400")
    unavailable = set(range(9000, 9150))
    available = _available(info, unavailable)
    assert len(available) <= BATCH_PORT_VERIFICATION_SIZE

    assert PortSelector().select(info, BATCH_PORT_VERIFICATION_SIZE, unavailable) == available


def test_wide_declaration_keeps_lowest_half_so_low_forwarding_still_passes():
    """A host that forwards only the bottom of a wide range passes today; it must keep passing."""
    info = _executor_info(port_range="40000-65535")
    open_ports = set(range(40000, 40010))

    new = PortSelector().select(info, BATCH_PORT_VERIFICATION_SIZE, set())

    head = BATCH_PORT_VERIFICATION_SIZE // 2
    assert new[:head] == _available(info)[:head]
    assert sum(p.external in open_ports for p in new) == len(open_ports)


@pytest.mark.parametrize("n", [301, 302, 450, 599, 600, 601, 25536, 45535])
def test_sample_is_bounded_distinct_sorted_and_spans_both_ends(n):
    ports = [PortPair(p, p) for p in range(20000, 20000 + n)]

    picked = sample_ports(ports, BATCH_PORT_VERIFICATION_SIZE)

    assert len(picked) == BATCH_PORT_VERIFICATION_SIZE
    assert picked == sorted(set(picked), key=lambda p: p.internal)
    assert picked[0] == ports[0]
    assert picked[-1] == ports[-1]
    head = BATCH_PORT_VERIFICATION_SIZE // 2
    assert picked[:head] == ports[:head]
    gaps = [b.internal - a.internal for a, b in zip(picked[head:], picked[head + 1 :])]
    assert max(gaps) - min(gaps) <= 1


def test_sample_is_deterministic_across_cycles():
    info = _executor_info(port_range="40000-65535")
    selector = PortSelector()
    rented = {40001, 50000}
    first = selector.select(info, BATCH_PORT_VERIFICATION_SIZE, rented)

    for _ in range(5):
        assert selector.select(info, BATCH_PORT_VERIFICATION_SIZE, rented) == first


@pytest.mark.parametrize("size", [0, 1, 2, 3])
def test_sample_tiny_budgets_match_lowest_cut(size):
    ports = [PortPair(p, p) for p in range(9000, 9010)]
    if size <= 2:
        assert sample_ports(ports, size) == ports[:size]
    else:
        assert sample_ports(ports, size) == [ports[0], ports[1], ports[-1]]


def test_sample_empty_list():
    assert sample_ports([], BATCH_PORT_VERIFICATION_SIZE) == []


def test_tally_small_range_is_one_entry():
    declared = [PortPair(p, p) for p in range(9000, 9101)]
    probed = declared[:50]
    answered = declared[:3]

    assert tally_port_ranges(declared, probed, answered) == (
        PortRangeResult(first=9000, last=9100, declared=101, probed=50, answered=3),
    )


def test_tally_wide_range_splits_at_bucket_boundaries_by_external_port():
    # the offset is not a multiple of the bucket width, so bucketing by internal port gives other bounds
    declared = [PortPair(8000 + i, 38017 + i) for i in range(12001)]

    tallies = tally_port_ranges(declared, declared[-2:], declared[-1:])

    assert [(t.first, t.last, t.declared) for t in tallies] == [
        (38017, 39999, 1983),
        (40000, 44999, 5000),
        (45000, 49999, 5000),
        (50000, 50017, 18),
    ]
    assert [(t.probed, t.answered) for t in tallies] == [(0, 0), (0, 0), (0, 0), (2, 1)]
    assert tallies[-1].as_dict() == {"range": "50000-50017", "declared": 18, "probed": 2, "answered": 1}


def test_tally_nothing_declared():
    assert tally_port_ranges([], [], []) == ()


@pytest.mark.asyncio
async def test_orchestrator_no_ports_still_reports_declared_ranges(mocker):
    info = _executor_info(port_range="9000-9001")
    port_probe = mocker.Mock(spec=PortProbe)
    dind_probe = mocker.Mock(spec=DindProbe)

    result = await ConnectivityOrchestrator(PortSelector(), port_probe, dind_probe).verify(
        executor_info=info,
        miner_hotkey="miner",
        sysbox_runtime=False,
        unavailable_ports=[9000, 9001],
        ssh_client=mocker.Mock(),
    )

    assert result.status == "no_ports"
    assert result.port_ranges == (PortRangeResult(first=9000, last=9001, declared=2, probed=0, answered=0),)


def test_select_calls_sample_ports(monkeypatch):
    """Production's selection is sample_ports: a select() that inlines its own copy fails here."""
    calls = []

    def spy(ports, size):
        calls.append((list(ports), size))
        return sample_ports(ports, size)

    monkeypatch.setattr(port_selector_module, "sample_ports", spy)
    info = _executor_info(port_range="40000-65535")

    result = PortSelector().select(info, BATCH_PORT_VERIFICATION_SIZE, {40001})

    assert calls == [(_available(info, {40001}), BATCH_PORT_VERIFICATION_SIZE)]
    assert result == sample_ports(_available(info, {40001}), BATCH_PORT_VERIFICATION_SIZE)


@pytest.mark.asyncio
async def test_orchestrator_parses_the_declaration_once(monkeypatch, mocker):
    parsed = []

    def counting_get_all_ports(*args):
        parsed.append(args)
        return get_all_ports(*args)

    monkeypatch.setattr(port_selector_module, "get_all_ports", counting_get_all_ports)
    port_probe = mocker.Mock(spec=PortProbe)
    port_probe.probe = mocker.AsyncMock(return_value=PortProbeResult(successful=(), failed=()))
    dind_probe = mocker.Mock(spec=DindProbe)
    dind_probe.verify = mocker.AsyncMock(
        return_value=DindProbeResult(success=False, sysbox_runtime=False, port=None)
    )

    await ConnectivityOrchestrator(PortSelector(), port_probe, dind_probe).verify(
        executor_info=_executor_info(port_range="40000-65535"),
        miner_hotkey="miner",
        sysbox_runtime=False,
        unavailable_ports=[],
        ssh_client=mocker.Mock(),
    )

    assert len(parsed) == 1


def _answering(info, open_ports):
    return sum(p.external in open_ports for p in PortSelector().select(info, BATCH_PORT_VERIFICATION_SIZE, set()))


@pytest.mark.parametrize("block, verified", [(342, 3), (341, 2), (513, 4)])
def test_limit_top_block_needs_about_342_ports_on_40000_65535(block, verified):
    info = _executor_info(port_range="40000-65535")

    assert _answering(info, set(range(65536 - block, 65536))) == verified


def test_limit_block_at_positions_151_to_300_of_a_wide_range_is_newly_hidden():
    info = _executor_info(port_range="40000-65535")
    open_ports = set(range(40150, 40300))

    old = _available(info)[:BATCH_PORT_VERIFICATION_SIZE]
    assert sum(p.external in open_ports for p in old) == 150
    assert _answering(info, open_ports) < MIN_PORT_COUNT


def test_limit_low_block_of_300_verifies_151():
    info = _executor_info(port_range="40000-65535")

    assert _answering(info, set(range(40000, 40300))) == 151


def test_tally_counts_duplicate_ports_and_shared_externals_once():
    declared = [PortPair(9000, 40000), PortPair(9001, 40000), PortPair(9002, 40001), PortPair(9002, 40001)]

    (tally,) = tally_port_ranges(declared, declared, declared[:2])

    assert (tally.declared, tally.probed, tally.answered) == (2, 2, 1)


def test_tally_drops_ports_outside_1_65535():
    declared = [PortPair(1, 0), PortPair(2, 70000), PortPair(3, -5), PortPair(4, 9000)]

    assert tally_port_ranges(declared, declared, declared) == (
        PortRangeResult(first=9000, last=9000, declared=1, probed=1, answered=1),
    )


def test_tally_caps_entries_with_an_other_bucket(monkeypatch):
    monkeypatch.setattr(port_selector_module, "PORT_RANGE_BUCKET_WIDTH", 10)
    declared = [PortPair(p, p) for p in range(10000, 10000 + 10 * 40, 10)]

    tallies = tally_port_ranges(declared, declared[-3:], declared[-1:])

    assert len(tallies) == PORT_RANGE_MAX_ENTRIES
    assert not any(t.other for t in tallies[:-1])
    other = tallies[-1]
    assert (other.other, other.first, other.last, other.declared) == (True, 10310, 10390, 9)
    assert (other.probed, other.answered) == (3, 1)
    assert other.as_dict()["range"] == "other 10310-10390"
    assert sum(t.declared for t in tallies) == len(declared)
