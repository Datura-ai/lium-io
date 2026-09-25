import random

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo

from services.const import BATCH_PORT_VERIFICATION_SIZE
from services.executor_connectivity import port_selector as port_selector_module
from services.executor_connectivity.dind_probe import DindProbe
from services.executor_connectivity.models import (
    DindProbeResult,
    PortPair,
    PortProbeResult,
    PortRangeResult,
    SecondPass,
)
from services.executor_connectivity.orchestrator import ConnectivityOrchestrator
from services.executor_connectivity.port_probe import PortProbe
from services.executor_connectivity.port_selector import PortSelector, spread_ports, tally_port_ranges
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
    assert tallies[-1].as_dict() == {"pass": 1, "range": "50000-50017", "declared": 18, "probed": 2, "answered": 1}


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
    assert result.second_pass == SecondPass.NO_PORTS_LEFT


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


def test_tally_counts_duplicate_ports_and_shared_externals_once():
    declared = [PortPair(9000, 40000), PortPair(9001, 40000), PortPair(9002, 40001), PortPair(9002, 40001)]

    (tally,) = tally_port_ranges(declared, declared, declared[:2])

    assert (tally.declared, tally.probed, tally.answered) == (2, 2, 1)


def test_tally_drops_ports_outside_1_65535():
    declared = [PortPair(1, 0), PortPair(2, 70000), PortPair(3, -5), PortPair(4, 9000)]

    assert tally_port_ranges(declared, declared, declared) == (
        PortRangeResult(first=9000, last=9000, declared=1, probed=1, answered=1),
    )


def _main_selection(info, size, unavailable):
    """Main's PortSelector.select, verbatim: the lowest `size` free declared ports."""
    all_ports = get_all_ports(info.port_range, info.port_mappings, info.ssh_port)
    available_ports = [
        PortPair(internal, external) for internal, external in all_ports if external not in unavailable
    ]
    return available_ports[:size]


def _random_shapes(seed, count):
    rng = random.Random(seed)
    shapes = []
    for _ in range(count):
        kind = rng.choice(["range", "range", "list", "mappings"])
        if kind == "range":
            lo = rng.randint(1024, 65000)
            hi = min(65535, lo + rng.choice([0, 5, 150, 299, 300, 301, 1000, 25535, 45535]))
            kwargs = {"port_range": f"{lo}-{hi}"}
            pool = list(range(lo, hi + 1))
        elif kind == "list":
            pool = sorted(rng.sample(range(1024, 65535), rng.randint(1, 700)))
            kwargs = {"port_range": ",".join(map(str, rng.sample(pool, len(pool))))}
        else:
            internals = rng.sample(range(1024, 65535), rng.randint(1, 700))
            pairs = [[i, rng.randint(1024, 65535)] for i in internals]
            kwargs = {"port_mappings": str(pairs)}
            pool = [e for _, e in pairs]
        ssh = rng.choice([22, rng.choice(pool)])
        rented = set(rng.sample(pool, min(len(pool), rng.choice([0, 0, 5, 200]))))
        shapes.append((kwargs, ssh, rented))
    return shapes


@pytest.mark.parametrize("kwargs, ssh, rented", _random_shapes(1463, 300))
def test_pass_one_is_mains_selection(kwargs, ssh, rented):
    info = _executor_info(ssh_port=ssh, **kwargs)

    assert PortSelector().select(info, BATCH_PORT_VERIFICATION_SIZE, rented) == _main_selection(
        info, BATCH_PORT_VERIFICATION_SIZE, rented
    )


def test_pass_two_spread_excludes_pass_one_and_rented_ports_and_includes_the_highest():
    info = _executor_info(port_range="40000-65535")
    rented = {40001, 50000, 65535}
    selector = PortSelector()
    declared = _available(info)
    one = selector.select(info, BATCH_PORT_VERIFICATION_SIZE, rented)

    two = selector.select_spread(declared, BATCH_PORT_VERIFICATION_SIZE, rented, pass_one_ports=one)

    assert len(two) == BATCH_PORT_VERIFICATION_SIZE
    assert not {p.external for p in two} & ({p.external for p in one} | rented)
    assert two[0] == PortPair(40301, 40301)
    assert two[-1] == PortPair(65534, 65534)
    assert two == sorted(set(two), key=lambda p: p.internal)


def test_pass_two_is_deterministic_for_the_same_declaration_and_rental_set():
    info = _executor_info(port_range="40000-65535")
    rented = {40001, 50000}
    selector = PortSelector()
    declared = _available(info)
    one = selector.select(info, BATCH_PORT_VERIFICATION_SIZE, rented)
    first = selector.select_spread(declared, BATCH_PORT_VERIFICATION_SIZE, rented, pass_one_ports=one)

    for _ in range(5):
        again_one = selector.select(info, BATCH_PORT_VERIFICATION_SIZE, rented)
        assert selector.select_spread(declared, BATCH_PORT_VERIFICATION_SIZE, rented, pass_one_ports=again_one) == first


@pytest.mark.parametrize("n", [1, 2, 299, 300, 301, 302, 450, 600, 25236, 45236])
def test_spread_is_bounded_distinct_sorted_even_and_keeps_the_highest(n):
    ports = [PortPair(p, p) for p in range(20000, 20000 + n)]

    picked = spread_ports(ports, BATCH_PORT_VERIFICATION_SIZE)

    assert len(picked) == min(n, BATCH_PORT_VERIFICATION_SIZE)
    assert picked == sorted(set(picked), key=lambda p: p.internal)
    assert picked[-1] == ports[-1]
    if n > BATCH_PORT_VERIFICATION_SIZE:
        assert picked[0] == ports[0]
        gaps = [b.internal - a.internal for a, b in zip(picked, picked[1:])]
        assert max(gaps) - min(gaps) <= 1


@pytest.mark.parametrize("size, expected", [(0, []), (1, [9009]), (2, [9000, 9009]), (3, [9000, 9004, 9009])])
def test_spread_tiny_budgets(size, expected):
    ports = [PortPair(p, p) for p in range(9000, 9010)]

    assert [p.internal for p in spread_ports(ports, size)] == expected
