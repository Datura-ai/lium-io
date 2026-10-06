import pytest
from core.config import settings
from datura.requests.miner_requests import ExecutorSSHInfo

from services.const import BATCH_PORT_VERIFICATION_SIZE
from services.executor_connectivity.dind_probe import DindProbe
from services.executor_connectivity.models import (
    PortPair,
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
    result = selector.select(info, size=5, unavailable_ports={9001}, declared=_available(info))

    ports = {(p.internal, p.external) for p in result}
    assert (22, 2200) not in ports
    assert (9001, 9001) not in ports
    assert (9000, 9000) in ports
    assert (9002, 9002) in ports


def test_port_selector_from_range_uses_range_list():
    info = _executor_info(port_range="9000,9001,9002")

    selector = PortSelector()
    result = selector.select(info, size=2, unavailable_ports=set(), declared=_available(info))

    assert len(result) == 2
    for port in result:
        assert port.internal in {9000, 9001, 9002}
        assert port.external == port.internal


def test_port_selector_from_range_excludes_ssh_port():
    info = _executor_info(port_range="22,9000,9001")

    selector = PortSelector()
    result = selector.select(info, size=2, unavailable_ports=set(), declared=_available(info))

    assert all(p.internal != 22 for p in result)


def test_port_selector_default_range_when_missing():
    info = _executor_info(port_range=None)

    selector = PortSelector()
    result = selector.select(info, size=3, unavailable_ports=set(), declared=_available(info))

    assert len(result) == 3
    for port in result:
        assert 20000 <= port.internal <= 65535
        assert port.external == port.internal


def test_port_selector_empty_when_all_rented():
    info = _executor_info(port_range="9000-9001")

    selector = PortSelector()
    result = selector.select(info, size=2, unavailable_ports={9000, 9001}, declared=_available(info))

    assert result == []


def _available(info, unavailable=frozenset()):
    """Every declared pair the selector may pick from, in the order it sees them."""
    return [
        PortPair(i, e)
        for i, e in get_all_ports(info.port_range, info.port_mappings, info.ssh_port)
        if e not in unavailable
    ]


def two_passes(info, unavailable=frozenset()):
    """The declared ports and the two passes' picks, as the orchestrator selects them."""
    selector, declared = PortSelector(), _available(info)
    one = selector.select(info, BATCH_PORT_VERIFICATION_SIZE, set(unavailable), declared=declared)
    two = selector.select_spread(declared, BATCH_PORT_VERIFICATION_SIZE, set(unavailable), pass_one_ports=one)
    return declared, one, two


_WIDE = [PortPair(8000 + i, 38017 + i) for i in range(12001)]
_SMALL = [PortPair(p, p) for p in range(9000, 9101)]
_ODD = [PortPair(9000, 40000), PortPair(9001, 40000), PortPair(9002, 40001), PortPair(9002, 40001)]
_OUTSIDE = [PortPair(1, 0), PortPair(2, 70000), PortPair(3, -5), PortPair(4, 9000)]


@pytest.mark.parametrize(
    "declared, probed, answered, expected",
    [
        pytest.param(_SMALL, _SMALL[:50], _SMALL[:3], [(9000, 9100, 101, 50, 3)], id="small-range-is-one-entry"),
        # the offset is not a multiple of the bucket width, so bucketing by internal port gives other bounds
        pytest.param(
            _WIDE, _WIDE[-2:], _WIDE[-1:],
            [(38017, 39999, 1983, 0, 0), (40000, 44999, 5000, 0, 0), (45000, 49999, 5000, 0, 0), (50000, 50017, 18, 2, 1)],
            id="wide-range-splits-at-external-bucket-bounds",
        ),
        pytest.param([], [], [], [], id="nothing-declared"),
        pytest.param(_ODD, _ODD, _ODD[:2], [(40000, 40001, 2, 2, 1)], id="duplicates-and-shared-externals-count-once"),
        pytest.param(_OUTSIDE, _OUTSIDE, _OUTSIDE, [(9000, 9000, 1, 1, 1)], id="outside-1-65535-dropped"),
    ],
)
def test_tally_port_ranges(declared, probed, answered, expected):
    assert tally_port_ranges(declared, probed, answered) == tuple(
        PortRangeResult(first=lo, last=hi, declared=d, probed=p, answered=a) for lo, hi, d, p, a in expected
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("topup", [False, True])
async def test_orchestrator_no_ports_still_reports_declared_ranges(mocker, monkeypatch, topup):
    monkeypatch.setattr(settings, "PORT_PROBE_TOPUP_BELOW_FLOOR", topup)
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
    assert result.second_pass == (SecondPass.NO_PORTS_LEFT if topup else None)


def test_pass_two_spread_excludes_pass_one_and_rented_ports_and_includes_the_highest():
    info = _executor_info(port_range="40000-65535")
    rented = {40001, 50000, 65535}

    _, one, two = two_passes(info, rented)

    assert len(two) == BATCH_PORT_VERIFICATION_SIZE
    assert not {p.external for p in two} & ({p.external for p in one} | rented)
    assert two[0] == PortPair(40301, 40301)
    assert two[-1] == PortPair(65534, 65534)
    assert two == sorted(set(two), key=lambda p: p.internal)


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
