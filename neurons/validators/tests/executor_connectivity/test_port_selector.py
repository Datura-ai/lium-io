import pytest
from datura.requests.miner_requests import ExecutorSSHInfo

from services.const import BATCH_PORT_VERIFICATION_SIZE, MIN_PORT_COUNT
from services.executor_connectivity.dind_probe import DindProbe
from services.executor_connectivity.models import DindProbeResult, PortPair, PortProbeResult
from services.executor_connectivity.orchestrator import ConnectivityOrchestrator
from services.executor_connectivity.port_probe import PortProbe
from services.executor_connectivity.port_selector import PortSelector, sample_ports
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


def test_reston_regression_wide_range_open_only_high():
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
async def test_reston_regression_orchestrator_verifies_high_ports(mocker):
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
