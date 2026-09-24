
import pytest
from datura.requests.miner_requests import ExecutorSSHInfo

from services.const import BATCH_PORT_VERIFICATION_SIZE, MIN_PORT_COUNT
from services.executor_connectivity.models import PortPair
from services.executor_connectivity.port_selector import PortSelector
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


def test_port_selector_wide_range_reaches_ports_forwarded_only_at_the_top():
    info = _executor_info(port_range="40000-65535")
    open_ports = set(range(60000, 65536))

    result = PortSelector().select(info, size=BATCH_PORT_VERIFICATION_SIZE, unavailable_ports=set())

    all_ports = get_all_ports(info.port_range, info.port_mappings, info.ssh_port)
    old_selection = all_ports[:BATCH_PORT_VERIFICATION_SIZE]
    assert sum(internal in open_ports for internal, _ in old_selection) < MIN_PORT_COUNT
    assert sum(p.internal in open_ports for p in result) >= MIN_PORT_COUNT
    assert len(result) == BATCH_PORT_VERIFICATION_SIZE
    assert len({p.internal for p in result}) == BATCH_PORT_VERIFICATION_SIZE
    assert result[0].internal == 40000
    assert result[-1].internal == 65535
    assert [PortPair(p, p) for p in range(40000, 40150)] == result[:150]


@pytest.mark.parametrize("port_range", ["9000-9299", "9000-9099", "9000,9005,9010"])
def test_port_selector_small_range_selects_the_same_ports_as_before(port_range):
    info = _executor_info(port_range=port_range)

    result = PortSelector().select(
        info, size=BATCH_PORT_VERIFICATION_SIZE, unavailable_ports={9001}
    )

    all_ports = get_all_ports(info.port_range, info.port_mappings, info.ssh_port)
    before = [PortPair(i, e) for i, e in all_ports if e != 9001][:BATCH_PORT_VERIFICATION_SIZE]
    assert result == before


def test_port_selector_wide_mappings_sample_across_all_pairs_after_unavailable():
    mappings = str([[p, p + 10000] for p in range(20000, 21000)])
    info = _executor_info(port_mappings=mappings)

    result = PortSelector().select(info, size=300, unavailable_ports={30000})

    assert len(result) == 300
    assert all(p.external == p.internal + 10000 and p.external != 30000 for p in result)
    assert result[0] == PortPair(20001, 30001)
    assert result[-1] == PortPair(20999, 30999)
    assert [p.internal for p in result] == sorted({p.internal for p in result})


def test_port_selector_empty_when_all_rented():
    info = _executor_info(port_range="9000-9001")

    selector = PortSelector()
    result = selector.select(info, size=2, unavailable_ports={9000, 9001})

    assert result == []
