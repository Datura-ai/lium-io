import pytest

from datura.requests.miner_requests import ExecutorSSHInfo

from services.executor_connectivity.models import PortPair
from services.executor_connectivity.port_selector import PortSelector


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
