"""DAH-5269: the scrape names each RDMA device's kernel driver, and the EFA probe records hosts whose
live ports run on the `efa` driver."""

import os
from pathlib import Path

from datura.requests.miner_requests import ExecutorSSHInfo
from neurons.validators.tests.helpers import build_scrape_namespace
from services.efa_link_probe import SPEC_KEY, efa_device_count, measure_and_attach
from services.task.models import JobResult

SRC = Path(__file__).resolve().parents[1] / "src"


def _port(device: str, driver: str, state: str = "4: ACTIVE", link_layer: str = "Unknown") -> dict:
    return {
        "device": device,
        "port": "1",
        "driver": driver,
        "state": state,
        "link_layer": link_layer,
    }


def _result(ports: list[dict], is_rented: bool = False) -> JobResult:
    return JobResult(
        spec={"infiniband_ports": ports},
        executor_info=ExecutorSSHInfo(
            uuid="e1",
            address="198.51.100.9",
            port=9001,
            ssh_username="u",
            ssh_port=22,
            python_path="",
            root_dir="",
        ),
        score=1.0,
        job_score=1.0,
        job_batch_id="b",
        log_status="info",
        log_text="ok",
        is_rented=is_rented,
    )


def test_counts_distinct_live_efa_devices_only() -> None:
    result = _result(
        [
            _port("rdmap0", "efa"),
            _port("rdmap1", "efa"),
            _port("rdmap2", "efa", state="1: DOWN"),
            _port("mlx5_0", "mlx5_core", link_layer="Ethernet"),
        ]
    )
    assert efa_device_count(result) == 2


def test_free_efa_host_gets_the_fact() -> None:
    result = _result([_port("rdmap0", "efa"), _port("rdmap1", "efa")])
    measure_and_attach([result], {})
    assert result.spec[SPEC_KEY]["device_count"] == 2
    assert result.spec[SPEC_KEY]["gigabits_per_second"] is None


def test_rented_and_non_efa_hosts_are_left_alone() -> None:
    rented = _result([_port("rdmap0", "efa")], is_rented=True)
    roce = _result([_port("mlx5_0", "mlx5_core", link_layer="Ethernet")])
    bare = _result([])
    bare.spec = None
    measure_and_attach([rented, roce, bare], {})
    assert SPEC_KEY not in rented.spec
    assert SPEC_KEY not in roce.spec
    assert bare.spec is None


def test_scrape_reads_the_device_driver(tmp_path: Path) -> None:
    namespace = build_scrape_namespace(
        SRC / "miner_jobs" / "machine_scrape.py", {"read_driver_name"}, {"os": os}
    )
    driver_dir = tmp_path / "bus" / "pci" / "drivers" / "efa"
    driver_dir.mkdir(parents=True)
    (tmp_path / "rdmap0" / "device").mkdir(parents=True)
    (tmp_path / "rdmap0" / "device" / "driver").symlink_to(driver_dir)

    assert namespace["read_driver_name"](str(tmp_path / "rdmap0")) == "efa"
    assert namespace["read_driver_name"](str(tmp_path / "missing")) == ""
