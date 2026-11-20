"""The scrape names each RDMA device's kernel driver (e.g. AWS EFA ports, which report no IB/Ethernet link layer)."""

import os
from pathlib import Path

from neurons.validators.tests.helpers import build_scrape_namespace

SRC = Path(__file__).resolve().parents[1] / "src"


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
