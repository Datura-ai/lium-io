import time
from unittest.mock import AsyncMock

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo
from services import efa_link_probe
from services.task.models import JobResult

EFA_PORT = {"device": "rdmap87s0", "driver": "efa", "link_layer": "Unknown", "state": "4: ACTIVE"}
SELF_MANAGED_IB_PORT = {"device": "mlx5_0", "driver": "mlx5_core", "link_layer": "InfiniBand", "state": "4: ACTIVE"}

FI_PINGPONG_OUTPUT = """bytes   #sent   #ack     total       time     MB/sec    usec/xfer   Mxfers/sec
64k     1000    =1000    125m        0.10s   1250.00      52.43       0.02
"""


def host(uuid: str, ports: list[dict], is_rented: bool = False) -> JobResult:
    return JobResult(
        executor_info=ExecutorSSHInfo(
            uuid=uuid,
            address="10.0.0.1",
            port=8080,
            ssh_username="root",
            ssh_port=22,
            python_path="/usr/bin/python3",
            root_dir="/tmp",
        ),
        score=0,
        job_score=0,
        job_batch_id="batch",
        log_status="info",
        log_text="",
        is_rented=is_rented,
        spec={"infiniband_ports": ports},
    )


def test_every_pair_of_free_efa_hosts_is_measured_and_rented_or_non_efa_hosts_are_not():
    hosts = [
        host("c", [SELF_MANAGED_IB_PORT, EFA_PORT]),
        host("a", [EFA_PORT]),
        host("b", [EFA_PORT]),
        host("rented", [EFA_PORT], is_rented=True),
        host("ib-only", [SELF_MANAGED_IB_PORT]),
        host("efa-down", [{**EFA_PORT, "state": "1: DOWN"}]),
    ]

    pairs = efa_link_probe.pairs_to_measure(hosts)

    assert [(s.executor_info.uuid, c.executor_info.uuid) for s, c in pairs] == [("a", "b"), ("a", "c"), ("b", "c")]


def test_the_rate_is_read_from_fi_pingpongs_table_and_a_run_without_one_measures_nothing():
    assert efa_link_probe.measured_gigabits_per_second(FI_PINGPONG_OUTPUT) == 10.0
    assert efa_link_probe.measured_gigabits_per_second("fi_getinfo(): No data available\n") is None


@pytest.mark.asyncio
async def test_each_pair_is_recorded_on_both_hosts_under_its_own_spec_key(monkeypatch):
    a, b = host("a", [EFA_PORT]), host("b", [EFA_PORT])

    async def measure(server, client, key, addresses):
        addresses.update({"a": "10.0.1.1", "b": "10.0.1.2"})
        return 10.0

    monkeypatch.setattr(efa_link_probe, "_measure_pair", AsyncMock(side_effect=measure))

    await efa_link_probe.measure_and_attach([a, b], "key", {}, time.monotonic())

    assert [(m["peer_executor_uuid"], m["gigabits_per_second"]) for m in a.spec["efa_link_measurements"]] == [("b", 10.0)]
    assert [(m["peer_executor_uuid"], m["peer_address"]) for m in b.spec["efa_link_measurements"]] == [("a", "10.0.1.1")]
    assert "roce_link_measurements" not in a.spec


@pytest.mark.asyncio
async def test_a_probe_error_records_a_failed_pair_and_never_reaches_the_miners_job(monkeypatch):
    a, b = host("a", [EFA_PORT]), host("b", [EFA_PORT])
    monkeypatch.setattr(efa_link_probe, "_measure_pair", AsyncMock(side_effect=OSError("ssh closed")))

    await efa_link_probe.measure_and_attach([a, b], "key", {}, time.monotonic())

    assert a.spec["efa_link_measurements"][0]["gigabits_per_second"] is None
