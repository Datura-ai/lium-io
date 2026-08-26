"""DAH-2667 — the validator measures a RoCE fabric before the backend sells it as a cluster.

Port shapes are the ones prod executors report (measured 2026-08-05): mlx5 answers on an Ethernet
link layer and carries the host's address as the IPv4-mapped entry of the GID table.
"""

import asyncio

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo

from services import roce_link_probe
from services.roce_link_probe import (
    _client_command,
    _server_command,
    RoceLinkMeasurement,
    attach_measurement,
    measure_and_attach,
    measured_gigabits_per_second,
    pairs_to_measure,
    roce_fabric_host_of,
)
from services.task.models import JobResult

LINK_LOCAL_GID = "fe80:0000:0000:0000:0e42:a1ff:fe4c:9a10"


def roce_port(address: str, device: str = "mlx5_0") -> dict:
    return {
        "device": device,
        "port": "1",
        "link_layer": "Ethernet",
        "state": "4: ACTIVE",
        "gids": [LINK_LOCAL_GID, f"0000:0000:0000:0000:0000:ffff:{address}"],
    }


def infiniband_port() -> dict:
    return {
        "device": "mlx5_1",
        "port": "1",
        "link_layer": "InfiniBand",
        "state": "4: ACTIVE",
        "gids": [LINK_LOCAL_GID],
    }


def job_result(uuid: str, ports: list[dict], *, is_rented: bool = False) -> JobResult:
    return JobResult(
        executor_info=ExecutorSSHInfo(
            uuid=uuid,
            address="203.0.113.10",
            port=8080,
            ssh_username="root",
            ssh_port=22,
            python_path="/usr/bin/python3",
            root_dir="/root/app",
        ),
        spec={"infiniband_ports": ports},
        score=1.0,
        job_score=1.0,
        job_batch_id="batch-1",
        log_status="success",
        log_text="",
        is_rented=is_rented,
    )


def test_a_host_with_one_live_roce_rail_names_its_address_and_segment():
    host = roce_fabric_host_of(job_result("a", [roce_port("ac10:0506")]))

    assert host is not None
    assert host.roce_address == "172.16.5.6"
    assert host.segment == "172.16.5.0/24"


def test_a_host_that_also_holds_a_live_infiniband_port_is_not_measured():
    assert roce_fabric_host_of(job_result("a", [roce_port("ac10:0506"), infiniband_port()])) is None


def test_a_host_whose_rails_answer_on_two_segments_is_not_measured():
    ports = [roce_port("ac10:0506"), roce_port("0a00:0007", device="mlx5_2")]

    assert roce_fabric_host_of(job_result("a", ports)) is None


def test_a_host_reporting_no_rdma_port_is_not_measured():
    assert roce_fabric_host_of(job_result("a", [])) is None


def test_two_free_hosts_on_one_segment_make_one_pair():
    results = [job_result("a", [roce_port("ac10:0506")]), job_result("b", [roce_port("ac10:0507")])]

    pairs = pairs_to_measure(results)

    assert [(server.executor_info.uuid, client.executor_info.uuid) for server, client in pairs] == [("a", "b")]


def test_a_rented_host_is_never_measured():
    results = [
        job_result("a", [roce_port("ac10:0506")]),
        job_result("b", [roce_port("ac10:0507")], is_rented=True),
    ]

    assert pairs_to_measure(results) == []


def test_hosts_on_different_segments_are_not_paired():
    results = [job_result("a", [roce_port("ac10:0506")]), job_result("b", [roce_port("0a00:0007")])]

    assert pairs_to_measure(results) == []


def test_two_hosts_claiming_one_address_are_not_paired():
    results = [job_result("a", [roce_port("ac10:0506")]), job_result("b", [roce_port("ac10:0506")])]

    assert pairs_to_measure(results) == []


def test_four_hosts_on_one_segment_measure_every_pair():
    results = [
        job_result("d", [roce_port("ac10:0509")]),
        job_result("a", [roce_port("ac10:0506")]),
        job_result("c", [roce_port("ac10:0508")]),
        job_result("b", [roce_port("ac10:0507")]),
    ]

    pairs = pairs_to_measure(results)

    assert [(server.executor_info.uuid, client.executor_info.uuid) for server, client in pairs] == [
        ("a", "b"),
        ("a", "c"),
        ("a", "d"),
        ("b", "c"),
        ("b", "d"),
        ("c", "d"),
    ]


def test_an_odd_host_on_a_segment_is_still_measured_against_both_others():
    results = [
        job_result("a", [roce_port("ac10:0506")]),
        job_result("b", [roce_port("ac10:0507")]),
        job_result("c", [roce_port("ac10:0508")]),
    ]

    assert len(pairs_to_measure(results)) == 3


def test_the_clients_table_yields_the_average_gigabits():
    output = (
        " #bytes     #iterations    BW peak[Gb/sec]    BW average[Gb/sec]   MsgRate[Mpps]\n"
        " 65536      5000             92.99              92.51              0.176478\n"
        "---------------------------------------------------------------------------------------\n"
    )

    assert measured_gigabits_per_second(output) == pytest.approx(92.51)


def test_a_run_that_printed_no_table_measured_nothing():
    assert measured_gigabits_per_second("Unable to open device mlx5_0\n") is None


def _measurement(peer_uuid: str, peer_address: str, gigabits: float | None) -> RoceLinkMeasurement:
    return RoceLinkMeasurement(
        peer_executor_uuid=peer_uuid,
        peer_address=peer_address,
        gigabits_per_second=gigabits,
        measured_at="2026-08-25T13:00:00+00:00",
    )


def test_the_measurement_lands_in_the_spec_of_the_host():
    result = job_result("a", [roce_port("ac10:0506")])

    attach_measurement(result, _measurement("b", "172.16.5.7", 92.51))

    assert result.spec["roce_link_measurements"] == [
        {
            "peer_executor_uuid": "b",
            "peer_address": "172.16.5.7",
            "gigabits_per_second": 92.51,
            "measured_at": "2026-08-25T13:00:00+00:00",
        }
    ]


def test_every_peer_of_the_segment_keeps_its_own_entry():
    result = job_result("a", [roce_port("ac10:0506")])

    attach_measurement(result, _measurement("b", "172.16.5.7", 92.51))
    attach_measurement(result, _measurement("c", "172.16.5.8", 91.02))

    assert [entry["peer_address"] for entry in result.spec["roce_link_measurements"]] == [
        "172.16.5.7",
        "172.16.5.8",
    ]


def test_a_second_run_against_one_peer_replaces_that_peers_entry():
    result = job_result("a", [roce_port("ac10:0506")])

    attach_measurement(result, _measurement("b", "172.16.5.7", 92.51))
    attach_measurement(result, _measurement("b", "172.16.5.7", None))

    assert result.spec["roce_link_measurements"] == [
        {
            "peer_executor_uuid": "b",
            "peer_address": "172.16.5.7",
            "gigabits_per_second": None,
            "measured_at": "2026-08-25T13:00:00+00:00",
        }
    ]


def test_a_pair_that_could_not_talk_is_recorded_as_a_failure():
    result = job_result("a", [roce_port("ac10:0506")])

    attach_measurement(result, _measurement("b", "172.16.5.7", None))

    assert result.spec["roce_link_measurements"][0]["gigabits_per_second"] is None


@pytest.mark.asyncio
async def test_a_sweep_that_hangs_gives_up_instead_of_failing_the_miners_cycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The caller's timeout cancels the whole job and scores the miner zero, so the probe stops first."""

    async def never_finishes(*_args: object, **_kwargs: object) -> None:
        await asyncio.sleep(60)

    monkeypatch.setattr(roce_link_probe.settings, "ROCE_LINK_PROBE_ENABLED", True)
    monkeypatch.setattr(roce_link_probe, "PROBE_BUDGET_SECONDS", 0.05)
    monkeypatch.setattr(roce_link_probe, "_measure_every_pair", never_finishes)
    results = [job_result("a", [roce_port("ac10:0506")]), job_result("b", [roce_port("ac10:0507")])]

    await measure_and_attach(results, "private-key", {})

    assert all(result.spec.get("roce_link_measurements") is None for result in results)


def test_both_probe_containers_can_pin_the_memory_a_real_card_registers():
    """`ibv_reg_mr` fails without these, and Soft-RoCE never shows it (DAH-2571)."""
    for command in (_server_command(), _client_command("10.0.0.5")):
        assert "--cap-add IPC_LOCK" in command
        assert "--ulimit memlock=-1:-1" in command


@pytest.mark.asyncio
async def test_a_probe_that_raises_never_reaches_the_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    """An exception here would abort the miner's whole job and score it zero for the cycle."""

    def explode(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("specs are not what the probe expected")

    monkeypatch.setattr(roce_link_probe.settings, "ROCE_LINK_PROBE_ENABLED", True)
    monkeypatch.setattr(roce_link_probe, "pairs_to_measure", explode)

    await measure_and_attach([job_result("a", [roce_port("ac10:0506")])], "private-key", {})
