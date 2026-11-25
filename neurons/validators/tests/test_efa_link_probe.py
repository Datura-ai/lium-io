import asyncio
import shlex
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo
from services import efa_link_probe
from services.task.models import JobResult

EFA_PORT = {"device": "rdmap87s0", "driver": "efa", "link_layer": "Unknown", "state": "4: ACTIVE"}
SELF_MANAGED_IB_PORT = {"device": "mlx5_0", "driver": "mlx5_core", "link_layer": "InfiniBand", "state": "4: ACTIVE"}

FI_PINGPONG_OUTPUT = """bytes   #sent   #ack     total       time     MB/sec    usec/xfer   Mxfers/sec
64k     1000    =1000    125m        0.10s   1250.00      52.43       0.02
"""


@pytest.fixture(autouse=True)
def reviewed_probe_digest(monkeypatch):
    monkeypatch.setattr(efa_link_probe, "EFA_PROBE_IMAGE_DIGEST", "sha256:" + "0" * 64)


def host(uuid: str, ports: list[dict], is_rented: bool = False) -> JobResult:
    return JobResult(
        executor_info=ExecutorSSHInfo(
            uuid=uuid,
            address="192.0.2.1",
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


@pytest.mark.parametrize("c_address", ["192.0.2.13", "192.0.2.12"])
@pytest.mark.asyncio
async def test_each_pair_is_recorded_on_both_hosts_under_its_own_spec_key(monkeypatch, c_address):
    a, b, c = [host(uuid, [EFA_PORT]) for uuid in "abc"]

    async def measure(server, client, key, addresses):
        addresses.update({"a": "192.0.2.11", "b": "192.0.2.12", "c": c_address})
        return 10.0 if (server.executor_info.uuid, client.executor_info.uuid) == ("a", "b") else None

    monkeypatch.setattr(efa_link_probe, "_measure_pair", measure)

    await efa_link_probe.measure_and_attach([a, b, c], "key", {}, time.monotonic())

    for node, expected in [(a, {"b": 10.0, "c": None}), (b, {"a": 10.0, "c": None}), (c, {"a": None, "b": None})]:
        assert {m["peer_executor_uuid"]: m["gigabits_per_second"] for m in node.spec["efa_link_measurements"]} == expected
        assert "roce_link_measurements" not in node.spec
    assert [(m["peer_executor_uuid"], m["peer_address"]) for m in b.spec["efa_link_measurements"]][0] == ("a", "192.0.2.11")


@pytest.mark.asyncio
async def test_no_probe_runs_until_a_reviewed_image_digest_is_pinned(monkeypatch):
    monkeypatch.setattr(efa_link_probe, "EFA_PROBE_IMAGE_DIGEST", "")
    measured = []

    async def measure(server, client, key, addresses):
        measured.append((server, client))
        return 10.0

    monkeypatch.setattr(efa_link_probe, "_measure_pair", measure)
    a, b = [host(uuid, [EFA_PORT]) for uuid in "ab"]

    await efa_link_probe.measure_and_attach([a, b], "key", {}, time.monotonic())

    assert measured == []
    assert "efa_link_measurements" not in a.spec


@pytest.mark.parametrize("failure", ["ssh_closed", "blackhole"])
@pytest.mark.asyncio
async def test_an_unreachable_host_is_recorded_as_failed_and_the_healthy_pair_is_still_measured(monkeypatch, failure):
    a, b, c = [host(uuid, [EFA_PORT]) for uuid in "abc"]
    ips = {"a": "198.51.100.1", "b": "203.0.113.1", "c": "203.0.113.2"}

    @asynccontextmanager
    async def connected(info, key):
        async def run(command, **kwargs):
            if command.endswith(" address"):
                return SimpleNamespace(stdout=ips[info.uuid])
            if " client " in command:
                if command.endswith(ips["a"]):
                    if failure == "ssh_closed":
                        raise OSError("ssh closed")
                    if "--entrypoint timeout" in command:
                        words = shlex.split(command)
                        await asyncio.sleep(float(words[words.index(efa_link_probe._probe_image()) + 1]) / 1000)
                        return SimpleNamespace(stdout="")
                    await asyncio.sleep(kwargs["timeout"] / 1000)
                    raise asyncio.TimeoutError()
                return SimpleNamespace(stdout=FI_PINGPONG_OUTPUT)
            return SimpleNamespace(stdout="")

        yield SimpleNamespace(run=run)

    monkeypatch.setattr(efa_link_probe, "_connected", connected)
    monkeypatch.setattr(efa_link_probe.asyncssh, "import_private_key", lambda _: None)
    # the real 180 s budget and 120 s ssh timeout, scaled by 1000
    monkeypatch.setattr(efa_link_probe, "budget_left_for_probe", lambda _: 0.18)
    monkeypatch.setattr(efa_link_probe, "PROBE_MINIMUM_BUDGET_SECONDS", 0)
    monkeypatch.setattr(efa_link_probe, "CLIENT_RETRY_DELAY_SECONDS", 0.002)

    await efa_link_probe.measure_and_attach([a, b, c], "key", {}, time.monotonic())

    assert {m["peer_executor_uuid"]: m["gigabits_per_second"] for m in b.spec["efa_link_measurements"]} == {
        "a": None,
        "c": 10.0,
    }


def fake_connection(monkeypatch, client_run=None, cleanup=None, server_run=None):
    @asynccontextmanager
    async def connected(info, key):
        async def run(command, **kwargs):
            if command == f"docker rm -f {efa_link_probe.PROBE_CONTAINER_NAME} >/dev/null 2>&1" and cleanup:
                return await cleanup()
            if command.endswith(" address"):
                return SimpleNamespace(stdout="203.0.113.1")
            if " client " in command and client_run:
                return await client_run()
            if " server " in command and server_run:
                return await server_run()
            return SimpleNamespace(stdout=FI_PINGPONG_OUTPUT)

        yield SimpleNamespace(run=run)

    monkeypatch.setattr(efa_link_probe, "_connected", connected)
    monkeypatch.setattr(efa_link_probe.asyncssh, "import_private_key", lambda _: None)


@pytest.mark.asyncio
async def test_stopping_the_server_cannot_hold_the_sweep_past_the_client_deadline(monkeypatch):
    a, b = [host(uuid, [EFA_PORT]) for uuid in "ab"]
    monkeypatch.setattr(efa_link_probe, "CLIENT_DEADLINE_SECONDS", 0.01)

    async def hanging_cleanup():
        await asyncio.sleep(60)

    fake_connection(monkeypatch, cleanup=hanging_cleanup)

    gigabits = await asyncio.wait_for(efa_link_probe._measure_pair(a, b, "key", {}), timeout=5)

    assert gigabits == 10.0


@pytest.mark.asyncio
async def test_a_failing_server_stop_does_not_swallow_the_budget_cancel(monkeypatch):
    a, b = [host(uuid, [EFA_PORT]) for uuid in "ab"]
    client_started = asyncio.Event()

    async def hanging_client():
        client_started.set()
        await asyncio.sleep(60)

    async def failing_cleanup():
        raise OSError("connection lost")

    fake_connection(monkeypatch, client_run=hanging_client, cleanup=failing_cleanup)
    task = asyncio.create_task(efa_link_probe._measure_pair(a, b, "key", {}))
    await client_started.wait()

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_a_cancel_while_the_server_starts_still_stops_it(monkeypatch):
    a, b = [host(uuid, [EFA_PORT]) for uuid in "ab"]
    server_starting = asyncio.Event()
    stopped = []

    async def hanging_server_start():
        server_starting.set()
        await asyncio.sleep(60)

    async def cleanup():
        stopped.append(True)
        return SimpleNamespace(stdout="")

    fake_connection(monkeypatch, cleanup=cleanup, server_run=hanging_server_start)
    task = asyncio.create_task(efa_link_probe._measure_pair(a, b, "key", {}))
    await server_starting.wait()

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped == [True]


def test_the_probe_runs_the_reviewed_0_0_2_image_by_digest(monkeypatch):
    monkeypatch.undo()  # drop the autouse placeholder digest
    assert efa_link_probe._probe_image() == (
        "daturaai/lium-rdma-probe@sha256:8e6bf7f085c70c2f16c1862fdb0ab67b2c218c09fe4a91b41be79b235f5491e2"
    )
