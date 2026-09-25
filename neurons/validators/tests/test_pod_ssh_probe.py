"""Observe only: the once-per-cycle SSH probe of every rented pod.

What one probe reads (``tcp_connect_fault``), which pods are probed and when the cycle's observations
say the outage is ours (``probe_rented_pods``), how they ride on the node's result (``attach_pod_ssh``),
and the observations-only result for a rented node the cycle has no result for (``pod_ssh_only_results``).
"""

from __future__ import annotations

import asyncio
import errno
import socket
from unittest.mock import AsyncMock, patch

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo
from protocol.vc_protocol.compute_requests import (
    ManualRentalInfo,
    RentedExecutor,
    RentedExecutorsResponse,
    RentedPod,
)
from protocol.vc_protocol.validator_requests import (
    ExecutorSpecRequest,
    PodSshObservation,
    PodSshResult,
)
from services.pod_ssh_probe import (
    EXECUTOR_RESULT_MISSING,
    attach_pod_ssh,
    fleet_is_ok,
    pod_ssh_only_results,
    probe_rented_pods,
)
from services.task.checks.rented_pod_ssh import ConnectOutcome, tcp_connect_fault
from services.task.checks.ssh_identification import SSH_PRE_BANNER_LINES_MAX, is_ssh2_identification
from services.task.models import JobResult

from services import pod_ssh_probe

SSH2 = b"SSH-2.0-OpenSSH_9.6p1 Ubuntu-3ubuntu13\r\n"
JOB_BATCH_ID = "2026-09-25 10:15:00"


async def _serve(handler):
    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    return server, server.sockets[0].getsockname()[1]


def _greet(payload: bytes):
    def handler(_reader, writer):
        writer.write(payload)
        writer.close()

    return handler


# --- one probe ------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_closed_port_is_refused_with_its_errno():
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    closed_port = probe.getsockname()[1]
    probe.close()

    assert await tcp_connect_fault("127.0.0.1", closed_port, timeout=2.0) == ConnectOutcome(
        PodSshResult.REFUSED, errno.ECONNREFUSED
    )


@pytest.mark.asyncio
async def test_an_accept_then_close_is_no_banner():
    # docker-proxy accepts on the host while sshd inside the container is down
    server, port = await _serve(lambda _r, w: w.close())
    try:
        assert await tcp_connect_fault("127.0.0.1", port, timeout=2.0) == ConnectOutcome(
            PodSshResult.NO_BANNER
        )
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_a_banner_split_across_segments_is_read_whole():
    async def greet_in_two_segments(_reader, writer):
        writer.write(b"SSH-2")
        await writer.drain()
        await asyncio.sleep(0.05)
        writer.write(b".0-OpenSSH_9.6p1 Ubuntu-3ubuntu13\r\n")
        await writer.drain()
        writer.close()

    server, port = await _serve(greet_in_two_segments)
    try:
        assert await tcp_connect_fault("127.0.0.1", port, timeout=2.0) == ConnectOutcome(
            PodSshResult.BANNER
        )
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_a_connect_that_never_completes_is_timeout():
    async def hang(*_args, **_kwargs):
        await asyncio.sleep(10)

    with patch("asyncio.open_connection", new=hang):
        assert await tcp_connect_fault("127.0.0.1", 22, timeout=0.05) == ConnectOutcome(
            PodSshResult.TIMEOUT
        )


@pytest.mark.asyncio
async def test_connect_and_banner_read_share_one_deadline():
    accepted: list[asyncio.StreamWriter] = []
    silent, port = await _serve(lambda _r, w: accepted.append(w))
    real_open_connection = asyncio.open_connection

    async def slow_connect(*args, **kwargs):
        await asyncio.sleep(0.2)
        return await real_open_connection(*args, **kwargs)

    loop = asyncio.get_running_loop()
    try:
        with patch("asyncio.open_connection", new=slow_connect):
            started = loop.time()
            outcome = await tcp_connect_fault("127.0.0.1", port, timeout=0.3)
            elapsed = loop.time() - started
    finally:
        for writer in accepted:
            writer.close()
        silent.close()
        await silent.wait_closed()

    assert outcome == ConnectOutcome(PodSshResult.NO_BANNER)
    assert 0.25 <= elapsed < 0.45, elapsed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        pytest.param(
            [b"Welcome\r\n", b"\r\n", SSH2], PodSshResult.BANNER, id="pre_banner_lines_skipped"
        ),
        pytest.param(
            [b"x\r\n"] * SSH_PRE_BANNER_LINES_MAX + [SSH2],
            PodSshResult.BANNER,
            id="at_the_line_bound",
        ),
        pytest.param(
            [b"x\r\n"] * (SSH_PRE_BANNER_LINES_MAX + 1) + [SSH2],
            PodSshResult.NO_BANNER,
            id="past_the_line_bound",
        ),
        pytest.param(
            [b"y" * 300 + b"\r\n", SSH2], PodSshResult.NO_BANNER, id="pre_banner_line_too_long"
        ),
        pytest.param([b"SSH-1.99-OpenSSH_3.9p1\r\n"], PodSshResult.NO_BANNER, id="ssh_1.99"),
        pytest.param([b"HTTP/1.1 400 Bad Request\r\n\r\n"], PodSshResult.NO_BANNER, id="http"),
        pytest.param([b"SSH-2.0-\r\n"], PodSshResult.NO_BANNER, id="no_softwareversion"),
        pytest.param(
            [b"SSH-2.0-OpenSSH_9.6p1"], PodSshResult.NO_BANNER, id="closed_before_the_line_ended"
        ),
    ],
)
async def test_the_identification_line_decides_banner(lines, expected):
    server, port = await _serve(_greet(b"".join(lines)))
    try:
        assert (await tcp_connect_fault("127.0.0.1", port, timeout=2.0)).result is expected
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.parametrize(
    ("line", "ok"),
    [
        (b"SSH-2.0-OpenSSH_9.6p1 Ubuntu-3ubuntu13\r\n", True),
        (b"SSH-2.0-dropbear_2024.85\n", True),
        (b"SSH-2.0-" + b"x" * 245 + b"\r\n", True),
        (b"SSH-2.0-" + b"x" * 246 + b"\r\n", False),
        (b"SSH-1.5-Cisco-1.25\r\n", False),
        (b"ssh-2.0-OpenSSH_9.6p1\r\n", False),
        (b"", False),
    ],
)
def test_is_ssh2_identification(line, ok):
    assert is_ssh2_identification(line) is ok


@pytest.mark.parametrize("port", [0, -1, 65536, 70000])
def test_a_mapped_port_outside_1_65535_is_read_as_no_mapped_port(port):
    # asyncio.open_connection raises OverflowError, not OSError, on such a port
    assert RentedPod(pod_id="p", container_name="c", ssh_port=port).ssh_port is None


# --- the cycle's probe ------------------------------------------------------------------------------------------


def _pod(pod_id: str, *, ssh_port: int | None = 40100, status: str | None = "RUNNING") -> RentedPod:
    return RentedPod(pod_id=pod_id, container_name=f"c-{pod_id}", ssh_port=ssh_port, status=status)


def _rented(
    executors: dict[str, list[RentedPod]], *, manual: tuple[str, ...] = ()
) -> RentedExecutorsResponse:
    return RentedExecutorsResponse(
        executors={
            uuid: RentedExecutor(
                miner_hotkey=f"hk-{uuid}",
                executor_ip_address=f"198.51.100.{i + 1}",
                executor_ip_port="9001",
                pods=pods,
            )
            for i, (uuid, pods) in enumerate(executors.items())
        },
        manual_rental_executors={
            uuid: ManualRentalInfo(gpu_model="H100", gpu_count=1) for uuid in manual
        },
    )


def _probe_with(outcomes: dict[tuple[str, int], ConnectOutcome]):
    calls: list[tuple[str, int]] = []

    async def fake(host: str, port: int, timeout: float) -> ConnectOutcome:
        calls.append((host, port))
        return outcomes.get((host, port), ConnectOutcome(PodSshResult.BANNER))

    return patch.object(pod_ssh_probe, "tcp_connect_fault", new=fake), calls


@pytest.mark.asyncio
async def test_only_running_pods_with_a_port_on_non_manual_nodes_are_probed():
    rented = _rented(
        {
            "E1": [_pod("p1"), _pod("p2", status="STOPPED"), _pod("p3", ssh_port=None)],
            "E2": [_pod("p4", ssh_port=40200)],
            "MANUAL": [_pod("p5")],
        },
        manual=("MANUAL",),
    )
    fake, calls = _probe_with(
        {("198.51.100.2", 40200): ConnectOutcome(PodSshResult.REFUSED, errno.ECONNREFUSED)}
    )
    with fake:
        observations = await probe_rented_pods(
            rented, timeout=1, concurrency=4, job_batch_id=JOB_BATCH_ID
        )

    assert sorted(calls) == [("198.51.100.1", 40100), ("198.51.100.2", 40200)]
    assert observations == {
        "e1": [
            PodSshObservation(pod_id="p1", result=PodSshResult.BANNER, errno=None, fleet_ok=True)
        ],
        "e2": [
            PodSshObservation(pod_id="p4", result=PodSshResult.REFUSED, errno=111, fleet_ok=True)
        ],
    }


@pytest.mark.asyncio
async def test_most_of_the_fleet_silent_marks_every_observation_as_our_outage():
    rented = _rented({f"E{i}": [_pod(f"p{i}", ssh_port=40100 + i)] for i in range(6)})
    silent = {
        (f"198.51.100.{i + 1}", 40100 + i): ConnectOutcome(PodSshResult.TIMEOUT) for i in range(3)
    }
    fake, _ = _probe_with(silent)
    with fake:
        observations = await probe_rented_pods(
            rented, timeout=1, concurrency=2, job_batch_id=JOB_BATCH_ID
        )

    assert {o.fleet_ok for obs in observations.values() for o in obs} == {False}
    assert observations["e5"][0].result is PodSshResult.BANNER


@pytest.mark.parametrize(
    ("results", "ok"),
    [
        ([PodSshResult.TIMEOUT] * 4, True),  # too few pods for a share to mean anything
        ([PodSshResult.BANNER] * 3 + [PodSshResult.REFUSED] * 2, True),
        (
            [PodSshResult.BANNER] * 3
            + [PodSshResult.REFUSED, PodSshResult.NO_BANNER, PodSshResult.TIMEOUT],
            False,
        ),
        ([], True),
    ],
)
def test_fleet_is_ok(results, ok):
    assert fleet_is_ok(results) is ok


@pytest.mark.asyncio
async def test_concurrency_bounds_the_probes_in_flight():
    rented = _rented({f"E{i}": [_pod(f"p{i}", ssh_port=40100 + i)] for i in range(10)})
    in_flight = peak = 0

    async def fake(host, port, timeout):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return ConnectOutcome(PodSshResult.BANNER)

    with patch.object(pod_ssh_probe, "tcp_connect_fault", new=fake):
        observations = await probe_rented_pods(
            rented, timeout=1, concurrency=3, job_batch_id=JOB_BATCH_ID
        )

    assert peak == 3
    assert len(observations) == 10


@pytest.mark.asyncio
async def test_no_rented_list_probes_nothing():
    assert await probe_rented_pods(None, timeout=1, concurrency=1, job_batch_id=JOB_BATCH_ID) == {}


# --- the results --------------------------------------------------------------------------------------------------


def _result(uuid: str, score: float = 1.0) -> JobResult:
    return JobResult(
        executor_info=ExecutorSSHInfo(
            uuid=uuid,
            address="198.51.100.9",
            port=9001,
            ssh_username="u",
            ssh_port=22,
            python_path="",
            root_dir="",
        ),
        score=score,
        job_score=score,
        job_batch_id=JOB_BATCH_ID,
        log_status="info",
        log_text="ok",
    )


def _obs(pod_id: str, result: PodSshResult = PodSshResult.BANNER) -> PodSshObservation:
    return PodSshObservation(pod_id=pod_id, result=result)


def test_observations_ride_on_the_nodes_result_whatever_the_uuids_case():
    with_pods, without = _result("E1-UPPER"), _result("e2")
    reported = attach_pod_ssh({"hk": [with_pods, without]}, {"e1-upper": [_obs("p1")]})

    assert with_pods.pod_ssh == [_obs("p1")]
    assert without.pod_ssh is None
    assert reported == {"e1-upper", "e2"}


def test_a_probed_node_without_a_result_gets_one_carrying_only_its_observations():
    rented = _rented({"E1": [_pod("p1")], "E2": [_pod("p2")], "E3": [_pod("p3")]})
    observations = {"e1": [_obs("p1")], "e2": [_obs("p2", PodSshResult.TIMEOUT)]}

    by_hotkey = pod_ssh_only_results(
        rented, observations, reported={"e1"}, job_batch_id=JOB_BATCH_ID
    )

    (result,) = by_hotkey["hk-E2"]
    assert list(by_hotkey) == ["hk-E2"]
    assert result.executor_info.uuid == "e2"
    assert (result.executor_info.address, result.executor_info.port) == ("198.51.100.2", 9001)
    assert (result.score, result.job_score, result.spec, result.gpu_count) == (0, 0, None, 0)
    assert result.validation_event.reason_code == EXECUTOR_RESULT_MISSING
    assert result.failure_reason_code == EXECUTOR_RESULT_MISSING
    assert result.availability_errors is None  # [] would clear the node's stored errors
    assert result.scored_at is None and result.is_successful is False
    assert result.pod_ssh == [_obs("p2", PodSshResult.TIMEOUT)]


def test_a_node_that_left_the_rented_list_gets_nothing():
    assert (
        pod_ssh_only_results(
            _rented({}), {"e9": [_obs("p9")]}, reported=set(), job_batch_id=JOB_BATCH_ID
        )
        == {}
    )


@pytest.mark.asyncio
async def test_the_published_spec_carries_the_observations_and_the_backend_reads_them():
    from services.miner_service import MinerService

    result = pod_ssh_only_results(
        _rented({"E2": [_pod("p2")]}),
        {
            "e2": [
                _obs("p2", PodSshResult.REFUSED).model_copy(
                    update={"errno": 111, "fleet_ok": False}
                )
            ]
        },
        reported=set(),
        job_batch_id=JOB_BATCH_ID,
    )["hk-E2"]
    service = MinerService.__new__(MinerService)
    service.redis_service = AsyncMock()

    with patch("services.miner_service.settings") as settings:
        settings.DRY_RUN = False
        settings.POD_STATES_REPORT_ENABLED = False
        await MinerService.publish_machine_specs(
            service, result, "hk-E2", "ck-E2", is_whole_miner_batch=False
        )

    (_channel, payload), _ = service.redis_service.publish.call_args
    assert payload["pod_ssh"] == [
        {"pod_id": "p2", "result": "refused", "errno": 111, "fleet_ok": False}
    ]
    assert payload["batch_total"] is None and payload["scored_at"] is None
    spec = ExecutorSpecRequest(
        **{k: v for k, v in payload.items() if k in ExecutorSpecRequest.model_fields},
        validator_hotkey="v",
    )
    assert spec.pod_ssh[0].result is PodSshResult.REFUSED
    assert spec.validation_event.reason_code == EXECUTOR_RESULT_MISSING
