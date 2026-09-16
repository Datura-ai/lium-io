"""DAH-3490: after a rental's container is removed, the validator names who broke the card, posts the result as
its own request, and delists a node the workload broke that still does not answer — with no penalty reason."""

from contextlib import asynccontextmanager, contextmanager
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest
import services.docker_service as module
from payload_models.payloads import ContainerDeleteRequest, WorkloadKind
from protocol.vc_protocol.compute_requests import GpuFaultProbeRequest
from protocol.vc_protocol.validator_requests import ResetVerifiedJobReason
from services.docker_service import DockerService
from services.gpu_xid_attribution import ECC_QUERY_COMMAND, XID_LOG_COMMAND

from tests.helpers import default_executor

EXECUTOR = "d51b8008-7338-4c49-a5ae-d37e876ff79f"
POD = "11655dc5-53ba-4a8d-a341-fe6c9d12bda7"
STARTED = datetime.now(UTC) - timedelta(hours=2)
LOST_CARD = "Unable to determine the device handle for GPU 0000:81:00.0: Unknown Error"


def xid(minutes_after_start: int, code: int, pid: int | None = 4242) -> str:
    stamp = (STARTED + timedelta(minutes=minutes_after_start)).strftime("%Y-%m-%dT%H:%M:%S")
    pid_part = f"pid={pid}, name=python, " if pid is not None else ""
    return f"{stamp},000000+00:00 NVRM: Xid (PCI:0000:81:00): {code}, {pid_part}Ch 00000008"


def payload(kind: WorkloadKind = WorkloadKind.CUSTOMER_RENTAL) -> ContainerDeleteRequest:
    return ContainerDeleteRequest(
        miner_hotkey="miner", executor_id=EXECUTOR, pod_id=POD, container_name=f"pod_{POD}", workload_kind=kind
    )


class FakeSSH:
    """The host's answers to the probe's two commands, over the probe's own SSH session."""

    def __init__(self, *, xid_lines: str, gpu_exit: int, gpu_out: str, gpu_err: str = ""):
        self.answers = {XID_LOG_COMMAND: (0, xid_lines, ""), ECC_QUERY_COMMAND: (gpu_exit, gpu_out, gpu_err)}
        self.calls: list[str] = []

    async def run(self, cmd, input=None):
        self.calls.append(cmd)
        exit_status, stdout, stderr = self.answers[cmd]
        return Mock(exit_status=exit_status, stdout=stdout, stderr=stderr)


@contextmanager
def probe_flag(enabled: bool = True):
    with patch("services.docker_service.settings") as s:
        s.RENTAL_GPU_FAULT_PROBE_ENABLED = enabled
        yield


def service(ssh: FakeSSH) -> DockerService:
    redis = AsyncMock()
    redis.get_verified_job_info.return_value = {"spec": "NVIDIA H100:8", "uuids": "GPU-a"}
    svc = DockerService(ssh_service=Mock(), redis_service=redis, attestation_service=Mock(), backend_client=AsyncMock())

    @asynccontextmanager
    async def connect(**kwargs):
        yield ssh

    svc._connect_for_test = connect
    return svc


async def run_probe(svc: DockerService, started_at=STARTED, kind=WorkloadKind.CUSTOMER_RENTAL, pids=None):
    with patch.object(module.asyncssh, "connect", svc._connect_for_test):
        return await svc._probe_rental_end_gpu_fault(payload(kind), default_executor(), Mock(), Mock(), started_at, pids, {})


@pytest.mark.asyncio
async def test_a_workload_fault_on_a_node_that_no_longer_answers_delists_it_without_a_penalty_reason():
    """On origin/main the undeploy ends at ContainerDeleted: nothing reads the kernel log, ResetVerifiedJobReason has no
    workload member, and the node keeps its verification while its lost card fails every later cycle at score 0 with a
    reason the backend penalises. Here: one probe, one report, one reset with the no-penalty reason."""
    ssh = FakeSSH(xid_lines=xid(30, 31), gpu_exit=15, gpu_out="", gpu_err=LOST_CARD)
    svc = service(ssh)

    report = await run_probe(svc)

    assert ssh.calls == [XID_LOG_COMMAND, ECC_QUERY_COMMAND]
    assert set(report) == set(GpuFaultProbeRequest.model_fields)  # the wire body, nothing undeclared
    assert report["phase"] == "rental_end" and report["attribution"] == "workload"
    assert report["node_answers"] is False and LOST_CARD in report["nvidia_smi_error"]
    assert report["container_pids_known"] is False and report["dmesg_unavailable"] is False
    svc.backend_client.report_gpu_fault_probe.assert_awaited_once_with(EXECUTOR, report)
    svc.redis_service.clear_verified_job_info.assert_awaited_once()
    reset = svc.redis_service.clear_verified_job_info.await_args.kwargs
    assert reset["reason"] == ResetVerifiedJobReason.GPU_FAULT_AFTER_RENTAL_WORKLOAD
    assert reset["reason"].value == 2  # the wire value the backend reads as "delist, no penalty"
    assert reset["executor_id"] == EXECUTOR and reset["miner_hotkey"] == "miner"
    assert reset["evidence"]["reason_code"] == "GPU_FAULT_AFTER_RENTAL_WORKLOAD"
    assert reset["evidence"]["check_id"] == "docker.delete.rental_end_gpu_fault"
    assert reset["evidence"]["pod_id"] == POD and len(reset["evidence"]["workload_xids"]) == 1


@pytest.mark.asyncio
async def test_another_pods_process_is_not_this_renters_when_the_pids_were_read_beside_another_tenant():
    # a multi-pod node: the PID set read through the SDK before the stop places the line on the other tenant
    ssh = FakeSSH(xid_lines=xid(30, 31, pid=9999), gpu_exit=15, gpu_out="", gpu_err=LOST_CARD)
    svc = service(ssh)
    report = await run_probe(svc, pids={4242, 4243})
    assert report["attribution"] == "none" and report["other_container"] == 1 and report["container_pids_known"] is True
    svc.redis_service.clear_verified_job_info.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_pids_are_read_only_beside_another_tenant_and_before_this_delete_drops_its_own_entry():
    """The rented-machine record still lists this pod's own container when the window is read (this delete removes
    it later), so "another tenant" is decided by name; a single-tenant node leaves the PID set None and the
    renter's exited process is placed by its timestamp."""
    svc = service(FakeSSH(xid_lines="", gpu_exit=0, gpu_out=""))
    docker_client = AsyncMock()
    docker_client.container_started_at.return_value = "2026-09-16T09:00:00.000000000Z"
    docker_client.container_pids.return_value = {4242}
    with probe_flag():
        svc.redis_service.get_rented_machine.return_value = {
            "owner_flag": False,
            "containers": [{"name": f"pod_{POD}", "pod_id": POD}],
        }
        started_at, pids = await svc._read_rental_window(docker_client, payload(), default_executor(), Mock())
        assert started_at is not None and pids is None
        docker_client.container_pids.assert_not_awaited()

        svc.redis_service.get_rented_machine.return_value = {
            "owner_flag": False,
            "containers": [{"name": f"pod_{POD}", "pod_id": POD}, {"name": "pod_other", "pod_id": "other"}],
        }
        started_at, pids = await svc._read_rental_window(docker_client, payload(), default_executor(), Mock())
        # the PIDs are read from this pod's container, not the other tenant's, and only now
        docker_client.container_pids.assert_awaited_once_with(container_name=f"pod_{POD}")
        assert pids is not None


@pytest.mark.asyncio
async def test_a_kernel_log_that_cannot_be_read_attributes_nothing():
    ssh = FakeSSH(xid_lines="DMESG_UNAVAILABLE\n", gpu_exit=15, gpu_out="", gpu_err=LOST_CARD)
    svc = service(ssh)
    report = await run_probe(svc)
    assert report["attribution"] == "none" and report["dmesg_unavailable"] is True
    svc.redis_service.clear_verified_job_info.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_workload_fault_on_a_node_that_answers_is_reported_and_nothing_is_reset():
    # the card came back after the container died: the renter and provider were told mid-rental; the node stays listed
    ssh = FakeSSH(xid_lines=xid(30, 31), gpu_exit=0, gpu_out="00000000:81:00.0, GPU-a, 0\n")
    svc = service(ssh)
    report = await run_probe(svc)
    assert report["attribution"] == "workload" and report["node_answers"] is True
    svc.backend_client.report_gpu_fault_probe.assert_awaited_once()
    svc.redis_service.clear_verified_job_info.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_hardware_fault_is_not_this_path_even_when_the_node_does_not_answer():
    # Xid 79 (off the bus) is the provider's: the existing checks handle the node; no no-penalty reset from here
    ssh = FakeSSH(xid_lines=xid(30, 79, pid=None), gpu_exit=15, gpu_out="", gpu_err=LOST_CARD)
    svc = service(ssh)
    report = await run_probe(svc)
    assert report["attribution"] == "hardware"
    svc.redis_service.clear_verified_job_info.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_window_start_means_no_verdict():
    ssh = FakeSSH(xid_lines=xid(30, 31), gpu_exit=15, gpu_out="", gpu_err=LOST_CARD)
    svc = service(ssh)
    report = await run_probe(svc, started_at=None)
    assert report["attribution"] == "none" and report["container_started_at"] is None
    svc.redis_service.clear_verified_job_info.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_host_that_cannot_be_reached_reports_nothing_and_raises_nothing():
    svc = service(FakeSSH(xid_lines="", gpu_exit=0, gpu_out=""))

    @asynccontextmanager
    async def refuse(**kwargs):
        raise OSError("connection refused")
        yield  # pragma: no cover

    svc._connect_for_test = refuse
    assert await run_probe(svc) is None
    svc.backend_client.report_gpu_fault_probe.assert_not_awaited()
    svc.redis_service.clear_verified_job_info.assert_not_awaited()


def test_only_a_customer_rental_is_probed_and_only_under_the_flag():
    svc = service(FakeSSH(xid_lines="", gpu_exit=0, gpu_out=""))
    with probe_flag():
        assert svc._rental_end_gpu_fault_probe_applies(payload()) is True
        assert svc._rental_end_gpu_fault_probe_applies(payload(WorkloadKind.FILLER)) is False
        # the validator's own synthetic rental probe (rental_probe.py) tears down a pod the backend never saw
        synthetic = payload().model_copy(update={"gpu_fault_probe": False})
        assert svc._rental_end_gpu_fault_probe_applies(synthetic) is False
        # no backend client (tests, tools): the report is the point, so no probe
        svc.backend_client = None
        assert svc._rental_end_gpu_fault_probe_applies(payload()) is False
    svc.backend_client = AsyncMock()
    with probe_flag(enabled=False):
        assert svc._rental_end_gpu_fault_probe_applies(payload()) is False


@pytest.mark.asyncio
async def test_the_probe_is_scheduled_not_awaited_and_tracked_until_done():
    svc = service(FakeSSH(xid_lines="", gpu_exit=0, gpu_out=""))
    started = []

    async def probe(*args):
        started.append(args[0].pod_id)
        return {"attribution": "none"}

    with patch.object(svc, "_probe_rental_end_gpu_fault", probe):
        task = svc._schedule_rental_end_gpu_fault_probe(payload(), default_executor(), Mock(), Mock(), STARTED, None, {})
        assert task in svc.rental_end_gpu_fault_tasks and started == []  # nothing ran before the caller yields
        assert await task == {"attribution": "none"}
    assert started == [POD]
    assert task not in svc.rental_end_gpu_fault_tasks
