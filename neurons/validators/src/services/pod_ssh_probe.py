"""The once-per-cycle SSH probe of every rented pod (observe only).

At the start of each cycle ``probe_rented_pods`` connects to the mapped SSH port of every RUNNING pod
of every rented node in the backend's rented list (manual rentals excluded: the renter holds those
at root) and reads the SSH identification line. Each pod gives one ``PodSshObservation``: ``banner``,
``refused`` (with the errno), ``no_banner`` or ``timeout``. The probe needs no key and never touches
the container, and nothing here changes a score or a verdict.

The fleet decides whether the observations say anything about the pods. When at least
``FLEET_SHARE_THAT_MEANS_OUR_OWN_OUTAGE`` of a fleet of at least
``SMALLEST_FLEET_THAT_CAN_SHOW_AN_OUTAGE`` pods fails to send a banner, the validator's own network
is the suspect: every observation of the cycle carries ``fleet_ok=False`` and the backend reads it
as no observation. The share is computed right here, since the probe holds the whole cycle.

Each node's observations ride on its ``JobResult`` (``attach_pod_ssh``). A rented node the cycle
has no result for (the miner timed out, failed, or left it out of its answer) gets a minimal result
carrying only its observations, reason ``EXECUTOR_RESULT_MISSING`` (``pod_ssh_only_results``),
with ``RENTED_POD_SSH_RESULT_MISSING_REPORT_ENABLED`` on.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable, Mapping

from datura.requests.miner_requests import ExecutorSSHInfo
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse
from protocol.vc_protocol.validator_requests import PodSshObservation, PodSshResult, ValidationEvent
from services.task.availability import (
    FLEET_SHARE_THAT_MEANS_OUR_OWN_OUTAGE,
    SMALLEST_FLEET_THAT_CAN_SHOW_AN_OUTAGE,
)
from services.task.checks.rented_pod_ssh import ConnectOutcome, tcp_connect_fault
from services.task.models import JobResult, build_msg

from core.utils import _m, get_extra_info

logger = logging.getLogger(__name__)

POD_STATUS_RUNNING = "RUNNING"
EXECUTOR_RESULT_MISSING = "EXECUTOR_RESULT_MISSING"


def _targets(rented: RentedExecutorsResponse) -> list[tuple[str, str, str, int]]:
    """(executor uuid, pod id, host, port) for every RUNNING pod with a mapped SSH port."""
    manual = {str(uuid).lower() for uuid in (rented.manual_rental_executors or {})}
    targets = []
    for raw_uuid, executor in rented.executors.items():
        executor_uuid = str(raw_uuid).lower()
        if executor_uuid in manual:
            continue
        for pod in executor.pods:
            if pod.status == POD_STATUS_RUNNING and pod.ssh_port:
                targets.append(
                    (executor_uuid, pod.pod_id, executor.executor_ip_address, pod.ssh_port)
                )
    return targets


def fleet_is_ok(results: Iterable[PodSshResult]) -> bool:
    """False when most of a fleet large enough to show it sent no banner: the outage is ours."""
    results = list(results)
    if len(results) < SMALLEST_FLEET_THAT_CAN_SHOW_AN_OUTAGE:
        return True
    silent = sum(1 for result in results if result is not PodSshResult.BANNER)
    return silent / len(results) < FLEET_SHARE_THAT_MEANS_OUR_OWN_OUTAGE


async def probe_rented_pods(
    rented: RentedExecutorsResponse | None, *, timeout: float, concurrency: int, job_batch_id: str
) -> dict[str, list[PodSshObservation]]:
    """This cycle's observations, keyed by the lower-cased executor uuid. Empty without a rented list."""
    if not rented:
        return {}
    targets = _targets(rented)
    if not targets:
        return {}
    limit = asyncio.Semaphore(concurrency)

    async def probe(host: str, port: int) -> ConnectOutcome:
        async with limit:
            return await tcp_connect_fault(host, port, timeout)

    outcomes = await asyncio.gather(*(probe(host, port) for _, _, host, port in targets))
    fleet_ok = fleet_is_ok(outcome.result for outcome in outcomes)
    observations: dict[str, list[PodSshObservation]] = {}
    for (executor_uuid, pod_id, _, _), outcome in zip(targets, outcomes, strict=True):
        observations.setdefault(executor_uuid, []).append(
            PodSshObservation(
                pod_id=pod_id, result=outcome.result, errno=outcome.errno, fleet_ok=fleet_ok
            )
        )
    counts: dict[str, int] = {}
    for outcome in outcomes:
        counts[outcome.result.value] = counts.get(outcome.result.value, 0) + 1
    log = logger.info if fleet_ok else logger.warning
    log(
        _m(
            "[sync] rented pod SSH probe"
            if fleet_ok
            else "[sync] rented pod SSH probe: most pods silent, our outage",
            extra=get_extra_info(
                {
                    "job_batch_id": job_batch_id,
                    "pods": len(targets),
                    "results": counts,
                    "fleet_ok": fleet_ok,
                }
            ),
        )
    )
    return observations


def attach_pod_ssh(
    job_results: Mapping[str, Iterable[JobResult]],
    observations: Mapping[str, list[PodSshObservation]],
) -> set[str]:
    """Put each node's observations on its results; returns the lower-cased uuids that have a result."""
    reported: set[str] = set()
    for results in job_results.values():
        for result in results:
            executor_uuid = str(result.executor_info.uuid).lower()
            reported.add(executor_uuid)
            if executor_uuid in observations:
                result.pod_ssh = observations[executor_uuid]
    return reported


def build_result_missing_event(
    *, executor_uuid: str, miner_hotkey: str, pods: int
) -> ValidationEvent:
    return build_msg(
        event="No result from the miner for this rented node",
        reason=EXECUTOR_RESULT_MISSING,
        severity="error",
        category="runtime",
        impact="No validation this cycle; the report carries only the rented pods' SSH observations.",
        remediation="Check that the miner is up and returns this node in its answer to the validator.",
        what={"executor_uuid": executor_uuid, "miner_hotkey": miner_hotkey, "observed_pods": pods},
    )


def _result_missing_job_result(
    executor_uuid: str,
    executor: RentedExecutor,
    observations: list[PodSshObservation],
    job_batch_id: str,
) -> JobResult:
    try:
        executor_port = int(executor.executor_ip_port)
    except (TypeError, ValueError):
        executor_port = 0
    event = build_result_missing_event(
        executor_uuid=executor_uuid, miner_hotkey=executor.miner_hotkey, pods=len(observations)
    )
    return JobResult(
        spec=None,
        executor_info=ExecutorSSHInfo(
            uuid=executor_uuid,
            address=executor.executor_ip_address,
            port=executor_port,
            # The miner never handed over SSH details; these only satisfy the model.
            ssh_username="",
            ssh_port=0,
            python_path="",
            root_dir="",
        ),
        score=0,
        job_score=0,
        collateral_deposited=False,
        job_batch_id=job_batch_id,
        log_status="error",
        log_text=event.event,
        validation_event=event,
        gpu_model=None,
        gpu_count=0,
        sysbox_runtime=False,
        is_rented=True,
        # None, not []: an empty list would clear the node's stored availability errors.
        availability_errors=None,
        failure_reason_code=EXECUTOR_RESULT_MISSING,
        pod_ssh=observations,
    )


def pod_ssh_only_results(
    rented: RentedExecutorsResponse | None,
    observations: Mapping[str, list[PodSshObservation]],
    reported: set[str],
    job_batch_id: str,
) -> dict[str, list[JobResult]]:
    """One observations-only result per probed rented node the cycle has no result for, by miner hotkey.

    A node with a result this cycle (the pipeline's, a not-listed or manual-rental synthesis) is never
    given a second one. The miner hotkey comes from the backend's rented list, not from a miner answer.
    """
    if not rented:
        return {}
    executors = {str(uuid).lower(): executor for uuid, executor in rented.executors.items()}
    by_hotkey: dict[str, list[JobResult]] = {}
    for executor_uuid, node_observations in observations.items():
        executor = executors.get(executor_uuid)
        if executor is None or executor_uuid in reported:
            continue
        by_hotkey.setdefault(executor.miner_hotkey, []).append(
            _result_missing_job_result(executor_uuid, executor, node_observations, job_batch_id)
        )
    return by_hotkey
