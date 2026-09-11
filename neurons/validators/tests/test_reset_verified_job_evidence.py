"""DAH-3386: the reset that flips a rented node inactive names the check that fired and what it saw.

Backend side (lium-platform DAH-3385) writes these fields into the penalty it raises. The regressions each test
guards against, from the 90-day penalty-reversal pass (lium-ads reports/PENALTY_ACCURACY_20260910.md):

  1 test_pod_not_running_reset_carries_the_containers_death_diagnostics
        a renter's own entrypoint exiting (container present, its own exit code, not OOM-killed) was penalised as
        "executor inactive"; the row said nothing about the container — the reviewer had to SSH to the host.
  1b test_host_answers_to_docker_inspect_are_typed_and_bounded_before_they_travel
        the values come from the provider's host; an oversize or mistyped inspect state must not reach the backend's
        JSONB row and log line as sent (PR_PROCESS §5 bounded input).
  2 test_a_check_that_clears_the_job_without_naming_itself_is_named_by_the_pipeline
        spec change, GPU fingerprint, duplicate executor, banned GPU and NVML spoof all reach the backend as
        reason DEFAULT (0); the pipeline fills reason_code/check_id so they are told apart on the row.
  3 test_a_check_that_does_not_clear_the_job_adds_no_evidence
        the transport-unreachable result must not start carrying clear-evidence (it deliberately does not flip).
  4 test_the_publish_splits_names_from_evidence_and_an_empty_evidence_is_none
        the wire shape the backend's ResetVerifiedJobRequest expects; an older validator's publish (no keys) is
        still valid because every new field is optional.
  5 test_the_published_reset_reaches_the_backend_queue_with_its_names_and_evidence
        compute_client.py reads the three new keys off the redis payload; drop them and every reset reaches the
        backend unnamed while every other test stays green.
"""

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from clients.compute_client import ComputeClient
from neurons.validators.src.services.redis_service import RESET_VERIFIED_JOB_CHANNEL, RedisService
from neurons.validators.src.services.task.checks.rented_machine import (
    _PENALTY_EVIDENCE_STR_MAX,
    TenantEnforcementCheck,
    _penalty_evidence_from_diagnostics,
)
from neurons.validators.src.services.task.messages import TenantEnforcementMessages as Msg
from neurons.validators.src.services.task.models import ValidationEvent
from neurons.validators.src.services.task.pipeline import CheckResult, updates_with_clear_verified_job_evidence
from protocol.vc_protocol.validator_requests import ResetVerifiedJobReason, ResetVerifiedJobRequest
from test_rented_machine_check import (
    DummyBackendClient,
    DummyScoreCalculator,
    DummySSHClient,
    MockContainerCleanup,
    build_rented_data,
)

from helpers import build_context_config, build_services, build_state


def _event(reason_code: str, check_id: str | None) -> ValidationEvent:
    return ValidationEvent(
        event="x", reason_code=reason_code, severity="warning", impact="none", check_id=check_id,
        when=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_pod_not_running_reset_carries_the_containers_death_diagnostics(context_factory):
    ssh = DummySSHClient(pod_running=False)
    backend = DummyBackendClient(active=True)
    ctx = context_factory(
        services=build_services(
            score_calculator=DummyScoreCalculator(actual_score=1.0, job_score=1.0, warning=""),
            container_cleanup=MockContainerCleanup(),
            backend=backend,
        ),
        config=build_context_config(),
        state=build_state(
            gpu_processes=[], gpu_details=[], gpu_model="NVIDIA RTX 4090",
            rented_data=build_rented_data(
                "executor-123", {"containers": [{"name": "tenant-123", "pod_id": "pod-1"}], "owner_flag": False}
            ),
        ),
        ssh=ssh,
        collateral_deposited=True,
        is_rental_succeed=True,
        contract_version="v1.0.0",
    )

    result = await TenantEnforcementCheck().run(ctx)

    assert result.passed is False
    assert result.updates["clear_verified_job_reason"] == ResetVerifiedJobReason.POD_NOT_RUNNING.value
    evidence = result.updates["clear_verified_job_evidence"]
    assert evidence["reason_code"] == Msg.POD_NOT_RUNNING.reason
    assert evidence["check_id"] == TenantEnforcementCheck().check_id
    assert evidence["pod_id"] == "pod-1" and evidence["container_name"] == "tenant-123"
    # The DummySSHClient's `docker inspect` answers Status exited / ExitCode 255 / FinishedAt — the fields that
    # separate "the workload exited" from "the host lost the container" reach the backend as sent.
    assert evidence["container"]["container_status"] == "exited"
    assert evidence["container"]["container_exit_code"] == 255
    assert evidence["container"]["container_finished_at"] == ssh.container_finished_at
    assert evidence["container"]["container_missing"] is False
    # Logs and host context stay in Loki; they are not part of the penalty row.
    assert "container_logs_tail" not in evidence["container"]
    assert "container_host_context" not in evidence["container"]


def test_host_answers_to_docker_inspect_are_typed_and_bounded_before_they_travel():
    """A host that answers `docker inspect` with a 10 kB error string, a string exit code, a non-bool OOM flag and a
    nested object for a field must not get any of that onto the penalty row: strings are cut, wrong types dropped."""
    hostile = {
        "container_status": "x" * 5000,
        "container_exit_code": "137",
        "container_oom_killed": "yes",
        "container_missing": 0,
        "container_error": {"nested": ["a" * 5000]},
        "container_finished_at": None,
        "container_logs_tail": "must not travel",
        "diagnostics_capture_error": ["inspect: " + "e" * 5000],
    }
    evidence = _penalty_evidence_from_diagnostics(hostile)
    assert evidence == {"container_status": "x" * _PENALTY_EVIDENCE_STR_MAX, "container_finished_at": None}
    assert len(json.dumps(evidence)) < 2 * _PENALTY_EVIDENCE_STR_MAX
    # And the honest shape passes through with its types intact.
    honest = {"container_status": "exited", "container_exit_code": 2, "container_oom_killed": False,
              "container_missing": False, "container_finished_at": "2026-09-10T13:45:00Z"}
    assert _penalty_evidence_from_diagnostics(honest) == honest


def test_a_check_that_clears_the_job_without_naming_itself_is_named_by_the_pipeline():
    result = CheckResult(passed=False, event=_event("SPEC_CHANGED", None), updates={"clear_verified_job_info": True})

    updates = updates_with_clear_verified_job_evidence(result, "gpu.validate.spec_change")

    assert updates["clear_verified_job_info"] is True
    assert updates["clear_verified_job_evidence"] == {"reason_code": "SPEC_CHANGED", "check_id": "gpu.validate.spec_change"}
    # A check that attached its own evidence keeps every key of it — the names it chose win over the event's.
    rich = CheckResult(
        passed=False,
        event=_event("EVENT_SAYS_OTHERWISE", "event.says.otherwise"),
        updates={
            "clear_verified_job_info": True,
            "clear_verified_job_evidence": {
                "reason_code": "POD_NOT_RUNNING",
                "check_id": "executor.validate.rented_state",
                "container": {"container_exit_code": 2},
            },
        },
    )
    assert updates_with_clear_verified_job_evidence(rich, "pipeline.says.otherwise")["clear_verified_job_evidence"] == {
        "reason_code": "POD_NOT_RUNNING",
        "check_id": "executor.validate.rented_state",
        "container": {"container_exit_code": 2},
    }


def test_a_check_that_does_not_clear_the_job_adds_no_evidence():
    result = CheckResult(
        passed=False, event=_event("EXECUTOR_TRANSPORT_UNREACHABLE", "executor.validate.rented_state"),
        updates={"default_extra": {"rented": True}},
    )
    assert updates_with_clear_verified_job_evidence(result, "executor.validate.rented_state") == {"default_extra": {"rented": True}}


@pytest.mark.asyncio
async def test_the_publish_splits_names_from_evidence_and_an_empty_evidence_is_none():
    service = RedisService.__new__(RedisService)
    service.hset = AsyncMock()
    service.publish = AsyncMock()

    await service.clear_verified_job_info(
        miner_hotkey="hk", executor_id="ex-1", prev_info={}, reason=ResetVerifiedJobReason.POD_NOT_RUNNING,
        evidence={"reason_code": "POD_NOT_RUNNING", "check_id": "executor.validate.rented_state",
                  "container": {"container_exit_code": 2, "container_oom_killed": False}},
    )
    channel, payload = service.publish.await_args.args
    assert payload["reason"] == ResetVerifiedJobReason.POD_NOT_RUNNING.value
    assert payload["reason_code"] == "POD_NOT_RUNNING"
    assert payload["check_id"] == "executor.validate.rented_state"
    assert payload["evidence"] == {"container": {"container_exit_code": 2, "container_oom_killed": False}}

    await service.clear_verified_job_info(miner_hotkey="hk", executor_id="ex-2", prev_info={})
    _, bare = service.publish.await_args.args
    assert bare["reason"] == ResetVerifiedJobReason.DEFAULT.value
    assert bare["reason_code"] is None and bare["check_id"] is None and bare["evidence"] is None


class _OneMessagePubsub:
    """A pubsub that delivers one redis message and then stops the client's forever-loop."""

    def __init__(self, channel: str, payload: dict) -> None:
        self.message = {"channel": channel.encode(), "data": json.dumps(payload).encode()}
        self.closed = False

    async def listen(self):
        yield self.message
        raise asyncio.CancelledError  # the only way out of `while True`; not caught by `except Exception`

    async def aclose(self) -> None:
        self.closed = True


async def _queued_reset_request(payload: dict) -> ResetVerifiedJobRequest:
    """Feed one RESET_VERIFIED_JOB_CHANNEL message through compute_client's redis loop; the queued request."""
    client = ComputeClient.__new__(ComputeClient)
    client.lock = asyncio.Lock()
    client.message_queue = []
    client.logging_extra = {}
    client.my_hotkey = lambda: "vk"
    pubsub = _OneMessagePubsub(RESET_VERIFIED_JOB_CHANNEL, payload)
    client.miner_service = MagicMock()
    client.miner_service.redis_service = AsyncMock()
    client.miner_service.redis_service.subscribe = AsyncMock(return_value=pubsub)
    with pytest.raises(asyncio.CancelledError):
        await client.subscribe_mesages_from_redis()
    assert pubsub.closed
    (request,) = client.message_queue
    assert isinstance(request, ResetVerifiedJobRequest)
    return request


@pytest.mark.asyncio
async def test_the_published_reset_reaches_the_backend_queue_with_its_names_and_evidence():
    """compute_client.py builds the ResetVerifiedJobRequest from the redis payload; if it stopped reading the
    three new keys every reset would reach the backend unnamed again while every other test stayed green
    (taiberium, 11 Sep). The payload is the one `clear_verified_job_info` publishes, not a hand-written copy."""
    service = RedisService.__new__(RedisService)
    service.hset = AsyncMock()
    service.publish = AsyncMock()
    await service.clear_verified_job_info(
        miner_hotkey="hk", executor_id="ex-1", prev_info={}, reason=ResetVerifiedJobReason.POD_NOT_RUNNING,
        evidence={"reason_code": "POD_NOT_RUNNING", "check_id": "executor.validate.rented_state", "pod_id": "pod-1",
                  "container_name": "tenant-123", "container": {"container_exit_code": 2, "container_oom_killed": False}},
    )
    _, payload = service.publish.await_args.args

    request = await _queued_reset_request(payload)

    assert request.validator_hotkey == "vk" and request.miner_hotkey == "hk" and request.executor_uuid == "ex-1"
    assert request.reason == ResetVerifiedJobReason.POD_NOT_RUNNING
    assert request.reason_code == "POD_NOT_RUNNING" and request.check_id == "executor.validate.rented_state"
    assert request.evidence.pod_id == "pod-1" and request.evidence.container_name == "tenant-123"
    assert request.evidence.container.container_exit_code == 2 and request.evidence.container.container_oom_killed is False
    assert request.evidence.container.container_status is None
    # what send_model puts on the wire (model_dump_json, nothing excluded) is the redis payload's evidence
    # unchanged: the six container fields the host did not answer are absent, not null — the backend's JSONB
    # row (lium-platform#325) must not fill up with nulls the validator never saw
    wire = json.loads(request.model_dump_json())
    assert wire["evidence"] == payload["evidence"] == {
        "pod_id": "pod-1", "container_name": "tenant-123",
        "container": {"container_exit_code": 2, "container_oom_killed": False},
    }

    await service.clear_verified_job_info(miner_hotkey="hk", executor_id="ex-2", prev_info={})
    _, bare = service.publish.await_args.args
    plain = await _queued_reset_request(bare)
    assert plain.reason == ResetVerifiedJobReason.DEFAULT
    assert plain.reason_code is None and plain.check_id is None and plain.evidence is None
