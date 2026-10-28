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

from unittest.mock import AsyncMock

import pytest
from services.task.result_handler import ResultHandler

from helpers import build_state


@pytest.mark.asyncio
async def test_the_reset_carries_the_address_this_cycle_checked(context_factory):
    # DAH-2865: the backend drops a reset that arrives after the node moved to another address under the same id
    redis = AsyncMock()
    ctx = context_factory(state=build_state()).model_copy(
        update={"clear_verified_job_info": True, "clear_verified_job_evidence": {"reason_code": "SPEC_CHANGED"}}
    )

    await ResultHandler(redis_service=redis, dry_run=False)._persist_verification_data(
        "miner-hotkey", "executor-123", {"count": 7, "failed": 0, "spec": "s", "uuids": "GPU-a"}, ctx, success=False
    )

    assert redis.clear_verified_job_info.await_args.kwargs["evidence"] == {
        "reason_code": "SPEC_CHANGED",
        "executor_ip": ctx.executor.address,
        "executor_port": ctx.executor.port,
    }
