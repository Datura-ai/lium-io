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
async def test_the_reset_carries_the_address_and_cycle_this_check_ran_in(context_factory):
    # DAH-2865: the backend drops a reset that arrives after the node moved to another address under the same id,
    # timed by its cycle however late it arrives
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
        "job_batch_id": ctx.config.job_batch_id,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("unrented_only", [True, False])
async def test_an_idle_only_reset_travels_as_its_own_field_not_in_the_evidence(context_factory, unrented_only):
    # DAH-6784: the backend may cut an oversized evidence to its keys; the idle-only mark must not ride in it
    from services.redis_service import RedisService

    redis = AsyncMock()
    ctx = context_factory(state=build_state()).model_copy(
        update={
            "clear_verified_job_info": True,
            "clear_verified_job_evidence": {"reason_code": "NODE_ACCOUNT_MISMATCH", "container_logs": "x" * 8192},
            "clear_verified_job_unrented_only": unrented_only,
        }
    )
    await ResultHandler(redis_service=redis, dry_run=False)._persist_verification_data(
        "miner-hotkey", "executor-123", {"count": 7, "failed": 0, "spec": "s", "uuids": "GPU-a"}, ctx, success=False
    )
    kwargs = redis.clear_verified_job_info.await_args.kwargs
    assert kwargs["unrented_only"] is unrented_only

    service = object.__new__(RedisService)
    service.hset, service.publish = AsyncMock(), AsyncMock()
    await RedisService.clear_verified_job_info(service, "miner-hotkey", "executor-123", **{
        k: kwargs[k] for k in ("prev_info", "reason", "evidence", "unrented_only")
    })
    published = service.publish.await_args.args[1]
    assert published["unrented_only"] is unrented_only
    assert "unrented_only" not in published["evidence"]


def test_a_later_clearing_check_never_inherits_an_idle_only_mark():
    # the pipeline copies each check's updates onto the context: a clearing check without the mark resets it to False
    from datetime import UTC, datetime

    from services.task.models import ValidationEvent
    from services.task.pipeline import CheckResult, updates_with_clear_verified_job_evidence

    event = ValidationEvent(
        event="banned", reason_code="GPU_BANNED", severity="error", impact="", check_id="gpu.validate.banned",
        when=datetime.now(UTC),
    )
    plain = CheckResult(passed=False, event=event, updates={"clear_verified_job_info": True})
    marked = CheckResult(
        passed=False, event=event, updates={"clear_verified_job_info": True, "clear_verified_job_unrented_only": True}
    )
    assert updates_with_clear_verified_job_evidence(plain, "x")["clear_verified_job_unrented_only"] is False
    assert updates_with_clear_verified_job_evidence(marked, "x")["clear_verified_job_unrented_only"] is True


def test_a_passing_idle_only_check_appends_its_id_and_a_failed_one_does_not():
    # DAH-6784: the backend lifts a hold an idle-only check left only when that check's id is listed
    from datetime import UTC, datetime

    from services.task.models import ValidationEvent
    from services.task.pipeline import CheckResult, updates_with_clear_verified_job_evidence

    def result(passed):
        event = ValidationEvent(
            event="probe", reason_code="PROBE", severity="info", impact="", check_id="executor.validate.rental_probe",
            when=datetime.now(UTC),
        )
        return CheckResult(passed=passed, event=event, updates={"idle_check_passed": True})

    assert updates_with_clear_verified_job_evidence(result(True), "x", ("executor.validate.node_account",)) == {
        "idle_checks_passed": ("executor.validate.node_account", "executor.validate.rental_probe")
    }
    assert updates_with_clear_verified_job_evidence(result(False), "x", ("a",)) == {}


def test_a_check_that_names_its_proven_parts_appends_those_ids():
    # DAH-6784: node_account lists the parts it proved, so a hold one part left lifts only on proof of that part
    from datetime import UTC, datetime

    from services.task.models import ValidationEvent
    from services.task.pipeline import CheckResult, updates_with_clear_verified_job_evidence

    event = ValidationEvent(
        event="ok", reason_code="NODE_ACCOUNT_OK", severity="info", impact="", check_id="executor.validate.node_account",
        when=datetime.now(UTC),
    )
    parts = ("executor.validate.node_account.host_key",)
    marked = CheckResult(passed=True, event=event, updates={"idle_check_passed": parts})
    assert updates_with_clear_verified_job_evidence(marked, "x", ("a",)) == {"idle_checks_passed": ("a", *parts)}
    for bad in ((), ("x", 1), ["x"], "x"):
        odd = CheckResult(passed=True, event=event, updates={"idle_check_passed": bad})
        assert updates_with_clear_verified_job_evidence(odd, "x") == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("success", [True, False])
async def test_the_result_carries_the_idle_only_checks_that_passed_once_each(context_factory, success):
    from types import SimpleNamespace

    ctx = context_factory(state=build_state(specs={"gpu": {"count": 1}})).model_copy(
        update={"idle_checks_passed": ("executor.validate.rental_probe", "executor.validate.rental_probe")}
    )
    result = await ResultHandler(redis_service=AsyncMock(), dry_run=True).handle_result(
        context=ctx,
        miner_info=SimpleNamespace(miner_hotkey="miner-hotkey", job_batch_id="batch-1"),
        executor_info=ctx.executor,
        verified_job_info={},
        log_text="x",
        success=success,
    )
    assert result.idle_checks_passed == ["executor.validate.rental_probe"]
