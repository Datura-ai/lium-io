"""DAH-6710: the validator reports, per rented pod, a dead /proc FUSE (RENTED_POD_PROC_FUSE_DISCONNECTED)."""

from __future__ import annotations

import copy
import os
from types import SimpleNamespace

import pytest
from neurons.validators.src.services.task.checks.rented_pod_ownership import RentedPodOwnershipCheck
from neurons.validators.src.services.task.result_handler import ResultHandler
from protocol.vc_protocol import validator_requests

from tests.helpers import build_state

# test_protocol_compat puts lium_protocol on sys.path
from tests.test_protocol_compat import _same_wire_view, _validator_parse_validator_message, lium_protocol, recorded
from tests.test_rented_pod_ownership_check import _ctx as _ownership_ctx
from tests.test_rented_pod_ownership_check import _proc
from tests.test_rented_pod_ownership_check import _ssh as _ownership_ssh

PROC_FUSE = "RENTED_POD_PROC_FUSE_DISCONNECTED"


def _old_spec() -> dict:
    """A recorded ExecutorSpecRequest: what a validator that predates 2.17.0 sends."""
    message = next(e["message"] for e in recorded("validator_to_backend") if e["expect"] == "ExecutorSpecRequest")
    assert "rented_pod_faults" not in message
    return copy.deepcopy(message)


def test_an_old_shape_spec_parses_with_the_field_absent():
    message = _old_spec()
    ours = lium_protocol.VALIDATOR_MESSAGES.parse_obj(message)
    theirs = _validator_parse_validator_message(message)
    assert ours.rented_pod_faults is None and theirs.rented_pod_faults is None


def test_the_field_reaches_the_wire_the_same_on_both_sides():
    message = {**_old_spec(), "rented_pod_faults": {"pod-1": [PROC_FUSE], "pod-2": []}}
    ours = lium_protocol.VALIDATOR_MESSAGES.parse_obj(message)
    theirs = _validator_parse_validator_message(message)
    _same_wire_view(message, ours, theirs)
    assert isinstance(theirs, validator_requests.ExecutorSpecRequest)
    assert theirs.rented_pod_faults == {"pod-1": [PROC_FUSE], "pod-2": []}


async def _job_result(ctx):
    return await ResultHandler(redis_service=None, dry_run=True).handle_result(
        context=ctx,
        miner_info=SimpleNamespace(miner_hotkey="miner-hotkey", job_batch_id="batch-1"),
        executor_info=ctx.executor,
        verified_job_info={},
        log_text="x",
        success=True,
    )


@pytest.mark.asyncio
async def test_the_cycle_result_carries_the_verdicts(context_factory):
    state = build_state(rented_pod_faults={"1": [PROC_FUSE]})
    result = await _job_result(context_factory(state=state))
    assert result.rented_pod_faults == {"1": [PROC_FUSE]}


@pytest.mark.asyncio
async def test_a_cycle_that_judged_no_pod_reports_no_verdicts(context_factory):
    result = await _job_result(context_factory(state=build_state()))
    assert result.rented_pod_faults is None


@pytest.mark.asyncio
async def test_a_shifted_pod_gets_no_per_pod_code(context_factory, tmp_path):
    # ownership shift (and the runtime-restart read behind NVLink) stays an event: no per-pod verdict is reported
    me = os.getuid()
    _proc(tmp_path, 4242, f"         0     {me + 200000}      65536\n")
    _proc(tmp_path, 4343, f"         0     {me}      65536\n")
    result = await RentedPodOwnershipCheck().run(_ownership_ctx(context_factory, _ownership_ssh(str(tmp_path))))
    assert result.updates == {}
