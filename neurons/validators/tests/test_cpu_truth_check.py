"""DAH-2671 item 2a — CpuTruthCheck.

RED→GREEN: before the check existed, nothing compared the advertised CPU(s) count against the
kernel-present population, so a host advertising 176 cores over 44 real ones passed unnoticed. The
mismatch test fails closed; an inability to measure never fails.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from neurons.validators.src.services.task.checks.cpu_truth import (
    CpuTruthCheck,
)
from neurons.validators.src.services.task.messages import CpuTruthMessages as Msg


def _fake_ssh(*, present="0-127", raise_exc=None):
    async def run(cmd, *args, **kwargs):
        if raise_exc is not None:
            raise raise_exc
        if "/sys/devices/system/cpu/present" in cmd:
            return SimpleNamespace(exit_status=0, stdout=present, stderr="")
        return SimpleNamespace(exit_status=1, stdout="", stderr="")

    return SimpleNamespace(run=run)


def _state(advertised):
    from tests.helpers import build_state

    return build_state(specs={"cpu": {"count": advertised}})


# ---------------------------- _count_cpu_list (pure) ----------------------------


# ---------------------------- behavior ----------------------------


@pytest.mark.asyncio
async def test_mismatch_fails(context_factory):
    # advertised 176 over a 44-core present population — the spoof signature.
    ctx = context_factory(state=_state(176), ssh=_fake_ssh(present="0-43"))
    result = await CpuTruthCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.CPU_MISMATCH.reason
    assert result.event.what_we_saw["advertised"] == 176
    assert result.event.what_we_saw["present"] == 44
    # The check is non-fatal and ScoreCheck runs after it, so the verdict travels as a context
    # flag calculate_scores gates on — see test_mismatch_zeroes_the_final_score.
    assert result.updates["cpu_truth_passed"] is False
    assert "clear_verified_job_info" not in result.updates


@pytest.mark.asyncio
async def test_offlined_cores_not_a_spoof(context_factory):
    # honest host with cores offlined: present (== lscpu CPU(s)) equals advertised.
    ctx = context_factory(state=_state(128), ssh=_fake_ssh(present="0-127"))
    result = await CpuTruthCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.CPU_OK.reason


