"""DAH-2671 item 2a — CpuTruthCheck.

RED→GREEN: before the check existed, nothing compared the advertised CPU(s) count against the
kernel-present population, so a host advertising 176 cores over 44 real ones passed unnoticed. The
mismatch test fails closed only under CPU_TRUTH_ENFORCEMENT_ENABLED; shadow observes and passes; an
inability to measure never fails.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from neurons.validators.src.services.task.checks.cpu_truth import (
    CpuTruthCheck,
    _count_cpu_list,
)
from neurons.validators.src.services.task.messages import CpuTruthMessages as Msg


def _fake_ssh(*, cpuinfo="128", present="0-127", ncpu="128", raise_exc=None):
    async def run(cmd, *args, **kwargs):
        if raise_exc is not None:
            raise raise_exc
        if "/proc/cpuinfo" in cmd:
            return SimpleNamespace(exit_status=0, stdout=cpuinfo, stderr="")
        if "/sys/devices/system/cpu/present" in cmd:
            return SimpleNamespace(exit_status=0, stdout=present, stderr="")
        if "docker info" in cmd:
            return SimpleNamespace(exit_status=0, stdout=ncpu, stderr="")
        return SimpleNamespace(exit_status=1, stdout="", stderr="")

    return SimpleNamespace(run=run)


def _state(advertised):
    from tests.helpers import build_state

    return build_state(specs={"cpu": {"count": advertised}})


# ---------------------------- _count_cpu_list (pure) ----------------------------


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("0-127", 128),
        ("0-19,40-59", 40),
        ("0", 1),
        ("", None),
        ("garbage", None),
    ],
)
def test_count_cpu_list(spec, expected):
    assert _count_cpu_list(spec) == expected


# ---------------------------- behavior ----------------------------


@pytest.mark.asyncio
async def test_match_passes(context_factory):
    ctx = context_factory(state=_state(128), ssh=_fake_ssh(present="0-127"))
    with patch("neurons.validators.src.services.task.checks.cpu_truth.settings") as s:
        s.CPU_TRUTH_CHECK_ENABLED = True
        s.CPU_TRUTH_ENFORCEMENT_ENABLED = False
        result = await CpuTruthCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.CPU_OK.reason
    assert result.event.what_we_saw["advertised"] == 128
    assert result.event.what_we_saw["present"] == 128


@pytest.mark.asyncio
async def test_mismatch_emits_under_shadow_and_still_passes(context_factory):
    # advertised 176 over a 44-core present population — the spoof signature.
    ctx = context_factory(state=_state(176), ssh=_fake_ssh(present="0-43", cpuinfo="44"))
    with patch("neurons.validators.src.services.task.checks.cpu_truth.settings") as s:
        s.CPU_TRUTH_CHECK_ENABLED = True
        s.CPU_TRUTH_ENFORCEMENT_ENABLED = False
        result = await CpuTruthCheck().run(ctx)

    assert result.passed is True  # shadow: never changes score
    assert result.event.reason_code == Msg.CPU_MISMATCH.reason
    assert result.event.what_we_saw["advertised"] == 176
    assert result.event.what_we_saw["present"] == 44


@pytest.mark.asyncio
async def test_mismatch_fails_under_enforcement(context_factory):
    ctx = context_factory(state=_state(176), ssh=_fake_ssh(present="0-43", cpuinfo="44"))
    with patch("neurons.validators.src.services.task.checks.cpu_truth.settings") as s:
        s.CPU_TRUTH_CHECK_ENABLED = True
        s.CPU_TRUTH_ENFORCEMENT_ENABLED = True
        result = await CpuTruthCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.CPU_MISMATCH.reason


@pytest.mark.asyncio
async def test_ssh_error_passes_with_unknown_reason(context_factory):
    ctx = context_factory(state=_state(176), ssh=_fake_ssh(raise_exc=OSError("ssh down")))
    with patch("neurons.validators.src.services.task.checks.cpu_truth.settings") as s:
        s.CPU_TRUTH_CHECK_ENABLED = True
        s.CPU_TRUTH_ENFORCEMENT_ENABLED = True  # even under enforcement, cannot-measure never fails
        result = await CpuTruthCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.CPU_UNKNOWN.reason


@pytest.mark.asyncio
async def test_offlined_cores_not_a_spoof(context_factory):
    # honest host with cores offlined: present (== lscpu CPU(s)) equals advertised; online is lower.
    ctx = context_factory(state=_state(128), ssh=_fake_ssh(present="0-127", cpuinfo="64"))
    with patch("neurons.validators.src.services.task.checks.cpu_truth.settings") as s:
        s.CPU_TRUTH_CHECK_ENABLED = True
        s.CPU_TRUTH_ENFORCEMENT_ENABLED = True
        result = await CpuTruthCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.CPU_OK.reason


@pytest.mark.asyncio
async def test_missing_advertised_count_is_unknown(context_factory):
    from tests.helpers import build_state

    ctx = context_factory(state=build_state(specs={"cpu": {}}), ssh=_fake_ssh())
    with patch("neurons.validators.src.services.task.checks.cpu_truth.settings") as s:
        s.CPU_TRUTH_CHECK_ENABLED = True
        s.CPU_TRUTH_ENFORCEMENT_ENABLED = True
        result = await CpuTruthCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.CPU_UNKNOWN.reason


@pytest.mark.asyncio
async def test_disabled_master_switch_skips(context_factory):
    ctx = context_factory(state=_state(176), ssh=_fake_ssh(present="0-43"))
    with patch("neurons.validators.src.services.task.checks.cpu_truth.settings") as s:
        s.CPU_TRUTH_CHECK_ENABLED = False
        s.CPU_TRUTH_ENFORCEMENT_ENABLED = True
        result = await CpuTruthCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.SKIPPED.reason
