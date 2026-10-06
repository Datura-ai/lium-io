"""One machine earns under one miner hotkey per cycle.

Regression: one executor process can sign for two miner hotkeys (MINER_HOTKEY_SS58_ADDRESS plus
DEFAULT_MINER_HOTKEY), so both hotkeys list the same machine and both were scored for it in the
same cycle (13 executors in June 2026, 46 in August, 16 in September). DuplicateExecutorCheck
could not see it: its Redis set comes from the backend, keyed per hotkey, and the backend keeps
one row for the shared executor UUID.

`keep_one_miner_per_executor` joins copies by executor UUID, ip:port and GPU UUID; the hotkey that
sorts first keeps the score, the others score 0 when DUPLICATE_EXECUTOR_DRY_RUN is off.
"""

import logging

import pytest
from core.config import settings
from datura.requests.miner_requests import ExecutorSSHInfo
from incentive.miner_incentive_log import MinerLogLine
from services.task.checks.duplicate_executor import (
    ACROSS_MINERS_OUTCOME,
    MATCH_EXECUTOR_UUID,
    MATCH_GPU_UUID,
    MATCH_IP_PORT,
    keep_one_miner_per_executor,
)
from services.task.models import JobResult

HOTKEY_A = "5CDmAsUADeCP3dFkB58Po3xq9NpLr2Xc3GW7mCKF4Sdy1ySy"
HOTKEY_B = "5DcgzqMmpRTdm6ajx4NbutvRGe257PRFgSavP8np6jqAEpDr"
HOTKEY_C = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
H200 = "NVIDIA H200"
REASON = "EXECUTOR_DUPLICATE_ACROSS_MINERS"


def _result(
    executor_uuid: str,
    *,
    address: str = "198.51.100.7",
    port: int = 8001,
    gpu_uuids: tuple[str, ...] = (),
    score: float = 1.0,
) -> JobResult:
    return JobResult(
        spec={"gpu": {"count": len(gpu_uuids) or 8, "details": [{"name": H200, "uuid": u} for u in gpu_uuids]}},
        executor_info=ExecutorSSHInfo(
            uuid=executor_uuid,
            address=address,
            port=port,
            ssh_username="root",
            ssh_port=2200,
            python_path="/usr/bin/python3",
            root_dir="/root/app",
        ),
        score=score,
        job_score=score,
        job_batch_id="2026-10-06 15:00:00",
        log_status="info",
        log_text="Validation task completed",
        gpu_model=H200,
        gpu_count=8,
    )


@pytest.fixture
def enforce(monkeypatch):
    monkeypatch.setattr(settings, "DUPLICATE_EXECUTOR_DRY_RUN", False)
    monkeypatch.setattr(settings, "DUPLICATE_EXECUTOR_MATCH_GPU_UUID", True)


def _assert_zeroed(result: JobResult, kept_by: str) -> None:
    assert (result.score, result.job_score) == (0, 0)
    assert not result.is_successful
    assert result.duplicate_kept_by == kept_by
    assert result.failure_reason_code == REASON
    assert result.validation_event.reason_code == REASON
    assert MinerLogLine.validation_failure_code(result) == REASON


def _across_miner_logs(caplog) -> list:
    return [r.msg for r in caplog.records if getattr(r.msg, "extra", {}).get("outcome") == ACROSS_MINERS_OUTCOME]


def _assert_untouched(result: JobResult) -> None:
    assert (result.score, result.job_score) == (1.0, 1.0)
    assert result.duplicate_kept_by is None
    assert result.failure_reason_code is None


@pytest.mark.parametrize("order", [(HOTKEY_A, HOTKEY_B), (HOTKEY_B, HOTKEY_A)], ids=["a-first", "b-first"])
def test_one_executor_uuid_under_two_hotkeys_is_scored_once_under_the_first_hotkey(enforce, order):
    copies = {HOTKEY_A: _result("0ec12c01-5f62-41d9-89b7-cabb9dd6b95a"), HOTKEY_B: _result("0EC12C01-5f62-41d9-89b7-cabb9dd6b95a")}
    job_results = {hotkey: [copies[hotkey]] for hotkey in order}

    duplicates = keep_one_miner_per_executor(job_results)

    _assert_untouched(copies[HOTKEY_A])
    _assert_zeroed(copies[HOTKEY_B], kept_by=HOTKEY_A)
    assert [(d.kept_hotkey, d.dropped_hotkey, d.enforced) for d in duplicates] == [(HOTKEY_A, HOTKEY_B, True)]
    assert MATCH_EXECUTOR_UUID in duplicates[0].matched_on


def test_two_executor_uuids_on_one_ip_port_are_one_machine(enforce):
    kept, dropped = _result("uuid-a"), _result("uuid-b")

    duplicates = keep_one_miner_per_executor({HOTKEY_B: [dropped], HOTKEY_A: [kept]})

    _assert_untouched(kept)
    _assert_zeroed(dropped, kept_by=HOTKEY_A)
    assert duplicates[0].matched_on == (MATCH_IP_PORT,)


def test_a_shared_gpu_uuid_joins_two_executors_on_different_addresses(enforce):
    kept = _result("uuid-a", address="198.51.100.7", gpu_uuids=("GPU-1", "GPU-2"))
    dropped = _result("uuid-b", address="203.0.113.9", gpu_uuids=("GPU-2", "GPU-3"))

    duplicates = keep_one_miner_per_executor({HOTKEY_A: [kept], HOTKEY_B: [dropped]})

    _assert_untouched(kept)
    _assert_zeroed(dropped, kept_by=HOTKEY_A)
    assert duplicates[0].matched_on == (MATCH_GPU_UUID,)


def test_gpu_uuid_matching_off_leaves_a_gpu_only_match_alone(enforce, monkeypatch):
    monkeypatch.setattr(settings, "DUPLICATE_EXECUTOR_MATCH_GPU_UUID", False)
    first = _result("uuid-a", address="198.51.100.7", gpu_uuids=("GPU-1",))
    second = _result("uuid-b", address="203.0.113.9", gpu_uuids=("GPU-1",))

    assert keep_one_miner_per_executor({HOTKEY_A: [first], HOTKEY_B: [second]}) == []
    _assert_untouched(first)
    _assert_untouched(second)


def test_matches_are_transitive_and_every_other_hotkey_loses_its_copy(enforce):
    # B shares the executor uuid with A; C shares only B's GPU: all three are one machine.
    a = _result("uuid-a", address="198.51.100.7", port=8001)
    b = _result("uuid-a", address="198.51.100.7", port=8002, gpu_uuids=("GPU-9",))
    c = _result("uuid-c", address="203.0.113.9", gpu_uuids=("GPU-9",))

    duplicates = keep_one_miner_per_executor({HOTKEY_C: [c], HOTKEY_B: [b], HOTKEY_A: [a]})

    _assert_untouched(a)
    _assert_zeroed(b, kept_by=HOTKEY_A)
    _assert_zeroed(c, kept_by=HOTKEY_A)
    assert sorted(d.dropped_hotkey for d in duplicates) == [HOTKEY_B, HOTKEY_C]


def test_a_single_hotkey_is_unaffected(enforce):
    # one hotkey's own repeats are MinerService's job; distinct machines never match
    own_repeat = [_result("uuid-a"), _result("uuid-b")]
    other = [_result("uuid-x", address="203.0.113.9", gpu_uuids=("GPU-7",))]

    assert keep_one_miner_per_executor({HOTKEY_A: own_repeat, HOTKEY_B: other}) == []
    for result in own_repeat + other:
        _assert_untouched(result)


def test_an_unscored_copy_takes_no_part(enforce):
    scored, failed = _result("uuid-a"), _result("uuid-a", score=0)

    assert keep_one_miner_per_executor({HOTKEY_A: [failed], HOTKEY_B: [scored]}) == []
    assert scored.score == 1.0 and scored.duplicate_kept_by is None


def test_dry_run_logs_both_hotkeys_and_changes_nothing(monkeypatch, caplog):
    monkeypatch.setattr(settings, "DUPLICATE_EXECUTOR_DRY_RUN", True)
    first, second = _result("uuid-a"), _result("uuid-a")

    with caplog.at_level(logging.WARNING):
        duplicates = keep_one_miner_per_executor({HOTKEY_A: [first], HOTKEY_B: [second]})

    _assert_untouched(first)
    _assert_untouched(second)
    assert [(d.kept_hotkey, d.dropped_hotkey, d.enforced) for d in duplicates] == [(HOTKEY_A, HOTKEY_B, False)]
    logged = _across_miner_logs(caplog)
    assert len(logged) == 1
    assert (logged[0].extra["kept_by_hotkey"], logged[0].extra["miner_hotkey"]) == (HOTKEY_A, HOTKEY_B)
    assert logged[0].extra["enforced"] is False and "observe mode" in logged[0].message


def test_enforce_logs_both_hotkeys(enforce, caplog):
    with caplog.at_level(logging.WARNING):
        keep_one_miner_per_executor({HOTKEY_A: [_result("uuid-a")], HOTKEY_B: [_result("uuid-a")]})

    logged = _across_miner_logs(caplog)
    assert len(logged) == 1
    assert (logged[0].extra["kept_by_hotkey"], logged[0].extra["miner_hotkey"]) == (HOTKEY_A, HOTKEY_B)
    assert logged[0].extra["enforced"] is True and "observe mode" not in logged[0].message
