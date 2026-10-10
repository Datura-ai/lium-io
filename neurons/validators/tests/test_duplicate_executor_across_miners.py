"""One machine earns under one miner hotkey per cycle.

One executor process can sign for two miner hotkeys (MINER_HOTKEY_SS58_ADDRESS plus
DEFAULT_MINER_HOTKEY), so both hotkeys list the same machine and both were scored for it in the
same cycle. DuplicateExecutorCheck cannot see it: its Redis set comes from the backend, keyed per
hotkey, and the backend keeps one row for the shared executor UUID.

Copies the validator reached on one SSH endpoint are one machine and one hotkey keeps the score;
matches on node-reported values (executor UUID, listed ip:port, GPU UUID) are logged only.
"""


import pytest
from core.config import settings
from datura.requests.miner_requests import ExecutorSSHInfo
from incentive.miner_incentive_log import MinerLogLine
from services.task.checks.duplicate_executor import (
    MATCH_SSH_ENDPOINT,
    keep_one_miner_per_executor,
)
from services.task.models import JobResult

# Substrate dev keys; in SS58 string order BOB < CHARLIE < ALICE.
ALICE = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
BOB = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
CHARLIE = "5FLSigC9HGRKVhB9FiEo4Y3koPsNmBmLJbpXg2mp1hXcS59Y"
H200 = "NVIDIA H200"
REASON = "EXECUTOR_DUPLICATE_ACROSS_MINERS"


def _result(
    executor_uuid: str,
    *,
    address: str = "198.51.100.7",
    port: int = 8001,
    ssh_port: int = 2200,
    gpu_uuids: tuple[str, ...] = (),
    score: float = 1.0,
    is_rented: bool = False,
) -> JobResult:
    return JobResult(
        spec={"gpu": {"count": 8, "details": [{"name": H200, "uuid": u} for u in gpu_uuids]}},
        executor_info=ExecutorSSHInfo(
            uuid=executor_uuid,
            address=address,
            port=port,
            ssh_username="root",
            ssh_port=ssh_port,
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
        is_rented=is_rented,
    )


@pytest.fixture
def enforce(monkeypatch):
    monkeypatch.setattr(settings, "DUPLICATE_EXECUTOR_DRY_RUN", False)


def _assert_zeroed(result: JobResult, kept_by: str) -> None:
    assert (result.score, result.job_score) == (0, 0)
    assert not result.is_successful
    assert result.duplicate_kept_by == kept_by
    assert result.failure_reason_code == REASON
    assert result.validation_event.reason_code == REASON
    assert MinerLogLine.validation_failure_code(result) == REASON


def _assert_untouched(result: JobResult) -> None:
    assert (result.score, result.job_score) == (1.0, 1.0)
    assert result.duplicate_kept_by is None
    assert result.failure_reason_code is None


@pytest.mark.parametrize("order", [(BOB, ALICE), (ALICE, BOB)], ids=["keeper-first", "keeper-last"])
def test_one_executor_under_two_hotkeys_is_scored_once_under_the_first_hotkey(enforce, order):
    copies = {BOB: _result("0ec12c01-5f62"), ALICE: _result("0EC12C01-5f62")}

    duplicates = keep_one_miner_per_executor({hotkey: [copies[hotkey]] for hotkey in order})

    _assert_untouched(copies[BOB])
    _assert_zeroed(copies[ALICE], kept_by=BOB)
    assert copies[ALICE].duplicate_shares_kept_row
    assert [(d.hotkey, d.kept_by, d.matched_on, d.enforced) for d in duplicates] == [
        (ALICE, BOB, MATCH_SSH_ENDPOINT, True)
    ]


def test_a_rented_copy_keeps_the_score(enforce):
    idle, rented = _result("uuid-a"), _result("uuid-b", is_rented=True)

    keep_one_miner_per_executor({BOB: [idle], ALICE: [rented]})

    _assert_untouched(rented)
    _assert_zeroed(idle, kept_by=ALICE)


def test_every_other_hotkey_on_the_endpoint_loses_its_copy(enforce):
    a, b, c = _result("uuid-a"), _result("uuid-b"), _result("uuid-c")

    duplicates = keep_one_miner_per_executor({ALICE: [a], BOB: [b], CHARLIE: [c]})

    _assert_untouched(b)
    _assert_zeroed(a, kept_by=BOB)
    _assert_zeroed(c, kept_by=BOB)
    assert sorted(d.hotkey for d in duplicates) == sorted([ALICE, CHARLIE])


