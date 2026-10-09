"""One machine earns under one miner hotkey per cycle.

One executor process can sign for two miner hotkeys (MINER_HOTKEY_SS58_ADDRESS plus
DEFAULT_MINER_HOTKEY), so both hotkeys list the same machine and both were scored for it in the
same cycle. DuplicateExecutorCheck cannot see it: its Redis set comes from the backend, keyed per
hotkey, and the backend keeps one row for the shared executor UUID.

Copies the validator reached on one SSH endpoint are one machine and one hotkey keeps the score;
matches on node-reported values (executor UUID, listed ip:port, GPU UUID) are logged only.
"""

import logging

import pytest
from core import validator as validator_module
from core.config import settings
from core.validator import settle_cycle_results, specs_to_publish
from datura.requests.miner_requests import ExecutorSSHInfo
from incentive.miner_incentive_log import MinerLogLine
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse
from services.task.checks.duplicate_executor import (
    ACROSS_MINERS_OUTCOME,
    MATCH_EXECUTOR_UUID,
    MATCH_GPU_UUID,
    MATCH_IP_PORT,
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


def _across_miner_logs(caplog) -> list:
    return [r.msg for r in caplog.records if getattr(r.msg, "extra", {}).get("outcome") == ACROSS_MINERS_OUTCOME]


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


def test_two_executor_uuids_on_one_ssh_endpoint_are_one_machine(enforce):
    kept, dropped = _result("uuid-a"), _result("uuid-b")

    keep_one_miner_per_executor({ALICE: [dropped], BOB: [kept]})

    _assert_untouched(kept)
    _assert_zeroed(dropped, kept_by=BOB)
    assert not dropped.duplicate_shares_kept_row


def test_a_rented_copy_keeps_the_score(enforce):
    idle, rented = _result("uuid-a"), _result("uuid-b", is_rented=True)

    keep_one_miner_per_executor({BOB: [idle], ALICE: [rented]})

    _assert_untouched(rented)
    _assert_zeroed(idle, kept_by=ALICE)


def test_the_hotkey_the_rental_names_keeps_a_shared_rented_executor(enforce):
    # both copies read as rented: the rented list is keyed by executor UUID alone
    holder, other = _result("uuid-a", is_rented=True), _result("uuid-a", is_rented=True)
    rented = RentedExecutorsResponse(
        executors={
            "UUID-A": RentedExecutor(
                miner_hotkey=ALICE, executor_ip_address="198.51.100.7", executor_ip_port="8001", pods=[]
            )
        }
    )

    settle_cycle_results({BOB: [other], ALICE: [holder]}, {}, rented)

    _assert_untouched(holder)
    _assert_zeroed(other, kept_by=ALICE)


def test_a_forced_pass_the_validator_did_not_log_in_for_proves_no_endpoint(enforce):
    first = _result("uuid-a", ssh_port=0, is_rented=True)
    second = _result("uuid-b", ssh_port=0, is_rented=True)

    keep_one_miner_per_executor({BOB: [first], ALICE: [second]})

    _assert_untouched(first)
    _assert_untouched(second)


def test_every_other_hotkey_on_the_endpoint_loses_its_copy(enforce):
    a, b, c = _result("uuid-a"), _result("uuid-b"), _result("uuid-c")

    duplicates = keep_one_miner_per_executor({ALICE: [a], BOB: [b], CHARLIE: [c]})

    _assert_untouched(b)
    _assert_zeroed(a, kept_by=BOB)
    _assert_zeroed(c, kept_by=BOB)
    assert sorted(d.hotkey for d in duplicates) == sorted([ALICE, CHARLIE])


@pytest.mark.parametrize(
    "first, second, matched_on",
    [
        (_result("uuid-a", address="198.51.100.7"), _result("uuid-a", address="203.0.113.9"), MATCH_EXECUTOR_UUID),
        (_result("uuid-a", ssh_port=2200), _result("uuid-b", ssh_port=2201), MATCH_IP_PORT),
        (
            _result("uuid-a", address="198.51.100.7", gpu_uuids=("GPU-1", "GPU-2")),
            _result("uuid-b", address="203.0.113.9", gpu_uuids=("GPU-2",)),
            MATCH_GPU_UUID,
        ),
    ],
    ids=["executor-uuid", "listed-ip-port", "gpu-uuid"],
)
def test_a_node_reported_match_on_another_endpoint_is_logged_and_never_zeroed(enforce, caplog, first, second, matched_on):
    with caplog.at_level(logging.WARNING):
        duplicates = keep_one_miner_per_executor({BOB: [first], ALICE: [second]})

    _assert_untouched(first)
    _assert_untouched(second)
    assert sorted((d.hotkey, d.matched_on, d.kept_by, d.enforced) for d in duplicates) == sorted(
        [(BOB, matched_on, None, False), (ALICE, matched_on, None, False)]
    )
    assert {log.extra["miner_hotkey"] for log in _across_miner_logs(caplog)} == {BOB, ALICE}


def test_a_single_hotkey_is_unaffected(enforce):
    own_repeat = [_result("uuid-a"), _result("uuid-b")]
    other = [_result("uuid-x", address="203.0.113.9", gpu_uuids=("GPU-7",))]

    assert keep_one_miner_per_executor({BOB: own_repeat, ALICE: other}) == []
    for result in own_repeat + other:
        _assert_untouched(result)


def test_an_unscored_copy_takes_no_part(enforce):
    scored, failed = _result("uuid-a"), _result("uuid-a", score=0)

    assert keep_one_miner_per_executor({BOB: [failed], ALICE: [scored]}) == []
    _assert_untouched(scored)


def test_dry_run_logs_both_hotkeys_and_changes_nothing(monkeypatch, caplog):
    monkeypatch.setattr(settings, "DUPLICATE_EXECUTOR_DRY_RUN", True)
    first, second = _result("uuid-a"), _result("uuid-a")

    with caplog.at_level(logging.WARNING):
        duplicates = keep_one_miner_per_executor({BOB: [first], ALICE: [second]})

    _assert_untouched(first)
    _assert_untouched(second)
    assert not second.duplicate_shares_kept_row
    assert [(d.hotkey, d.kept_by, d.enforced) for d in duplicates] == [(ALICE, BOB, False)]
    logged = _across_miner_logs(caplog)
    assert [(log.extra["miner_hotkey"], log.extra["kept_by_hotkey"], log.extra["enforced"]) for log in logged] == [
        (ALICE, BOB, False)
    ]
    assert "observe mode" in logged[0].message


def test_enforce_logs_both_hotkeys(enforce, caplog):
    with caplog.at_level(logging.WARNING):
        keep_one_miner_per_executor({BOB: [_result("uuid-a")], ALICE: [_result("uuid-a")]})

    logged = _across_miner_logs(caplog)
    assert [(log.extra["miner_hotkey"], log.extra["kept_by_hotkey"], log.extra["enforced"]) for log in logged] == [
        (ALICE, BOB, True)
    ]


@pytest.mark.parametrize("dry_run, gpus", [(True, 16), (False, 8)], ids=["dry-run", "enforce"])
def test_the_tier_counts_a_shared_machine_once_only_when_enforced(monkeypatch, dry_run, gpus):
    monkeypatch.setattr(settings, "DUPLICATE_EXECUTOR_DRY_RUN", dry_run)
    results = {BOB: [_result("uuid-a")], ALICE: [_result("uuid-a")]}

    assert settle_cycle_results(results, {}) == {H200: gpus}


def test_a_failing_duplicate_pass_leaves_the_cycle_scoring(monkeypatch):
    def broken(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(validator_module, "keep_one_miner_per_executor", broken)
    results = {BOB: [_result("uuid-a")], ALICE: [_result("uuid-a")]}

    assert settle_cycle_results(results, {}) == {H200: 16}


def test_only_a_zeroed_copy_of_the_keepers_own_row_is_not_published(enforce):
    same_row = {BOB: [_result("uuid-a")], ALICE: [_result("uuid-a")]}
    own_row = {BOB: [_result("uuid-a")], ALICE: [_result("uuid-b")]}
    keep_one_miner_per_executor(same_row)
    keep_one_miner_per_executor(own_row)

    assert specs_to_publish(same_row[ALICE]) == []
    assert specs_to_publish(own_row[ALICE]) == own_row[ALICE]
    assert specs_to_publish(same_row[BOB]) == same_row[BOB]
