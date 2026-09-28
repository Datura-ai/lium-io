"""The per-executor job-result dump after scoring is DEBUG; the summary lines stay at INFO."""

import logging
from time import time

from datura.requests.miner_requests import ExecutorSSHInfo

from incentive.utils import log_for_monitoring
from services.task_service import JobResult


def _job(uuid: str) -> JobResult:
    return JobResult(
        executor_info=ExecutorSSHInfo(
            uuid=uuid,
            address="203.0.113.7",
            port=8080,
            ssh_username="root",
            ssh_port=22,
            python_path="/usr/bin/python3",
            root_dir="/tmp",
        ),
        score=1.0,
        job_score=1.0,
        job_batch_id="batch",
        log_status="success",
        log_text="ok",
    )


def test_per_executor_job_result_dump_is_debug_and_summary_stays_info(caplog):
    caplog.set_level(logging.DEBUG, logger="incentive.utils")

    log_for_monitoring({"miner": [_job("exec-1"), _job("exec-2")]}, time())

    dumps = [r for r in caplog.records if r.getMessage() == ""]
    assert [r.levelno for r in dumps] == [logging.DEBUG, logging.DEBUG]
    summary = [r for r in caplog.records if r.getMessage() == "Incentive_results"]
    assert [r.levelno for r in summary] == [logging.INFO]


def test_per_executor_job_result_dump_is_skipped_at_info(caplog):
    caplog.set_level(logging.INFO, logger="incentive.utils")

    log_for_monitoring({"miner": [_job("exec-1")]}, time())

    assert not [r for r in caplog.records if r.getMessage() == ""]
    assert [r.getMessage() for r in caplog.records] == ["Incentive_results"]
