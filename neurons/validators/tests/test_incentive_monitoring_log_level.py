"""The per-executor job-result dump after scoring is DEBUG; the summary lines stay at INFO."""

import logging
from time import time

import pytest

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


@pytest.mark.parametrize(("level", "dumps"), [(logging.DEBUG, 2), (logging.INFO, 0)])
def test_per_executor_job_result_dump_is_debug_and_summary_stays_info(caplog, level, dumps):
    caplog.set_level(level, logger="incentive.utils")

    log_for_monitoring({"miner": [_job("exec-1"), _job("exec-2")]}, time())

    records = [(r.getMessage(), r.levelno) for r in caplog.records]
    assert [r for r in records if r[1] >= logging.INFO] == [("Incentive_results", logging.INFO)]
    assert records.count(("", logging.DEBUG)) == dumps
