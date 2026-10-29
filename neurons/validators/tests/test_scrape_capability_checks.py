"""run_checks_concurrently() in machine_scrape.py: the sysbox and storage-limit capability tests
(one `docker run --gpus all` each) run side by side, and each result comes back in its own slot.

machine_scrape.py is a script, not a module, so the helper is extracted by ast (helpers.build_scrape_namespace).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest
from neurons.validators.tests.helpers import build_scrape_namespace

SRC = Path(__file__).resolve().parents[1] / "src"


@pytest.fixture
def scrape() -> dict[str, Any]:
    return build_scrape_namespace(
        SRC / "miner_jobs" / "machine_scrape.py",
        {"run_checks_concurrently", "store_check_result"},
        {"threading": threading},
    )


def test_capability_checks_run_at_the_same_time(scrape: dict[str, Any]) -> None:
    # Arrange: each check returns only once the other one is running too
    both_running = threading.Barrier(2, timeout=5)

    def check() -> tuple[bool, str]:
        both_running.wait()
        return True, "ok"

    # Act
    results = scrape["run_checks_concurrently"]([check, check])

    # Assert
    assert results == [(True, "ok"), (True, "ok")]


def test_each_capability_result_keeps_its_checks_position(scrape: dict[str, Any]) -> None:
    # Arrange
    def sysbox_check() -> tuple[bool, str]:
        return False, "Sysbox runtime does not support GPU access."

    def storage_limit_check() -> tuple[bool, str]:
        return True, "Storage limit is supported."

    # Act
    results = scrape["run_checks_concurrently"]([sysbox_check, storage_limit_check])

    # Assert
    assert results == [
        (False, "Sysbox runtime does not support GPU access."),
        (True, "Storage limit is supported."),
    ]
