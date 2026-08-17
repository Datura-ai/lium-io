"""File-backed PodLog store (DAH-2391: per-host postgres removed).

The store must keep the old SQL semantics (rows for the container plus rows
with no container name, oldest first) and must never raise: a full or broken
disk degrades pod-log collection only, never the agent.
"""

import asyncio
from datetime import datetime

import pytest

from models.pod_log import PodLog
from services import pod_log_store
from services.pod_log_service import PodLogService


@pytest.fixture()
def store_file(tmp_path, monkeypatch):
    path = tmp_path / "pod_logs.jsonl"
    monkeypatch.setattr(pod_log_store, "POD_LOG_FILE", path)
    return path


def test_append_and_find_roundtrip_keeps_sql_semantics(store_file):
    # Arrange: two events for the container, one global error row, one foreign row
    newer = PodLog(container_name="container_a", event="die", exit_code=137,
                   reason="immediate_termination", created_at=datetime(2026, 7, 10, 12, 5))
    older = PodLog(container_name="container_a", event="start", created_at=datetime(2026, 7, 10, 12, 0))
    global_error = PodLog(error="Docker API error", created_at=datetime(2026, 7, 10, 12, 1))
    foreign = PodLog(container_name="container_b", event="start", created_at=datetime(2026, 7, 10, 12, 2))

    for log in [newer, older, global_error, foreign]:
        pod_log_store.append_pod_log(log)

    # Act
    result = pod_log_store.find_by_container_name("container_a")

    # Assert: foreign row excluded, null-container row included, sorted oldest first
    assert [log.uuid for log in result] == [older.uuid, global_error.uuid, newer.uuid]
    assert result[2].exit_code == 137
    assert result[2].reason == "immediate_termination"


def test_find_skips_corrupt_lines(store_file):
    # Arrange: a valid line surrounded by garbage and a torn write
    valid = PodLog(container_name="container_a", event="start")
    pod_log_store.append_pod_log(valid)
    with open(store_file, "a", encoding="utf-8") as f:
        f.write("{not json\n")
        f.write('{"container_name": "container_a", "created_at": 123')  # torn, no newline

    # Act
    result = pod_log_store.find_by_container_name("container_a")

    # Assert
    assert [log.uuid for log in result] == [valid.uuid]


def test_find_returns_empty_when_file_missing(store_file):
    # Act
    result = pod_log_store.find_by_container_name("container_a")

    # Assert
    assert result == []


def test_append_never_raises_on_disk_failure(store_file, monkeypatch):
    # Arrange: every open() fails as on a full disk
    def broken_open(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("builtins.open", broken_open)

    # Act / Assert: no exception reaches the caller
    pod_log_store.append_pod_log(PodLog(container_name="container_a", event="start"))


def test_append_over_cap_drops_oldest_events_and_keeps_the_newest(store_file, monkeypatch):
    # Arrange: ten equally sized lines, then a cap that only half of them fit under
    written = [
        PodLog(container_name="container_a", event=f"event_{i}",
               created_at=datetime(2026, 7, 10, 12, i))
        for i in range(10)
    ]
    for log in written:
        pod_log_store.append_pod_log(log)
    line_size = store_file.stat().st_size // len(written)
    monkeypatch.setattr(pod_log_store, "MAX_FILE_SIZE_BYTES", 5 * line_size)
    monkeypatch.setattr(pod_log_store, "KEEP_TAIL_BYTES", 5 * line_size)
    fresh = PodLog(container_name="container_a", event="event_new",
                   created_at=datetime(2026, 7, 10, 12, 10))

    # Act
    pod_log_store.append_pod_log(fresh)

    # Assert: the five oldest events are gone, the newest five plus the fresh one stay
    result = pod_log_store.find_by_container_name("container_a")
    assert [log.uuid for log in result] == [log.uuid for log in written[5:]] + [fresh.uuid]


def test_pod_log_service_reads_from_store(store_file):
    # Arrange
    log = PodLog(container_name="container_a", event="start")
    pod_log_store.append_pod_log(log)

    # Act
    result = asyncio.run(PodLogService().find_by_continer_name("container_a"))

    # Assert
    assert [entry.uuid for entry in result] == [log.uuid]
