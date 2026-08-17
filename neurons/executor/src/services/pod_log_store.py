"""File-backed store for PodLog container lifecycle events.

Replaces the per-host Postgres that existed only for this one diagnostic
table (DAH-2391): the DB container dying used to crash-loop the whole agent
via the alembic gate in run.sh. Events are appended as JSON lines by
monitor.py and read back by /pod_logs. Every operation is best-effort: a
full or failing disk degrades pod-log collection but can never take the
agent (or its /ping and /upload_ssh_key endpoints) down.
"""

import os
from pathlib import Path

from core.logger import get_logger
from models.pod_log import PodLog

logger = get_logger(__name__)

POD_LOG_FILE = Path(os.getenv("POD_LOG_FILE", "/var/lib/executor-pod-logs/pod_logs.jsonl"))
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024


def append_pod_log(log: PodLog) -> None:
    # best-effort append that never raises, so a full disk cannot break the caller
    try:
        POD_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        if POD_LOG_FILE.exists() and POD_LOG_FILE.stat().st_size > MAX_FILE_SIZE_BYTES:
            logger.warning(
                "Pod log file %s exceeds %d bytes, dropping event",
                POD_LOG_FILE,
                MAX_FILE_SIZE_BYTES,
            )
            return
        with open(POD_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log.model_dump_json() + "\n")
    except Exception as e:
        logger.warning("Failed to append pod log: %s", e)


def find_by_container_name(container_name: str) -> list[PodLog]:
    # same result set as the old SQL query: rows for this container plus rows
    # without a container name (monitor-level errors), oldest first
    logs: list[PodLog] = []
    try:
        with open(POD_LOG_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    log = PodLog.model_validate_json(line)
                except Exception:
                    # torn or corrupt line (e.g. a write cut short by a full disk)
                    continue
                if log.container_name == container_name or log.container_name is None:
                    logs.append(log)
    except FileNotFoundError:
        return []
    except Exception as e:
        logger.warning("Failed to read pod logs: %s", e)
        return []
    logs.sort(key=lambda log: log.created_at)
    return logs
