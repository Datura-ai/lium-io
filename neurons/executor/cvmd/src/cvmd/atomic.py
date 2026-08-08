"""Durable atomic file replacement — tmp + fsync + rename + fsync(dir).

Both the state machine and the replay store depend on a write either landing whole or not at all,
and on it being on disk before the caller continues. The replay store is the strict case: its
record must be durable *before* the request it covers reaches a handler, so a crash cannot leave
an already-executed request replayable.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def write_json_durable(path: Path, payload: Any) -> None:
    """Replace `path` with `payload` as JSON, atomically, and return only once it is on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True).encode()

    # NamedTemporaryFile in the same directory: rename(2) is only atomic within a filesystem.
    handle = tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    )
    tmp_path = Path(handle.name)
    try:
        with handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    # The rename itself needs flushing, or a crash can leave the directory entry pointing at
    # nothing even though the file's own contents were synced.
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def read_json(path: Path) -> Any | None:
    """Read JSON from `path`, or None if it is absent, unreadable, or not valid JSON.

    Callers decide what a None means; nothing here treats a corrupt file as an empty one.
    """
    try:
        with path.open("rb") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
