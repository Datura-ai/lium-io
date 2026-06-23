"""Cross-process mutex for docker image pulls.

Two pullers can run on an executor host at the same time:
  * the on-boot cache pre-pull task inside the executor FastAPI app, and
  * the validator-launched ``gpus_utility.py`` script (legacy fallback path).

Both should never ``docker pull`` the same image concurrently. This module
provides a best-effort, non-blocking file lock both can acquire around their
pulls. It deliberately imports nothing heavy (no settings, no docker) so the
standalone ``gpus_utility.py`` script can use it without pulling in the whole
executor configuration.
"""

import fcntl
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Both pullers run inside the same executor container, so a shared path in /tmp
# is visible to both processes.
CACHE_PULL_LOCK_PATH = "/tmp/lium_cache_pull.lock"


@contextmanager
def cache_pull_lock(path: str = CACHE_PULL_LOCK_PATH):
    """Best-effort cross-process lock around a docker pull.

    Yields ``True`` if the lock was acquired (caller may pull), ``False`` if
    another puller currently holds it (caller should skip this cycle). Never
    blocks and never raises on lock contention.
    """
    acquired = False
    try:
        handle = open(path, "w")
    except Exception as e:
        # Failing to even open the lock file should not block pulling; degrade
        # to "no lock" so the cache still gets populated.
        logger.warning(f"cache pull lock unavailable ({e}); proceeding without it")
        yield True
        return

    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError:
            acquired = False
        yield acquired
    finally:
        if acquired:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
        handle.close()
