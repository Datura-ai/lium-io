"""A stand-in supervisor that detaches exactly as the real one does, then waits.

Used by `test_supervisor.py` in place of `cvmd.dstack.child`. It calls the real `_detach`, so
what the test asserts about sessions, parents and reaping is asserted about the shipping code —
only the part that would start QEMU is replaced.

Invoked with the same argv `spawn` builds: <scripts_dir> <vm_dir> <kp_port>.
"""

import sys
import time
from pathlib import Path

from cvmd.dstack.child import _detach

SLEEP_SECONDS = 120


def main() -> int:
    _detach(Path(sys.argv[2]))
    # Deliberately NOT flushed, exactly as `run_instance` prints the QEMU command line. Whether
    # this reaches the console log while the process is still running is entirely down to `-u`.
    print("detached")
    time.sleep(SLEEP_SECONDS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
