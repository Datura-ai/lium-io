"""The executor host's kernel.pid_max, as the rental create and the validation both read it (DAH-7041).

The create refuses a rental when the host's pid_max is unusable (docker_service `HOST_READ_PID_MAX_STEP`): with no
safe pids limit a tenant fork bomb starves the host's sshd/dockerd. HostPidMaxCheck reads the same file with the same
parser and threshold, so a node the create would refuse on every rent is not listed in the first place.
"""

from core.config import settings

READ_HOST_PID_MAX_COMMAND = "cat /proc/sys/kernel/pid_max"
# Floor for a rental container's pids.max when it is clamped to a low host kernel.pid_max: a value
# of 0 or below would read as "unlimited" to Docker, so never emit one.
MIN_RENTAL_PIDS_LIMIT = 512
# what a miss keeps of the host's answer for the log: enough to tell a banner from an empty read
_OUTPUT_KEPT_CHARS = 80


def parse_host_pid_max(exit_status: int | None, stdout: str | None) -> tuple[int | None, str]:
    """(pid_max, "") for a usable answer, else (None, why). Only for a read the host answered (an exit status)."""
    if exit_status != 0:
        return None, f"exit status {exit_status}"
    try:
        value = int((stdout or "").strip())
    except (TypeError, ValueError):
        return None, f"unparsable output {(stdout or '')[:_OUTPUT_KEPT_CHARS]!r}"
    if value <= 0:
        return None, "not positive"
    return value, ""


def pid_max_refused(host_pid_max: int | None) -> bool:
    """True when the create refuses: the pids limit is on and the host PID margin cannot be guaranteed."""
    return settings.RENTAL_PIDS_LIMIT_PER_CPU > 0 and (
        host_pid_max is None or host_pid_max - settings.RENTAL_PIDS_LIMIT_HOST_MARGIN < MIN_RENTAL_PIDS_LIMIT
    )
