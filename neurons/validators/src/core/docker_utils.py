import asyncio
import json
import shlex
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import asyncssh

from services.const import DOCKER_DIND_IMAGE

# Bounded so a crash-looping worker can't flood the log line; keep the tail,
# not the head — a dying entrypoint prints its fatal error last.
_CONTAINER_DEATH_LOG_TAIL_LINES = 100
_CONTAINER_DEATH_LOG_MAX_CHARS = 4000


class DockerCommand:
    """Builds safe docker command strings."""

    @staticmethod
    def run_alpine(name: str, script: str, network_mode: str, timeout: int) -> str:
        """Build docker run command for Alpine netcat container."""
        heredoc = "__NC_EOF__"
        network_flag = f"--network={network_mode}" if network_mode == "host" else network_mode
        return (
            f"/usr/bin/docker run -d --rm --name {name} "
            f"{network_flag} docker.io/library/alpine:3.19 sh -c "
            f"'cat << \"{heredoc}\" > /tmp/nc.sh\n{script}\n{heredoc}\n"
            f"timeout {timeout} sh /tmp/nc.sh'"
        )

    @staticmethod
    def run_dind(name: str, port: int, public_key: str, sysbox: bool) -> str:
        """Build docker run command for DinD container."""
        runtime = "--runtime=sysbox-runc " if sysbox else ""
        ssh_cmd = (
            "sh -c 'mkdir -p ~/.ssh && echo "
            f"\"{public_key}\" >> ~/.ssh/authorized_keys "
            "&& ssh-keygen -A && service ssh start && tail -f /dev/null'"
        )
        return (
            f"/usr/bin/docker run -d {runtime} --name {name} --gpus all "
            f"-p {port}:22 {DOCKER_DIND_IMAGE} {ssh_cmd}"
        )

    @staticmethod
    def remove_with_volumes(name: str) -> str:
        """Build docker rm command that also removes anonymous volumes."""
        return f"/usr/bin/docker rm -fv {name}"

    @staticmethod
    def ps_filter(*name_patterns: str) -> str:
        """Build docker ps command with one or more filters."""
        filters = ' '.join(f'--filter "name={pattern}"' for pattern in name_patterns)
        return f'/usr/bin/docker ps -a {filters} --format "{{{{.Names}}}}"'

    @staticmethod
    def inspect_status(container_id: str) -> str:
        """Build docker inspect command for status."""
        return f"/usr/bin/docker ps -a --filter id={container_id} --format '{{{{.Status}}}}|||{{{{.State}}}}' 2>&1"

    @staticmethod
    def logs(container_id: str) -> str:
        """Build docker logs command."""
        return f"/usr/bin/docker logs {container_id} 2>&1 | head -20"

    @staticmethod
    def inspect_exit_code(container_id: str) -> str:
        """Build docker inspect for exit code."""
        return f"/usr/bin/docker inspect {container_id} --format '{{{{.State.ExitCode}}}}' 2>&1"

    @staticmethod
    def volume_remove(*volume_names: str) -> str:
        """Build docker volume rm command, tolerating per-volume failures."""
        names = " ".join(shlex.quote(name) for name in volume_names)
        return f"/usr/bin/docker volume rm {names} 2>/dev/null || true"

    @staticmethod
    def volume_ls_dangling() -> str:
        """Build docker volume ls command listing dangling volume names."""
        return "/usr/bin/docker volume ls -qf dangling=true"

    @staticmethod
    def inspect_created_timestamp(container_id: str) -> str:
        """Build docker inspect command to get creation timestamp in seconds."""
        return (
            f"/usr/bin/docker inspect {shlex.quote(container_id)} "
            "--format '{{json .Created}}' | "
            "xargs -I {} date -d {} +%s"
        )

    @staticmethod
    def ps_running(container_name: str) -> str:
        """Build docker ps command to check if container is running."""
        return f"/usr/bin/docker ps -q -f name={container_name}"

    @staticmethod
    def exec_command(container_name: str, command: str) -> str:
        """Build docker exec command."""
        return f"/usr/bin/docker exec -i {container_name} sh -c '{command}'"


@dataclass
class ContainerDeathDiagnostics:
    """Why a container died, plus host context.

    Shared post-mortem for create-time failures (docker_service) and run-time
    failures (rented_machine checks) so both land in Loki with one flat field
    shape and are queryable together (DAH-2395 / DAH-2193). exit_code / oom_killed
    are top-level, not nested under a state object, so a Loki query filters them
    directly without digging through JSON.
    """

    status: str | None = None
    exit_code: int | None = None
    oom_killed: bool | None = None
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    logs_tail: str | None = None
    host_context: dict[str, str] | None = None
    capture_error: str | None = None

    def to_log_fields(self) -> dict[str, object]:
        """Flatten into the shared Loki field shape used at both call sites."""
        return {
            "container_status": self.status,
            "container_exit_code": self.exit_code,
            "container_oom_killed": self.oom_killed,
            "container_error": self.error,
            "container_started_at": self.started_at,
            "container_finished_at": self.finished_at,
            "container_logs_tail": self.logs_tail,
            "container_host_context": self.host_context,
            "diagnostics_capture_error": self.capture_error,
        }


class ContainerDeathKind(str, Enum):
    """Why a dead filler container died — decides whether the provider is punishable.

    Only REMOVED and STOPPED are external kills (a container cannot rm or SIGTERM itself
    from outside its own process tree); everything else is the filler's or the host's own
    failure and belongs to self-heal (DAH-2419), never to incentive withholding.
    """

    REMOVED = "removed"  # container gone entirely — external `docker rm`
    STOPPED = "stopped"  # exited by SIGKILL/SIGTERM (137/143), not OOM — external stop
    HOST_REBOOT = "host_reboot"  # SIGTERM that coincided with a host reboot / executor restart
    SELF_CRASHED = "self_crashed"  # nonzero exit other than the kill signals
    OOM_KILLED = "oom_killed"  # kernel OOM kill (reports 137 + OOMKilled=true)
    NEVER_STARTED = "never_started"  # created but never ran, or start failed
    CLEAN_EXIT = "clean_exit"  # exit 0
    UNKNOWN = "unknown"  # diagnostics incomplete — fail open


_ZERO_DOCKER_TIMESTAMP_PREFIX = "0001-01-01"
_KILL_SIGNAL_EXIT_CODES = (137, 143)  # 128+SIGKILL, 128+SIGTERM
_REMOVED_MARKERS = ("no such object", "no such container")  # docker casing varies; match lowercased
# A reboot/compose-restart SIGTERMs every container at once, so the executor stack restarts around
# the same time. If it (re)started no earlier than this many seconds before the filler died, the
# SIGTERM was collateral of that restart, not a filler-targeted `docker stop`.
_HOST_RESTART_WINDOW_SECONDS = 600


def _never_started(diagnostics: ContainerDeathDiagnostics) -> bool:
    if diagnostics.status == "created":
        return True
    started = diagnostics.started_at or ""
    return started.startswith(_ZERO_DOCKER_TIMESTAMP_PREFIX) if started else False


def _stop_coincided_with_host_restart(diagnostics: ContainerDeathDiagnostics) -> bool:
    """True when the executor stack restarted around the filler's death (reboot/compose restart).

    Uses the executor container's start time (a UTC docker timestamp, same clock as FinishedAt) to
    avoid the timezone ambiguity of `uptime -s`. A filler-targeted `docker stop` leaves the executor
    stack untouched, so its start time stays hours/days before the death.
    """
    context = diagnostics.host_context or {}
    executor_started = _parse_docker_timestamp(context.get("executor_container_started_at"))
    finished = _parse_docker_timestamp(diagnostics.finished_at)
    if executor_started is None or finished is None:
        return False
    return (executor_started - finished).total_seconds() > -_HOST_RESTART_WINDOW_SECONDS


def classify_container_death(diagnostics: ContainerDeathDiagnostics) -> ContainerDeathKind:
    capture_error = (diagnostics.capture_error or "").lower()
    logs_tail = (diagnostics.logs_tail or "").lower()
    if any(marker in capture_error or marker in logs_tail for marker in _REMOVED_MARKERS):
        return ContainerDeathKind.REMOVED
    if diagnostics.oom_killed:
        return ContainerDeathKind.OOM_KILLED
    if _never_started(diagnostics):
        return ContainerDeathKind.NEVER_STARTED
    if diagnostics.exit_code in _KILL_SIGNAL_EXIT_CODES:
        if _stop_coincided_with_host_restart(diagnostics):
            return ContainerDeathKind.HOST_REBOOT
        return ContainerDeathKind.STOPPED
    if diagnostics.exit_code == 0 and diagnostics.status is not None:
        return ContainerDeathKind.CLEAN_EXIT
    if isinstance(diagnostics.exit_code, int) and diagnostics.exit_code != 0:
        return ContainerDeathKind.SELF_CRASHED
    return ContainerDeathKind.UNKNOWN


def _parse_docker_timestamp(value: str | None) -> datetime | None:
    if not value or value.startswith(_ZERO_DOCKER_TIMESTAMP_PREFIX):
        return None
    # Docker prints RFC3339 with nanoseconds; fromisoformat takes at most microseconds.
    trimmed = value.rstrip("Z")
    if "." in trimmed:
        seconds_part, fraction = trimmed.split(".", 1)
        trimmed = f"{seconds_part}.{fraction[:6]}"
    try:
        return datetime.fromisoformat(trimmed)
    except ValueError:
        return None


def container_uptime_seconds(started_at: str | None, finished_at: str | None) -> float | None:
    started = _parse_docker_timestamp(started_at)
    finished = _parse_docker_timestamp(finished_at)
    if started is None or finished is None:
        return None
    return (finished - started).total_seconds()


async def collect_container_death_diagnostics(
    ssh_client: asyncssh.SSHClientConnection,
    container_name: str,
) -> ContainerDeathDiagnostics:
    """Read why a container died over an already-open SSH session.

    Best effort: every failure folds into `capture_error` instead of raising
    (cancellation excepted). At both call sites this runs right before the
    container is force-removed, and the executor host belongs to the miner —
    losing the rest of the evidence to an exception is worse than a partial
    record.
    """
    diagnostics = ContainerDeathDiagnostics()
    capture_errors: list[str] = []
    quoted_name = shlex.quote(container_name)

    try:
        inspect_result = await ssh_client.run(
            f"/usr/bin/docker inspect --format '{{{{json .State}}}}' {quoted_name}"
        )
        raw_state = (inspect_result.stdout or "").strip()
        if raw_state:
            parsed_state = json.loads(raw_state)
            if isinstance(parsed_state, dict):
                diagnostics.status = parsed_state.get("Status")
                diagnostics.exit_code = parsed_state.get("ExitCode")
                diagnostics.oom_killed = parsed_state.get("OOMKilled")
                diagnostics.error = parsed_state.get("Error")
                diagnostics.started_at = parsed_state.get("StartedAt")
                diagnostics.finished_at = parsed_state.get("FinishedAt")
            else:
                capture_errors.append(f"inspect: non-object state JSON: {parsed_state!r}")
        else:
            stderr = (getattr(inspect_result, "stderr", "") or "").strip()
            capture_errors.append(f"inspect: {stderr or 'empty stdout'}")
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        capture_errors.append(f"inspect: {exc}")

    try:
        logs_result = await ssh_client.run(
            f"/usr/bin/docker logs --tail {_CONTAINER_DEATH_LOG_TAIL_LINES} {quoted_name} 2>&1 || true"
        )
        tail = (logs_result.stdout or "")[-_CONTAINER_DEATH_LOG_MAX_CHARS:]
        diagnostics.logs_tail = tail or None
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        capture_errors.append(f"logs: {exc}")

    host_context = await _collect_host_context(ssh_client)
    diagnostics.host_context = host_context or None

    if capture_errors:
        diagnostics.capture_error = "; ".join(capture_errors)

    return diagnostics


async def _collect_host_context(ssh_client: asyncssh.SSHClientConnection) -> dict[str, str]:
    """Executor-host context that separates a real container death from a host reboot
    or executor-container restart.

    Two independent signals:
      - `host_boot_time` from `uptime -s` — if it is close to the container's FinishedAt,
        the host likely rebooted and the container did not come back up on its own.
      - `executor_container_started_at` — if far newer than `host_boot_time`, the executor
        container itself restarted (watchtower, compose restart, autoheal) without a full
        host reboot. The executor container is located via the compose service label so the
        lookup survives renames.

    All failures are swallowed into `*_error` fields — this runs on the already-open SSH
    session, so we prefer empty signal over raising and losing the rest of the event.
    """
    host_context: dict[str, str] = {}

    try:
        uptime_result = await ssh_client.run("uptime -s")
        boot_time = (uptime_result.stdout or "").strip()
        if boot_time:
            host_context["host_boot_time"] = boot_time
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        host_context["host_boot_error"] = str(exc)

    try:
        name_cmd = (
            "/usr/bin/docker ps -a "
            "--filter label=com.docker.compose.service=executor "
            "--format '{{.Names}}' | head -n 1"
        )
        name_result = await ssh_client.run(name_cmd)
        executor_names = (name_result.stdout or "").strip().splitlines()[0:1]
        executor_name = executor_names[0] if executor_names else ""
        if executor_name:
            quoted_executor = shlex.quote(executor_name)
            start_cmd = (
                f"/usr/bin/docker inspect --format '{{{{.State.StartedAt}}}}' {quoted_executor}"
            )
            start_result = await ssh_client.run(start_cmd)
            started_at = (start_result.stdout or "").strip()
            if started_at:
                host_context["executor_container_name"] = executor_name
                host_context["executor_container_started_at"] = started_at
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        host_context["executor_container_error"] = str(exc)

    return host_context
