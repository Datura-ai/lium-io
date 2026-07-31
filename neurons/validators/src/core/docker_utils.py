import asyncio
import json
import shlex
from dataclasses import dataclass

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
        """Build docker exec command.

        `-u 0` because the exec would otherwise inherit the image's USER and lose
        access to the root-owned paths we write to. Numeric, so it does not need a
        root entry in the image's /etc/passwd (DAH-2534).
        """
        return f"/usr/bin/docker exec -u 0 -i {container_name} sh -c '{command}'"


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


ALPINE_HELPER_IMAGE = "docker.io/library/alpine:3.19"


async def df_available_bytes(ssh_client: asyncssh.SSHClientConnection, host_path: str) -> int:
    """Free bytes on the filesystem holding `host_path`, measured THROUGH the docker daemon.

    Three facts make this non-obvious enough to keep in one place. The validator's SSH session lands
    inside the miner's executor container, so a host path like the docker data root does not exist
    there and a plain df would measure the wrong filesystem — hence the bind-mounted helper container.
    Busybox df has no `--output`, so the request is POSIX `-P`, whose contract is one unwrapped line
    per filesystem. Available is then column 4 of the data line.

    Raises on anything unexpected; callers decide whether that is fatal.
    """
    result = await ssh_client.run(
        f"/usr/bin/docker run --rm -v {shlex.quote(host_path)}:/hostfs:ro "
        f"{ALPINE_HELPER_IMAGE} df -P -B1 /hostfs"
    )
    if getattr(result, "exit_status", 0) != 0:
        raise Exception(f"df via helper container failed: {getattr(result, 'stderr', '')}")
    lines = (result.stdout or "").strip().splitlines()
    if len(lines) < 2:
        raise Exception(f"Unexpected df output: {result.stdout!r}")
    columns = lines[1].split()
    if len(columns) < 4 or not columns[3].isdigit():
        raise Exception(f"Unexpected df output: {result.stdout!r}")
    return int(columns[3])


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
