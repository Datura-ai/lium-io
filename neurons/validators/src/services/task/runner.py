import asyncio
import time
import uuid
from datetime import UTC, datetime
from typing import Optional

import asyncssh
from pydantic import BaseModel

# error_type when ssh.run returned but the host sent neither an exit status nor an exit signal
NO_EXIT_STATUS = "no_exit_status"


class SSHCommandResult(BaseModel):
    command: str
    command_id: str
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    started_at: datetime
    finished_at: datetime
    success: bool
    error_type: Optional[str] = None
    error_message: Optional[str] = None


class SSHCommandRunner:
    """Safe, observable SSH command runner with metrics and retries."""

    def __init__(self, ssh_client: asyncssh.SSHClientConnection, *, max_retries: int = 1):
        self.ssh = ssh_client
        self.max_retries = max_retries

    async def run(
        self,
        cmd: str,
        *,
        timeout: int = 60,
        check: bool = False,
        retryable: bool = True,
        stdin_text: str | None = None,
    ) -> SSHCommandResult:
        """Run a command and capture stdout/stderr safely."""
        attempt = 0
        last_exc: Exception | None = None

        while attempt <= self.max_retries:
            attempt += 1
            cid = str(uuid.uuid4())
            start = datetime.now(UTC)
            t0 = time.perf_counter()
            try:
                result = await asyncio.wait_for(self.ssh.run(cmd, input=stdin_text), timeout=timeout)
                dt = int((time.perf_counter() - t0) * 1000)
                stdout = str(result.stdout) or ""
                stderr = str(result.stderr) or ""
                if result.exit_status is None:
                    # asyncssh does not raise when the connection drops or the channel closes after
                    # it opened: it returns what arrived with neither an exit status nor a signal
                    # (a signal reads -1). Either end can cause that, and it is not a success.
                    if not retryable or attempt > self.max_retries:
                        return SSHCommandResult(
                            command=cmd,
                            command_id=cid,
                            exit_code=-1,
                            stdout=stdout.strip(),
                            stderr=stderr.strip(),
                            duration_ms=dt,
                            started_at=start,
                            finished_at=datetime.now(UTC),
                            success=False,
                            error_type=NO_EXIT_STATUS,
                            error_message="the SSH channel closed without an exit status",
                        )
                    await asyncio.sleep(0.5 * attempt)
                    continue

                exit_code = result.exit_status
                ok = exit_code == 0

                if check and not ok:
                    raise RuntimeError(f"Command failed with exit code {exit_code}: {cmd}")

                return SSHCommandResult(
                    command=cmd,
                    command_id=cid,
                    exit_code=exit_code,
                    stdout=stdout.strip(),
                    stderr=stderr.strip(),
                    duration_ms=dt,
                    started_at=start,
                    finished_at=datetime.now(UTC),
                    success=ok,
                )

            except asyncio.TimeoutError as exc:
                last_exc = exc
                if not retryable or attempt > self.max_retries:
                    return SSHCommandResult(
                        command=cmd,
                        command_id=cid,
                        exit_code=-1,
                        stdout="",
                        stderr="",
                        duration_ms=int((time.perf_counter() - t0) * 1000),
                        started_at=start,
                        finished_at=datetime.now(UTC),
                        success=False,
                        error_type="timeout",
                        error_message=str(exc),
                    )
                await asyncio.sleep(1.0 * attempt)

            except asyncssh.Error as exc:
                last_exc = exc
                if not retryable or attempt > self.max_retries:
                    return SSHCommandResult(
                        command=cmd,
                        command_id=cid,
                        exit_code=-1,
                        stdout="",
                        stderr="",
                        duration_ms=int((time.perf_counter() - t0) * 1000),
                        started_at=start,
                        finished_at=datetime.now(UTC),
                        success=False,
                        error_type=exc.__class__.__name__,
                        error_message=str(exc),
                    )
                await asyncio.sleep(0.5 * attempt)

            except Exception as exc:
                return SSHCommandResult(
                    command=cmd,
                    command_id=cid,
                    exit_code=-1,
                    stdout="",
                    stderr="",
                    duration_ms=int((time.perf_counter() - t0) * 1000),
                    started_at=start,
                    finished_at=datetime.now(UTC),
                    success=False,
                    error_type=exc.__class__.__name__,
                    error_message=str(exc),
                )

        raise last_exc or RuntimeError(f"Unknown SSH error for cmd: {cmd}")
