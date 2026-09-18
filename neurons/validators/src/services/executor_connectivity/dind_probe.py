import asyncio
import logging
from typing import Any

import asyncssh
from asyncssh import SSHClientConnection, SSHKey

from core.docker_utils import DockerCommand
from core.utils import _m, get_extra_info
from services.executor_connectivity.models import DindProbeResult, PortPair
from services.ssh_service import SSHService

logger = logging.getLogger(__name__)

# DAH-2588: how long sshd inside the DinD container may take to accept a login. Measured in
# prod the container is ready in ~3s (p99 ~7s, max ~10s), so the fixed 5s wait this replaced
# sat right on the median and refused roughly one probe in 130 — and a refused probe reads as
# "no sysbox", which delists an unrented executor for the whole cycle. Polling is also cheaper
# than a longer sleep: a ready container is caught at ~3s instead of always waiting.
#
# The per-attempt timeout is deliberately not tightened to the poll interval: asyncssh counts TCP
# connect, key exchange AND authentication against it, none of which carry over to the next
# attempt, so a host slow enough to need 6s would fail every single try. An attempt is shortened
# only when less than 12s of the deadline is left, which is time it could not have used anyway.
DIND_SSH_READY_TIMEOUT_SECONDS = 30
DIND_SSH_CONNECT_TIMEOUT_SECONDS = 12
DIND_SSH_POLL_INTERVAL_SECONDS = 1.5

# DAH-2856: when the container started but sshd never answered, the validator used to know only
# "connection refused" and the node was scored as having no sysbox — with "install sysbox" as
# the advice. The image's entrypoint waits for the inner dockerd before sshd starts, so the
# container's own logs hold the real cause. Read before removal; the first matching pattern wins.
# (pattern in the log, cause code, plain words for the provider)
DIND_LOG_CAUSES: tuple[tuple[str, str, str], ...] = (
    (
        "can't initialize iptables table",
        "DIND_INNER_DOCKERD_IPTABLES",
        "the inner dockerd cannot use legacy iptables: the host runs iptables in nf_tables mode and the "
        "ip_tables/iptable_nat kernel modules are not loaded. Fix on the host: "
        "`sudo modprobe -a ip_tables iptable_nat iptable_filter` (nvidia_docker_sysbox_setup.sh --check "
        "shows it)",
    ),
    (
        "failed to start daemon",
        "DIND_INNER_DOCKERD_DOWN",
        "the inner dockerd did not start",
    ),
)
# the validator log gets this much of the container log; the provider-facing error gets the one matching
# line, head first (dockerd's real iptables line measured 333 chars on EC2, run 20260917T180704Z-2185)
DIND_LOG_EXCERPT_MAX_CHARS = 1500
DIND_LOG_LINE_MAX_CHARS = 400
DIND_SSHD_NOT_READY_WORDS = (
    f"sshd inside the DinD container did not answer within {DIND_SSH_READY_TIMEOUT_SECONDS}s and its "
    "log shows no dockerd error"
)


# DAH-3634: `docker run` itself refused. 534 of 694 SYSBOX_REQUIRED_MISSING zero cycles in 7 d
# (10 to 17 Sep, 21 executors) were the NVIDIA container hook refusing the probe's container —
# the node cannot start any GPU container, sysbox or not, and "install sysbox" was the advice.
# (pattern in docker's stderr, cause code, plain words for the provider); first match wins. Only
# NVML failures are classified: a hook `mount error` is sysbox's shiftfs fallback
# (nvidia_docker_sysbox_setup.sh --check) and a bound host port says nothing about the GPU stack,
# so both keep the plain SYSBOX_REQUIRED_MISSING. The fix text lives on the matching
# SysboxRequiredMessages template, keyed by the code. `NVIDIA_RUNTIME_MISMATCH` is the executor
# updater's name for the same host condition (DAH-3481), so the docs table has one row for it.
DOCKER_RUN_CAUSES: tuple[tuple[str, str, str], ...] = (
    (
        "nvml error: driver/library version mismatch",
        "NVIDIA_RUNTIME_MISMATCH",
        "the NVIDIA container hook cannot start a GPU container: the kernel driver and the user-space "
        "NVIDIA library are different versions (a driver update without a reboot)",
    ),
    (
        "nvml error:",
        "NVIDIA_CONTAINER_HOOK_FAILED",
        "the NVIDIA container hook cannot start a GPU container: NVML failed on the host",
    ),
)
DOCKER_RUN_ERROR_LINE_MAX_CHARS = 400
_HOOK_LINE_START = "nvidia-container-cli:"


def diagnose_docker_run_error(stderr: str | None) -> tuple[str, str] | None:
    """Return (cause code, plain words) when `docker run` of the DinD container was refused for
    an NVML reason the provider can act on; None for anything else (a bound host port, a name
    conflict, a sysbox mount error).

    The words quote the hook's own line (from `nvidia-container-cli:` on, capped), so the provider
    sees the hook's error in the event and not only the validator's reading of it.
    """
    text = stderr or ""
    for pattern, code, words in DOCKER_RUN_CAUSES:
        if pattern in text:
            line = next((ln for ln in text.splitlines() if pattern in ln), "")
            start = line.find(_HOOK_LINE_START)
            line = line[start if start >= 0 else 0 :].strip()[:DOCKER_RUN_ERROR_LINE_MAX_CHARS]
            return code, f"{words}. docker said: {line}" if line else words
    return None


def diagnose_dind_log(log_text: str | None) -> tuple[str, str]:
    """Return (cause code, plain words) for a DinD container whose sshd never answered.

    The words carry the matching log line (capped), so the provider sees dockerd's own error in
    the event and not only the validator's reading of it.
    """
    text = log_text or ""
    for pattern, code, words in DIND_LOG_CAUSES:
        if pattern in text:
            line = next((ln for ln in text.splitlines() if pattern in ln), "")
            line = line.strip()[:DIND_LOG_LINE_MAX_CHARS]
            return code, f"{words}. dockerd said: {line}" if line else words
    return "DIND_SSHD_NOT_READY", DIND_SSHD_NOT_READY_WORDS


class DindVerifier:
    """Verifies Docker-in-Docker capability."""

    def __init__(self, ssh_service: SSHService):
        self.ssh_service = ssh_service

    async def verify(
        self,
        port: PortPair,
        *,
        ssh_client: SSHClientConnection,
        host: str,
        container_name_prefix: str,
        sysbox: bool,
        log_ctx: dict | None = None,
    ) -> DindProbeResult:
        """Verify DinD on port."""
        name = f"{container_name_prefix}_{port.external}"
        log_ctx = {**(log_ctx or {}), "port": port.internal, "sysbox_requested": sysbox}
        created = False
        sshd_answered = False

        try:
            logger.info(_m("DinD start", extra=get_extra_info(log_ctx)))

            private_key, public_key = self.ssh_service.generate_keypair()
            cmd = DockerCommand.run_dind(name, port.internal, public_key.strip(), sysbox)
            logger.debug("run: %s...", cmd[:100])

            result = await ssh_client.run(cmd)
            if result.exit_status != 0:
                error_msg = result.stderr.strip() if result.stderr and isinstance(result.stderr, str) else "unknown error"
                # carried whether or not the probe asked for sysbox-runc: the executor's own sysbox
                # self-report (machine_scrape.check_sysbox_gpu_compatibility) runs the same hook on the
                # same host, so on 530 of 533 refused zero cycles sysbox_requested was already False
                cause = diagnose_docker_run_error(error_msg)
                logger.error(
                    _m(
                        "DinD creation failed",
                        extra=get_extra_info(
                            {**log_ctx, "error": error_msg, **({"cause": cause[0]} if cause else {})}
                        ),
                    )
                )
                await ssh_client.run(DockerCommand.remove_with_volumes(name))
                return DindProbeResult(
                    success=False,
                    log_text=f"dind: check failed port={port.internal}",
                    sysbox_runtime=sysbox,
                    port=port,
                    error=f"{cause[0]}: {cause[1]}" if cause else None,
                )

            logger.info(_m("DinD container created", extra=get_extra_info(log_ctx)))
            created = True

            # Test SSH
            pkey = asyncssh.import_private_key(private_key)
            async with await self._connect_retrying_until_sshd_answers(
                host, port, pkey, log_ctx
            ) as ssh:
                sshd_answered = True
                # Test sysbox
                if sysbox:
                    # daturaai/dind bundles the hello-world image into the inner dockerd
                    # at container start (DAH-1959), so `docker run` resolves it locally with no
                    # registry round-trip. If the bundled load failed for any reason the local
                    # image is absent and docker falls back to a Docker Hub pull, matching the
                    # previous behaviour.
                    try:
                        result = await asyncio.wait_for(
                            ssh.run("docker run --rm hello-world"), timeout=30
                        )
                        sysbox_ok = result.exit_status == 0
                        error_msg = (
                            result.stderr.strip()
                            if not sysbox_ok and result.stderr and isinstance(result.stderr, str)
                            else "unknown error"
                        )
                    except asyncio.TimeoutError:
                        sysbox_ok = False
                        error_msg = "sysbox check timed out after 30s"

                    if not sysbox_ok:
                        logger.warning(
                            _m("Sysbox check failed", extra=get_extra_info({**log_ctx, "error": error_msg}))
                        )
                        sysbox = False
                    else:
                        logger.info(_m("Sysbox check ok", extra=get_extra_info(log_ctx)))

            await ssh_client.run(DockerCommand.remove_with_volumes(name))
            logger.info(_m("DinD check ok", extra=get_extra_info({**log_ctx, "sysbox_result": sysbox})))

            return DindProbeResult(
                success=True,
                log_text=f"dind: check ok port={port.internal}",
                sysbox_runtime=sysbox,
                port=port,
            )

        except Exception as e:
            logger.error(
                _m("DinD check failed", extra=get_extra_info({**log_ctx, "error": str(e)})),
                exc_info=True,
            )
            error: str | None = None
            if created and not sshd_answered:
                error = await self._diagnose_unanswered_container(ssh_client, name, e, log_ctx)
            await ssh_client.run(DockerCommand.remove_with_volumes(name))
            return DindProbeResult(
                success=False,
                log_text=f"dind: check failed port={port.internal}",
                sysbox_runtime=sysbox,
                port=port,
                error=error,
            )

    async def _diagnose_unanswered_container(
        self,
        ssh_client: SSHClientConnection,
        name: str,
        ssh_error: Exception,
        log_ctx: dict[str, Any],
    ) -> str:
        """Read the container's logs while it still exists and name the cause (DAH-2856).

        Best-effort: a failed read still yields the generic cause, never an exception — the
        caller is on its way to remove the container and report the miss.
        """
        try:
            result = await asyncio.wait_for(
                ssh_client.run(DockerCommand.dind_diagnostics(name)), timeout=15
            )
            stdout = result.stdout if isinstance(result.stdout, str) else ""
        except Exception as read_error:
            # diagnosis must not mask the probe result
            stdout = f"(could not read the container log: {read_error})"
        code, words = diagnose_dind_log(stdout)
        if code == "DIND_SSHD_NOT_READY":
            words = f"{words} (ssh: {str(ssh_error)[:120]})"
        logger.warning(
            _m(
                "DinD container started but sshd never answered",
                extra=get_extra_info(
                    {
                        **log_ctx,
                        "cause": code,
                        "ssh_error": str(ssh_error),
                        "log_excerpt": stdout[-DIND_LOG_EXCERPT_MAX_CHARS:],
                    }
                ),
            )
        )
        return f"{code}: {words}"

    async def _connect_retrying_until_sshd_answers(
        self,
        host: str,
        port: PortPair,
        pkey: SSHKey,
        log_ctx: dict[str, Any],
    ) -> SSHClientConnection:
        """Connect to the freshly started DinD container, retrying until sshd answers.

        The deadline caps the whole wait, not just the moment an attempt starts: a hanging
        attempt gets whatever is left of the budget rather than a fresh 12s on top of it, so a
        blackholed port cannot stretch the probe past DIND_SSH_READY_TIMEOUT_SECONDS (DAH-2272
        exists because one such hang stalled the pipeline). No attempt starts once the deadline
        has passed either — a poll that resumes late would otherwise get a negative budget and
        report a timeout in place of the real connection error. The last failure is raised as is,
        so the caller still logs that error (a refused connection and an unreachable host are
        different diagnoses).
        """
        loop = asyncio.get_running_loop()
        started_at = loop.time()
        deadline = started_at + DIND_SSH_READY_TIMEOUT_SECONDS
        attempts = 0
        last_error: Exception | None = None

        while loop.time() < deadline:
            attempts += 1
            attempt_timeout_seconds = min(
                DIND_SSH_CONNECT_TIMEOUT_SECONDS, deadline - loop.time()
            )
            try:
                connection = await asyncssh.connect(
                    host=host,
                    port=port.external,
                    username="root",
                    client_keys=[pkey],
                    known_hosts=None,
                    connect_timeout=attempt_timeout_seconds,
                    login_timeout=attempt_timeout_seconds,
                )
            except Exception as error:
                last_error = error
                if loop.time() + DIND_SSH_POLL_INTERVAL_SECONDS >= deadline:
                    break
                await asyncio.sleep(DIND_SSH_POLL_INTERVAL_SECONDS)
                continue

            logger.info(
                _m(
                    "DinD SSH connected",
                    extra=get_extra_info(
                        {
                            **log_ctx,
                            "ssh_ready_sec": round(loop.time() - started_at, 2),
                            "ssh_attempts": attempts,
                        }
                    ),
                )
            )
            return connection

        logger.warning(
            _m(
                "DinD SSH not ready before deadline",
                extra=get_extra_info(
                    {
                        **log_ctx,
                        "ssh_waited_sec": round(loop.time() - started_at, 2),
                        "ssh_attempts": attempts,
                    }
                ),
            )
        )
        raise last_error


class DindProbe:
    """Runs DinD verification via published SSH port."""

    def __init__(self, verifier: DindVerifier):
        self.verifier = verifier

    async def verify(
        self,
        port: PortPair,
        *,
        ssh_client: SSHClientConnection,
        host: str,
        container_name_prefix: str,
        sysbox_runtime: bool,
        log_ctx: dict | None = None,
    ) -> DindProbeResult:
        return await self.verifier.verify(
            port,
            ssh_client=ssh_client,
            host=host,
            container_name_prefix=container_name_prefix,
            sysbox=sysbox_runtime,
            log_ctx=log_ctx,
        )
