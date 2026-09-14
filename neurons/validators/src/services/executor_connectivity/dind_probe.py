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

# B-176: reason codes for a failed probe, carried to the backend in `specs.runtime_probe` so
# the provider sees why (the probe used to log at ERROR and vanish). The DinD container is
# started with `--gpus all`, so the NVIDIA container runtime hook runs first; when the host's
# driver and its user-space libraries disagree the hook fails before any command runs and the
# executor's own container (`deploy.resources.reservations.devices: nvidia`) cannot start
# either. That is the case an image update must wait for (ticket-0325).
RUNTIME_PROBE_NVIDIA_MISMATCH = "NVIDIA_RUNTIME_MISMATCH"
DIND_PROBE_FAILED = "DIND_PROBE_FAILED"
RUNTIME_PROBE_ERROR_MAX_CHARS = 500
_NVIDIA_MISMATCH_MARKERS = (
    "driver/library version mismatch",
    "nvidia-container-cli: initialization error",
    "nvml error",
)


def classify_runtime_probe_error(error: str | None) -> str:
    """Map the stderr of a failed `docker run --gpus all` to a reason code."""
    text = (error or "").lower()
    if any(marker in text for marker in _NVIDIA_MISMATCH_MARKERS):
        return RUNTIME_PROBE_NVIDIA_MISMATCH
    return DIND_PROBE_FAILED


def _bounded_error(error: str | None) -> str | None:
    if not error:
        return None
    return error[:RUNTIME_PROBE_ERROR_MAX_CHARS]


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

        try:
            logger.info(_m("DinD start", extra=get_extra_info(log_ctx)))

            private_key, public_key = self.ssh_service.generate_keypair()
            cmd = DockerCommand.run_dind(name, port.internal, public_key.strip(), sysbox)
            logger.debug("run: %s...", cmd[:100])

            result = await ssh_client.run(cmd)
            if result.exit_status != 0:
                error_msg = result.stderr.strip() if result.stderr and isinstance(result.stderr, str) else "unknown error"
                reason_code = classify_runtime_probe_error(error_msg)
                logger.error(
                    _m(
                        "DinD creation failed",
                        extra=get_extra_info({**log_ctx, "error": error_msg, "reason_code": reason_code}),
                    )
                )
                await ssh_client.run(DockerCommand.remove_with_volumes(name))
                return DindProbeResult(
                    success=False,
                    log_text=f"dind: check failed port={port.internal}",
                    sysbox_runtime=sysbox,
                    port=port,
                    reason_code=reason_code,
                    error=_bounded_error(error_msg),
                )

            logger.info(_m("DinD container created", extra=get_extra_info(log_ctx)))

            # Test SSH
            pkey = asyncssh.import_private_key(private_key)
            async with await self._connect_retrying_until_sshd_answers(
                host, port, pkey, log_ctx
            ) as ssh:
                # Test sysbox
                if sysbox:
                    # daturaai/dind:0.0.1 bundles the hello-world image into the inner dockerd
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
            await ssh_client.run(DockerCommand.remove_with_volumes(name))
            return DindProbeResult(
                success=False,
                log_text=f"dind: check failed port={port.internal}",
                sysbox_runtime=sysbox,
                port=port,
                reason_code=DIND_PROBE_FAILED,
                error=_bounded_error(str(e)),
            )

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
