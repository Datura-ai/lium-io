import asyncio
import logging
from collections.abc import Awaitable, Callable
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
# The liveness read on a prestarted container (`docker inspect` over the host SSH) after a failed
# connect: bounded so it cannot stretch the probe past the ready timeout by itself.
DIND_LIVENESS_READ_TIMEOUT_SECONDS = 5


class DindContainerGone(Exception):
    """liumd phase 2c/3: the prestarted container is no longer running on the host — a rental's
    `wait_for_port_check_containers` force-removed it, or it exited — so no amount of waiting will
    get sshd to answer. Raised instead of running the ready timeout down."""


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
        prestarted=None,
    ) -> DindProbeResult:
        """Verify DinD on port.

        `prestarted` (liumd phase 2c, `local_verify_facts.PreparedDind`): the executor already ran
        this very `docker run` from the validator's signed intent — same name, same port, the key
        pair the validator minted. Only the `docker run` round trip and the container's boot are
        skipped; the connection with the validator's private key, the sysbox proof and the removal
        are this method's own, so the executor's word ("started") decides nothing on its own: a
        container that does not answer the key is a failed probe, exactly as today.
        """
        name = f"{container_name_prefix}_{port.external}"
        log_ctx = {**(log_ctx or {}), "port": port.internal, "sysbox_requested": sysbox}
        if prestarted is not None:
            if prestarted.name != name or prestarted.port != port:
                # Not the container this probe is about: start our own and leave it UNconsumed,
                # so the pipeline's settle step still removes it.
                prestarted = None
            else:
                prestarted.consumed = True  # ours from here: removed on every path below

        try:
            logger.info(_m("DinD start", extra=get_extra_info({**log_ctx, "prestarted": prestarted is not None})))

            still_running = None
            if prestarted is not None:
                private_key = prestarted.private_key

                async def still_running() -> bool:
                    # Only read once a connection attempt has failed: a container that a rental
                    # removed the moment it landed (DAH-2272) would otherwise cost the whole ready
                    # timeout (30 s) before today's probe runs on a proven port. One `inspect`
                    # (≈ 0.3 s) per failed attempt, never on the path of a container that answers.
                    result = await asyncio.wait_for(
                        ssh_client.run(DockerCommand.inspect_running(name)),
                        timeout=DIND_LIVENESS_READ_TIMEOUT_SECONDS,
                    )
                    return result.exit_status == 0 and (result.stdout or "").strip() == "true"
            else:
                private_key, public_key = self.ssh_service.generate_keypair()
                cmd = DockerCommand.run_dind(name, port.internal, public_key.strip(), sysbox)
                logger.debug("run: %s...", cmd[:100])

                result = await ssh_client.run(cmd)
                if result.exit_status != 0:
                    error_msg = result.stderr.strip() if result.stderr and isinstance(result.stderr, str) else "unknown error"
                    logger.error(_m("DinD creation failed", extra=get_extra_info({**log_ctx, "error": error_msg})))
                    await ssh_client.run(DockerCommand.remove_with_volumes(name))
                    return DindProbeResult(
                        success=False,
                        log_text=f"dind: check failed port={port.internal}",
                        sysbox_runtime=sysbox,
                        port=port,
                    )

                logger.info(_m("DinD container created", extra=get_extra_info(log_ctx)))

            # Test SSH
            pkey = asyncssh.import_private_key(private_key)
            async with await self._connect_retrying_until_sshd_answers(
                host, port, pkey, log_ctx, still_running=still_running
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

        except DindContainerGone as e:
            # Not the host's failure: the container the validator was coming for was removed
            # under it (a rental landing in the window). The orchestrator runs today's probe next.
            logger.warning(_m("DinD prestart gone", extra=get_extra_info({**log_ctx, "error": str(e)})))
            await ssh_client.run(DockerCommand.remove_with_volumes(name))
            return DindProbeResult(
                success=False,
                log_text=f"dind: prestart gone port={port.internal}",
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
            )

    async def _connect_retrying_until_sshd_answers(
        self,
        host: str,
        port: PortPair,
        pkey: SSHKey,
        log_ctx: dict[str, Any],
        still_running: Callable[[], Awaitable[bool]] | None = None,
    ) -> SSHClientConnection:
        """Connect to the freshly started DinD container, retrying until sshd answers.

        `still_running` (an async callable, prestarted containers only): asked after a failed
        attempt whether the container is still there; `False` raises `DindContainerGone` at once
        instead of polling a removed container until the deadline. An error in the question itself
        is not an answer — the poll goes on as before.

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
                if still_running is not None and await self._is_gone(still_running, log_ctx):
                    raise DindContainerGone(
                        f"prestarted DinD container is gone after {attempts} attempt(s): {error}"
                    ) from error
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

    @staticmethod
    async def _is_gone(still_running: Callable[[], Awaitable[bool]], log_ctx: dict[str, Any]) -> bool:
        try:
            return not await still_running()
        except Exception as error:  # noqa: BLE001 — an unreadable host is not "gone"; keep polling
            logger.info(_m("DinD liveness read failed; polling on", extra=get_extra_info({**log_ctx, "error": str(error)})))
            return False


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
        prestarted=None,
    ) -> DindProbeResult:
        return await self.verifier.verify(
            port,
            ssh_client=ssh_client,
            host=host,
            container_name_prefix=container_name_prefix,
            sysbox=sysbox_runtime,
            log_ctx=log_ctx,
            prestarted=prestarted,
        )
