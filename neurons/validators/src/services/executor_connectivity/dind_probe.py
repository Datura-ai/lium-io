import asyncio
import logging
import random

import asyncssh
from asyncssh import SSHClientConnection

from core.docker_utils import DockerCommand
from core.utils import _m, get_extra_info
from services.const import DIND_PROBE_IMAGE
from services.executor_connectivity.models import DindProbeResult, PortPair
from services.ssh_service import SSHService

logger = logging.getLogger(__name__)

# public.ecr.aws enforces 1 unauthenticated image pull / sec / IP. Miners with several
# executors behind one NAT see concurrent waves of probes from a single validator, plus
# concurrent waves from multiple validators on the subnet, which still trips the limit
# even after we left docker.io. Retrying once with a randomised 2–10 s wait lets the
# bursts de-phase across executors of the same IP and brings residual 429s back down to
# real-failure noise. We only retry on the throttling signature so genuine "daemon dead"
# failures continue to surface immediately.
_RATELIMIT_MARKERS = ("toomanyrequests", "rate exceeded")
_PROBE_RETRY_MIN_S = 2
_PROBE_RETRY_MAX_S = 10


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
                logger.error(_m("DinD creation failed", extra=get_extra_info({**log_ctx, "error": error_msg})))
                await ssh_client.run(DockerCommand.remove(name))
                return DindProbeResult(
                    success=False,
                    log_text=f"dind: check failed port={port.internal}",
                    sysbox_runtime=sysbox,
                    port=port,
                )

            logger.info(_m("DinD container created", extra=get_extra_info(log_ctx)))
            await asyncio.sleep(5)

            # Test SSH
            pkey = asyncssh.import_private_key(private_key)
            async with asyncssh.connect(
                host=host, port=port.external, username="root", client_keys=[pkey], known_hosts=None
            ) as ssh:
                logger.info(_m("DinD SSH connected", extra=get_extra_info(log_ctx)))

                # Probe that the inner dockerd can actually pull+run a tiny image. We use a
                # sha256-pinned mirror on public.ecr.aws (same bytes as docker.io/library/hello-world)
                # to avoid Docker Hub anonymous-pull rate limits, which were the original failure mode.
                # This is also the only end-to-end signal that sysbox-runc is wired up correctly:
                # without sysbox, nested dockerd cannot start the container, so a successful run
                # implies the runtime works. We do NOT trust this probe as proof of sysbox by itself —
                # an attacker could rig the inner daemon to lie — it is a functional health check.
                if sysbox:
                    probe_cmd = f"docker run --rm {DIND_PROBE_IMAGE}"
                    result = await ssh.run(probe_cmd)
                    probe_ok = result.exit_status == 0
                    error_msg = (
                        result.stderr.strip()
                        if result.stderr and isinstance(result.stderr, str)
                        else "unknown error"
                    )

                    # Retry once on registry rate-limit. ECR public's 1 pull/sec/IP limit gets
                    # exhausted whenever a miner runs many executors behind one NAT and the
                    # whole subnet's validators dispatch their probe waves at the same block.
                    # A single retry with random 2-10 s wait de-phases the bursts so each
                    # executor of one IP lands in a different one-second window.
                    if not probe_ok and any(m in error_msg.lower() for m in _RATELIMIT_MARKERS):
                        backoff = random.uniform(_PROBE_RETRY_MIN_S, _PROBE_RETRY_MAX_S)
                        logger.info(
                            _m(
                                "DinD run probe ratelimited, retrying",
                                extra=get_extra_info({**log_ctx, "backoff_s": round(backoff, 2)}),
                            )
                        )
                        await asyncio.sleep(backoff)
                        result = await ssh.run(probe_cmd)
                        probe_ok = result.exit_status == 0
                        if not probe_ok:
                            error_msg = (
                                result.stderr.strip()
                                if result.stderr and isinstance(result.stderr, str)
                                else "unknown error"
                            )

                    if not probe_ok:
                        logger.warning(
                            _m("DinD run probe failed", extra=get_extra_info({**log_ctx, "error": error_msg}))
                        )
                        sysbox = False
                    else:
                        logger.info(_m("DinD run probe ok", extra=get_extra_info(log_ctx)))

            await ssh_client.run(DockerCommand.remove(name))
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
            await ssh_client.run(DockerCommand.remove(name))
            return DindProbeResult(
                success=False,
                log_text=f"dind: check failed port={port.internal}",
                sysbox_runtime=sysbox,
                port=port,
            )


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
