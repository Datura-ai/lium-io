import asyncio
import logging

import asyncssh
from asyncssh import SSHClientConnection

from core.docker_utils import DockerCommand
from core.utils import _m, get_extra_info
from services.executor_connectivity.models import DindProbeResult, PortPair
from services.ssh_service import SSHService

logger = logging.getLogger(__name__)


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

                # Test sysbox
                if sysbox:
                    result = await ssh.run("docker pull hello-world")
                    sysbox_ok = result.exit_status == 0
                    if not sysbox_ok:
                        error_msg = result.stderr.strip() if result.stderr and isinstance(result.stderr, str) else "unknown error"
                        logger.warning(
                            _m("Sysbox check failed", extra=get_extra_info({**log_ctx, "error": error_msg}))
                        )
                        sysbox = False
                    else:
                        logger.info(_m("Sysbox check ok", extra=get_extra_info(log_ctx)))

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
