import logging

from asyncssh import SSHClientConnection

from services.executor_connectivity.docker_command import DockerCommand

logger = logging.getLogger(__name__)


class ContainerCleanupService:
    """Removes old containers not in active pod list."""

    async def cleanup(
        self,
        ssh: SSHClientConnection,
        active_pod_names: list[str],
        container_name_prefix: str,
    ) -> None:
        try:
            result = await ssh.run(DockerCommand.ps_filter(f"^{container_name_prefix}"))
            if not result.stdout or not isinstance(result.stdout, str) or not result.stdout.strip():
                return

            names = [
                n
                for n in result.stdout.strip().split("\n")
                if n and n not in active_pod_names
            ]
            if names:
                logger.info(f"cleanup: found {len(names)} containers, {names=}")
                await ssh.run(f"/usr/bin/docker rm {' '.join(names)} -f")
                await ssh.run(DockerCommand.volume_prune())
                logger.info(f"cleanup: removed {len(names)} containers, {names=}")
        except Exception as e:
            logger.error("cleanup failed: %s", e, exc_info=True)
