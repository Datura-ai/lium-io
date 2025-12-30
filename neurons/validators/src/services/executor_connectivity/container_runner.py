import asyncio

from asyncssh import SSHClientConnection

from services.executor_connectivity.docker_command import DockerCommand
from services.executor_connectivity.models import ContainerStartResult


class ContainerRunner:
    """Runs netcat test containers and reports diagnostics."""

    async def run(
        self,
        ssh_client: SSHClientConnection,
        name: str,
        script: str,
        network_mode: str,
        timeout: int,
    ) -> ContainerStartResult:
        """Start Alpine container with netcat script and return diagnostics."""
        cmd = DockerCommand.run_alpine(name, script, network_mode, timeout)

        result = await ssh_client.run(cmd)
        if result.exit_status != 0:
            error_msg = result.stderr.strip() if result.stderr and isinstance(result.stderr, str) else "unknown error"
            return ContainerStartResult(
                ok=False,
                container_id=None,
                status=error_msg,
                logs=None,
            )

        if not result.stdout or not isinstance(result.stdout, str):
            return ContainerStartResult(
                ok=False,
                container_id=None,
                status="container start returned no output",
                logs=None,
            )

        container_id = result.stdout.strip()
        await asyncio.sleep(1.0)

        # Verify running
        result = await ssh_client.run(DockerCommand.inspect_status(container_id))
        if result.stdout and isinstance(result.stdout, str) and "Up" in result.stdout:
            return ContainerStartResult(
                ok=True,
                container_id=container_id,
                status=result.stdout.strip(),
                logs=None,
            )

        # Failed - get diagnostics
        status = result.stdout.strip() if result.stdout and isinstance(result.stdout, str) else "no status"
        logs_result = await ssh_client.run(DockerCommand.logs(container_id))
        logs = logs_result.stdout.strip()[:500] if logs_result.stdout and isinstance(logs_result.stdout, str) else "no logs"
        return ContainerStartResult(
            ok=False,
            container_id=container_id,
            status=status,
            logs=logs,
        )

    async def cleanup(self, ssh_client: SSHClientConnection, name: str):
        """Remove container."""
        await ssh_client.run(DockerCommand.remove(name))
