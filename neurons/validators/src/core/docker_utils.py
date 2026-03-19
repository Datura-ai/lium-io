from services.const import DOCKER_DIND_IMAGE


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
    def remove(name: str) -> str:
        """Build docker rm command."""
        return f"/usr/bin/docker rm -f {name} 2>/dev/null || true"

    @staticmethod
    def ps_filter(name_pattern: str) -> str:
        """Build docker ps command with filter."""
        return f'/usr/bin/docker ps -a --filter "name={name_pattern}" --format "{{{{.Names}}}}"'

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
    def volume_prune() -> str:
        """Build docker volume prune command."""
        return "/usr/bin/docker volume prune -af"