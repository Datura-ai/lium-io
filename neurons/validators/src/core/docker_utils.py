import shlex

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
        return f"/usr/bin/docker rm -f {name}"

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
    def volume_prune() -> str:
        """Build docker volume prune command."""
        return "/usr/bin/docker volume prune -af"

    @staticmethod
    def volume_remove(volume_name: str) -> str:
        """Build docker volume rm command."""
        return f"/usr/bin/docker volume rm {volume_name} 2>/dev/null || true"

    @staticmethod
    def volume_ls_dangling() -> str:
        """Build docker volume ls command listing dangling volume names."""
        return "/usr/bin/docker volume ls -qf dangling=true"

    @staticmethod
    def volume_inspect_created(volume_names: list[str]) -> str:
        """Build docker volume inspect command printing name and creation time per line."""
        names = " ".join(shlex.quote(name) for name in volume_names)
        return (
            "/usr/bin/docker volume inspect "
            f"--format '{{{{.Name}}}} {{{{.CreatedAt}}}}' {names} 2>/dev/null"
        )

    @staticmethod
    def volume_remove_batch(volume_names: list[str]) -> str:
        """Build docker volume rm command for multiple volumes, tolerating per-volume failures."""
        names = " ".join(shlex.quote(name) for name in volume_names)
        return f"/usr/bin/docker volume rm {names} 2>/dev/null || true"

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
        """Build docker exec command."""
        return f"/usr/bin/docker exec -i {container_name} sh -c '{command}'"
