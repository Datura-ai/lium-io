"""DAH-2534: the legacy shell exec path must also pin the user to root."""

from neurons.validators.src.core.docker_utils import DockerCommand


def test_exec_command_forces_root_user() -> None:
    command: str = DockerCommand.exec_command("pod_abc", "cat /root/.ssh/authorized_keys")

    assert "-u root" in command
    assert command.startswith("/usr/bin/docker exec -u root -i pod_abc ")
