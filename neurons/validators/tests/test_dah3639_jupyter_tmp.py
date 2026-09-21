"""DAH-3639: with an encrypted local volume (or no local volume) the jupyter script
is installed by the validator. `docker cp` into the container fails here with
"Could not find the file /tmp in container" even after the directory exists (the
daemon resolves the destination against the container's rootfs; the container's own
exec namespace sees it), so the script is piped into the container through
`docker exec` and never through `docker cp`."""

import pytest

from tests.test_jupyter_command_quoting import _RecordingDockerService, _inner_script_of


@pytest.mark.asyncio
async def test_encrypted_branch_installs_the_script_without_docker_cp():
    service = _RecordingDockerService()

    await service.run_jupyter(
        ssh_client=None,
        container_name="pod_x",
        jupyter_token="tok",
        jupyter_port=8888,
        log_tag="t",
        log_extra={},
        local_volume=None,
        local_volume_path="/root",
        encrypted_local_volume=True,
    )

    assert all("docker cp" not in command for command in service.commands), service.commands

    install_commands = [c for c in service.commands if "run_jupyter.sh |" in c]
    assert len(install_commands) == 1, service.commands
    install = install_commands[0]
    assert " -i " in install
    assert "cat /root/app/run_jupyter.sh |" in install
    assert _inner_script_of(install) == "cat > /root/run_jupyter.sh && chmod +x /root/run_jupyter.sh"
    # nothing is fed from the executor process: the host shell pipes the file in
    install_index = service.commands.index(install)
    assert service.stdin_datas[install_index] is None

    run_steps = [c for c in service.commands if "--password=" in c]
    assert run_steps and "/root/run_jupyter.sh --password" in _inner_script_of(run_steps[0])
