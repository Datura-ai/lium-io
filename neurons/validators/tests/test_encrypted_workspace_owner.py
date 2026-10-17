"""DAH-2534: the encrypted workspace is mounted by root, so it has to end up
writable by the image's own user — and that is decided by a real write probe,
not by assuming the chown was enough."""

from types import SimpleNamespace

import pytest

from services.docker_service import DockerService


class _FakeSshClient:
    """Records commands and replays a scripted result per call."""

    def __init__(self, results: list[tuple[int, str]]) -> None:
        self._results: list[tuple[int, str]] = results
        self.commands_called: list[str] = []

    async def run(self, command: str, check: bool = True):
        self.commands_called.append(command)
        exit_status, stdout = self._results.pop(0)

        class _Result:
            pass

        result = _Result()
        result.exit_status = exit_status
        result.stdout = stdout
        result.stderr = ""
        return result


async def _grant(ssh_client, inspect: tuple[int, str], plaintext_path: str = "/workspace") -> str | None:
    # the host's `docker inspect` of the image USER, run by the caller beside the setup exec
    inspect_exit_status, inspect_stdout = inspect
    return await DockerService._grant_workspace_to_container_user(
        DockerService.__new__(DockerService),
        ssh_client=ssh_client,
        container_q="pod_x",
        plaintext_path=plaintext_path,
        user_inspect_result=SimpleNamespace(
            exit_status=inspect_exit_status, stdout=inspect_stdout, stderr=""
        ),
        log_extra={},
    )


@pytest.mark.asyncio
async def test_probe_name_is_unique_so_it_cannot_delete_renter_data():
    # the workspace may already hold renter files on a remount
    probes: list[str] = []
    for _ in range(2):
        ssh_client = _FakeSshClient([(0, ""), (0, "")])
        await _grant(ssh_client, (0, "prism\n"))
        probes.append(ssh_client.commands_called[1])

    assert probes[0] != probes[1]


@pytest.mark.asyncio
async def test_workspace_path_is_quoted_into_the_shell():
    ssh_client = _FakeSshClient([(0, ""), (0, "")])

    await _grant(ssh_client, (0, "prism\n"), plaintext_path="/root'$(id)'x")

    for command in ssh_client.commands_called:
        assert "$(id)" in command
        # quoted twice (inner sh -c, outer host shell), so the host never expands it
        assert "'\"'\"'" in command or "\\'" in command
