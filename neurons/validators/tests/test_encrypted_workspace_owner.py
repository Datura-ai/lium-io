"""DAH-2534: the encrypted workspace is mounted by root, so it has to be handed
back to the image's own user or a non-root renter cannot write to it."""

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


async def _grant(ssh_client, plaintext_path: str = "/workspace") -> str | None:
    return await DockerService._grant_workspace_to_container_user(
        DockerService.__new__(DockerService),
        ssh_client=ssh_client,
        container_q="pod_x",
        plaintext_path=plaintext_path,
        log_extra={},
    )


@pytest.mark.asyncio
async def test_workspace_is_chowned_to_a_non_root_image_user():
    ssh_client = _FakeSshClient([(0, "101\n101\n"), (0, "")])

    await _grant(ssh_client)

    assert "chown 101:101" in ssh_client.commands_called[1]
    # the probe must NOT force a user — it reads the image's own default
    assert "-u 0" not in ssh_client.commands_called[0]


@pytest.mark.asyncio
async def test_root_image_needs_no_chown():
    ssh_client = _FakeSshClient([(0, "0\n0\n")])

    await _grant(ssh_client)

    assert len(ssh_client.commands_called) == 1


@pytest.mark.asyncio
async def test_failed_chown_is_reported_so_the_rental_fails():
    # a known-non-root user whose chown failed means the workspace is unusable —
    # billing a pod for that is worse than failing the deploy
    ssh_client = _FakeSshClient([(0, "101\n101\n"), (1, "")])

    error = await _grant(ssh_client)

    assert error is not None and "cannot write" in error


@pytest.mark.asyncio
async def test_unresolvable_user_is_only_a_warning():
    # without a uid we cannot tell whether a chown was needed at all
    ssh_client = _FakeSshClient([(127, "")])

    assert await _grant(ssh_client) is None


@pytest.mark.asyncio
async def test_workspace_path_is_quoted_into_the_shell():
    ssh_client = _FakeSshClient([(0, "101\n101\n"), (0, "")])

    await _grant(ssh_client, plaintext_path="/root'$(id)'x")

    chown_command: str = ssh_client.commands_called[1]
    assert "$(id)" in chown_command
    # quoted twice (inner sh -c, outer host shell), so the host never expands it
    assert "'\"'\"'" in chown_command or "\\'" in chown_command
