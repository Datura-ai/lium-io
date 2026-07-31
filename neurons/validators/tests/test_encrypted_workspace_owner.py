"""DAH-2534: the encrypted workspace is mounted by root, so it has to end up
writable by the image's own user — and that is decided by a real write probe,
not by assuming the chown was enough."""

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
async def test_non_root_image_is_chowned_then_probed():
    ssh_client = _FakeSshClient([(0, "prism\n"), (0, ""), (0, "")])

    assert await _grant(ssh_client) is None

    inspect_command, chown_command, probe_command = ssh_client.commands_called
    assert "docker inspect" in inspect_command
    assert "chown prism" in chown_command
    assert "-u prism" in probe_command and ".lium-write-probe" in probe_command


@pytest.mark.asyncio
async def test_root_image_needs_no_chown_or_probe():
    ssh_client = _FakeSshClient([(0, "\n")])

    assert await _grant(ssh_client) is None
    assert len(ssh_client.commands_called) == 1


@pytest.mark.asyncio
async def test_unreadable_image_user_fails_the_rental():
    # a probe we cannot run must not be mistaken for a probe that passed
    ssh_client = _FakeSshClient([(1, "")])

    error = await _grant(ssh_client)

    assert error is not None and "could not read the image USER" in error


@pytest.mark.asyncio
async def test_unwritable_workspace_fails_even_when_chown_succeeded():
    # chown-ing the mountpoint says nothing about traversing its parents
    ssh_client = _FakeSshClient([(0, "prism\n"), (0, ""), (1, "")])

    error = await _grant(ssh_client)

    assert error is not None and "not writable by the image user" in error


@pytest.mark.asyncio
async def test_failed_chown_still_passes_when_the_probe_succeeds():
    # the probe is the verdict; the chown is only remediation
    ssh_client = _FakeSshClient([(0, "prism\n"), (1, ""), (0, "")])

    assert await _grant(ssh_client) is None


@pytest.mark.asyncio
async def test_workspace_path_is_quoted_into_the_shell():
    ssh_client = _FakeSshClient([(0, "prism\n"), (0, ""), (0, "")])

    await _grant(ssh_client, plaintext_path="/root'$(id)'x")

    for command in ssh_client.commands_called[1:]:
        assert "$(id)" in command
        # quoted twice (inner sh -c, outer host shell), so the host never expands it
        assert "'\"'\"'" in command or "\\'" in command
