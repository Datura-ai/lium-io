"""DAH-2534: setup_encrypted_local_volume must actually consult the write probe
and fail the rental when it says the workspace is unusable. Every other test in
the suite mocks this method out, so nothing else covers the wiring."""

import pytest

from services.docker_service import DockerService


class _ScriptedSshClient:
    """Answers each `run` from a scripted list, matched by call order."""

    def __init__(self, results: list[tuple[int, str]]) -> None:
        self._results: list[tuple[int, str]] = results
        self.commands_called: list[str] = []

    async def run(self, command: str, check: bool = True):
        self.commands_called.append(command)
        exit_status, stdout = self._results.pop(0) if self._results else (0, "")

        class _Result:
            pass

        result = _Result()
        result.exit_status = exit_status
        result.stdout = stdout
        result.stderr = ""
        return result


def _service() -> DockerService:
    service = DockerService.__new__(DockerService)

    async def _noop(*_, **__) -> None:
        return None

    service.stream_log = _noop
    return service


async def _setup(ssh_client) -> None:
    await DockerService.setup_encrypted_local_volume(
        _service(),
        ssh_client=ssh_client,
        container_name="pod_x",
        plaintext_path="/workspace",
        volume_name="vol",
        pod_id="11111111-1111-1111-1111-111111111111",
        log_tag="t",
        log_extra={},
    )


# upload, run setup, wipe tmp, verify mount, inspect USER, chown, probe
_UP_TO_PROBE: list[tuple[int, str]] = [(0, ""), (0, ""), (0, ""), (0, ""), (0, "prism\n"), (0, "")]


@pytest.mark.asyncio
async def test_failed_write_probe_fails_the_whole_setup():
    ssh_client = _ScriptedSshClient([*_UP_TO_PROBE, (1, "")])

    with pytest.raises(RuntimeError, match="not writable by the image user"):
        await _setup(ssh_client)


@pytest.mark.asyncio
async def test_successful_probe_lets_the_setup_finish():
    ssh_client = _ScriptedSshClient([*_UP_TO_PROBE, (0, "")])

    await _setup(ssh_client)

    probe_commands = [c for c in ssh_client.commands_called if ".lium-write-probe-" in c]
    assert probe_commands, f"the probe never ran: {ssh_client.commands_called}"


@pytest.mark.asyncio
async def test_root_image_skips_chown_and_probe_entirely():
    # inspect returns an empty USER -> nothing to hand over
    ssh_client = _ScriptedSshClient([(0, ""), (0, ""), (0, ""), (0, ""), (0, "\n")])

    await _setup(ssh_client)

    assert not [c for c in ssh_client.commands_called if ".lium-write-probe-" in c]
