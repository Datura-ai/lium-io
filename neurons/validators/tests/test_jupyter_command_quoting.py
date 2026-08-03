"""DAH-2534: local_volume_path comes from the renter's template and is spliced
into a `docker exec ... sh -c <script>` run on the miner host, so it has to stay
one literal token at BOTH layers — the host shell and the container's own sh."""

import shlex

import pytest

from services.docker_service import DockerService


# no quotes of its own: only real quoting keeps `id` from becoming a command
HOSTILE_PATH = "/root;id;x"


class _RecordingDockerService(DockerService):
    """Captures the command strings run_jupyter would send, runs nothing."""

    def __init__(self) -> None:
        self.commands: list[str] = []

    async def execute_and_stream_logs(self, *, command: str, **_) -> tuple[bool, str]:
        self.commands.append(command)
        return True, ""

    async def stream_log(self, *_, **__) -> None:
        return None


def _inner_script_of(command: str) -> str:
    """The argument the container's `sh -c` receives, as the host shell parses it."""
    tokens: list[str] = shlex.split(command)
    return tokens[tokens.index("-c") + 1]


@pytest.mark.asyncio
@pytest.mark.parametrize("encrypted", [False, True])
async def test_renter_volume_path_stays_one_token_at_both_layers(encrypted: bool):
    service = _RecordingDockerService()

    await service.run_jupyter(
        ssh_client=None,
        container_name="pod_x",
        jupyter_token="tok",
        jupyter_port=8888,
        log_tag="t",
        log_extra={},
        local_volume=None if encrypted else "vol",
        local_volume_path=HOSTILE_PATH,
        encrypted_local_volume=encrypted,
    )

    exec_commands: list[str] = [c for c in service.commands if "docker exec" in c]
    assert exec_commands, "run_jupyter issued no exec command to check"

    for command in exec_commands:
        inner_script: str = _inner_script_of(command)
        # punctuation_chars=True makes shlex surface ';' as its own token, the way
        # the container's sh would — without it an injected path looks like one word
        inner_tokens: list[str] = list(shlex.shlex(inner_script, punctuation_chars=True))
        assert ";" not in inner_tokens, f"path broke out into commands: {inner_script!r}"
        assert HOSTILE_PATH in inner_script, inner_script
