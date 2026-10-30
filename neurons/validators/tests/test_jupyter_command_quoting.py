"""DAH-2534: local_volume_path comes from the renter's template and is spliced
into a `docker exec ... sh -c <script>` run on the miner host, so it has to stay
one literal token at BOTH layers — the host shell and the container's own sh."""

import pytest

from services.docker_service import DockerService


# no quotes of its own: only real quoting keeps `id` from becoming a command
HOSTILE_PATH = "/root;id;x"


class _RecordingDockerService(DockerService):
    """Captures the command strings run_jupyter would send, runs nothing."""

    def __init__(self) -> None:
        self.commands: list[str] = []
        self.stdin_datas: list[str | None] = []

    async def execute_and_stream_logs(self, *, command: str, stdin_data: str | None = None, **_) -> tuple[bool, str]:
        self.commands.append(command)
        self.stdin_datas.append(stdin_data)
        return True, ""

    async def stream_log(self, *_, **__) -> None:
        return None


@pytest.mark.asyncio
@pytest.mark.parametrize("encrypted", [False, True])
async def test_a_hostile_renter_volume_path_is_refused_before_any_command_runs(encrypted: bool):
    # DAH-5426: the path also lands in a sed program on the host, so it must be a plain path, not just quoted
    service = _RecordingDockerService()

    with pytest.raises(ValueError, match="not a plain absolute path"):
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

    assert service.commands == []
