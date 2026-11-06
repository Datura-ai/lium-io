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


@pytest.mark.asyncio
async def test_volume_script_copy_finishes_before_the_temp_container_is_removed():
    # a detached (`-d`) copy can be force-removed by the cleanup before cp runs, so the chmod
    # then fails with "No such file or directory"
    service = _RecordingDockerService()

    await service.run_jupyter(
        ssh_client=None,
        container_name="pod_x",
        jupyter_token="tok",
        jupyter_port=8888,
        log_tag="t",
        log_extra={},
        local_volume="vol",
    )

    copy_index = next(i for i, c in enumerate(service.commands) if "/root/app/run_jupyter.sh /mnt/" in c)
    assert " -d " not in service.commands[copy_index]
    # a renter-placed FIFO at the target must be unlinked, not opened
    assert "cp --remove-destination " in service.commands[copy_index]
    assert "docker rm -fv" in service.commands[copy_index + 1]


class _FakeProcess:
    def __init__(self, exit_status: int | None) -> None:
        self.exit_status = exit_status
        self.exit_signal = None if exit_status is not None else ("KILL", False, "", "")
        self.stdin = _NullStdin()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_) -> None:
        return None

    @property
    def stdout(self):
        return _empty_lines()

    @property
    def stderr(self):
        return _empty_lines()

    async def wait_closed(self) -> None:
        return None


class _NullStdin:
    def write(self, _data: str) -> None:
        return None

    def write_eof(self) -> None:
        return None


async def _empty_lines():
    for line in ():
        yield line


class _FakeSSH:
    """Every command exits 0 and silently, except the script copy, which exits with `copy_status`."""

    def __init__(self, copy_status: int | None) -> None:
        self.copy_status = copy_status
        self.commands: list[str] = []

    def create_process(self, command: str) -> _FakeProcess:
        self.commands.append(command)
        is_copy = "/root/app/run_jupyter.sh /mnt/" in command
        return _FakeProcess(self.copy_status if is_copy else 0)


class _SilentDockerService(DockerService):
    def __init__(self) -> None:
        pass

    async def stream_log(self, *_, **__) -> None:
        return None


@pytest.mark.asyncio
@pytest.mark.parametrize("copy_status", [0, 137, None])
async def test_a_silently_failed_script_copy_stops_before_chmod_and_still_cleans_up(copy_status: int | None):
    # a cp killed after creating an empty target exits nonzero (or on a signal) with no stderr
    ssh = _FakeSSH(copy_status)
    service = _SilentDockerService()

    run = service.run_jupyter(
        ssh_client=ssh,
        container_name="pod_x",
        jupyter_token="tok",
        jupyter_port=8888,
        log_tag="t",
        log_extra={},
        local_volume="vol",
    )
    if copy_status == 0:
        await run
    else:
        with pytest.raises(Exception, match="script copy"):
            await run

    assert any("docker rm -fv temp_jupyter_copy_" in c for c in ssh.commands)
    reached_chmod = any("chmod +x" in c for c in ssh.commands)
    assert reached_chmod is (copy_status == 0)


@pytest.mark.asyncio
async def test_the_volume_script_copy_points_jupyter_at_the_template_path():
    # DAH-5426: the copied script serves the path the volume is mounted at, not /root
    service = _RecordingDockerService()

    await service.run_jupyter(
        ssh_client=None,
        container_name="pod_x",
        jupyter_token="tok",
        jupyter_port=8888,
        log_tag="t",
        log_extra={},
        local_volume="vol",
        local_volume_path="/workspace",
    )

    copy = next(c for c in service.commands if "/root/app/run_jupyter.sh /mnt/" in c)
    inner = shlex.split(copy)[-1]
    assert inner.startswith("cp --remove-destination /root/app/run_jupyter.sh /mnt/ && sed -i ")
    assert shlex.split(inner)[-2] == "s#--ServerApp.root_dir=/root #--ServerApp.root_dir=/workspace #"
