"""An install-jupyter request on a running pod carries no encryption flag. On a pod whose volume
is gocryptfs-encrypted the script must be piped into the container, not copied into the
ciphertext volume, or `chmod /root/run_jupyter.sh` fails and every install comes back FAILED."""

import contextlib
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import services.docker_service as docker_service_module
from payload_models.payloads import InstallJupyterServerRequest, JupyterServerInstalled
from services.docker_service import DockerService, _VolumeEncryptionState, _jupyter_root_sed


class _InstallJupyterService(DockerService):
    def __init__(self, encryption_state: _VolumeEncryptionState) -> None:
        self.encryption_state = encryption_state
        self.run_jupyter_kwargs: dict = {}
        self.ssh_service = SimpleNamespace(decrypt_payload=lambda *_: "key")

    async def _prepare_known_hosts_policy(self, *_):
        return None

    async def _local_volume_encryption_state(self, ssh_client, container_name):
        return self.encryption_state

    async def run_jupyter(self, **kwargs):
        self.run_jupyter_kwargs = kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("state", "encrypted"),
    [(_VolumeEncryptionState.ENCRYPTED, True), (_VolumeEncryptionState.PLAIN, False)],
)
async def test_install_jupyter_passes_the_pods_volume_encryption(monkeypatch, state, encrypted):
    @contextlib.asynccontextmanager
    async def fake_connect(**_):
        yield object()

    async def fake_sock(*_):
        return {}

    monkeypatch.setattr(docker_service_module.asyncssh, "connect", fake_connect)
    monkeypatch.setattr(docker_service_module.asyncssh, "import_private_key", lambda _: "pkey")
    monkeypatch.setattr(docker_service_module, "public_host_sock", fake_sock)
    service = _InstallJupyterService(state)
    payload = InstallJupyterServerRequest(
        miner_hotkey="hk",
        executor_id="ex",
        pod_id="pod",
        container_name="pod_pod",
        jupyter_port_map=(60110, 60110),
        local_volume="volume_pod",
        local_volume_path="/root",
    )
    executor_info = SimpleNamespace(address="1.2.3.4", port=8000, ssh_username="root", ssh_port=22)

    result = await service.install_jupyter_server(payload, executor_info, SimpleNamespace(ss58_address="v"), "enc")

    assert isinstance(result, JupyterServerInstalled)
    assert service.run_jupyter_kwargs["encrypted_local_volume"] is encrypted


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("local_volume", "local_volume_path", "encrypted", "root"),
    [
        ("volume_pod", "/workspace", False, "/workspace"),
        ("volume_pod", "/root", False, "/root"),
        ("volume_pod", "/workspace", True, "/workspace"),
        (None, "/root", False, "/root"),
    ],
)
async def test_run_jupyter_serves_the_pod_volume_path(local_volume, local_volume_path, encrypted, root):
    # DAH-5426: notebooks land on the pod volume, wherever the template mounts it
    service = _InstallJupyterService(_VolumeEncryptionState.PLAIN)
    service.execute_and_stream_logs = AsyncMock(return_value=(True, ""))

    await DockerService.run_jupyter(
        service,
        ssh_client=object(),
        container_name="pod_pod",
        jupyter_token="t",
        jupyter_port=8888,
        log_tag="t",
        log_extra={},
        local_volume=local_volume,
        local_volume_path=local_volume_path,
        encrypted_local_volume=encrypted,
    )

    copy = next(c.kwargs["command"] for c in service.execute_and_stream_logs.await_args_list if "sed " in c.kwargs["command"])
    assert f"--ServerApp.root_dir={root} #" in copy


def test_jupyter_root_sed_rewrites_the_executor_script():
    script = Path(__file__).parents[3] / "neurons/executor/run_jupyter.sh"

    out = subprocess.run(
        f"sed {_jupyter_root_sed('/workspace')} {script}", shell=True, capture_output=True, text=True, check=True
    ).stdout

    assert "--ServerApp.root_dir=/workspace \\" in out
    assert "--ServerApp.root_dir=/root" not in out


@pytest.mark.parametrize(
    "root",
    ["/workspace;printf${IFS}SED_EXECUTED;true#e;#", "/work#space", "/root/../etc", "workspace", "/a b"],
)
def test_jupyter_root_sed_refuses_a_path_that_is_not_plain(root):
    with pytest.raises(ValueError):
        _jupyter_root_sed(root)


@pytest.mark.asyncio
async def test_run_jupyter_runs_nothing_for_a_hostile_volume_path():
    service = _InstallJupyterService(_VolumeEncryptionState.ENCRYPTED)
    service.execute_and_stream_logs = AsyncMock(return_value=(True, ""))

    with pytest.raises(ValueError):
        await DockerService.run_jupyter(
            service,
            ssh_client=object(),
            container_name="pod_pod",
            jupyter_token="t",
            jupyter_port=8888,
            log_tag="t",
            log_extra={},
            local_volume="volume_pod",
            local_volume_path="/workspace;printf${IFS}SED_EXECUTED;true#e;#",
            encrypted_local_volume=True,
        )

    assert all("sed " not in c.kwargs["command"] for c in service.execute_and_stream_logs.await_args_list)
