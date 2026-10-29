"""An install-jupyter request on a running pod carries no encryption flag. On a pod whose volume
is gocryptfs-encrypted the script must be piped into the container, not copied into the
ciphertext volume, or `chmod /root/run_jupyter.sh` fails and every install comes back FAILED."""

import contextlib
from types import SimpleNamespace

import pytest

import services.docker_service as docker_service_module
from payload_models.payloads import InstallJupyterServerRequest, JupyterServerInstalled
from services.docker_service import DockerService, _VolumeEncryptionState


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
