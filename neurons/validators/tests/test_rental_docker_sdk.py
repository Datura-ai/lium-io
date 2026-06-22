import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from datura.requests.miner_requests import ExecutorSSHInfo
from services.rental_docker_sdk import (
    ContainerExecSpec,
    ContainerRunSpec,
    DeviceMount,
    GpuDeviceRequest,
    PortBinding,
    RentalDockerConnectionError,
    RentalDockerSdkClient,
    RentalDockerSdkClientFactory,
    VolumeMount,
    build_authorized_keys_exec_spec,
    build_environment_exec_spec,
)


class FakeApiClient:
    def __init__(self):
        self.host_config_kwargs = None
        self.created_container = None
        self.started = []
        self.exec_created = []
        self.exec_started = []
        self.exec_inspected = []
        self.closed = False

    def create_host_config(self, **kwargs):
        self.host_config_kwargs = kwargs
        return {"host_config": True}

    def create_container(self, **kwargs):
        self.created_container = kwargs
        return {"Id": "container-id"}

    def start(self, container_name):
        self.started.append(container_name)

    def exec_create(self, **kwargs):
        self.exec_created.append(kwargs)
        return {"Id": "exec-id"}

    def exec_start(self, exec_id, **kwargs):
        self.exec_started.append((exec_id, kwargs))
        return (b"stdout", b"stderr")

    def exec_inspect(self, exec_id):
        self.exec_inspected.append(exec_id)
        return {"ExitCode": 0}

    def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_run_container_maps_spec_to_docker_sdk_api():
    api_client = FakeApiClient()
    client = RentalDockerSdkClient(api_client)

    await client.run_container(
        ContainerRunSpec(
            image="registry.example/app:tag",
            name="pod_test",
            command=("sh", "-c", "echo inside"),
            environment={"HOSTILE_ENV": "value'; echo SHOULD_NOT_RUN; $(echo nope)"},
            ports=(PortBinding(container_port=22, host_port=30022),),
            volumes=(VolumeMount(source="volume_test", target="/workspace"),),
            restart_policy="unless-stopped",
            cap_add=("NET_ADMIN",),
            sysctls={"net.ipv4.conf.all.src_valid_mark": "1"},
            devices=(
                DeviceMount(
                    path_on_host="/dev/nvidia0",
                    path_in_container="/dev/nvidia0",
                ),
            ),
            device_requests=(
                GpuDeviceRequest(device_ids=("GPU-test",)),
            ),
            cpu_count=2,
            memory_gb=8,
            storage_limit_gb=20,
            shm_size="1g",
        )
    )

    assert api_client.created_container["image"] == "registry.example/app:tag"
    assert api_client.created_container["name"] == "pod_test"
    assert api_client.created_container["command"] == ["sh", "-c", "echo inside"]
    assert api_client.created_container["environment"]["HOSTILE_ENV"].startswith("value'")
    assert api_client.host_config_kwargs["port_bindings"] == {"22/tcp": 30022}
    assert api_client.host_config_kwargs["binds"] == ["volume_test:/workspace:rw"]
    assert api_client.host_config_kwargs["restart_policy"] == {"Name": "unless-stopped"}
    assert api_client.host_config_kwargs["devices"] == [
        "/dev/nvidia0:/dev/nvidia0:rwm"
    ]
    assert dict(api_client.host_config_kwargs["device_requests"][0]) == {
        "Driver": "",
        "Count": 0,
        "DeviceIDs": ["GPU-test"],
        "Capabilities": [["gpu"]],
        "Options": {},
    }
    assert api_client.host_config_kwargs["nano_cpus"] == 2_000_000_000
    assert api_client.host_config_kwargs["mem_limit"] == "8g"
    assert api_client.host_config_kwargs["storage_opt"] == {"size": "20g"}
    assert api_client.started == ["pod_test"]


@pytest.mark.asyncio
async def test_exec_in_container_passes_argv_and_environment_as_data():
    api_client = FakeApiClient()
    client = RentalDockerSdkClient(api_client)

    result = await client.exec_in_container(
        ContainerExecSpec(
            container_name="pod_exec",
            argv=("sh", "-c", "cat /tmp/file"),
            environment={"A": "B"},
        )
    )

    assert result.exit_status == 0
    assert result.stdout == "stdout"
    assert result.stderr == "stderr"
    assert api_client.exec_created == [
        {
            "container": "pod_exec",
            "cmd": ["sh", "-c", "cat /tmp/file"],
            "stdin": False,
            "environment": {"A": "B"},
        }
    ]
    assert api_client.exec_started == [("exec-id", {"demux": True})]


def test_stdin_exec_spec_builders_keep_values_out_of_argv():
    public_key = 'ssh-ed25519 AAAA user"; echo KEY_MARKER; $(echo nope)'
    key_spec = build_authorized_keys_exec_spec(
        container_name="pod_key",
        public_keys=[public_key],
    )
    env_spec = build_environment_exec_spec(
        container_name="pod_env",
        environment={
            "HOSTILE_ENV": "value'; echo ENV_MARKER; $(echo env)",
            "MULTILINE_ENV": "line1\nENV_NEWLINE_MARKER",
        },
    )

    assert key_spec is not None
    assert public_key not in " ".join(key_spec.argv)
    assert key_spec.stdin == f"{public_key}\n"
    assert env_spec is not None
    assert "ENV_MARKER" not in " ".join(env_spec.argv)
    assert "ENV_NEWLINE_MARKER" not in " ".join(env_spec.argv)
    assert "HOSTILE_ENV=value'; echo ENV_MARKER; $(echo env)\n" in env_spec.stdin
    assert "MULTILINE_ENV=line1\nENV_NEWLINE_MARKER\n" in env_spec.stdin


@pytest.mark.asyncio
async def test_factory_restores_home_and_closes_client(monkeypatch):
    original_home = os.environ.get("HOME")
    created = {}
    api_client = FakeApiClient()

    def api_client_factory(**kwargs):
        ssh_home = Path(os.environ["HOME"])
        created["kwargs"] = kwargs
        created["ssh_home"] = ssh_home
        created["key_mode"] = (ssh_home / ".ssh" / "id_executor").stat().st_mode & 0o777
        created["known_hosts"] = (ssh_home / ".ssh" / "known_hosts").read_text()
        return api_client

    monkeypatch.setattr(
        "services.rental_docker_sdk._validate_paramiko_known_hosts",
        Mock(),
    )
    factory = RentalDockerSdkClientFactory(api_client_factory=api_client_factory)
    executor_info = ExecutorSSHInfo(
        uuid="executor-id",
        address="127.0.0.1",
        port=8000,
        ssh_username="root",
        ssh_port=2222,
        python_path="/usr/bin/python",
        root_dir="/root",
        ssh_host_key="ssh-ed25519 AAAATESTKEY",
    )

    async with factory.connect(
        executor_info=executor_info,
        private_key="PRIVATE KEY",
    ) as client:
        assert isinstance(client, RentalDockerSdkClient)

    assert created["kwargs"]["base_url"] == "ssh://root@127.0.0.1:2222"
    assert created["kwargs"]["use_ssh_client"] is False
    assert "PRIVATE KEY" not in str(created["kwargs"])
    assert created["key_mode"] == 0o600
    assert "[127.0.0.1]:2222 ssh-ed25519 AAAATESTKEY" in created["known_hosts"]
    assert os.environ.get("HOME") == original_home
    assert api_client.closed is True


@pytest.mark.asyncio
async def test_factory_fails_closed_without_executor_host_key():
    api_client_factory = Mock()
    factory = RentalDockerSdkClientFactory(api_client_factory=api_client_factory)
    executor_info = ExecutorSSHInfo(
        uuid="executor-id",
        address="127.0.0.1",
        port=8000,
        ssh_username="root",
        ssh_port=2222,
        python_path="/usr/bin/python",
        root_dir="/root",
        ssh_host_key=None,
    )

    with pytest.raises(RentalDockerConnectionError):
        async with factory.connect(
            executor_info=executor_info,
            private_key="PRIVATE KEY",
        ):
            pass

    api_client_factory.assert_not_called()
