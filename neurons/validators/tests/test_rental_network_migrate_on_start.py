"""A stopped pod still on the default bridge is moved onto the rental network when it is started again."""

import pytest
from docker.errors import NotFound

import services.docker_service as docker_service
from services.rental_docker_sdk import (
    RENTAL_NETWORK_ICC_OPTION,
    RENTAL_NETWORK_NAME,
    RentalDockerSdkClient,
)


class FakeApi:
    def __init__(self, *, running=False, networks=("bridge",), mode="default", rental_network_exists=True):
        self.events = []
        self.container = {
            "State": {"Running": running, "Restarting": False},
            "HostConfig": {"NetworkMode": mode},
            "NetworkSettings": {"Networks": {name: {} for name in networks}},
        }
        self.networks = {}
        if rental_network_exists:
            self.networks[RENTAL_NETWORK_NAME] = {"Driver": "bridge", "Options": {RENTAL_NETWORK_ICC_OPTION: "false"}}

    def inspect_container(self, name):
        self.events.append(("inspect_container", name))
        return self.container

    def inspect_network(self, name):
        self.events.append(("inspect_network", name))
        if name not in self.networks:
            raise NotFound(f"network {name} not found")
        return self.networks[name]

    def create_network(self, name, **kwargs):
        self.events.append(("create_network", name))
        self.networks[name] = {"Driver": kwargs["driver"], "Options": kwargs["options"]}

    def disconnect_container_from_network(self, container, network):
        self.events.append(("disconnect", container, network))

    def connect_container_to_network(self, container, network):
        self.events.append(("connect", container, network))

    def start(self, container):
        self.events.append(("start", container))


@pytest.mark.asyncio
async def test_start_moves_a_stopped_default_bridge_pod_onto_the_rental_network():
    api = FakeApi()
    client = RentalDockerSdkClient(api)

    await client.start(container_name="pod_a", network=RENTAL_NETWORK_NAME)

    assert api.events == [
        ("inspect_container", "pod_a"),
        ("inspect_network", RENTAL_NETWORK_NAME),
        ("disconnect", "pod_a", "bridge"),
        ("connect", "pod_a", RENTAL_NETWORK_NAME),
        ("start", "pod_a"),
    ]


@pytest.mark.asyncio
async def test_start_creates_the_rental_network_on_a_host_that_has_none_before_moving():
    api = FakeApi(rental_network_exists=False)
    client = RentalDockerSdkClient(api)

    await client.start(container_name="pod_a", network=RENTAL_NETWORK_NAME)

    assert ("create_network", RENTAL_NETWORK_NAME) in api.events
    assert api.events[-2:] == [("connect", "pod_a", RENTAL_NETWORK_NAME), ("start", "pod_a")]


@pytest.mark.asyncio
async def test_start_leaves_a_pod_already_on_the_rental_network_alone():
    api = FakeApi(networks=(RENTAL_NETWORK_NAME,), mode=RENTAL_NETWORK_NAME)
    client = RentalDockerSdkClient(api)

    await client.start(container_name="pod_a", network=RENTAL_NETWORK_NAME)

    assert api.events == [("inspect_container", "pod_a"), ("start", "pod_a")]


@pytest.mark.asyncio
async def test_start_does_not_move_a_running_pod():
    api = FakeApi(running=True)
    client = RentalDockerSdkClient(api)

    await client.start(container_name="pod_a", network=RENTAL_NETWORK_NAME)

    assert api.events == [("inspect_container", "pod_a"), ("start", "pod_a")]


@pytest.mark.asyncio
async def test_start_does_not_move_a_pod_on_the_host_network():
    api = FakeApi(networks=("host",), mode="host")
    client = RentalDockerSdkClient(api)

    await client.start(container_name="pod_a", network=RENTAL_NETWORK_NAME)

    assert api.events == [("inspect_container", "pod_a"), ("start", "pod_a")]


@pytest.mark.asyncio
async def test_start_without_a_network_only_starts():
    api = FakeApi()
    client = RentalDockerSdkClient(api)

    await client.start(container_name="pod_a")

    assert api.events == [("start", "pod_a")]


def test_restart_network_is_none_when_the_flag_is_off(monkeypatch):
    monkeypatch.setattr(docker_service.settings, "RENTAL_NETWORK_MIGRATE_ON_START_ENABLED", False)

    network = docker_service._restart_network()

    assert network is None


def test_restart_network_is_the_rental_network_when_the_flag_is_on(monkeypatch):
    monkeypatch.setattr(docker_service.settings, "RENTAL_NETWORK_MIGRATE_ON_START_ENABLED", True)

    network = docker_service._restart_network()

    assert network == RENTAL_NETWORK_NAME
