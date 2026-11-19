"""A stopped pod still on the default bridge is moved onto the rental network when it is started again."""

import pytest
from docker.errors import NotFound

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



# DAH-6737: each rental has a bridge of its own; no two tenants share a link
OWN = "lium-rentals-pod_a"


class _Networks(dict):
    """FakeApi's name -> network map that also answers docker's `networks(filters=...)` list call."""

    def __init__(self, known, listing):
        super().__init__(known)
        self.listing = listing

    def __call__(self, filters=None):
        assert filters == {"label": "io.lium.network=pod"}
        return self.listing


class PerPodApi(FakeApi):
    def __init__(self, *, template_options=None, containers=(), pod_networks=(), **kwargs):
        super().__init__(**kwargs)
        if template_options is not None:
            self.networks[RENTAL_NETWORK_NAME]["Options"] = template_options
        self.listed = list(containers)
        self.networks = _Networks(self.networks, list(pod_networks))
        self.removed = []

    def create_network(self, name, **kwargs):
        super().create_network(name, **kwargs)
        self.networks[name]["Labels"] = kwargs["labels"]

    def containers(self, all=False):
        assert all is True  # a stopped pod still owns its bridge
        return self.listed

    def remove_network(self, net_id):
        self.removed.append(net_id)


def _listed(*networks):
    return {"NetworkSettings": {"Networks": {name: {} for name in networks}}}


OLD = "2026-01-01T00:00:00.123456789Z"


@pytest.mark.asyncio
async def test_start_moves_a_stopped_pod_off_the_shared_rentals_bridge_onto_its_own():
    api = PerPodApi(networks=(RENTAL_NETWORK_NAME,), mode=RENTAL_NETWORK_NAME)

    await RentalDockerSdkClient(api).start(container_name="pod_a", network=OWN)

    assert ("create_network", OWN) in api.events
    assert api.events[-3:] == [
        ("disconnect", "pod_a", RENTAL_NETWORK_NAME),
        ("connect", "pod_a", OWN),
        ("start", "pod_a"),
    ]
    assert api.networks[OWN]["Labels"]["io.lium.network"] == "pod"
    assert api.networks[OWN]["Options"][RENTAL_NETWORK_ICC_OPTION] == "false"


@pytest.mark.asyncio
async def test_a_pod_bridge_takes_the_mtu_the_provider_set_on_the_template():
    api = PerPodApi(template_options={RENTAL_NETWORK_ICC_OPTION: "false", "com.docker.network.driver.mtu": "1400"})

    await RentalDockerSdkClient(api).start(container_name="pod_a", network=OWN)

    assert api.networks[OWN]["Options"]["com.docker.network.driver.mtu"] == "1400"


@pytest.mark.asyncio
async def test_creating_a_pod_bridge_removes_only_old_unreferenced_pod_bridges():
    api = PerPodApi(
        containers=[_listed("lium-rentals-pod_stopped")],
        pod_networks=[
            {"Id": "gone", "Name": "lium-rentals-pod_gone", "Created": OLD},
            {"Id": "stopped", "Name": "lium-rentals-pod_stopped", "Created": OLD},
            # a racing create has made its bridge but not yet its container
            {"Id": "fresh", "Name": "lium-rentals-pod_fresh", "Created": "2999-01-01T00:00:00Z"},
            {"Id": "tmpl", "Name": RENTAL_NETWORK_NAME, "Created": OLD},
            {"Id": "foreign", "Name": "someone-else", "Created": OLD},
        ],
    )

    await RentalDockerSdkClient(api).start(container_name="pod_a", network=OWN)

    assert api.removed == ["gone"]


@pytest.mark.asyncio
async def test_a_failed_cleanup_never_blocks_the_pod():
    api = PerPodApi()

    def broken(all=False):
        raise RuntimeError("daemon busy")

    api.containers = broken
    await RentalDockerSdkClient(api).start(container_name="pod_a", network=OWN)

    assert api.events[-1] == ("start", "pod_a")
