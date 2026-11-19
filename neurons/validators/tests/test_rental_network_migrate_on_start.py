"""A stopped pod still on the default bridge is moved onto the rental network when it is started again."""

import json

import pytest
from docker.errors import NotFound

from services.rental_docker_sdk import (
    RENTAL_NETWORK_ICC_OPTION,
    RENTAL_NETWORK_NAME,
    RentalDockerSdkClient,
    RentalPodBridgeGone,
    RentalPodNetworkMoveFailed,
)


class FakeApi:
    def __init__(self, *, running=False, networks=("bridge",), mode="default", rental_network_exists=True):
        self.events = []
        self.connect_requests = []
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
        self.networks[name] = {"Id": f"id-{name}", "Driver": kwargs["driver"], "Options": kwargs["options"]}

    def disconnect_container_from_network(self, container, network):
        self.events.append(("disconnect", container, network))

    def connect_container_to_network(self, container, network):
        self.events.append(("connect", container, network))

    # docker-py's low-level request path, as a pod's own bridge is connected by ID with EndpointConfig.NetworkID
    def _url(self, path, *args):
        return path.format(*args)

    def _post_json(self, url, data=None):
        self.connect_requests.append(json.loads(json.dumps(data)))  # what goes on the wire
        self.events.append(("connect", data["Container"], url.split("/")[2]))

    def _raise_for_status(self, response):
        return None

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

    assert api.events == [("inspect_container", "pod_a"), ("inspect_network", RENTAL_NETWORK_NAME), ("start", "pod_a")]


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




class OldNetworkApi(FakeApi):
    """Its disconnect fails as Docker can for a network removed while the pod was stopped, or for any other reason."""

    def __init__(self, *, disconnect_error, **kwargs):
        super().__init__(**kwargs)
        self.disconnect_error = disconnect_error

    def disconnect_container_from_network(self, container, network):
        super().disconnect_container_from_network(container, network)
        raise self.disconnect_error


@pytest.mark.asyncio
async def test_leaving_an_old_network_already_removed_is_a_no_op_and_the_pod_still_moves():
    # DAH-6737: the old shared network was removed while the pod was stopped; nothing to leave
    api = OldNetworkApi(networks=("lium-old",), disconnect_error=NotFound("network lium-old not found"))
    client = RentalDockerSdkClient(api)

    await client.start(container_name="pod_a", network=RENTAL_NETWORK_NAME)

    assert api.events[-3:] == [
        ("inspect_network", "lium-old"),
        ("connect", "pod_a", RENTAL_NETWORK_NAME),
        ("start", "pod_a"),
    ]


@pytest.mark.asyncio
async def test_a_refused_move_off_a_network_that_still_exists_is_ours_and_never_starts_the_pod():
    api = OldNetworkApi(disconnect_error=RuntimeError("500 Server Error: endpoint busy"))
    api.networks["bridge"] = {"Id": "id-bridge"}
    client = RentalDockerSdkClient(api)

    with pytest.raises(RentalPodNetworkMoveFailed):
        await client.start(container_name="pod_a", network=RENTAL_NETWORK_NAME)

    assert not [e for e in api.events if e[0] in ("connect", "start")]


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
    client = RentalDockerSdkClient(api)

    guarded = await client.ensure_network(OWN)  # the guard's step, before its egress drop
    await client.start(container_name="pod_a", network=OWN, network_id=guarded)

    assert guarded == f"id-{OWN}" and api.events.count(("create_network", OWN)) == 1
    # bound to the ID the drop was applied to, never the name
    assert api.events[-3:] == [
        ("disconnect", "pod_a", RENTAL_NETWORK_NAME),
        ("connect", "pod_a", guarded),
        ("start", "pod_a"),
    ]
    # F-36: the ID in the body too, else dockerd keeps the attachment by name and resolves it again at start
    assert api.connect_requests[-1] == {"Container": "pod_a", "EndpointConfig": {"NetworkID": guarded}}
    assert api.networks[OWN]["Labels"]["io.lium.network"] == "pod"
    assert api.networks[OWN]["Options"][RENTAL_NETWORK_ICC_OPTION] == "false"


@pytest.mark.parametrize("bridge", ["gone", "made_again"])
@pytest.mark.parametrize("mode", [RENTAL_NETWORK_NAME, "default"])
@pytest.mark.asyncio
async def test_start_never_makes_or_uses_a_pod_bridge_other_than_the_guarded_one(bridge, mode):
    # DAH-6737 review (F-28, finding 6): the bridge is removed after its egress drop, and maybe made again under the
    # same name with no drop on it; the start is refused, nothing is created, moved or started
    api = PerPodApi(networks=(RENTAL_NETWORK_NAME,), mode=mode)
    client = RentalDockerSdkClient(api)
    guarded = await client.ensure_network(OWN)
    del api.networks[OWN]
    if bridge == "made_again":
        api.networks[OWN] = {"Id": "id-other", "Driver": "bridge", "Options": {RENTAL_NETWORK_ICC_OPTION: "false"}}
    api.events.clear()

    with pytest.raises(RentalPodBridgeGone):
        await client.start(container_name="pod_a", network=OWN, network_id=guarded)

    assert not [e for e in api.events if e[0] in ("create_network", "disconnect", "connect", "start")]


@pytest.mark.asyncio
async def test_a_stopped_pod_bound_to_a_bridge_since_removed_is_refused_not_started():
    # created on its own bridge by ID; that bridge was removed while it was stopped and the guard made a new one
    api = PerPodApi(networks=(OWN,), mode="id-old")
    client = RentalDockerSdkClient(api)
    guarded = await client.ensure_network(OWN)
    api.events.clear()

    with pytest.raises(RentalPodBridgeGone):
        await client.start(container_name="pod_a", network=OWN, network_id=guarded)

    assert ("start", "pod_a") not in api.events


@pytest.mark.asyncio
async def test_a_pod_bridge_takes_the_mtu_the_provider_set_on_the_template():
    api = PerPodApi(template_options={RENTAL_NETWORK_ICC_OPTION: "false", "com.docker.network.driver.mtu": "1400"})

    await RentalDockerSdkClient(api).ensure_network(OWN)

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

    await RentalDockerSdkClient(api).ensure_network(OWN)

    assert api.removed == ["gone"]


@pytest.mark.asyncio
async def test_a_failed_cleanup_never_blocks_the_pod():
    api = PerPodApi()

    def broken(all=False):
        raise RuntimeError("daemon busy")

    api.containers = broken
    client = RentalDockerSdkClient(api)
    await client.start(container_name="pod_a", network=OWN, network_id=await client.ensure_network(OWN))

    assert api.events[-1] == ("start", "pod_a")


@pytest.mark.asyncio
async def test_a_moved_pod_whose_template_was_pruned_gets_it_back_before_its_start():
    # moved earlier: on its own bridge, NetworkMode still the template, which a prune removed while it was stopped
    api = PerPodApi(networks=(OWN,), mode=RENTAL_NETWORK_NAME, rental_network_exists=False)
    api.networks[OWN] = {"Driver": "bridge", "Options": {RENTAL_NETWORK_ICC_OPTION: "false"}}

    await RentalDockerSdkClient(api).start(container_name="pod_a", network=OWN)

    assert api.events.index(("create_network", RENTAL_NETWORK_NAME)) < api.events.index(("start", "pod_a"))
    assert not [e for e in api.events if e[0] in ("disconnect", "connect")]


def test_bridge_age_reads_the_offset_docker_writes(monkeypatch):
    from datetime import datetime, timezone

    from services import rental_docker_sdk

    class Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 11, 18, 12, 0, 0, tzinfo=timezone.utc)

    monkeypatch.setattr(rental_docker_sdk, "datetime", Frozen)
    # the same instant, 11:50 UTC, in UTC and at +02:00: both 600 s old, not 7800 s and -6600 s
    assert rental_docker_sdk._age_seconds("2026-11-18T11:50:00.123456789Z") == pytest.approx(600, abs=1)
    assert rental_docker_sdk._age_seconds("2026-11-18T13:50:00.123456789+02:00") == pytest.approx(600, abs=1)
    assert rental_docker_sdk._age_seconds("garbage") == 0.0


@pytest.mark.asyncio
async def test_the_cleanup_ages_each_bridge_by_its_own_offset(monkeypatch):
    from datetime import datetime, timezone

    from services import rental_docker_sdk

    class Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 11, 18, 12, 0, 0, tzinfo=timezone.utc)

    monkeypatch.setattr(rental_docker_sdk, "datetime", Frozen)
    api = PerPodApi(
        pod_networks=[
            # 11:45 UTC, 15 min old: removed (read as UTC it would look 2 h in the future and stay forever)
            {"Id": "old", "Name": "lium-rentals-pod_old", "Created": "2026-11-18T13:45:00.5+02:00"},
            # 11:55 UTC, 5 min old, a racing create's: kept (read as UTC it would look 5 h old and be removed)
            {"Id": "fresh", "Name": "lium-rentals-pod_fresh", "Created": "2026-11-18T06:55:00.5-05:00"},
        ],
    )

    await RentalDockerSdkClient(api).ensure_network(OWN)

    assert api.removed == ["old"]


def test_the_reaper_keeps_a_bridge_a_create_here_guarded_and_has_not_run_on_yet():
    # review on lium-platform#3246: a create whose run (retries, an image flatten) outlasts the reap age leaves its
    # guarded bridge unreferenced; another create's reaper must not remove it, so a gone bridge is only ever the host's
    from unittest.mock import Mock

    from services.rental_docker_sdk import bridge_in_use

    held, idle = f"{RENTAL_NETWORK_NAME}-pod_held", f"{RENTAL_NETWORK_NAME}-pod_ended"
    api = Mock()
    api.containers.return_value = []
    api.networks.return_value = [{"Name": n, "Id": n, "Created": "2020-01-01T00:00:00Z"} for n in (held, idle)]
    client = RentalDockerSdkClient(api)

    with bridge_in_use(held):
        client._remove_unused_pod_networks_sync()
    assert [c.args[0] for c in api.remove_network.call_args_list] == [idle]

    api.remove_network.reset_mock()
    client._remove_unused_pod_networks_sync()  # the create ended: an unused bridge is reaped as before
    assert {c.args[0] for c in api.remove_network.call_args_list} == {held, idle}


def test_a_bridge_reserved_while_the_reaper_removes_it_is_made_again_before_its_guard():
    # DAH-6737 review (F-38): the reaper checked the old bridge unreserved and is removing it when a create reserves it
    # and ensures it for its guard; the guard must get a bridge that exists after the removal, never the removed one
    import threading

    from services.rental_docker_sdk import bridge_in_use

    stale = {"Id": "old-id", "Name": OWN, "Created": OLD, "Driver": "bridge", "Options": {RENTAL_NETWORK_ICC_OPTION: "false"}}
    api = PerPodApi(pod_networks=[stale])
    api.networks[OWN] = stale
    removing, release = threading.Event(), threading.Event()

    def remove_network(net_id):
        removing.set()
        release.wait(5)
        api.removed.append(net_id)
        del api.networks[OWN]

    api.remove_network = remove_network
    client = RentalDockerSdkClient(api)
    reaper = threading.Thread(target=client._remove_unused_pod_networks_sync)
    reaper.start()
    assert removing.wait(5)
    guarded: list[str] = []
    with bridge_in_use(OWN):
        ensure = threading.Thread(target=lambda: guarded.append(client._ensure_rental_network_sync(OWN)))
        ensure.start()
        ensure.join(0.3)  # the ensure waits on the reaper's removal of that bridge
        release.set()
        reaper.join(5)
        ensure.join(5)

    assert api.removed == ["old-id"]
    assert guarded == [f"id-{OWN}"] and api.networks[OWN]["Id"] == guarded[0]
