"""The rental container's run spec — ONE definition for the validator (which builds it) and the
executor (which, under liumd `POST /rent`, creates the container from it locally).

The validator's SSH path hands `ContainerRunSpec` to docker-py over an SSH tunnel
(`neurons/validators/src/services/rental_docker_sdk.py`); the executor's local path hands the SAME
spec to the SAME docker-py calls on the host (`neurons/executor/src/services/local_rent_service.py`).
Both build the HostConfig with `build_host_config_kwargs` below, so what the container is made of
cannot drift between the two paths: a field added to the spec is added here, once, and both ends
pick it up.

The wire form (`spec_to_wire` / `spec_from_wire`) is JSON-safe and bounded; `spec_from_wire` refuses
anything outside the shape rather than guess. It carries NO secret: the validator sends a spec
locally only when `environment`, `command` and `entrypoint` are the defaults (`carries_only_public_fields`),
because the executor's API port is plain HTTP — a renter's env vars and startup command stay on the
SSH path until the executor has a key of its own.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# --- the spec -----------------------------------------------------------------------------------


@dataclass(slots=True)
class PortBinding:
    container_port: int
    host_port: int
    protocol: str = "tcp"


@dataclass(slots=True)
class VolumeMount:
    source: str
    target: str
    read_only: bool = False


@dataclass(slots=True)
class DeviceMount:
    path_on_host: str
    path_in_container: str | None = None
    permissions: str = "rwm"


DEFAULT_GPU_CAPABILITIES: tuple[tuple[str, ...], ...] = (("gpu",),)


@dataclass(slots=True)
class GpuDeviceRequest:
    count: int | None = None
    device_ids: tuple[str, ...] = ()
    capabilities: tuple[tuple[str, ...], ...] = DEFAULT_GPU_CAPABILITIES


@dataclass(slots=True)
class ContainerUlimit:
    name: str
    soft: int
    hard: int


@dataclass(slots=True)
class ContainerRunSpec:
    image: str
    name: str
    command: tuple[str, ...] = ()
    environment: dict[str, str] = field(default_factory=dict)
    ports: tuple[PortBinding, ...] = ()
    volumes: tuple[VolumeMount, ...] = ()
    restart_policy: str | None = "unless-stopped"
    runtime: str | None = None
    cap_add: tuple[str, ...] = ()
    sysctls: dict[str, str] = field(default_factory=dict)
    ulimits: tuple[ContainerUlimit, ...] = ()
    devices: tuple[DeviceMount, ...] = ()
    device_requests: tuple[GpuDeviceRequest, ...] = ()
    cpu_count: int | None = None
    memory_gb: int | None = None
    storage_limit_gb: int | None = None
    shm_size: str | None = None
    entrypoint: str | None = None
    # None keeps the daemon's default bridge (the CVM quote broker talks over unix sockets only);
    # the validator names its icc-off rental network here (DAH-3199).
    network: str | None = None


# --- the rental network (DAH-3199) --------------------------------------------------------------

# Every rental on a host shares this user-defined bridge instead of docker0. The daemon's default
# bridge allows inter-container traffic, and a pod holds NET_ADMIN, so two rentals on a split host
# could otherwise reach each other's unpublished ports. Docker enforces ICC=false with a FORWARD
# drop between ports of this bridge; published ports still arrive through the host and NAT egress
# is untouched. One network per host — nothing to remove at teardown. The validator names it in
# `ContainerRunSpec.network`; whichever side creates the container makes sure it exists first
# (`ensure_rental_network`) — the SAME check on both paths, so a locally made rental is never the
# one on docker0.
RENTAL_NETWORK_NAME = "lium-rentals"
RENTAL_NETWORK_ICC_OPTION = "com.docker.network.bridge.enable_icc"
RENTAL_NETWORK_OPTIONS = {RENTAL_NETWORK_ICC_OPTION: "false"}
RENTAL_NETWORK_LABELS = {"io.lium.purpose": "rental-isolation"}


class RentalNetworkError(RuntimeError):
    """The rental network cannot be used: it could not be created, or a same-named network on the
    host does not turn inter-container traffic off. The container is NOT created."""


def ensure_rental_network(api_client: Any, name: str) -> None:
    """The container's network exists on the host and has inter-container traffic off.

    Runs before every rental `create_container`, so the isolation holds on a host that has never
    seen a rental, on one whose network was removed by hand, and for two creates racing on the
    same host (the loser's `create_network` conflicts and the network is inspected again). A
    network of that name whose options do not turn ICC off is refused rather than used: running
    the pod on it would silently restore the docker0 behaviour this network exists to end.
    Blocking (docker-py); callers run it off the event loop."""
    network = _inspect_network_or_none(api_client, name)
    if network is None:
        try:
            api_client.create_network(
                name,
                driver="bridge",
                options=dict(RENTAL_NETWORK_OPTIONS),
                labels=dict(RENTAL_NETWORK_LABELS),
            )
        except Exception as exc:
            network = _inspect_network_or_none(api_client, name)
            if network is None:
                detail = str(exc) or exc.__class__.__name__
                raise RentalNetworkError(f"Docker SDK create network {name} failed: {detail}") from exc
        else:
            network = _inspect_network_or_none(api_client, name)
            if network is None:
                raise RentalNetworkError(f"Docker network {name} was created but cannot be inspected")
    require_icc_off(name, network)


def require_icc_off(name: str, network: dict) -> None:
    options = network.get("Options") or {}
    if network.get("Driver") == "bridge" and options.get(RENTAL_NETWORK_ICC_OPTION) == "false":
        return
    raise RentalNetworkError(
        f"Docker network {name} exists on the executor but is not a bridge with "
        f"{RENTAL_NETWORK_ICC_OPTION}=false (driver={network.get('Driver')!r}, options={options!r}); "
        "refusing to run the rental on it. Remove the network once it is empty so the validator "
        "recreates it with inter-container traffic off."
    )


def _inspect_network_or_none(api_client: Any, name: str) -> dict | None:
    try:
        return api_client.inspect_network(name)
    except Exception as exc:
        if _is_not_found(exc):
            return None
        raise


def _is_not_found(exc: Exception) -> bool:
    # docker-py's NotFound / ImageNotFound, or any APIError whose daemon answer was a 404
    if exc.__class__.__name__ in {"ImageNotFound", "NotFound"}:
        return True
    return getattr(getattr(exc, "response", None), "status_code", None) == 404


# --- docker-py arguments (the one HostConfig) ---------------------------------------------------


def build_host_config_kwargs(spec: ContainerRunSpec) -> dict:
    """`APIClient.create_host_config(**kwargs)` for this spec. Both ends call this."""
    kwargs = {
        "port_bindings": {_port_key(port): port.host_port for port in spec.ports},
        "binds": [
            f"{volume.source}:{volume.target}:{'ro' if volume.read_only else 'rw'}"
            for volume in spec.volumes
        ],
        "restart_policy": {"Name": spec.restart_policy} if spec.restart_policy else None,
        "runtime": spec.runtime,
        "cap_add": list(spec.cap_add) or None,
        "sysctls": spec.sysctls or None,
        "ulimits": _ulimits(spec.ulimits),
        "devices": [_device_arg(device) for device in spec.devices],
        "device_requests": _device_requests(spec.device_requests),
        "nano_cpus": spec.cpu_count * 1_000_000_000 if spec.cpu_count else None,
        "mem_limit": f"{spec.memory_gb}g" if spec.memory_gb else None,
        "storage_opt": ({"size": f"{spec.storage_limit_gb}g"} if spec.storage_limit_gb else None),
        "shm_size": spec.shm_size,
        "network_mode": spec.network,
    }
    return {key: value for key, value in kwargs.items() if value is not None}


def create_and_start(
    api_client: Any,
    spec: ContainerRunSpec,
    *,
    labels: dict[str, str] | None = None,
    on_created: Callable[[str], None] | None = None,
) -> str | None:
    """The one `docker run -d` of a rental, as docker-py `APIClient` calls: the validator issues
    them through its SSH tunnel, the executor's `POST /rent` on the host — the same calls, so the
    container is the same whichever side made it (the executor adds its rollback label, nothing
    else). `on_created` gets the id the moment the daemon answers the create, before `start` — so
    a start that fails still leaves the id with the caller. Returns the id when the daemon gave
    one. Blocking; callers run it off the event loop."""
    host_config = api_client.create_host_config(**build_host_config_kwargs(spec))
    created = api_client.create_container(
        image=spec.image,
        command=list(spec.command) or None,
        detach=True,
        ports=container_ports(spec.ports) or None,
        environment=spec.environment or None,
        volumes=container_volumes(spec.volumes) or None,
        name=spec.name,
        entrypoint=spec.entrypoint or None,
        host_config=host_config,
        **({"labels": labels} if labels else {}),
    )
    container_id = created.get("Id") if isinstance(created, dict) else None
    if on_created is not None and container_id:
        on_created(container_id)
    api_client.start(spec.name)
    return container_id


def container_ports(ports: tuple[PortBinding, ...]) -> list[tuple[int, str]]:
    """`APIClient.create_container(ports=…)`: the container-side ports to expose."""
    return [(port.container_port, port.protocol) for port in ports]


def container_volumes(volumes: tuple[VolumeMount, ...]) -> list[str]:
    """`APIClient.create_container(volumes=…)`: the container-side mount points."""
    return [volume.target for volume in volumes]


def _port_key(port: PortBinding) -> str:
    return f"{port.container_port}/{port.protocol}"


def _device_arg(device: DeviceMount) -> str:
    target = device.path_in_container or device.path_on_host
    return f"{device.path_on_host}:{target}:{device.permissions}"


def _ulimits(ulimits: tuple[ContainerUlimit, ...]) -> list | None:
    if not ulimits:
        return None
    from docker.types import Ulimit  # docker-py is a dependency of both ends, not of datura

    return [Ulimit(name=ulimit.name, soft=ulimit.soft, hard=ulimit.hard) for ulimit in ulimits]


def _device_requests(device_requests: tuple[GpuDeviceRequest, ...]) -> list:
    if not device_requests:
        return []
    from docker.types import DeviceRequest

    return [
        DeviceRequest(
            count=device_request.count,
            device_ids=list(device_request.device_ids) or None,
            capabilities=[list(capability) for capability in device_request.capabilities],
        )
        for device_request in device_requests
    ]


# --- the wire form ------------------------------------------------------------------------------

# The env the validator sets on every rental container itself (docker_service.build_run_spec);
# anything else in `environment` is the renter's and never crosses the plain-HTTP call.
PUBLIC_ENVIRONMENT = {"NVIDIA_DRIVER_CAPABILITIES": "all"}

WIRE_FIELDS = (
    "image", "name", "command", "environment", "ports", "volumes", "restart_policy", "runtime",
    "cap_add", "sysctls", "ulimits", "devices", "device_requests", "cpu_count", "memory_gb",
    "storage_limit_gb", "shm_size", "entrypoint", "network",
)  # fmt: skip

# Bounds: what one rental container legitimately has, with room. `spec_from_wire` refuses more.
MAX_STR = 512
MAX_LIST = 64
MAX_DICT = 64
NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,254}$")  # docker's own name rule
# The validator names every rental container `pod_<pod_id>` and every filler `filler_<pod_id>`
# (`DockerService.get_container_name`); the executor refuses a signed intent for any other name.
RENTAL_CONTAINER_NAME_PREFIXES = ("pod_", "filler_")
IMAGE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./:@+-]{0,511}$")
PROTOCOLS = ("tcp", "udp")
RESTART_POLICIES = ("no", "always", "on-failure", "unless-stopped")
PERMISSIONS_PATTERN = re.compile(r"^[rwm]{1,3}$")


class WireError(ValueError):
    """The wire document is not a rental spec: which field, why."""


def carries_only_public_fields(spec: ContainerRunSpec) -> bool:
    """True when nothing in the spec is the renter's private input: no startup command, no
    entrypoint, no environment beyond what the validator sets itself. Only such a spec may travel
    on the executor's plain-HTTP API; everything else keeps the SSH tunnel."""
    return (
        not spec.command
        and spec.entrypoint is None
        and dict(spec.environment) == PUBLIC_ENVIRONMENT
    )


def spec_to_wire(spec: ContainerRunSpec) -> dict[str, Any]:
    return {
        "image": spec.image,
        "name": spec.name,
        "command": list(spec.command),
        "environment": dict(spec.environment),
        "ports": [
            {"container_port": p.container_port, "host_port": p.host_port, "protocol": p.protocol}
            for p in spec.ports
        ],
        "volumes": [
            {"source": v.source, "target": v.target, "read_only": v.read_only} for v in spec.volumes
        ],
        "restart_policy": spec.restart_policy,
        "runtime": spec.runtime,
        "cap_add": list(spec.cap_add),
        "sysctls": dict(spec.sysctls),
        "ulimits": [{"name": u.name, "soft": u.soft, "hard": u.hard} for u in spec.ulimits],
        "devices": [
            {
                "path_on_host": d.path_on_host,
                "path_in_container": d.path_in_container,
                "permissions": d.permissions,
            }
            for d in spec.devices
        ],
        "device_requests": [
            {
                "count": r.count,
                "device_ids": list(r.device_ids),
                "capabilities": [list(c) for c in r.capabilities],
            }
            for r in spec.device_requests
        ],
        # 0 and None both mean "no limit" to `build_host_config_kwargs` (a pod's ram_total defaults
        # to 0); the wire says it one way, so the executor's strict parser reads the same HostConfig.
        "cpu_count": spec.cpu_count or None,
        "memory_gb": spec.memory_gb or None,
        "storage_limit_gb": spec.storage_limit_gb or None,
        "shm_size": spec.shm_size,
        "entrypoint": spec.entrypoint,
        # the icc-off rental bridge (DAH-3199); None = the daemon's default bridge
        "network": spec.network,
    }


def spec_from_wire(raw: Any) -> ContainerRunSpec:
    """The inverse of `spec_to_wire`, refusing (WireError) any shape, type or size outside a
    rental container's. Unknown keys are refused too: a newer validator's field must not be
    silently dropped by an older executor."""
    if not isinstance(raw, dict):
        raise WireError("spec is not an object")
    unknown = set(raw) - set(WIRE_FIELDS)
    if unknown:
        raise WireError(f"unknown field(s): {sorted(unknown)}")
    image = _str(raw, "image", required=True)
    if not IMAGE_PATTERN.fullmatch(image):
        raise WireError("image is not an image reference")
    name = _str(raw, "name", required=True)
    if not NAME_PATTERN.fullmatch(name):
        raise WireError("name is not a container name")
    ports = tuple(
        PortBinding(
            container_port=_port(p, "container_port"),
            host_port=_port(p, "host_port"),
            protocol=_choice(p, "protocol", PROTOCOLS, default="tcp"),
        )
        for p in _objects(raw, "ports")
    )
    volumes = tuple(
        VolumeMount(
            source=_str(v, "source", required=True),
            target=_str(v, "target", required=True),
            read_only=_bool(v, "read_only", default=False),
        )
        for v in _objects(raw, "volumes")
    )
    for volume in volumes:
        if not volume.target.startswith("/"):
            raise WireError("volume target is not an absolute path")
    ulimits = tuple(
        ContainerUlimit(
            name=_str(u, "name", required=True),
            soft=_int(u, "soft", required=True, lo=-1),
            hard=_int(u, "hard", required=True, lo=-1),
        )
        for u in _objects(raw, "ulimits")
    )
    devices = tuple(
        DeviceMount(
            path_on_host=_str(d, "path_on_host", required=True),
            path_in_container=_str(d, "path_in_container"),
            permissions=_str(d, "permissions", default="rwm"),
        )
        for d in _objects(raw, "devices")
    )
    for device in devices:
        if not device.path_on_host.startswith("/dev/") or not PERMISSIONS_PATTERN.fullmatch(
            device.permissions
        ):
            raise WireError("device is not a /dev path with rwm permissions")
        if device.path_in_container is not None and not device.path_in_container.startswith("/"):
            raise WireError("device path in container is not absolute")
    device_requests = tuple(
        GpuDeviceRequest(
            count=_int(r, "count", lo=-1),
            device_ids=tuple(_strings(r, "device_ids")),
            # absent = the dataclass default (the inverse of `spec_to_wire`), not "no capability"
            capabilities=(
                tuple(tuple(_strings({"c": c}, "c")) for c in _list(r, "capabilities"))
                if "capabilities" in r
                else DEFAULT_GPU_CAPABILITIES
            ),
        )
        for r in _objects(raw, "device_requests")
    )
    restart_policy = _str(raw, "restart_policy", default="unless-stopped")
    if restart_policy is not None and restart_policy not in RESTART_POLICIES:
        raise WireError(f"restart_policy is not one of {list(RESTART_POLICIES)}")
    network = _str(raw, "network")
    if network is not None and not NAME_PATTERN.fullmatch(network):
        raise WireError("network is not a docker network name")
    return ContainerRunSpec(
        image=image,
        name=name,
        command=tuple(_strings(raw, "command")),
        environment=_str_dict(raw, "environment"),
        ports=ports,
        volumes=volumes,
        restart_policy=restart_policy,
        runtime=_str(raw, "runtime"),
        cap_add=tuple(_strings(raw, "cap_add")),
        sysctls=_str_dict(raw, "sysctls"),
        ulimits=ulimits,
        devices=devices,
        device_requests=device_requests,
        cpu_count=_int(raw, "cpu_count", lo=1),
        memory_gb=_int(raw, "memory_gb", lo=1),
        storage_limit_gb=_int(raw, "storage_limit_gb", lo=1),
        shm_size=_str(raw, "shm_size"),
        entrypoint=_str(raw, "entrypoint"),
        network=network,
    )


def _str(obj: dict, key: str, *, required: bool = False, default: str | None = None) -> str | None:
    value = obj.get(key, default)
    if value is None:
        if required:
            raise WireError(f"{key} is missing")
        return None
    if not isinstance(value, str) or len(value) > MAX_STR or (required and not value):
        raise WireError(f"{key} is not a string of 1..{MAX_STR} characters")
    return value or default  # an optional "" is "not set": the default, as the dataclass says


def _int(obj: dict, key: str, *, required: bool = False, lo: int, hi: int = 2**31) -> int | None:
    value = obj.get(key)
    if value is None:
        if required:
            raise WireError(f"{key} is missing")
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        raise WireError(f"{key} is not an integer in {lo}..{hi}")
    return value


def _port(obj: dict, key: str) -> int:
    return _int(obj, key, required=True, lo=1, hi=65535)  # type: ignore[return-value]


def _bool(obj: dict, key: str, *, default: bool) -> bool:
    value = obj.get(key, default)
    if not isinstance(value, bool):
        raise WireError(f"{key} is not a boolean")
    return value


def _choice(obj: dict, key: str, choices: tuple[str, ...], *, default: str) -> str:
    value = obj.get(key, default)
    if value not in choices:
        raise WireError(f"{key} is not one of {list(choices)}")
    return value


def _list(obj: dict, key: str) -> list:
    value = obj.get(key, [])
    if not isinstance(value, list) or len(value) > MAX_LIST:
        raise WireError(f"{key} is not a list of at most {MAX_LIST}")
    return value


def _objects(obj: dict, key: str) -> list[dict]:
    items = _list(obj, key)
    if not all(isinstance(item, dict) for item in items):
        raise WireError(f"{key} is not a list of objects")
    return items


def _strings(obj: dict, key: str) -> list[str]:
    items = _list(obj, key)
    if not all(isinstance(item, str) and len(item) <= MAX_STR for item in items):
        raise WireError(f"{key} is not a list of strings of at most {MAX_STR} characters")
    return items


def _str_dict(obj: dict, key: str) -> dict[str, str]:
    value = obj.get(key, {})
    if (
        not isinstance(value, dict)
        or len(value) > MAX_DICT
        or not all(
            isinstance(k, str) and isinstance(v, str) and 0 < len(k) <= MAX_STR and len(v) <= MAX_STR
            for k, v in value.items()
        )
    ):
        raise WireError(f"{key} is not an object of at most {MAX_DICT} short string pairs")
    return dict(value)
