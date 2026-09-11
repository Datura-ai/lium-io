"""Warm container pool — the pure half (flag `WARM_POOL_ENABLED`, default off).

A *slot* is a container this validator created on an idle executor and never started:
`warm_<uuid>`, the executor's pre-pulled template image, a fresh sparse volume `volume_<uuid>`
mounted where a rental mounts its own, all GPUs as device requests, a fixed block of the
executor's ports, and the rental HostConfig (`_build_rental_container_run_spec`). No keys, no
renter env, no command override: the image's own entrypoint, exactly what a rental of the image
gets. A stopped container holds no GPU, CPU or memory, so fillers run as before.

A whole-host rental of that image *adopts* the slot instead of creating a volume and running a
fresh container: one host command lists the slots and the image (`find_slots_command`), the
rental's own volume sizing bounds the slot from below (`slot_disk_sizes_fit`, the request cap from
above), :func:`slot_matches` proves the slot is what the rental would create right now, and one host
command renames it to the pod's name, applies the rental's CPU/memory limits and starts it
(`adopt_command`). Anything else — no slot, a slot that differs in any field, a failed rename or
start — is a *miss*, logged with its reason, and the rental takes the path that exists today.

Trust: the validator created the slot, but the miner owns the daemon in between. Adoption
therefore never trusts the slot's labels alone; it compares the live `docker inspect` output
against the spec it would use now (image id, mounts and tmpfs, ports, devices, GPU requests, runtime,
network, namespaces, capabilities, sysctls, ulimits, cgroup limits, restart policy, storage-opt, env,
cmd, entrypoint) and requires
`State.Status == created` with a zero `StartedAt` — a container that ever ran is not a slot. The
slot's volume is inspected as well (`volume_mismatch`): the size the rental is granted is the one
the volume plugin recorded, never the label alone.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from payload_models.payloads import (
    ContainerCreateRequest,
    CustomOptions,
    PayloadPortMapping,
    WorkloadKind,
)
from services.const import WARM_CONTAINER_PREFIX
from services.rental_docker_sdk import RENTAL_NETWORK_ICC_OPTION, ContainerRunSpec, GpuDeviceRequest

WARM_POOL_LABEL = "lium.warm_pool"
WARM_POOL_CREATED_AT_LABEL = "lium.warm_pool.created_at"
WARM_POOL_VOLUME_LIMIT_LABEL = "lium.warm_pool.volume_limit_gb"
WARM_POOL_STORAGE_LIMIT_LABEL = "lium.warm_pool.storage_limit_gb"
# Docker's zero time: what `State.StartedAt` reads on a container that was never started.
_NEVER_STARTED = "0001-01-01T00:00:00Z"
_FIND_SEPARATOR = "__LIUM_WARM_POOL__"
# Cap on the slot-inspect part of `find_slots_command`'s output: a slot inspect is ~8 KB; room for a
# handful of slots, not for a host-sized document.
_SLOT_INSPECT_MAX_BYTES = 256 * 1024
# dockerd's default /dev/shm when the create request carries no ShmSize.
_DEFAULT_SHM_SIZE = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class WarmSlot:
    name: str
    volume_name: str
    image_id: str
    volume_limit_gb: int | None
    storage_limit_gb: int | None
    created_at: datetime | None
    inspect: dict


@dataclass(frozen=True, slots=True)
class WarmPoolAdoption:
    slot: WarmSlot
    port_maps: list[tuple[int, int, int]]
    image_doc: dict


@dataclass(frozen=True, slots=True)
class FindSlotsOutput:
    """What `find_slots_command` printed, parsed.

    `image_doc` is the image's inspect, None when the image is not on the host (or its inspect did
    not parse). `slot_docs` is the inspect of every created slot: `[]` when the host listed none
    (xargs -r prints nothing) and None when it printed something that is not a JSON list — cut by
    `_SLOT_INSPECT_MAX_BYTES`, or not JSON at all. None is not an empty pool: a caller that read it
    as "no slots" would add one slot per filler start to a host with too many slot documents."""

    image_doc: dict | None
    slot_docs: list[dict] | None


def slot_name(slot_id: str) -> str:
    return f"{WARM_CONTAINER_PREFIX}{slot_id}"


def slot_volume_name(slot_id: str) -> str:
    return f"volume_{slot_id}"


def slot_labels(
    *, volume_limit_gb: int | None, storage_limit_gb: int | None, now: datetime
) -> dict[str, str]:
    labels = {WARM_POOL_LABEL: "1", WARM_POOL_CREATED_AT_LABEL: now.astimezone(UTC).isoformat()}
    if volume_limit_gb is not None:
        labels[WARM_POOL_VOLUME_LIMIT_LABEL] = str(volume_limit_gb)
    if storage_limit_gb is not None:
        labels[WARM_POOL_STORAGE_LIMIT_LABEL] = str(storage_limit_gb)
    return labels


def adopt_block_reason(
    payload: ContainerCreateRequest,
    custom_options: CustomOptions,
    *,
    is_custom_build: bool,
    image_managed_jupyter: bool,
    wants_quote_socket: bool = False,
) -> str | None:
    """Why this rental cannot adopt any slot, decided from the request and the host class alone;
    None when it can.

    A slot is one fixed container: the image's own command and entrypoint, one local volume, all
    GPUs, no renter environment. A rental that needs anything else is created fresh.
    """
    if payload.workload_kind != WorkloadKind.CUSTOMER_RENTAL:
        return "not a customer rental"
    if is_custom_build:
        return "custom build"
    if wants_quote_socket:
        # the TDX quote-broker socket is a create-time bind mount a slot never carries
        return "cvm quote socket is create-time"
    if payload.local_volume:
        return "reboot reuses its volume"
    if payload.pod_mapping:
        return "reboot reuses its ports"
    if payload.bootstrap_restore is not None:
        # the backup is restored into the rental's volume before `docker run`; had that volume been a
        # slot's, a step-4 miss would remove the slot and the restored data with it
        return "restore writes the volume before create"
    if payload.external_volume_info is not None:
        return "external volume"
    if payload.cluster_membership is not None:
        return "cluster member"
    if payload.disk_share is None or payload.disk_share < 1.0:
        return "partial-host rental"
    if payload.storage_limit_gb is None:
        return "storage-opt unsupported on this host"
    if payload.enable_jupyter:
        return "jupyter port"
    if custom_options.startup_commands and custom_options.startup_commands.strip():
        return "startup_commands override"
    if custom_options.entrypoint and custom_options.entrypoint.strip():
        return "entrypoint override"
    if custom_options.environment:
        return "renter environment is create-time"
    if custom_options.volumes and [v.split(":")[-1] for v in custom_options.volumes] != ["/root"]:
        return "non-default volume path"
    if custom_options.shm_size:
        return "shm_size override"
    if custom_options.internal_ports or custom_options.initial_port_count:
        return "template publishes its own ports"
    if image_managed_jupyter:
        return "image-managed jupyter needs a create-time token"
    return None


def find_slots_command(image: str) -> str:
    """One host command: the image's inspect, a separator, then the inspect of every created slot.

    The host decides how many labelled containers exist, so the slot part is capped; a truncated
    document does not parse and is reported as unreadable (None), never as "no slots"."""
    return (
        f"/usr/bin/docker image inspect --format '{{{{json .}}}}' {shlex.quote(image)}; "
        f"echo {_FIND_SEPARATOR}; "
        f"/usr/bin/docker ps -aq --filter label={WARM_POOL_LABEL}=1 --filter status=created "
        f"| xargs -r /usr/bin/docker inspect | head -c {_SLOT_INSPECT_MAX_BYTES}"
    )


def parse_find_slots_output(stdout: str) -> FindSlotsOutput:
    """The image inspect and the slot inspects from the output of `find_slots_command`; see
    `FindSlotsOutput` for what None means in each field."""
    head, sep, tail = (stdout or "").partition(_FIND_SEPARATOR)
    if not sep:
        return FindSlotsOutput(image_doc=None, slot_docs=None)
    image = _load_json(head.strip())
    image_doc = image if isinstance(image, dict) else None
    tail = tail.strip()
    if not tail:
        return FindSlotsOutput(image_doc=image_doc, slot_docs=[])
    slots = _load_json(tail)
    if not isinstance(slots, list):
        return FindSlotsOutput(image_doc=image_doc, slot_docs=None)
    return FindSlotsOutput(image_doc=image_doc, slot_docs=[s for s in slots if isinstance(s, dict)])


def slot_from_inspect(
    doc: dict, *, image_id: str, now: datetime, max_age: timedelta
) -> WarmSlot | None:
    """A slot the rental may consider, or None when the container is not a fresh slot of this image."""
    name = (doc.get("Name") or "").lstrip("/")
    labels = (doc.get("Config") or {}).get("Labels") or {}
    state = doc.get("State") or {}
    if not name.startswith(WARM_CONTAINER_PREFIX) or labels.get(WARM_POOL_LABEL) != "1":
        return None
    if (
        state.get("Status") != "created"
        or (state.get("StartedAt") or _NEVER_STARTED) != _NEVER_STARTED
    ):
        return None
    if doc.get("Image") != image_id:
        return None
    created_at = _parse_time(labels.get(WARM_POOL_CREATED_AT_LABEL))
    if created_at is None or now - created_at > max_age:
        return None
    volumes = [
        m.get("Name")
        for m in doc.get("Mounts") or []
        if m.get("Type") == "volume"
        and m.get("Name")
        and str(m.get("Driver") or "").startswith("vloopback")
    ]
    if len(volumes) != 1 or volumes[0] != slot_volume_name(slot_id_from_name(name)):
        # the slot's own never-used volume, and no other: a slot the daemon's owner pointed at a
        # previous renter's vloopback volume would hand that data to the next renter at /root
        return None
    return WarmSlot(
        name=name,
        volume_name=volumes[0],
        image_id=image_id,
        volume_limit_gb=_int_or_none(labels.get(WARM_POOL_VOLUME_LIMIT_LABEL)),
        storage_limit_gb=_int_or_none(labels.get(WARM_POOL_STORAGE_LIMIT_LABEL)),
        created_at=created_at,
        inspect=doc,
    )


def slot_port_maps(
    slot: WarmSlot,
    available_ports: list[PayloadPortMapping] | None,
    rental_port_maps: list[tuple[int, int, int]],
) -> list[tuple[int, int, int]] | None:
    """The rental's (docker_port, internal_port, external_port) triples from the slot's bindings —
    the same container ports `generate_portMappings` chose for this rental, each bound to a host port
    the backend offers it — or None."""
    by_internal = {p.internal_port: p.external_port for p in available_ports or []}
    maps: list[tuple[int, int, int]] = []
    for key, bindings in ((slot.inspect.get("HostConfig") or {}).get("PortBindings") or {}).items():
        docker_port = int(key.split("/")[0])
        if len(bindings or []) != 1:
            # a rental binds each container port to exactly one host port
            return None
        for binding in bindings or []:
            internal = int(binding.get("HostPort") or 0)
            if internal not in by_internal:
                return None
            maps.append((docker_port, internal, by_internal[internal]))
    if not maps or {d for d, _, _ in maps} != {d for d, _, _ in rental_port_maps}:
        return None
    return sorted(maps)


# `slot_disk_sizes_fit` reasons that mean the slot is smaller than the host sizes a rental now — the
# caller removes such a slot, where one that is merely larger than this rental's cap is kept for the next.
SLOT_VOLUME_BELOW_SIZING = "slot volume smaller than the rental's sizing"
SLOT_STORAGE_BELOW_SIZING = "slot storage-opt smaller than the rental's sizing"
SLOT_BELOW_SIZING_REASONS = frozenset({SLOT_VOLUME_BELOW_SIZING, SLOT_STORAGE_BELOW_SIZING})


def slot_disk_sizes_fit(
    slot: WarmSlot,
    payload: ContainerCreateRequest,
    *,
    sized_volume_gb: int | None,
    sized_storage_gb: int | None,
) -> str | None:
    """Why the slot's fixed volume and storage-opt sizes are not what this rental may be given now;
    None when they fit.

    The floor is the sizing the rental computes now (`resolve_volume_sizing` against the host as it
    is at rent time, `sized_*`): a slot sized while a filler's data held the disk is smaller than
    that, and adopting it would hand the renter less than a fresh create would. The ceiling is the
    backend's request cap: the sizing caps the slice at 1.5x the request and hands 2/3 of it to the
    volume and 1/3 to storage-opt, so volume <= request and storage <= request / 2. The two are not
    the same number because the slot was sized without a request cap and the disk drifts by a few GB
    between its create and the rental."""
    if slot.volume_limit_gb is None or slot.storage_limit_gb is None:
        return "slot has no recorded sizes"
    if payload.volume_limit_gb is not None:
        if slot.volume_limit_gb > payload.volume_limit_gb:
            return "slot volume larger than the request cap"
        if slot.storage_limit_gb > max(payload.volume_limit_gb // 2, 1):
            return "slot storage-opt larger than the request cap"
    if sized_volume_gb is not None and slot.volume_limit_gb < sized_volume_gb:
        return SLOT_VOLUME_BELOW_SIZING
    if sized_storage_gb is not None and slot.storage_limit_gb < sized_storage_gb:
        return SLOT_STORAGE_BELOW_SIZING
    return None


def inspect_volume_command(volume_name: str) -> str:
    """Driver, declared size, sparse flag and the plugin's own size record of one volume."""
    return (
        f"/usr/bin/docker volume inspect {shlex.quote(volume_name)} --format "
        "'{{.Driver}}|{{index .Options \"size\"}}|{{index .Options \"sparse\"}}|{{index .Status \"size-max\"}}'"
    )


def volume_mismatch(slot: WarmSlot, inspect_output: str) -> str | None:
    """Why the slot's live volume is not the one its labels describe; None when it is.

    The size labels are on a container the miner's daemon holds, so the rental's volume limit is
    proved against the volume plugin's own record before adoption: a vloopback volume, sparse,
    whose declared size is the labelled one."""
    lines = [line for line in (inspect_output or "").splitlines() if line.strip()]
    if len(lines) != 1:
        return "volume not inspectable"
    driver, size_option, sparse, size_max = (lines[0].strip().split("|") + ["", "", "", ""])[:4]
    if not driver.startswith("vloopback"):
        return "volume driver"
    declared_gb = _size_to_gb(size_option)
    if declared_gb is None:
        declared_gb = _size_to_gb(size_max)
    if declared_gb is None or declared_gb != slot.volume_limit_gb:
        return "volume size"
    if sparse != "true":
        return "volume not sparse"
    return None


def inspect_network_command(network_name: str) -> str:
    """Driver and inter-container-traffic option of one docker network."""
    return (
        f"/usr/bin/docker network inspect {shlex.quote(network_name)} --format "
        f"'{{{{.Driver}}}}|{{{{index .Options \"{RENTAL_NETWORK_ICC_OPTION}\"}}}}'"
    )


def network_mismatch(inspect_output: str) -> str | None:
    """Why the rental network a slot sits on is not the ICC-off bridge the rental requires; None when
    it is. A `docker create` proves this through `_ensure_rental_network_sync` (DAH-3199); a slot was
    created hours ago and holds no endpoint until it starts, so the network could have been removed
    and recreated with traffic on in between — adoption re-reads it, as the create would."""
    lines = [line for line in (inspect_output or "").splitlines() if line.strip()]
    if len(lines) != 1:
        return "network not inspectable"
    driver, icc = (lines[0].strip().split("|") + ["", ""])[:2]
    if driver != "bridge" or icc != "false":
        return "network not an ICC-off bridge"
    return None


def slot_matches(slot: WarmSlot, spec: ContainerRunSpec, image_doc: dict) -> str | None:
    """Why the live slot differs from the container the rental would create now; None when equal.

    Every HostConfig field `_build_rental_container_run_spec` sets is compared; CPU and memory are
    not (they are applied at adoption with `docker update`) and must be unset on the slot.
    """
    doc = slot.inspect
    host = doc.get("HostConfig") or {}
    config = doc.get("Config") or {}
    image_config = image_doc.get("Config") or {}

    if set(host.get("Binds") or []) != {
        f"{v.source}:{v.target}:{'ro' if v.read_only else 'rw'}" for v in spec.volumes
    }:
        return "binds"
    # An image `VOLUME` line gives every container of the image an anonymous mount at that path,
    # the slot and a fresh rental alike; the rental's own volumes are the rest.
    if set(m.get("Destination") for m in doc.get("Mounts") or []) != {
        v.target for v in spec.volumes
    } | set((image_config.get("Volumes") or {}).keys()):
        return "mounts"
    if _port_bindings(host) != {
        f"{p.container_port}/{p.protocol}": p.host_port for p in spec.ports
    }:
        return "ports"
    if set(_inspected_device_keys(host)) != {
        f"{d.path_on_host}:{d.path_in_container or d.path_on_host}:{d.permissions}"
        for d in spec.devices
    }:
        return "devices"
    if _device_requests(host) != _expected_device_requests(spec.device_requests):
        return "gpu device requests"
    if (host.get("Runtime") or "runc") != (spec.runtime or "runc"):
        return "runtime"
    if set(host.get("CapAdd") or []) != set(spec.cap_add):
        return "capabilities"
    if (host.get("Sysctls") or {}) != (spec.sysctls or {}):
        return "sysctls"
    if {(u.get("Name"), u.get("Soft"), u.get("Hard")) for u in host.get("Ulimits") or []} != {
        (u.name, u.soft, u.hard) for u in spec.ulimits
    }:
        return "ulimits"
    if ((host.get("RestartPolicy") or {}).get("Name") or "no") != (spec.restart_policy or "no"):
        return "restart policy"
    if (host.get("StorageOpt") or {}) != (
        {"size": f"{spec.storage_limit_gb}g"} if spec.storage_limit_gb else {}
    ):
        return "storage-opt"
    if host.get("NanoCpus") or host.get("Memory"):
        return "slot carries cpu/memory limits"
    if spec.shm_size or (host.get("ShmSize") or _DEFAULT_SHM_SIZE) != _DEFAULT_SHM_SIZE:
        return "shm_size"
    # dockerd merges the request env over the image env by key, so a key set by both (the NVIDIA
    # images ship NVIDIA_DRIVER_CAPABILITIES) appears once, with the request's value.
    if _env_dict(config.get("Env")) != {**_env_dict(image_config.get("Env")), **spec.environment}:
        return "environment"
    if (config.get("Cmd") or None) != (image_config.get("Cmd") or None) or spec.command:
        return "command"
    if (config.get("Entrypoint") or None) != (
        image_config.get("Entrypoint") or None
    ) or spec.entrypoint:
        return "entrypoint"
    if (config.get("Image") or "") != spec.image:
        return "image reference"
    # Fields the rental spec never sets must be at dockerd's defaults: a container re-created on the
    # host with any of them changed is not the container the rental would create.
    if host.get("Privileged"):
        return "privileged"
    if (host.get("PidMode") or "") or (host.get("IpcMode") or "private") not in ("", "private", "shareable"):
        return "namespace mode"
    if host.get("UsernsMode") or host.get("UTSMode"):
        return "namespace mode"
    # `--tmpfs` is recorded only in HostConfig.Tmpfs — it never appears in `.Mounts` — so a slot
    # with a tmpfs over the renter's data path would pass the mount checks above; `--mount` lands
    # in HostConfig.Mounts (the spec mounts through Binds only).
    if host.get("Tmpfs") or host.get("Mounts"):
        return "tmpfs or mount"
    # The slot sits on the network the rental would run on — the ICC-off `lium-rentals` bridge
    # (DAH-3199, `spec.network`); a slot on docker0 or `host` would put the pod back on the network
    # that bridge exists to end. A spec without a network expects dockerd's default bridge.
    network_mode = host.get("NetworkMode") or "default"
    expected_network = spec.network or "default"
    if network_mode != expected_network and not (
        expected_network == "default" and network_mode == "bridge"
    ):
        return "network"
    if any(
        host.get(field)
        for field in (
            "SecurityOpt",
            "CapDrop",
            "DeviceCgroupRules",
            "VolumesFrom",
            "GroupAdd",
            "Links",
            "ExtraHosts",
            "PublishAllPorts",
            "ReadonlyRootfs",
            "Init",
            "CgroupParent",
            "Dns",
            "DnsOptions",
            "DnsSearch",
            "OomScoreAdj",
            "AutoRemove",
            "VolumeDriver",
            "Cgroup",
            # OCI annotations go straight to the runtime (runc reads `org.systemd.property.*` as
            # cgroup unit properties) and are omitted from the document when empty
            "Annotations",
            # cgroup limits the spec never sets and `docker update --cpus/--memory` at adoption
            # does not reset: a slot pinned to one core or capped on pids would throttle the renter
            "CpuShares",
            "CpuPeriod",
            "CpuQuota",
            "CpuRealtimePeriod",
            "CpuRealtimeRuntime",
            "CpusetCpus",
            "CpusetMems",
            "MemoryReservation",
            "MemorySwap",
            "KernelMemory",
            "PidsLimit",
            "BlkioWeight",
            "BlkioWeightDevice",
            "BlkioDeviceReadBps",
            "BlkioDeviceWriteBps",
            "BlkioDeviceReadIOps",
            "BlkioDeviceWriteIOps",
            "OomKillDisable",
        )
    ):
        return "extra host config"
    # `--memory-swappiness` is null by default and 0 is a real setting, so a truthiness test would
    # let a slot that pins swappiness pass; any value set means the slot can swap the renter's pod
    if host.get("MemorySwappiness") is not None:
        return "extra host config"
    if config.get("User"):
        return "user"
    # What runs inside the pod besides Cmd/Entrypoint: a healthcheck is a command dockerd runs in
    # the container on a timer, so it must be the image's own; the working directory and stop
    # signal likewise.
    if (config.get("Healthcheck") or None) != (image_config.get("Healthcheck") or None):
        return "healthcheck"
    if (config.get("WorkingDir") or "") != (image_config.get("WorkingDir") or ""):
        return "working dir"
    if (config.get("StopSignal") or "") != (image_config.get("StopSignal") or ""):
        return "stop signal"
    if any(
        (binding.get("HostIp") or "") not in ("", "0.0.0.0")
        for bindings in (host.get("PortBindings") or {}).values()
        for binding in bindings or []
    ):
        return "port host ip"
    if any(m.get("Type") != "volume" or m.get("RW") is False for m in doc.get("Mounts") or []):
        return "mount type"
    return None


def adopt_command(
    slot: WarmSlot, pod_name: str, *, cpu_count: int | None, memory_gb: int | None
) -> str:
    """rename → apply the rental's CPU/memory limits → start, in one host command."""
    parts = [f"/usr/bin/docker rename {shlex.quote(slot.name)} {shlex.quote(pod_name)}"]
    update = []
    if cpu_count:
        update.append(f"--cpus {int(cpu_count)}")
    if memory_gb:
        # `_build_host_config_kwargs` sets mem_limit only, and dockerd then allows swap up to the
        # memory limit (MemorySwap = 2 x Memory); state the same explicitly.
        update.append(f"--memory {int(memory_gb)}g --memory-swap {2 * int(memory_gb)}g")
    if update:
        parts.append(
            f"/usr/bin/docker update {' '.join(update)} {shlex.quote(pod_name)} >/dev/null"
        )
    parts.append(f"/usr/bin/docker start {shlex.quote(pod_name)} >/dev/null")
    return " && ".join(parts)


def remove_slot_command(container_name: str, volume_name: str) -> str:
    """Force-remove a slot (under whichever name it carries now) and its never-used volume."""
    return (
        f"/usr/bin/docker rm -f {shlex.quote(container_name)} >/dev/null 2>&1; "
        f"/usr/bin/docker volume rm {shlex.quote(volume_name)} >/dev/null 2>&1; true"
    )


def slot_volumes_command(*, tag: str | None = None) -> str:
    """The volume name of every created slot on the host, one per line — `resolve_volume_sizing`
    leaves them out of the declared-size sum, since a sparse slot volume holds no bytes yet.
    With `tag`, each line is `<tag>\\t<name>`, so the listing can be one section of a larger
    probe command (the volume host probe, lium-io#1332) whose parser reads tagged records."""
    # `docker inspect --format` parses the template as-is (no `\t` → tab pass, unlike `docker ps`),
    # so the tab is a template action, as the newline already is
    prefix = f'{tag}{{{{"\\t"}}}}' if tag else ""
    return (
        f"/usr/bin/docker ps -aq --filter label={WARM_POOL_LABEL}=1 --filter status=created "
        "| xargs -r /usr/bin/docker inspect --format "
        f'\'{{{{range .Mounts}}}}{{{{if eq .Type "volume"}}}}{prefix}{{{{.Name}}}}{{{{"\\n"}}}}{{{{end}}}}{{{{end}}}}\''
    )


def list_slots_command() -> str:
    """Every slot on the host with its age label, one `name<TAB>created_at` per line."""
    return (
        f"/usr/bin/docker ps -a --filter label={WARM_POOL_LABEL}=1 "
        f"--format '{{{{.Names}}}}\t{{{{.Label \"{WARM_POOL_CREATED_AT_LABEL}\"}}}}'"
    )


def stale_slots(listing: str, *, now: datetime, max_age: timedelta) -> list[str]:
    """Slot names from `list_slots_command` output that are past `max_age` or unreadable."""
    stale: list[str] = []
    for line in (listing or "").splitlines():
        name, _, created = line.partition("\t")
        name = name.strip()
        if not name.startswith(WARM_CONTAINER_PREFIX):
            continue
        created_at = _parse_time(created.strip())
        if created_at is None or now - created_at > max_age:
            stale.append(name)
    return stale


def slot_id_from_name(name: str) -> str:
    return name.removeprefix(WARM_CONTAINER_PREFIX)


def _env_dict(env: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in env or []:
        key, _, value = str(item).partition("=")
        out[key] = value
    return out


def _port_bindings(host: dict) -> dict[str, int]:
    out: dict[str, int] = {}
    for key, bindings in (host.get("PortBindings") or {}).items():
        for binding in bindings or []:
            out[key] = int(binding.get("HostPort") or 0)
    return out


def _inspected_device_keys(host: dict) -> list[str]:
    """`host:container:permissions` per device in a HostConfig — the key `slot_matches` compares with
    the rental spec's devices."""
    return [
        f"{d.get('PathOnHost')}:{d.get('PathInContainer') or d.get('PathOnHost')}:{d.get('CgroupPermissions') or 'rwm'}"
        for d in host.get("Devices") or []
    ]


def _device_requests(host: dict) -> list[tuple[int, tuple[str, ...], tuple[tuple[str, ...], ...]]]:
    out = []
    for req in host.get("DeviceRequests") or []:
        caps = tuple(tuple(sorted(group)) for group in req.get("Capabilities") or [])
        out.append((int(req.get("Count") or 0), tuple(sorted(req.get("DeviceIDs") or [])), caps))
    return sorted(out)


def _expected_device_requests(
    requests: tuple[GpuDeviceRequest, ...],
) -> list[tuple[int, tuple[str, ...], tuple[tuple[str, ...], ...]]]:
    out = []
    for req in requests:
        # docker-py sends count=-1 when device ids are given; docker reports Count 0 for id lists.
        count = 0 if req.device_ids else int(req.count or 0)
        out.append(
            (
                count,
                tuple(sorted(req.device_ids)),
                tuple(tuple(sorted(g)) for g in req.capabilities),
            )
        )
    return sorted(out)


def _load_json(text: str):
    if not text:
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _int_or_none(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _size_to_gb(value: str) -> int | None:
    """Whole gigabytes from a vloopback size: the `size` option (`40g`, `40G`, `40gb`) or the
    plugin's `size-max` byte count; None for anything else or a size that is not whole GB."""
    text = (value or "").strip().lower()
    if not text:
        return None
    if text.isdigit():
        size_bytes = int(text)
        return size_bytes // 1024**3 if size_bytes % 1024**3 == 0 else None
    number = text.removesuffix("gb").removesuffix("g")
    return int(number) if number != text and number.isdigit() else None
