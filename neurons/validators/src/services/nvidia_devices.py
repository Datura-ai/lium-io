"""Host device-node discovery for `docker run --device` flags.

Named for NVIDIA because that is why it was written; it now also resolves the RDMA verbs nodes a
rental container needs to use an InfiniBand or RoCE card (DAH-2571).

Why this module exists
----------------------
`docker run --gpus ...` installs a transient device cgroup via the
nvidia-container-runtime hook. On hosts running cgroup v2 with the systemd
cgroup driver, `systemctl daemon-reload` (triggered by routine apt upgrades)
re-evaluates the docker-<id>.scope and overwrites its device program with one
that doesn't know about NVIDIA — `open(/dev/nvidia*)` returns EPERM and NVML
reports "Unknown Error" inside the container. /dev nodes stay visible but
unreadable.

Forwarding the same nodes via explicit `--device /dev/nvidia*` puts them into
HostConfig.Devices, so Docker reapplies the cgroup policy after the scope is
reconfigured. The set of nodes is host-specific (driver version, GPU topology,
MIG, NVSwitch, IMEX) — we probe at runtime instead of hardcoding.

Public surface
--------------
    flags = await build_gpu_flags(ssh_client, gpu_uuids)
        # full ready-to-paste string, e.g.
        # --gpus all --device=/dev/nvidia0 --device=/dev/nvidiactl ...

Failure handling
----------------
`build_gpu_flags` never raises: any probe failure (SSH error, missing
nvidia-smi, unknown UUID, etc.) is logged at WARNING and the function falls
back to the legacy `--gpus`-only string. The pod still gets created — it
just doesn't get the daemon-reload protection. This trades a known
regression (back to today's behaviour) for resilience against unforeseen
executor-side issues.
"""
from __future__ import annotations

import logging
import shlex
import xml.etree.ElementTree as ET
from collections.abc import Sequence

import asyncssh

from core.config import settings
from services.rental_docker_sdk import GpuDockerConfig, build_gpu_docker_config

logger = logging.getLogger(__name__)

_PROC_GPU_INFO_CMD = (
    "for f in /proc/driver/nvidia/gpus/*/information; do "
    '[ -r "$f" ] || continue; '
    "awk -F: '"
    '$1 == "GPU UUID" { uuid = $2 } '
    '$1 == "Device Minor" { minor = $2 } '
    "END { "
    'gsub(/^[[:space:]]+|[[:space:]]+$/, "", uuid); '
    'gsub(/^[[:space:]]+|[[:space:]]+$/, "", minor); '
    'if (uuid != "" && minor != "") printf "%s, %s\\n", uuid, minor; '
    "}' \"$f\"; "
    "done 2>/dev/null"
)


async def build_gpu_flags(
    ssh_client: asyncssh.SSHClientConnection,
    gpu_uuids: Sequence[str] | None,
) -> str:
    """Assemble the full GPU flag block for `docker run`.

    Combines two layers:
    - `--gpus` (or `--gpus '"device=<uuid>,..."'` for partial rentals): triggers
      the nvidia-container-runtime hook which bind-mounts userspace libs and the
      `nvidia-smi` binary into the container.
    - `--device /dev/nvidia*`: persists the device cgroup across systemd
      `daemon-reload` and `systemctl restart containerd`.

    Falls back to a legacy `--gpus`-only string on any probe failure (logged
    at WARNING). The pod will still be created in the legacy path; it just
    won't survive `systemctl daemon-reload` on the executor host.
    """
    gpu_config = await build_gpu_docker_config_for_executor(ssh_client, gpu_uuids)
    device_flags = _device_flags(
        tuple(device.path_on_host for device in gpu_config.device_mounts)
    )
    return " ".join(flag for flag in (_gpus_flag(gpu_uuids), device_flags) if flag)


async def build_gpu_docker_config_for_executor(
    ssh_client: asyncssh.SSHClientConnection,
    gpu_uuids: Sequence[str] | None,
) -> GpuDockerConfig:
    """Resolve structured GPU Docker options for SDK container creation."""
    try:
        if gpu_uuids:
            per_gpu, host_total = await _query_gpu_nodes_for_uuids(ssh_client, gpu_uuids)
            is_partial_rental = len(per_gpu) < host_total
        else:
            per_gpu = await _query_all_gpu_nodes(ssh_client)
            is_partial_rental = False

        # On partial rentals (some-but-not-all GPUs on the host), withhold every host-wide node:
        # /dev/nvidia-caps/* are per-GPU/per-MIG control nodes that would let a tenant peek at or
        # manipulate another tenant's GPU, and the RDMA verbs devices belong to cards the other
        # tenant may be renting (DAH-2571). We don't sell MIG slices today, but stripping both
        # under partial rental closes the leak before either ever ships.
        shared = await _query_shared_nodes(ssh_client, is_whole_host_rental=not is_partial_rental)
        return build_gpu_docker_config(gpu_uuids, device_nodes=(*per_gpu, *shared))
    except Exception:
        logger.warning(
            "nvidia_devices: probe failed, falling back to legacy --gpus only "
            "(pod will not survive systemd daemon-reload on the executor)",
            exc_info=True,
            extra={"gpu_uuids": list(gpu_uuids) if gpu_uuids else None},
        )
        return build_gpu_docker_config(gpu_uuids)


def _gpus_flag(gpu_uuids: Sequence[str] | None) -> str:
    if gpu_uuids:
        device_arg = f'"device={",".join(gpu_uuids)}"'
        return f"--gpus {shlex.quote(device_arg)}"
    return "--gpus all"


def _device_flags(nodes: Sequence[str]) -> str:
    return " ".join(f"--device={node}" for node in nodes)


async def _query_all_gpu_nodes(ssh: asyncssh.SSHClientConnection) -> tuple[str, ...]:
    res = await ssh.run("ls -1d /dev/nvidia[0-9]* 2>/dev/null || true")
    return _stdout_lines(res.stdout)


async def _query_gpu_nodes_for_uuids(
    ssh: asyncssh.SSHClientConnection,
    gpu_uuids: Sequence[str],
) -> tuple[tuple[str, ...], int]:
    """Resolve requested UUIDs to /dev/nvidiaN nodes, plus return host GPU count.

    Returns (per_gpu_nodes_in_request_order, host_total_gpu_count). The host
    total lets the caller decide whether this is a partial-host rental.
    """
    errors: list[str] = []
    try:
        uuid_to_minor = await _query_gpu_minor_map_from_proc(ssh)
    except RuntimeError as exc:
        errors.append(str(exc))
        uuid_to_minor = {}

    if not uuid_to_minor or _missing_gpu_uuids(gpu_uuids, uuid_to_minor):
        try:
            xml_uuid_to_minor = await _query_gpu_minor_map_from_nvidia_smi_xml(ssh)
        except RuntimeError as exc:
            errors.append(str(exc))
        else:
            if xml_uuid_to_minor:
                uuid_to_minor = xml_uuid_to_minor

    if not uuid_to_minor:
        details = f": {'; '.join(errors)}" if errors else ""
        raise RuntimeError(f"GPU minor discovery returned no UUID/minor rows{details}")

    missing = _missing_gpu_uuids(gpu_uuids, uuid_to_minor)
    if missing:
        raise RuntimeError(
            f"GPU {missing[0]!r} requested by tenant not present on executor; "
            f"visible: {sorted(uuid_to_minor)}"
        )

    # Deduplicated, request order kept: the caller compares this length against the host GPU count
    # to decide whether the rental covers the whole host, so a UUID repeated N times would pass a
    # single-GPU rental off as whole-host and hand it every host-wide device.
    per_gpu = tuple(dict.fromkeys(f"/dev/nvidia{uuid_to_minor[uuid]}" for uuid in gpu_uuids))
    return per_gpu, len(uuid_to_minor)


def _missing_gpu_uuids(gpu_uuids: Sequence[str], uuid_to_minor: dict[str, int]) -> list[str]:
    return [uuid for uuid in gpu_uuids if uuid not in uuid_to_minor]


async def _query_gpu_minor_map_from_proc(
    ssh: asyncssh.SSHClientConnection,
) -> dict[str, int]:
    res = await ssh.run(_PROC_GPU_INFO_CMD)
    if res.exit_status != 0:
        raise RuntimeError(
            "NVIDIA /proc GPU minor query failed on executor: "
            f"exit_status={res.exit_status}, stdout={res.stdout!r}, stderr={res.stderr!r}"
        )
    return _parse_uuid_minor_csv(res.stdout)


async def _query_gpu_minor_map_from_nvidia_smi_xml(
    ssh: asyncssh.SSHClientConnection,
) -> dict[str, int]:
    res = await ssh.run("nvidia-smi -q -x")
    if res.exit_status != 0:
        raise RuntimeError(
            "nvidia-smi XML query failed on executor: "
            f"exit_status={res.exit_status}, stdout={res.stdout!r}, stderr={res.stderr!r}"
        )
    return _parse_nvidia_smi_xml_minor_map(res.stdout)


async def _query_shared_nodes(
    ssh: asyncssh.SSHClientConnection,
    *,
    is_whole_host_rental: bool = True,
) -> tuple[str, ...]:
    """Enumerate shared NVIDIA control nodes, and the RDMA verbs nodes, that exist on the host.

    On a partial-host rental this skips /dev/nvidia-caps/*, the IMEX channel nodes and every RDMA
    device — all three belong to the host as a whole, and forwarding them would hand a tenant
    control nodes of a GPU or a card another tenant is renting on the same box.

    A whole-host rental gets the verbs devices unconditionally. It rode a separate
    ENABLE_RDMA_DEVICE_PASSTHROUGH switch while RDMA from inside a container was unproven; that
    switch has been on in prod since DAH-2571 and a multi-node cluster rental (DAH-2620) cannot
    work without the devices at all, so the second rubber-stamp is gone and the only gate left is
    the whole-host rule below. Only the `uverbs*` nodes and `rdma_cm` are ever forwarded, never the
    /dev/infiniband directory:
    that also carries `issm*`, the subnet-manager interface, and `umad*`, raw MAD access. A renter
    holding `issm` can interfere with the fabric every other tenant on it depends on (DAH-2571).
    """
    globs = (
        "/dev/nvidiactl /dev/nvidia-modeset /dev/nvidia-uvm "
        "/dev/nvidia-uvm-tools /dev/nvidia-nvswitchctl "
        "/dev/nvidia-nvswitch[0-9]* /dev/nvidia-nvlink[0-9]*"
    )
    if is_whole_host_rental:
        globs += " /dev/infiniband/uverbs[0-9]* /dev/infiniband/rdma_cm"
    cmd = (
        f"for p in {globs}; do "
        '[ -e "$p" ] && printf "%s\\n" "$p"; '
        "done"
    )
    if is_whole_host_rental:
        cmd += (
            "; find /dev/nvidia-caps /dev/nvidia-caps-imex-channels "
            "-mindepth 1 -maxdepth 1 -print 2>/dev/null || true"
        )
    res = await ssh.run(cmd)
    return _stdout_lines(res.stdout)


def _stdout_lines(stdout: str) -> tuple[str, ...]:
    return tuple(line.strip() for line in stdout.splitlines() if line.strip())


def _parse_uuid_minor_csv(stdout: str) -> dict[str, int]:
    uuid_to_minor: dict[str, int] = {}
    for line in _stdout_lines(stdout):
        uuid, _, minor = line.partition(",")
        try:
            uuid_to_minor[uuid.strip()] = int(minor.strip())
        except ValueError:
            continue
    return uuid_to_minor


def _parse_nvidia_smi_xml_minor_map(stdout: str) -> dict[str, int]:
    try:
        root = ET.fromstring(stdout)
    except ET.ParseError as exc:
        raise RuntimeError("nvidia-smi XML output could not be parsed") from exc

    uuid_to_minor: dict[str, int] = {}
    for gpu in root.iter():
        if _xml_local_name(gpu.tag) != "gpu":
            continue

        uuid = _xml_child_text(gpu, "uuid")
        minor = _xml_child_text(gpu, "minor_number")
        if not uuid or not minor:
            continue
        try:
            uuid_to_minor[uuid.strip()] = int(minor.strip())
        except ValueError:
            continue
    return uuid_to_minor


def _xml_child_text(parent: ET.Element, child_name: str) -> str | None:
    for child in parent:
        if _xml_local_name(child.tag) == child_name:
            return child.text
    return None


def _xml_local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


if __name__ == "__main__":
    # Ad-hoc smoke run against a real GPU host. Connects, prints what
    # build_gpu_flags() would emit for whole-host and partial rentals.
    import asyncio

    HOST = "69.19.136.107"
    USER = "shadeform"

    async def _main() -> None:
        async with asyncssh.connect(HOST, username=USER, known_hosts=None) as ssh:
            print("=" * 60)
            print(f"host: {USER}@{HOST}")
            uuids_raw = await ssh.run(
                "nvidia-smi --query-gpu=uuid --format=csv,noheader"
            )
            visible = _stdout_lines(uuids_raw.stdout)
            print(f"visible GPUs: {visible}")
            print("=" * 60)

            print("\n[whole-host rental]")
            print(await build_gpu_flags(ssh, gpu_uuids=None))

            if visible:
                first = visible[0].rstrip(",")
                print(f"\n[partial rental: {first}]")
                print(await build_gpu_flags(ssh, gpu_uuids=[first]))

    asyncio.run(_main())
