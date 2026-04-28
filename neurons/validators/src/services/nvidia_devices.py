"""NVIDIA device-node discovery for `docker run --device` flags.

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
from collections.abc import Sequence

import asyncssh

logger = logging.getLogger(__name__)


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
    try:
        if gpu_uuids:
            per_gpu, host_total = await _query_gpu_nodes_for_uuids(ssh_client, gpu_uuids)
            is_partial_rental = len(per_gpu) < host_total
        else:
            per_gpu = await _query_all_gpu_nodes(ssh_client)
            is_partial_rental = False

        # On partial rentals (some-but-not-all GPUs on the host), skip /dev/nvidia-caps/*.
        # Caps are per-GPU/per-MIG control nodes; forwarding all of them lets a tenant
        # peek at or manipulate MIG state of another tenant's GPU on the same host.
        # We don't sell MIG slices today, but stripping caps under partial rental
        # closes the leak before that ever ships.
        shared = await _query_shared_nodes(ssh_client, include_caps=not is_partial_rental)
        device_flags = _device_flags((*per_gpu, *shared))
        return " ".join(flag for flag in (_gpus_flag(gpu_uuids), device_flags) if flag)
    except Exception:
        logger.warning(
            "nvidia_devices: probe failed, falling back to legacy --gpus only "
            "(pod will not survive systemd daemon-reload on the executor)",
            exc_info=True,
            extra={"gpu_uuids": list(gpu_uuids) if gpu_uuids else None},
        )
        return _gpus_flag(gpu_uuids)


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
    # `minor_number` is the documented driver-assigned minor that maps to /dev/nvidiaN,
    # unlike `index` which is the CUDA enumeration order (PCI BDF) and is not formally
    # guaranteed to equal the device-node minor.
    res = await ssh.run("nvidia-smi --query-gpu=uuid,minor_number --format=csv,noheader")
    if res.exit_status != 0:
        raise RuntimeError(f"nvidia-smi query failed on executor: {res.stderr!r}")

    uuid_to_minor: dict[str, int] = {}
    for line in _stdout_lines(res.stdout):
        uuid, _, minor = line.partition(",")
        try:
            uuid_to_minor[uuid.strip()] = int(minor.strip())
        except ValueError:
            continue

    missing = [uuid for uuid in gpu_uuids if uuid not in uuid_to_minor]
    if missing:
        raise RuntimeError(
            f"GPU {missing[0]!r} requested by tenant not present on executor; "
            f"visible: {sorted(uuid_to_minor)}"
        )

    per_gpu = tuple(f"/dev/nvidia{uuid_to_minor[uuid]}" for uuid in gpu_uuids)
    return per_gpu, len(uuid_to_minor)


async def _query_shared_nodes(
    ssh: asyncssh.SSHClientConnection,
    *,
    include_caps: bool = True,
) -> tuple[str, ...]:
    """Enumerate shared NVIDIA control nodes that exist on the host.

    `include_caps=False` skips /dev/nvidia-caps/* and IMEX channel nodes — used
    for partial-host rentals so we don't leak per-GPU MIG/IMEX caps belonging
    to neighbouring tenants on the same host.
    """
    cmd = (
        "for p in /dev/nvidiactl /dev/nvidia-modeset /dev/nvidia-uvm "
        "/dev/nvidia-uvm-tools /dev/nvidia-nvswitchctl "
        "/dev/nvidia-nvswitch[0-9]* /dev/nvidia-nvlink[0-9]*; do "
        '[ -e "$p" ] && printf "%s\\n" "$p"; '
        "done"
    )
    if include_caps:
        cmd += (
            "; find /dev/nvidia-caps /dev/nvidia-caps-imex-channels "
            "-mindepth 1 -maxdepth 1 -print 2>/dev/null || true"
        )
    res = await ssh.run(cmd)
    return _stdout_lines(res.stdout)


def _stdout_lines(stdout: str) -> tuple[str, ...]:
    return tuple(line.strip() for line in stdout.splitlines() if line.strip())


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
