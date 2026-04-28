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

    # Lower-level pieces (exposed for tests / future reuse):
    plan = await discover_nvidia_devices(ssh_client, gpu_uuids)  # NvidiaDevicePlan
    flags = plan.as_device_flags()                               # only --device=... part
"""
from __future__ import annotations

from dataclasses import dataclass

import asyncssh


@dataclass(frozen=True)
class NvidiaDevicePlan:
    """Set of /dev/nvidia* nodes resolved on a specific executor host.

    `per_gpu` — minor nodes for the GPUs being rented (whole host or subset).
    `shared`  — control / MIG / NVSwitch / IMEX nodes that exist on the host.
    """

    per_gpu: tuple[str, ...]
    shared: tuple[str, ...]

    def as_device_flags(self) -> str:
        nodes = (*self.per_gpu, *self.shared)
        return " ".join(f"--device={n}" for n in nodes)


async def build_gpu_flags(
    ssh_client: asyncssh.SSHClientConnection,
    gpu_uuids: list[str] | None,
) -> str:
    """Assemble the full GPU flag block for `docker run`.

    Combines two layers:
    - `--gpus` (or `--gpus '"device=<uuid>,..."'` for partial rentals): triggers
      the nvidia-container-runtime hook which bind-mounts userspace libs and the
      `nvidia-smi` binary into the container.
    - `--device /dev/nvidia*`: persists the device cgroup across systemd
      `daemon-reload` and `systemctl restart containerd`.
    """
    plan = await discover_nvidia_devices(ssh_client, gpu_uuids)
    return f"{_gpus_flag(gpu_uuids)} {plan.as_device_flags()}".strip()


async def discover_nvidia_devices(
    ssh_client: asyncssh.SSHClientConnection,
    gpu_uuids: list[str] | None,
) -> NvidiaDevicePlan:
    per_gpu = await _per_gpu_nodes(ssh_client, gpu_uuids)
    shared = await _shared_nodes(ssh_client)
    return NvidiaDevicePlan(per_gpu=tuple(per_gpu), shared=tuple(shared))


def _gpus_flag(gpu_uuids: list[str] | None) -> str:
    if gpu_uuids:
        return f'--gpus \'"device={",".join(gpu_uuids)}"\''
    return "--gpus all"


async def _per_gpu_nodes(
    ssh: asyncssh.SSHClientConnection,
    gpu_uuids: list[str] | None,
) -> list[str]:
    if not gpu_uuids:
        res = await ssh.run("ls -1d /dev/nvidia[0-9]* 2>/dev/null || true")
        return [line.strip() for line in res.stdout.splitlines() if line.strip()]

    res = await ssh.run("nvidia-smi --query-gpu=uuid,index --format=csv,noheader")
    if res.exit_status != 0:
        raise RuntimeError(f"nvidia-smi query failed on executor: {res.stderr!r}")

    uuid_to_idx: dict[str, int] = {}
    for line in res.stdout.splitlines():
        uuid, _, idx = line.partition(",")
        try:
            uuid_to_idx[uuid.strip()] = int(idx.strip())
        except ValueError:
            continue

    nodes: list[str] = []
    for uuid in gpu_uuids:
        if uuid not in uuid_to_idx:
            raise RuntimeError(
                f"GPU {uuid!r} requested by tenant not present on executor; "
                f"visible: {sorted(uuid_to_idx)}"
            )
        nodes.append(f"/dev/nvidia{uuid_to_idx[uuid]}")
    return nodes


async def _shared_nodes(ssh: asyncssh.SSHClientConnection) -> list[str]:
    # One round-trip enumerates every nvidia control / MIG / NVSwitch / IMEX node
    # that actually exists on this host.
    cmd = (
        "for p in /dev/nvidiactl /dev/nvidia-modeset /dev/nvidia-uvm "
        "/dev/nvidia-uvm-tools /dev/nvidia-nvswitchctl; do "
        '[ -e "$p" ] && echo "$p"; '
        "done; "
        "ls -1 /dev/nvidia-nvswitch[0-9]* /dev/nvidia-nvlink[0-9]* 2>/dev/null; "
        "find /dev/nvidia-caps /dev/nvidia-caps-imex-channels "
        "-mindepth 1 -maxdepth 1 2>/dev/null"
    )
    res = await ssh.run(cmd)
    return [line.strip() for line in res.stdout.splitlines() if line.strip()]
