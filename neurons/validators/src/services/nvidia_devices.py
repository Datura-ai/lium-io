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
"""
from __future__ import annotations

import shlex
from collections.abc import Sequence

import asyncssh


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
    """
    if gpu_uuids:
        per_gpu = await _query_gpu_nodes_for_uuids(ssh_client, gpu_uuids)
    else:
        per_gpu = await _query_all_gpu_nodes(ssh_client)

    shared = await _query_shared_nodes(ssh_client)
    device_flags = _device_flags((*per_gpu, *shared))
    return " ".join(flag for flag in (_gpus_flag(gpu_uuids), device_flags) if flag)


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
) -> tuple[str, ...]:
    res = await ssh.run("nvidia-smi --query-gpu=uuid,index --format=csv,noheader")
    if res.exit_status != 0:
        raise RuntimeError(f"nvidia-smi query failed on executor: {res.stderr!r}")

    uuid_to_index: dict[str, int] = {}
    for line in _stdout_lines(res.stdout):
        uuid, _, index = line.partition(",")
        try:
            uuid_to_index[uuid.strip()] = int(index.strip())
        except ValueError:
            continue

    missing = [uuid for uuid in gpu_uuids if uuid not in uuid_to_index]
    if missing:
        raise RuntimeError(
            f"GPU {missing[0]!r} requested by tenant not present on executor; "
            f"visible: {sorted(uuid_to_index)}"
        )

    return tuple(f"/dev/nvidia{uuid_to_index[uuid]}" for uuid in gpu_uuids)


async def _query_shared_nodes(ssh: asyncssh.SSHClientConnection) -> tuple[str, ...]:
    # One round-trip enumerates every nvidia control / MIG / NVSwitch / IMEX node
    # that actually exists on this host.
    cmd = (
        "for p in /dev/nvidiactl /dev/nvidia-modeset /dev/nvidia-uvm "
        "/dev/nvidia-uvm-tools /dev/nvidia-nvswitchctl "
        "/dev/nvidia-nvswitch[0-9]* /dev/nvidia-nvlink[0-9]*; do "
        '[ -e "$p" ] && printf "%s\\n" "$p"; '
        "done; "
        "find /dev/nvidia-caps /dev/nvidia-caps-imex-channels "
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
