#!/usr/bin/env python3
"""DAH-2620: the inner daemon's default runtime, so a nested container gets the fabric for free.

A renter who runs their job inside their own image should not have to know that RDMA needs
`--device /dev/infiniband`, `--cap-add IPC_LOCK` and an unlimited memlock. Docker can default a
ulimit and nothing else, so the devices and the capability are injected here: this wrapper is
registered as `default-runtime`, edits the OCI spec on `create`, and then hands off to runc.

Only `uverbs*` and `rdma_cm` are ever added. `issm*` is the subnet-manager interface and `umad*` is
raw MAD access — a tenant holding either can disturb a fabric every other tenant shares, which is
the same allowlist the validator applies when it forwards devices into the pod itself.
"""

import json
import os
import stat
import sys

VERBS_DIR = "/dev/infiniband"
RLIMIT_INFINITY = 0xFFFFFFFFFFFFFFFF


def _bundle_path(argv: list[str]) -> str | None:
    for index, argument in enumerate(argv):
        if argument in ("--bundle", "-b") and index + 1 < len(argv):
            return argv[index + 1]
    return None


def _verbs_devices() -> list[str]:
    if not os.path.isdir(VERBS_DIR):
        return []
    names = sorted(os.listdir(VERBS_DIR))
    return [
        f"{VERBS_DIR}/{name}"
        for name in names
        if name.startswith("uverbs") or name == "rdma_cm"
    ]


def _add_devices(spec: dict) -> None:
    linux = spec.setdefault("linux", {})
    devices = linux.setdefault("devices", [])
    cgroup_devices = linux.setdefault("resources", {}).setdefault("devices", [])
    already_there = {device.get("path") for device in devices}

    for host_path in _verbs_devices():
        if host_path in already_there:
            continue
        info = os.stat(host_path)
        major, minor = os.major(info.st_rdev), os.minor(info.st_rdev)
        devices.append(
            {
                "path": host_path,
                "type": "c",
                "major": major,
                "minor": minor,
                "fileMode": stat.S_IMODE(info.st_mode),
                "uid": info.st_uid,
                "gid": info.st_gid,
            }
        )
        cgroup_devices.append(
            {"allow": True, "type": "c", "major": major, "minor": minor, "access": "rwm"}
        )


def _add_memory_pinning(spec: dict) -> None:
    """Registering a memory region pins pages: the capability and the limit are needed together."""
    process = spec.setdefault("process", {})
    capabilities = process.setdefault("capabilities", {})
    for bucket in ("bounding", "effective", "permitted", "inheritable", "ambient"):
        held = capabilities.get(bucket)
        if held is not None and "CAP_IPC_LOCK" not in held:
            held.append("CAP_IPC_LOCK")

    rlimits = [
        limit for limit in process.get("rlimits", []) if limit.get("type") != "RLIMIT_MEMLOCK"
    ]
    rlimits.append(
        {"type": "RLIMIT_MEMLOCK", "hard": RLIMIT_INFINITY, "soft": RLIMIT_INFINITY}
    )
    process["rlimits"] = rlimits


def inject(spec: dict) -> dict:
    _add_devices(spec)
    _add_memory_pinning(spec)
    return spec


def main(argv: list[str]) -> None:
    bundle = _bundle_path(argv)
    if bundle:
        spec_path = os.path.join(bundle, "config.json")
        if os.path.isfile(spec_path):
            with open(spec_path) as handle:
                spec = json.load(handle)
            with open(spec_path, "w") as handle:
                json.dump(inject(spec), handle)

    os.execvp("runc", ["runc", *argv])


if __name__ == "__main__":
    main(sys.argv[1:])
