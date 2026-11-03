"""NVLink multicast probe, run on the executor by NvlinkFaultCheck (`python -I -`, this file on stdin).

Standard library only. On a board whose GPUs support NVLink multicast (HGX/NVSwitch: H100, H200, B200,
B300), it creates one multicast object over every GPU, adds each GPU and binds a small allocation from
each, which is what NCCL does to start NVLink SHARP (NVLS). After a sticky uncorrectable NVLink error the
links still show as active and every single-GPU check passes, yet the bind fails on every GPU with
CUDA_ERROR_ILLEGAL_STATE until the GPUs and the fabric are reset (an 8x B300, 23 Oct): that is a fault.
A board without multicast support, a driver without the multicast API, or any other error is
"unsupported" or "error": the validator does not penalise what it could not measure.

Prints one line `NVLINK_MULTICAST_PROBE_JSON: {...}`. Exit 0 ok/unsupported, 1 fault, 2 could not run.
"""

from __future__ import annotations

import ctypes
import json
import multiprocessing
import os
import sys

JSON_MARKER = "NVLINK_MULTICAST_PROBE_JSON:"
BUDGET_SECONDS = 45.0
CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED = 132
CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 1  # what NCCL requests for NVLS
TWO_MB = 2 * 1024 * 1024
# CUresult codes that mean the NVLink fabric is broken: NVLink uncorrectable, and illegal state (the NVLS
# bind after a sticky NVLink error). Every other code is "could not measure".
NVLINK_FAULT_CODES = frozenset({220, 401})


class McProp(ctypes.Structure):
    _fields_ = [
        ("numDevices", ctypes.c_uint),
        ("size", ctypes.c_size_t),
        ("handleTypes", ctypes.c_ulonglong),
        ("flags", ctypes.c_ulonglong),
    ]


class MemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class AllocFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]


class AllocProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", MemLocation),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", AllocFlags),
    ]


class Unsupported(Exception):
    pass


class Fault(Exception):
    pass


class ProbeError(Exception):
    pass


def probe(lib) -> dict:
    """Bind one multicast object across every GPU: {"status": "ok", ...}, or raise the verdict."""
    for name in ("cuMulticastCreate", "cuMulticastAddDevice", "cuMulticastBindMem", "cuMemCreate"):
        if not hasattr(lib, name):
            raise Unsupported(f"driver has no {name}")
    u64, size_t = ctypes.c_uint64, ctypes.c_size_t
    lib.cuMulticastAddDevice.argtypes = [u64, ctypes.c_int]
    lib.cuMulticastBindMem.argtypes = [u64, size_t, u64, size_t, size_t, u64]
    lib.cuMemCreate.argtypes = [ctypes.POINTER(u64), size_t, ctypes.POINTER(AllocProp), u64]

    def call(fn, *args, fault_codes=frozenset()):
        code = getattr(lib, fn)(*args)
        if code != 0:
            message = f"{fn} -> CUresult {code}"
            raise Fault(message) if code in fault_codes else ProbeError(message)

    call("cuInit", 0)
    count = ctypes.c_int()
    call("cuDeviceGetCount", ctypes.byref(count))
    if count.value < 2:
        raise Unsupported(f"{count.value} GPU(s)")
    devices = []
    for index in range(count.value):
        device, supported = ctypes.c_int(), ctypes.c_int()
        call("cuDeviceGet", ctypes.byref(device), index)
        call(
            "cuDeviceGetAttribute",
            ctypes.byref(supported),
            CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED,
            device,
        )
        if not supported.value:
            raise Unsupported(f"gpu {index} does not support multicast")
        devices.append(device.value)

    prop = McProp(len(devices), 0, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0)
    granularity = ctypes.c_size_t()
    call("cuMulticastGetGranularity", ctypes.byref(granularity), ctypes.byref(prop), 0)
    prop.size = max(granularity.value, TWO_MB)  # both powers of two: a multiple of each
    handle = ctypes.c_uint64()
    call(
        "cuMulticastCreate",
        ctypes.byref(handle),
        ctypes.byref(prop),
        fault_codes=NVLINK_FAULT_CODES,
    )
    for device in devices:
        call("cuMulticastAddDevice", handle.value, device, fault_codes=NVLINK_FAULT_CODES)
    for device in devices:
        context = ctypes.c_void_p()
        call("cuDevicePrimaryCtxRetain", ctypes.byref(context), device)
        call("cuCtxSetCurrent", context)
        alloc = AllocProp(type=1, requestedHandleTypes=CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)
        alloc.location = MemLocation(1, device)  # pinned device memory on this GPU
        memory = ctypes.c_uint64()
        call("cuMemCreate", ctypes.byref(memory), prop.size, ctypes.byref(alloc), 0)
        try:
            call(
                "cuMulticastBindMem",
                handle.value,
                0,
                memory.value,
                0,
                prop.size,
                0,
                fault_codes=NVLINK_FAULT_CODES,
            )
        except Fault as exc:
            raise Fault(f"gpu {device}: {exc}") from None
    return {"status": "ok", "gpus": len(devices), "size_mb": prop.size // (1024 * 1024)}


def _worker(conn) -> None:
    # quiet: a forked worker inherits the SSH session's stdout/stderr
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    try:
        lib = None
        for name in ("libcuda.so.1", "libcuda.so"):
            try:
                lib = ctypes.CDLL(name)
                break
            except OSError:
                continue
        if lib is None:
            raise ProbeError("libcuda.so.1 not found")
        report = probe(lib)
    except Unsupported as exc:
        report = {"status": "unsupported", "error": str(exc)}
    except Fault as exc:
        report = {"status": "fault", "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - the parent must always get a verdict
        report = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    conn.send(report)
    conn.close()


def main() -> int:
    # in a fork: a crash or hang in the driver takes the worker, not the verdict
    mp = multiprocessing.get_context("fork")
    parent_conn, child_conn = mp.Pipe(duplex=False)
    process = mp.Process(target=_worker, args=(child_conn,))
    process.start()
    child_conn.close()
    try:
        report = (
            parent_conn.recv()
            if parent_conn.poll(BUDGET_SECONDS)
            else {"status": "error", "error": "probe hung"}
        )
    except EOFError:
        report = {"status": "error", "error": "probe worker died"}
    if process.is_alive():
        process.kill()
    print(JSON_MARKER, json.dumps(report, sort_keys=True))
    return {"ok": 0, "unsupported": 0, "fault": 1}.get(report["status"], 2)


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
