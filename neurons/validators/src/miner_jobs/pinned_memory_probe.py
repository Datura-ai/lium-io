"""Pinned-host-memory copy probe, run on the executor by PinnedMemoryProbeCheck (`python -I -`, this file on stdin).

Standard library only. Every GPU gets a page-locked host buffer (cuMemAllocHost, what PyTorch's
pin_memory() and cudaHostAlloc hand out) filled with a random pattern, copied to the device and back
with the async copy engines on a stream (what `.to("cuda", non_blocking=True)` does), and the result is
compared byte for byte on the host. A card that cannot DMA into pinned guest memory, the TDX guest with
the GPU outside confidential-computing mode a renter hit on 13 Oct (F-1599: "the GPU does not see pinned
memory buffers"), returns wrong data, a CUDA error or never completes the copy: that is a fault. So
is a driver-start call (init, enumeration, device, context) answering with an error that names broken
hardware or a driver that cannot start (BROKEN_GPU_CODES): nvidia-smi lists such a card, yet no renter can run CUDA on it. Any other probe
that cannot start (no libcuda, a context or allocation the driver or the container's memlock limit
refuses) is an error, not a fault: the validator does not penalise what it could not measure.

The report also carries what the executor says about TDX and GPU confidential computing (the cpuinfo
`tdx_guest` flag, /dev/tdx_guest, `nvidia-smi conf-compute -f`) as evidence; the copy decides the verdict.

Prints one line `PINNED_MEMORY_PROBE_JSON: {...}`. Exit 0 ok, 1 fault, 2 could not run.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import multiprocessing
import os
import subprocess
import sys
import time

MB = 1024 * 1024
JSON_MARKER = "PINNED_MEMORY_PROBE_JSON:"
# A 64 MB copy takes a few milliseconds on PCIe; a card that has not finished it in this budget, setup
# included, is hung. The check's SSH cap sits above it.
WORKER_BUDGET_SECONDS = 30.0
COUNT_BUDGET_SECONDS = 20.0
NVIDIA_SMI_TIMEOUT_SECONDS = 10
MAX_MISMATCH_OFFSETS = 4
# CUresult codes that mean the GPU or its driver is broken, not that the probe could not measure:
# ECC and NVLink uncorrectable, hardware stack error, system not ready (fabric manager down), and
# unknown (a wedged card: cuInit fails while nvidia-smi still lists it).
BROKEN_GPU_CODES = frozenset({214, 220, 714, 802, 999})


class ProbeError(Exception):
    """The probe could not start on this device: reported as an error, not a fault."""


class CopyFault(Exception):
    """A copy call failed once the buffers existed, the data did not survive the round trip, or a call
    driver-start call answered with one of BROKEN_GPU_CODES."""


class Cuda:
    """The slice of the driver API the probe uses, through ctypes."""

    def __init__(self) -> None:
        self.lib = None
        for name in ("libcuda.so.1", "libcuda.so"):
            try:
                self.lib = ctypes.CDLL(name)
                break
            except OSError:
                continue
        if self.lib is None:
            raise ProbeError("libcuda.so.1 not found")
        lib = self.lib
        lib.cuGetErrorName.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        # ctypes defaults every argument to a C int: 64-bit pointers and size_t lengths must be declared
        lib.cuMemcpyHtoDAsync_v2.argtypes = [
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]
        lib.cuMemcpyDtoHAsync_v2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]
        lib.cuMemsetD8Async.argtypes = [
            ctypes.c_uint64,
            ctypes.c_ubyte,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]
        lib.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        lib.cuMemFree_v2.argtypes = [ctypes.c_uint64]
        lib.cuMemAllocHost_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        lib.cuMemFreeHost.argtypes = [ctypes.c_void_p]
        lib.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
        lib.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
        lib.cuStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
        lib.cuStreamSynchronize.argtypes = [ctypes.c_void_p]
        lib.cuDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]

    def error_name(self, code: int) -> str:
        name = ctypes.c_char_p()
        if self.lib.cuGetErrorName(code, ctypes.byref(name)) == 0 and name.value:
            return name.value.decode()
        return f"CUDA_ERROR_{code}"

    def call(
        self, fn: str, *args, raise_as_fault: bool = True, broken_gpu_is_fault: bool = False
    ) -> None:
        code = getattr(self.lib, fn)(*args)
        if code != 0:
            message = f"{fn} -> {self.error_name(code)} ({code})"
            if raise_as_fault or (broken_gpu_is_fault and code in BROKEN_GPU_CODES):
                raise CopyFault(message)
            raise ProbeError(message)


def probe_device(index: int, size_mb: int, on_phase=None) -> dict:
    """Round-trip a random pattern through pinned host memory and the device; raise CopyFault if it does not survive."""
    size = size_mb * MB
    cuda = Cuda()
    # setup: a refusal is an error, not a fault, unless a driver-start call
    # (init, device, context) names a broken GPU (BROKEN_GPU_CODES)
    cuda.call("cuInit", 0, raise_as_fault=False, broken_gpu_is_fault=True)
    device = ctypes.c_int()
    cuda.call(
        "cuDeviceGet", ctypes.byref(device), index, raise_as_fault=False, broken_gpu_is_fault=True
    )
    name = ctypes.create_string_buffer(128)
    cuda.call("cuDeviceGetName", name, 128, device, raise_as_fault=False)
    context = ctypes.c_void_p()
    cuda.call(
        "cuCtxCreate_v2",
        ctypes.byref(context),
        0,
        device,
        raise_as_fault=False,
        broken_gpu_is_fault=True,
    )
    stream = ctypes.c_void_p()
    cuda.call("cuStreamCreate", ctypes.byref(stream), 0, raise_as_fault=False)
    h_src = ctypes.c_void_p()
    h_dst = ctypes.c_void_p()
    cuda.call("cuMemAllocHost_v2", ctypes.byref(h_src), size, raise_as_fault=False)
    cuda.call("cuMemAllocHost_v2", ctypes.byref(h_dst), size, raise_as_fault=False)
    d_buf = ctypes.c_uint64()
    cuda.call("cuMemAlloc_v2", ctypes.byref(d_buf), size, raise_as_fault=False)
    pattern = os.urandom(size)
    ctypes.memmove(h_src, pattern, size)
    ctypes.memset(h_dst, 0, size)

    if on_phase:
        on_phase("copy")
    started = time.perf_counter()
    # the device buffer starts as a sentinel, so a DMA that never reads the host buffer cannot pass as a copy
    cuda.call("cuMemsetD8Async", d_buf.value, 0xA5, size, stream)
    cuda.call("cuMemcpyHtoDAsync_v2", d_buf.value, h_src, size, stream)
    cuda.call("cuMemcpyDtoHAsync_v2", h_dst, d_buf.value, size, stream)
    cuda.call("cuStreamSynchronize", stream)
    copy_ms = round((time.perf_counter() - started) * 1000, 2)

    got = ctypes.string_at(h_dst, size)
    report = {
        "index": index,
        "name": name.value.decode(errors="replace"),
        "size_mb": size_mb,
        "copy_ms": copy_ms,
    }
    if got != pattern:
        mismatched, offsets = _mismatches(pattern, got)
        if got.count(0) == size:
            what = "host buffer untouched (device-to-host copy wrote nothing)"
        elif got.count(0xA5) == size:
            what = "device buffer still the sentinel (host-to-device copy read nothing)"
        else:
            what = "data corrupted"
        raise CopyFault(
            f"pinned H2D/D2H round trip: {what}; {mismatched} of {size // MB} MB chunks differ, "
            f"first at byte offsets {offsets}"
        )
    cuda.call("cuMemFree_v2", d_buf.value)
    cuda.call("cuMemFreeHost", h_src)
    cuda.call("cuMemFreeHost", h_dst)
    cuda.call("cuCtxDestroy_v2", context)
    return report


def _mismatches(want: bytes, got: bytes) -> tuple[int, list[int]]:
    # compare in 1 MB chunks: the count says how much was lost, the offsets where
    chunks = 0
    offsets: list[int] = []
    for start in range(0, len(want), MB):
        if want[start : start + MB] != got[start : start + MB]:
            chunks += 1
            if len(offsets) < MAX_MISMATCH_OFFSETS:
                first = next(
                    i for i in range(start, min(start + MB, len(want))) if want[i] != got[i]
                )
                offsets.append(first)
    return chunks, offsets


def _quiet_child() -> None:
    # a forked worker inherits the SSH session's stdout/stderr; children talk to the parent over their pipe only
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.close(devnull)


def _worker(index: int, size_mb: int, conn) -> None:
    _quiet_child()

    def on_phase(phase: str) -> None:
        # the parent classifies a worker that never answers by the phase it reached: setup -> error, copy -> fault
        conn.send({"phase": phase})

    try:
        report = probe_device(index, size_mb, on_phase)
        report["status"] = "ok"
    except CopyFault as exc:
        report = {"index": index, "status": "fault", "error": str(exc)}
    except ProbeError as exc:
        report = {"index": index, "status": "error", "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - the parent must always get a verdict
        report = {"index": index, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
    conn.send(report)
    conn.close()


def _count_worker(conn) -> None:
    _quiet_child()
    try:
        cuda = Cuda()
        cuda.call("cuInit", 0, raise_as_fault=False, broken_gpu_is_fault=True)
        count = ctypes.c_int()
        cuda.call(
            "cuDeviceGetCount", ctypes.byref(count), raise_as_fault=False, broken_gpu_is_fault=True
        )
        conn.send({"count": count.value})
    except CopyFault as exc:
        conn.send({"fault": str(exc)})
    except Exception as exc:  # noqa: BLE001
        conn.send({"error": str(exc)})
    conn.close()


def _reap(process) -> None:
    process.join(1)
    if process.is_alive():
        process.kill()
        process.join(5)


def _device_count(mp) -> int:
    # in its own fork: a process that has called cuInit cannot fork CUDA-capable children
    parent_conn, child_conn = mp.Pipe(duplex=False)
    process = mp.Process(target=_count_worker, args=(child_conn,))
    process.start()
    child_conn.close()
    try:
        reply = (
            parent_conn.recv()
            if parent_conn.poll(COUNT_BUDGET_SECONDS)
            else {"error": "device enumeration hung"}
        )
    except EOFError:
        reply = {"error": "device enumeration worker died"}
    _reap(process)
    if "fault" in reply:
        raise CopyFault(reply["fault"])
    if "error" in reply:
        raise ProbeError(reply["error"])
    return reply["count"]


def drain_workers(workers: list, deadline: float) -> list[dict]:
    """Collect one verdict per worker by the shared deadline. A worker that dies or hangs in the copy is a
    fault; one that never got past setup is an error."""
    reports = []
    for index, process, conn in workers:
        phase = "setup"
        report = None
        while report is None:
            remaining = deadline - time.perf_counter()
            try:
                # past the deadline a verdict already in the pipe still counts: poll(0) reads it
                if not conn.poll(max(remaining, 0)):
                    break
                message = conn.recv()
            except EOFError:
                break
            if "phase" in message:
                phase = message["phase"]
            else:
                report = message
        if report is None:
            status = "fault" if phase == "copy" else "error"
            alive = process.is_alive()
            what = "hung" if alive else "died"
            report = {"index": index, "status": status, "error": f"worker {what} during {phase}"}
        _reap(process)
        reports.append(report)
    return reports


def tee_evidence() -> dict:
    """What the executor says about TDX and GPU confidential computing. Evidence only, never the verdict."""
    evidence: dict = {"dev_tdx_guest": os.path.exists("/dev/tdx_guest")}
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    evidence["cpuinfo_tdx_guest"] = "tdx_guest" in line.split()
                    break
    except OSError as exc:
        evidence["cpuinfo_error"] = str(exc)[:200]
    try:
        out = subprocess.run(
            ["nvidia-smi", "conf-compute", "-f"],
            capture_output=True,
            text=True,
            timeout=NVIDIA_SMI_TIMEOUT_SECONDS,
        )
        evidence["gpu_cc_mode"] = (out.stdout or out.stderr).strip()[:200]
    except (OSError, subprocess.SubprocessError) as exc:
        evidence["gpu_cc_mode_error"] = f"{type(exc).__name__}"[:200]
    return evidence


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--size-mb", type=int, default=64, help="pinned buffer per GPU (default 64)"
    )
    args = parser.parse_args(argv)

    started = time.perf_counter()
    result: dict = {"status": "ok", "devices": [], "faults": [], "tee": tee_evidence()}
    # every CUDA call happens in a forked worker: a crash or hang in the driver takes the worker, not the verdict
    mp = multiprocessing.get_context("fork")
    try:
        indices = list(range(_device_count(mp)))
        if not indices:
            raise ProbeError("no CUDA devices")
    except ProbeError as exc:
        result.update(
            status="error", error=str(exc), elapsed_s=round(time.perf_counter() - started, 2)
        )
        print(JSON_MARKER, json.dumps(result, sort_keys=True))
        return 2
    except CopyFault as exc:
        result.update(
            status="fault", faults=[str(exc)], elapsed_s=round(time.perf_counter() - started, 2)
        )
        print(JSON_MARKER, json.dumps(result, sort_keys=True))
        return 1

    workers = []
    for index in indices:
        parent_conn, child_conn = mp.Pipe(duplex=False)
        process = mp.Process(target=_worker, args=(index, args.size_mb, child_conn))
        process.start()
        child_conn.close()
        workers.append((index, process, parent_conn))
    result["devices"] = drain_workers(workers, time.perf_counter() + WORKER_BUDGET_SECONDS)

    for report in result["devices"]:
        if report["status"] == "fault":
            result["faults"].append(f"gpu {report['index']}: {report['error']}")
    errors = [r for r in result["devices"] if r["status"] == "error"]
    if result["faults"]:
        result["status"] = "fault"
    elif errors:
        result["status"] = "error"
        result["error"] = "; ".join(f"gpu {r['index']}: {r['error']}" for r in errors)
    result["elapsed_s"] = round(time.perf_counter() - started, 2)
    print(JSON_MARKER, json.dumps(result, sort_keys=True))
    return {"ok": 0, "fault": 1, "error": 2}[result["status"]]


if __name__ == "__main__":
    code = main(sys.argv[1:])
    # flush, then leave without the exit handlers so a child the driver still holds cannot keep the verdict
    # from the validator
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
