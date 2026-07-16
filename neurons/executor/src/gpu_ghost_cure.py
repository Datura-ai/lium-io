"""Ghost GPU cure: clear the Blackwell GSP "engine busy" utilization latch (DAH-2431).

A ghost GPU reads 100% utilization with ~0 MiB used and zero compute processes
after a process died while a CUDA kernel was executing. The latch is
telemetry-only (no performance impact for renters) but it makes idle Blackwell
GPUs look busy for days. It is cleared by opening and cleanly closing a CUDA
context on the affected GPU from a fresh process.

This module is self-contained (stdlib + libcuda.so via ctypes, no nvcc, no
pynvml): the trivial kernel ships as PTX and is JIT-compiled by the driver, so
the same file works on any GPU architecture and can also be copied to a host
and run standalone.

Usage:
- as a script: detects ghost GPUs via nvidia-smi and cures each one
  (each cure runs in a child process — the clean process exit is part of
  the verified cure recipe);
- as a subprocess payload: GHOST_CURE_UUID=<gpu-uuid> python3 gpu_ghost_cure.py
- from code (executor watchdog): detect_ghosts() + spawn_cure(uuid).

Env knobs (testing):
- GHOST_CURE_MODE=ctx|kernel  (default kernel; ctx = context create/destroy only)
- GHOST_CURE_SECONDS=2.0      (wall-clock of kernel activity)
"""

import ctypes
import logging
import os
import subprocess
import sys
import time

logger = logging.getLogger(__name__)

CUDA_SUCCESS = 0
GHOST_UTIL_MIN = 99
GHOST_MEM_MIB_MAX = 8
CURE_KERNEL_SECONDS_DEFAULT = 2.0

# Trivial FMA spin kernel as PTX (sm_75 baseline, JIT-compiled forward by the driver).
_SPIN_PTX = b"""
.version 7.0
.target sm_75
.address_size 64

.visible .entry spin(
    .param .u64 pbuf,
    .param .u32 iters
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<4>;
    .reg .f32 %f<3>;
    .reg .b64 %rd<3>;

    ld.param.u64 %rd1, [pbuf];
    cvta.to.global.u64 %rd2, %rd1;
    ld.param.u32 %r1, [iters];
    mov.u32 %r2, 0;
    mov.f32 %f1, 0f3F800000;

$L__loop:
    fma.rn.f32 %f1, %f1, 0f3F7FFFF0, 0f3DCCCCCD;
    add.s32 %r2, %r2, 1;
    setp.lt.u32 %p1, %r2, %r1;
    @%p1 bra $L__loop;

    st.global.f32 [%rd2], %f1;
    ret;
}
\x00"""


class CudaError(RuntimeError):
    pass


def _check(lib: ctypes.CDLL, res: int, what: str) -> None:
    if res != CUDA_SUCCESS:
        msg = ctypes.c_char_p()
        lib.cuGetErrorString(res, ctypes.byref(msg))
        detail = msg.value.decode() if msg.value else "?"
        raise CudaError(f"{what} failed: CUresult={res} ({detail})")


def _cuda_device_by_uuid(lib: ctypes.CDLL, gpu_uuid: str) -> int:
    # resolve an NVML-style uuid ("GPU-8c6b85ff-...") to a CUDA device ordinal
    want = gpu_uuid.lower().removeprefix("gpu-").replace("-", "")
    count = ctypes.c_int()
    _check(lib, lib.cuDeviceGetCount(ctypes.byref(count)), "cuDeviceGetCount")
    for i in range(count.value):
        dev = ctypes.c_int()
        _check(lib, lib.cuDeviceGet(ctypes.byref(dev), i), "cuDeviceGet")
        raw = (ctypes.c_ubyte * 16)()
        _check(lib, lib.cuDeviceGetUuid(ctypes.byref(raw), dev), "cuDeviceGetUuid")
        if bytes(raw).hex() == want:
            return dev.value
    raise CudaError(f"no CUDA device with uuid {gpu_uuid}")


def cure_gpu(
    gpu_uuid: str, mode: str = "kernel", seconds: float = CURE_KERNEL_SECONDS_DEFAULT
) -> None:
    # open a clean CUDA context on the latched GPU, optionally spin a kernel, tear down cleanly
    lib = ctypes.CDLL("libcuda.so.1")
    _check(lib, lib.cuInit(0), "cuInit")
    dev = _cuda_device_by_uuid(lib, gpu_uuid)

    ctx = ctypes.c_void_p()
    _check(lib, lib.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev), "cuCtxCreate_v2")
    try:
        if mode == "kernel":
            module = ctypes.c_void_p()
            _check(
                lib,
                lib.cuModuleLoadData(ctypes.byref(module), ctypes.c_char_p(_SPIN_PTX)),
                "cuModuleLoadData",
            )
            func = ctypes.c_void_p()
            _check(lib, lib.cuModuleGetFunction(ctypes.byref(func), module, b"spin"), "cuModuleGetFunction")

            dptr = ctypes.c_uint64()
            _check(lib, lib.cuMemAlloc_v2(ctypes.byref(dptr), 16 * 1024 * 1024), "cuMemAlloc_v2")

            iters = ctypes.c_uint32(500_000)
            params = (ctypes.c_void_p * 2)(
                ctypes.cast(ctypes.byref(dptr), ctypes.c_void_p),
                ctypes.cast(ctypes.byref(iters), ctypes.c_void_p),
            )
            deadline = time.monotonic() + seconds
            launches = 0
            while time.monotonic() < deadline:
                _check(
                    lib,
                    lib.cuLaunchKernel(func, 2048, 1, 1, 256, 1, 1, 0, None, params, None),
                    "cuLaunchKernel",
                )
                _check(lib, lib.cuCtxSynchronize(), "cuCtxSynchronize")
                launches += 1
            logger.info("cure kernel done: %d launches in %.1fs", launches, seconds)

            _check(lib, lib.cuMemFree_v2(dptr), "cuMemFree_v2")
            _check(lib, lib.cuModuleUnload(module), "cuModuleUnload")
    finally:
        _check(lib, lib.cuCtxDestroy_v2(ctx), "cuCtxDestroy_v2")


def _smi(args: list[str]) -> list[list[str]]:
    out = subprocess.run(
        ["nvidia-smi", *args, "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    ).stdout
    return [
        [cell.strip() for cell in line.split(",")] for line in out.strip().splitlines() if line.strip()
    ]


def detect_ghosts() -> list[str]:
    # ghost = util pinned at >=99% with (almost) no memory and zero compute processes on that GPU
    gpus = _smi(["--query-gpu=uuid,utilization.gpu,memory.used"])
    busy_uuids = {row[0] for row in _smi(["--query-compute-apps=gpu_uuid"]) if row}
    return [
        uuid
        for uuid, util, mem in gpus
        if int(util) >= GHOST_UTIL_MIN and int(mem) <= GHOST_MEM_MIB_MAX and uuid not in busy_uuids
    ]


def spawn_cure(gpu_uuid: str, timeout: float = 30.0) -> bool:
    # run the cure in a child process (clean process exit is part of the verified recipe),
    # then re-check whether the GPU is still a ghost
    env = os.environ | {"GHOST_CURE_UUID": gpu_uuid}
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__)],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        logger.error("cure subprocess failed for %s: %s", gpu_uuid, proc.stderr.strip())
        return False
    time.sleep(3)
    return gpu_uuid not in detect_ghosts()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    mode = os.environ.get("GHOST_CURE_MODE", "kernel")
    seconds = float(os.environ.get("GHOST_CURE_SECONDS", str(CURE_KERNEL_SECONDS_DEFAULT)))

    payload_uuid = os.environ.get("GHOST_CURE_UUID")
    if payload_uuid:
        cure_gpu(payload_uuid, mode=mode, seconds=seconds)
        return

    ghosts = detect_ghosts()
    if not ghosts:
        logger.info("no ghost GPUs detected")
        return
    for uuid in ghosts:
        logger.info("ghost GPU detected: %s — running cure (mode=%s)", uuid, mode)
        cured = spawn_cure(uuid)
        logger.info("cure result for %s: %s", uuid, "CLEARED" if cured else "STILL LATCHED")


if __name__ == "__main__":
    main()
