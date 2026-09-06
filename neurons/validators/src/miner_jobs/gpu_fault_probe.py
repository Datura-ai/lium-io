"""GPU kernel-fault probe, run on the executor by GpuFaultProbeCheck (`python -I -`, this file on stdin).

Standard library only. The CUDA driver API is reached through ctypes (`libcuda.so.1` is in every
container the NVIDIA runtime prepares), the kernels are PTX the driver JIT-compiles, NVML is read
through nvidia-ml-py when it is importable. One forked worker per GPU runs, for a few seconds, the
access patterns a cuBLAS matmul never exercises: a random permutation built on the device, a gather
through it, a scatter back (plain stores and atomics), a dependent pointer chase, and a pinned-memory
async H2D/D2H round-trip. Every result is verified on the device (mismatch counters) and a sample of
the chase on the host. A fault is any non-zero CUDA return code once the context exists
(CUDA_ERROR_ILLEGAL_ADDRESS is what Blender prints as "Illegal address in CUDA queue"), a data
mismatch, a worker that crashes or hangs, or an uncorrected-ECC / remapped-row / recovery-action
change in NVML across the run. A probe that cannot start (no libcuda, cuInit, PTX JIT) is an error,
not a fault: the validator does not penalise what it could not measure.

Prints one line `GPU_FAULT_PROBE_JSON: {...}`. Exit 0 ok, 1 fault, 2 could not run.
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
JSON_MARKER = "GPU_FAULT_PROBE_JSON:"
BLOCK = 256
CHASE_STEPS = 32
CHASE_SAMPLES = 64
MIN_LOG2_N = 20  # 1 Mi elements = 4 MB per buffer
MAX_LOG2_N = 28  # 256 Mi elements = 1 GB per buffer
BUFFERS = 4  # in, idx, out, aux
VRAM_RESERVE_MB = (
    1024  # left free on the card: the runtime's own context and whatever else idles there
)
WORKER_GRACE_SECONDS = 30  # on top of --seconds: JIT, allocations, copies, host verification

# fmix32-style mixer whose every step is a bijection on [0, 2**k): (x + seed) mod 2**k, odd multiplier
# mod 2**k, xorshift within k bits. Mirrored in PTX by k_perm; the host replays it to check the chase.
PERM_MUL = (0x9E3779B1, 0x85EBCA6B, 0xC2B2AE35)
PERM_SHIFT = (15, 13, 16)

PTX = r"""
.version 6.0
.target sm_50
.address_size 64

// idx[i] = perm(i): the permutation every other kernel gathers and scatters through
.visible .entry k_perm(.param .u64 p_idx, .param .u32 p_n, .param .u32 p_seed, .param .u32 p_mask)
{
    .reg .pred %p<2>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [p_idx];
    ld.param.u32 %r1, [p_n];
    ld.param.u32 %r2, [p_seed];
    ld.param.u32 %r3, [p_mask];
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ntid.x;
    mov.u32 %r6, %tid.x;
    mad.lo.s32 %r7, %r4, %r5, %r6;
    setp.ge.u32 %p1, %r7, %r1;
    @%p1 bra L_end;
    add.u32 %r8, %r7, %r2;
    and.b32 %r8, %r8, %r3;
    mov.u32 %r10, 0x9E3779B1;
    mul.lo.u32 %r8, %r8, %r10;
    and.b32 %r8, %r8, %r3;
    shr.u32 %r9, %r8, 15;
    xor.b32 %r8, %r8, %r9;
    mov.u32 %r10, 0x85EBCA6B;
    mul.lo.u32 %r8, %r8, %r10;
    and.b32 %r8, %r8, %r3;
    shr.u32 %r9, %r8, 13;
    xor.b32 %r8, %r8, %r9;
    mov.u32 %r10, 0xC2B2AE35;
    mul.lo.u32 %r8, %r8, %r10;
    and.b32 %r8, %r8, %r3;
    shr.u32 %r9, %r8, 16;
    xor.b32 %r8, %r8, %r9;
    cvta.to.global.u64 %rd2, %rd1;
    mul.wide.u32 %rd3, %r7, 4;
    add.s64 %rd4, %rd2, %rd3;
    st.global.u32 [%rd4], %r8;
L_end:
    ret;
}

// out[i] = i
.visible .entry k_iota(.param .u64 p_out, .param .u32 p_n)
{
    .reg .pred %p<2>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [p_out];
    ld.param.u32 %r1, [p_n];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra L_end;
    cvta.to.global.u64 %rd2, %rd1;
    mul.wide.u32 %rd3, %r5, 4;
    add.s64 %rd4, %rd2, %rd3;
    st.global.u32 [%rd4], %r5;
L_end:
    ret;
}

// out[i] = in[idx[i]]  (random reads)
.visible .entry k_gather(.param .u64 p_out, .param .u64 p_in, .param .u64 p_idx, .param .u32 p_n)
{
    .reg .pred %p<2>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<12>;
    ld.param.u64 %rd1, [p_out];
    ld.param.u64 %rd2, [p_in];
    ld.param.u64 %rd3, [p_idx];
    ld.param.u32 %r1, [p_n];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra L_end;
    cvta.to.global.u64 %rd4, %rd1;
    cvta.to.global.u64 %rd5, %rd2;
    cvta.to.global.u64 %rd6, %rd3;
    mul.wide.u32 %rd7, %r5, 4;
    add.s64 %rd8, %rd6, %rd7;
    ld.global.u32 %r6, [%rd8];
    mul.wide.u32 %rd9, %r6, 4;
    add.s64 %rd10, %rd5, %rd9;
    ld.global.u32 %r7, [%rd10];
    add.s64 %rd11, %rd4, %rd7;
    st.global.u32 [%rd11], %r7;
L_end:
    ret;
}

// out[idx[i]] = i  (random writes; out becomes the inverse permutation)
.visible .entry k_scatter(.param .u64 p_out, .param .u64 p_idx, .param .u32 p_n)
{
    .reg .pred %p<2>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<12>;
    ld.param.u64 %rd1, [p_out];
    ld.param.u64 %rd3, [p_idx];
    ld.param.u32 %r1, [p_n];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra L_end;
    cvta.to.global.u64 %rd4, %rd1;
    cvta.to.global.u64 %rd6, %rd3;
    mul.wide.u32 %rd7, %r5, 4;
    add.s64 %rd8, %rd6, %rd7;
    ld.global.u32 %r6, [%rd8];
    mul.wide.u32 %rd9, %r6, 4;
    add.s64 %rd10, %rd4, %rd9;
    st.global.u32 [%rd10], %r5;
L_end:
    ret;
}

// cnt[idx[i]] += 1  (random atomics; every slot ends at exactly 1)
.visible .entry k_scatter_add(.param .u64 p_cnt, .param .u64 p_idx, .param .u32 p_n)
{
    .reg .pred %p<2>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<12>;
    ld.param.u64 %rd1, [p_cnt];
    ld.param.u64 %rd3, [p_idx];
    ld.param.u32 %r1, [p_n];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra L_end;
    cvta.to.global.u64 %rd4, %rd1;
    cvta.to.global.u64 %rd6, %rd3;
    mul.wide.u32 %rd7, %r5, 4;
    add.s64 %rd8, %rd6, %rd7;
    ld.global.u32 %r6, [%rd8];
    mul.wide.u32 %rd9, %r6, 4;
    add.s64 %rd10, %rd4, %rd9;
    red.global.add.u32 [%rd10], 1;
L_end:
    ret;
}

// j = i; steps times: j = idx[j]; out[i] = j  (dependent random reads)
.visible .entry k_chase(.param .u64 p_out, .param .u64 p_idx, .param .u32 p_n, .param .u32 p_steps)
{
    .reg .pred %p<3>;
    .reg .b32 %r<12>;
    .reg .b64 %rd<12>;
    ld.param.u64 %rd1, [p_out];
    ld.param.u64 %rd3, [p_idx];
    ld.param.u32 %r1, [p_n];
    ld.param.u32 %r10, [p_steps];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra L_end;
    cvta.to.global.u64 %rd4, %rd1;
    cvta.to.global.u64 %rd6, %rd3;
    mov.u32 %r8, %r5;
    mov.u32 %r9, 0;
    setp.ge.u32 %p2, %r9, %r10;
    @%p2 bra L_store;
L_loop:
    mul.wide.u32 %rd7, %r8, 4;
    add.s64 %rd8, %rd6, %rd7;
    ld.global.u32 %r8, [%rd8];
    add.u32 %r9, %r9, 1;
    setp.lt.u32 %p2, %r9, %r10;
    @%p2 bra L_loop;
L_store:
    mul.wide.u32 %rd9, %r5, 4;
    add.s64 %rd10, %rd4, %rd9;
    st.global.u32 [%rd10], %r8;
L_end:
    ret;
}

// ctr += (a[i] != b[i])
.visible .entry k_count_neq(.param .u64 p_a, .param .u64 p_b, .param .u32 p_n, .param .u64 p_ctr)
{
    .reg .pred %p<3>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<12>;
    ld.param.u64 %rd1, [p_a];
    ld.param.u64 %rd2, [p_b];
    ld.param.u32 %r1, [p_n];
    ld.param.u64 %rd3, [p_ctr];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra L_end;
    cvta.to.global.u64 %rd4, %rd1;
    cvta.to.global.u64 %rd5, %rd2;
    mul.wide.u32 %rd7, %r5, 4;
    add.s64 %rd8, %rd4, %rd7;
    ld.global.u32 %r6, [%rd8];
    add.s64 %rd9, %rd5, %rd7;
    ld.global.u32 %r7, [%rd9];
    setp.eq.u32 %p2, %r6, %r7;
    @%p2 bra L_end;
    cvta.to.global.u64 %rd10, %rd3;
    red.global.add.u32 [%rd10], 1;
L_end:
    ret;
}

// ctr += (a[i] != v)
.visible .entry k_count_neq_const(.param .u64 p_a, .param .u32 p_n, .param .u32 p_v, .param .u64 p_ctr)
{
    .reg .pred %p<3>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<12>;
    ld.param.u64 %rd1, [p_a];
    ld.param.u32 %r1, [p_n];
    ld.param.u32 %r7, [p_v];
    ld.param.u64 %rd3, [p_ctr];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra L_end;
    cvta.to.global.u64 %rd4, %rd1;
    mul.wide.u32 %rd7, %r5, 4;
    add.s64 %rd8, %rd4, %rd7;
    ld.global.u32 %r6, [%rd8];
    setp.eq.u32 %p2, %r6, %r7;
    @%p2 bra L_end;
    cvta.to.global.u64 %rd10, %rd3;
    red.global.add.u32 [%rd10], 1;
L_end:
    ret;
}

// ctr += (inv[idx[i]] != i)
.visible .entry k_check_inverse(.param .u64 p_inv, .param .u64 p_idx, .param .u32 p_n, .param .u64 p_ctr)
{
    .reg .pred %p<3>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<12>;
    ld.param.u64 %rd1, [p_inv];
    ld.param.u64 %rd2, [p_idx];
    ld.param.u32 %r1, [p_n];
    ld.param.u64 %rd3, [p_ctr];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra L_end;
    cvta.to.global.u64 %rd4, %rd1;
    cvta.to.global.u64 %rd5, %rd2;
    mul.wide.u32 %rd7, %r5, 4;
    add.s64 %rd8, %rd5, %rd7;
    ld.global.u32 %r6, [%rd8];
    mul.wide.u32 %rd9, %r6, 4;
    add.s64 %rd10, %rd4, %rd9;
    ld.global.u32 %r7, [%rd10];
    setp.eq.u32 %p2, %r7, %r5;
    @%p2 bra L_end;
    cvta.to.global.u64 %rd11, %rd3;
    red.global.add.u32 [%rd11], 1;
L_end:
    ret;
}
"""

KERNELS = (
    "k_perm",
    "k_iota",
    "k_gather",
    "k_scatter",
    "k_scatter_add",
    "k_chase",
    "k_count_neq",
    "k_count_neq_const",
    "k_check_inverse",
)


def perm(i: int, seed: int, mask: int) -> int:
    x = (i + seed) & mask
    for mul, shift in zip(PERM_MUL, PERM_SHIFT):
        x = (x * mul) & mask
        x ^= x >> shift
    return x


class ProbeError(Exception):
    """The probe could not start on this device (library, cuInit, JIT): reported as an error, not a fault."""


class CudaFault(Exception):
    """A CUDA call failed once the context existed, or a result did not verify: the device is faulty."""


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
        self.lib.cuGetErrorName.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        self.lib.cuGetErrorName.restype = ctypes.c_int

    def error_name(self, code: int) -> str:
        name = ctypes.c_char_p()
        if self.lib.cuGetErrorName(code, ctypes.byref(name)) == 0 and name.value:
            return name.value.decode()
        return f"CUDA_ERROR_{code}"

    def call(self, fn: str, *args, fault: bool = True) -> None:
        code = getattr(self.lib, fn)(*args)
        if code != 0:
            message = f"{fn} -> {self.error_name(code)} ({code})"
            raise CudaFault(message) if fault else ProbeError(message)


class DevPtr(int):
    """A CUdeviceptr; distinguishes 64-bit pointer arguments from 32-bit scalars in _launch."""


def _launch(cuda: Cuda, func, stream, n: int, *args) -> None:
    # kernel arguments travel as an array of pointers to their ctypes values, which must outlive the call
    values = [ctypes.c_uint64(a) if isinstance(a, DevPtr) else ctypes.c_uint32(a) for a in args]
    params = (ctypes.c_void_p * len(values))(*[ctypes.addressof(v) for v in values])
    grid = (n + BLOCK - 1) // BLOCK
    cuda.call("cuLaunchKernel", func, grid, 1, 1, BLOCK, 1, 1, 0, stream, params, None)


def _read_counter(cuda: Cuda, d_ctr: int, h_ctr, stream) -> int:
    cuda.call("cuMemcpyDtoHAsync_v2", h_ctr, ctypes.c_uint64(d_ctr), ctypes.c_size_t(4), stream)
    cuda.call("cuStreamSynchronize", stream)
    return ctypes.c_uint32.from_address(h_ctr.value).value


def probe_device(index: int, seconds: float, vram_mb: int) -> dict:
    report: dict = {"index": index}
    cuda = Cuda()
    lib = cuda.lib
    lib.cuLaunchKernel.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
    ]
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
    lib.cuMemsetD32Async.argtypes = [
        ctypes.c_uint64,
        ctypes.c_uint,
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    lib.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
    lib.cuMemFree_v2.argtypes = [ctypes.c_uint64]
    lib.cuMemAllocHost_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    lib.cuMemFreeHost.argtypes = [ctypes.c_void_p]
    lib.cuMemGetInfo_v2.argtypes = [
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
    lib.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
    lib.cuStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
    lib.cuStreamSynchronize.argtypes = [ctypes.c_void_p]
    lib.cuModuleLoadDataEx.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.cuModuleGetFunction.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    lib.cuDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]

    cuda.call("cuInit", 0, fault=False)
    device = ctypes.c_int()
    cuda.call("cuDeviceGet", ctypes.byref(device), index, fault=False)
    name = ctypes.create_string_buffer(256)
    lib.cuDeviceGetName(name, 256, device)
    report["name"] = name.value.decode(errors="replace")

    context = ctypes.c_void_p()
    cuda.call("cuCtxCreate_v2", ctypes.byref(context), 0, device)
    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    cuda.call("cuMemGetInfo_v2", ctypes.byref(free), ctypes.byref(total))
    report["vram_free_mb"] = free.value // MB

    budget = min(vram_mb * MB, max(free.value - VRAM_RESERVE_MB * MB, 0))
    log2_n = MIN_LOG2_N
    while log2_n < MAX_LOG2_N and BUFFERS * 4 * (1 << (log2_n + 1)) <= budget:
        log2_n += 1
    n = 1 << log2_n
    mask = n - 1
    report["elements"] = n
    report["working_set_mb"] = BUFFERS * 4 * n // MB

    t_jit = time.perf_counter()
    module = ctypes.c_void_p()
    error_log = ctypes.create_string_buffer(8192)
    options = (ctypes.c_int * 2)(
        5, 6
    )  # CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
    option_values = (ctypes.c_void_p * 2)(ctypes.addressof(error_log), 8192)
    code = lib.cuModuleLoadDataEx(ctypes.byref(module), PTX.encode(), 2, options, option_values)
    if code != 0:
        raise ProbeError(
            f"PTX JIT failed: {cuda.error_name(code)}: {error_log.value.decode(errors='replace')}"
        )
    funcs = {}
    for kernel in KERNELS:
        func = ctypes.c_void_p()
        cuda.call("cuModuleGetFunction", ctypes.byref(func), module, kernel.encode(), fault=False)
        funcs[kernel] = func
    report["jit_ms"] = int((time.perf_counter() - t_jit) * 1000)

    stream = ctypes.c_void_p()
    cuda.call("cuStreamCreate", ctypes.byref(stream), 1)  # CU_STREAM_NON_BLOCKING
    device_buffers = []
    for _ in range(BUFFERS):
        ptr = ctypes.c_uint64()
        cuda.call("cuMemAlloc_v2", ctypes.byref(ptr), n * 4)
        device_buffers.append(DevPtr(ptr.value))
    d_in, d_idx, d_out, d_aux = device_buffers
    d_ctr = ctypes.c_uint64()
    cuda.call("cuMemAlloc_v2", ctypes.byref(d_ctr), 16)
    d_ctr = DevPtr(d_ctr.value)
    h_ctr = ctypes.c_void_p()
    cuda.call("cuMemAllocHost_v2", ctypes.byref(h_ctr), 16)
    copy_bytes = min(32 * MB, n * 4)
    h_src, h_dst = ctypes.c_void_p(), ctypes.c_void_p()
    cuda.call("cuMemAllocHost_v2", ctypes.byref(h_src), copy_bytes)
    cuda.call("cuMemAllocHost_v2", ctypes.byref(h_dst), copy_bytes)
    pattern = os.urandom(copy_bytes)
    ctypes.memmove(h_src, pattern, copy_bytes)
    h_sample = ctypes.c_void_p()
    cuda.call("cuMemAllocHost_v2", ctypes.byref(h_sample), CHASE_SAMPLES * 4)
    report["copy_mb"] = copy_bytes // MB

    def launch(kernel: str, *args) -> None:
        _launch(cuda, funcs[kernel], stream, n, *args)

    def counter_check(what: str, kernel: str, *args) -> None:
        cuda.call("cuMemsetD32Async", d_ctr, 0, 4, stream)
        launch(kernel, *args)
        mismatches = _read_counter(cuda, d_ctr, h_ctr, stream)
        if mismatches:
            raise CudaFault(f"{what}: {mismatches} of {n} elements wrong")

    launch("k_iota", d_in, n)
    rounds = 0
    t_work = time.perf_counter()
    while rounds < 2 or (time.perf_counter() - t_work < seconds and rounds < 64):
        seed = (rounds * 0x9E37 + 12345) & mask
        launch("k_perm", d_idx, n, seed, mask)
        launch("k_gather", d_out, d_in, d_idx, n)
        counter_check("gather", "k_count_neq", d_out, d_idx, n, d_ctr)
        cuda.call("cuMemsetD32Async", d_aux, 0, n, stream)
        launch("k_scatter_add", d_aux, d_idx, n)
        counter_check("scatter_add", "k_count_neq_const", d_aux, n, 1, d_ctr)
        launch("k_scatter", d_aux, d_idx, n)
        counter_check("scatter", "k_check_inverse", d_aux, d_idx, n, d_ctr)
        launch("k_chase", d_out, d_idx, n, CHASE_STEPS)
        # pinned-memory round trip through the copy engines, both directions, verified on the host
        cuda.call("cuMemcpyHtoDAsync_v2", d_aux, h_src, copy_bytes, stream)
        cuda.call("cuMemcpyDtoHAsync_v2", h_dst, d_aux, copy_bytes, stream)
        # a host sample of the chase: 64 dependent-load chains replayed in Python
        starts = [(perm(k, seed ^ 0x5BD1, mask)) for k in range(CHASE_SAMPLES)]
        for k, start in enumerate(starts):
            cuda.call(
                "cuMemcpyDtoHAsync_v2",
                ctypes.c_void_p(h_sample.value + 4 * k),
                ctypes.c_uint64(d_out + 4 * start),
                ctypes.c_size_t(4),
                stream,
            )
        cuda.call("cuStreamSynchronize", stream)
        cuda.call("cuCtxSynchronize")
        if ctypes.string_at(h_dst, copy_bytes) != pattern:
            raise CudaFault("async memcpy round trip: data differs")
        sample = (ctypes.c_uint32 * CHASE_SAMPLES).from_address(h_sample.value)
        for k, start in enumerate(starts):
            j = start
            for _ in range(CHASE_STEPS):
                j = perm(j, seed, mask)
            if sample[k] != j:
                raise CudaFault(f"chase: start {start} reached {sample[k]}, expected {j}")
        rounds += 1
    report["rounds"] = rounds
    report["work_s"] = round(time.perf_counter() - t_work, 2)

    for ptr in device_buffers + [d_ctr]:
        cuda.call("cuMemFree_v2", ptr)
    for ptr in (h_ctr, h_src, h_dst, h_sample):
        cuda.call("cuMemFreeHost", ptr)
    cuda.call("cuCtxSynchronize")
    cuda.call("cuCtxDestroy_v2", context)
    return report


def _worker(index: int, seconds: float, vram_mb: int, conn) -> None:
    started = time.perf_counter()
    try:
        report = probe_device(index, seconds, vram_mb)
        report["status"] = "ok"
    except CudaFault as exc:
        report = {"index": index, "status": "fault", "error": str(exc)}
    except ProbeError as exc:
        report = {"index": index, "status": "error", "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - the parent must always get a verdict
        report = {"index": index, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
    report["elapsed_s"] = round(time.perf_counter() - started, 2)
    conn.send(report)
    conn.close()


def _count_worker(conn) -> None:
    try:
        cuda = Cuda()
        cuda.call("cuInit", 0, fault=False)
        count = ctypes.c_int()
        cuda.call("cuDeviceGetCount", ctypes.byref(count), fault=False)
        conn.send({"count": count.value})
    except Exception as exc:  # noqa: BLE001
        conn.send({"error": str(exc)})
    conn.close()


def _device_count(mp) -> int:
    # in its own fork: a process that has called cuInit cannot fork CUDA-capable children
    parent_conn, child_conn = mp.Pipe(duplex=False)
    process = mp.Process(target=_count_worker, args=(child_conn,))
    process.start()
    child_conn.close()
    reply = (
        parent_conn.recv()
        if parent_conn.poll(WORKER_GRACE_SECONDS)
        else {"error": "device enumeration hung"}
    )
    process.join(5)
    if "error" in reply:
        raise ProbeError(reply["error"])
    return reply["count"]


def nvml_snapshot() -> dict:
    """Per-GPU counters that move when hardware faults: uncorrected ECC, remapped rows, recovery action."""
    try:
        import pynvml
    except ImportError:
        return {"available": False}
    try:
        pynvml.nvmlInit()
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": str(exc)}
    gpus = []
    try:
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            gpu: dict = {"index": i}
            for key, read in (
                ("uuid", lambda h: pynvml.nvmlDeviceGetUUID(h)),
                (
                    "ecc_uncorrected",
                    lambda h: pynvml.nvmlDeviceGetTotalEccErrors(
                        h, pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED, pynvml.NVML_VOLATILE_ECC
                    ),
                ),
                ("remapped_rows", lambda h: list(pynvml.nvmlDeviceGetRemappedRows(h))),
                ("recovery_action", lambda h: pynvml.nvmlDeviceGetGpuRecoveryAction(h)),
            ):
                try:
                    value = read(handle)
                    gpu[key] = value.decode() if isinstance(value, bytes) else value
                except Exception:  # noqa: BLE001 - NotSupported on consumer cards, missing on old bindings
                    gpu[key] = None
            gpus.append(gpu)
    finally:
        pynvml.nvmlShutdown()
    return {"available": True, "gpus": gpus}


def nvml_faults(before: dict, after: dict) -> list[str]:
    faults = []
    for b, a in zip(before.get("gpus", []), after.get("gpus", [])):
        if a.get("ecc_uncorrected") is not None and b.get("ecc_uncorrected") is not None:
            if a["ecc_uncorrected"] > b["ecc_uncorrected"]:
                faults.append(
                    f"gpu {a['index']}: uncorrected ECC errors {b['ecc_uncorrected']} -> {a['ecc_uncorrected']}"
                )
        rows = a.get("remapped_rows")
        if rows and len(rows) >= 4 and (rows[2] or rows[3]):
            faults.append(f"gpu {a['index']}: remapped rows pending={rows[2]} failure={rows[3]}")
        if a.get("recovery_action"):
            faults.append(f"gpu {a['index']}: NVML recovery action {a['recovery_action']} required")
    return faults


def xid_lines() -> dict:
    """The kernel log's NVRM Xid lines when the container may read it; most cannot, and that is fine."""
    try:
        out = subprocess.run(["dmesg"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": str(exc)}
    if out.returncode != 0:
        return {"available": False, "error": (out.stderr or "").strip()[:200]}
    lines = [line.strip() for line in out.stdout.splitlines() if "NVRM: Xid" in line]
    return {"available": True, "count": len(lines), "last": lines[-5:]}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--seconds", type=float, default=4.0, help="kernel work per GPU (default 4)"
    )
    parser.add_argument(
        "--vram-mb", type=int, default=2048, help="working set cap per GPU (default 2048)"
    )
    parser.add_argument(
        "--device", type=int, action="append", help="probe only this index (repeatable)"
    )
    args = parser.parse_args(argv)

    started = time.perf_counter()
    result: dict = {"status": "ok", "devices": [], "faults": []}
    # every CUDA call happens in a forked worker (one per GPU, plus one to count them): a crash or hang
    # in the driver takes the worker, not the verdict, and the parent never initialises CUDA itself
    mp = multiprocessing.get_context("fork")
    try:
        indices = args.device if args.device else list(range(_device_count(mp)))
        if not indices:
            raise ProbeError("no CUDA devices")
    except ProbeError as exc:
        result.update(
            status="error", error=str(exc), elapsed_s=round(time.perf_counter() - started, 2)
        )
        print(JSON_MARKER, json.dumps(result, sort_keys=True))
        return 2

    result["nvml_before"] = nvml_snapshot()
    xid_before = xid_lines()

    workers = []
    for index in indices:
        parent_conn, child_conn = mp.Pipe(duplex=False)
        process = mp.Process(target=_worker, args=(index, args.seconds, args.vram_mb, child_conn))
        process.start()
        child_conn.close()
        workers.append((index, process, parent_conn))

    deadline = time.perf_counter() + args.seconds + WORKER_GRACE_SECONDS
    for index, process, conn in workers:
        report = None
        while time.perf_counter() < deadline:
            if conn.poll(0.2):
                try:
                    report = conn.recv()
                except EOFError:
                    pass
                break
            if not process.is_alive():
                break
        process.join(timeout=max(0.0, deadline - time.perf_counter()))
        if process.is_alive():
            process.kill()
            process.join(5)
            report = {
                "index": index,
                "status": "fault",
                "error": f"hung: no result after {int(args.seconds + WORKER_GRACE_SECONDS)}s",
            }
        elif report is None:
            code = process.exitcode
            report = {
                "index": index,
                "status": "fault",
                "error": f"worker died with exit code {code}",
            }
        result["devices"].append(report)

    result["nvml_after"] = nvml_snapshot()
    result["xid"] = xid_lines()
    if xid_before.get("available") and result["xid"].get("available"):
        new_xids = result["xid"]["count"] - xid_before["count"]
        if new_xids > 0:
            result["faults"].append(f"{new_xids} new NVRM Xid line(s): {result['xid']['last']}")
    result["faults"].extend(nvml_faults(result["nvml_before"], result["nvml_after"]))
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
    sys.exit(main(sys.argv[1:]))
