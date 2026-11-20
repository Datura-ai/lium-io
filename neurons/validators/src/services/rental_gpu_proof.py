"""Rental create proves every rented GPU with CUDA before the pod is reported created.

A host can make nvidia-smi and /proc inside the renter's container list cards CUDA cannot use (20 Nov 2026:
nvidia-smi listed 8, torch.cuda.device_count() was 1). Billing starts on ContainerCreated, so
the create itself asks CUDA, from inside the renter's container: every rented GPU UUID must be a CUDA
device there that opens a context and runs a 1 MiB memset. A short count raises RentalGpuProofFailed,
which the create's post-run handler turns into FailedContainerRequest (rent_failed, never billed).

False rejections it avoids:
- the renter image sets CUDA_VISIBLE_DEVICES: the probe runs under `env -u CUDA_VISIBLE_DEVICES`;
- the renter's own workload already holds the card (out of memory, exclusive-process compute mode): the card
  is still enumerated by CUDA under its own UUID, so it counts as present;
- an image with no python3 or no env (distroless, slim): the same probe runs on the host pinned to the rented
  UUIDs (CUDA_VISIBLE_DEVICES), the gpu_wedge cure's path, instead of failing the create.
"""

import json
import logging
import shlex
import time

import asyncssh

from core.utils import _m, get_extra_info

logger = logging.getLogger(__name__)

PROOF_MARKER = "RENTAL_GPU_PROOF_JSON:"
# An 8-GPU TDX guest takes ~16 s for cuInit and ~7.5 s for 8 contexts (DAH-6451, pinned_memory_probe);
# a bare-metal 8-GPU host about 1-3 s in all. 45 s fails a hung driver without failing the slow guests.
PROOF_TIMEOUT_SECONDS = 45
# docker exec / env: the image has no such executable (126 also covers an OCI "exec failed")
_NO_INTERPRETER_EXIT = (126, 127)
# CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_DEVICE_UNAVAILABLE: the card is real and busy with the renter's own work
_BUSY_RC = (2, 46)

_PROBE_CODE = f"""
import ctypes, json
c = ctypes
try:
    lib = c.CDLL("libcuda.so.1")
except OSError as e:
    print({PROOF_MARKER!r} + json.dumps({{"init_rc": "no_libcuda", "cards": []}})); raise SystemExit(0)
lib.cuMemsetD8_v2.argtypes = [c.c_uint64, c.c_ubyte, c.c_size_t]
lib.cuMemFree_v2.argtypes = [c.c_uint64]
n = c.c_int(0)
rc = lib.cuInit(0) or lib.cuDeviceGetCount(c.byref(n))
cards = []
for i in range(n.value if rc == 0 else 0):
    d = c.c_int(); u = (c.c_ubyte * 16)()
    lib.cuDeviceGet(c.byref(d), i); lib.cuDeviceGetUuid(u, d)
    h = bytes(u).hex()
    ctx = c.c_void_p(); r = lib.cuCtxCreate_v2(c.byref(ctx), 0, d)
    if r == 0:
        p = c.c_uint64(); r = lib.cuMemAlloc_v2(c.byref(p), c.c_size_t(1 << 20))
        if r == 0:
            r = lib.cuMemsetD8_v2(p, 0x5A, 1 << 20) or lib.cuCtxSynchronize()
            lib.cuMemFree_v2(p)
        lib.cuCtxDestroy_v2(ctx)
    cards.append({{"uuid": "GPU-%s-%s-%s-%s-%s" % (h[:8], h[8:12], h[12:16], h[16:20], h[20:]), "rc": r}})
print({PROOF_MARKER!r} + json.dumps({{"init_rc": rc, "cards": cards}}))
"""


class RentalGpuProofFailed(RuntimeError):
    pass


def judge_gpu_proof(expected_uuids: list[str], stdout: str) -> str:
    """'' when every expected UUID is a CUDA device that ran the op (or is busy with real work), else why not."""
    line = next((ln for ln in reversed(stdout.splitlines()) if ln.startswith(PROOF_MARKER)), None)
    if line is None:
        return "the CUDA probe gave no answer"
    try:
        verdict = json.loads(line[len(PROOF_MARKER):])
        seen = {card["uuid"].lower(): card["rc"] for card in verdict["cards"]}
    except (ValueError, KeyError, TypeError, AttributeError):
        return "the CUDA probe gave an unreadable answer"
    usable = [u for u in expected_uuids if seen.get(u.lower(), -1) == 0 or seen.get(u.lower()) in _BUSY_RC]
    if len(usable) == len(expected_uuids):
        return ""
    # the image's own python3 wrote this answer and the backend reads the error for GPU-fault text: only an int or
    # the probe's own marker goes into it
    init_rc = verdict.get("init_rc")
    if not (isinstance(init_rc, int) or init_rc == "no_libcuda"):
        init_rc = "?"
    # count only, never which card or which signal disagreed (anti-treadmill, as the all-cards proof)
    return (
        f"CUDA can use {len(usable)} of the {len(expected_uuids)} rented GPUs "
        f"(cuInit={init_rc}, CUDA devices={len(seen)})"
    )


async def prove_rented_gpus(
    ssh_client, *, container: str, python_path: str, gpu_uuids: list[str], log_extra: dict
) -> None:
    """Raise RentalGpuProofFailed unless CUDA in the container (or, for an image with no python3, on the
    host pinned to the rented UUIDs) uses every rented GPU."""
    if not gpu_uuids:
        # shortcut: a whole-node pod (--gpus all) names no UUIDs to prove; listing's all-cards proof covers it
        return
    started = time.monotonic()
    where = "container"
    command = shlex.join(
        ("/usr/bin/docker", "exec", "-u", "0", container, "env", "-u", "CUDA_VISIBLE_DEVICES",
         "python3", "-c", _PROBE_CODE)
    )
    try:
        result = await ssh_client.run(command, timeout=PROOF_TIMEOUT_SECONDS, check=False)
        if result.exit_status in _NO_INTERPRETER_EXIT:
            where = "host"
            pin = shlex.quote(",".join(gpu_uuids))
            command = f"CUDA_VISIBLE_DEVICES={pin} {shlex.quote(python_path)} -c {shlex.quote(_PROBE_CODE)}"
            result = await ssh_client.run(command, timeout=PROOF_TIMEOUT_SECONDS, check=False)
    except (TimeoutError, asyncssh.TimeoutError):
        failure = f"the CUDA probe did not finish in {PROOF_TIMEOUT_SECONDS}s"
    else:
        failure = judge_gpu_proof(gpu_uuids, result.stdout or "")
    elapsed = round(time.monotonic() - started, 3)
    extra = {**log_extra, "gpu_count": len(gpu_uuids), "proved_in": where, "elapsed_seconds": elapsed,
             "failure": failure}
    (logger.warning if failure else logger.info)(_m("Rental GPU CUDA proof", extra=get_extra_info(extra)))
    if failure:
        raise RentalGpuProofFailed(f"GPU check failed before billing: {failure}")
