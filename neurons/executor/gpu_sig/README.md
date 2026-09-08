# gpu_sig — on-host GPU hardware-signature prober (DAH-3137)

A small, self-contained binary that ships **pre-built in the executor image** and
answers a validator's nonce-bound challenge with a **sealed** per-GPU hardware
signature. It is a building block toward `liumd` (DAH-2834). Full design and
threat model: `~/lium-ads/verification-binary/DESIGN.md`.

## What it measures (per card, pinned via `CUDA_VISIBLE_DEVICES`)
- sustained **FP32 tiled-SGEMM throughput** (TFLOPS), seeded from the nonce
- device-to-device **VRAM bandwidth** (GB/s)
- **kernel-reported identity** (GPU UUID / model / PCI) from
  `/proc/driver/nvidia/gpus/*/information` — outside NVML, so a userspace
  `nvidia-smi`/`libnvidia-ml` shim (DAH-2662) does not control it

It seals the result so the miner-controlled shell cannot tamper the numbers:

```
key = HMAC_SHA256(MASTER_KEY, kernel_uuid)          # validator derives the same
sig = HMAC_SHA256(key, "nonce|device|uuid|pci|tflops|gbps|ms")
```

The signed `msg` string is authoritative; the validator parses the numbers out of
it and verifies `sig`, so C/Python float formatting can never diverge.

## Build
```
make gpu_sig LIUM_SIG_KEY=<hex-seal-key-matching-the-validator>   # needs nvcc + CUDA
make test_seal                                                    # host-only crypto interop (cc)
```
`GPU_ARCHS` in the `Makefile` covers Ampere→Hopper (sm_80/86/89/90). Blackwell
(sm_100/sm_120) needs CUDA ≥ 12.8; add its `-gencode` there. At runtime the binary
needs only the driver `libcuda.so.1` (built `-cudart static`), which the NVIDIA
container runtime injects.

## Ship in the executor image
Opt-in build args on `neurons/executor/Dockerfile` (default off — a normal build
pulls no CUDA and installs nothing):
```
docker build \
  --build-arg GPUSIG_BUILDER=nvidia/cuda:12.6.2-devel-ubuntu22.04 \
  --build-arg INSTALL_GPU_SIGNATURE=true \
  --build-arg LIUM_SIG_KEY=<hex> ...
```
Installs to `/root/app/bin/gpu_sig`. The validator finds it at
`GPU_SIGNATURE_BINARY_RELATIVE` under the executor root and skips cleanly if absent.

## Calibration (required before enforcement)
Run on one real card of each class and record `tflops` / `gbps`:
```
./gpu_sig --nonce $(openssl rand -hex 32) --device 0
```
Feed generous floors into `GPU_SIGNATURE_ENVELOPE`
(`neurons/validators/src/services/gpu_signature.py`) and set `calibrated=True`
only for classes measured with THIS binary. Until then the envelope gates
nothing (fail-open) and the check is observe-only.

## Honest scope
`MASTER_KEY` is baked into the (signed, digest-pinned) image, so a root provider
who reverse-engineers the binary can still forge a seal over fabricated numbers.
This closes the cheap/scalable spoofs (NVML shims, canned/replayed answers,
wrong-class numbers, count serialisation). Full unforgeability needs the
derived-key seal folded into liumd and/or NVIDIA CC attestation.
