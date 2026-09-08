# Build the gpu_sig hardware-signature prober (DAH-3137).
#
# The GPU binary needs a CUDA toolkit (nvcc). The seal-interop test builds with
# a plain C compiler (no CUDA), so CI/dev can verify the HMAC matches Python.
#
#   make gpu_sig        # needs nvcc + a CUDA-capable build host
#   make test_seal      # host-only crypto interop harness (cc)
#   make LIUM_SIG_KEY=<hex> gpu_sig   # bake a release seal key

NVCC ?= nvcc
CC ?= cc
LIUM_SIG_KEY ?= lium-gpu-sig-dev-key-do-not-use-in-prod
# Target the datacenter + consumer arches the fleet runs (Ampere..Blackwell).
GPU_ARCHS ?= -gencode arch=compute_80,code=sm_80 \
             -gencode arch=compute_86,code=sm_86 \
             -gencode arch=compute_89,code=sm_89 \
             -gencode arch=compute_90,code=sm_90

gpu_sig: gpu_sig.cu sha256.h
	$(NVCC) -O3 $(GPU_ARCHS) \
	  -DLIUM_SIG_MASTER_KEY="\"$(LIUM_SIG_KEY)\"" \
	  -o gpu_sig gpu_sig.cu

# Portable single-arch build for a pod whose exact SM you know (faster to build):
#   make ARCH=sm_89 gpu_sig_single
gpu_sig_single: gpu_sig.cu sha256.h
	$(NVCC) -O3 -arch=$(ARCH) \
	  -DLIUM_SIG_MASTER_KEY="\"$(LIUM_SIG_KEY)\"" \
	  -o gpu_sig gpu_sig.cu

test_seal: test_seal.c sha256.h
	$(CC) -O2 -o test_seal test_seal.c

clean:
	rm -f gpu_sig gpu_sig_single test_seal

.PHONY: clean
