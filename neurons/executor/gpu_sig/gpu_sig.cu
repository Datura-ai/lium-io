/*
 * gpu_sig — nonce-bound, sealed per-GPU hardware-signature prober (DAH-3137).
 *
 * Ships pre-built in the executor image (NOT uploaded per check). The validator
 * runs it over the existing SSH channel, pinned to one card:
 *
 *     CUDA_VISIBLE_DEVICES=<i> <root>/bin/gpu_sig --nonce <64hex> --device 0
 *
 * It measures a hardware signature that is expensive to fake:
 *   - sustained FP32 matmul throughput (TFLOPS), seeded from the nonce
 *   - device VRAM bandwidth (GB/s)
 *   - kernel-reported GPU identity (UUID/model/PCI) read from
 *     /proc/driver/nvidia/gpus/<pci>/information — OUTSIDE NVML, so a userspace
 *     nvidia-smi/libnvidia-ml shim (DAH-2662) does not control it.
 *
 * It seals the result so the miner-controlled shell cannot tamper the numbers:
 *   key  = HMAC_SHA256(MASTER_KEY, kernel_uuid)      # validator derives the same
 *   sig  = HMAC_SHA256(key, "nonce|device|uuid|pci|tflops|gbps|ms")
 * The message is the EXACT string tokens printed in the JSON, so the C and
 * Python HMAC inputs are byte-identical regardless of float formatting.
 *
 * Honest scope: MASTER_KEY is baked into the (signed, digest-pinned) executor
 * image, so a root provider who reverse-engineers the binary can still forge a
 * seal over fabricated numbers. This closes the cheap/scalable spoofs (NVML
 * shims, canned/replayed answers, wrong-class numbers, count serialisation) and
 * gathers ground truth; full unforgeability needs the derived-key seal folded
 * into liumd (DAH-2834) and/or NVIDIA CC attestation. See DESIGN.md §8.
 */
#include <cuda_runtime.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>  /* strcasecmp */

#include "sha256.h"

#ifndef LIUM_SIG_MASTER_KEY
/* Overridden at build time: nvcc ... -DLIUM_SIG_MASTER_KEY="\"<hex>\"".
 * This default is a well-known dev key; a release image bakes a rotated key
 * that the validator holds via shared config. */
#define LIUM_SIG_MASTER_KEY "lium-gpu-sig-dev-key-do-not-use-in-prod"
#endif

#define TILE 32

static void emit_error(const char *nonce, int device, const char *code) {
    /* structured, single-line, always exit != 0 on failure */
    printf("{\"gpu_sig\": 1, \"ok\": false, \"nonce\": \"%s\", \"device\": %d, \"error\": \"%s\"}\n",
           nonce ? nonce : "", device, code);
    fflush(stdout);
}

#define CUDA_OK(call, nonce, dev, code)                          \
    do {                                                         \
        cudaError_t _e = (call);                                 \
        if (_e != cudaSuccess) {                                 \
            char _buf[128];                                      \
            snprintf(_buf, sizeof(_buf), "%s:%s", code, cudaGetErrorName(_e)); \
            emit_error(nonce, dev, _buf);                        \
            return 3;                                            \
        }                                                        \
    } while (0)

/* Tiled single-precision GEMM: C = A * B (N x N), shared-memory blocked. */
__global__ void sgemm_tiled(const float *A, const float *B, float *C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;
    for (int t = 0; t < (N + TILE - 1) / TILE; ++t) {
        int ac = t * TILE + threadIdx.x;
        int br = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < N && ac < N) ? A[row * N + ac] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (br < N && col < N) ? B[br * N + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < N && col < N) C[row * N + col] = acc;
}

/* Deterministic nonce-seeded fill so a precomputed answer for a different nonce
 * does not reuse this run's inputs. */
__global__ void fill_seeded(float *buf, long n, unsigned int seed) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned int x = (unsigned int)(idx * 1103515245u + seed + 12345u);
        x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
        buf[idx] = (float)(x & 0xffffu) / 65536.0f;
    }
}

/* Find the kernel /proc entry whose PCI matches the CUDA-reported bus id and
 * extract "GPU UUID:" and "Model:". Returns 1 on success. */
static int read_kernel_identity(const char *pci_lower, char *uuid_out, size_t uuid_sz,
                                char *model_out, size_t model_sz) {
    uuid_out[0] = '\0';
    model_out[0] = '\0';
    const char *base = "/proc/driver/nvidia/gpus";
    DIR *d = opendir(base);
    if (!d) return 0;
    struct dirent *ent;
    char match_dir[512];
    match_dir[0] = '\0';
    while ((ent = readdir(d)) != NULL) {
        if (ent->d_name[0] == '.') continue;
        /* CUDA pci is like 0000:65:00.0; kernel dir is the same, case-insensitive */
        if (strcasecmp(ent->d_name, pci_lower) == 0) {
            snprintf(match_dir, sizeof(match_dir), "%s/%s/information", base, ent->d_name);
            break;
        }
    }
    closedir(d);
    if (match_dir[0] == '\0') return 0;
    FILE *f = fopen(match_dir, "r");
    if (!f) return 0;
    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        char *p;
        if ((p = strstr(line, "GPU UUID:")) != NULL) {
            p += strlen("GPU UUID:");
            while (*p == ' ' || *p == '\t') p++;
            char *e = p + strlen(p);
            while (e > p && (e[-1] == '\n' || e[-1] == '\r' || e[-1] == ' ' || e[-1] == '\t')) *--e = '\0';
            strncpy(uuid_out, p, uuid_sz - 1); uuid_out[uuid_sz - 1] = '\0';
        } else if ((p = strstr(line, "Model:")) != NULL) {
            p += strlen("Model:");
            while (*p == ' ' || *p == '\t') p++;
            char *e = p + strlen(p);
            while (e > p && (e[-1] == '\n' || e[-1] == '\r' || e[-1] == ' ' || e[-1] == '\t')) *--e = '\0';
            strncpy(model_out, p, model_sz - 1); model_out[model_sz - 1] = '\0';
        }
    }
    fclose(f);
    return uuid_out[0] != '\0';
}

static void json_escape(const char *in, char *out, size_t out_sz) {
    size_t o = 0;
    for (size_t i = 0; in[i] && o + 2 < out_sz; ++i) {
        char c = in[i];
        if (c == '"' || c == '\\') { out[o++] = '\\'; out[o++] = c; }
        else if (c == '\n' || c == '\r' || c == '\t') { out[o++] = ' '; }
        else out[o++] = c;
    }
    out[o] = '\0';
}

int main(int argc, char **argv) {
    const char *nonce = "";
    int device = 0;
    int mm_n = 8192;      /* matmul square dimension */
    int mm_iters = 30;    /* timed matmul iterations */
    int bw_iters = 50;    /* timed bandwidth iterations */

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--nonce") && i + 1 < argc) nonce = argv[++i];
        else if (!strcmp(argv[i], "--device") && i + 1 < argc) device = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--mm-n") && i + 1 < argc) mm_n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--mm-iters") && i + 1 < argc) mm_iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bw-iters") && i + 1 < argc) bw_iters = atoi(argv[++i]);
    }
    if (strlen(nonce) < 16) { emit_error(nonce, device, "bad_nonce"); return 2; }

    const char *master = getenv("LIUM_SIG_KEY");
    if (!master || !*master) master = LIUM_SIG_MASTER_KEY;

    CUDA_OK(cudaSetDevice(device), nonce, device, "set_device");

    cudaDeviceProp prop;
    CUDA_OK(cudaGetDeviceProperties(&prop, device), nonce, device, "get_props");
    char pci[32];
    CUDA_OK(cudaDeviceGetPCIBusId(pci, sizeof(pci), device), nonce, device, "get_pci");
    for (char *p = pci; *p; ++p) if (*p >= 'A' && *p <= 'Z') *p += 32; /* lowercase */

    char kern_uuid[128], kern_model[128];
    int have_kern = read_kernel_identity(pci, kern_uuid, sizeof(kern_uuid), kern_model, sizeof(kern_model));
    if (!have_kern) {
        /* Identity anchor is the whole point; a missing /proc view is reported,
         * not guessed. The validator decides how to weight a kernel-blind run. */
        kern_uuid[0] = '\0';
        snprintf(kern_model, sizeof(kern_model), "%s", prop.name);
    }

    unsigned int seed = 0;
    for (int i = 0; i < 8 && nonce[i]; ++i) {
        char c = nonce[i];
        unsigned int v = (c >= '0' && c <= '9') ? (c - '0')
                       : (c >= 'a' && c <= 'f') ? (c - 'a' + 10)
                       : (c >= 'A' && c <= 'F') ? (c - 'A' + 10) : 0;
        seed = (seed << 4) | v;
    }

    /* ---- matmul throughput ---- */
    long elems = (long)mm_n * mm_n;
    size_t bytes = (size_t)elems * sizeof(float);
    float *dA, *dB, *dC;
    CUDA_OK(cudaMalloc(&dA, bytes), nonce, device, "malloc_a");
    CUDA_OK(cudaMalloc(&dB, bytes), nonce, device, "malloc_b");
    CUDA_OK(cudaMalloc(&dC, bytes), nonce, device, "malloc_c");

    int fill_threads = 256;
    long fill_blocks = (elems + fill_threads - 1) / fill_threads;
    fill_seeded<<<fill_blocks, fill_threads>>>(dA, elems, seed);
    fill_seeded<<<fill_blocks, fill_threads>>>(dB, elems, seed ^ 0x9e3779b9u);
    CUDA_OK(cudaDeviceSynchronize(), nonce, device, "fill");

    dim3 block(TILE, TILE);
    dim3 grid((mm_n + TILE - 1) / TILE, (mm_n + TILE - 1) / TILE);

    for (int w = 0; w < 3; ++w) sgemm_tiled<<<grid, block>>>(dA, dB, dC, mm_n);  /* warmup */
    CUDA_OK(cudaDeviceSynchronize(), nonce, device, "warmup");

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int it = 0; it < mm_iters; ++it) sgemm_tiled<<<grid, block>>>(dA, dB, dC, mm_n);
    cudaEventRecord(t1);
    CUDA_OK(cudaEventSynchronize(t1), nonce, device, "mm_sync");
    float mm_ms = 0.0f; cudaEventElapsedTime(&mm_ms, t0, t1);
    double flop = 2.0 * (double)mm_n * mm_n * mm_n * mm_iters;
    double tflops = (mm_ms > 0.0) ? (flop / (mm_ms / 1000.0) / 1e12) : 0.0;

    /* ---- VRAM bandwidth (device-to-device copy) ---- */
    size_t bw_bytes = bytes; /* reuse the matmul-sized buffer */
    cudaEventRecord(t0);
    for (int it = 0; it < bw_iters; ++it)
        cudaMemcpy(dC, dA, bw_bytes, cudaMemcpyDeviceToDevice);
    cudaEventRecord(t1);
    CUDA_OK(cudaEventSynchronize(t1), nonce, device, "bw_sync");
    float bw_ms = 0.0f; cudaEventElapsedTime(&bw_ms, t0, t1);
    /* read + write => 2x bytes moved per copy */
    double gbps = (bw_ms > 0.0) ? (2.0 * (double)bw_bytes * bw_iters / (bw_ms / 1000.0) / 1e9) : 0.0;

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cudaEventDestroy(t0); cudaEventDestroy(t1);

    /* ---- format numbers as the exact strings we will both print and seal ---- */
    char tflops_s[32], gbps_s[32], ms_s[32];
    snprintf(tflops_s, sizeof(tflops_s), "%.3f", tflops);
    snprintf(gbps_s, sizeof(gbps_s), "%.3f", gbps);
    snprintf(ms_s, sizeof(ms_s), "%.3f", mm_ms);

    /* seal key = HMAC(master, kernel_uuid); the validator derives the same from
     * the UUID it independently expects, so a fabricated UUID -> unverifiable. */
    uint8_t perkey[32];
    hmac_sha256((const uint8_t *)master, strlen(master),
                (const uint8_t *)kern_uuid, strlen(kern_uuid), perkey);

    char msg[512];
    snprintf(msg, sizeof(msg), "%s|%d|%s|%s|%s|%s|%s",
             nonce, device, kern_uuid, pci, tflops_s, gbps_s, ms_s);
    char sig[65];
    hmac_sha256_hex(perkey, 32, (const uint8_t *)msg, strlen(msg), sig);

    char uuid_j[160], model_j[160], pci_j[64], msg_j[600];
    json_escape(kern_uuid, uuid_j, sizeof(uuid_j));
    json_escape(kern_model, model_j, sizeof(model_j));
    json_escape(pci, pci_j, sizeof(pci_j));
    json_escape(msg, msg_j, sizeof(msg_j));

    /* `msg` is the authoritative, signed record: the validator parses the
     * numbers/identity OUT of it (not the cosmetic numeric fields) and verifies
     * `sig` over it, so C/Python float formatting can never diverge. */
    printf("{\"gpu_sig\": 1, \"ok\": true, \"nonce\": \"%s\", \"device\": %d, "
           "\"kernel_uuid\": \"%s\", \"kernel_model\": \"%s\", \"cuda_name\": \"%s\", "
           "\"pci\": \"%s\", \"vram_mb\": %llu, \"mm_n\": %d, \"mm_iters\": %d, "
           "\"tflops\": %s, \"gbps\": %s, \"ms\": %s, \"have_kernel\": %s, "
           "\"msg\": \"%s\", \"sig\": \"%s\"}\n",
           nonce, device, uuid_j, model_j, prop.name, pci_j,
           (unsigned long long)(prop.totalGlobalMem / (1024ULL * 1024ULL)),
           mm_n, mm_iters, tflops_s, gbps_s, ms_s, have_kern ? "true" : "false",
           msg_j, sig);
    fflush(stdout);
    return 0;
}
