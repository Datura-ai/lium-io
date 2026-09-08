/*
 * Host-only interop harness for the gpu_sig seal (DAH-3137).
 *
 * Reproduces the exact key derivation + seal the CUDA binary uses, with no
 * CUDA dependency, so CI/dev can assert byte-parity with the validator's
 * Python HMAC (neurons/validators/tests/test_gpu_signature.py).
 *
 *   make test_seal
 *   ./test_seal <master_key> <kernel_uuid> "<nonce|device|uuid|pci|tflops|gbps|ms>"
 *
 * Prints the lowercase hex seal on one line.
 */
#include <stdio.h>
#include <string.h>
#include "sha256.h"

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <master_key> <kernel_uuid> <message>\n", argv[0]);
        return 2;
    }
    const char *master = argv[1];
    const char *uuid = argv[2];
    const char *msg = argv[3];

    uint8_t perkey[32];
    hmac_sha256((const uint8_t *)master, strlen(master),
                (const uint8_t *)uuid, strlen(uuid), perkey);
    char sig[65];
    hmac_sha256_hex(perkey, 32, (const uint8_t *)msg, strlen(msg), sig);
    printf("%s\n", sig);
    return 0;
}
