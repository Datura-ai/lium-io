/*
 * Self-contained SHA-256 + HMAC-SHA256 (public-domain style, no OpenSSL).
 *
 * Header-only so gpu_sig.cu builds with just nvcc/gcc and no -lssl on the
 * executor image. The seal format MUST stay byte-identical to the validator's
 * Python HMAC (neurons/validators/src/services/gpu_signature.py) — see
 * test_seal.c for the interop check against Python's hmac module.
 */
#ifndef LIUM_SHA256_H
#define LIUM_SHA256_H

#include <stdint.h>
#include <string.h>

typedef struct {
    uint32_t state[8];
    uint64_t bitlen;
    uint8_t data[64];
    uint32_t datalen;
} sha256_ctx;

static const uint32_t sha256_k[64] = {
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
    0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
    0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
    0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
    0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
    0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
};

#define SHA256_ROTR(a,b) (((a) >> (b)) | ((a) << (32-(b))))

static void sha256_transform(sha256_ctx *ctx, const uint8_t *data) {
    uint32_t a,b,c,d,e,f,g,h,i,j,t1,t2,m[64];
    for (i = 0, j = 0; i < 16; ++i, j += 4)
        m[i] = ((uint32_t)data[j] << 24) | ((uint32_t)data[j+1] << 16) |
               ((uint32_t)data[j+2] << 8) | ((uint32_t)data[j+3]);
    for (; i < 64; ++i)
        m[i] = (SHA256_ROTR(m[i-2],17) ^ SHA256_ROTR(m[i-2],19) ^ (m[i-2] >> 10)) + m[i-7] +
               (SHA256_ROTR(m[i-15],7) ^ SHA256_ROTR(m[i-15],18) ^ (m[i-15] >> 3)) + m[i-16];
    a=ctx->state[0]; b=ctx->state[1]; c=ctx->state[2]; d=ctx->state[3];
    e=ctx->state[4]; f=ctx->state[5]; g=ctx->state[6]; h=ctx->state[7];
    for (i = 0; i < 64; ++i) {
        t1 = h + (SHA256_ROTR(e,6) ^ SHA256_ROTR(e,11) ^ SHA256_ROTR(e,25)) + ((e & f) ^ (~e & g)) + sha256_k[i] + m[i];
        t2 = (SHA256_ROTR(a,2) ^ SHA256_ROTR(a,13) ^ SHA256_ROTR(a,22)) + ((a & b) ^ (a & c) ^ (b & c));
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    ctx->state[0]+=a; ctx->state[1]+=b; ctx->state[2]+=c; ctx->state[3]+=d;
    ctx->state[4]+=e; ctx->state[5]+=f; ctx->state[6]+=g; ctx->state[7]+=h;
}

static void sha256_init(sha256_ctx *ctx) {
    ctx->datalen = 0; ctx->bitlen = 0;
    ctx->state[0]=0x6a09e667u; ctx->state[1]=0xbb67ae85u; ctx->state[2]=0x3c6ef372u; ctx->state[3]=0xa54ff53au;
    ctx->state[4]=0x510e527fu; ctx->state[5]=0x9b05688cu; ctx->state[6]=0x1f83d9abu; ctx->state[7]=0x5be0cd19u;
}

static void sha256_update(sha256_ctx *ctx, const uint8_t *data, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        ctx->data[ctx->datalen++] = data[i];
        if (ctx->datalen == 64) {
            sha256_transform(ctx, ctx->data);
            ctx->bitlen += 512;
            ctx->datalen = 0;
        }
    }
}

static void sha256_final(sha256_ctx *ctx, uint8_t *hash) {
    uint32_t i = ctx->datalen;
    ctx->data[i++] = 0x80;
    if (ctx->datalen < 56) {
        while (i < 56) ctx->data[i++] = 0x00;
    } else {
        while (i < 64) ctx->data[i++] = 0x00;
        sha256_transform(ctx, ctx->data);
        memset(ctx->data, 0, 56);
    }
    ctx->bitlen += (uint64_t)ctx->datalen * 8;
    for (int k = 0; k < 8; ++k)
        ctx->data[63 - k] = (uint8_t)(ctx->bitlen >> (8 * k));
    sha256_transform(ctx, ctx->data);
    for (i = 0; i < 4; ++i)
        for (int k = 0; k < 8; ++k)
            hash[i + k * 4] = (uint8_t)((ctx->state[k] >> (24 - i * 8)) & 0xff);
}

static void sha256_bytes(const uint8_t *data, size_t len, uint8_t out[32]) {
    sha256_ctx ctx; sha256_init(&ctx); sha256_update(&ctx, data, len); sha256_final(&ctx, out);
}

/* HMAC-SHA256(key, msg) -> out[32] */
static void hmac_sha256(const uint8_t *key, size_t keylen,
                        const uint8_t *msg, size_t msglen, uint8_t out[32]) {
    uint8_t k[64], k_ipad[64], k_opad[64], inner[32];
    memset(k, 0, 64);
    if (keylen > 64) {
        sha256_bytes(key, keylen, k);
    } else {
        memcpy(k, key, keylen);
    }
    for (int i = 0; i < 64; ++i) { k_ipad[i] = k[i] ^ 0x36; k_opad[i] = k[i] ^ 0x5c; }
    sha256_ctx ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, k_ipad, 64);
    sha256_update(&ctx, msg, msglen);
    sha256_final(&ctx, inner);
    sha256_init(&ctx);
    sha256_update(&ctx, k_opad, 64);
    sha256_update(&ctx, inner, 32);
    sha256_final(&ctx, out);
}

static void hmac_sha256_hex(const uint8_t *key, size_t keylen,
                            const uint8_t *msg, size_t msglen, char out_hex[65]) {
    uint8_t mac[32];
    hmac_sha256(key, keylen, msg, msglen, mac);
    static const char *hx = "0123456789abcdef";
    for (int i = 0; i < 32; ++i) {
        out_hex[i * 2] = hx[(mac[i] >> 4) & 0xf];
        out_hex[i * 2 + 1] = hx[mac[i] & 0xf];
    }
    out_hex[64] = '\0';
}

#endif /* LIUM_SHA256_H */
