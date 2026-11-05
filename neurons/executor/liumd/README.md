# liumd

The executor image installs the vendored `liumd-amd64` or `liumd-arm64` binary at
`/usr/local/bin/liumd`.

Both files are the unsigned CI artifacts of
[Datura-ai/lium-platform#2222](https://github.com/Datura-ai/lium-platform/pull/2222)
(source head `d040d1022def`, merge `3b66b4f94aec`,
[run 41797756992](https://github.com/Datura-ai/lium-platform/actions/runs/41797756992)):

| binary | sha256 |
| --- | --- |
| `liumd-amd64` | `9fea1b3a0edaa5b11eae92b04a01e804c50b80520682f6ab1917214899c19ad2` |
| `liumd-arm64` | `7a5450395d4c5e1102053db565e056407423243c0f16ac22657b5f0d1dcf3aee` |

`release-amd64.json`/`release-arm64.json` and `provenance-amd64.json`/`provenance-arm64.json`
are the same run's metadata, kept beside the binaries for traceability only. There is no source
manifest, signature, or verification step at build or runtime; the image digest is the delivery
trust boundary.

Refresh both binaries and both metadata pairs from the same lium-platform CI run.
