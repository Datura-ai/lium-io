# liumd

The executor image installs the vendored `liumd-amd64` or `liumd-arm64` binary at
`/usr/local/bin/liumd`.

Both files are the unsigned CI artifacts of
[Datura-ai/lium-platform#2222](https://github.com/Datura-ai/lium-platform/pull/2222)
(source head `698c7bf26fac`, merge `2a213566629d`,
[run 42020290731](https://github.com/Datura-ai/lium-platform/actions/runs/42020290731)):

| binary | sha256 |
| --- | --- |
| `liumd-amd64` | `3f8d0ca8e9641b4be5f9fbd9b057a848b07ad2bd31047f0b0a6c6a86b47fefe9` |
| `liumd-arm64` | `0188698f51bcca9bedde34935639b6295187a762d2ef846643e16d1e4c98ff01` |

`release-amd64.json`/`release-arm64.json` and `provenance-amd64.json`/`provenance-arm64.json`
are the same run's metadata, kept beside the binaries for traceability only. There is no source
manifest, signature, or verification step at build or runtime; the image digest is the delivery
trust boundary.

Refresh both binaries and both metadata pairs from the same lium-platform source build.
