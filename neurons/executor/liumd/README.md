# liumd

The executor image installs the vendored `liumd-amd64` or `liumd-arm64` binary at
`/usr/local/bin/liumd`.

Both files are built from
[Datura-ai/lium-platform#2222](https://github.com/Datura-ai/lium-platform/pull/2222)
(source head `c85e86b4a4aa`, merge `f4fbaa9d987c`, local keyless `ci/static_build.sh`):

| binary | sha256 |
| --- | --- |
| `liumd-amd64` | `62f414165c349d2c0c7ee9ee9f6bf6f75c9073aabb4b837973545311666f2e09` |
| `liumd-arm64` | `8d1adbc1b440f52c02dc658d040de154e63b49b8a25e1d872df7b1b2003a0afa` |

`release-amd64.json`/`release-arm64.json` and `provenance-amd64.json`/`provenance-arm64.json`
are the same build's metadata, kept beside the binaries for traceability only. There is no source
manifest, signature, or verification step at build or runtime; the image digest is the delivery
trust boundary.

Refresh both binaries and both metadata pairs from the same lium-platform source build.
