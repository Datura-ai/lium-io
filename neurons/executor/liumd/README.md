# liumd

The executor image installs the vendored `liumd-amd64` or `liumd-arm64` binary at
`/usr/local/bin/liumd`.

Both files are the unsigned build artifacts built from the liumd merged source `6ea408d468b9`:

| binary | sha256 |
| --- | --- |
| `liumd-amd64` | `23ce96f5486a7a19e9dff0a4acbe8f6558cce7d681e1ed3d12ecc1375bde7b56` |
| `liumd-arm64` | `d6e4755f749116a8347b3ab1a0869e0db2fc7b047c9151e2809f085d86de347d` |

`provenance-amd64.json`/`provenance-arm64.json` are the same run's metadata, kept beside the
binaries for traceability. `manifest.json` is the validator refresh manifest: it names the
`build_sha` and the sha256 of each binary. The image digest is still the normal delivery trust
boundary.

Refresh both binaries and both provenance files from the same liumd source build.
