# liumd

The executor image installs the vendored `liumd-amd64` or `liumd-arm64` binary at
`/usr/local/bin/liumd`.

Both files are the unsigned CI artifacts built from the lium-platform main source head
`53999a8d15b3`:

| binary | sha256 |
| --- | --- |
| `liumd-amd64` | `3566c44a3f0814351254f74d45eee4e1185737e3f4770003276a78e43d2e8f6b` |
| `liumd-arm64` | `276f72398dea9bb07025f2255c5a613cb8d9e9009bd0ea82b07706dfd516e8a6` |

`provenance-amd64.json`/`provenance-arm64.json` are the same run's metadata, kept beside the
binaries for traceability only. The build emits no release manifest. There is no source manifest,
signature, or verification step at build or runtime; the image digest is the delivery trust
boundary.

Refresh both binaries and both provenance files from the same lium-platform main build.
