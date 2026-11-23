# liumd

The executor image installs the vendored `liumd-amd64` or `liumd-arm64` binary at
`/usr/local/bin/liumd`.

Both files are the unsigned build artifacts built from the liumd merged source `aa6045bfe0f8`:

| binary | sha256 |
| --- | --- |
| `liumd-amd64` | `fff7a3e72f2860ba7fbd0925963eea710441827d80acd8c04e401c4be671c037` |
| `liumd-arm64` | `ef6054b879961c7bf53e266d83d6bead7ed5f99280935a8494fec8260835e9dc` |

`provenance-amd64.json`/`provenance-arm64.json` are the same run's metadata, kept beside the
binaries for traceability. `manifest.json` is the validator refresh manifest: it names the
`build_sha` and the sha256 of each binary. The image digest is still the normal delivery trust
boundary.

Refresh both binaries and both provenance files from the same liumd source build.
