# liumd

The executor image installs the vendored `liumd-amd64` or `liumd-arm64` binary at
`/usr/local/bin/liumd`.

Both files are the unsigned build artifacts built from the liumd merged source `6948d46d68fa`:

| binary | sha256 |
| --- | --- |
| `liumd-amd64` | `b50d16966fdab64c04f0a4a00e5fb7364967aa310675b01ea9d28c1f91e4c4e8` |
| `liumd-arm64` | `d12d7a884a7dc2af1e67b3dd9316008cb848bc37ed8255dc944f1544e997039c` |

`provenance-amd64.json`/`provenance-arm64.json` are the same run's metadata, kept beside the
binaries for traceability. `manifest.json` is the validator refresh manifest: it names the
`build_sha` and the sha256 of each binary. The image digest is still the normal delivery trust
boundary.

Refresh both binaries and both provenance files from the same liumd source build.
