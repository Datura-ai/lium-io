# liumd

The executor image installs the vendored `liumd-amd64` or `liumd-arm64` binary at
`/usr/local/bin/liumd`.

Both files are the unsigned build artifacts built from the liumd merged source `7f4249f9c6fe`:

| binary | sha256 |
| --- | --- |
| `liumd-amd64` | `042583398bdd26daa1e789815496dc33c91e051f5ab0551f4ac262c5cf66f1da` |
| `liumd-arm64` | `6c5977e3713918c3910e32a8ec223ed0f20be1a04eb499f4941e38bada4d889d` |

This build adds `data_sysbox_layer_reset_version` (lium-platform#3411) and
`hard_disk_unsized_containers` (lium-platform#3405).
It also keeps an NVIDIA GB10 card whose driver reports its memory as Not Supported, and gives the
local-volume walk its own time budget after the layer reads.

`provenance-amd64.json`/`provenance-arm64.json` are the same run's metadata, kept beside the
binaries for traceability. `manifest.json` is the validator refresh manifest: it names the
`build_sha` and the sha256 of each binary. The image digest is still the normal delivery trust
boundary.

Refresh both binaries and both provenance files from the same liumd source build.
