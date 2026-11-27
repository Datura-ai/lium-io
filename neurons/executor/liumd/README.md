# liumd

The executor image installs the vendored `liumd-amd64` or `liumd-arm64` binary at
`/usr/local/bin/liumd`.

Both files are the unsigned build artifacts built from the liumd merged source `e5611e3a3d9c`:

| binary | sha256 |
| --- | --- |
| `liumd-amd64` | `8453893fbb76b8bd47ff09d00d4b8a37ba59b55bdd06e32491eb82f7ac8f9444` |
| `liumd-arm64` | `564d8a42da76df90dfaabc3e3a7053298a83ddad4c15cbf45ffed547870cedc5` |

This build adds `data_sysbox_layer_reset_version` (lium-platform#3411) and
`hard_disk_unsized_containers` (lium-platform#3405).
It also keeps an NVIDIA GB10 card whose driver reports its memory as Not Supported, and gives the
local-volume walk its own time budget after the layer reads. Each pod whose disk size is unknown
carries the reason.

`provenance-amd64.json`/`provenance-arm64.json` are the same run's metadata, kept beside the
binaries for traceability. `manifest.json` is the validator refresh manifest: it names the
`build_sha` and the sha256 of each binary. The image digest is still the normal delivery trust
boundary.

Refresh both binaries and both provenance files from the same liumd source build.
