# liumd

The executor image installs the vendored `liumd-amd64` or `liumd-arm64` binary at
`/usr/local/bin/liumd`.

Both files are the unsigned build artifacts built from the liumd merged source `d8e536959238`:

| binary | sha256 |
| --- | --- |
| `liumd-amd64` | `60fe2cb3f3ad239dcd8180e89b669b11e299d3a05c62272f7bd531913b858d68` |
| `liumd-arm64` | `ff43d228fa02ba1a17196e445bb490e827222c1868aa24f96d0a45e29149c8b0` |

This build adds `data_sysbox_layer_reset_version` (lium-platform#3411) and
`hard_disk_unsized_containers` (lium-platform#3405).
It also keeps an NVIDIA GB10 card whose driver reports its memory as Not Supported, and gives the
local-volume walk its own time budget after the layer reads. Each pod whose disk size is unknown
carries the reason, and liumd's own network probe never joins a renter's bridge network.
The embedded verifyx prover is the DAH-7126 build (a stalled package download still measures upload),
the same change `libverifyx.so` already carries; the other embedded provers are unchanged.

`provenance-amd64.json`/`provenance-arm64.json` are the same run's metadata, kept beside the
binaries for traceability. `manifest.json` is the validator refresh manifest: it names the
`build_sha` and the sha256 of each binary. The image digest is still the normal delivery trust
boundary.

Refresh both binaries and both provenance files from the same liumd source build.
