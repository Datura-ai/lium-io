# liumd

The executor image installs the vendored `liumd-amd64` or `liumd-arm64` binary at
`/usr/local/bin/liumd`.

Both files are the unsigned build artifacts built from the liumd merged source `1ab1c2cf2dab`:

| binary | sha256 |
| --- | --- |
| `liumd-amd64` | `ca846779772b7bec3da38f6c6a9f27d8519fb84e62263f27399da402f8a76a45` |
| `liumd-arm64` | `ed17105758fc683a91a31e15aae7f114ee2c91118aec434867fca8776079777c` |

This build adds `data_sysbox_layer_reset_version` (lium-platform#3411) and
`hard_disk_unsized_containers` (lium-platform#3405).
It also keeps an NVIDIA GB10 card whose driver reports its memory as Not Supported.

`provenance-amd64.json`/`provenance-arm64.json` are the same run's metadata, kept beside the
binaries for traceability. `manifest.json` is the validator refresh manifest: it names the
`build_sha` and the sha256 of each binary. The image digest is still the normal delivery trust
boundary.

Refresh both binaries and both provenance files from the same liumd source build.
