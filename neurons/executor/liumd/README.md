# liumd

The executor image installs the vendored `liumd-amd64` or `liumd-arm64` binary at
`/usr/local/bin/liumd`.

Both files are the unsigned build artifacts built from the liumd merged source `bb90cf69caa5`:

| binary | sha256 |
| --- | --- |
| `liumd-amd64` | `ea491db17e071bdd4bac2c228e5ca1c130b5b373d78e98e7ab0fff82a0993e79` |
| `liumd-arm64` | `6fe276645c2dc2228ecfc058e42ecf61260d5ab8c93e82a80b4a45514f8b09dd` |

This build adds `data_sysbox_layer_reset_version` (lium-platform#3411) and
`hard_disk_unsized_containers` (lium-platform#3405).

`provenance-amd64.json`/`provenance-arm64.json` are the same run's metadata, kept beside the
binaries for traceability. `manifest.json` is the validator refresh manifest: it names the
`build_sha` and the sha256 of each binary. The image digest is still the normal delivery trust
boundary.

Refresh both binaries and both provenance files from the same liumd source build.
