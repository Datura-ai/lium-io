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
`build_sha` and the sha256 of each binary. `manifest.json.sig` is the ed25519 signature over the
manifest's exact bytes, made by the liumd release key (ed25519:04d614eaf59dbbb8) when a `liumd-v*`
release is tagged. The validator's refresh pins that key, fetches the signature with the manifest and
installs nothing unless it verifies; a missing or foreign signature keeps the node's current binary.
The signed manifest also carries `release`, the release tag's X.Y.Z. The validator keeps the highest release it has
accepted, for each node and overall, and refuses a lower one, so an older signed manifest cannot roll nodes back.
It records that release before it installs anything, and installs nothing when it cannot. The validator image
also carries this `manifest.json` and its signature, and the release of that copy is a floor a Redis flush cannot
lower, so vendoring a signed release here raises it with no other edit.
The image digest is still the normal delivery trust boundary.

Refresh both binaries, both provenance files, `manifest.json` and `manifest.json.sig` together from
the same signed liumd release.
