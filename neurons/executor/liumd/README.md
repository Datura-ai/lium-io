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
