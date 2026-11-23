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
