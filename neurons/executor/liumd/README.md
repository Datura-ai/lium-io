# liumd

The executor image installs the vendored `liumd-amd64` or `liumd-arm64` binary at
`/usr/local/bin/liumd`.

Both files are the unsigned build artifacts built from the liumd merged source `6ea408d468b9`:

| binary | sha256 |
| --- | --- |
| `liumd-amd64` | `23ce96f5486a7a19e9dff0a4acbe8f6558cce7d681e1ed3d12ecc1375bde7b56` |
| `liumd-arm64` | `d6e4755f749116a8347b3ab1a0869e0db2fc7b047c9151e2809f085d86de347d` |

`provenance-amd64.json`/`provenance-arm64.json` are the same run's metadata, kept beside the
binaries for traceability. `manifest.json` is the validator refresh manifest: it names the
`build_sha` and the sha256 of each binary. `manifest.json.sig` is the ed25519 signature over the
manifest's exact bytes, made by the liumd release key (ed25519:04d614eaf59dbbb8) when a `liumd-v*`
release is tagged. The validator's refresh pins that key, fetches the signature with the manifest and
installs nothing unless it verifies; a missing or foreign signature keeps the node's current binary.
The image digest is still the normal delivery trust boundary.

Refresh both binaries, both provenance files, `manifest.json` and `manifest.json.sig` together from
the same signed liumd release.
