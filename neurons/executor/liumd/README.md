# liumd

The executor image installs the vendored `liumd-amd64` or `liumd-arm64` binary at
`/usr/local/bin/liumd`.

Both files are the unsigned build artifacts built from the liumd merged source `00dfc5278593`:

| binary | sha256 |
| --- | --- |
| `liumd-amd64` | `ffa2512bb114ac0fb5c8b5a37e54a2dce207a3f804b53b224c43c0aaf9b3e82d` |
| `liumd-arm64` | `791bbfbede1cd97c3b65ed021ee564ca416fcac7d87647c098e3dff5be055a07` |

`provenance-amd64.json`/`provenance-arm64.json` are the same run's metadata, kept beside the
binaries for traceability. `manifest.json` is the validator refresh manifest: it names the
`build_sha` and the sha256 of each binary. The image digest is still the normal delivery trust
boundary.

Refresh both binaries and both provenance files from the same liumd source build.
