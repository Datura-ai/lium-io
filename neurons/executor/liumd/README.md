# liumd

The executor image installs the vendored `liumd-amd64` or `liumd-arm64` binary at
`/usr/local/bin/liumd`.

Both files are the unsigned CI artifacts built from the liumd source head `f92fb6473f61`:

| binary | sha256 |
| --- | --- |
| `liumd-amd64` | `03fd59d996a435eb42bc0be59b3dbe77a107ff9416f40056267fc775f8ed57e5` |
| `liumd-arm64` | `b0a13e8b20f1eb99c07103fdf6b2594cbbe66f8e4f3a710eb748caa95a479657` |

`provenance-amd64.json`/`provenance-arm64.json` are the same run's metadata, kept beside the
binaries for traceability. `manifest.json` is the validator refresh manifest: it names the
`build_sha` and the sha256 of each binary. The image digest is still the normal delivery trust
boundary.

Refresh both binaries and both provenance files from the same liumd source build.
