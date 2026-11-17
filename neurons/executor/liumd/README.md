# liumd

The executor image installs the vendored `liumd-amd64` or `liumd-arm64` binary at
`/usr/local/bin/liumd`.

Both files are the unsigned CI artifacts built from the liumd source head `25f8f05508ac`:

| binary | sha256 |
| --- | --- |
| `liumd-amd64` | `1b88754fd33a515cd98e68c94a21b0aaa833f1a0e06d56e5176da3876b6d1555` |
| `liumd-arm64` | `cc12eda4fb8042a6aa60d957e52a8ed674d67d9bda29f26ec6d4a765ccba4625` |

`provenance-amd64.json`/`provenance-arm64.json` are the same run's metadata, kept beside the
binaries for traceability. `manifest.json` is the validator refresh manifest: it names the
`build_sha` and the sha256 of each binary. The image digest is still the normal delivery trust
boundary.

Refresh both binaries and both provenance files from the same liumd source build.
