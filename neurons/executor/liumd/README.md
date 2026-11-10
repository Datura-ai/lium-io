# liumd

The executor image installs the vendored `liumd-amd64` or `liumd-arm64` binary at
`/usr/local/bin/liumd`.

Both files are the unsigned CI artifacts built from the lium-platform main source head
`49a75a372a9c`:

| binary | sha256 |
| --- | --- |
| `liumd-amd64` | `0cb8cd0d86ffa1cd1dd7b2ecb7f728aa1db68e0521691a97fbbe395e865e7481` |
| `liumd-arm64` | `ab784c232f09a4717038d222b7323681ee667d968459682f54f0b33791eaa4c5` |

`provenance-amd64.json`/`provenance-arm64.json` are the same run's metadata, kept beside the
binaries for traceability only. The build emits no release manifest. There is no source manifest,
signature, or verification step at build or runtime; the image digest is the delivery trust
boundary.

Refresh both binaries and both provenance files from the same lium-platform main build.
