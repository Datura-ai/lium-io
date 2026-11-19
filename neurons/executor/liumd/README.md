# liumd

The executor image installs the vendored `liumd-amd64` or `liumd-arm64` binary at
`/usr/local/bin/liumd`.

Both files are the unsigned build artifacts built from the liumd merged source `6ea408d468b9`:

| binary | sha256 |
| --- | --- |
| `liumd-amd64` | `6721baf2623c41a639459fbbfeeff7f53fca49ae71de5ff8dbe2b2cccddc8710` |
| `liumd-arm64` | `ba99d6078f95f771e173cdc5cdf2a0efd396972f7bd2dd61ad407c9c07ee496a` |

`provenance-amd64.json`/`provenance-arm64.json` are the same run's metadata, kept beside the
binaries for traceability. `manifest.json` is the validator refresh manifest: it names the
`build_sha` and the sha256 of each binary. The image digest is still the normal delivery trust
boundary.

Refresh both binaries and both provenance files from the same liumd source build.
