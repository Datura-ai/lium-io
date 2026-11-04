# liumd

The executor image installs the vendored `liumd-amd64` or `liumd-arm64` binary at
`/usr/local/bin/liumd`. There is no source manifest, signature, or verification step; the image
digest is the delivery trust boundary.

Replace both binaries from the lium-platform CI artifact when refreshing this directory.
