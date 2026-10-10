#!/bin/sh
# /usr/local/bin/liumd in the executor image. The binary is a keyless dev build, whose baked child
# manifest pins no digests, so it would refuse every child it spawns; the image hashes its own
# children at build (/etc/liumd/children.json) and this hands that manifest over. A signed release
# build ignores LIUMD_CHILD_MANIFEST_FILE and trusts only the manifest it baked.
LIUMD_CHILD_MANIFEST_FILE=/etc/liumd/children.json
export LIUMD_CHILD_MANIFEST_FILE
exec /usr/local/lib/liumd/liumd "$@"
