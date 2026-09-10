#!/bin/sh
# Build-time sshd configuration of the executor image (run by the Dockerfile after the
# executor's own sshd_config lines are appended).
#
# OpenSSH 9.8+ enables PerSourcePenalties: a source address whose connections keep failing
# authentication is refused for a while ("Not allowed at this time", or the connection is
# closed during the handshake). On an executor whose port 2200 collapses every client into
# one source address — a CVM behind QEMU slirp (every connection arrives from 10.0.2.2), a
# Docker userland-proxy host — scanner noise fills that one bucket and the validator is
# locked out; after a silent hour the backend fires EXECUTOR_INACTIVE_MID_RENTAL
# (DAH-3236, ticket-0287, ticket-0302).
#
# The directive is unknown to OpenSSH < 9.8, where sshd refuses to start on it, so it is
# written only when this sshd accepts it. It goes into a drop-in that sshd_config's
# `Include /etc/ssh/sshd_config.d/*.conf` reads ahead of the `Match User liumuser` block
# (a line appended after that block would become part of the Match).
#
# Whatever the sshd version, the last step is `sshd -T` over the rendered config: the image
# build fails on any directive sshd would refuse at container start.
set -eu

SSHD_CONFIG="${SSHD_CONFIG:-/etc/ssh/sshd_config}"
SSHD_CONFIG_DIR="${SSHD_CONFIG_DIR:-/etc/ssh/sshd_config.d}"
SSHD_PRIVSEP_DIR="${SSHD_PRIVSEP_DIR:-/run/sshd}"
DROP_IN="$SSHD_CONFIG_DIR/lium.conf"

# The image ships no host keys (run.sh generates them at start) and `sshd -T` exits
# without one, so the checks use a throwaway key. OpenSSH < 9.8's `sshd -T` also wants
# the privilege-separation directory the init script creates at start.
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
ssh-keygen -q -t ed25519 -N '' -f "$tmp/hostkey"
mkdir -p "$SSHD_PRIVSEP_DIR"
sshd_test() { sshd -T -f "$SSHD_CONFIG" -h "$tmp/hostkey" "$@"; }
sshd_version="$(ssh -V 2>&1 | head -n 1)"

# Does this sshd know the directive? Only "Bad configuration option: PerSourcePenalties"
# means no; any other failure is a broken config and fails the build here.
if probe="$(sshd_test -o PerSourcePenalties=no 2>&1 >/dev/null)"; then
    supported=yes
elif printf '%s\n' "$probe" | grep -q 'Bad configuration option: PerSourcePenalties'; then
    supported=no
else
    echo "sshd_setup: sshd -T failed: $probe" >&2
    exit 1
fi

mkdir -p "$SSHD_CONFIG_DIR"
if [ "$supported" = yes ]; then
    printf '# written by sshd_setup.sh at image build (DAH-3236); that script says why\nPerSourcePenalties no\n' >"$DROP_IN"
    if ! sshd_test | grep -qx 'persourcepenalties no'; then
        echo "sshd_setup: $DROP_IN is not in effect; does $SSHD_CONFIG include $SSHD_CONFIG_DIR/*.conf?" >&2
        exit 1
    fi
    echo "sshd_setup: PerSourcePenalties no ($sshd_version)"
else
    rm -f "$DROP_IN"
    echo "sshd_setup: this sshd predates PerSourcePenalties, nothing to turn off ($sshd_version)"
fi

sshd_test >/dev/null
echo "sshd_setup: sshd -T accepts $SSHD_CONFIG"
