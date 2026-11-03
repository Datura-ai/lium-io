#!/bin/sh
# Build-time sshd configuration of the executor image (run by the Dockerfile after the
# executor's own sshd_config lines are appended), and `sshd_setup.sh start`, which run.sh
# calls before sshd starts to pick the per-source limits for this host.
#
# Per-source limits (DAH-5323, ticket-0870): a scanner that holds ~30 unauthenticated
# connections pushed sshd past MaxStartups 10:30:100, which then dropped about half of all new
# connections, the validator's included. The drop-in caps one source at PerSourceMaxStartups 10
# pending connections, turns PerSourcePenalties on (repeat failures from one address are refused
# for a while), cuts LoginGraceTime to 20s and raises MaxStartups to 60:30:200. The validator
# logs in with one valid key in well under a second, so it earns no penalty and never comes near
# the grace time.
#
# OpenSSH 9.8+ enables PerSourcePenalties: a source address whose connections keep failing
# authentication is refused for a while ("Not allowed at this time", or the connection is
# closed during the handshake). On an executor whose port 2200 collapses every client into
# one source address — a CVM behind QEMU slirp (every connection arrives from 10.0.2.2), a
# Docker userland-proxy host — scanner noise fills that one bucket and the validator is
# locked out; after a silent hour the backend fires EXECUTOR_INACTIVE_MID_RENTAL
# (DAH-3236, ticket-0287, ticket-0302). Such hosts turn both per-source limits off:
# `start` puts the off drop-in in place when EXECUTOR_SSH_SOURCE_LIMITS=false or the executor
# runs in a CVM (the dstack socket is there; QEMU slirp shows sshd 10.0.2.2 for everyone), and
# sshd_log_receiver.py puts it in place (and reloads sshd) on the first connection from a private
# address (sshd's VERBOSE "Connection from" line, before authentication), which is what such a host
# shows for every client, so scanners cannot fill PerSourceMaxStartups ahead of the validator. Private
# ranges and the validator's own addresses are also on PerSourcePenaltyExemptList, never penalised.
#
# The directive is unknown to OpenSSH < 9.8, where sshd refuses to start on it, so it is
# written only when this sshd accepts it. It goes into a drop-in that sshd_config's
# `Include /etc/ssh/sshd_config.d/*.conf` reads ahead of the `Match User liumuser` block
# (a line appended after that block would become part of the Match).
#
# Whatever the version, the drop-in also sets LogLevel VERBOSE (DAH-4629): every login then logs
# "Accepted publickey for <user> from <ip> port <n> ssh2: <type> SHA256:<fingerprint>", the one record
# that names the key behind a session into this container. run.sh sends sshd's log to a file the
# validator reads each cycle.
#
# Whatever the sshd version, the last step is `sshd -T` over the rendered config: the image
# build fails on any directive sshd would refuse at container start.
set -eu

SSHD_CONFIG="${SSHD_CONFIG:-/etc/ssh/sshd_config}"
SSHD_CONFIG_DIR="${SSHD_CONFIG_DIR:-/etc/ssh/sshd_config.d}"
SSHD_PRIVSEP_DIR="${SSHD_PRIVSEP_DIR:-/run/sshd}"
DROP_IN="$SSHD_CONFIG_DIR/lium.conf"
# written at build, copied in by `start`; sshd keeps the first value it reads and 00- sorts before lium.conf
SOURCE_OFF="${SSHD_SOURCE_OFF:-/etc/ssh/lium-source-limits-off.conf}"
SOURCE_OFF_IN="$SSHD_CONFIG_DIR/00-lium-source-limits-off.conf"

if [ "${1:-}" = start ]; then
    limits="$(printf '%s' "${EXECUTOR_SSH_SOURCE_LIMITS:-true}" | tr '[:upper:]' '[:lower:]')"
    if [ ! -f "$SOURCE_OFF" ]; then
        echo "sshd_setup: this sshd has no per-source limits"
    elif [ "$limits" = false ] || [ "$limits" = 0 ] || [ "$limits" = no ] || [ -S "${DSTACK_SOCKET_PATH:-/var/run/dstack.sock}" ]; then
        cp "$SOURCE_OFF" "$SOURCE_OFF_IN"
        echo "sshd_setup: per-source limits off (EXECUTOR_SSH_SOURCE_LIMITS=$limits, or a CVM)"
    else
        rm -f "$SOURCE_OFF_IN"
        echo "sshd_setup: per-source limits on"
    fi
    exit 0
fi

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
rm -f "$SOURCE_OFF_IN"
printf '# written by sshd_setup.sh at image build (DAH-3236, DAH-4629, DAH-5323); that script says why\nLogLevel VERBOSE\nLoginGraceTime 20\nMaxStartups 60:30:200\n' >"$DROP_IN"
expect() { # expect <sshd -T line>...: each line must be in the rendered config
    out="$(sshd_test)"
    for line in "$@"; do
        if ! printf '%s\n' "$out" | grep -qx "$line"; then
            echo "sshd_setup: '$line' is not in effect; does $SSHD_CONFIG include $SSHD_CONFIG_DIR/*.conf?" >&2
            exit 1
        fi
    done
}
if [ "$supported" = yes ]; then
    # never penalised: the validator's own addresses (a slow or early login must not lock it out) and the
    # private, CGNAT and loopback ranges a NAT, Docker gateway or tunnel shows for every client
    exempt="10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,100.64.0.0/10,127.0.0.0/8,::1/128,fc00::/7"
    sources="$(python3 - "$(dirname "$0")/src/core" <<'PY'
import ast, pathlib, sys

core = pathlib.Path(sys.argv[1])  # same choice as core.config._resolve_validator_ssh_sources


def literal(path, name):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)


override = core / "config_override.py"
if override.exists():
    print(literal(override, "_VALIDATOR_SSH_SOURCES") or "")
else:
    print(literal(core / "config.py", "_BUILTIN_VALIDATOR_SSH_SOURCES") or "")
PY
)"
    printf 'PerSourceMaxStartups 10\nPerSourcePenalties yes\nPerSourcePenaltyExemptList %s\n' "$exempt${sources:+,$sources}" >>"$DROP_IN"
    expect 'persourcemaxstartups 10'
    if sshd_test | grep -qx 'persourcepenalties no'; then
        echo "sshd_setup: PerSourcePenalties yes from $DROP_IN is not in effect" >&2
        exit 1
    fi
    printf '# copied into sshd_config.d by `sshd_setup.sh start` on hosts where every client is one address (DAH-3236)\nPerSourceMaxStartups none\nPerSourcePenalties no\n' >"$SOURCE_OFF"
    cp "$SOURCE_OFF" "$SOURCE_OFF_IN"
    expect 'persourcemaxstartups none' 'persourcepenalties no' 'loglevel VERBOSE'
    rm -f "$SOURCE_OFF_IN"
    echo "sshd_setup: per-source limits on, off drop-in checked ($sshd_version)"
else
    rm -f "$SOURCE_OFF"
    echo "sshd_setup: this sshd predates PerSourcePenalties, no per-source limits ($sshd_version)"
fi
expect 'loglevel VERBOSE' 'logingracetime 20' 'maxstartups 60:30:200'

sshd_test >/dev/null
echo "sshd_setup: sshd -T accepts $SSHD_CONFIG"
