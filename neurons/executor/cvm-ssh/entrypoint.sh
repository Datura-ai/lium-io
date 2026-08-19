#!/bin/sh
# The renter CVM's guest-SSH service (DAH-2684): an sshd whose every session is opened
# inside the guest OS, not inside this container.
#
# How a session reaches the guest, in the order it happens:
#
#   1. sshd authenticates the renter here, in this container, against the keys the
#      backend put in this service's environment (LIUM_SSH_AUTHORIZED_KEYS). Those keys are
#      part of the measured compose, so the quote covers who may log in.
#   2. sshd chroots the session into /host — the guest's own root filesystem, bind-mounted
#      in by the compose (`/:/host`). From this point the session sees the guest's files
#      and nothing of this image; sftp and scp therefore read and write the guest.
#   3. The renter's login shell is /run/lium-ssh/shell INSIDE that chroot — a script this
#      entrypoint stages into the guest's /run — which moves the process into PID 1's
#      remaining namespaces (mount, uts, ipc, net, pid) with a statically linked busybox
#      nsenter staged beside it, then execs the guest's own /bin/sh. The guest ships no
#      nsenter and no bash: the shell is the guest's busybox ash, and that is the point —
#      the renter is on the guest, with what the guest has.
#
# What this needs from the compose, and refuses to start without: `pid: host` (so PID 1 is
# the guest's init), `privileged: true` (setns into another process's namespaces), the
# guest root at /host, and `network_mode: host` (so the port sshd binds is the guest port
# the host forwards). A container missing any of these could still start an sshd — one
# whose sessions land in this container — and that is exactly the wrong thing to hand a
# renter who was told they are logging into their CVM, so it is refused with the reason.
#
# Never a password path. Root here has no password, PasswordAuthentication is off, and the
# only identities accepted are the ones the compose named.

set -eu

log() { printf 'lium-cvm-ssh: %s\n' "$*" >&2; }
die() { log "refusing to start: $*"; exit 64; }

PORT="${LIUM_SSH_PORT:-2200}"
GUEST_ROOT="${LIUM_SSH_GUEST_ROOT:-/host}"
HOSTKEY_DIR="${LIUM_SSH_HOSTKEY_DIR:-/etc/ssh/hostkeys}"
AUTHORIZED_DIR=/etc/ssh/authorized
# Inside the guest's /run — a tmpfs, so it is writable on a read-only rootfs and gone on
# reboot, which is right: it is re-staged every time this container starts.
STAGE_REL=/run/lium-ssh
STAGE="$GUEST_ROOT$STAGE_REL"

# ---------------------------------------------------------------- preconditions

case "$PORT" in
    ''|*[!0-9]*) die "LIUM_SSH_PORT must be a port number, got '$PORT'" ;;
esac
[ "$PORT" -ge 1 ] && [ "$PORT" -le 65535 ] || die "LIUM_SSH_PORT out of range: $PORT"

[ "$(id -u)" = "0" ] || die "must run as root to enter the guest's namespaces"

# The guest root must be mounted, and it must be a DIFFERENT filesystem from ours: a compose
# that forgot the bind would leave /host an empty directory, and a session chrooted into an
# empty directory has no shell to run.
[ -d "$GUEST_ROOT/proc" ] && [ -x "$GUEST_ROOT/bin/sh" ] \
    || die "the guest root is not mounted at $GUEST_ROOT (the compose must bind '/:$GUEST_ROOT')"

# `pid: host` — PID 1 in our view must be the guest's init. Under a private pid namespace
# this very shell is PID 1, and nsenter would "enter" our own namespaces.
[ "$$" != "1" ] || die "this container has its own pid namespace (the compose must set 'pid: host')"

# `privileged: true` — reading another process's namespace links and setns into them need
# CAP_SYS_PTRACE and CAP_SYS_ADMIN. Probed with the same static nsenter the sessions use.
if [ -z "$(readlink /proc/1/ns/mnt 2>/dev/null)" ]; then
    die "cannot read PID 1's namespaces (the compose must set 'privileged: true')"
fi
/bin/busybox.static nsenter --target 1 --mount -- /bin/true 2>/dev/null \
    || die "cannot enter PID 1's namespaces (the compose must set 'privileged: true')"

# `network_mode: host` — sshd binds $PORT in whatever network namespace this container has.
# If that is a private one, the guest port cvmd forwards has no listener and the CVM is
# never declared ready. PID 1's net namespace must be ours.
[ "$(readlink /proc/self/ns/net)" = "$(readlink /proc/1/ns/net)" ] \
    || die "this container is not in the guest's network namespace (the compose must set 'network_mode: host')"

# `/dev/pts:/dev/pts` — sshd allocates a session's terminal from OUR /dev/pts, and the
# session then runs in the guest's mount namespace, where the terminal must exist under the
# same name. Docker gives a container a private devpts instance by default; a pty from that
# instance is invisible on the guest and every interactive login reads as "not a tty". So
# the compose binds the guest's instance over ours, and this checks they are one and the
# same device.
[ "$(stat -c %d /dev/pts 2>/dev/null)" = "$(stat -c %d "$GUEST_ROOT/dev/pts" 2>/dev/null)" ] \
    || die "/dev/pts is not the guest's (the compose must bind '/dev/pts:/dev/pts'); sessions would have no terminal"

[ -n "${LIUM_SSH_AUTHORIZED_KEYS:-}" ] \
    || die "LIUM_SSH_AUTHORIZED_KEYS is empty; an sshd nobody can log into is not a service"

# --------------------------------------------------------------- authorized keys

# One key per line, exactly as the backend wrote them into the compose. CRs stripped so a
# key pasted from a Windows client is not silently ignored by sshd. Written to a root-owned
# path outside any home directory: sshd reads it BEFORE the chroot, so it lives in this
# image, and StrictModes wants it and its directory root-owned and not group/world-writable.
mkdir -p "$AUTHORIZED_DIR"
chmod 0755 "$AUTHORIZED_DIR"
printf '%s\n' "$LIUM_SSH_AUTHORIZED_KEYS" | tr -d '\r' | grep -v '^[[:space:]]*$' > "$AUTHORIZED_DIR/root"
chmod 0644 "$AUTHORIZED_DIR/root"
KEY_COUNT="$(wc -l < "$AUTHORIZED_DIR/root" | tr -d ' ')"
[ "$KEY_COUNT" -gt 0 ] || die "LIUM_SSH_AUTHORIZED_KEYS holds no key lines"

# ---------------------------------------------------------------- host keys

# Persisted in a named volume by the compose. The launch report carries this fingerprint
# and the renter pins it; a restart of this container must not change it.
mkdir -p "$HOSTKEY_DIR"
chmod 0700 "$HOSTKEY_DIR"
for type in ed25519 rsa ecdsa; do
    key="$HOSTKEY_DIR/ssh_host_${type}_key"
    if [ ! -f "$key" ]; then
        ssh-keygen -q -N '' -t "$type" -f "$key"
        log "generated a new $type host key"
    fi
done

# ---------------------------------------------------------------- the guest-side stage

# The chrooted session can only run what exists under the guest root, so the two things it
# needs are copied there: a static busybox (for nsenter — the guest's own busybox is built
# without that applet) and the login shell that uses it.
mkdir -p "$STAGE" || die "cannot write $STAGE_REL under the guest root; is the guest's /run writable?"
cp /bin/busybox.static "$STAGE/busybox"
chmod 0755 "$STAGE/busybox"

# The login shell, as sshd runs it after the chroot: with no arguments for an interactive
# session, or `-c <command>` for `ssh host <command>`. Both end in the guest's /bin/sh with
# the process fully inside PID 1's namespaces. Its shebang is the GUEST's /bin/sh — this
# script executes inside the chroot, so `/bin/sh` here is the guest's busybox ash.
cat > "$STAGE/shell" <<'EOF'
#!/bin/sh
# lium-cvm-ssh login shell: staged by the guest-SSH container, runs chrooted in the guest.
NSENTER="/run/lium-ssh/busybox nsenter --target 1 --mount --uts --ipc --net --pid"
# root's home as the GUEST names it, not as this container's passwd does.
GUEST_HOME="$(awk -F: '$1=="root"{print $6}' /etc/passwd 2>/dev/null)"
[ -n "$GUEST_HOME" ] && [ -d "$GUEST_HOME" ] || GUEST_HOME=/
export HOME="$GUEST_HOME"
if [ "${1:-}" = "-c" ]; then
    # sshd hands `ssh host <command>` over as exactly one argument after -c.
    exec $NSENTER --wd="$HOME" -- /bin/sh -c "$2"
fi
exec $NSENTER --wd="$HOME" -- /bin/sh -l
EOF
chmod 0755 "$STAGE/shell"

# Root's shell in THIS container's passwd is the staged script, by its path inside the
# chroot. sshd resolves the shell after chrooting, so the path must be the guest-side one.
sed -i "s#^root:\([^:]*\):0:0:\([^:]*\):\([^:]*\):.*#root:\1:0:0:\2:\3:$STAGE_REL/shell#" /etc/passwd
grep -q "^root:.*:$STAGE_REL/shell\$" /etc/passwd || die "could not set root's login shell"

# ---------------------------------------------------------------- sshd

cat > /etc/ssh/sshd_config <<EOF
# Written by lium-cvm-ssh at start; not user-editable. See entrypoint.sh.
Port $PORT
AddressFamily inet
ListenAddress 0.0.0.0
HostKey $HOSTKEY_DIR/ssh_host_ed25519_key
HostKey $HOSTKEY_DIR/ssh_host_rsa_key
HostKey $HOSTKEY_DIR/ssh_host_ecdsa_key

# Keys only. Root has no password and there is no other account.
PermitRootLogin prohibit-password
PubkeyAuthentication yes
AuthorizedKeysFile $AUTHORIZED_DIR/%u
PasswordAuthentication no
KbdInteractiveAuthentication no
PermitEmptyPasswords no
AllowUsers root
MaxAuthTries 6
LoginGraceTime 30
StrictModes yes

# Every session lives in the guest: chrooted into its root, then moved into PID 1's
# namespaces by the login shell staged at $STAGE_REL/shell.
ChrootDirectory $GUEST_ROOT
Subsystem sftp internal-sftp
PermitTTY yes

# Tunnels are the renter's, and they terminate in the guest's network namespace because this
# container shares it. No X11, no layer-3 tunnels.
AllowTcpForwarding yes
AllowAgentForwarding yes
AllowStreamLocalForwarding yes
GatewayPorts no
X11Forwarding no
PermitTunnel no
PermitUserEnvironment no

UseDNS no
PrintMotd no
ClientAliveInterval 60
ClientAliveCountMax 5
LogLevel INFO
PidFile /run/lium-cvm-sshd.pid
EOF

/usr/sbin/sshd -t -f /etc/ssh/sshd_config || die "the generated sshd_config does not pass sshd -t"

log "guest SSH on port $PORT; $KEY_COUNT authorized key(s); sessions land in the guest via $STAGE_REL/shell"
log "host key fingerprint: $(ssh-keygen -lf "$HOSTKEY_DIR/ssh_host_ed25519_key.pub" | awk '{print $2}')"

exec /usr/sbin/sshd -D -e -f /etc/ssh/sshd_config
