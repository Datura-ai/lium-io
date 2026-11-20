#!/bin/sh
# Install the pod's ed25519 SSH host key and make a running sshd serve it.
#
# Usage: sh pod_ssh_host_key.sh '<public key line>' < private_key
#
# The validator derives the same key for a pod every time it creates the pod's
# container, so a reboot (which recreates the container) keeps the host key the
# renter's known_hosts already trusts. A no-op when the key is already in place.
set -eu

RUN_DIR="${LIUM_RUN_DIR:-/run}"
HOST_KEY_DIR="${LIUM_SSH_HOST_KEY_DIR:-/etc/ssh}"
LOCK_DIR="$RUN_DIR/lium-ssh-setup.lock"
LOCK_TIMEOUT_SECS=30
KEY="$HOST_KEY_DIR/ssh_host_ed25519_key"
PUBLIC_KEY="${1:-}"

case "$PUBLIC_KEY" in
    ssh-ed25519\ *) ;;
    *)
        printf '%s\n' "Expected an ssh-ed25519 public key as the first argument"
        exit 2
        ;;
esac

if [ -f "$KEY" ] && [ -f "$KEY.pub" ]; then
    current="$(cut -d' ' -f1,2 "$KEY.pub" 2>/dev/null || true)"
    if [ "$current" = "$(printf '%s' "$PUBLIC_KEY" | cut -d' ' -f1,2)" ]; then
        cat > /dev/null
        printf '%s\n' "Pod SSH host key already installed"
        exit 0
    fi
fi

# Shared with sshd_bootstrap.sh and lium images' /start.sh: their `ssh-keygen -A`
# must not race this write. A busy lock is waited out, then ignored.
waited=0
LOCK_HELD=0
mkdir -p "$RUN_DIR"
while true; do
    if mkdir "$LOCK_DIR" 2>/dev/null; then
        LOCK_HELD=1
        break
    fi
    if [ "$waited" -ge "$LOCK_TIMEOUT_SECS" ]; then
        break
    fi
    sleep 1
    waited=$((waited + 1))
done
release_lock() {
    if [ "$LOCK_HELD" -eq 1 ]; then
        rmdir "$LOCK_DIR" 2>/dev/null || true
    fi
}
trap release_lock EXIT

umask 077
mkdir -p "$HOST_KEY_DIR"
cat > "$KEY.lium-new"
if [ ! -s "$KEY.lium-new" ]; then
    rm -f "$KEY.lium-new"
    printf '%s\n' "No private key on stdin"
    exit 2
fi
mv -f "$KEY.lium-new" "$KEY"
printf '%s\n' "$PUBLIC_KEY" > "$KEY.pub"
chmod 600 "$KEY"
chmod 644 "$KEY.pub"
printf '%s\n' "Installed pod SSH host key"

# sshd loads host keys at start; SIGHUP makes the master re-exec and load the new
# one. Through the pidfile only: per-session sshd processes must not get the signal.
for pidfile in "$RUN_DIR/sshd.pid" /var/run/sshd.pid; do
    [ -f "$pidfile" ] || continue
    sshd_pid="$(cat "$pidfile" 2>/dev/null || true)"
    case "${sshd_pid:-}" in
        *[!0-9]* | "") continue ;;
    esac
    if kill -0 "$sshd_pid" 2>/dev/null; then
        kill -HUP "$sshd_pid" 2>/dev/null || true
        printf '%s\n' "Sent SIGHUP to sshd (pid $sshd_pid)"
        exit 0
    fi
done
printf '%s\n' "No running sshd found; it loads the key when it starts"
