#!/bin/sh
# sshd host keys that survive a recreated container (DAH-5729). The image ships none (Dockerfile) and every runner
# update recreates the executor, so without this the node would present a new host key after each release.
# The keys live in SSH_HOST_KEYS_DIR, a volume (docker-compose.app.yml). Each start restores them into /etc/ssh,
# generates only the types still missing, and stores any key the volume lacks: the first start with an empty volume
# keeps the keys this container already has (a restart) or generates them.
# SSH_ROOT is a test hook: `ssh-keygen -A -f` writes under $SSH_ROOT/etc/ssh.
set -eu

KEYS_DIR="${SSH_HOST_KEYS_DIR:-/var/lib/executor-ssh-host-keys}"
SSH_DIR="${SSH_ROOT:-}/etc/ssh"

mkdir -p "$KEYS_DIR" "$SSH_DIR"
chmod 700 "$KEYS_DIR"
if ls "$KEYS_DIR"/ssh_host_* >/dev/null 2>&1; then
    cp -p "$KEYS_DIR"/ssh_host_* "$SSH_DIR"/
fi
if [ -n "${SSH_ROOT:-}" ]; then ssh-keygen -A -f "$SSH_ROOT"; else ssh-keygen -A; fi
for key in "$SSH_DIR"/ssh_host_*; do
    [ -e "$KEYS_DIR/$(basename "$key")" ] || cp -p "$key" "$KEYS_DIR"/
done
