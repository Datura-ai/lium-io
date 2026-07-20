#!/bin/bash
# Disk reserve for validator writes (DAH-2448).
#
# On machines without XFS/vloopback a renter can fill the host disk to 100%;
# the validator's upload then fails (UPLOAD_FAILED) and the miner collects a
# false penalty. This script pre-allocates a 5G loop-backed ext4 inside the
# reserve_data volume and mounts it over every path the validation cycle
# writes to, so those writes keep working on a full disk:
#   /root/app   — overlay (upper in reserve): validator sftp uploads
#   /tmp        — bind: machine_scrape onefile extraction + tempfiles
#   /root/.ssh  — bind: authorized_keys appends before each cycle
#
# /run is deliberately NOT covered: /var/run symlinks to /run, so any mount
# there shadows the bind-mounted docker.sock/dstack.sock. Consequence: an
# executor RESTART while the disk is already full may fail on /run/sshd —
# same as today; the reserve protects the running executor.
#
# Everything is fail-open: any failure logs a warning and the executor runs
# with partial or no protection (today's behavior).
#
# Usage: setup_disk_reserve.sh [setup|janitor]

RESERVE_DIR=/var/lium-reserve
IMG="$RESERVE_DIR/reserve.img"
MARKER="$RESERVE_DIR/.initialized"
MNT=/mnt/lium-reserve
SIZE_MB=5120

log() { echo "[disk-reserve] $*"; }

ensure_loop_node() {
    # loop devices are global, but their /dev nodes are not created in a fresh
    # container (no udev) — recreate the node if missing
    [ -b "$1" ] || mknod "$1" b 7 "${1#/dev/loop}"
}

setup_backing_fs() {
    # create/attach the loop-backed ext4 whose blocks are pre-allocated on the
    # host fs — the reservation must exist before the disk ever fills
    [ -d "$RESERVE_DIR" ] || { log "$RESERVE_DIR not mounted (compose volume missing)"; return 1; }

    # idempotent: extends/repairs the allocation, no-op when already full-size
    fallocate -l "${SIZE_MB}M" "$IMG" || { log "fallocate ${SIZE_MB}M failed (host disk too full?)"; return 1; }

    local loop
    loop=$(losetup -j "$IMG" 2>/dev/null | head -1 | cut -d: -f1)
    if [ -n "$loop" ]; then
        ensure_loop_node "$loop" || return 1
    else
        [ -f "$MARKER" ] && log "WARNING: reserve initialized earlier but no loop attached — previous device may have leaked"
        loop=$(losetup -f 2>&1) || { log "losetup -f failed: $loop"; return 1; }
        ensure_loop_node "$loop" || return 1
        losetup "$loop" "$IMG" || { log "losetup $loop $IMG failed"; return 1; }
    fi

    # nodiscard is mandatory: default mkfs.ext4 discards blocks and silently
    # destroys the fallocate reservation. lazy init must be off too: the
    # kernel's post-mount ext4lazyinit zeroes inode tables via WRITE_ZEROES,
    # which loop turns into hole-punching in the backing file (~80M of the
    # reservation lost; a write into a hole on a full disk aborts the journal
    # and flips the reserve read-only)
    local mkfs_opts="nodiscard,lazy_itable_init=0,lazy_journal_init=0"
    if [ "$(blkid -o value -s TYPE "$loop" 2>/dev/null)" != "ext4" ]; then
        mkfs.ext4 -q -F -E "$mkfs_opts" -m 0 -L lium-reserve "$loop" || { log "mkfs.ext4 $loop failed"; return 1; }
    fi

    mkdir -p "$MNT" || return 1
    if ! mountpoint -q "$MNT" && ! mount "$loop" "$MNT"; then
        # reserve content is disposable — rebuild a corrupted fs instead of
        # staying unprotected forever
        log "mount failed — re-creating reserve fs on $loop"
        mkfs.ext4 -q -F -E "$mkfs_opts" -m 0 -L lium-reserve "$loop" || { log "mkfs.ext4 $loop failed"; return 1; }
        mount "$loop" "$MNT" || { log "mount $loop failed"; return 1; }
    fi
    # an fs with a previously aborted journal can still mount, but read-only —
    # same rebuild rule as a failed mount
    if ! touch "$MNT/.rw_probe" 2>/dev/null; then
        log "reserve mounted read-only — re-creating fs on $loop"
        umount "$MNT" || return 1
        mkfs.ext4 -q -F -E "$mkfs_opts" -m 0 -L lium-reserve "$loop" || { log "mkfs.ext4 $loop failed"; return 1; }
        mount "$loop" "$MNT" || { log "mount $loop failed"; return 1; }
    fi
    rm -f "$MNT/.rw_probe"
    touch "$MARKER" || true
}

mount_app_overlay() {
    # upper/work wiped every boot: uploads are ephemeral and a stale upper
    # must not shadow files after an image upgrade
    rm -rf "$MNT/app-upper" "$MNT/app-work" &&
        mkdir -p "$MNT/app-upper" "$MNT/app-work" &&
        mount -t overlay overlay \
            -o "lowerdir=/root/app,upperdir=$MNT/app-upper,workdir=$MNT/app-work" /root/app
}

mount_tmp() {
    rm -rf "$MNT/tmp" &&
        mkdir -p "$MNT/tmp" &&
        chmod 1777 "$MNT/tmp" &&
        mount --bind "$MNT/tmp" /tmp
}

mount_ssh() {
    # not wiped: keys now survive container recreation (previously lost)
    mkdir -p "$MNT/ssh" &&
        chmod 700 "$MNT/ssh" &&
        if [ -f /root/.ssh/authorized_keys ] && [ ! -f "$MNT/ssh/authorized_keys" ]; then
            cp -p /root/.ssh/authorized_keys "$MNT/ssh/authorized_keys"
        fi &&
        mount --bind "$MNT/ssh" /root/.ssh
}

evict_upload_dirs_under_pressure() {
    # delete oldest upload dirs until reserve usage drops below 80%; only
    # meaningful when the reserve is mounted — otherwise df would measure the
    # host disk and evict live upload dirs on any ≥80%-full machine
    mountpoint -q "$MNT" || return 0
    local usage oldest prev=""
    while :; do
        usage=$(df --output=pcent "$MNT" 2>/dev/null | tail -1 | tr -dc '0-9')
        { [ -n "$usage" ] && [ "$usage" -ge 80 ]; } || break
        oldest=$(find /root/app -maxdepth 1 -type d -regextype posix-extended \
            -regex '.*/[0-9a-f]{32}' -printf '%T@ %p\n' 2>/dev/null | sort -n | head -1 | cut -d' ' -f2-)
        if [ -z "$oldest" ] || [ "$oldest" = "$prev" ]; then
            find /tmp -mindepth 1 -maxdepth 1 -mmin +10 -exec rm -rf {} + 2>/dev/null
            break
        fi
        log "reserve ${usage}% full — evicting $oldest"
        rm -rf "$oldest"
        prev="$oldest"
    done
}

janitor_loop() {
    # steady-state cleanup, not just a crash backstop: the current validator
    # pipeline never deletes its upload dirs, and a scrape session killed
    # mid-run leaks its /tmp files
    while :; do
        # re-extend the backing file: any holes punched at runtime are
        # re-filled while the host disk still has room
        fallocate -l "${SIZE_MB}M" "$IMG" 2>/dev/null
        find /root/app -maxdepth 1 -type d -regextype posix-extended \
            -regex '.*/[0-9a-f]{32}' -mmin +90 -exec rm -rf {} + 2>/dev/null
        find /tmp -mindepth 1 -maxdepth 1 -mmin +90 -exec rm -rf {} + 2>/dev/null
        evict_upload_dirs_under_pressure
        sleep 900
    done
}

setup() {
    setup_backing_fs || { log "reserve unavailable; executor continues without disk protection"; return 1; }
    # each mount is an independent benefit — one failure must not skip the rest
    mount_app_overlay || log "WARNING: overlay on /root/app failed — validator uploads not protected"
    mount_tmp || log "WARNING: /tmp bind failed — scrape tempfiles not protected"
    mount_ssh || log "WARNING: /root/.ssh bind failed — ssh key appends not protected"
    log "reserve active: $(df -h --output=size,used,avail "$MNT" | tail -1 | tr -s ' ')"
}

case "${1:-setup}" in
    setup) setup ;;
    janitor) janitor_loop ;;
    *) log "unknown command: $1"; exit 1 ;;
esac
