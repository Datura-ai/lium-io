#!/bin/bash
# Undo the dirty page cache cap of executor-v1.151 (DAH-4978, reverted).
#
# vm.* sysctls are host-global and outlive the executor container, so removing
# setup_dirty_cap.sh from the image leaves every host at dirty_bytes=1 GiB until its
# next reboot. This restores the kernel defaults (dirty_ratio=20, dirty_background_ratio=10;
# writing a ratio zeroes the byte caps) on hosts whose caps are exactly the values that
# script wrote. Any other cap belongs to the provider and is left alone.
#
# Fail-open: a failure logs a warning and the executor runs with whatever the host has.
# Remove this script once the fleet has rebooted or a new cap ships.

SYS_ROOT=${DIRTY_CAP_SYS_ROOT:-}          # test-only prefix for /proc
VM="$SYS_ROOT/proc/sys/vm"
OUR_CAP=$((1024 * 1024 * 1024))
OUR_BACKGROUND=$((OUR_CAP / 4))

log() { echo "[dirty-cap] $*"; }

cap=$(cat "$VM/dirty_bytes" 2>/dev/null) || { log "cannot read $VM/dirty_bytes"; exit 1; }
background=$(cat "$VM/dirty_background_bytes" 2>/dev/null) || { log "cannot read $VM/dirty_background_bytes"; exit 1; }

[ "$cap" -eq "$OUR_CAP" ] && [ "$background" -eq "$OUR_BACKGROUND" ] || exit 0

if ! { echo 20 > "$VM/dirty_ratio" && echo 10 > "$VM/dirty_background_ratio"; }; then
    log "write to $VM failed"
    exit 1
fi
log "restored dirty_ratio=20 dirty_background_ratio=10 (was dirty_bytes=$cap dirty_background_bytes=$background)"
