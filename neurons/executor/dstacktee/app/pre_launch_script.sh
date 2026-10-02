chmod +x /usr/bin/containerd-shim-runc-v2

# This file is sourced by /bin/app-compose.sh (bash) before it starts the compose. Sourced means
# `exit` here would kill the boot before the executor ever starts, and any `set -e` would change
# how the rest of app-compose.sh behaves -- so this script never does either, and cleans up the
# names it defines at the end.

lium_not_ready_marker=/dstack/sysbox-not-ready
lium_say() {
    # both the guest journal and the serial console, which is the host's only window in
    echo "lium: $*" >&2
}

# DAH-2780: docker.service is ordered before app-compose.service, so dockerd has already restored
# every `unless-stopped` container by the time this runs -- including customer rentals. Pulling the
# sysbox daemons out from under a live rental is what left a customer's container dead for nine
# hours on 26 August: it keeps handles on the old runtime and dies with exit 128. Stop rentals
# first: a container stopped here is not brought back by the docker restart below, and the
# validator will not start it either -- its only auto-recovery covers the DAH-2306 stale-mount
# signature -- so the pod stays down until the customer starts it and the node takes
# POD_NOT_RUNNING. That is the same end state as letting the swap wedge the container, minus the
# wedged containerd and the nine hours. The grace matches the validator's
# CONTAINER_STOP_GRACE_SECONDS -- killing GPU workloads fast wedges containerd/sysbox (DAH-2364),
# which is the worst thing that could happen immediately before a daemon swap. Only pod_/filler_
# are rentals; container_*/health_check_* are the validator's own throwaway probes and
# executor-runner is the executor itself.
if ! lium_rentals=$(docker ps -q \
    --filter status=running --filter status=restarting \
    --filter name=^pod_ --filter name=^filler_ 2>/dev/null); then
    lium_say "could not list containers -- dockerd not answering, rentals may survive the sysbox swap"
elif [ -n "$lium_rentals" ]; then
    lium_say "stopping rental containers before the sysbox swap"
    docker stop -t 30 $lium_rentals >/dev/null 2>&1 || \
        lium_say "some rental containers did not stop cleanly"
fi

# dstack-nvidia-0.5.11 bakes upstream sysbox CE 0.6.7, which rejects --gpus
# (docker device requests) and so fails the SN51 sysbox/GPU compatibility
# check. Stop the baked daemons and force-install the Datura sysbox build
# (GPU-capable, same topology as prod 0.5.5 CVMs). The installer skips itself
# whenever any sysbox-mgr.service unit-file exists -- on 0.5.11 the baked
# /usr/lib unit always matches -- so stub out its check_existing gate; the
# units it drops into /run/systemd/system then shadow the baked /usr/lib ones.
systemctl unmask sysbox-mgr sysbox-fs 2>/dev/null || true
systemctl stop sysbox-mgr sysbox-fs 2>/dev/null || true
systemctl disable sysbox-mgr sysbox-fs 2>/dev/null || true
# The stop above is best-effort, so record whatever survived it: the gate below has no other way to
# tell the daemon we are about to install from a baked 0.6.7 one that never died. Comparing the
# unit's MainPID alone cannot -- the unit name is the same either way.
lium_stale_mgr_pid=$(systemctl show -p MainPID --value sysbox-mgr 2>/dev/null)
lium_stale_fs_pid=$(systemctl show -p MainPID --value sysbox-fs 2>/dev/null)
# TODO: pin digest
# Whenever this digest moves, re-check that the sed anchor below still matches the installer's
# check_existing definition: sed exits 0 when it replaces nothing, so a renamed function would
# silently leave the gate in place and the baked 0.6.7 daemons would survive.
docker run --rm --privileged --pid=host --net=host -v /:/host dstacktee/dstack-sysbox-installer:1.0.0@sha256:2f5dbea99176f3ea0362b85346b31b1160bfb70c1d98d1c8d375d57782127dd1 bash -c "sed -i '/^check_existing() {/,/^}/c\\check_existing() { return 0; }' /usr/local/bin/install-sysbox-complete.sh && /usr/local/bin/install-sysbox-complete.sh"

# DAH-2780: prove the daemons we just installed are actually serving before anything is allowed to
# use them. Neither of the obvious signals is worth anything here: the units are Type=simple, so
# `systemctl is-active` only says a process exists, and sysbox-fs sends its readiness notification
# and logs "Ready" before it starts listening. A socket file left behind by a dead daemon also
# stays on disk, so `[ -S ... ]` passes while connect() gives ECONNREFUSED -- that was the 26 August
# shape exactly. /proc/net/unix is the one source that answers the real question: flags 00010000 is
# SO_ACCEPTCON, i.e. a socket that has reached listen(), not merely bind().
#
# The socket inode is matched against the unit's MainPID as well, so a socket entry left over from
# a dead daemon cannot answer for a live one. The MainPID is then checked against the pid that
# survived the best-effort stop above: if that stop silently failed, the baked 0.6.7 daemon is
# still listening on the same path *under the same unit*, so without that third comparison the
# gate would go green on a runtime that rejects --gpus.
lium_unit_is_serving() {  # <unit> <socket path> <pid that survived the stop>
    local inode pid
    inode=$(awk -v p="$2" '$4 == "00010000" && $8 == p { print $7; exit }' /proc/net/unix)
    [ -n "$inode" ] || return 1
    pid=$(systemctl show -p MainPID --value "$1" 2>/dev/null)
    [ -n "$pid" ] && [ "$pid" != "0" ] && [ "$pid" != "$3" ] || return 1
    ls -l "/proc/$pid/fd" 2>/dev/null | grep -q "socket:\[$inode\]"
}

lium_sysbox_is_serving() {
    lium_unit_is_serving sysbox-mgr /run/sysbox/sysmgr.sock "$lium_stale_mgr_pid" &&
        lium_unit_is_serving sysbox-fs /run/sysbox/sysfs.sock "$lium_stale_fs_pid"
}

lium_await_sysbox() {
    local left=30
    while [ "$left" -gt 0 ]; do
        lium_sysbox_is_serving && return 0
        sleep 1
        left=$((left - 1))
    done
    return 1
}

# A boot that gets this far owns the verdict, so a marker from an earlier failure goes now.
rm -f "$lium_not_ready_marker"
if ! lium_await_sysbox; then
    # --no-block is not optional: the daemon being restarted is the one that may be wedged in
    # uninterruptible sleep, and a blocking `systemctl restart` on it never returns. There is no
    # way to bound that from here, and a boot script that never finishes leaves a box whose only
    # door -- the executor -- never opens. Restarting by hand is what fixed the machine in one
    # second on 26 August.
    lium_say "sysbox is not serving; restarting the daemons"
    systemctl --no-block restart sysbox-mgr sysbox-fs 2>/dev/null || true
    if ! lium_await_sysbox; then
        # Keep booting on purpose. The executor is the only way anyone reaches this guest, so a
        # node that comes up visibly broken is worth more than one that never comes up. The line
        # below lands on the serial console, i.e. in the host's journal for the CVM unit.
        date -u '+%Y-%m-%dT%H:%M:%SZ sysbox daemons never started serving' > "$lium_not_ready_marker" 2>/dev/null
        lium_say "SYSBOX NOT READY -- rentals on this node will fail until sysbox is repaired"
    fi
fi

systemctl restart docker
# Warm the images the validator's sysbox/DinD probes run, so first-cycle
# probes don't burn their timeout budget on multi-GB registry pulls.
docker pull daturaai/dind:0.0.1 >/dev/null 2>&1 || true
docker pull daturaai/compute-subnet-executor:latest >/dev/null 2>&1 || true

unset -f lium_say lium_unit_is_serving lium_sysbox_is_serving lium_await_sysbox
unset lium_not_ready_marker lium_rentals lium_stale_mgr_pid lium_stale_fs_pid
