chmod +x /usr/bin/containerd-shim-runc-v2
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
# Keep the guest's own rsync (built against its glibc; sysbox needs it): the installer's is a static glibc build
# whose user/group lookups load the guest's libc.so.6 and die with a divide error in __libc_early_init (DAH-6549).
# Without a GPU-capable sysbox the node fails every validator sysbox check, so a failed step here stops
# the boot with docker's own error on the console instead of starting the executor without it.
sysbox_setup_failed() {
    echo "pre-launch: sysbox setup failed: $1" >&2
    [ -n "${2:-}" ] && printf '%s\n' "$2" | tail -n 20 >&2
    exit 1
}
SYSBOX_INSTALLER=dstacktee/dstack-sysbox-installer:1.0.0@sha256:2f5dbea99176f3ea0362b85346b31b1160bfb70c1d98d1c8d375d57782127dd1
# pulled only when not cached: a registry outage must not stop a CVM whose installer is already on disk
docker image inspect "$SYSBOX_INSTALLER" >/dev/null 2>&1 \
    || out=$(docker pull "$SYSBOX_INSTALLER" 2>&1) || sysbox_setup_failed "docker pull $SYSBOX_INSTALLER" "$out"
out=$(docker run --rm --privileged --pid=host --net=host -v /:/host "$SYSBOX_INSTALLER" bash -c "sed -i -e '/^check_existing() {/,/^}/c\\check_existing() { return 0; }' -e 's#cp /usr/local/bin/rsync /host/usr/bin/#:#' /usr/local/bin/install-sysbox-complete.sh && /usr/local/bin/install-sysbox-complete.sh" 2>&1) || sysbox_setup_failed "installer run" "$out"
out=$(systemctl restart docker 2>&1) || sysbox_setup_failed "docker restart" "$out"
docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"sysbox-runc"' \
    || sysbox_setup_failed "sysbox-runc missing from docker runtimes" "$(docker info 2>&1)"
# the same runtime + --gpus pair the validator's sysbox check runs
out=$(docker run --rm --runtime=sysbox-runc --gpus all daturaai/compute-subnet-executor:latest nvidia-smi -L 2>&1) \
    || sysbox_setup_failed "docker run --runtime=sysbox-runc --gpus all" "$out"
# Warm the images the validator's sysbox/DinD probes run, so first-cycle
# probes don't burn their timeout budget on multi-GB registry pulls.
docker pull daturaai/dind:0.0.1 >/dev/null 2>&1 || true
docker pull daturaai/compute-subnet-executor:latest >/dev/null 2>&1 || true
