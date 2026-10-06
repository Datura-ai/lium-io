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
# TODO: pin digest
docker run --rm --privileged --pid=host --net=host -v /:/host dstacktee/dstack-sysbox-installer:1.0.0@sha256:2f5dbea99176f3ea0362b85346b31b1160bfb70c1d98d1c8d375d57782127dd1 bash -c "sed -i '/^check_existing() {/,/^}/c\\check_existing() { return 0; }' /usr/local/bin/install-sysbox-complete.sh && /usr/local/bin/install-sysbox-complete.sh"
systemctl restart docker
# Warm the images the validator's sysbox/DinD probes run, so first-cycle
# probes don't burn their timeout budget on multi-GB registry pulls.
docker pull daturaai/dind:0.0.1 >/dev/null 2>&1 || true
docker pull daturaai/compute-subnet-executor:latest >/dev/null 2>&1 || true
