# The renter CVM's guest-SSH service (DAH-2684).
#
# A renter's SSH session lands IN THE CVM ITSELF — the guest OS: its processes, its network,
# its filesystem, its docker daemon and its GPUs — not inside a container the renter has to
# describe. The dstack production image (dstack-nvidia-0.5.11) ships no sshd, no login and a
# read-only dm-verity rootfs, so "SSH into the guest" cannot be a guest service. It is this
# container instead: sshd runs here, and every session it opens is moved into PID 1's
# namespaces (mount, uts, ipc, net, pid) before the renter's shell starts. See entrypoint.sh
# for exactly how, and README.md for what the renter can and cannot do from there.
#
# Shipped THROUGH the measured customer compose, like the attest-agent: the backend injects
# it (services/cvm_compose.py) pinned by digest, with the renter's authorized keys in the
# same service block. So the compose hash the hardware measures covers WHICH sshd runs and
# WHOSE keys it accepts, and the renter can check both against the quote.
#
# Everything this image contains is needed to open a session into the guest and nothing
# else. It is not the renter's toolbox — the guest is.

FROM alpine:3.20

# openssh-server: the daemon. busybox-static: a nsenter that runs from inside the guest's
# own root filesystem (see entrypoint.sh — the login shell is chrooted into the guest, where
# no nsenter exists, so one that needs no shared libraries is copied in).
RUN apk add --no-cache openssh-server busybox-static \
    && rm -f /etc/ssh/ssh_host_*_key* \
    && mkdir -p /etc/ssh/hostkeys /etc/ssh/authorized

COPY entrypoint.sh /usr/local/bin/lium-cvm-ssh
RUN chmod 0755 /usr/local/bin/lium-cvm-ssh

# The port is fleet convention (cvmd's cvm_ssh_guest_port, forwarded by cvm_ports); the
# entrypoint reads LIUM_SSH_PORT and this is documentation of the default only.
EXPOSE 2200

ENTRYPOINT ["/usr/local/bin/lium-cvm-ssh"]
