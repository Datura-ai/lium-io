# DAH-2620: the customer-facing multi-node cluster image.
#
# A renter who takes a group of IB nodes gets this image on every pod. The validator injects one
# env var per pod — LIUM_WIREGUARD_CONF_B64, the node's wg-quick config — and the entrypoint below
# raises wg0 from it and points NCCL at the overlay. The renter runs `torchrun --nnodes=N ...` and
# the InfiniBand transport engages on its own; nothing here has to be configured by hand.
#
# Built on the CUDA base so the card, the userspace RDMA libs and nvidia-smi are already present.
FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04

# wireguard-tools brings wg + wg-quick; the rest are what an unaddressed base image lacks and a
# distributed job needs (ibverbs for the RDMA path, iproute2 for the interface, openssh so the pod
# is reachable the usual way).
RUN apt-get update && apt-get install -y --no-install-recommends \
        wireguard-tools \
        iproute2 \
        libibverbs1 \
        ibverbs-providers \
        rdma-core \
        openssh-server \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY entrypoint.sh /usr/local/bin/lium-cluster-entrypoint
RUN chmod +x /usr/local/bin/lium-cluster-entrypoint

ENTRYPOINT ["/usr/local/bin/lium-cluster-entrypoint"]
CMD ["sleep", "infinity"]
