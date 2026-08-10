# DAH-2620: the customer-facing multi-node cluster image.
#
# Build and push as `daturaai/lium-cluster:<tag>`. That name is load-bearing: the backend refuses a
# cluster rental whose image does not start with it (CLUSTER_TEMPLATE_IMAGE_PREFIX), because only
# this entrypoint reads the injected config and raises wg0.
#
# A renter who takes a group of IB nodes gets this image on every pod. The validator injects one
# env var per pod — LIUM_WIREGUARD_CONF_B64, the node's wg-quick config — and the entrypoint below
# raises wg0 from it and points NCCL at the overlay. The renter runs `torchrun --nnodes=N ...` and
# the InfiniBand transport engages on its own; nothing here has to be configured by hand.
#
# daturaai/pytorch is the fleet's recommended image: PyTorch (so torchrun is present), CUDA, the
# userspace RDMA libs and nvidia-smi are all baked in. The renter runs their own torchrun on top.
FROM daturaai/pytorch:latest

# wireguard-tools brings wg + wg-quick, iproute2 the `ip` this entrypoint needs. Everything else a
# distributed job wants (torch, CUDA, ibverbs, sshd) is already in the base.
RUN apt-get update && apt-get install -y --no-install-recommends \
        wireguard-tools \
        iproute2 \
    && rm -rf /var/lib/apt/lists/*

COPY entrypoint.sh /usr/local/bin/lium-cluster-entrypoint
RUN chmod +x /usr/local/bin/lium-cluster-entrypoint

ENTRYPOINT ["/usr/local/bin/lium-cluster-entrypoint"]
CMD ["sleep", "infinity"]
