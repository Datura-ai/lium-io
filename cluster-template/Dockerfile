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
FROM daturaai/pytorch:2.12.0-py3.12-cuda12.8-devel-ubuntu24.04-dind

# wireguard-tools brings wg + wg-quick, iproute2 the `ip` this entrypoint needs.
#
# The RDMA userspace is NOT in the base, despite the devices being forwarded: without libibverbs and
# the provider plugins NCCL logs "Failed to open libibverbs.so[.1]" and quietly falls back to its
# socket transport, so the job would crawl over the overlay while every device check still passes.
# ibverbs-utils is the diagnosis the renter reaches for first (ibv_devinfo), and perftest is how
# anyone confirms the fabric end to end (ib_write_bw).
RUN apt-get update && apt-get install -y --no-install-recommends \
        wireguard-tools \
        iproute2 \
        libibverbs1 \
        librdmacm1 \
        ibverbs-providers \
        ibverbs-utils \
        perftest \
    && rm -rf /var/lib/apt/lists/*

COPY entrypoint.sh /usr/local/bin/lium-cluster-entrypoint
COPY lium-rdma-runc.py /usr/local/bin/lium-rdma-runc
RUN chmod +x /usr/local/bin/lium-cluster-entrypoint /usr/local/bin/lium-rdma-runc

# The base image starts the inner Docker daemon only when asked. A cluster pod always wants it: the
# renter's job usually runs in their own image, and that is where the fabric has to reach.
ENV ENABLE_DIND=true

ENTRYPOINT ["/usr/local/bin/lium-cluster-entrypoint"]
# The base image's own start script, which our entrypoint execs into after raising the overlay.
CMD ["/start.sh"]
