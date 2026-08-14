# The renter CVM's attest-agent (DAH-2579).
#
# Shipped THROUGH the measured customer compose rather than baked into the OS image. That
# is the point of the task: the agent can be updated without minting a new image hash and
# re-approving it fleet-wide, and the customer can read exactly what runs beside their
# workload, because it is a service in the compose they submitted.
#
# Everything this container can reach is something a trust check needs — there is no shell
# tooling to pivot with, no docker client, and the compose block that runs it grants no socket
# and no volume into the workload. The NVIDIA attestation stack below is the one bulky thing
# here, and it is not optional: without it the agent has no GPU evidence to return, and the
# GPUs a renter paid for cannot be proven to exist.

FROM python:3.12-slim

# Never root inside the guest. The agent reads the dstack guest socket and NVML, both of
# which are group-granted; nothing it does needs uid 0, and a compromise of the agent must
# not be a compromise of the CVM.
RUN groupadd --system attest && useradd --system --gid attest --home /var/lib/lium-attest-agent attest

WORKDIR /opt/attest-agent
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir . \
    # The CVM-only dependencies, added here rather than declared in pyproject.toml so the
    # package stays installable (and testable) outside a CVM. See the comment there.
    #
    # nv-ppcie-verifier is the NVIDIA attestation stack — the same package and version the
    # executor image installs under INSTALL_GPU_ATTESTATION, so both collectors produce
    # evidence NRAS is known to accept on this fleet. Without it the agent answers every
    # trust check with no GPU evidence, and the GPU set of a rented CVM stays a guest-made
    # claim that nothing can confirm.
    #
    # It brings its own pinned nvidia-ml-py (== 12.550.52, which is pynvml). Do not ask for a
    # newer one alongside it: the two constraints are unsatisfiable and the build fails.
    && pip install --no-cache-dir "dstack-sdk>=0.5.0" "nv-ppcie-verifier==1.5.0" \
    && mkdir -p /var/lib/lium-attest-agent \
    && chown -R attest:attest /var/lib/lium-attest-agent

USER attest

# Not /opt/attest-agent. The NVIDIA verifier package opens a log file in the working
# directory as it is imported, and this container runs unprivileged with a read-only rootfs
# (the compose block sets `read_only: true`), so every other path raises PermissionError or
# EROFS before a single piece of GPU evidence is collected. The state volume is the one
# writable place the agent has, and it is where that log belongs anyway.
WORKDIR /var/lib/lium-attest-agent

EXPOSE 8451
ENTRYPOINT ["attest-agent"]
