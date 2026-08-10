# The renter CVM's attest-agent (DAH-2579).
#
# Shipped THROUGH the measured customer compose rather than baked into the OS image. That
# is the point of the task: the agent can be updated without minting a new image hash and
# re-approving it fleet-wide, and the customer can read exactly what runs beside their
# workload, because it is a service in the compose they submitted.
#
# The image is deliberately thin. Everything this container can reach is something a trust
# check needs — there is no shell tooling to pivot with, no docker client, and the compose
# block that runs it grants no socket and no volume into the workload.

FROM python:3.12-slim

# Never root inside the guest. The agent reads the dstack guest socket and NVML, both of
# which are group-granted; nothing it does needs uid 0, and a compromise of the agent must
# not be a compromise of the CVM.
RUN groupadd --system attest && useradd --system --gid attest --home /var/lib/lium-attest-agent attest

WORKDIR /opt/attest-agent
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir . \
    # The two CVM-only dependencies, added here rather than declared in pyproject.toml so
    # the package stays installable (and testable) outside a CVM. See the comment there.
    && pip install --no-cache-dir "dstack-sdk>=0.5.0" "nvidia-ml-py>=12.560.30" \
    && mkdir -p /var/lib/lium-attest-agent \
    && chown -R attest:attest /var/lib/lium-attest-agent

USER attest
EXPOSE 8451
ENTRYPOINT ["attest-agent"]
