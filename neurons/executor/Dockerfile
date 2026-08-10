FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV ENV=prod
ENV PYTHONUNBUFFERED=1

WORKDIR /root/app

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
     build-essential \
     bzip2 \
     ca-certificates \
     curl \
     e2fsprogs \
     fuse3 \
     git \
     gnupg \
     libffi-dev \
     libpq-dev \
     lsb-release \
     openssh-client \
     openssh-server \
     pkg-config \
     speedtest-cli \
     ssh \
     awscli \
     tar \
     util-linux \
  && install -m 0755 -d /etc/apt/keyrings \
  && curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
  && chmod a+r /etc/apt/keyrings/docker.gpg \
  && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian bookworm stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null \
  && apt-get update \
  && apt-get install -y --no-install-recommends docker-ce-cli \
  && rm -rf /var/lib/apt/lists/* \
  && docker --version

ARG RESTIC_VERSION=0.19.1
ARG RESTIC_LINUX_AMD64_SHA256=f415415624dcc452f2a02b8c33641791a8c6d6d3b65bbb3543fcf9a25151585c
RUN curl -fsSLo /tmp/restic.bz2 \
      "https://github.com/restic/restic/releases/download/v${RESTIC_VERSION}/restic_${RESTIC_VERSION}_linux_amd64.bz2" \
  && echo "${RESTIC_LINUX_AMD64_SHA256}  /tmp/restic.bz2" | sha256sum -c - \
  && bunzip2 /tmp/restic.bz2 \
  && install -m 0755 /tmp/restic /usr/local/bin/restic \
  && rm -f /tmp/restic \
  && test "$(restic version | awk '{print $2}')" = "${RESTIC_VERSION}"

ARG GOCRYPTFS_VERSION=2.6.1
ARG GOCRYPTFS_LINUX_AMD64_SHA256=49b8c0eb0f6373b6ac99c394a52909d8478e74c08d0961527c1162967cc28c44
RUN curl -fsSLo /tmp/gocryptfs.tgz \
      "https://github.com/rfjakob/gocryptfs/releases/download/v${GOCRYPTFS_VERSION}/gocryptfs_v${GOCRYPTFS_VERSION}_linux-static_amd64.tar.gz" \
  && echo "${GOCRYPTFS_LINUX_AMD64_SHA256}  /tmp/gocryptfs.tgz" | sha256sum -c - \
  && tar -xzf /tmp/gocryptfs.tgz -C /usr/local/bin gocryptfs \
  && chmod 0755 /usr/local/bin/gocryptfs \
  && rm -f /tmp/gocryptfs.tgz \
  && gocryptfs --version

RUN pip install -U pdm
ENV PDM_CHECK_UPDATE=false

COPY pyproject.toml pdm.lock README.md ./
COPY --from=datura . /datura

RUN pdm install --prod

# G1 phase-0 (DAH-2338) — NVIDIA confidential-compute GPU attestation stack.
# Deliberately OUTSIDE the pdm lock: the nv-ppcie-verifier dependency tree is
# heavy and fragile, so it lives in its own opt-in layer for CVM runner builds
# only. The single pin pulls nv_attestation_sdk, the local GPU verifier
# (cc_admin) and pynvml transitively — same pin as the private-ml-sdk reference.
#
# Install into an ISOLATED prefix and expose it to the pdm venv via a
# low-priority .pth. Installing into the venv directly downgrades pinned deps
# (typing_extensions/setuptools) and breaks pydantic; `pdm run pip install`
# targets the base interpreter, which the venv the executor runs in never sees.
ARG INSTALL_GPU_ATTESTATION=false
RUN if [ "$INSTALL_GPU_ATTESTATION" = "true" ]; then \
      pip install --target /opt/gpu-attest nv-ppcie-verifier==1.5.0 && \
      echo /opt/gpu-attest > /root/app/.venv/lib/python3.11/site-packages/gpu-attest.pth; \
    fi

COPY . .

RUN mv /root/app/libdmcompverify.so /usr/lib/ \
  && mv /root/app/libverifyx.so /usr/lib/ \
  && mv /root/app/libinspector.so /usr/lib/

# Remove existing SSH host keys
RUN rm -f /etc/ssh/ssh_host_*

RUN mkdir -p /etc/docker \
  && mkdir -p /etc/nvidia-container-runtime \
  && mkdir -p /root/.ssh

RUN useradd -m -s /bin/bash liumuser && \
    mkdir -p /home/liumuser/.ssh && \
    chmod 700 /home/liumuser/.ssh && \
    chown -R liumuser:liumuser /home/liumuser/.ssh

RUN echo 'AcceptEnv CONTAINER_NAME' >>/etc/ssh/sshd_config && \
    printf 'Match User liumuser\n    ForceCommand bash -c '"'"'[ -n "$CONTAINER_NAME" ] && exec docker exec -it "$CONTAINER_NAME" bash || { echo "CONTAINER_NAME not set"; exit 1; }'"'"'\n' >>/etc/ssh/sshd_config

CMD ["bash", "run.sh"]
