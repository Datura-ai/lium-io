FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV ENV=prod
ENV PYTHONUNBUFFERED=1

WORKDIR /root/app

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
     build-essential \
     ca-certificates \
     curl \
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
  && install -m 0755 -d /etc/apt/keyrings \
  && curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
  && chmod a+r /etc/apt/keyrings/docker.gpg \
  && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian bookworm stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null \
  && apt-get update \
  && apt-get install -y --no-install-recommends docker-ce-cli \
  && rm -rf /var/lib/apt/lists/* \
  && docker --version

RUN pip install -U pdm
ENV PDM_CHECK_UPDATE=false

COPY pyproject.toml pdm.lock README.md ./
COPY --from=datura . /datura

RUN pdm install --prod

COPY . .

RUN mv /root/app/libdmcompverify.so /usr/lib/ \
  && mv /root/app/libverifyx.so /usr/lib/

# Remove existing SSH host keys
RUN rm -f /etc/ssh/ssh_host_*

RUN mkdir -p /etc/docker \
  && mkdir -p /etc/nvidia-container-runtime \
  && mkdir -p /root/.ssh

ARG SSH_PUBLIC_KEY

RUN useradd -m -s /bin/bash liumuser && \
    mkdir -p /home/liumuser/.ssh && \
    echo "$SSH_PUBLIC_KEY" > /home/liumuser/.ssh/authorized_keys && \
    chmod 700 /home/liumuser/.ssh && chmod 600 /home/liumuser/.ssh/authorized_keys && \
    chown -R liumuser:liumuser /home/liumuser/.ssh

RUN echo 'AcceptEnv CONTAINER_NAME' >>/etc/ssh/sshd_config && \
    printf 'Match User liumuser\n    ForceCommand bash -c '"'"'[ -n "$CONTAINER_NAME" ] && exec docker exec -it "$CONTAINER_NAME" bash || { echo "CONTAINER_NAME not set"; exit 1; }'"'"'\n' >>/etc/ssh/sshd_config

CMD ["bash", "run.sh"]
