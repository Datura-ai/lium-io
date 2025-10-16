ARG BASE_IMAGE=daturaai/ubuntu:24.04-py3.11


FROM $BASE_IMAGE AS base-image
LABEL builder=true

WORKDIR /root/app/

RUN pip3 install -U pdm
ENV PDM_CHECK_UPDATE=false

RUN apt-get update
COPY pyproject.toml pdm.lock README.md ./
COPY --from=datura . /datura

RUN --mount=type=cache,target=/tmp/pdm_cache \
  pdm config cache_dir /tmp/pdm_cache && \
  pdm config python.use_venv False && \
  PDM_BUILD_SCM_VERSION=0 pdm sync --prod --group :all
RUN mkdir -p /opt/ && mv __pypackages__/3.11/ /opt/pypackages/

FROM $BASE_IMAGE AS secondary-image
LABEL builder=false

RUN apt-get update -y \
  && apt-get upgrade -y \
  && apt-get install -y speedtest-cli \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /root/app/

COPY . .
COPY --from=base-image /opt/pypackages/ /opt/pypackages/
RUN mv /root/app/libdmcompverify.so /usr/lib/
RUN mv /root/app/libverifyx.so /usr/lib/

# cli fix
RUN sed -i '1s|.*|#!/usr/bin/env python3|' /opt/pypackages/bin/alembic
RUN sed -i '1s|.*|#!/usr/bin/env python3|' /opt/pypackages/bin/speedcheck
RUN sed -i '1s|.*|#!/usr/bin/env python3|' /opt/pypackages/bin/netmeasure
# Remove existing SSH host keys
RUN rm -f /etc/ssh/ssh_host_*

ENV ENV=prod
ENV PYTHONUNBUFFERED=1
ENV PATH=/opt/pypackages/bin:$PATH
ENV PYTHONPATH=/opt/pypackages/lib:/root/app

RUN echo "export PYTHONPATH=$PYTHONPATH" >> /etc/profile
RUN echo "export PYTHONPATH=$PYTHONPATH" >> ~/.bashrc
RUN echo "export PYTHONPATH=$PYTHONPATH" >> ~/.bash_profile

RUN mkdir -p /etc/docker
RUN mkdir -p /etc/nvidia-container-runtime
RUN mkdir -p /root/.ssh

ARG SSH_PUBLIC_KEY

RUN useradd -m -s /bin/bash liumuser && \
    mkdir -p /home/liumuser/.ssh && \
    echo "$SSH_PUBLIC_KEY" > /home/liumuser/.ssh/authorized_keys && \
    chmod 700 /home/liumuser/.ssh && chmod 600 /home/liumuser/.ssh/authorized_keys && \
    chown -R liumuser:liumuser /home/liumuser/.ssh
    
RUN echo 'AcceptEnv CONTAINER_NAME' >>/etc/ssh/sshd_config && \
    printf 'Match User liumuser\n    ForceCommand bash -c '"'"'[ -n "$CONTAINER_NAME" ] && exec docker exec -it "$CONTAINER_NAME" bash || { echo "CONTAINER_NAME not set"; exit 1; }'"'"'\n' >>/etc/ssh/sshd_config

CMD ["bash", "run.sh"]
