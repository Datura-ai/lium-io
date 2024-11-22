ARG BASE_IMAGE=python:3.11-slim


FROM $BASE_IMAGE AS base-image
LABEL builder=true

WORKDIR /root/app/

RUN pip3 install -U pdm
ENV PDM_CHECK_UPDATE=false

RUN apt-get update && apt-get install -y git
COPY pyproject.toml pdm.lock README.md ./
# RUN pdm lock --check
COPY --from=datura . /datura

RUN --mount=type=cache,target=/tmp/pdm_cache \
  pdm config cache_dir /tmp/pdm_cache && \
  pdm config python.use_venv False && \
  # compute-horde from path computes version from git tag which is unavailable here, 
  # hence we pass PDM_BUILD_SCM_VERSION=0 manually
  PDM_BUILD_SCM_VERSION=0 pdm sync --prod --group :all
RUN mkdir -p /opt/ && mv __pypackages__/3.11/ /opt/pypackages/

ENV PATH=/opt/pypackages/bin:$PATH
ENV PYTHONPATH=/opt/pypackages/lib:/root/app

COPY . .

FROM $BASE_IMAGE AS secondary-image
LABEL builder=false

RUN apt-get update \
  && apt-get install -y wget speedtest-cli docker.io docker-compose openssh-server \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /root/app/

# Remove existing SSH host keys
RUN rm -f /etc/ssh/ssh_host_*

ENV ENV=prod
ENV PYTHONUNBUFFERED=1
ENV PATH=/opt/pypackages/bin:$PATH
ENV PYTHONPATH=/opt/pypackages/lib:/root/app

RUN echo "export PYTHONPATH=$PYTHONPATH" >> /etc/profile
RUN echo "export PYTHONPATH=$PYTHONPATH" >> ~/.bashrc
RUN echo "export PYTHONPATH=$PYTHONPATH" >> ~/.bash_profile

COPY --from=base-image /root/app/ /root/app/
COPY --from=base-image /opt/pypackages/ /opt/pypackages/

RUN mkdir -p /etc/docker
RUN mkdir -p /etc/nvidia-container-runtime

LABEL version="3.1.0"

CMD ["bash", "run.sh"]