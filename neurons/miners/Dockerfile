ARG BASE_IMAGE=python:3.11-slim

FROM $BASE_IMAGE

WORKDIR /root/src/

RUN apt-get update \
  && apt-get install -y git curl wget nano docker.io docker-compose openssh-server \
  && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /run/sshd

RUN curl -sSL https://pdm-project.org/install-pdm.py | python3 -
RUN python -m pip install --upgrade pip

COPY pyproject.toml pdm.lock ./
RUN pdm lock --check

COPY --from=datura . /datura

RUN --mount=type=cache,target=/tmp/pdm_cache \
  pdm config cache_dir /tmp/pdm_cache && \
  pdm config python.use_venv False && \
  PDM_BUILD_SCM_VERSION=0 pdm sync --prod --group :all

RUN mkdir -p /opt/ && mv __pypackages__/3.11/ /opt/pypackages/

ENV PYTHONUNBUFFERED=1
ENV PATH=/opt/pypackages/bin:$PATH
ENV PYTHONPATH=/opt/pypackages/lib:$PYTHONPATH

COPY . .

RUN /usr/sbin/sshd

EXPOSE 8000

ENTRYPOINT [ "./run.sh" ]
