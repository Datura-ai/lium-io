ARG BASE_IMAGE=python:3.11-slim

FROM $BASE_IMAGE

WORKDIR /root/app

RUN apt-get update \
  && apt-get install -y wget git \
  && rm -rf /var/lib/apt/lists/*

RUN pip install -U pdm
ENV PDM_CHECK_UPDATE=false

COPY pyproject.toml pdm.lock README.md ./
COPY --from=datura . /datura

RUN pdm lock --check
RUN pdm install --prod

COPY . .

ENV ENV=prod
ENV PYTHONUNBUFFERED=1

CMD ["bash", "run.sh"]