ARG BASE_IMAGE=python:3.11-slim

FROM $BASE_IMAGE

RUN apt-get update \
  && apt-get install -y make wget gcc ccache patchelf openssh-client g++ \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install -U pdm
ENV PDM_CHECK_UPDATE=false

COPY pyproject.toml pdm.lock README.md ./
COPY --from=datura . /datura

RUN pdm lock --check
RUN pdm install --prod --no-editable

COPY . .

RUN mv libdmcompverify.so /usr/lib/
RUN mv libverifyx.so /usr/lib/

ENV ENV=prod
ENV USE_TORCH=0
ENV PYTHONUNBUFFERED=1

CMD ["bash", "run.sh"]
