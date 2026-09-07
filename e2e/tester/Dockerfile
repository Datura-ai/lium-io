# The validator image (neurons/validators/Dockerfile, step for step) plus pytest. The suites run inside it and call
# the validator's own services against the stack's miner and executor; build context is the repo root.
ARG BASE_IMAGE=python:3.11-slim
FROM $BASE_IMAGE

RUN apt-get update \
  && apt-get install -y --no-install-recommends make wget gcc ccache patchelf openssh-client g++ \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install -U pdm
ENV PDM_CHECK_UPDATE=false

COPY neurons/validators/pyproject.toml neurons/validators/pdm.lock neurons/validators/README.md ./
COPY --from=datura . /datura

RUN pdm lock --check
RUN pdm install --prod
RUN .venv/bin/pip install --no-cache-dir pytest==8.4.2 pytest-timeout==2.4.0 pytest-asyncio==1.2.0

COPY neurons/validators/ .

RUN mv libdmcompverify.so /usr/lib/ && mv libverifyx.so /usr/lib/ && mv libinspector.so /usr/lib/

ENV ENV=prod
ENV USE_TORCH=0
ENV PYTHONUNBUFFERED=1

COPY e2e/tester/entrypoint.sh /e2e-entrypoint.sh
RUN chmod +x /e2e-entrypoint.sh
ENTRYPOINT ["/e2e-entrypoint.sh"]
CMD ["pytest", "--help"]
