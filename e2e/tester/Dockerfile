# The validator image (neurons/validators/Dockerfile, step for step) plus pytest. Context is neurons/validators
# so its .dockerignore keeps a local .venv / .pdm-python from replacing the image interpreter.
ARG BASE_IMAGE=python:3.11-slim
FROM $BASE_IMAGE

RUN apt-get update \
  && apt-get install -y --no-install-recommends make wget gcc ccache patchelf openssh-client g++ \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install -U pdm
ENV PDM_CHECK_UPDATE=false

COPY pyproject.toml pdm.lock README.md ./
COPY --from=datura . /datura

RUN pdm lock --check
RUN pdm install --prod
RUN pdm run python -m ensurepip --upgrade >/dev/null && pdm run python -m pip install --no-cache-dir -q pytest==8.4.2 pytest-timeout==2.4.0 requests

COPY . .

RUN mv libdmcompverify.so /usr/lib/ && mv libverifyx.so /usr/lib/ && mv libinspector.so /usr/lib/

ENV ENV=prod
ENV USE_TORCH=0
ENV PYTHONUNBUFFERED=1

COPY --from=e2e tester/entrypoint.sh /e2e-entrypoint.sh
RUN chmod +x /e2e-entrypoint.sh
ENTRYPOINT ["/e2e-entrypoint.sh"]
CMD ["pytest", "--help"]
