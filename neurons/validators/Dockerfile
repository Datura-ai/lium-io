ARG BASE_IMAGE=python:3.11-slim

FROM $BASE_IMAGE

RUN apt-get update \
  && apt-get install -y make wget gcc ccache patchelf openssh-client g++ \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# pdm[locked]: pdm's own dependency pins; plain pdm took hishel 2.0.0, which crashes `pdm install` (DAH-5825)
RUN pip install "pdm[locked]==2.29.3" "unearth<0.19"
ENV PDM_CHECK_UPDATE=false

COPY pyproject.toml pdm.lock README.md ./
COPY --from=datura . /datura

RUN pdm lock --check
RUN pdm install --prod

COPY . .

RUN mv libdmcompverify.so /usr/lib/
RUN mv libverifyx.so /usr/lib/
RUN mv libinspector.so /usr/lib/
# the aarch64 builds the validator checks ARM executors against (core/checksums.arch_lib_path)
RUN for f in lib*-aarch64.so; do [ ! -e "$f" ] || mv "$f" /usr/lib/; done

ENV ENV=prod
ENV USE_TORCH=0
ENV PYTHONUNBUFFERED=1
# websockets reconnect backoff (pdm.lock pins 16.1, which reads these at import): after a backend restart the
# connector must be back within the backend's 10 s rent wait; the default grows to 90 s (55.6 s waits seen 12 Nov)
ENV WEBSOCKETS_BACKOFF_INITIAL_DELAY=1 WEBSOCKETS_BACKOFF_MIN_DELAY=1 WEBSOCKETS_BACKOFF_MAX_DELAY=8

CMD ["bash", "run.sh"]
