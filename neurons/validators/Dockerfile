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
RUN pdm install --prod

COPY . .

RUN mv libdmcompverify.so /usr/lib/
RUN mv libverifyx.so /usr/lib/
RUN mv libinspector.so /usr/lib/
# DAH-3137: GPU hardware-signature verifier (built in celium-gpu-verifier, published here).
# Provenance: the file must match the SHA-256 recorded in libgpusig.so.sha256 (copied from
# celium-gpu-verifier's gpusig/SHA256SUMS at publish time), or the image build fails. A
# blob-only change without a matching digest line is therefore never shipped unreviewed.
RUN sha256sum -c libgpusig.so.sha256 && mv libgpusig.so /usr/lib/

ENV ENV=prod
ENV USE_TORCH=0
ENV PYTHONUNBUFFERED=1

CMD ["bash", "run.sh"]
