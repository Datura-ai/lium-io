ARG BASE_IMAGE=python:3.11-slim


FROM $BASE_IMAGE AS base-image
LABEL builder=true

WORKDIR /root/app/

RUN pip3 install -U pdm
ENV PDM_CHECK_UPDATE=false

RUN apt-get update && apt-get install -y git
COPY pyproject.toml pdm.lock README.md ./
RUN pdm lock --check
COPY --from=datura . /datura

RUN --mount=type=cache,target=/tmp/pdm_cache \
  pdm config cache_dir /tmp/pdm_cache && \
  pdm config python.use_venv False && \
  PDM_BUILD_SCM_VERSION=0 pdm sync --prod --group :all
RUN mkdir -p /opt/ && mv __pypackages__/3.11/ /opt/pypackages/

ENV PATH=/opt/pypackages/bin:$PATH
ENV PYTHONPATH=/opt/pypackages/lib:/root/app

COPY . .

FROM nvidia/cuda:12.8.0-devel-ubuntu24.04 AS cuda-build

RUN apt-get update && apt-get install -y cmake g++ wget \
  && apt-get install -y nvidia-cuda-toolkit
  
FROM $BASE_IMAGE AS secondary-image
LABEL builder=false

RUN apt-get update \
  && apt-get install -y make wget gcc ccache patchelf \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /root/app/

ENV ENV=prod
ENV USE_TORCH=0
ENV PYTHONUNBUFFERED=1
ENV PATH=/opt/pypackages/bin:$PATH
ENV PYTHONPATH=/opt/pypackages/lib:/root/app
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

RUN echo "export PYTHONPATH=$PYTHONPATH" >> /etc/profile
RUN echo "export PYTHONPATH=$PYTHONPATH" >> ~/.bashrc
RUN echo "export PYTHONPATH=$PYTHONPATH" >> ~/.bash_profile

COPY --from=base-image /root/app/ /root/app/
COPY --from=base-image /opt/pypackages/ /opt/pypackages/
COPY --from=cuda-build /usr/local/cuda /usr/local/cuda

CMD ["bash", "run.sh"]
