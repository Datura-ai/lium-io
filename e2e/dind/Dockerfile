# The executor host's docker daemon for the e2e stack (CI, no GPU).
#
# The validator starts every rental with `docker run --gpus …` (services/nvidia_devices.py). dockerd resolves that
# flag through its `nvidia` device driver, which only needs an `nvidia-container-runtime-hook` binary on its PATH;
# with no GPU on the machine a hook that does nothing lets the container start GPU-less, so the whole rental
# path — pull, run, port mapping, ssh key, exec, delete — runs on any runner. With E2E_GPU=1 the stack does not
# use this daemon at all: docker-compose.gpu.yml points the executor at the host's own dockerd and GPUs.
FROM docker:27-dind
COPY nvidia-container-runtime-hook /usr/bin/nvidia-container-runtime-hook
RUN chmod +x /usr/bin/nvidia-container-runtime-hook
