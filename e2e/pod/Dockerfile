# The rental image the e2e validator starts on the executor: the smallest thing that looks like a Lium template to
# sshd_bootstrap.sh (openssh-server present, /root/.ssh writable, stays up). No CUDA: GPU access is asserted through
# the device nodes docker injects, not through libraries in the image.
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends openssh-server ca-certificates && rm -rf /var/lib/apt/lists/* \
  && mkdir -p /run/sshd /root/.ssh && chmod 700 /root/.ssh
CMD ["sleep", "infinity"]
