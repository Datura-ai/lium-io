# The rental image the e2e validator starts on the executor: the smallest thing that looks like a Lium template to
# sshd_bootstrap.sh (openssh-server present, /root/.ssh writable, stays up). No CUDA: GPU access is asserted through
# the device nodes docker injects, not through libraries in the image.
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends openssh-server ca-certificates && rm -rf /var/lib/apt/lists/* \
  && mkdir -p /run/sshd /root/.ssh && chmod 700 /root/.ssh
# bash traps TERM: docker stop sends it to PID 1, and a bare sleep ignores it until the delete's 30 s grace
# (CONTAINER_STOP_GRACE_SECONDS) expires, which the rental suite pays on every teardown
CMD ["bash", "-c", "trap 'exit 0' TERM INT; sleep infinity & wait"]
