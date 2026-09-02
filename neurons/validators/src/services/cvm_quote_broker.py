"""Quote-only broker in front of the dstack guest agent for customer pods on CVM nodes (DAH-2828).

A TEE workload wants to take its own TDX quote from inside the pod, and the dstack SDK dials
``/var/run/dstack.sock`` for that. The guest agent behind that socket also serves ``GetKey`` /
``GetTlsKey`` (derive the guest's app keys, shared by the executor and every pod on the node) and
``EmitEvent`` (extends RTMR3), so the raw socket must never reach a renter. The validator therefore
runs one nginx container per guest that listens on its own socket, forwards only ``GetQuote``,
``Info`` and ``Version`` to the agent and answers everything else with 403. The pod gets THAT
socket bind-mounted at the SDK's default path.
"""

import asyncio
import hashlib
import logging

import asyncssh

from core.utils import _m, get_extra_info
from services.rental_docker_sdk import (
    ContainerExecSpec,
    ContainerRunSpec,
    RentalDockerOperationError,
    RentalDockerSdkClient,
    VolumeMount,
)

logger = logging.getLogger(__name__)

# The dstack guest agent's socket on the guest — also the path the dstack SDK dials inside a pod.
DSTACK_GUEST_SOCKET_PATH = "/var/run/dstack.sock"

QUOTE_BROKER_CONTAINER_NAME = "lium-dstack-quote-broker"
QUOTE_BROKER_SOCKET_DIR = "/var/run/lium-dstack"
QUOTE_BROKER_SOCKET_PATH = f"{QUOTE_BROKER_SOCKET_DIR}/dstack.sock"
# nginx:1.27-alpine, pinned by digest: the broker runs on the provider's guest next to the pod
# that trusts it, so a moving tag must not be able to change what the pod talks to.
QUOTE_BROKER_IMAGE = (
    "docker.io/library/nginx@sha256:65645c7bb6a0661892a8b03b89d0743208a18dd2f3f17a54ef4b76fb8e2f2a10"
)
QUOTE_BROKER_ALLOWED_METHODS = ("GetQuote", "Info", "Version")
# The SDK posts to "/<Method>"; the "/prpc/DstackGuest.<Method>" spelling is the agent's own.
QUOTE_BROKER_ALLOWED_PATH_RE = r"^/(prpc/)?(DstackGuest\.)?(GetQuote|Info|Version)$"
QUOTE_BROKER_REV_ENV = "LIUM_QUOTE_BROKER_REV"
QUOTE_BROKER_SOCKET_WAIT_SECONDS = 15
_ALLOWED_METHODS_TEXT = ", ".join(QUOTE_BROKER_ALLOWED_METHODS)

# `user root`: the workers must connect to the agent socket, whose mode is dstack's, not ours.
_NGINX_CONF = f"""\
user root;
worker_processes 1;
error_log /dev/stderr warn;
pid /tmp/nginx.pid;
events {{ worker_connections 64; }}
http {{
  access_log /dev/stdout;
  default_type application/json;
  upstream dstack_guest {{ server unix:{DSTACK_GUEST_SOCKET_PATH}; }}
  server {{
    listen unix:{QUOTE_BROKER_SOCKET_PATH};
    location ~ {QUOTE_BROKER_ALLOWED_PATH_RE} {{
      proxy_pass http://dstack_guest;
      proxy_http_version 1.1;
      proxy_set_header Host $host;
    }}
    location / {{
      return 403 '{{"error":"lium quote broker: only {_ALLOWED_METHODS_TEXT} reach the dstack guest agent from a pod"}}';
    }}
  }}
}}
"""

# A pod bind that raced the broker (guest reboot) leaves a directory at the socket path and a
# previous broker leaves a stale socket; nginx refuses to bind over either.
_ENTRYPOINT = (
    f"rm -rf {QUOTE_BROKER_SOCKET_PATH}"
    " && printf '%s' \"$LIUM_QUOTE_BROKER_NGINX_CONF\" > /etc/nginx/nginx.conf"
    " && exec nginx -g 'daemon off;'"
)


def quote_broker_revision() -> str:
    # anything that changes what the broker does changes the revision, so a running broker
    # from an older validator release gets replaced instead of trusted
    digest = hashlib.sha256(f"{QUOTE_BROKER_IMAGE}\n{_ENTRYPOINT}\n{_NGINX_CONF}".encode()).hexdigest()
    return digest[:16]


def quote_broker_run_spec() -> ContainerRunSpec:
    return ContainerRunSpec(
        image=QUOTE_BROKER_IMAGE,
        name=QUOTE_BROKER_CONTAINER_NAME,
        entrypoint="/bin/sh",
        command=("-c", _ENTRYPOINT),
        environment={
            "LIUM_QUOTE_BROKER_NGINX_CONF": _NGINX_CONF,
            QUOTE_BROKER_REV_ENV: quote_broker_revision(),
        },
        volumes=(
            VolumeMount(source=DSTACK_GUEST_SOCKET_PATH, target=DSTACK_GUEST_SOCKET_PATH, read_only=True),
            VolumeMount(source=QUOTE_BROKER_SOCKET_DIR, target=QUOTE_BROKER_SOCKET_DIR),
        ),
        restart_policy="unless-stopped",
    )


def quote_socket_pod_mount() -> VolumeMount:
    # connect() on a unix socket does not need a writable mount, so the pod gets it read-only
    return VolumeMount(source=QUOTE_BROKER_SOCKET_PATH, target=DSTACK_GUEST_SOCKET_PATH, read_only=True)


async def ensure_quote_broker(
    docker_client: RentalDockerSdkClient,
    ssh_client: asyncssh.SSHClientConnection,
    *,
    log_extra: dict,
) -> None:
    # bring the guest's broker to this release's revision and wait until its socket exists
    spec = quote_broker_run_spec()
    revision = spec.environment[QUOTE_BROKER_REV_ENV]
    if not await _broker_running_at_revision(ssh_client, revision):
        await _start_broker(docker_client, ssh_client, spec, revision, log_extra)
    # also on the fast path: "Running" means the entrypoint shell started, not that nginx binds yet
    try:
        await asyncio.wait_for(_wait_for_broker_socket(docker_client), QUOTE_BROKER_SOCKET_WAIT_SECONDS)
    except TimeoutError:
        raise RuntimeError(
            f"quote broker socket {QUOTE_BROKER_SOCKET_PATH} did not appear "
            f"within {QUOTE_BROKER_SOCKET_WAIT_SECONDS}s"
        ) from None


async def _start_broker(
    docker_client: RentalDockerSdkClient,
    ssh_client: asyncssh.SSHClientConnection,
    spec: ContainerRunSpec,
    revision: str,
    log_extra: dict,
) -> None:
    logger.info(_m("Starting the CVM quote broker", extra=get_extra_info({**log_extra, "broker_revision": revision})))
    await ssh_client.run(f"/usr/bin/docker rm -fv {QUOTE_BROKER_CONTAINER_NAME} 2>/dev/null || true")
    if not await docker_client.image_exists(image=spec.image):
        await docker_client.pull(image=spec.image)
    try:
        await docker_client.run_container(spec)
    except RentalDockerOperationError:
        # two pod creates on one guest raced here; the other one's broker is the same broker
        if not await _broker_running_at_revision(ssh_client, revision):
            raise


async def _broker_running_at_revision(ssh_client: asyncssh.SSHClientConnection, revision: str) -> bool:
    result = await ssh_client.run(
        f"/usr/bin/docker inspect -f '{{{{.State.Running}}}} {{{{.Config.Env}}}}' "
        f"{QUOTE_BROKER_CONTAINER_NAME} 2>/dev/null || true"
    )
    output = (result.stdout or "").strip()
    return output.startswith("true ") and f"{QUOTE_BROKER_REV_ENV}={revision}" in output


async def _wait_for_broker_socket(docker_client: RentalDockerSdkClient) -> None:
    # nginx creates the listening socket a moment after the container starts; a pod bind
    # issued before that would make dockerd create a directory at the socket path instead
    probe = ContainerExecSpec(
        container_name=QUOTE_BROKER_CONTAINER_NAME,
        argv=("test", "-S", QUOTE_BROKER_SOCKET_PATH),
    )
    while (await docker_client.exec_in_container(probe)).exit_status != 0:
        await asyncio.sleep(0.5)
