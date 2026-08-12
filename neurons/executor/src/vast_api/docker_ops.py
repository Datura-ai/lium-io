import base64
import logging
from pathlib import Path

import docker

from vast_api.config import VastSettings
from vast_api.errors import ApiFailure

logger = logging.getLogger(__name__)

ASSETS_DIR = Path(__file__).parent / "assets"

# Verbatim from SUCCESS-PATH.md step 3 — do not "improve".
VAST_UNS_ENTRYPOINT_CMD = (
    "rm -f /etc/systemd/system/docker.service /etc/systemd/system/docker.socket "
    "/etc/systemd/system/containerd.service; mkdir -p /run/sshd; exec /sbin/init"
)

NESTED_GPU_TEST_IMAGE = "nvidia/cuda:12.4.1-base-ubuntu22.04"

# docker-py has no per-call exec timeout; the client-wide HTTP timeout is the only
# knob and it must cover the longest blocking calls (kaalia install ~5 min, image
# build, nested docker pulls) — the library default of 60s would cut them off.
DOCKER_API_TIMEOUT_SECONDS = 3600


class DockerOps:
    """Host-docker operations over docker-py (the shell mounts /var/run/docker.sock)."""

    def __init__(self, settings: VastSettings):
        self.settings = settings
        self._client = None

    @property
    def client(self):
        # lazy: from_env() negotiates the API version over the socket, and a docker
        # hiccup at boot must not stop the executor app from starting
        if self._client is None:
            self._client = docker.from_env(timeout=DOCKER_API_TIMEOUT_SECONDS)
        return self._client

    # --- G0 ---

    def executor_running(self) -> bool:
        try:
            container = self.client.containers.get(self.settings.EXECUTOR_CONTAINER_NAME)
        except docker.errors.NotFound:
            return False
        return container.status == "running"

    def executor_network(self) -> str | None:
        # the network the executor container itself sits on (--network value for vast-uns);
        # the name-contains scan is only a fallback when that read fails
        try:
            container = self.client.containers.get(self.settings.EXECUTOR_CONTAINER_NAME)
            networks = container.attrs["NetworkSettings"]["Networks"]
            if networks:
                return next(iter(networks))
        except Exception:
            logger.warning("reading executor container networks failed", exc_info=True)
        for network in self.client.networks.list():
            if "executor" in network.name:
                return network.name
        return None

    # --- image / container lifecycle ---

    def image_exists(self) -> bool:
        try:
            self.client.images.get(self.settings.VAST_UNS_IMAGE_TAG)
            return True
        except docker.errors.ImageNotFound:
            return False

    def build_image(self) -> None:
        logger.info("building %s from %s", self.settings.VAST_UNS_IMAGE_TAG, ASSETS_DIR)
        self.client.images.build(
            path=str(ASSETS_DIR),
            dockerfile="Dockerfile.vast-uns",
            tag=self.settings.VAST_UNS_IMAGE_TAG,
        )

    def get_vast_uns(self):
        try:
            return self.client.containers.get(self.settings.VAST_UNS_NAME)
        except docker.errors.NotFound:
            return None

    def vast_uns_has_state_mount(self, container) -> bool:
        # the identity-preserving mount is mandatory (state_mount_missing rail)
        for mount in container.attrs.get("Mounts", []):
            if mount.get("Destination") == "/var/lib/vastai_kaalia":
                return True
        return False

    def run_vast_uns(self, network: str) -> None:
        # exact flags from SUCCESS-PATH step 3 + the state/DMI mounts from G0 notes
        s = self.settings
        ports = {}
        for port in range(s.PORT_RANGE_START, s.PORT_RANGE_END + 1):
            ports[f"{port}/tcp"] = port
            ports[f"{port}/udp"] = port
        self.client.containers.run(
            s.VAST_UNS_IMAGE_TAG,
            name=s.VAST_UNS_NAME,
            detach=True,
            privileged=True,
            network=network,
            cgroupns="private",
            security_opt=["label=disable"],
            tmpfs={"/run": "", "/run/lock": ""},
            ports=ports,
            volumes={
                s.DATA_ROOT_MOUNT: {"bind": "/var/lib/docker", "mode": "rw"},
                s.STATE_DIR_HOST: {"bind": "/var/lib/vastai_kaalia", "mode": "rw"},
                s.DMI_BIN_HOST: {"bind": "/var/lib/dmi.bin", "mode": "ro"},
            },
            device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])],
            entrypoint="/bin/bash",
            command=["-c", VAST_UNS_ENTRYPOINT_CMD],
        )

    def vast_rental_containers(self) -> list[str]:
        # a Vast contract IS a container C.<id> in the nested dockerd — and it must be
        # counted in ANY state: a client-stopped instance still holds the contract and
        # its storage, and can restart anytime (verified live on 147063: C.47368067
        # Exited while the client kept the instance; rentals_running showed 0)
        rc, output = self.exec_in_uns(["docker", "ps", "-a", "--format", "{{.Names}}"])
        if rc != 0:
            raise ApiFailure("host_command_failed", f"nested docker ps failed: {output[:300]}")
        return [name for name in output.split() if name.startswith("C.")]

    def delete_vast_uns(self) -> bool:
        container = self.get_vast_uns()
        if container is None:
            return False
        container.remove(force=True)
        return True

    # --- exec into vast-uns ---

    def exec_in_uns(self, cmd: list[str] | str, env: dict[str, str] | None = None) -> tuple[int, str]:
        # run one command inside the vast-uns container, return (rc, combined output);
        # blocking time is bounded by the client-wide DOCKER_API_TIMEOUT_SECONDS
        container = self.get_vast_uns()
        if container is None:
            raise ApiFailure("host_command_failed", "vast-uns container is not running")
        exit_code, output = container.exec_run(cmd, environment=env)
        return exit_code, output.decode(errors="replace") if isinstance(output, bytes) else str(output)

    def write_file_in_uns(self, path: str, content: str) -> None:
        # base64 round-trip for the content; path travels as an env var, never
        # interpolated into the shell line — safe by construction
        encoded = base64.b64encode(content.encode()).decode()
        rc, output = self.exec_in_uns(
            ["bash", "-c", 'mkdir -p "$(dirname "$DEST")" && echo "$CONTENT_B64" | base64 -d > "$DEST"'],
            env={"DEST": path, "CONTENT_B64": encoded},
        )
        if rc != 0:
            raise ApiFailure("host_command_failed", f"writing {path} in vast-uns failed: {output[:500]}")

    def install_nested_daemon(self) -> None:
        # SUCCESS-PATH step 4: nested daemon.json + nvidia-ctk --in-place + docker + chattr +i
        daemon_json = (ASSETS_DIR / "daemon.json").read_text()
        self.exec_in_uns(["chattr", "-i", "/etc/docker/daemon.json"])  # ok to fail on fresh box
        self.write_file_in_uns("/etc/docker/daemon.json", daemon_json)
        rc, output = self.exec_in_uns(
            ["nvidia-ctk", "config", "--in-place", "--set", "nvidia-container-cli.no-cgroups=false"]
        )
        if rc != 0:
            raise ApiFailure("host_command_failed", f"nvidia-ctk config failed: {output[:500]}")
        # restart (not plain start) so a changed daemon.json is actually reloaded;
        # on the fresh path docker is stopped and restart degrades to start.
        rc, output = self.exec_in_uns(["systemctl", "restart", "docker"])
        if rc != 0:
            raise ApiFailure("host_command_failed", f"nested docker start failed: {output[:500]}")
        # immutability is the only protection against kaalia rewriting daemon.json
        rc, output = self.exec_in_uns(["chattr", "+i", "/etc/docker/daemon.json"])
        if rc != 0:
            raise ApiFailure("host_command_failed", f"chattr +i daemon.json failed: {output[-300:]}")

    def nested_daemon_matches_asset(self) -> bool:
        rc, output = self.exec_in_uns(["cat", "/etc/docker/daemon.json"])
        if rc != 0:
            return False
        asset = (ASSETS_DIR / "daemon.json").read_text()
        return output.strip() == asset.strip()

    def nested_docker_active(self) -> bool:
        rc, output = self.exec_in_uns(["systemctl", "is-active", "docker"])
        return rc == 0 and output.strip() == "active"

    def nested_gpu_ok(self) -> bool:
        # a nested --runtime=nvidia container must see the GPU (G1 check)
        rc, output = self.exec_in_uns(
            ["docker", "run", "--rm", "--runtime=nvidia", NESTED_GPU_TEST_IMAGE, "nvidia-smi", "-L"]
        )
        return rc == 0 and "GPU" in output

    # --- status probes ---

    def executor_health(self) -> str:
        try:
            container = self.client.containers.get(self.settings.EXECUTOR_CONTAINER_NAME)
        except docker.errors.NotFound:
            return "missing"
        health = container.attrs.get("State", {}).get("Health", {}).get("Status")
        return health or container.status

    def filler_running(self) -> bool:
        # the Lium idle filler runs as host containers named filler_*
        for container in self.client.containers.list():
            if container.name.startswith("filler_"):
                return True
        return False
