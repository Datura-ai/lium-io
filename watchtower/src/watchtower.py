import docker
import requests
import bittensor
import time
from datetime import datetime, UTC
from typing import Optional
from docker.models.containers import Container

from config import settings, WATCHTOWER_ENDPOINT_URL, WATCHTOWER_VALIDATOR_HOTKEY
from logger import get_logger, _m
from models import WatchtowerDigestResponse

logger = get_logger(__name__)

EXECUTOR_RUNNER_CONTAINER_NAME = "executor-runner"
# The compose service name of the runner in neurons/executor/docker-compose.yml. The container
# is `<project>-executor-runner-1` there (`executor-executor-runner-1` under the default project),
# so the standard stack is found by this label when the CVM name above does not exist.
EXECUTOR_RUNNER_SERVICE_LABEL = "com.docker.compose.service=executor-runner"
# Docker Hub's registry host. `registry-mirrors` in the daemon apply to `docker.io` names only,
# so a reference that names this host goes straight to Docker Hub.
CANONICAL_REGISTRY_HOST = "registry-1.docker.io"


class DigestMismatchError(Exception):
    """The image the daemon returned for `image@digest` does not carry that digest."""


class RunnerLookupError(Exception):
    """The runner container could not be identified this cycle (ambiguous label or a listing error)."""


def image_has_digest(image, digest: str) -> bool:
    """True when one of the image's RepoDigests ends in `@<digest>`."""
    return any(entry.split("@")[-1] == digest for entry in image.attrs.get("RepoDigests", []) or [])


def pull_image_by_digest(client: docker.DockerClient, image_name: str, remote_digest: str) -> str:
    """
    Pull `image_name@remote_digest` and return the reference that names the pulled image.

    A by-digest pull is content-addressed: the daemon checks the manifest hash, so a
    registry mirror in `/etc/docker/daemon.json` can answer with the right bytes or with
    an error, never with an old image under a new name (that is what a tag pull lets it
    do). When the normal path fails, the same digest is pulled from
    `registry-1.docker.io` directly, which the daemon's `registry-mirrors` do not cover.

    Raises:
        docker.errors.APIError: both pulls failed
        DigestMismatchError: the daemon returned an image without the requested digest
    """
    reference = f"{image_name}@{remote_digest}"
    try:
        image = client.images.pull(reference)
        if not image_has_digest(image, remote_digest):
            raise DigestMismatchError(f"{reference} resolved to {image.attrs.get('RepoDigests')}")
        return reference
    except (docker.errors.APIError, DigestMismatchError) as e:
        logger.warning(_m("Pull by digest failed through the daemon's registry path, retrying from Docker Hub directly", {
            "image": reference,
            "error": str(e),
        }))

    reference = f"{CANONICAL_REGISTRY_HOST}/{image_name}@{remote_digest}"
    image = client.images.pull(reference)
    if not image_has_digest(image, remote_digest):
        raise DigestMismatchError(f"{reference} resolved to {image.attrs.get('RepoDigests')}")
    return reference


def find_runner_container(client: docker.DockerClient) -> Optional[Container]:
    """
    Find the executor-runner container: by the CVM name first, then by the compose service label.

    Returns:
        The container, or None when there is none (a first boot).

    Raises:
        RunnerLookupError: the label matches more than one container (for example while
            `docker compose up -d` renames the old runner and creates the new one), or the
            listing failed. The caller must not pull or create anything in that state.
    """
    container = find_container_by_name(client, EXECUTOR_RUNNER_CONTAINER_NAME)
    if container is not None:
        return container
    try:
        matches = client.containers.list(all=True, filters={"label": EXECUTOR_RUNNER_SERVICE_LABEL})
    except Exception as e:
        raise RunnerLookupError(f"listing containers by label {EXECUTOR_RUNNER_SERVICE_LABEL} failed: {e}") from e
    if len(matches) > 1:
        raise RunnerLookupError(
            f"{len(matches)} containers carry {EXECUTOR_RUNNER_SERVICE_LABEL}: {[c.name for c in matches]}"
        )
    if not matches:
        logger.info(_m("No runner container", {"name": EXECUTOR_RUNNER_CONTAINER_NAME, "label": EXECUTOR_RUNNER_SERVICE_LABEL}))
        return None
    logger.info(_m("Found runner container by label", {"name": matches[0].name, "id": matches[0].short_id}))
    return matches[0]


def container_image_digest(client: docker.DockerClient, container: Container) -> Optional[str]:
    """Digest (`sha256:…`) of the image a container runs, from the image's RepoDigests, or None."""
    try:
        image_id = container.attrs.get("Image")
        if not image_id:
            logger.warning(_m("Container has no image ID", {"container": container.name}))
            return None
        image = client.images.get(image_id)
        repo_digests = image.attrs.get("RepoDigests", [])
        if repo_digests:
            digest = repo_digests[0].split("@")[1]
            return digest
        return None
    except Exception as e:
        logger.error(_m("Error getting image digest from container", {"container": container.name, "error": str(e)}))
        return None


def verify_watchtower_signature(payload: WatchtowerDigestResponse) -> None:
    """
    Verify the signature of the watchtower digest response.

    Raises:
        Exception: If signature verification fails
    """
    try:
        keypair = bittensor.Keypair(ss58_address=WATCHTOWER_VALIDATOR_HOTKEY)
        signing_data = f"{payload.digest}:{payload.timestamp}"
        is_valid = keypair.verify(signing_data, payload.signature)

        # Verify that the timestamp is not too far in the future or past (e.g., within 5 minutes)
        now = int(datetime.now(UTC).timestamp())
        max_skew = 10 * 60  # 10 minutes
        if abs(payload.timestamp - now) > max_skew:
            raise Exception(
                f"Digest response timestamp out of allowed range (now={now}, ts={payload.timestamp})"
            )

        if not is_valid:
            raise Exception(
                f"Invalid signature from validator {WATCHTOWER_VALIDATOR_HOTKEY}"
            )

    except Exception as e:
        logger.error(_m("Signature verification failed", {"error": str(e)}))
        raise


def fetch_verified_digest() -> Optional[str]:
    """
    Fetch the latest authorized digest from the validator endpoint.
    Verifies the signature before returning.

    Returns:
        Verified digest string or None if fetch/verification fails
    """
    try:
        response = requests.get(
            WATCHTOWER_ENDPOINT_URL,
            timeout=30
        )
        response.raise_for_status()

        data = response.json()
        payload = WatchtowerDigestResponse(**data)

        verify_watchtower_signature(payload)

        logger.info(_m("Successfully fetched and verified digest", {
            "digest": payload.digest,
            "timestamp": payload.timestamp
        }))

        return payload.digest

    except requests.RequestException as e:
        logger.error(_m("Failed to fetch digest from endpoint", {
            "url": WATCHTOWER_ENDPOINT_URL,
            "error": str(e)
        }))
        return None
    except Exception as e:
        logger.error(_m("Error in fetch_verified_digest", {"error": str(e)}))
        return None


def find_container_by_name(client: docker.DockerClient, container_name: str) -> Optional[Container]:
    """
    Find a container by its name.

    Args:
        client: Docker client instance
        container_name: Name of the container to find

    Returns:
        Container object or None if not found
    """
    try:
        container = client.containers.get(container_name)
        logger.info(_m("Found container", {
            "name": container_name,
            "id": container.short_id,
            "status": container.status
        }))
        return container
    except docker.errors.NotFound:
        logger.info(_m("Container not found", {"name": container_name}))
        return None
    except Exception as e:
        logger.error(_m("Error finding container", {"name": container_name, "error": str(e)}))
        return None

def _inherited_or_none(container_value, image_value):
    """A Cmd/Entrypoint the old container took from its image is left to the new image."""
    if container_value is None or container_value == image_value:
        return None
    return container_value


def container_create_kwargs(client: docker.DockerClient, old: Container, image_reference: str) -> dict:
    """
    Arguments for `client.api.create_container` that rebuild `old` from `image_reference`.

    Name, command, entrypoint, environment, labels, working directory, user, the whole
    HostConfig (binds, restart policy, ports, privileges) and the network endpoints are
    copied from the old container, so a runner created by compose keeps its compose
    labels, its `.env` bind and its restart policy. A Cmd or Entrypoint that the old
    container inherited from its image is not copied: the new image decides it.
    """
    attrs = old.attrs
    config = attrs.get("Config") or {}
    old_image_config = {}
    try:
        old_image_config = client.images.get(attrs["Image"]).attrs.get("Config") or {}
    except Exception as e:
        logger.warning(_m("Old image not inspectable, copying Cmd/Entrypoint as they are", {"error": str(e)}))

    networks = (attrs.get("NetworkSettings") or {}).get("Networks") or {}
    endpoints = {}
    for network_name, endpoint in networks.items():
        # docker adds the short container id as an alias; the new container gets its own
        aliases = [alias for alias in (endpoint.get("Aliases") or []) if not old.id.startswith(alias)]
        endpoints[network_name] = client.api.create_endpoint_config(aliases=aliases or None)

    return {
        "image": image_reference,
        "name": attrs.get("Name", "").lstrip("/") or None,
        "command": _inherited_or_none(config.get("Cmd"), old_image_config.get("Cmd")),
        "entrypoint": _inherited_or_none(config.get("Entrypoint"), old_image_config.get("Entrypoint")),
        # Env is copied whole: compose-set variables must survive, and the image-provided
        # part (PATH and the like) is the same in every runner build.
        "environment": config.get("Env"),
        "labels": config.get("Labels"),
        "working_dir": config.get("WorkingDir") or None,
        "user": config.get("User") or None,
        "host_config": attrs.get("HostConfig") or {},
        "networking_config": client.api.create_networking_config(endpoints) if endpoints else None,
    }


def recreate_container(client: docker.DockerClient, old: Container, image_reference: str) -> Container:
    """
    Replace `old` with a container of the same configuration built from `image_reference`.

    Order: the old container is renamed aside, the new one is created under the old name,
    the old one is stopped, the new one is started, and only then is the old one removed.
    A failure to create puts the old name back. A failure to stop the old one or to start
    the new one removes the new container, puts the old name back and starts the old one
    again. The host keeps a runner in every case.
    """
    kwargs = container_create_kwargs(client, old, image_reference)
    name = kwargs["name"]
    previous_name = f"{name}-previous-{old.short_id}"
    logger.info(_m("Renaming existing container aside", {"name": name, "id": old.short_id, "to": previous_name}))
    old.rename(previous_name)
    try:
        logger.info(_m("Creating new container with updated image", {"name": name, "image": image_reference}))
        created = client.api.create_container(**kwargs)
    except Exception:
        old.rename(name)
        raise
    try:
        logger.info(_m("Stopping existing container", {"name": previous_name, "id": old.short_id}))
        old.stop(timeout=10)
        client.api.start(created["Id"])
    except Exception:
        logger.error(_m("Switch to the new container failed, restoring the old one", {"name": name, "image": image_reference}))
        client.api.remove_container(created["Id"], force=True)
        old.rename(name)
        old.start()
        raise
    logger.info(_m("Removing old container", {"name": previous_name}))
    old.remove(force=True)
    new_container = client.containers.get(created["Id"])
    logger.info(_m("Successfully created new container", {"name": new_container.name, "id": new_container.short_id}))
    return new_container


def pull_and_restart_containers(client: docker.DockerClient, image_name: str, remote_digest: str) -> bool:
    """
    Pull the image by digest and restart the executor-runner container from it.

    An existing runner (CVM name or compose service label) is rebuilt with its own
    configuration. When there is none, a runner is created with the CVM stack's
    configuration (docker socket and the executor `.env` file).

    Args:
        client: Docker client instance
        image_name: Image to pull
        remote_digest: The digest of the remote image

    Returns:
        True if successful, False otherwise
    """
    container_name = EXECUTOR_RUNNER_CONTAINER_NAME

    try:
        # Pull the new image
        logger.info(_m("Pulling new image", {"image": f"{image_name}@{remote_digest}"}))
        image_reference = pull_image_by_digest(client, image_name, remote_digest)
        logger.info(_m("Successfully pulled new image", {"image": image_reference}))

        # Find the existing container
        try:
            container = find_runner_container(client)
        except RunnerLookupError as e:
            logger.warning(_m("Runner container not identified, nothing recreated", {"error": str(e)}))
            return False

        if container:
            try:
                recreate_container(client, container, image_reference)
                return True
            except Exception as e:
                logger.error(_m("Failed to recreate container", {
                    "name": container.name,
                    "image": image_reference,
                    "error": str(e)
                }))
                return False

        # No runner yet: create one with the CVM stack's configuration
        try:
            logger.info(_m("Creating new container with updated image", {
                "name": container_name,
                "image": image_reference
            }))

            new_container = client.containers.run(
                image_reference,
                name=container_name,
                detach=True,
                restart_policy={"Name": "unless-stopped"},
                volumes={
                    "/var/run/docker.sock": {"bind": "/var/run/docker.sock", "mode": "rw"},
                    settings.WATCHTOWER_ENV_FILE_PATH: {"bind": "/root/executor/.env", "mode": "rw"}
                },
            )

            logger.info(_m("Successfully created new container", {
                "name": container_name,
                "id": new_container.short_id
            }))
            return True

        except Exception as e:
            logger.error(_m("Failed to create new container", {
                "name": container_name,
                "image": image_reference,
                "error": str(e)
            }))
            return False

    except docker.errors.APIError as e:
        logger.error(_m("Docker API error during pull/restart", {"error": str(e)}))
        return False
    except Exception as e:
        logger.error(_m("Unexpected error during pull/restart", {"error": str(e)}))
        return False


def check_and_update() -> None:
    """
    Single iteration: check for updates and apply if needed.
    """
    try:
        client = docker.from_env()
        image_name = settings.WATCHTOWER_IMAGE

        try:
            container = find_runner_container(client)
        except RunnerLookupError as e:
            logger.warning(_m("Runner container not identified this cycle, nothing pulled", {"error": str(e)}))
            return
        current_digest = container_image_digest(client, container) if container else None
        logger.info(_m("Current image digest", {
            "digest": current_digest,
            "container": container.name if container else None,
        }))

        remote_digest = fetch_verified_digest()
        if not remote_digest:
            logger.warning(_m("Could not fetch/verify remote digest, continuing..."))
            return

        logger.info(_m("Remote verified digest", {"digest": remote_digest}))

        if current_digest == remote_digest:
            logger.info(_m("Image is up to date", extra={"image": image_name}))
            return

        logger.info(_m("Image update detected", {
            "current": current_digest,
            "remote": remote_digest
        }))

        success = pull_and_restart_containers(client, image_name, remote_digest)
        if success:
            logger.info(_m("Successfully updated to new image", extra={"image": image_name}))
        else:
            logger.error(_m("Failed to update image", extra={"image": image_name}))

    except Exception as e:
        logger.error(_m("Error in check_and_update", {"error": str(e)}), exc_info=True)


def main():
    """
    Main loop: run watchtower permanently.
    """
    if not settings.WATCHTOWER_ENABLED:
        logger.info("Watchtower is disabled (WATCHTOWER_ENABLED=False)")
        return

    logger.info(_m("Starting watchtower", {
        "image": settings.WATCHTOWER_IMAGE,
        "interval": settings.WATCHTOWER_INTERVAL,
        "endpoint": WATCHTOWER_ENDPOINT_URL,
        "validator_hotkey": WATCHTOWER_VALIDATOR_HOTKEY
    }))

    while True:
        try:
            check_and_update()
        except Exception as e:
            logger.error(_m("Unexpected error in main loop", {"error": str(e)}), exc_info=True)

        logger.info(_m("Sleeping until next check", {"interval_seconds": settings.WATCHTOWER_INTERVAL}))
        time.sleep(settings.WATCHTOWER_INTERVAL)


if __name__ == "__main__":
    main()
