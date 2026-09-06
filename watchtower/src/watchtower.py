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


def get_current_image_digest(client: docker.DockerClient, container_name: str) -> Optional[str]:
    """
    Get the current digest of the image running inside the given container.

    Args:
        client: Docker client instance
        container_name: Name of the running container (e.g., "executor-runner")

    Returns:
        Image digest (e.g., "sha256:abc123...") or None if not found
    """
    try:
        container = client.containers.get(container_name)
        image_id = container.attrs.get("Image")
        if not image_id:
            logger.warning(_m("Container has no image ID", {"container": container_name}))
            return None
        image = client.images.get(image_id)
        repo_digests = image.attrs.get("RepoDigests", [])
        if repo_digests:
            digest = repo_digests[0].split("@")[1]
            return digest
        return None
    except docker.errors.NotFound:
        logger.warning(_m("Container not found", {"container": container_name}))
        return None
    except Exception as e:
        logger.error(_m("Error getting image digest from container", {"container": container_name, "error": str(e)}))
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


def pull_and_restart_containers(client: docker.DockerClient, image_name: str, remote_digest: str) -> bool:
    """
    Pull the latest image and restart the executor-runner container.
    Stops and removes the old container, then creates a new one with proper configuration.

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
        client.images.pull(f"{image_name}@{remote_digest}")
        logger.info(_m("Successfully pulled new image", {"image": f"{image_name}@{remote_digest}"}))

        # Find the existing container
        container = find_container_by_name(client, container_name)

        if container:
            # Stop and remove the old container
            try:
                logger.info(_m("Stopping existing container", {
                    "name": container_name,
                    "id": container.short_id
                }))
                container.stop(timeout=10)
                logger.info(_m("Removing old container", {"name": container_name}))
                container.remove()
                logger.info(_m("Successfully removed old container", {"name": container_name}))
            except Exception as e:
                logger.error(_m("Failed to stop/remove old container", {
                    "name": container_name,
                    "error": str(e)
                }))
                return False

        # Create new container with proper configuration from docker-compose.yml
        try:
            logger.info(_m("Creating new container with updated image", {
                "name": container_name,
                "image": f"{image_name}@{remote_digest}"
            }))

            new_container = client.containers.run(
                f"{image_name}@{remote_digest}",
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
                "image": f"{image_name}@{remote_digest}",
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
        container_name = EXECUTOR_RUNNER_CONTAINER_NAME

        current_digest = get_current_image_digest(client, container_name)
        logger.info(_m("Current image digest", {"digest": current_digest, "container": container_name}))

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
