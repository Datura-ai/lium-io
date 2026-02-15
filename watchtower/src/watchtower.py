import docker
import requests
import bittensor
import time
import json
from typing import Optional
from docker.models.containers import Container

from config import settings
from logger import get_logger, _m
from models import WatchtowerDigestResponse

logger = get_logger(__name__)


def get_current_image_digest(client: docker.DockerClient, image_name: str) -> Optional[str]:
    """
    Get the current digest of the monitored Docker image.

    Args:
        client: Docker client instance
        image_name: Full image name (e.g., "daturaai/compute-subnet-executor-runner")

    Returns:
        Image digest (e.g., "sha256:abc123...") or None if not found
    """
    try:
        image = client.images.get(image_name)
        repo_digests = image.attrs.get('RepoDigests', [])
        if repo_digests:
            digest = repo_digests[0].split('@')[1]
            return digest
        return None
    except docker.errors.ImageNotFound:
        logger.warning(_m("Image not found locally", {"image": image_name}))
        return None
    except Exception as e:
        logger.error(_m("Error getting image digest", {"image": image_name, "error": str(e)}))
        return None


def verify_watchtower_signature(payload: WatchtowerDigestResponse) -> None:
    """
    Verify the signature of the watchtower digest response.

    Raises:
        Exception: If signature verification fails
    """
    try:
        keypair = bittensor.Keypair(ss58_address=settings.WATCHTOWER_VALIDATOR_HOTKEY)

        signing_data = {
            "digest": payload.digest,
            "timestamp": payload.timestamp,
        }
        message = json.dumps(signing_data, sort_keys=True)

        signature = payload.signature
        if not signature.startswith('0x'):
            signature = '0x' + signature

        is_valid = keypair.verify(message, signature)

        if not is_valid:
            raise Exception(
                f"Invalid signature from validator {settings.WATCHTOWER_VALIDATOR_HOTKEY}"
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
            settings.WATCHTOWER_ENDPOINT_URL,
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
            "url": settings.WATCHTOWER_ENDPOINT_URL,
            "error": str(e)
        }))
        return None
    except Exception as e:
        logger.error(_m("Error in fetch_verified_digest", {"error": str(e)}))
        return None


def find_containers_by_image(client: docker.DockerClient, image_name: str) -> list[Container]:
    """
    Find all running containers using the specified image.

    Args:
        client: Docker client instance
        image_name: Image name to match

    Returns:
        List of Container objects
    """
    try:
        containers = client.containers.list(filters={"ancestor": image_name})
        logger.info(_m("Found containers using image", {
            "image": image_name,
            "count": len(containers),
            "containers": [c.name for c in containers]
        }))
        return containers
    except Exception as e:
        logger.error(_m("Error finding containers", {"error": str(e)}))
        return []


def pull_and_restart_containers(client: docker.DockerClient, image_name: str) -> bool:
    """
    Pull the latest image and restart all containers using it.

    Args:
        client: Docker client instance
        image_name: Image to pull

    Returns:
        True if successful, False otherwise
    """
    try:
        containers = find_containers_by_image(client, image_name)
        if not containers:
            logger.warning(_m("No containers found to restart", {"image": image_name}))
            return False

        logger.info(_m("Pulling new image", {"image": image_name}))
        client.images.pull(image_name)
        logger.info(_m("Successfully pulled new image", {"image": image_name}))

        for container in containers:
            try:
                logger.info(_m("Restarting container", {
                    "name": container.name,
                    "id": container.short_id
                }))
                container.restart(timeout=10)
                logger.info(_m("Successfully restarted container", {
                    "name": container.name
                }))
            except Exception as e:
                logger.error(_m("Failed to restart container", {
                    "name": container.name,
                    "error": str(e)
                }))

        return True

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

        current_digest = get_current_image_digest(client, image_name)
        if not current_digest:
            logger.warning(_m("Could not retrieve current image digest", {"image": image_name}))
            return

        logger.info(_m("Current image digest", {"digest": current_digest}))

        remote_digest = fetch_verified_digest()
        if not remote_digest:
            logger.warning(_m("Could not fetch/verify remote digest, continuing..."))
            return

        logger.info(_m("Remote verified digest", {"digest": remote_digest}))

        if current_digest == remote_digest:
            logger.info(_m("Image is up to date"))
            return

        logger.info(_m("Image update detected", {
            "current": current_digest,
            "remote": remote_digest
        }))

        success = pull_and_restart_containers(client, image_name)
        if success:
            logger.info(_m("Successfully updated to new image"))
        else:
            logger.error(_m("Failed to update image"))

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
        "endpoint": settings.WATCHTOWER_ENDPOINT_URL,
        "validator_hotkey": settings.WATCHTOWER_VALIDATOR_HOTKEY
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
