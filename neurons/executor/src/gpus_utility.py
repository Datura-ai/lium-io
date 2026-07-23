import asyncio
import logging
import subprocess
import aiohttp
import click
import pynvml
import psutil

from services.pull_lock import cache_pull_lock

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _local_tag_matches_digest(template: str, expected_digest: str) -> bool:
    # True when the locally cached template already carries the expected RepoDigest
    inspect_result = subprocess.run(
        ['docker', 'image', 'inspect', '--format', '{{json .RepoDigests}}', template],
        capture_output=True,
        text=True
    )
    if inspect_result.returncode != 0:
        return False
    return expected_digest in inspect_result.stdout


def _pull_template_image(docker_image: str, docker_image_tag: str, expected_digest: str | None) -> bool:
    # pull the template; with a backend digest, pin the pull (repo@sha256:…) and re-point
    # the tag — tag pulls resolve through provider registry mirrors, which can serve a
    # stale manifest, while digest pulls are content-addressed and bypass them (DAH-2461)
    template = f"{docker_image}:{docker_image_tag}"
    if expected_digest:
        pinned_ref = f"{docker_image}@{expected_digest}"
        pull_result = subprocess.run(
            ['docker', 'pull', pinned_ref],
            capture_output=True,
            text=True
        )
        if pull_result.returncode != 0:
            logger.error(f"Failed to pull {pinned_ref}: {pull_result.stderr}")
            return False
        tag_result = subprocess.run(
            ['docker', 'tag', pinned_ref, template],
            capture_output=True,
            text=True
        )
        if tag_result.returncode != 0:
            logger.error(f"Failed to tag {pinned_ref} as {template}: {tag_result.stderr}")
            return False
        logger.info(f"Successfully pulled {template} at {expected_digest}")
        return True

    pull_result = subprocess.run(
        ['docker', 'pull', template],
        capture_output=True,
        text=True
    )
    if pull_result.returncode != 0:
        logger.error(f"Failed to pull {template}: {pull_result.stderr}")
        return False
    logger.info(f"Successfully pulled {template}")
    return True


async def manage_docker_images(
    compute_rest_app_url: str,
    min_disk_space_multiplier: float = 3.0,  # Minimum disk space required (3x image size)
    first_interval: int = 60 * 5,  # Retry interval when errors occur (network, Docker commands, etc.)
    success_interval: int = 2 * 60 * 60,  # Wait interval after successful Docker pull
):
    """
    Manages Docker images on the executor by:
    1. Syncing with backend for most used templates
    2. Cleaning up unused images
    3. Ensuring sufficient disk space

    Args:
        compute_rest_app_url (str): URL of the compute REST application
        min_disk_space_multiplier (float): Minimum disk space required as multiple of image size
    """
    # Get GPU name using NVML
    gpu_name = "unknown"
    driver_version = "unknown"
    try:
        pynvml.nvmlInit()
        # finally: a leaked NVML handle makes this process the "in use by another client"
        # blocker for any later per-GPU maintenance (DAH-2427 review finding).
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Get first GPU
            gpu_name = pynvml.nvmlDeviceGetName(handle)
            driver_version = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode("utf-8")
        finally:
            pynvml.nvmlShutdown()
    except Exception as e:
        logger.error(f"Failed to get GPU name: {e}")

    backend_url = f"{compute_rest_app_url}/executors/default-docker-image?gpu_model={gpu_name}&driver_version={driver_version}"

    async with aiohttp.ClientSession() as session:
        while True:
            try:
                # Get disk space info
                disk = psutil.disk_usage('/')
                available_space = disk.free

                async with session.get(backend_url) as response:
                    if response.status != 200:
                        logger.error(f"Failed to get template. Status: {response.status}")
                        await asyncio.sleep(first_interval)
                        continue

                    docker_images_data = await response.json()
                    logger.info(f"Received {len(docker_images_data)} docker image templates")
                    
                    # Check if no templates were returned
                    if not docker_images_data:
                        logger.warning(f"No templates found for GPU {gpu_name}")
                        await asyncio.sleep(first_interval)
                        continue
                    
                    had_error = False
                    for data in docker_images_data:
                        docker_image = data['docker_image']
                        docker_image_tag = data['docker_image_tag']
                        docker_image_size = data['docker_image_size']
                        # DAH-2461: optional bare "sha256:…" digest resolved by the backend
                        # from Docker Hub; absent on older backends.
                        expected_digest = data.get('docker_image_digest')
                        
                        # Check if no templates were found for the GPU
                        if not docker_image or not docker_image_tag:
                            logger.warning(f"No templates found for GPU {gpu_name}")
                            await asyncio.sleep(first_interval)
                            continue

                        template = f"{docker_image}:{docker_image_tag}"
                        logger.info(f"Processing template: {template}")

                        # Clean up unused images
                        docker_list_result = subprocess.run(
                            ['docker', 'images', '--format', '{{.Repository}}:{{.Tag}}'],
                            capture_output=True,
                            text=True
                        )
                        
                        if docker_list_result.returncode == 0:
                            local_images = docker_list_result.stdout.strip().split('\n')
                            
                            for image in local_images:
                                if not image:
                                    continue
                                    
                                repo, tag = image.split(':')
                                if repo == docker_image and tag != docker_image_tag:
                                    remove_result = subprocess.run(
                                        ['docker', 'rmi', image],
                                        capture_output=True,
                                        text=True
                                    )
                                    if remove_result.returncode == 0:
                                        logger.info(f"Removed unused image: {image}")
                                    else:
                                        logger.warning(f"Failed to remove image {image}: {remove_result.stderr}")
                        else:
                            logger.warning(f"Failed to list Docker images: {docker_list_result.stderr}")

                        # DAH-2461: with a backend digest, skip pulling when the local tag
                        # already matches — otherwise this loop's unconditional tag pull
                        # would re-poison the tag on hosts behind a stale registry mirror.
                        if expected_digest and _local_tag_matches_digest(template, expected_digest):
                            logger.info(
                                f"Template {template} already at backend digest ({expected_digest}); skipping pull"
                            )
                            continue

                        # Process template and pull image
                        required_space = int(docker_image_size * min_disk_space_multiplier)
                        
                        if available_space < required_space:
                            logger.warning(
                                f"Skipping pull of {template} - insufficient disk space. "
                                f"Required: {required_space}, Available: {available_space}"
                            )
                            await asyncio.sleep(first_interval)
                            continue

                        # Pull the image (guarded so we don't pull the same image
                        # concurrently with the executor's on-boot cache pre-pull).
                        with cache_pull_lock() as acquired:
                            if not acquired:
                                logger.info(
                                    f"Another puller holds the lock; skipping {template} this cycle"
                                )
                                continue
                            pulled = _pull_template_image(docker_image, docker_image_tag, expected_digest)

                        if not pulled:
                            had_error = True

                    # Sleep once per full sweep: retry sooner if anything failed.
                    await asyncio.sleep(first_interval if had_error else success_interval)

            except aiohttp.ClientError as e:
                logger.error(f"Network error: {e}")
                await asyncio.sleep(first_interval)
            except subprocess.CalledProcessError as e:
                logger.error(f"Docker command failed: {e.stderr}")
                await asyncio.sleep(first_interval)
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                await asyncio.sleep(first_interval)


@click.command()
@click.option("--compute_rest_app_url", prompt="Compute-app Url", help="Compute-app Url")
def main(compute_rest_app_url: str):
    asyncio.run(manage_docker_images(compute_rest_app_url))


if __name__ == "__main__":
    main()
