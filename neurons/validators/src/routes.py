import time
import aiohttp
from fastapi import APIRouter
from cache import AsyncTTL
from core.config import settings


router = APIRouter()


@AsyncTTL(time_to_live=60 * 10, maxsize=1)
async def get_docker_hub_digest(image: str, tag: str = "latest") -> str:
    # Get auth token
    auth_url = f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:daturaai/{image}:pull"
    async with aiohttp.ClientSession() as session:
        async with session.get(auth_url) as resp:
            resp.raise_for_status()
            data = await resp.json()
            token = data["token"]

        # Get manifest with digest
        manifest_url = f"https://registry-1.docker.io/v2/daturaai/{image}/manifests/{tag}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.docker.distribution.manifest.v2+json"
        }
        async with session.head(manifest_url, headers=headers) as resp:
            resp.raise_for_status()
            return resp.headers.get("Docker-Content-Digest")


@router.get("/digest")
async def get_watchtower_digest(image: str = settings.DEFAULT_DOCKER_IMAGE, tag: str = settings.DEFAULT_DOCKER_TAG):
    import datetime
    digest = await get_docker_hub_digest(image, tag)
    timestamp = int(datetime.datetime.now(datetime.UTC).timestamp())
    validator_keypair = settings.get_bittensor_wallet().get_hotkey()
    signature = validator_keypair.sign(f"{digest}:{timestamp}".encode()).hex()
    return {
        "digest": digest,
        "signature": f"0x{signature}",
        "timestamp": timestamp,
    }
