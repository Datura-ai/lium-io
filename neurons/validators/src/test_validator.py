import asyncio
import bittensor

from core.config import settings
from fastapi.testclient import TestClient
from concurrent.futures import ThreadPoolExecutor, as_completed
from services.docker_service import DockerService
from services.ioc import ioc

from validator import app

client = TestClient(app)


def send_post_request():
    response = client.post(
        "/miner_request",
        json={
            "miner_hotkey": "5EHgHZBfx4ZwU7GzGCS8VCMBLBEKo5eaCvXKiu6SASwWT6UY",
            "miner_address": "localhost",
            "miner_port": 8000
        },
    )
    assert response.status_code == 200


def test_socket_connections():
    num_requests = 10  # Number of simultaneous requests
    with ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = [executor.submit(send_post_request) for _ in range(num_requests)]

        for future in as_completed(futures):
            response = future.result()
            assert response.status_code == 200


async def check_docker_port_mappings():
    docker_service: DockerService = ioc["DockerService"]
    miner_hotkey = '5Df8qGLMd19BXByefGCZFN57fWv6jDm5hUbnQeUTu2iqNBhT'
    executor_id = 'c272060f-8eae-4265-8e26-1d83ac96b498'
    port_mappings = await docker_service.generate_portMappings(miner_hotkey, executor_id)
    print('port_mappings ==>', port_mappings)

if __name__ == "__main__":
    # test_socket_connections()
    asyncio.run(check_docker_port_mappings())

    config = settings.get_bittensor_config()
    subtensor = bittensor.subtensor(config=config)
    node = subtensor.substrate

    netuid = settings.BITTENSOR_NETUID
    tempo = subtensor.tempo(netuid)
    weights_rate_limit = node.query("SubtensorModule", "WeightsSetRateLimit", [netuid]).value
    server_rate_limit = node.query("SubtensorModule", "WeightsSetRateLimit", [netuid]).value
    serving_rate_limit = node.query("SubtensorModule", "ServingRateLimit", [netuid]).value
    print('rate limit ===>', tempo, weights_rate_limit, serving_rate_limit)
