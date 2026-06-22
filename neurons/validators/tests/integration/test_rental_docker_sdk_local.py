import os
import uuid

import pytest

from services.rental_docker_sdk import (
    ContainerExecSpec,
    ContainerRunSpec,
    RentalDockerSdkClient,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_RENTAL_DOCKER_SDK_INTEGRATION") != "1",
    reason="set RUN_RENTAL_DOCKER_SDK_INTEGRATION=1 to run local Docker smoke",
)


@pytest.mark.asyncio
async def test_local_docker_sdk_rental_smoke():
    import docker
    from docker.utils import kwargs_from_env

    image = "alpine:3.20"
    container_name = f"lium-rental-sdk-smoke-{uuid.uuid4().hex[:12]}"
    hostile_env = "value'; echo SHOULD_NOT_RUN; $(echo nope)"
    stdin_marker = "LOCAL_SDK_STDIN_MARKER"

    api_client = docker.APIClient(timeout=60, **kwargs_from_env())
    client = RentalDockerSdkClient(api_client)
    try:
        await client.pull(image=image)
        await client.run_container(
            ContainerRunSpec(
                image=image,
                name=container_name,
                command=("sh", "-c", "trap : TERM INT; sleep infinity & wait"),
                environment={"HOSTILE_ENV": hostile_env},
                restart_policy=None,
            )
        )

        inspect = api_client.inspect_container(container_name)
        assert f"HOSTILE_ENV={hostile_env}" in (inspect["Config"].get("Env") or [])

        result = await client.exec_in_container(
            ContainerExecSpec(
                container_name=container_name,
                argv=("sh", "-c", "cat > /tmp/sdk-smoke && cat /tmp/sdk-smoke"),
                stdin=f"{stdin_marker}\nsecond-line\n",
            )
        )
        assert result.exit_status == 0
        assert stdin_marker in result.stdout

        key_result = await client.exec_in_container(
            ContainerExecSpec(
                container_name=container_name,
                argv=("sh", "-c", "mkdir -p /root/.ssh && cat >> /root/.ssh/authorized_keys"),
                stdin='ssh-ed25519 AAAA user"; echo KEY_MARKER; $(echo nope)\n',
            )
        )
        assert key_result.exit_status == 0

        await client.stop(container_name=container_name)
        assert api_client.inspect_container(container_name)["State"]["Running"] is False

        await client.start(container_name=container_name)
        assert api_client.inspect_container(container_name)["State"]["Running"] is True

        await client.remove_container(container_name=container_name, force=True, remove_volumes=True)
        with pytest.raises(docker.errors.NotFound):
            api_client.inspect_container(container_name)
    finally:
        try:
            api_client.remove_container(container_name, force=True, v=True)
        except Exception:
            pass
        await client.aclose()
