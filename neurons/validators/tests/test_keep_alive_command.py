from unittest.mock import Mock

from payload_models.payloads import ContainerCreateRequest, CustomOptions
from services.docker_service import DockerService
from services.rental_docker_sdk import KEEP_ALIVE_COMMAND, GpuDockerConfig, keep_alive_command_for


def test_bare_shell_cmd_gets_keep_alive():
    assert keep_alive_command_for((), ("/bin/bash",)) == KEEP_ALIVE_COMMAND


def test_bare_shell_cmd_behind_exec_wrapper_entrypoint_gets_keep_alive():
    assert (
        keep_alive_command_for(("/opt/nvidia/nvidia_entrypoint.sh",), ("bash",))
        == KEEP_ALIVE_COMMAND
    )


def test_no_cmd_and_no_entrypoint_gets_keep_alive():
    assert keep_alive_command_for((), ()) == KEEP_ALIVE_COMMAND


def test_image_with_own_server_command_is_kept():
    assert keep_alive_command_for((), ("python", "-m", "server")) == ()


def test_shell_with_arguments_is_kept():
    assert (
        keep_alive_command_for((), ("/bin/bash", "-c", "service ssh start && tail -f /dev/null"))
        == ()
    )


def test_entrypoint_without_cmd_is_kept():
    assert keep_alive_command_for(("/start.sh",), ()) == ()


def _run_spec(startup_commands: str | None, image_command_fallback: tuple[str, ...]):
    docker_service = DockerService(
        ssh_service=Mock(),
        redis_service=Mock(),
        attestation_service=Mock(),
        rental_docker_client_factory=Mock(),
    )
    return docker_service._build_rental_container_run_spec(
        payload=ContainerCreateRequest(
            miner_hotkey="hk",
            executor_id="ex",
            pod_id="pod",
            docker_image="img:tag",
            gpu_uuids=["g0"],
        ),
        container_name="pod",
        custom_options=CustomOptions(startup_commands=startup_commands),
        port_maps=[],
        local_volume="volume_pod",
        local_volume_path="/root",
        encrypted_local_volume=False,
        external_volume_name=None,
        gpu_devices=GpuDockerConfig(),
        effective_storage_limit_gb=None,
        cpu_count=None,
        image_command_fallback=image_command_fallback,
    )


def test_run_spec_uses_fallback_when_renter_gave_no_command():
    assert _run_spec(None, KEEP_ALIVE_COMMAND).command == KEEP_ALIVE_COMMAND


def test_run_spec_prefers_renter_command_over_fallback():
    assert _run_spec("python app.py", KEEP_ALIVE_COMMAND).command == ("python", "app.py")
