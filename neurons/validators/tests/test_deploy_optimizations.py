"""DAH-1524 — deploy-time optimizations for cached-template pods.

Covers (from the plan's Test Plan):
- Pull-skip (#1): skip when image present, pull when absent, fail-open on probe
  error, and `check=False` on the inspect probe.
- SSH bootstrap (#2): run the idempotent bootstrap for the default rental path
  (`ships_sshd` None/False), and SKIP it for templates that set `ships_sshd`
  (DAH-2265) — the image ships+starts sshd itself, so the validator only
  re-ensures ~/.ssh for key injection. Includes the DEFAULT-PATH REGRESSION
  GUARD. When `ships_sshd` is set with `enable_jupyter`, the validator forwards
  JUPYTER_PASSWORD as a docker-run env var instead of running run_jupyter — that
  is the only Jupyter variable the image's start.sh reads — and falls back to
  run_jupyter if the mapped docker port is not the 8888 the image hardcodes.
- Profile summary log (#3): emitted once on success, survives the mixed-shape
  profilers list, and NOT emitted on the failure path.
- Backward compat: payloads omitting `ships_sshd` deserialize with None.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
import services.docker_service as ds_module
from pydantic import ValidationError
from datura.requests.miner_requests import ExecutorSSHInfo
from payload_models.payloads import (
    ContainerCreated,
    ContainerCreateRequest,
    CustomOptions,
    FailedContainerRequest,
    PayloadPortMapping,
    ProfilerStep,
    ProfilerStepName,
    now_ms,
)
from services.docker_service import DockerService
from services.rental_docker_sdk import ContainerExecResult, build_gpu_docker_config

# ------------------------------------------------------------------
# Fixtures / helpers
# ------------------------------------------------------------------


@pytest.fixture
def svc():
    return DockerService(
        ssh_service=Mock(),
        redis_service=Mock(),
        attestation_service=Mock(),
    )


class _ConnCtx:
    def __init__(self, ssh):
        self.ssh = ssh

    async def __aenter__(self):
        return self.ssh

    async def __aexit__(self, *exc):
        return None


def _ssh_result(exit_status: int = 0, stdout: str = "", stderr: str = ""):
    r = Mock()
    r.exit_status = exit_status
    r.stdout = stdout
    r.stderr = stderr
    return r


def _ssh_client(*, inspect_exit: int = 0, inspect_raises: bool = False):
    """AsyncMock ssh connection plus SDK image-probe controls.

    The image presence probe moved from host-side `docker image inspect` to
    Docker SDK data calls; keep the old test inputs so the test intent stays
    readable while `_patch_happy` maps them onto the fake SDK client.
    """
    client = AsyncMock()
    client.image_exists_result = inspect_exit == 0
    client.image_exists_error = RuntimeError("probe boom") if inspect_raises else None

    def _side(cmd, *args, **kwargs):
        return _ssh_result(exit_status=0)

    client.run = AsyncMock(side_effect=_side)
    return client


class _FakeRentalDockerClient:
    def __init__(self, *, image_exists_result: bool, image_exists_error=None):
        self.image_exists_result = image_exists_result
        self.image_exists_error = image_exists_error
        self.image_exists_calls = []
        self.pulled_images = []
        self.run_specs = []
        self.exec_specs = []
        self.login_calls = []

    async def login(self, *, username: str, password: str) -> None:
        self.login_calls.append({"username": username, "password": password})

    async def image_exists(self, *, image: str) -> bool:
        self.image_exists_calls.append(image)
        if self.image_exists_error is not None:
            raise self.image_exists_error
        return self.image_exists_result

    async def pull(self, *, image: str) -> None:
        self.pulled_images.append(image)

    async def run_container(self, spec) -> None:
        self.run_specs.append(spec)

    async def exec_in_container(self, spec) -> ContainerExecResult:
        self.exec_specs.append(spec)
        return ContainerExecResult(exit_status=0)


class _FakeRentalDockerFactory:
    def __init__(self, client):
        self.client = client

    def connect(self, *, executor_info: ExecutorSSHInfo, private_key: str):
        return self

    async def __aenter__(self):
        return self.client

    async def __aexit__(self, *exc):
        return None


def _payload(**over) -> ContainerCreateRequest:
    base = dict(
        miner_hotkey="miner",
        executor_id=str(uuid4()),
        pod_id=str(uuid4()),
        docker_image="daturaai/pytorch:1.0.0",
        user_public_keys=["ssh-ed25519 test-key"],
        gpu_uuids=["GPU-test"],
        cpu_count=1,
        memory_gb=1,
        volume_limit_gb=2,
        storage_limit_gb=1,
        available_ports=[PayloadPortMapping(internal_port=20001, external_port=20001)],
        pod_mapping=[],
        active_container_names=[],
        active_volume_names=[],
    )
    base.update(over)
    return ContainerCreateRequest(**base)


def _executor_info(payload: ContainerCreateRequest) -> ExecutorSSHInfo:
    return ExecutorSSHInfo(
        uuid=payload.executor_id,
        address="127.0.0.1",
        port=8080,
        ssh_username="root",
        ssh_port=2200,
        python_path="/usr/bin/python",
        root_dir="/root/app",
        ssh_host_key="ssh-ed25519 AAAATESTKEY",
    )


def _patch_happy(svc, monkeypatch, ssh_client):
    """Stub the whole deploy flow so only the new branch logic is exercised and
    create_container reaches a real ContainerCreated."""
    docker_client = _FakeRentalDockerClient(
        image_exists_result=ssh_client.image_exists_result,
        image_exists_error=ssh_client.image_exists_error,
    )
    svc.rental_docker_client_factory = _FakeRentalDockerFactory(docker_client)
    monkeypatch.setattr(
        "services.docker_service.asyncssh.connect",
        Mock(return_value=_ConnCtx(ssh_client)),
    )
    monkeypatch.setattr("services.docker_service.asyncssh.import_private_key", Mock())
    monkeypatch.setattr(
        "services.docker_service.build_gpu_docker_config_for_executor",
        AsyncMock(return_value=build_gpu_docker_config(["GPU-test"])),
    )
    svc.ssh_service.decrypt_payload = Mock(return_value="private-key")
    svc.redis_service.add_pending_pod = AsyncMock()
    svc.redis_service.remove_pending_pod = AsyncMock()
    svc.redis_service.add_rented_pod = AsyncMock()
    lock = AsyncMock()
    lock.__aenter__ = AsyncMock(return_value=lock)
    lock.__aexit__ = AsyncMock(return_value=None)
    svc.redis_service.acquire_executor_lock = Mock(return_value=lock)
    monkeypatch.setattr(svc, "_prepare_known_hosts_policy", AsyncMock(return_value=None))
    monkeypatch.setattr(
        svc, "generate_portMappings",
        AsyncMock(return_value=([(22, 20001, 20001)], None)),
    )
    monkeypatch.setattr(svc, "clean_existing_containers", AsyncMock())
    monkeypatch.setattr(svc, "clean_stale_vloopback_volumes", AsyncMock())
    monkeypatch.setattr(
        svc, "resolve_volume_sizing",
        AsyncMock(return_value=Mock(volume_limit_gb=10, storage_limit_gb=20)),
    )
    monkeypatch.setattr(svc, "create_local_volume", AsyncMock())
    monkeypatch.setattr(
        svc, "wait_for_port_check_containers",
        AsyncMock(return_value=(True, "ok")),
    )
    monkeypatch.setattr(svc, "_run_rental_docker_create_with_port_retry", AsyncMock())
    monkeypatch.setattr(svc, "check_container_running", AsyncMock(return_value=True))
    monkeypatch.setattr(
        svc,
        "install_open_ssh_server_and_start_ssh_service_with_rental_docker",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(svc, "run_jupyter", AsyncMock())
    monkeypatch.setattr(svc, "execute_and_stream_logs", AsyncMock(return_value=(True, "")))
    monkeypatch.setattr(svc, "stream_log", AsyncMock())
    monkeypatch.setattr(svc, "finish_stream_logs", AsyncMock())
    monkeypatch.setattr(svc, "handle_stream_logs", AsyncMock())


async def _run(svc, payload):
    return await svc.create_container(
        payload=payload,
        executor_info=_executor_info(payload),
        keypair=Mock(ss58_address="validator-hotkey"),
        private_key="encrypted",
    )


def _docker_client(svc):
    return svc.rental_docker_client_factory.client


def _pulled_images(svc):
    return _docker_client(svc).pulled_images


def _exec_argv_texts(svc):
    return [" ".join(spec.argv) for spec in _docker_client(svc).exec_specs]


def _ssh_run_cmds(ssh_client):
    return [c.args[0] for c in ssh_client.run.await_args_list if c.args]


def _created_run_spec(svc):
    """The ContainerRunSpec handed to _run_rental_docker_create_with_port_retry
    (carries the create-time `docker run` environment)."""
    return svc._run_rental_docker_create_with_port_retry.await_args.kwargs["run_spec"]


# ------------------------------------------------------------------
# #1 — pull-skip
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pull_skipped_when_image_present(svc, monkeypatch):
    ssh_client = _ssh_client(inspect_exit=0)  # image present
    _patch_happy(svc, monkeypatch, ssh_client)

    result = await _run(svc, _payload())

    assert isinstance(result, ContainerCreated)
    assert _pulled_images(svc) == [], "docker pull must NOT be issued when image is present"
    pull_step = next(p for p in result.profilers if p.name == ProfilerStepName.DOCKER_PULL)
    assert pull_step.skipped is True


@pytest.mark.asyncio
async def test_pull_runs_when_image_absent(svc, monkeypatch):
    ssh_client = _ssh_client(inspect_exit=1)  # image absent
    _patch_happy(svc, monkeypatch, ssh_client)

    result = await _run(svc, _payload(docker_image="daturaai/pytorch:1.2.3"))

    assert isinstance(result, ContainerCreated)
    assert _pulled_images(svc) == ["daturaai/pytorch:1.2.3"]
    pull_step = next(p for p in result.profilers if p.name == ProfilerStepName.DOCKER_PULL)
    assert not pull_step.skipped


@pytest.mark.asyncio
async def test_probe_error_falls_through_to_pull(svc, monkeypatch):
    """Fail-open: a raised probe error must pull (not a CCF)."""
    ssh_client = _ssh_client(inspect_raises=True)
    _patch_happy(svc, monkeypatch, ssh_client)

    result = await _run(svc, _payload(docker_image="daturaai/pytorch:9.9.9"))

    assert isinstance(result, ContainerCreated)
    assert _pulled_images(svc) == ["daturaai/pytorch:9.9.9"]


@pytest.mark.asyncio
async def test_inspect_probe_uses_sdk_data_not_host_shell(svc, monkeypatch):
    ssh_client = _ssh_client(inspect_exit=0)
    _patch_happy(svc, monkeypatch, ssh_client)

    payload = _payload()
    await _run(svc, payload)

    assert _docker_client(svc).image_exists_calls == [payload.docker_image]
    assert not any(
        "image inspect" in command for command in _ssh_run_cmds(ssh_client)
    )


# ------------------------------------------------------------------
# #2 — SSH bootstrap
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ships_sshd_true_skips_bootstrap(svc, monkeypatch):
    """DAH-2265: templates that ship sshd must NOT run the validator bootstrap.

    The image's start.sh starts sshd itself; the validator only re-ensures
    ~/.ssh so key injection can write authorized_keys.
    """
    ssh_client = _ssh_client(inspect_exit=0)
    _patch_happy(svc, monkeypatch, ssh_client)

    result = await _run(svc, _payload(ships_sshd=True))

    assert isinstance(result, ContainerCreated)
    # The image ships & starts sshd itself, so the validator bootstrap is skipped.
    # (~/.ssh is created by the authorized_keys injection that runs just before.)
    svc.install_open_ssh_server_and_start_ssh_service_with_rental_docker.assert_not_awaited()


@pytest.mark.asyncio
async def test_default_ships_sshd_none_runs_full_bootstrap(svc, monkeypatch):
    """DEFAULT-PATH REGRESSION GUARD.

    Fails loudly if the `ships_sshd` field default is ever flipped away from
    None — which would silently surrender sshd hardening (`PasswordAuthentication
    no`) and the 30s self-heal watchdog for ALL pods. NB: this unit seam mocks
    the bootstrap, so it cannot assert bash-level hardening/watchdog presence;
    that is the staging verification step.
    """
    ssh_client = _ssh_client(inspect_exit=0)
    _patch_happy(svc, monkeypatch, ssh_client)

    payload = _payload()
    assert payload.ships_sshd is None  # default must stay None

    result = await _run(svc, payload)

    assert isinstance(result, ContainerCreated)
    svc.install_open_ssh_server_and_start_ssh_service_with_rental_docker.assert_awaited_once()


@pytest.mark.asyncio
async def test_ships_sshd_false_runs_install(svc, monkeypatch):
    ssh_client = _ssh_client(inspect_exit=0)
    _patch_happy(svc, monkeypatch, ssh_client)

    result = await _run(svc, _payload(ships_sshd=False))

    assert isinstance(result, ContainerCreated)
    svc.install_open_ssh_server_and_start_ssh_service_with_rental_docker.assert_awaited_once()


@pytest.mark.asyncio
async def test_ships_sshd_false_preserves_lenient_bootstrap_failure(svc, monkeypatch):
    ssh_client = _ssh_client(inspect_exit=0)
    _patch_happy(svc, monkeypatch, ssh_client)
    svc.install_open_ssh_server_and_start_ssh_service_with_rental_docker.return_value = False

    result = await _run(svc, _payload(ships_sshd=False))

    assert isinstance(result, ContainerCreated)
    svc.install_open_ssh_server_and_start_ssh_service_with_rental_docker.assert_awaited_once()


@pytest.mark.asyncio
async def test_ships_sshd_true_forwards_jupyter_password_instead_of_run_jupyter(svc, monkeypatch):
    """DAH-2265: ships_sshd + enable_jupyter → forward JUPYTER_PASSWORD as a
    docker-run env var and let the image launch Jupyter, rather than running
    the validator's run_jupyter path."""
    ssh_client = _ssh_client(inspect_exit=0)
    _patch_happy(svc, monkeypatch, ssh_client)
    # Provide a jupyter port map so the jupyter branch fires.
    monkeypatch.setattr(
        svc, "generate_portMappings",
        AsyncMock(return_value=([(22, 20001, 20001)], (8888, 30888))),
    )

    result = await _run(svc, _payload(ships_sshd=True, enable_jupyter=True, **_CREDS))

    assert isinstance(result, ContainerCreated)
    # Neither the validator sshd bootstrap nor run_jupyter should run.
    svc.install_open_ssh_server_and_start_ssh_service_with_rental_docker.assert_not_awaited()
    svc.run_jupyter.assert_not_awaited()

    # WHY (oracle B10 fold, DAH-2382): the image-managed bundle also skips the docker
    # login — ships_sshd is the default-image signal, so even with renter credentials
    # attached the login must not run and its profiler step must say skipped.
    assert _docker_client(svc).login_calls == []
    assert _login_step(result).skipped is True

    # JUPYTER_PASSWORD is forwarded as create-time container env: it is the ONLY
    # Jupyter variable the image's start.sh reads (`if [[ $JUPYTER_PASSWORD ]]` →
    # `jupyter lab --ServerApp.token=$JUPYTER_PASSWORD`). Forwarding ENABLE_JUPYTER /
    # JUPYTER_TOKEN instead left the gate closed and Jupyter never started.
    env = _created_run_spec(svc).environment
    token = env.get("JUPYTER_PASSWORD")
    assert token

    # The returned URL uses the same token the image will serve Jupyter as.
    assert result.jupyter_url == f"http://127.0.0.1:30888/lab?token={token}"


@pytest.mark.asyncio
async def test_image_managed_jupyter_falls_back_when_docker_port_is_not_8888(svc, monkeypatch):
    """The image's start.sh hardcodes `jupyter lab --port=8888`. If the mapped docker
    port ever moved off 8888 the image's Jupyter would be unreachable, so the validator
    must fall back to its own run_jupyter rather than silently serving nothing."""
    ssh_client = _ssh_client(inspect_exit=0)
    _patch_happy(svc, monkeypatch, ssh_client)
    monkeypatch.setattr(
        svc, "generate_portMappings",
        AsyncMock(return_value=([(22, 20001, 20001)], (9999, 30888))),
    )

    result = await _run(svc, _payload(ships_sshd=True, enable_jupyter=True))

    assert isinstance(result, ContainerCreated)
    # sshd is still the image's job, but Jupyter falls back to the validator.
    svc.install_open_ssh_server_and_start_ssh_service_with_rental_docker.assert_not_awaited()
    svc.run_jupyter.assert_awaited_once()
    assert "JUPYTER_PASSWORD" not in _created_run_spec(svc).environment


@pytest.mark.asyncio
async def test_ships_sshd_none_with_jupyter_uses_run_jupyter(svc, monkeypatch):
    """Regression guard: the default path (ships_sshd None) still runs the
    validator's run_jupyter and does NOT forward JUPYTER_PASSWORD."""
    ssh_client = _ssh_client(inspect_exit=0)
    _patch_happy(svc, monkeypatch, ssh_client)
    monkeypatch.setattr(
        svc, "generate_portMappings",
        AsyncMock(return_value=([(22, 20001, 20001)], (8888, 30888))),
    )

    result = await _run(svc, _payload(enable_jupyter=True, **_CREDS))

    assert isinstance(result, ContainerCreated)
    svc.install_open_ssh_server_and_start_ssh_service_with_rental_docker.assert_awaited_once()
    svc.run_jupyter.assert_awaited_once()
    assert "JUPYTER_PASSWORD" not in _created_run_spec(svc).environment

    # WHY (oracle B11 fold, DAH-2382): the validator-managed branch keeps the docker
    # login — ships_sshd None with credentials is no default-image signal, so the
    # login must run and the DOCKER_LOGIN profiler step must not claim skipped.
    assert _docker_client(svc).login_calls == [{"username": "renter", "password": "renter-secret"}]
    assert _login_step(result).skipped is False


@pytest.mark.asyncio
async def test_ships_sshd_with_startup_commands_runs_bootstrap_and_run_jupyter(svc, monkeypatch):
    """Regression (reviewer arhangel66, CHANGES_REQUESTED): a non-empty
    startup_commands is passed as the container command and REPLACES the image
    CMD (/start.sh) — the script that starts sshd and launches Jupyter. With
    ships_sshd set BUT startup_commands present, start.sh never runs, so the
    validator must NOT take the skip path: it must run its own sshd bootstrap
    AND run_jupyter, else the pod comes up with dead SSH and a Jupyter URL that
    resets. Port is 8888 so the ONLY thing forcing the fallback is the override,
    not the port guard."""
    ssh_client = _ssh_client(inspect_exit=0)
    _patch_happy(svc, monkeypatch, ssh_client)
    monkeypatch.setattr(
        svc, "generate_portMappings",
        AsyncMock(return_value=([(22, 20001, 20001)], (8888, 30888))),
    )

    result = await _run(
        svc,
        _payload(
            ships_sshd=True,
            enable_jupyter=True,
            custom_options=CustomOptions(startup_commands="sleep infinity"),
        ),
    )

    assert isinstance(result, ContainerCreated)
    # Both validator-managed paths must run — the image's start.sh was replaced.
    svc.install_open_ssh_server_and_start_ssh_service_with_rental_docker.assert_awaited_once()
    svc.run_jupyter.assert_awaited_once()
    # No image-managed Jupyter env is forwarded on the validator-managed fallback.
    assert "JUPYTER_PASSWORD" not in _created_run_spec(svc).environment


@pytest.mark.asyncio
async def test_ships_sshd_with_entrypoint_override_runs_bootstrap(svc, monkeypatch):
    """Same failure mode via the other override: a renter entrypoint replaces
    /pytorch-entrypoint.sh (the ENTRYPOINT that execs the CMD /start.sh), so
    start.sh never runs and the image starts no sshd. The validator must fall
    back to its own bootstrap even though ships_sshd is set."""
    ssh_client = _ssh_client(inspect_exit=0)
    _patch_happy(svc, monkeypatch, ssh_client)

    result = await _run(
        svc,
        _payload(
            ships_sshd=True,
            custom_options=CustomOptions(entrypoint="bash"),
        ),
    )

    assert isinstance(result, ContainerCreated)
    svc.install_open_ssh_server_and_start_ssh_service_with_rental_docker.assert_awaited_once()


@pytest.mark.asyncio
async def test_key_injection_runs_when_template_ships_sshd(svc, monkeypatch):
    """DAH-2265: even when the validator skips the sshd bootstrap (ships_sshd),
    the customer's public keys are still injected so the image-started sshd finds
    authorized_keys in place."""
    ssh_client = _ssh_client(inspect_exit=0)
    _patch_happy(svc, monkeypatch, ssh_client)

    keys = ["ssh-ed25519 key-one", "ssh-ed25519 key-two"]
    result = await _run(svc, _payload(ships_sshd=True, user_public_keys=keys))

    assert isinstance(result, ContainerCreated)
    # Bootstrap is skipped on the ships_sshd path...
    svc.install_open_ssh_server_and_start_ssh_service_with_rental_docker.assert_not_awaited()
    # ...but key injection (which also `mkdir -p /root/.ssh`) still ran exactly once.
    authorized = [
        spec
        for spec in _docker_client(svc).exec_specs
        if "authorized_keys" in " ".join(spec.argv)
    ]
    assert len(authorized) == 1
    assert "key-one" in authorized[0].stdin
    assert "key-two" in authorized[0].stdin


# ------------------------------------------------------------------
# #3 — profile summary log
# ------------------------------------------------------------------


def _summary_calls(mock_logger):
    return [
        c
        for c in mock_logger.info.call_args_list
        if c.args and getattr(c.args[0], "message", None) == "Deployment profile summary"
    ]


@pytest.mark.asyncio
async def test_summary_emitted_on_success(svc, monkeypatch):
    ssh_client = _ssh_client(inspect_exit=0)
    _patch_happy(svc, monkeypatch, ssh_client)
    monkeypatch.setattr(ds_module.settings, "ENABLE_INSPECTOR", False)
    mock_logger = Mock()
    monkeypatch.setattr(ds_module, "logger", mock_logger)

    result = await _run(svc, _payload())

    assert isinstance(result, ContainerCreated)
    summaries = _summary_calls(mock_logger)
    assert len(summaries) == 1
    extra = summaries[0].args[0].extra
    assert isinstance(extra["total_duration_ms"], int)
    names = {s["name"] for s in extra["profile_steps"]}
    assert "Docker pull step finished" in names
    assert "Inspector collector start step finished" in names
    assert "Finished in subnet." in names
    inspector = next(
        s for s in extra["profile_steps"]
        if s["name"] == "Inspector collector start step finished"
    )
    assert inspector["skipped"] is True


@pytest.mark.asyncio
async def test_summary_marks_skipped_steps(svc, monkeypatch):
    ssh_client = _ssh_client(inspect_exit=0)  # image present -> pull skipped
    _patch_happy(svc, monkeypatch, ssh_client)
    mock_logger = Mock()
    monkeypatch.setattr(ds_module, "logger", mock_logger)

    await _run(svc, _payload(ships_sshd=True))

    extra = _summary_calls(mock_logger)[0].args[0].extra
    pull = next(s for s in extra["profile_steps"] if s["name"] == "Docker pull step finished")
    assert pull["skipped"] is True
    assert pull["duration_ms"] is not None and pull["duration_ms"] >= 0


@pytest.mark.asyncio
async def test_summary_survives_timestamp_only_first_entry(svc, monkeypatch):
    """With payload.timestamp set, the first profiler step has `duration is
    None` (timestamp-only anchor). The summary builder must not raise and must
    exclude it from the total."""
    ssh_client = _ssh_client(inspect_exit=0)
    _patch_happy(svc, monkeypatch, ssh_client)
    mock_logger = Mock()
    monkeypatch.setattr(ds_module, "logger", mock_logger)

    result = await _run(svc, _payload(timestamp=1_700_000_000_000))

    assert isinstance(result, ContainerCreated)
    extra = _summary_calls(mock_logger)[0].args[0].extra
    requested = next(s for s in extra["profile_steps"] if s["name"] == "Requested from backend")
    assert requested["duration_ms"] is None
    assert isinstance(extra["total_duration_ms"], int)


@pytest.mark.asyncio
async def test_summary_not_emitted_on_failure(svc, monkeypatch):
    ssh_client = _ssh_client(inspect_exit=1)
    _patch_happy(svc, monkeypatch, ssh_client)
    monkeypatch.setattr(
        svc, "_run_rental_docker_create_with_port_retry",
        AsyncMock(side_effect=RuntimeError("docker run failed")),
    )
    mock_logger = Mock()
    monkeypatch.setattr(ds_module, "logger", mock_logger)

    result = await _run(svc, _payload())

    assert isinstance(result, FailedContainerRequest)
    assert _summary_calls(mock_logger) == []


# ------------------------------------------------------------------
# Backward compatibility
# ------------------------------------------------------------------


def test_payload_without_ships_sshd_deserializes():
    legacy = dict(
        message_type="ContainerCreateRequest",
        miner_hotkey="miner",
        executor_id="00000000-0000-0000-0000-000000000001",
        pod_id="00000000-0000-0000-0000-0000000000aa",
        docker_image="daturaai/pytorch:1.0.0",
        gpu_uuids=["GPU-x"],
    )
    req = ContainerCreateRequest(**legacy)
    assert req.ships_sshd is None


# ------------------------------------------------------------------
# DAH-1524 — cleanup step no longer blocks the critical path on a 10s sleep
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deploy_cleanup_runs_without_blocking_sleep(svc, monkeypatch):
    """The container-cleanup GC still runs on the deploy path, but the former
    `sleep=10` is gone: create_container must NOT request a blocking sleep
    (defaults to 0), so the cleanup no longer adds ~10s to the critical path.
    The GC itself (clean_existing_containers) is still invoked."""
    ssh_client = _ssh_client(inspect_exit=0)
    _patch_happy(svc, monkeypatch, ssh_client)

    result = await _run(svc, _payload())

    assert isinstance(result, ContainerCreated)
    svc.clean_existing_containers.assert_awaited_once()
    sleep_arg = svc.clean_existing_containers.await_args.kwargs.get("sleep", 0)
    assert sleep_arg == 0, f"deploy path must not request a blocking sleep, got sleep={sleep_arg}"


# ------------------------------------------------------------------
# DAH-1524 — ProfilerStep schema: typed `name` enum + wire-shape preservation
# ------------------------------------------------------------------


def test_profiler_step_rejects_unknown_name():
    """`name` is constrained to the ProfilerStepName enum."""
    with pytest.raises(ValidationError):
        ProfilerStep(name="not a real step", duration=5)


def test_profiler_step_since_builds_duration_step():
    """`since()` is the factory used at every deploy step: it fills `duration`
    from the elapsed time and carries the `skipped` flag, leaving `timestamp`
    unset (so the wire shape stays name+duration[, skipped])."""
    start = now_ms()
    step = ProfilerStep.since(ProfilerStepName.DOCKER_RUN, start)
    assert step.name is ProfilerStepName.DOCKER_RUN
    assert step.duration is not None and step.duration >= 0
    assert step.timestamp is None
    assert step.skipped is False
    assert set(step.model_dump()) == {"name", "duration"}

    skipped = ProfilerStep.since(ProfilerStepName.DOCKER_PULL, start, skipped=True)
    assert skipped.skipped is True
    assert set(skipped.model_dump()) == {"name", "duration", "skipped"}


def test_profiler_step_serializes_to_historical_wire_shape():
    """The typed model must serialize to exactly the keys we emitted before it
    became a model — no `null`/`false` keys may leak — so lium-io-backend keeps
    parsing byte-identical JSON."""
    duration_step = ProfilerStep(name=ProfilerStepName.STARTED_IN_SUBNET, duration=508)
    assert duration_step.model_dump() == {"name": "Started in subnet", "duration": 508}
    # A non-skipped, non-anchor step carries neither `skipped` nor `timestamp`.
    assert "skipped" not in duration_step.model_dump()
    assert "timestamp" not in duration_step.model_dump()

    anchor = ProfilerStep(
        name=ProfilerStepName.REQUESTED_FROM_BACKEND, timestamp=1_700_000_000_000
    )
    assert anchor.model_dump() == {
        "name": "Requested from backend",
        "timestamp": 1_700_000_000_000,
    }

    skipped = ProfilerStep(name=ProfilerStepName.DOCKER_PULL, duration=127, skipped=True)
    assert skipped.model_dump() == {
        "name": "Docker pull step finished",
        "duration": 127,
        "skipped": True,
    }


def test_container_created_profilers_round_trip():
    """ContainerCreated with typed profilers serializes and re-parses cleanly —
    the validator round-trips responses through JSON."""
    response = ContainerCreated(
        miner_hotkey="miner",
        executor_id="00000000-0000-0000-0000-000000000001",
        pod_id="00000000-0000-0000-0000-0000000000aa",
        container_name="c",
        volume_name="v",
        port_maps=[],
        profilers=[
            ProfilerStep(
                name=ProfilerStepName.REQUESTED_FROM_BACKEND, timestamp=1_700_000_000_000
            ),
            ProfilerStep(name=ProfilerStepName.STARTED_IN_SUBNET, duration=508),
            ProfilerStep(name=ProfilerStepName.DOCKER_PULL, duration=127, skipped=True),
        ],
    )

    parsed = ContainerCreated.model_validate_json(response.model_dump_json())

    assert [p.name for p in parsed.profilers] == [
        ProfilerStepName.REQUESTED_FROM_BACKEND,
        ProfilerStepName.STARTED_IN_SUBNET,
        ProfilerStepName.DOCKER_PULL,
    ]
    assert parsed.profilers[0].timestamp == 1_700_000_000_000
    assert parsed.profilers[0].duration is None
    assert parsed.profilers[2].skipped is True


# ------------------------------------------------------------------
# #4 — docker-login skip for default cache-template images
# ------------------------------------------------------------------

_DEFAULT_IMAGE = "daturaai/pytorch:2.12.0-py3.12-cuda12.8-devel-ubuntu24.04-dind"
_CREDS = {"docker_username": "renter", "docker_password": "renter-secret"}


def _login_step(result):
    return next(p for p in result.profilers if p.name == ProfilerStepName.DOCKER_LOGIN)


@pytest.mark.asyncio
async def test_docker_login_skipped_for_default_image_even_with_credentials(svc, monkeypatch):
    """The default images are public and pre-cached, so the login is pure latency —
    skip it even though the backend attached the renter's saved credentials.

    `ships_sshd` IS the "renter selected a default image" signal: the backend sets it
    from the same check that resolves the recommended image (`ships_sshd=is_cached`).
    The validator trusts it rather than re-deriving default-ness from the image ref.
    """
    ssh_client = _ssh_client(inspect_exit=0)
    _patch_happy(svc, monkeypatch, ssh_client)

    result = await _run(svc, _payload(docker_image=_DEFAULT_IMAGE, ships_sshd=True, **_CREDS))

    assert isinstance(result, ContainerCreated)
    assert _docker_client(svc).login_calls == [], "docker login must NOT run for a default image"
    assert _login_step(result).skipped is True


@pytest.mark.asyncio
async def test_docker_login_runs_for_non_default_image_with_credentials(svc, monkeypatch):
    """DEFAULT-PATH REGRESSION GUARD: a user image with credentials still logs in —
    a private registry pull depends on it. The backend sends ships_sshd=False here."""
    ssh_client = _ssh_client(inspect_exit=1)
    _patch_happy(svc, monkeypatch, ssh_client)

    result = await _run(
        svc, _payload(docker_image="private/repo:1.0.0", ships_sshd=False, **_CREDS)
    )

    assert isinstance(result, ContainerCreated)
    assert _docker_client(svc).login_calls == [{"username": "renter", "password": "renter-secret"}]
    assert _login_step(result).skipped is False


@pytest.mark.asyncio
async def test_docker_login_runs_when_ships_sshd_is_none(svc, monkeypatch):
    """LEGACY-PATH REGRESSION GUARD: retry/filler rentals omit ships_sshd (None). That
    is not a default-image signal, so the login must still run when credentials exist."""
    ssh_client = _ssh_client(inspect_exit=1)
    _patch_happy(svc, monkeypatch, ssh_client)

    result = await _run(svc, _payload(docker_image="private/repo:1.0.0", **_CREDS))

    assert isinstance(result, ContainerCreated)
    assert _docker_client(svc).login_calls == [{"username": "renter", "password": "renter-secret"}]
    assert _login_step(result).skipped is False


@pytest.mark.asyncio
async def test_docker_login_step_marked_skipped_when_no_credentials(svc, monkeypatch):
    """No credentials was always a no-op login; now the profile says so instead of
    reporting a 0ms step that looks like real work."""
    ssh_client = _ssh_client(inspect_exit=0)
    _patch_happy(svc, monkeypatch, ssh_client)

    result = await _run(svc, _payload(docker_image="private/repo:1.0.0"))

    assert isinstance(result, ContainerCreated)
    assert _docker_client(svc).login_calls == []
    assert _login_step(result).skipped is True


@pytest.mark.asyncio
async def test_docker_login_runs_for_custom_build(svc, monkeypatch):
    """A custom build's `FROM` may pull a private base image, so the login stays. The
    backend guarantees this by forcing is_cached=False (→ ships_sshd=False) for custom
    builds, even when the base image happens to be a default ref."""
    ssh_client = _ssh_client(inspect_exit=0)
    _patch_happy(svc, monkeypatch, ssh_client)
    monkeypatch.setattr(svc, "_custom_build_image", AsyncMock(return_value=(True, None)))

    result = await _run(
        svc,
        _payload(
            docker_image=_DEFAULT_IMAGE,
            dockerfile_content=f"FROM {_DEFAULT_IMAGE}\nRUN echo hi\n",
            ships_sshd=False,
            **_CREDS,
        ),
    )

    assert isinstance(result, ContainerCreated)
    assert _docker_client(svc).login_calls == [{"username": "renter", "password": "renter-secret"}]
    assert _login_step(result).skipped is False
