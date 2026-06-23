"""DAH-1524 — deploy-time optimizations for cached-template pods.

Covers (from the plan's Test Plan):
- Pull-skip (#1): skip when image present, pull when absent, fail-open on probe
  error, and `check=False` on the inspect probe.
- SSH bootstrap (#2): run the idempotent bootstrap for every rental path,
  including templates that set `ships_sshd`. Includes the DEFAULT-PATH
  REGRESSION GUARD.
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
    FailedContainerRequest,
    PayloadPortMapping,
    ProfilerStep,
    ProfilerStepName,
    now_ms,
)
from services.docker_service import DockerService

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
    """AsyncMock ssh connection whose `run` answers the image-inspect probe
    deterministically and returns exit 0 for everything else."""
    client = AsyncMock()

    def _side(cmd, *args, **kwargs):
        if "image inspect" in cmd:
            if inspect_raises:
                raise RuntimeError("probe boom")
            return _ssh_result(exit_status=inspect_exit)
        return _ssh_result(exit_status=0)

    client.run = AsyncMock(side_effect=_side)
    return client


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
    )


def _patch_happy(svc, monkeypatch, ssh_client):
    """Stub the whole deploy flow so only the new branch logic is exercised and
    create_container reaches a real ContainerCreated."""
    monkeypatch.setattr(
        "services.docker_service.asyncssh.connect",
        Mock(return_value=_ConnCtx(ssh_client)),
    )
    monkeypatch.setattr("services.docker_service.asyncssh.import_private_key", Mock())
    monkeypatch.setattr("services.docker_service.build_gpu_flags", AsyncMock(return_value=""))
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
    monkeypatch.setattr(svc, "_run_docker_create_with_port_retry", AsyncMock())
    monkeypatch.setattr(svc, "check_container_running", AsyncMock(return_value=True))
    monkeypatch.setattr(
        svc,
        "install_open_ssh_server_and_start_ssh_service",
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


def _pull_commands(svc):
    return [
        c.kwargs.get("command", "")
        for c in svc.execute_and_stream_logs.await_args_list
        if "docker pull" in c.kwargs.get("command", "")
    ]


def _ssh_run_cmds(ssh_client):
    return [c.args[0] for c in ssh_client.run.await_args_list if c.args]


# ------------------------------------------------------------------
# #1 — pull-skip
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pull_skipped_when_image_present(svc, monkeypatch):
    ssh_client = _ssh_client(inspect_exit=0)  # image present
    _patch_happy(svc, monkeypatch, ssh_client)

    result = await _run(svc, _payload())

    assert isinstance(result, ContainerCreated)
    assert _pull_commands(svc) == [], "docker pull must NOT be issued when image is present"
    pull_step = next(p for p in result.profilers if p.name == ProfilerStepName.DOCKER_PULL)
    assert pull_step.skipped is True


@pytest.mark.asyncio
async def test_pull_runs_when_image_absent(svc, monkeypatch):
    ssh_client = _ssh_client(inspect_exit=1)  # image absent
    _patch_happy(svc, monkeypatch, ssh_client)

    result = await _run(svc, _payload(docker_image="daturaai/pytorch:1.2.3"))

    assert isinstance(result, ContainerCreated)
    assert any("/usr/bin/docker pull daturaai/pytorch:1.2.3" in c for c in _pull_commands(svc))
    pull_step = next(p for p in result.profilers if p.name == ProfilerStepName.DOCKER_PULL)
    assert not pull_step.skipped


@pytest.mark.asyncio
async def test_probe_error_falls_through_to_pull(svc, monkeypatch):
    """Fail-open: a raised probe error must pull (not a CCF)."""
    ssh_client = _ssh_client(inspect_raises=True)
    _patch_happy(svc, monkeypatch, ssh_client)

    result = await _run(svc, _payload(docker_image="daturaai/pytorch:9.9.9"))

    assert isinstance(result, ContainerCreated)
    assert any("/usr/bin/docker pull daturaai/pytorch:9.9.9" in c for c in _pull_commands(svc))


@pytest.mark.asyncio
async def test_inspect_probe_uses_check_false(svc, monkeypatch):
    ssh_client = _ssh_client(inspect_exit=0)
    _patch_happy(svc, monkeypatch, ssh_client)

    await _run(svc, _payload())

    inspect_calls = [
        c for c in ssh_client.run.await_args_list if c.args and "image inspect" in c.args[0]
    ]
    assert inspect_calls, "expected a docker image inspect probe"
    assert all(c.kwargs.get("check") is False for c in inspect_calls)


# ------------------------------------------------------------------
# #2 — SSH bootstrap
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ships_sshd_true_still_runs_bootstrap(svc, monkeypatch):
    ssh_client = _ssh_client(inspect_exit=0)
    _patch_happy(svc, monkeypatch, ssh_client)

    result = await _run(svc, _payload(ships_sshd=True))

    assert isinstance(result, ContainerCreated)
    svc.install_open_ssh_server_and_start_ssh_service.assert_awaited_once()


@pytest.mark.asyncio
async def test_ships_sshd_true_fails_create_when_bootstrap_fails(svc, monkeypatch):
    ssh_client = _ssh_client(inspect_exit=0)
    _patch_happy(svc, monkeypatch, ssh_client)
    svc.install_open_ssh_server_and_start_ssh_service.return_value = False

    result = await _run(svc, _payload(ships_sshd=True))

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "ssh_bootstrap"
    svc.install_open_ssh_server_and_start_ssh_service.assert_awaited_once()


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
    svc.install_open_ssh_server_and_start_ssh_service.assert_awaited_once()


@pytest.mark.asyncio
async def test_ships_sshd_false_runs_install(svc, monkeypatch):
    ssh_client = _ssh_client(inspect_exit=0)
    _patch_happy(svc, monkeypatch, ssh_client)

    result = await _run(svc, _payload(ships_sshd=False))

    assert isinstance(result, ContainerCreated)
    svc.install_open_ssh_server_and_start_ssh_service.assert_awaited_once()


@pytest.mark.asyncio
async def test_ships_sshd_true_with_jupyter(svc, monkeypatch):
    ssh_client = _ssh_client(inspect_exit=0)
    _patch_happy(svc, monkeypatch, ssh_client)
    # Provide a jupyter port map so the jupyter branch fires.
    monkeypatch.setattr(
        svc, "generate_portMappings",
        AsyncMock(return_value=([(22, 20001, 20001)], (8888, 30888))),
    )

    result = await _run(svc, _payload(ships_sshd=True, enable_jupyter=True))

    assert isinstance(result, ContainerCreated)
    svc.install_open_ssh_server_and_start_ssh_service.assert_awaited_once()
    svc.run_jupyter.assert_awaited_once()


@pytest.mark.asyncio
async def test_key_injection_runs_after_shipped_sshd_bootstrap(svc, monkeypatch):
    ssh_client = _ssh_client(inspect_exit=0)
    _patch_happy(svc, monkeypatch, ssh_client)

    keys = ["ssh-ed25519 key-one", "ssh-ed25519 key-two"]
    result = await _run(svc, _payload(ships_sshd=True, user_public_keys=keys))

    assert isinstance(result, ContainerCreated)
    authorized = [c for c in _ssh_run_cmds(ssh_client) if "authorized_keys" in c]
    assert len(authorized) == len(keys)
    assert any("key-one" in c for c in authorized)
    assert any("key-two" in c for c in authorized)


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
    assert "Finished in subnet." in names


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
        svc, "_run_docker_create_with_port_retry",
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
