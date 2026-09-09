"""DAH-3258 — rental post-run concurrency (RENTAL_POSTRUN_CONCURRENT_ENABLED).

After `docker run` a rent ran, serially: the gocryptfs mount (4 SSH commands), the authorized_keys
exec, the sshd bootstrap / Jupyter, the /etc/environment exec and the inspector collector start.
With the flag on the inspector start runs alongside the mount and the keys and environment are one
exec; the mount → keys order and every step's completion before ContainerCreated are unchanged.
Covered here:

- the combined exec spec: without environment it IS the keys spec; with it the keys stay on stdin,
  the lines ride in one exec-process variable, argv and the log fields carry no renter value;
- the combined script through `sh -c` writes both files with the same bytes the two specs wrote;
- `add_ssh_public_keys_with_rental_docker(environment=…)` runs one exec and names both on failure;
- `create_container`: flag off → the serial order and two execs; flag on → the collector starts
  before the mount finishes and is awaited before the return, one exec carries keys + environment,
  the environment step is skipped; no environment → the plain keys exec; an environment above
  MAX_ENVIRONMENT_EXEC_VAR_BYTES keeps its own exec; a failed mount settles the collector (stop when
  no other rented container) before the cleanup; ENABLE_INSPECTOR off → no task.
"""

from __future__ import annotations

import asyncio
import subprocess
from unittest.mock import AsyncMock, Mock

import pytest
from core.config import settings
from payload_models.payloads import CustomOptions, FailedContainerRequest, ProfilerStepName
from services.docker_service import DockerService
from services.rental_docker_observability import rental_exec_spec_log_fields
from services.rental_docker_sdk import (
    ENVIRONMENT_LINES_EXEC_VAR,
    MAX_ENVIRONMENT_EXEC_VAR_BYTES,
    ContainerExecResult,
    build_authorized_keys_and_environment_exec_spec,
    build_authorized_keys_exec_spec,
    build_environment_exec_spec,
    environment_fits_exec_variable,
)
from test_deploy_optimizations import (
    _patch_happy,
    _payload as _deploy_payload,
    _run as _run_create_container,
    _ssh_client as _deploy_ssh_client,
)

_KEYS = ["ssh-ed25519 AAAAkey1 a@b", "ssh-rsa AAAAkey2"]
_ENV = {"HF_TOKEN": "hf_secret_value", "EMPTY": "", " ": "x", "MODEL": "llama"}


@pytest.fixture
def svc():
    return DockerService(ssh_service=Mock(), redis_service=Mock(), attestation_service=Mock())


# ------------------------------------------------------------------
# the combined exec spec
# ------------------------------------------------------------------


def test_combined_spec_without_environment_is_the_keys_spec():
    combined = build_authorized_keys_and_environment_exec_spec(
        container_name="pod_x", public_keys=_KEYS, environment=None
    )
    assert combined == build_authorized_keys_exec_spec(container_name="pod_x", public_keys=_KEYS)
    assert combined.environment == {}
    blank_only = build_authorized_keys_and_environment_exec_spec(
        container_name="pod_x", public_keys=_KEYS, environment={"EMPTY": "", " ": "x"}
    )
    assert blank_only == combined


def test_combined_spec_keys_on_stdin_environment_in_one_exec_variable():
    keys_spec = build_authorized_keys_exec_spec(container_name="pod_x", public_keys=_KEYS)
    env_spec = build_environment_exec_spec(container_name="pod_x", environment=_ENV)
    combined = build_authorized_keys_and_environment_exec_spec(
        container_name="pod_x", public_keys=_KEYS, environment=_ENV
    )
    assert combined.stdin == keys_spec.stdin
    assert combined.environment == {ENVIRONMENT_LINES_EXEC_VAR: env_spec.stdin}
    assert (
        env_spec.stdin == "HF_TOKEN=hf_secret_value\nMODEL=llama\n"
    )  # blank key / value filtered as before
    assert combined.argv[:2] == ("sh", "-c")
    assert combined.argv[2].startswith(keys_spec.argv[2] + " && ")
    assert f"printf '%s' \"${ENVIRONMENT_LINES_EXEC_VAR}\" >> /etc/environment" in combined.argv[2]
    assert len(combined.argv) == 3


def test_combined_spec_no_renter_value_in_argv_or_log_fields():
    combined = build_authorized_keys_and_environment_exec_spec(
        container_name="pod_x", public_keys=_KEYS, environment=_ENV
    )
    fields = rental_exec_spec_log_fields(combined)
    for secret in ("hf_secret_value", "llama", "AAAAkey1"):
        assert secret not in " ".join(combined.argv)
        assert secret not in repr(fields)
    assert fields["environment_keys"] == [ENVIRONMENT_LINES_EXEC_VAR]
    assert fields["stdin_bytes"] == len(combined.stdin.encode())


def test_combined_script_through_sh_writes_both_files_like_the_two_execs(tmp_path):
    keys_path = tmp_path / "home" / ".ssh" / "authorized_keys"
    env_path = tmp_path / "environment"
    env_path.write_text("PRE=1\n")
    combined = build_authorized_keys_and_environment_exec_spec(
        container_name="pod_x", public_keys=_KEYS, environment=_ENV, target_path=str(keys_path)
    )
    script = combined.argv[2].replace("/etc/environment", str(env_path))
    proc = subprocess.run(
        ["sh", "-c", script],
        input=combined.stdin,
        text=True,
        capture_output=True,
        env={"PATH": "/usr/bin:/bin", **combined.environment},
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert (
        keys_path.read_text()
        == build_authorized_keys_exec_spec(container_name="pod_x", public_keys=_KEYS).stdin
    )
    assert oct(keys_path.parent.stat().st_mode & 0o777) == "0o700"
    env_spec = build_environment_exec_spec(container_name="pod_x", environment=_ENV)
    assert env_path.read_text() == "PRE=1\n" + env_spec.stdin


def test_combined_script_percent_in_a_value_is_literal(tmp_path):
    env_path = tmp_path / "environment"
    keys_path = tmp_path / "authorized_keys"
    combined = build_authorized_keys_and_environment_exec_spec(
        container_name="pod_x",
        public_keys=_KEYS,
        environment={"PCT": "100%s done %d"},
        target_path=str(keys_path),
    )
    script = combined.argv[2].replace("/etc/environment", str(env_path))
    subprocess.run(
        ["sh", "-c", script],
        input=combined.stdin,
        text=True,
        env={"PATH": "/usr/bin:/bin", **combined.environment},
        check=True,
    )
    assert env_path.read_text() == "PCT=100%s done %d\n"


# ------------------------------------------------------------------
# add_ssh_public_keys_with_rental_docker(environment=…)
# ------------------------------------------------------------------


class _ExecClient:
    def __init__(self, exit_status: int = 0):
        self.exec_specs = []
        self.exit_status = exit_status

    async def exec_in_container(self, spec):
        self.exec_specs.append(spec)
        return ContainerExecResult(
            exit_status=self.exit_status, stdout="", stderr="boom" if self.exit_status else ""
        )


@pytest.mark.asyncio
async def test_add_keys_with_environment_runs_one_combined_exec(svc, monkeypatch):
    monkeypatch.setattr(svc, "stream_log", AsyncMock())
    client = _ExecClient()
    await svc.add_ssh_public_keys_with_rental_docker(
        client,
        container_name="pod_x",
        public_keys=_KEYS,
        log_tag="t",
        log_extra={},
        environment=_ENV,
    )
    assert client.exec_specs == [
        build_authorized_keys_and_environment_exec_spec(
            container_name="pod_x", public_keys=_KEYS, environment=_ENV
        )
    ]


@pytest.mark.asyncio
async def test_add_keys_without_environment_runs_the_keys_exec_as_before(svc, monkeypatch):
    monkeypatch.setattr(svc, "stream_log", AsyncMock())
    client = _ExecClient()
    await svc.add_ssh_public_keys_with_rental_docker(
        client, container_name="pod_x", public_keys=_KEYS, log_tag="t", log_extra={}
    )
    await svc.add_ssh_public_keys_with_rental_docker(
        client, container_name="pod_x", public_keys=_KEYS, log_tag="t", log_extra={}, environment={}
    )
    keys_spec = build_authorized_keys_exec_spec(container_name="pod_x", public_keys=_KEYS)
    assert client.exec_specs == [keys_spec, keys_spec]


@pytest.mark.asyncio
async def test_add_keys_with_environment_failure_names_both(svc, monkeypatch):
    monkeypatch.setattr(svc, "stream_log", AsyncMock())
    with pytest.raises(Exception, match="Failed to add SSH public keys and environment"):
        await svc.add_ssh_public_keys_with_rental_docker(
            _ExecClient(exit_status=1),
            container_name="pod_x",
            public_keys=_KEYS,
            log_tag="t",
            log_extra={},
            environment=_ENV,
        )
    with pytest.raises(Exception, match="Failed to add SSH public keys: "):
        await svc.add_ssh_public_keys_with_rental_docker(
            _ExecClient(exit_status=1),
            container_name="pod_x",
            public_keys=_KEYS,
            log_tag="t",
            log_extra={},
        )


# ------------------------------------------------------------------
# create_container wiring
# ------------------------------------------------------------------


class _Trace:
    """Records the post-run steps in the order they start and finish. The mount blocks until the
    inspector start has been observed (or a short timeout), so the trace shows whether the two
    overlapped; the inspector start itself takes 50 ms after that, so it can only be complete at
    the return when create_container awaited it."""

    async def cleanup(self, **_):
        self.events.append("cleanup")
        return False

    def __init__(self, *, mount_fails: bool = False):
        self.events: list[str] = []
        self.mount_fails = mount_fails
        self.inspector_started = asyncio.Event()
        self.lifecycle_actions: list[str] = []

    async def mount(self, **_):
        self.events.append("mount_start")
        try:
            await asyncio.wait_for(self.inspector_started.wait(), timeout=0.2)
        except asyncio.TimeoutError:
            pass
        self.events.append("mount_done")
        if self.mount_fails:
            raise RuntimeError("mount failed")

    async def lifecycle(self, *, action, **_):
        self.lifecycle_actions.append(action)
        self.events.append(f"inspector_{action}_start")
        if action == "start":
            self.inspector_started.set()
            await asyncio.sleep(0.05)
        else:
            await asyncio.sleep(0)
        self.events.append(f"inspector_{action}_done")


def _wire(
    svc, monkeypatch, *, trace: _Trace, inspector: bool = True, rented_elsewhere: bool = False
):
    ssh_client = _deploy_ssh_client()
    _patch_happy(svc, monkeypatch, ssh_client)
    monkeypatch.setattr(settings, "ENABLE_VOLUME_ENCRYPTION", True)
    monkeypatch.setattr(settings, "ENABLE_INSPECTOR", inspector)
    monkeypatch.setattr(svc, "_image_has_encrypted_volume_label", AsyncMock(return_value=True))
    monkeypatch.setattr(svc, "setup_encrypted_local_volume", trace.mount)
    monkeypatch.setattr(svc, "_run_inspector_collector_lifecycle", trace.lifecycle)
    monkeypatch.setattr(svc, "_has_rented_containers", AsyncMock(return_value=rented_elsewhere))
    monkeypatch.setattr(
        "services.docker_service.restore_tracked_gpu_power_limits", AsyncMock(return_value=0)
    )
    monkeypatch.setattr(
        "services.docker_service.raise_low_power_limits_to_default", AsyncMock(return_value=0)
    )
    original_keys = svc.add_ssh_public_keys_with_rental_docker
    original_env = svc.add_environment_variables_with_rental_docker

    async def _keys(**kw):
        trace.events.append("keys_exec")
        await asyncio.sleep(0)  # the SDK exec yields to the loop; the fake client does not
        result = await original_keys(**kw)
        trace.events.append("keys_exec_done")
        return result

    async def _env(**kw):
        trace.events.append("env_exec")
        return await original_env(**kw)

    monkeypatch.setattr(svc, "add_ssh_public_keys_with_rental_docker", _keys)
    monkeypatch.setattr(svc, "add_environment_variables_with_rental_docker", _env)
    return ssh_client


def _payload(**over):
    base = dict(
        enable_volume_encryption=True,
        is_sysbox=True,
        ships_sshd=True,
        custom_options=CustomOptions(environment={"HF_TOKEN": "hf_secret_value"}),
    )
    base.update(over)
    return _deploy_payload(**base)


def _exec_specs(svc):
    return svc.rental_docker_client_factory.client.exec_specs


def _step(result, name: ProfilerStepName):
    return next(p for p in result.profilers if p.name == name)


@pytest.mark.asyncio
async def test_flag_off_keeps_the_serial_order_and_two_execs(svc, monkeypatch):
    monkeypatch.setattr(settings, "RENTAL_POSTRUN_CONCURRENT_ENABLED", False)
    trace = _Trace()
    _wire(svc, monkeypatch, trace=trace)
    result = await _run_create_container(svc, _payload())
    assert type(result).__name__ == "ContainerCreated", getattr(result, "msg", "")
    assert trace.events == [
        "mount_start",
        "mount_done",
        "keys_exec",
        "keys_exec_done",
        "env_exec",
        "inspector_start_start",
        "inspector_start_done",
    ]
    specs = _exec_specs(svc)
    assert len(specs) == 2
    assert specs[0].environment == {} and "authorized_keys" in specs[0].argv[2]
    assert (
        specs[1].argv == ("sh", "-c", "cat >> /etc/environment")
        and specs[1].stdin == "HF_TOKEN=hf_secret_value\n"
    )
    assert _step(result, ProfilerStepName.ADDING_PUBLIC_KEYS).skipped is False
    assert _step(result, ProfilerStepName.INSPECTOR_START).skipped is False


@pytest.mark.asyncio
async def test_flag_on_inspector_overlaps_the_mount_and_one_exec_carries_keys_and_environment(
    svc, monkeypatch
):
    monkeypatch.setattr(settings, "RENTAL_POSTRUN_CONCURRENT_ENABLED", True)
    trace = _Trace()
    _wire(svc, monkeypatch, trace=trace)
    result = await _run_create_container(svc, _payload())
    assert type(result).__name__ == "ContainerCreated", getattr(result, "msg", "")
    ev = trace.events
    assert ev.index("inspector_start_start") < ev.index("mount_done"), ev
    assert ev.index("mount_done") < ev.index(
        "keys_exec"
    ), ev  # keys land on the mounted /root, never before
    assert "env_exec" not in ev
    # awaited before ContainerCreated: the start takes 50 ms after the mount let go of it, so it can
    # only be complete here because create_container waited for it (drop that await and this fails)
    assert ev[-1] == "inspector_start_done", ev
    assert _step(result, ProfilerStepName.INSPECTOR_START).duration is not None
    specs = _exec_specs(svc)
    assert len(specs) == 1
    assert specs[0] == build_authorized_keys_and_environment_exec_spec(
        container_name=result.container_name,
        public_keys=["ssh-ed25519 test-key"],
        environment={"HF_TOKEN": "hf_secret_value"},
    )
    assert _step(result, ProfilerStepName.ADDING_PUBLIC_KEYS).skipped is True
    assert _step(result, ProfilerStepName.INSPECTOR_START).skipped is False
    assert trace.lifecycle_actions == ["start"]


@pytest.mark.asyncio
async def test_flag_on_without_environment_runs_the_plain_keys_exec(svc, monkeypatch):
    monkeypatch.setattr(settings, "RENTAL_POSTRUN_CONCURRENT_ENABLED", True)
    trace = _Trace()
    _wire(svc, monkeypatch, trace=trace)
    result = await _run_create_container(svc, _payload(custom_options=None))
    assert type(result).__name__ == "ContainerCreated", getattr(result, "msg", "")
    specs = _exec_specs(svc)
    assert len(specs) == 1
    assert specs[0] == build_authorized_keys_exec_spec(
        container_name=result.container_name, public_keys=["ssh-ed25519 test-key"]
    )
    assert "env_exec" not in trace.events


@pytest.mark.asyncio
async def test_flag_on_unencrypted_pod_still_overlaps_the_inspector_with_the_exec(svc, monkeypatch):
    monkeypatch.setattr(settings, "RENTAL_POSTRUN_CONCURRENT_ENABLED", True)
    trace = _Trace()
    _wire(svc, monkeypatch, trace=trace)
    result = await _run_create_container(svc, _payload(enable_volume_encryption=False))
    assert type(result).__name__ == "ContainerCreated", getattr(result, "msg", "")
    ev = trace.events
    assert "mount_start" not in ev
    # the exec yields to the loop; the collector start runs during it, not after it
    assert ev.index("inspector_start_start") < ev.index("keys_exec_done"), ev
    assert ev[-1] == "inspector_start_done", ev
    assert trace.lifecycle_actions == ["start"]


@pytest.mark.asyncio
async def test_flag_on_failed_mount_settles_the_inspector_before_cleanup(svc, monkeypatch):
    monkeypatch.setattr(settings, "RENTAL_POSTRUN_CONCURRENT_ENABLED", True)
    trace = _Trace(mount_fails=True)
    _wire(svc, monkeypatch, trace=trace)
    monkeypatch.setattr(svc, "cleanup_failed_container_creation", trace.cleanup)
    result = await _run_create_container(svc, _payload())
    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "encrypted_volume_setup"
    assert trace.lifecycle_actions == [
        "start",
        "stop",
    ]  # no other rented container → the host is left as before
    ev = trace.events
    assert (
        ev.index("inspector_start_done")
        < ev.index("inspector_stop_start")
        < ev.index("inspector_stop_done")
        < ev.index("cleanup")
    ), ev
    assert ev.count("cleanup") == 1
    assert _exec_specs(svc) == []


@pytest.mark.asyncio
async def test_flag_on_failed_mount_keeps_the_inspector_when_another_pod_is_rented(
    svc, monkeypatch
):
    monkeypatch.setattr(settings, "RENTAL_POSTRUN_CONCURRENT_ENABLED", True)
    trace = _Trace(mount_fails=True)
    _wire(svc, monkeypatch, trace=trace, rented_elsewhere=True)
    monkeypatch.setattr(svc, "cleanup_failed_container_creation", AsyncMock(return_value=False))
    result = await _run_create_container(svc, _payload())
    assert isinstance(result, FailedContainerRequest)
    assert trace.lifecycle_actions == ["start"]


@pytest.mark.asyncio
async def test_flag_on_oversized_environment_keeps_its_own_exec(svc, monkeypatch):
    monkeypatch.setattr(settings, "RENTAL_POSTRUN_CONCURRENT_ENABLED", True)
    trace = _Trace()
    _wire(svc, monkeypatch, trace=trace)
    big = {"BLOB": "x" * (MAX_ENVIRONMENT_EXEC_VAR_BYTES + 1)}
    assert not environment_fits_exec_variable(big)
    assert environment_fits_exec_variable({"BLOB": "x" * (MAX_ENVIRONMENT_EXEC_VAR_BYTES - 10)})
    result = await _run_create_container(
        svc, _payload(custom_options=CustomOptions(environment=big))
    )
    assert type(result).__name__ == "ContainerCreated", getattr(result, "msg", "")
    specs = _exec_specs(svc)
    assert len(specs) == 2
    assert specs[0] == build_authorized_keys_exec_spec(
        container_name=result.container_name, public_keys=["ssh-ed25519 test-key"]
    )
    assert specs[1] == build_environment_exec_spec(
        container_name=result.container_name, environment=big
    )
    assert "env_exec" in trace.events
    assert _step(result, ProfilerStepName.ADDING_PUBLIC_KEYS).skipped is False


@pytest.mark.asyncio
async def test_flag_on_inspector_disabled_starts_nothing(svc, monkeypatch):
    monkeypatch.setattr(settings, "RENTAL_POSTRUN_CONCURRENT_ENABLED", True)
    trace = _Trace()
    _wire(svc, monkeypatch, trace=trace, inspector=False)
    result = await _run_create_container(svc, _payload())
    assert type(result).__name__ == "ContainerCreated", getattr(result, "msg", "")
    assert trace.lifecycle_actions == []
    assert _step(result, ProfilerStepName.INSPECTOR_START).skipped is True


@pytest.mark.asyncio
async def test_flag_off_failed_mount_never_started_the_inspector(svc, monkeypatch):
    monkeypatch.setattr(settings, "RENTAL_POSTRUN_CONCURRENT_ENABLED", False)
    trace = _Trace(mount_fails=True)
    _wire(svc, monkeypatch, trace=trace)
    monkeypatch.setattr(svc, "cleanup_failed_container_creation", AsyncMock(return_value=False))
    result = await _run_create_container(svc, _payload())
    assert isinstance(result, FailedContainerRequest)
    assert trace.lifecycle_actions == []
