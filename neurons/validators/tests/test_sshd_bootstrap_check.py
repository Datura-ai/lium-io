from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod

from core.config import settings
from services.docker_service import DockerService
from services.redis_service import SSHD_BOOTSTRAP_FAILED_PREFIX, SSHD_BOOTSTRAP_FAILED_TTL_SECONDS
from services.task.checks.sshd_bootstrap import SshdBootstrapCheck
from services.task.messages import SshdBootstrapMessages as Msg
from tests.helpers import FakeRedis, build_services, build_state

EXECUTOR = "executor-123"
KEY = f"{SSHD_BOOTSTRAP_FAILED_PREFIX}:{EXECUTOR}"
FILLER = "filler_run-1"
FAILED_RUN = "sshd verification failed after 15s\n(running=yes, listening=no)\n"


def _ctx(context_factory, redis, rented=False, filler_owner=None, ssh=None, backend=None):
    filler_fields = (
        {
            "all_filler_containers_by_executor": {EXECUTOR: [FILLER]},
            "default_job_owner_by_executor": {EXECUTOR: filler_owner},
        }
        if filler_owner
        else {}
    )
    rented_data = RentedExecutorsResponse(
        **filler_fields,
        executors={
            EXECUTOR: RentedExecutor(
                miner_hotkey="m",
                executor_ip_address="127.0.0.1",
                executor_ip_port="22",
                pods=[RentedPod(pod_id="p", container_name="c")],
            )
        }
        if rented
        else {}
    )
    services = build_services(redis=redis, **({"backend": backend} if backend else {}))
    return context_factory(state=build_state(rented_data=rented_data), services=services, ssh=ssh)


@pytest.mark.asyncio
async def test_a_stamped_unrented_node_fails(context_factory):
    redis = FakeRedis()
    redis.store[KEY] = "1"
    result = await SshdBootstrapCheck().run(_ctx(context_factory, redis))
    assert result.passed is False
    assert result.event.reason_code == Msg.FAILED.reason


@pytest.mark.asyncio
async def test_a_stamped_node_running_our_filler_fails_and_the_filler_is_removed(context_factory):
    # the broken nodes always run a filler (its failed create is the stamp): no exemption, the filler goes
    redis = FakeRedis()
    redis.store[KEY] = "1"
    ssh = AsyncMock()
    ssh.run.return_value = SimpleNamespace(exit_status=0, stdout="", stderr="")
    backend = build_services().backend
    result = await SshdBootstrapCheck().run(
        _ctx(context_factory, redis, filler_owner="lium", ssh=ssh, backend=backend)
    )
    assert result.passed is False
    assert result.event.reason_code == Msg.FAILED.reason
    assert ssh.run.await_args.args[0] == f"/usr/bin/docker rm -fv {FILLER}"
    backend.get_filler_run_active.assert_awaited_once_with("run-1", container_missing=True)
    assert result.event.what_we_saw["removed_fillers"] == [FILLER]


@pytest.mark.asyncio
@pytest.mark.parametrize("owner,remove_fillers", [("miner", True), ("lium", False)])
async def test_a_miners_own_job_or_a_dry_run_keeps_the_filler(context_factory, owner, remove_fillers):
    redis = FakeRedis()
    redis.store[KEY] = "1"
    ssh = AsyncMock()
    result = await SshdBootstrapCheck(remove_fillers=remove_fillers).run(
        _ctx(context_factory, redis, filler_owner=owner, ssh=ssh)
    )
    assert result.passed is False
    ssh.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failed_filler_removal_keeps_the_verdict(context_factory):
    redis = FakeRedis()
    redis.store[KEY] = "1"
    ssh = AsyncMock()
    ssh.run.side_effect = OSError("connection lost")
    backend = build_services().backend
    result = await SshdBootstrapCheck().run(
        _ctx(context_factory, redis, filler_owner="lium", ssh=ssh, backend=backend)
    )
    assert result.passed is False
    assert result.event.what_we_saw["removed_fillers"] == []
    backend.get_filler_run_active.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_redis_error_passes_the_node(context_factory):
    result = await SshdBootstrapCheck().run(_ctx(context_factory, FakeRedis(failing=True)))
    assert result.passed is True
    assert result.event.reason_code == Msg.OK.reason


@pytest.mark.asyncio
async def test_kill_switch_off_only_logs(context_factory):
    redis = FakeRedis()
    redis.store[KEY] = "1"
    with patch(
        "services.task.checks.sshd_bootstrap.settings.SSHD_BOOTSTRAP_ENFORCEMENT_ENABLED", False
    ):
        result = await SshdBootstrapCheck().run(_ctx(context_factory, redis))
    assert result.passed is True
    assert result.event.reason_code == Msg.FAILED_OBSERVED.reason


@pytest.mark.asyncio
@pytest.mark.parametrize("stamped,rented", [(False, False), (True, True)])
async def test_no_stamp_or_a_rented_node_passes(context_factory, stamped, rented):
    redis = FakeRedis()
    if stamped:
        redis.store[KEY] = "1"
    result = await SshdBootstrapCheck().run(_ctx(context_factory, redis, rented=rented))
    assert result.passed is True
    assert result.event.reason_code == Msg.OK.reason


def _service(redis) -> DockerService:
    service = DockerService(ssh_service=Mock(), redis_service=redis, attestation_service=Mock())
    service.stream_log = AsyncMock()
    return service


async def _bootstrap(service, run_result, executor_uuid=EXECUTOR):
    ok = SimpleNamespace(exit_status=0, stdout="", stderr="")
    with patch(
        "services.docker_service.exec_logged_rental_docker_sdk_operation",
        AsyncMock(side_effect=[ok, run_result]),
    ):
        return await service.install_open_ssh_server_and_start_ssh_service_with_rental_docker(
            Mock(),
            container_name="filler_x",
            log_tag="t",
            log_extra={},
            sshd_verdict_executor_uuid=executor_uuid,
        )


@pytest.mark.asyncio
async def test_sshd_not_serving_stamps_and_a_later_success_clears():
    redis = FakeRedis()
    service = _service(redis)
    failed = SimpleNamespace(
        exit_status=1, stdout="sshd verification failed after 30s\n", stderr=""
    )

    assert await _bootstrap(service, failed) is False
    assert redis.store[KEY] == "1"
    assert redis.ttl[KEY] == SSHD_BOOTSTRAP_FAILED_TTL_SECONDS

    assert await _bootstrap(service, SimpleNamespace(exit_status=0, stdout="", stderr="")) is True
    assert KEY not in redis.store


@pytest.mark.asyncio
async def test_other_failures_and_renter_creates_stamp_nothing():
    redis = FakeRedis()
    service = _service(redis)
    # apt-get failing exits before verify_sshd: not evidence about the node
    assert (
        await _bootstrap(
            service, SimpleNamespace(exit_status=100, stdout="E: Unable to fetch", stderr="")
        )
        is False
    )
    assert (
        await _bootstrap(
            service,
            SimpleNamespace(exit_status=1, stdout="sshd verification failed", stderr=""),
            executor_uuid=None,
        )
        is False
    )
    assert KEY not in redis.store


def _probe_ctx(context_factory, redis, stdout="", error=None, rented=False):
    ssh = AsyncMock()
    if error:
        ssh.run.side_effect = error
    else:
        # the probe answers first; any later call is the filler removal
        ssh.run.side_effect = [
            SimpleNamespace(exit_status=1 if stdout else 0, stdout=stdout, stderr=""),
            SimpleNamespace(exit_status=0, stdout="", stderr=""),
        ]
    backend = build_services().backend
    ctx = _ctx(context_factory, redis, rented=rented, filler_owner="lium", ssh=ssh, backend=backend)
    return ctx, ssh, backend


@pytest.mark.asyncio
async def test_a_running_filler_without_sshd_past_the_grace_fails_stamps_and_is_removed(
    context_factory,
):
    redis = FakeRedis()
    ctx, ssh, backend = _probe_ctx(context_factory, redis, stdout=FAILED_RUN)
    result = await SshdBootstrapCheck().run(ctx)
    assert result.passed is False
    assert result.event.reason_code == Msg.FAILED.reason
    assert redis.store[KEY] == "1"
    assert redis.ttl[KEY] == SSHD_BOOTSTRAP_FAILED_TTL_SECONDS
    probe = ssh.run.await_args_list[0].args[0]
    assert f"inspect -f '{{{{.State.Running}}}} {{{{.State.StartedAt}}}}' {FILLER}" in probe
    assert f"| base64 -d | /usr/bin/docker exec -u 0 -i {FILLER} sh -c" in probe
    assert ssh.run.await_args_list[1].args[0] == f"/usr/bin/docker rm -fv {FILLER}"
    backend.get_filler_run_active.assert_awaited_once_with("run-1", container_missing=True)


@pytest.mark.asyncio
async def test_a_dry_run_does_not_probe_a_running_filler(context_factory):
    redis = FakeRedis()
    ctx, ssh, _ = _probe_ctx(context_factory, redis, stdout=FAILED_RUN)
    result = await SshdBootstrapCheck(remove_fillers=False).run(ctx)
    assert result.passed is True
    assert KEY not in redis.store
    ssh.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_probe_timeout_is_capped_by_the_task_time_left(context_factory):
    redis = FakeRedis()
    ctx, ssh, _ = _probe_ctx(context_factory, redis)
    # 100 s left: JOB_TIME_OUT less the 120 s margin less the time already spent
    ctx = ctx.model_copy(update={"started_at_monotonic": ctx.started_at_monotonic - (settings.JOB_TIME_OUT - 220)})
    await SshdBootstrapCheck().run(ctx)
    assert 90 <= ssh.run.await_args.kwargs["timeout"] <= 100


@pytest.mark.asyncio
async def test_no_probe_runs_with_too_little_task_time_left(context_factory):
    redis = FakeRedis()
    ctx, ssh, _ = _probe_ctx(context_factory, redis, stdout=FAILED_RUN)
    ctx = ctx.model_copy(update={"started_at_monotonic": ctx.started_at_monotonic - settings.JOB_TIME_OUT})
    result = await SshdBootstrapCheck().run(ctx)
    assert result.passed is True
    ssh.run.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stdout,error",
    [
        ("sshd already running; adopting existing daemon\n", None),
        ("", None),
        ("E: Unable to locate package openssh-server\n", None),
        (None, OSError("connection lost")),
    ],
    ids=["serving", "young-or-not-running", "apt-failed", "ssh-error"],
)
async def test_a_serving_young_apt_failed_or_unprobed_filler_passes(context_factory, stdout, error):
    redis = FakeRedis()
    ctx, _, backend = _probe_ctx(context_factory, redis, stdout=stdout or "", error=error)
    result = await SshdBootstrapCheck().run(ctx)
    assert result.passed is True
    assert result.event.reason_code == Msg.OK.reason
    assert KEY not in redis.store
    backend.get_filler_run_active.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_rented_node_is_not_probed(context_factory):
    redis = FakeRedis()
    ctx, ssh, _ = _probe_ctx(context_factory, redis, stdout=FAILED_RUN, rented=True)
    result = await SshdBootstrapCheck().run(ctx)
    assert result.passed is True
    ssh.run.assert_not_awaited()


def test_the_probe_reruns_the_bootstrap_only_past_the_grace(tmp_path):
    import os
    import subprocess
    from datetime import datetime, timedelta, timezone

    from services.task.checks.sshd_bootstrap import _BOOTSTRAP_SCRIPT, running_filler_sshd_bootstrap_command

    # a fake docker: inspect reports the start time, exec saves the script it is fed on stdin
    docker = tmp_path / "docker"
    fed = tmp_path / "fed.sh"
    # the probe needs GNU `date -d`; on macOS it is coreutils' `gdate`
    date = tmp_path / "date"
    date.write_text('#!/bin/sh\ncommand -v gdate >/dev/null && exec gdate "$@"; exec /bin/date "$@"\n')
    date.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"}
    for minutes_ago, want in [(5, 3), (20, 0)]:
        fed.unlink(missing_ok=True)
        started_at = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
        started = started_at.strftime("%Y-%m-%dT%H:%M:%S.123456789Z")
        docker.write_text(
            f'#!/bin/sh\nif [ "$1" = inspect ]; then echo "true {started}"; else cat > {fed}; fi\n'
        )
        docker.chmod(0o755)
        probe = running_filler_sshd_bootstrap_command("filler_x").replace("/usr/bin/docker", str(docker))
        assert subprocess.run(["sh", "-c", probe], timeout=10, env=env).returncode == want
        assert fed.exists() is (want == 0)
    assert fed.read_bytes() == _BOOTSTRAP_SCRIPT.read_bytes()
