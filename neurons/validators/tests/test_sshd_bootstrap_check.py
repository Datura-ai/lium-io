from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod

from services.docker_service import DockerService
from services.redis_service import SSHD_BOOTSTRAP_FAILED_PREFIX, SSHD_BOOTSTRAP_FAILED_TTL_SECONDS
from services.task.checks.sshd_bootstrap import SshdBootstrapCheck
from services.task.messages import SshdBootstrapMessages as Msg
from tests.helpers import FakeRedis, build_services, build_state

EXECUTOR = "executor-123"
KEY = f"{SSHD_BOOTSTRAP_FAILED_PREFIX}:{EXECUTOR}"
FILLER = "filler_run-1"


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
        else {},
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
async def test_a_miners_own_job_or_a_dry_run_keeps_the_filler(
    context_factory, owner, remove_fillers
):
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


def _probe_ctx(context_factory, redis, exit_status=None, error=None, rented=False):
    ssh = AsyncMock()
    if error:
        ssh.run.side_effect = error
    else:
        # the probe answers first; any later call is the filler removal
        ssh.run.side_effect = [
            SimpleNamespace(exit_status=exit_status, stdout="", stderr=""),
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
    ctx, ssh, backend = _probe_ctx(context_factory, redis, exit_status=10)
    result = await SshdBootstrapCheck().run(ctx)
    assert result.passed is False
    assert result.event.reason_code == Msg.FAILED.reason
    assert redis.store[KEY] == "1"
    assert redis.ttl[KEY] == SSHD_BOOTSTRAP_FAILED_TTL_SECONDS
    probe = ssh.run.await_args_list[0].args[0]
    assert f"inspect -f '{{{{.State.Running}}}} {{{{.State.StartedAt}}}}' {FILLER}" in probe
    assert f"/usr/bin/docker exec -u 0 -i {FILLER} sh -c" in probe
    assert ssh.run.await_args_list[1].args[0] == f"/usr/bin/docker rm -fv {FILLER}"
    backend.get_filler_run_active.assert_awaited_once_with("run-1", container_missing=True)


@pytest.mark.asyncio
async def test_a_dry_run_judges_a_running_filler_without_writing_or_removing(context_factory):
    redis = FakeRedis()
    ctx, ssh, _ = _probe_ctx(context_factory, redis, exit_status=10)
    result = await SshdBootstrapCheck(remove_fillers=False).run(ctx)
    assert result.passed is False
    assert KEY not in redis.store
    assert ssh.run.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exit_status,error",
    [(0, None), (3, None), (255, None), (None, OSError("connection lost"))],
    ids=["listening", "young-or-not-running", "probe-error", "ssh-error"],
)
async def test_a_listening_young_or_unprobed_filler_passes(context_factory, exit_status, error):
    redis = FakeRedis()
    ctx, _, backend = _probe_ctx(context_factory, redis, exit_status=exit_status, error=error)
    result = await SshdBootstrapCheck().run(ctx)
    assert result.passed is True
    assert result.event.reason_code == Msg.OK.reason
    assert KEY not in redis.store
    backend.get_filler_run_active.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_rented_node_is_not_probed(context_factory):
    redis = FakeRedis()
    ctx, ssh, _ = _probe_ctx(context_factory, redis, exit_status=10, rented=True)
    result = await SshdBootstrapCheck().run(ctx)
    assert result.passed is True
    ssh.run.assert_not_awaited()


def test_the_listening_test_matches_the_bootstrap_script(tmp_path):
    import subprocess

    from services.task.checks.sshd_bootstrap import _LISTENING_TEST, running_filler_probe

    listen = "   0: 00000000:0016 00000000:0000 0A 00000000:00000000 00:00000000 00000000\n"
    other = "   0: 0100007F:1F90 00000000:0000 0A 00000000:00000000 00:00000000 00000000\n"
    for rows, want in [(listen, 0), (other, 10)]:
        (tmp_path / "tcp").write_text("  sl  local_address rem_address   st\n" + rows)
        script = _LISTENING_TEST.replace(
            "/proc/net/tcp /proc/net/tcp6", f"{tmp_path}/tcp {tmp_path}/none"
        )
        assert subprocess.run(["sh", "-c", script]).returncode == want
    # the whole probe against a fake docker: only a filler past the grace gets the exec verdict
    docker = tmp_path / "docker"
    for minutes_ago, exec_rc, want in [(20, 10, 10), (20, 0, 0), (5, 10, 3)]:
        started = subprocess.run(
            ["date", "-u", "-d", f"-{minutes_ago} min", "+%Y-%m-%dT%H:%M:%S.123456789Z"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        docker.write_text(
            f'#!/bin/sh\nif [ "$1" = inspect ]; then echo "true {started}"; else exit {exec_rc}; fi\n'
        )
        docker.chmod(0o755)
        probe = running_filler_probe("filler_x").replace("/usr/bin/docker", str(docker))
        assert subprocess.run(["sh", "-c", probe], timeout=10).returncode == want
