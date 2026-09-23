"""NO_OUTBOUND_INTERNET: a node whose renter pods cannot reach the internet fails validation.

ticket-0361: provider 5GUBuA7k's Tokyo 8x RTX 5090 nodes shipped pods with no outbound internet (4 of the
provider's 13 rent_failed in 24 h on one node) while every check passed: the speed rule is behind
FeatureFlag.VERIFYX_NETWORK_VALIDATION (off), a scrape whose every speed test failed only recorded the error,
and the scrape measures from the executor container, not from the rental network a pod runs on.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from contextlib import contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod

from services.docker_service import DockerService
from services.rental_docker_sdk import RENTAL_NETWORK_NAME
from services.task.checks import outbound_internet as module
from services.task.checks.outbound_internet import (
    EGRESS_PROBE_SCRIPT,
    OutboundInternetCheck,
    parse_egress_probe,
    pod_probe_command,
    scrape_egress_finding,
)
from services.task.messages import OutboundInternetMessages as Msg
from services.task.pipeline_factory import PipelineFactory
from services.task.runner import SSHCommandResult
from tests.helpers import build_state, default_executor, make_context

EXECUTOR = default_executor()
POD_OK = "lium_egress dns=ok\nlium_egress tool=wget http=200\n"
POD_DNS_FAIL = "lium_egress dns=fail\n"
POD_NO_ANSWER = "lium_egress dns=ok\nlium_egress tool=wget http=000\nwget: can't connect to remote host: Timed out\n"


def network(download, upload=None, **measurements) -> dict:
    return {
        "download_speed": download,
        "upload_speed": upload,
        "download_source": "cloudflare" if download else None,
        "measurements": measurements,
        "ema_download_speed": 480.0,
    }


HEALTHY = network(812.4, 640.1, speedtest_cli={"download_speed": 812.4, "upload_speed": 640.1})
CURL_FAILED = network(
    None,
    speedtest_cli={
        "download_speed": None,
        "network_speed_error": "RuntimeError('speedtest-cli: not found')",
    },
    cloudflare={
        "download_speed": None,
        "network_speed_error": "RuntimeError(\"run_cmd error cmd='curl ...' proc.returncode=6\")",
    },
    netmeasure={"download_speed": None, "network_speed_error": "RuntimeError('netmeasure')"},
    speedcheck={"download_speed": None, "network_speed_error": "RuntimeError('speedcheck')"},
)


def result(
    stdout: str = "", *, exit_code: int = 0, error_type: str | None = None
) -> SSHCommandResult:
    now = datetime.now(UTC)
    return SSHCommandResult(
        command="",
        command_id="c",
        exit_code=exit_code,
        stdout=stdout,
        stderr="",
        duration_ms=10,
        started_at=now,
        finished_at=now,
        success=exit_code == 0 and error_type is None,
        error_type=error_type,
        error_message="timed out" if error_type else None,
    )


class FakeRunner:
    def __init__(self, *answers: SSHCommandResult):
        self.answers = list(answers)
        self.commands: list[str] = []

    async def run(self, cmd, *, timeout=60, check=False, retryable=True, stdin_text=None):
        self.commands.append(cmd)
        return self.answers.pop(0) if self.answers else result()


def rented() -> RentedExecutorsResponse:
    return RentedExecutorsResponse(
        executors={
            EXECUTOR.uuid: RentedExecutor(
                miner_hotkey="miner-hotkey",
                executor_ip_address=EXECUTOR.address,
                executor_ip_port=str(EXECUTOR.port),
                pods=[RentedPod(pod_id="p1", container_name="pod_p1")],
            )
        }
    )


def make_ctx(
    net: dict | None = HEALTHY,
    *,
    pod: SSHCommandResult | None = None,
    pod_rerun: SSHCommandResult | None = None,
    sysbox=False,
    rented_data=None,
):
    """`pod` answers the pod probe; a re-run (after a no_egress reading) gets `pod_rerun`, or `pod` again."""
    pod = pod if pod is not None else result(POD_OK)
    runner = FakeRunner(pod, pod_rerun if pod_rerun is not None else pod)
    specs = {"network": net} if net is not None else {}
    ctx = make_context(
        state=build_state(specs=specs, sysbox_runtime=sysbox, rented_data=rented_data),
        runner=runner,
    )
    return ctx, runner


@contextmanager
def flags(*, check: bool = True, enforced: bool = True):
    fake = SimpleNamespace(
        NO_OUTBOUND_INTERNET_CHECK_ENABLED=check, NO_OUTBOUND_INTERNET_ENFORCEMENT_ENABLED=enforced
    )
    with patch.object(module, "settings", fake):
        yield


def test_defaults_log_only():
    """Regression: enforcement ships on and hides hosts before the shadow rows were read, or the check ships
    off and there are no shadow rows to read."""
    fields = type(module.settings).model_fields
    assert fields["NO_OUTBOUND_INTERNET_CHECK_ENABLED"].default is True
    assert fields["NO_OUTBOUND_INTERNET_ENFORCEMENT_ENABLED"].default is False


@pytest.mark.asyncio
@pytest.mark.parametrize("download", [None, 0, 0.0])
async def test_no_measured_download_fails_with_no_outbound_internet(download):
    """Regression: a scrape with no download measured (None) or a zero passes."""
    ctx, _ = make_ctx(network(download))
    with flags():
        res = await OutboundInternetCheck().run(ctx)
    assert res.passed is False
    assert res.event.reason_code == Msg.NO_OUTBOUND_INTERNET.reason
    assert res.event.what_we_saw["failed_by"] == ["scrape"]
    assert res.event.remediation == (
        "containers on this host cannot reach the internet; check the docker bridge / FORWARD chain and DNS"
    )


@pytest.mark.asyncio
async def test_every_speed_test_erroring_fails_and_carries_the_errors():
    """Regression: a curl to speed.cloudflare.com that errors (only network_speed_error recorded) passes."""
    ctx, _ = make_ctx(CURL_FAILED)
    with flags():
        res = await OutboundInternetCheck().run(ctx)
    assert res.passed is False and res.event.reason_code == Msg.NO_OUTBOUND_INTERNET.reason
    errors = res.event.what_we_saw["scrape"]["speed_errors"]
    assert set(errors) == {"speedtest_cli", "cloudflare", "netmeasure", "speedcheck"}
    assert "curl" in errors["cloudflare"]


@pytest.mark.asyncio
async def test_an_error_from_a_method_a_fallback_measured_past_is_not_a_finding():
    """Regression: a host without speedtest-cli whose Cloudflare curl measured fine fails for the recorded
    speedtest-cli error."""
    net = network(
        320.0,
        150.0,
        speedtest_cli={
            "download_speed": None,
            "network_speed_error": "RuntimeError('speedtest-cli: not found')",
        },
        cloudflare={"download_speed": 320.0, "upload_speed": 150.0},
    )
    ctx, _ = make_ctx(net)
    with flags():
        res = await OutboundInternetCheck().run(ctx)
    assert res.passed and res.event.reason_code == Msg.OUTBOUND_INTERNET_OK.reason


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pod_stdout,reason", [(POD_DNS_FAIL, "dns_failed"), (POD_NO_ANSWER, "no_http_response")]
)
async def test_a_pod_network_that_cannot_resolve_or_reach_out_fails_although_the_scrape_measured(
    pod_stdout, reason
):
    """Regression: the scrape measures from the executor container, so a host whose rental bridge drops
    FORWARD traffic or whose pod DNS is broken reports 800 Mbps and still ships pods without internet."""
    ctx, runner = make_ctx(HEALTHY, pod=result(pod_stdout))
    with flags():
        res = await OutboundInternetCheck().run(ctx)
    assert res.passed is False and res.event.reason_code == Msg.NO_OUTBOUND_INTERNET.reason
    assert res.event.what_we_saw["failed_by"] == ["pod_probe"]
    assert res.event.what_we_saw["pod_probe"]["reason"] == reason
    assert res.event.what_we_saw["pod_probe"]["first_reading"]["reason"] == reason
    assert len(runner.commands) == 2


@pytest.mark.asyncio
async def test_one_transient_pod_probe_miss_is_re_run_and_the_re_run_decides():
    """Regression (self-review): once enforcement is on, a single DNS or HTTP miss zeroes an idle node for
    the cycle; there is no second reading."""
    ctx, runner = make_ctx(HEALTHY, pod=result(POD_NO_ANSWER), pod_rerun=result(POD_OK))
    with flags():
        res = await OutboundInternetCheck().run(ctx)
    assert res.passed and res.event.reason_code == Msg.OUTBOUND_INTERNET_OK.reason
    pod_probe = res.event.what_we_saw["pod_probe"]
    assert pod_probe["verdict"] == "ok"
    assert pod_probe["first_reading"]["reason"] == "no_http_response"
    names = [command.split("--name ")[1].split()[0] for command in runner.commands]
    assert len(names) == 2 and names[0] != names[1]


@pytest.mark.asyncio
async def test_a_re_run_that_reaches_no_verdict_does_not_confirm_the_miss():
    ctx, _ = make_ctx(HEALTHY, pod=result(POD_DNS_FAIL), pod_rerun=result("", exit_code=125))
    with flags():
        res = await OutboundInternetCheck().run(ctx)
    assert res.passed and res.event.reason_code == Msg.OUTBOUND_INTERNET_UNMEASURED.reason
    assert res.event.what_we_saw["pod_probe"]["first_reading"]["reason"] == "dns_failed"


@pytest.mark.asyncio
async def test_a_healthy_node_starts_one_probe_container():
    """Regression: the re-run fires on every reading, doubling the containers started on every idle node."""
    ctx, runner = make_ctx(HEALTHY)
    with flags():
        res = await OutboundInternetCheck().run(ctx)
    assert res.passed and len(runner.commands) == 1
    assert "first_reading" not in res.event.what_we_saw["pod_probe"]


@pytest.mark.asyncio
async def test_a_slow_but_working_host_passes():
    """Regression: this check re-imposes a speed floor (VERIFYX_NETWORK_VALIDATION's rule, which is off)."""
    ctx, _ = make_ctx(network(3.2, 1.1, cloudflare={"download_speed": 3.2, "upload_speed": 1.1}))
    with flags():
        res = await OutboundInternetCheck().run(ctx)
    assert res.passed and res.event.reason_code == Msg.OUTBOUND_INTERNET_OK.reason


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "net,pod", [(network(None), result(POD_OK)), (HEALTHY, result(POD_DNS_FAIL))]
)
async def test_enforcement_off_only_logs(net, pod):
    """Regression: the finding fails the node while NO_OUTBOUND_INTERNET_ENFORCEMENT_ENABLED is off."""
    ctx, _ = make_ctx(net, pod=pod)
    with flags(enforced=False):
        res = await OutboundInternetCheck().run(ctx)
    assert res.passed is True
    assert res.event.reason_code == Msg.NO_OUTBOUND_INTERNET_OBSERVED.reason
    assert res.event.severity == "warning"
    assert res.event.what_we_saw["enforced"] is False


@pytest.mark.asyncio
async def test_a_pod_probe_that_did_not_run_is_no_verdict():
    """Regression: docker refusing the probe container (exit 125, image pull blocked) fails the node, or
    (self-review) is logged as OUTBOUND_INTERNET_OK, so the enforcement flip's counts include nodes the pod
    probe never measured."""
    ctx, _ = make_ctx(HEALTHY, pod=result("", exit_code=125))
    with flags():
        res = await OutboundInternetCheck().run(ctx)
    assert res.passed and res.event.reason_code == Msg.OUTBOUND_INTERNET_UNMEASURED.reason
    assert res.event.reason_code != Msg.OUTBOUND_INTERNET_OK.reason
    assert res.event.what_we_saw["pod_probe"]["verdict"] == "unmeasured"
    assert res.event.what_we_saw["pod_probe"]["reason"] == "docker_run_failed"


@pytest.mark.asyncio
async def test_a_timed_out_pod_probe_removes_its_container_and_is_no_verdict():
    """Regression: an SSH timeout leaves the probe container running on the host."""
    ctx, runner = make_ctx(HEALTHY, pod=result(error_type="timeout", exit_code=-1))
    with flags():
        res = await OutboundInternetCheck().run(ctx)
    assert res.passed and res.event.reason_code == Msg.OUTBOUND_INTERNET_UNMEASURED.reason
    assert res.event.what_we_saw["pod_probe"]["reason"] == "command_failed"
    name = runner.commands[0].split("--name ")[1].split()[0]
    assert runner.commands[1].startswith(f"/usr/bin/docker rm -f {name}")


@pytest.mark.asyncio
async def test_a_rented_node_is_left_alone():
    """Regression: a container starts next to a renter's pod, or a rented node is failed for its scrape."""
    ctx, runner = make_ctx(network(None), rented_data=rented())
    with flags():
        res = await OutboundInternetCheck().run(ctx)
    assert res.passed and res.event.reason_code == Msg.SKIPPED.reason
    assert res.event.what_we_saw["scrape"]["no_egress"] is True
    assert runner.commands == []


@pytest.mark.asyncio
async def test_check_off_starts_nothing():
    ctx, runner = make_ctx(network(None))
    with flags(check=False):
        res = await OutboundInternetCheck().run(ctx)
    assert res.passed and res.event.reason_code == Msg.SKIPPED.reason
    assert runner.commands == []


@pytest.mark.asyncio
async def test_dry_run_reads_the_scrape_without_a_container():
    ctx, runner = make_ctx(network(None))
    with flags():
        res = await OutboundInternetCheck(run_pod_probe=False).run(ctx)
    assert res.passed is False and res.event.what_we_saw["pod_probe"] is None
    assert runner.commands == []


def _rental_run_spec(*, is_sysbox: bool):
    payload = SimpleNamespace(
        is_sysbox=is_sysbox,
        cluster_membership=None,
        docker_image="daturaai/pytorch:2.7.0",
        memory_gb=None,
        workload_kind=None,
        cache_volumes=[],
    )
    custom_options = SimpleNamespace(
        environment={}, startup_commands=None, shm_size=None, entrypoint=None
    )
    gpu_devices = SimpleNamespace(device_mounts=[], device_requests=())
    with patch("services.docker_service._build_cache_volume_mounts", return_value=[]):
        return DockerService._build_rental_container_run_spec(
            DockerService.__new__(DockerService),
            payload=payload,
            container_name="pod_x",
            custom_options=custom_options,
            port_maps=[],
            local_volume="volume_x",
            local_volume_path="/root",
            encrypted_local_volume=False,
            external_volume_name=None,
            gpu_devices=gpu_devices,
            effective_storage_limit_gb=None,
            cpu_count=None,
        )


@pytest.mark.parametrize("sysbox", [False, True])
def test_pod_probe_container_is_networked_like_a_rental_pod(sysbox):
    """Regression: the probe runs on docker0 or the host network (where the scrape already measures), or with
    a DNS override the rental path does not set, so it cannot see the pod network's fault."""
    spec = _rental_run_spec(is_sysbox=sysbox)
    command = pod_probe_command("lium_egress_probe_abc", sysbox=sysbox)
    assert spec.network == RENTAL_NETWORK_NAME
    assert f"--network {spec.network} " in command
    assert ("--runtime=sysbox-runc" in command) is (spec.runtime == "sysbox-runc")
    assert "--dns" not in command and "--network=host" not in command
    assert (
        "docker network create --driver bridge -o com.docker.network.bridge.enable_icc=false"
        in command
    )


@pytest.mark.asyncio
async def test_the_check_sends_the_node_runtime_to_the_pod_probe():
    ctx, runner = make_ctx(HEALTHY, sysbox=True)
    with flags():
        await OutboundInternetCheck().run(ctx)
    assert "--runtime=sysbox-runc" in runner.commands[0]


def test_scrape_without_a_network_block_is_no_reading():
    assert scrape_egress_finding({}) is None
    assert scrape_egress_finding({"network": {"download_speed": float("nan")}})["no_egress"] is True


def _stub(directory, name: str, body: str) -> None:
    path = os.path.join(directory, name)
    with open(path, "w") as handle:
        handle.write(f"#!/bin/sh\n{body}\n")
    os.chmod(path, 0o755)


@pytest.mark.parametrize(
    "stubs,verdict,reason",
    [
        ({"getent": "exit 2", "curl": "printf 200"}, "no_egress", "dns_failed"),
        (
            {
                "getent": "echo 151.101.0.223 pypi.org",
                "curl": "printf 000; echo 'curl: (7) refused' >&2; exit 7",
            },
            "no_egress",
            "no_http_response",
        ),
        ({"getent": "echo 151.101.0.223 pypi.org", "curl": "printf 200"}, "ok", "http_response"),
        # alpine: busybox wget, no curl
        (
            {"getent": "echo 151.101.0.223 pypi.org", "wget": "echo '  HTTP/1.1 200 OK' >&2"},
            "ok",
            "http_response",
        ),
        (
            {
                "getent": "echo 151.101.0.223 pypi.org",
                "wget": "echo 'wget: bad address' >&2; exit 1",
            },
            "no_egress",
            "no_http_response",
        ),
        ({"getent": "echo 151.101.0.223 pypi.org"}, "unmeasured", "no_fetch_tool"),
    ],
)
def test_the_probe_script_under_sh(tmp_path, stubs, verdict, reason):
    """The script as the container's /bin/sh runs it, with getent/curl/wget stubbed: a broken quote or a
    busybox-incompatible construct turns every verdict into `unmeasured` (never a fail, never a pass)."""
    out = _run_script(tmp_path, stubs)
    probe = parse_egress_probe(out.stdout)
    assert out.returncode == 0
    assert (probe.verdict, probe.reason) == (verdict, reason)


def _run_script(tmp_path, stubs: dict[str, str]) -> subprocess.CompletedProcess:
    for name in ("awk", "tail", "cat"):
        os.symlink(shutil.which(name), tmp_path / name)
    for name, body in stubs.items():
        _stub(tmp_path, name, body)
    return subprocess.run(
        [shutil.which("sh"), "-c", EGRESS_PROBE_SCRIPT],
        env={"PATH": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=30,
        cwd=tmp_path,
    )


RESOLVES = "echo 151.101.0.223 pypi.org"


def test_the_wget_branch_asks_for_headers_only_under_a_total_deadline(tmp_path):
    """Regression (self-review): alpine's busybox wget fetched https://pypi.org/simple/ (46 MB) in full on
    every idle node every cycle, because its -T is a per-read timeout: no total bound, 4.4 GB a day per node,
    and a host under ~4 Mbps ran into the 90 s SSH timeout and never got a verdict."""
    out = _run_script(
        tmp_path,
        {
            "getent": RESOLVES,
            "timeout": f'echo "$@" > {tmp_path}/timeout.args; shift; exec "$@"',
            "wget": f"echo \"$@\" > {tmp_path}/wget.args; echo '  HTTP/1.1 200 OK' >&2",
        },
    )
    assert parse_egress_probe(out.stdout).verdict == "ok"
    timeout_args = (tmp_path / "timeout.args").read_text().split()
    wget_args = (tmp_path / "wget.args").read_text().split()
    assert timeout_args[:2] == ["10", "wget"]
    assert "--spider" in wget_args and "-O" not in wget_args
    assert wget_args[-1] == "https://pypi.org/robots.txt"
    assert "/simple" not in EGRESS_PROBE_SCRIPT


def test_a_wget_that_hangs_is_cut_off_by_the_deadline_and_reads_as_no_answer(tmp_path):
    """The total bound is real: a fetch still trickling in is killed and the verdict is no_http_response,
    not a probe that runs into the SSH timeout. The stubbed `timeout` shortens the 10 s to 1 s."""
    out = _run_script(
        tmp_path,
        {
            "getent": RESOLVES,
            "timeout": f'shift; exec {shutil.which("timeout")} 1 "$@"',
            "wget": f"exec {shutil.which('sleep')} 30",
        },
    )
    probe = parse_egress_probe(out.stdout)
    assert (probe.verdict, probe.reason) == ("no_egress", "no_http_response")


def test_the_curl_branch_asks_for_headers_only_under_a_total_deadline(tmp_path):
    out = _run_script(
        tmp_path,
        {"getent": RESOLVES, "curl": f'echo "$@" > {tmp_path}/curl.args; printf 200'},
    )
    assert parse_egress_probe(out.stdout).verdict == "ok"
    curl_args = (tmp_path / "curl.args").read_text().split()
    assert curl_args[curl_args.index("-m") + 1] == "10"
    assert "-I" in curl_args and curl_args[curl_args.index("-o") + 1] == "/dev/null"
    assert curl_args[-1] == "https://pypi.org/robots.txt"


def test_pipeline_runs_the_check_after_the_port_checks_and_before_the_rented_halt():
    ids = [check.check_id for check in PipelineFactory.build_checks()]
    index = ids.index(OutboundInternetCheck.check_id)
    assert ids[index - 1] == "executor.validate.port_count"
    assert index < ids.index("executor.validate.rented_state")
    dry = [
        c
        for c in PipelineFactory.build_dry_run_checks()
        if c.check_id == OutboundInternetCheck.check_id
    ]
    assert len(dry) == 1 and dry[0].run_pod_probe is False
