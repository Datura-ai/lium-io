"""NO_OUTBOUND_INTERNET: an idle node whose scrape ran its speed tests and measured neither direction fails.

ticket-0361: a scrape whose every speed test failed only recorded the error while every check passed (the speed
rule is behind FeatureFlag.VERIFYX_NETWORK_VALIDATION, off). The registry path that actually broke 14e704ba's
rents is RegistryPullCheck's (test_registry_pull_check.py); EGRESS_PROBE_SCRIPT is the rental probe's egress step.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod

from services.task.checks import outbound_internet as module
from services.task.checks.outbound_internet import (
    EGRESS_PROBE_SCRIPT,
    OutboundInternetCheck,
    parse_egress_probe,
    scrape_egress_finding,
)
from services.task.messages import OutboundInternetMessages as Msg
from services.task.pipeline_factory import PipelineFactory
from tests.helpers import build_state, default_executor, make_context

EXECUTOR = default_executor()


def network(download, upload=None, **measurements) -> dict:
    """A scrape's network block; without `measurements`, one Cloudflare test that read these figures."""
    if not measurements:
        measurements = {"cloudflare": {"download_speed": download, "upload_speed": upload}}
    return {
        "download_speed": download,
        "upload_speed": upload,
        "download_source": "cloudflare" if download else None,
        "upload_source": "cloudflare" if upload else None,
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


class FakeRunner:
    def __init__(self):
        self.commands: list[str] = []

    async def run(self, cmd, *, timeout=60, check=False, retryable=True, stdin_text=None):
        self.commands.append(cmd)
        raise AssertionError(f"OutboundInternetCheck ran a command on the executor: {cmd}")


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


def make_ctx(net: dict | None = HEALTHY, *, rented_data=None):
    runner = FakeRunner()
    specs = {"network": net} if net is not None else {}
    ctx = make_context(state=build_state(specs=specs, rented_data=rented_data), runner=runner)
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
@pytest.mark.parametrize("download,upload", [(None, None), (0, 0), (0.0, None), (None, 0.0)])
async def test_a_scrape_that_measured_neither_direction_fails_with_no_outbound_internet(
    download, upload
):
    """Regression: a scrape with neither a download nor an upload measured (None) or a zero passes."""
    ctx, _ = make_ctx(network(download, upload))
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
@pytest.mark.parametrize("download", [None, 0.0])
async def test_a_null_download_with_a_working_upload_passes(download):
    """Regression: 24 of one provider's 27 active nodes had no download (a Cloudflare download recorded as
    0) and fail NO_OUTBOUND_INTERNET for it once enforcement is on, although their pods reach out."""
    ctx, _ = make_ctx(
        network(download, 90.0, cloudflare={"download_speed": None, "upload_speed": 90.0})
    )
    with flags():
        res = await OutboundInternetCheck().run(ctx)
    assert res.passed and res.event.reason_code == Msg.OUTBOUND_INTERNET_OK.reason
    assert res.event.what_we_saw["scrape"]["no_egress"] is False


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
async def test_a_slow_but_working_host_passes():
    """Regression: this check re-imposes a speed floor (VERIFYX_NETWORK_VALIDATION's rule, which is off)."""
    ctx, _ = make_ctx(network(3.2, 1.1, cloudflare={"download_speed": 3.2, "upload_speed": 1.1}))
    with flags():
        res = await OutboundInternetCheck().run(ctx)
    assert res.passed and res.event.reason_code == Msg.OUTBOUND_INTERNET_OK.reason


@pytest.mark.asyncio
async def test_enforcement_off_only_logs():
    """Regression: the finding fails the node while NO_OUTBOUND_INTERNET_ENFORCEMENT_ENABLED is off."""
    ctx, _ = make_ctx(network(None))
    with flags(enforced=False):
        res = await OutboundInternetCheck().run(ctx)
    assert res.passed is True
    assert res.event.reason_code == Msg.NO_OUTBOUND_INTERNET_OBSERVED.reason
    assert res.event.severity == "warning"
    assert res.event.what_we_saw["enforced"] is False


@pytest.mark.asyncio
async def test_a_rented_node_is_left_alone():
    """Regression: a rented node is failed for its scrape."""
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


def test_scrape_without_a_network_block_is_no_reading():
    assert scrape_egress_finding({}) is None
    nan = network(float("nan"), cloudflare={"download_speed": float("nan")})
    assert scrape_egress_finding({"network": nan})["no_egress"] is True
    measured_up = {"network": network(None, 90.0)}
    assert scrape_egress_finding(measured_up)["no_egress"] is False
    assert scrape_egress_finding(measured_up)["upload_source"] == "cloudflare"


@pytest.mark.parametrize(
    "net",
    [
        {},
        {"download_speed": None, "upload_speed": None},
        {"download_speed": None, "measurements": {}},
    ],
    ids=["empty_block", "no_measurements", "empty_measurements"],
)
def test_a_scrape_that_ran_no_speed_test_is_no_reading(net):
    """Regression: lium-io#1419 (DAH-2774) removes the scrape's speed tests and leaves `network: {}`,
    which read as neither direction measured: every idle node NO_OUTBOUND_INTERNET."""
    assert scrape_egress_finding({"network": net}) is None


@pytest.mark.asyncio
async def test_a_scrape_without_speed_tests_is_logged_unmeasured_and_passes():
    """Regression: lium-io#1419's `network: {}` fails every idle node, or is logged as verified."""
    ctx, _ = make_ctx({})
    with flags():
        res = await OutboundInternetCheck().run(ctx)
    assert res.passed and res.event.reason_code == Msg.OUTBOUND_INTERNET_UNMEASURED.reason
    assert res.event.what_we_saw["scrape"] is None


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
    dry = [c.check_id for c in PipelineFactory.build_dry_run_checks()]
    assert OutboundInternetCheck.check_id in dry
