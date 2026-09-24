"""REGISTRY_PULL_FAILED: an idle node whose Docker daemon cannot pull a Docker Hub image through its own registry path.

ticket-0361 root cause (Muhammad, #loop-muhammad 23 Sep 18:46Z): 14e704ba failed 16 rents in 24 h, every one of
them a template image the node did not have cached. Its dockerd pulls through the registry mirror
docker.m.daocloud.io, whose DNS lookup times out; cached templates started fine. No check pulled anything, so the
node passed every one.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import subprocess
from collections import Counter
from contextlib import contextmanager, nullcontext
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod

from services.task.checks import registry_pull as module
from services.task.checks.registry_pull import (
    REGISTRY_PULL_IMAGE,
    REGISTRY_PULL_SCRIPT,
    RegistryPullCheck,
    classify_pull_error,
    parse_pull_probe,
)
from services.task.messages import RegistryPullMessages as Msg
from services.task.pipeline_factory import PipelineFactory
from services.task.runner import SSHCommandResult
from tests.helpers import build_services, build_state, default_executor, make_context

EXECUTOR = default_executor()
MIRROR = '["https://docker.m.daocloud.io/"]'
INFO = f"lium_pull driver=overlayfs mirrors={MIRROR}"
# Docker 29.1.3 (containerd image store) with that mirror configured and its DNS answers dropped: the 2.4 KB
# pull took 51 s on one run and over 60 s on another, and the 30 s bound ended it on both runs after; with the resolver refusing, the pull failed at once
MIRROR_DNS_TIMEOUT = f"{INFO}\nlium_pull cached=no cache=removed\nlium_pull exit=124 seconds=30\n"
MIRROR_DNS_REFUSED = (
    f"{INFO}\nlium_pull cached=no cache=removed\nlium_pull exit=1 seconds=0\n"
    f'Error response from daemon: failed to resolve reference "{REGISTRY_PULL_IMAGE}": failed to do request: '
    'Head "https://docker.m.daocloud.io/v2/library/hello-world/manifests/sha256:5e2309?ns=docker.io": dial tcp: '
    "lookup docker.m.daocloud.io on 127.0.0.53:53: read udp 127.0.0.1:43592->127.0.0.53:53: read: connection refused\n"
)
PULL_OK = f"{INFO}\nlium_pull cached=no cache=removed\nlium_pull exit=0 seconds=2\n{REGISTRY_PULL_IMAGE}\n"
RATE_LIMITED = (
    "lium_pull driver=overlayfs mirrors=[]\nlium_pull cached=no cache=removed\nlium_pull exit=1 seconds=1\n"
    "Error response from daemon: toomanyrequests: You have reached your unauthenticated pull rate limit. "
    "https://www.docker.com/increase-rate-limit\n"
)
# Docker 29.8.1, 23 Sep 2026, a REJECT rule in OUTPUT for every address of registry-1.docker.io (or the daemon's
# proxy), the committed hello-world digest pulled; the text is the daemon's own, verbatim
_HEAD = (
    f'Error response from daemon: failed to resolve reference "{REGISTRY_PULL_IMAGE}": failed to do request: '
    'Head "https://registry-1.docker.io/v2/library/hello-world/manifests/sha256:5e23090353324d887c48ad5e5c56d294eab81588df9605b07d1afe895f9cc8f8": '
)
REJECTED_TEXTS = {
    # containerd image store: --reject-with tcp-reset, icmp-host-unreachable, icmp-net-unreachable
    "tcp-reset": _HEAD + "dial tcp 184.193.200.226:443: connect: connection refused",
    "icmp-host-unreachable": _HEAD + "dial tcp 34.238.55.159:443: connect: no route to host",
    "icmp-net-unreachable": _HEAD + "dial tcp 52.45.63.121:443: connect: network is unreachable",
    # daemon.json "proxies": a proxy port nothing listens on, and a proxy host behind icmp-host-unreachable
    "proxy refused": _HEAD
    + "proxyconnect tcp: dial tcp 127.0.0.1:3999: connect: connection refused",
    "proxy no route": _HEAD
    + "proxyconnect tcp: dial tcp 192.0.2.10:3128: connect: no route to host",
    # the classic overlay2 store words it from the registry root
    "classic tcp-reset": 'Error response from daemon: Get "https://registry-1.docker.io/v2/": dial tcp 98.87.240.167:443: '
    "connect: connection refused",
    "classic icmp-net-unreachable": 'Error response from daemon: Get "https://registry-1.docker.io/v2/": dial tcp '
    "184.193.200.226:443: connect: network is unreachable",
}
REJECTED = (
    f"lium_pull driver=overlayfs mirrors=[]\nlium_pull cached=no cache=removed\nlium_pull exit=1 seconds=0\n"
    f"{REJECTED_TEXTS['tcp-reset']}\n"
)
HUB_UP = (True, "HTTP 401")
HUB_DOWN = (False, "ClientConnectorError: Cannot connect to host registry-1.docker.io:443")


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
        self.timeouts: list[int] = []

    async def run(self, cmd, *, timeout=60, check=False, retryable=True, stdin_text=None):
        self.commands.append(cmd)
        self.timeouts.append(timeout)
        return self.answers.pop(0) if self.answers else result(PULL_OK)


class FakeRedis:
    def __init__(self, *, unreadable: bool = False):
        self.store: dict[str, str] = {}
        self.ttl: dict[str, int | None] = {}
        self.unreadable = unreadable

    async def get(self, key):
        if self.unreadable:
            raise ConnectionError("redis down")
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value
        self.ttl[key] = ex

    async def delete(self, key):
        self.store.pop(key, None)


class FakeHub:
    """Stands in for probe_docker_hub: the validator's own GET of registry-1.docker.io/v2/."""

    def __init__(self, answer=HUB_UP):
        self.answer = answer
        self.calls = 0

    async def __call__(self):
        self.calls += 1
        return self.answer


class Clock:
    def __init__(self):
        self.now = 1_790_000_000.0

    def time(self) -> float:
        return self.now


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
    *answers: SSHCommandResult,
    redis: FakeRedis | None = None,
    rented_data=None,
    uuid: str | None = None,
):
    runner = FakeRunner(*answers)
    redis = redis if redis is not None else FakeRedis()
    ctx = make_context(
        executor=EXECUTOR.model_copy(update={"uuid": uuid}) if uuid else None,
        state=build_state(rented_data=rented_data),
        runner=runner,
        services=build_services(redis=redis),
    )
    return ctx, runner, redis


@contextmanager
def flags(
    *,
    check: bool = True,
    enforced: bool = True,
    interval_hours: float = 6.0,
    retry_minutes: float = 30.0,
    hub: FakeHub | None = None,
    phase_at_start: bool = True,
):
    """`phase_at_start` puts each node's phase where the clock stood when the node was first seen, so it pulls
    at once and every interval after; the fleet simulations turn it off and use each uuid's real phase."""
    fake = SimpleNamespace(
        REGISTRY_PULL_CHECK_ENABLED=check,
        REGISTRY_PULL_ENFORCEMENT_ENABLED=enforced,
        REGISTRY_PULL_PROBE_INTERVAL_HOURS=interval_hours,
        REGISTRY_PULL_PROBE_RETRY_MINUTES=retry_minutes,
    )
    clock = Clock()
    phases: dict[str, float] = {}
    phase = patch.object(
        module,
        "pull_phase_seconds",
        lambda uuid: phases.setdefault(uuid, clock.now % (interval_hours * 3600)),
    )
    with (
        patch.object(module, "settings", fake),
        patch.object(module, "time", clock),
        patch.object(module, "probe_docker_hub", hub or FakeHub()),
        phase if phase_at_start else nullcontext(),
    ):
        yield clock


def test_defaults_log_only_and_bounded():
    """Regression: enforcement ships on before the 48 h log-only review, or the pull runs every 15-minute cycle."""
    fields = type(module.settings).model_fields
    assert fields["REGISTRY_PULL_CHECK_ENABLED"].default is True
    assert fields["REGISTRY_PULL_ENFORCEMENT_ENABLED"].default is False
    assert fields["REGISTRY_PULL_PROBE_INTERVAL_HOURS"].default == 6.0
    assert fields["REGISTRY_PULL_PROBE_RETRY_MINUTES"].default == 30.0
    assert not [name for name in fields if name.startswith("REGISTRY_PULL_FLEET")]


def test_the_image_is_a_tiny_official_image_pinned_by_digest():
    """A tag can move or vanish upstream and turn into a fleet-wide manifest_unknown; a digest names fixed bytes."""
    name, _, digest = REGISTRY_PULL_IMAGE.partition("@")
    assert name == "docker.io/library/hello-world"
    assert digest.startswith("sha256:") and len(digest) == len("sha256:") + 64


@pytest.mark.parametrize(
    "exit_code,output,outcome",
    [
        (0, "", "ok"),
        (124, "", "timeout"),
        (137, "", "timeout"),
        (
            1,
            "Error response from daemon: Get https://registry-1.docker.io/v2/: net/http: TLS handshake timeout",
            "timeout",
        ),
        (
            1,
            "dial tcp: lookup docker.m.daocloud.io on 127.0.0.53:53: read udp 127.0.0.1:1->127.0.0.53:53: i/o timeout",
            "dns_error",
        ),
        (1, MIRROR_DNS_REFUSED.splitlines()[-1], "dns_error"),
        (1, "dial tcp: lookup registry-1.docker.io: no such host", "dns_error"),
        (1, RATE_LIMITED.splitlines()[-1], "rate_limited"),
        (
            1,
            "Error response from daemon: unexpected status code 429 Too Many Requests",
            "rate_limited",
        ),
        (1, "Error response from daemon: manifest unknown: manifest unknown", "manifest_unknown"),
        # Docker 29 (containerd image store) against Docker Hub, a digest it does not have
        (
            1,
            'Error response from daemon: failed to resolve reference "docker.io/library/hello-world@sha256:00": '
            "docker.io/library/hello-world@sha256:00: not found",
            "manifest_unknown",
        ),
        (1, "Error response from daemon: unauthorized: authentication required", "auth_error"),
        # a mirror that answers 503 for a blob, seen from docker.m.daocloud.io; the digest's hex holds a 429
        (
            1,
            "unexpected status from HEAD request to https://docker.m.daocloud.io/v2/library/hello-world/blobs/"
            "sha256:ab429cd: 503 Service Unavailable",
            "other",
        ),
        (1, "Cannot connect to the Docker daemon at unix:///var/run/docker.sock", "other"),
        # the CLI failing to reach dockerd's socket, not the registry: Docker 29.8.1's text with the daemon stopped,
        # and the refused variant
        (
            1,
            "failed to connect to the docker API at unix:///var/run/docker.sock; check if the path is correct and "
            "if the daemon is running: dial unix /var/run/docker.sock: connect: no such file or directory",
            "other",
        ),
        (1, "dial unix /var/run/docker.sock: connect: connection refused", "other"),
        # a resolver the host cannot reach is still a DNS failure
        (
            1,
            "dial tcp: lookup registry-1.docker.io on 10.0.0.2:53: dial udp 10.0.0.2:53: connect: network is unreachable",
            "dns_error",
        ),
        # a connection reset mid-transfer is not a rejected connect: no verdict, as before
        (1, "read tcp 10.0.0.5:51234->44.194.203.49:443: read: connection reset by peer", "other"),
    ],
)
def test_the_pull_error_is_classified(exit_code, output, outcome):
    assert classify_pull_error(exit_code, output) == outcome


@pytest.mark.parametrize("name", sorted(REJECTED_TEXTS))
def test_a_rejected_connection_to_docker_hub_or_its_proxy_is_unreachable_and_fails(name):
    """Regression (r4 M1): a firewall REJECT of Docker Hub read as `other`, no verdict, so a node failing every
    uncached rent never failed. Texts are Docker 29.8.1's own, under each REJECT flavour and store."""
    assert classify_pull_error(1, REJECTED_TEXTS[name]) == "unreachable"
    assert "unreachable" in module.FAILING_OUTCOMES


@pytest.mark.asyncio
async def test_a_node_whose_firewall_rejects_docker_hub_fails_twice_in_a_row():
    ctx, _, _ = make_ctx(result(REJECTED), result(REJECTED))
    check = RegistryPullCheck()
    with flags() as clock:
        first = await check.run(ctx)
        clock.now += 30 * 60
        second = await check.run(ctx)
    assert first.passed and first.event.reason_code == Msg.REGISTRY_PULL_FAILED_ONCE.reason
    assert second.passed is False and second.event.reason_code == Msg.REGISTRY_PULL_FAILED.reason
    assert second.event.what_we_saw["pull"]["outcome"] == "unreachable"
    assert "connect: connection refused" in second.event.what_we_saw["pull"]["detail"]


@pytest.mark.parametrize(
    "line,driver,store",
    [
        (f"lium_pull driver=overlayfs mirrors={MIRROR}", "overlayfs", "containerd"),
        (f"lium_pull driver=overlay2 mirrors={MIRROR}", "overlay2", "classic"),
        (f"lium_pull driver=zfs mirrors={MIRROR}", "zfs", None),
        # Docker 29.8.1's CLI with the daemon down still prints the format's literal text
        ("lium_pull driver= mirrors=", None, None),
        ("lium_pull mirrors=unknown", None, None),
    ],
)
def test_the_image_store_is_read_off_the_first_line(line, driver, store):
    reading = parse_pull_probe(PULL_OK.replace(INFO, line))
    assert (reading.driver, reading.image_store) == (driver, store)
    record = reading.as_record()
    assert (record["driver"], record["image_store"]) == (driver, store)
    assert record["seconds"] == 2


def test_a_daemon_the_cli_cannot_reach_leaves_the_mirrors_unknown():
    assert parse_pull_probe(PULL_OK.replace(INFO, "lium_pull driver= mirrors=")).mirrors is None


def test_the_event_records_the_mirrors_the_daemon_has():
    reading = parse_pull_probe(MIRROR_DNS_REFUSED)
    assert reading.outcome == "dns_error"
    assert reading.mirrors == ["https://docker.m.daocloud.io/"]
    assert "lookup docker.m.daocloud.io" in reading.detail
    assert parse_pull_probe(PULL_OK.replace(MIRROR, "[]")).mirrors == []
    assert parse_pull_probe(PULL_OK.replace(MIRROR, "null")).mirrors == []
    assert parse_pull_probe(PULL_OK.replace(MIRROR, "unknown")).mirrors is None


def test_an_image_still_cached_after_rmi_is_no_pull():
    reading = parse_pull_probe(
        f"lium_pull mirrors={MIRROR}\nlium_pull cached=yes cache=still_present\n"
    )
    assert reading.outcome == "not_run" and reading.cached_before is True


# ------------------------------------------------------------------ the script under sh, docker stubbed


def _stub(directory, name: str, body: str) -> None:
    path = os.path.join(directory, name)
    with open(path, "w") as handle:
        handle.write(f"#!/bin/sh\n{body}\n")
    os.chmod(path, 0o755)


def _docker_stub(tmp_path, *, pull: str, rmi_removes: bool = True, cached: bool = True) -> str:
    """A docker CLI whose image store is one file: `image inspect` finds it, `rmi` deletes it, `pull` makes it."""
    store = tmp_path / "hello-world.present"
    if cached:
        store.write_text("x")
    log = tmp_path / "docker.log"
    rmi = f"rm -f {store}" if rmi_removes else "true"
    return (
        f'echo "$*" >> {log}\n'
        'case "$1" in\n'
        f"  info) echo 'driver=overlay2 mirrors={MIRROR}' ;;\n"
        f"  image) [ -e {store} ] ;;\n"
        f"  rmi) {rmi} ;;\n"
        f"  pull) {pull} ;;\n"
        "esac"
    )


def _run_script(
    tmp_path, docker_body: str, extra: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("tail", "head", "date", "cat", "rm", "touch", "sleep"):
        os.symlink(shutil.which(name), bin_dir / name)
    _stub(bin_dir, "docker", docker_body)
    for name, body in (extra or {}).items():
        _stub(bin_dir, name, body)
    script = REGISTRY_PULL_SCRIPT.replace("d=/usr/bin/docker", f"d={bin_dir}/docker")
    return subprocess.run(
        [shutil.which("sh"), "-c", script],
        env={"PATH": str(bin_dir)},
        capture_output=True,
        text=True,
        timeout=60,
        cwd=tmp_path,
    )


def test_the_cached_image_is_removed_before_the_pull_and_after_it(tmp_path):
    """Regression: a node that already has the image answers the pull from its own store, so a broken mirror
    passes (14e704ba's cached templates started fine)."""
    store = tmp_path / "hello-world.present"
    out = _run_script(tmp_path, _docker_stub(tmp_path, pull=f"touch {store}; echo pulled"))
    reading = parse_pull_probe(out.stdout)
    assert out.returncode == 0
    assert (reading.outcome, reading.cached_before) == ("ok", True)
    calls = [line.split()[0] for line in (tmp_path / "docker.log").read_text().splitlines()]
    assert calls == ["info", "image", "rmi", "image", "pull", "rmi"]
    pull_args = (tmp_path / "docker.log").read_text().splitlines()[4].split()
    assert pull_args[-1] == REGISTRY_PULL_IMAGE
    assert not store.exists()
    assert (reading.driver, reading.image_store) == ("overlay2", "classic")
    info_args = (tmp_path / "docker.log").read_text().splitlines()[0]
    assert "driver={{.Driver}} mirrors={{json .RegistryConfig.Mirrors}}" in info_args


def test_a_docker_info_that_prints_nothing_leaves_store_and_mirrors_unknown(tmp_path):
    body = _docker_stub(tmp_path, pull="echo pulled", cached=False).replace(
        "  info) echo 'driver=overlay2 mirrors=", "  info) exit 1; echo '"
    )
    out = _run_script(tmp_path, body)
    reading = parse_pull_probe(out.stdout)
    assert out.stdout.splitlines()[0] == "lium_pull mirrors=unknown"
    assert (reading.outcome, reading.driver, reading.mirrors) == ("ok", None, None)


def test_an_image_rmi_cannot_remove_is_not_pulled(tmp_path):
    out = _run_script(tmp_path, _docker_stub(tmp_path, pull="echo pulled", rmi_removes=False))
    assert parse_pull_probe(out.stdout).outcome == "not_run"
    assert "pull" not in (tmp_path / "docker.log").read_text()


def test_the_pull_runs_under_a_30_s_total_bound(tmp_path):
    out = _run_script(
        tmp_path,
        _docker_stub(tmp_path, pull="echo pulled", cached=False),
        extra={"timeout": f'echo "$*" >> {tmp_path}/timeout.log; shift 3; exec "$@"'},
    )
    assert parse_pull_probe(out.stdout).outcome == "ok"
    bounds = (tmp_path / "timeout.log").read_text().splitlines()
    pull_bound = next(line for line in bounds if " pull " in line)
    assert pull_bound.split()[:3] == ["-k", "5", "30"]
    assert all(line.startswith("-k 5 ") for line in bounds)


def test_a_pull_that_hangs_is_cut_off_and_reads_as_timeout(tmp_path):
    """The bound is real: a pull still waiting on the mirror's lookup is killed and the outcome is timeout.
    The stubbed `timeout` shortens the 30 s to 1 s."""
    out = _run_script(
        tmp_path,
        _docker_stub(tmp_path, pull="exec sleep 30", cached=False),
        extra={"timeout": f'shift 3; exec {shutil.which("timeout")} 1 "$@"'},
    )
    reading = parse_pull_probe(out.stdout)
    assert (reading.outcome, reading.exit_code) == ("timeout", 124)
    assert reading.mirrors == ["https://docker.m.daocloud.io/"]


def test_the_script_runs_without_timeout_on_the_path(tmp_path):
    out = _run_script(tmp_path, _docker_stub(tmp_path, pull="echo pulled", cached=False))
    assert parse_pull_probe(out.stdout).outcome == "ok"


# ------------------------------------------------------------------ the check


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "enforced,passed,reason",
    [
        (True, False, Msg.REGISTRY_PULL_FAILED.reason),
        (False, True, Msg.REGISTRY_PULL_FAILED_OBSERVED.reason),
    ],
)
async def test_14e704ba_mirror_dns_timeout_twice_in_a_row(enforced, passed, reason):
    """Regression: 14e704ba's dockerd pulls through docker.m.daocloud.io, whose lookup times out; 16 rents of
    uncached templates failed in 24 h while the node passed every check."""
    ctx, runner, _ = make_ctx(result(MIRROR_DNS_TIMEOUT), result(MIRROR_DNS_TIMEOUT))
    check = RegistryPullCheck()
    with flags(enforced=enforced) as clock:
        first = await check.run(ctx)
        clock.now += 30 * 60
        second = await check.run(ctx)
    assert first.passed and first.event.reason_code == Msg.REGISTRY_PULL_FAILED_ONCE.reason
    assert second.passed is passed and second.event.reason_code == reason
    what = second.event.what_we_saw
    assert what["failures_in_a_row"] == 2 and what["probed_this_cycle"] is True
    assert what["pull"]["outcome"] == "timeout"
    assert what["pull"]["mirrors"] == ["https://docker.m.daocloud.io/"]
    assert what["pull"]["image"] == REGISTRY_PULL_IMAGE
    assert len(runner.commands) == 2


@pytest.mark.asyncio
async def test_a_mirror_dns_error_gives_registry_pull_failed_with_the_remediation():
    ctx, _, _ = make_ctx(result(MIRROR_DNS_REFUSED), result(MIRROR_DNS_REFUSED))
    check = RegistryPullCheck()
    with flags() as clock:
        await check.run(ctx)
        clock.now += 30 * 60
        res = await check.run(ctx)
    assert res.passed is False and res.event.reason_code == "REGISTRY_PULL_FAILED"
    assert res.event.what_we_saw["pull"]["outcome"] == "dns_error"
    assert "daemon.json" in res.event.remediation


@pytest.mark.asyncio
async def test_a_429_is_unmeasured_and_never_a_failure():
    """Regression: a provider's nodes behind one IP spend Docker Hub's anonymous quota, and the 429 fails them."""
    ctx, _, _ = make_ctx(result(RATE_LIMITED), result(RATE_LIMITED))
    check = RegistryPullCheck()
    with flags(interval_hours=0.1) as clock:
        first = await check.run(ctx)
        clock.now += 3600
        second = await check.run(ctx)
    for res in (first, second):
        assert res.passed and res.event.reason_code == Msg.REGISTRY_PULL_UNMEASURED.reason
        assert res.event.what_we_saw["pull"]["outcome"] == "rate_limited"
        assert res.event.what_we_saw["failures_in_a_row"] == 0


@pytest.mark.asyncio
async def test_a_429_between_two_failures_neither_counts_nor_resets():
    ctx, _, _ = make_ctx(
        result(MIRROR_DNS_TIMEOUT), result(RATE_LIMITED), result(MIRROR_DNS_TIMEOUT)
    )
    check = RegistryPullCheck()
    with flags(interval_hours=0.1) as clock:
        await check.run(ctx)
        clock.now += 3600
        between = await check.run(ctx)
        clock.now += 3600
        res = await check.run(ctx)
    assert between.passed and between.event.what_we_saw["failures_in_a_row"] == 1
    assert res.passed is False and res.event.reason_code == Msg.REGISTRY_PULL_FAILED.reason


@pytest.mark.asyncio
async def test_a_pull_that_works_passes_and_resets_the_count():
    ctx, _, redis = make_ctx(result(MIRROR_DNS_TIMEOUT), result(PULL_OK))
    check = RegistryPullCheck()
    with flags() as clock:
        await check.run(ctx)
        clock.now += 30 * 60
        res = await check.run(ctx)
    assert res.passed and res.event.reason_code == Msg.REGISTRY_PULL_OK.reason
    assert res.event.what_we_saw["failures_in_a_row"] == 0
    assert res.event.what_we_saw["pull"]["outcome"] == "ok"
    assert redis.ttl[f"registry_pull_probe:{EXECUTOR.uuid}"] == 7 * 24 * 3600


@pytest.mark.asyncio
async def test_one_failed_pull_is_not_a_finding():
    ctx, _, _ = make_ctx(result(MIRROR_DNS_TIMEOUT))
    with flags():
        res = await RegistryPullCheck().run(ctx)
    assert res.passed and res.event.reason_code == Msg.REGISTRY_PULL_FAILED_ONCE.reason
    assert res.event.what_we_saw["failures_in_a_row"] == 1


@pytest.mark.asyncio
async def test_a_rented_node_is_never_probed():
    """Regression: an image is removed and pulled next to a renter's pod."""
    ctx, runner, redis = make_ctx(rented_data=rented())
    with flags():
        res = await RegistryPullCheck().run(ctx)
    assert res.passed and res.event.reason_code == Msg.SKIPPED.reason
    assert res.event.what_we_saw["reason"] == "rented"
    assert runner.commands == [] and redis.store == {}


@pytest.mark.asyncio
async def test_after_a_pull_that_worked_the_next_one_waits_for_the_interval():
    """Regression: the pull runs every 15-minute cycle on every idle node (96 Docker Hub pulls a day each)."""
    ctx, runner, _ = make_ctx(result(PULL_OK), result(PULL_OK))
    check = RegistryPullCheck()
    with flags() as clock:
        await check.run(ctx)
        results = []
        for _ in range(23):
            clock.now += 15 * 60
            results.append(await check.run(ctx))
        assert len(runner.commands) == 1
        assert all(r.passed and r.event.reason_code == Msg.SKIPPED.reason for r in results)
        assert results[0].event.what_we_saw["reason"] == "not due"
        clock.now += 15 * 60
        await check.run(ctx)
    assert len(runner.commands) == 2


@pytest.mark.asyncio
async def test_after_a_failed_pull_the_next_one_waits_for_the_retry():
    ctx, runner, _ = make_ctx(result(MIRROR_DNS_TIMEOUT), result(MIRROR_DNS_TIMEOUT))
    check = RegistryPullCheck()
    with flags() as clock:
        await check.run(ctx)
        clock.now += 15 * 60
        early = await check.run(ctx)
        assert len(runner.commands) == 1
        assert early.passed and early.event.reason_code == Msg.SKIPPED.reason
        clock.now += 15 * 60
        await check.run(ctx)
    assert len(runner.commands) == 2


@pytest.mark.asyncio
async def test_a_standing_failure_holds_between_pulls():
    """Regression: with enforcement on, a node that failed twice passes the cycles in between pulls."""
    ctx, runner, _ = make_ctx(result(MIRROR_DNS_TIMEOUT), result(MIRROR_DNS_TIMEOUT))
    check = RegistryPullCheck()
    with flags() as clock:
        await check.run(ctx)
        clock.now += 30 * 60
        await check.run(ctx)
        clock.now += 15 * 60
        res = await check.run(ctx)
    assert len(runner.commands) == 2
    assert res.passed is False and res.event.reason_code == Msg.REGISTRY_PULL_FAILED.reason
    assert res.event.what_we_saw["probed_this_cycle"] is False


@pytest.mark.asyncio
async def test_redis_unreadable_pulls_nothing():
    """Without the last reading the interval cannot hold; pulling every cycle would spend the node's quota."""
    ctx, runner, _ = make_ctx(redis=FakeRedis(unreadable=True))
    with flags():
        res = await RegistryPullCheck().run(ctx)
    assert res.passed and res.event.reason_code == Msg.SKIPPED.reason
    assert runner.commands == []


@pytest.mark.asyncio
async def test_a_probe_that_did_not_run_is_no_verdict():
    ctx, runner, _ = make_ctx(result(error_type="timeout", exit_code=-1))
    with flags():
        res = await RegistryPullCheck().run(ctx)
    assert res.passed and res.event.reason_code == Msg.REGISTRY_PULL_UNMEASURED.reason
    assert res.event.what_we_saw["pull"]["outcome"] == "not_run"
    assert res.event.what_we_saw["failures_in_a_row"] == 0
    assert runner.timeouts == [module.REGISTRY_PULL_COMMAND_TIMEOUT_SECONDS]


@pytest.mark.asyncio
async def test_check_off_pulls_nothing():
    ctx, runner, _ = make_ctx()
    with flags(check=False):
        res = await RegistryPullCheck().run(ctx)
    assert res.passed and res.event.reason_code == Msg.SKIPPED.reason
    assert runner.commands == []


@pytest.mark.asyncio
async def test_the_command_is_the_script_under_sh():
    ctx, runner, _ = make_ctx()
    with flags():
        await RegistryPullCheck().run(ctx)
    assert runner.commands[0].startswith("/bin/sh -c ")
    assert REGISTRY_PULL_IMAGE in runner.commands[0]


def test_pipeline_runs_the_check_after_the_scrape_rule_and_before_the_rented_halt():
    ids = [check.check_id for check in PipelineFactory.build_checks()]
    index = ids.index(RegistryPullCheck.check_id)
    assert ids[index - 1] == "executor.validate.outbound_internet"
    assert index < ids.index("executor.validate.rented_state")
    dry = [c.check_id for c in PipelineFactory.build_dry_run_checks()]
    assert RegistryPullCheck.check_id not in dry


@pytest.mark.asyncio
async def test_an_open_streak_is_re_pulled_on_the_retry_after_a_429():
    """Regression: a 429 after a failure pushes the next pull out by the 6 h interval, so an enforced node
    fails for hours on a stale reading, or a fixed one waits hours to recover."""
    ctx, runner, _ = make_ctx(result(MIRROR_DNS_TIMEOUT), result(RATE_LIMITED), result(PULL_OK))
    check = RegistryPullCheck()
    with flags() as clock:
        await check.run(ctx)
        clock.now += 30 * 60
        await check.run(ctx)
        clock.now += 30 * 60
        res = await check.run(ctx)
    assert len(runner.commands) == 3
    assert res.passed and res.event.reason_code == Msg.REGISTRY_PULL_OK.reason


# ------------------------------------------------------------------ the Docker Hub control

HUB_TLS_TIMEOUT = (
    "lium_pull driver=overlayfs mirrors=[]\nlium_pull cached=no cache=removed\nlium_pull exit=1 seconds=10\n"
    'Error response from daemon: Get "https://registry-1.docker.io/v2/": net/http: TLS handshake timeout\n'
)


def _logged(caplog, reason: str) -> int:
    return sum(reason in record.getMessage() for record in caplog.records)


async def _confirming(check, ctx, clock: Clock):
    """The node's first failed pull, then its retry 30 minutes on: (first, confirming)."""
    first = await check.run(ctx)
    clock.now += 30 * 60
    return first, await check.run(ctx)


@pytest.mark.asyncio
async def test_a_failed_pull_while_docker_hub_is_down_from_the_validator_is_no_verdict():
    """Regression (r4 M2): with enforcement on, a Docker Hub outage that reads as a TLS handshake timeout failed
    every mirror-less idle node within about 30 minutes."""
    ctx, _, _ = make_ctx(result(HUB_TLS_TIMEOUT), result(HUB_TLS_TIMEOUT))
    hub = FakeHub(HUB_DOWN)
    check = RegistryPullCheck()
    with flags(hub=hub, interval_hours=0.1) as clock:
        first = await check.run(ctx)
        clock.now += 3600
        second = await check.run(ctx)
    for res in (first, second):
        assert res.passed and res.event.reason_code == Msg.REGISTRY_PULL_NO_VERDICT_HUB_DOWN.reason
        what = res.event.what_we_saw
        assert what["failures_in_a_row"] == 0 and what["no_verdict"] == "docker_hub_down"
        assert what["pull"]["outcome"] == "timeout"
        assert what["guard"]["docker_hub_control"]["reachable"] is False
    assert hub.calls == 2


@pytest.mark.asyncio
async def test_the_control_is_fetched_once_per_window_for_every_node(caplog):
    redis = FakeRedis()
    hub = FakeHub(HUB_DOWN)
    check = RegistryPullCheck()
    results = []
    with flags(hub=hub) as clock, caplog.at_level(logging.WARNING):
        for i in range(5):
            ctx, _, _ = make_ctx(result(HUB_TLS_TIMEOUT), redis=redis, uuid=f"node-{i}")
            results.append(await check.run(ctx))
            clock.now += 30
        assert hub.calls == 1 and _logged(caplog, "REGISTRY_PULL_NO_VERDICT_HUB_DOWN") == 1
        clock.now += module.HUB_CONTROL_TTL_SECONDS
        ctx, _, _ = make_ctx(result(HUB_TLS_TIMEOUT), redis=redis, uuid="node-late")
        results.append(await check.run(ctx))
    assert hub.calls == 2 and _logged(caplog, "REGISTRY_PULL_NO_VERDICT_HUB_DOWN") == 2
    assert {r.event.reason_code for r in results} == {Msg.REGISTRY_PULL_NO_VERDICT_HUB_DOWN.reason}
    assert redis.ttl["registry_pull_hub_control"] == module.HUB_CONTROL_TTL_SECONDS == 300


@pytest.mark.asyncio
async def test_with_docker_hub_up_a_node_whose_mirror_fails_still_counts():
    """The control guards against Docker Hub's outages, not the node's own mirror: 14e704ba still fails."""
    ctx, _, _ = make_ctx(result(MIRROR_DNS_TIMEOUT), result(MIRROR_DNS_TIMEOUT))
    hub = FakeHub(HUB_UP)
    check = RegistryPullCheck()
    with flags(hub=hub) as clock:
        first, second = await _confirming(check, ctx, clock)
    assert first.event.reason_code == Msg.REGISTRY_PULL_FAILED_ONCE.reason
    assert second.passed is False and second.event.reason_code == Msg.REGISTRY_PULL_FAILED.reason
    guard = second.event.what_we_saw["guard"]
    assert guard == {"docker_hub_control": {"at": clock.now, "reachable": True, "seen": "HTTP 401"}}
    assert "no_verdict" not in second.event.what_we_saw
    # 30 minutes apart: each failed pull has a control reading of its own
    assert hub.calls == 2


@pytest.mark.asyncio
async def test_a_pull_that_works_never_fetches_the_control():
    ctx, _, _ = make_ctx(result(PULL_OK))
    hub = FakeHub(HUB_DOWN)
    with flags(hub=hub):
        res = await RegistryPullCheck().run(ctx)
    assert res.event.reason_code == Msg.REGISTRY_PULL_OK.reason and hub.calls == 0


async def _serve(handler):
    from aiohttp import web

    app = web.Application()
    app.router.add_get("/v2/", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, site._server.sockets[0].getsockname()[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("status,reachable", [(401, True), (200, True), (503, False), (429, False)])
async def test_the_control_reads_401_or_200_as_reachable(status, reachable):
    from aiohttp import web

    async def handler(request):
        return web.Response(status=status)

    runner, port = await _serve(handler)
    try:
        with patch.object(module, "DOCKER_HUB_CONTROL_URL", f"http://127.0.0.1:{port}/v2/"):
            assert await module.probe_docker_hub() == (reachable, f"HTTP {status}")
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_the_control_reads_a_refused_connection_as_unreachable():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with patch.object(module, "DOCKER_HUB_CONTROL_URL", f"http://127.0.0.1:{port}/v2/"):
        reachable, seen = await module.probe_docker_hub()
    assert reachable is False and seen.startswith("ClientConnectorError")


@pytest.mark.asyncio
async def test_docker_hub_down_in_the_middle_of_a_streak_neither_counts_nor_resets_it():
    ctx, _, _ = make_ctx(*(result(HUB_TLS_TIMEOUT) for _ in range(3)))
    hub = FakeHub(HUB_UP)
    check = RegistryPullCheck()
    with flags(hub=hub) as clock:
        first = await check.run(ctx)
        hub.answer = HUB_DOWN
        clock.now += 30 * 60
        down = await check.run(ctx)
        hub.answer = HUB_UP
        clock.now += 30 * 60
        confirmed = await check.run(ctx)
    assert first.event.reason_code == Msg.REGISTRY_PULL_FAILED_ONCE.reason
    assert down.passed and down.event.reason_code == Msg.REGISTRY_PULL_NO_VERDICT_HUB_DOWN.reason
    assert down.event.what_we_saw["failures_in_a_row"] == 1
    assert (
        confirmed.passed is False and confirmed.event.reason_code == Msg.REGISTRY_PULL_FAILED.reason
    )


# ------------------------------------------------------------------ fleets over many cycles


class OutageRunner:
    """A node's pulls: Docker Hub's TLS timeout inside [start, end), or always when `broken` (or from
    `broken_from` on); else ok."""

    def __init__(
        self,
        clock: Clock,
        start: float,
        end: float,
        *,
        broken: bool = False,
        broken_from: float | None = None,
    ):
        self.clock, self.start, self.end, self.broken = clock, start, end, broken
        self.broken_from = broken_from
        self.pulls = 0

    async def run(self, cmd, *, timeout=60, check=False, retryable=True, stdin_text=None):
        self.pulls += 1
        failing = (
            self.broken
            or (self.broken_from is not None and self.clock.now >= self.broken_from)
            or self.start <= self.clock.now < self.end
        )
        return result(HUB_TLS_TIMEOUT if failing else PULL_OK)


def _node(uuid: str, miner: str, runner, redis: FakeRedis):
    return make_context(
        executor=EXECUTOR.model_copy(update={"uuid": uuid}),
        state=build_state(),
        runner=runner,
        services=build_services(redis=redis),
        miner_hotkey=miner,
    )


async def _cycles(
    check, nodes, clock: Clock, *, hours: float, active=None
) -> dict[str, list[tuple[float, str]]]:
    """The validator's 15-minute cycles: every idle node scored one after another, spread over the cycle.
    `active(ctx)` says whether a node is still idle (a rented one is not scored)."""
    seen: dict[str, list[tuple[float, str]]] = {ctx.executor.uuid: [] for ctx in nodes}
    for _ in range(int(hours * 4)):
        idle = [ctx for ctx in nodes if active is None or active(ctx)]
        step = 15 * 60 / len(idle)
        for ctx in idle:
            res = await check.run(ctx)
            seen[ctx.executor.uuid].append((clock.now, res.event.reason_code))
            clock.now += step
    return seen


def _fleet(
    clock: Clock, redis: FakeRedis, *, miners: list[str], start: float, end: float, tag: str = ""
):
    return [
        _node(f"exec{tag}-{i:03d}", miner, OutageRunner(clock, start, end), redis)
        for i, miner in enumerate(miners)
    ]


def _zeroed(seen) -> dict[str, float]:
    return {
        uuid: next(at for at, reason in rows if reason == Msg.REGISTRY_PULL_FAILED.reason)
        for uuid, rows in seen.items()
        if any(reason == Msg.REGISTRY_PULL_FAILED.reason for _, reason in rows)
    }


def _synchronise(redis: FakeRedis, nodes, at: float) -> None:
    """Every node's last pull at `at`, as after a deploy that pulled them all in one cycle."""
    for ctx in nodes:
        redis.store[f"registry_pull_probe:{ctx.executor.uuid}"] = json.dumps(
            {"at": at, "outcome": "ok"}
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("start", ["synchronised", "fresh"])
async def test_log_only_an_outage_only_the_nodes_see_zeroes_nobody_and_shows_in_the_observed_rows(
    start,
):
    """60 idle nodes of 6 miners; six hours in, a Docker Hub CDN incident only the nodes see (the control is up)
    lasts 2 h. With enforcement off, as it ships, nobody scores 0 and the nodes that pulled twice in it are
    REGISTRY_PULL_FAILED_OBSERVED: the fleet-wide pattern the review before enforcement reads."""
    redis = FakeRedis()
    check = RegistryPullCheck()
    with flags(enforced=False, phase_at_start=False) as clock:
        t0 = clock.now
        miners = [f"miner-{i % 6}" for i in range(60)]
        nodes = _fleet(clock, redis, miners=miners, start=t0 + 6 * 3600, end=t0 + 8 * 3600)
        if start == "synchronised":
            _synchronise(redis, nodes, t0)
        seen = await _cycles(check, nodes, clock, hours=14)
    reasons = {reason for rows in seen.values() for _, reason in rows}
    assert Msg.REGISTRY_PULL_FAILED.reason not in reasons
    observed = {
        uuid
        for uuid, rows in seen.items()
        if any(reason == Msg.REGISTRY_PULL_FAILED_OBSERVED.reason for _, reason in rows)
    }
    assert len(observed) >= 10
    # the pulls spread: no node pulls more than its schedule plus the retries of one streak
    assert max(ctx.runner.pulls for ctx in nodes) <= 8


@pytest.mark.asyncio
async def test_enforced_an_outage_only_the_nodes_see_zeroes_nodes_until_their_first_pull_after_it():
    """The accepted risk, with no fleet guard: enforced, a 2 h outage only the nodes see scores 0 every node that
    pulls twice in it; each is back on its first pull after it ends. The review before enforcement is the guard."""
    redis = FakeRedis()
    check = RegistryPullCheck()
    with flags(phase_at_start=False) as clock:
        t0 = clock.now
        start, end = t0 + 6 * 3600, t0 + 8 * 3600
        miners = [f"miner-{i % 6}" for i in range(60)]
        nodes = _fleet(clock, redis, miners=miners, start=start, end=end)
        seen = await _cycles(check, nodes, clock, hours=14)
    zeroed = _zeroed(seen)
    assert zeroed and all(start <= at < end for at in zeroed.values())
    for uuid in zeroed:
        # the retry comes 30 minutes after the last failed pull, in the next 15-minute cycle
        late = [reason for at, reason in seen[uuid] if at >= end + 45 * 60]
        assert late and Msg.REGISTRY_PULL_FAILED.reason not in late, uuid


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("stragglers", [0, 2])
async def test_27_broken_nodes_of_one_provider_and_stragglers_all_fail(stragglers, seed):
    """ticket-0361's shape: 27 nodes of one provider behind a mirror whose DNS times out, in a 60-node fleet, plus
    a couple of broken nodes of other providers. Every broken node fails on its retry after its first pull at its
    phase; no healthy one does."""
    redis = FakeRedis()
    check = RegistryPullCheck()
    with flags(phase_at_start=False) as clock:
        t0 = clock.now
        broken = [
            _node(
                f"ticket-s{seed}-{i:02d}",
                "provider-14e704ba",
                OutageRunner(clock, 0, 0, broken=True),
                redis,
            )
            for i in range(27)
        ] + [
            _node(
                f"straggler-s{seed}-{i}",
                f"straggler-{i}",
                OutageRunner(clock, 0, 0, broken=True),
                redis,
            )
            for i in range(stragglers)
        ]
        healthy = [
            _node(f"exec-s{seed}-{i:02d}", f"miner-{i % 5}", OutageRunner(clock, 0, 0), redis)
            for i in range(33 - stragglers)
        ]
        seen = await _cycles(check, broken + healthy, clock, hours=10)
    zeroed = _zeroed(seen)
    assert set(zeroed) == {ctx.executor.uuid for ctx in broken}
    for ctx in broken:
        uuid = ctx.executor.uuid
        first = next(at for at, reason in seen[uuid] if reason != Msg.SKIPPED.reason)
        assert zeroed[uuid] - first <= 45 * 60, uuid
    assert max(zeroed.values()) - t0 <= 7 * 3600


def test_the_phase_is_stable_per_node_and_spreads_the_fleet_over_the_interval():
    with flags(phase_at_start=False):
        interval = 6 * 3600
        phase = module.pull_phase_seconds("14e704ba-96d8-45b0-a334-c12da67bd516")
        assert phase == module.pull_phase_seconds("14e704ba-96d8-45b0-a334-c12da67bd516")
        assert 0 <= phase < interval
        hours = Counter(int(module.pull_phase_seconds(f"exec-{i}") // 3600) for i in range(600))
    assert sorted(hours) == [0, 1, 2, 3, 4, 5]
    assert all(70 <= count <= 130 for count in hours.values()), hours


@pytest.mark.asyncio
async def test_a_node_first_seen_waits_for_its_phase_then_pulls_every_interval():
    """r5: every idle node pulled in the first cycle after deploy and every 6 h after, in the same cycle."""
    uuid = "14e704ba-96d8-45b0-a334-c12da67bd516"
    ctx, runner, _ = make_ctx(result(PULL_OK), result(PULL_OK), uuid=uuid)
    check = RegistryPullCheck()
    with flags(phase_at_start=False) as clock:
        slot = module.next_scheduled_pull(uuid, clock.now)
        first = await check.run(ctx)
        assert runner.commands == []
        assert first.event.reason_code == Msg.SKIPPED.reason
        assert first.event.what_we_saw["last"]["next_pull_at"] == slot
        clock.now = slot
        pulled = await check.run(ctx)
        after = await check.run(ctx)
    assert pulled.event.reason_code == Msg.REGISTRY_PULL_OK.reason and len(runner.commands) == 1
    assert after.event.what_we_saw["last"]["next_pull_at"] == slot + 6 * 3600
