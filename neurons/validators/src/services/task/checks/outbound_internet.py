from __future__ import annotations

import logging
import math
import shlex
import uuid
from dataclasses import dataclass, replace
from typing import Any, Literal

from core.config import settings
from core.docker_utils import ALPINE_HELPER_IMAGE
from core.utils import _m, get_extra_info
from services.rental_docker_sdk import (
    RENTAL_NETWORK_LABELS,
    RENTAL_NETWORK_NAME,
    RENTAL_NETWORK_OPTIONS,
)

from ..messages import OutboundInternetMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context

logger = logging.getLogger(__name__)

EGRESS_PROBE_HOST = "pypi.org"
# 325 bytes, asked for headers only: https://pypi.org/simple/ is 46 MB, and busybox wget's -T is a read
# timeout, so fetching it pulled the whole index from every idle node every cycle
EGRESS_PROBE_URL = "https://pypi.org/robots.txt"
EGRESS_FETCH_TIMEOUT_SECONDS = 10
EGRESS_PROBE_MARKER = "lium_egress"
EGRESS_PROBE_CONTAINER_PREFIX = "lium_egress_probe_"
# getent's resolver timeouts plus the fetch's 10 s total bound, plus a `docker run` of a 3 MB image that
# may be pulled
POD_PROBE_TIMEOUT_SECONDS = 90
_TAIL_CHARS = 300

# `getent hosts pypi.org && curl -sS -m 10 -o /dev/null -I -w '%{http_code}' https://pypi.org/robots.txt`, in
# stages so the verdict says which one failed. POSIX sh: the host-side probe runs it in alpine (busybox wget,
# no curl), the rental probe in the renter image (curl). curl's -m is a total deadline; wget has none (-T is
# per read), so it runs under `timeout` wherever the image has one. An image without getent goes straight
# to the fetch, whose own resolution then decides. Exit 0 always: the marker lines are the answer.
EGRESS_PROBE_SCRIPT = (
    "if command -v getent >/dev/null 2>&1; then "
    f"if ! getent hosts {EGRESS_PROBE_HOST} >/dev/null 2>&1; then echo '{EGRESS_PROBE_MARKER} dns=fail'; exit 0; fi; "
    f"echo '{EGRESS_PROBE_MARKER} dns=ok'; fi; "
    "if command -v curl >/dev/null 2>&1; then tool=curl; "
    f"code=$(curl -sS -m {EGRESS_FETCH_TIMEOUT_SECONDS} -o /dev/null -I -w '%{{http_code}}' {EGRESS_PROBE_URL} "
    "2>/tmp/lium_egress.err); "
    "elif command -v wget >/dev/null 2>&1; then tool=wget; "
    f"bound=; if command -v timeout >/dev/null 2>&1; then bound='timeout {EGRESS_FETCH_TIMEOUT_SECONDS}'; fi; "
    f"code=$($bound wget -S --spider -T {EGRESS_FETCH_TIMEOUT_SECONDS} {EGRESS_PROBE_URL} 2>/tmp/lium_egress.err; "
    "awk '/^ *HTTP\\//{c=$2} END{print c}' /tmp/lium_egress.err); "
    "else tool=none; fi; "
    f'echo "{EGRESS_PROBE_MARKER} tool=$tool http=${{code:-000}}"; '
    "tail -c 300 /tmp/lium_egress.err 2>/dev/null; exit 0"
)

Verdict = Literal["ok", "no_egress", "unmeasured"]


@dataclass(frozen=True)
class EgressProbe:
    """What one run of EGRESS_PROBE_SCRIPT said. `unmeasured` is no verdict: the probe itself did not run."""

    verdict: Verdict
    reason: str
    tool: str | None = None
    http_code: str | None = None
    detail: str | None = None
    # the reading a re-run replaced: a no_egress is only a verdict once a second run says it too
    first_reading: dict[str, Any] | None = None

    def as_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {"verdict": self.verdict, "reason": self.reason}
        if self.tool:
            record["tool"] = self.tool
        if self.http_code:
            record["http_code"] = self.http_code
        if self.detail:
            record["detail"] = self.detail[-_TAIL_CHARS:]
        if self.first_reading:
            record["first_reading"] = self.first_reading
        return record

    def summary(self) -> str:
        text = f"{self.reason} ({self.tool or 'no tool'}, http {self.http_code or '000'})"
        return f"{text}: {self.detail[-_TAIL_CHARS:]}" if self.detail else text


def parse_egress_probe(stdout: str) -> EgressProbe:
    fields: dict[str, str] = {}
    other_lines: list[str] = []
    for line in (stdout or "").splitlines():
        line = line.strip()
        if line.startswith(EGRESS_PROBE_MARKER):
            for token in line[len(EGRESS_PROBE_MARKER) :].split():
                key, _, value = token.partition("=")
                fields[key] = value
        elif line:
            other_lines.append(line)
    detail = "\n".join(other_lines)[-_TAIL_CHARS:] or None

    if fields.get("dns") == "fail":
        return EgressProbe("no_egress", "dns_failed", detail=detail)
    tool = fields.get("tool")
    if tool is None:
        return EgressProbe("unmeasured", "no_probe_output", detail=detail)
    if tool == "none":
        return EgressProbe("unmeasured", "no_fetch_tool", tool=tool, detail=detail)
    code = fields.get("http") or "000"
    if code.isdigit() and 100 <= int(code) <= 599:
        # any HTTP answer is egress; how fast it came is the speed rule's business
        return EgressProbe("ok", "http_response", tool=tool, http_code=code)
    return EgressProbe("no_egress", "no_http_response", tool=tool, http_code=code, detail=detail)


def pod_probe_command(container_name: str, *, sysbox: bool) -> str:
    """Run EGRESS_PROBE_SCRIPT in a container networked the way a renter's pod is.

    DockerService._build_rental_container_run_spec puts every pod on the user-defined bridge
    RENTAL_NETWORK_NAME (ICC off) with no DNS override, so it resolves through dockerd's embedded DNS
    and leaves the host through that bridge's own FORWARD/NAT rules, not docker0's, and under
    sysbox-runc when the node has it. The scrape measures from the executor container instead, which
    is why a host can report a good speed and still ship pods without internet. The network is
    created the way the rental path creates it when the host has never run a rental.
    """
    options = " ".join(f"-o {key}={value}" for key, value in RENTAL_NETWORK_OPTIONS.items())
    labels = " ".join(f"--label {key}={value}" for key, value in RENTAL_NETWORK_LABELS.items())
    runtime = "--runtime=sysbox-runc " if sysbox else ""
    return (
        f"/usr/bin/docker network inspect {RENTAL_NETWORK_NAME} >/dev/null 2>&1 || "
        f"/usr/bin/docker network create --driver bridge {options} {labels} {RENTAL_NETWORK_NAME} >/dev/null 2>&1; "
        f"/usr/bin/docker run --rm --name {shlex.quote(container_name)} --network {RENTAL_NETWORK_NAME} {runtime}"
        f"--entrypoint /bin/sh {ALPINE_HELPER_IMAGE} -c {shlex.quote(EGRESS_PROBE_SCRIPT)}"
    )


def _is_positive_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def scrape_egress_finding(specs: dict[str, Any] | None) -> dict[str, Any] | None:
    """What the scrape's speed tests say about egress; None when the scrape carried no network block.

    benchmark_network_speed keeps the first download and upload any method measured and each method's
    `network_speed_error` under `measurements`. A finding only when neither direction was measured: a
    missing download alone is not one (ticket-0361: 24 of one provider's 27 active nodes had no download,
    most of them a Cloudflare download recorded as 0, and 14e704ba's upload measured 77-105 Mbps), nor
    is an error from a method a later one measured past.
    """
    network = (specs or {}).get("network")
    if not isinstance(network, dict):
        return None
    errors: dict[str, str] = {}
    if network.get("network_speed_error"):
        errors["network"] = str(network["network_speed_error"])[:_TAIL_CHARS]
    for method, measurement in (network.get("measurements") or {}).items():
        if isinstance(measurement, dict) and measurement.get("network_speed_error"):
            errors[method] = str(measurement["network_speed_error"])[:_TAIL_CHARS]
    download = network.get("download_speed")
    upload = network.get("upload_speed")
    return {
        "no_egress": not _is_positive_number(download) and not _is_positive_number(upload),
        "download_speed": download,
        "upload_speed": upload,
        "download_source": network.get("download_source"),
        "upload_source": network.get("upload_source"),
        "speed_errors": errors,
    }


def _is_rented(ctx: Context) -> bool:
    rented_data = ctx.state.rented_data
    rented_executor = rented_data.executors.get(ctx.executor.uuid) if rented_data else None
    return rented_executor is not None and len(rented_executor.pods) > 0


class OutboundInternetCheck:
    """Fail an idle node whose containers cannot reach the internet (NO_OUTBOUND_INTERNET).

    Two readings, either one is enough: a container on the rental network (`pod_probe_command`, the
    pod's network and runtime) could not resolve pypi.org or got no HTTP answer from it, or the scrape
    measured neither a download nor an upload. The pod probe is the one that sees a pod-network-only
    fault (14e704ba: no download, 77-105 Mbps upload, 3 renters without internet). A
    slow host passes: speed stays behind FeatureFlag.VERIFYX_NETWORK_VALIDATION. A pod probe that did
    not run (docker refused it, the SSH command timed out) reaches no verdict, never fails the node and
    is logged as OUTBOUND_INTERNET_UNMEASURED, not as verified. A pod probe that says no_egress is run
    a second time (only then, so a healthy node still starts one container per cycle) and the second
    run decides.

    Under NO_OUTBOUND_INTERNET_ENFORCEMENT_ENABLED the finding fails the node the way INSUFFICIENT_PORTS
    does; without it the check logs NO_OUTBOUND_INTERNET_OBSERVED and passes. A rented node is left
    alone like PortCountCheck leaves it: no container next to the renter's pod, no verdict.
    """

    check_id = "executor.validate.outbound_internet"
    fatal = True

    def __init__(self, *, run_pod_probe: bool = True):
        # the dry-run pipeline starts no containers on the executor
        self.run_pod_probe = run_pod_probe

    async def run(self, ctx: Context) -> CheckResult:
        if not settings.NO_OUTBOUND_INTERNET_CHECK_ENABLED:
            return self._skipped(ctx, "NO_OUTBOUND_INTERNET_CHECK_ENABLED is off")

        scrape = scrape_egress_finding(ctx.state.specs)
        if _is_rented(ctx):
            return self._skipped(ctx, "rented", scrape=scrape)

        pod_probe = await self._pod_probe_twice_on_no_egress(ctx) if self.run_pod_probe else None
        what: dict[str, Any] = {
            "scrape": scrape,
            "pod_probe": pod_probe.as_record() if pod_probe else None,
            "enforced": settings.NO_OUTBOUND_INTERNET_ENFORCEMENT_ENABLED,
        }
        failed_by = []
        if scrape is not None and scrape["no_egress"]:
            failed_by.append("scrape")
        if pod_probe is not None and pod_probe.verdict == "no_egress":
            failed_by.append("pod_probe")
        if not failed_by:
            template = (
                Msg.OUTBOUND_INTERNET_UNMEASURED
                if pod_probe is not None and pod_probe.verdict == "unmeasured"
                else Msg.OUTBOUND_INTERNET_OK
            )
            event = render_message(template, ctx=ctx, check_id=self.check_id, what=what)
            return CheckResult(passed=True, event=event)

        what["failed_by"] = failed_by
        if settings.NO_OUTBOUND_INTERNET_ENFORCEMENT_ENABLED:
            event = render_message(
                Msg.NO_OUTBOUND_INTERNET, ctx=ctx, check_id=self.check_id, what=what
            )
            return CheckResult(passed=False, event=event)
        event = render_message(
            Msg.NO_OUTBOUND_INTERNET_OBSERVED, ctx=ctx, check_id=self.check_id, what=what
        )
        return CheckResult(passed=True, event=event)

    async def _pod_probe_twice_on_no_egress(self, ctx: Context) -> EgressProbe:
        """One transient DNS or HTTP miss must not zero an idle node: a no_egress reading is re-run once
        and the second run is the verdict (a second run that reaches none is no verdict either)."""
        first = await self._pod_probe(ctx)
        if first.verdict != "no_egress":
            return first
        second = await self._pod_probe(ctx)
        return replace(second, first_reading=first.as_record())

    async def _pod_probe(self, ctx: Context) -> EgressProbe:
        container_name = f"{EGRESS_PROBE_CONTAINER_PREFIX}{uuid.uuid4().hex[:12]}"
        result = await ctx.runner.run(
            pod_probe_command(container_name, sysbox=ctx.state.sysbox_runtime),
            timeout=POD_PROBE_TIMEOUT_SECONDS,
            retryable=False,
        )
        if result.error_type is not None:
            # a timeout can leave the container behind; --rm only fires once its process exits
            await ctx.runner.run(
                f"/usr/bin/docker rm -f {shlex.quote(container_name)} >/dev/null 2>&1; true",
                timeout=30,
                retryable=False,
            )
            return EgressProbe(
                "unmeasured",
                "command_failed",
                detail=f"{result.error_type}: {result.error_message}",
            )
        probe = parse_egress_probe(result.stdout)
        if probe.verdict == "unmeasured":
            probe = EgressProbe(
                "unmeasured",
                "docker_run_failed" if result.exit_code != 0 else probe.reason,
                detail=f"exit {result.exit_code}: {(result.stderr or result.stdout)[-_TAIL_CHARS:]}",
            )
            logger.info(
                _m(
                    "Outbound internet pod probe reached no verdict",
                    extra=get_extra_info({**ctx.default_extra, **probe.as_record()}),
                )
            )
        return probe

    def _skipped(
        self, ctx: Context, reason: str, *, scrape: dict[str, Any] | None = None
    ) -> CheckResult:
        what: dict[str, Any] = {"reason": reason}
        if scrape is not None:
            what["scrape"] = scrape
        event = render_message(Msg.SKIPPED, ctx=ctx, check_id=self.check_id, what=what)
        return CheckResult(passed=True, event=event)
