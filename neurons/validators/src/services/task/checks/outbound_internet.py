from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from core.config import settings

from ..messages import OutboundInternetMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context

EGRESS_PROBE_HOST = "pypi.org"
# 325 bytes, asked for headers only: https://pypi.org/simple/ is 46 MB, and busybox wget's -T is a read
# timeout, not a total one
EGRESS_PROBE_URL = "https://pypi.org/robots.txt"
EGRESS_FETCH_TIMEOUT_SECONDS = 10
EGRESS_PROBE_MARKER = "lium_egress"
_TAIL_CHARS = 300

# `getent hosts pypi.org && curl -sS -m 10 -o /dev/null -I -w '%{http_code}' https://pypi.org/robots.txt`, in
# stages so the verdict says which one failed. POSIX sh: the rental probe runs it in the renter image (curl,
# or busybox wget where there is no curl). curl's -m is a total deadline; wget has none (-T is
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

    def as_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {"verdict": self.verdict, "reason": self.reason}
        if self.tool:
            record["tool"] = self.tool
        if self.http_code:
            record["http_code"] = self.http_code
        if self.detail:
            record["detail"] = self.detail[-_TAIL_CHARS:]
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


def _is_positive_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def scrape_egress_finding(specs: dict[str, Any] | None) -> dict[str, Any] | None:
    """What the scrape's speed tests say about egress; None when the scrape ran no speed test.

    benchmark_network_speed keeps the first download and upload any method measured and each method's
    result under `measurements`; with neither measured, all four methods ran. A finding only when
    neither direction was measured: a missing download alone is not one (ticket-0361: 24 of one
    provider's 27 active nodes had no download, most of them a Cloudflare download recorded as 0, and
    14e704ba's upload measured 77-105 Mbps), nor is an error from a method a later one measured past.
    A network block without `measurements` is no reading: lium-io#1419 (DAH-2774) removes the scrape's
    speed tests and leaves `{}`, which must not read as every node without egress.
    """
    network = (specs or {}).get("network")
    if not isinstance(network, dict):
        return None
    if not isinstance(network.get("measurements"), dict) or not network["measurements"]:
        return None
    errors: dict[str, str] = {}
    if network.get("network_speed_error"):
        errors["network"] = str(network["network_speed_error"])[:_TAIL_CHARS]
    for method, measurement in network["measurements"].items():
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
    """Fail an idle node whose scrape ran its speed tests and measured neither direction (NO_OUTBOUND_INTERNET).

    A null download alone is not a finding (ticket-0361: 24 of one provider's 27 active nodes had one), nor
    is a scrape that ran no speed test (lium-io#1419), which is logged as OUTBOUND_INTERNET_UNMEASURED. A
    slow host passes: speed stays behind FeatureFlag.VERIFYX_NETWORK_VALIDATION. The in-pod view is the
    rental probe's `egress` step; a registry path that cannot pull is RegistryPullCheck's.

    Under NO_OUTBOUND_INTERNET_ENFORCEMENT_ENABLED the finding fails the node the way INSUFFICIENT_PORTS
    does; without it the check logs NO_OUTBOUND_INTERNET_OBSERVED and passes. A rented node is left
    alone like PortCountCheck leaves it.
    """

    check_id = "executor.validate.outbound_internet"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        if not settings.NO_OUTBOUND_INTERNET_CHECK_ENABLED:
            return self._skipped(ctx, "NO_OUTBOUND_INTERNET_CHECK_ENABLED is off")

        scrape = scrape_egress_finding(ctx.state.specs)
        if _is_rented(ctx):
            return self._skipped(ctx, "rented", scrape=scrape)

        what: dict[str, Any] = {
            "scrape": scrape,
            "enforced": settings.NO_OUTBOUND_INTERNET_ENFORCEMENT_ENABLED,
        }
        if scrape is None or not scrape["no_egress"]:
            template = (
                Msg.OUTBOUND_INTERNET_UNMEASURED if scrape is None else Msg.OUTBOUND_INTERNET_OK
            )
            event = render_message(template, ctx=ctx, check_id=self.check_id, what=what)
            return CheckResult(passed=True, event=event)

        what["failed_by"] = ["scrape"]
        if settings.NO_OUTBOUND_INTERNET_ENFORCEMENT_ENABLED:
            event = render_message(
                Msg.NO_OUTBOUND_INTERNET, ctx=ctx, check_id=self.check_id, what=what
            )
            return CheckResult(passed=False, event=event)
        event = render_message(
            Msg.NO_OUTBOUND_INTERNET_OBSERVED, ctx=ctx, check_id=self.check_id, what=what
        )
        return CheckResult(passed=True, event=event)

    def _skipped(
        self, ctx: Context, reason: str, *, scrape: dict[str, Any] | None = None
    ) -> CheckResult:
        what: dict[str, Any] = {"reason": reason}
        if scrape is not None:
            what["scrape"] = scrape
        event = render_message(Msg.SKIPPED, ctx=ctx, check_id=self.check_id, what=what)
        return CheckResult(passed=True, event=event)
