from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

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
