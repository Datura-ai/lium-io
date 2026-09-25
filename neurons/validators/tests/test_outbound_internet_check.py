"""EGRESS_PROBE_SCRIPT, the rental probe's egress step (NO_OUTBOUND_INTERNET), run under sh.

ticket-0361: the fail signals are the paths a renter takes, RegistryPullCheck's real pull
(test_registry_pull_check.py) and this step inside the renter container.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from services.task.checks.outbound_internet import EGRESS_PROBE_SCRIPT, parse_egress_probe


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
