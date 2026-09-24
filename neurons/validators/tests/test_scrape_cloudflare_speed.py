"""cloudflare_speed() in machine_scrape.py: a refusal is an error, not a measured 0 Mbps.

ticket-0361 follow-up: 24 of one provider's 27 active nodes reported no download (the ticket-0361 node's upload measured
77-105 Mbps). Cloudflare answers a request it will not serve with HTTP 403 and a 1-byte body (seen for
__down?bytes=100000000 on 23 Sep 2026) and curl exits 0 on it, so the speed read alone was 0.0 Mbps, which
benchmark_network_speed then drops as "no figure" without any error saying why.

machine_scrape.py is a script, not a module, so the helpers are extracted by ast (tests/helpers.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from neurons.validators.tests.helpers import build_scrape_namespace

SRC = Path(__file__).resolve().parents[1] / "src"
HELPERS = {
    "CLOUDFLARE_DOWN_BYTES",
    "CLOUDFLARE_UP_BYTES",
    "CLOUDFLARE_MAX_SECONDS",
    "CURL_EXIT_OPERATION_TIMEDOUT",
    "CLOUDFLARE_UP_CUT_STATUSES",
    "cloudflare_transfer_mbps",
    "cloudflare_speed",
}
DOWN_OK = (0, "200 50000000 12500000", "")  # 100 Mbps
UP_OK = (0, "200 26214400 11250000", "")  # 90 Mbps
REFUSED = (0, "403 1 35", "")  # what Cloudflare sent for a size it would not serve


def scrape(*answers: tuple[int, str, str]) -> tuple[dict[str, Any], list[str]]:
    """The helpers, with run_cmd_status answering the download, then the upload."""
    commands: list[str] = []
    queue = list(answers)

    def run_cmd_status(cmd: str) -> tuple[int, str, str]:
        commands.append(cmd)
        return queue.pop(0)

    namespace = build_scrape_namespace(
        SRC / "miner_jobs" / "machine_scrape.py", HELPERS, {"run_cmd_status": run_cmd_status}
    )
    return namespace, commands


def test_both_directions_measured():
    namespace, commands = scrape(DOWN_OK, UP_OK)

    data = namespace["cloudflare_speed"]()

    assert data == {"download_speed": 100.0, "upload_speed": 90.0}
    assert "__down?bytes=50000000" in commands[0] and "%{http_code}" in commands[0]
    assert "__up" in commands[1] and "%{http_code}" in commands[1]


def test_a_refused_download_is_an_error_and_the_upload_is_still_measured():
    """Regression: a 403 with a 1-byte body read as 0.0 Mbps, the node's only download figure, with no error."""
    namespace, _ = scrape(REFUSED, UP_OK)

    data = namespace["cloudflare_speed"]()

    assert data["download_speed"] is None
    assert data["upload_speed"] == 90.0
    assert "HTTP 403, 1 bytes" in data["network_speed_error"]


def test_a_failed_download_does_not_skip_the_upload():
    """Regression: download and upload shared one try, so a download error left upload unmeasured too."""
    namespace, commands = scrape((6, "000 0 0", "curl: (6) Could not resolve host"), UP_OK)

    data = namespace["cloudflare_speed"]()

    assert len(commands) == 2
    assert data["upload_speed"] == 90.0
    assert "curl exit 6" in data["network_speed_error"]


def test_a_transfer_cut_at_max_time_is_a_slow_measurement():
    """Regression: 50 MB within --max-time 15 needs 26.7 Mbps; a slower host's curl exit 28 was read as no
    figure at all instead of its average speed."""
    namespace, _ = scrape((28, "200 18750000 1250000", ""), UP_OK)

    data = namespace["cloudflare_speed"]()

    assert data["download_speed"] == 10.0
    assert "network_speed_error" not in data


# curl 8.5 posting 25 MB to a local server that reads slowly, --max-time 3: `100 1900544 633113` exit 28
# with curl's default `Expect: 100-continue`, `000 1966080 654831` exit 28 with `-H 'Expect:'`
@pytest.mark.parametrize("status", ["000", "100"], ids=["no_expect", "expect_100_continue"])
def test_an_upload_cut_at_max_time_is_a_slow_measurement(status):
    """Regression (r3): __up answers only once the whole body is in, so an upload cut at --max-time has no
    final status (000, or 100 after Expect: 100-continue) and a slow host's upload was read as refused."""
    namespace, commands = scrape(
        DOWN_OK, (28, f"{status} 1966080 654831", "curl: (28) Operation timed out")
    )

    data = namespace["cloudflare_speed"]()

    assert data == {"download_speed": 100.0, "upload_speed": 5.24}
    assert "-H 'Expect:'" in commands[1]


def test_a_cut_transfer_without_a_measurement_raises():
    """A download cut before any answer has no 200; an upload cut before a byte went out measured nothing."""
    namespace, _ = scrape()
    cut_before_anything = (28, "000 0 0", "curl: (28) Operation timed out")

    with pytest.raises(RuntimeError, match="HTTP 000, 0 bytes"):
        namespace["cloudflare_transfer_mbps"](
            *cut_before_anything, namespace["CLOUDFLARE_DOWN_BYTES"]
        )
    with pytest.raises(RuntimeError, match="0 bytes in 15 s"):
        namespace["cloudflare_transfer_mbps"](
            *cut_before_anything,
            namespace["CLOUDFLARE_UP_BYTES"],
            namespace["CLOUDFLARE_UP_CUT_STATUSES"],
        )


@pytest.mark.parametrize(
    "answer,error",
    [
        ((0, "200 1000 50000000", ""), "1000 of 50000000 bytes"),
        ((28, "200 0 0", ""), "0 bytes in 15 s"),
        ((0, "garbage", ""), "curl exit 0"),
    ],
    ids=["short_body", "nothing_before_max_time", "unparseable"],
)
def test_an_answer_that_is_no_measurement_raises(answer, error):
    namespace, _ = scrape()

    with pytest.raises(RuntimeError, match=error):
        namespace["cloudflare_transfer_mbps"](*answer, namespace["CLOUDFLARE_DOWN_BYTES"])
