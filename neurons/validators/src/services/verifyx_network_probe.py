"""Which failed VerifyX network probes are Cloudflare's fault, not the host's.

Kept free of settings and SSH imports: the preflight image (Dockerfile.preflight) ships only
`preflight/` and `services/` with nvidia-ml-py, and its network verdict must match the validator's.
"""

import re

# Transport / Cloudflare-side faults only. The word "cloudflare" in an error is not enough:
# a host that failed the probe for its own reason can mention that URL. The verifier's own texts for a probe-side fault (celium-gpu-verifier network.rs): Cloudflare's 429,
# a direction timeout, or the transport error to speed.cloudflare.com itself.
CLOUDFLARE_PROBE_FAIL_RX = re.compile(
    r"429|timeout|timed out|outage|connection refused|rate.?limit"
    r"|\bDownload request failed for https://speed\.cloudflare\.com/\S*: error sending request\b",
    re.I,
)


def is_positive_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def is_cloudflare_probe_failure(network_execution: dict) -> bool:
    """True only for a probe-side transport or Cloudflare fault in the probe's own error.

    A failed probe with no error, or an error that only names the Cloudflare URL, is a
    host-caused failure and does not fall back to the package download.
    """
    capacity = (network_execution.get("speedtest") or {}).get("download_mbps")
    if is_positive_number(capacity):
        return False
    err = str(network_execution.get("error") or "").strip()
    if not err:
        return False
    return bool(CLOUDFLARE_PROBE_FAIL_RX.search(err))
