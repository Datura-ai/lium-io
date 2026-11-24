"""External reachability prober: fetches http://host:port/ from AWS Lambda's shared egress, outside the validator's
network, and returns the first bytes of each answer (None when nothing answered).

DAH-4537. Deployed as Lambda `lium-external-reachability-prober` (python3.12, handler `handler.handler`) behind a
function URL with no VPC, so requests leave from AWS's shared egress. `PROBE_TOKEN` is the bearer token the validator
sends as EXTERNAL_PORT_PROBE_TOKEN; the URL is EXTERNAL_PORT_PROBE_URL.
"""

import base64
import hmac
import ipaddress
import json
import os
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

MAX_TARGETS = 64
# A node's answer is "<token>:<port>". Same cap as the validator's own probe (PortTester.PROBE_BODY_LIMIT): a longer
# answer is no answer on both paths, so the two can never disagree on a port.
MAX_BODY = 256
# Fetched with every request: if the prober cannot reach this known-open URL, its own egress is down and an empty
# answer would read as "every node blocks the outside", so it answers 503 (no verdict) instead.
CANARY_URL = os.environ.get("CANARY_URL", "http://checkip.amazonaws.com/")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    # a redirect could lead to a private address the public-IP check never saw; the 3xx itself is the answer
    def redirect_request(self, *args, **kwargs):
        return None


_opener = urllib.request.build_opener(_NoRedirect)


def _fetch(host: str, port: int, timeout: float) -> str | None:
    return _get(f"http://{host}:{port}/", timeout)


def _get(url: str, timeout: float, read=None) -> str | None:
    # one handler for opening and reading: an error status is an answer too, and a node that stalls or drops mid-body
    # is no answer for that target, never a failure of the whole invocation
    try:
        try:
            resp = _opener.open(url, timeout=timeout)
        except urllib.error.HTTPError as e:
            resp = e
        with resp:
            return (read or _capped)(resp)
    except Exception:
        return None


def _canary_up(timeout: float) -> bool:
    # the canary proves egress by answering at all; the node-answer size rule is not about it
    return _get(CANARY_URL, timeout, read=lambda resp: "") is not None


def _capped(resp) -> str | None:
    data = resp.read(MAX_BODY + 1)
    return None if len(data) > MAX_BODY else data.decode(errors="replace")


def _public(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_global
    except ValueError:
        return False


def handler(event, context):
    headers = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
    token = os.environ["PROBE_TOKEN"]
    if not hmac.compare_digest(headers.get("authorization", ""), f"Bearer {token}"):
        return {"statusCode": 401, "body": "unauthorized"}
    try:
        raw = event.get("body") or "{}"
        body = json.loads(base64.b64decode(raw) if event.get("isBase64Encoded") else raw)
        targets = [(str(t["host"]), int(t["port"])) for t in body["targets"]]
        timeout = min(float(body.get("timeout", 3)), 10.0)
    except (ValueError, KeyError, TypeError):
        return {"statusCode": 400, "body": "bad request"}
    # only public IPs: this must not become a way into anything private
    if not targets or len(targets) > MAX_TARGETS or not all(_public(h) and 0 < p < 65536 for h, p in targets):
        return {"statusCode": 400, "body": "bad targets"}
    with ThreadPoolExecutor(max_workers=16) as pool:
        canary = pool.submit(_canary_up, timeout)
        bodies = list(pool.map(lambda t: _fetch(t[0], t[1], timeout), targets))
        if not canary.result():
            return {"statusCode": 503, "body": "prober egress failed"}
    results = [{"host": h, "port": p, "body": b} for (h, p), b in zip(targets, bodies)]
    return {"statusCode": 200, "headers": {"content-type": "application/json"}, "body": json.dumps({"results": results})}
