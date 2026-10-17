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


def _fetch(host: str, port: int, timeout: float) -> str | None:
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/", timeout=timeout) as resp:
            return resp.read(256).decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.read(256).decode(errors="replace")
    except Exception:
        return None


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
        bodies = list(pool.map(lambda t: _fetch(t[0], t[1], timeout), targets))
    results = [{"host": h, "port": p, "body": b} for (h, p), b in zip(targets, bodies)]
    return {"statusCode": 200, "headers": {"content-type": "application/json"}, "body": json.dumps({"results": results})}
