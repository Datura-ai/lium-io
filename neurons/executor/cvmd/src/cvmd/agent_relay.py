"""Dial the in-guest attest-agent through its host-side forward (DAH-2675).

cvmd is the only party that can reach the agent when its forward is bound to loopback — the
default `ports.py` gives a three-part spec — and that binding is wanted: it keeps the agent off
the public internet without publishing a per-node port. So the validator asks cvmd, cvmd asks
the agent, and the answer goes back verbatim.

Relaying adds no trust. The agent's answer is a TDX quote whose `report_data` binds the agent's
TLS key, the GPU set and the caller's nonce (attest-agent/identity.py), so this host can
withhold the answer but cannot forge or replay one — the verifier on the validator's side checks
the binding against the nonce it issued. That is also why TLS verification is off here: the
certificate is the agent's self-signed one, and the quote binding — not a CA — is what
authenticates the exchange, the same stance the validator's own relay takes toward cvmd.
"""

import json
import logging
import ssl
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

ATTEST_PATH = "/v1/attest"


class AgentRelayError(Exception):
    """The agent was not reached, or answered something that is not a JSON object."""


def dial_address(bind_address: str) -> str:
    """Where to dial for a forward bound at `bind_address`, from this host itself.

    A loopback or wildcard bind is reachable at 127.0.0.1 from here; an explicit address is
    dialed as written. An empty address is a record predating the address field — treat it as
    the loopback default the port parser would have applied.
    """
    if not bind_address or bind_address == "0.0.0.0":
        return "127.0.0.1"
    return bind_address


def relay_attest(
    *,
    address: str,
    host_port: int,
    nonce: str,
    timeout_seconds: float,
) -> tuple[int, dict]:
    """POST the caller's nonce to the agent and return (status, body) exactly as answered.

    HTTP error statuses are answers, not failures: a 422 or 503 from the agent carries the
    distinction the validator acts on, so it is passed through rather than flattened into one
    relay error. Only "no usable answer at all" raises.
    """
    url = f"https://{dial_address(address)}:{host_port}{ATTEST_PATH}"
    body = json.dumps({"nonce": nonce}).encode()

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE

    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds, context=context) as answer:
            return answer.status, _decode(answer.read())
    except urllib.error.HTTPError as exc:
        return exc.code, _decode(exc.read())
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise AgentRelayError(f"the attest-agent at {url} could not be reached: {exc}") from exc


def _decode(raw: bytes) -> dict:
    try:
        parsed = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise AgentRelayError("the attest-agent answered with something that is not JSON") from exc
    if not isinstance(parsed, dict):
        raise AgentRelayError(
            f"the attest-agent answered with a {type(parsed).__name__}, not an object"
        )
    return parsed
