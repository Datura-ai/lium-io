"""Forward a platform-signed cvmd call to a CVM host, unchanged (DAH-2580).

This validator is transport for renter provisioning and nothing else. It receives four headers
and a body the backend signed, opens a connection to the host, and replays them. It cannot
alter any of it: the signature covers the method, the request target and the body bytes, so an
edit anywhere produces a request cvmd refuses. It cannot originate one either — cvmd gives the
`renter` scope to the platform key alone, and this process does not hold that key.

That split is the point of the design rather than a consequence of it. "The validator never
triggers renter provisioning and never destroys CVMs" is a property of what this process can
sign, not a rule it is trusted to follow, so a compromised validator cannot start or stop a
customer's CVM — the worst it can do is fail to relay, which shows up as a rental that did not
start rather than as one nobody ordered.

**TLS.** cvmd serves HTTPS with a certificate the host generated for itself, so there is no CA
that could vouch for it and certificate verification is not what authenticates this exchange —
the signature in the request and the measurements in the response are. Verification is
therefore off, deliberately and narrowly: the connection carries a bearer capability that is
already scoped to one call on one host, and it carries no secret in the response.

**Timeouts.** A launch waits for a guest to boot and a teardown waits for a host's memory to
come back; both are minutes, and a teardown of a large-memory guest is tens of minutes. So the
budget is per-operation and generous. Nothing here retries: a repeated launch could land a
second CVM request on a node already committed to the first, and cvmd's own 409 is a better
answer to that than a client that keeps asking.
"""

import json
import logging
from dataclasses import dataclass

import aiohttp

logger = logging.getLogger(__name__)

# A launch holds until the guest answers, which cvmd itself bounds at 900 s by default. The
# relay's budget sits above that so the host's own timeout is the one that decides, and its
# reason — which names what did not come up — is the one that reaches the backend.
DEFAULT_PROVISION_TIMEOUT_SECONDS = 1200

# A verified teardown holds until all four release conditions hold together, which cvmd bounds
# at 1800 s. Same reasoning, same ordering.
DEFAULT_TEARDOWN_TIMEOUT_SECONDS = 2100


class CvmdRelayError(Exception):
    """The host was not reached, or answered something that is not a JSON body."""


@dataclass(frozen=True)
class RelayResult:
    """What the host answered. `ok` is the host's own verdict, not this relay's."""

    status: int
    body: dict

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def reason(self) -> str:
        """The host's own explanation, or a description of an answer that carried none."""
        detail = self.body.get("detail")
        if isinstance(detail, str) and detail:
            return detail
        return f"cvmd answered {self.status} with no detail"


class CvmdRelay:
    """Replays one signed call. Holds no key and makes no decisions about what to send."""

    def __init__(self, *, session_factory=None) -> None:
        # Injectable so a test can assert on exactly the bytes and headers that go out. The
        # claim being tested is that nothing is altered in transit, and that is only checkable
        # from the outgoing side.
        self._session_factory = session_factory or aiohttp.ClientSession

    async def forward(
        self,
        *,
        base_url: str,
        method: str,
        path: str,
        body: str,
        headers: dict[str, str],
        timeout_seconds: int,
    ) -> RelayResult:
        url = f"{base_url.rstrip('/')}{path}"
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)

        try:
            async with self._session_factory(timeout=timeout) as session:
                async with session.request(
                    method,
                    url,
                    # `data` rather than `json`: the signature covers these exact bytes, and
                    # letting aiohttp serialize an object would re-encode them.
                    data=body.encode(),
                    headers={**headers, "Content-Type": "application/json"},
                    ssl=False,
                ) as response:
                    text = await response.text()
                    return RelayResult(status=response.status, body=_as_dict(text))
        except CvmdRelayError:
            raise
        except Exception as exc:
            # Includes the timeout. Reported as "not reached" rather than as a failed launch,
            # because a launch that timed out on this side may still be running on the host —
            # and treating it as failed is how a node ends up holding a CVM nobody records.
            raise CvmdRelayError(f"cvmd at {url} could not be reached: {exc}") from exc


def _as_dict(text: str) -> dict:
    """Parse a cvmd response body. A non-JSON answer is reported, never guessed at."""
    if not text.strip():
        return {}
    try:
        parsed = json.loads(text)
    except ValueError as exc:
        raise CvmdRelayError(f"cvmd answered with something that is not JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise CvmdRelayError(f"cvmd answered with a {type(parsed).__name__}, not an object")
    return parsed
