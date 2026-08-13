"""Which scope a request needs — resolved from the method, the path, and the body's `kind`.

The two authorized keys hold disjoint scopes: the validator key creates validation CVMs, the
platform key creates and destroys renter CVMs. Reads are open to either.

`POST /v1/cvm` is the one route whose required scope depends on the body, which is why the
middleware parses the JSON. It parses it **once** and hands the parsed object to the handler:
two parsers over the same bytes only need to disagree once — duplicate keys, unicode escapes,
number coercion — for the scope check and the handler to act on different values.
"""

from dataclasses import dataclass
from typing import Any

from cvmd.auth.clients import Scope

CVM_PATH = "/v1/cvm"
STATE_PATH = "/v1/state"
HEALTH_PATH = "/health"
# DAH-2675: open to any authorized key via the default below, like the other non-mutating
# routes. Requesting a quote grants nothing — the agent's answer is hardware-bound to the
# caller's nonce — and the narrow ATTEST scope exists so a key can be authorized for exactly
# this and hold no launch or teardown right.
ATTEST_PATH = "/v1/attest"

# Routes open to any authorized key.
EITHER_KEY = None

# `kind` values accepted by POST /v1/cvm, and the scope each one demands.
KIND_TO_SCOPE = {
    "validation": Scope.VALIDATION,
    "renter": Scope.RENTER,
}


@dataclass(frozen=True)
class BodyRejected:
    """The body is absent, unparseable, or not a shape the scope rules understand → 422.

    Returned, not raised. `detail` goes straight into the response, and a reason read off a
    caught exception is what CodeQL's `py/stack-trace-exposure` flags: a raised exception
    carries its traceback, and the rule does not try to guess how much of it the message
    repeats. A plain value keeps the reason exactly as precise as it was — none of these
    three is derived from an exception in the first place — without routing it through one.

    Only ever produced after the signature verified, so this is a well-formedness complaint
    about an authenticated request — never a path to a scope bypass.
    """

    detail: str


def required_scope(method: str, path: str, parsed_body: Any) -> Scope | BodyRejected | None:
    """Return the scope this request needs, or EITHER_KEY if any authorized key may make it.

    Returns a BodyRejected when the body is required but unusable.
    """
    if path == CVM_PATH and method == "POST":
        if not isinstance(parsed_body, dict):
            return BodyRejected("body must be a JSON object")
        kind = parsed_body.get("kind")
        if not isinstance(kind, str):
            return BodyRejected("body has no `kind`")
        scope = KIND_TO_SCOPE.get(kind)
        if scope is None:
            allowed = ", ".join(sorted(KIND_TO_SCOPE))
            return BodyRejected(f"unknown kind {kind!r} (allowed: {allowed})")
        return scope

    if path == CVM_PATH and method == "DELETE":
        return Scope.RENTER

    # Reads, and any path the router will 404 on. An authorized caller who mistypes a path gets
    # the 404; an unauthorized one never gets far enough to learn which paths exist.
    return EITHER_KEY
