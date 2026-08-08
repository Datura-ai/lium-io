"""The full 2-key x 5-endpoint-class scope matrix, plus single-parse enforcement.

The `client` fixture's host has no launch configuration, which is what makes the authorized
cells readable: an authorized create answers **503** ("this host has no launch configuration
for ..."), a code only the handler can produce. Reaching a 503 therefore proves the request
cleared auth and scope, exactly as the 501 did before DAH-2576 filled the handlers in.
Launching for real is `test_launch.py`'s job; this file is about who may ask.
"""

import json

import pytest
from conftest import signed_request

HASH_A = "a" * 64
HASH_B = "b" * 64

# The triple is required on a validation create, so the matrix has to send a complete body —
# otherwise an authorized cell would fail model validation and report 422, which the scope
# layer also produces, and the cell would stop distinguishing anything.
VALIDATION_BODY = {
    "kind": "validation",
    "qemu": "10.1.0",
    "os_image_hash": HASH_A,
    "compose_hash": HASH_B,
}

VALIDATION_CREATE = ("POST", "/v1/cvm", VALIDATION_BODY)
RENTER_CREATE = ("POST", "/v1/cvm", {"kind": "renter"})
RENTER_DESTROY = ("DELETE", "/v1/cvm", None)
STATE_READ = ("GET", "/v1/state", None)
HEALTH = ("GET", "/health", None)

# Every cell of {validator, platform} x {the five endpoint classes}.
#
# 403 is a scope violation on an otherwise valid request. 501 is the renter create, whose body
# DAH-2580 defines. 503 and 200 are handlers that ran.
MATRIX = [
    ("validator", VALIDATION_CREATE, 503),
    ("validator", RENTER_CREATE, 403),
    ("validator", RENTER_DESTROY, 403),
    ("validator", STATE_READ, 200),
    ("platform", VALIDATION_CREATE, 403),
    ("platform", RENTER_CREATE, 501),
    ("platform", RENTER_DESTROY, 200),
    ("platform", STATE_READ, 200),
]


@pytest.mark.parametrize(
    ("key_name", "call", "expected"),
    MATRIX,
    ids=[f"{key}-{call[0]}{call[1]}-{(call[2] or {}).get('kind', '')}" for key, call, _ in MATRIX],
)
def test_scope_matrix(client, validator_key, platform_key, key_name, call, expected):
    keypair = validator_key if key_name == "validator" else platform_key
    method, path, payload = call
    body = json.dumps(payload).encode() if payload is not None else b""

    response = signed_request(client, keypair, method, path, body=body)
    assert response.status_code == expected


def test_health_needs_no_key(client):
    """The fifth endpoint class: open to everyone, signed or not."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "version" in response.json()


def test_unknown_kind_is_422_not_a_scope_bypass(client, platform_key):
    body = json.dumps({"kind": "something-else"}).encode()
    response = signed_request(client, platform_key, "POST", "/v1/cvm", body=body)
    assert response.status_code == 422


def test_missing_kind_is_422(client, platform_key):
    body = json.dumps({"image": "sha256:abc"}).encode()
    response = signed_request(client, platform_key, "POST", "/v1/cvm", body=body)
    assert response.status_code == 422


def test_unparseable_body_after_valid_signature_is_422(client, platform_key):
    """A validly signed non-JSON body is a well-formedness complaint, never a scope bypass."""
    body = b"this is not json"
    response = signed_request(client, platform_key, "POST", "/v1/cvm", body=body)
    assert response.status_code == 422


def test_empty_body_on_create_is_422(client, platform_key):
    response = signed_request(client, platform_key, "POST", "/v1/cvm", body=b"")
    assert response.status_code == 422


def test_signed_request_to_unknown_path_is_404_not_403(client, validator_key):
    """An authorized caller who mistypes a path gets the router's 404.

    An unauthorized one never gets that far, so the 404 does not leak which paths exist.
    """
    assert signed_request(client, validator_key, "GET", "/v1/nope").status_code == 404
    assert client.get("/v1/nope").status_code == 401


class TestSingleParse:
    """The middleware parses the body once, and that one object decides scope and reaches the
    handler.

    Two parsers over the same bytes only need to disagree once — duplicate keys, unicode escapes,
    number coercion — for the scope check and the handler to act on different values. A duplicate
    key is the cheapest way to make two parsers disagree: `json` keeps the last, several other
    parsers keep the first.
    """

    DUPLICATE_KIND = (
        b'{"kind":"renter","kind":"validation","qemu":"10.1.0",'
        b'"os_image_hash":"' + HASH_A.encode() + b'",'
        b'"compose_hash":"' + HASH_B.encode() + b'"}'
    )

    def test_scope_follows_the_last_duplicate(self, client, validator_key):
        """cvmd's parse resolves `kind` to `validation`, so the validator key is the one allowed.

        503 rather than 501 is the tell: the request reached the *validation* handler. Had any
        layer read `renter` — the first value — it would have been 501 instead.
        """
        response = signed_request(
            client, validator_key, "POST", "/v1/cvm", body=self.DUPLICATE_KIND
        )
        assert response.status_code == 503

    def test_the_other_key_is_refused_on_the_same_bytes(self, client, platform_key):
        """The mirror of the above. If any part of the stack read `renter` — the *first* value —
        this would have reached a handler and the two halves would be disagreeing about the
        same request.
        """
        response = signed_request(client, platform_key, "POST", "/v1/cvm", body=self.DUPLICATE_KIND)
        assert response.status_code == 403

    def test_handler_reads_state_not_the_stream(self, client, platform_key):
        """The middleware consumes the request stream to enforce the size cap.

        A handler that re-read the body would find it spent, so reaching the renter handler's
        501 here is positive evidence that the handler is using the object the middleware
        passed forward.
        """
        body = json.dumps({"kind": "renter", "image": "sha256:abc"}).encode()
        response = signed_request(client, platform_key, "POST", "/v1/cvm", body=body)
        assert response.status_code == 501

    def test_required_scope_reads_the_parsed_object(self):
        """The unit-level statement of the same property, with no HTTP in the way."""
        from cvmd.auth.clients import Scope
        from cvmd.auth.scope import required_scope

        parsed = json.loads(self.DUPLICATE_KIND)
        assert required_scope("POST", "/v1/cvm", parsed) is Scope.VALIDATION
