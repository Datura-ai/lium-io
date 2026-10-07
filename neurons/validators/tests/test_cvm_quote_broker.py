"""A customer pod on a CVM node reaches the dstack guest agent only through the quote-only broker.

The broker is one nginx container per guest, started by the validator over the guest's docker
socket: it forwards ``GetQuote``/``Info``/``Version`` to ``/var/run/dstack.sock`` and answers
everything else (key derivation, RTMR3 extension) with 403. The pod gets the broker's socket at the
dstack SDK's default path; bare-metal nodes and fillers get nothing.
"""

import re
from unittest.mock import Mock

import pytest

from services.cvm_quote_broker import (
    QUOTE_BROKER_ALLOWED_PATH_RE,
)
from services.docker_service import DockerService


@pytest.fixture
def docker_service() -> DockerService:
    # _build_rental_container_run_spec is pure (no I/O), so mocked dependencies suffice.
    return DockerService(
        ssh_service=Mock(),
        redis_service=Mock(),
        attestation_service=Mock(),
        rental_docker_client_factory=Mock(),
    )


# --- what the broker lets through -------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "/GetKey",  # derives the guest's app keys, shared with the executor and other pods
        "/GetTlsKey",
        "/Sign",
        "/EmitEvent",  # extends RTMR3 — would change the node's own measurement
        "/Attest",
        "/Verify",
        "/prpc/DstackGuest.GetKey",
        "/GetQuoteX",
        "/GetQuote/extra",
        "/",
    ],
)
def test_allow_list_rejects_everything_else(path):
    assert re.match(QUOTE_BROKER_ALLOWED_PATH_RE, path) is None


# --- the broker container ----------------------------------------------------------------------


# --- ensure_quote_broker -----------------------------------------------------------------------


# --- what the pod gets --------------------------------------------------------------------------


