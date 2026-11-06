"""The executor's public GET routes: answered without a miner signature, with what the node is set up for."""

import secrets

import pytest

from tests import lib

pytestmark = pytest.mark.timeout(120)


def test_executor_reports_its_configured_miner_hotkey():
    """GET /miner-identity names the miner hotkey the executor is configured for and echoes the caller's
    nonce, so the answer belongs to this request (lium-io#1628)."""
    nonce = secrets.token_hex(16)
    r = lib.http("GET", f"{lib.EXECUTOR_URL}/miner-identity", params={"nonce": nonce})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["miner_hotkey"] == lib.MINER_HOTKEY, body
    assert body["nonce"] == nonce, body
    # the stack sets no MINER_ACCOUNT_CLAIM, and the route leaves out what is unset
    assert "account_claim" not in body, body


def test_executor_miner_identity_needs_a_hex_nonce():
    """Without a nonce, or with one that is not 16-64 bytes of hex, there is no answer (lium-io#1628)."""
    r = lib.http("GET", f"{lib.EXECUTOR_URL}/miner-identity")
    assert r.status_code == 422, r.text
    r = lib.http("GET", f"{lib.EXECUTOR_URL}/miner-identity", params={"nonce": "not-hex"})
    assert r.status_code == 422, r.text


def test_executor_update_status_answers_without_a_signature():
    """GET /update-status is public: it answered 401 on every node until lium-io#1422. The stack has no
    runner container and no digest endpoint (COMPUTE_REST_API_URL is empty), so the update state is
    unknown and says why, and the route still answers 200."""
    r = lib.http("GET", f"{lib.EXECUTOR_URL}/update-status")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["version"], body
    runner = body["runner"]
    assert runner["update_pending"] is None, body
    assert "COMPUTE_REST_API_URL not set" in (runner["error"] or ""), body
    assert "running_digest" in body["executor"], body
