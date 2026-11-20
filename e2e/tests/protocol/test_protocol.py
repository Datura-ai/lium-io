"""The wire contract between validator, miner and executor — accepted for the right keys, refused for everything else.

What a real cycle does before any check runs: the validator signs into the miner (four headers over
AuthenticationPayload), the miner forwards the validator's SSH public key to each executor with its own signature on
top (/upload_ssh_key), the executor installs it and answers with its SSH coordinates, and at the end the key is
removed. Each step here is the real HTTP call the code makes, plus the refusals a spoofer must hit.
"""

import socket
import time

import asyncssh
import pytest

from tests import lib

pytestmark = pytest.mark.timeout(120)


# ---------------------------------------------------------------- registration ---------------------------------------


def test_executor_answers_version():
    r = lib.http("GET", f"{lib.EXECUTOR_URL}/version")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["version"] and body["version"] != "unknown", body
    # the executor always serves the one-call /verify (DAH-4912)
    assert body["capabilities"] == ["local_verify/1"], body


def test_miner_lists_the_executor_for_its_validator():
    """POST /executors: the validator signs its own hotkey; the miner returns the executors assigned to it."""
    vk = lib.validator_keypair()
    r = lib.http("POST", f"{lib.MINER_URL}/executors", json={"signature": lib.sign(vk, vk.ss58_address), "validator_hotkey": vk.ss58_address})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["validator_hotkey"] == vk.ss58_address
    listed = {(e["uuid"], e["address"], e["port"]) for e in body["executors"]}
    assert (lib.EXECUTOR_UUID, lib.EXECUTOR_IP, lib.EXECUTOR_PORT) in listed, body
    assert (lib.DEAD_EXECUTOR_UUID, lib.DEAD_EXECUTOR_IP, lib.EXECUTOR_PORT) in listed, "the seeded offline executor must be listed too"


def test_miner_refuses_a_stranger_listing_executors():
    sk = lib.stranger_keypair()
    r = lib.http("POST", f"{lib.MINER_URL}/executors", json={"signature": lib.sign(sk, sk.ss58_address), "validator_hotkey": sk.ss58_address})
    # a stranger with a valid signature over its own hotkey is not a registered validator: refused outright
    assert r.status_code == 403, r.text
    vk = lib.validator_keypair()
    r = lib.http("POST", f"{lib.MINER_URL}/executors", json={"signature": lib.sign(sk, vk.ss58_address), "validator_hotkey": vk.ss58_address})
    assert r.status_code == 401, f"a stranger's signature under the validator's hotkey must be refused: {r.status_code} {r.text}"


# ---------------------------------------------------------- validator → miner REST auth --------------------------------


def _submit(headers: dict, body: dict | None = None):
    priv, pub = lib.ssh_keypair()
    vk = lib.validator_keypair()
    body = body or {
        "message_type": "SSHPubKeySubmitRequest",
        "public_key": pub,
        "validator_signature": lib.sign(vk, pub),
        "validator_miner_signature": lib.sign(vk, lib.ssh_pubkey_signing_blob(pub, None, lib.MINER_HOTKEY)),
        "miner_hotkey": lib.MINER_HOTKEY,
    }
    return lib.http("POST", f"{lib.MINER_URL}/api/validator/ssh-pubkey-submit", json=body, headers=headers, timeout=60), priv, pub


def test_validator_headers_are_accepted_and_the_executor_installs_the_key():
    vk = lib.validator_keypair()
    r, priv, pub = _submit(lib.validator_rest_headers(vk))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("message_type") == "AcceptSSHKeyRequest", body
    assert len(body["executors"]) == 1, body
    # the miner also names every executor it knows, so the validator can tell "unreachable" from "not listed"
    assert set(body["known_executor_ids"]) == {lib.EXECUTOR_UUID, lib.DEAD_EXECUTOR_UUID}, body
    ex = body["executors"][0]
    assert ex["uuid"] == lib.EXECUTOR_UUID and ex["address"] == lib.EXECUTOR_IP
    assert ex["ssh_port"] == lib.EXECUTOR_SSH_PORT and ex["ssh_username"] and ex["python_path"] and ex["root_dir"]
    # the key the validator minted now opens the executor — the SSH hop every check runs over
    rc, out = lib.run(lib.ssh_run(ex["address"], ex["ssh_port"], ex["ssh_username"], priv, "id -un && test -d " + ex["root_dir"]))
    assert rc == 0 and out.strip() == ex["ssh_username"], (rc, out)
    # remove it the way the validator does at the end of the cycle
    r = lib.http("POST", f"{lib.MINER_URL}/api/validator/ssh-pubkey-remove", headers=lib.validator_rest_headers(vk),
                 json={"message_type": "SSHPubKeyRemoveRequest", "public_key": pub, "validator_signature": lib.sign(vk, pub), "miner_hotkey": lib.MINER_HOTKEY}, timeout=60)
    assert r.status_code == 200, r.text
    with pytest.raises(asyncssh.PermissionDenied):
        lib.run(lib.ssh_run(ex["address"], ex["ssh_port"], ex["ssh_username"], priv, "true", timeout=15))


def test_stranger_signature_is_refused_401():
    sk = lib.stranger_keypair()
    vk = lib.validator_keypair()
    h = lib.validator_rest_headers(vk)
    h["X-Signature"] = lib.validator_rest_headers(sk)["X-Signature"]  # right hotkey, wrong signer
    r, _, _ = _submit(h)
    assert r.status_code == 401, r.text


def test_unregistered_validator_is_refused_403():
    sk = lib.stranger_keypair()
    r, _, _ = _submit(lib.validator_rest_headers(sk))
    assert r.status_code == 403, r.text


def test_stale_timestamp_is_refused_401():
    vk = lib.validator_keypair()
    r, _, _ = _submit(lib.validator_rest_headers(vk, timestamp=int(time.time()) - 600))
    assert r.status_code == 401 and "old" in r.text.lower(), r.text


def test_future_timestamp_is_refused_401():
    vk = lib.validator_keypair()
    r, _, _ = _submit(lib.validator_rest_headers(vk, timestamp=int(time.time()) + 600))
    assert r.status_code == 401 and "future" in r.text.lower(), r.text


def test_headers_naming_another_miner_are_refused_403():
    """X-Miner-Hotkey is part of the signed blob; outside CENTRAL_MODE the miner serves one hotkey and refuses a
    correctly signed sign-in made for another miner (dependencies/auth.py, lium-io#1324)."""
    vk = lib.validator_keypair()
    r, _, _ = _submit(lib.validator_rest_headers(vk, miner_hotkey=lib.stranger_keypair().ss58_address))
    assert r.status_code == 403 and "another miner" in r.text, r.text


# ------------------------------------------------------------ miner → executor /upload_ssh_key -------------------------


def test_executor_sshd_listens_on_the_port_it_advertises():
    """SSH_PORT is what the executor reports; nothing in the image makes sshd listen there (stock config: :22, bridged in
    production by a docker port map this stack does not have). The compose `command:` sets sshd's Port from SSH_PORT —
    this reads the SSH banner on the advertised port. Fails with ECONNREFUSED when sshd is still on :22 (the port is 2222
    on purpose, so the CPU path proves the mechanism the GPU host relies on)."""
    priv, pub = lib.ssh_keypair()
    r = lib.http("POST", f"{lib.EXECUTOR_URL}/upload_ssh_key", json=lib.upload_ssh_key_payload(lib.miner_keypair(), lib.validator_keypair(), pub))
    assert r.status_code == 200, r.text
    port = r.json()["ssh_port"]
    assert port == lib.EXECUTOR_SSH_PORT != 22, "the stack must advertise a non-default port or the check proves nothing"
    with socket.create_connection((lib.EXECUTOR_IP, port), timeout=10) as s:
        banner = s.recv(64)
    assert banner.startswith(b"SSH-2.0-"), banner
    lib.http("POST", f"{lib.EXECUTOR_URL}/remove_ssh_key", json=lib.upload_ssh_key_payload(lib.miner_keypair(), lib.validator_keypair(), pub))


def test_executor_accepts_the_double_signature():
    priv, pub = lib.ssh_keypair()
    r = lib.http("POST", f"{lib.EXECUTOR_URL}/upload_ssh_key", json=lib.upload_ssh_key_payload(lib.miner_keypair(), lib.validator_keypair(), pub))
    assert r.status_code == 200, r.text
    info = r.json()
    assert info["ssh_port"] == lib.EXECUTOR_SSH_PORT and info["ssh_username"]
    rc, _ = lib.run(lib.ssh_run(lib.EXECUTOR_IP, info["ssh_port"], info["ssh_username"], priv, "true"))
    assert rc == 0
    r = lib.http("POST", f"{lib.EXECUTOR_URL}/remove_ssh_key", json=lib.upload_ssh_key_payload(lib.miner_keypair(), lib.validator_keypair(), pub))
    assert r.status_code == 200, r.text


def test_executor_refuses_a_spoofed_validator_signature():
    _, pub = lib.ssh_keypair()
    r = lib.http("POST", f"{lib.EXECUTOR_URL}/upload_ssh_key", json=lib.upload_ssh_key_payload(lib.miner_keypair(), lib.stranger_keypair(), pub))
    assert r.status_code == 401, r.text


def test_executor_refuses_a_spoofed_miner_signature():
    _, pub = lib.ssh_keypair()
    r = lib.http("POST", f"{lib.EXECUTOR_URL}/upload_ssh_key", json=lib.upload_ssh_key_payload(lib.stranger_keypair(), lib.validator_keypair(), pub))
    # refused by the miner-signature middleware before the route runs
    assert r.status_code == 401, r.text


def test_executor_refuses_a_substituted_public_key():
    """Issue #744: a valid miner signature over one key must not install a different key."""
    _, pub = lib.ssh_keypair()
    _, other = lib.ssh_keypair()
    body = lib.upload_ssh_key_payload(lib.miner_keypair(), lib.validator_keypair(), pub)
    body["public_key"] = other
    body["validator_signature"] = lib.sign(lib.validator_keypair(), other)
    r = lib.http("POST", f"{lib.EXECUTOR_URL}/upload_ssh_key", json=body)
    # the miner signature over data_to_sign is valid, so the route's key-consistency check answers
    assert r.status_code == 400 and "mismatch" in r.text.lower(), r.text


# --------------------------------------------------- executor: key bound to a miner, dated nonce ------------------------


def _upload_ssh_key_to_executor(body: dict):
    return lib.http("POST", f"{lib.EXECUTOR_URL}/upload_ssh_key", json=body)


def _remove_ssh_key_from_executor(body: dict) -> None:
    lib.http("POST", f"{lib.EXECUTOR_URL}/remove_ssh_key", json=body)


def test_executor_accepts_a_key_bound_to_the_pool_hotkey():
    """A provider on Lium-held payouts is served under the pool hotkey: a key the validator bound to it is accepted
    like one bound to the executor's own miner (lium-io#1923)."""
    priv, pub = lib.ssh_keypair()
    body = lib.upload_ssh_key_payload(lib.miner_keypair(), lib.validator_keypair(), pub, bound_to=lib.POOL_HOTKEY)
    r = _upload_ssh_key_to_executor(body)
    assert r.status_code == 200, r.text
    try:
        rc, _ = lib.run(lib.ssh_run(lib.EXECUTOR_IP, r.json()["ssh_port"], r.json()["ssh_username"], priv, "true"))
        assert rc == 0
    finally:
        _remove_ssh_key_from_executor(body)


def test_executor_refuses_a_key_bound_to_another_miner_403():
    _, pub = lib.ssh_keypair()
    body = lib.upload_ssh_key_payload(lib.miner_keypair(), lib.validator_keypair(), pub, bound_to=lib.stranger_keypair().ss58_address)
    r = _upload_ssh_key_to_executor(body)
    assert r.status_code == 403, r.text


def test_executor_refuses_a_key_without_a_miner_binding_403():
    """The stack's executor enforces the binding (REQUIRE_MINER_BOUND_SSH_KEY): a legacy upload without one is refused."""
    _, pub = lib.ssh_keypair()
    r = _upload_ssh_key_to_executor(lib.upload_ssh_key_payload(lib.miner_keypair(), lib.validator_keypair(), pub, bound_to=None))
    assert r.status_code == 403, r.text


def test_executor_refuses_a_binding_signed_by_a_stranger_403():
    _, pub = lib.ssh_keypair()
    body = lib.upload_ssh_key_payload(lib.miner_keypair(), lib.validator_keypair(), pub)
    body["validator_miner_signature"] = lib.sign(lib.stranger_keypair(), lib.ssh_pubkey_signing_blob(pub, None, lib.MINER_HOTKEY))
    r = _upload_ssh_key_to_executor(body)
    assert r.status_code == 403, r.text


def _dated_nonce(issued_at: int) -> str:
    # the shape of datura's fresh_upload_nonce, minted at `issued_at`
    import secrets

    from datura.requests.validator_requests import UPLOAD_NONCE_MAGIC

    return (UPLOAD_NONCE_MAGIC + issued_at.to_bytes(8, "big") + secrets.token_bytes(20)).hex()


def test_executor_accepts_a_fresh_dated_nonce():
    from datura.requests.validator_requests import fresh_upload_nonce

    priv, pub = lib.ssh_keypair()
    body = lib.upload_ssh_key_payload(lib.miner_keypair(), lib.validator_keypair(), pub, nonce=fresh_upload_nonce())
    r = _upload_ssh_key_to_executor(body)
    assert r.status_code == 200, r.text
    try:
        rc, _ = lib.run(lib.ssh_run(lib.EXECUTOR_IP, r.json()["ssh_port"], r.json()["ssh_username"], priv, "true"))
        assert rc == 0
    finally:
        _remove_ssh_key_from_executor(body)


def test_executor_refuses_a_stale_nonce_401():
    """A captured upload is worth at most UPLOAD_NONCE_MAX_AGE_SECONDS (600 s): one minted an hour ago is refused."""
    _, pub = lib.ssh_keypair()
    r = _upload_ssh_key_to_executor(lib.upload_ssh_key_payload(lib.miner_keypair(), lib.validator_keypair(), pub, nonce=_dated_nonce(int(time.time()) - 3600)))
    assert r.status_code == 401 and "nonce" in r.text.lower(), r.text


def test_executor_refuses_a_stripped_or_swapped_nonce_401():
    """The validator signature covers the nonce: dropping it or replacing it in transit breaks the signature."""
    from datura.requests.validator_requests import fresh_upload_nonce

    _, pub = lib.ssh_keypair()
    signed = lib.upload_ssh_key_payload(lib.miner_keypair(), lib.validator_keypair(), pub, nonce=fresh_upload_nonce())
    stripped = {k: v for k, v in signed.items() if k != "nonce"}
    r = _upload_ssh_key_to_executor(stripped)
    assert r.status_code == 401, r.text
    r = _upload_ssh_key_to_executor({**signed, "nonce": fresh_upload_nonce()})
    assert r.status_code == 401, r.text
