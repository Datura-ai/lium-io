"""The wire contract between validator, miner and executor — accepted for the right keys, refused for everything else.

What a real cycle does before any check runs: the validator signs into the miner (four headers over
AuthenticationPayload), the miner forwards the validator's SSH public key to each executor with its own signature on
top (/upload_ssh_key), the executor installs it and answers with its SSH coordinates, and at the end the key is
removed. Each step here is the real HTTP call the code makes, plus the refusals a spoofer must hit.
"""

import asyncio
import time

import pytest

from tests import lib

pytestmark = pytest.mark.timeout(120)


# ---------------------------------------------------------------- registration ---------------------------------------


def test_executor_answers_version():
    r = lib.http("GET", f"{lib.EXECUTOR_URL}/version")
    assert r.status_code == 200, r.text
    assert r.json(), "executor /version body is empty"


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
    # a valid signature over the wrong hotkey: the route only proves key ownership, and a stranger owns no executors
    assert r.status_code == 200 and r.json()["executors"] == [], r.text
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
    ex = body["executors"][0]
    assert ex["uuid"] == lib.EXECUTOR_UUID and ex["address"] == lib.EXECUTOR_IP
    assert ex["ssh_port"] == lib.EXECUTOR_SSH_PORT and ex["ssh_username"] and ex["python_path"] and ex["root_dir"]
    # the key the validator minted now opens the executor — the SSH hop every check runs over
    rc, out = asyncio.run(lib.ssh_run(ex["address"], ex["ssh_port"], ex["ssh_username"], priv, "id -un && test -d " + ex["root_dir"]))
    assert rc == 0 and out.strip() == ex["ssh_username"], (rc, out)
    # remove it the way the validator does at the end of the cycle
    r = lib.http("POST", f"{lib.MINER_URL}/api/validator/ssh-pubkey-remove", headers=lib.validator_rest_headers(vk),
                 json={"message_type": "SSHPubKeyRemoveRequest", "public_key": pub, "validator_signature": lib.sign(vk, pub), "miner_hotkey": lib.MINER_HOTKEY}, timeout=60)
    assert r.status_code == 200, r.text
    with pytest.raises(Exception):
        asyncio.run(lib.ssh_run(ex["address"], ex["ssh_port"], ex["ssh_username"], priv, "true", timeout=15))


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


def test_headers_for_another_miner_are_refused():
    vk = lib.validator_keypair()
    r, _, _ = _submit(lib.validator_rest_headers(vk, miner_hotkey=lib.stranger_keypair().ss58_address))
    assert r.status_code in (401, 403), r.text


# ------------------------------------------------------------ miner → executor /upload_ssh_key -------------------------


def test_executor_accepts_the_double_signature():
    priv, pub = lib.ssh_keypair()
    r = lib.http("POST", f"{lib.EXECUTOR_URL}/upload_ssh_key", json=lib.upload_ssh_key_payload(lib.miner_keypair(), lib.validator_keypair(), pub))
    assert r.status_code == 200, r.text
    info = r.json()
    assert info["ssh_port"] == lib.EXECUTOR_SSH_PORT and info["ssh_username"]
    rc, _ = asyncio.run(lib.ssh_run(lib.EXECUTOR_IP, info["ssh_port"], info["ssh_username"], priv, "true"))
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
    assert r.status_code in (401, 403), r.text


def test_executor_refuses_a_substituted_public_key():
    """Issue #744: a valid miner signature over one key must not install a different key."""
    _, pub = lib.ssh_keypair()
    _, other = lib.ssh_keypair()
    body = lib.upload_ssh_key_payload(lib.miner_keypair(), lib.validator_keypair(), pub)
    body["public_key"] = other
    body["validator_signature"] = lib.sign(lib.validator_keypair(), other)
    r = lib.http("POST", f"{lib.EXECUTOR_URL}/upload_ssh_key", json=body)
    assert r.status_code in (400, 401), r.text
