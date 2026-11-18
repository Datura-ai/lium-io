"""DAH-6430: POST /executors accepts the signed headers (bound to this miner and a 10 s window); the
signature over the validator's own hotkey stays accepted only while ACCEPT_UNBOUND_SIGNATURES."""

import asyncio
import time
from unittest.mock import MagicMock

import bittensor
import pytest
from datura.requests.validator_requests import AuthenticationPayload, SimpleValidatorRequest
from fastapi import HTTPException

import dependencies.auth as auth_module

VALIDATOR = bittensor.Keypair.create_from_uri("//LiumTestValidatorExecList")
MINER = bittensor.Keypair.create_from_uri("//LiumTestMinerExecList")
OTHER = bittensor.Keypair.create_from_uri("//LiumTestOtherMinerExecList")


@pytest.fixture(autouse=True, params=[False, True], ids=["standard", "central"])
def this_miner(monkeypatch, request):
    wallet = MagicMock()
    wallet.get_hotkey.return_value = MINER
    settings = MagicMock(CENTRAL_MODE=request.param)
    settings.get_bittensor_wallet.return_value = wallet
    monkeypatch.setattr(auth_module.core_config, "settings", settings)


def _service():
    service = MagicMock()
    service.is_valid_validator.return_value = True
    return service


def _legacy():
    return SimpleValidatorRequest(
        signature=f"0x{VALIDATOR.sign(VALIDATOR.ss58_address).hex()}", validator_hotkey=VALIDATOR.ss58_address
    )


def _verify(miner_hotkey=None, timestamp=None, request=None):
    kwargs = {}
    if miner_hotkey is not None:
        payload = AuthenticationPayload(
            validator_hotkey=VALIDATOR.ss58_address, miner_hotkey=miner_hotkey, timestamp=timestamp
        )
        kwargs = dict(
            x_validator_hotkey=VALIDATOR.ss58_address,
            x_miner_hotkey=miner_hotkey,
            x_timestamp=timestamp,
            x_signature=f"0x{VALIDATOR.sign(payload.blob_for_signing()).hex()}",
        )
    return asyncio.run(auth_module.verify_simple_validator_signature(request or _legacy(), _service(), **kwargs))


def test_signed_headers_for_this_miner_are_accepted():
    assert _verify(MINER.ss58_address, int(time.time())) == VALIDATOR.ss58_address


@pytest.mark.parametrize("miner,age", [("own", 60), ("other", 0)], ids=["stale", "another-miner"])
def test_stale_or_other_miner_headers_are_refused(miner, age, monkeypatch):
    hotkey = MINER.ss58_address if miner == "own" else OTHER.ss58_address
    # during the rollout the valid body signature still carries the request (e.g. a skewed clock)
    assert _verify(hotkey, int(time.time()) - age) == VALIDATOR.ss58_address

    monkeypatch.setattr(auth_module, "ACCEPT_UNBOUND_SIGNATURES", False)
    with pytest.raises(HTTPException) as err:
        _verify(hotkey, int(time.time()) - age)
    assert err.value.status_code in (401, 403)


def test_legacy_body_signature_only_while_accepted(monkeypatch):
    assert _verify() == VALIDATOR.ss58_address

    monkeypatch.setattr(auth_module, "ACCEPT_UNBOUND_SIGNATURES", False)
    with pytest.raises(HTTPException) as err:
        _verify()
    assert err.value.status_code == 401


def test_headers_for_another_miner_with_a_bad_body_signature_are_refused():
    # replayed headers: valid and fresh, but signed for another miner; the body signature is not valid
    forged = SimpleValidatorRequest(signature="0x" + "00" * 64, validator_hotkey=VALIDATOR.ss58_address)
    with pytest.raises(HTTPException) as err:
        _verify(OTHER.ss58_address, int(time.time()), request=forged)
    assert err.value.status_code == 401
