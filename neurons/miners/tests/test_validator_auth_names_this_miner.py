"""
A validator sign-in is accepted only when its signed payload names this miner, and every message after it
names the same miner (WebSocket consumer and REST routes).

Ephemeral keypairs sign the real payload shape (AuthenticationPayload.blob_for_signing); the consumer's socket
and the services are the only things replaced. The REST dependency and the route functions are called directly
(the miners lock has no httpx, so no TestClient); lium-io#1312's e2e stack exercises the same routes over HTTP.
"""

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import bittensor
import pytest
from datura.requests.miner_requests import AcceptJobRequest, UnAuthorizedRequest
from datura.requests.validator_requests import (
    AuthenticateRequest,
    AuthenticationPayload,
    GetPodLogsRequest,
    SSHPubKeyRemoveRequest,
    SSHPubKeySubmitRequest,
)
from fastapi import HTTPException

import consumers.validator_consumer as consumer_module
import dependencies.auth as auth_module
import routes.validator_interface as routes
from consumers.validator_consumer import ValidatorConsumer
from services.executor_service import ExecutorService
from services.ssh_service import MinerSSHService
from services.validator_service import ValidatorService

_CONTAINER = "pod_0b2a6d1e-9e2e-4e2e-8e2e-000000000e2e"
_EXECUTOR_ID = "0b2a6d1e-9e2e-4e2e-8e2e-000000000e2e"


@pytest.fixture(scope="module")
def validator_keypair():
    return bittensor.Keypair.create_from_uri("//LiumTestValidatorNames")


@pytest.fixture(scope="module")
def miner_keypair():
    return bittensor.Keypair.create_from_uri("//LiumTestMinerNames")


@pytest.fixture(scope="module")
def other_miner_keypair():
    return bittensor.Keypair.create_from_uri("//LiumTestOtherMinerNames")


def _settings(miner_keypair, central_mode: bool) -> MagicMock:
    # Pydantic Settings refuses setattr of non-fields, so the module's settings object is replaced whole
    wallet = MagicMock()
    wallet.get_hotkey.return_value = miner_keypair
    settings = MagicMock()
    settings.get_bittensor_wallet.return_value = wallet
    settings.CENTRAL_MODE = central_mode
    settings.RENTAL_REQUEST_HOOK = None
    return settings


def _auth_request(validator_keypair, miner_hotkey: str, *, signer=None):
    payload = AuthenticationPayload(
        validator_hotkey=validator_keypair.ss58_address,
        miner_hotkey=miner_hotkey,
        timestamp=int(time.time()),
    )
    signer = signer or validator_keypair
    return AuthenticateRequest(
        payload=payload, signature=f"0x{signer.sign(payload.blob_for_signing()).hex()}"
    )


# ---------------------------------------------------------------- WebSocket consumer ----------------------------------


@pytest.fixture()
def consumer(validator_keypair, miner_keypair, monkeypatch):
    def make(central_mode: bool = False) -> ValidatorConsumer:
        monkeypatch.setattr(consumer_module, "settings", _settings(miner_keypair, central_mode))
        validator_service = MagicMock(spec=ValidatorService)
        validator_service.is_valid_validator.return_value = True
        executor_service = MagicMock(spec=ExecutorService)
        # one executor, so an accepted sign-in is answered with AcceptJobRequest and the session stays open
        executor_service.get_executors_for_validator = AsyncMock(
            return_value=[
                SimpleNamespace(uuid=UUID(_EXECUTOR_ID), address="203.0.113.10", port=8001)
            ]
        )
        executor_service.get_pod_logs = AsyncMock(return_value=[])
        executor_service.register_pubkey = AsyncMock(return_value=[])
        executor_service.deregister_pubkey = AsyncMock(return_value=None)
        c = ValidatorConsumer(
            websocket=MagicMock(),
            validator_key=validator_keypair.ss58_address,
            ssh_service=MagicMock(spec=MinerSSHService),
            validator_service=validator_service,
            executor_service=executor_service,
        )
        c.send_message = AsyncMock()
        c.disconnect = AsyncMock()
        return c

    return make


def test_a_sign_in_naming_this_miner_verifies(consumer, validator_keypair, miner_keypair):
    c = consumer()
    assert c.verify_auth_msg(_auth_request(validator_keypair, miner_keypair.ss58_address)) == (
        True,
        "",
    )


def test_a_sign_in_naming_another_miner_is_refused(
    consumer, validator_keypair, other_miner_keypair
):
    """The validator signed this payload for another miner; a standard miner is not that miner."""
    c = consumer()
    ok, reason = c.verify_auth_msg(
        _auth_request(validator_keypair, other_miner_keypair.ss58_address)
    )
    assert ok is False
    assert reason.startswith("wrong miner hotkey"), reason


def test_a_bad_signature_is_a_refusal_not_none(consumer, validator_keypair, miner_keypair):
    stranger = bittensor.Keypair.create_from_uri("//LiumTestStrangerNames")
    c = consumer()
    assert c.verify_auth_msg(
        _auth_request(validator_keypair, miner_keypair.ss58_address, signer=stranger)
    ) == (
        False,
        "invalid signature",
    )


def test_the_central_miner_accepts_a_sign_in_for_any_portal_miner(
    consumer, validator_keypair, other_miner_keypair
):
    """CENTRAL_MODE serves many hotkeys; the name is bound to the session instead (tests below)."""
    c = consumer(central_mode=True)
    assert c.verify_auth_msg(
        _auth_request(validator_keypair, other_miner_keypair.ss58_address)
    ) == (True, "")


@pytest.mark.asyncio
async def test_a_refused_sign_in_closes_the_socket_before_anything_runs(
    consumer, validator_keypair, other_miner_keypair
):
    c = consumer()
    await c.handle_message(_auth_request(validator_keypair, other_miner_keypair.ss58_address))

    assert c.validator_authenticated is False
    sent = c.send_message.await_args.args[0]
    assert isinstance(sent, UnAuthorizedRequest) and "wrong miner hotkey" in sent.details
    c.disconnect.assert_awaited_once()
    c.executor_service.get_executors_for_validator.assert_not_awaited()


@pytest.mark.asyncio
async def test_after_sign_in_a_request_naming_another_miner_is_refused(
    consumer, validator_keypair, miner_keypair, other_miner_keypair
):
    c = consumer(central_mode=True)
    await c.handle_message(_auth_request(validator_keypair, miner_keypair.ss58_address))
    assert (
        c.validator_authenticated is True
        and c.authenticated_miner_hotkey == miner_keypair.ss58_address
    )
    c.send_message.reset_mock()

    await c.handle_message(
        GetPodLogsRequest(
            executor_id=_EXECUTOR_ID,
            container_name=_CONTAINER,
            miner_hotkey=other_miner_keypair.ss58_address,
        )
    )

    c.executor_service.get_pod_logs.assert_not_awaited()
    sent = c.send_message.await_args.args[0]
    assert (
        isinstance(sent, UnAuthorizedRequest) and other_miner_keypair.ss58_address in sent.details
    )
    c.disconnect.assert_awaited_once()
    assert c.validator_authenticated is False, "a refused message ends the session"


@pytest.mark.asyncio
async def test_a_request_queued_before_sign_in_is_held_to_the_same_name(
    consumer, validator_keypair, miner_keypair, other_miner_keypair
):
    """Queued messages run right after authentication; they get the same binding, not a free pass."""
    c = consumer(central_mode=True)
    await c.handle_message(
        SSHPubKeySubmitRequest(
            public_key=b"ssh-ed25519 AAAA test",
            validator_signature="0x00",
            miner_hotkey=other_miner_keypair.ss58_address,
        )
    )
    assert c.msg_queue and c.validator_authenticated is False

    await c.handle_message(_auth_request(validator_keypair, miner_keypair.ss58_address))

    c.executor_service.register_pubkey.assert_not_awaited()
    assert any(
        isinstance(call.args[0], UnAuthorizedRequest) for call in c.send_message.await_args_list
    )
    c.disconnect.assert_awaited_once()
    # the refusal ended the session before the executor list was offered
    c.executor_service.get_executors_for_validator.assert_not_awaited()


@pytest.mark.asyncio
async def test_requests_naming_the_signed_in_miner_are_served(
    consumer, validator_keypair, miner_keypair
):
    c = consumer()
    await c.handle_message(_auth_request(validator_keypair, miner_keypair.ss58_address))
    assert isinstance(c.send_message.await_args.args[0], AcceptJobRequest)

    await c.handle_message(
        GetPodLogsRequest(
            executor_id=_EXECUTOR_ID,
            container_name=_CONTAINER,
            miner_hotkey=miner_keypair.ss58_address,
        )
    )
    await c.handle_message(
        SSHPubKeyRemoveRequest(
            public_key=b"ssh-ed25519 AAAA test",
            validator_signature="0x00",
            miner_hotkey=miner_keypair.ss58_address,
        )
    )

    c.executor_service.get_pod_logs.assert_awaited_once_with(
        validator_keypair.ss58_address, miner_keypair.ss58_address, _EXECUTOR_ID, _CONTAINER
    )
    c.executor_service.deregister_pubkey.assert_awaited_once()
    c.disconnect.assert_not_awaited()


# ---------------------------------------------------------------- REST dependency and routes -------------------------


def _headers(validator_keypair, miner_hotkey: str, *, signer=None) -> dict:
    """The four headers MinerService._generate_auth_headers sends, as the dependency receives them."""
    payload = AuthenticationPayload(
        validator_hotkey=validator_keypair.ss58_address,
        miner_hotkey=miner_hotkey,
        timestamp=int(time.time()),
    )
    signer = signer or validator_keypair
    return {
        "x_validator_hotkey": validator_keypair.ss58_address,
        "x_miner_hotkey": miner_hotkey,
        "x_timestamp": payload.timestamp,
        "x_signature": f"0x{signer.sign(payload.blob_for_signing()).hex()}",
    }


@pytest.fixture()
def rest(miner_keypair, monkeypatch):
    def make(central_mode: bool = False) -> tuple[MagicMock, MagicMock]:
        monkeypatch.setattr(
            auth_module.core_config, "settings", _settings(miner_keypair, central_mode)
        )
        validator_service = MagicMock(spec=ValidatorService)
        validator_service.is_valid_validator.return_value = True
        executor_service = MagicMock(spec=ExecutorService)
        executor_service.get_pod_logs = AsyncMock(return_value=[])
        executor_service.register_pubkey = AsyncMock(return_value=[])
        executor_service.deregister_pubkey = AsyncMock(return_value=None)
        return validator_service, executor_service

    return make


async def test_rest_headers_naming_this_miner_authenticate(rest, validator_keypair, miner_keypair):
    validator_service, _ = rest()
    headers = _headers(validator_keypair, miner_keypair.ss58_address)

    assert (
        await auth_module.verify_validator_auth_from_headers(
            **headers, validator_service=validator_service
        )
        == validator_keypair.ss58_address
    )
    assert (
        await auth_module.authenticated_miner_hotkey(
            headers["x_miner_hotkey"], validator_keypair.ss58_address
        )
        == miner_keypair.ss58_address
    )


async def test_rest_headers_naming_another_miner_are_refused_403(
    rest, validator_keypair, other_miner_keypair
):
    """The docstring always promised this 403; the signature is valid, the miner is not us."""
    validator_service, _ = rest()

    with pytest.raises(HTTPException) as refused:
        await auth_module.verify_validator_auth_from_headers(
            **_headers(validator_keypair, other_miner_keypair.ss58_address),
            validator_service=validator_service,
        )

    assert refused.value.status_code == 403
    assert refused.value.detail == "Authentication names another miner"


async def test_rest_central_miner_accepts_headers_for_any_portal_miner(
    rest, validator_keypair, other_miner_keypair
):
    validator_service, _ = rest(central_mode=True)
    headers = _headers(validator_keypair, other_miner_keypair.ss58_address)

    assert (
        await auth_module.verify_validator_auth_from_headers(
            **headers, validator_service=validator_service
        )
        == validator_keypair.ss58_address
    )


async def test_rest_a_stranger_signature_is_still_401_for_this_miner(
    rest, validator_keypair, miner_keypair
):
    validator_service, _ = rest()
    stranger = bittensor.Keypair.create_from_uri("//LiumTestStrangerNames")

    with pytest.raises(HTTPException) as refused:
        await auth_module.verify_validator_auth_from_headers(
            **_headers(validator_keypair, miner_keypair.ss58_address, signer=stranger),
            validator_service=validator_service,
        )

    assert refused.value.status_code == 401


def _pod_logs_request(miner_hotkey: str) -> GetPodLogsRequest:
    return GetPodLogsRequest(
        executor_id=_EXECUTOR_ID, container_name=_CONTAINER, miner_hotkey=miner_hotkey
    )


async def test_rest_pod_logs_names_the_signed_in_miner_and_is_served(
    rest, validator_keypair, miner_keypair
):
    _, executor_service = rest()
    me = miner_keypair.ss58_address

    response = await routes.get_pod_logs(
        _pod_logs_request(me),
        authenticated_validator=validator_keypair.ss58_address,
        authenticated_miner=me,
        executor_service=executor_service,
    )

    assert response.message_type.value == "PodLogsResponse"
    executor_service.get_pod_logs.assert_awaited_once_with(
        validator_keypair.ss58_address, me, _EXECUTOR_ID, _CONTAINER
    )


async def test_rest_body_naming_a_miner_the_headers_did_not_is_refused_403(
    rest, validator_keypair, miner_keypair, other_miner_keypair
):
    """CENTRAL_MODE: the headers may name any portal miner, the body must name that same one — on all three routes,
    and as a 403, not as a FailedRequest body the route's except would otherwise turn it into."""
    _, executor_service = rest(central_mode=True)
    me, other = miner_keypair.ss58_address, other_miner_keypair.ss58_address
    vk = validator_keypair.ss58_address
    submit = SSHPubKeySubmitRequest(
        public_key=b"ssh-ed25519 AAAA test", validator_signature="0x00", miner_hotkey=other
    )
    remove = SSHPubKeyRemoveRequest(
        public_key=b"ssh-ed25519 AAAA test", validator_signature="0x00", miner_hotkey=other
    )

    for call in (
        routes.get_pod_logs(
            _pod_logs_request(other),
            authenticated_validator=vk,
            authenticated_miner=me,
            executor_service=executor_service,
        ),
        routes.submit_ssh_pubkey(
            submit,
            authenticated_validator=vk,
            authenticated_miner=me,
            executor_service=executor_service,
            ssh_service=MagicMock(),
        ),
        routes.remove_ssh_pubkey(
            remove,
            authenticated_validator=vk,
            authenticated_miner=me,
            executor_service=executor_service,
        ),
    ):
        with pytest.raises(HTTPException) as refused:
            await call
        assert refused.value.status_code == 403
        assert refused.value.detail == "Request names a miner the authentication did not"

    executor_service.get_pod_logs.assert_not_awaited()
    executor_service.register_pubkey.assert_not_awaited()
    executor_service.deregister_pubkey.assert_not_awaited()


def test_require_request_names_authenticated_miner_accepts_the_same_name(miner_keypair):
    auth_module.require_request_names_authenticated_miner(
        miner_keypair.ss58_address, miner_keypair.ss58_address
    )


def _dependency_names(dependant) -> set[str]:
    names = {dependant.call.__name__} if dependant.call is not None else set()
    for sub in dependant.dependencies:
        names |= _dependency_names(sub)
    return names


def test_the_three_rest_routes_are_wired_to_the_miner_binding():
    """FastAPI resolves both dependencies on each route: the signed headers and the miner hotkey they named."""
    from fastapi.routing import APIRoute

    wanted = {
        "/api/validator/ssh-pubkey-submit",
        "/api/validator/ssh-pubkey-remove",
        "/api/validator/pod-logs",
    }
    seen = set()
    for route in routes.validator_router.routes:
        if isinstance(route, APIRoute) and route.path in wanted:
            names = _dependency_names(route.dependant)
            assert {"verify_validator_auth_from_headers", "authenticated_miner_hotkey"} <= names, (
                route.path,
                names,
            )
            seen.add(route.path)
    assert seen == wanted
