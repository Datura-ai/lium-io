"""
Tests for ExecutorService SSH key payload construction.

Focus: verify that validator_signature is correctly forwarded in the POST
body sent to the executor when registering or removing SSH public keys.
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import bittensor
import aiohttp
import pytest
from datura.requests.miner_requests import ExecutorSSHInfo

from models.executor import Executor
from protocol.miner_portal_request import (
    DeleteExecutorRequest,
    ExecutorDeleted,
    SyncExecutorPayload,
)
import services.executor_service as es
from services.executor_service import ExecutorService


_SSH_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITestKey123456789abcdef user@host"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def validator_keypair():
    return bittensor.Keypair.create_from_uri("//LiumTestValidator")


@pytest.fixture(scope="module")
def miner_keypair():
    return bittensor.Keypair.create_from_uri("//LiumTestMiner")


@pytest.fixture
def executor_service(miner_keypair, monkeypatch):
    """ExecutorService with mocked DAO and wallet — no real DB or wallet needed."""
    service = ExecutorService.__new__(ExecutorService)
    service.executor_dao = MagicMock()
    service.ssh_service = MagicMock()

    # Replace settings in executor_service with a mock (Pydantic Settings does not allow
    # setattr of non-fields like get_bittensor_wallet).
    import services.executor_service as es_module

    mock_wallet = MagicMock()
    mock_wallet.get_hotkey.return_value = miner_keypair
    mock_settings = MagicMock()
    mock_settings.get_bittensor_wallet.return_value = mock_wallet
    mock_settings.CENTRAL_MODE = False
    monkeypatch.setattr(es_module, "settings", mock_settings)

    return service


@pytest.fixture
def test_executor():
    return Executor(
        uuid=uuid4(),
        validator="//TestValidator",
        address="127.0.0.1",
        port=8001,
    )


_SSH_ANSWER = json.dumps({"ssh_username": "root", "ssh_port": 22, "python_path": "/p", "root_dir": "/r"}).encode()


def _response(status: int = 200, body: bytes = b"", error: Exception | None = None, content_length: int | None = None):
    """A mock aiohttp response whose body streams as one chunk, or raises `error` while reading."""
    response = MagicMock(status=status, content_length=content_length)

    async def chunks(_size):
        if error is not None:
            raise error
        yield body

    response.content.iter_chunked = chunks
    return response


def _make_mock_session(response_status: int = 500):
    """Return a mock aiohttp.ClientSession that captures the POST payload."""
    mock_response = _response(response_status, b"test error")

    mock_post_ctx = MagicMock()
    mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post_ctx.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    mock_session.post = MagicMock(return_value=mock_post_ctx)

    return mock_session


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_pubkey_to_executor_includes_validator_signature(
    executor_service, test_executor, validator_keypair
):
    """validator_signature is forwarded verbatim in the POST body to the executor."""
    # Arrange
    validator_sig = "0x" + validator_keypair.sign(_SSH_KEY).hex()
    mock_session = _make_mock_session()

    # Act
    with patch("services.executor_service.aiohttp.ClientSession", return_value=mock_session):
        await executor_service.send_pubkey_to_executor(test_executor, _SSH_KEY, validator_sig)

    # Assert — the executor was called with a JSON body
    assert mock_session.post.called
    sent_payload = mock_session.post.call_args.kwargs["json"]

    # Assert — validator_signature is present and unchanged in the forwarded payload
    assert "validator_signature" in sent_payload
    assert sent_payload["validator_signature"] == validator_sig


@pytest.mark.asyncio
async def test_send_pubkey_to_executor_relays_the_miner_bound_signature(executor_service, test_executor):
    """miner_hotkey and validator_miner_signature reach the executor verbatim; absent when not sent."""
    mock_session = _make_mock_session()
    with patch("services.executor_service.aiohttp.ClientSession", return_value=mock_session):
        await executor_service.send_pubkey_to_executor(
            test_executor, _SSH_KEY, "0xaa", miner_hotkey="5Miner", validator_miner_signature="0xbb"
        )
        sent = mock_session.post.call_args.kwargs["json"]
        assert (sent["miner_hotkey"], sent["validator_miner_signature"]) == ("5Miner", "0xbb")

        await executor_service.send_pubkey_to_executor(test_executor, _SSH_KEY, "0xaa")
        sent = mock_session.post.call_args.kwargs["json"]
        assert "miner_hotkey" not in sent and "validator_miner_signature" not in sent


def _identity_answer(session, hotkey: str | None, status: int = 200):
    """Make the mock session's GET /miner-identity answer `hotkey`, echoing the request's nonce."""

    def get(url, params=None, timeout=None, allow_redirects=True):
        assert url.endswith("/miner-identity")
        assert allow_redirects is False  # a node's redirect is never followed
        answer = {"miner_hotkey": hotkey, "nonce": params["nonce"]} if hotkey else {}
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=_response(status, json.dumps(answer).encode()))
        ctx.__aexit__ = AsyncMock(return_value=None)
        return ctx

    session.get = MagicMock(side_effect=get)


def _post_answers(session, *answers):
    """POST /upload_ssh_key answers these responses in turn; session.sent keeps a copy of each body as sent."""
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(side_effect=list(answers))
    ctx.__aexit__ = AsyncMock(return_value=None)
    session.sent = []

    def post(url, json=None, timeout=None, allow_redirects=True):
        assert allow_redirects is False  # a node's redirect is never followed
        session.sent.append(dict(json))
        return ctx

    session.post = MagicMock(side_effect=post)


@pytest.fixture(autouse=True)
def _no_known_identities():
    es._executor_own_hotkeys.clear()
    yield
    es._executor_own_hotkeys.clear()


@pytest.mark.asyncio
async def test_a_big_owner_map_reaches_the_executor_as_its_one_entry(executor_service, test_executor):
    """DAH-6740: 2,000 owner hotkeys (~370 KiB of signatures, past the executor's 256 KiB body limit) relay as one entry,
    and the upload never waits on the identity read."""
    signatures = {f"5Owner{i:05d}" + "x" * 41: "0x" + "ab" * 64 for i in range(2000)}
    assert len(json.dumps(signatures)) > 256 * 1024
    own = sorted(signatures)[1234]
    mock_session = _make_mock_session()
    with patch("services.executor_service.aiohttp.ClientSession", return_value=mock_session):
        _identity_answer(mock_session, own)
        # identity not known yet: the upload goes at once without an owner entry, the read runs beside it
        await executor_service.send_pubkey_to_executor(test_executor, _SSH_KEY, "0xaa", owner_hotkey_signatures=signatures)
        assert "owner_hotkey_signatures" not in mock_session.post.call_args.kwargs["json"]
        await asyncio.gather(*es._identity_reads)

        # learned: later uploads carry just that entry with no further read
        mock_session.get.reset_mock()
        await executor_service.send_pubkey_to_executor(test_executor, _SSH_KEY, "0xaa", owner_hotkey_signatures=signatures)
        sent = mock_session.post.call_args.kwargs["json"]
        assert sent["owner_hotkey_signatures"] == {own: signatures[own]}
        assert len(json.dumps(sent)) < 256 * 1024
        assert not mock_session.get.called

        # no owner map: no identity read at all
        es._executor_own_hotkeys.clear()
        await executor_service.send_pubkey_to_executor(test_executor, _SSH_KEY, "0xaa")
        assert not mock_session.get.called
        assert "owner_hotkey_signatures" not in mock_session.post.call_args.kwargs["json"]


@pytest.mark.asyncio
async def test_a_slow_identity_read_never_holds_a_listing_bound_upload(executor_service, test_executor):
    """The upload answers while /miner-identity is still hanging; the read finishes on its own afterwards."""
    release = asyncio.Event()
    session = _make_mock_session()

    def hanging_get(url, params=None, timeout=None, allow_redirects=True):
        async def wait():
            await release.wait()
            raise asyncio.TimeoutError()

        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(side_effect=wait)
        ctx.__aexit__ = AsyncMock(return_value=None)
        return ctx

    session.get = MagicMock(side_effect=hanging_get)
    _post_answers(session, _response(200, _SSH_ANSWER))
    with patch("services.executor_service.aiohttp.ClientSession", return_value=session):
        info = await asyncio.wait_for(
            executor_service.send_pubkey_to_executor(
                test_executor, _SSH_KEY, "0xaa", owner_hotkey_signatures={"5Owner": "0xcc"}
            ),
            timeout=1,
        )
        assert info is not None and session.post.call_count == 1
        release.set()
        await asyncio.gather(*es._identity_reads)
    assert str(test_executor.uuid) not in es._executor_own_hotkeys


@pytest.mark.asyncio
async def test_a_binding_refusal_retries_with_the_executors_one_entry(executor_service, test_executor):
    """A 403 makes the miner read the executor's hotkey and retry once with that entry; a hotkey outside the owner
    set (or a second 403) ends the upload."""
    signatures = {"5OwnerA": "0xaa01", "5OwnerB": "0xbb02"}
    refused = lambda: _response(403, b'{"detail":"SSH key was not issued for this miner"}')  # noqa: E731
    session = _make_mock_session()
    with patch("services.executor_service.aiohttp.ClientSession", return_value=session), \
            patch("services.executor_service._learn_own_hotkey_later"):
        _identity_answer(session, "5OwnerB")
        _post_answers(session, refused(), _response(200, _SSH_ANSWER))
        info = await executor_service.send_pubkey_to_executor(test_executor, _SSH_KEY, "0xaa", owner_hotkey_signatures=signatures)
        assert info is not None and session.post.call_count == 2
        first, second = session.sent
        assert "owner_hotkey_signatures" not in first
        assert second["owner_hotkey_signatures"] == {"5OwnerB": "0xbb02"}

        # the learned entry refused again, then a hotkey outside the owner set: no second retry either way
        for hotkey in ("5OwnerB", "5Elsewhere"):
            _identity_answer(session, hotkey)
            _post_answers(session, refused(), refused())
            assert await executor_service.send_pubkey_to_executor(
                test_executor, _SSH_KEY, "0xaa", owner_hotkey_signatures=signatures
            ) is None
            assert session.post.call_count == 1 and session.get.call_count == 1


@pytest.mark.asyncio
async def test_a_refusal_on_the_last_attempt_still_sends_the_signed_retry(executor_service, test_executor):
    """Two dropped uploads, then a binding refusal on the third: the one-entry retry is its own step and is sent."""
    signatures = {"5OwnerA": "0xaa01", "5OwnerB": "0xbb02"}
    refused = _response(403, b'{"detail":"SSH key was not issued for this miner"}')
    session = _make_mock_session()
    with patch("services.executor_service.aiohttp.ClientSession", return_value=session), \
            patch("services.executor_service._learn_own_hotkey_later"), \
            patch("services.executor_service.asyncio.sleep", AsyncMock()):
        _identity_answer(session, "5OwnerB")
        _post_answers(session, asyncio.TimeoutError(), asyncio.TimeoutError(), refused, _response(200, _SSH_ANSWER))
        info = await executor_service.send_pubkey_to_executor(
            test_executor, _SSH_KEY, "0xaa", owner_hotkey_signatures=signatures, deadline=time.monotonic() + 60
        )
    assert info is not None and session.post.call_count == es.UPLOAD_SSH_KEY_ATTEMPTS + 1
    assert session.sent[-1]["owner_hotkey_signatures"] == {"5OwnerB": "0xbb02"}


@pytest.mark.asyncio
async def test_a_failed_identity_read_after_a_refusal_is_read_again_within_the_deadline(executor_service, test_executor):
    """A 503 from /miner-identity after a refusal is read again while time remains; with no time left it gives up."""
    signatures = {"5OwnerA": "0xaa01", "5OwnerB": "0xbb02"}
    refused = lambda: _response(403, b'{"detail":"SSH key was not issued for this miner"}')  # noqa: E731
    session = _make_mock_session()
    answers = iter([None, "5OwnerB"])

    def flaky_get(url, params=None, timeout=None, allow_redirects=True):
        hotkey = next(answers)
        body = json.dumps({"miner_hotkey": hotkey, "nonce": params["nonce"]} if hotkey else {}).encode()
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=_response(200 if hotkey else 503, body))
        ctx.__aexit__ = AsyncMock(return_value=None)
        return ctx

    with patch("services.executor_service.aiohttp.ClientSession", return_value=session), \
            patch("services.executor_service._learn_own_hotkey_later"), \
            patch("services.executor_service.asyncio.sleep", AsyncMock()):
        session.get = MagicMock(side_effect=flaky_get)
        _post_answers(session, refused(), _response(200, _SSH_ANSWER))
        info = await executor_service.send_pubkey_to_executor(
            test_executor, _SSH_KEY, "0xaa", owner_hotkey_signatures=signatures, deadline=time.monotonic() + 60
        )
        assert info is not None and session.get.call_count == 2
        assert session.sent[-1]["owner_hotkey_signatures"] == {"5OwnerB": "0xbb02"}

        # too little time for another read plus the retry: one read, then the upload ends
        es._executor_own_hotkeys.clear()
        _identity_answer(session, None, status=503)
        _post_answers(session, refused())
        assert await executor_service.send_pubkey_to_executor(
            test_executor, _SSH_KEY, "0xaa", owner_hotkey_signatures=signatures, deadline=time.monotonic() + 5
        ) is None
        assert session.get.call_count == 1 and session.post.call_count == 1


@pytest.mark.parametrize(
    "status, body_error, expected_posts, accepted",
    [
        (None, None, 2, True),  # the request timed out before any answer
        (200, aiohttp.ClientPayloadError("Response payload is not completed"), 2, True),
        (200, json.JSONDecodeError("invalid JSON", "{", 1), 1, False),
        (503, None, 1, False),
    ],
)
@pytest.mark.asyncio
async def test_send_pubkey_to_executor_retries_an_executor_that_did_not_answer(
    executor_service, test_executor, status, body_error, expected_posts, accepted
):
    """A timed-out or cut-off upload is asked again, so one dropped request does not leave the node out of
    the answer; a complete bad answer is not; a node that never answers gives up inside the deadline."""
    answered = _response(200, _SSH_ANSWER)
    if status is None:
        first = asyncio.TimeoutError()
    else:
        bad_json = isinstance(body_error, json.JSONDecodeError)
        first = _response(status, b"{" if bad_json else b"unavailable", None if bad_json else body_error)
    post_ctx = MagicMock()
    post_ctx.__aenter__ = AsyncMock(side_effect=[first, answered])
    post_ctx.__aexit__ = AsyncMock(return_value=None)
    session = _make_mock_session()
    session.post = MagicMock(return_value=post_ctx)

    with patch("services.executor_service.aiohttp.ClientSession", return_value=session), \
            patch("services.executor_service.asyncio.sleep", AsyncMock()):
        info = await executor_service.send_pubkey_to_executor(test_executor, _SSH_KEY, "0xaa")
        assert (info is not None and info.uuid == str(test_executor.uuid)) is accepted
        assert session.post.call_count == expected_posts

        post_ctx.__aenter__ = AsyncMock(side_effect=asyncio.TimeoutError())
        session.post.reset_mock()
        assert await executor_service.send_pubkey_to_executor(test_executor, _SSH_KEY, "0xaa") is None
        assert session.post.call_count == es.UPLOAD_SSH_KEY_ATTEMPTS

        # no time left for a second attempt: one try, then the miner answers without the node
        session.post.reset_mock()
        assert await executor_service.send_pubkey_to_executor(
            test_executor, _SSH_KEY, "0xaa", deadline=time.monotonic() + 1
        ) is None
        assert session.post.call_count == 1



@pytest.mark.parametrize("content_length", [None, es.MAX_EXECUTOR_ANSWER_BYTES + 1])
@pytest.mark.asyncio
async def test_send_pubkey_reads_no_more_than_the_answer_limit(executor_service, test_executor, content_length):
    """An oversized answer, declared or streamed, is refused without reading it whole or asking again."""
    read = []

    async def endless(_size):
        while True:
            read.append(1)
            yield b"x" * 65536

    answer = _response(200, content_length=content_length)
    answer.content.iter_chunked = endless
    session = _make_mock_session()
    session.post.return_value.__aenter__ = AsyncMock(return_value=answer)

    with patch("services.executor_service.aiohttp.ClientSession", return_value=session):
        assert await executor_service.send_pubkey_to_executor(test_executor, _SSH_KEY, "0xaa") is None

    assert session.post.call_count == 1
    assert len(read) <= es.MAX_EXECUTOR_ANSWER_BYTES // 65536 + 1


@pytest.mark.parametrize("content_length", [None, es.MAX_POD_LOGS_ANSWER_BYTES + 1])
@pytest.mark.asyncio
async def test_get_pod_logs_reads_no_more_than_its_limit(executor_service, test_executor, content_length):
    """The /pod_logs answer is bounded too, at its own larger limit; a normal answer still parses."""
    read = []

    async def endless(_size):
        while True:
            read.append(1)
            yield b"x" * 65536

    answer = _response(200, content_length=content_length)
    answer.content.iter_chunked = endless
    session = _make_mock_session()
    session.post.return_value.__aenter__ = AsyncMock(return_value=answer)
    executor_service.get_executors_for_validator = AsyncMock(return_value=[test_executor])

    with patch("services.executor_service.aiohttp.ClientSession", return_value=session):
        with pytest.raises(es.ExecutorAnswerTooLarge):
            await executor_service.get_pod_logs("v", "m", str(test_executor.uuid), "c")
        assert len(read) <= es.MAX_POD_LOGS_ANSWER_BYTES // 65536 + 1

        session.post.return_value.__aenter__ = AsyncMock(return_value=_response(200, b"[]"))
        assert await executor_service.get_pod_logs("v", "m", str(test_executor.uuid), "c") == []


@pytest.mark.parametrize(
    "is_rental_request, hook, lookup_seconds, expected_left",
    [
        (False, None, 0, 25), (False, None, 15, 10),
        (False, "http://hook", 15, 10), (True, None, 15, 10),
        (True, "http://hook", 0, 20), (True, "http://hook", 15, 5),
    ],
)
@pytest.mark.asyncio
async def test_register_pubkey_reserves_the_hook_time_only_when_the_hook_runs(
    executor_service, test_executor, is_rental_request, hook, lookup_seconds, expected_left
):
    """Ordinary checks never call the rental hook, so their uploads keep the whole wait minus the reply margin."""
    clock = [100.0]

    async def slow_lookup(*_args, **_kwargs):
        clock[0] += lookup_seconds
        return [test_executor]

    es.settings.RENTAL_REQUEST_HOOK = hook
    executor_service.get_executors_for_validator = slow_lookup
    executor_service.send_pubkey_to_executor = AsyncMock(return_value=None)
    with patch("services.executor_service.time.monotonic", side_effect=lambda: clock[0]):
        await executor_service.register_pubkey("v", "m", _SSH_KEY.encode(), "0xaa", is_rental_request=is_rental_request)

    deadline = executor_service.send_pubkey_to_executor.call_args.kwargs["deadline"]
    assert deadline - clock[0] == expected_left
    # uploads plus a stalled hook still end inside the validator's 30 s wait, leaving its connect/auth and reply time
    hook_time = es.RENTAL_REQUEST_HOOK_TIMEOUT_SECONDS if is_rental_request and hook else 0
    overhead = es.UPLOAD_SSH_KEY_RESPONSE_MARGIN_SECONDS
    assert lookup_seconds + expected_left + hook_time + overhead <= es.VALIDATOR_SSH_SUBMIT_TIMEOUT_SECONDS


@pytest.mark.parametrize("is_rental_request", [False, True])
@pytest.mark.parametrize("hook", [None, "http://hook"])
@pytest.mark.parametrize("lookup_seconds", [0, 15])
@pytest.mark.asyncio
async def test_register_pubkey_first_upload_keeps_mains_full_timeout(
    executor_service, test_executor, is_rental_request, hook, lookup_seconds
):
    """Retries must not cost a first-attempt success main accepted: attempt 1 always gets the full 10 s."""
    clock = [100.0]

    async def slow_lookup(*_args, **_kwargs):
        clock[0] += lookup_seconds
        return [test_executor]

    answered = _response(200, _SSH_ANSWER)
    post_ctx = MagicMock()
    post_ctx.__aenter__ = AsyncMock(return_value=answered)
    post_ctx.__aexit__ = AsyncMock(return_value=None)
    session = _make_mock_session()
    session.post = MagicMock(return_value=post_ctx)
    es.settings.RENTAL_REQUEST_HOOK = hook
    executor_service.get_executors_for_validator = slow_lookup

    with patch("services.executor_service.aiohttp.ClientSession", return_value=session), \
            patch("services.executor_service.time.monotonic", side_effect=lambda: clock[0]):
        registration = await executor_service.register_pubkey(
            "v", "m", _SSH_KEY.encode(), "0xaa", is_rental_request=is_rental_request
        )

    assert len(registration.accepted) == 1
    assert session.post.call_args.kwargs["timeout"].total == es.UPLOAD_SSH_KEY_ATTEMPT_TIMEOUT_SECONDS


@pytest.mark.asyncio
async def test_register_pubkey_answers_by_the_deadline_with_the_uploads_that_finished(
    executor_service, test_executor, monkeypatch
):
    """DAH-5734 (4 Nov): dead executors kept the answer past the validator's wait and it skipped the whole hotkey."""
    dead = Executor(uuid=uuid4(), validator="//TestValidator", address="127.0.0.2", port=8001)
    accepted = ExecutorSSHInfo.parse_obj(
        {
            "ssh_username": "root", "ssh_port": 22, "python_path": "/p", "root_dir": "/r",
            **test_executor.model_dump(mode="json"),
        }
    )

    async def upload(executor, *_args, **_kwargs):
        if executor is dead:
            await asyncio.sleep(10)
        return accepted

    monkeypatch.setattr(es, "UPLOAD_SSH_KEY_BUDGET_SECONDS", 0.2)
    es.settings.RENTAL_REQUEST_HOOK = None
    executor_service.get_executors_for_validator = AsyncMock(return_value=[test_executor, dead])
    executor_service.send_pubkey_to_executor = upload

    started = time.monotonic()
    registration = await executor_service.register_pubkey("v", "m", _SSH_KEY.encode(), "0xaa")

    assert time.monotonic() - started < 1
    assert registration.accepted == [accepted]
    assert registration.known_executor_ids == [str(test_executor.uuid), str(dead.uuid)]


@pytest.mark.asyncio
async def test_register_pubkey_warns_with_the_ids_of_executors_cut_off_at_the_deadline(
    executor_service, test_executor, monkeypatch
):
    """A cut-off upload logs nothing itself; the provider needs the executor ids to find the slow node."""
    dead = Executor(uuid=uuid4(), validator="//TestValidator", address="192.0.2.2", port=8001)

    async def upload(executor, *_args, **_kwargs):
        if executor is dead:
            await asyncio.sleep(10)
        return None

    monkeypatch.setattr(es, "UPLOAD_SSH_KEY_BUDGET_SECONDS", 0.2)
    es.settings.RENTAL_REQUEST_HOOK = None
    executor_service.get_executors_for_validator = AsyncMock(return_value=[test_executor, dead])
    executor_service.send_pubkey_to_executor = upload

    with patch.object(es.logger, "warning") as warning:
        await executor_service.register_pubkey("v", "m", _SSH_KEY.encode(), "0xaa")

    assert warning.call_args.args[0].extra["cut_off_executor_ids"] == [str(dead.uuid)]


@pytest.mark.asyncio
async def test_register_pubkey_counts_the_deadline_from_the_validators_signed_timestamp(
    executor_service, test_executor, monkeypatch
):
    """DAH-5734 (4 Nov 14:16Z): picked up 6 s after the validator signed, the answer left as its 30 s ran out."""
    dead = Executor(uuid=uuid4(), validator="//TestValidator", address="127.0.0.2", port=8001)

    async def upload(executor, *_args, **_kwargs):
        if executor is dead:
            await asyncio.sleep(10)
        return None

    monkeypatch.setattr(es, "UPLOAD_SSH_KEY_BUDGET_SECONDS", 5.0)
    es.settings.RENTAL_REQUEST_HOOK = None
    executor_service.get_executors_for_validator = AsyncMock(return_value=[test_executor, dead])
    executor_service.send_pubkey_to_executor = upload

    started = time.monotonic()
    await executor_service.register_pubkey("v", "m", _SSH_KEY.encode(), "0xaa", requested_at=time.time() - 4.8)

    assert time.monotonic() - started < 1

@pytest.mark.asyncio
async def test_send_pubkey_does_not_retry_when_the_delay_overran_the_deadline(executor_service, test_executor):
    """A retry delay that ended past the deadline gives up instead of starting an attempt the validator cannot wait for."""
    clock = [100.0]

    async def late_sleep(_seconds):
        clock[0] += 30  # the event loop was busy: the 1 s retry delay ended well after the deadline

    answered = _response(200, _SSH_ANSWER)
    post_ctx = MagicMock()
    post_ctx.__aenter__ = AsyncMock(side_effect=[asyncio.TimeoutError(), answered])
    post_ctx.__aexit__ = AsyncMock(return_value=None)
    session = _make_mock_session()
    session.post = MagicMock(return_value=post_ctx)

    with patch("services.executor_service.aiohttp.ClientSession", return_value=session), \
            patch("services.executor_service.asyncio.sleep", late_sleep), \
            patch("services.executor_service.time.monotonic", side_effect=lambda: clock[0]):
        info = await executor_service.send_pubkey_to_executor(test_executor, _SSH_KEY, "0xaa", deadline=clock[0] + 20)

    assert info is None
    assert session.post.call_count == 1


# ---------------------------------------------------------------------------
# A failed write rolls the shared session back
# ---------------------------------------------------------------------------


def _payload(test_executor):
    return SyncExecutorPayload(
        uuid=test_executor.uuid,
        validator=test_executor.validator,
        address=test_executor.address,
        port=test_executor.port,
        price_per_gpu=0.5,
    )


# ---------------------------------------------------------------------------
# A portal delete removes only the node it names (ticket-0591)
# ---------------------------------------------------------------------------


def test_delete_removes_the_named_node_not_another_at_its_address(executor_service, test_executor):
    dao = executor_service.executor_dao
    dao.delete_by_uuid.return_value = False

    payload = _payload(test_executor)

    result = executor_service.delete(DeleteExecutorRequest(executor=payload))

    assert isinstance(result, ExecutorDeleted)
    dao.delete_by_uuid.assert_called_once_with(payload.uuid)
    dao.delete_by_address_port.assert_not_called()


@pytest.mark.parametrize(
    "messages, expected",
    [
        # an address edit after the provider changed price here, before the portal synced it
        ([{"address": "10.0.0.9", "port": 2222}], ("10.0.0.9", 2222, "5MinerValidator", 3.0)),
        # a portal price edit, then an address edit: the address edit undoes neither
        (
            [
                {"validator": "5PortalValidator", "address": "10.0.0.1", "port": 22, "price_per_gpu": 3.5},
                {"address": "10.0.0.9", "port": 2222},
            ],
            ("10.0.0.9", 2222, "5PortalValidator", 3.5),
        ),
        # a full update still writes every field it carries
        (
            [{"validator": "5PortalValidator", "address": "10.0.0.9", "port": 2222, "price_per_gpu": 2.5}],
            ("10.0.0.9", 2222, "5PortalValidator", 2.5),
        ),
    ],
    ids=["address_only", "price_edit_then_address_edit", "full_update"],
)
def test_an_update_writes_only_the_fields_the_portal_sent(tmp_path, messages, expected):
    from sqlmodel import Session, SQLModel, create_engine

    from daos.executor import ExecutorDao
    from protocol.miner_portal_request import ExecutorUpdated, UpdateExecutorRequest

    engine = create_engine(f"sqlite:///{tmp_path / 'update.db'}")
    SQLModel.metadata.create_all(engine)
    node_id = uuid4()
    with Session(engine) as session:
        session.add(Executor(uuid=node_id, address="10.0.0.1", port=22, validator="5MinerValidator", price_per_gpu=3.0))
        session.commit()
    with Session(engine) as session:
        service = ExecutorService.__new__(ExecutorService)
        service.executor_dao = ExecutorDao(session)

        for fields in messages:
            sent = UpdateExecutorRequest.model_validate_json(
                json.dumps({"executor": {"uuid": str(node_id), **fields}})
            )
            assert isinstance(service.update(sent), ExecutorUpdated)

    with Session(engine) as session:
        node = session.get(Executor, node_id)
        assert (node.address, node.port, node.validator, node.price_per_gpu) == expected


def test_an_update_for_a_node_not_on_this_miner_is_answered_as_such(executor_service, test_executor):
    from protocol.miner_portal_request import ExecutorUpdateFailed, UpdateExecutorRequest

    executor_service.executor_dao.find_by_uuid.return_value = None

    result = executor_service.update(UpdateExecutorRequest(executor=_payload(test_executor)))

    assert isinstance(result, ExecutorUpdateFailed)
    assert result.not_on_miner is True
    assert result.executor_id == test_executor.uuid
    executor_service.executor_dao.session.commit.assert_not_called()


@pytest.mark.asyncio
async def test_central_miner_lists_each_node_with_its_registered_host_key_from_one_portal_read(executor_service):
    """DAH-5729: the host key comes from the portal read the node list came from. The snapshot expiring while the
    uploads run (the second read would then be a slow bulk refresh) must not cause another portal call."""
    from datura.requests.miner_requests import ExecutorSSHInfo

    from clients.miner_portal_api import MinerPortalAPI

    keyed_node, unkeyed_node = str(uuid4()), str(uuid4())
    portal_items = [
        {"id": keyed_node.upper(), "validator_hotkey": "v", "executor_ip_address": "node.example.com",
         "executor_ip_port": "8001", "ssh_host_key": "ssh-ed25519 AAAAregistered",
         "node_hotkeys": ["5Account", "5Pool"], "account_claim": "claim-1"},
        {"id": unkeyed_node, "validator_hotkey": "v", "executor_ip_address": "203.0.113.5", "executor_ip_port": "8001"},
    ]
    es.settings.CENTRAL_MODE = True
    es.settings.RENTAL_REQUEST_HOOK = None
    reads = AsyncMock(side_effect=[portal_items, RuntimeError("portal read after the uploads")])

    async def answered(executor, *_args, **_kwargs):
        MinerPortalAPI._snapshot_fetched_at = None  # the cache expires mid-gather
        return ExecutorSSHInfo(
            uuid=str(executor.uuid), address=executor.address, port=executor.port, ssh_username="root",
            ssh_port=22, python_path="/p", root_dir="/r",
        )

    executor_service.send_pubkey_to_executor = answered
    with patch.object(MinerPortalAPI, "fetch_executors", reads):
        registration = await executor_service.register_pubkey("v", "m", _SSH_KEY.encode(), "0xaa")

    assert reads.await_count == 1
    by_id = {info.uuid: info for info in registration.accepted}
    assert by_id[keyed_node.lower()].expected_ssh_host_key == "ssh-ed25519 AAAAregistered"
    assert by_id[unkeyed_node].expected_ssh_host_key is None
    # DAH-6539: the account binding from the same read; none for a node the portal lists without it
    assert by_id[keyed_node.lower()].expected_node_hotkeys == ["5Account", "5Pool"]
    assert by_id[keyed_node.lower()].expected_account_claim == "claim-1"
    assert by_id[unkeyed_node].expected_node_hotkeys is None


@pytest.mark.asyncio
async def test_central_miner_gives_a_validator_nothing_of_a_miner_whose_nodes_are_assigned_elsewhere(executor_service):
    """Central mode accepts any signed X-Miner-Hotkey; what the request reaches is the portal's list of that miner's
    nodes assigned to the SIGNING validator. A validator naming a miner whose nodes it was not assigned acts on none."""
    from clients.miner_portal_api import MinerPortalAPI

    others_nodes = [
        {"id": str(uuid4()), "validator_hotkey": "assigned-validator", "executor_ip_address": "node.example.com",
         "executor_ip_port": "8001"},
    ]
    es.settings.CENTRAL_MODE = True
    es.settings.RENTAL_REQUEST_HOOK = None
    executor_service.send_pubkey_to_executor = AsyncMock()
    executor_service.remove_pubkey_from_executor = AsyncMock()

    with patch.object(MinerPortalAPI, "fetch_executors", AsyncMock(return_value=others_nodes)):
        assert await executor_service.get_executors_for_validator("signing-validator", "other-miner") == []
        registration = await executor_service.register_pubkey("signing-validator", "other-miner", _SSH_KEY.encode(), "0xaa")
        await executor_service.deregister_pubkey("signing-validator", "other-miner", _SSH_KEY.encode(), "0xaa")
        with pytest.raises(Exception, match="not found executor"):
            await executor_service.get_pod_logs("signing-validator", "other-miner", others_nodes[0]["id"], "c")

    assert registration.accepted == []
    executor_service.send_pubkey_to_executor.assert_not_awaited()
    executor_service.remove_pubkey_from_executor.assert_not_awaited()


@pytest.mark.asyncio
async def test_standalone_miner_sends_no_expected_host_key(executor_service, test_executor):
    from datura.requests.miner_requests import ExecutorSSHInfo

    executor_service.executor_dao.get_executors_for_validator.return_value = [test_executor]
    es.settings.RENTAL_REQUEST_HOOK = None
    executor_service.send_pubkey_to_executor = AsyncMock(return_value=ExecutorSSHInfo(
        uuid=str(test_executor.uuid), address="127.0.0.1", port=8001, ssh_username="root",
        ssh_port=22, python_path="/p", root_dir="/r",
    ))
    registration = await executor_service.register_pubkey("v", "m", _SSH_KEY.encode(), "0xaa")
    assert registration.accepted[0].expected_ssh_host_key is None


@pytest.mark.asyncio
async def test_register_pubkey_relays_the_answer_of_a_node_that_refused_the_key(executor_service, test_executor):
    """22 Nov: a node with its clock off answered 401 "Stale upload nonce" 14 times and the validator said "could not
    reach". The miner now names the refusal, so the validator can tell a reachable node that said no from a dead one."""
    mock_session = _make_mock_session()
    refusal = _response(401, b'{"detail":"Stale upload nonce"}')
    mock_session.post.return_value.__aenter__ = AsyncMock(return_value=refusal)
    es.settings.RENTAL_REQUEST_HOOK = None
    executor_service.get_executors_for_validator = AsyncMock(return_value=[test_executor])

    with patch("services.executor_service.aiohttp.ClientSession", return_value=mock_session):
        registration = await executor_service.register_pubkey("v", "m", _SSH_KEY.encode(), "0xaa")

    assert registration.accepted == []
    assert registration.refused == {str(test_executor.uuid): "HTTP 401: Stale upload nonce"}


def test_a_refusal_that_is_no_fastapi_error_is_its_body_cut_short():
    assert es._refusal_text(500, "Internal\n  Server Error") == "HTTP 500: Internal Server Error"
    assert len(es._refusal_text(502, "x" * 5000)) == es.MAX_REFUSAL_CHARS


@pytest.mark.asyncio
async def test_a_node_answering_a_field_outside_its_format_is_not_accepted(executor_service, test_executor):
    """The miner parses the node's answer with the shared model, so a node whose SSH fields are not a plain user
    name, absolute path, port list or SSH key is left out of the answer, and asked once."""
    answer = json.dumps({"ssh_username": "root", "ssh_port": 22, "python_path": "/bin/true; id", "root_dir": "/r"})
    session = _make_mock_session()
    session.post.return_value.__aenter__ = AsyncMock(return_value=_response(200, answer.encode()))

    with patch("services.executor_service.aiohttp.ClientSession", return_value=session):
        assert await executor_service.send_pubkey_to_executor(test_executor, _SSH_KEY, "0xaa") is None
    assert session.post.call_count == 1

    session.post.return_value.__aenter__ = AsyncMock(return_value=_response(200, _SSH_ANSWER))
    with patch("services.executor_service.aiohttp.ClientSession", return_value=session):
        assert (await executor_service.send_pubkey_to_executor(test_executor, _SSH_KEY, "0xaa")).python_path == "/p"
