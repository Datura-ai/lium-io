"""DAH-6590: the executor trusts the validator named in a message the subnet owner signed.

Real signatures with ephemeral keypairs; a temp dir stands in for the reserve_data volume.
"""

import asyncio
import json
from unittest.mock import MagicMock

import pytest
import services.validator_trust_service as trust_service
from bittensor_wallet import Keypair
from datura.validator_trust import SignedTrustMessage, trust_line, verify_trust_message
from services.validator_trust_service import STATE_FILE, ValidatorTrust

import dependencies.auth as auth


@pytest.fixture(scope="module")
def owner():
    return Keypair.create_from_uri("//LiumTrustOwner")


@pytest.fixture(scope="module")
def old_validator():
    return Keypair.create_from_uri("//LiumTrustOldValidator")


@pytest.fixture(scope="module")
def new_validator():
    return Keypair.create_from_uri("//LiumTrustNewValidator")


@pytest.fixture(scope="module")
def new_owner():
    return Keypair.create_from_uri("//LiumTrustNewOwner")


def _signed(owner: Keypair, validator: Keypair, revision: int) -> SignedTrustMessage:
    line = trust_line(validator.ss58_address, revision)
    return SignedTrustMessage(message=line, signature="0x" + owner.sign(line).hex())


def _backend(monkeypatch, get) -> None:
    """The backend answers with `get()`: a response-like object with `status_code` and `json()`, or it raises."""

    async def fetch(url, body=None):
        response = get(url)
        return None if response.status_code == 404 else response.json()

    monkeypatch.setattr(trust_service, "fetch_json", fetch)


def _serves(monkeypatch, signed: SignedTrustMessage) -> None:
    monkeypatch.setattr(trust_service.settings, "COMPUTE_REST_API_URL", "http://backend")
    _backend(monkeypatch, lambda *a, **k: MagicMock(status_code=200, json=MagicMock(return_value=signed._asdict())))


def _down(*args, **kwargs):
    raise ConnectionError("down")


def _chain(monkeypatch, read) -> None:
    """Every public chain node answers `read(url)`, or it raises."""

    async def owner(url):
        return read(url)

    monkeypatch.setattr(trust_service, "read_subnet_owner_coldkey", owner)


def _trust(tmp_path, image_message=None) -> ValidatorTrust:
    """A node starting on the volume `tmp_path`."""
    trust = ValidatorTrust(state_dir=tmp_path, image_message=image_message or tmp_path / "absent.json")
    trust.load()
    return trust


def _image(tmp_path, signed: SignedTrustMessage):
    image = tmp_path / "image.json"
    image.write_text(json.dumps(signed._asdict()))
    return image


def _accepts(monkeypatch, trust: ValidatorTrust, validator: Keypair, baked: dict[str, str]) -> str | None:
    monkeypatch.setattr(auth, "validator_trust", trust)
    monkeypatch.setattr(auth, "VALIDATOR_HOTKEYS_SS58", baked)
    return auth.match_validator_hotkey("ping_request", "0x" + validator.sign("ping_request").hex())


@pytest.fixture(autouse=True)
def release_owner(monkeypatch, owner):
    monkeypatch.setattr(trust_service, "RELEASE_SUBNET_OWNER_COLDKEY", owner.ss58_address)
    monkeypatch.setattr(trust_service.settings, "COMPUTE_REST_API_URL", "http://backend")


def test_acceptance_new_validator_survives_recreate_without_backend_or_chain(
    tmp_path, monkeypatch, owner, old_validator, new_validator
):
    baked = {"current": old_validator.ss58_address}
    running = _trust(tmp_path)
    assert _accepts(monkeypatch, running, old_validator, baked) == "current"

    # the backend serves the owner's message naming the new validator
    _serves(monkeypatch, _signed(owner, new_validator, 1))
    asyncio.run(running.refresh_message())
    assert _accepts(monkeypatch, running, new_validator, baked) == "signed"
    assert _accepts(monkeypatch, running, old_validator, baked) is None

    # container recreated; backend and chain both unreachable
    _backend(monkeypatch, _down)
    _chain(monkeypatch, _down)
    recreated = _trust(tmp_path)
    asyncio.run(recreated.refresh_owner())
    asyncio.run(recreated.refresh_message())
    assert recreated.report() == {"validator": new_validator.ss58_address, "revision": 1, "subnet_owner": owner.ss58_address}
    assert _accepts(monkeypatch, recreated, new_validator, baked) == "signed"
    assert _accepts(monkeypatch, recreated, old_validator, baked) is None


def test_refuses_another_signer_and_a_revision_not_higher(tmp_path, owner, old_validator, new_validator):
    trust = _trust(tmp_path)
    assert not trust.offer(_signed(new_validator, new_validator, 5))  # not the owner
    assert trust.offer(_signed(owner, new_validator, 2))
    assert not trust.offer(_signed(owner, old_validator, 2))  # same revision
    assert not trust.offer(_signed(owner, old_validator, 1))  # rollback
    assert trust.offer(_signed(owner, old_validator, 3))
    assert trust.held.validator == old_validator.ss58_address


@pytest.mark.parametrize(
    "message",
    ["lium-validator:{v}:01", "lium-validator:{v}:1 ", "lium-validator:{v}", "other:{v}:1", "lium-validator:5Fbad:1"],
)
def test_the_line_is_exact(owner, new_validator, message):
    line = message.format(v=new_validator.ss58_address)
    with pytest.raises(ValueError):
        verify_trust_message(SignedTrustMessage(line, "0x" + owner.sign(line).hex()), owner.ss58_address)


def test_start_takes_the_higher_revision_of_image_and_volume(tmp_path, monkeypatch, owner, old_validator, new_validator):
    image = _image(tmp_path, _signed(owner, old_validator, 4))
    trust = _trust(tmp_path)
    _serves(monkeypatch, _signed(owner, new_validator, 3))
    asyncio.run(trust.refresh_message())
    assert _trust(tmp_path, image).held.validator == old_validator.ss58_address
    _serves(monkeypatch, _signed(owner, new_validator, 5))
    asyncio.run(trust.refresh_message())
    assert _trust(tmp_path, image).held.validator == new_validator.ss58_address


def test_an_image_message_taken_at_start_is_kept_for_a_later_start_without_it(
    tmp_path, owner, old_validator, new_validator
):
    kept = _trust(tmp_path)
    assert kept.offer(_signed(owner, old_validator, 3))
    asyncio.run(kept.save())
    started = _trust(tmp_path, _image(tmp_path, _signed(owner, new_validator, 4)))
    asyncio.run(started.save())  # the next refresh
    assert _trust(tmp_path).held.validator == new_validator.ss58_address


def test_the_image_copy_needs_the_release_owner(tmp_path, new_owner, new_validator):
    assert _trust(tmp_path, _image(tmp_path, _signed(new_owner, new_validator, 1))).held is None


@pytest.mark.parametrize("kept", ["{not json", json.dumps({"owner": "5Fbad", "signer": None, "message": None}), "bad-sig"])
def test_an_unreadable_state_falls_back_to_the_image_copy_else_baked(
    tmp_path, monkeypatch, owner, old_validator, new_validator, kept
):
    if kept == "bad-sig":
        signed = _signed(owner, new_validator, 9)
        kept = json.dumps({"owner": owner.ss58_address, "signer": owner.ss58_address,
                           "message": {"message": signed.message, "signature": "0x" + "00" * 64}})
    (tmp_path / STATE_FILE).write_text(kept)
    baked = {"current": old_validator.ss58_address}
    assert _trust(tmp_path).hotkeys(baked) == baked
    image = _image(tmp_path, _signed(owner, new_validator, 1))
    assert _trust(tmp_path, image).hotkeys(baked) == {"signed": new_validator.ss58_address}


def test_a_404_or_garbage_keeps_the_held_message(tmp_path, monkeypatch, owner, new_validator):
    trust = _trust(tmp_path)
    trust.offer(_signed(owner, new_validator, 1))
    for answer in (MagicMock(status_code=404), MagicMock(status_code=200, json=MagicMock(return_value={"x": 1}))):
        _backend(monkeypatch, lambda *a, **k: answer)
        asyncio.run(trust.refresh_message())
        assert trust.held.validator == new_validator.ss58_address


def test_no_message_keeps_the_baked_hotkeys(tmp_path, monkeypatch, old_validator):
    assert _accepts(monkeypatch, _trust(tmp_path), old_validator, {"current": old_validator.ss58_address}) == "current"


def test_a_non_prod_build_never_takes_a_message(tmp_path, owner, new_validator):
    trust = ValidatorTrust(state_dir=tmp_path, image_message=tmp_path / "absent.json", enabled=False)
    assert not trust.offer(_signed(owner, new_validator, 1))
    assert trust.hotkeys({"current": "x"}) == {"current": "x"}


def test_an_owner_change_changes_only_the_verifying_key(
    tmp_path, monkeypatch, owner, new_owner, old_validator, new_validator
):
    trust = _trust(tmp_path)
    _serves(monkeypatch, _signed(owner, new_validator, 7))
    asyncio.run(trust.refresh_message())
    _chain(monkeypatch, lambda url: new_owner.ss58_address)
    assert asyncio.run(trust.refresh_owner())
    # the held validator stays, across a restart too
    restarted = _trust(tmp_path)
    assert (restarted.owner, restarted.held.validator, restarted.held.revision) == (
        new_owner.ss58_address, new_validator.ss58_address, 7
    )
    assert not restarted.offer(_signed(owner, old_validator, 8))  # the former owner no longer signs
    assert not restarted.offer(_signed(new_owner, old_validator, 7))  # one counter: not higher
    assert restarted.offer(_signed(new_owner, old_validator, 8))


def test_one_public_node_cannot_name_an_owner(tmp_path, monkeypatch, owner):
    trust = _trust(tmp_path)
    liar = Keypair.create_from_uri("//LiumTrustLiar").ss58_address
    answers = iter([liar, owner.ss58_address])
    _chain(monkeypatch, lambda url: next(answers))
    assert not asyncio.run(trust.refresh_owner())
    assert trust.owner == owner.ss58_address

    def one_down(url):
        if url == trust_service.PUBLIC_CHAIN_NODES[0]:
            raise ConnectionError("down")
        return liar

    _chain(monkeypatch, one_down)
    assert not asyncio.run(trust.refresh_owner())
    assert trust.owner == owner.ss58_address


@pytest.mark.parametrize("chain", ["down", "disagree"])
def test_a_chain_outage_does_not_block_a_validator_change(tmp_path, monkeypatch, owner, new_validator, chain):
    trust = _trust(tmp_path)
    if chain == "down":
        _chain(monkeypatch, _down)
    else:
        answers = iter([owner.ss58_address, Keypair.create_from_uri("//LiumTrustLiar").ss58_address])
        _chain(monkeypatch, lambda url: next(answers))
    assert not asyncio.run(trust.refresh_owner())
    _serves(monkeypatch, _signed(owner, new_validator, 1))
    asyncio.run(trust.refresh_message())
    assert trust.held.validator == new_validator.ss58_address


def test_a_failed_write_is_retried_at_the_next_refresh(tmp_path, monkeypatch, owner, new_validator):
    trust = _trust(tmp_path)
    real_write = trust_service._write_json

    def full(path, data):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(trust_service, "_write_json", full)
    _serves(monkeypatch, _signed(owner, new_validator, 7))
    asyncio.run(trust.refresh_message())
    assert trust.held.validator == new_validator.ss58_address  # trusted for this run
    assert not (tmp_path / STATE_FILE).exists()
    monkeypatch.setattr(trust_service, "_write_json", real_write)
    _backend(monkeypatch, lambda *a, **k: MagicMock(status_code=404))
    asyncio.run(trust.refresh_message())
    assert _trust(tmp_path).held == trust.held


async def _serve(handler):
    """A local HTTP server answering every request with `handler`; returns (runner, url)."""
    from aiohttp import web

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


async def _dribble(request):
    """Keeps the reply open: one byte at a time, each well inside any per-read timeout."""
    from aiohttp import web

    response = web.StreamResponse()
    await response.prepare(request)
    for _ in range(1000):
        await response.write(b" ")
        await asyncio.sleep(0.05)
    return response


def test_a_slowly_streamed_reply_ends_at_the_deadline(monkeypatch):
    monkeypatch.setattr(trust_service, "FETCH_TIMEOUT_SECONDS", 0.3)

    async def run():
        runner, url = await _serve(_dribble)
        try:
            started = asyncio.get_running_loop().time()
            with pytest.raises(asyncio.TimeoutError):
                await trust_service.fetch_json(url)
            return asyncio.get_running_loop().time() - started
        finally:
            await runner.cleanup()

    assert asyncio.run(run()) < 1.0


def test_an_oversized_reply_is_refused(monkeypatch):
    from aiohttp import web

    async def big(request):
        return web.Response(body=b"[" + b"1," * trust_service.FETCH_LIMIT_BYTES + b"1]")

    async def run():
        runner, url = await _serve(big)
        try:
            with pytest.raises(ValueError, match="bytes"):
                await trust_service.fetch_json(url)
        finally:
            await runner.cleanup()

    asyncio.run(run())


def test_the_owner_read_posts_the_storage_request(monkeypatch, owner):
    from aiohttp import web
    from datura.validator_trust import subnet_owner_request
    from scalecodec.utils.ss58 import ss58_decode

    sent = []

    async def node(request):
        sent.append(await request.json())
        return web.json_response({"result": "0x" + ss58_decode(owner.ss58_address)})

    async def run():
        runner, url = await _serve(node)
        try:
            return await trust_service.read_subnet_owner_coldkey(url)
        finally:
            await runner.cleanup()

    assert asyncio.run(run()) == owner.ss58_address
    assert sent == [subnet_owner_request()]


def test_a_slow_backend_does_not_stop_later_refreshes(tmp_path, monkeypatch, owner):
    monkeypatch.setattr(trust_service, "FETCH_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(trust_service, "MESSAGE_REFRESH_SECONDS", 0)
    monkeypatch.setattr(trust_service, "OWNER_REFRESH_SECONDS", 0)
    trust = _trust(tmp_path, owner)
    monkeypatch.setattr(trust_service, "validator_trust", trust)
    owner_reads, message_reads = [], []
    _chain(monkeypatch, lambda url: owner_reads.append(url) or owner.ss58_address)

    async def dribble(request):
        message_reads.append(request.path)
        return await _dribble(request)

    async def run():
        runner, url = await _serve(dribble)
        monkeypatch.setattr(trust_service.settings, "COMPUTE_REST_API_URL", url)
        try:
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(trust_service.run_validator_trust_refresh(), 1.0)
        finally:
            await runner.cleanup()

    asyncio.run(run())
    assert len(message_reads) >= 3
    assert len(owner_reads) >= 3 * len(trust_service.PUBLIC_CHAIN_NODES)


def test_a_compressed_reply_is_refused_unexpanded(monkeypatch):
    import gzip

    from aiohttp import web

    seen = []

    async def bomb(request):
        seen.append(request.headers.get("Accept-Encoding"))
        body = gzip.compress(b" " * (8 * trust_service.FETCH_LIMIT_BYTES))
        return web.Response(body=body, headers={"Content-Encoding": "gzip", "Content-Type": "application/json"})

    async def run():
        runner, url = await _serve(bomb)
        try:
            with pytest.raises(ValueError, match="encoded"):
                await trust_service.fetch_json(url)
        finally:
            await runner.cleanup()

    asyncio.run(run())
    assert seen == ["identity"]


def test_the_image_copy_is_signed_by_the_release_time_owner_coldkey():
    from datura.validator_trust import RELEASE_SUBNET_OWNER_COLDKEY, verify_trust_message

    shipped = trust_service._signed(json.loads(trust_service.IMAGE_MESSAGE_PATH.read_text()))
    assert verify_trust_message(shipped, RELEASE_SUBNET_OWNER_COLDKEY).revision == 1
