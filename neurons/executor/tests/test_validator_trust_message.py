"""DAH-6590: the executor trusts the validator named in a message the subnet owner signed.

Real signatures with ephemeral keypairs; a temp dir stands in for the reserve_data volume.
"""

import asyncio
import json
from unittest.mock import MagicMock

import pytest
from bittensor_wallet import Keypair

import dependencies.auth as auth
import services.validator_trust_service as trust_service
from datura.validator_trust import SignedTrustMessage, trust_line, verify_trust_message
from services.validator_trust_service import STATE_FILE, ValidatorTrust


@pytest.fixture(scope="module")
def owner():
    return Keypair.create_from_uri("//LiumTrustOwner")


@pytest.fixture(scope="module")
def old_validator():
    return Keypair.create_from_uri("//LiumTrustOldValidator")


@pytest.fixture(scope="module")
def new_validator():
    return Keypair.create_from_uri("//LiumTrustNewValidator")


def _signed(owner: Keypair, validator: Keypair, revision: int) -> SignedTrustMessage:
    line = trust_line(validator.ss58_address, revision)
    return SignedTrustMessage(message=line, signature="0x" + owner.sign(line).hex())


def _backend(monkeypatch, get) -> None:
    """The backend answers with `get()`: a response-like object with `status_code` and `json()`, or it raises."""

    async def fetch(url, body=None):
        response = get(url)
        return None if response.status_code == 404 else response.json()

    monkeypatch.setattr(trust_service, "fetch_json", fetch)


def _chain(monkeypatch, read) -> None:
    """Every public chain node answers `read(url)`, or it raises."""

    async def owner(url):
        return read(url)

    monkeypatch.setattr(trust_service, "read_subnet_owner_hotkey", owner)


def _state(tmp_path) -> dict:
    path = tmp_path / STATE_FILE
    return json.loads(path.read_text()) if path.exists() else {"messages": {}}


def _keep(tmp_path, signer: Keypair, signed: SignedTrustMessage, held: bool = True) -> None:
    """A message on the volume, as a node that took it from `signer` keeps it."""
    state = _state(tmp_path)
    state["messages"][signer.ss58_address] = signed._asdict()
    state.setdefault("owner", signer.ss58_address)
    if held:
        state["held"] = signer.ss58_address
    (tmp_path / STATE_FILE).write_text(json.dumps(state))


def _trust(tmp_path, owner: Keypair, image_message=None, confirmed: bool = True) -> ValidatorTrust:
    """Start a node whose kept owner is `owner`; `confirmed`: its first chain read has named that owner."""
    state = _state(tmp_path)
    state["owner"] = owner.ss58_address
    (tmp_path / STATE_FILE).write_text(json.dumps(state))
    trust = ValidatorTrust(state_dir=tmp_path, image_message=image_message or tmp_path / "absent.json")
    trust.load()
    trust.owner_confirmed = confirmed
    return trust


def _accepts(monkeypatch, trust: ValidatorTrust, validator: Keypair, baked: dict[str, str]) -> str | None:
    monkeypatch.setattr(auth, "validator_trust", trust)
    monkeypatch.setattr(auth, "VALIDATOR_HOTKEYS_SS58", baked)
    return auth.match_validator_hotkey("ping_request", "0x" + validator.sign("ping_request").hex())


def test_acceptance_new_validator_survives_recreate_without_backend_or_chain(
    tmp_path, monkeypatch, owner, old_validator, new_validator
):
    baked = {"current": old_validator.ss58_address}
    running = _trust(tmp_path, owner)
    assert _accepts(monkeypatch, running, old_validator, baked) == "current"

    # the backend serves the owner's message naming the new validator
    answer = MagicMock(status_code=200)
    answer.json.return_value = _signed(owner, new_validator, 1)._asdict()
    _backend(monkeypatch, lambda *a, **k: answer)
    asyncio.run(running.refresh_message())
    assert _accepts(monkeypatch, running, new_validator, baked) == "signed"
    assert _accepts(monkeypatch, running, old_validator, baked) is None

    # container recreated; backend and chain both unreachable
    def down(*args, **kwargs):
        raise ConnectionError("down")

    _backend(monkeypatch, down)
    _chain(monkeypatch, down)
    recreated = _trust(tmp_path, owner)
    asyncio.run(recreated.refresh_message())
    asyncio.run(recreated.refresh_owner())
    assert recreated.report() == {
        "validator": new_validator.ss58_address,
        "revision": 1,
        "subnet_owner": owner.ss58_address,
    }
    assert _accepts(monkeypatch, recreated, new_validator, baked) == "signed"
    assert _accepts(monkeypatch, recreated, old_validator, baked) is None


def test_refuses_another_signer_and_a_revision_not_higher(tmp_path, owner, old_validator, new_validator):
    trust = _trust(tmp_path, owner)
    assert not trust.offer(_signed(new_validator, new_validator, 5))  # not the owner
    assert trust.offer(_signed(owner, new_validator, 2))
    assert not trust.offer(_signed(owner, old_validator, 2))  # same revision
    assert not trust.offer(_signed(owner, old_validator, 1))  # rollback
    assert trust.offer(_signed(owner, old_validator, 3))  # two changes on one day both pass
    assert trust.held.validator == old_validator.ss58_address


def test_start_takes_the_higher_revision_of_image_and_volume(tmp_path, monkeypatch, owner, old_validator, new_validator):
    # the image copy is checked against the owner at release
    monkeypatch.setattr(trust_service, "RELEASE_SUBNET_OWNER_HOTKEY", owner.ss58_address)
    image = tmp_path / "image.json"
    image.write_text(_dump(_signed(owner, old_validator, 4)))
    _keep(tmp_path, owner, _signed(owner, new_validator, 3))
    assert _trust(tmp_path, owner, image).held.validator == old_validator.ss58_address
    (tmp_path / STATE_FILE).unlink()
    _keep(tmp_path, owner, _signed(owner, new_validator, 5))
    assert _trust(tmp_path, owner, image).held.validator == new_validator.ss58_address


def test_a_404_or_garbage_keeps_the_held_message(tmp_path, monkeypatch, owner, new_validator):
    trust = _trust(tmp_path, owner)
    trust.offer(_signed(owner, new_validator, 1))
    for answer in (MagicMock(status_code=404), MagicMock(status_code=200, json=MagicMock(return_value={"x": 1}))):
        _backend(monkeypatch, lambda *a, **k: answer)
        asyncio.run(trust.refresh_message())
        assert trust.held.validator == new_validator.ss58_address


def test_owner_change_is_kept_and_checks_the_next_message(tmp_path, monkeypatch, owner, new_validator):
    new_owner = Keypair.create_from_uri("//LiumTrustNewOwner")
    trust = _trust(tmp_path, owner)
    _chain(monkeypatch, lambda url: new_owner.ss58_address)
    asyncio.run(trust.refresh_owner())
    assert _trust(tmp_path, new_owner).owner == new_owner.ss58_address  # kept on the volume
    assert trust.offer(_signed(new_owner, new_validator, 1))


def test_no_message_keeps_the_baked_hotkeys(tmp_path, monkeypatch, owner, old_validator):
    assert _accepts(monkeypatch, _trust(tmp_path, owner), old_validator, {"current": old_validator.ss58_address}) == "current"


def test_a_non_prod_build_never_takes_a_message(tmp_path, owner, new_validator):
    trust = ValidatorTrust(state_dir=tmp_path, image_message=tmp_path / "absent.json", enabled=False)
    assert not trust.offer(_signed(owner, new_validator, 1))
    assert trust.hotkeys({"current": "x"}) == {"current": "x"}


@pytest.mark.parametrize(
    "message",
    ["lium-validator:{v}:01", "lium-validator:{v}:1 ", "lium-validator:{v}", "other:{v}:1", "lium-validator:5Fbad:1"],
)
def test_the_line_is_exact(owner, new_validator, message):
    line = message.format(v=new_validator.ss58_address)
    with pytest.raises(ValueError):
        verify_trust_message(SignedTrustMessage(line, "0x" + owner.sign(line).hex()), owner.ss58_address)


def _dump(signed: SignedTrustMessage) -> str:
    import json

    return json.dumps(signed._asdict())


def test_an_owner_change_keeps_the_held_validator_across_a_restart(tmp_path, monkeypatch, owner, new_validator):
    # review finding: owner A signed the held message, the chain then names owner C, the node restarts
    new_owner = Keypair.create_from_uri("//LiumTrustNewOwner")
    trust = _trust(tmp_path, owner)
    assert trust.offer(_signed(owner, new_validator, 7))
    _chain(monkeypatch, lambda url: new_owner.ss58_address)
    asyncio.run(trust.refresh_owner())
    restarted = ValidatorTrust(state_dir=tmp_path, image_message=tmp_path / "absent.json")
    restarted.load()
    assert restarted.owner == new_owner.ss58_address
    assert restarted.report()["validator"] == new_validator.ss58_address
    assert restarted.held.revision == 7


def test_a_new_owner_starts_its_own_revisions(tmp_path, monkeypatch, owner, old_validator, new_validator):
    new_owner = Keypair.create_from_uri("//LiumTrustNewOwner")
    trust = _trust(tmp_path, owner)
    assert trust.offer(_signed(owner, new_validator, 50))
    _chain(monkeypatch, lambda url: new_owner.ss58_address)
    asyncio.run(trust.refresh_owner())
    assert not trust.offer(_signed(owner, old_validator, 51))  # the old owner no longer signs
    assert trust.offer(_signed(new_owner, old_validator, 1))
    assert not trust.offer(_signed(new_owner, new_validator, 1))  # and the floor holds within the new owner


def test_a_failed_write_is_written_once_the_volume_takes_it(tmp_path, monkeypatch, owner, new_validator):
    # review finding: a full volume at acceptance must not leave the revision unsaved for good
    trust = _trust(tmp_path, owner)
    real_write = trust_service._write_json

    def full(path, data):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(trust_service, "_write_json", full)
    assert trust.offer(_signed(owner, new_validator, 7))
    assert _state(tmp_path)["messages"] == {}
    monkeypatch.setattr(trust_service, "_write_json", real_write)
    _backend(monkeypatch, lambda *a, **k: MagicMock(status_code=404))
    asyncio.run(trust.refresh_message())
    restarted = _trust(tmp_path, owner)
    assert restarted.held == trust.held


def test_one_public_node_cannot_name_an_owner(tmp_path, monkeypatch, owner):
    trust = _trust(tmp_path, owner)
    liar = Keypair.create_from_uri("//LiumTrustLiar").ss58_address
    answers = iter([liar, owner.ss58_address])
    _chain(monkeypatch, lambda url: next(answers))
    asyncio.run(trust.refresh_owner())
    assert trust.owner == owner.ss58_address

    def one_down(url):
        if url == trust_service.PUBLIC_CHAIN_NODES[0]:
            raise ConnectionError("down")
        return liar

    _chain(monkeypatch, one_down)
    asyncio.run(trust.refresh_owner())
    assert trust.owner == owner.ss58_address


def test_the_image_copy_needs_the_release_owner(tmp_path, owner, new_validator):
    image = tmp_path / "image.json"
    image.write_text(_dump(_signed(owner, new_validator, 1)))  # not signed by the release owner
    assert _trust(tmp_path, owner, image).held is None


def test_owner_a_b_a_prefers_the_current_owners_message(tmp_path, monkeypatch, owner, old_validator, new_validator):
    # review finding: image copy signed by A (revision 8), kept message signed by B, the owner file still says B
    owner_b = Keypair.create_from_uri("//LiumTrustNewOwner")
    monkeypatch.setattr(trust_service, "RELEASE_SUBNET_OWNER_HOTKEY", owner.ss58_address)
    image = tmp_path / "image.json"
    image.write_text(_dump(_signed(owner, new_validator, 8)))
    _keep(tmp_path, owner_b, _signed(owner_b, old_validator, 3))
    trust = _trust(tmp_path, owner_b, image)
    assert trust.held.validator == old_validator.ss58_address  # B is still the owner the node knows
    # the chain names A again: the image message A signed is held at once, with no backend
    _chain(monkeypatch, lambda url: owner.ss58_address)
    _backend(monkeypatch, lambda *a, **k: (_ for _ in ()).throw(ConnectionError()))
    asyncio.run(trust.refresh_owner())
    assert (trust.held.validator, trust.held.revision) == (new_validator.ss58_address, 8)
    # a restart with the owner file now saying A keeps it, and A's lower revisions stay refused
    restarted = _trust(tmp_path, owner, image)
    assert (restarted.held.validator, restarted.held.revision) == (new_validator.ss58_address, 8)
    assert not restarted.offer(_signed(owner, old_validator, 5))


def test_an_owner_change_never_restores_an_older_message_of_the_former_owner(
    tmp_path, monkeypatch, owner, old_validator, new_validator
):
    # review finding: owner A, image revision 8 (replacement), kept revision 3 (revoked); the chain then names B
    owner_b = Keypair.create_from_uri("//LiumTrustNewOwner")
    monkeypatch.setattr(trust_service, "RELEASE_SUBNET_OWNER_HOTKEY", owner.ss58_address)
    image = tmp_path / "image.json"
    image.write_text(_dump(_signed(owner, new_validator, 8)))
    _keep(tmp_path, owner, _signed(owner, old_validator, 3), held=False)
    trust = _trust(tmp_path, owner, image)
    assert (trust.held.validator, trust.held.revision) == (new_validator.ss58_address, 8)
    _chain(monkeypatch, lambda url: owner_b.ss58_address)
    asyncio.run(trust.refresh_owner())
    assert (trust.held.validator, trust.held.revision) == (new_validator.ss58_address, 8)
    restarted = ValidatorTrust(state_dir=tmp_path, image_message=image)
    restarted.load()
    assert restarted.owner == owner_b.ss58_address
    assert (restarted.held.validator, restarted.held.revision) == (new_validator.ss58_address, 8)


def test_a_restart_never_compares_revisions_of_different_owners(
    tmp_path, monkeypatch, owner, old_validator, new_validator
):
    # review finding: image copy A revision 8 (old validator), B then signs revision 1 (new validator), the chain
    # names C, who has published nothing; the node restarts
    owner_b = Keypair.create_from_uri("//LiumTrustNewOwner")
    owner_c = Keypair.create_from_uri("//LiumTrustThirdOwner")
    monkeypatch.setattr(trust_service, "RELEASE_SUBNET_OWNER_HOTKEY", owner.ss58_address)
    image = tmp_path / "image.json"
    image.write_text(_dump(_signed(owner, old_validator, 8)))
    trust = _trust(tmp_path, owner, image)
    for new_owner in (owner_b, owner_c):
        _chain(monkeypatch, lambda url, o=new_owner: o.ss58_address)
        asyncio.run(trust.refresh_owner())
        if new_owner is owner_b:
            assert trust.offer(_signed(owner_b, new_validator, 1))
    assert trust.held.validator == new_validator.ss58_address
    restarted = ValidatorTrust(state_dir=tmp_path, image_message=image)
    restarted.load()
    assert restarted.owner == owner_c.ss58_address
    assert (restarted.held.validator, restarted.held.revision) == (new_validator.ss58_address, 1)


def test_a_returning_owner_keeps_its_highest_revision(tmp_path, monkeypatch, owner, old_validator, new_validator):
    # review finding: A's image copy is revision 4, the node takes A revision 8, B signs, then the chain names A again
    owner_b = Keypair.create_from_uri("//LiumTrustNewOwner")
    monkeypatch.setattr(trust_service, "RELEASE_SUBNET_OWNER_HOTKEY", owner.ss58_address)
    image = tmp_path / "image.json"
    image.write_text(_dump(_signed(owner, old_validator, 4)))
    trust = _trust(tmp_path, owner, image)
    assert trust.offer(_signed(owner, new_validator, 8))
    for chain_owner in (owner_b, owner):
        _chain(monkeypatch, lambda url, o=chain_owner: o.ss58_address)
        asyncio.run(trust.refresh_owner())
        if chain_owner is owner_b:
            assert trust.offer(_signed(owner_b, old_validator, 1))
    assert (trust.held.validator, trust.held.revision) == (new_validator.ss58_address, 8)
    restarted = ValidatorTrust(state_dir=tmp_path, image_message=image)
    restarted.load()
    assert (restarted.held.validator, restarted.held.revision) == (new_validator.ss58_address, 8)
    assert not restarted.offer(_signed(owner, old_validator, 5))


def test_a_restarted_node_reads_the_owner_before_taking_a_served_message(
    tmp_path, monkeypatch, owner, old_validator, new_validator
):
    # review finding: kept owner A, the chain already names B, the backend still serves a higher A revision
    owner_b = Keypair.create_from_uri("//LiumTrustNewOwner")
    _keep(tmp_path, owner, _signed(owner, new_validator, 2))
    trust = _trust(tmp_path, owner, confirmed=False)
    answer = MagicMock(status_code=200)
    answer.json.return_value = _signed(owner, old_validator, 9)._asdict()
    _backend(monkeypatch, lambda *a, **k: answer)
    monkeypatch.setattr(trust_service.settings, "COMPUTE_REST_API_URL", "http://backend")
    asyncio.run(trust.refresh_message())  # no chain read yet: nothing is taken
    assert (trust.held.validator, trust.held.revision) == (new_validator.ss58_address, 2)

    def down(url):
        raise ConnectionError("down")

    _chain(monkeypatch, down)
    asyncio.run(trust.refresh_owner())  # the chain unreachable: the last held message stays, still nothing taken
    asyncio.run(trust.refresh_message())
    assert (trust.held.validator, trust.held.revision) == (new_validator.ss58_address, 2)
    _chain(monkeypatch, lambda url: owner_b.ss58_address)
    asyncio.run(trust.refresh_owner())
    asyncio.run(trust.refresh_message())  # A no longer signs
    assert (trust.held.validator, trust.held.revision) == (new_validator.ss58_address, 2)
    answer.json.return_value = _signed(owner_b, old_validator, 1)._asdict()
    asyncio.run(trust.refresh_message())
    assert (trust.held.validator, trust.held_owner) == (old_validator.ss58_address, owner_b.ss58_address)


def test_the_owner_and_the_held_message_are_kept_together(tmp_path, monkeypatch, owner, old_validator, new_validator):
    # review finding: the owner write failed, the message write landed, a restart read owner A with B's message
    owner_b = Keypair.create_from_uri("//LiumTrustNewOwner")
    trust = _trust(tmp_path, owner)
    assert trust.offer(_signed(owner, new_validator, 3))
    real_write = trust_service._write_json
    monkeypatch.setattr(trust_service, "_write_json", lambda path, data: (_ for _ in ()).throw(OSError(28, "full")))
    _chain(monkeypatch, lambda url: owner_b.ss58_address)
    asyncio.run(trust.refresh_owner())
    assert _state(tmp_path)["owner"] == owner.ss58_address  # nothing of the change landed
    monkeypatch.setattr(trust_service, "_write_json", real_write)
    assert trust.offer(_signed(owner_b, old_validator, 1))
    state = _state(tmp_path)
    assert (state["owner"], state["held"]) == (owner_b.ss58_address, owner_b.ss58_address)
    restarted = ValidatorTrust(state_dir=tmp_path, image_message=tmp_path / "absent.json")
    restarted.load()
    assert (restarted.owner, restarted.held.validator) == (owner_b.ss58_address, old_validator.ss58_address)
    assert not restarted.offer(_signed(owner, new_validator, 99))  # the former owner signs nothing any more


@pytest.mark.parametrize("second_read", ["down", "disagree"])
def test_a_failed_owner_read_after_a_good_one_takes_no_served_message(
    tmp_path, monkeypatch, owner, old_validator, new_validator, second_read
):
    other = Keypair.create_from_uri("//LiumTrustNewOwner")
    _keep(tmp_path, owner, _signed(owner, new_validator, 2))
    trust = _trust(tmp_path, owner, confirmed=False)
    _chain(monkeypatch, lambda url: owner.ss58_address)
    asyncio.run(trust.refresh_owner())
    assert trust.owner_confirmed
    if second_read == "down":

        def read(url):
            raise ConnectionError("down")
    else:
        answers = iter([owner.ss58_address, other.ss58_address] * 4)

        def read(url):
            return next(answers)

    _chain(monkeypatch, read)
    asyncio.run(trust.refresh_owner())
    assert not trust.owner_confirmed
    answer = MagicMock(status_code=200)
    answer.json.return_value = _signed(owner, old_validator, 9)._asdict()
    _backend(monkeypatch, lambda *a, **k: answer)
    monkeypatch.setattr(trust_service.settings, "COMPUTE_REST_API_URL", "http://backend")
    asyncio.run(trust.refresh_message())
    assert (trust.held.validator, trust.held.revision) == (new_validator.ss58_address, 2)
    _chain(monkeypatch, lambda url: owner.ss58_address)
    asyncio.run(trust.refresh_owner())  # the next round reads the owner again
    asyncio.run(trust.refresh_message())
    assert (trust.held.validator, trust.held.revision) == (old_validator.ss58_address, 9)


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
    from scalecodec.utils.ss58 import ss58_decode

    from datura.validator_trust import subnet_owner_request

    sent = []

    async def node(request):
        sent.append(await request.json())
        return web.json_response({"result": "0x" + ss58_decode(owner.ss58_address)})

    async def run():
        runner, url = await _serve(node)
        try:
            return await trust_service.read_subnet_owner_hotkey(url)
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


def test_refreshes_write_the_volume_off_the_event_loop(tmp_path, monkeypatch, owner, new_validator):
    import threading

    new_owner = Keypair.create_from_uri("//LiumTrustNewOwner")
    trust = _trust(tmp_path, owner)
    real_write = trust_service._write_json
    writers = []

    def write(path, data):
        writers.append(threading.current_thread() is threading.main_thread())
        real_write(path, data)

    monkeypatch.setattr(trust_service, "_write_json", write)
    monkeypatch.setattr(trust_service.settings, "COMPUTE_REST_API_URL", "http://backend")
    _chain(monkeypatch, lambda url: new_owner.ss58_address)
    answer = MagicMock(status_code=200)
    answer.json.return_value = _signed(new_owner, new_validator, 1)._asdict()
    _backend(monkeypatch, lambda url: answer)
    asyncio.run(trust.refresh_owner())  # owner change written
    asyncio.run(trust.refresh_message())  # new message written
    assert trust.held.validator == new_validator.ss58_address
    assert len(writers) >= 2 and not any(writers)
