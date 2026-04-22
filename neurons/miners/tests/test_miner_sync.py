import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

import core.miner as miner_module
from core.miner import Miner
from models.validator import Validator


def _make_miner() -> Miner:
    miner = Miner.__new__(Miner)
    miner.netuid = 51
    miner.axon = object()
    miner.subtensor = None
    miner.last_announced_block = 0
    miner.should_exit = False
    miner.default_extra = {"external_ip": "127.0.0.1", "external_port": 8000}
    miner.wallet = SimpleNamespace(
        get_hotkey=lambda: SimpleNamespace(ss58_address="test-hotkey")
    )
    return miner


@pytest.mark.asyncio
async def test_announce_awaits_serve_axon_when_tempo_elapsed():
    """Tempo threshold should trigger async axon serving exactly once."""
    # Arrange
    miner = _make_miner()
    miner.last_announced_block = 5
    miner.get_current_block = AsyncMock(return_value=25)
    miner.get_tempo = AsyncMock(return_value=10)
    serve_axon = AsyncMock(return_value=True)
    miner.subtensor = SimpleNamespace(serve_axon=serve_axon)

    # Act
    await miner.announce()

    # Assert — crossing one tempo interval should trigger async announce
    serve_axon.assert_awaited_once_with(netuid=miner.netuid, axon=miner.axon)

    # Assert — the last announced block should advance to the current block
    assert miner.last_announced_block == 25


@pytest.mark.asyncio
async def test_fetch_validators_awaits_async_subtensor_calls(monkeypatch):
    """Validator sync should use async metagraph APIs and keep the existing stake filter."""
    # Arrange
    miner = _make_miner()
    monkeypatch.setattr(miner_module.settings, "MIN_ALPHA_STAKE", 10)
    monkeypatch.setattr(miner_module.settings, "MIN_TOTAL_STAKE", 50)

    eligible = SimpleNamespace(uid=0, hotkey="validator-1", stake=SimpleNamespace(tao=15))
    low_alpha = SimpleNamespace(uid=1, hotkey="validator-2", stake=SimpleNamespace(tao=5))
    low_total = SimpleNamespace(uid=2, hotkey="validator-3", stake=SimpleNamespace(tao=15))

    get_metagraph_info = AsyncMock(return_value=SimpleNamespace(total_stake=[70, 70, 10]))
    metagraph = AsyncMock(return_value=SimpleNamespace(neurons=[eligible, low_alpha, low_total]))
    miner.subtensor = SimpleNamespace(
        get_metagraph_info=get_metagraph_info,
        metagraph=metagraph,
    )

    # Act
    validators = await miner.fetch_validators()

    # Assert — both async chain reads should be awaited during sync
    get_metagraph_info.assert_awaited_once_with(miner.netuid)
    metagraph.assert_awaited_once_with(netuid=miner.netuid)

    # Assert — only the validator meeting both stake thresholds should remain
    assert [validator.hotkey for validator in validators] == ["validator-1"]


@pytest.mark.asyncio
async def test_save_validators_offloads_persistence_to_thread(monkeypatch):
    """Validator persistence should leave the event loop by using asyncio.to_thread."""
    # Arrange
    miner = _make_miner()
    validators = [SimpleNamespace(hotkey="validator-1"), SimpleNamespace(hotkey="validator-2")]
    captured: dict[str, object] = {}

    async def fake_to_thread(func, *args, **kwargs):
        captured["func"] = func
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(miner_module.asyncio, "to_thread", fake_to_thread)

    # Act
    await miner.save_validators(validators)

    # Assert — save_validators should delegate sync DB work to the thread helper
    assert captured["func"] == miner._save_validators_sync
    assert captured["args"] == (["validator-1", "validator-2"],)
    assert captured["kwargs"] == {}


@pytest.mark.asyncio
async def test_sync_allows_other_coroutines_to_run_while_waiting_on_async_chain_calls():
    """Sync should yield control while awaiting chain I/O instead of monopolizing the loop."""
    # Arrange
    miner = _make_miner()
    execution_order: list[str] = []

    async def set_subtensor():
        execution_order.append("set_subtensor:start")
        await asyncio.sleep(0.01)
        execution_order.append("set_subtensor:end")

    async def announce():
        execution_order.append("announce:start")
        await asyncio.sleep(0.01)
        execution_order.append("announce:end")

    async def fetch_validators():
        execution_order.append("fetch:start")
        await asyncio.sleep(0.01)
        execution_order.append("fetch:end")
        return [SimpleNamespace(hotkey="validator-1")]

    async def save_validators(validators):
        execution_order.append(f"save:{len(validators)}")

    async def concurrent_marker():
        await asyncio.sleep(0)
        execution_order.append("marker")

    miner.set_subtensor = set_subtensor
    miner.announce = announce
    miner.fetch_validators = fetch_validators
    miner.save_validators = save_validators

    # Act
    await asyncio.gather(miner.sync(), concurrent_marker())

    # Assert — another coroutine should run before sync reaches its final save step
    assert execution_order.index("marker") < execution_order.index("save:1")


def test_save_validators_sync_inserts_only_missing_validators_and_commits_once(tmp_path, monkeypatch):
    """Batch persistence should avoid duplicate inserts and perform one commit for new validators."""
    # Arrange
    test_engine = create_engine(f"sqlite:///{tmp_path / 'miner-sync.db'}")
    SQLModel.metadata.create_all(test_engine)

    commit_calls: list[str] = []

    class SpySession(Session):
        def commit(self):
            commit_calls.append("commit")
            return super().commit()

    with Session(test_engine) as session:
        session.add(Validator(validator_hotkey="validator-1", active=True))
        session.commit()

    commit_calls.clear()
    monkeypatch.setattr(miner_module, "engine", test_engine)
    monkeypatch.setattr(miner_module, "Session", SpySession)

    # Act
    Miner._save_validators_sync(["validator-1", "validator-2", "validator-2", "validator-3"])

    # Assert — only one commit should be used for the batch insert of missing validators
    assert commit_calls == ["commit"]

    with Session(test_engine) as session:
        validators = session.exec(select(Validator).order_by(Validator.validator_hotkey)).all()

    # Assert — existing validators remain and only missing hotkeys are inserted once
    assert [validator.validator_hotkey for validator in validators] == [
        "validator-1",
        "validator-2",
        "validator-3",
    ]
