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
    miner.bootstrap_complete = False
    miner.should_exit = False
    miner.default_extra = {"external_ip": "127.0.0.1", "external_port": 8000}
    miner.wallet = SimpleNamespace(
        get_hotkey=lambda: SimpleNamespace(ss58_address="test-hotkey")
    )
    return miner


@pytest.mark.asyncio
async def test_ensure_axon_registration_awaits_serve_axon_when_on_chain_info_differs():
    """Startup axon verification should announce only when local config differs from chain."""
    # Arrange
    miner = _make_miner()
    get_neuron = AsyncMock(
        return_value=SimpleNamespace(
            axon_info=SimpleNamespace(ip="203.0.113.10", port=9000, is_serving=True)
        )
    )
    serve_axon = AsyncMock(return_value=True)
    miner.subtensor = SimpleNamespace(
        get_neuron_for_pubkey_and_subnet=get_neuron,
        serve_axon=serve_axon,
    )

    # Act
    await miner.ensure_axon_registration()

    # Assert — startup verification should read the current neuron state first
    get_neuron.assert_awaited_once_with(
        hotkey_ss58="test-hotkey",
        netuid=miner.netuid,
    )

    # Assert — mismatched axon info should trigger a fresh announce
    serve_axon.assert_awaited_once_with(netuid=miner.netuid, axon=miner.axon)


@pytest.mark.asyncio
async def test_ensure_axon_registration_skips_serve_axon_when_on_chain_info_matches():
    """Startup axon verification should avoid announcing when chain state already matches."""
    # Arrange
    miner = _make_miner()
    get_neuron = AsyncMock(
        return_value=SimpleNamespace(
            axon_info=SimpleNamespace(
                ip=miner.default_extra["external_ip"],
                port=miner.default_extra["external_port"],
                is_serving=True,
            )
        )
    )
    serve_axon = AsyncMock(return_value=True)
    miner.subtensor = SimpleNamespace(
        get_neuron_for_pubkey_and_subnet=get_neuron,
        serve_axon=serve_axon,
    )

    # Act
    await miner.ensure_axon_registration()

    # Assert
    serve_axon.assert_not_awaited()


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
async def test_bootstrap_registers_axon_and_syncs_validators_once():
    """Bootstrap should verify axon state before the one-time validator sync."""
    # Arrange
    miner = _make_miner()
    execution_order: list[str | tuple[str, int]] = []

    async def ensure_axon_registration():
        execution_order.append("axon")

    async def fetch_validators():
        execution_order.append("fetch:start")
        execution_order.append("fetch:end")
        return [SimpleNamespace(hotkey="validator-1")]

    async def save_validators(validators):
        execution_order.append(("save", len(validators)))

    miner.ensure_axon_registration = ensure_axon_registration
    miner.fetch_validators = fetch_validators
    miner.save_validators = save_validators

    # Act
    await miner.bootstrap()

    # Assert
    assert execution_order == ["axon", "fetch:start", "fetch:end", ("save", 1)]
    assert miner.bootstrap_complete is True


@pytest.mark.asyncio
async def test_sync_bootstraps_only_once_per_connection():
    """The repeating loop should skip axon and validator bootstrap after the first success."""
    # Arrange
    miner = _make_miner()
    bootstrap_calls: list[str] = []
    miner.set_subtensor = AsyncMock()

    async def bootstrap():
        bootstrap_calls.append("bootstrap")
        miner.bootstrap_complete = True

    miner.bootstrap = bootstrap

    # Act
    await miner.sync()
    await miner.sync()

    # Assert
    assert miner.set_subtensor.await_count == 2
    assert bootstrap_calls == ["bootstrap"]


@pytest.mark.asyncio
async def test_sync_retries_bootstrap_after_reinitialize_on_failure():
    """A failed startup bootstrap should reinitialize subtensor and retry on the next cycle."""
    # Arrange
    miner = _make_miner()
    miner.set_subtensor = AsyncMock()
    bootstrap_attempts = 0
    reinitializations = 0

    async def bootstrap():
        nonlocal bootstrap_attempts
        bootstrap_attempts += 1
        if bootstrap_attempts == 1:
            raise RuntimeError("temporary bootstrap failure")
        miner.bootstrap_complete = True

    async def initialize_subtensor():
        nonlocal reinitializations
        reinitializations += 1
        miner.subtensor = object()
        miner.bootstrap_complete = False

    miner.bootstrap = bootstrap
    miner.initialize_subtensor = initialize_subtensor

    # Act
    await miner.sync()
    await miner.sync()

    # Assert
    assert reinitializations == 1
    assert bootstrap_attempts == 2
    assert miner.bootstrap_complete is True


@pytest.mark.asyncio
async def test_sync_skips_reinitialize_when_shutdown_is_in_progress():
    """Shutdown should suppress reconnect attempts after a sync failure."""
    # Arrange
    miner = _make_miner()
    miner.should_exit = True
    miner.set_subtensor = AsyncMock()
    miner.bootstrap = AsyncMock(side_effect=RuntimeError("temporary bootstrap failure"))
    miner.initialize_subtensor = AsyncMock()

    # Act
    await miner.sync()

    # Assert — shutdown should prevent creating a fresh chain connection
    miner.initialize_subtensor.assert_not_awaited()


@pytest.mark.asyncio
async def test_initialize_subtensor_closes_new_connection_when_shutdown_starts(monkeypatch):
    """A reconnect finishing during shutdown should be closed immediately and not retained."""
    # Arrange
    miner = _make_miner()
    miner.config = object()
    miner.close_subtensor = AsyncMock()
    miner.check_registered = AsyncMock()
    new_subtensor = SimpleNamespace(close=AsyncMock())

    async def initialize():
        miner.should_exit = True
        return new_subtensor

    async_subtensor_factory = SimpleNamespace(initialize=AsyncMock(side_effect=initialize))
    monkeypatch.setattr(
        miner_module.bittensor,
        "AsyncSubtensor",
        lambda config: async_subtensor_factory,
    )

    # Act
    await miner.initialize_subtensor()

    # Assert — shutdown should close the fresh connection instead of keeping it on the miner
    miner.close_subtensor.assert_awaited_once()
    new_subtensor.close.assert_awaited_once()
    miner.check_registered.assert_not_awaited()
    assert miner.subtensor is None


@pytest.mark.asyncio
async def test_initialize_subtensor_sets_connection_and_checks_registration(monkeypatch):
    """Normal initialization should retain the new subtensor and perform registration checks."""
    # Arrange
    miner = _make_miner()
    miner.config = object()
    miner.close_subtensor = AsyncMock()
    miner.check_registered = AsyncMock()
    new_subtensor = SimpleNamespace(close=AsyncMock())
    async_subtensor_factory = SimpleNamespace(initialize=AsyncMock(return_value=new_subtensor))
    monkeypatch.setattr(
        miner_module.bittensor,
        "AsyncSubtensor",
        lambda config: async_subtensor_factory,
    )

    # Act
    await miner.initialize_subtensor()

    # Assert — successful initialization should keep the new connection and validate registration
    miner.close_subtensor.assert_awaited_once()
    miner.check_registered.assert_awaited_once()
    new_subtensor.close.assert_not_awaited()
    assert miner.subtensor is new_subtensor


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
