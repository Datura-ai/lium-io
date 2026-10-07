import logging
import os
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo
from lium_core.shared_config import DEFAULT_SHARED_CONFIG

from constants import TOTAL_BURN_EMISSION

# Settings is instantiated during test collection through app imports. These
# defaults keep unit tests self-contained without requiring a local validator env.
os.environ.setdefault("BITTENSOR_WALLET_NAME", "test-wallet")
os.environ.setdefault("BITTENSOR_WALLET_HOTKEY_NAME", "test-hotkey")
os.environ.setdefault("SQLALCHEMY_DATABASE_URI", "sqlite:///:memory:")
os.environ.setdefault("ASYNC_SQLALCHEMY_DATABASE_URI", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("VOLUME_MASTER_SECRET", "test-master-secret-32-chars-long!!")

# Prevent network calls during module-level SharedConfigClient instantiation.
# core/config.py creates `shared_client = SharedConfigClient(...)` at import time, which
# otherwise performs a blocking HTTP fetch against the live API. We make _fetch return a
# hermetic config pinned to the production burn value (TOTAL_BURN_EMISSION) so tests
# exercise production behaviour, not the lium-core offline fallback (intentionally left
# stale at the old value). This patch must run before test collection.
_TEST_SHARED_CONFIG = DEFAULT_SHARED_CONFIG.model_copy(
    update={"total_burn_emission": TOTAL_BURN_EMISSION}
)
_shared_config_patcher = patch(
    "lium_core.shared_config.client.SharedConfigClient._fetch",
    return_value=_TEST_SHARED_CONFIG,
)
_shared_config_patcher.start()

from helpers import make_context

if TYPE_CHECKING:
    from services.task_service import JobResult

# Prevent wallet KeyFileError during module-level PriceProvider() instantiation in rental_price.py.
# rental_price.py creates PriceProvider() as a class variable at import time; after the
# price_provider.py update, PriceProvider.__init__ calls SubtensorClient.get_instance() which
# reads the wallet keypair file. This patch must run before test collection.
_subtensor_patcher = patch(
    "clients.subtensor_client.SubtensorClient.get_instance",
    return_value=MagicMock(),
)
_subtensor_patcher.start()


@pytest.fixture
def make_pcc_job() -> Callable[..., "JobResult"]:
    """Factory for a rental-price `JobResult` (one node, one GPU model, N cards) with the fields the
    per-count-cap algorithm reads, for any rental-price test that needs one without importing another
    test module's private helper (a rename there would break collection)."""
    from services.task_service import JobResult  # after the subtensor patch above, like the flow test's helper

    def _make(
        executor_id: str,
        gpu_model: str,
        gpu_count: int,
        *,
        is_rented: bool = False,
        supports_gpu_splitting: bool = False,
        gpu_splitting_min_count: int | None = None,
    ) -> JobResult:
        return JobResult(
            executor_info=ExecutorSSHInfo(
                uuid=executor_id, address="10.0.0.1", port=8080,
                ssh_username="root", ssh_port=22,
                python_path="/usr/bin/python3", root_dir="/tmp",
            ),
            score=1.0, job_score=1.0, job_batch_id="pcc-batch",
            log_status="success", log_text="ok",
            gpu_model=gpu_model, gpu_count=gpu_count, is_rented=is_rented,
            collateral_deposited=True, sysbox_runtime=True,
            supports_gpu_splitting=supports_gpu_splitting,
            gpu_splitting_min_count=gpu_splitting_min_count,
        )

    return _make


@pytest.fixture(autouse=True)
def no_docker_hub_digest_on_rent_path():
    """create_container asks Docker Hub for a present image's digest; no answer keeps tests offline."""
    with patch("services.docker_service.fetch_docker_hub_digest", AsyncMock(return_value=None)) as lookup:
        yield lookup


@pytest.fixture(autouse=True)
def no_kept_upload_probes_carried_between_tests():
    """The VerifyX check counts kept upload probes per (hotkey, uuid) for the process; tests share both."""
    yield
    for name in ("services.task.checks.verifyx", "neurons.validators.src.services.task.checks.verifyx"):
        module = sys.modules.get(name)
        if module is not None:
            module._kept_upload_probes.clear()


@pytest.fixture(scope="session", autouse=True)
def setup_sql_logging():
    """Enable SQL query logging for all tests."""
    # Configure root logging to show INFO level
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)

    # Enable SQLAlchemy engine logging - just set level, let basicConfig handle output
    sql_logger = logging.getLogger('sqlalchemy.engine')
    sql_logger.setLevel(logging.INFO)

    # Also enable pool logging for connection monitoring
    pool_logger = logging.getLogger('sqlalchemy.pool')
    pool_logger.setLevel(logging.DEBUG)

    print("✅ SQL logging enabled for all tests")


@pytest.fixture(autouse=True)
def _no_collateral_rpc():
    """CollateralStatusCheck never reaches a real RPC: no miner has an EVM address unless a test says so.

    Not monkeypatch: an autouse fixture that requests it moves its undo after the event loop's teardown,
    and a test that patched time.monotonic with a finite iterator then fails there."""
    from services import collateral_status

    async def _refuse(batch):
        raise AssertionError("collateral RPC called in a unit test")

    saved = collateral_status._reader
    collateral_status._reader = collateral_status.CollateralStatusReader(
        rpc=_refuse, evm_address_for_hotkey=lambda _hotkey: None
    )
    yield
    collateral_status._reader = saved


@pytest.fixture
def context_factory():
    def _factory(**overrides):
        return make_context(**overrides)

    return _factory


