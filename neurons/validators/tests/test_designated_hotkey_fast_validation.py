"""The fast-validation profile for nodes registered under designated miner hotkeys.

Applied ONLY to the FIRST, unscored verification (the express lane's `first_pass=True` call) of an
executor whose miner hotkey is in DESIGNATED_MINER_HOTKEYS, and only with
DESIGNATED_HOTKEY_FAST_VALIDATION_ENABLED on. Skipped: the collateral read and VerifyX (both deferred
to the first scored cycle). Resized: the capability matmul (kept, at the first-pass VRAM
budget). Kept whole: every check that catches a broken node. A provider node — whatever it reports
about itself — takes today's full pipeline; so does every scored cycle of the designated-hotkey node.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from neurons.validators.src.protocol.vc_protocol.compute_requests import ExecutorHealthCheckResponse
from neurons.validators.src.services.container_cleanup import ContainerCleanup
from neurons.validators.src.services.task import pipeline_factory as pipeline_factory_module
from neurons.validators.src.services.task.checks.capability import CapabilityCheck
from neurons.validators.src.services.task.checks.collateral import CollateralCheck
from neurons.validators.src.services.task.checks.gpu_count import GpuCountCheck
from neurons.validators.src.services.task.checks.gpu_vram_precheck import GpuVramPrecheck
from neurons.validators.src.services.task.checks.port_count import PortCountCheck
from neurons.validators.src.services.task.checks.rental_verification import RentalVerificationCheck
from neurons.validators.src.services.task.checks.sysbox_required import SysboxRequiredCheck
from neurons.validators.src.services.task.checks.verifyx import VerifyXCheck
from neurons.validators.src.services.task.messages import CapabilityMessages as CapMsg
from neurons.validators.src.services.task.messages import CollateralMessages as ColMsg
from neurons.validators.src.services.task.messages import GpuCountMessages as GpuCountMsg
from neurons.validators.src.services.task.messages import PortCountMessages as PortMsg
from neurons.validators.src.services.task.messages import RentalVerificationMessages as RentMsg
from neurons.validators.src.services.task.messages import SysboxRequiredMessages as SysboxMsg
from neurons.validators.src.services.task.messages import VerifyXMessages as VxMsg
from neurons.validators.src.services.task.pipeline_factory import PipelineFactory
from neurons.validators.src.services.task.score_calculator import calculate_scores

from core.config import settings
from services.const import MIN_PORT_COUNT
from tests.helpers import (
    build_context_config,
    build_services,
    build_state,
    default_executor,
    make_context,
)
from tests.test_capability_check import DummyValidationService
from tests.test_collateral_check import DummyCollateralService
from tests.test_rental_verification_check import DummyBackendClient
from tests.test_verifyx_check import DummyVerifyXService

# fixture names, not keys: the profile compares the authenticated miner hotkey string as-is
DESIGNATED_HOTKEY = "designated-hotkey-fixture-1"
OTHER_DESIGNATED_HOTKEY = "designated-hotkey-fixture-2"
PROVIDER_HOTKEY = "provider-hotkey-fixture"


@pytest.fixture
def profile_on(monkeypatch):
    monkeypatch.setattr(settings, "DESIGNATED_HOTKEY_FAST_VALIDATION_ENABLED", True)
    monkeypatch.setattr(
        settings, "DESIGNATED_MINER_HOTKEYS", f"{DESIGNATED_HOTKEY}, {OTHER_DESIGNATED_HOTKEY}"
    )


# --- who gets the profile ---------------------------------------------------------------------


def test_flag_and_hotkeys_default_off():
    assert settings.DESIGNATED_HOTKEY_FAST_VALIDATION_ENABLED is False
    assert settings.DESIGNATED_MINER_HOTKEYS == ""
    assert settings.designated_miner_hotkeys() == frozenset()
    assert settings.is_designated_hotkey_first_pass(DESIGNATED_HOTKEY, first_pass=True) is False


def test_hotkey_list_is_parsed_trimmed_and_ignores_blanks(profile_on, monkeypatch):
    monkeypatch.setattr(
        settings, "DESIGNATED_MINER_HOTKEYS", f" {DESIGNATED_HOTKEY} ,, {OTHER_DESIGNATED_HOTKEY},"
    )
    assert settings.designated_miner_hotkeys() == frozenset(
        {DESIGNATED_HOTKEY, OTHER_DESIGNATED_HOTKEY}
    )


# --- the dedicated-hotkey rule ----------------------------------------------------------------
# A miner hotkey names an account, not a person: custodied (wallet-free) provider accounts all list
# under one shared pool hotkey. That hotkey in the designated list would give every such provider's
# first pass the profile, so config load refuses the overlap and startup names the unchecked gap.

POOL_HOTKEY = "lium-pool-hotkey-fixture"


@pytest.mark.parametrize("flag", [True, False])
def test_config_load_refuses_a_designated_hotkey_that_is_a_pool_hotkey(flag):
    from pydantic import ValidationError

    from core.config import Settings

    with pytest.raises(ValidationError, match="dedicated hotkey"):
        Settings(
            _env_file=None,
            DESIGNATED_HOTKEY_FAST_VALIDATION_ENABLED=flag,
            DESIGNATED_MINER_HOTKEYS=f"{DESIGNATED_HOTKEY}, {POOL_HOTKEY}",
            LIUM_POOL_HOTKEYS=f"{POOL_HOTKEY},other-pool-hotkey-fixture",
        )


def test_config_load_accepts_dedicated_hotkeys_beside_a_filled_pool_mirror():
    from core.config import Settings

    s = Settings(
        _env_file=None,
        DESIGNATED_HOTKEY_FAST_VALIDATION_ENABLED=True,
        DESIGNATED_MINER_HOTKEYS=f"{DESIGNATED_HOTKEY},{OTHER_DESIGNATED_HOTKEY}",
        LIUM_POOL_HOTKEYS=POOL_HOTKEY,
    )
    assert s.lium_pool_hotkeys() == frozenset({POOL_HOTKEY})
    assert s.is_designated_hotkey_first_pass(DESIGNATED_HOTKEY, first_pass=True) is True
    assert s.is_designated_hotkey_first_pass(POOL_HOTKEY, first_pass=True) is False
    assert s.designated_hotkey_startup_warnings() == []


@pytest.mark.parametrize(
    ("flag", "designated", "pool", "expected_fragment"),
    [
        (False, DESIGNATED_HOTKEY, "", None),  # flag off: nothing to say
        (True, "", "", "empty DESIGNATED_MINER_HOTKEYS"),  # on, selects nobody
        (True, DESIGNATED_HOTKEY, "", "LIUM_POOL_HOTKEYS is empty"),  # on, rule unchecked
        (True, DESIGNATED_HOTKEY, POOL_HOTKEY, None),  # on, rule checked at load
    ],
)
def test_startup_warning_names_the_unchecked_dedicated_hotkey_rule(
    monkeypatch, flag, designated, pool, expected_fragment
):
    monkeypatch.setattr(settings, "DESIGNATED_HOTKEY_FAST_VALIDATION_ENABLED", flag)
    monkeypatch.setattr(settings, "DESIGNATED_MINER_HOTKEYS", designated)
    monkeypatch.setattr(settings, "LIUM_POOL_HOTKEYS", pool)
    warnings = settings.designated_hotkey_startup_warnings()
    if expected_fragment is None:
        assert warnings == []
    else:
        assert len(warnings) == 1 and expected_fragment in warnings[0]


@pytest.mark.parametrize(
    "flag,hotkeys,miner_hotkey,first_pass,expected",
    [
        # the four things that must ALL hold
        (True, DESIGNATED_HOTKEY, DESIGNATED_HOTKEY, True, True),
        # flag off: nobody, even the designated hotkey on its first pass
        (False, DESIGNATED_HOTKEY, DESIGNATED_HOTKEY, True, False),
        # flag on, empty list: nobody
        (True, "", DESIGNATED_HOTKEY, True, False),
        # a provider's hotkey never matches
        (True, DESIGNATED_HOTKEY, PROVIDER_HOTKEY, True, False),
        # a scored cycle (the wave never passes first_pass) never takes the profile
        (True, DESIGNATED_HOTKEY, DESIGNATED_HOTKEY, False, False),
        # a prefix / superstring of the designated hotkey is not the designated hotkey
        (True, DESIGNATED_HOTKEY, DESIGNATED_HOTKEY[:-1], True, False),
        (True, DESIGNATED_HOTKEY, DESIGNATED_HOTKEY + "0", True, False),
        (True, DESIGNATED_HOTKEY, None, True, False),
        (True, DESIGNATED_HOTKEY, "", True, False),
    ],
)
def test_profile_needs_flag_listed_hotkey_and_first_pass(
    monkeypatch, flag, hotkeys, miner_hotkey, first_pass, expected
):
    monkeypatch.setattr(settings, "DESIGNATED_HOTKEY_FAST_VALIDATION_ENABLED", flag)
    monkeypatch.setattr(settings, "DESIGNATED_MINER_HOTKEYS", hotkeys)
    assert settings.is_designated_hotkey_first_pass(miner_hotkey, first_pass) is expected


async def _build_context(
    miner_hotkey: str, first_pass: bool, executor_overrides: dict | None = None
):
    """build_context with everything stubbed, capturing the ContextConfig the factory assembles."""
    redis = SimpleNamespace(
        get_verified_job_info=AsyncMock(return_value={}),
        is_elem_exists_in_set=AsyncMock(return_value=False),
    )
    factory = PipelineFactory.__new__(PipelineFactory)
    factory.redis_service = redis
    for name in (
        "ssh_service",
        "validation_service",
        "verifyx_validation_service",
        "inspector_validation_service",
        "collateral_contract_service",
        "executor_connectivity_service",
        "backend_client",
        "pod_recovery",
        "container_cleanup",
    ):
        setattr(factory, name, MagicMock())
    shell = SimpleNamespace(ssh_client=MagicMock())
    executor_fields = dict(
        uuid="executor-123",
        address="203.0.113.10",
        port=8000,
        ssh_username="root",
        ssh_port=22,
        root_dir="/root/app",
    )
    executor_fields.update(executor_overrides or {})
    executor = SimpleNamespace(**executor_fields)
    encrypted_files = SimpleNamespace(
        encrypt_key="k",
        machine_scrape_file_name="scrape",
        machine_scrape_source=None,
        all_keys={},
        tmp_directory="/tmp/x",
    )
    miner_info = SimpleNamespace(
        job_batch_id="b",
        miner_hotkey=miner_hotkey,
        miner_coldkey="c",
        miner_address="203.0.113.1",
        miner_port=1,
    )
    return await factory.build_context(
        shell=shell,
        miner_info=miner_info,
        executor_info=executor,
        keypair=None,
        private_key="p",
        public_key="q",
        encrypted_files=encrypted_files,
        rented_data=None,
        default_docker_image_digests={},
        first_pass=first_pass,
    )


@pytest.mark.asyncio
async def test_build_context_marks_the_designated_hotkey_first_pass(profile_on, monkeypatch):
    monkeypatch.setattr(pipeline_factory_module, "Context", lambda **kw: SimpleNamespace(**kw))
    ctx = await _build_context(DESIGNATED_HOTKEY, first_pass=True)
    assert ctx.config.designated_hotkey_first_pass is True
    # independent of DAH-3011's flag: FIRST_PASS_FAST_PATH_ENABLED is off here and the profile is whole
    assert ctx.config.first_pass is False


@pytest.mark.asyncio
async def test_build_context_scored_cycle_of_the_designated_hotkey_node_is_unmarked(
    profile_on, monkeypatch
):
    monkeypatch.setattr(pipeline_factory_module, "Context", lambda **kw: SimpleNamespace(**kw))
    ctx = await _build_context(DESIGNATED_HOTKEY, first_pass=False)
    assert ctx.config.designated_hotkey_first_pass is False


@pytest.mark.asyncio
async def test_build_context_provider_with_every_designated_looking_field_is_unmarked(
    profile_on, monkeypatch
):
    """The profile is keyed on the miner hotkey the validator authenticated. A provider node that
    reports every field a designated-hotkey node would — the same address family, root dir, username,
    even the designated hotkey as its *executor* fields — still runs the full pipeline: nothing an executor
    reports is read."""
    monkeypatch.setattr(pipeline_factory_module, "Context", lambda **kw: SimpleNamespace(**kw))
    ctx = await _build_context(
        PROVIDER_HOTKEY,
        first_pass=True,
        executor_overrides={
            "uuid": DESIGNATED_HOTKEY,
            "miner_hotkey": DESIGNATED_HOTKEY,
            "designated": True,
            "fast_validation": True,
            "provider_display_name": "Designated",
            "operator_managed": True,
        },
    )
    assert ctx.config.designated_hotkey_first_pass is False


@pytest.mark.asyncio
async def test_build_context_flag_off_keeps_the_designated_hotkey_unmarked(monkeypatch):
    monkeypatch.setattr(settings, "DESIGNATED_HOTKEY_FAST_VALIDATION_ENABLED", False)
    monkeypatch.setattr(settings, "DESIGNATED_MINER_HOTKEYS", DESIGNATED_HOTKEY)
    monkeypatch.setattr(pipeline_factory_module, "Context", lambda **kw: SimpleNamespace(**kw))
    ctx = await _build_context(DESIGNATED_HOTKEY, first_pass=True)
    assert ctx.config.designated_hotkey_first_pass is False


def test_the_check_list_is_the_same_for_every_node():
    """No check is removed from the pipeline: the profile decides inside each affected check, so
    the checks that catch a broken node are on the list for a designated-hotkey node as for any other."""
    ids = [type(c).__name__ for c in PipelineFactory.build_checks()]
    for kept in (
        "MachineSpecScrapeCheck",
        "GpuCountCheck",
        "GpuModelValidCheck",
        "GpuVramPrecheck",
        "DiskHealthCheck",
        "NvmlDigestCheck",
        "GpuFingerprintCheck",
        "PortConnectivityCheck",
        "PortCountCheck",
        "SysboxRequiredCheck",
        "GpuUsageCheck",
        "CapabilityCheck",
        "RentalVerificationCheck",
        "CollateralCheck",
        "VerifyXCheck",
    ):
        assert kept in ids


# --- skipped step: collateral ------------------------------------------------------------------


def _collateral_ctx(
    context_factory, designated: bool, service: DummyCollateralService, enable_no_collateral=False
):
    specs = {"gpu": {"count": 8, "details": [{"name": "NVIDIA B300 SXM6 AC"}]}}
    return context_factory(
        services=build_services(collateral=service),
        config=build_context_config(
            designated_hotkey_first_pass=designated, enable_no_collateral=enable_no_collateral
        ),
        state=build_state(specs=specs, gpu_count=8, gpu_details=specs["gpu"]["details"]),
        miner_hotkey=DESIGNATED_HOTKEY if designated else PROVIDER_HOTKEY,
    )


@pytest.mark.asyncio
async def test_designated_hotkey_first_pass_skips_the_collateral_read(context_factory):
    service = DummyCollateralService(deposited=False, error="no bond", contract_version="1.0.2")
    result = await CollateralCheck().run(
        _collateral_ctx(context_factory, designated=True, service=service)
    )

    assert result.passed is True
    assert result.event.reason_code == ColMsg.DESIGNATED_HOTKEY_SKIPPED.reason
    assert result.event.what_we_saw["skipped"] is True
    assert service.called_with is None  # no on-chain read
    # the published row says what is true
    assert result.updates == {
        "collateral_deposited": False,
        "collateral_error_message": None,
        "contract_version": None,
    }


@pytest.mark.asyncio
async def test_provider_node_still_pays_the_collateral_read(context_factory):
    service = DummyCollateralService(deposited=False, error="no bond", contract_version="1.0.2")
    result = await CollateralCheck().run(
        _collateral_ctx(context_factory, designated=False, service=service)
    )

    assert result.passed is False
    assert result.event.reason_code == ColMsg.MISSING.reason
    assert service.called_with is not None


# --- skipped step: VerifyX ---------------------------------------------------------------------


def _verifyx_ctx(context_factory, designated: bool, service: DummyVerifyXService):
    scrape_specs = {
        "gpu": {"count": 8},
        "ram": {"total": 2_000_000},
        "hard_disk": {"total": 900, "free": 800, "utilization": 11},
        "network": {"download_speed": 812.5, "upload_speed": 640.0},
    }
    return context_factory(
        services=build_services(verifyx=service),
        config=build_context_config(verifyx_enabled=True, designated_hotkey_first_pass=designated),
        state=build_state(specs=scrape_specs),
        miner_hotkey=DESIGNATED_HOTKEY if designated else PROVIDER_HOTKEY,
    )


@pytest.mark.asyncio
async def test_designated_hotkey_first_pass_skips_verifyx_and_publishes_the_scrape_readings(
    context_factory,
):
    service = DummyVerifyXService(success=True, updated_specs={"ram": {"total": 1}})
    ctx = _verifyx_ctx(context_factory, designated=True, service=service)
    result = await VerifyXCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == VxMsg.DESIGNATED_HOTKEY_SKIPPED.reason
    assert result.event.what_we_saw["bandwidth_gate"] == "deferred_to_first_scored_cycle"
    assert result.event.what_we_saw["network"] == {"download_speed": 812.5, "upload_speed": 640.0}
    assert service.called_with is None  # the probe never ran
    # state untouched: the scrape's readings are what the backend gets, and no VerifyX EMA is seeded,
    # so the first scored cycle bootstraps from its own sample as on any never-measured node
    assert result.updates == {}
    assert "ema_verifyx_download_speed" not in ctx.state.specs["network"]


@pytest.mark.asyncio
async def test_provider_node_still_runs_verifyx(context_factory):
    service = DummyVerifyXService(success=False, error_msg="probe failed")
    result = await VerifyXCheck().run(
        _verifyx_ctx(context_factory, designated=False, service=service)
    )

    assert result.passed is False
    assert service.called_with is not None


@pytest.mark.asyncio
async def test_verifyx_disabled_wins_over_the_profile(context_factory):
    """Nothing new is run when VerifyX is off fleet-wide; the profile only ever removes work."""
    service = DummyVerifyXService(success=True)
    ctx = context_factory(
        services=build_services(verifyx=service),
        config=build_context_config(verifyx_enabled=False, designated_hotkey_first_pass=True),
        state=build_state(specs={"gpu": {"count": 8}}),
    )
    result = await VerifyXCheck().run(ctx)
    assert result.event.reason_code == VxMsg.DISABLED.reason
    assert service.called_with is None


# --- kept, resized: the GPU compute probe --------------------------------------------------------


class _SizingAwareValidationService(DummyValidationService):
    async def validate_gpu_model_and_process_job(
        self, *, ssh_client, executor_info, default_extra, machine_spec, **kw
    ):
        self.sizing_kwargs = kw
        return await super().validate_gpu_model_and_process_job(
            ssh_client=ssh_client,
            executor_info=executor_info,
            default_extra=default_extra,
            machine_spec=machine_spec,
        )


def _capability_ctx(context_factory, designated: bool, service):
    return context_factory(
        services=build_services(validation=service),
        config=build_context_config(designated_hotkey_first_pass=designated),
        state=build_state(specs={"gpu": {"count": 8}}),
    )


@pytest.mark.asyncio
async def test_designated_hotkey_first_pass_keeps_the_matmul_at_the_first_pass_budget(
    context_factory, monkeypatch
):
    monkeypatch.setattr(
        settings, "FIRST_PASS_FAST_PATH_ENABLED", False
    )  # the profile does not need DAH-3011's flag
    service = _SizingAwareValidationService(success=True)
    result = await CapabilityCheck().run(
        _capability_ctx(context_factory, designated=True, service=service)
    )

    assert result.passed is True
    assert result.event.reason_code == CapMsg.VERIFY_OK.reason
    assert service.sizing_kwargs == {"vram_budget_mb": settings.FIRST_PASS_MATMUL_VRAM_MB}
    assert (
        result.event.what_we_saw["first_pass_vram_budget_mb"] == settings.FIRST_PASS_MATMUL_VRAM_MB
    )


@pytest.mark.asyncio
async def test_designated_hotkey_first_pass_still_fails_a_card_that_cannot_compute(context_factory):
    service = _SizingAwareValidationService(success=False, error_message="UUID mismatch")
    result = await CapabilityCheck().run(
        _capability_ctx(context_factory, designated=True, service=service)
    )
    assert result.passed is False


@pytest.mark.asyncio
async def test_provider_node_matmul_fills_the_card(context_factory, monkeypatch):
    monkeypatch.setattr(settings, "FIRST_PASS_FAST_PATH_ENABLED", False)
    service = _SizingAwareValidationService(success=True)
    result = await CapabilityCheck().run(
        _capability_ctx(context_factory, designated=False, service=service)
    )
    assert result.passed is True
    assert service.sizing_kwargs == {}


# --- kept whole: the checks that catch a broken node ----------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("designated", [True, False])
async def test_gpu_count_gate_is_the_same_for_both(context_factory, designated):
    ctx = context_factory(
        config=build_context_config(max_gpu_count=8, designated_hotkey_first_pass=designated),
        state=build_state(specs={"gpu": {"count": 9}}, gpu_count=9),
    )
    result = await GpuCountCheck().run(ctx)
    assert result.passed is False
    assert result.event.reason_code == GpuCountMsg.COUNT_EXCEEDS.reason


@pytest.mark.asyncio
@pytest.mark.parametrize("designated", [True, False])
async def test_vram_gate_is_the_same_for_both(context_factory, designated):
    details = [{"name": "NVIDIA H100 80GB HBM3", "uuid": "GPU-abc", "capacity": 24064}]
    ctx = context_factory(
        config=build_context_config(designated_hotkey_first_pass=designated),
        state=build_state(gpu_model="NVIDIA H100 80GB HBM3", gpu_count=1, gpu_details=details),
    )
    result = await GpuVramPrecheck().run(ctx)
    assert result.passed is False
    assert result.event.reason_code == "GPU_VRAM_MISMATCH"


@pytest.mark.asyncio
@pytest.mark.parametrize("designated", [True, False])
async def test_port_bind_gate_is_the_same_for_both(context_factory, designated):
    ctx = context_factory(
        config=build_context_config(designated_hotkey_first_pass=designated),
        state=build_state(verified_port_count=MIN_PORT_COUNT - 1),
    )
    result = await PortCountCheck().run(ctx)
    assert result.passed is False
    assert result.event.reason_code == PortMsg.INSUFFICIENT_PORTS.reason


@pytest.mark.asyncio
@pytest.mark.parametrize("designated", [True, False])
async def test_sysbox_gate_is_the_same_for_both(context_factory, designated):
    ctx = context_factory(
        config=build_context_config(designated_hotkey_first_pass=designated),
        state=build_state(sysbox_runtime=False),
    )
    result = await SysboxRequiredCheck().run(ctx)
    assert result.passed is False
    assert result.event.reason_code == SysboxMsg.SYSBOX_MISSING.reason


@pytest.mark.asyncio
@pytest.mark.parametrize("designated", [True, False])
async def test_rental_verification_still_rents_the_probe_container_for_both(designated):
    backend = DummyBackendClient(
        response=ExecutorHealthCheckResponse(
            success=False, error="Container not responding", details={"timeout": True}
        )
    )
    ctx = make_context(
        services=build_services(backend=backend, container_cleanup=ContainerCleanup()),
        config=build_context_config(designated_hotkey_first_pass=designated),
        state=build_state(specs={"verified_ports": [8080]}),
        miner_hotkey=DESIGNATED_HOTKEY if designated else PROVIDER_HOTKEY,
    )
    with patch(
        "neurons.validators.src.services.task.checks.rental_verification.settings"
    ) as mock_settings:
        mock_settings.SKIP_RENTAL_VERIFICATION = False
        result = await RentalVerificationCheck().run(ctx)

    assert backend.called_with is not None
    assert backend.called_with["container_port"] == 8080
    assert result.passed is False
    assert result.event.reason_code == RentMsg.FAILED.reason


# --- the score the backend lists on -----------------------------------------------------------


def _score_ctx(
    designated: bool, specs: dict, *, price_per_gpu=None, collateral_deposited=False, **flags
):
    executor = default_executor().model_copy(update={"price_per_gpu": price_per_gpu})
    return make_context(
        executor=executor,
        state=build_state(specs=specs),
        config=build_context_config(designated_hotkey_first_pass=designated),
        collateral_deposited=collateral_deposited,
        **flags,
    )


def test_designated_hotkey_first_pass_scores_positive_without_verifyx_ema_or_collateral(
    monkeypatch,
):
    monkeypatch.setattr(
        settings, "ENABLE_NO_COLLATERAL", False
    )  # the strict setting: collateral fatal for providers
    ctx = _score_ctx(True, {"network": {"download_speed": 812.5}}, collateral_deposited=False)
    actual, job, warning = calculate_scores(ctx, rented=False)
    assert (actual, job) == (1.0, 1.0)
    assert "deferred to the first scored cycle" in warning


def test_provider_node_without_verifyx_ema_scores_zero(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_NO_COLLATERAL", False)
    ctx = _score_ctx(False, {"network": {"download_speed": 812.5}}, collateral_deposited=False)
    actual, job, warning = calculate_scores(ctx, rented=False)
    assert (actual, job) == (0.0, 0.0)
    assert "unavailable" in warning


def test_provider_node_without_collateral_scores_zero_when_collateral_is_required(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_NO_COLLATERAL", False)
    ctx = _score_ctx(
        False, {"network": {"ema_verifyx_download_speed": 500.0}}, collateral_deposited=False
    )
    actual, job, warning = calculate_scores(ctx, rented=False)
    assert (actual, job) == (0.0, 0.0)
    assert "Collateral required" in warning


@pytest.mark.parametrize(
    "flags,fragment",
    [
        ({"cpu_truth_passed": False}, "CPU cores"),
        ({"provider_side_load_passed": False}, "Provider-side workload"),
        ({"inspector_passed": False}, "Inspector"),
    ],
)
def test_designated_hotkey_first_pass_keeps_every_broken_or_hostile_host_gate(flags, fragment):
    ctx = _score_ctx(True, {"network": {"download_speed": 812.5}}, **flags)
    actual, job, warning = calculate_scores(ctx, rented=False)
    assert (actual, job) == (0.0, 0.0)
    assert fragment in warning


def test_designated_hotkey_first_pass_keeps_the_price_cap(monkeypatch):
    from core.config import shared_client

    capped = shared_client.config.model_copy(
        update={"machine_prices": {"NVIDIA B300 SXM6 AC": 5.0}, "machine_max_price_rate": 2.0}
    )
    monkeypatch.setattr(shared_client, "_config", capped)
    ctx = make_context(
        executor=default_executor().model_copy(update={"price_per_gpu": 50.0}),
        state=build_state(
            specs={"network": {"download_speed": 812.5}}, gpu_model="NVIDIA B300 SXM6 AC"
        ),
        config=build_context_config(designated_hotkey_first_pass=True),
    )
    actual, job, warning = calculate_scores(ctx, rented=False)
    assert actual == 0.0
    assert "price exceeds" in warning
