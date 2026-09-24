"""The fast-validation profile for nodes registered under designated miner hotkeys.

Applied ONLY to the FIRST, unscored verification (the express lane's `first_pass=True` call) of an
executor whose miner hotkey is in DESIGNATED_MINER_HOTKEYS, and only with
DESIGNATED_HOTKEY_FAST_VALIDATION_ENABLED on. The profile only resizes the capability matmul (kept,
at the first-pass VRAM budget). The collateral read, VerifyX and every check that catches a broken
node run and gate as on every pass. A provider node — whatever it reports about itself — takes
today's full pipeline; so does every scored cycle of the designated-hotkey node.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from neurons.validators.src.protocol.vc_protocol.compute_requests import ExecutorHealthCheckResponse
from neurons.validators.src.services.container_cleanup import ContainerCleanup
from neurons.validators.src.services.task import pipeline_factory as pipeline_factory_module
from neurons.validators.src.services.task.checks.capability import CapabilityCheck
from neurons.validators.src.services.task.checks.gpu_count import GpuCountCheck
from neurons.validators.src.services.task.checks.gpu_vram_precheck import GpuVramPrecheck
from neurons.validators.src.services.task.checks.port_count import PortCountCheck
from neurons.validators.src.services.task.checks.rental_verification import RentalVerificationCheck
from neurons.validators.src.services.task.checks.sysbox_required import SysboxRequiredCheck
from neurons.validators.src.services.task.checks.verifyx import VerifyXCheck
from neurons.validators.src.services.task.messages import CapabilityMessages as CapMsg
from neurons.validators.src.services.task.messages import GpuCountMessages as GpuCountMsg
from neurons.validators.src.services.task.messages import PortCountMessages as PortMsg
from neurons.validators.src.services.task.messages import RentalVerificationMessages as RentMsg
from neurons.validators.src.services.task.messages import SysboxRequiredMessages as SysboxMsg
from neurons.validators.src.services.task.messages import VerifyXMessages as VxMsg
from neurons.validators.src.services.task.pipeline import Context
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
from tests.test_rental_verification_check import DummyBackendClient
from tests.test_verifyx_check import DummyVerifyXService

# lium-io#1440 removes the validator collateral check; these tests then skip instead of failing.
try:
    from neurons.validators.src.services.task.checks.collateral import CollateralCheck
    from neurons.validators.src.services.task.messages import CollateralMessages as ColMsg
    from tests.test_collateral_check import DummyCollateralService

    HAS_COLLATERAL = True
except ImportError:
    HAS_COLLATERAL = False

needs_collateral = pytest.mark.skipif(
    not HAS_COLLATERAL, reason="the validator collateral check is removed"
)

# ss58 addresses of fixed byte patterns, not anyone's keys: the lists only take ss58 hotkeys, and
# the profile compares the authenticated miner hotkey string as-is
DESIGNATED_HOTKEY = "5C62Ck4UrFPiBtoCmeSrgF7x9yv9mn38446dhCpsi2mLHiFT"
OTHER_DESIGNATED_HOTKEY = "5C7LYpP2ZH3tpKbvVvwiVe54AapxErdPBbvkYhe6y9ZBkqWt"
PROVIDER_HOTKEY = "5C8etthaGJi5SkQeEDSaK32ABBjkhwDeK9ksQCTLEGM3EH14"


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
# first pass the profile, so config load refuses the overlap, and with the flag on it refuses an
# empty pool mirror, where the overlap cannot be checked.

POOL_HOTKEY = "5C9yEy27yLNG5BDMxVwS8RyGBneZB1ouShazFhGZVP8thK5z"
OTHER_POOL_HOTKEY = "5CBHb3LfgN2Shc25gnSHwpvNCPZMe6QAaFR77C5nkVvkAK1o"
UNRELATED_POOL_HOTKEY = "5CCbw7fDPPgdL2poR4w9mDsUCzUA7AzRhoFDxgu21cibdUmW"


@pytest.mark.parametrize("flag", [True, False])
def test_config_load_refuses_a_designated_hotkey_that_is_a_pool_hotkey(flag):
    from pydantic import ValidationError

    from core.config import Settings

    with pytest.raises(ValidationError, match="dedicated hotkey"):
        Settings(
            _env_file=None,
            DESIGNATED_HOTKEY_FAST_VALIDATION_ENABLED=flag,
            DESIGNATED_MINER_HOTKEYS=f"{DESIGNATED_HOTKEY}, {POOL_HOTKEY}",
            LIUM_POOL_HOTKEYS=f"{POOL_HOTKEY},{OTHER_POOL_HOTKEY}",
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


@pytest.mark.parametrize("designated", [DESIGNATED_HOTKEY, ""])
@pytest.mark.parametrize("pool", ["", " , ", "[]", " [ ] ", '[""]', '[" ", ""]'])
def test_config_load_refuses_the_flag_on_with_an_empty_pool_mirror(designated, pool):
    from pydantic import ValidationError

    from core.config import Settings

    with pytest.raises(ValidationError, match="LIUM_POOL_HOTKEYS empty"):
        Settings(
            _env_file=None,
            DESIGNATED_HOTKEY_FAST_VALIDATION_ENABLED=True,
            DESIGNATED_MINER_HOTKEYS=designated,
            LIUM_POOL_HOTKEYS=pool,
        )


def test_config_load_accepts_an_empty_pool_mirror_with_the_flag_off():
    from core.config import Settings

    s = Settings(
        _env_file=None,
        DESIGNATED_HOTKEY_FAST_VALIDATION_ENABLED=False,
        DESIGNATED_MINER_HOTKEYS=DESIGNATED_HOTKEY,
        LIUM_POOL_HOTKEYS="",
    )
    assert s.lium_pool_hotkeys() == frozenset()


# The portal's env form of LIUM_POOL_HOTKEYS is a JSON list; the mirror takes it verbatim.


@pytest.mark.parametrize("flag", [True, False])
@pytest.mark.parametrize(
    "designated",
    [f"{DESIGNATED_HOTKEY},{POOL_HOTKEY}", f'["{DESIGNATED_HOTKEY}", "{POOL_HOTKEY}"]'],
)
def test_config_load_refuses_a_pool_hotkey_given_in_the_portals_json_form(flag, designated):
    from pydantic import ValidationError

    from core.config import Settings

    with pytest.raises(ValidationError, match="dedicated hotkey"):
        Settings(
            _env_file=None,
            DESIGNATED_HOTKEY_FAST_VALIDATION_ENABLED=flag,
            DESIGNATED_MINER_HOTKEYS=designated,
            LIUM_POOL_HOTKEYS=f'["{POOL_HOTKEY}", "{OTHER_POOL_HOTKEY}"]',
        )


@pytest.mark.parametrize(
    "pool",
    [
        f'["{POOL_HOTKEY}", "{OTHER_POOL_HOTKEY}"]',
        f'  [ " {POOL_HOTKEY} " , "", "{OTHER_POOL_HOTKEY}" ]  ',
        f"{POOL_HOTKEY}, {OTHER_POOL_HOTKEY}",
    ],
)
@pytest.mark.parametrize(
    "designated",
    [
        f"{DESIGNATED_HOTKEY},{OTHER_DESIGNATED_HOTKEY}",
        f'["{DESIGNATED_HOTKEY}", "{OTHER_DESIGNATED_HOTKEY}"]',
    ],
)
def test_config_load_reads_the_json_and_comma_forms_alike(designated, pool):
    from core.config import Settings

    s = Settings(
        _env_file=None,
        DESIGNATED_HOTKEY_FAST_VALIDATION_ENABLED=True,
        DESIGNATED_MINER_HOTKEYS=designated,
        LIUM_POOL_HOTKEYS=pool,
    )
    assert s.lium_pool_hotkeys() == frozenset({POOL_HOTKEY, OTHER_POOL_HOTKEY})
    assert s.designated_miner_hotkeys() == frozenset({DESIGNATED_HOTKEY, OTHER_DESIGNATED_HOTKEY})
    assert s.is_designated_hotkey_first_pass(DESIGNATED_HOTKEY, first_pass=True) is True
    assert s.is_designated_hotkey_first_pass(POOL_HOTKEY, first_pass=True) is False


# (value, the refusal it gets)
NOT_JSON = "starts with '\\[' but is not a valid JSON list"
NOT_STRINGS = "entry \\d+ is not a string; .* must be a JSON list of strings"
NOT_SS58 = "entry \\d+ is not an ss58 hotkey"
MALFORMED_HOTKEY_LISTS = [
    (f'["{POOL_HOTKEY}"', NOT_JSON),  # unclosed JSON list
    (f'["{POOL_HOTKEY}",]', NOT_JSON),  # trailing comma: not JSON
    (f"['{POOL_HOTKEY}']", NOT_JSON),  # Python repr, not JSON
    (f'[1, "{POOL_HOTKEY}"]', NOT_STRINGS),
    (f'["{POOL_HOTKEY}", null]', NOT_STRINGS),
    (f'[["{POOL_HOTKEY}"]]', NOT_STRINGS),
    (f'{{"hotkeys": ["{POOL_HOTKEY}"]}}', NOT_SS58),  # JSON object
    (f'"{POOL_HOTKEY}"', NOT_SS58),  # a quoted string
    (f'"{POOL_HOTKEY}", "{OTHER_POOL_HOTKEY}"', NOT_SS58),  # a JSON list without brackets
    (f"{POOL_HOTKEY} {OTHER_POOL_HOTKEY}", NOT_SS58),  # space-separated
    (f"{POOL_HOTKEY}\t{OTHER_POOL_HOTKEY}", NOT_SS58),  # tab-separated
    (f"{POOL_HOTKEY};{OTHER_POOL_HOTKEY}", NOT_SS58),  # semicolon-separated
    (f"{POOL_HOTKEY}|{OTHER_POOL_HOTKEY}", NOT_SS58),  # pipe-separated
    (f'["{POOL_HOTKEY};{OTHER_POOL_HOTKEY}"]', NOT_SS58),
    (f"{POOL_HOTKEY}\u200b,{OTHER_POOL_HOTKEY}", NOT_SS58),  # zero-width space: strip() keeps it
    (f"\u200b{POOL_HOTKEY}", NOT_SS58),
    (f'["{POOL_HOTKEY}\u200b"]', NOT_SS58),
    ("lium-pool-hotkey", NOT_SS58),  # not an address
    ("0x" + "ab" * 32, NOT_SS58),  # a hex public key
    ("F7NZ", NOT_SS58),  # a valid ss58 index address, not a hotkey
    (POOL_HOTKEY[:-1], NOT_SS58),  # one character short
    (POOL_HOTKEY + "1", NOT_SS58),  # one character long
    (POOL_HOTKEY[:5] + "0" + POOL_HOTKEY[6:], NOT_SS58),  # 0, O, I and l are not base58
    (POOL_HOTKEY[:5] + "O" + POOL_HOTKEY[6:], NOT_SS58),
    (POOL_HOTKEY[:5] + "I" + POOL_HOTKEY[6:], NOT_SS58),
    (POOL_HOTKEY[:5] + "l" + POOL_HOTKEY[6:], NOT_SS58),
    (POOL_HOTKEY[:-1] + "1", NOT_SS58),  # base58, right length, bad checksum
]


@pytest.mark.parametrize("flag", [True, False])
@pytest.mark.parametrize(("pool", "refusal"), MALFORMED_HOTKEY_LISTS)
def test_config_load_refuses_a_malformed_pool_mirror(flag, pool, refusal):
    from pydantic import ValidationError

    from core.config import Settings

    with pytest.raises(ValidationError, match=f"LIUM_POOL_HOTKEYS .*{refusal}"):
        Settings(
            _env_file=None,
            DESIGNATED_HOTKEY_FAST_VALIDATION_ENABLED=flag,
            DESIGNATED_MINER_HOTKEYS=POOL_HOTKEY,
            LIUM_POOL_HOTKEYS=pool,
        )


@pytest.mark.parametrize(("designated", "refusal"), MALFORMED_HOTKEY_LISTS)
def test_config_load_refuses_a_malformed_designated_list(designated, refusal):
    from pydantic import ValidationError

    from core.config import Settings

    with pytest.raises(ValidationError, match=f"DESIGNATED_MINER_HOTKEYS .*{refusal}"):
        Settings(
            _env_file=None,
            DESIGNATED_HOTKEY_FAST_VALIDATION_ENABLED=True,
            DESIGNATED_MINER_HOTKEYS=designated,
            LIUM_POOL_HOTKEYS=UNRELATED_POOL_HOTKEY,
        )


@pytest.mark.parametrize("name", ["DESIGNATED_MINER_HOTKEYS", "LIUM_POOL_HOTKEYS"])
@pytest.mark.parametrize(
    "value",
    [
        f"{DESIGNATED_HOTKEY},,{OTHER_DESIGNATED_HOTKEY[:-1]}",
        f'["{DESIGNATED_HOTKEY}", "", "{OTHER_DESIGNATED_HOTKEY[:-1]}"]',
    ],
)
def test_the_refusal_names_the_entry_index_not_its_value(name, value):
    from pydantic import ValidationError

    from core.config import Settings

    lists = {
        "DESIGNATED_MINER_HOTKEYS": PROVIDER_HOTKEY,
        "LIUM_POOL_HOTKEYS": UNRELATED_POOL_HOTKEY,
        name: value,
    }
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None, DESIGNATED_HOTKEY_FAST_VALIDATION_ENABLED=True, **lists)
    message = str(excinfo.value)
    assert f"{name} entry 2 is not an ss58 hotkey" in message
    assert OTHER_DESIGNATED_HOTKEY[:-1] not in message
    assert DESIGNATED_HOTKEY not in message


def test_config_load_reads_the_portals_json_mirror_from_the_environment(monkeypatch):
    from pydantic import ValidationError

    from core.config import Settings

    monkeypatch.setenv("DESIGNATED_HOTKEY_FAST_VALIDATION_ENABLED", "true")
    monkeypatch.setenv("DESIGNATED_MINER_HOTKEYS", POOL_HOTKEY)
    monkeypatch.setenv("LIUM_POOL_HOTKEYS", f'["{POOL_HOTKEY}"]')
    with pytest.raises(ValidationError, match="dedicated hotkey"):
        Settings(_env_file=None)


@pytest.mark.parametrize(
    ("flag", "designated", "pool", "expected_fragment"),
    [
        (False, DESIGNATED_HOTKEY, "", None),  # flag off: nothing to say
        (True, "", POOL_HOTKEY, "empty DESIGNATED_MINER_HOTKEYS"),  # on, selects nobody
        (True, DESIGNATED_HOTKEY, POOL_HOTKEY, None),  # on, rule checked at load
    ],
)
def test_startup_warning_names_a_flag_that_selects_nobody(
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
    miner_hotkey: str, first_pass: bool, executor_overrides: dict[str, object] | None = None
) -> Context:
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
    # independent of FIRST_PASS_FAST_PATH_ENABLED: it is off here and the profile is whole
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
        "VerifyXCheck",
        *(("CollateralCheck",) if HAS_COLLATERAL else ()),
    ):
        assert kept in ids


# --- kept step: collateral --------------------------------------------------------------------


def _collateral_ctx(context_factory, designated: bool, service, enable_no_collateral=False):
    specs = {"gpu": {"count": 8, "details": [{"name": "NVIDIA B300 SXM6 AC"}]}}
    return context_factory(
        services=build_services(collateral=service),
        config=build_context_config(
            designated_hotkey_first_pass=designated, enable_no_collateral=enable_no_collateral
        ),
        state=build_state(specs=specs, gpu_count=8, gpu_details=specs["gpu"]["details"]),
        miner_hotkey=DESIGNATED_HOTKEY if designated else PROVIDER_HOTKEY,
    )


@needs_collateral
@pytest.mark.asyncio
async def test_designated_hotkey_first_pass_still_reads_collateral(context_factory):
    service = DummyCollateralService(deposited=False, error="no bond", contract_version="1.0.2")
    result = await CollateralCheck().run(
        _collateral_ctx(context_factory, designated=True, service=service)
    )

    assert result.passed is False
    assert result.event.reason_code == ColMsg.MISSING.reason
    assert service.called_with is not None


@needs_collateral
@pytest.mark.asyncio
async def test_provider_node_still_pays_the_collateral_read(context_factory):
    service = DummyCollateralService(deposited=False, error="no bond", contract_version="1.0.2")
    result = await CollateralCheck().run(
        _collateral_ctx(context_factory, designated=False, service=service)
    )

    assert result.passed is False
    assert result.event.reason_code == ColMsg.MISSING.reason
    assert service.called_with is not None


# --- kept step: VerifyX -----------------------------------------------------------------------


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
async def test_designated_hotkey_first_pass_still_runs_verifyx(context_factory):
    service = DummyVerifyXService(success=False, error_msg="probe failed")
    result = await VerifyXCheck().run(
        _verifyx_ctx(context_factory, designated=True, service=service)
    )

    assert result.passed is False
    assert service.called_with is not None


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
    """VerifyX off fleet-wide stays off on the profile."""
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
    )  # the profile does not need FIRST_PASS_FAST_PATH_ENABLED
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
    designated: bool,
    specs: dict[str, object],
    *,
    price_per_gpu: float | None = None,
    collateral_deposited: bool = False,
    **flags: bool,
) -> Context:
    executor = default_executor().model_copy(update={"price_per_gpu": price_per_gpu})
    return make_context(
        executor=executor,
        state=build_state(specs=specs),
        config=build_context_config(designated_hotkey_first_pass=designated),
        collateral_deposited=collateral_deposited,
        **flags,
    )


@pytest.fixture
def collateral_required(monkeypatch):
    if HAS_COLLATERAL:
        monkeypatch.setattr(settings, "ENABLE_NO_COLLATERAL", False)


@pytest.mark.parametrize("designated", [True, False])
@pytest.mark.parametrize(
    "specs,collateral_deposited,fragment",
    [
        ({"network": {"download_speed": 812.5}}, True, "unavailable"),
        pytest.param(
            {"network": {"ema_verifyx_download_speed": 500.0}},
            False,
            "Collateral required",
            marks=needs_collateral,
        ),
    ],
)
def test_designated_hotkey_first_pass_keeps_the_verifyx_and_collateral_gates(
    collateral_required, designated, specs, collateral_deposited, fragment
):
    ctx = _score_ctx(designated, specs, collateral_deposited=collateral_deposited)
    actual, job, warning = calculate_scores(ctx, rented=False)
    assert (actual, job) == (0.0, 0.0)
    assert fragment in warning


def test_provider_node_without_verifyx_ema_scores_zero(collateral_required):
    ctx = _score_ctx(False, {"network": {"download_speed": 812.5}}, collateral_deposited=False)
    actual, job, warning = calculate_scores(ctx, rented=False)
    assert (actual, job) == (0.0, 0.0)
    assert "unavailable" in warning


@needs_collateral
def test_provider_node_without_collateral_scores_zero_when_collateral_is_required(
    collateral_required,
):
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
