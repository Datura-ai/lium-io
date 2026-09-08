import pytest

from neurons.validators.src.services.task.checks.tdx_host import TdxHostCheck
from neurons.validators.src.services.task.messages import TdxHostMessages as Msg
from tests.helpers import build_context_config, build_services, build_state


@pytest.mark.parametrize(
    "cpu_model,expected",
    [
        # TDX-capable: 5th Gen Xeon Scalable (Emerald Rapids)
        ("Intel(R) Xeon(R) Platinum 8592+ @ 1.90GHz", True),
        ("Intel(R) Xeon(R) Platinum 8580", True),
        ("Intel(R) Xeon(R) Gold 6554S", True),
        ("Intel(R) Xeon(R) Gold 6526Y", True),
        # TDX-capable: 4th Gen Xeon Scalable (Sapphire Rapids)
        ("Intel(R) Xeon(R) Platinum 8490H", True),
        ("Intel(R) Xeon(R) Platinum 8460Y+", True),
        ("Intel(R) Xeon(R) Gold 6442Y", True),
        ("Intel(R) Xeon(R) Gold 6430", True),
        # TDX-capable: Intel Xeon 6 P-core (Granite Rapids)
        ("Intel(R) Xeon(R) 6980P", True),
        ("Intel(R) Xeon(R) 6960P", True),
        # TDX-capable: Intel Xeon 6 E-core (Sierra Forest)
        ("Intel(R) Xeon(R) 6780E", True),
        ("Intel(R) Xeon(R) 6766E", True),
        # TDX-capable: case-insensitive match
        ("INTEL(R) XEON(R) PLATINUM 8592+", True),
        # TDX-capable: inside a CVM lscpu reports the raw "family/model" CPUID instead of a name
        ("06/cf", True),  # Emerald Rapids, the H200 CVMs
        ("06/ad", True),  # Granite Rapids, the B200 CVM
        ("06/8f", True),  # Sapphire Rapids
        ("06/af", True),  # Sierra Forest
        ("06/AE", True),  # Granite Rapids D, uppercase hex
        # Not TDX-capable: raw CPUID of a family 6 model without TDX (Alder Lake client)
        ("06/97", False),
        # Not TDX-capable: a TDX model id under a different family
        ("0f/cf", False),
        # Not TDX-capable: 3rd Gen Xeon Scalable (Ice Lake)
        ("Intel(R) Xeon(R) Platinum 8380", False),
        ("Intel(R) Xeon(R) Gold 6330", False),
        # Not TDX-capable: 2nd Gen Xeon Scalable (Cascade Lake)
        ("Intel(R) Xeon(R) Platinum 8280", False),
        # Not TDX-capable: AMD
        ("AMD EPYC 9654 96-Core Processor", False),
        ("AMD EPYC 7452 32-Core Processor", False),
        # Not TDX-capable: consumer/workstation Intel
        ("Intel(R) Core(TM) i9-13900K", False),
        ("Intel(R) Xeon(R) E5-2686 v4", False),
        # Not TDX-capable: empty string
        ("", False),
    ],
)
def test_is_tdx_capable(cpu_model, expected):
    # is_tdx_capable is a pure staticmethod — no context or async needed
    assert TdxHostCheck.is_tdx_capable(cpu_model) is expected


@pytest.mark.parametrize(
    "specs,expected_pass,expected_reason,expected_tdx_flag",
    [
        # TDX-capable CPU — check passes and sets tdx_host_supported=True
        (
            {"cpu": {"model": "Intel(R) Xeon(R) Platinum 8592+"}},
            True,
            Msg.TDX_SUPPORTED.reason,
            True,
        ),
        # Non-TDX CPU — check passes (non-fatal) and sets tdx_host_supported=False
        (
            {"cpu": {"model": "AMD EPYC 7452 32-Core Processor"}},
            True,
            Msg.TDX_NOT_SUPPORTED.reason,
            False,
        ),
        # Missing cpu key in specs — safe default False
        (
            {},
            True,
            Msg.TDX_NOT_SUPPORTED.reason,
            False,
        ),
        # cpu key present but model is empty string
        (
            {"cpu": {"model": ""}},
            True,
            Msg.TDX_NOT_SUPPORTED.reason,
            False,
        ),
    ],
)
@pytest.mark.asyncio
async def test_tdx_host_check_run(specs, expected_pass, expected_reason, expected_tdx_flag, context_factory):
    # Arrange
    services = build_services()
    config = build_context_config()
    state = build_state(specs=specs)
    ctx = context_factory(services=services, config=config, state=state)

    # Act
    result = await TdxHostCheck().run(ctx)

    # Assert — check is always non-fatal regardless of TDX support
    assert result.passed is expected_pass
    assert result.event.reason_code == expected_reason
    # tdx_host_supported must be written into the updated specs
    assert result.updates["state"].specs["tdx_host_supported"] is expected_tdx_flag
