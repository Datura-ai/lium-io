import pytest

from neurons.validators.src.services.task.checks.collateral import CollateralCheck
from neurons.validators.src.services.task.messages import CollateralMessages as Msg

from tests.helpers import build_context_config, build_services, build_state


class DummyCollateralService:
    def __init__(self, *, deposited: bool, error: str | None, contract_version: str | None):
        self.deposited = deposited
        self.error = error
        self.contract_version = contract_version
        self.called_with: dict[str, str | int | None] | None = None

    async def is_eligible_executor(
        self,
        *,
        miner_hotkey: str,
        executor_uuid: str,
        gpu_model: str | None,
        gpu_count: int,
    ):
        self.called_with = {
            "miner_hotkey": miner_hotkey,
            "executor_uuid": executor_uuid,
            "gpu_model": gpu_model,
            "gpu_count": gpu_count,
        }
        return self.deposited, self.error, self.contract_version



@pytest.mark.parametrize(
    "deposited,error,enable_no_collateral,expected_pass,expected_reason",
    [
        (True, None, False, True, Msg.VERIFIED.reason),
        (False, "insufficient bond", False, False, Msg.MISSING.reason),
        (False, None, True, True, Msg.MISSING.reason),
    ],
)
@pytest.mark.asyncio
async def test_collateral_check(
    deposited,
    error,
    enable_no_collateral,
    expected_pass,
    expected_reason,
    context_factory,
):
    service = DummyCollateralService(
        deposited=deposited,
        error=error,
        contract_version="1.0.2",
    )
    services = build_services(collateral=service)
    config = build_context_config(enable_no_collateral=enable_no_collateral)
    specs = {"gpu": {"count": 4, "details": [{"name": "NVIDIA RTX 3090"}]}}
    state = build_state(specs=specs, gpu_count=4, gpu_details=specs["gpu"]["details"])

    ctx = context_factory(services=services, config=config, state=state)

    result = await CollateralCheck().run(ctx)

    assert result.passed is expected_pass
    assert result.event.reason_code == expected_reason
    assert result.updates["collateral_deposited"] is deposited
    assert result.updates["contract_version"] == "1.0.2"

    assert service.called_with == {
        "miner_hotkey": "miner-hotkey",
        "executor_uuid": "executor-123",
        "gpu_model": "NVIDIA RTX 3090",
        "gpu_count": 4,
    }
