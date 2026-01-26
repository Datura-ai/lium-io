import pytest
from unittest.mock import patch

from neurons.validators.src.protocol.vc_protocol.compute_requests import ExecutorHealthCheckResponse
from neurons.validators.src.services.task.checks.rental_verification import RentalVerificationCheck
from neurons.validators.src.services.task.messages import RentalVerificationMessages as Msg

from tests.helpers import build_services, build_state


class DummyBackendClient:
    def __init__(self, *, response: ExecutorHealthCheckResponse | None):
        self.response = response
        self.called_with: dict | None = None

    async def check_executor_health(
        self,
        miner_address: str,
        miner_port: int,
        miner_hotkey: str,
        container_port: int,
        executor_id: str | None = None,
    ):
        self.called_with = {
            "miner_address": miner_address,
            "miner_port": miner_port,
            "miner_hotkey": miner_hotkey,
            "container_port": container_port,
            "executor_id": executor_id,
        }
        return self.response


@pytest.mark.parametrize(
    "skip_verification,expected_pass,expected_reason",
    [
        (True, True, Msg.SKIPPED.reason),
        (False, True, Msg.VERIFIED.reason),
    ],
)
@pytest.mark.asyncio
async def test_rental_verification_skip(
    skip_verification,
    expected_pass,
    expected_reason,
    context_factory,
):
    """Test that rental verification is skipped when SKIP_RENTAL_VERIFICATION is enabled."""
    backend_client = DummyBackendClient(
        response=ExecutorHealthCheckResponse(success=True, error=None, details={})
    )
    services = build_services(backend=backend_client)
    state = build_state(specs={"verified_ports": [8080]})

    ctx = context_factory(services=services, state=state)

    with patch("neurons.validators.src.services.task.checks.rental_verification.settings") as mock_settings:
        mock_settings.SKIP_RENTAL_VERIFICATION = skip_verification
        result = await RentalVerificationCheck().run(ctx)

    assert result.passed is expected_pass
    assert result.event.reason_code == expected_reason

    # Should not call API when skipped
    if skip_verification:
        assert backend_client.called_with is None
    else:
        assert backend_client.called_with is not None


@pytest.mark.asyncio
async def test_rental_verification_no_ports():
    """Test that check fails when no verified ports are available."""
    backend_client = DummyBackendClient(
        response=ExecutorHealthCheckResponse(success=True, error=None, details={})
    )
    services = build_services(backend=backend_client)
    # No verified ports
    state = build_state(specs={})

    from tests.helpers import make_context
    ctx = make_context(services=services, state=state)

    with patch("neurons.validators.src.services.task.checks.rental_verification.settings") as mock_settings:
        mock_settings.SKIP_RENTAL_VERIFICATION = False
        result = await RentalVerificationCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.FAILED.reason
    assert "No verified ports available" in result.event.what_we_saw["error"]
    # Should not call API when no ports
    assert backend_client.called_with is None


@pytest.mark.asyncio
async def test_rental_verification_success():
    """Test successful rental verification."""
    backend_client = DummyBackendClient(
        response=ExecutorHealthCheckResponse(
            success=True,
            error=None,
            details={"container_healthy": True}
        )
    )
    services = build_services(backend=backend_client)
    state = build_state(specs={"verified_ports": [8080, 8081, 8082]})

    from tests.helpers import make_context
    ctx = make_context(services=services, state=state)

    with patch("neurons.validators.src.services.task.checks.rental_verification.settings") as mock_settings:
        mock_settings.SKIP_RENTAL_VERIFICATION = False
        result = await RentalVerificationCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.VERIFIED.reason
    assert result.event.what_we_saw["verified"] is True
    assert result.event.what_we_saw["details"]["container_healthy"] is True

    # Verify API was called with correct params
    assert backend_client.called_with == {
        "miner_address": "127.0.0.1",
        "miner_port": 22,
        "miner_hotkey": "miner-hotkey",
        "container_port": 8080,  # First verified port
        "executor_id": "executor-123",
    }


@pytest.mark.asyncio
async def test_rental_verification_failed():
    """Test rental verification failure."""
    backend_client = DummyBackendClient(
        response=ExecutorHealthCheckResponse(
            success=False,
            error="Container not responding",
            details={"timeout": True}
        )
    )
    services = build_services(backend=backend_client)
    state = build_state(specs={"verified_ports": [8080]})

    from tests.helpers import make_context
    ctx = make_context(services=services, state=state)

    with patch("neurons.validators.src.services.task.checks.rental_verification.settings") as mock_settings:
        mock_settings.SKIP_RENTAL_VERIFICATION = False
        result = await RentalVerificationCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.FAILED.reason
    assert result.event.what_we_saw["verified"] is False
    assert result.event.what_we_saw["error"] == "Container not responding"
    assert result.event.what_we_saw["details"]["timeout"] is True


@pytest.mark.asyncio
async def test_rental_verification_api_error():
    """Test handling of API errors (returns None)."""
    backend_client = DummyBackendClient(response=None)  # API failure
    services = build_services(backend=backend_client)
    state = build_state(specs={"verified_ports": [8080]})

    from tests.helpers import make_context
    ctx = make_context(services=services, state=state)

    with patch("neurons.validators.src.services.task.checks.rental_verification.settings") as mock_settings:
        mock_settings.SKIP_RENTAL_VERIFICATION = False
        result = await RentalVerificationCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.API_ERROR.reason
    assert "API returned None" in result.event.what_we_saw["error"]


@pytest.mark.asyncio
async def test_rental_verification_exception():
    """Test handling of exceptions during API call."""
    class ExceptionBackendClient:
        async def check_executor_health(self, **kwargs):
            raise Exception("Network timeout")

    backend_client = ExceptionBackendClient()
    services = build_services(backend=backend_client)
    state = build_state(specs={"verified_ports": [8080]})

    from tests.helpers import make_context
    ctx = make_context(services=services, state=state)

    with patch("neurons.validators.src.services.task.checks.rental_verification.settings") as mock_settings:
        mock_settings.SKIP_RENTAL_VERIFICATION = False
        result = await RentalVerificationCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.API_ERROR.reason
    assert "Network timeout" in result.event.what_we_saw["error"]


@pytest.mark.asyncio
async def test_rental_verification_uses_first_verified_port():
    """Test that the check uses the first verified port from the list."""
    backend_client = DummyBackendClient(
        response=ExecutorHealthCheckResponse(success=True, error=None, details={})
    )
    services = build_services(backend=backend_client)
    # Multiple verified ports
    state = build_state(specs={"verified_ports": [9001, 9002, 9003]})

    from tests.helpers import make_context
    ctx = make_context(services=services, state=state)

    with patch("neurons.validators.src.services.task.checks.rental_verification.settings") as mock_settings:
        mock_settings.SKIP_RENTAL_VERIFICATION = False
        result = await RentalVerificationCheck().run(ctx)

    assert result.passed is True
    # Should use first port (9001)
    assert backend_client.called_with["container_port"] == 9001
