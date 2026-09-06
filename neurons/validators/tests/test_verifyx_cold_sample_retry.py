"""DAH-2959: a never-measured executor's first VerifyX download sample decides its first cycle alone
(the EMA bootstraps from it). With VERIFYX_COLD_SAMPLE_RETRY_ENABLED the check takes one more sample
inside the same task and gates on the better one; flag off is today's single sample."""

import pytest

from core.config import settings
from neurons.validators.src.services.task.checks.verifyx import VerifyXCheck
from neurons.validators.src.services.task.messages import VerifyXMessages as Msg
from protocol.vc_protocol.compute_requests import NetworkEMA, RentedExecutorsResponse

from tests.helpers import build_context_config, build_services, build_state
from tests.test_verifyx_check import MockVerifyXResponse

EXECUTOR = "executor-123"  # tests.helpers.default_executor().uuid


def _probe(download: float | None, *, success: bool = True, upload: float | None = 40.0) -> MockVerifyXResponse:
    network = {} if download is None else {"download_speed": download, "upload_speed": upload}
    return MockVerifyXResponse(
        data={"success": success, "ram": {"total": 64}, "hard_disk": {"total": 1000}, "network": network}
    )


class _ProbeSequence:
    """Returns the queued responses in order and counts the calls."""

    def __init__(self, *responses: MockVerifyXResponse):
        self._responses = list(responses)
        self.calls = 0

    async def validate_verifyx_and_process_job(self, *, shell, executor_info, default_extra, machine_spec):
        self.calls += 1
        return self._responses.pop(0)


def _never_measured() -> RentedExecutorsResponse:
    return RentedExecutorsResponse(executors={}, banned_guids=[], network_ema={})


def _known(download: float) -> RentedExecutorsResponse:
    return RentedExecutorsResponse(
        executors={},
        banned_guids=[],
        network_ema={EXECUTOR: NetworkEMA(ema_verifyx_download_speed=download, ema_verifyx_upload_speed=50.0)},
    )


def _ctx(context_factory, service, rented_data):
    return context_factory(
        services=build_services(verifyx=service),
        config=build_context_config(verifyx_enabled=True),
        state=build_state(specs={"gpu": {"count": 1}}, rented_data=rented_data),
    )


@pytest.fixture
def retry_on(monkeypatch):
    monkeypatch.setattr(settings, "VERIFYX_COLD_SAMPLE_RETRY_ENABLED", True)


@pytest.mark.asyncio
async def test_flag_off_is_todays_single_cold_sample(monkeypatch, context_factory):
    monkeypatch.setattr(settings, "VERIFYX_COLD_SAMPLE_RETRY_ENABLED", False)
    service = _ProbeSequence(_probe(59.9), _probe(760.0))

    result = await VerifyXCheck().run(_ctx(context_factory, service, _never_measured()))

    assert service.calls == 1
    assert result.passed is False
    assert result.event.reason_code == Msg.VERIFY_FAILED_NETWORK_SPEED_TOO_SLOW.reason
    assert result.updates["state"].specs["network"]["ema_verifyx_download_speed"] == pytest.approx(59.9)
    assert "cold_sample_retry" not in result.event.what_we_saw


@pytest.mark.asyncio
async def test_cold_first_sample_is_re_measured_and_the_better_one_seeds_the_ema(retry_on, context_factory):
    # b3e4ca69, 4 Sep: 59.9 Mbps on the first sample, ~760 Mbps once the link was free.
    service = _ProbeSequence(_probe(59.9, upload=20.0), _probe(760.0, upload=110.0))

    result = await VerifyXCheck().run(_ctx(context_factory, service, _never_measured()))

    assert service.calls == 2
    assert result.passed is True
    assert result.event.reason_code == Msg.VERIFY_SUCCESS.reason
    network = result.updates["state"].specs["network"]
    assert network["verifyx_download_speed"] == pytest.approx(760.0)
    assert network["ema_verifyx_download_speed"] == pytest.approx(760.0)
    assert network["verifyx_upload_speed"] == pytest.approx(110.0)
    assert result.event.what_we_saw["cold_sample_retry"] == {
        "first_download_speed_mbps": 59.9,
        "retry_download_speed_mbps": 760.0,
        "used": "retry",
    }


@pytest.mark.asyncio
async def test_two_slow_samples_still_fail_the_unchanged_gate(retry_on, context_factory):
    service = _ProbeSequence(_probe(40.0), _probe(88.0))

    result = await VerifyXCheck().run(_ctx(context_factory, service, _never_measured()))

    assert service.calls == 2
    assert result.passed is False
    assert result.event.reason_code == Msg.VERIFY_FAILED_NETWORK_SPEED_TOO_SLOW.reason
    # The better of the two is what the node is judged on and what seeds the EMA.
    assert result.updates["state"].specs["network"]["ema_verifyx_download_speed"] == pytest.approx(88.0)
    assert result.event.what_we_saw["cold_sample_retry"]["used"] == "retry"


@pytest.mark.asyncio
async def test_failed_network_probe_on_first_sample_is_retried(retry_on, context_factory):
    # download_speed None (probe timed out) would bootstrap the EMA to 0.0.
    service = _ProbeSequence(_probe(None), _probe(300.0))

    result = await VerifyXCheck().run(_ctx(context_factory, service, _never_measured()))

    assert service.calls == 2
    assert result.passed is True
    assert result.updates["state"].specs["network"]["ema_verifyx_download_speed"] == pytest.approx(300.0)
    assert result.event.what_we_saw["cold_sample_retry"]["first_download_speed_mbps"] is None


@pytest.mark.asyncio
async def test_a_worse_retry_keeps_the_first_sample(retry_on, context_factory):
    service = _ProbeSequence(_probe(70.0), _probe(30.0))

    result = await VerifyXCheck().run(_ctx(context_factory, service, _never_measured()))

    assert service.calls == 2
    assert result.passed is False
    assert result.updates["state"].specs["network"]["ema_verifyx_download_speed"] == pytest.approx(70.0)
    assert result.event.what_we_saw["cold_sample_retry"]["used"] == "first"


@pytest.mark.asyncio
async def test_a_retry_whose_probe_failed_outright_is_not_used(retry_on, context_factory):
    service = _ProbeSequence(_probe(70.0), _probe(900.0, success=False))

    result = await VerifyXCheck().run(_ctx(context_factory, service, _never_measured()))

    assert service.calls == 2
    assert result.passed is False
    assert result.updates["state"].specs["network"]["ema_verifyx_download_speed"] == pytest.approx(70.0)


@pytest.mark.asyncio
async def test_known_host_below_the_gate_is_not_retried(retry_on, context_factory):
    # prev EMA 110 + sample 20 -> EMA 65: a real slow host, judged as today.
    service = _ProbeSequence(_probe(20.0), _probe(900.0))

    result = await VerifyXCheck().run(_ctx(context_factory, service, _known(110.0)))

    assert service.calls == 1
    assert result.passed is False
    assert result.updates["state"].specs["network"]["ema_verifyx_download_speed"] == pytest.approx(65.0)


@pytest.mark.asyncio
async def test_fast_first_sample_is_not_retried(retry_on, context_factory):
    service = _ProbeSequence(_probe(500.0), _probe(1.0))

    result = await VerifyXCheck().run(_ctx(context_factory, service, _never_measured()))

    assert service.calls == 1
    assert result.passed is True
    assert "cold_sample_retry" not in result.event.what_we_saw


@pytest.mark.asyncio
async def test_without_backend_data_nobody_looks_new_and_nothing_is_retried(retry_on, context_factory):
    # rented_data None: every executor would look never-measured; a backend blip must not double VerifyX fleet-wide.
    service = _ProbeSequence(_probe(59.9), _probe(760.0))

    result = await VerifyXCheck().run(_ctx(context_factory, service, None))

    assert service.calls == 1
    assert result.passed is False


@pytest.mark.asyncio
async def test_ram_or_disk_failure_on_the_first_sample_is_not_retried(retry_on, context_factory):
    # The retry is for the network sample only; a probe that failed on RAM/disk is judged as today.
    service = _ProbeSequence(_probe(59.9, success=False), _probe(760.0))

    result = await VerifyXCheck().run(_ctx(context_factory, service, _never_measured()))

    assert service.calls == 1
    assert result.passed is False
    assert result.event.reason_code == Msg.VERIFY_FAILED.reason
