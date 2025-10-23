import asyncio

import pytest

from datura.requests.miner_requests import ExecutorSSHInfo

from neurons.validators.src.services.task.checks.gpu_count import GpuCountCheck
from neurons.validators.src.services.task.pipeline import (
    Context,
    ContextConfig,
    ContextServices,
    ContextState,
)


class DummyRedis:
    ...


def make_context(*, gpu_count: int | None, max_gpu_count: int | None) -> Context:
    executor = ExecutorSSHInfo(
        uuid="executor-123",
        address="127.0.0.1",
        port=22,
        ssh_username="root",
        ssh_port=22,
        python_path="/usr/bin/python",
        root_dir="/root/app",
    )

    services = ContextServices(
        ssh=None,
        redis=DummyRedis(),
        collateral=None,
        validation=None,
        verifyx=None,
        connectivity=None,
        shell=None,
        port_mapping=None,
        score_calculator=lambda *_, **__: (0.0, 0.0, ""),
    )

    config = ContextConfig(
        executor_root="/root/app",
        compute_rest_app_url="http://validator",
        gpu_monitor_script_relative="src/gpus_utility.py",
        machine_scrape_filename="scrape.sh",
        machine_scrape_timeout=300,
        obfuscation_keys={},
        validator_keypair=None,
        max_gpu_count=max_gpu_count,
        gpu_model_rates={},
        nvml_digest_map={},
        enable_no_collateral=False,
        verifyx_enabled=False,
        port_private_key=None,
        port_public_key=None,
        job_batch_id="batch-1",
    )

    specs = {"gpu": {"count": gpu_count or 0}}
    state = ContextState(specs=specs, gpu_count=gpu_count)

    return Context(
        executor=executor,
        miner_hotkey="miner-hotkey",
        ssh=None,
        runner=None,
        verified={},
        settings={},
        encrypt_key=None,
        default_extra={},
        services=services,
        config=config,
        state=state,
        is_rental_succeed=False,
    )


@pytest.mark.parametrize(
    "gpu_count,max_gpu_count,expected_pass,expected_reason",
    [
        (2, 3, True, "GPU_COUNT_OK"),
        (5, None, False, "GPU_COUNT_POLICY_MISSING"),
        (5, 4, False, "GPU_COUNT_EXCEEDS_MAX"),
    ],
)
@pytest.mark.asyncio
async def test_gpu_count_check(gpu_count, max_gpu_count, expected_pass, expected_reason):
    ctx = make_context(gpu_count=gpu_count, max_gpu_count=max_gpu_count)
    result = await GpuCountCheck().run(ctx)

    assert result.passed is expected_pass
    assert result.event.reason_code == expected_reason

    if expected_pass:
        assert result.event.what["count"] == (gpu_count or 0)
