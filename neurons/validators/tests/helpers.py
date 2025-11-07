from __future__ import annotations

from datura.requests.miner_requests import ExecutorSSHInfo

from neurons.validators.src.services.task.pipeline import (
    Context,
    ContextConfig,
    ContextServices,
    ContextState,
)


class DummyPortMapping:
    async def get_successful_ports_count(self, executor_uuid: str) -> int:  # pragma: no cover
        return 0


class DummyScoreCalc:
    def __call__(self, *args, **kwargs):  # pragma: no cover
        return 0.0, 0.0, ""


def default_executor() -> ExecutorSSHInfo:
    return ExecutorSSHInfo(
        uuid="executor-123",
        address="127.0.0.1",
        port=22,
        ssh_username="root",
        ssh_port=22,
        python_path="/usr/bin/python",
        root_dir="/root/app",
    )


def build_context_config(**overrides) -> ContextConfig:
    base = dict(
        executor_root="/root/app",
        compute_rest_app_url="http://validator",
        gpu_monitor_script_relative="src/gpus_utility.py",
        machine_scrape_filename="scrape.sh",
        machine_scrape_timeout=300,
        obfuscation_keys={},
        validator_keypair=None,
        max_gpu_count=None,
        gpu_model_rates={},
        nvml_digest_map={},
        enable_no_collateral=False,
        verifyx_enabled=False,
        port_private_key=None,
        port_public_key=None,
        job_batch_id="batch-1",
    )
    base.update(overrides)
    return ContextConfig(**base)


def build_services(**overrides) -> ContextServices:
    base = dict(
        ssh=None,
        redis=None,
        collateral=None,
        validation=None,
        verifyx=None,
        connectivity=None,
        shell=None,
        port_mapping=DummyPortMapping(),
        score_calculator=DummyScoreCalc(),
    )
    base.update(overrides)
    return ContextServices(**base)


def build_state(**overrides) -> ContextState:
    base = dict(specs={})
    base.update(overrides)
    return ContextState(**base)


def make_context(
    *,
    executor: ExecutorSSHInfo | None = None,
    services: ContextServices | None = None,
    config: ContextConfig | None = None,
    state: ContextState | None = None,
    miner_hotkey: str = "miner-hotkey",
    pipeline_id: str = "test-pipeline-id",
    **extra,
) -> Context:
    executor_obj = executor or default_executor()
    services_obj = services or build_services()
    config_obj = config or build_context_config()
    state_obj = state or build_state()

    base_kwargs = dict(
        pipeline_id=pipeline_id,
        executor=executor_obj,
        miner_hotkey=miner_hotkey,
        ssh=None,
        runner=None,
        verified={},
        settings={},
        encrypt_key=None,
        default_extra={},
        services=services_obj,
        config=config_obj,
        state=state_obj,
        is_rental_succeed=False,
    )
    base_kwargs.update(extra)

    return Context.model_construct(**base_kwargs)
