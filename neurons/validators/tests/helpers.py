from __future__ import annotations

from datura.requests.miner_requests import ExecutorSSHInfo

from neurons.validators.src.services.task.pipeline import (
    Context,
    ContextConfig,
    ContextServices,
    ContextState,
)




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
        price_per_gpu=0.5,
    )


def build_context_config(**overrides) -> ContextConfig:
    base = dict(
        executor_root="/root/app",
        compute_rest_app_url="http://validator",
        gpu_monitor_script_relative="src/gpus_utility.py",
        machine_scrape_filename="scrape.sh",
        machine_scrape_timeout=300,
        obfuscation_keys={},
        default_docker_image_digests={},
        validator_keypair=None,
        max_gpu_count=None,
        gpu_model_rates={},
        nvml_digest_map={},
        enable_no_collateral=False,
        verifyx_enabled=False,
        inspector_enabled=False,
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
        inspector=None,
        connectivity=None,
        shell=None,
        score_calculator=DummyScoreCalc(),
        backend=None,
        container_cleanup=None,
    )
    base.update(overrides)
    return ContextServices(**base)


def build_state(**overrides) -> ContextState:
    base = dict(specs={}, verified_port_count=0)
    base.update(overrides)
    return ContextState(**base)


def make_context(
    *,
    executor: ExecutorSSHInfo | None = None,
    services: ContextServices | None = None,
    config: ContextConfig | None = None,
    state: ContextState | None = None,
    miner_hotkey: str = "miner-hotkey",
    miner_address: str = "127.0.0.1",
    miner_port: int = 8000,
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
        miner_address=miner_address,
        miner_port=miner_port,
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


# Incentive logging assertion helpers

def assert_incentive_log_present(log_text: str) -> None:
    """Verify 'Incentive Scores Calculation Logs:' header is present."""
    assert "Incentive Scores Calculation Logs:" in log_text


def assert_executor_has_log(log_text: str, executor_id: str) -> None:
    """Verify executor_id appears in the logs."""
    assert executor_id in log_text


def assert_log_contains_keys(log_text: str, expected_keys: list[str]) -> None:
    """Verify log_text contains all expected key strings."""
    for key in expected_keys:
        assert key in log_text, f"Expected key '{key}' not found in log_text"


def extract_incentive_section(log_text: str) -> str:
    """Extract the incentive logs section from log_text."""
    if "Incentive Scores Calculation Logs:" not in log_text:
        return ""
    return log_text.split("Incentive Scores Calculation Logs:")[1]


def count_incentive_log_entries(log_text: str) -> int:
    """Count incentive log entries by counting 'executor_id:' occurrences."""
    section = extract_incentive_section(log_text)
    return section.count("executor_id:")


# Rental price incentive log (rental_price.py lines 146-165): message + extra keys
RENTAL_PRICE_INCENTIVE_LOG_MESSAGE = (
    "Rental price incentive for executor is calculated successfully. "
    "Formula: rental_share * gpu_count * effective_rate / total_rental_cost"
)
RENTAL_PRICE_INCENTIVE_EXTRA_KEYS = [
    "hotkey",
    "executor_id",
    "gpu_model",
    "gpu_count",
    "hourly_rate",
    "unrented_cap_multiplier",
    "effective_rate",
    "total_unrented_by_gpu_type",
    "max_cap",
    "cap_dilution_applied",
    "rental_share",
    "burn_share",
    "incentive",
    "total_rental_cost",
]


def assert_rental_price_incentive_log_full_content(full_log_text: str) -> None:
    """Assert full_log_text contains the rental price incentive log message and all extra keys.

    Covers the logic in incentive/rental_price.py that appends the rental price
    incentive log (message + get_extra_info(...)). Verifies the message and
    every extra field is present in the log.
    """
    assert_incentive_log_present(full_log_text)
    assert (
        RENTAL_PRICE_INCENTIVE_LOG_MESSAGE in full_log_text
    ), f"Expected rental price incentive message not found in full_log_text"
    assert_log_contains_keys(full_log_text, RENTAL_PRICE_INCENTIVE_EXTRA_KEYS)
