"""End-to-end pipeline tests demonstrating full validation flows.

These tests show how the entire pipeline executes from start to finish,
serving as examples for testing custom scenarios and understanding the
complete validation flow.
"""

import pytest
from unittest.mock import Mock

from neurons.validators.src.services.task.pipeline import Pipeline, LoggerSink
from neurons.validators.src.services.task.checks import (
    BannedGpuCheck,
    CapabilityCheck,
    CollateralCheck,
    DuplicateExecutorCheck,
    FinalizeCheck,
    GpuCountCheck,
    GpuFingerprintCheck,
    GpuModelValidCheck,
    GpuUsageCheck,
    MachineSpecScrapeCheck,
    NvmlDigestCheck,
    PortConnectivityCheck,
    PortCountCheck,
    TenantEnforcementCheck,
    ScoreCheck,
    StartGPUMonitorCheck,
    SpecChangeCheck,
    UploadFilesCheck,
    VerifyXCheck,
)

from tests.helpers import build_context_config, build_services, build_state


class DummyLogger:
    """Mock logger for capturing pipeline events."""

    def __init__(self):
        self.events = []

    def info(self, msg):
        self.events.append(("info", msg))

    def warning(self, msg):
        self.events.append(("warning", msg))

    def error(self, msg):
        self.events.append(("error", msg))


class DummySSHCommandRunner:
    """Mock SSH command runner for successful operations."""

    def __init__(self, machine_specs_encrypted: str = "encrypted_payload"):
        self.machine_specs_encrypted = machine_specs_encrypted
        self.commands_called = []

    async def run(self, command: str, timeout: int = 300, retryable: bool = False):
        self.commands_called.append(command)

        from neurons.validators.src.services.task.runner import SSHCommandResult
        from datetime import datetime, UTC

        # Mock successful responses for different commands
        if "ps aux | grep" in command:
            # GPU monitor not running initially
            stdout = ""
        elif "pip install" in command:
            stdout = "Successfully installed packages"
        elif "nohup" in command:
            stdout = ""
        elif "chmod +x" in command and "scrape.sh" in command:
            # Return encrypted machine specs
            stdout = self.machine_specs_encrypted
        else:
            stdout = ""

        return SSHCommandResult(
            command=command,
            command_id="cmd-123",
            exit_code=0,
            stdout=stdout,
            stderr="",
            duration_ms=100,
            started_at=datetime.now(UTC),
            finished_at=datetime.now(UTC),
            success=True,
        )


class DummySFTPClient:
    """Mock SFTP client for file uploads."""

    async def put(self, local_path: str, remote_path: str, recurse: bool = False):
        pass


class DummySSHClient:
    """Mock SSH client with SFTP support."""

    def __init__(self):
        self.sftp_client = DummySFTPClient()

    def start_sftp_client(self):
        return self

    async def __aenter__(self):
        return self.sftp_client

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def run(self, command: str):
        """Mock SSH run for various commands."""
        result = Mock()

        # Docker container checks for rented machines
        if "docker ps" in command:
            result.stdout = ""  # Not rented
        else:
            result.stdout = ""

        return result


class DummySSHService:
    """Mock SSH service for decryption."""

    def decrypt_payload(self, encrypt_key: str, payload: str) -> str:
        import json
        # Return mock machine specs
        return json.dumps({
            "gpu": {
                "count": 2,
                "driver": "535.104.05",
                "details": [
                    {"name": "NVIDIA RTX 4090", "uuid": "GPU-abc123", "gpu_utilization": 0, "memory_utilization": 0, "capacity": 24},
                    {"name": "NVIDIA RTX 4090", "uuid": "GPU-def456", "gpu_utilization": 0, "memory_utilization": 0, "capacity": 24},
                ],
            },
            "gpu_processes": [],
            "sysbox_runtime": True,
            "md5_checksums": {
                "libnvidia_ml": "expected_digest_for_535.104.05",
            },
        })


class DummyRedisService:
    """Mock Redis service."""

    async def get_rented_machine(self, executor):
        return None  # Not rented

    async def lrange(self, key: str):
        return []

    async def get_banned_guids(self):
        return set()  # No banned GPUs

    async def is_elem_exists_in_set(self, set_name: str, value: str):
        return False  # Not a duplicate

    async def renting_in_progress(self, miner_hotkey: str, executor_uuid: str):
        return False  # Not renting in progress


class DummyCollateralService:
    """Mock collateral contract service."""

    async def is_eligible_executor(self, miner_hotkey: str, executor_uuid: str, gpu_model: str, gpu_count: int):
        return True, None, "v1.0.0"  # collateral_deposited, error_message, contract_version


class DummyValidationService:
    """Mock validation service for GPU capability checks."""

    async def validate_gpu_model_and_process_job(self, *, ssh_client, executor_info, default_extra, machine_spec):
        return True


class DummyVerifyXService:
    """Mock VerifyX validation service."""

    async def validate_verifyx_and_process_job(self, *, shell, executor_info, default_extra, machine_spec):
        result = Mock()
        result.data = {
            "success": True,
            "ram": 128000,
            "hard_disk": 1000000,
            "network": 10000,
        }
        result.error = None
        return result


class DummyConnectivityService:
    """Mock executor connectivity service."""

    async def verify_ports(self, *args, **kwargs):
        result = Mock()
        result.success = True
        result.sysbox_runtime = kwargs.get("sysbox_runtime", True)
        result.log_text = "Port connectivity verified"
        return result


class DummyPortMappingService:
    """Mock port mapping service."""

    async def get_successful_ports_count(self, executor_uuid: str) -> int:
        return 10  # Sufficient ports


def dummy_score_calculator(gpu_model: str, collateral: bool, rental_succeed: bool, contract_version: str, rented: bool, port_count: int):
    """Mock score calculator."""
    return 1.0, 1.0, ""  # actual_score, job_score, warning


@pytest.mark.asyncio
async def test_successful_unrented_pipeline_flow(context_factory):
    """Test a complete successful pipeline run for an unrented machine.

    This demonstrates the full validation flow from start to finish:
    1. Preparation phase (GPU monitor, upload, scrape)
    2. Validation phase (GPU checks, fingerprints, policies)
    3. Rental check (not rented, continues)
    4. Capability validation (usage, ports, verifyx, capability)
    5. Finalization (score, finalize)

    Use this as a template for testing your own scenarios.
    """
    # Setup mock keypair
    mock_keypair = Mock()
    mock_keypair.sign.return_value = b"\x00" * 64
    mock_keypair.ss58_address = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"

    # Setup mock executor
    mock_executor = Mock()
    mock_executor.uuid = "executor-123"
    mock_executor.root_dir = "/root/app"
    mock_executor.python_path = "/usr/bin/python3"

    # Create encrypted payload
    encrypted_payload = "encrypted_machine_specs_here"

    # Setup mocks
    runner = DummySSHCommandRunner(machine_specs_encrypted=encrypted_payload)
    ssh_client = DummySSHClient()
    ssh_service = DummySSHService()
    redis_service = DummyRedisService()
    collateral_service = DummyCollateralService()
    validation_service = DummyValidationService()
    verifyx_service = DummyVerifyXService()
    connectivity_service = DummyConnectivityService()
    port_mapping_service = DummyPortMappingService()

    # Setup services
    services = build_services(
        ssh=ssh_service,
        redis=redis_service,
        collateral=collateral_service,
        validation=validation_service,
        verifyx=verifyx_service,
        connectivity=connectivity_service,
        port_mapping=port_mapping_service,
        score_calculator=dummy_score_calculator,
    )

    # Setup config with all required fields
    config = build_context_config(
        validator_keypair=mock_keypair,
        compute_rest_app_url="http://validator:8000",
        gpu_monitor_script_relative="src/gpus_utility.py",
        machine_scrape_filename="scrape.sh",
        machine_scrape_timeout=300,
        obfuscation_keys={},
        max_gpu_count=8,
        gpu_model_rates={"NVIDIA RTX 4090": 1.0},
        nvml_digest_map={"535.104.05": "expected_digest_for_535.104.05"},
        enable_no_collateral=False,
        verifyx_enabled=True,
        port_private_key="private_key",
        port_public_key="public_key",
        job_batch_id="batch-123",
    )

    # Setup state
    state = build_state(
        upload_local_dir="/tmp/validator/files",
    )

    # Create context
    ctx = context_factory(
        executor=mock_executor,
        miner_hotkey="miner-hotkey-123",
        ssh=ssh_client,
        runner=runner,
        services=services,
        config=config,
        state=state,
        encrypt_key="test-encrypt-key",
        is_rental_succeed=True,
        verified={"spec": "", "uuids": ""},  # No previous verification
    )

    # Create logger
    logger = DummyLogger()

    # Define the full pipeline (same as in create_task)
    checks = [
        StartGPUMonitorCheck(),
        UploadFilesCheck(),
        MachineSpecScrapeCheck(),
        GpuCountCheck(),
        GpuModelValidCheck(),
        NvmlDigestCheck(),
        SpecChangeCheck(),
        GpuFingerprintCheck(),
        BannedGpuCheck(),
        DuplicateExecutorCheck(),
        CollateralCheck(),
        TenantEnforcementCheck(),
        GpuUsageCheck(),
        PortConnectivityCheck(),
        VerifyXCheck(),
        CapabilityCheck(),
        PortCountCheck(),
        ScoreCheck(),
        FinalizeCheck(),
    ]

    # Run the pipeline
    pipeline = Pipeline(checks, sink=LoggerSink(logger))
    ok, events, final_ctx = await pipeline.run(ctx)

    # Verify pipeline succeeded
    assert ok is True, "Pipeline should complete successfully"
    assert len(events) == 19, "Should have events from all 19 checks"

    # Verify final context state
    assert final_ctx.success is True
    assert final_ctx.score == 1.0
    assert final_ctx.job_score == 1.0
    assert final_ctx.rented is False
    assert final_ctx.state.gpu_count == 2
    assert final_ctx.state.gpu_model == "NVIDIA RTX 4090"
    assert final_ctx.state.gpu_model_count == "NVIDIA RTX 4090:2"
    assert final_ctx.state.gpu_uuids == "GPU-abc123,GPU-def456"
    assert final_ctx.state.sysbox_runtime is True
    assert final_ctx.collateral_deposited is True

    # Verify all checks emitted events
    assert all(event.reason_code is not None for event in events)

    # Verify key checkpoints
    reason_codes = [event.reason_code for event in events]
    assert "MONITOR_STARTED" in reason_codes or "MONITOR_RUNNING" in reason_codes
    assert "UPLOAD_OK" in reason_codes
    assert "SCRAPE_OK" in reason_codes
    assert "GPU_COUNT_OK" in reason_codes
    assert "GPU_MODEL_OK" in reason_codes
    assert "EXECUTOR_NOT_RENTED" in reason_codes
    assert "SCORE_COMPUTED" in reason_codes
    assert "VALIDATION_COMPLETED" in reason_codes

    print(f"\n✅ Pipeline completed successfully!")
    print(f"   - Executed {len(events)} checks")
    print(f"   - Final score: {final_ctx.score}")
    print(f"   - GPU: {final_ctx.state.gpu_model_count}")
    print(f"   - Commands executed: {len(runner.commands_called)}")


@pytest.mark.asyncio
async def test_successful_rented_pipeline_flow(context_factory):
    """Test a complete successful pipeline run for a rented machine.

    Demonstrates the rented path which halts after TenantEnforcementCheck:
    1. Preparation phase (GPU monitor, upload, scrape)
    2. Validation phase (GPU checks, fingerprints, policies)
    3. Rental check (rented, verifies container, computes score, HALTS)

    Note: Remaining checks (GpuUsage, PortConnectivity, etc.) don't run.
    """
    # Setup mock keypair
    mock_keypair = Mock()
    mock_keypair.sign.return_value = b"\x00" * 64
    mock_keypair.ss58_address = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"

    # Setup mock executor
    mock_executor = Mock()
    mock_executor.uuid = "executor-123"
    mock_executor.root_dir = "/root/app"
    mock_executor.python_path = "/usr/bin/python3"

    # Setup SSH client that returns rented container info
    class RentedSSHClient:
        def __init__(self):
            self.sftp_client = DummySFTPClient()

        def start_sftp_client(self):
            return self

        async def __aenter__(self):
            return self.sftp_client

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def run(self, command: str):
            """Mock SSH run for various commands."""
            result = Mock()
            if "docker ps" in command:
                result.stdout = "container-id-123"  # Container is running
            elif "authorized_keys" in command:
                result.stdout = "ssh-rsa AAA..."  # SSH keys present
            else:
                result.stdout = ""
            return result

    ssh_client = RentedSSHClient()

    # Setup SSH service
    ssh_service = DummySSHService()

    # Setup Redis service that returns rental info
    class RentedRedisService:
        async def get_rented_machine(self, executor):
            return {
                "container_name": "tenant-container-123",
                "owner_flag": False,
            }

        async def lrange(self, key: str):
            return []

        async def get_banned_guids(self):
            return set()  # No banned GPUs

        async def is_elem_exists_in_set(self, set_name: str, value: str):
            return False  # Not a duplicate

        async def renting_in_progress(self, miner_hotkey: str, executor_uuid: str):
            return False  # Not renting in progress

    # Create runner and other services
    runner = DummySSHCommandRunner()
    redis_service = RentedRedisService()
    collateral_service = DummyCollateralService()
    port_mapping_service = DummyPortMappingService()

    # Setup services
    services = build_services(
        ssh=ssh_service,
        redis=redis_service,
        collateral=collateral_service,
        port_mapping=port_mapping_service,
        score_calculator=dummy_score_calculator,
    )

    # Setup config
    config = build_context_config(
        validator_keypair=mock_keypair,
        compute_rest_app_url="http://validator:8000",
        gpu_monitor_script_relative="src/gpus_utility.py",
        machine_scrape_filename="scrape.sh",
        machine_scrape_timeout=300,
        obfuscation_keys={},
        max_gpu_count=8,
        gpu_model_rates={"NVIDIA RTX 4090": 1.0},
        nvml_digest_map={"535.104.05": "expected_digest_for_535.104.05"},
    )

    # Setup state with GPU processes in the correct container
    state = build_state(
        upload_local_dir="/tmp/validator/files",
        gpu_processes=[
            {"container_name": "tenant-container-123", "pid": 1234}
        ],
        gpu_details=[
            {"gpu_utilization": 50, "memory_utilization": 60}
        ],
    )

    # Create context
    ctx = context_factory(
        executor=mock_executor,
        miner_hotkey="miner-hotkey-123",
        ssh=ssh_client,
        runner=runner,
        services=services,
        config=config,
        state=state,
        encrypt_key="test-encrypt-key",
        is_rental_succeed=True,
        verified={"spec": "", "uuids": ""},
    )

    # Create logger
    logger = DummyLogger()

    # Define the full pipeline
    checks = [
        StartGPUMonitorCheck(),
        UploadFilesCheck(),
        MachineSpecScrapeCheck(),
        GpuCountCheck(),
        GpuModelValidCheck(),
        NvmlDigestCheck(),
        SpecChangeCheck(),
        GpuFingerprintCheck(),
        BannedGpuCheck(),
        DuplicateExecutorCheck(),
        CollateralCheck(),
        TenantEnforcementCheck(),
        # These checks below should NOT run because pipeline halts
        GpuUsageCheck(),
        PortConnectivityCheck(),
        VerifyXCheck(),
        CapabilityCheck(),
        PortCountCheck(),
        ScoreCheck(),
        FinalizeCheck(),
    ]

    # Run the pipeline
    pipeline = Pipeline(checks, sink=LoggerSink(logger))
    ok, events, final_ctx = await pipeline.run(ctx)

    # Debug: print which check failed if not ok
    if not ok:
        print(f"\n❌ Pipeline failed at check: {events[-1].check_id}")
        print(f"   Reason: {events[-1].reason_code}")
        print(f"   Event: {events[-1].event}")
        print(f"   Total checks run: {len(events)}")

    # Verify pipeline succeeded but halted early
    assert ok is True, f"Pipeline should complete successfully (halted), but failed at {events[-1].check_id if events else 'unknown'}"
    assert len(events) == 12, "Should only have events up to TenantEnforcementCheck (12 checks)"

    # Verify final context state
    assert final_ctx.success is True
    assert final_ctx.rented is True
    assert final_ctx.score == 1.0
    assert final_ctx.job_score == 1.0

    # Verify the last event is from TenantEnforcementCheck
    last_event = events[-1]
    assert last_event.reason_code == "RENTED"
    assert last_event.check_id == "executor.validate.rented_state"

    # Verify checks after TenantEnforcementCheck did NOT run
    reason_codes = [event.reason_code for event in events]
    assert "GPU_USAGE_OK" not in reason_codes, "GpuUsageCheck should not run for rented machines"
    assert "PORT_CONNECTIVITY_OK" not in reason_codes, "PortConnectivityCheck should not run for rented machines"

    print(f"\n✅ Rented pipeline completed successfully!")
    print(f"   - Executed {len(events)} checks (halted early)")
    print(f"   - Final score: {final_ctx.score}")
    print(f"   - Rented: {final_ctx.rented}")
