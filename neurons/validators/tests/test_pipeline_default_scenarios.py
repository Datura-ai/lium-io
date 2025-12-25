"""End-to-end pipeline tests demonstrating full validation flows.

These tests show how the entire pipeline executes from start to finish,
serving as examples for testing custom scenarios and understanding the
complete validation flow.
"""

import pytest
from unittest.mock import Mock

from datura.requests.miner_requests import ExecutorSSHInfo
from neurons.validators.src.services.matrix_validation_service import ValidationResult
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
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse, RentedExecutor, RentedPod

from datura.requests.miner_requests import ExecutorSSHInfo
from helpers import build_context_config, build_services, build_state
from protocol.vc_protocol.compute_requests import (
    RentedExecutor,
    RentedExecutorsResponse,
    RentedPod,
)


def make_executor(uuid: str = "executor-123") -> ExecutorSSHInfo:
    """Create a real ExecutorSSHInfo for tests."""
    return ExecutorSSHInfo(
        uuid=uuid,
        address="192.168.1.100",
        port=8080,
        ssh_username="root",
        ssh_port=22,
        python_path="/usr/bin/python3",
        root_dir="/root/app",
    )


class DummyKeypair:
    """Keypair that can be serialized (unlike Mock)."""

    def __init__(self):
        self.ss58_address = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"

    def sign(self, data: bytes) -> bytes:
        return b"\x00" * 64


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


class SimpleSSHResult:
    """Simple SSH result that is JSON serializable."""

    def __init__(self, stdout: str = ""):
        self.stdout = stdout


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
        # Docker container checks for rented machines
        if "docker ps" in command:
            return SimpleSSHResult(stdout="")  # Not rented
        return SimpleSSHResult(stdout="")


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
        return ValidationResult(
            success=True,
            expected_uuid="test-uuid-123",
            returned_uuid="test-uuid-123",
            stdout="UUID: test-uuid-123",
            stderr=""
        )


class VerifyXResult:
    """Result from VerifyX validation (serializable)."""

    def __init__(self):
        self.data = {
            "success": True,
            "ram": 128000,
            "hard_disk": 1000000,
            "network": {"download_speed": 10000},
        }
        self.error = None


class DummyVerifyXService:
    """Mock VerifyX validation service."""

    async def validate_verifyx_and_process_job(self, *, shell, executor_info, default_extra, machine_spec):
        return VerifyXResult()


class ConnectivityResult:
    """Result from port connectivity check (serializable)."""

    def __init__(self, sysbox_runtime: bool = True):
        self.success = True
        self.sysbox_runtime = sysbox_runtime
        self.log_text = "Port connectivity verified"
        self.verified_port_count = 10


class DummyConnectivityService:
    """Mock executor connectivity service."""

    async def verify_ports(self, *args, **kwargs):
        return ConnectivityResult(kwargs.get("sysbox_runtime", True))


class DummyPortMappingService:
    """Mock port mapping service."""

    async def get_successful_ports_count(self, executor_uuid: str) -> int:
        return 10  # Sufficient ports


class DummyKeypair:
    """Mock keypair that is JSON serializable."""

    def __init__(self, ss58_address: str = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"):
        self.ss58_address = ss58_address

    def sign(self, data: bytes) -> bytes:
        return b"\x00" * 64


def dummy_score_calculator(ctx, rented: bool):
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
    # Setup keypair (not Mock - to avoid Pydantic serialization issues)
    keypair = DummyKeypair()

    # Setup real executor (not Mock - to avoid Pydantic serialization issues)
    executor = make_executor("executor-123")

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
        validator_keypair=keypair,
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

    # Setup state with rented_data (not rented = empty executors)
    rented_data = RentedExecutorsResponse(executors={}, banned_guids=[])
    state = build_state(
        upload_local_dir="/tmp/validator/files",
        rented_data=rented_data,
    )

    # Create context
    ctx = context_factory(
        executor=executor,
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
    executor_uuid = "executor-123"

    # Setup keypair (not Mock - to avoid Pydantic serialization issues)
    keypair = DummyKeypair()

    # Setup real executor (not Mock - to avoid Pydantic serialization issues)
    executor = make_executor(executor_uuid)

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
            if "docker ps" in command:
                return SimpleSSHResult(stdout="container-id-123")  # Container is running
            elif "authorized_keys" in command:
                return SimpleSSHResult(stdout="ssh-rsa AAA...")  # SSH keys present
            return SimpleSSHResult(stdout="")

    ssh_client = RentedSSHClient()

    # Setup SSH service
    ssh_service = DummySSHService()

    # Build rented_data for this executor
    rented_data = RentedExecutorsResponse(
        executors={
            executor_uuid: RentedExecutor(
                miner_hotkey="miner-hotkey-123",
                executor_ip_address="192.168.1.100",
                executor_ip_port="8080",
                pods=[
                    RentedPod(
                        pod_id="pod-123",
                        container_name="tenant-container-123",
                        rented_ports=[],
                    )
                ],
                owner_flag=False,
                rented_ports=[],
            )
        },
        banned_guids=[],
    )

    # Create runner and other services
    runner = DummySSHCommandRunner()
    redis_service = DummyRedisService()
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
        validator_keypair=keypair,
        compute_rest_app_url="http://validator:8000",
        gpu_monitor_script_relative="src/gpus_utility.py",
        machine_scrape_filename="scrape.sh",
        machine_scrape_timeout=300,
        obfuscation_keys={},
        max_gpu_count=8,
        gpu_model_rates={"NVIDIA RTX 4090": 1.0},
        nvml_digest_map={"535.104.05": "expected_digest_for_535.104.05"},
    )

    # Setup state with GPU processes in the correct container and rented_data
    state = build_state(
        upload_local_dir="/tmp/validator/files",
        gpu_processes=[
            {"container_name": "tenant-container-123", "pid": 1234}
        ],
        gpu_details=[
            {"gpu_utilization": 50, "memory_utilization": 60}
        ],
        rented_data=rented_data,
    )

    # Create context
    ctx = context_factory(
        executor=executor,
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
