import enum
from datetime import datetime

from datura.requests.base import BaseRequest
from datura.requests.miner_requests import PodLog
from pydantic import BaseModel, Field, field_validator, model_serializer


class CustomOptions(BaseModel):
    volumes: list[str] | None = None
    environment: dict[str, str] | None = None
    entrypoint: str | None = None
    internal_ports: list[int] | None = None
    startup_commands: str | None = None
    shm_size: str | None = None
    initial_port_count: int | None = None

    @classmethod
    def sanitize(cls, custom_options: 'CustomOptions | None') -> 'CustomOptions':
        """Sanitize CustomOptions to prevent command injection attacks."""
        if not custom_options:
            return cls()

        # Sanitize volumes - only allow valid host:container path format
        volumes = cls._sanitize_volumes(custom_options.volumes or [])
        
        # Sanitize entrypoint - only allow single command, no flags
        entrypoint = cls._sanitize_entrypoint(custom_options.entrypoint)
        
        # Sanitize shm size - only allow valid size format
        shm_size = cls._sanitize_shm_size(custom_options.shm_size)
        
        # Sanitize environment variables
        environment = cls._sanitize_environment(custom_options.environment or {})
        
        return cls(
            volumes=volumes if volumes else None,
            environment=environment if environment else None,
            entrypoint=entrypoint,
            internal_ports=custom_options.internal_ports,
            initial_port_count=custom_options.initial_port_count,
            startup_commands=custom_options.startup_commands,
            shm_size=shm_size,
        )

    @staticmethod
    def _sanitize_volumes(volumes: list[str]) -> list[str]:
        """Sanitize volume mounts to prevent command injection."""
        sanitized = []
        for volume in volumes:
            if not volume or not volume.strip():
                continue
                
            # Remove any extra flags or commands
            clean_volume = volume.strip().split()[0]
            
            # Validate format: must be host_path:container_path
            if ':' not in clean_volume:
                continue
                
            host_path, container_path = clean_volume.split(':', 1)
            
            # Basic validation - reject dangerous paths
            if CustomOptions._is_dangerous_path(host_path) or CustomOptions._is_dangerous_path(container_path):
                continue
                
            sanitized.append(clean_volume)
            
        return sanitized

    @staticmethod
    def _sanitize_entrypoint(entrypoint: str | None) -> str | None:
        """Sanitize entrypoint to prevent command injection."""
        import re
        if not entrypoint or not entrypoint.strip():
            return None
            
        # Only allow single command, no flags or arguments
        clean_entrypoint = entrypoint.strip().split()[0]
        
        # Basic validation - only allow alphanumeric, dots, slashes, hyphens, underscores
        # Allow relative paths (./), absolute paths (/), and simple commands (abc/)
        if not re.match(r'^[a-zA-Z0-9./_-]+$', clean_entrypoint):
            return None
            
        return clean_entrypoint

    @staticmethod
    def _sanitize_shm_size(shm_size: str | None) -> str | None:
        """Sanitize shared memory size to prevent command injection."""
        import re
        if not shm_size or not shm_size.strip():
            return None
            
        # Only allow valid size format (e.g., "1g", "512m", "1024")
        clean_size = shm_size.strip().split()[0]
        
        # Validate format: number followed by optional unit (case insensitive)
        if not re.match(r'^\d+[kmg]?$', clean_size.lower()):
            return None
            
        return clean_size

    @staticmethod
    def _sanitize_environment(environment: dict[str, str]) -> dict[str, str]:
        """Sanitize environment variables to prevent command injection."""
        sanitized = {}
        for key, value in environment.items():
            if not key or not value or not key.strip() or not str(value).strip():
                continue
                
            # Basic validation for key and value
            clean_key = key.strip()
            clean_value = str(value).strip()
            
            # Reject keys that might be dangerous
            if CustomOptions._is_dangerous_env_key(clean_key):
                continue
                
            sanitized[clean_key] = clean_value
            
        return sanitized

    @staticmethod
    def _is_dangerous_path(path: str) -> bool:
        """Check if path is potentially dangerous."""
        dangerous_patterns = [
            '/etc', '/proc', '/sys', '/dev', '/var/run/docker.sock',
            '/usr/bin/docker', '/bin', '/sbin', '/usr/sbin'
        ]
        
        path_lower = path.lower()
        return any(dangerous in path_lower for dangerous in dangerous_patterns)

    @staticmethod
    def _is_dangerous_env_key(key: str) -> bool:
        """Check if environment key is potentially dangerous."""
        dangerous_keys = [
            'PATH', 'LD_LIBRARY_PATH', 'LD_PRELOAD', 'PYTHONPATH',
            'NODE_PATH', 'RUBYLIB', 'PERL5LIB'
        ]
        
        return key.upper() in dangerous_keys


class MinerJobRequestPayload(BaseModel):
    job_batch_id: str
    miner_hotkey: str
    miner_coldkey: str
    miner_address: str
    miner_port: int


class MinerJobEnryptedFiles(BaseModel):
    encrypt_key: str
    all_keys: dict
    tmp_directory: str
    machine_scrape_file_name: str
    # score_file_name: str


class ResourceType(BaseModel):
    cpu: int
    gpu: int
    memory: str
    volume: str

    @field_validator("cpu", "gpu")
    def validate_positive_int(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"{v} should be a valid non-negative integer string.")
        return v

    @field_validator("memory", "volume")
    def validate_memory_format(cls, v: str) -> str:
        if not v[:-2].isdigit() or v[-2:].upper() not in ["MB", "GB"]:
            raise ValueError(f"{v} is not a valid format.")
        return v


class ContainerRequestType(enum.Enum):
    ContainerCreateRequest = "ContainerCreateRequest"
    ContainerStartRequest = "ContainerStartRequest"
    ContainerStopRequest = "ContainerStopRequest"
    ContainerDeleteRequest = "ContainerDeleteRequest"
    AddSshPublicKey = "AddSshPublicKey"
    RemoveSshPublicKeysRequest = "RemoveSshPublicKeysRequest"
    DuplicateExecutorsResponse = "DuplicateExecutorsResponse"
    ExecutorRentFinished = "ExecutorRentFinished"
    GetPodLogsRequestFromServer = "GetPodLogsRequestFromServer"
    AddDebugSshKeyRequest = "AddDebugSshKeyRequest"
    BackupContainerRequest = "BackupContainerRequest"
    RestoreContainerRequest = "RestoreContainerRequest"
    InstallJupyterServer = "InstallJupyterServer"


class WorkloadKind(enum.Enum):
    CUSTOMER_RENTAL = "CUSTOMER_RENTAL"
    FILLER = "FILLER"


class BaseServerRequest(BaseRequest):
    message_type: ContainerRequestType
    miner_hotkey: str
    miner_address: str | None = None
    miner_port: int | None = None
    executor_id: str


class ContainerBaseRequest(BaseServerRequest):
    pod_id: str
    workload_kind: WorkloadKind = WorkloadKind.CUSTOMER_RENTAL


class ExternalVolumeInfo(BaseModel):
    name: str
    plugin: str
    iam_user_access_key: str
    iam_user_secret_key: str


class PayloadPortMapping(BaseModel):
    docker_port: int | None = None
    internal_port: int
    external_port: int


class GpuPowerLimit(BaseModel):
    # DAH-2356: one GPU's power cap for the Lium PEARL default job (see gpu_power_limits below).
    # gt=0: reject a buggy backend value (0/negative) at the boundary, not as a hardware-minimum cap.
    gpu_uuid: str
    watts: int = Field(gt=0)


class ContainerCreateRequest(ContainerBaseRequest):
    """Container creation request from the backend.

    Disk sizing contract: when ``disk_share`` is set, the validator computes
    effective volume/storage limits from fresh on-host disk state and
    ``volume_limit_gb``/``storage_limit_gb`` act only as upper-bound caps.
    When ``disk_share`` is absent, ``volume_limit_gb``/``storage_limit_gb``
    are exact sizes (legacy behavior).
    """

    message_type: ContainerRequestType = ContainerRequestType.ContainerCreateRequest
    docker_image: str
    user_public_keys: list[str] = []
    gpu_uuids: list[str]
    cpu_count: int | None = None
    memory_gb: int | None = None
    custom_options: CustomOptions | None = None
    debug: bool | None = None
    local_volume: str | None = None
    volume_limit_gb: int | None = None
    storage_limit_gb: int | None = None
    disk_share: float | None = None  # pod's share of machine disk (rented_gpus/total_gpus); None -> legacy sizing
    min_volume_gb: int | None = None  # reject floor for fresh sizing; None -> never reject (shrink only)
    external_volume_info: ExternalVolumeInfo | None = None
    is_sysbox: bool | None = None
    docker_username: str | None = None  # when edit pod, docker_username is required
    # when edit pod, docker_password is required
    docker_password: str | None = Field(default=None, repr=False)
    timestamp: int | None = None
    backup_log_id: str | None = None
    restore_path: str | None = None
    enable_jupyter: bool | None = None
    available_ports: list[PayloadPortMapping] | None = None
    pod_mapping: list[PayloadPortMapping] | None = None
    active_container_names: list[str] | None = None
    active_volume_names: list[str] | None = None
    # DAH-2211 (custom-dockerfile pod): when present and non-empty the validator
    # builds the image from this Dockerfile on the executor host instead of pulling
    # `docker_image`. None or "" keeps the normal image-pull path.
    dockerfile_content: str | None = None
    # DAH-1524 introduced this flag ("skip the bootstrap for cached templates");
    # its semantics have since changed TWICE — do not trust older comments:
    #   - 2345af85 reverted the skip: the sshd bootstrap ALWAYS runs, because a
    #     renter's custom startup_commands replaces the image CMD, so the image
    #     alone can never be trusted to bring sshd up.
    #   - DAH-2341 made the bootstrap converge with a self-starting image
    #     instead of racing it (grace-wait, adopt a running sshd, no keygen
    #     when adopted; see sshd_bootstrap.sh header).
    # Today ships_sshd=True (set by lium-io-backend for cached recommended
    # images) means exactly one thing: a bootstrap failure is FATAL to the
    # deploy (RuntimeError -> FailedContainerRequest). None/False keeps
    # bootstrap failures lenient (logged, deploy continues).
    ships_sshd: bool | None = None
    # DAH-2356: per-GPU power caps for the Lium PEARL default-job (FILLER) container. Applied
    # host-side via `nvidia-smi -pl` (clamped to the GPU's live min/max) before the filler starts;
    # the pre-cap limit is recorded and restored on delete. None -> no cap (customer rentals, DPHN,
    # miner default jobs). MUST be declared here or pydantic drops it on deserialization of the
    # backend's request.
    gpu_power_limits: list[GpuPowerLimit] | None = None


class ExecutorRentFinishedRequest(ContainerBaseRequest):
    message_type: ContainerRequestType = ContainerRequestType.ExecutorRentFinished


class ContainerStartRequest(ContainerBaseRequest):
    message_type: ContainerRequestType = ContainerRequestType.ContainerStartRequest
    container_name: str


class AddSshPublicKeyRequest(ContainerBaseRequest):
    message_type: ContainerRequestType = ContainerRequestType.AddSshPublicKey
    container_name: str
    user_public_keys: list[str] = []


class RemoveSshPublicKeysRequest(ContainerBaseRequest):
    message_type: ContainerRequestType = ContainerRequestType.RemoveSshPublicKeysRequest
    container_name: str
    user_public_keys: list[str] = []


class AddDebugSshKeyRequest(BaseServerRequest):
    message_type: ContainerRequestType = ContainerRequestType.AddDebugSshKeyRequest
    public_key: str


class InstallJupyterServerRequest(ContainerBaseRequest):
    message_type: ContainerRequestType = ContainerRequestType.InstallJupyterServer
    container_name: str
    jupyter_port_map: tuple[int, int]
    local_volume: str | None = None
    local_volume_path: str = "/root"


class ContainerStopRequest(ContainerBaseRequest):
    message_type: ContainerRequestType = ContainerRequestType.ContainerStopRequest
    container_name: str


class ContainerDeleteRequest(ContainerBaseRequest):
    message_type: ContainerRequestType = ContainerRequestType.ContainerDeleteRequest
    container_name: str
    local_volume: str | None = None
    external_volume: str | None = None


class GetPodLogsRequestFromServer(ContainerBaseRequest):
    message_type: ContainerRequestType = ContainerRequestType.GetPodLogsRequestFromServer
    container_name: str


class BackupContainerRequest(ContainerBaseRequest):
    message_type: ContainerRequestType = ContainerRequestType.BackupContainerRequest
    source_volume: str
    backup_volume_info: ExternalVolumeInfo  # S3 backup volume with credentials
    backup_path: str
    source_volume_path: str
    backup_target_path: str
    auth_token: str  # JWT for progress updates
    backup_log_id: str


class RestoreContainerRequest(ContainerBaseRequest):
    message_type: ContainerRequestType = ContainerRequestType.RestoreContainerRequest
    target_volume: str
    backup_volume_info: ExternalVolumeInfo  # S3 backup volume with credentials
    backup_source_path: str  # path in backup S3 volume
    target_volume_path: str  # local volume mounted path
    auth_token: str  # JWT for progress updates
    restore_log_id: str
    restore_path: str


##############################################################
# Response payloads
##############################################################

class ContainerResponseType(enum.Enum):
    ContainerCreated = "ContainerCreated"
    ContainerStarted = "ContainerStarted"
    ContainerStopped = "ContainerStopped"
    ContainerDeleted = "ContainerDeleted"
    SshPubKeyAdded = "SshPubKeyAdded"
    FailedRequest = "FailedRequest"
    PodLogsResponseToServer = "PodLogsResponseToServer"
    FailedGetPodLogs = "FailedGetPodLogs"
    DebugSshKeyAdded = "DebugSshKeyAdded"
    FailedAddDebugSshKey = "FailedAddDebugSshKey"
    SshPubKeyRemoved = "SshPubKeyRemoved"
    JupyterServerInstalled = "JupyterServerInstalled"
    JupyterInstallationFailed = "JupyterInstallationFailed"


class BaseValidatorResponse(BaseRequest):
    message_type: ContainerResponseType
    miner_hotkey: str
    executor_id: str


class ContainerWarningCode(enum.Enum):
    ExternalVolumeFailed = "ExternalVolumeFailed"


class ContainerBaseResponse(BaseValidatorResponse):
    pod_id: str
    workload_kind: WorkloadKind = WorkloadKind.CUSTOMER_RENTAL

class ProfilerStepName(str, enum.Enum):
    """Stable identifiers for each deploy-profiling step (DAH-1524).

    The string *values* double as Loki query keys and are consumed by
    lium-io-backend, so they must not be changed without coordinating both.
    Adding a new deploy step means adding a member here.
    """

    REQUESTED_FROM_BACKEND = "Requested from backend"
    STARTED_IN_SUBNET = "Started in subnet"
    PORT_MAPPINGS_GENERATED = "Port mappings generated"
    SSH_CONNECTION_ESTABLISHED = "SSH connection established"
    DOCKER_LOGIN = "Docker login step finished"
    CUSTOM_DOCKER_BUILD = "Custom docker build step finished"
    DOCKER_PULL = "Docker pull step finished"
    CONTAINER_CLEANING = "Container cleaning step finished"
    DOCKER_VOLUME_CREATION = "Docker volume creation step finished"
    DOCKER_VOLUME_CREATION_FAILED = "Docker volume creation step failed"
    GPU_DEVICE_PROBE = "GPU device probe step finished"
    PORT_CHECK_WAIT = "Port-check wait step finished"
    DOCKER_RUN = "Docker run step finished"
    CONTAINER_RUNNING_CHECK = "Container running check step finished"
    SSH_SERVICE_INSTALLATION = "SSH service installation step finished"
    ADDING_PUBLIC_KEYS = "Adding public keys step finished"
    INSPECTOR_START = "Inspector collector start step finished"
    FINISHED_IN_SUBNET = "Finished in subnet."


def now_ms() -> int:
    """Current wall-clock time as integer milliseconds (the deploy-profiling clock)."""
    return int(datetime.utcnow().timestamp() * 1000)


class ProfilerStep(BaseModel):
    """One step in a deploy profile (DAH-1524).

    Typed replacement for the former free-form ``dict``. The anchor step
    carries ``timestamp`` (no ``duration``); every other step carries a
    ``duration`` in ms; ``skipped`` marks a short-circuited step.

    ``_serialize`` reproduces the historical wire shape exactly — it emits
    only the keys that were present before this became a model — so
    lium-io-backend keeps receiving byte-identical JSON.
    """

    name: ProfilerStepName
    duration: int | None = None
    timestamp: int | None = None
    skipped: bool = False

    @classmethod
    def since(
        cls, name: ProfilerStepName, prev_ms: int, *, skipped: bool = False
    ) -> "ProfilerStep":
        """Build a step whose ``duration`` is the time elapsed since ``prev_ms`` (ms).

        Collapses the repeated ``ProfilerStep(name=..., duration=now_ms() -
        prev_timestamp)`` at every deploy step into one readable call.
        """
        return cls(name=name, duration=now_ms() - prev_ms, skipped=skipped)

    @model_serializer
    def _serialize(self) -> dict:
        data: dict = {"name": self.name.value}
        if self.timestamp is not None:
            data["timestamp"] = self.timestamp
        if self.duration is not None:
            data["duration"] = self.duration
        if self.skipped:
            data["skipped"] = self.skipped
        return data


class ContainerCreated(ContainerBaseResponse):
    message_type: ContainerResponseType = ContainerResponseType.ContainerCreated
    container_name: str
    volume_name: str
    port_maps: list[tuple[int, int]]
    profilers: list[ProfilerStep] = []
    backup_log_id: str | None = None
    restore_path: str | None = None
    jupyter_url: str | None = None
    warnings: list[ContainerWarningCode] | None = None
    storage_limit_gb: int | None = None
    volume_limit_gb: int | None = None
    local_volume_path: str | None = None


class ContainerStarted(ContainerBaseResponse):
    message_type: ContainerResponseType = ContainerResponseType.ContainerStarted
    container_name: str


class ContainerStopped(ContainerBaseResponse):
    message_type: ContainerResponseType = ContainerResponseType.ContainerStopped
    container_name: str


class ContainerDeleted(ContainerBaseResponse):
    message_type: ContainerResponseType = ContainerResponseType.ContainerDeleted


class SshPubKeyAdded(ContainerBaseResponse):
    message_type: ContainerResponseType = ContainerResponseType.SshPubKeyAdded
    user_public_keys: list[str] = []


class SshPubKeyRemoved(ContainerBaseResponse):
    message_type: ContainerResponseType = ContainerResponseType.SshPubKeyRemoved
    user_public_keys: list[str] = []


class DebugSshKeyAdded(BaseValidatorResponse):
    message_type: ContainerResponseType = ContainerResponseType.DebugSshKeyAdded
    address: str
    port: int
    ssh_username: str
    ssh_port: int


class FailedContainerErrorCodes(enum.Enum):
    UnknownError = "UnknownError"
    NoSshKeys = "NoSshKeys"
    ContainerNotRunning = "ContainerNotRunning"
    NoPortMappings = "NoPortMappings"
    InvalidExecutorId = "InvalidExecutorId"
    ExceptionError = "ExceptionError"
    FailedMsgFromMiner = "FailedMsgFromMiner"
    RentingInProgress = "RentingInProgress"
    NoJupyterPortMapping = "NoJupyterPortMapping"
    AttestationError = "AttestationError"


class FailedContainerErrorTypes(enum.Enum):
    ContainerCreationFailed = "ContainerCreationFailed"
    ContainerDeletionFailed = "ContainerDeletionFailed"
    ContainerStopFailed = "ContainerStopFailed"
    ContainerStartFailed = "ContainerStartFailed"
    AddSSkeyFailed = "AddSSkeyFailed"
    UnknownRequest = "UnknownRequest"


class FailedContainerRequest(ContainerBaseResponse):
    message_type: ContainerResponseType = ContainerResponseType.FailedRequest
    error_type: FailedContainerErrorTypes = FailedContainerErrorTypes.ContainerCreationFailed
    msg: str
    error_code: FailedContainerErrorCodes | None = None
    failure_step: str | None = None


class DuplicateExecutorsResponse(BaseModel):
    message_type: ContainerRequestType = ContainerRequestType.DuplicateExecutorsResponse
    executors: dict[str, list]
    rental_succeed_executors: list[str] | None = None


class PodLogsResponseToServer(ContainerBaseResponse):
    message_type: ContainerResponseType = ContainerResponseType.PodLogsResponseToServer
    container_name: str
    logs: list[PodLog] = []


class FailedGetPodLogs(ContainerBaseResponse):
    message_type: ContainerResponseType = ContainerResponseType.FailedGetPodLogs
    container_name: str
    msg: str


class FailedAddDebugSshKey(BaseValidatorResponse):
    message_type: ContainerResponseType = ContainerResponseType.FailedAddDebugSshKey
    msg: str


class JupyterServerInstalled(ContainerBaseResponse):
    message_type: ContainerResponseType = ContainerResponseType.JupyterServerInstalled
    jupyter_url: str


class JupyterInstallationFailed(ContainerBaseResponse):
    message_type: ContainerResponseType = ContainerResponseType.JupyterInstallationFailed
    msg: str


class GetEstimateRequest(BaseModel):
    request_id: str = ""
    gpu_model: str
    gpu_count: int = 1
    is_rented: bool = False
    gpu_splitting: bool = False
    gpu_splitting_min_count: int | None = None
    sysbox_runtime: bool = True
    collateral_deposited: bool = True
