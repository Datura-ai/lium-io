from pydantic_settings import BaseSettings, SettingsConfigDict


class VastSettings(BaseSettings):
    """Standalone settings for the vast_api module.

    Deliberately independent from core.config: the executor Settings requires
    MINER_HOTKEY_SS58_ADDRESS/DB_URI env vars the sidecar shell must not need.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    VAST_API_TOKEN: str
    VAST_ACCOUNT_KEY_FILE: str = "/run/secrets/vast_host_key"
    EXECUTOR_CONTAINER_NAME: str = "executor-executor-1"
    VAST_UNS_NAME: str = "vast-uns"
    VAST_UNS_IMAGE_TAG: str = "vast-uns-kaalia:img"
    STATE_DIR_HOST: str = "/var/lib/vast-uns-state"
    DMI_BIN_HOST: str = "/var/lib/vast-dmi.bin"
    DATA_ROOT_IMG: str = "/ephemeral/vast-dockerd.img"
    DATA_ROOT_MOUNT: str = "/mnt/vast-dockerd"
    DATA_ROOT_SIZE_GB: int = 400
    PORT_RANGE_START: int = 40000
    PORT_RANGE_END: int = 40300
    RUNS_DIR: str = "/var/lib/vast-api/runs"
    LISTEN_PORT: int = 8151
