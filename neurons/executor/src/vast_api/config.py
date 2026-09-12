from pydantic_settings import BaseSettings, SettingsConfigDict


class VastSettings(BaseSettings):
    """Settings for the vast_api module.

    Kept separate from core.config Settings on purpose: every field here has a
    working default, so the executor boots unchanged on boxes that never touch
    Vast.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

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
